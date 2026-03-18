"""Qwen 2.5 VL 7B text encoder for Capybara (HunyuanVideo 1.5).

Wraps the Qwen 2.5 VL language model (text-only forward, no vision inputs)
as a text encoder for Capybara image/video generation and editing.

Prompts are wrapped in a JSON chat template via ``apply_chat_template()``,
encoded through the LLM, then the instruction prefix is cropped from the
output so only the user-prompt tokens remain.

Two task modes:

**image** (default):
  System prompt describing image attributes, followed by user prompt.

**video**:
  System prompt describing video attributes (motion, camera, temporal),
  followed by user prompt.

The output hidden state is extracted from layer ``-(skip_layer + 1)``
(default ``skip_layer=2`` means the 3rd-from-last layer).  Shape is
``(B, seq_len, 3584)`` with an attention mask ``(B, seq_len)``.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any
import os

from text.clip import TextOutput

__all__ = [
    "Qwen25VLEncoder",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Special token IDs (Qwen 2.5 tokenizer)
# ---------------------------------------------------------------------------
_IM_START_TOKEN_ID = 151644  # <|im_start|>
_IM_END_TOKEN_ID = 151645    # <|im_end|>
_PAD_TOKEN_ID = 151643       # <|endoftext|> used as pad

# ---------------------------------------------------------------------------
# Chat templates -- matches Capybara's PROMPT_TEMPLATE definitions exactly
# ---------------------------------------------------------------------------
_IMAGE_TEMPLATE = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Describe the image by detailing the following aspects: \
        1. The main content and theme of the image. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. The background environment, light, style and atmosphere.",
    },
    {"role": "user", "content": "{}"},
]

_VIDEO_TEMPLATE = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video.",
    },
    {"role": "user", "content": "{}"},
]


def _apply_text_to_template(text: str, template: list[dict]) -> list[dict]:
    """Insert user text into a chat template, replacing ``{}`` placeholders."""
    result = deepcopy(template)
    for item in result:
        if isinstance(item, dict) and "content" in item:
            item["content"] = item["content"].format(text if text else " ")
    return result


def _resolve_qwen25vl_snapshot() -> Path | None:
    base = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = base / "models--Qwen--Qwen2.5-VL-7B-Instruct"
    if not repo_dir.exists():
        return None
    refs_main = repo_dir / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text().strip()
        snapshot = repo_dir / "snapshots" / revision
        if snapshot.exists():
            return snapshot
    snapshots = sorted((repo_dir / "snapshots").glob("*")) if (repo_dir / "snapshots").exists() else []
    return snapshots[-1] if snapshots else None


def _infer_qwen25vl_config_from_state_dict(state_dict: dict[str, torch.Tensor]) -> "Qwen2_5_VLConfig":
    from transformers import Qwen2_5_VLConfig, Qwen3Config

    embed_key = "model.embed_tokens.weight"
    if embed_key not in state_dict:
        embed_key = "embed_tokens.weight"
    if embed_key not in state_dict:
        raise KeyError(
            f"Cannot find embedding weight. Available keys (first 10): "
            f"{sorted(state_dict.keys())[:10]}"
        )

    vocab_size, hidden_size = state_dict[embed_key].shape
    num_layers = 0
    while True:
        q_key = f"model.layers.{num_layers}.self_attn.q_proj.weight"
        if q_key not in state_dict:
            break
        num_layers += 1

    gate_key = "model.layers.0.mlp.gate_proj.weight"
    intermediate_size = state_dict[gate_key].shape[0] if gate_key in state_dict else hidden_size * 4
    q_key = "model.layers.0.self_attn.q_proj.weight"
    q_out = state_dict[q_key].shape[0]
    head_dim = 128
    num_heads = q_out // head_dim
    k_key = "model.layers.0.self_attn.k_proj.weight"
    k_out = state_dict[k_key].shape[0]
    num_kv_heads = k_out // head_dim

    text_config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=40960,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        tie_word_embeddings=True,
    )

    return Qwen2_5_VLConfig(text_config=text_config, tie_word_embeddings=True)


class Qwen25VLEncoder:
    """Qwen 2.5 VL 7B text encoder for Capybara.

    Loads ``Qwen2_5_VLForConditionalGeneration`` (text-only path) and
    extracts a hidden state as the conditioning tensor.

    Args:
        model_path: Local directory with model weights.  Loaded eagerly
            if given.
        dtype: Torch dtype for model weights (default ``bfloat16``).
        device: Target device string.
        hidden_state_skip_layer: Which hidden layer to extract.
            ``0`` = last layer, ``2`` = 3rd from last (Capybara default).
        max_length: Maximum prompt tokens after cropping (default 1000).
    """

    def __init__(
        self,
        model_path: str | None = None,
        dtype: Any = None,
        device: str = "cpu",
        hidden_state_skip_layer: int = 2,
        max_length: int = 1000,
    ) -> None:
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._dtype = dtype
        self._device = device
        self._skip_layer = hidden_state_skip_layer
        self._max_length = max_length

        # Cached crop_start values (auto-calculated on first encode)
        self._crop_start_image: int | None = None
        self._crop_start_video: int | None = None

        if model_path is not None:
            self.load(model_path)

    # -- lifecycle -----------------------------------------------------------

    def load(self, model_path: str) -> None:
        """Load a Qwen 2.5 VL model and tokenizer from a local directory.

        Uses ``Qwen2_5_VLForConditionalGeneration`` from transformers.
        Always sets ``local_files_only=True`` -- never downloads anything.
        """
        import torch

        if self._dtype is None:
            self._dtype = torch.bfloat16

        logger.info("Loading Qwen25VL encoder from %s", model_path)

        try:
            from transformers import (  # type: ignore[import-untyped]
                AutoTokenizer,
                Qwen2_5_VLForConditionalGeneration,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers is required to load Qwen 2.5 VL models. "
                "Install it with: pip install transformers"
            ) from exc

        if model_path.endswith(".safetensors"):
            self._load_from_safetensors(model_path)
            return

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="right",
            local_files_only=True,
        )

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self._dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(self._device)
        self._model.eval()
        self._model.requires_grad_(False)

        # Pre-calculate crop_start for both templates
        self._crop_start_image = self._calculate_crop_start("image")
        self._crop_start_video = self._calculate_crop_start("video")
        logger.info(
            "Crop offsets: image=%d, video=%d",
            self._crop_start_image,
            self._crop_start_video,
        )

    def unload(self) -> None:
        """Release model and tokenizer, freeing memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._crop_start_image = None
        self._crop_start_video = None

    @property
    def is_loaded(self) -> bool:
        """``True`` if a model is currently loaded."""
        return self._model is not None

    # -- crop calculation ----------------------------------------------------

    def _calculate_crop_start(self, task_type: str) -> int:
        """Find where user content starts in the tokenized chat template.

        Tokenizes a dummy prompt through the template, then searches for
        the ``<|im_start|>user\\n`` marker.  Returns the token index
        immediately after that marker.
        """
        template = _IMAGE_TEMPLATE if task_type == "image" else _VIDEO_TEMPLATE
        dummy_messages = _apply_text_to_template("a photo of a cat", template)

        dummy_tokens = self._tokenizer.apply_chat_template(
            dummy_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = dummy_tokens["input_ids"][0].tolist()

        # Find "<|im_start|>user\n" marker
        marker = "<|im_start|>user\n"
        marker_tokens = self._tokenizer(marker, add_special_tokens=False)["input_ids"]

        for i in range(len(input_ids) - len(marker_tokens) + 1):
            if input_ids[i : i + len(marker_tokens)] == marker_tokens:
                return i + len(marker_tokens)

        # Fallback: find second <|im_start|> + skip 3 tokens
        count = 0
        for i, tid in enumerate(input_ids):
            if tid == _IM_START_TOKEN_ID:
                count += 1
                if count == 2:
                    return min(i + 3, len(input_ids))

        logger.warning("Could not find crop_start marker, defaulting to 0")
        return 0

    # -- encoding ------------------------------------------------------------

    def encode(
        self,
        text: str,
        task_type: str = "image",
        max_length: int | None = None,
    ) -> tuple[TextOutput, Any]:
        """Tokenize and encode *text*, extracting a hidden state.

        The chat-template prefix is cropped from the output so only
        user-prompt tokens remain.

        Args:
            text: Input prompt string.
            task_type: ``"image"`` or ``"video"``.
            max_length: Override max prompt tokens (default from init).

        Returns:
            A tuple ``(TextOutput, attention_mask)`` where:
            - ``TextOutput.hidden_states``: ``(1, seq_len, 3584)``
            - ``TextOutput.pooled_output``: always ``None``
            - ``attention_mask``: ``(1, seq_len)`` or ``None``
        """
        import torch

        if not self.is_loaded:
            raise RuntimeError("Qwen25VLEncoder is not loaded. Call load() first.")

        if task_type not in ("image", "video"):
            raise ValueError(f"Unsupported task_type: {task_type!r}")

        max_len = max_length if max_length is not None else self._max_length

        # Select template and crop offset
        if task_type == "image":
            template = _IMAGE_TEMPLATE
            crop_start = self._crop_start_image or 0
        else:
            template = _VIDEO_TEMPLATE
            crop_start = self._crop_start_video or 0

        # Apply prompt to template
        messages = _apply_text_to_template(text, template)

        # Tokenize via apply_chat_template (JSON chat format)
        tokens = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            truncation=True,
            max_length=max_len + crop_start,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].to(self._device)
        attn_mask = tokens["attention_mask"].to(self._device)

        # Forward pass -- text only, no pixel_values
        need_hidden = self._skip_layer is not None and self._skip_layer > 0
        with torch.inference_mode():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=need_hidden,
            )

        # Extract the target hidden state
        if need_hidden and outputs.hidden_states is not None:
            # hidden_states is a tuple: (embedding, layer_0, ..., layer_N)
            hidden = outputs.hidden_states[-(self._skip_layer + 1)]
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden = outputs.hidden_states[-1]
        else:
            raise RuntimeError(
                "Model output has no hidden states. "
                "Check that output_hidden_states=True is supported."
            )

        # Crop template prefix
        if crop_start > 0:
            hidden = hidden[:, crop_start:]
            attn_mask = attn_mask[:, crop_start:]

        # Return None for mask if all positions are attended
        attention_mask_out: Any = None
        if attn_mask.sum() != attn_mask.numel():
            attention_mask_out = attn_mask

        return TextOutput(hidden_states=hidden, pooled_output=None), attention_mask_out

    def _load_from_safetensors(self, safetensors_path: str) -> None:
        """Load Qwen2.5-VL from a single safetensors file using cached config when available."""
        from transformers import (
            Qwen2_5_VLConfig,
            Qwen2_5_VLForConditionalGeneration,
            Qwen2Tokenizer,
        )

        tokenizer_dir = os.path.join(os.path.dirname(__file__), "qwen25_tokenizer")
        if not os.path.isdir(tokenizer_dir):
            raise FileNotFoundError(
                f"Bundled tokenizer not found at {tokenizer_dir}. "
                "Copy ComfyUI's qwen25_tokenizer/ directory into text/."
            )
        self._tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_dir)
        logger.info("Loaded Qwen2Tokenizer from %s", tokenizer_dir)

        config_dir = _resolve_qwen25vl_snapshot()
        if config_dir is None:
            from safetensors.torch import load_file

            logger.info("Loading safetensors weights from %s", safetensors_path)
            state_dict = load_file(safetensors_path, device="cpu")
            logger.info("Loaded %d tensors from safetensors", len(state_dict))
            config = _infer_qwen25vl_config_from_state_dict(state_dict)
            self._model = Qwen2_5_VLForConditionalGeneration(config)
            self._model.load_state_dict(state_dict, strict=False)
            self._model = self._model.to(device=self._device, dtype=self._dtype)
        else:
            from accelerate import init_empty_weights
            from accelerate.utils import load_checkpoint_in_model

            config = Qwen2_5_VLConfig.from_pretrained(str(config_dir), local_files_only=True)
            logger.info(
                "Streaming Qwen2.5-VL weights from %s using config from %s",
                safetensors_path,
                config_dir,
            )
            with init_empty_weights():
                self._model = Qwen2_5_VLForConditionalGeneration(config)
            self._model.to_empty(device="cpu")

            if hasattr(self._model, "tie_weights"):
                self._model.tie_weights()

            load_checkpoint_in_model(
                self._model,
                checkpoint=safetensors_path,
                device_map={"": "cpu"},
                dtype=self._dtype,
                strict=False,
            )
            self._model = self._model.to(device=self._device)

        self._model.eval()
        self._model.requires_grad_(False)

        self._crop_start_image = self._calculate_crop_start("image")
        self._crop_start_video = self._calculate_crop_start("video")
        logger.info(
            "Crop offsets: image=%d, video=%d",
            self._crop_start_image,
            self._crop_start_video,
        )
