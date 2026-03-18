"""CLIP text encoder -- supports CLIP-L and CLIP-G (OpenCLIP ViT-bigG)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "CLIPEncoder",
    "CLIPType",
    "TextOutput",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class CLIPType(str, Enum):
    """Supported CLIP model variants."""

    CLIP_L = "clip_l"
    CLIP_G = "clip_g"


@dataclass
class TextOutput:
    """Output from a text encoder forward pass.

    Attributes:
        hidden_states: Encoder hidden states, shape ``(batch, seq_len, dim)``.
        pooled_output: Pooled representation, shape ``(batch, dim)``.
            ``None`` for encoders that do not produce a pooled output (e.g. T5).
    """

    hidden_states: Any  # torch.Tensor at runtime
    pooled_output: Any | None = None  # torch.Tensor | None


# ---------------------------------------------------------------------------
# CLIP hidden dimensions
# ---------------------------------------------------------------------------

_CLIP_DIMS: dict[CLIPType, int] = {
    CLIPType.CLIP_L: 768,
    CLIPType.CLIP_G: 1280,
}


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class CLIPEncoder:
    """Load and run a CLIP text encoder.

    Supports both CLIP-L (768-dim, used by SD15/SDXL/Flux) and
    CLIP-G (1280-dim OpenCLIP ViT-bigG, used by SDXL).

    All ``transformers`` imports are lazy so the module can be imported
    without the library installed.

    Args:
        model_path: HuggingFace repo or local path.  Loaded eagerly if given.
        dtype: Torch dtype for model weights.
        device: Target device string.
        use_projection: When ``True``, load ``CLIPTextModelWithProjection``
            instead of ``CLIPTextModel``.  Required for CLIP-G (SDXL).
    """

    def __init__(
        self,
        model_path: str | None = None,
        dtype: Any = None,
        device: str = "cpu",
        use_projection: bool = False,
    ) -> None:
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._dtype = dtype
        self._device = device
        self._use_projection = use_projection
        if model_path is not None:
            self.load(model_path)

    # -- lifecycle -----------------------------------------------------------

    def load(
        self,
        model_path: str,
        subfolder: str | None = None,
        tokenizer_subfolder: str | None = None,
    ) -> None:
        """Load a CLIP model and tokenizer from *model_path*.

        Accepts either:
        - A single ``.safetensors`` file (ComfyUI-style, with companion
          ``.tokenizer.json`` next to it)
        - A HuggingFace repo ID or local directory with config + tokenizer

        Args:
            model_path: HuggingFace repo ID, local directory, or ``.safetensors`` file.
            subfolder: Optional subfolder for the model weights
                (e.g. ``"text_encoder_2"`` for SDXL CLIP-G).
            tokenizer_subfolder: Optional subfolder for the tokenizer.
                Defaults to *subfolder* when not specified.
        """
        import torch

        if self._dtype is None:
            self._dtype = torch.float16

        if model_path.endswith(".safetensors"):
            self._load_from_safetensors(model_path)
            return

        try:
            from transformers import CLIPTokenizer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "transformers is required to load CLIP models. "
                "Install it with: pip install transformers"
            ) from exc

        tok_sf = tokenizer_subfolder or subfolder
        tok_kwargs: dict[str, str] = {}
        model_kwargs: dict[str, str] = {}
        if tok_sf:
            tok_kwargs["subfolder"] = tok_sf
        if subfolder:
            model_kwargs["subfolder"] = subfolder

        logger.info("Loading CLIP model from %s (subfolder=%s)", model_path, subfolder)
        self._tokenizer = CLIPTokenizer.from_pretrained(model_path, **tok_kwargs)

        if self._use_projection:
            from transformers import CLIPTextModelWithProjection  # type: ignore[import-untyped]

            self._model = CLIPTextModelWithProjection.from_pretrained(
                model_path,
                torch_dtype=self._dtype,
                **model_kwargs,
            ).to(self._device)
        else:
            from transformers import CLIPTextModel  # type: ignore[import-untyped]

            self._model = CLIPTextModel.from_pretrained(
                model_path,
                torch_dtype=self._dtype,
                **model_kwargs,
            ).to(self._device)
        self._model.eval()

    def _load_from_safetensors(self, safetensors_path: str) -> None:
        """Load from a single .safetensors file (ComfyUI-style).

        Infers model config from weight tensor shapes and loads the
        companion ``.tokenizer.json`` file from the same directory.
        """
        import os

        import torch
        from safetensors.torch import load_file
        from transformers import (
            CLIPTextConfig,
            CLIPTextModel,
            CLIPTextModelWithProjection,
            PreTrainedTokenizerFast,
        )

        logger.info("Loading CLIP from safetensors: %s", safetensors_path)
        state_dict = load_file(safetensors_path, device="cpu")

        # Detect architecture from shapes
        embed_weight = state_dict["text_model.embeddings.token_embedding.weight"]
        vocab_size, hidden_size = embed_weight.shape

        # Count layers
        num_layers = 0
        while f"text_model.encoder.layers.{num_layers}.layer_norm1.weight" in state_dict:
            num_layers += 1

        has_projection = "text_projection.weight" in state_dict
        num_heads = hidden_size // 64  # CLIP uses head_dim=64
        intermediate_size = hidden_size * 4

        logger.info(
            "CLIP config: hidden=%d, layers=%d, heads=%d, projection=%s",
            hidden_size, num_layers, num_heads, has_projection,
        )

        config_kwargs: dict[str, Any] = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads,
            "max_position_embeddings": 77,
        }
        if has_projection:
            config_kwargs["projection_dim"] = state_dict["text_projection.weight"].shape[0]

        config = CLIPTextConfig(**config_kwargs)

        if self._use_projection or has_projection:
            model = CLIPTextModelWithProjection(config)
        else:
            model = CLIPTextModel(config)

        model.load_state_dict(state_dict, strict=True)
        self._model = model.to(device=self._device, dtype=self._dtype)
        self._model.eval()

        # Load tokenizer from companion .tokenizer.json
        stem = os.path.splitext(safetensors_path)[0]
        tokenizer_path = stem + ".tokenizer.json"
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Companion tokenizer not found at {tokenizer_path}. "
                "Expected a .tokenizer.json file next to the .safetensors file."
            )

        self._tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        # CLIP-L (OpenAI, vocab=49408): bos=49406, eos=49407
        # CLIP-G (OpenCLIP, vocab=49408): bos=0, eos=49407
        if vocab_size <= 49408 and not self._use_projection:
            # CLIP-L
            self._tokenizer.bos_token_id = 49406
            self._tokenizer.eos_token_id = 49407
            self._tokenizer.pad_token_id = 49407
        else:
            # CLIP-G (OpenCLIP / SDXL text_encoder_2)
            self._tokenizer.bos_token_id = 0
            self._tokenizer.eos_token_id = 49407
            self._tokenizer.pad_token_id = 0
        self._tokenizer.model_max_length = 77
        logger.info("Loaded CLIP tokenizer from %s", tokenizer_path)

    def unload(self) -> None:
        """Release model and tokenizer, freeing memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        """``True`` if a model is currently loaded."""
        return self._model is not None

    # -- encoding ------------------------------------------------------------

    def encode(
        self,
        text: str,
        max_length: int = 77,
        clip_skip: int = 0,
    ) -> TextOutput:
        """Tokenize and encode *text*, applying prompt weights to embeddings.

        Parses ``(word:1.5)`` syntax and scales per-token hidden states
        accordingly.  When all weights are 1.0 the fast path is used.

        Args:
            text: Input prompt string (may contain weight syntax).
            max_length: Maximum token length (CLIP default is 77).
            clip_skip: Number of final encoder layers to skip.

        Returns:
            :class:`TextOutput` with hidden states and pooled output.
        """
        import torch

        from text.tokenizer import (
            has_non_default_weights,
            parse_prompt_weights,
            split_segments_at_break,
        )

        if not self.is_loaded:
            raise RuntimeError("CLIPEncoder is not loaded. Call load() first.")

        segments = parse_prompt_weights(text)

        # Fast path -- no weighting needed
        if not has_non_default_weights(segments):
            return self._encode_unweighted(text, max_length, clip_skip)

        # Split at BREAK boundaries
        groups = split_segments_at_break(segments)

        all_hidden: list[Any] = []
        pooled: Any | None = None

        for group in groups:
            hidden_chunk, pooled_chunk = self._encode_weighted_group(
                group, max_length, clip_skip,
            )
            all_hidden.append(hidden_chunk)
            if pooled is None:
                pooled = pooled_chunk

        if len(all_hidden) == 1:
            hidden = all_hidden[0]
        else:
            hidden = torch.cat(all_hidden, dim=1)

        return TextOutput(hidden_states=hidden, pooled_output=pooled)

    def _encode_unweighted(
        self,
        text: str,
        max_length: int,
        clip_skip: int,
    ) -> TextOutput:
        """Fast-path encoding with no weight application."""
        import torch

        tokens = self._tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        # CLIP models don't accept token_type_ids
        tokens.pop("token_type_ids", None)

        with torch.no_grad():
            outputs = self._model(
                **tokens,
                output_hidden_states=True,
            )

        hidden = self._select_hidden_state(outputs, clip_skip)

        if self._use_projection:
            pooled = getattr(outputs, "text_embeds", None)
        else:
            pooled = getattr(outputs, "pooler_output", None)

        return TextOutput(hidden_states=hidden, pooled_output=pooled)

    def _encode_weighted_group(
        self,
        group: list[tuple[str, float]],
        max_length: int,
        clip_skip: int,
    ) -> tuple[Any, Any | None]:
        """Encode a single BREAK-group with per-token weight scaling."""
        import torch

        from text.tokenizer import build_token_weight_map

        bos_id = self._tokenizer.bos_token_id
        eos_id = self._tokenizer.eos_token_id
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_id

        def _tokenize_bare(text: str) -> list[int]:
            return self._tokenizer.encode(text, add_special_tokens=False)

        token_ids, weights = build_token_weight_map(
            group,
            tokenize_fn=_tokenize_bare,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            max_length=max_length,
        )

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self._device)
        attention_mask = torch.tensor(
            [[1 if t != pad_id else 0 for t in token_ids]],
            dtype=torch.long,
            device=self._device,
        )

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden = self._select_hidden_state(outputs, clip_skip)

        weight_tensor = torch.tensor(
            weights, dtype=hidden.dtype, device=hidden.device,
        ).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        hidden = hidden * weight_tensor

        if self._use_projection:
            pooled = getattr(outputs, "text_embeds", None)
        else:
            pooled = getattr(outputs, "pooler_output", None)

        return hidden, pooled

    def _select_hidden_state(self, outputs: Any, clip_skip: int) -> Any:
        """Select the appropriate hidden state given *clip_skip*."""
        if clip_skip > 0 and len(outputs.hidden_states) > clip_skip:
            hidden = outputs.hidden_states[-(clip_skip + 1)]
            if hasattr(self._model, "text_model") and hasattr(
                self._model.text_model, "final_layer_norm",
            ):
                hidden = self._model.text_model.final_layer_norm(hidden)
        else:
            hidden = outputs.last_hidden_state
        return hidden
