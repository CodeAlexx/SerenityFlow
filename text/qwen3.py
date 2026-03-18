"""Qwen3 text encoder for Klein and ZImage models.

Supports two modes:

**Klein mode** (``mode="klein"``):
  Extract hidden states at layers [9, 18, 27] and stack them along
  the hidden dimension to produce the joint_attention_dim embedding:
  - Klein 4B: hidden_size=2560, joint_dim = 3 * 2560 = 7680
  - Klein 9B: hidden_size=4096, joint_dim = 3 * 4096 = 12288

**ZImage mode** (``mode="zimage"``):
  Extract the penultimate layer hidden state (layer_idx=-2).
  - Output dim: 2560 (Qwen3 4B hidden size)

Chat templates are hardcoded -- we do NOT rely on transformers'
``apply_chat_template`` since tokenizer files may not include one.
"""

from __future__ import annotations

import logging
from typing import Any

from text.clip import TextOutput

__all__ = [
    "Qwen3Encoder",
]

logger = logging.getLogger(__name__)

# Default extraction layers for Klein models
_KLEIN_EXTRACT_LAYERS: tuple[int, int, int] = (9, 18, 27)

# Hardcoded chat templates -- these MUST match the training conventions.
# Klein: includes <think> tags (thinking-mode prompt)
_KLEIN_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
# ZImage: NO thinking tags (plain assistant prompt)
_ZIMAGE_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


class Qwen3Encoder:
    """Qwen3 text encoder for Klein and ZImage models.

    Loads a Qwen3 causal LM and extracts hidden states according to
    the selected *mode*:

    - ``"klein"``: Stacks layers [9, 18, 27] into ``(B, seq, 3*D)``.
    - ``"zimage"``: Returns penultimate layer as ``(B, seq, D)``.

    Args:
        model_path: HuggingFace repo or local path. Loaded eagerly if given.
        mode: ``"klein"`` or ``"zimage"``.
        extract_layers: Tuple of layer indices for Klein mode (ignored in ZImage).
        dtype: Torch dtype for model weights.
        device: Target device string.
    """

    def __init__(
        self,
        model_path: str | None = None,
        mode: str = "klein",
        extract_layers: tuple[int, ...] = _KLEIN_EXTRACT_LAYERS,
        dtype: Any = None,
        device: str = "cpu",
    ) -> None:
        if mode not in ("klein", "zimage"):
            raise ValueError(f"Unsupported mode: {mode!r}. Must be 'klein' or 'zimage'.")

        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._dtype = dtype
        self._device = device
        self._mode = mode
        self._extract_layers = extract_layers
        if model_path is not None:
            self.load(model_path)

    # -- lifecycle -----------------------------------------------------------

    def load(self, model_path: str) -> None:
        """Load a Qwen3 causal LM and tokenizer.

        Accepts either:
        - A single ``.safetensors`` file (ComfyUI-style single-file checkpoint)
        - A HuggingFace repo ID or local directory with config.json + tokenizer files
        """
        try:
            from transformers import AutoModelForCausalLM, Qwen2Tokenizer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "transformers is required to load Qwen3 models. "
                "Install it with: pip install transformers"
            ) from exc

        import os
        import torch

        if self._dtype is None:
            self._dtype = torch.float16

        logger.info("Loading Qwen3 encoder (mode=%s) from %s", self._mode, model_path)

        if model_path.endswith(".safetensors"):
            self._load_from_safetensors(model_path)
        else:
            # HuggingFace directory or repo ID
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=self._dtype, trust_remote_code=True,
            ).to(self._device)
            self._model.eval()

    def _load_from_safetensors(self, safetensors_path: str) -> None:
        """Load from a single .safetensors file (ComfyUI-style).

        Uses bundled tokenizer files and infers model config from the
        weight tensor shapes.
        """
        import os

        import torch
        from safetensors.torch import load_file
        from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3ForCausalLM

        # 1. Load tokenizer from bundled files
        tokenizer_dir = os.path.join(os.path.dirname(__file__), "qwen25_tokenizer")
        if not os.path.isdir(tokenizer_dir):
            raise FileNotFoundError(
                f"Bundled tokenizer not found at {tokenizer_dir}. "
                "Copy ComfyUI's qwen25_tokenizer/ directory into text/."
            )
        self._tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_dir)
        logger.info("Loaded Qwen2Tokenizer from %s", tokenizer_dir)

        # 2. Load weights
        logger.info("Loading safetensors weights from %s", safetensors_path)
        state_dict = load_file(safetensors_path, device="cpu")
        logger.info("Loaded %d tensors from safetensors", len(state_dict))

        # 3. Infer config from weight shapes
        # Qwen3 4B: hidden=2560, intermediate=6912, 36 layers, 32 heads
        # Qwen3 8B: hidden=4096, intermediate=11008, 36 layers, 32 heads
        # Detect from embedding weight shape
        embed_key = "model.embed_tokens.weight"
        if embed_key not in state_dict:
            # Try without model. prefix
            embed_key = "embed_tokens.weight"
        if embed_key not in state_dict:
            raise KeyError(
                f"Cannot find embedding weight. Available keys (first 10): "
                f"{sorted(state_dict.keys())[:10]}"
            )

        vocab_size, hidden_size = state_dict[embed_key].shape
        logger.info("Detected: vocab_size=%d, hidden_size=%d", vocab_size, hidden_size)

        # Count layers
        num_layers = 0
        while True:
            k = f"model.layers.{num_layers}.self_attn.q_proj.weight"
            if k not in state_dict:
                break
            num_layers += 1
        logger.info("Detected %d layers", num_layers)

        # Infer intermediate size from gate_proj
        gate_key = "model.layers.0.mlp.gate_proj.weight"
        intermediate_size = state_dict[gate_key].shape[0] if gate_key in state_dict else hidden_size * 4

        # Infer num_heads from q_proj shape
        q_key = "model.layers.0.self_attn.q_proj.weight"
        q_out = state_dict[q_key].shape[0]
        head_dim = 128  # standard for Qwen
        num_heads = q_out // head_dim

        # Infer num_kv_heads from k_proj
        k_key = "model.layers.0.self_attn.k_proj.weight"
        k_out = state_dict[k_key].shape[0]
        num_kv_heads = k_out // head_dim

        logger.info(
            "Config: hidden=%d, layers=%d, heads=%d, kv_heads=%d, intermediate=%d",
            hidden_size, num_layers, num_heads, num_kv_heads, intermediate_size,
        )

        # 4. Build config and instantiate model
        # MUST use Qwen3Config (not Qwen2) — Qwen3 has independent head_dim=128
        # whereas Qwen2 derives head_dim = hidden_size/num_heads which is wrong.
        # rope_theta=1e6 matches ComfyUI's Qwen3_4BConfig / Qwen3_8BConfig.
        config = Qwen3Config(
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

        logger.info("Creating Qwen3ForCausalLM from config...")
        self._model = Qwen3ForCausalLM(config)

        # 5. Load state dict (strict=False: rotary embeddings are computed, not stored)
        self._model.load_state_dict(state_dict, strict=False)
        self._model = self._model.to(device=self._device, dtype=self._dtype)
        self._model.eval()
        logger.info("Qwen3 model loaded successfully on %s", self._device)

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

    # -- chat template -------------------------------------------------------

    def _format_prompt(self, text: str) -> str:
        """Apply the hardcoded chat template for the current mode.

        Klein: includes ``<think>`` tags.
        ZImage: plain assistant prompt, no thinking tags.
        """
        if self._mode == "zimage":
            return _ZIMAGE_TEMPLATE.format(prompt=text)
        else:
            return _KLEIN_TEMPLATE.format(prompt=text)

    # -- encoding ------------------------------------------------------------

    def encode(
        self,
        text: str,
        max_length: int = 512,
    ) -> TextOutput:
        """Tokenize and encode *text*, extracting hidden states.

        For Klein mode: stacks layers [9, 18, 27] into
        ``(B, seq, N * hidden_size)``.

        For ZImage mode: returns penultimate layer as
        ``(B, seq, hidden_size)``.

        Applies prompt weight syntax ``(word:1.5)`` when non-default
        weights are detected.

        Args:
            text: Input prompt string (may contain weight syntax).
            max_length: Maximum token length (default 512).

        Returns:
            :class:`TextOutput` with hidden states.
            ``pooled_output`` is always ``None``.
        """
        import torch

        from text.tokenizer import (
            has_non_default_weights,
            parse_prompt_weights,
            split_segments_at_break,
        )

        if not self.is_loaded:
            raise RuntimeError("Qwen3Encoder is not loaded. Call load() first.")

        segments = parse_prompt_weights(text)

        if not has_non_default_weights(segments):
            return self._encode_unweighted(text, max_length)

        groups = split_segments_at_break(segments)

        all_hidden: list[Any] = []
        for group in groups:
            all_hidden.append(self._encode_weighted_group(group, max_length))

        if len(all_hidden) == 1:
            hidden = all_hidden[0]
        else:
            hidden = torch.cat(all_hidden, dim=1)

        return TextOutput(hidden_states=hidden, pooled_output=None)

    # -- hidden state extraction ---------------------------------------------

    def _extract_hidden(self, hidden_states: tuple[Any, ...]) -> Any:
        """Extract hidden states according to the current mode.

        Klein: stack layers [9, 18, 27] and concatenate along hidden dim.
        ZImage: return penultimate layer directly.
        """
        if self._mode == "zimage":
            return self._extract_penultimate(hidden_states)
        else:
            return self._extract_and_stack(hidden_states)

    def _extract_and_stack(self, hidden_states: tuple[Any, ...]) -> Any:
        """Klein mode: extract layers and concatenate along hidden dimension.

        Given the full tuple of hidden states from all layers, extract
        the states at ``self._extract_layers`` and concatenate them
        along the last dimension.

        Input per layer: ``(B, seq, hidden_size)``
        Output: ``(B, seq, len(extract_layers) * hidden_size)``
        """
        import torch

        selected = [hidden_states[layer_idx] for layer_idx in self._extract_layers]

        # Stack: [B, N, seq, D] then reshape to [B, seq, N*D]
        stacked = torch.stack(selected, dim=1)
        batch_size, num_layers, seq_len, hidden_dim = stacked.shape
        return stacked.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, num_layers * hidden_dim,
        )

    def _extract_penultimate(self, hidden_states: tuple[Any, ...]) -> Any:
        """ZImage mode: return the penultimate hidden state.

        Input: tuple of ``(B, seq, hidden_size)`` for all layers.
        Output: ``(B, seq, hidden_size)`` from layer index -2.
        """
        if len(hidden_states) >= 2:
            return hidden_states[-2]
        # Fallback: last layer if model somehow has fewer than 2 layers
        return hidden_states[-1]

    # -- unweighted encoding -------------------------------------------------

    def _encode_unweighted(self, text: str, max_length: int) -> TextOutput:
        """Fast-path encoding with no weight application."""
        import torch

        formatted = self._format_prompt(text)

        tokens = self._tokenizer(
            formatted,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(
                **tokens,
                output_hidden_states=True,
            )

        hidden = self._extract_hidden(outputs.hidden_states)

        return TextOutput(hidden_states=hidden, pooled_output=None)

    # -- weighted encoding ---------------------------------------------------

    def _encode_weighted_group(
        self,
        group: list[tuple[str, float]],
        max_length: int,
    ) -> Any:
        """Encode a single BREAK-group with per-token weight scaling."""
        import torch

        from text.tokenizer import build_token_weight_map

        bos_id = getattr(self._tokenizer, "bos_token_id", None)
        eos_id = self._tokenizer.eos_token_id
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_id if eos_id is not None else 0

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

        hidden = self._extract_hidden(outputs.hidden_states)

        # Apply per-token weights
        weight_tensor = torch.tensor(
            weights, dtype=hidden.dtype, device=hidden.device,
        ).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        hidden = hidden * weight_tensor

        return hidden
