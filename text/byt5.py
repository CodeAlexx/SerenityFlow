"""ByT5-v2 encoder for HunyuanVideo 1.5 multilingual glyph rendering.

ByT5 is a byte-level T5 model that operates directly on UTF-8 bytes,
making it language-agnostic and capable of rendering any script without
a learned tokenizer.  HunyuanVideo 1.5 uses ByT5-Small with a custom
checkpoint that adds color/font special tokens for glyph control.

Architecture (ByT5-Small):
  - d_model: 1472
  - d_ff: 3584
  - num_heads: 6
  - d_kv: 64
  - encoder_layers: 12
  - vocab_size: 384 (259 bytes + special tokens + color/font tokens)
  - Activation: gated GELU

Output: ``(B, max_length, 1472)`` hidden states.

Weight format:
    encoder.embed_tokens.weight               (vocab, 1472)
    encoder.block.{0-11}.layer.0.SelfAttention.{q,k,v,o}.weight
    encoder.block.{0-11}.layer.0.layer_norm.weight
    encoder.block.{0-11}.layer.1.DenseReluDense.{wi_0,wi_1,wo}.weight
    encoder.block.{0-11}.layer.1.layer_norm.weight
    encoder.final_layer_norm.weight
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from torch import Tensor

from text.clip import TextOutput

__all__ = [
    "ByT5Encoder",
]

logger = logging.getLogger(__name__)


class ByT5Encoder:
    """Load and run ByT5 for HunyuanVideo 1.5 glyph encoding.

    ByT5-Small specifics:
      - d_model: 1472
      - d_ff: 3584
      - num_heads: 6
      - d_kv: 64 (1472 / 6 ≈ 245, but T5 uses d_kv independently)
      - num_layers: 12 (encoder only)
      - vocab_size: 384+ (259 bytes + learned special tokens)
      - Activation: gated GELU (tanh approximation)
      - Tokenizer: HuggingFace AutoTokenizer (byte-level)

    All transformers imports are lazy.
    """

    def __init__(
        self,
        dtype: Any = None,
        device: str = "cpu",
    ) -> None:
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._dtype = dtype
        self._device = device

    def load(self, model_path: str) -> None:
        """Load ByT5 encoder from a HuggingFace directory or safetensors.

        Parameters
        ----------
        model_path : str
            Path to ByT5 model directory (HuggingFace format) or a
            safetensors file with custom checkpoint weights.
        """
        from transformers import AutoTokenizer, T5Config, T5EncoderModel

        if self._dtype is None:
            self._dtype = torch.float16

        logger.info("Loading ByT5 encoder from %s", model_path)

        if os.path.isdir(model_path):
            # HuggingFace directory — load directly
            self._model = T5EncoderModel.from_pretrained(
                model_path,
                torch_dtype=self._dtype,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            # Safetensors file — build from config + load weights
            from safetensors.torch import load_file

            state_dict = load_file(model_path, device="cpu")
            embed_key = next(
                (k for k in ("encoder.embed_tokens.weight", "shared.weight") if k in state_dict),
                None,
            )
            if embed_key is None:
                raise ValueError(f"Cannot find embedding weights in {model_path}")

            vocab_size, d_model = state_dict[embed_key].shape

            # Count encoder layers
            num_layers = 0
            while f"encoder.block.{num_layers}.layer.0.SelfAttention.q.weight" in state_dict:
                num_layers += 1

            d_ff = state_dict["encoder.block.0.layer.1.DenseReluDense.wi_0.weight"].shape[0]

            # Detect d_kv from q weight shape: (num_heads * d_kv, d_model)
            q_shape = state_dict["encoder.block.0.layer.0.SelfAttention.q.weight"].shape[0]
            # ByT5-Small: num_heads=6, d_kv=64 → q shape = 384
            # Try common d_kv values
            for d_kv in (64, 128, 256):
                if q_shape % d_kv == 0:
                    num_heads = q_shape // d_kv
                    break
            else:
                d_kv = 64
                num_heads = q_shape // d_kv

            logger.info(
                "ByT5 config: d_model=%d, layers=%d, heads=%d, d_ff=%d, vocab=%d, d_kv=%d",
                d_model, num_layers, num_heads, d_ff, vocab_size, d_kv,
            )

            config = T5Config(
                vocab_size=vocab_size,
                d_model=d_model,
                d_ff=d_ff,
                d_kv=d_kv,
                num_heads=num_heads,
                num_layers=num_layers,
                num_decoder_layers=0,
                is_encoder_decoder=False,
                use_cache=False,
                dense_act_fn="gelu_pytorch_tanh",
                feed_forward_proj="gated-gelu",
                tie_word_embeddings=False,
            )

            model = T5EncoderModel(config)
            model.load_state_dict(state_dict, strict=False)
            self._model = model

            # Try to load tokenizer from same directory
            model_dir = os.path.dirname(model_path)
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
            except Exception:
                # Fall back to base ByT5 tokenizer (byte-level, always works)
                logger.info("Loading base ByT5 tokenizer from google/byt5-small")
                self._tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

        self._model = self._model.to(device=self._device, dtype=self._dtype)
        self._model.eval()
        logger.info(
            "ByT5 loaded on %s with dtype %s (vocab=%d)",
            self._device, self._dtype,
            self._model.config.vocab_size,
        )

    def unload(self) -> None:
        """Release model and tokenizer."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def encode(
        self,
        text: str,
        max_length: int = 256,
    ) -> tuple[TextOutput, Tensor | None]:
        """Tokenize and encode text, returning hidden states and attention mask.

        Parameters
        ----------
        text : str
            Input text (supports any language via byte-level tokenization).
        max_length : int
            Maximum token count (default 256 for HunyuanVideo 1.5).

        Returns
        -------
        tuple[TextOutput, Tensor | None]
            ``(output, attention_mask)`` where output contains hidden states
            ``(1, max_length, d_model)`` and attention_mask ``(1, max_length)``.
        """
        if not self.is_loaded:
            raise RuntimeError("ByT5 not loaded. Call load() first.")

        # Tokenize
        tokens = self._tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self._device)
        attention_mask = tokens["attention_mask"].to(self._device)

        # Forward pass
        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        hidden_states = outputs.last_hidden_state.to(dtype=self._dtype)

        # Zero out padding positions
        mask_expanded = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        hidden_states = hidden_states * mask_expanded

        return (
            TextOutput(hidden_states=hidden_states, pooled_output=None),
            attention_mask,
        )
