"""UMT5-XXL text encoder for Wan 2.1/2.2 — pure transformers, no diffusers.

UMT5-XXL is architecturally identical to T5-XXL encoder but with different
config (vocab=256384 vs 32128, different d_ff=10240 vs 16384, uses
gated GELU (tanh) activation, and uses a SentencePiece tokenizer instead
of a BPE tokenizer).

Output: (B, L, 4096) text embeddings.

Weight format (ComfyUI-style single safetensors):
    shared.weight / encoder.embed_tokens.weight   (256384, 4096)
    encoder.block.{0-23}.layer.0.SelfAttention.{q,k,v,o}.weight
    encoder.block.{0-23}.layer.0.SelfAttention.relative_attention_bias.weight  (block 0 only)
    encoder.block.{0-23}.layer.0.layer_norm.weight
    encoder.block.{0-23}.layer.1.DenseReluDense.{wi_0,wi_1,wo}.weight
    encoder.block.{0-23}.layer.1.layer_norm.weight
    encoder.final_layer_norm.weight
"""
from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from text.clip import TextOutput

__all__ = [
    "UMT5Encoder",
]

logger = logging.getLogger(__name__)


class UMT5Encoder:
    """Load and run UMT5-XXL for Wan text encoding.

    UMT5-XXL specifics:
      - d_model: 4096
      - d_ff: 10240
      - num_heads: 64
      - d_kv: 64
      - num_layers: 24
      - vocab_size: 256384
      - Activation: gated GELU (tanh approximation)
      - Tokenizer: SentencePiece

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

    def load(self, model_path: str, tokenizer_path: str | None = None) -> None:
        """Load UMT5-XXL from a safetensors file.

        Args:
            model_path: Path to .safetensors weights.
            tokenizer_path: Path to SentencePiece .model file. If None,
                looks for spiece.model in same directory, or tries to extract
                from the safetensors file itself (ComfyUI embeds it).
        """
        import torch
        from safetensors.torch import load_file
        from transformers import T5Config, T5EncoderModel

        if self._dtype is None:
            self._dtype = torch.float16

        logger.info("Loading UMT5-XXL from %s", model_path)
        state_dict = load_file(model_path, device="cpu")

        # Dequantize FP8 scaled weights: weight_real = fp8_weight * scale_weight
        # FP8 e4m3fn stores raw quantized values; scale factors are separate scalars.
        # Without this, weights are 3-4 orders of magnitude wrong!
        scale_keys = [k for k in state_dict if k.endswith(".scale_weight")]
        if scale_keys:
            logger.info("Dequantizing %d FP8 scaled weights", len(scale_keys))
            for scale_key in scale_keys:
                weight_key = scale_key.removesuffix(".scale_weight") + ".weight"
                if weight_key in state_dict and state_dict[weight_key].dtype == torch.float8_e4m3fn:
                    scale = state_dict[scale_key].float()
                    state_dict[weight_key] = state_dict[weight_key].float() * scale
                del state_dict[scale_key]

        # Auto-detect config from shapes
        embed_key = "shared.weight" if "shared.weight" in state_dict else "encoder.embed_tokens.weight"
        vocab_size, d_model = state_dict[embed_key].shape

        # Count layers
        num_layers = 0
        while f"encoder.block.{num_layers}.layer.0.SelfAttention.q.weight" in state_dict:
            num_layers += 1

        d_ff = state_dict["encoder.block.0.layer.1.DenseReluDense.wi_0.weight"].shape[0]
        d_kv = state_dict["encoder.block.0.layer.0.SelfAttention.q.weight"].shape[0] // (d_model // 64)
        num_heads = d_model // d_kv

        logger.info(
            "UMT5 config: d_model=%d, layers=%d, heads=%d, d_ff=%d, vocab=%d, d_kv=%d",
            d_model, num_layers, num_heads, d_ff, vocab_size, d_kv,
        )

        # Build T5Config for UMT5
        config = T5Config(
            vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            d_kv=d_kv,
            num_heads=num_heads,
            num_layers=num_layers,
            num_decoder_layers=0,
            relative_attention_num_buckets=32,
            relative_attention_max_distance=128,
            is_encoder_decoder=False,
            use_cache=False,
            dense_act_fn="gelu_pytorch_tanh",
            feed_forward_proj="gated-gelu",
            tie_word_embeddings=False,
        )

        model = T5EncoderModel(config)

        # UMT5 has per-block relative attention bias (unlike standard T5 which
        # shares block 0's bias). HuggingFace T5 only creates the bias embedding
        # for block 0, so we must manually add it to blocks 1+ before loading.
        has_per_block_bias = any(
            k.startswith("encoder.block.1.layer.0.SelfAttention.relative_attention_bias")
            for k in state_dict
        )
        if has_per_block_bias:
            logger.info("UMT5: enabling per-block relative attention bias for %d blocks", num_layers)
            for i in range(1, num_layers):
                attn = model.encoder.block[i].layer[0].SelfAttention
                attn.has_relative_attention_bias = True
                attn.relative_attention_num_buckets = 32
                attn.relative_attention_max_distance = 128
                attn.relative_attention_bias = nn.Embedding(32, num_heads)

        model.load_state_dict(state_dict, strict=False)

        # Monkey-patch T5Attention forward to force per-block bias computation.
        # HuggingFace T5 passes position_bias from block 0 to all subsequent
        # blocks, so they never compute their own. For UMT5, each block has
        # unique bias weights that must be used.
        if has_per_block_bias:
            def _make_bias_patch(orig_forward):
                def _patched(*args, **kwargs):
                    kwargs["position_bias"] = None
                    return orig_forward(*args, **kwargs)
                return _patched

            patched = 0
            for block in model.encoder.block:
                attn = block.layer[0].SelfAttention
                if attn.has_relative_attention_bias:
                    attn.forward = _make_bias_patch(attn.forward)
                    patched += 1
            logger.info("UMT5: patched %d/%d blocks for per-block attention bias", patched, num_layers)

        self._model = model.to(device=self._device, dtype=self._dtype)
        self._model.eval()
        logger.info("UMT5 model loaded on %s with dtype %s", self._device, self._dtype)

        # Load tokenizer
        self._load_tokenizer(model_path, tokenizer_path, state_dict)

    def _load_tokenizer(
        self,
        model_path: str,
        tokenizer_path: str | None,
        state_dict: dict[str, Tensor],
    ) -> None:
        """Load the SentencePiece tokenizer for UMT5."""
        import sentencepiece

        # Strategy 1: Explicit path
        if tokenizer_path and os.path.isfile(tokenizer_path):
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=tokenizer_path)
            logger.info("Loaded SentencePiece tokenizer from %s", tokenizer_path)
            return

        # Strategy 2: Look for spiece.model in same directory
        model_dir = os.path.dirname(model_path)
        for name in ("spiece.model", "tokenizer.model", "spiece_model.model"):
            candidate = os.path.join(model_dir, name)
            if os.path.isfile(candidate):
                self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=candidate)
                logger.info("Loaded SentencePiece tokenizer from %s", candidate)
                return

        # Strategy 3: Check for embedded tokenizer in state_dict (ComfyUI format)
        if "spiece_model" in state_dict:
            model_proto = state_dict["spiece_model"]
            if torch.is_tensor(model_proto):
                model_proto = model_proto.numpy().tobytes()
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=model_proto)
            logger.info("Loaded embedded SentencePiece tokenizer from state_dict")
            return

        # Strategy 4: Try to find it in known ComfyUI locations
        comfy_paths = [
            "/home/alex/SwarmUI/dlbackend/ComfyUI/comfy/text_encoders/",
        ]
        for base in comfy_paths:
            for name in ("spiece.model", "spiece_model.model"):
                candidate = os.path.join(base, name)
                if os.path.isfile(candidate):
                    self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=candidate)
                    logger.info("Loaded SentencePiece tokenizer from %s", candidate)
                    return

        raise FileNotFoundError(
            f"Could not find SentencePiece tokenizer. Tried:\n"
            f"  - Explicit path: {tokenizer_path}\n"
            f"  - Same directory as model: {model_dir}/spiece.model\n"
            f"  - Embedded in state_dict key 'spiece_model'\n"
            f"Please provide tokenizer_path explicitly."
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
        max_length: int = 512,
    ) -> tuple[TextOutput, Tensor | None]:
        """Tokenize and encode text, returning hidden states and attention mask.

        Args:
            text: Input text string.
            max_length: Maximum token length (UMT5 default is 512).

        Returns:
            Tuple of (TextOutput, attention_mask):
                - TextOutput with hidden_states shape (1, seq_len, 4096)
                - attention_mask shape (1, seq_len) or None
        """
        import torch

        if not self.is_loaded:
            raise RuntimeError("UMT5Encoder is not loaded. Call load() first.")

        # Tokenize
        token_ids = self._tokenizer.encode(text)
        # Add EOS token (id=1)
        token_ids = token_ids + [1]

        # Truncate if needed
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        # Pad to min_length=512 with pad_token=0
        original_len = len(token_ids)
        if len(token_ids) < max_length:
            token_ids = token_ids + [0] * (max_length - len(token_ids))

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self._device)
        attention_mask = torch.zeros(1, len(token_ids), dtype=torch.long, device=self._device)
        attention_mask[0, :original_len] = 1

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)

        # Zero out masked positions (ComfyUI does zero_out_masked=True)
        hidden = outputs.last_hidden_state
        hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)

        return TextOutput(hidden_states=hidden, pooled_output=None), attention_mask
