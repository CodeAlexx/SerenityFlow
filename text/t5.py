"""T5 text encoder -- T5-XXL for Flux 1, Chroma, SD3, and Wan."""

from __future__ import annotations

import logging
from typing import Any

from text.clip import TextOutput

__all__ = [
    "T5Encoder",
]

logger = logging.getLogger(__name__)


class T5Encoder:
    """Load and run a T5 encoder model (typically T5-XXL).

    T5-XXL specifics:
      - Hidden dimension: 4096
      - Default max tokens: 512
      - No pooled output (encoder-only usage)

    Used by: Flux 1, Chroma, Wan, SD3.

    All ``transformers`` imports are lazy so the module can be imported
    without the library installed.
    """

    def __init__(
        self,
        model_path: str | None = None,
        dtype: Any = None,
        device: str = "cpu",
    ) -> None:
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._dtype = dtype
        self._device = device
        if model_path is not None:
            self.load(model_path)

    # -- lifecycle -----------------------------------------------------------

    def load(self, model_path: str) -> None:
        """Load a T5 encoder model and tokenizer from *model_path*.

        Accepts either:
        - A single ``.safetensors`` file (with companion ``.tokenizer.json``)
        - A HuggingFace repo ID or local directory with config + tokenizer
        """
        import torch

        if self._dtype is None:
            self._dtype = torch.float16

        if model_path.endswith(".safetensors"):
            self._load_from_safetensors(model_path)
            return

        try:
            from transformers import AutoTokenizer, T5EncoderModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "transformers is required to load T5 models. "
                "Install it with: pip install transformers"
            ) from exc

        logger.info("Loading T5 encoder from %s", model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = T5EncoderModel.from_pretrained(
            model_path,
            torch_dtype=self._dtype,
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
            PreTrainedTokenizerFast,
            T5Config,
            T5EncoderModel,
        )

        logger.info("Loading T5 from safetensors: %s", safetensors_path)
        state_dict = load_file(safetensors_path, device="cpu")

        # Detect architecture from shapes
        embed_weight = state_dict.get("shared.weight")
        if embed_weight is None:
            embed_weight = state_dict["encoder.embed_tokens.weight"]
        vocab_size, d_model = embed_weight.shape

        # Count layers
        num_layers = 0
        while f"encoder.block.{num_layers}.layer.0.SelfAttention.q.weight" in state_dict:
            num_layers += 1

        # Infer FFN dim from wi_0
        d_ff = state_dict["encoder.block.0.layer.1.DenseReluDense.wi_0.weight"].shape[0]

        # Infer num_heads (head_dim=64 for T5-XXL: 4096/64=64 heads)
        num_heads = d_model // 64

        # Check relative attention bias
        rel_attn_key = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
        rel_attn_buckets = state_dict[rel_attn_key].shape[0] if rel_attn_key in state_dict else 32

        logger.info(
            "T5 config: d_model=%d, layers=%d, heads=%d, d_ff=%d",
            d_model, num_layers, num_heads, d_ff,
        )

        config = T5Config(
            vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            num_layers=num_layers,
            num_decoder_layers=0,
            relative_attention_num_buckets=rel_attn_buckets,
            relative_attention_max_distance=128,
            is_encoder_decoder=False,
            use_cache=False,
            dense_act_fn="gelu_new",
            feed_forward_proj="gated-gelu",
        )

        model = T5EncoderModel(config)
        # shared.weight and encoder.embed_tokens.weight may both exist
        # T5EncoderModel ties them; load with strict=False for flexibility
        model.load_state_dict(state_dict, strict=False)
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
        self._tokenizer.eos_token_id = 1
        self._tokenizer.pad_token_id = 0
        self._tokenizer.model_max_length = 512
        logger.info("Loaded T5 tokenizer from %s", tokenizer_path)

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
        max_length: int = 512,
    ) -> TextOutput:
        """Tokenize and encode *text*, applying prompt weights to embeddings.

        Parses ``(word:1.5)`` syntax and scales per-token hidden states
        accordingly.  When all weights are 1.0 the fast path is used.

        Args:
            text: Input prompt string (may contain weight syntax).
            max_length: Maximum token length (T5-XXL default is 512).

        Returns:
            :class:`TextOutput` with hidden states. ``pooled_output`` is
            always ``None`` for T5 (encoder-only, no pooling head).
        """
        import torch

        from text.tokenizer import (
            has_non_default_weights,
            parse_prompt_weights,
            split_segments_at_break,
        )

        if not self.is_loaded:
            raise RuntimeError("T5Encoder is not loaded. Call load() first.")

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

    def _encode_unweighted(self, text: str, max_length: int) -> TextOutput:
        """Fast-path encoding with no weight application."""
        import torch

        tokens = self._tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        # T5 doesn't accept token_type_ids
        tokens.pop("token_type_ids", None)

        with torch.no_grad():
            outputs = self._model(**tokens)

        return TextOutput(
            hidden_states=outputs.last_hidden_state,
            pooled_output=None,
        )

    def _encode_weighted_group(
        self,
        group: list[tuple[str, float]],
        max_length: int,
    ) -> Any:
        """Encode a single BREAK-group with per-token weight scaling."""
        import torch

        from text.tokenizer import build_token_weight_map

        eos_id = self._tokenizer.eos_token_id
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0  # T5 typically uses 0 as pad

        def _tokenize_bare(text: str) -> list[int]:
            return self._tokenizer.encode(text, add_special_tokens=False)

        token_ids, weights = build_token_weight_map(
            group,
            tokenize_fn=_tokenize_bare,
            bos_token_id=None,  # T5 has no BOS
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
            )

        hidden = outputs.last_hidden_state

        weight_tensor = torch.tensor(
            weights, dtype=hidden.dtype, device=hidden.device,
        ).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        hidden = hidden * weight_tensor

        return hidden
