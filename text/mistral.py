"""Mistral text encoder for Flux 2 Dev models.

Extracts hidden states at layers [10, 20, 30] and stacks them along
the hidden dimension to produce the joint_attention_dim embedding:

- Mistral3 24B (Small): hidden_size=5120, joint_dim = 3 * 5120 = 15360

Uses Mistral chat template with system prompt for structured image
description reasoning.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from text.clip import TextOutput

__all__ = [
    "MistralEncoder",
]

logger = logging.getLogger(__name__)

# Default extraction layers for Flux 2 Dev (Mistral3 24B)
_DEFAULT_EXTRACT_LAYERS: tuple[int, int, int] = (10, 20, 30)

# Mistral chat template used by Flux 2 Dev
_MISTRAL_CHAT_TEMPLATE = (
    "[SYSTEM_PROMPT]You are an AI that reasons about image descriptions. "
    "You give structured responses focusing on object relationships, object\n"
    "attribution and actions without speculation.[/SYSTEM_PROMPT]"
    "[INST]{}[/INST]"
)


def _truthy_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


class MistralEncoder:
    """Mistral text encoder for Flux 2 Dev models.

    Loads a Mistral3 24B (Small) causal LM and extracts hidden states
    from specific intermediate layers, stacking them to form the text
    conditioning tensor expected by the Flux 2 Dev transformer.

    The stacking logic matches the Klein/Qwen3 pattern:

    1. Extract hidden states from layers [10, 20, 30]
    2. Stack along a new dimension
    3. Permute and reshape to concatenate along hidden dim
    4. Output shape: ``(B, seq, 3 * 5120)`` = ``(B, seq, 15360)``

    Args:
        model_path: HuggingFace repo or local path. Loaded eagerly if given.
        extract_layers: Tuple of layer indices to extract and stack.
        dtype: Torch dtype for model weights.
        device: Target device string.
    """

    def __init__(
        self,
        model_path: str | None = None,
        extract_layers: tuple[int, ...] = _DEFAULT_EXTRACT_LAYERS,
        dtype: Any = None,
        device: str = "cpu",
        load_in_8bit: bool | None = None,
    ) -> None:
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._dtype = dtype
        self._device = device
        self._extract_layers = extract_layers
        self._load_in_8bit = (
            _truthy_env("SERENITY_MISTRAL_8BIT", default=False)
            if load_in_8bit is None
            else bool(load_in_8bit)
        )
        default_max_length = 256 if self._load_in_8bit else 512
        self._default_max_length = max(
            32,
            _int_env("SERENITY_MISTRAL_MAX_LENGTH", default_max_length),
        )
        if model_path is not None:
            self.load(model_path)

    # -- lifecycle -----------------------------------------------------------

    def load(self, model_path: str) -> None:
        """Load a Mistral3 causal LM and tokenizer from *model_path*."""
        try:
            from transformers import (  # type: ignore[import-untyped]
                AutoModel,
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers is required to load Mistral models. "
                "Install it with: pip install transformers"
            ) from exc

        import torch

        if self._dtype is None:
            self._dtype = torch.float16

        model_dir = Path(model_path).expanduser()
        tokenizer_dir = model_dir
        if not (tokenizer_dir / "tokenizer.json").exists():
            sibling = model_dir.parent / "tokenizer"
            if (sibling / "tokenizer.json").exists():
                tokenizer_dir = sibling

        logger.info("Loading Mistral encoder from %s (tokenizer=%s)", model_dir, tokenizer_dir)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_dir),
                trust_remote_code=True,
                use_fast=True,
            )
        except Exception:
            # FLUX.2-dev tokenizer is Llama-compatible and can be loaded directly.
            from transformers import LlamaTokenizerFast  # type: ignore[import-untyped]

            self._tokenizer = LlamaTokenizerFast.from_pretrained(str(tokenizer_dir))

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        quantized = False
        if self._load_in_8bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            load_kwargs["device_map"] = "auto"
            if torch.cuda.is_available():
                # Keep headroom on 24 GB cards to avoid int8 dequant OOM spikes.
                gpu_budget_gb = max(8, _int_env("SERENITY_MISTRAL_8BIT_GPU_GB", 18))
                cpu_budget_gb = max(16, _int_env("SERENITY_MISTRAL_8BIT_CPU_GB", 96))
                load_kwargs["max_memory"] = {
                    0: f"{gpu_budget_gb}GiB",
                    "cpu": f"{cpu_budget_gb}GiB",
                }
            quantized = True
            logger.info("Loading Mistral encoder in 8-bit mode (bitsandbytes).")
        else:
            load_kwargs["torch_dtype"] = self._dtype

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                **load_kwargs,
            )
        except ValueError as exc:
            # Some installs expose FLUX.2's text encoder as Mistral3Config
            # under AutoModel (not AutoModelForCausalLM). Fall back cleanly.
            if "Unrecognized configuration class" not in str(exc):
                raise
            logger.warning(
                "AutoModelForCausalLM does not support this Mistral config; "
                "falling back to AutoModel."
            )
            self._model = AutoModel.from_pretrained(
                str(model_dir),
                **load_kwargs,
            )
        if not quantized:
            self._model = self._model.to(self._device)
        self._model.eval()

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
        """Apply Mistral chat template for Flux 2 Dev.

        Uses the hardcoded Mistral template:
        ``[SYSTEM_PROMPT]You are a helpful assistant.[/SYSTEM_PROMPT][INST]{prompt}[/INST]``

        This matches the template used by the official Flux 2 Dev pipeline.
        """
        return _MISTRAL_CHAT_TEMPLATE.format(text)

    # -- encoding ------------------------------------------------------------

    def encode(
        self,
        text: str,
        max_length: int | None = None,
    ) -> TextOutput:
        """Tokenize and encode *text*, extracting stacked hidden states.

        Applies prompt weight syntax ``(word:1.5)`` when non-default
        weights are detected.

        Args:
            text: Input prompt string (may contain weight syntax).
            max_length: Maximum token length (default 512).

        Returns:
            :class:`TextOutput` with stacked hidden states of shape
            ``(B, seq, N * hidden_size)`` where N = len(extract_layers).
            For Mistral3 24B: ``(B, seq, 15360)``.
            ``pooled_output`` is always ``None``.
        """
        import torch

        from text.tokenizer import (
            has_non_default_weights,
            parse_prompt_weights,
            split_segments_at_break,
        )

        if not self.is_loaded:
            raise RuntimeError("MistralEncoder is not loaded. Call load() first.")
        if max_length is None:
            max_length = self._default_max_length

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

    def _extract_and_stack(self, hidden_states: tuple[Any, ...]) -> Any:
        """Extract layers and concatenate along hidden dimension.

        Given the full tuple of hidden states from all layers, extract
        the states at ``self._extract_layers`` and concatenate them
        along the last dimension.

        Input per layer: ``(B, seq, hidden_size)``
        Output: ``(B, seq, len(extract_layers) * hidden_size)``

        This uses the same stack-movedim-reshape pattern as Klein/Qwen3.
        """
        import torch

        selected = [hidden_states[layer_idx] for layer_idx in self._extract_layers]

        # Stack: [B, N, seq, D] then reshape to [B, seq, N*D]
        stacked = torch.stack(selected, dim=1)
        batch_size, num_layers, seq_len, hidden_dim = stacked.shape
        return stacked.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, num_layers * hidden_dim,
        )

    def _encode_unweighted(self, text: str, max_length: int) -> TextOutput:
        """Fast-path encoding with no weight application."""
        import torch

        formatted = self._format_prompt(text)
        input_device = self._inference_device()

        tokens = self._tokenizer(
            formatted,
            max_length=max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        ).to(input_device)

        with torch.no_grad():
            outputs = self._model(
                **tokens,
                output_hidden_states=True,
            )

        hidden = self._extract_and_stack(outputs.hidden_states)

        return TextOutput(hidden_states=hidden, pooled_output=None)

    def _inference_device(self) -> str:
        if self._model is None:
            return self._device
        device_map = getattr(self._model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for target in device_map.values():
                if isinstance(target, str) and target.startswith("cuda"):
                    return target
                if isinstance(target, int):
                    return f"cuda:{target}"
        try:
            return str(next(self._model.parameters()).device)
        except Exception:
            return self._device

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

        input_device = self._inference_device()
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=input_device)
        attention_mask = torch.tensor(
            [[1 if t != pad_id else 0 for t in token_ids]],
            dtype=torch.long,
            device=input_device,
        )

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden = self._extract_and_stack(outputs.hidden_states)

        # Apply per-token weights
        weight_tensor = torch.tensor(
            weights, dtype=hidden.dtype, device=hidden.device,
        ).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        hidden = hidden * weight_tensor

        return hidden
