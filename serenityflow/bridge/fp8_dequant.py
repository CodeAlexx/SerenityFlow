"""Universal scaled FP8 dequantization for state dicts.

Handles both naming conventions:
  - LTX-style:  *.weight_scale  (scale for *.weight)
  - Flux-style:  *.scale_weight  (scale for *.weight)
  - Input scales: *.input_scale / *.scale_input  (dropped, not needed for inference)

Works on raw state dicts BEFORE any key renaming/conversion.
No model-specific logic — pure state dict operation.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "dequantize_fp8",
    "dequant_scaled_fp8",
    "dequant_scaled_fp8_in_model",
    "has_fp8_scales",
]

# FP8 dtypes
_FP8_DTYPES = set()
try:
    _FP8_DTYPES.add(torch.float8_e4m3fn)
    _FP8_DTYPES.add(torch.float8_e5m2)
except AttributeError:
    pass


def _is_fp8(tensor: torch.Tensor) -> bool:
    return tensor.dtype in _FP8_DTYPES


def _maybe_expand_scale_to_tensor_shape(scale: torch.Tensor, target_shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
    """Expand per-tensor / per-row / blockwise scales to match a weight tensor."""
    target_shape = tuple(target_shape)
    if scale.shape == target_shape or scale.numel() == 1:
        return scale

    # Common weight-only layouts: [out] -> [out, 1], [in] -> [1, in]
    if scale.dim() == 1 and len(target_shape) == 2:
        if scale.shape[0] == target_shape[0]:
            return scale.unsqueeze(-1)
        if scale.shape[0] == target_shape[1]:
            return scale.unsqueeze(0)

    expanded = scale
    while expanded.dim() < len(target_shape):
        expanded = expanded.unsqueeze(-1)

    if all(src == dst or src == 1 for src, dst in zip(expanded.shape, target_shape, strict=True)):
        return expanded

    for dim, (src, dst) in enumerate(zip(expanded.shape, target_shape, strict=True)):
        if src == dst:
            continue
        if src <= 0 or dst % src != 0:
            raise ValueError(f"Cannot expand FP8 scale shape {tuple(scale.shape)} to {target_shape}")
        expanded = expanded.repeat_interleave(dst // src, dim=dim)
    return expanded


def dequantize_fp8(
    fp8_weight: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an FP8 weight tensor following torchao's weight-only contract."""
    weight_f32 = fp8_weight.to(torch.float32)
    scale_f32 = _maybe_expand_scale_to_tensor_shape(scale, weight_f32.shape).to(
        device=weight_f32.device,
        dtype=torch.float32,
    )
    return (weight_f32 * scale_f32).to(out_dtype)


def _find_scale_pairs(keys: list[str]) -> dict[str, str]:
    """Map weight keys to their scale keys.

    Handles:
      *.weight_scale  → scale for *.weight  (LTX naming)
      *.scale_weight  → scale for *.weight  (Flux naming)
    """
    key_set = set(keys)
    pairs: dict[str, str] = {}  # weight_key → scale_key

    for key in keys:
        # LTX pattern: foo.weight_scale → foo.weight
        if key.endswith(".weight_scale"):
            weight_key = key.removesuffix("_scale")
            if weight_key in key_set:
                pairs[weight_key] = key

        # Flux pattern: foo.scale_weight → foo.weight
        elif key.endswith(".scale_weight"):
            weight_key = key.removesuffix(".scale_weight") + ".weight"
            if weight_key in key_set:
                pairs[weight_key] = key

    return pairs


def has_fp8_scales(state_dict: dict[str, Any]) -> bool:
    """Check if a state dict contains scaled FP8 weight/scale pairs."""
    keys = list(state_dict.keys())
    for key in keys:
        if key.endswith(".weight_scale") or key.endswith(".scale_weight"):
            return True
    return False


def dequant_scaled_fp8(state_dict: dict[str, Any], target_dtype: torch.dtype = torch.bfloat16) -> dict[str, Any]:
    """Dequant scaled FP8 weights in a state dict, in-place.

    For each weight+scale pair:
      dequanted = (fp8_weight.to(float32) * expanded_scale.to(float32)).to(target_dtype)

    Removes scale keys and input_scale/scale_input keys from the dict.
    Returns the same dict (modified in-place) for convenience.
    """
    keys = list(state_dict.keys())
    pairs = _find_scale_pairs(keys)

    if not pairs:
        return state_dict

    dequanted_count = 0
    for weight_key, scale_key in pairs.items():
        weight = state_dict[weight_key]
        scale = state_dict[scale_key]

        if _is_fp8(weight):
            state_dict[weight_key] = dequantize_fp8(weight, scale, out_dtype=target_dtype)
            dequanted_count += 1
        elif weight.dtype in (torch.float16, torch.bfloat16, torch.float32):
            # Already cast (e.g. by diffusers from_single_file) — but values are wrong.
            # Can't recover from a bad cast. Log a warning.
            logger.warning(
                "Scale key %s found but weight %s is already %s (not FP8). "
                "Values may be incorrect if loaded by a converter that cast without scaling.",
                scale_key, weight_key, weight.dtype,
            )

        # Remove the scale key
        del state_dict[scale_key]

    # Remove input_scale / scale_input keys (not needed for inference)
    drop_keys = [k for k in state_dict if k.endswith(".input_scale") or k.endswith(".scale_input")]
    for k in drop_keys:
        del state_dict[k]

    logger.info(
        "FP8 dequant: %d weights dequanted to %s, %d scale keys removed, %d input_scale keys removed",
        dequanted_count, target_dtype, len(pairs), len(drop_keys),
    )
    return state_dict


def dequant_scaled_fp8_in_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    prefix: str = "",
    target_dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Dequant scaled FP8 weights in a loaded model by re-reading the checkpoint.

    Use when the model was loaded by a converter (e.g. diffusers from_single_file)
    that cast FP8→bf16 without applying scales. Re-reads the original FP8 values
    and scales from the safetensors file and writes correct values in-place.

    Args:
        model: The loaded model with (incorrect) bf16 weights.
        checkpoint_path: Path to the original safetensors file.
        prefix: Key prefix to strip when matching model parameter names
                (e.g. "model.diffusion_model." for LTX).
        target_dtype: Target dtype for dequanted weights.

    Returns:
        Number of weights fixed.
    """
    from safetensors import safe_open

    # Build name→parameter map for the model.
    # Also build suffix map for fuzzy matching (handles wrapper modules like
    # velocity_model.X where checkpoint has just X).
    param_map: dict[str, torch.nn.Parameter] = {}
    suffix_map: dict[str, torch.nn.Parameter] = {}
    for name, param in model.named_parameters():
        param_map[name] = param
        # Store by suffix: if name is "velocity_model.blocks.0.weight",
        # also store under "blocks.0.weight"
        parts = name.split(".", 1)
        if len(parts) == 2:
            suffix_map[parts[1]] = param

    fixed = 0
    with safe_open(checkpoint_path, framework="pt") as f:
        all_keys = list(f.keys())
        pairs = _find_scale_pairs(all_keys)

        for weight_key, scale_key in pairs.items():
            # Strip prefix to get the module path
            if prefix and weight_key.startswith(prefix):
                param_name = weight_key[len(prefix):]
            else:
                param_name = weight_key

            param = param_map.get(param_name)
            if param is None:
                param = suffix_map.get(param_name)
            if param is None:
                continue

            fp8_weight = f.get_tensor(weight_key)
            scale = f.get_tensor(scale_key)

            if not _is_fp8(fp8_weight):
                continue

            dequanted = dequantize_fp8(fp8_weight, scale, out_dtype=target_dtype)
            param.data = dequanted.to(device=param.device)
            fixed += 1

    if fixed > 0:
        logger.info("Fixed %d FP8 weights in model from %s", fixed, checkpoint_path)

    return fixed
