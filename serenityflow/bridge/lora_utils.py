"""Standalone LoRA loading and merging for SerenityFlow.

Simple weight-space LoRA merge — loads LoRA state dict and applies
delta weights directly to model parameters.

No runtime imports from serenity, comfy, or diffusers.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["load_lora", "merge_lora_into_model"]


def load_lora(path: str) -> dict:
    """Load LoRA state dict from safetensors or torch checkpoint."""
    from serenityflow.bridge.model_utils import load_state_dict
    return load_state_dict(path)


def _find_lora_pairs(lora_sd: dict) -> dict[str, dict[str, torch.Tensor]]:
    """Group LoRA state dict into {base_key: {"down": tensor, "up": tensor, ...}} pairs.

    Handles multiple naming conventions:
    - Diffusers: ``*.lora_A.weight`` / ``*.lora_B.weight``
    - Kohya/ComfyUI: ``*.lora_down.weight`` / ``*.lora_up.weight``
    - With optional ``lora_te_`` / ``lora_unet_`` prefixes
    """
    pairs: dict[str, dict[str, torch.Tensor]] = {}

    for key, tensor in lora_sd.items():
        # Skip non-weight keys (alpha, metadata, etc.)
        if key.endswith(".alpha"):
            # Extract base key and store alpha
            base = key.rsplit(".alpha", 1)[0]
            pairs.setdefault(base, {})["alpha"] = tensor
            continue

        # Match down/A weight
        down_match = re.match(r"(.+?)\.(lora_down|lora_A)\.weight$", key)
        if down_match:
            base = down_match.group(1)
            pairs.setdefault(base, {})["down"] = tensor
            continue

        # Match up/B weight
        up_match = re.match(r"(.+?)\.(lora_up|lora_B)\.weight$", key)
        if up_match:
            base = up_match.group(1)
            pairs.setdefault(base, {})["up"] = tensor
            continue

    return pairs


def _resolve_target_key(base_key: str, model_keys: set[str]) -> str | None:
    """Resolve a LoRA base key to the actual model parameter key.

    Strips common prefixes and tries to find a matching ``.weight`` parameter.
    """
    # Try direct match
    target = f"{base_key}.weight"
    if target in model_keys:
        return target

    # Strip common LoRA prefixes
    for prefix in ("lora_unet_", "lora_te_", "lora_te1_", "lora_te2_",
                    "transformer.", "unet.", "text_encoder."):
        if base_key.startswith(prefix):
            stripped = base_key[len(prefix):]
            target = f"{stripped}.weight"
            if target in model_keys:
                return target

    # Try replacing dots with underscores and vice versa (Kohya format)
    alt = base_key.replace("_", ".")
    target = f"{alt}.weight"
    if target in model_keys:
        return target

    return None


def merge_lora_into_model(
    model: nn.Module,
    lora_sd: dict,
    strength: float = 1.0,
) -> None:
    """Merge LoRA weights into model parameters in-place.

    Computes ``weight += strength * (up @ down) * scale`` for each LoRA pair
    and adds the delta directly to the base model weight.
    """
    if strength == 0.0:
        return

    pairs = _find_lora_pairs(lora_sd)
    if not pairs:
        logger.warning("No LoRA pairs found in state dict (%d keys)", len(lora_sd))
        return

    model_sd = {k: v for k, v in model.named_parameters()}
    model_keys = set(model_sd.keys())
    applied = 0
    skipped = 0

    for base_key, components in pairs.items():
        down = components.get("down")
        up = components.get("up")
        if down is None or up is None:
            continue

        target_key = _resolve_target_key(base_key, model_keys)
        if target_key is None:
            skipped += 1
            continue

        param = model_sd[target_key]

        # Compute scale from alpha and rank
        alpha = components.get("alpha")
        rank = down.shape[0]
        if alpha is not None:
            scale = float(alpha) / rank
        else:
            scale = 1.0

        # Compute delta: up @ down for linear, with reshape for conv
        with torch.no_grad():
            up_f = up.to(dtype=param.dtype, device=param.device)
            down_f = down.to(dtype=param.dtype, device=param.device)

            if up_f.ndim == 2 and down_f.ndim == 2:
                # Linear layer: delta = up @ down
                delta = up_f @ down_f
            elif up_f.ndim == 4 and down_f.ndim == 4:
                # Conv2d layer: use grouped conv or reshape
                delta = torch.nn.functional.conv2d(
                    down_f.permute(1, 0, 2, 3), up_f
                ).permute(1, 0, 2, 3)
            else:
                # Mixed dims — try matmul on 2D views
                delta = (up_f.squeeze() @ down_f.squeeze()).view(param.shape)

            param.data.add_(delta * (strength * scale))
            applied += 1

    logger.info("LoRA merge: %d layers applied, %d skipped (strength=%.2f)", applied, skipped, strength)
