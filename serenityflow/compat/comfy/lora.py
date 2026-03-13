"""Compatibility shim for comfy.lora.

LoRA weight calculation and loading.
"""
from __future__ import annotations

import torch

import comfy.utils


def load_lora(lora_path, to_load):
    """Load LoRA weights from file and match to model keys."""
    sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
    return sd


def model_lora_keys_unet(model, key_map=None):
    if key_map is None:
        key_map = {}
    if hasattr(model, "state_dict"):
        sd = model.state_dict()
        for k in sd:
            key_map[k] = k
    return key_map


def model_lora_keys_clip(model, key_map=None):
    if key_map is None:
        key_map = {}
    return key_map


def calculate_weight(patches, weight, key, intermediate_dtype=None):
    """Apply patch list to a weight tensor. Core LoRA math."""
    for p in patches:
        strength = p[0]
        v = p[1]
        strength_model = p[2] if len(p) > 2 else 1.0

        if isinstance(v, torch.Tensor):
            weight = weight + strength * strength_model * v.to(weight.device, dtype=weight.dtype)
        elif isinstance(v, (list, tuple)):
            if len(v) == 1:
                # Full diff
                diff = v[0].to(weight.device, dtype=weight.dtype)
                weight = weight + strength * strength_model * diff
            elif len(v) == 2:
                # LoRA
                up = v[0].to(weight.device, dtype=weight.dtype)
                down = v[1].to(weight.device, dtype=weight.dtype)
                if len(weight.shape) == 4:
                    if len(down.shape) == 4:
                        delta = torch.nn.functional.conv2d(
                            down.permute(1, 0, 2, 3), up,
                        ).permute(1, 0, 2, 3)
                    else:
                        delta = (up @ down).reshape(weight.shape)
                else:
                    delta = up @ down
                weight = weight + strength * strength_model * delta
            elif len(v) >= 3:
                # LoRA with alpha
                up = v[0].to(weight.device, dtype=weight.dtype)
                down = v[1].to(weight.device, dtype=weight.dtype)
                alpha = v[2]
                if alpha is not None:
                    if isinstance(alpha, torch.Tensor):
                        alpha = alpha.item()
                    rank = down.shape[0]
                    scale = alpha / rank
                else:
                    scale = 1.0
                if len(weight.shape) == 4:
                    if len(down.shape) == 4:
                        delta = torch.nn.functional.conv2d(
                            down.permute(1, 0, 2, 3), up,
                        ).permute(1, 0, 2, 3)
                    else:
                        delta = (up @ down).reshape(weight.shape)
                else:
                    delta = up @ down
                weight = weight + strength * strength_model * delta * scale

    return weight
