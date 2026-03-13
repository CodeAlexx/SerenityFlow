from __future__ import annotations

"""
Conditioning is list[tuple[Tensor, dict]].
The tensor is the text embedding. The dict carries metadata.
This module validates, normalizes, and merges conditioning.
"""
import torch
from typing import Any

OPTIONAL_KEYS = {
    "pooled_output":          torch.Tensor,
    "area":                   tuple,
    "strength":               (int, float),
    "set_area_to_bounds":     bool,
    "mask":                   torch.Tensor,
    "mask_strength":          (int, float),
    "timestep_start":         (int, float),
    "timestep_end":           (int, float),
    "concat_latent_image":    torch.Tensor,
    "concat_mask":            torch.Tensor,
    "control":                object,  # ControlNetHandle — loose type
    "cross_attn_controlnet":  torch.Tensor,
    "noise_augmentation":     (int, float),
    "gligen":                 tuple,
    "hooks":                  object,
}

def validate(cond: Any) -> list[str]:
    """Returns list of errors/warnings. Empty = valid."""
    errors = []
    if not isinstance(cond, list):
        return [f"CONDITIONING must be list, got {type(cond).__name__}"]
    for i, entry in enumerate(cond):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            errors.append(f"Entry {i}: must be (Tensor, dict), got {type(entry).__name__} len={len(entry) if hasattr(entry, '__len__') else '?'}")
            continue
        tensor, meta = entry
        if not isinstance(tensor, torch.Tensor):
            errors.append(f"Entry {i}: first element must be Tensor, got {type(tensor).__name__}")
        if not isinstance(meta, dict):
            errors.append(f"Entry {i}: second element must be dict, got {type(meta).__name__}")
            continue
        for key, value in meta.items():
            expected = OPTIONAL_KEYS.get(key)
            if expected is None:
                errors.append(f"Entry {i}: unknown key '{key}' (warning)")
                continue
            if expected is object:
                continue  # Accept anything
            if not isinstance(value, expected):
                errors.append(f"Entry {i}: key '{key}' expected {expected}, got {type(value).__name__}")
    return errors

def normalize(cond: list) -> list:
    """Strip unknown keys, enforce types. Returns new list."""
    result = []
    for tensor, meta in cond:
        clean = {}
        for key, value in meta.items():
            if key in OPTIONAL_KEYS:
                clean[key] = value
        result.append((tensor, clean))
    return result

def merge(a: list, b: list) -> list:
    """Concatenate two conditioning lists."""
    return a + b

def set_area(cond: list, area: tuple, strength: float = 1.0) -> list:
    """Return new conditioning with area applied to all entries."""
    result = []
    for tensor, meta in cond:
        new_meta = dict(meta)
        new_meta["area"] = area
        new_meta["strength"] = strength
        result.append((tensor, new_meta))
    return result

def set_mask(cond: list, mask: torch.Tensor, strength: float = 1.0) -> list:
    result = []
    for tensor, meta in cond:
        new_meta = dict(meta)
        new_meta["mask"] = mask
        new_meta["mask_strength"] = strength
        result.append((tensor, new_meta))
    return result

def set_timestep_range(cond: list, start: float, end: float) -> list:
    result = []
    for tensor, meta in cond:
        new_meta = dict(meta)
        new_meta["timestep_start"] = start
        new_meta["timestep_end"] = end
        result.append((tensor, new_meta))
    return result

def zero_out(cond: list) -> list:
    result = []
    for tensor, meta in cond:
        result.append((torch.zeros_like(tensor), dict(meta)))
    return result
