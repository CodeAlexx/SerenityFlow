from __future__ import annotations

"""
Wire types that flow between nodes. Model handles that wrap runtime state.
This module defines the vocabulary of the entire engine.
"""
import uuid
import copy
import torch
from dataclasses import dataclass, replace, field
from typing import Any, Optional

# ─── Wire Type Registry ───

WIRE_TYPES = {
    "MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT",
    "IMAGE", "MASK", "CONTROL_NET", "CLIP_VISION", "CLIP_VISION_OUTPUT",
    "STYLE_MODEL", "GLIGEN", "UPSCALE_MODEL", "AUDIO",
    "STRING", "INT", "FLOAT", "BOOLEAN", "COMBO",
}

# ─── Validation ───

def validate_image(t: torch.Tensor) -> list[str]:
    """Validate IMAGE wire type. Returns list of errors. Empty = valid."""
    errors = []
    if not isinstance(t, torch.Tensor):
        return [f"IMAGE must be Tensor, got {type(t).__name__}"]
    if t.ndim != 4:
        errors.append(f"IMAGE must be 4D (BHWC), got {t.ndim}D shape {tuple(t.shape)}")
    elif t.shape[-1] not in (1, 3, 4):
        errors.append(f"IMAGE channels (dim -1) must be 1/3/4, got {t.shape[-1]}")
    if t.dtype != torch.float32:
        errors.append(f"IMAGE must be float32, got {t.dtype}")
    if t.numel() > 0:
        if t.min() < -0.01 or t.max() > 1.01:
            errors.append(f"IMAGE range must be [0,1], got [{t.min():.3f}, {t.max():.3f}]")
    return errors

def validate_mask(t: torch.Tensor) -> list[str]:
    errors = []
    if not isinstance(t, torch.Tensor):
        return [f"MASK must be Tensor, got {type(t).__name__}"]
    if t.ndim not in (2, 3):
        errors.append(f"MASK must be 2D (HW) or 3D (BHW), got {t.ndim}D")
    if t.dtype != torch.float32:
        errors.append(f"MASK must be float32, got {t.dtype}")
    if t.numel() > 0:
        if t.min() < -0.01 or t.max() > 1.01:
            errors.append(f"MASK range must be [0,1], got [{t.min():.3f}, {t.max():.3f}]")
    return errors

def validate_latent(d: dict) -> list[str]:
    errors = []
    if not isinstance(d, dict):
        return [f"LATENT must be dict, got {type(d).__name__}"]
    if "samples" not in d:
        errors.append("LATENT must contain 'samples' key")
    elif not isinstance(d["samples"], torch.Tensor):
        errors.append(f"LATENT['samples'] must be Tensor, got {type(d['samples']).__name__}")
    elif d["samples"].ndim != 4:
        errors.append(f"LATENT['samples'] must be 4D (BCHW), got {d['samples'].ndim}D")
    return errors

def validate_conditioning(cond: list) -> list[str]:
    """Defer to conditioning.py for full validation."""
    from serenityflow.core.conditioning import validate as cond_validate
    return cond_validate(cond)

VALIDATORS = {
    "IMAGE": validate_image,
    "MASK": validate_mask,
    "LATENT": validate_latent,
    "CONDITIONING": validate_conditioning,
}

def validate_wire(type_name: str, value: Any) -> list[str]:
    """Validate a value against its declared wire type. Returns errors."""
    validator = VALIDATORS.get(type_name)
    if validator is None:
        return []  # No validator = always valid (opaque types, scalars)
    return validator(value)

# ─── Model Handles ───

@dataclass(frozen=True)
class ModelHandle:
    handle_id: str
    arch: str
    config: dict
    path: str
    dtype: torch.dtype
    patch_ledger: Any  # PatchLedger — typed as Any to avoid circular import
    stagehand_ref: Any = None
    model_options: dict = field(default_factory=dict)

    def cache_key(self) -> str:
        fp = self.patch_ledger.fingerprint() if self.patch_ledger else "none"
        return f"{self.handle_id}:{fp}"

    def with_patches(self, ledger) -> ModelHandle:
        return replace(self, handle_id=uuid.uuid4().hex, patch_ledger=ledger)

    def with_options(self, opts: dict) -> ModelHandle:
        merged = _deep_merge(self.model_options, opts)
        return replace(self, handle_id=uuid.uuid4().hex, model_options=merged)

@dataclass(frozen=True)
class ClipHandle:
    handle_id: str
    path: str
    clip_type: str = ""
    stagehand_ref: Any = None

    def cache_key(self) -> str:
        return self.handle_id

@dataclass(frozen=True)
class VaeHandle:
    handle_id: str
    path: str
    stagehand_ref: Any = None

    def cache_key(self) -> str:
        return self.handle_id

@dataclass(frozen=True)
class ControlNetHandle:
    handle_id: str
    path: str
    stagehand_ref: Any = None

    def cache_key(self) -> str:
        return self.handle_id

# ─── Utility ───

def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result
