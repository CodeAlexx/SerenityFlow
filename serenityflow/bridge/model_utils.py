"""Standalone model loading utilities for SerenityFlow.

Provides model architecture detection, state dict loading, VAE key
conversion, and thin VAE wrapper classes. No runtime imports from
serenity, comfy, or diffusers.

VAE key conversion derived from:
  - ComfyUI (MIT License): comfy/diffusers_convert.py
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = [
    "ModelArchitecture",
    "ModelConfig",
    "detect_from_file",
    "load_state_dict",
    "extract_vae_state_dict",
    "safe_load_state_dict",
    "convert_ldm_vae_to_diffusers",
    "VAEDecoder",
    "VAEEncoder",
    "get_vae_scaling_factor",
    "get_prediction_type",
    "get_prediction_kwargs",
    "find_qwen_cache_snapshot",
]


# --------------------------------------------------------------------------- #
# ModelArchitecture enum — values match serenity-inference strings so
# comparisons like ``si_config.architecture == ModelArchitecture.ZIMAGE``
# work across enum boundaries (both are ``str, Enum``).
# --------------------------------------------------------------------------- #


class ModelArchitecture(str, Enum):
    SD15 = "sd15"
    SD20 = "sd20"
    SD21 = "sd21"
    SDXL = "sdxl"
    SDXL_REFINER = "sdxl_refiner"
    SD3 = "sd3"
    FLUX_DEV = "flux_dev"
    FLUX_FILL = "flux_fill"
    FLUX_SCHNELL = "flux_schnell"
    FLUX_2_DEV = "flux_2_dev"
    FLUX_2_KLEIN_4B = "flux_2_klein_4b"
    FLUX_2_KLEIN_9B = "flux_2_klein_9b"
    LTXV = "ltxv"
    WAN = "wan"
    ZIMAGE = "zimage"
    QWEN = "qwen"
    LUMINA = "lumina"
    HUNYUAN_VIDEO = "hunyuan_video"
    CHROMA = "chroma"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# ModelConfig
# --------------------------------------------------------------------------- #


@dataclass
class ModelConfig:
    """Lightweight model configuration from detection."""

    architecture: ModelArchitecture
    config: dict = field(default_factory=dict)
    unet_config: dict = field(default_factory=dict)
    confidence: float = 1.0
    explanation: str = ""


# --------------------------------------------------------------------------- #
# Architecture detection — wraps SF's own model_detect.py
# --------------------------------------------------------------------------- #

# Map model_detect pipeline names → enum values
_PIPELINE_TO_ARCH = {
    "sd15": ModelArchitecture.SD15,
    "sdxl": ModelArchitecture.SDXL,
    "sd3": ModelArchitecture.SD3,
    "flux": ModelArchitecture.FLUX_DEV,
    "ltxv": ModelArchitecture.LTXV,
    "wan": ModelArchitecture.WAN,
    "klein": ModelArchitecture.FLUX_2_KLEIN_4B,
    "zimage": ModelArchitecture.ZIMAGE,
    "qwen": ModelArchitecture.QWEN,
    "hunyuan_video": ModelArchitecture.HUNYUAN_VIDEO,
}


def detect_from_file(path: str) -> Optional[ModelConfig]:
    """Detect model architecture from checkpoint file.

    Uses SF's own ``model_detect.py`` internally.
    Returns :class:`ModelConfig` or ``None`` if detection fails.
    """
    from serenityflow.bridge.model_detect import detect_architecture

    try:
        pipeline_name, config_dict, confidence, explanation = detect_architecture(path)
        arch = _PIPELINE_TO_ARCH.get(pipeline_name, ModelArchitecture.UNKNOWN)
        return ModelConfig(
            architecture=arch,
            config=config_dict,
            unet_config=config_dict,
            confidence=confidence,
            explanation=explanation,
        )
    except Exception:
        logger.debug("detect_from_file failed for %s", path, exc_info=True)
        return None


# --------------------------------------------------------------------------- #
# State dict loading
# --------------------------------------------------------------------------- #


def load_state_dict(path: str) -> dict:
    """Load state dict from safetensors, GGUF, or torch checkpoint.

    Returns raw state dict with no key conversion.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".safetensors":
        from safetensors.torch import load_file
        return load_file(path, device="cpu")

    if ext == ".gguf":
        try:
            from serenity_safetensors import load_gguf_state_dict
            return load_gguf_state_dict(path)
        except ImportError:
            raise RuntimeError(
                f"GGUF loading requires serenity-safetensors package: {path}"
            )

    # .pt, .ckpt, .bin — torch checkpoint
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict):
        if "state_dict" in sd:
            sd = sd["state_dict"]
        elif "model" in sd:
            sd = sd["model"]
    return sd


def extract_vae_state_dict(sd: dict) -> dict:
    """Extract VAE sub-dict from a combined checkpoint state dict.

    Handles prefixes: ``first_stage_model.``, ``vae.``
    Returns empty dict if no VAE keys found.
    """
    for prefix in ("first_stage_model.", "vae."):
        sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if sub:
            return sub
    return {}


def safe_load_state_dict(model: nn.Module, sd: dict) -> None:
    """Load state dict into model with ``strict=False``.

    Tries ``assign=True`` for zero-copy loading; falls back without it
    for older PyTorch versions. Logs missing/unexpected keys at debug level.
    """
    try:
        result = model.load_state_dict(sd, strict=False, assign=True)
    except TypeError:
        result = model.load_state_dict(sd, strict=False)
    if result.missing_keys:
        logger.debug("safe_load_state_dict: %d missing keys", len(result.missing_keys))
    if result.unexpected_keys:
        logger.debug("safe_load_state_dict: %d unexpected keys", len(result.unexpected_keys))


# --------------------------------------------------------------------------- #
# VAE key conversion: LDM / CompVis → diffusers AutoencoderKL
#
# Derived from ComfyUI comfy/diffusers_convert.py (MIT License).
# The original converts diffusers→LDM; this reverses the mapping.
# --------------------------------------------------------------------------- #


def _build_vae_conversion_map() -> list[tuple[str, str]]:
    """Build (ldm_key_part, diffusers_key_part) pairs."""
    mapping: list[tuple[str, str]] = [
        # (LDM, diffusers)
        ("nin_shortcut", "conv_shortcut"),
        ("norm_out", "conv_norm_out"),
        ("mid.attn_1.", "mid_block.attentions.0."),
    ]

    for i in range(4):
        for j in range(2):
            ldm = f"encoder.down.{i}.block.{j}."
            hf = f"encoder.down_blocks.{i}.resnets.{j}."
            mapping.append((ldm, hf))

        if i < 3:
            mapping.append((f"down.{i}.downsample.", f"down_blocks.{i}.downsamplers.0."))
            mapping.append((f"up.{3 - i}.upsample.", f"up_blocks.{i}.upsamplers.0."))

        for j in range(3):
            ldm = f"decoder.up.{3 - i}.block.{j}."
            hf = f"decoder.up_blocks.{i}.resnets.{j}."
            mapping.append((ldm, hf))

    for i in range(2):
        ldm = f"mid.block_{i + 1}."
        hf = f"mid_block.resnets.{i}."
        mapping.append((ldm, hf))

    return mapping


def _build_vae_attn_map() -> list[tuple[str, str]]:
    """Build attention sub-key mapping (LDM → diffusers)."""
    return [
        ("norm.", "group_norm."),
        ("q.", "to_q."),
        ("k.", "to_k."),
        ("v.", "to_v."),
        ("proj_out.", "to_out.0."),
    ]


_VAE_MAP = _build_vae_conversion_map()
_VAE_ATTN_MAP = _build_vae_attn_map()


def convert_ldm_vae_to_diffusers(vae_state_dict: dict) -> dict:
    """Convert LDM/CompVis/Stability VAE keys to diffusers AutoencoderKL format.

    Derived from ComfyUI ``comfy/diffusers_convert.py`` (MIT License),
    reversed to go LDM → diffusers instead of diffusers → LDM.
    """
    mapping = {k: k for k in vae_state_dict.keys()}

    # Apply structural key conversions (LDM → diffusers)
    for k, v in mapping.items():
        for ldm_part, hf_part in _VAE_MAP:
            v = v.replace(ldm_part, hf_part)
        mapping[k] = v

    # Apply attention sub-key conversions
    for k, v in mapping.items():
        if "attentions" in v:
            for ldm_part, hf_part in _VAE_ATTN_MAP:
                v = v.replace(ldm_part, hf_part)
            mapping[k] = v

    return {mapping[k]: vae_state_dict[k] for k in vae_state_dict}


# --------------------------------------------------------------------------- #
# VAE wrappers
# --------------------------------------------------------------------------- #


class VAEDecoder:
    """Thin wrapper around a diffusers VAE model for decoding latents to images."""

    __slots__ = ("vae", "dtype", "device", "scaling_factor")

    def __init__(
        self,
        vae_model: nn.Module,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
        scaling_factor: float = 0.18215,
    ) -> None:
        self.vae = vae_model
        self.dtype = dtype
        self.device = device
        self.scaling_factor = scaling_factor

    def decode(self, latents: Tensor, **kwargs) -> Tensor:
        """Decode latents to images. Returns tensor in [0, 1] range."""
        latents = latents / self.scaling_factor
        latents = latents.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            images = self.vae.decode(latents, **kwargs).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        return images


class VAEEncoder:
    """Thin wrapper around a diffusers VAE model for encoding images to latents."""

    __slots__ = ("vae", "dtype", "device", "scaling_factor")

    def __init__(
        self,
        vae_model: nn.Module,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
        scaling_factor: float = 0.18215,
    ) -> None:
        self.vae = vae_model
        self.dtype = dtype
        self.device = device
        self.scaling_factor = scaling_factor

    def encode(self, images: Tensor, **kwargs) -> Tensor:
        """Encode images to latents. Expects tensor in [0, 1] range."""
        images = 2 * images - 1  # [0,1] → [-1,1]
        images = images.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            latents = self.vae.encode(images, **kwargs).latent_dist.sample()
        latents = latents * self.scaling_factor
        return latents


# --------------------------------------------------------------------------- #
# Per-architecture defaults
# --------------------------------------------------------------------------- #


_VAE_SCALING_FACTORS: dict[str, float] = {
    "sd15": 0.18215,
    "sd20": 0.18215,
    "sd21": 0.18215,
    "sdxl": 0.13025,
    "sd3": 1.5305,
    "flux_dev": 0.3611,
    "flux_fill": 0.3611,
    "flux_schnell": 0.3611,
    "flux_2_dev": 1.0,
    "flux_2_klein_4b": 0.3611,
    "flux_2_klein_9b": 0.3611,
    "ltxv": 1.0,
    "wan": 1.0,
    "zimage": 0.3611,
    "qwen": 0.3611,
    "lumina": 0.3611,
    "chroma": 0.3611,
    "hunyuan_video": 1.0,
}


def get_vae_scaling_factor(arch: ModelArchitecture | str) -> float:
    """Return default VAE scaling factor for the given architecture."""
    key = arch.value if isinstance(arch, ModelArchitecture) else str(arch)
    return _VAE_SCALING_FACTORS.get(key, 0.18215)


_PREDICTION_TYPES: dict[str, str] = {
    "sd15": "eps",
    "sd20": "v_prediction",
    "sd21": "v_prediction",
    "sdxl": "eps",
    "sd3": "flow",
    "flux_dev": "flow_flux",
    "flux_fill": "flow_flux",
    "flux_schnell": "flow_flux",
    "flux_2_dev": "flow_flux",
    "flux_2_klein_4b": "flow_flux",
    "flux_2_klein_9b": "flow_flux",
    "ltxv": "flow",
    "wan": "flow",
    "zimage": "flow",
    "qwen": "flow",
    "lumina": "flow",
    "chroma": "flow_flux",
    "hunyuan_video": "flow",
}


def get_prediction_type(arch: ModelArchitecture | str) -> str:
    """Return default prediction type for the given architecture."""
    key = arch.value if isinstance(arch, ModelArchitecture) else str(arch)
    return _PREDICTION_TYPES.get(key, "eps")


_PREDICTION_KWARGS: dict[str, dict[str, Any]] = {
    "sd15": {},
    "sd20": {},
    "sd21": {},
    "sdxl": {},
    "sd3": {"shift": 3.0, "multiplier": 1000.0},
    "flux_dev": {"base_shift": 0.5, "max_shift": 1.15},
    "flux_fill": {"base_shift": 0.5, "max_shift": 1.15},
    "flux_schnell": {"base_shift": 0.5, "max_shift": 1.15},
    "flux_2_dev": {"base_shift": 0.5, "max_shift": 1.15},
    "flux_2_klein_4b": {"base_shift": 0.5, "max_shift": 1.15},
    "flux_2_klein_9b": {"base_shift": 0.5, "max_shift": 1.15},
    "ltxv": {"shift": 1.0, "multiplier": 1000.0},
    "wan": {"shift": 1.0, "multiplier": 1000.0},
    "zimage": {"shift": 1.0, "multiplier": 1000.0},
    "qwen": {"shift": 1.0, "multiplier": 1000.0},
    "lumina": {"shift": 1.0, "multiplier": 1000.0},
    "chroma": {"base_shift": 0.5, "max_shift": 1.15},
    "hunyuan_video": {"shift": 1.0, "multiplier": 1000.0},
}


def get_prediction_kwargs(arch: ModelArchitecture | str) -> dict[str, Any]:
    """Return default prediction kwargs for the given architecture."""
    key = arch.value if isinstance(arch, ModelArchitecture) else str(arch)
    return _PREDICTION_KWARGS.get(key, {})


# --------------------------------------------------------------------------- #
# Qwen cache snapshot discovery
# --------------------------------------------------------------------------- #


def find_qwen_cache_snapshot() -> Optional[str]:
    """Find a local Qwen-Image pipeline snapshot in ``~/.serenity/models/``.

    Scans for directories containing a ``transformer`` subfolder with
    ``config.json`` (the diffusers pipeline layout).  Returns the path as
    a string, or ``None`` if no snapshot is found.
    """
    from pathlib import Path

    base = Path.home() / ".serenity" / "models"
    candidates = [
        base / "checkpoints",
        base / "diffusion_models",
        base,
    ]
    for search_dir in candidates:
        if not search_dir.exists():
            continue
        for d in search_dir.iterdir():
            if not d.is_dir():
                continue
            if "qwen" in d.name.lower():
                transformer_dir = d / "transformer"
                if transformer_dir.exists() and (transformer_dir / "config.json").exists():
                    return str(d)
    return None
