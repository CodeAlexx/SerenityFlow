"""Standalone FLUX.2 loading utilities for SerenityFlow.

Provides variant probing, config generation, and single-file loading helpers.
No runtime imports from serenity.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "check_not_lora",
    "flux2_config_dir",
    "format_load_diagnostic",
    "probe_flux2_variant",
    "strip_accelerate_hooks",
]


# --------------------------------------------------------------------------- #
# Flux2Transformer2DModel config templates
#
# These match the __init__ signature of diffusers Flux2Transformer2DModel.
# Keys: patch_size, in_channels, num_layers, num_single_layers,
#        attention_head_dim, num_attention_heads, joint_attention_dim,
#        timestep_guidance_channels, mlp_ratio, axes_dims_rope, rope_theta, eps
# --------------------------------------------------------------------------- #

_FLUX2_DEV_CONFIG: dict[str, Any] = {
    "_class_name": "Flux2Transformer2DModel",
    "_diffusers_version": "0.36.0",
    "patch_size": 1,
    "in_channels": 128,
    "num_layers": 8,
    "num_single_layers": 48,
    "attention_head_dim": 128,
    "num_attention_heads": 48,
    "joint_attention_dim": 15360,
    "timestep_guidance_channels": 256,
    "mlp_ratio": 3.0,
    "axes_dims_rope": [32, 32, 32, 32],
    "rope_theta": 2000,
    "eps": 1e-6,
}

_KLEIN_9B_CONFIG: dict[str, Any] = {
    "_class_name": "Flux2Transformer2DModel",
    "_diffusers_version": "0.36.0",
    "patch_size": 1,
    "in_channels": 128,
    "num_layers": 10,
    "num_single_layers": 20,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 7680,
    "timestep_guidance_channels": 256,
    "mlp_ratio": 3.0,
    "axes_dims_rope": [32, 32, 32, 32],
    "rope_theta": 2000,
    "eps": 1e-6,
}

_KLEIN_4B_CONFIG: dict[str, Any] = {
    "_class_name": "Flux2Transformer2DModel",
    "_diffusers_version": "0.36.0",
    "patch_size": 1,
    "in_channels": 128,
    "num_layers": 5,
    "num_single_layers": 10,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 7680,
    "timestep_guidance_channels": 256,
    "mlp_ratio": 3.0,
    "axes_dims_rope": [32, 32, 32, 32],
    "rope_theta": 2000,
    "eps": 1e-6,
}


# --------------------------------------------------------------------------- #
# Variant probing
# --------------------------------------------------------------------------- #


def probe_flux2_variant(path: str) -> tuple[str, dict[str, Any]]:
    """Detect FLUX.2 variant from safetensors header keys.

    Reads tensor names only (no data loaded) and counts double_blocks /
    single_blocks to distinguish Klein 4B, Klein 9B, and Dev.

    Returns ``(variant_name, config_dict)`` where *config_dict* is suitable
    for writing to a temporary ``config.json`` consumed by
    ``Flux2Transformer2DModel.from_single_file``.
    """
    from serenityflow.bridge.model_detect import read_tensor_names

    names = read_tensor_names(path)
    double_count = sum(
        1 for k in names
        if k.startswith("double_blocks.") and k.endswith(".img_attn.qkv.weight")
    )
    single_count = sum(
        1 for k in names
        if k.startswith("single_blocks.") and k.endswith(".linear1.weight")
    )

    logger.debug(
        "probe_flux2_variant: %s — double=%d single=%d",
        os.path.basename(path), double_count, single_count,
    )

    # Klein 4B: 5 double + 10 single
    if double_count <= 5 and single_count <= 10:
        return "klein_4b", dict(_KLEIN_4B_CONFIG)

    # Klein 9B: ~10 double + ~20 single
    if double_count <= 10 and single_count <= 20:
        return "klein_9b", dict(_KLEIN_9B_CONFIG)

    # Dev (or larger): 8 double + 48 single (default config)
    return "dev", dict(_FLUX2_DEV_CONFIG)


# --------------------------------------------------------------------------- #
# Config directory context manager
# --------------------------------------------------------------------------- #


@contextmanager
def flux2_config_dir(config: dict[str, Any]):
    """Create a temporary directory containing ``config.json``.

    Yields the directory path for use as ``config=`` argument to
    ``from_single_file``.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)
        yield tmpdir


# --------------------------------------------------------------------------- #
# Single-file loading helpers
# --------------------------------------------------------------------------- #


def check_not_lora(path: str) -> None:
    """Raise ``ValueError`` if the file appears to be a LoRA checkpoint."""
    from serenityflow.bridge.model_detect import read_tensor_names

    names = read_tensor_names(path)
    lora_indicators = ("lora_a.", "lora_b.", "lora_down.", "lora_up.",
                       "lora_A.", "lora_B.")
    for name in names:
        low = name.lower()
        if any(ind.lower() in low for ind in lora_indicators):
            raise ValueError(
                f"File appears to be a LoRA, not a model checkpoint: {path}"
            )


def format_load_diagnostic(path: str, model_type: str, exc: Exception) -> str:
    """Format a human-readable load failure message."""
    return (
        f"Failed to load {model_type} from {os.path.basename(path)}: {exc}"
    )


def strip_accelerate_hooks(model) -> None:
    """Remove accelerate dispatch hooks that interfere with direct model use.

    After ``from_single_file`` with accelerate, modules may have
    ``_hf_hook`` / ``_old_forward`` attributes that redirect ``forward()``
    through the accelerate dispatcher. Stripping them restores normal
    PyTorch behaviour.
    """
    for module in model.modules():
        if hasattr(module, "_hf_hook"):
            del module._hf_hook
        if hasattr(module, "_old_forward"):
            module.forward = module._old_forward
            del module._old_forward
