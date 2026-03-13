"""Compatibility shim for comfy.model_detection.

Model architecture detection from state dict keys.
"""
from __future__ import annotations

import comfy.utils


def count_blocks(state_dict_keys, prefix_string):
    count = 0
    while True:
        if any(k.startswith(f"{prefix_string}{count}") for k in state_dict_keys):
            count += 1
        else:
            break
    return count


def detect_unet_config(state_dict, key_prefix, dtype=None):
    """Detect UNet config from state dict keys."""
    return {}


def unet_config_from_diffusers_unet(state_dict, dtype=None):
    return {}


def model_config_from_unet_config(unet_config, state_dict=None):
    return None


def model_config_from_unet(state_dict, unet_key_prefix, dtype=None):
    return None


def convert_config(unet_config):
    return unet_config
