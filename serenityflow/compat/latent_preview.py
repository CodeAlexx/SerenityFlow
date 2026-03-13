"""Compatibility shim for latent_preview module."""
from __future__ import annotations

from enum import Enum


class LatentPreviewMethod(Enum):
    NoPreviews = "none"
    Auto = "auto"
    Latent2RGB = "latent2rgb"
    TAESD = "taesd"


def get_previewer(device, latent_format=None):
    return None


def set_preview_method(method):
    pass
