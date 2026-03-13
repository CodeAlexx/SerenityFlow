"""Compatibility shim for comfy.comfy_types.node_typing."""
from __future__ import annotations

from typing import Any


class IO:
    IMAGE = "IMAGE"
    MASK = "MASK"
    LATENT = "LATENT"
    MODEL = "MODEL"
    CLIP = "CLIP"
    VAE = "VAE"
    CONDITIONING = "CONDITIONING"
    CONTROL_NET = "CONTROL_NET"
    STRING = "STRING"
    INT = "INT"
    FLOAT = "FLOAT"
    BOOLEAN = "BOOLEAN"


class ComfyNodeABC:
    """Base class for ComfyUI-style nodes."""
    RETURN_TYPES: tuple = ()
    RETURN_NAMES: tuple = ()
    FUNCTION: str = "execute"
    CATEGORY: str = ""
    OUTPUT_NODE: bool = False

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}


InputTypeDict = dict
