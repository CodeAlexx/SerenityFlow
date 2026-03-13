"""Compatibility shim for comfy_api.latest.io.

New-style node API used by newer custom nodes.
"""
from __future__ import annotations

from typing import Any


class NodeOutput:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class NodeInput:
    def __init__(self, name, type_name, **kwargs):
        self.name = name
        self.type_name = type_name
        self.default = kwargs.get("default")
        self.optional = kwargs.get("optional", False)


# Type constants
IMAGE = "IMAGE"
MASK = "MASK"
LATENT = "LATENT"
MODEL = "MODEL"
CLIP = "CLIP"
VAE = "VAE"
CONDITIONING = "CONDITIONING"
STRING = "STRING"
INT = "INT"
FLOAT = "FLOAT"
BOOLEAN = "BOOLEAN"
