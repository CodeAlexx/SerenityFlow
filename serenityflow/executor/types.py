"""Type system matching ComfyUI's type labels.

These are string constants used for documentation and graph planner identification.
No runtime type checking -- nodes accept whatever they receive.
"""
from __future__ import annotations

# Model types (tracked by graph planner)
MODEL = "MODEL"
CLIP = "CLIP"
VAE = "VAE"
CONTROL_NET = "CONTROL_NET"
CLIP_VISION = "CLIP_VISION"
STYLE_MODEL = "STYLE_MODEL"

# Data types (not tracked by graph planner)
CONDITIONING = "CONDITIONING"
LATENT = "LATENT"
IMAGE = "IMAGE"
MASK = "MASK"

# Scalar types
INT = "INT"
FLOAT = "FLOAT"
STRING = "STRING"

# Set of types the graph planner considers "model-carrying"
MODEL_TYPES = {MODEL, CLIP, VAE, CONTROL_NET, CLIP_VISION, STYLE_MODEL}

__all__ = [
    "MODEL", "CLIP", "VAE", "CONTROL_NET", "CLIP_VISION", "STYLE_MODEL",
    "CONDITIONING", "LATENT", "IMAGE", "MASK",
    "INT", "FLOAT", "STRING",
    "MODEL_TYPES",
]
