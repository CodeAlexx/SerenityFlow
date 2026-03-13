"""Compatibility shim for comfy.ldm.lightricks.model.

LTX Video model utilities. KJNodes imports apply_rotary_emb from here.
"""
from __future__ import annotations

import torch


def apply_rotary_emb(x, freqs):
    """Apply rotary embeddings to input tensor."""
    cos = freqs.cos()
    sin = freqs.sin()

    # Split x into pairs for rotation
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    # Apply rotation
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos

    return torch.stack([y1, y2], dim=-1).flatten(-2)
