"""SDXL-specific nodes -- refiner encoder, latent."""
from __future__ import annotations

import torch

from serenityflow.nodes.registry import registry
from serenityflow.bridge.types import wrap_latent


@registry.register(
    "CLIPTextEncodeSDXLRefiner",
    return_types=("CONDITIONING",),
    category="conditioning/sdxl",
    input_types={"required": {
        "clip": ("CLIP",),
        "ascore": ("FLOAT",),
        "width": ("INT",), "height": ("INT",),
        "text": ("STRING",),
    }},
)
def clip_text_encode_sdxl_refiner(clip, ascore, width, height, text):
    from serenityflow.bridge.serenity_api import encode_text
    conditioning = encode_text(clip, text)
    out = []
    for c in conditioning:
        n = dict(c)
        n["aesthetic_score"] = ascore
        n["width"] = width
        n["height"] = height
        out.append(n)
    return (out,)


@registry.register(
    "EmptySDXLLatentImage",
    return_types=("LATENT",),
    category="latent/sdxl",
    input_types={"required": {
        "width": ("INT",), "height": ("INT",), "batch_size": ("INT",),
    }},
)
def empty_sdxl_latent(width, height, batch_size=1):
    # SDXL: 4 channels at 1/8 resolution (same as SD1.5)
    latent = torch.zeros(batch_size, 4, height // 8, width // 8)
    return (wrap_latent(latent),)
