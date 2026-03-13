"""Conditioning nodes -- text encode, combine, concat, area, mask, zero."""
from __future__ import annotations

import torch

from serenityflow.nodes.registry import registry
from serenityflow.bridge.types import find_cross_attn_key


@registry.register(
    "CLIPTextEncode",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {"clip": ("CLIP",), "text": ("STRING",)}},
)
def clip_text_encode(clip, text):
    from serenityflow.bridge.serenity_api import encode_text
    conditioning = encode_text(clip, text)
    return (conditioning,)


@registry.register(
    "ConditioningCombine",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {
        "conditioning_1": ("CONDITIONING",),
        "conditioning_2": ("CONDITIONING",),
    }},
)
def conditioning_combine(conditioning_1, conditioning_2):
    return (conditioning_1 + conditioning_2,)


@registry.register(
    "ConditioningConcat",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {
        "conditioning_to": ("CONDITIONING",),
        "conditioning_from": ("CONDITIONING",),
    }},
)
def conditioning_concat(conditioning_to, conditioning_from):
    out = []
    for i in range(min(len(conditioning_to), len(conditioning_from))):
        t = dict(conditioning_to[i])
        key_to = find_cross_attn_key(t)
        key_from = find_cross_attn_key(conditioning_from[i])
        if key_to and key_from:
            t[key_to] = torch.cat([t[key_to], conditioning_from[i][key_from]], dim=1)
        out.append(t)
    return (out,)


@registry.register(
    "ConditioningSetArea",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {
        "conditioning": ("CONDITIONING",),
        "width": ("INT",), "height": ("INT",),
        "x": ("INT",), "y": ("INT",),
        "strength": ("FLOAT",),
    }},
)
def conditioning_set_area(conditioning, width, height, x, y, strength):
    out = []
    for c in conditioning:
        n = dict(c)
        n["area"] = (height // 8, width // 8, y // 8, x // 8)
        n["strength"] = strength
        n["set_area_to_bounds"] = False
        out.append(n)
    return (out,)


@registry.register(
    "ConditioningSetMask",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {
        "conditioning": ("CONDITIONING",),
        "mask": ("MASK",), "strength": ("FLOAT",),
        "set_cond_area": ("STRING",),
    }},
)
def conditioning_set_mask(conditioning, mask, strength, set_cond_area="default"):
    out = []
    for c in conditioning:
        n = dict(c)
        n["mask"] = mask
        n["strength"] = strength
        n["set_area_to_bounds"] = (set_cond_area != "default")
        out.append(n)
    return (out,)


@registry.register(
    "ConditioningZeroOut",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {"conditioning": ("CONDITIONING",)}},
)
def conditioning_zero_out(conditioning):
    out = []
    for c in conditioning:
        n = dict(c)
        key = find_cross_attn_key(n)
        if key:
            n[key] = torch.zeros_like(n[key])
        if "pooled_output" in n:
            n["pooled_output"] = torch.zeros_like(n["pooled_output"])
        out.append(n)
    return (out,)


@registry.register(
    "ConditioningSetAreaPercentage",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {
        "conditioning": ("CONDITIONING",),
        "width": ("FLOAT",), "height": ("FLOAT",),
        "x": ("FLOAT",), "y": ("FLOAT",),
        "strength": ("FLOAT",),
    }},
)
def conditioning_set_area_percentage(conditioning, width, height, x, y, strength):
    out = []
    for c in conditioning:
        n = dict(c)
        n["area"] = ("percentage", height, width, y, x)
        n["strength"] = strength
        n["set_area_to_bounds"] = False
        out.append(n)
    return (out,)


@registry.register(
    "ConditioningSetTimestepRange",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {
        "conditioning": ("CONDITIONING",),
        "start": ("FLOAT",), "end": ("FLOAT",),
    }},
)
def conditioning_set_timestep_range(conditioning, start, end):
    out = []
    for c in conditioning:
        n = dict(c)
        n["timestep_start"] = start
        n["timestep_end"] = end
        out.append(n)
    return (out,)


@registry.register(
    "CLIPTextEncodeSDXL",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {
        "clip": ("CLIP",),
        "width": ("INT",), "height": ("INT",),
        "crop_w": ("INT",), "crop_h": ("INT",),
        "target_width": ("INT",), "target_height": ("INT",),
        "text_g": ("STRING",), "text_l": ("STRING",),
    }},
)
def clip_text_encode_sdxl(clip, width, height, crop_w, crop_h,
                          target_width, target_height, text_g, text_l):
    from serenityflow.bridge.serenity_api import encode_text
    # SDXL: encode using the primary text (text_g typically)
    # The CLIPWrapper handles dual encoding internally based on arch
    conditioning = encode_text(clip, text_g)
    # Attach SDXL-specific metadata
    out = []
    for c in conditioning:
        n = dict(c)
        n["width"] = width
        n["height"] = height
        n["crop_w"] = crop_w
        n["crop_h"] = crop_h
        n["target_width"] = target_width
        n["target_height"] = target_height
        n["text_l"] = text_l
        out.append(n)
    return (out,)


@registry.register(
    "CLIPTextEncodeFlux",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {
        "clip": ("CLIP",), "clip_l": ("STRING",), "t5xxl": ("STRING",),
        "guidance": ("FLOAT",),
    }},
)
def clip_text_encode_flux(clip, clip_l, t5xxl, guidance=3.5):
    from serenityflow.bridge.serenity_api import encode_text
    # Flux uses T5 as primary, CLIP-L as secondary
    conditioning = encode_text(clip, t5xxl)
    out = []
    for c in conditioning:
        n = dict(c)
        n["guidance"] = guidance
        out.append(n)
    return (out,)


@registry.register(
    "FluxGuidance",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {
        "conditioning": ("CONDITIONING",),
        "guidance": ("FLOAT",),
    }},
)
def flux_guidance(conditioning, guidance=3.5):
    out = []
    for c in conditioning:
        n = dict(c)
        n["guidance"] = guidance
        out.append(n)
    return (out,)


@registry.register(
    "InstructPixToPixConditioning",
    return_types=("CONDITIONING", "CONDITIONING", "LATENT"),
    return_names=("positive", "negative", "latent"),
    category="conditioning",
    input_types={"required": {
        "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",),
        "vae": ("VAE",),
        "pixels": ("IMAGE",),
    }},
)
def instruct_pix_to_pix_conditioning(positive, negative, vae, pixels):
    raise NotImplementedError("InstructPixToPixConditioning")


@registry.register(
    "InpaintModelConditioning",
    return_types=("CONDITIONING", "CONDITIONING", "LATENT"),
    return_names=("positive", "negative", "latent"),
    category="conditioning",
    input_types={"required": {
        "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",),
        "vae": ("VAE",),
        "pixels": ("IMAGE",),
        "mask": ("MASK",),
    }},
)
def inpaint_model_conditioning(positive, negative, vae, pixels, mask):
    raise NotImplementedError("InpaintModelConditioning")
