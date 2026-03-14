"""Latent nodes -- empty latent, VAE decode/encode, upscale, noise mask."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from serenityflow.nodes.registry import registry
from serenityflow.bridge.types import bchw_to_bhwc, bhwc_to_bchw, unwrap_latent, wrap_latent


@registry.register(
    "EmptyLatentImage",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "width": ("INT", {"default": 512, "min": 16, "max": 16384, "step": 8}),
        "height": ("INT", {"default": 512, "min": 16, "max": 16384, "step": 8}),
        "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
    }},
)
def empty_latent_image(width, height, batch_size=1):
    latent = torch.zeros(batch_size, 4, height // 8, width // 8)
    return (wrap_latent(latent),)


@registry.register(
    "EmptySD3LatentImage",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
        "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
        "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
    }},
)
def empty_sd3_latent(width, height, batch_size=1):
    latent = torch.zeros(batch_size, 16, height // 8, width // 8)
    return (wrap_latent(latent),)


@registry.register(
    "VAEDecode",
    return_types=("IMAGE",),
    category="latent",
    input_types={"required": {"samples": ("LATENT",), "vae": ("VAE",)}},
)
def vae_decode_node(samples, vae):
    from serenityflow.bridge.serenity_api import vae_decode

    latent = unwrap_latent(samples)
    # Free GPU VRAM before VAE decode — the diffusion model may still be
    # resident and VAE needs room to work
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    image = vae_decode(vae, latent)
    # Serenity returns BCHW, convert to BHWC for ComfyUI
    image = bchw_to_bhwc(image)
    return (image,)


@registry.register(
    "VAEEncode",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {"pixels": ("IMAGE",), "vae": ("VAE",)}},
)
def vae_encode_node(pixels, vae):
    from serenityflow.bridge.serenity_api import vae_encode

    # ComfyUI images are BHWC, Serenity expects BCHW
    image = bhwc_to_bchw(pixels)
    latent = vae_encode(vae, image)
    return (wrap_latent(latent),)


@registry.register(
    "VAEDecodeTiled",
    return_types=("IMAGE",),
    category="latent",
    input_types={"required": {
        "samples": ("LATENT",), "vae": ("VAE",), "tile_size": ("INT",),
    }},
)
def vae_decode_tiled_node(samples, vae, tile_size=512):
    from serenityflow.bridge.serenity_api import vae_decode_tiled

    latent = unwrap_latent(samples)
    image = vae_decode_tiled(vae, latent, tile_size=tile_size)
    image = bchw_to_bhwc(image)
    return (image,)


@registry.register(
    "VAEEncodeTiled",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "pixels": ("IMAGE",), "vae": ("VAE",), "tile_size": ("INT",),
    }},
)
def vae_encode_tiled_node(pixels, vae, tile_size=512):
    from serenityflow.bridge.serenity_api import vae_encode_tiled

    image = bhwc_to_bchw(pixels)
    latent = vae_encode_tiled(vae, image, tile_size=tile_size)
    return (wrap_latent(latent),)


@registry.register(
    "SetLatentNoiseMask",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {"samples": ("LATENT",), "mask": ("MASK",)}},
)
def set_latent_noise_mask(samples, mask):
    s = dict(samples) if isinstance(samples, dict) else {"samples": samples}
    s["noise_mask"] = mask
    return (s,)


@registry.register(
    "LatentUpscale",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "samples": ("LATENT",), "upscale_method": ("STRING",),
        "width": ("INT",), "height": ("INT",), "crop": ("STRING",),
    }},
)
def latent_upscale(samples, upscale_method, width, height, crop="disabled"):
    latent = unwrap_latent(samples)
    target_h = height // 8
    target_w = width // 8
    mode = "nearest" if upscale_method in ("nearest", "nearest-exact") else "bilinear"
    align = None if mode == "nearest" else False
    result = F.interpolate(latent, size=(target_h, target_w), mode=mode, align_corners=align)
    return (wrap_latent(result),)


@registry.register(
    "LatentUpscaleBy",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "samples": ("LATENT",), "upscale_method": ("STRING",),
        "scale_by": ("FLOAT",),
    }},
)
def latent_upscale_by(samples, upscale_method, scale_by):
    latent = unwrap_latent(samples)
    _, _, h, w = latent.shape
    new_h = round(h * scale_by)
    new_w = round(w * scale_by)
    mode = "nearest" if upscale_method in ("nearest", "nearest-exact") else "bilinear"
    align = None if mode == "nearest" else False
    result = F.interpolate(latent, size=(new_h, new_w), mode=mode, align_corners=align)
    return (wrap_latent(result),)


@registry.register(
    "LatentComposite",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "samples_to": ("LATENT",), "samples_from": ("LATENT",),
        "x": ("INT",), "y": ("INT",),
        "feather": ("INT",),
    }},
)
def latent_composite(samples_to, samples_from, x, y, feather=0):
    dest = unwrap_latent(samples_to).clone()
    src = unwrap_latent(samples_from)
    # Coordinates in latent space (already divided by 8 by caller if needed)
    lx, ly = x // 8, y // 8
    _, _, sh, sw = src.shape
    _, _, dh, dw = dest.shape
    # Clamp to dest bounds
    eh = min(ly + sh, dh)
    ew = min(lx + sw, dw)
    sy = max(0, -ly)
    sx = max(0, -lx)
    ly = max(0, ly)
    lx = max(0, lx)
    dest[:, :, ly:eh, lx:ew] = src[:, :, sy:sy + (eh - ly), sx:sx + (ew - lx)]
    return (wrap_latent(dest),)


@registry.register(
    "LatentCrop",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "samples": ("LATENT",),
        "width": ("INT",), "height": ("INT",),
        "x": ("INT",), "y": ("INT",),
    }},
)
def latent_crop(samples, width, height, x, y):
    latent = unwrap_latent(samples)
    lx, ly = x // 8, y // 8
    lw, lh = width // 8, height // 8
    result = latent[:, :, ly:ly + lh, lx:lx + lw]
    return (wrap_latent(result.contiguous()),)


@registry.register(
    "LatentFlip",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "samples": ("LATENT",), "flip_method": ("STRING",),
    }},
)
def latent_flip(samples, flip_method="x-axis"):
    latent = unwrap_latent(samples)
    if flip_method == "x-axis":
        result = torch.flip(latent, [2])  # flip height
    else:
        result = torch.flip(latent, [3])  # flip width
    return (wrap_latent(result),)


@registry.register(
    "LatentRotate",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "samples": ("LATENT",), "rotation": ("STRING",),
    }},
)
def latent_rotate(samples, rotation="none"):
    latent = unwrap_latent(samples)
    k = {"none": 0, "90 degrees": 1, "180 degrees": 2, "270 degrees": 3}.get(rotation, 0)
    if k > 0:
        latent = torch.rot90(latent, k=k, dims=[2, 3])
    return (wrap_latent(latent),)


@registry.register(
    "LatentBatch",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "samples1": ("LATENT",), "samples2": ("LATENT",),
    }},
)
def latent_batch(samples1, samples2):
    l1 = unwrap_latent(samples1)
    l2 = unwrap_latent(samples2)
    # Resize if spatial dims differ — match to first
    if l1.shape[2:] != l2.shape[2:]:
        l2 = F.interpolate(l2, size=l1.shape[2:], mode="bilinear", align_corners=False)
    result = torch.cat([l1, l2], dim=0)
    return (wrap_latent(result),)


@registry.register(
    "RepeatLatentBatch",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "samples": ("LATENT",), "amount": ("INT",),
    }},
)
def repeat_latent_batch(samples, amount=1):
    latent = unwrap_latent(samples)
    result = latent.repeat(amount, 1, 1, 1)
    return (wrap_latent(result),)
