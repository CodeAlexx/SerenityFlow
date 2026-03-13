"""Other model-specific nodes -- HiDream, Chroma, ZImage."""
from __future__ import annotations

import torch

from serenityflow.nodes.registry import registry
from serenityflow.bridge.types import wrap_latent


@registry.register(
    "EmptyHiDreamLatentImage",
    return_types=("LATENT",),
    category="latent/hidream",
    input_types={"required": {
        "width": ("INT",), "height": ("INT",), "batch_size": ("INT",),
    }},
)
def empty_hidream_latent(width, height, batch_size=1):
    # HiDream: 16 channels, 1/8 resolution
    latent = torch.zeros(batch_size, 16, height // 8, width // 8)
    return (wrap_latent(latent),)


@registry.register(
    "EmptyChromaLatentImage",
    return_types=("LATENT",),
    category="latent/chroma",
    input_types={"required": {
        "width": ("INT",), "height": ("INT",), "batch_size": ("INT",),
    }},
)
def empty_chroma_latent(width, height, batch_size=1):
    # Chroma: 16 channels, 1/8 resolution (Flux-based)
    latent = torch.zeros(batch_size, 16, height // 8, width // 8)
    return (wrap_latent(latent),)


@registry.register(
    "EmptyZImageLatentImage",
    return_types=("LATENT",),
    category="latent/zimage",
    input_types={"required": {
        "width": ("INT",), "height": ("INT",), "batch_size": ("INT",),
    }},
)
def empty_zimage_latent(width, height, batch_size=1):
    # ZImage: 16 channels, 1/8 resolution (Flux-based)
    latent = torch.zeros(batch_size, 16, height // 8, width // 8)
    return (wrap_latent(latent),)


@registry.register(
    "Load3D",
    return_types=("MESH",),
    category="3d",
    input_types={"required": {"model_file": ("STRING",)}},
)
def load_3d(model_file):
    return ({"filepath": model_file},)


@registry.register(
    "EmptyLatentHunyuan3Dv2",
    return_types=("LATENT",),
    category="latent/3d",
    input_types={"required": {
        "resolution": ("INT",), "batch_size": ("INT",),
    }},
)
def empty_latent_hunyuan_3dv2(resolution, batch_size=1):
    latent = torch.zeros(batch_size, 16, resolution // 8, resolution // 8)
    return (wrap_latent(latent),)


@registry.register(
    "VoxelToMesh",
    return_types=("MESH",),
    category="3d",
    input_types={"required": {"voxels": ("VOXELS",)}},
)
def voxel_to_mesh(voxels):
    raise NotImplementedError("VoxelToMesh requires marching cubes implementation")


@registry.register(
    "VAEDecodeHunyuan3D",
    return_types=("MESH",),
    category="3d",
    input_types={"required": {
        "samples": ("LATENT",),
        "vae": ("VAE",),
    }},
)
def vae_decode_hunyuan_3d(samples, vae):
    raise NotImplementedError("VAEDecodeHunyuan3D requires bridge.decode_3d()")


@registry.register(
    "Hunyuan3Dv2Conditioning",
    return_types=("CONDITIONING",),
    category="conditioning/3d",
    input_types={"required": {
        "clip": ("CLIP",),
        "image": ("IMAGE",),
    }},
)
def hunyuan_3dv2_conditioning(clip, image):
    raise NotImplementedError("Hunyuan3Dv2Conditioning requires bridge.encode_3d_conditioning()")


@registry.register(
    "Hunyuan3Dv2ConditioningMultiView",
    return_types=("CONDITIONING",),
    category="conditioning/3d",
    input_types={"required": {
        "clip": ("CLIP",),
        "front": ("IMAGE",),
        "back": ("IMAGE",),
        "left": ("IMAGE",),
        "right": ("IMAGE",),
    }},
)
def hunyuan_3dv2_conditioning_multi_view(clip, front, back, left, right):
    raise NotImplementedError("Hunyuan3Dv2ConditioningMultiView requires bridge.encode_3d_conditioning()")
