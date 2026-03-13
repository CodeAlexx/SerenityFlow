"""Mock node implementations for testing the executor.

These don't do real inference -- they produce dummy tensors of the right shape
and type so the executor's routing, linking, and ordering can be verified.
"""
from __future__ import annotations

import torch

from serenityflow.nodes.registry import registry


@registry.register(
    "CheckpointLoaderSimple",
    return_types=("MODEL", "CLIP", "VAE"),
    return_names=("MODEL", "CLIP", "VAE"),
    category="loaders",
    input_types={"required": {"ckpt_name": ("STRING",)}},
)
def mock_checkpoint_loader(ckpt_name):
    return (f"mock_model:{ckpt_name}", f"mock_clip:{ckpt_name}", f"mock_vae:{ckpt_name}")


@registry.register(
    "UNETLoader",
    return_types=("MODEL",),
    category="loaders",
    input_types={"required": {"unet_name": ("STRING",), "weight_dtype": ("STRING",)}},
)
def mock_unet_loader(unet_name, weight_dtype="default"):
    return (f"mock_unet:{unet_name}",)


@registry.register(
    "CLIPLoader",
    return_types=("CLIP",),
    category="loaders",
    input_types={"required": {"clip_name": ("STRING",), "type": ("STRING",)}},
)
def mock_clip_loader(clip_name, type="stable_diffusion"):
    return (f"mock_clip:{clip_name}",)


@registry.register(
    "DualCLIPLoader",
    return_types=("CLIP",),
    category="loaders",
    input_types={"required": {"clip_name1": ("STRING",), "clip_name2": ("STRING",), "type": ("STRING",)}},
)
def mock_dual_clip_loader(clip_name1, clip_name2, type="flux"):
    return (f"mock_dualclip:{clip_name1}+{clip_name2}",)


@registry.register(
    "VAELoader",
    return_types=("VAE",),
    category="loaders",
    input_types={"required": {"vae_name": ("STRING",)}},
)
def mock_vae_loader(vae_name):
    return (f"mock_vae:{vae_name}",)


@registry.register(
    "CLIPTextEncode",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {"clip": ("CLIP",), "text": ("STRING",)}},
)
def mock_clip_text_encode(clip, text):
    return ([{"cross_attn": torch.zeros(1, 77, 768), "pooled": torch.zeros(1, 768), "text": text}],)


@registry.register(
    "EmptyLatentImage",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {"width": ("INT",), "height": ("INT",), "batch_size": ("INT",)}},
)
def mock_empty_latent(width, height, batch_size=1):
    return ({"samples": torch.zeros(batch_size, 4, height // 8, width // 8)},)


@registry.register(
    "EmptySD3LatentImage",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {"width": ("INT",), "height": ("INT",), "batch_size": ("INT",)}},
)
def mock_empty_sd3_latent(width, height, batch_size=1):
    return ({"samples": torch.zeros(batch_size, 16, height // 8, width // 8)},)


@registry.register(
    "KSampler",
    return_types=("LATENT",),
    category="sampling",
    input_types={"required": {
        "model": ("MODEL",), "seed": ("INT",), "steps": ("INT",),
        "cfg": ("FLOAT",), "sampler_name": ("STRING",),
        "scheduler": ("STRING",), "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",), "latent_image": ("LATENT",),
        "denoise": ("FLOAT",),
    }},
)
def mock_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                  positive, negative, latent_image, denoise=1.0):
    latent = latent_image["samples"] if isinstance(latent_image, dict) else latent_image
    return ({"samples": torch.randn_like(latent)},)


@registry.register(
    "VAEDecode",
    return_types=("IMAGE",),
    category="latent",
    input_types={"required": {"samples": ("LATENT",), "vae": ("VAE",)}},
)
def mock_vae_decode(samples, vae):
    latent = samples["samples"] if isinstance(samples, dict) else samples
    b, c, h, w = latent.shape
    return (torch.rand(b, h * 8, w * 8, 3),)


@registry.register(
    "VAEEncode",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {"pixels": ("IMAGE",), "vae": ("VAE",)}},
)
def mock_vae_encode(pixels, vae):
    b, h, w, c = pixels.shape
    return ({"samples": torch.randn(b, 4, h // 8, w // 8)},)


@registry.register(
    "LoraLoader",
    return_types=("MODEL", "CLIP"),
    category="loaders",
    input_types={"required": {
        "model": ("MODEL",), "clip": ("CLIP",),
        "lora_name": ("STRING",),
        "strength_model": ("FLOAT",), "strength_clip": ("FLOAT",),
    }},
)
def mock_lora_loader(model, clip, lora_name, strength_model, strength_clip):
    return (f"{model}+lora:{lora_name}@{strength_model}", f"{clip}+lora:{lora_name}@{strength_clip}")


@registry.register(
    "LoraLoaderModelOnly",
    return_types=("MODEL",),
    category="loaders",
    input_types={"required": {
        "model": ("MODEL",), "lora_name": ("STRING",),
        "strength_model": ("FLOAT",),
    }},
)
def mock_lora_loader_model_only(model, lora_name, strength_model):
    return (f"{model}+lora:{lora_name}@{strength_model}",)


@registry.register(
    "SaveImage",
    return_types=(),
    category="image",
    is_output=True,
    input_types={"required": {"images": ("IMAGE",), "filename_prefix": ("STRING",)}},
)
def mock_save_image(images, filename_prefix="SerenityFlow"):
    return {"ui": {"images": [f"{filename_prefix}_mock.png"]}}


@registry.register(
    "PreviewImage",
    return_types=(),
    category="image",
    is_output=True,
    input_types={"required": {"images": ("IMAGE",)}},
)
def mock_preview_image(images):
    return {"ui": {"images": ["preview_mock.png"]}}


@registry.register(
    "ControlNetLoader",
    return_types=("CONTROL_NET",),
    category="loaders",
    input_types={"required": {"control_net_name": ("STRING",)}},
)
def mock_controlnet_loader(control_net_name):
    return (f"mock_controlnet:{control_net_name}",)


@registry.register(
    "ControlNetApplyAdvanced",
    return_types=("CONDITIONING", "CONDITIONING"),
    category="conditioning",
    input_types={"required": {
        "positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
        "control_net": ("CONTROL_NET",), "image": ("IMAGE",),
        "strength": ("FLOAT",), "start_percent": ("FLOAT",),
        "end_percent": ("FLOAT",),
    }},
)
def mock_controlnet_apply(positive, negative, control_net, image,
                          strength=1.0, start_percent=0.0, end_percent=1.0):
    return (positive, negative)


@registry.register(
    "CLIPSetLastLayer",
    return_types=("CLIP",),
    category="conditioning",
    input_types={"required": {"clip": ("CLIP",), "stop_at_clip_layer": ("INT",)}},
)
def mock_clip_set_last_layer(clip, stop_at_clip_layer):
    return (f"{clip}@layer{stop_at_clip_layer}",)


@registry.register(
    "LoadImage",
    return_types=("IMAGE", "MASK"),
    category="image",
    input_types={"required": {"image": ("STRING",)}},
)
def mock_load_image(image):
    return (torch.rand(1, 512, 512, 3), torch.ones(1, 512, 512))


@registry.register(
    "ImageScale",
    return_types=("IMAGE",),
    category="image",
    input_types={"required": {
        "image": ("IMAGE",), "upscale_method": ("STRING",),
        "width": ("INT",), "height": ("INT",), "crop": ("STRING",),
    }},
)
def mock_image_scale(image, upscale_method, width, height, crop="disabled"):
    return (torch.rand(image.shape[0], height, width, 3),)


# --- New Phase 2 mock nodes ---

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
def mock_cond_set_area_pct(conditioning, width, height, x, y, strength):
    return (conditioning,)


@registry.register(
    "ConditioningSetTimestepRange",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {
        "conditioning": ("CONDITIONING",),
        "start": ("FLOAT",), "end": ("FLOAT",),
    }},
)
def mock_cond_set_timestep_range(conditioning, start, end):
    return (conditioning,)


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
def mock_clip_text_encode_sdxl(clip, width, height, crop_w, crop_h,
                                target_width, target_height, text_g, text_l):
    return ([{"cross_attn": torch.zeros(1, 77, 768), "pooled_output": torch.zeros(1, 1280)}],)


@registry.register(
    "CLIPTextEncodeFlux",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {
        "clip": ("CLIP",), "clip_l": ("STRING",), "t5xxl": ("STRING",),
        "guidance": ("FLOAT",),
    }},
)
def mock_clip_text_encode_flux(clip, clip_l, t5xxl, guidance=3.5):
    return ([{"cross_attn": torch.zeros(1, 256, 4096), "guidance": guidance}],)


@registry.register(
    "FluxGuidance",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {
        "conditioning": ("CONDITIONING",),
        "guidance": ("FLOAT",),
    }},
)
def mock_flux_guidance(conditioning, guidance=3.5):
    return (conditioning,)


@registry.register(
    "LatentUpscaleBy",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "samples": ("LATENT",), "upscale_method": ("STRING",),
        "scale_by": ("FLOAT",),
    }},
)
def mock_latent_upscale_by(samples, upscale_method, scale_by):
    latent = samples["samples"] if isinstance(samples, dict) else samples
    _, c, h, w = latent.shape
    return ({"samples": torch.zeros(1, c, round(h * scale_by), round(w * scale_by))},)


@registry.register(
    "LatentComposite",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "samples_to": ("LATENT",), "samples_from": ("LATENT",),
        "x": ("INT",), "y": ("INT",), "feather": ("INT",),
    }},
)
def mock_latent_composite(samples_to, samples_from, x, y, feather=0):
    return (samples_to,)


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
def mock_latent_crop(samples, width, height, x, y):
    return ({"samples": torch.zeros(1, 4, height // 8, width // 8)},)


@registry.register(
    "LatentFlip",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "samples": ("LATENT",), "flip_method": ("STRING",),
    }},
)
def mock_latent_flip(samples, flip_method="x-axis"):
    return (samples,)


@registry.register(
    "LatentRotate",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "samples": ("LATENT",), "rotation": ("STRING",),
    }},
)
def mock_latent_rotate(samples, rotation="none"):
    return (samples,)


@registry.register(
    "LatentBatch",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "samples1": ("LATENT",), "samples2": ("LATENT",),
    }},
)
def mock_latent_batch(samples1, samples2):
    l1 = samples1["samples"] if isinstance(samples1, dict) else samples1
    l2 = samples2["samples"] if isinstance(samples2, dict) else samples2
    return ({"samples": torch.cat([l1, l2], dim=0)},)


@registry.register(
    "RepeatLatentBatch",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "samples": ("LATENT",), "amount": ("INT",),
    }},
)
def mock_repeat_latent_batch(samples, amount=1):
    latent = samples["samples"] if isinstance(samples, dict) else samples
    return ({"samples": latent.repeat(amount, 1, 1, 1)},)


@registry.register(
    "ImageBatch",
    return_types=("IMAGE",),
    category="image",
    input_types={"required": {
        "image1": ("IMAGE",), "image2": ("IMAGE",),
    }},
)
def mock_image_batch(image1, image2):
    return (torch.cat([image1, image2], dim=0),)


@registry.register(
    "ImageInvert",
    return_types=("IMAGE",),
    category="image",
    input_types={"required": {"image": ("IMAGE",)}},
)
def mock_image_invert(image):
    return (1.0 - image,)


@registry.register(
    "ImagePadForOutpaint",
    return_types=("IMAGE", "MASK"),
    category="image",
    input_types={"required": {
        "image": ("IMAGE",),
        "left": ("INT",), "top": ("INT",),
        "right": ("INT",), "bottom": ("INT",),
        "feathering": ("INT",),
    }},
)
def mock_image_pad_for_outpaint(image, left, top, right, bottom, feathering=40):
    b, h, w, c = image.shape
    return (torch.zeros(b, h + top + bottom, w + left + right, c), torch.ones(b, h + top + bottom, w + left + right))
