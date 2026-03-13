"""Model operation nodes -- merge, CFG rescale, FreeU, attention guidance, etc."""
from __future__ import annotations

import torch

from serenityflow.nodes.registry import registry


# ---------------------------------------------------------------------------
# Model merging
# ---------------------------------------------------------------------------

@registry.register(
    "ModelMergeSimple",
    return_types=("MODEL",),
    category="advanced/model_merging",
    input_types={"required": {
        "model1": ("MODEL",),
        "model2": ("MODEL",),
        "ratio": ("FLOAT",),
    }},
)
def model_merge_simple(model1, model2, ratio=1.0):
    # TODO: bridge.merge_models()
    raise NotImplementedError("ModelMergeSimple requires bridge.merge_models()")


@registry.register(
    "ModelMergeBlocks",
    return_types=("MODEL",),
    category="advanced/model_merging",
    input_types={"required": {
        "model1": ("MODEL",),
        "model2": ("MODEL",),
        "input": ("FLOAT",),
        "middle": ("FLOAT",),
        "out": ("FLOAT",),
    }},
)
def model_merge_blocks(model1, model2, input=1.0, middle=1.0, out=1.0):
    # TODO: bridge.merge_models_by_blocks()
    raise NotImplementedError("ModelMergeBlocks requires bridge.merge_models_by_blocks()")


@registry.register(
    "CLIPMergeSimple",
    return_types=("CLIP",),
    category="advanced/model_merging",
    input_types={"required": {
        "clip1": ("CLIP",),
        "clip2": ("CLIP",),
        "ratio": ("FLOAT",),
    }},
)
def clip_merge_simple(clip1, clip2, ratio=1.0):
    # TODO: bridge.merge_clips()
    raise NotImplementedError("CLIPMergeSimple requires bridge.merge_clips()")


@registry.register(
    "CLIPMergeSubtract",
    return_types=("CLIP",),
    category="advanced/model_merging",
    input_types={"required": {
        "clip1": ("CLIP",),
        "clip2": ("CLIP",),
        "multiplier": ("FLOAT",),
    }},
)
def clip_merge_subtract(clip1, clip2, multiplier=1.0):
    # TODO: bridge.merge_clips_subtract()
    raise NotImplementedError("CLIPMergeSubtract requires bridge.merge_clips_subtract()")


# ---------------------------------------------------------------------------
# Model sampling overrides
# ---------------------------------------------------------------------------

@registry.register(
    "ModelSamplingDiscrete",
    return_types=("MODEL",),
    category="advanced/model",
    input_types={"required": {
        "model": ("MODEL",),
        "sampling": ("STRING",),
        "zsnr": ("BOOLEAN",),
    }},
)
def model_sampling_discrete(model, sampling="eps", zsnr=False):
    if hasattr(model, "with_options"):
        return (model.with_options({"sampling_type": sampling, "zsnr": zsnr}),)
    return (model,)


@registry.register(
    "ModelSamplingContinuousEDM",
    return_types=("MODEL",),
    category="advanced/model",
    input_types={"required": {
        "model": ("MODEL",),
        "sampling": ("STRING",),
        "sigma_max": ("FLOAT",),
        "sigma_min": ("FLOAT",),
    }},
)
def model_sampling_continuous_edm(model, sampling="v_prediction", sigma_max=120.0, sigma_min=0.002):
    if hasattr(model, "with_options"):
        return (model.with_options({
            "sampling_type": "continuous_edm",
            "prediction_type": sampling,
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
        }),)
    return (model,)


@registry.register(
    "ModelSamplingContinuousV",
    return_types=("MODEL",),
    category="advanced/model",
    input_types={"required": {
        "model": ("MODEL",),
        "sampling": ("STRING",),
        "sigma_max": ("FLOAT",),
        "sigma_min": ("FLOAT",),
    }},
)
def model_sampling_continuous_v(model, sampling="v_prediction", sigma_max=500.0, sigma_min=0.002):
    if hasattr(model, "with_options"):
        return (model.with_options({
            "sampling_type": "continuous_v",
            "prediction_type": sampling,
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
        }),)
    return (model,)


@registry.register(
    "ModelSamplingFlux",
    return_types=("MODEL",),
    category="advanced/model",
    input_types={"required": {
        "model": ("MODEL",),
        "max_shift": ("FLOAT",),
        "base_shift": ("FLOAT",),
        "width": ("INT",), "height": ("INT",),
    }},
)
def model_sampling_flux(model, max_shift=1.15, base_shift=0.5, width=1024, height=1024):
    if hasattr(model, "with_options"):
        return (model.with_options({
            "sampling_type": "flux",
            "max_shift": max_shift,
            "base_shift": base_shift,
            "width": width,
            "height": height,
        }),)
    return (model,)


# ---------------------------------------------------------------------------
# CFG modifications
# ---------------------------------------------------------------------------

@registry.register(
    "RescaleCFG",
    return_types=("MODEL",),
    category="advanced/model",
    input_types={"required": {
        "model": ("MODEL",),
        "multiplier": ("FLOAT",),
    }},
)
def rescale_cfg(model, multiplier=0.7):
    if hasattr(model, "with_options"):
        return (model.with_options({"rescale_cfg_multiplier": multiplier}),)
    return (model,)


# ---------------------------------------------------------------------------
# FreeU
# ---------------------------------------------------------------------------

@registry.register(
    "FreeU",
    return_types=("MODEL",),
    category="advanced/model",
    input_types={"required": {
        "model": ("MODEL",),
        "b1": ("FLOAT",), "b2": ("FLOAT",),
        "s1": ("FLOAT",), "s2": ("FLOAT",),
    }},
)
def freeu(model, b1=1.1, b2=1.2, s1=0.9, s2=0.2):
    if hasattr(model, "with_options"):
        return (model.with_options({
            "freeu": {"b1": b1, "b2": b2, "s1": s1, "s2": s2, "version": 1},
        }),)
    return (model,)


@registry.register(
    "FreeU_V2",
    return_types=("MODEL",),
    category="advanced/model",
    input_types={"required": {
        "model": ("MODEL",),
        "b1": ("FLOAT",), "b2": ("FLOAT",),
        "s1": ("FLOAT",), "s2": ("FLOAT",),
    }},
)
def freeu_v2(model, b1=1.3, b2=1.4, s1=0.9, s2=0.2):
    if hasattr(model, "with_options"):
        return (model.with_options({
            "freeu": {"b1": b1, "b2": b2, "s1": s1, "s2": s2, "version": 2},
        }),)
    return (model,)


# ---------------------------------------------------------------------------
# Attention guidance
# ---------------------------------------------------------------------------

@registry.register(
    "PerturbedAttentionGuidance",
    return_types=("MODEL",),
    category="advanced/model",
    input_types={"required": {
        "model": ("MODEL",),
        "scale": ("FLOAT",),
    }},
)
def perturbed_attention_guidance(model, scale=3.0):
    if hasattr(model, "with_options"):
        return (model.with_options({"pag_scale": scale}),)
    return (model,)


@registry.register(
    "SelfAttentionGuidance",
    return_types=("MODEL",),
    category="advanced/model",
    input_types={"required": {
        "model": ("MODEL",),
        "scale": ("FLOAT",),
        "blur_sigma": ("FLOAT",),
    }},
)
def self_attention_guidance(model, scale=0.5, blur_sigma=2.0):
    if hasattr(model, "with_options"):
        return (model.with_options({"sag_scale": scale, "sag_blur_sigma": blur_sigma}),)
    return (model,)


# ---------------------------------------------------------------------------
# Architecture-specific block merge (use same underlying mechanism)
# ---------------------------------------------------------------------------

def _block_merge_node(name, category, block_count):
    """Factory for architecture-specific block merge nodes."""
    input_spec = {
        "model1": ("MODEL",),
        "model2": ("MODEL",),
    }
    for i in range(block_count):
        input_spec[f"block_{i}"] = ("FLOAT",)

    @registry.register(
        name,
        return_types=("MODEL",),
        category=category,
        input_types={"required": input_spec},
    )
    def merge_fn(**kwargs):
        # TODO: bridge.merge_models_by_blocks()
        raise NotImplementedError(f"{name} requires bridge.merge_models_by_blocks()")
    return merge_fn


_block_merge_node("ModelMergeSD1", "advanced/model_merging/sd1", 25)
_block_merge_node("ModelMergeSDXL", "advanced/model_merging/sdxl", 25)
_block_merge_node("ModelMergeSD3", "advanced/model_merging/sd3", 38)
_block_merge_node("ModelMergeFlux1", "advanced/model_merging/flux", 19)


# ---------------------------------------------------------------------------
# Upscale model
# ---------------------------------------------------------------------------

@registry.register(
    "UpscaleModelLoader",
    return_types=("UPSCALE_MODEL",),
    category="loaders",
    input_types={"required": {"model_name": ("STRING",)}},
)
def upscale_model_loader(model_name):
    # TODO: bridge.load_upscale_model()
    raise NotImplementedError("UpscaleModelLoader requires bridge.load_upscale_model()")


@registry.register(
    "ImageUpscaleWithModel",
    return_types=("IMAGE",),
    category="image/upscaling",
    input_types={"required": {
        "upscale_model": ("UPSCALE_MODEL",),
        "image": ("IMAGE",),
    }},
)
def image_upscale_with_model(upscale_model, image):
    # TODO: bridge.upscale_image()
    raise NotImplementedError("ImageUpscaleWithModel requires bridge.upscale_image()")


# ---------------------------------------------------------------------------
# CLIP Vision / Style
# ---------------------------------------------------------------------------

@registry.register(
    "CLIPVisionLoader",
    return_types=("CLIP_VISION",),
    category="loaders",
    input_types={"required": {"clip_name": ("STRING",)}},
)
def clip_vision_loader(clip_name):
    from serenityflow.bridge.model_paths import get_model_paths
    # TODO: bridge.load_clip_vision()
    raise NotImplementedError("CLIPVisionLoader requires bridge.load_clip_vision()")


@registry.register(
    "CLIPVisionEncode",
    return_types=("CLIP_VISION_OUTPUT",),
    category="conditioning",
    input_types={"required": {
        "clip_vision": ("CLIP_VISION",),
        "image": ("IMAGE",),
    }},
)
def clip_vision_encode(clip_vision, image):
    # TODO: bridge.encode_clip_vision()
    raise NotImplementedError("CLIPVisionEncode requires bridge.encode_clip_vision()")


@registry.register(
    "StyleModelLoader",
    return_types=("STYLE_MODEL",),
    category="loaders",
    input_types={"required": {"style_model_name": ("STRING",)}},
)
def style_model_loader(style_model_name):
    # TODO: bridge.load_style_model()
    raise NotImplementedError("StyleModelLoader requires bridge.load_style_model()")


@registry.register(
    "StyleModelApply",
    return_types=("CONDITIONING",),
    category="conditioning/style_model",
    input_types={"required": {
        "conditioning": ("CONDITIONING",),
        "style_model": ("STYLE_MODEL",),
        "clip_vision_output": ("CLIP_VISION_OUTPUT",),
    }},
)
def style_model_apply(conditioning, style_model, clip_vision_output):
    # TODO: bridge.apply_style_model()
    raise NotImplementedError("StyleModelApply requires bridge.apply_style_model()")


@registry.register(
    "unCLIPConditioning",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {
        "conditioning": ("CONDITIONING",),
        "clip_vision_output": ("CLIP_VISION_OUTPUT",),
        "strength": ("FLOAT",),
        "noise_augmentation": ("FLOAT",),
    }},
)
def unclip_conditioning(conditioning, clip_vision_output, strength=1.0, noise_augmentation=0.0):
    out = []
    for c in conditioning:
        n = dict(c)
        n["unclip_conditioning"] = {
            "clip_vision_output": clip_vision_output,
            "strength": strength,
            "noise_augmentation": noise_augmentation,
        }
        out.append(n)
    return (out,)


# ---------------------------------------------------------------------------
# Load image as mask
# ---------------------------------------------------------------------------

@registry.register(
    "LoadImageMask",
    return_types=("MASK",),
    category="mask",
    input_types={"required": {
        "image": ("STRING",),
        "channel": ("STRING",),
    }},
)
def load_image_mask(image, channel="alpha"):
    import os
    import numpy as np
    from PIL import Image as PILImage
    from serenityflow.bridge.model_paths import get_model_paths

    if os.path.isabs(image):
        filepath = image
    else:
        paths = get_model_paths()
        filepath = os.path.join(paths.base_dir, "input", image)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image not found: {filepath}")

    img = PILImage.open(filepath).convert("RGBA")
    img_array = np.array(img).astype(np.float32) / 255.0

    ch_map = {"red": 0, "green": 1, "blue": 2, "alpha": 3}
    idx = ch_map.get(channel, 3)
    mask = torch.from_numpy(img_array[:, :, idx]).unsqueeze(0)
    if channel == "alpha":
        mask = 1.0 - mask  # ComfyUI convention: mask=1 means masked
    return (mask,)


# ---------------------------------------------------------------------------
# SD3 / AuraFlow / misc model sampling overrides
# ---------------------------------------------------------------------------

@registry.register(
    "ModelSamplingSD3",
    return_types=("MODEL",),
    category="advanced/model",
    input_types={"required": {
        "model": ("MODEL",),
        "shift": ("FLOAT",),
    }},
)
def model_sampling_sd3(model, shift=3.0):
    if hasattr(model, "with_options"):
        return (model.with_options({"sampling_type": "sd3", "shift": shift}),)
    return (model,)


@registry.register(
    "ModelSamplingAuraFlow",
    return_types=("MODEL",),
    category="advanced/model",
    input_types={"required": {
        "model": ("MODEL",),
        "shift": ("FLOAT",),
    }},
)
def model_sampling_aura_flow(model, shift=1.73):
    if hasattr(model, "with_options"):
        return (model.with_options({"sampling_type": "aura_flow", "shift": shift}),)
    return (model,)


@registry.register(
    "DifferentialDiffusion",
    return_types=("MODEL",),
    category="advanced/model",
    input_types={"required": {"model": ("MODEL",)}},
)
def differential_diffusion(model):
    if hasattr(model, "with_options"):
        return (model.with_options({"differential_diffusion": True}),)
    return (model,)


@registry.register(
    "SkipLayerGuidanceDiT",
    return_types=("MODEL",),
    category="advanced/model",
    input_types={"required": {
        "model": ("MODEL",),
        "skip_blocks": ("STRING",),
        "start_percent": ("FLOAT",),
        "end_percent": ("FLOAT",),
    }},
)
def skip_layer_guidance_dit(model, skip_blocks="", start_percent=0.0, end_percent=1.0):
    if hasattr(model, "with_options"):
        return (model.with_options({
            "skip_layer_guidance": {
                "skip_blocks": skip_blocks,
                "start_percent": start_percent,
                "end_percent": end_percent,
            },
        }),)
    return (model,)


@registry.register(
    "UNetTemporalAttentionMultiply",
    return_types=("MODEL",),
    category="advanced/model",
    input_types={"required": {
        "model": ("MODEL",),
        "self_structural": ("FLOAT",),
        "self_temporal": ("FLOAT",),
        "cross_structural": ("FLOAT",),
        "cross_temporal": ("FLOAT",),
    }},
)
def unet_temporal_attention_multiply(model, self_structural=1.0, self_temporal=1.0,
                                     cross_structural=1.0, cross_temporal=1.0):
    if hasattr(model, "with_options"):
        return (model.with_options({
            "temporal_attention_multiply": {
                "self_structural": self_structural,
                "self_temporal": self_temporal,
                "cross_structural": cross_structural,
                "cross_temporal": cross_temporal,
            },
        }),)
    return (model,)


@registry.register(
    "CFGZeroStar",
    return_types=("MODEL",),
    category="advanced/model",
    input_types={"required": {"model": ("MODEL",)}},
)
def cfg_zero_star(model):
    if hasattr(model, "with_options"):
        return (model.with_options({"cfg_zero_star": True}),)
    return (model,)


@registry.register(
    "EasyCache",
    return_types=("MODEL",),
    category="advanced/model",
    input_types={"required": {"model": ("MODEL",)}},
)
def easy_cache(model):
    if hasattr(model, "with_options"):
        return (model.with_options({"cache_enabled": True}),)
    return (model,)


@registry.register(
    "HunyuanVideo15SuperResolution",
    return_types=("MODEL",),
    category="advanced/model",
    input_types={"required": {
        "model": ("MODEL",),
        "upscale_model": ("LATENT_UPSCALE_MODEL",),
    }},
)
def hunyuan_video_15_super_resolution(model, upscale_model):
    raise NotImplementedError("HunyuanVideo15SuperResolution requires bridge integration")


@registry.register(
    "HunyuanVideo15LatentUpscaleWithModel",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "samples": ("LATENT",),
        "upscale_model": ("LATENT_UPSCALE_MODEL",),
    }},
)
def hunyuan_video_15_latent_upscale_with_model(samples, upscale_model):
    raise NotImplementedError("HunyuanVideo15LatentUpscaleWithModel requires bridge integration")
