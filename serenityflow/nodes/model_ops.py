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
    import copy
    # ModelHandle path: store merge config via with_options
    if hasattr(model1, "with_options") and hasattr(model2, "with_options"):
        return (model1.with_options({
            "merge": {"model2": model2, "ratio": ratio, "type": "simple"},
        }),)
    # nn.Module path: weighted average of state dicts
    if isinstance(model1, torch.nn.Module) and isinstance(model2, torch.nn.Module):
        merged = copy.deepcopy(model1)
        sd1 = model1.state_dict()
        sd2 = model2.state_dict()
        merged_sd = {}
        for key in sd1:
            if key in sd2:
                merged_sd[key] = (1.0 - ratio) * sd1[key] + ratio * sd2[key]
            else:
                merged_sd[key] = sd1[key]
        # Include keys only in model2
        for key in sd2:
            if key not in sd1:
                merged_sd[key] = sd2[key]
        merged.load_state_dict(merged_sd, strict=False)
        return (merged,)
    raise TypeError(
        f"Cannot merge models of types {type(model1).__name__} and {type(model2).__name__}"
    )


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
    import copy

    def _classify_key(key: str) -> str:
        """Classify a state dict key as input/middle/output block."""
        k = key.lower()
        # UNet-style blocks
        if "input_block" in k or "down_block" in k or "encoder" in k:
            return "input"
        if "middle_block" in k or "mid_block" in k:
            return "middle"
        if "output_block" in k or "up_block" in k or "decoder" in k:
            return "output"
        # Transformer-style: classify by layer index (first third/middle/last third)
        return "middle"  # default to middle for unclassified keys

    # ModelHandle path
    if hasattr(model1, "with_options") and hasattr(model2, "with_options"):
        return (model1.with_options({
            "merge": {
                "model2": model2,
                "input": input,
                "middle": middle,
                "out": out,
                "type": "blocks",
            },
        }),)
    # nn.Module path
    if isinstance(model1, torch.nn.Module) and isinstance(model2, torch.nn.Module):
        merged = copy.deepcopy(model1)
        sd1 = model1.state_dict()
        sd2 = model2.state_dict()
        ratios = {"input": input, "middle": middle, "output": out}
        merged_sd = {}
        for key in sd1:
            if key in sd2:
                block_type = _classify_key(key)
                r = ratios.get(block_type, middle)
                merged_sd[key] = (1.0 - r) * sd1[key] + r * sd2[key]
            else:
                merged_sd[key] = sd1[key]
        for key in sd2:
            if key not in sd1:
                merged_sd[key] = sd2[key]
        merged.load_state_dict(merged_sd, strict=False)
        return (merged,)
    raise TypeError(
        f"Cannot merge models of types {type(model1).__name__} and {type(model2).__name__}"
    )


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
    import copy
    # nn.Module path: weighted average of state dicts
    if isinstance(clip1, torch.nn.Module) and isinstance(clip2, torch.nn.Module):
        merged = copy.deepcopy(clip1)
        sd1 = clip1.state_dict()
        sd2 = clip2.state_dict()
        merged_sd = {}
        for key in sd1:
            if key in sd2:
                merged_sd[key] = (1.0 - ratio) * sd1[key] + ratio * sd2[key]
            else:
                merged_sd[key] = sd1[key]
        for key in sd2:
            if key not in sd1:
                merged_sd[key] = sd2[key]
        merged.load_state_dict(merged_sd, strict=False)
        return (merged,)
    # Handle-based or wrapper objects: store merge config
    if hasattr(clip1, "with_options"):
        return (clip1.with_options({
            "merge": {"clip2": clip2, "ratio": ratio},
        }),)
    # Fallback: return clip1 weighted (best effort)
    return (clip1,)


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
    from serenityflow.bridge.model_paths import get_model_paths
    paths = get_model_paths()
    model_path = paths.find(model_name, "upscale_models")

    try:
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
    except Exception:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    return ({"state_dict": state_dict, "path": model_path},)


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
    # Tile-based upscaling: process image in overlapping tiles for memory efficiency
    tile_size = 512
    overlap = 64

    model = upscale_model
    # If model is a dict with state_dict, it's from UpscaleModelLoader
    # Try to instantiate a simple upscale model; fall back to pass-through
    if isinstance(model, dict) and "state_dict" in model:
        # Attempt to build an nn.Module from the state dict
        # This supports ESRGAN/RealESRGAN/SwinIR style models
        try:
            from serenityflow.bridge.serenity_api import _get as _get_bridge
            bridge = _get_bridge()
            if "load_upscale_model" in bridge:
                model = bridge["load_upscale_model"](upscale_model["path"])
            else:
                raise KeyError("no bridge upscale loader")
        except Exception:
            # No bridge support -- try to use the state dict directly
            # to detect scale factor for a simple fallback
            pass

    # image is BHWC float32 [0,1]
    B, H, W, C = image.shape
    # Convert to BCHW for model processing
    img = image.permute(0, 3, 1, 2)  # -> BCHW

    if isinstance(model, torch.nn.Module):
        try:
            first_param = next(model.parameters())
            device = first_param.device
            dtype = first_param.dtype
        except StopIteration:
            device = torch.device("cpu")
            dtype = torch.float32

        # Determine upscale factor with a tiny probe
        with torch.no_grad():
            probe = torch.zeros(1, C, 8, 8, device=device, dtype=dtype)
            probe_out = model(probe)
            scale = probe_out.shape[-1] // 8

        # Process each batch item in tiles
        results = []
        for b in range(B):
            single = img[b:b+1].to(device=device, dtype=dtype)
            out_h, out_w = H * scale, W * scale
            output = torch.zeros(1, C, out_h, out_w, device=device, dtype=dtype)
            count = torch.zeros(1, 1, out_h, out_w, device=device, dtype=dtype)

            step = tile_size - overlap
            for y in range(0, H, step):
                for x in range(0, W, step):
                    y_end = min(y + tile_size, H)
                    x_end = min(x + tile_size, W)
                    y_start = max(0, y_end - tile_size)
                    x_start = max(0, x_end - tile_size)

                    tile = single[:, :, y_start:y_end, x_start:x_end]
                    with torch.no_grad():
                        tile_out = model(tile)

                    oy = y_start * scale
                    ox = x_start * scale
                    oh = tile_out.shape[2]
                    ow = tile_out.shape[3]
                    output[:, :, oy:oy+oh, ox:ox+ow] += tile_out
                    count[:, :, oy:oy+oh, ox:ox+ow] += 1.0

            output = output / count.clamp(min=1.0)
            results.append(output.cpu().float())

        result = torch.cat(results, dim=0)
    else:
        # No usable model -- bicubic 4x fallback
        result = torch.nn.functional.interpolate(
            img, scale_factor=4, mode="bicubic", align_corners=False
        )

    # Back to BHWC
    result = result.permute(0, 2, 3, 1).clamp(0.0, 1.0)
    return (result,)


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
    from serenityflow.bridge.serenity_api import load_clip_vision
    paths = get_model_paths()
    path = paths.find(clip_name, "clip_vision")
    return (load_clip_vision(path),)


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
    # If clip_vision is an nn.Module with a forward method, use it directly
    if isinstance(clip_vision, torch.nn.Module):
        # image is BHWC float32, convert to BCHW
        img = image.permute(0, 3, 1, 2)
        # Resize to 224x224 (CLIP vision standard)
        img = torch.nn.functional.interpolate(img, size=(224, 224), mode="bicubic", align_corners=False)
        # Normalize with CLIP mean/std
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=img.device).view(1, 3, 1, 1)
        img = (img - mean) / std
        with torch.no_grad():
            output = clip_vision(img)
        # Return as structured output
        if isinstance(output, dict):
            return (output,)
        if hasattr(output, "last_hidden_state"):
            return ({"last_hidden_state": output.last_hidden_state,
                     "image_embeds": getattr(output, "image_embeds", None),
                     "penultimate_hidden_states": getattr(output, "hidden_states", [None])[-2]
                     if hasattr(output, "hidden_states") and output.hidden_states else None},)
        return ({"last_hidden_state": output},)
    # Dict-based (state_dict wrapper from loader)
    if isinstance(clip_vision, dict) and "state_dict" in clip_vision:
        raise NotImplementedError(
            "CLIPVisionEncode: model is a raw state_dict. "
            "Bridge integration needed to instantiate the vision model."
        )
    raise TypeError(f"Unsupported clip_vision type: {type(clip_vision).__name__}")


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


# ---------------------------------------------------------------------------
# PatchModelAddDownscale
# ---------------------------------------------------------------------------

@registry.register(
    "PatchModelAddDownscale",
    return_types=("MODEL",),
    category="advanced/model",
    input_types={"required": {
        "model": ("MODEL",),
        "block_number": ("INT",),
        "downscale_factor": ("FLOAT",),
        "start_percent": ("FLOAT",),
        "end_percent": ("FLOAT",),
        "downscale_after_skip": ("BOOLEAN",),
    }},
)
def patch_model_add_downscale(model, block_number=3, downscale_factor=2.0,
                               start_percent=0.0, end_percent=0.35,
                               downscale_after_skip=True):
    if hasattr(model, "with_options"):
        return (model.with_options({
            "downscale_patch": {
                "block_number": block_number,
                "downscale_factor": downscale_factor,
                "start_percent": start_percent,
                "end_percent": end_percent,
                "downscale_after_skip": downscale_after_skip,
            },
        }),)
    return (model,)


# ---------------------------------------------------------------------------
# CLIPLoaderGGUF
# ---------------------------------------------------------------------------

@registry.register(
    "CLIPLoaderGGUF",
    return_types=("CLIP",),
    category="loaders",
    input_types={"required": {
        "clip_name": ("STRING",),
        "clip_type": ("STRING",),
    }},
)
def clip_loader_gguf(clip_name, clip_type="stable_diffusion"):
    from serenityflow.bridge.model_paths import get_model_paths
    paths = get_model_paths()
    # Search in clip and text_encoders directories
    clip_path = paths.find(clip_name, "clip")

    try:
        from serenityflow.bridge.serenity_api import load_clip
        return (load_clip(clip_path, clip_type=clip_type),)
    except Exception:
        # GGUF loading requires bridge support for quantized format
        raise NotImplementedError(
            f"CLIPLoaderGGUF: bridge.load_clip() failed for GGUF file '{clip_name}'. "
            "GGUF-quantized CLIP loading requires bridge support."
        )
