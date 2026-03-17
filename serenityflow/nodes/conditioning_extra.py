"""Extra conditioning nodes -- average, schedule, GLIGEN, IP-Adapter."""
from __future__ import annotations

import torch

from serenityflow.nodes.registry import registry
from serenityflow.bridge.types import find_cross_attn_key


@registry.register(
    "ConditioningAverage",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {
        "conditioning_to": ("CONDITIONING",),
        "conditioning_from": ("CONDITIONING",),
        "conditioning_to_strength": ("FLOAT",),
    }},
)
def conditioning_average(conditioning_to, conditioning_from, conditioning_to_strength=1.0):
    out = []
    for i in range(min(len(conditioning_to), len(conditioning_from))):
        ct = dict(conditioning_to[i])
        cf = conditioning_from[i]
        key_to = find_cross_attn_key(ct)
        key_from = find_cross_attn_key(cf)
        if key_to and key_from:
            tw = conditioning_to_strength
            # Pad shorter to match longer
            t_emb = ct[key_to]
            f_emb = cf[key_from]
            if t_emb.shape[1] != f_emb.shape[1]:
                max_len = max(t_emb.shape[1], f_emb.shape[1])
                if t_emb.shape[1] < max_len:
                    t_emb = torch.nn.functional.pad(t_emb, (0, 0, 0, max_len - t_emb.shape[1]))
                if f_emb.shape[1] < max_len:
                    f_emb = torch.nn.functional.pad(f_emb, (0, 0, 0, max_len - f_emb.shape[1]))
            ct[key_to] = t_emb * tw + f_emb * (1.0 - tw)
        out.append(ct)
    return (out,)


@registry.register(
    "ConditioningSetAreaStrength",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {
        "conditioning": ("CONDITIONING",),
        "strength": ("FLOAT",),
    }},
)
def conditioning_set_area_strength(conditioning, strength=1.0):
    out = []
    for c in conditioning:
        n = dict(c)
        n["strength"] = strength
        out.append(n)
    return (out,)


@registry.register(
    "ConditioningCombineMultiple",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={
        "required": {"conditioning_1": ("CONDITIONING",)},
        "optional": {
            "conditioning_2": ("CONDITIONING",),
            "conditioning_3": ("CONDITIONING",),
            "conditioning_4": ("CONDITIONING",),
            "conditioning_5": ("CONDITIONING",),
        },
    },
)
def conditioning_combine_multiple(conditioning_1, conditioning_2=None,
                                   conditioning_3=None, conditioning_4=None,
                                   conditioning_5=None):
    result = list(conditioning_1)
    for c in [conditioning_2, conditioning_3, conditioning_4, conditioning_5]:
        if c is not None:
            result.extend(c)
    return (result,)


@registry.register(
    "GLIGENTextBoxApply",
    return_types=("CONDITIONING",),
    category="conditioning/gligen",
    input_types={"required": {
        "conditioning_to": ("CONDITIONING",),
        "clip": ("CLIP",),
        "gligen_textbox_model": ("GLIGEN",),
        "text": ("STRING",),
        "width": ("INT",), "height": ("INT",),
        "x": ("INT",), "y": ("INT",),
    }},
)
def gligen_textbox_apply(conditioning_to, clip, gligen_textbox_model, text,
                         width, height, x, y):
    out = []
    for c in conditioning_to:
        n = dict(c)
        n["gligen"] = {
            "model": gligen_textbox_model,
            "text": text,
            "area": (x, y, width, height),
        }
        out.append(n)
    return (out,)


@registry.register(
    "GLIGENLoader",
    return_types=("GLIGEN",),
    category="loaders",
    input_types={"required": {"gligen_name": ("STRING",)}},
)
def gligen_loader(gligen_name):
    # TODO: bridge.load_gligen()
    raise NotImplementedError("GLIGENLoader requires bridge.load_gligen()")


@registry.register(
    "IPAdapterApply",
    return_types=("MODEL",),
    category="conditioning/ipadapter",
    input_types={"required": {
        "model": ("MODEL",),
        "clip_vision_output": ("CLIP_VISION_OUTPUT",),
        "weight": ("FLOAT",),
        "weight_type": ("STRING",),
    },
    "optional": {
        "start_at": ("FLOAT",),
        "end_at": ("FLOAT",),
    }},
)
def ipadapter_apply(model, clip_vision_output, weight=1.0,
                    weight_type="linear", start_at=0.0, end_at=1.0):
    if hasattr(model, "with_options"):
        return (model.with_options({
            "ipadapter": {
                "clip_vision_output": clip_vision_output,
                "weight": weight,
                "weight_type": weight_type,
                "start_at": start_at,
                "end_at": end_at,
            },
        }),)
    return (model,)


@registry.register(
    "ConditioningSetDefault",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {
        "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",),
    }},
)
def conditioning_set_default(positive, negative):
    # Pass-through that labels positive/negative for UI clarity
    return (positive,)


@registry.register(
    "LTXVConditioning",
    return_types=("CONDITIONING", "CONDITIONING"),
    category="conditioning",
    input_types={
        "required": {
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
        },
        "optional": {
            "frame_rate": ("FLOAT",),
        },
    },
)
def ltxv_conditioning(positive, negative, frame_rate=25.0):
    pos_out = []
    for c in positive:
        n = dict(c)
        n["frame_rate"] = frame_rate
        pos_out.append(n)
    neg_out = []
    for c in negative:
        n = dict(c)
        n["frame_rate"] = frame_rate
        neg_out.append(n)
    return (pos_out, neg_out)


@registry.register(
    "SVD_img2vid_Conditioning",
    return_types=("CONDITIONING", "CONDITIONING", "LATENT"),
    category="conditioning",
    input_types={"required": {
        "clip_vision": ("CLIP_VISION",),
        "init_image": ("IMAGE",),
        "vae": ("VAE",),
        "width": ("INT",), "height": ("INT",),
        "video_frames": ("INT",),
        "motion_bucket_id": ("INT",),
        "fps": ("INT",),
        "augmentation_level": ("FLOAT",),
    }},
)
def svd_img2vid_conditioning(clip_vision, init_image, vae, width, height,
                             video_frames, motion_bucket_id, fps, augmentation_level):
    raise NotImplementedError("SVD_img2vid_Conditioning requires bridge.encode_svd()")


@registry.register(
    "LoraLoaderStack",
    return_types=("MODEL", "CLIP"),
    category="loaders",
    input_types={"required": {
        "model": ("MODEL",),
        "clip": ("CLIP",),
        "lora_name_1": ("STRING",),
        "strength_1": ("FLOAT",),
    },
    "optional": {
        "lora_name_2": ("STRING",),
        "strength_2": ("FLOAT",),
        "lora_name_3": ("STRING",),
        "strength_3": ("FLOAT",),
        "lora_name_4": ("STRING",),
        "strength_4": ("FLOAT",),
        "lora_name_5": ("STRING",),
        "strength_5": ("FLOAT",),
    }},
)
def lora_loader_stack(model, clip, lora_name_1, strength_1=1.0,
                      lora_name_2=None, strength_2=1.0,
                      lora_name_3=None, strength_3=1.0,
                      lora_name_4=None, strength_4=1.0,
                      lora_name_5=None, strength_5=1.0):
    from serenityflow.bridge.serenity_api import apply_lora, apply_lora_clip
    from serenityflow.bridge.model_paths import get_model_paths
    paths = get_model_paths()

    for lora_name, strength in [(lora_name_1, strength_1),
                                 (lora_name_2, strength_2),
                                 (lora_name_3, strength_3),
                                 (lora_name_4, strength_4),
                                 (lora_name_5, strength_5)]:
        if lora_name:
            lora_path = paths.find(lora_name, "loras")
            model = apply_lora(model, lora_path, strength=strength)
            clip = apply_lora_clip(clip, lora_path, strength=strength)
    return (model, clip)


@registry.register(
    "LatentFromBatch",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "samples": ("LATENT",),
        "batch_index": ("INT",),
        "length": ("INT",),
    }},
)
def latent_from_batch(samples, batch_index=0, length=1):
    from serenityflow.bridge.types import unwrap_latent, wrap_latent
    latent = unwrap_latent(samples)
    end = min(batch_index + length, latent.shape[0])
    batch_index = max(0, batch_index)
    return (wrap_latent(latent[batch_index:end]),)


@registry.register(
    "LatentInterpolate",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "samples1": ("LATENT",),
        "samples2": ("LATENT",),
        "ratio": ("FLOAT",),
    }},
)
def latent_interpolate(samples1, samples2, ratio=0.5):
    from serenityflow.bridge.types import unwrap_latent, wrap_latent
    l1 = unwrap_latent(samples1)
    l2 = unwrap_latent(samples2)
    result = l1 * (1.0 - ratio) + l2 * ratio
    return (wrap_latent(result),)


@registry.register(
    "LatentAdd",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "samples1": ("LATENT",),
        "samples2": ("LATENT",),
    }},
)
def latent_add(samples1, samples2):
    from serenityflow.bridge.types import unwrap_latent, wrap_latent
    return (wrap_latent(unwrap_latent(samples1) + unwrap_latent(samples2)),)


@registry.register(
    "LatentSubtract",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "samples1": ("LATENT",),
        "samples2": ("LATENT",),
    }},
)
def latent_subtract(samples1, samples2):
    from serenityflow.bridge.types import unwrap_latent, wrap_latent
    return (wrap_latent(unwrap_latent(samples1) - unwrap_latent(samples2)),)


@registry.register(
    "LatentMultiply",
    return_types=("LATENT",),
    category="latent",
    input_types={"required": {
        "samples": ("LATENT",),
        "multiplier": ("FLOAT",),
    }},
)
def latent_multiply(samples, multiplier=1.0):
    from serenityflow.bridge.types import unwrap_latent, wrap_latent
    return (wrap_latent(unwrap_latent(samples) * multiplier),)


@registry.register(
    "CheckpointSave",
    return_types=(),
    category="advanced",
    is_output=True,
    input_types={"required": {
        "model": ("MODEL",),
        "clip": ("CLIP",),
        "vae": ("VAE",),
        "filename_prefix": ("STRING",),
    }},
)
def checkpoint_save(model, clip, vae, filename_prefix="checkpoint"):
    # TODO: bridge.save_checkpoint()
    raise NotImplementedError("CheckpointSave requires bridge.save_checkpoint()")


@registry.register(
    "CLIPSave",
    return_types=(),
    category="advanced",
    is_output=True,
    input_types={"required": {
        "clip": ("CLIP",),
        "filename_prefix": ("STRING",),
    }},
)
def clip_save(clip, filename_prefix="clip"):
    # TODO: bridge.save_clip()
    raise NotImplementedError("CLIPSave requires bridge.save_clip()")


@registry.register(
    "VAESave",
    return_types=(),
    category="advanced",
    is_output=True,
    input_types={"required": {
        "vae": ("VAE",),
        "filename_prefix": ("STRING",),
    }},
)
def vae_save(vae, filename_prefix="vae"):
    # TODO: bridge.save_vae()
    raise NotImplementedError("VAESave requires bridge.save_vae()")


@registry.register(
    "ReferenceLatent",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {
        "conditioning": ("CONDITIONING",),
        "latent": ("LATENT",),
    }},
)
def reference_latent(conditioning, latent):
    """Attach a reference latent to conditioning for Flux2/Klein image editing.

    The model reads the reference through attention — the latent is packed
    into the conditioning dict so the sampler can concatenate reference tokens
    with noise tokens during each forward pass.
    """
    from serenityflow.bridge.types import unwrap_latent
    ref = unwrap_latent(latent)
    out = []
    for c in conditioning:
        n = dict(c)
        n["reference_latent"] = ref
        out.append(n)
    return (out,)


@registry.register(
    "TextEncodeQwenImageEditPlus",
    return_types=("CONDITIONING",),
    category="conditioning",
    input_types={"required": {
        "clip": ("CLIP",),
        "text": ("STRING", {"multiline": True}),
        "image": ("IMAGE",),
    }},
)
def text_encode_qwen_image_edit_plus(clip, text, image):
    """Encode text + image together for Qwen image edit models.

    Qwen edit models expect the source image context alongside the edit
    instruction for conditional generation.
    """
    from serenityflow.bridge.serenity_api import encode_text
    from serenityflow.bridge.types import bhwc_to_bchw

    conditioning = encode_text(clip, text)
    # Attach the source image to conditioning for the edit model
    image_bchw = bhwc_to_bchw(image) if image.ndim == 4 and image.shape[-1] in (1, 3, 4) else image
    out = []
    for c in conditioning:
        n = dict(c)
        n["edit_image"] = image_bchw
        out.append(n)
    return (out,)
