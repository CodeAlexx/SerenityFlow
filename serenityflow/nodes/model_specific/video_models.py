"""Video model-specific nodes -- LTX-V, WAN, HunyuanVideo schedulers and utilities."""
from __future__ import annotations

import torch

from serenityflow.nodes.registry import registry
from serenityflow.bridge.types import wrap_latent


@registry.register(
    "LTXVScheduler",
    return_types=("SIGMAS",),
    category="sampling/custom_sampling/schedulers",
    input_types={"required": {
        "steps": ("INT",),
        "shift": ("FLOAT",),
        "denoise": ("FLOAT",),
    }},
)
def ltxv_scheduler(steps, shift=5.0, denoise=1.0):
    # LTX-Video uses shifted flow-matching sigmas
    total_steps = steps
    if denoise < 1.0:
        total_steps = int(steps / denoise)
    sigmas = torch.linspace(1.0, 0.0, total_steps + 1)
    if shift != 1.0:
        sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    if denoise < 1.0:
        sigmas = sigmas[-(steps + 1):]
    return (sigmas,)


@registry.register(
    "LTXVAddGuide",
    return_types=("GUIDER",),
    category="sampling/custom_sampling/guiders",
    input_types={"required": {
        "guider": ("GUIDER",),
        "guide_image": ("IMAGE",),
        "guide_strength": ("FLOAT",),
    }},
)
def ltxv_add_guide(guider, guide_image, guide_strength=1.0):
    result = dict(guider)
    result["guide_image"] = guide_image
    result["guide_strength"] = guide_strength
    return (result,)


@registry.register(
    "WanImageToVideo",
    return_types=("CONDITIONING",),
    category="conditioning/wan",
    input_types={"required": {
        "conditioning": ("CONDITIONING",),
        "image": ("IMAGE",),
    },
    "optional": {
        "strength": ("FLOAT",),
    }},
)
def wan_image_to_video(conditioning, image, strength=1.0):
    out = []
    for c in conditioning:
        n = dict(c)
        n["wan_guide_image"] = image
        n["wan_guide_strength"] = strength
        out.append(n)
    return (out,)


@registry.register(
    "WanCameraControl",
    return_types=("CONDITIONING",),
    category="conditioning/wan",
    input_types={"required": {
        "conditioning": ("CONDITIONING",),
        "camera_type": ("STRING",),
        "speed": ("FLOAT",),
    }},
)
def wan_camera_control(conditioning, camera_type="pan_left", speed=1.0):
    out = []
    for c in conditioning:
        n = dict(c)
        n["wan_camera_type"] = camera_type
        n["wan_camera_speed"] = speed
        out.append(n)
    return (out,)


@registry.register(
    "WanVaceToVideo",
    return_types=("CONDITIONING",),
    category="conditioning/wan",
    input_types={"required": {
        "conditioning": ("CONDITIONING",),
        "vace_image": ("IMAGE",),
    },
    "optional": {
        "vace_mask": ("MASK",),
        "strength": ("FLOAT",),
    }},
)
def wan_vace_to_video(conditioning, vace_image, vace_mask=None, strength=1.0):
    out = []
    for c in conditioning:
        n = dict(c)
        n["wan_vace_image"] = vace_image
        if vace_mask is not None:
            n["wan_vace_mask"] = vace_mask
        n["wan_vace_strength"] = strength
        out.append(n)
    return (out,)


@registry.register(
    "WanFunControlToVideo",
    return_types=("CONDITIONING",),
    category="conditioning/wan",
    input_types={"required": {
        "conditioning": ("CONDITIONING",),
        "control_video": ("IMAGE",),
        "control_type": ("STRING",),
    },
    "optional": {
        "strength": ("FLOAT",),
    }},
)
def wan_fun_control_to_video(conditioning, control_video, control_type="depth", strength=1.0):
    out = []
    for c in conditioning:
        n = dict(c)
        n["wan_control_video"] = control_video
        n["wan_control_type"] = control_type
        n["wan_control_strength"] = strength
        out.append(n)
    return (out,)


@registry.register(
    "HunyuanVideoSampler",
    return_types=("LATENT",),
    category="sampling/hunyuan",
    input_types={"required": {
        "model": ("MODEL",),
        "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",),
        "latent_image": ("LATENT",),
        "seed": ("INT",),
        "steps": ("INT",),
        "cfg": ("FLOAT",),
        "denoise": ("FLOAT",),
    }},
)
def hunyuan_video_sampler(model, positive, negative, latent_image,
                          seed=0, steps=30, cfg=6.0, denoise=1.0):
    # TODO: bridge.sample() with hunyuan-specific options
    raise NotImplementedError("HunyuanVideoSampler requires bridge video sampling")


@registry.register(
    "VideoLinearCFGGuidance",
    return_types=("MODEL",),
    category="sampling/video",
    input_types={"required": {
        "model": ("MODEL",),
        "min_cfg": ("FLOAT",),
    }},
)
def video_linear_cfg_guidance(model, min_cfg=1.0):
    if hasattr(model, "with_options"):
        return (model.with_options({"video_linear_cfg_min": min_cfg}),)
    return (model,)


@registry.register(
    "WanFunInpaintToVideo",
    return_types=("LATENT",),
    category="sampling/video",
    input_types={"required": {
        "model": ("MODEL",),
        "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",),
        "vae": ("VAE",),
        "image": ("IMAGE",),
        "mask": ("MASK",),
        "latent": ("LATENT",),
    }},
)
def wan_fun_inpaint_to_video(model, positive, negative, vae, image, mask, latent):
    raise NotImplementedError("WanFunInpaintToVideo requires bridge video inpainting")


@registry.register(
    "WanCameraEmbedding",
    return_types=("WAN_CAMERA_EMBEDDING",),
    category="conditioning/video",
    input_types={"required": {"camera_poses": ("STRING",)}},
)
def wan_camera_embedding(camera_poses):
    return ({"camera_poses": camera_poses},)


@registry.register(
    "WanCameraImageToVideo",
    return_types=("LATENT",),
    category="sampling/video",
    input_types={"required": {
        "model": ("MODEL",),
        "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",),
        "vae": ("VAE",),
        "image": ("IMAGE",),
        "camera_embedding": ("WAN_CAMERA_EMBEDDING",),
        "latent": ("LATENT",),
    }},
)
def wan_camera_image_to_video(model, positive, negative, vae, image, camera_embedding, latent):
    raise NotImplementedError("WanCameraImageToVideo requires bridge camera-guided video sampling")


@registry.register(
    "Wan22FunControlToVideo",
    return_types=("MODEL",),
    category="conditioning/video",
    input_types={"required": {
        "model": ("MODEL",),
        "control_video": ("IMAGE",),
    }},
)
def wan22_fun_control_to_video(model, control_video):
    raise NotImplementedError("Wan22FunControlToVideo requires bridge video control")


@registry.register(
    "WanFirstLastFrameToVideo",
    return_types=("LATENT",),
    category="sampling/video",
    input_types={"required": {
        "model": ("MODEL",),
        "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",),
        "vae": ("VAE",),
        "first_image": ("IMAGE",),
        "last_image": ("IMAGE",),
        "latent": ("LATENT",),
    }},
)
def wan_first_last_frame_to_video(model, positive, negative, vae, first_image, last_image, latent):
    raise NotImplementedError("WanFirstLastFrameToVideo requires bridge first/last frame video sampling")
