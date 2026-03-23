"""Video model-specific nodes -- LTX-V, WAN, HunyuanVideo schedulers and utilities."""
from __future__ import annotations

import logging

import torch

from serenityflow.nodes.registry import registry
from serenityflow.bridge.types import wrap_latent

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LTX-V Model Loading & Sampling
# ---------------------------------------------------------------------------


@registry.register(
    "LTXVLoader",
    return_types=("LTXV_MODEL",),
    return_names=("ltxv_model",),
    category="loaders/video",
    input_types={"required": {
        "checkpoint_path": ("STRING",),
        "gemma_path": ("STRING",),
    },
    "optional": {
        "dtype": (["bfloat16", "float16"], {"default": "bfloat16"}),
        "spatial_upsampler_path": ("STRING", {"default": ""}),
        "distilled_lora_path": ("STRING", {"default": ""}),
        "lora_path": ("STRING", {"default": ""}),
        "lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
        "quantization": (["auto", "none", "fp8-cast", "fp8-scaled-mm"], {"default": "auto"}),
        "backend": (["auto", "official", "legacy_stagehand"], {"default": "auto"}),
        "serenityfp8_path": ("STRING", {"default": ""}),
    }},
)
def ltxv_loader(
    checkpoint_path,
    gemma_path,
    dtype="bfloat16",
    spatial_upsampler_path="",
    distilled_lora_path="",
    lora_path="",
    lora_strength=1.0,
    quantization="auto",
    backend="auto",
    serenityfp8_path="",
):
    """Load LTX-V 19B model (transformer + VAE + text encoder)."""
    import os
    from serenityflow.bridge.serenity_api import load_ltxv_model

    # Resolve relative names against known model directories
    if not os.path.exists(checkpoint_path):
        for base in [
            os.path.expanduser("~/EriDiffusion/Models/diffusion_models"),
            os.path.expanduser("~/EriDiffusion/Models"),
            os.path.expanduser("~/models"),
            os.path.expanduser("~/models/LTX-2"),
            os.path.expanduser("~/SwarmUI/Models/ltx2"),
        ]:
            candidate = os.path.join(base, checkpoint_path)
            if os.path.exists(candidate):
                checkpoint_path = candidate
                break

    if not os.path.exists(gemma_path):
        for base in [
            os.path.expanduser("~/EriDiffusion/Models/clip"),
            os.path.expanduser("~/EriDiffusion/Models"),
            os.path.expanduser("~/models"),
        ]:
            candidate = os.path.join(base, gemma_path)
            if os.path.isdir(candidate):
                gemma_path = candidate
                break

    # Resolve serenityfp8 slab path
    if serenityfp8_path and not os.path.exists(serenityfp8_path):
        for base in [
            os.path.expanduser("~/.serenity/models/checkpoints"),
            os.path.expanduser("~/serenity/results/serenityfp8"),
        ]:
            candidate = os.path.join(base, serenityfp8_path)
            if os.path.exists(candidate):
                serenityfp8_path = candidate
                break

    log.info("LTXVLoader: checkpoint=%s, gemma=%s, serenityfp8=%s", checkpoint_path, gemma_path, serenityfp8_path or "(none)")
    model = load_ltxv_model(
        checkpoint_path,
        gemma_path,
        dtype=dtype,
        spatial_upsampler_path=spatial_upsampler_path or None,
        distilled_lora_path=distilled_lora_path or None,
        lora_paths=(lora_path,) if lora_path else (),
        lora_strengths=(float(lora_strength),) if lora_path else (),
        quantization=quantization,
        backend="legacy_stagehand" if backend == "legacy_stagehand" else backend,
        serenityfp8_path=serenityfp8_path or "",
    )
    return (model,)


@registry.register(
    "LTXVSampler",
    return_types=("IMAGE", "VIDEO", "AUDIO"),
    return_names=("frames", "video", "audio"),
    category="sampling/video",
    is_output=False,
    input_types={"required": {
        "ltxv_model": ("LTXV_MODEL",),
        "prompt": ("STRING", {"multiline": True}),
        "width": ("INT", {"default": 768, "min": 64, "max": 1920, "step": 32}),
        "height": ("INT", {"default": 512, "min": 64, "max": 1088, "step": 32}),
        "num_frames": ("INT", {"default": 25, "min": 1, "max": 257, "step": 8}),
        "steps": ("INT", {"default": 8, "min": 1, "max": 200}),
        "cfg": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.5}),
        "seed": ("INT", {"default": 42}),
    },
    "optional": {
        "negative_prompt": ("STRING", {"multiline": True}),
        "frame_rate": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 60.0}),
        "stg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0}),
        "mode": (["auto", "distilled", "dev"], {"default": "auto"}),
        "guide_image": ("IMAGE",),
        "guide_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
        "guide_frame_idx": ("INT", {"default": 0, "min": 0, "max": 512}),
        "audio": ("AUDIO",),
        "audio_start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.1}),
        "audio_duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.1}),
        "max_shift": ("FLOAT", {"default": 2.05, "min": 0.0, "max": 10.0, "step": 0.05}),
        "base_shift": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 10.0, "step": 0.05}),
        "decode_timestep": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.005}),
        "decode_noise_scale": ("FLOAT", {"default": 0.025, "min": 0.0, "max": 1.0, "step": 0.005}),
    }},
)
def ltxv_sampler(ltxv_model, prompt, width=768, height=512, num_frames=25,
                 steps=8, cfg=3.0, seed=42, negative_prompt="",
                 frame_rate=25.0, stg_scale=1.0, mode="auto",
                 guide_image=None, guide_strength=1.0, guide_frame_idx=0,
                 audio=None, audio_start_time=0.0, audio_duration=0.0,
                 max_shift=2.05, base_shift=0.95,
                 decode_timestep=0.05, decode_noise_scale=0.025):
    """Generate video frames using LTX-V 19B.

    Mode controls the denoising schedule:
      - "auto": detect from checkpoint name (distilled vs dev)
      - "distilled": fixed 8-step sigma schedule, no CFG
      - "dev": LTX2Scheduler with CFG/STG guidance, more steps

    Returns decoded video frames as IMAGE tensor [F, H, W, C].
    """
    from serenityflow.bridge.serenity_api import sample_ltxv

    result = sample_ltxv(
        model=ltxv_model,
        prompt=prompt,
        negative_prompt=negative_prompt or "",
        width=width,
        height=height,
        num_frames=num_frames,
        steps=steps,
        guidance_scale=cfg,
        stg_scale=stg_scale,
        seed=seed,
        frame_rate=frame_rate,
        mode=mode,
        guide_image=guide_image,
        guide_strength=guide_strength,
        guide_frame_idx=guide_frame_idx,
        audio=audio,
        audio_start_time=audio_start_time,
        audio_duration=audio_duration if audio_duration and audio_duration > 0 else None,
        max_shift=max_shift,
        base_shift=base_shift,
        decode_timestep=decode_timestep,
        decode_noise_scale=decode_noise_scale,
    )

    video = result["video"]
    log.info("Raw video tensor: shape=%s dtype=%s", list(video.shape), video.dtype)

    # vae_decode_video returns uint8 [0-255] in [F,H,W,C] — normalize to float [0,1]
    frames = video.float()
    if video.dtype == torch.uint8 or frames.max() > 1.5:
        frames = frames / 255.0

    # Ensure [F, H, W, C] layout
    if frames.dim() == 5:
        # [B, C, T, H, W] → [T, H, W, C]
        frames = frames[0].permute(1, 2, 3, 0)
    elif frames.dim() == 4:
        if frames.shape[-1] > 4:
            # [C, T, H, W] → [T, H, W, C]
            frames = frames.permute(1, 2, 3, 0)
        # else already [T, H, W, C]
    elif frames.dim() == 3:
        frames = frames.unsqueeze(-1)

    frames = frames.clamp(0, 1)
    log.info("Generated %d frames at %dx%d", frames.shape[0], frames.shape[2], frames.shape[1])
    audio_out = result.get("audio") or {"path": None, "waveform": None, "sample_rate": None}
    video_out = {"frames": frames, "fps": frame_rate, "audio": audio_out}
    return (frames, video_out, audio_out)


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
