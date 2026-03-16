"""RES4LYF / ClownSampler advanced sampling nodes."""
from __future__ import annotations

import torch

from serenityflow.nodes.registry import registry


# ---------------------------------------------------------------------------
# Sampler name choices
# ---------------------------------------------------------------------------

_CLOWN_SAMPLER_NAMES = [
    "euler", "dpmpp_2m", "dpmpp_3m", "dpmpp_2s", "dpmpp_3s", "dpmpp_sde", "heun",
]

_SAMPLER_MODES = ["standard", "unsample", "resample"]

_SHIFT_SCALING = ["exponential", "linear"]


# ---------------------------------------------------------------------------
# Core samplers
# ---------------------------------------------------------------------------

@registry.register(
    "ClownSampler",
    return_types=("LATENT",),
    category="sampling/res4lyf",
    input_types={
        "required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "latent": ("LATENT",),
            "sampler_mode": (_SAMPLER_MODES,),
            "sampler_name": (_CLOWN_SAMPLER_NAMES,),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "denoise_alt": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            "shift": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.01}),
            "shift_scaling": (_SHIFT_SCALING,),
            "eta": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            "d_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 2**53}),
        },
        "optional": {
            "guides": ("GUIDES",),
            "options": ("CLOWN_OPTIONS",),
        },
    },
)
def clown_sampler(model, positive, negative, latent, sampler_mode="standard",
                  sampler_name="euler", steps=20, denoise=1.0, denoise_alt=1.0,
                  cfg=7.5, shift=1.0, base_shift=0.5, shift_scaling="exponential",
                  eta=0.0, s_noise=1.0, d_noise=1.0, noise_seed=0,
                  guides=None, options=None):
    config = {
        "type": "clown_sampler",
        "model": model,
        "positive": positive,
        "negative": negative,
        "latent": latent,
        "sampler_mode": sampler_mode,
        "sampler_name": sampler_name,
        "steps": steps,
        "denoise": denoise,
        "denoise_alt": denoise_alt,
        "cfg": cfg,
        "shift": shift,
        "base_shift": base_shift,
        "shift_scaling": shift_scaling,
        "eta": eta,
        "s_noise": s_noise,
        "d_noise": d_noise,
        "noise_seed": noise_seed,
    }
    if guides is not None:
        config["guides"] = guides
    if options is not None:
        config.update(_merge_options(options))
    # Config node -- actual sampling delegated to bridge/executor at runtime
    return (config,)


@registry.register(
    "ClownSamplerAdvanced",
    return_types=("LATENT",),
    category="sampling/res4lyf",
    input_types={
        "required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "latent": ("LATENT",),
            "sampler_mode": (_SAMPLER_MODES,),
            "sampler_name": (_CLOWN_SAMPLER_NAMES,),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "denoise_alt": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            "shift": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.01}),
            "shift_scaling": (_SHIFT_SCALING,),
            "eta": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            "d_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 2**53}),
            "start_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "end_step": ("INT", {"default": 20, "min": 0, "max": 10000}),
            "add_noise": ("BOOLEAN", {"default": True}),
            "return_leftover_noise": ("BOOLEAN", {"default": False}),
        },
        "optional": {
            "guides": ("GUIDES",),
            "options": ("CLOWN_OPTIONS",),
        },
    },
)
def clown_sampler_advanced(model, positive, negative, latent,
                           sampler_mode="standard", sampler_name="euler",
                           steps=20, denoise=1.0, denoise_alt=1.0,
                           cfg=7.5, shift=1.0, base_shift=0.5,
                           shift_scaling="exponential",
                           eta=0.0, s_noise=1.0, d_noise=1.0, noise_seed=0,
                           start_step=0, end_step=20,
                           add_noise=True, return_leftover_noise=False,
                           guides=None, options=None):
    config = {
        "type": "clown_sampler_advanced",
        "model": model,
        "positive": positive,
        "negative": negative,
        "latent": latent,
        "sampler_mode": sampler_mode,
        "sampler_name": sampler_name,
        "steps": steps,
        "denoise": denoise,
        "denoise_alt": denoise_alt,
        "cfg": cfg,
        "shift": shift,
        "base_shift": base_shift,
        "shift_scaling": shift_scaling,
        "eta": eta,
        "s_noise": s_noise,
        "d_noise": d_noise,
        "noise_seed": noise_seed,
        "start_step": start_step,
        "end_step": end_step,
        "add_noise": add_noise,
        "return_leftover_noise": return_leftover_noise,
    }
    if guides is not None:
        config["guides"] = guides
    if options is not None:
        config.update(_merge_options(options))
    return (config,)


# ---------------------------------------------------------------------------
# Chain sampler
# ---------------------------------------------------------------------------

@registry.register(
    "ChainSampler",
    return_types=("LATENT",),
    category="sampling/res4lyf",
    input_types={
        "required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "latent": ("LATENT",),
            "stage_1_sampler": (_CLOWN_SAMPLER_NAMES,),
            "stage_1_steps": ("INT", {"default": 10, "min": 1, "max": 10000}),
            "stage_1_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "stage_1_cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            "stage_2_sampler": (_CLOWN_SAMPLER_NAMES,),
            "stage_2_steps": ("INT", {"default": 10, "min": 1, "max": 10000}),
            "stage_2_denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "stage_2_cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**53}),
        },
        "optional": {
            "stage_3_sampler": (_CLOWN_SAMPLER_NAMES,),
            "stage_3_steps": ("INT", {"default": 5, "min": 1, "max": 10000}),
            "stage_3_denoise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            "stage_3_cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1}),
        },
    },
)
def chain_sampler(model, positive, negative, latent,
                  stage_1_sampler="euler", stage_1_steps=10,
                  stage_1_denoise=1.0, stage_1_cfg=7.5,
                  stage_2_sampler="dpmpp_2m", stage_2_steps=10,
                  stage_2_denoise=0.5, stage_2_cfg=5.0,
                  seed=0,
                  stage_3_sampler=None, stage_3_steps=5,
                  stage_3_denoise=0.3, stage_3_cfg=3.0):
    stages = [
        {
            "sampler": stage_1_sampler,
            "steps": stage_1_steps,
            "denoise": stage_1_denoise,
            "cfg": stage_1_cfg,
        },
        {
            "sampler": stage_2_sampler,
            "steps": stage_2_steps,
            "denoise": stage_2_denoise,
            "cfg": stage_2_cfg,
        },
    ]
    if stage_3_sampler is not None:
        stages.append({
            "sampler": stage_3_sampler,
            "steps": stage_3_steps,
            "denoise": stage_3_denoise,
            "cfg": stage_3_cfg,
        })
    config = {
        "type": "chain_sampler",
        "model": model,
        "positive": positive,
        "negative": negative,
        "latent": latent,
        "stages": stages,
        "seed": seed,
    }
    return (config,)


# ---------------------------------------------------------------------------
# Guide system
# ---------------------------------------------------------------------------

_GUIDE_TYPES = ["epsilon_projection", "velocity", "v_prediction"]


@registry.register(
    "ClownGuide",
    return_types=("GUIDES",),
    category="sampling/res4lyf/guides",
    input_types={"required": {
        "guide_type": (_GUIDE_TYPES,),
        "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
    }},
)
def clown_guide(guide_type, weight=1.0):
    return ({"type": guide_type, "weight": weight},)


@registry.register(
    "ClownGuides",
    return_types=("GUIDES",),
    category="sampling/res4lyf/guides",
    input_types={
        "required": {
            "guide_1": ("GUIDES",),
        },
        "optional": {
            "guide_2": ("GUIDES",),
            "guide_3": ("GUIDES",),
            "guide_4": ("GUIDES",),
            "guide_5": ("GUIDES",),
        },
    },
)
def clown_guides(guide_1, guide_2=None, guide_3=None, guide_4=None, guide_5=None):
    guides = [guide_1]
    for g in (guide_2, guide_3, guide_4, guide_5):
        if g is not None:
            guides.append(g)
    return (guides,)


@registry.register(
    "ClownGuide_Style",
    return_types=("GUIDES",),
    category="sampling/res4lyf/guides",
    input_types={"required": {
        "style_image": ("IMAGE",),
        "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
        "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
    }},
)
def clown_guide_style(style_image, weight=1.0, start_at=0.0, end_at=1.0):
    return ({
        "type": "style",
        "style_image": style_image,
        "weight": weight,
        "start_at": start_at,
        "end_at": end_at,
    },)


@registry.register(
    "ClownGuide_FrequencySeparation",
    return_types=("GUIDES",),
    category="sampling/res4lyf/guides",
    input_types={"required": {
        "low_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
        "high_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
        "cutoff": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
    }},
)
def clown_guide_frequency_separation(low_weight=1.0, high_weight=1.0, cutoff=0.5):
    return ({
        "type": "frequency_separation",
        "low_weight": low_weight,
        "high_weight": high_weight,
        "cutoff": cutoff,
    },)


@registry.register(
    "ClownGuide_AdaIN_MMDiT",
    return_types=("GUIDES",),
    category="sampling/res4lyf/guides",
    input_types={"required": {
        "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
        "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
    }},
)
def clown_guide_adain_mmdit(weight=1.0, start_at=0.0, end_at=1.0):
    return ({
        "type": "adain_mmdit",
        "weight": weight,
        "start_at": start_at,
        "end_at": end_at,
    },)


# ---------------------------------------------------------------------------
# Options system
# ---------------------------------------------------------------------------

def _merge_options(options):
    """Flatten a CLOWN_OPTIONS dict into config-compatible keys."""
    if isinstance(options, list):
        merged = {}
        for opt in options:
            merged.update(opt)
        return merged
    return dict(options)


@registry.register(
    "ClownOptions_Combine",
    return_types=("CLOWN_OPTIONS",),
    category="sampling/res4lyf/options",
    input_types={
        "required": {
            "options_1": ("CLOWN_OPTIONS",),
        },
        "optional": {
            "options_2": ("CLOWN_OPTIONS",),
            "options_3": ("CLOWN_OPTIONS",),
            "options_4": ("CLOWN_OPTIONS",),
            "options_5": ("CLOWN_OPTIONS",),
        },
    },
)
def clown_options_combine(options_1, options_2=None, options_3=None,
                          options_4=None, options_5=None):
    merged = dict(options_1)
    for opt in (options_2, options_3, options_4, options_5):
        if opt is not None:
            merged.update(opt)
    return (merged,)


@registry.register(
    "ClownOptions_SDE",
    return_types=("CLOWN_OPTIONS",),
    category="sampling/res4lyf/options",
    input_types={"required": {
        "noise_type": (["brownian", "gaussian"],),
        "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
        "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
    }},
)
def clown_options_sde(noise_type="brownian", eta=1.0, s_noise=1.0):
    return ({
        "sde_noise_type": noise_type,
        "sde_eta": eta,
        "sde_s_noise": s_noise,
    },)


@registry.register(
    "ClownOptions_Momentum",
    return_types=("CLOWN_OPTIONS",),
    category="sampling/res4lyf/options",
    input_types={"required": {
        "momentum": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
        "momentum_sign": (["positive", "negative", "alternate"],),
    }},
)
def clown_options_momentum(momentum=0.0, momentum_sign="positive"):
    return ({
        "momentum": momentum,
        "momentum_sign": momentum_sign,
    },)


@registry.register(
    "ClownOptions_Tile",
    return_types=("CLOWN_OPTIONS",),
    category="sampling/res4lyf/options",
    input_types={"required": {
        "tile_width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
        "tile_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
        "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
    }},
)
def clown_options_tile(tile_width=512, tile_height=512, overlap=64):
    return ({
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tile_overlap": overlap,
    },)


@registry.register(
    "ClownOptions_DetailBoost",
    return_types=("CLOWN_OPTIONS",),
    category="sampling/res4lyf/options",
    input_types={"required": {
        "boost_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
        "start_at": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
        "end_at": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
    }},
)
def clown_options_detail_boost(boost_strength=0.5, start_at=0.3, end_at=0.7):
    return ({
        "detail_boost_strength": boost_strength,
        "detail_boost_start_at": start_at,
        "detail_boost_end_at": end_at,
    },)


# ---------------------------------------------------------------------------
# Regional conditioning
# ---------------------------------------------------------------------------

@registry.register(
    "ClownRegionalConditioning",
    return_types=("CONDITIONING",),
    category="sampling/res4lyf/regional",
    input_types={"required": {
        "conditioning": ("CONDITIONING",),
        "mask": ("MASK",),
        "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
    }},
)
def clown_regional_conditioning(conditioning, mask, strength=1.0):
    out = []
    for c in conditioning:
        n = dict(c)
        n["mask"] = mask
        n["strength"] = strength
        n["set_area_to_bounds"] = False
        out.append(n)
    return (out,)


@registry.register(
    "ClownRegionalConditioning2",
    return_types=("CONDITIONING",),
    category="sampling/res4lyf/regional",
    input_types={"required": {
        "conditioning_1": ("CONDITIONING",),
        "mask_1": ("MASK",),
        "conditioning_2": ("CONDITIONING",),
        "mask_2": ("MASK",),
    }},
)
def clown_regional_conditioning_2(conditioning_1, mask_1, conditioning_2, mask_2):
    region_1 = []
    for c in conditioning_1:
        n = dict(c)
        n["mask"] = mask_1
        n["strength"] = 1.0
        n["set_area_to_bounds"] = False
        region_1.append(n)
    region_2 = []
    for c in conditioning_2:
        n = dict(c)
        n["mask"] = mask_2
        n["strength"] = 1.0
        n["set_area_to_bounds"] = False
        region_2.append(n)
    return (region_1 + region_2,)


@registry.register(
    "ClownRegionalConditioning3",
    return_types=("CONDITIONING",),
    category="sampling/res4lyf/regional",
    input_types={"required": {
        "conditioning_1": ("CONDITIONING",),
        "mask_1": ("MASK",),
        "conditioning_2": ("CONDITIONING",),
        "mask_2": ("MASK",),
        "conditioning_3": ("CONDITIONING",),
        "mask_3": ("MASK",),
    }},
)
def clown_regional_conditioning_3(conditioning_1, mask_1,
                                  conditioning_2, mask_2,
                                  conditioning_3, mask_3):
    out = []
    for cond, m in ((conditioning_1, mask_1), (conditioning_2, mask_2),
                    (conditioning_3, mask_3)):
        for c in cond:
            n = dict(c)
            n["mask"] = m
            n["strength"] = 1.0
            n["set_area_to_bounds"] = False
            out.append(n)
    return (out,)


# ---------------------------------------------------------------------------
# Temporal prompting
# ---------------------------------------------------------------------------

@registry.register(
    "ClownOptions_FrameWeights",
    return_types=("CLOWN_OPTIONS",),
    category="sampling/res4lyf/options",
    input_types={"required": {
        "frame_weights": ("STRING", {"default": "0:1.0, 10:0.5, 20:0.8"}),
        "total_frames": ("INT", {"default": 24, "min": 1, "max": 10000}),
    }},
)
def clown_options_frame_weights(frame_weights="0:1.0, 10:0.5, 20:0.8",
                                total_frames=24):
    keyframes = _parse_frame_weights(frame_weights)
    weights = _interpolate_frame_weights(keyframes, total_frames)
    return ({
        "frame_weights": weights,
        "total_frames": total_frames,
    },)


def _parse_frame_weights(spec: str) -> list[tuple[int, float]]:
    """Parse 'frame:weight, frame:weight, ...' into sorted keyframe list."""
    keyframes = []
    for part in spec.split(","):
        part = part.strip()
        if ":" not in part:
            continue
        frame_str, weight_str = part.split(":", 1)
        keyframes.append((int(frame_str.strip()), float(weight_str.strip())))
    keyframes.sort(key=lambda x: x[0])
    return keyframes


def _interpolate_frame_weights(keyframes: list[tuple[int, float]],
                                total_frames: int) -> list[float]:
    """Linearly interpolate between keyframes to produce per-frame weights."""
    if not keyframes:
        return [1.0] * total_frames
    weights = []
    for f in range(total_frames):
        if f <= keyframes[0][0]:
            weights.append(keyframes[0][1])
        elif f >= keyframes[-1][0]:
            weights.append(keyframes[-1][1])
        else:
            # Find surrounding keyframes
            for i in range(len(keyframes) - 1):
                if keyframes[i][0] <= f <= keyframes[i + 1][0]:
                    span = keyframes[i + 1][0] - keyframes[i][0]
                    if span == 0:
                        weights.append(keyframes[i][1])
                    else:
                        t = (f - keyframes[i][0]) / span
                        w = keyframes[i][1] * (1.0 - t) + keyframes[i + 1][1] * t
                        weights.append(w)
                    break
    return weights


# ---------------------------------------------------------------------------
# Utility nodes
# ---------------------------------------------------------------------------

@registry.register(
    "CLIPTextEncodeFluxUnguided",
    return_types=("CONDITIONING",),
    category="sampling/res4lyf/utility",
    input_types={"required": {
        "clip": ("CLIP",),
        "text": ("STRING",),
    }},
)
def clip_text_encode_flux_unguided(clip, text):
    from serenityflow.bridge.serenity_api import encode_text
    conditioning = encode_text(clip, text)
    # Strip guidance metadata if present
    out = []
    for c in conditioning:
        n = dict(c)
        n.pop("guidance", None)
        out.append(n)
    return (out,)


_NOISE_TYPES = ["gaussian", "uniform", "simplex", "fractal"]


@registry.register(
    "AdvancedNoise",
    return_types=("NOISE",),
    category="sampling/res4lyf/noise",
    input_types={"required": {
        "seed": ("INT", {"default": 0, "min": 0, "max": 2**53}),
        "noise_type": (_NOISE_TYPES,),
        "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
    }},
)
def advanced_noise(seed, noise_type="gaussian", scale=1.0):
    return ({
        "type": noise_type,
        "seed": seed,
        "scale": scale,
    },)
