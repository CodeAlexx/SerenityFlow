"""FLUX-specific nodes -- latents, guidance, scheduler, KV cache."""
from __future__ import annotations

import torch

from serenityflow.nodes.registry import registry
from serenityflow.bridge.types import wrap_latent


@registry.register(
    "EmptyFlux2LatentImage",
    return_types=("LATENT",),
    category="latent/flux",
    input_types={"required": {
        "width": ("INT",), "height": ("INT",), "batch_size": ("INT",),
    }},
)
def empty_flux2_latent(width, height, batch_size=1):
    # FLUX uses 16-channel latent at 1/8 resolution
    latent = torch.zeros(batch_size, 16, height // 8, width // 8)
    return (wrap_latent(latent),)


@registry.register(
    "FluxDisableGuidance",
    return_types=("CONDITIONING",),
    category="conditioning/flux",
    input_types={"required": {"conditioning": ("CONDITIONING",)}},
)
def flux_disable_guidance(conditioning):
    out = []
    for c in conditioning:
        n = dict(c)
        n["guidance"] = 0.0
        out.append(n)
    return (out,)


@registry.register(
    "Flux2Scheduler",
    return_types=("SIGMAS",),
    category="sampling/custom_sampling/schedulers",
    input_types={"required": {
        "steps": ("INT",),
        "shift": ("FLOAT",),
        "denoise": ("FLOAT",),
    }},
)
def flux2_scheduler(steps, shift=1.0, denoise=1.0):
    # Flux uses shifted linear sigmas in flow-matching space
    total_steps = steps
    if denoise < 1.0:
        total_steps = int(steps / denoise)
    sigmas = torch.linspace(1.0, 0.0, total_steps + 1)
    # Apply shift
    if shift != 1.0:
        sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    # Truncate for denoise < 1.0
    if denoise < 1.0:
        sigmas = sigmas[-(steps + 1):]
    return (sigmas,)


@registry.register(
    "FluxKVCache",
    return_types=("MODEL",),
    category="advanced/flux",
    input_types={"required": {
        "model": ("MODEL",),
        "cache_steps": ("INT",),
    }},
)
def flux_kv_cache(model, cache_steps=2):
    if hasattr(model, "with_options"):
        return (model.with_options({"kv_cache_steps": cache_steps}),)
    return (model,)
