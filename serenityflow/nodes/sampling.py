"""Sampling nodes -- KSampler, KSamplerAdvanced."""
from __future__ import annotations

from serenityflow.nodes.registry import registry


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
def ksampler(model, seed, steps, cfg, sampler_name, scheduler,
             positive, negative, latent_image, denoise=1.0):
    from serenityflow.bridge.serenity_api import sample
    from serenityflow.bridge.types import unwrap_latent, wrap_latent

    latent = unwrap_latent(latent_image)

    result = sample(
        model=model, latent=latent,
        positive=positive, negative=negative,
        seed=seed, steps=steps, cfg=cfg,
        sampler_name=sampler_name, scheduler=scheduler,
        denoise=denoise,
    )

    return (wrap_latent(result),)


@registry.register(
    "KSamplerAdvanced",
    return_types=("LATENT",),
    category="sampling",
    input_types={"required": {
        "model": ("MODEL",), "add_noise": ("STRING",), "noise_seed": ("INT",),
        "steps": ("INT",), "cfg": ("FLOAT",), "sampler_name": ("STRING",),
        "scheduler": ("STRING",), "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",), "latent_image": ("LATENT",),
        "start_at_step": ("INT",), "end_at_step": ("INT",),
        "return_with_leftover_noise": ("STRING",),
    }},
)
def ksampler_advanced(model, add_noise, noise_seed, steps, cfg,
                      sampler_name, scheduler, positive, negative,
                      latent_image, start_at_step, end_at_step,
                      return_with_leftover_noise="disable"):
    from serenityflow.bridge.serenity_api import sample
    from serenityflow.bridge.types import unwrap_latent, wrap_latent

    latent = unwrap_latent(latent_image)

    result = sample(
        model=model, latent=latent,
        positive=positive, negative=negative,
        seed=noise_seed, steps=steps, cfg=cfg,
        sampler_name=sampler_name, scheduler=scheduler,
        denoise=1.0,
        start_step=start_at_step,
        end_step=end_at_step,
        add_noise=(add_noise == "enable"),
        return_with_leftover_noise=(return_with_leftover_noise == "enable"),
    )

    return (wrap_latent(result),)
