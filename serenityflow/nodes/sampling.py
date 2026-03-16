"""Sampling nodes -- KSampler, KSamplerAdvanced."""
from __future__ import annotations

from serenityflow.nodes.registry import registry


_SAMPLER_NAMES = ["euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral",
                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde",
                  "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
                  "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ipndm", "ipndm_v",
                  "deis", "ddim", "uni_pc", "uni_pc_bh2"]
_SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple",
                    "ddim_uniform", "beta", "linear_quadratic"]

@registry.register(
    "KSampler",
    return_types=("LATENT",),
    category="sampling",
    input_types={
        "required": {
            "model": ("MODEL",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**53}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            "sampler_name": (_SAMPLER_NAMES,),
            "scheduler": (_SCHEDULER_NAMES,),
            "positive": ("CONDITIONING",),
            "latent_image": ("LATENT",),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        },
        "optional": {
            "negative": ("CONDITIONING",),
            "lanpaint_mode": ("BOOLEAN", {"default": False,
                "tooltip": "When enabled with a noise mask, use LanPaint Langevin dynamics for higher quality inpainting."}),
            "lanpaint_thinking_steps": ("INT", {"default": 5, "min": 1, "max": 50,
                "tooltip": "LanPaint reasoning iterations per denoise step."}),
        },
    },
)
def ksampler(model, seed, steps, cfg, sampler_name, scheduler,
             positive, latent_image, denoise=1.0, negative=None,
             lanpaint_mode=False, lanpaint_thinking_steps=5):
    from serenityflow.bridge.types import unwrap_latent, wrap_latent

    latent = unwrap_latent(latent_image)
    noise_mask = latent_image.get("noise_mask") if isinstance(latent_image, dict) else None

    # Auto-route to LanPaint when enabled and a mask is present
    if lanpaint_mode and noise_mask is not None:
        from serenityflow.nodes.sampling_lanpaint import _sample_with_lanpaint
        result = _sample_with_lanpaint(
            model=model, latent=latent, positive=positive, negative=negative,
            seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name,
            scheduler=scheduler, denoise=denoise,
            thinking_steps=lanpaint_thinking_steps, lambda_strength=7.0,
            friction=15.0, step_size=0.3, beta=1.0, mode="image_first",
            early_stop_threshold=0.0, early_stop_patience=1,
            noise_mask=noise_mask,
        )
        return (wrap_latent(result),)

    from serenityflow.bridge.serenity_api import sample
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
