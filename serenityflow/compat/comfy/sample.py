"""Compatibility shim for comfy.sample.

High-level sampling functions.
"""
from __future__ import annotations

import torch


def prepare_noise(latent_image, seed, noise_inds=None):
    generator = torch.manual_seed(seed)
    noise = torch.randn(
        latent_image.size(), dtype=latent_image.dtype,
        layout=latent_image.layout, generator=generator,
    )
    return noise


def cleanup_additional_models(models):
    for m in models:
        if hasattr(m, "cleanup"):
            m.cleanup()


def sample(model, noise, steps, cfg, sampler_name, scheduler, positive,
           negative, latent_image, denoise=1.0, disable_noise=False,
           start_step=None, last_step=None, force_full_denoise=False,
           noise_mask=None, sigmas=None, callback=None, disable_pbar=False,
           seed=None):
    return latent_image


def sample_custom(model, noise, cfg, sampler, sigmas, positive, negative,
                  latent_image, noise_mask=None, callback=None,
                  disable_pbar=False, seed=None):
    return latent_image
