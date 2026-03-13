"""Custom sampling nodes -- composable noise, guiders, schedulers, samplers."""
from __future__ import annotations

import math

import torch

from serenityflow.nodes.registry import registry


# ---------------------------------------------------------------------------
# Noise generators
# ---------------------------------------------------------------------------

@registry.register(
    "RandomNoise",
    return_types=("NOISE",),
    category="sampling/custom_sampling/noise",
    input_types={"required": {"noise_seed": ("INT",)}},
)
def random_noise(noise_seed):
    return ({"type": "random", "seed": noise_seed},)


@registry.register(
    "EmptyNoise",
    return_types=("NOISE",),
    category="sampling/custom_sampling/noise",
    input_types={"required": {}},
)
def empty_noise():
    return ({"type": "empty"},)


# ---------------------------------------------------------------------------
# Guiders
# ---------------------------------------------------------------------------

@registry.register(
    "BasicGuider",
    return_types=("GUIDER",),
    category="sampling/custom_sampling/guiders",
    input_types={"required": {
        "model": ("MODEL",),
        "conditioning": ("CONDITIONING",),
    }},
)
def basic_guider(model, conditioning):
    return ({"type": "basic", "model": model, "positive": conditioning},)


@registry.register(
    "CFGGuider",
    return_types=("GUIDER",),
    category="sampling/custom_sampling/guiders",
    input_types={"required": {
        "model": ("MODEL",),
        "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",),
        "cfg": ("FLOAT",),
    }},
)
def cfg_guider(model, positive, negative, cfg=8.0):
    return ({"type": "cfg", "model": model, "positive": positive,
             "negative": negative, "cfg": cfg},)


@registry.register(
    "DualCFGGuider",
    return_types=("GUIDER",),
    category="sampling/custom_sampling/guiders",
    input_types={"required": {
        "model": ("MODEL",),
        "cond1": ("CONDITIONING",),
        "cond2": ("CONDITIONING",),
        "negative": ("CONDITIONING",),
        "cfg_conds": ("FLOAT",),
        "cfg_cond2_negative": ("FLOAT",),
    }},
)
def dual_cfg_guider(model, cond1, cond2, negative, cfg_conds=8.0, cfg_cond2_negative=8.0):
    return ({"type": "dual_cfg", "model": model, "cond1": cond1, "cond2": cond2,
             "negative": negative, "cfg_conds": cfg_conds,
             "cfg_cond2_negative": cfg_cond2_negative},)


@registry.register(
    "PerpNegGuider",
    return_types=("GUIDER",),
    category="sampling/custom_sampling/guiders",
    input_types={"required": {
        "model": ("MODEL",),
        "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",),
        "empty_conditioning": ("CONDITIONING",),
        "cfg": ("FLOAT",),
        "neg_scale": ("FLOAT",),
    }},
)
def perp_neg_guider(model, positive, negative, empty_conditioning, cfg=8.0, neg_scale=1.0):
    return ({"type": "perp_neg", "model": model, "positive": positive,
             "negative": negative, "empty_conditioning": empty_conditioning,
             "cfg": cfg, "neg_scale": neg_scale},)


# ---------------------------------------------------------------------------
# Schedulers — pure math, no bridge needed
# ---------------------------------------------------------------------------

@registry.register(
    "BasicScheduler",
    return_types=("SIGMAS",),
    category="sampling/custom_sampling/schedulers",
    input_types={"required": {
        "model": ("MODEL",),
        "scheduler": ("STRING",),
        "steps": ("INT",),
        "denoise": ("FLOAT",),
    }},
)
def basic_scheduler(model, scheduler, steps, denoise=1.0):
    # Delegate to bridge for model-aware sigma generation
    # TODO: bridge.get_sigmas()
    raise NotImplementedError("BasicScheduler requires bridge.get_sigmas()")


@registry.register(
    "KarrasScheduler",
    return_types=("SIGMAS",),
    category="sampling/custom_sampling/schedulers",
    input_types={"required": {
        "steps": ("INT",),
        "sigma_max": ("FLOAT",),
        "sigma_min": ("FLOAT",),
        "rho": ("FLOAT",),
    }},
)
def karras_scheduler(steps, sigma_max=14.614642, sigma_min=0.0291675, rho=7.0):
    ramp = torch.linspace(0, 1, steps + 1)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return (sigmas,)


@registry.register(
    "ExponentialScheduler",
    return_types=("SIGMAS",),
    category="sampling/custom_sampling/schedulers",
    input_types={"required": {
        "steps": ("INT",),
        "sigma_max": ("FLOAT",),
        "sigma_min": ("FLOAT",),
    }},
)
def exponential_scheduler(steps, sigma_max=14.614642, sigma_min=0.0291675):
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), steps + 1).exp()
    return (sigmas,)


@registry.register(
    "PolyexponentialScheduler",
    return_types=("SIGMAS",),
    category="sampling/custom_sampling/schedulers",
    input_types={"required": {
        "steps": ("INT",),
        "sigma_max": ("FLOAT",),
        "sigma_min": ("FLOAT",),
        "rho": ("FLOAT",),
    }},
)
def polyexponential_scheduler(steps, sigma_max=14.614642, sigma_min=0.0291675, rho=1.0):
    ramp = torch.linspace(1, 0, steps + 1)
    sigmas = torch.where(
        ramp > 0,
        sigma_min ** (1.0 / rho) + ramp * (sigma_max ** (1.0 / rho) - sigma_min ** (1.0 / rho)),
        torch.zeros_like(ramp),
    ) ** rho
    # Last sigma should be exactly 0 (denoised)
    sigmas[-1] = 0.0
    return (sigmas,)


@registry.register(
    "BetaSamplingScheduler",
    return_types=("SIGMAS",),
    category="sampling/custom_sampling/schedulers",
    input_types={"required": {
        "model": ("MODEL",),
        "steps": ("INT",),
        "alpha": ("FLOAT",),
        "beta": ("FLOAT",),
    }},
)
def beta_sampling_scheduler(model, steps, alpha=0.6, beta=0.6):
    # Beta distribution CDF for timestep selection
    ts = 1.0 - torch.linspace(0, 1, steps + 1)
    # Apply beta CDF approximation
    from math import lgamma, exp
    def beta_cdf(x, a, b):
        # Simple approximation using regularized incomplete beta
        # For node purposes, linear interpolation in beta-warped space
        return x ** a / (x ** a + (1 - x) ** b)
    warped = torch.tensor([beta_cdf(t.item(), alpha, beta) if t > 0 else 0.0 for t in ts])
    # Map to sigma range (model-dependent, use reasonable defaults)
    sigma_max = 14.614642
    sigma_min = 0.0291675
    sigmas = sigma_min + warped * (sigma_max - sigma_min)
    sigmas = sigmas.flip(0)
    sigmas[-1] = 0.0
    return (sigmas,)


@registry.register(
    "SDTurboScheduler",
    return_types=("SIGMAS",),
    category="sampling/custom_sampling/schedulers",
    input_types={"required": {
        "model": ("MODEL",),
        "steps": ("INT",),
        "denoise": ("FLOAT",),
    }},
)
def sd_turbo_scheduler(model, steps, denoise=1.0):
    # SD Turbo uses a very short sigma schedule
    sigma_max = 14.614642 * denoise
    sigma_min = 0.0291675
    if steps == 1:
        sigmas = torch.tensor([sigma_max, 0.0])
    else:
        sigmas = torch.linspace(sigma_max, 0.0, steps + 1)
    return (sigmas,)


@registry.register(
    "AlignYourStepsScheduler",
    return_types=("SIGMAS",),
    category="sampling/custom_sampling/schedulers",
    input_types={"required": {
        "model_type": ("STRING",),
        "steps": ("INT",),
        "denoise": ("FLOAT",),
    }},
)
def align_your_steps_scheduler(model_type, steps, denoise=1.0):
    # AYS reference sigmas for known model types
    ays_sigmas = {
        "sd15": [14.6146, 6.4745, 3.8636, 2.6946, 1.8841, 1.3943, 0.9598,
                 0.6522, 0.3977, 0.1522, 0.0292],
        "sdxl": [14.6146, 6.3184, 3.7681, 2.1811, 1.3405, 0.8620, 0.5552,
                 0.3466, 0.1553, 0.0292],
        "sd3": [14.6146, 7.3855, 3.7337, 2.0867, 1.1584, 0.6418, 0.3573,
                0.1632, 0.0292],
        "flux": [14.6146, 5.1653, 2.3859, 1.1283, 0.5363, 0.2539, 0.1175,
                 0.0292],
    }
    ref = ays_sigmas.get(model_type, ays_sigmas["sdxl"])
    # Interpolate to requested step count
    ref_t = torch.tensor(ref, dtype=torch.float32)
    if steps + 1 == len(ref_t):
        sigmas = ref_t
    else:
        x_ref = torch.linspace(0, 1, len(ref_t))
        x_new = torch.linspace(0, 1, steps + 1)
        sigmas = torch.zeros(steps + 1)
        for i, xn in enumerate(x_new):
            idx = torch.searchsorted(x_ref, xn).clamp(1, len(x_ref) - 1).item()
            t = (xn - x_ref[idx - 1]) / (x_ref[idx] - x_ref[idx - 1] + 1e-8)
            sigmas[i] = ref_t[idx - 1] * (1 - t) + ref_t[idx] * t
    if denoise < 1.0:
        sigmas = sigmas[max(0, int((1 - denoise) * steps)):]
    return (sigmas,)


@registry.register(
    "GITSScheduler",
    return_types=("SIGMAS",),
    category="sampling/custom_sampling/schedulers",
    input_types={"required": {
        "model": ("MODEL",),
        "steps": ("INT",),
        "coeff": ("FLOAT",),
        "denoise": ("FLOAT",),
    }},
)
def gits_scheduler(model, steps, coeff=1.2, denoise=1.0):
    # GITS (Growing Iterative Timestep Sampling) — geometric spacing
    sigma_max = 14.614642 * denoise
    sigma_min = 0.0291675
    # Geometric with coefficient adjustment
    ratio = (sigma_min / sigma_max) ** (1.0 / steps)
    sigmas = torch.tensor([sigma_max * (ratio ** i) for i in range(steps)] + [0.0])
    sigmas = sigmas * (coeff / 1.2)  # Scale by coefficient
    sigmas[-1] = 0.0
    return (sigmas,)


@registry.register(
    "LaplaceScheduler",
    return_types=("SIGMAS",),
    category="sampling/custom_sampling/schedulers",
    input_types={"required": {
        "steps": ("INT",),
        "sigma_max": ("FLOAT",),
        "sigma_min": ("FLOAT",),
        "mu": ("FLOAT",),
        "beta": ("FLOAT",),
    }},
)
def laplace_scheduler(steps, sigma_max=14.614642, sigma_min=0.0291675, mu=0.0, beta=0.5):
    # Laplace distribution CDF for sigma scheduling
    ramp = torch.linspace(0, 1, steps + 1)
    # Map through Laplace quantile function
    sigmas = torch.where(
        ramp <= 0.5,
        mu + beta * torch.log(2.0 * ramp.clamp(min=1e-8)),
        mu - beta * torch.log(2.0 * (1.0 - ramp).clamp(min=1e-8)),
    )
    # Rescale to sigma range
    sigmas = (sigmas - sigmas.min()) / (sigmas.max() - sigmas.min() + 1e-8)
    sigmas = sigma_min + sigmas * (sigma_max - sigma_min)
    sigmas = sigmas.flip(0)
    sigmas[-1] = 0.0
    return (sigmas,)


# ---------------------------------------------------------------------------
# Sigma manipulation
# ---------------------------------------------------------------------------

@registry.register(
    "SplitSigmas",
    return_types=("SIGMAS", "SIGMAS"),
    return_names=("high_sigmas", "low_sigmas"),
    category="sampling/custom_sampling/sigmas",
    input_types={"required": {
        "sigmas": ("SIGMAS",),
        "step": ("INT",),
    }},
)
def split_sigmas(sigmas, step):
    return (sigmas[:step + 1], sigmas[step:])


@registry.register(
    "SplitSigmasDenoise",
    return_types=("SIGMAS", "SIGMAS"),
    return_names=("high_sigmas", "low_sigmas"),
    category="sampling/custom_sampling/sigmas",
    input_types={"required": {
        "sigmas": ("SIGMAS",),
        "denoise": ("FLOAT",),
    }},
)
def split_sigmas_denoise(sigmas, denoise=1.0):
    steps = len(sigmas) - 1
    split_at = max(0, int(steps * (1.0 - denoise)))
    return (sigmas[:split_at + 1], sigmas[split_at:])


@registry.register(
    "FlipSigmas",
    return_types=("SIGMAS",),
    category="sampling/custom_sampling/sigmas",
    input_types={"required": {"sigmas": ("SIGMAS",)}},
)
def flip_sigmas(sigmas):
    return (sigmas.flip(0),)


@registry.register(
    "SetSigmaStart",
    return_types=("SIGMAS",),
    category="sampling/custom_sampling/sigmas",
    input_types={"required": {
        "sigmas": ("SIGMAS",),
        "sigma_start": ("FLOAT",),
    }},
)
def set_sigma_start(sigmas, sigma_start):
    result = sigmas.clone()
    result[0] = sigma_start
    return (result,)


@registry.register(
    "SetSigmaEnd",
    return_types=("SIGMAS",),
    category="sampling/custom_sampling/sigmas",
    input_types={"required": {
        "sigmas": ("SIGMAS",),
        "sigma_end": ("FLOAT",),
    }},
)
def set_sigma_end(sigmas, sigma_end):
    result = sigmas.clone()
    result[-1] = sigma_end
    return (result,)


# ---------------------------------------------------------------------------
# Sampler algorithms
# ---------------------------------------------------------------------------

@registry.register(
    "SamplerEuler",
    return_types=("SAMPLER",),
    category="sampling/custom_sampling/samplers",
    input_types={"required": {}},
)
def sampler_euler():
    return ({"type": "euler"},)


@registry.register(
    "SamplerEulerAncestral",
    return_types=("SAMPLER",),
    category="sampling/custom_sampling/samplers",
    input_types={"required": {
        "eta": ("FLOAT",),
        "s_noise": ("FLOAT",),
    }},
)
def sampler_euler_ancestral(eta=1.0, s_noise=1.0):
    return ({"type": "euler_ancestral", "eta": eta, "s_noise": s_noise},)


@registry.register(
    "SamplerDPMPP_2M",
    return_types=("SAMPLER",),
    category="sampling/custom_sampling/samplers",
    input_types={"required": {}},
)
def sampler_dpmpp_2m():
    return ({"type": "dpmpp_2m"},)


@registry.register(
    "SamplerDPMPP_SDE",
    return_types=("SAMPLER",),
    category="sampling/custom_sampling/samplers",
    input_types={"required": {
        "eta": ("FLOAT",),
        "s_noise": ("FLOAT",),
        "noise_device": ("STRING",),
    }},
)
def sampler_dpmpp_sde(eta=1.0, s_noise=1.0, noise_device="gpu"):
    return ({"type": "dpmpp_sde", "eta": eta, "s_noise": s_noise,
             "noise_device": noise_device},)


@registry.register(
    "SamplerDPMPP_2S_Ancestral",
    return_types=("SAMPLER",),
    category="sampling/custom_sampling/samplers",
    input_types={"required": {
        "eta": ("FLOAT",),
        "s_noise": ("FLOAT",),
    }},
)
def sampler_dpmpp_2s_ancestral(eta=1.0, s_noise=1.0):
    return ({"type": "dpmpp_2s_ancestral", "eta": eta, "s_noise": s_noise},)


@registry.register(
    "SamplerDPMPP_3M_SDE",
    return_types=("SAMPLER",),
    category="sampling/custom_sampling/samplers",
    input_types={"required": {
        "eta": ("FLOAT",),
        "s_noise": ("FLOAT",),
        "noise_device": ("STRING",),
    }},
)
def sampler_dpmpp_3m_sde(eta=1.0, s_noise=1.0, noise_device="gpu"):
    return ({"type": "dpmpp_3m_sde", "eta": eta, "s_noise": s_noise,
             "noise_device": noise_device},)


@registry.register(
    "SamplerLCM",
    return_types=("SAMPLER",),
    category="sampling/custom_sampling/samplers",
    input_types={"required": {}},
)
def sampler_lcm():
    return ({"type": "lcm"},)


@registry.register(
    "SamplerDDIM",
    return_types=("SAMPLER",),
    category="sampling/custom_sampling/samplers",
    input_types={"required": {
        "eta": ("FLOAT",),
    }},
)
def sampler_ddim(eta=0.0):
    return ({"type": "ddim", "eta": eta},)


@registry.register(
    "SamplerUniPC",
    return_types=("SAMPLER",),
    category="sampling/custom_sampling/samplers",
    input_types={"required": {
        "order": ("INT",),
    }},
)
def sampler_unipc(order=3):
    return ({"type": "unipc", "order": order},)


@registry.register(
    "SamplerHeun",
    return_types=("SAMPLER",),
    category="sampling/custom_sampling/samplers",
    input_types={"required": {}},
)
def sampler_heun():
    return ({"type": "heun"},)


@registry.register(
    "SamplerDPMAdaptive",
    return_types=("SAMPLER",),
    category="sampling/custom_sampling/samplers",
    input_types={"required": {
        "order": ("INT",),
        "rtol": ("FLOAT",),
        "atol": ("FLOAT",),
        "h_init": ("FLOAT",),
    }},
)
def sampler_dpm_adaptive(order=3, rtol=0.05, atol=0.0078, h_init=0.05):
    return ({"type": "dpm_adaptive", "order": order, "rtol": rtol,
             "atol": atol, "h_init": h_init},)


# ---------------------------------------------------------------------------
# Top-level composable samplers
# ---------------------------------------------------------------------------

@registry.register(
    "SamplerCustom",
    return_types=("LATENT", "LATENT"),
    return_names=("output", "denoised_output"),
    category="sampling/custom_sampling",
    input_types={"required": {
        "model": ("MODEL",),
        "add_noise": ("BOOLEAN",),
        "noise_seed": ("INT",),
        "cfg": ("FLOAT",),
        "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",),
        "sampler": ("SAMPLER",),
        "sigmas": ("SIGMAS",),
        "latent_image": ("LATENT",),
    }},
)
def sampler_custom(model, add_noise, noise_seed, cfg, positive, negative,
                   sampler, sigmas, latent_image):
    # TODO: bridge.sample_custom()
    raise NotImplementedError("SamplerCustom requires bridge.sample_custom()")


@registry.register(
    "SamplerCustomAdvanced",
    return_types=("LATENT", "LATENT"),
    return_names=("output", "denoised_output"),
    category="sampling/custom_sampling",
    input_types={"required": {
        "noise": ("NOISE",),
        "guider": ("GUIDER",),
        "sampler": ("SAMPLER",),
        "sigmas": ("SIGMAS",),
        "latent_image": ("LATENT",),
    }},
)
def sampler_custom_advanced(noise, guider, sampler, sigmas, latent_image):
    # TODO: bridge.sample_custom()
    raise NotImplementedError("SamplerCustomAdvanced requires bridge.sample_custom()")


# ---------------------------------------------------------------------------
# Sampler selection by name
# ---------------------------------------------------------------------------

@registry.register(
    "KSamplerSelect",
    return_types=("SAMPLER",),
    category="sampling/custom_sampling/samplers",
    input_types={"required": {"sampler_name": ("STRING",)}},
)
def ksampler_select(sampler_name):
    return ({"type": sampler_name},)


# ---------------------------------------------------------------------------
# Disable noise (alias for empty noise with distinct class_type)
# ---------------------------------------------------------------------------

@registry.register(
    "DisableNoise",
    return_types=("NOISE",),
    category="sampling/custom_sampling/noise",
    input_types={"required": {}},
)
def disable_noise():
    return ({"type": "empty"},)
