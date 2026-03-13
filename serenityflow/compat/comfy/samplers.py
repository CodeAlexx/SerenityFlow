"""Compatibility shim for comfy.samplers.

Sampler/scheduler names and pure-math scheduler functions.
"""
from __future__ import annotations

import math
from typing import Any

import torch

SAMPLER_NAMES = [
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp",
    "heun", "heunpp2", "dpm_2", "dpm_2_ancestral",
    "lms", "dpm_fast", "dpm_adaptive",
    "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde", "dpmpp_sde_gpu",
    "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde", "dpmpp_3m_sde_gpu",
    "ddpm", "lcm", "ipndm", "ipndm_v", "deis", "res_momentumized",
    "ddim", "uni_pc", "uni_pc_bh2",
]

SCHEDULER_NAMES = [
    "normal", "karras", "exponential", "sgm_uniform", "simple",
    "ddim_uniform", "beta", "linear_quadratic", "kl_optimal",
]


# === Scheduler functions ===


def normal_scheduler(model_sampling, steps, sgm=False, floor=False):
    s = model_sampling
    start = s.timestep(s.sigma_max) if hasattr(s, "timestep") else 0
    end = s.timestep(s.sigma_min) if hasattr(s, "timestep") else 1
    timesteps = torch.linspace(start, end, steps + 1)[:-1] if not sgm else torch.linspace(start, end, steps)
    sigmas = torch.zeros(steps + 1)
    for i, t in enumerate(timesteps):
        sigmas[i] = s.sigma(t) if hasattr(s, "sigma") else t
    sigmas[-1] = 0.0
    return sigmas


def karras_scheduler(n, sigma_min, sigma_max, rho=7.0):
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, torch.zeros(1)])


def exponential_scheduler(n, sigma_min, sigma_max):
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n).exp()
    return torch.cat([sigmas, torch.zeros(1)])


def simple_scheduler(model_sampling, steps):
    s = model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs.append(float(s.sigmas[int(x * ss)]))
    sigs.append(0.0)
    return torch.FloatTensor(sigs)


def ddim_scheduler(model_sampling, steps):
    s = model_sampling
    sigs = []
    ss = max(len(s.sigmas) // steps, 1)
    x = 1
    while x < len(s.sigmas):
        sigs.append(float(s.sigmas[x]))
        x += ss
    sigs = sigs[:steps]
    sigs.reverse()
    sigs.append(0.0)
    return torch.FloatTensor(sigs)


def sgm_uniform_scheduler(model_sampling, steps):
    s = model_sampling
    start = s.timestep(s.sigma_max) if hasattr(s, "timestep") else 0
    end = s.timestep(s.sigma_min) if hasattr(s, "timestep") else 1
    timesteps = torch.linspace(start, end, steps + 1)[1:]
    sigmas = torch.zeros(steps + 1)
    for i, t in enumerate(timesteps):
        sigmas[i] = s.sigma(t) if hasattr(s, "sigma") else t
    sigmas[-1] = 0.0
    return sigmas


def beta_scheduler(model_sampling, steps, alpha=0.6, beta=0.6):
    ts = 1.0 - torch.linspace(0, 1, steps)
    ts = (ts ** alpha) / ((ts ** alpha + (1 - ts) ** beta))
    sigmas = torch.zeros(steps + 1)
    s = model_sampling
    for i, t in enumerate(ts):
        idx = int(t * (len(s.sigmas) - 1))
        sigmas[i] = s.sigmas[idx]
    sigmas[-1] = 0.0
    return sigmas


def calculate_sigmas(model_sampling, scheduler_name, steps):
    if scheduler_name == "karras":
        sigma_min = float(model_sampling.sigma_min)
        sigma_max = float(model_sampling.sigma_max)
        return karras_scheduler(steps, sigma_min, sigma_max)
    elif scheduler_name == "exponential":
        sigma_min = float(model_sampling.sigma_min)
        sigma_max = float(model_sampling.sigma_max)
        return exponential_scheduler(steps, sigma_min, sigma_max)
    elif scheduler_name == "simple":
        return simple_scheduler(model_sampling, steps)
    elif scheduler_name == "ddim_uniform":
        return ddim_scheduler(model_sampling, steps)
    elif scheduler_name == "sgm_uniform":
        return sgm_uniform_scheduler(model_sampling, steps)
    elif scheduler_name == "beta":
        return beta_scheduler(model_sampling, steps)
    else:
        return normal_scheduler(model_sampling, steps)


# === KSampler ===


class KSampler:
    SCHEDULERS = SCHEDULER_NAMES
    SAMPLERS = SAMPLER_NAMES

    def __init__(self, model, steps, device, sampler=None, scheduler=None,
                 denoise=None, model_options=None):
        self.model = model
        self.device = device
        self.steps = steps
        self.sampler = sampler
        self.scheduler = scheduler
        self.denoise = denoise or 1.0
        self.model_options = model_options or {}
        self.sigmas = None

    def sample(self, noise, positive, negative, cfg, latent_image=None,
               start_step=None, last_step=None, force_full_denoise=False,
               denoise_mask=None, sigmas=None, callback=None, disable_pbar=False,
               seed=None):
        return noise


class KSAMPLER:
    SAMPLERS = SAMPLER_NAMES
    SCHEDULERS = SCHEDULER_NAMES


class CFGGuider:
    def __init__(self, model_patcher):
        self.model_patcher = model_patcher
        self.conds = {}
        self.cfg = 1.0
        self.model_options = model_patcher.model_options.copy() if model_patcher else {}

    def set_conds(self, **kwargs):
        self.conds.update(kwargs)

    def set_cfg(self, cfg):
        self.cfg = cfg

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None,
               callback=None, disable_pbar=False, seed=None):
        return noise

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        return torch.zeros_like(x)


def ksampler(sampler_name, extra_options=None, inpaint_options=None):
    return None


def sampler_object(name):
    return None


def cfg_function(model, cond_pred, uncond_pred, cond_scale, x, timestep,
                 model_options=None, cond=None, uncond=None):
    return uncond_pred + (cond_pred - uncond_pred) * cond_scale


def sampling_function(model, x, timestep, uncond, cond, cond_scale,
                      model_options=None, seed=None):
    return x


def resolve_areas_and_cond_masks(conditions, h, w, device):
    pass


def calculate_start_end_timesteps(model, conds):
    pass


def encode_model_conds(model_function, conds, noise, device, prompt_type,
                       **kwargs):
    return conds


def pre_run_control(model, conds):
    pass


def cleanup_additional_models(models):
    pass


def prepare_sampling(model, noise_shape, conds):
    return model, conds, conds


def calc_cond_batch(model, conds, x_in, timestep, model_options):
    return x_in
