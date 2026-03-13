"""Compatibility shim for comfy.model_sampling.

Sigma/timestep math classes used by custom nodes and schedulers.
"""
from __future__ import annotations

import math

import torch
import numpy as np


class ModelSamplingDiscrete(torch.nn.Module):
    def __init__(self, model_config=None, num_timesteps=1000):
        super().__init__()
        self._register_schedule(num_timesteps=num_timesteps)

    def _register_schedule(self, given_betas=None, beta_schedule="linear",
                           timesteps=1000, linear_start=0.00085,
                           linear_end=0.012, cosine_s=8e-3, num_timesteps=None):
        if num_timesteps is not None:
            timesteps = num_timesteps
        if given_betas is not None:
            betas = given_betas
        elif beta_schedule == "linear":
            betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5,
                                   timesteps, dtype=torch.float64) ** 2
        elif beta_schedule == "cosine":
            t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64) / timesteps
            alphas_cumprod = torch.cos((t + cosine_s) / (1 + cosine_s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = torch.clamp(betas, max=0.999)
        else:
            betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5,
                                   timesteps, dtype=torch.float64) ** 2

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.register_buffer("sigmas", sigmas.float())
        self.register_buffer("log_sigmas", sigmas.log().float())

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.view(log_sigma.shape[:1] + (1,) * (log_sigma.ndim - 1)) - self.log_sigmas
        return dists.abs().argmin(dim=-1).view(sigma.shape).float()

    def sigma(self, timestep):
        t = timestep.float()
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        w = t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp()

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return self.sigma_max
        if percent >= 1.0:
            return torch.tensor(0.0)
        idx = int(percent * len(self.sigmas))
        return self.sigmas[max(0, min(idx, len(self.sigmas) - 1))]


class ModelSamplingContinuousEDM(torch.nn.Module):
    def __init__(self, model_config=None, sigma_min=0.002, sigma_max=120.0):
        super().__init__()
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        sigmas = torch.linspace(math.log(sigma_min), math.log(sigma_max), 1000).exp()
        self.register_buffer("sigmas", sigmas)

    @property
    def sigma_min(self):
        return self._sigma_min

    @property
    def sigma_max(self):
        return self._sigma_max

    def timestep(self, sigma):
        return 0.25 * sigma.log()

    def sigma(self, timestep):
        return (timestep / 0.25).exp()

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return torch.tensor(self._sigma_max)
        if percent >= 1.0:
            return torch.tensor(0.0)
        log_s = math.log(self._sigma_max) + percent * (math.log(self._sigma_min) - math.log(self._sigma_max))
        return torch.tensor(math.exp(log_s))


class ModelSamplingContinuousV(ModelSamplingContinuousEDM):
    pass


class ModelSamplingFlux(torch.nn.Module):
    def __init__(self, model_config=None, shift=1.0):
        super().__init__()
        self.shift = shift
        sigmas = torch.linspace(1.0, 0.0, 1000)
        self.register_buffer("sigmas", sigmas)

    @property
    def sigma_min(self):
        return torch.tensor(0.0)

    @property
    def sigma_max(self):
        return torch.tensor(1.0)

    def timestep(self, sigma):
        return sigma

    def sigma(self, timestep):
        return timestep

    def percent_to_sigma(self, percent):
        return torch.tensor(1.0 - percent)


class ModelSamplingDiscreteDistilled(ModelSamplingDiscrete):
    pass


class StableCascadeSampling(torch.nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        self.register_buffer("sigmas", torch.linspace(1.0, 0.0, 1000))

    @property
    def sigma_min(self):
        return torch.tensor(0.0)

    @property
    def sigma_max(self):
        return torch.tensor(1.0)

    def timestep(self, sigma):
        return sigma

    def sigma(self, timestep):
        return timestep

    def percent_to_sigma(self, percent):
        return torch.tensor(1.0 - percent)
