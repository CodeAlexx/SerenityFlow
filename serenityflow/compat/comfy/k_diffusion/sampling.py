"""Compatibility shim for comfy.k_diffusion.sampling.

K-diffusion sampler implementations.
"""
from __future__ import annotations

import torch

from comfy.model_management import throw_exception_if_processing_interrupted


class BrownianTreeNoiseSampler:
    def __init__(self, x, sigma_min, sigma_max, seed=None, cpu=False):
        self.x = x
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.seed = seed
        self.cpu = cpu

    def __call__(self, sigma, sigma_next):
        return torch.randn_like(self.x)


class BatchedBrownianTree:
    def __init__(self, x, t0, t1, seed=None, **kwargs):
        self.x = x

    def __call__(self, t0, t1):
        return torch.randn_like(self.x)


def to_d(x, sigma, denoised):
    """Convert a denoiser output to a d (velocity) estimate."""
    return (x - denoised) / sigma.view(-1, *([1] * (x.ndim - 1)))


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    if not eta:
        return sigma_to, torch.tensor(0.0)
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


@torch.no_grad()
def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None,
                 s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0):
    extra_args = extra_args or {}
    for i in range(len(sigmas) - 1):
        throw_exception_if_processing_interrupted()
        sigma = sigmas[i]
        denoised = model(x, sigma * x.new_ones([x.shape[0]]), **extra_args)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "denoised": denoised})
        dt = sigmas[i + 1] - sigma
        x = x + d * dt
    return x


@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None,
                            disable=None, eta=1.0, s_noise=1.0,
                            noise_sampler=None):
    extra_args = extra_args or {}
    noise_sampler = noise_sampler or default_noise_sampler(x)
    for i in range(len(sigmas) - 1):
        throw_exception_if_processing_interrupted()
        sigma = sigmas[i]
        denoised = model(x, sigma * x.new_ones([x.shape[0]]), **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigma, sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "denoised": denoised})
        d = to_d(x, sigma, denoised)
        dt = sigma_down - sigma
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigma, sigmas[i + 1]) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    extra_args = extra_args or {}
    old_denoised = None
    for i in range(len(sigmas) - 1):
        throw_exception_if_processing_interrupted()
        sigma = sigmas[i]
        denoised = model(x, sigma * x.new_ones([x.shape[0]]), **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "denoised": denoised})
        t, t_next = -sigma.log(), -sigmas[i + 1].log()
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma / sigmas[i + 1]) * x - (-h).expm1() * denoised
        else:
            h_last = t - (-sigmas[i - 1]).log()
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma / sigmas[i + 1]) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


@torch.no_grad()
def sample_dpmpp_2m_sde(model, x, sigmas, extra_args=None, callback=None,
                         disable=None, eta=1.0, s_noise=1.0,
                         noise_sampler=None, solver_type="midpoint"):
    return sample_dpmpp_2m(model, x, sigmas, extra_args=extra_args,
                           callback=callback, disable=disable)
