"""Standalone sampling math for SerenityFlow — sigma schedules, CFG, prediction types.

Derived from:
  - ComfyUI (MIT License): comfy/model_sampling.py, comfy/samplers.py,
    comfy/k_diffusion/sampling.py, comfy_extras/nodes_model_advanced.py
  - diffusers (Apache 2.0): schedulers/scheduling_euler_discrete.py

No runtime imports from serenity, comfy, or diffusers.
"""
from __future__ import annotations

import math
import logging
from enum import Enum

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = [
    "SchedulerType",
    "compute_sigmas",
    "create_noise",
    "apply_cfg",
    "PredictionType",
    "Prediction",
    "EpsPrediction",
    "VPrediction",
    "FlowPrediction",
    "FluxPrediction",
    "get_prediction",
    "PipelineCounters",
    "sample",
]


# --------------------------------------------------------------------------- #
# Sigma schedules
# --------------------------------------------------------------------------- #


class SchedulerType(str, Enum):
    NORMAL = "normal"
    KARRAS = "karras"
    EXPONENTIAL = "exponential"
    SGM_UNIFORM = "sgm_uniform"
    SIMPLE = "simple"
    DDIM_UNIFORM = "ddim_uniform"


def _normal_scheduler(n: int, sigma_min: float, sigma_max: float) -> Tensor:
    """Log-space linear interpolation (ComfyUI normal_scheduler without model_sampling)."""
    log_min = math.log(sigma_min)
    log_max = math.log(sigma_max)
    sigmas = torch.linspace(log_max, log_min, n).exp()
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def _karras_scheduler(n: int, sigma_min: float, sigma_max: float, rho: float = 7.0) -> Tensor:
    """Karras et al. 2022 — k-diffusion get_sigmas_karras."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def _exponential_scheduler(n: int, sigma_min: float, sigma_max: float) -> Tensor:
    """k-diffusion get_sigmas_exponential."""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n).exp()
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def _sgm_uniform_scheduler(n: int, sigma_min: float, sigma_max: float) -> Tensor:
    """SGM Uniform — log-space, excludes endpoint, then appends 0."""
    log_sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n + 1)[:-1]
    sigmas = log_sigmas.exp()
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def _simple_scheduler(n: int, sigma_min: float, sigma_max: float) -> Tensor:
    """Linear interpolation between sigma_max and sigma_min."""
    sigmas = torch.linspace(sigma_max, sigma_min, n)
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def _ddim_uniform_scheduler(n: int, sigma_min: float, sigma_max: float) -> Tensor:
    """DDIM-style uniform spacing in sigma space."""
    sigmas = torch.linspace(sigma_max, sigma_min, n)
    return torch.cat([sigmas, sigmas.new_zeros(1)])


_SCHEDULER_MAP = {
    SchedulerType.NORMAL: _normal_scheduler,
    SchedulerType.KARRAS: _karras_scheduler,
    SchedulerType.EXPONENTIAL: _exponential_scheduler,
    SchedulerType.SGM_UNIFORM: _sgm_uniform_scheduler,
    SchedulerType.SIMPLE: _simple_scheduler,
    SchedulerType.DDIM_UNIFORM: _ddim_uniform_scheduler,
}


def compute_sigmas(
    scheduler: SchedulerType | str,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    **kwargs,
) -> Tensor:
    """Compute a sigma schedule of length ``num_steps + 1`` (final element is 0).

    Parameters
    ----------
    scheduler : SchedulerType or str
        Which noise schedule to use.
    num_steps : int
        Number of denoising steps.
    sigma_min, sigma_max : float
        Noise level bounds.
    **kwargs
        Extra arguments forwarded to the scheduler function.
    """
    scheduler = SchedulerType(scheduler)
    fn = _SCHEDULER_MAP[scheduler]
    return fn(num_steps, sigma_min, sigma_max, **kwargs)


# --------------------------------------------------------------------------- #
# Noise creation
# --------------------------------------------------------------------------- #


def create_noise(
    seed: int,
    shape: tuple[int, ...],
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create a reproducible noise tensor from a seed.

    Uses ``torch.Generator`` for deterministic results across runs.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(shape, generator=generator, device="cpu", dtype=dtype)
    return noise.to(device)


# --------------------------------------------------------------------------- #
# CFG
# --------------------------------------------------------------------------- #


def apply_cfg(
    cond_pred: Tensor,
    uncond_pred: Tensor,
    cfg_scale: float,
) -> Tensor:
    """Standard classifier-free guidance.

    Returns ``uncond + cfg_scale * (cond - uncond)``.
    Short-circuits when cfg_scale == 1.0.
    """
    if cfg_scale == 1.0:
        return cond_pred
    return uncond_pred + cfg_scale * (cond_pred - uncond_pred)


# --------------------------------------------------------------------------- #
# Prediction types
# --------------------------------------------------------------------------- #


def _reshape_sigma(sigma: Tensor, target: Tensor) -> Tensor:
    """Reshape sigma for broadcasting: scalar or (B,) -> (B, 1, 1, ...)."""
    if sigma.nelement() == 1:
        return sigma.view(())
    return sigma.view(sigma.shape[:1] + (1,) * (target.ndim - 1))


class PredictionType(str, Enum):
    EPS = "eps"
    V_PREDICTION = "v_prediction"
    FLOW = "flow"
    FLOW_FLUX = "flow_flux"


class Prediction:
    """Base prediction type — defines the interface for noise/denoised conversion."""

    def __init__(self, sigma_data: float = 1.0, **kwargs) -> None:
        self.sigma_data = sigma_data

    def calculate_input(self, sigma: Tensor, noise: Tensor) -> Tensor:
        """Scale noisy input for the model (c_in)."""
        sigma = _reshape_sigma(sigma, noise)
        return noise / (sigma ** 2 + self.sigma_data ** 2) ** 0.5

    def calculate_denoised(
        self, sigma: Tensor, model_output: Tensor, model_input: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    def noise_scaling(
        self, sigma: Tensor, noise: Tensor, latent: Tensor, max_denoise: bool = False,
    ) -> Tensor:
        sigma = _reshape_sigma(sigma, noise)
        if max_denoise:
            noise = noise * torch.sqrt(1.0 + sigma ** 2.0)
        else:
            noise = noise * sigma
        return noise + latent

    def sigma_to_timestep(self, sigma: Tensor) -> Tensor:
        return sigma


class EpsPrediction(Prediction):
    """Epsilon (noise) prediction — SD 1.5, SDXL.  ComfyUI ``EPS``."""

    def calculate_denoised(
        self, sigma: Tensor, model_output: Tensor, model_input: Tensor,
    ) -> Tensor:
        sigma = _reshape_sigma(sigma, model_output)
        return model_input - model_output * sigma


class VPrediction(Prediction):
    """V-prediction — SD 2.x.  ComfyUI ``V_PREDICTION``."""

    def calculate_denoised(
        self, sigma: Tensor, model_output: Tensor, model_input: Tensor,
    ) -> Tensor:
        sigma = _reshape_sigma(sigma, model_output)
        sd2 = self.sigma_data ** 2
        return (
            model_input * sd2 / (sigma ** 2 + sd2)
            - model_output * sigma * self.sigma_data / (sigma ** 2 + sd2) ** 0.5
        )


class FlowPrediction(Prediction):
    """Flow matching — Wan, Qwen, Lumina, ZImage, SD3.  ComfyUI ``CONST``."""

    def __init__(self, sigma_data: float = 1.0, shift: float = 1.0, multiplier: float = 1000.0, **kwargs) -> None:
        super().__init__(sigma_data=sigma_data)
        self.shift = shift
        self.multiplier = multiplier

    def calculate_input(self, sigma: Tensor, noise: Tensor) -> Tensor:
        return noise

    def calculate_denoised(
        self, sigma: Tensor, model_output: Tensor, model_input: Tensor,
    ) -> Tensor:
        sigma = _reshape_sigma(sigma, model_output)
        return model_input - model_output * sigma

    def noise_scaling(
        self, sigma: Tensor, noise: Tensor, latent: Tensor, max_denoise: bool = False,
    ) -> Tensor:
        """``sigma * noise + (1 - sigma) * latent`` — ComfyUI CONST.noise_scaling."""
        sigma = _reshape_sigma(sigma, noise)
        return sigma * noise + (1.0 - sigma) * latent

    def sigma_to_timestep(self, sigma: Tensor) -> Tensor:
        return sigma * self.multiplier


# -- Flux prediction (mu-based exponential sigma shifting) --


def _flux_time_shift(mu: float, sigma: float, t: Tensor) -> Tensor:
    """Exponential time shift — ComfyUI flux_time_shift."""
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)


def _calculate_flux_mu(
    seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Compute mu for Flux exponential sigma shifting based on sequence length."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return float(m * seq_len + b)


class FluxPrediction(FlowPrediction):
    """Flux-specific flow matching with mu-based exponential sigma shifting.

    Based on ComfyUI ``ModelSamplingFlux``.
    """

    def __init__(
        self,
        sigma_data: float = 1.0,
        mu: float | None = None,
        seq_len: int = 4096,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        **kwargs,
    ) -> None:
        super().__init__(sigma_data=sigma_data, shift=1.0, multiplier=1.0)
        if mu is not None:
            self.mu = mu
        else:
            self.mu = _calculate_flux_mu(
                seq_len=seq_len,
                base_seq_len=base_seq_len,
                max_seq_len=max_seq_len,
                base_shift=base_shift,
                max_shift=max_shift,
            )

    def sigma_to_timestep(self, sigma: Tensor) -> Tensor:
        return sigma

    def apply_sigma_shift(self, sigmas: Tensor) -> Tensor:
        """Apply exponential time shift to a sigma schedule."""
        return _flux_time_shift(self.mu, 1.0, sigmas)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


_PREDICTION_MAP = {
    PredictionType.EPS: EpsPrediction,
    PredictionType.V_PREDICTION: VPrediction,
    PredictionType.FLOW: FlowPrediction,
    PredictionType.FLOW_FLUX: FluxPrediction,
}


def get_prediction(prediction_type: PredictionType | str, **kwargs) -> Prediction:
    """Create a Prediction instance for the given type."""
    prediction_type = PredictionType(prediction_type)
    cls = _PREDICTION_MAP[prediction_type]
    return cls(**kwargs)


# --------------------------------------------------------------------------- #
# Pipeline counters (stub — real counters are optional)
# --------------------------------------------------------------------------- #


class PipelineCounters:
    """Minimal stub — real counters are optional."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def wrap_model_fn(self, fn):
        return fn

    def make_callback(self, cb):
        return cb


# --------------------------------------------------------------------------- #
# Denoising samplers
#
# Ported from ComfyUI k-diffusion (MIT License):
#   comfy/k_diffusion/sampling.py
#   Copyright (c) 2022 Katherine Crowson (https://github.com/crowsonkb)
#
# No runtime imports from serenity, comfy, or diffusers.
# --------------------------------------------------------------------------- #


def _append_dims(sigma: Tensor, ndim: int) -> Tensor:
    """Reshape sigma for broadcasting: ``(B,) -> (B, 1, 1, ...)``."""
    return sigma.view(sigma.shape[:1] + (1,) * (ndim - 1))


def _to_d(x: Tensor, sigma: Tensor, denoised: Tensor) -> Tensor:
    """Convert denoiser output to a Karras ODE derivative."""
    return (x - denoised) / _append_dims(sigma, x.ndim)


def _get_ancestral_step(sigma_from: Tensor, sigma_to: Tensor, eta: float = 1.0):
    """Calculate sigma_down and sigma_up for ancestral sampling."""
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(
        sigma_to,
        eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5,
    )
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


# -- Individual samplers ---------------------------------------------------- #


@torch.no_grad()
def _sample_euler(model_fn, x: Tensor, sigmas: Tensor, callback=None) -> Tensor:
    """Euler method (Karras Algorithm 2)."""
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        sigma_hat = sigmas[i]
        denoised = model_fn(x, sigma_hat * s_in)
        d = _to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        dt = sigmas[i + 1] - sigma_hat
        # Euler step
        x = x + d * dt
    return x


@torch.no_grad()
def _sample_euler_ancestral(model_fn, x: Tensor, sigmas: Tensor, callback=None, eta: float = 1.0) -> Tensor:
    """Ancestral sampling with Euler steps."""
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        denoised = model_fn(x, sigmas[i] * s_in)
        sigma_down, sigma_up = _get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigma_down == 0:
            x = denoised
        else:
            d = _to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt + torch.randn_like(x) * sigma_up
    return x


@torch.no_grad()
def _sample_euler_ancestral_rf(model_fn, x: Tensor, sigmas: Tensor, callback=None, eta: float = 1.0) -> Tensor:
    """Ancestral Euler for rectified-flow (CONST) models."""
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        denoised = model_fn(x, sigmas[i] * s_in)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
            sigma_down = sigmas[i + 1] * downstep_ratio
            alpha_ip1 = 1 - sigmas[i + 1]
            alpha_down = 1 - sigma_down
            renoise_coeff = (sigmas[i + 1] ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2) ** 0.5
            sigma_down_i_ratio = sigma_down / sigmas[i]
            x = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * denoised
            if eta > 0:
                x = (alpha_ip1 / alpha_down) * x + torch.randn_like(x) * renoise_coeff
    return x


@torch.no_grad()
def _sample_dpmpp_2m(model_fn, x: Tensor, sigmas: Tensor, callback=None) -> Tensor:
    """DPM-Solver++(2M)."""
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in range(len(sigmas) - 1):
        denoised = model_fn(x, sigmas[i] * s_in)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


@torch.no_grad()
def _sample_dpmpp_2m_sde(model_fn, x: Tensor, sigmas: Tensor, callback=None, eta: float = 1.0, solver_type: str = "midpoint") -> Tensor:
    """DPM-Solver++(2M) SDE.

    Uses simple randn noise (no BrownianTree dependency).
    """
    if len(sigmas) <= 1:
        return x

    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None
    h_last = None

    for i in range(len(sigmas) - 1):
        denoised = model_fn(x, sigmas[i] * s_in)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Final denoising step
            x = denoised
        else:
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            h_eta = h * (eta + 1)

            x = (sigma_fn(t_next) / sigma_fn(t)) * (-h * eta).exp() * x + sigma_fn(t_next) * (-h_eta).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == "heun":
                    x = x + sigma_fn(t_next) * ((-h_eta).expm1().neg() / (-h_eta) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == "midpoint":
                    x = x + 0.5 * sigma_fn(t_next) * (-h_eta).expm1().neg() * (1 / r) * (denoised - old_denoised)

            if eta > 0:
                x = x + torch.randn_like(x) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt()

            h_last = h
        old_denoised = denoised
    return x


# -- Dispatcher ------------------------------------------------------------- #


_SAMPLER_MAP = {
    "euler": _sample_euler,
    "euler_ancestral": _sample_euler_ancestral,
    "euler_ancestral_rf": _sample_euler_ancestral_rf,
    "dpmpp_2m": _sample_dpmpp_2m,
    "dpmpp_2m_sde": _sample_dpmpp_2m_sde,
    "dpmpp_sde": _sample_dpmpp_2m_sde,
}


@torch.no_grad()
def sample(
    model_fn,
    noise: Tensor,
    sigmas: Tensor,
    sampler_type: str = "euler",
    callback=None,
) -> Tensor:
    """Run a denoising loop.

    Parameters
    ----------
    model_fn : callable
        ``(x, sigma) -> denoised``.  Sigma is broadcast as ``sigmas[i] * s_in``.
    noise : Tensor
        Initial noisy latent (already scaled by the sigma schedule).
    sigmas : Tensor
        1-D sigma schedule with final element == 0.
    sampler_type : str
        One of ``euler``, ``euler_ancestral``, ``dpmpp_2m``, ``dpmpp_2m_sde``,
        ``dpmpp_sde``.  Unknown types fall back to ``euler`` with a warning.
    callback : callable or None
        ``callback({'x': ..., 'i': ..., 'sigma': ..., 'sigma_hat': ..., 'denoised': ...})``.
    """
    fn = _SAMPLER_MAP.get(sampler_type)
    if fn is None:
        logger.warning("Unknown sampler_type %r — falling back to euler", sampler_type)
        fn = _sample_euler
    return fn(model_fn, noise, sigmas, callback=callback)
