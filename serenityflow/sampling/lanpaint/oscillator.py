"""Stochastic Harmonic Oscillator for Langevin dynamics.

Simulates the SDE:
    dy(t) = q(t) dt
    dq(t) = -Gamma A y(t) dt + Gamma C dt + Gamma D dw(t) - Gamma q(t) dt

With v(t) = q(t) / sqrt(Gamma) for numerical stability.

Ported from LanPaint (https://github.com/scraed/LanPaint).
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Numerically stable helper functions for exponential / hyperbolic combos
# ---------------------------------------------------------------------------

def _expm1_x(x: torch.Tensor) -> torch.Tensor:
    """Compute (exp(x) - 1) / x with Taylor fallback near zero."""
    result = torch.special.expm1(x) / x
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
    mask = torch.abs(x) < 1e-2
    taylor = 1 + x / 2.0 + x**2 / 6.0
    return torch.where(mask, taylor, result)


def _expm1mx_x2(x: torch.Tensor) -> torch.Tensor:
    """Compute (exp(x) - 1 - x) / x^2 with Taylor fallback."""
    result = (torch.special.expm1(x) - x) / x**2
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
    mask = torch.abs(x**2) < 1e-2
    taylor = 1 / 2.0 + x / 6 + x**2 / 24 + x**3 / 120
    return torch.where(mask, taylor, result)


def _expm1mxmhx2_x3(x: torch.Tensor) -> torch.Tensor:
    """Compute (exp(x) - 1 - x - x^2/2) / x^3 with Taylor fallback."""
    result = (torch.special.expm1(x) - x - x**2 / 2) / x**3
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
    mask = torch.abs(x**3) < 1e-2
    taylor = 1 / 6 + x / 24 + x**2 / 120 + x**3 / 720 + x**4 / 5040
    return torch.where(mask, taylor, result)


def _exp_1mcosh_GD(gamma_t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Compute e^(-Gt) * (1 - cosh(Gt*sqrt(D))) / ((Gt)^2 * D)."""
    is_positive = delta > 0
    sqrt_abs_delta = torch.sqrt(torch.abs(delta))
    gamma_t_sqrt_delta = gamma_t * sqrt_abs_delta

    numerator_pos = (
        torch.exp(-gamma_t)
        - (torch.exp(gamma_t * (sqrt_abs_delta - 1))
           + torch.exp(gamma_t * (-sqrt_abs_delta - 1))) / 2
    )
    numerator_neg = torch.exp(-gamma_t) * (1 - torch.cos(gamma_t * sqrt_abs_delta))
    numerator = torch.where(is_positive, numerator_pos, numerator_neg)
    result = numerator / (delta * gamma_t**2)
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))

    mask = torch.abs(gamma_t_sqrt_delta**2) < 5e-2
    taylor = (-0.5 - gamma_t**2 / 24 * delta - gamma_t**4 / 720 * delta**2) * torch.exp(-gamma_t)
    return torch.where(mask, taylor, result)


def _exp_sinh_GsqrtD(gamma_t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Compute e^(-Gt) * sinh(Gt*sqrt(D)) / (Gt*sqrt(D))."""
    is_positive = delta > 0
    sqrt_abs_delta = torch.sqrt(torch.abs(delta))
    gamma_t_sqrt_delta = gamma_t * sqrt_abs_delta

    numerator_pos = (
        torch.exp(gamma_t * (sqrt_abs_delta - 1))
        - torch.exp(gamma_t * (-sqrt_abs_delta - 1))
    ) / 2
    result_pos = numerator_pos / gamma_t_sqrt_delta
    result_pos = torch.where(torch.isfinite(result_pos), result_pos, torch.zeros_like(result_pos))

    mask = torch.abs(gamma_t_sqrt_delta) < 1e-2
    taylor = (1 + gamma_t**2 / 6 * delta + gamma_t**4 / 120 * delta**2) * torch.exp(-gamma_t)
    result_pos = torch.where(mask, taylor, result_pos)

    result_neg = torch.exp(-gamma_t) * torch.special.sinc(gamma_t_sqrt_delta / torch.pi)
    return torch.where(is_positive, result_pos, result_neg)


def _exp_cosh(gamma_t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Compute e^(-Gt) * cosh(Gt*sqrt(D))."""
    ecg = _exp_1mcosh_GD(gamma_t, delta)
    return torch.exp(-gamma_t) - gamma_t**2 * delta * ecg


def _exp_sinh_sqrtD(gamma_t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Compute e^(-Gt) * sinh(Gt*sqrt(D)) / sqrt(D)."""
    return gamma_t * _exp_sinh_GsqrtD(gamma_t, delta)


def _zeta1(gamma_t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    half = gamma_t / 2
    ec = _exp_cosh(half, delta)
    es = _exp_sinh_sqrtD(half, delta)

    numerator = 1 - (ec + es)
    denominator = gamma_t * (1 - delta) / 4
    result = 1 - numerator / denominator
    result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))

    mask = torch.abs(denominator) < 5e-3
    t1 = _expm1_x(-gamma_t)
    t2 = _expm1mx_x2(-gamma_t)
    t3 = _expm1mxmhx2_x3(-gamma_t)
    taylor = (
        t1
        + (0.5 + t1 - 3 * t2) * denominator
        + (-1 / 6.0 + t1 / 2 - 4 * t2 + 10 * t3) * denominator**2
    )
    return torch.where(mask, taylor, result)


def _zeta2(gamma_t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return _exp_sinh_GsqrtD(gamma_t / 2, delta)


def _sig11(gamma_t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return (
        1 - torch.exp(-gamma_t)
        + gamma_t**2 * _exp_1mcosh_GD(gamma_t, delta)
        + _exp_sinh_sqrtD(gamma_t, delta)
    )


def _Zcoefs(gamma_t: torch.Tensor, delta: torch.Tensor):
    z1 = _zeta1(gamma_t, delta)
    z2 = _zeta2(gamma_t, delta)

    sq_total = 1 - z1 + gamma_t * (delta - 1) * (z1 - 1)**2 / 8
    amplitude = torch.sqrt(sq_total)

    zc1 = (gamma_t**0.5 * z2 / 2**0.5) / amplitude
    zc2 = zc1 * gamma_t * (-2 * _exp_1mcosh_GD(gamma_t, delta) / _sig11(gamma_t, delta))**0.5
    zc3 = torch.sqrt(torch.maximum(1 - zc1**2 - zc2**2, sq_total.new_zeros(sq_total.shape)))

    return zc1 * amplitude, zc2 * amplitude, zc3 * amplitude, amplitude


# ---------------------------------------------------------------------------
# Stochastic Harmonic Oscillator
# ---------------------------------------------------------------------------

class StochasticHarmonicOscillator:
    """Simulates a stochastic harmonic oscillator.

    SDE:
        dy(t) = q(t) dt
        dq(t) = -Gamma*A*y(t) dt + Gamma*C dt + Gamma*D dw(t) - Gamma*q(t) dt

    Parameters:
        Gamma: damping coefficient
        A: harmonic potential strength
        C: constant force term
        D: noise amplitude
    """

    def __init__(self, Gamma, A, C, D):
        self.Gamma = Gamma
        self.A = A
        self.C = C
        self.D = D
        self.Delta = 1 - 4 * A / Gamma

    def _sig11(self, gamma_t, delta):
        return _sig11(gamma_t, delta)

    def _sig22(self, gamma_t, delta):
        return 1 - _zeta1(2 * gamma_t, delta) + 2 * gamma_t * _exp_1mcosh_GD(gamma_t, delta)

    def dynamics(self, y0: torch.Tensor, v0: torch.Tensor | None, t) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance (position, velocity) by time t.

        Args:
            y0: initial position
            v0: initial velocity v(0) = q(0)/sqrt(Gamma). None → random init.
            t: time step

        Returns:
            (y(t), v(t))
        """
        zero = y0.new_zeros(1)
        Delta = self.Delta + zero
        Gamma_hat = self.Gamma * t + zero
        A = self.A + zero
        C = self.C + zero
        D = self.D + zero
        Gamma = self.Gamma + zero

        z1 = _zeta1(Gamma_hat, Delta)
        z2 = _zeta2(Gamma_hat, Delta)
        EE = 1 - Gamma_hat * z2

        if v0 is None:
            v0 = torch.randn_like(y0) * D / 2**0.5

        # Mean position and velocity
        term1 = (1 - z1) * (C * t - A * t * y0) + z2 * (Gamma**0.5) * v0 * t
        y_mean = term1 + y0
        v_mean = (1 - EE) * (C - A * y0) / (Gamma**0.5) + (EE - A * t * (1 - z1)) * v0

        # Covariance
        cov_yy = D**2 * t * self._sig22(Gamma_hat, Delta)
        cov_vv = D**2 * self._sig11(Gamma_hat, Delta) / 2
        cov_yv = (_zeta2(Gamma_hat, Delta) * Gamma_hat * D)**2 / 2 / (Gamma**0.5)

        # Cholesky factorization (manual for stability)
        batch_shape = y0.shape
        tol = 1e-8
        cov_yy = torch.clamp(cov_yy, min=tol)
        sd_yy = torch.sqrt(cov_yy)

        scale_tril = torch.zeros(*batch_shape, 2, 2, device=y0.device, dtype=y0.dtype)
        scale_tril[..., 0, 0] = sd_yy
        scale_tril[..., 1, 0] = cov_yv / sd_yy
        scale_tril[..., 1, 1] = torch.clamp(cov_vv - cov_yv**2 / cov_yy, min=tol)**0.5

        mean = torch.zeros(*batch_shape, 2, device=y0.device, dtype=y0.dtype)
        mean[..., 0] = y_mean
        mean[..., 1] = v_mean

        new_yv = torch.distributions.MultivariateNormal(
            loc=mean, scale_tril=scale_tril,
        ).sample()

        return new_yv[..., 0], new_yv[..., 1]


__all__ = ["StochasticHarmonicOscillator"]
