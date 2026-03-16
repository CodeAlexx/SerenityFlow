"""Early stopping logic for LanPaint Langevin iterations.

Monitors convergence of the masked region and stops iterations early
when the denoised prediction stabilizes.

Based on contributions by godnight10061 to the LanPaint project.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from .types import LangevinState


def _clamp01(val: float) -> float:
    return max(0.0, min(1.0, val))


def _abt_scale(abt_val: float) -> float:
    """Smooth scale based on noise level: 0 at extremes, 1 at mid-schedule."""
    abt_val = _clamp01(abt_val)
    return _clamp01(4.0 * abt_val * (1.0 - abt_val))


def _boundary_weight(latent_mask: torch.Tensor, inpaint_weight: torch.Tensor) -> Optional[torch.Tensor]:
    """4-neighbor boundary weight: unknown pixels adjacent to known pixels."""
    if latent_mask.dim() != 4:
        return None

    known = latent_mask > 0.5
    neighbor_known = torch.zeros_like(known)
    neighbor_known[:, :, 1:, :] |= known[:, :, :-1, :]
    neighbor_known[:, :, :-1, :] |= known[:, :, 1:, :]
    neighbor_known[:, :, :, 1:] |= known[:, :, :, :-1]
    neighbor_known[:, :, :, :-1] |= known[:, :, :, 1:]

    boundary = (~known) & neighbor_known
    return boundary.to(dtype=torch.float32) * inpaint_weight


def _weighted_mse(t1: torch.Tensor, t2: torch.Tensor, weight: torch.Tensor) -> float:
    diff_sq = (t1.to(dtype=torch.float32) - t2.to(dtype=torch.float32)) ** 2
    denom = torch.sum(weight) + 1e-12
    return float((torch.sum(diff_sq * weight) / denom).item())


class LanPaintEarlyStopper:
    """Per-step early-stop logic for LanPaint inner (Langevin) iterations."""

    @classmethod
    def create(
        cls,
        *,
        latent_mask: torch.Tensor,
        abt: torch.Tensor,
        threshold: float = 0.0,
        patience: int = 1,
    ) -> Optional[LanPaintEarlyStopper]:
        """Create a stopper if threshold > 0, else return None."""
        enabled = (threshold > 0.0) and (patience > 0)
        if not enabled:
            return None

        patience_eff = max(1, patience) + 1

        try:
            abt_val = float(torch.mean(abt).item())
        except (TypeError, ValueError):
            abt_val = 0.0

        threshold_eff = threshold * _abt_scale(abt_val)
        if threshold_eff <= 0.0:
            return None

        inpaint_weight = (1 - latent_mask).to(dtype=torch.float32)
        if float(torch.sum(inpaint_weight).item()) < 1e-6:
            return None

        ring_weight = _boundary_weight(latent_mask, inpaint_weight)

        return cls(
            threshold_eff=threshold_eff,
            patience_eff=patience_eff,
            inpaint_weight=inpaint_weight,
            ring_weight=ring_weight,
        )

    def __init__(
        self,
        *,
        threshold_eff: float,
        patience_eff: int,
        inpaint_weight: torch.Tensor,
        ring_weight: Optional[torch.Tensor],
    ) -> None:
        self.threshold_eff = threshold_eff
        self.patience_eff = patience_eff
        self.inpaint_weight = inpaint_weight
        self.ring_weight = ring_weight
        self.patience_counter = 0
        self.x0_anchor: Optional[torch.Tensor] = None

    def step(
        self,
        *,
        x_t_before: torch.Tensor,
        x_t_after: torch.Tensor,
        prev_args: Any,
        args: Any,
    ) -> bool:
        """Check convergence. Returns True if should stop."""
        inpaint = self.inpaint_weight

        def _get_x0(arg: Any) -> Optional[torch.Tensor]:
            if isinstance(arg, LangevinState):
                return arg.x0
            return None

        x0_prev = _get_x0(prev_args)
        x0_cur = _get_x0(args)

        if x0_prev is not None and x0_cur is not None:
            dist_inpaint = _weighted_mse(x0_cur, x0_prev, inpaint)
            dist_ring = _weighted_mse(x0_cur, x0_prev, self.ring_weight) if self.ring_weight is not None else None
            dist = dist_inpaint if dist_ring is None else max(dist_inpaint, dist_ring)
        else:
            dist = _weighted_mse(x_t_after, x_t_before, inpaint)
            x0_cur = None

        # Drift guard
        if x0_cur is not None:
            if dist <= self.threshold_eff:
                if self.x0_anchor is None:
                    self.x0_anchor = x0_cur.detach()
                else:
                    drift_inpaint = _weighted_mse(x0_cur, self.x0_anchor, inpaint)
                    drift_ring = (
                        _weighted_mse(x0_cur, self.x0_anchor, self.ring_weight)
                        if self.ring_weight is not None else None
                    )
                    dist_drift = drift_inpaint if drift_ring is None else max(drift_inpaint, drift_ring)
                    dist = max(dist, dist_drift)
            else:
                self.x0_anchor = None

        if dist <= self.threshold_eff:
            self.patience_counter += 1
        else:
            self.patience_counter = 0
            self.x0_anchor = None

        return self.patience_counter >= self.patience_eff


__all__ = ["LanPaintEarlyStopper"]
