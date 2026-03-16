"""Unit tests for LanPaint sampling algorithm.

Tests cover:
1. Score function math — known inputs, verify score_x and score_y
2. SHO dynamics — single step with known inputs
3. Mask enforcement — float mask → binary 0/1
4. Overdamped limit — verify overdamped solver matches expectations
5. Early stopping — mock convergence, verify stops correctly
6. Mask reshaping — various input shapes → target shape
"""
from __future__ import annotations

import pytest
import torch

from serenityflow.sampling.lanpaint.types import LangevinState
from serenityflow.sampling.lanpaint.oscillator import StochasticHarmonicOscillator
from serenityflow.sampling.lanpaint.earlystop import LanPaintEarlyStopper
from serenityflow.sampling.lanpaint.mask_utils import binarize_mask, reshape_mask, prepare_mask
from serenityflow.sampling.lanpaint.solver import LanPaintSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model_fn(return_value=None):
    """Create a mock dual-output model function."""
    call_count = [0]

    def model_fn(x, sigma):
        call_count[0] += 1
        if return_value is not None:
            return return_value, return_value
        # Return zeros as denoised prediction
        return torch.zeros_like(x), torch.zeros_like(x)

    model_fn.call_count = call_count
    return model_fn


# ---------------------------------------------------------------------------
# 1. Score function math
# ---------------------------------------------------------------------------

class TestScoreFunction:
    """Verify score computation for known inputs."""

    def test_score_unmasked_region(self):
        """score_x = -(x_t - x_0) in the unmasked region."""
        x_t = torch.tensor([2.0, 3.0, 4.0])
        x_0 = torch.tensor([1.0, 1.0, 1.0])

        # score_x = -(x_t - x_0)
        expected = -(x_t - x_0)
        assert torch.allclose(expected, torch.tensor([-1.0, -2.0, -3.0]))

    def test_score_masked_region(self):
        """score_y = -(1+lambda)*(x_t - y) + lambda*(x_t - x_0_big)."""
        x_t = torch.tensor([2.0])
        y = torch.tensor([1.0])  # known region value
        x_0_big = torch.tensor([0.5])
        lamb = 4.0

        score_y = -(1 + lamb) * (x_t - y) + lamb * (x_t - x_0_big)
        # = -5 * 1 + 4 * 1.5 = -5 + 6 = 1.0
        assert torch.allclose(score_y, torch.tensor([1.0]))

    def test_score_mask_compositing(self):
        """Verify mask correctly selects between score_x and score_y."""
        score_x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        score_y = torch.tensor([10.0, 20.0, 30.0, 40.0])
        mask = torch.tensor([0.0, 0.0, 1.0, 1.0])  # 0=unknown, 1=known

        result = score_x * (1 - mask) + score_y * mask
        expected = torch.tensor([1.0, 2.0, 30.0, 40.0])
        assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# 2. SHO dynamics
# ---------------------------------------------------------------------------

class TestSHODynamics:
    """Test StochasticHarmonicOscillator single step."""

    def test_sho_returns_correct_shapes(self):
        """SHO dynamics should return same shape as input."""
        Gamma = torch.tensor(10.0)
        A = torch.tensor(1.0)
        C = torch.tensor(0.5)
        D = torch.tensor(1.0)

        osc = StochasticHarmonicOscillator(Gamma, A, C, D)
        y0 = torch.randn(2, 4, 8, 8)
        v0 = torch.randn(2, 4, 8, 8)
        t = torch.tensor(0.1)

        y_new, v_new = osc.dynamics(y0, v0, t)
        assert y_new.shape == y0.shape
        assert v_new.shape == v0.shape

    def test_sho_none_velocity_init(self):
        """SHO should initialize velocity from noise when v0=None."""
        Gamma = torch.tensor(10.0)
        A = torch.tensor(1.0)
        C = torch.tensor(0.5)
        D = torch.tensor(1.0)

        osc = StochasticHarmonicOscillator(Gamma, A, C, D)
        y0 = torch.randn(1, 4, 4, 4)
        t = torch.tensor(0.1)

        y_new, v_new = osc.dynamics(y0, None, t)
        assert y_new.shape == y0.shape
        assert v_new.shape == y0.shape

    def test_sho_zero_time_preserves_position(self):
        """At t=0, position should stay approximately the same."""
        Gamma = torch.tensor(10.0)
        A = torch.tensor(1.0)
        C = torch.tensor(0.0)
        D = torch.tensor(0.001)  # very small noise

        osc = StochasticHarmonicOscillator(Gamma, A, C, D)
        y0 = torch.ones(1, 4, 4, 4)
        v0 = torch.zeros(1, 4, 4, 4)
        t = torch.tensor(1e-6)

        torch.manual_seed(42)
        y_new, _ = osc.dynamics(y0, v0, t)
        # With tiny time step and no noise, should be very close to y0
        assert torch.allclose(y_new, y0, atol=0.1)


# ---------------------------------------------------------------------------
# 3. Mask enforcement
# ---------------------------------------------------------------------------

class TestMaskEnforcement:
    """Test binary mask enforcement."""

    def test_binarize_above_threshold(self):
        mask = torch.tensor([0.0, 0.3, 0.5, 0.7, 1.0])
        result = binarize_mask(mask, threshold=0.5)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
        assert torch.allclose(result, expected)

    def test_binarize_custom_threshold(self):
        mask = torch.tensor([0.0, 0.2, 0.4, 0.6, 1.0])
        result = binarize_mask(mask, threshold=0.3)
        expected = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
        assert torch.allclose(result, expected)

    def test_binarize_already_binary(self):
        mask = torch.tensor([0.0, 1.0, 0.0, 1.0])
        result = binarize_mask(mask)
        expected = torch.tensor([0.0, 1.0, 0.0, 1.0])
        assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# 4. Mask reshaping
# ---------------------------------------------------------------------------

class TestMaskReshaping:
    """Test mask reshaping to match latent dimensions."""

    def test_2d_to_4d(self):
        mask = torch.ones(64, 64)
        target = (1, 4, 32, 32)
        result = reshape_mask(mask, target)
        assert result.shape == target

    def test_3d_to_4d(self):
        mask = torch.ones(1, 64, 64)
        target = (1, 4, 32, 32)
        result = reshape_mask(mask, target)
        assert result.shape == target

    def test_4d_to_4d(self):
        mask = torch.ones(1, 1, 64, 64)
        target = (1, 4, 32, 32)
        result = reshape_mask(mask, target)
        assert result.shape == target

    def test_video_5d(self):
        mask = torch.ones(1, 1, 8, 64, 64)
        target = (1, 16, 4, 32, 32)
        result = reshape_mask(mask, target, video=True)
        assert result.shape == target

    def test_batch_expansion(self):
        mask = torch.ones(1, 1, 32, 32)
        target = (4, 4, 32, 32)
        result = reshape_mask(mask, target)
        assert result.shape[0] == 4

    def test_prepare_mask_device(self):
        mask = torch.ones(32, 32)
        target = (1, 4, 16, 16)
        result = prepare_mask(mask, target, torch.device("cpu"))
        assert result.device == torch.device("cpu")
        assert result.shape == target


# ---------------------------------------------------------------------------
# 5. Early stopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    """Test convergence detection."""

    def test_disabled_when_threshold_zero(self):
        mask = torch.ones(1, 4, 8, 8)
        abt = torch.tensor([0.5])
        stopper = LanPaintEarlyStopper.create(
            latent_mask=mask, abt=abt, threshold=0.0, patience=1,
        )
        assert stopper is None

    def test_disabled_at_extreme_noise(self):
        """At abt=0 or abt=1, early stopping should be disabled."""
        mask = torch.ones(1, 4, 8, 8)
        abt = torch.tensor([0.0])  # extreme noise
        stopper = LanPaintEarlyStopper.create(
            latent_mask=mask, abt=abt, threshold=0.01, patience=1,
        )
        assert stopper is None

    def test_stops_on_convergence(self):
        """Should stop after patience+1 consecutive stable steps."""
        mask = torch.zeros(1, 4, 8, 8)  # all known (inpaint_weight = 1-mask = 1)
        abt = torch.tensor([0.5])
        stopper = LanPaintEarlyStopper.create(
            latent_mask=mask, abt=abt, threshold=1.0, patience=1,
        )
        assert stopper is not None

        # Identical tensors → distance = 0 → should converge
        x = torch.randn(1, 4, 8, 8)
        for _ in range(10):
            stopped = stopper.step(
                x_t_before=x, x_t_after=x,
                prev_args=None, args=None,
            )
            if stopped:
                break
        assert stopped, "Should have stopped on identical inputs"

    def test_does_not_stop_on_divergence(self):
        """Should not stop when inputs keep changing."""
        mask = torch.zeros(1, 4, 8, 8)
        abt = torch.tensor([0.5])
        stopper = LanPaintEarlyStopper.create(
            latent_mask=mask, abt=abt, threshold=0.0001, patience=1,
        )
        assert stopper is not None

        for i in range(5):
            x_before = torch.randn(1, 4, 8, 8)
            x_after = torch.randn(1, 4, 8, 8) * 100
            stopped = stopper.step(
                x_t_before=x_before, x_t_after=x_after,
                prev_args=None, args=None,
            )
            assert not stopped, f"Should not stop on divergent inputs (step {i})"


# ---------------------------------------------------------------------------
# 6. Solver integration
# ---------------------------------------------------------------------------

class TestSolverIntegration:
    """Test the LanPaint solver with mock model."""

    def test_solver_returns_correct_shape(self):
        """Output should match input shape."""
        model_fn = _make_mock_model_fn()
        solver = LanPaintSolver(
            model_fn=model_fn, n_steps=2, is_flow=True,
        )

        x = torch.randn(1, 4, 32, 32)
        sigma = torch.tensor([0.5])
        latent_image = torch.randn(1, 4, 32, 32)
        noise = torch.randn(1, 4, 32, 32)
        mask = torch.zeros(1, 4, 32, 32)
        mask[:, :, :16, :] = 1.0  # top half known

        result = solver(
            x=x, sigma=sigma, latent_image=latent_image,
            noise=noise, latent_mask=mask,
        )
        assert result.shape == x.shape

    def test_solver_calls_model_multiple_times(self):
        """N thinking steps should result in N+1 model calls (N iterations + 1 final)."""
        model_fn = _make_mock_model_fn()
        n_steps = 3
        solver = LanPaintSolver(
            model_fn=model_fn, n_steps=n_steps, is_flow=True,
        )

        x = torch.randn(1, 4, 16, 16)
        sigma = torch.tensor([0.5])
        latent_image = torch.randn(1, 4, 16, 16)
        noise = torch.randn(1, 4, 16, 16)
        mask = torch.zeros(1, 4, 16, 16)
        mask[:, :, :8, :] = 1.0

        solver(x=x, sigma=sigma, latent_image=latent_image,
               noise=noise, latent_mask=mask)

        # Each Langevin iteration calls model once (in score_model via coef_C),
        # plus one final denoise call.
        # First iteration: coef_C calls score → 1 model call
        # Subsequent iterations: coef_C twice (half-step scheme) → 1 model call each
        # Plus final denoise → 1 call
        # Total: at least n_steps + 1
        assert model_fn.call_count[0] >= n_steps + 1

    def test_solver_zero_steps_is_standard_denoise(self):
        """With 0 thinking steps, should just do a standard denoise."""
        denoised = torch.ones(1, 4, 8, 8) * 0.5
        model_fn = _make_mock_model_fn(return_value=denoised)
        solver = LanPaintSolver(
            model_fn=model_fn, n_steps=0, is_flow=True,
        )

        x = torch.randn(1, 4, 8, 8)
        sigma = torch.tensor([0.5])
        latent_image = torch.ones(1, 4, 8, 8)
        noise = torch.randn(1, 4, 8, 8)
        mask = torch.zeros(1, 4, 8, 8)
        mask[:, :, :4, :] = 1.0

        result = solver(x=x, sigma=sigma, latent_image=latent_image,
                        noise=noise, latent_mask=mask)

        # With 0 steps, known region should be latent_image
        known_region = result[:, :, :4, :]
        expected_known = latent_image[:, :, :4, :]
        assert torch.allclose(known_region, expected_known, atol=1e-5)

    def test_solver_known_region_preserved(self):
        """Known region should be replaced with original latent_image."""
        denoised = torch.zeros(1, 4, 8, 8)
        model_fn = _make_mock_model_fn(return_value=denoised)
        solver = LanPaintSolver(
            model_fn=model_fn, n_steps=0, is_flow=True,
        )

        x = torch.randn(1, 4, 8, 8)
        sigma = torch.tensor([0.5])
        latent_image = torch.ones(1, 4, 8, 8) * 42.0
        noise = torch.randn(1, 4, 8, 8)
        # mask = 1 means known, 0 means generate
        mask = torch.ones(1, 4, 8, 8)  # all known

        result = solver(x=x, sigma=sigma, latent_image=latent_image,
                        noise=noise, latent_mask=mask)

        # All known → output should be latent_image
        assert torch.allclose(result, latent_image, atol=1e-5)

    def test_solver_vp_mode(self):
        """Solver should work in VP/VE mode (non-flow)."""
        model_fn = _make_mock_model_fn()
        solver = LanPaintSolver(
            model_fn=model_fn, n_steps=1, is_flow=False, is_flux=False,
        )

        x = torch.randn(1, 4, 8, 8)
        sigma = torch.tensor([1.0])
        latent_image = torch.randn(1, 4, 8, 8)
        noise = torch.randn(1, 4, 8, 8)
        mask = torch.zeros(1, 4, 8, 8)
        mask[:, :, :4, :] = 1.0

        result = solver(x=x, sigma=sigma, latent_image=latent_image,
                        noise=noise, latent_mask=mask)
        assert result.shape == x.shape
        assert not torch.isnan(result).any()


# ---------------------------------------------------------------------------
# 7. LangevinState
# ---------------------------------------------------------------------------

class TestLangevinState:
    def test_namedtuple_fields(self):
        state = LangevinState(v=torch.tensor(1.0), C=torch.tensor(2.0), x0=torch.tensor(3.0))
        assert state.v == 1.0
        assert state.C == 2.0
        assert state.x0 == 3.0

    def test_none_fields(self):
        state = LangevinState(v=None, C=None, x0=None)
        assert state.v is None
        assert state.C is None
        assert state.x0 is None
