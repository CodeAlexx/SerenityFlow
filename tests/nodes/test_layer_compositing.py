"""Tests for layer_compositing nodes -- blend modes, color adjustments,
filters, layer effects, utility generators, and mask utilities."""
from __future__ import annotations

import os
import tempfile

import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def img_4x4():
    """4x4 RGB image, batch=1, filled with 0.5."""
    return torch.full((1, 4, 4, 3), 0.5)


@pytest.fixture
def img_32():
    """32x32 random RGB image, batch=1."""
    return torch.rand(1, 32, 32, 3)


@pytest.fixture
def black_4x4():
    return torch.zeros(1, 4, 4, 3)


@pytest.fixture
def white_4x4():
    return torch.ones(1, 4, 4, 3)


@pytest.fixture
def mask_32():
    return torch.ones(1, 32, 32)


# ---------------------------------------------------------------------------
# ImageBlendAdvanced — blend modes
# ---------------------------------------------------------------------------

class TestImageBlendAdvanced:
    def test_output_shape_and_dtype(self, img_32):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        result, mask = image_blend_advanced(img_32, img_32, "normal", 1.0)
        assert result.shape == img_32.shape
        assert result.dtype == torch.float32
        assert mask.shape == (1, 32, 32)

    def test_normal_blend_full_opacity(self, black_4x4, white_4x4):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        result, _ = image_blend_advanced(black_4x4, white_4x4, "normal", 1.0)
        assert torch.allclose(result, white_4x4, atol=1e-5)

    def test_normal_blend_half_opacity(self, black_4x4, white_4x4):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        result, _ = image_blend_advanced(black_4x4, white_4x4, "normal", 0.5)
        assert torch.allclose(result, torch.full_like(result, 0.5), atol=1e-5)

    def test_multiply_black_anything(self, img_4x4, black_4x4):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        result, _ = image_blend_advanced(img_4x4, black_4x4, "multiply", 1.0)
        assert result.max() < 0.01

    def test_multiply_white_identity(self, img_4x4, white_4x4):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        result, _ = image_blend_advanced(img_4x4, white_4x4, "multiply", 1.0)
        assert torch.allclose(result, img_4x4, atol=1e-5)

    def test_screen_white_anything(self, img_4x4, white_4x4):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        result, _ = image_blend_advanced(img_4x4, white_4x4, "screen", 1.0)
        assert torch.allclose(result, torch.ones_like(result), atol=1e-5)

    def test_screen_black_identity(self, img_4x4, black_4x4):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        result, _ = image_blend_advanced(img_4x4, black_4x4, "screen", 1.0)
        assert torch.allclose(result, img_4x4, atol=1e-5)

    def test_overlay_output_range(self, img_32):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        fg = torch.rand(1, 32, 32, 3)
        result, _ = image_blend_advanced(img_32, fg, "overlay", 1.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_soft_light_output_range(self, img_32):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        fg = torch.rand(1, 32, 32, 3)
        result, _ = image_blend_advanced(img_32, fg, "soft_light", 1.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_hard_light_output_range(self, img_32):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        fg = torch.rand(1, 32, 32, 3)
        result, _ = image_blend_advanced(img_32, fg, "hard_light", 1.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_color_dodge_output_clamped(self, img_4x4):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        fg = torch.full((1, 4, 4, 3), 0.9)
        result, _ = image_blend_advanced(img_4x4, fg, "color_dodge", 1.0)
        assert result.max() <= 1.0

    def test_color_burn_output_clamped(self, img_4x4):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        fg = torch.full((1, 4, 4, 3), 0.1)
        result, _ = image_blend_advanced(img_4x4, fg, "color_burn", 1.0)
        assert result.min() >= 0.0

    def test_darken(self, img_4x4, black_4x4):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        result, _ = image_blend_advanced(img_4x4, black_4x4, "darken", 1.0)
        assert torch.allclose(result, black_4x4, atol=1e-5)

    def test_lighten(self, img_4x4, white_4x4):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        result, _ = image_blend_advanced(img_4x4, white_4x4, "lighten", 1.0)
        assert torch.allclose(result, white_4x4, atol=1e-5)

    def test_difference_same_is_black(self, img_4x4):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        result, _ = image_blend_advanced(img_4x4, img_4x4, "difference", 1.0)
        assert result.max() < 0.01

    def test_exclusion_same(self, img_4x4):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        # exclusion(a,a) = a+a-2*a*a = 2a - 2a^2 = 2*0.5 - 2*0.25 = 0.5
        result, _ = image_blend_advanced(img_4x4, img_4x4, "exclusion", 1.0)
        assert torch.allclose(result, torch.full_like(result, 0.5), atol=1e-5)

    def test_add_clamped(self, img_4x4):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        result, _ = image_blend_advanced(img_4x4, img_4x4, "add", 1.0)
        assert result.max() <= 1.0

    def test_subtract(self, img_4x4, black_4x4):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        result, _ = image_blend_advanced(img_4x4, img_4x4, "subtract", 1.0)
        assert result.max() < 0.01

    def test_with_mask(self, black_4x4, white_4x4):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        mask = torch.zeros(1, 4, 4)
        mask[:, :2, :] = 1.0  # top half visible
        result, comp_mask = image_blend_advanced(black_4x4, white_4x4, "normal", 1.0, mask=mask)
        # Top half should be white, bottom half black
        assert result[:, :2, :, :].mean() > 0.9
        assert result[:, 2:, :, :].mean() < 0.1

    def test_scale(self, img_32):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        fg = torch.ones(1, 32, 32, 3)
        result, _ = image_blend_advanced(img_32, fg, "normal", 1.0, scale=0.5)
        assert result.shape == img_32.shape

    def test_rotation(self, img_32):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        fg = torch.ones(1, 32, 32, 3)
        result, _ = image_blend_advanced(img_32, fg, "normal", 0.5, rotation=45.0)
        assert result.shape == img_32.shape

    def test_position(self, img_32):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        fg = torch.ones(1, 8, 8, 3)
        result, _ = image_blend_advanced(img_32, fg, "normal", 1.0, x_percent=0.0, y_percent=0.0)
        assert result.shape == img_32.shape

    def test_batch_handling(self):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        bg = torch.rand(3, 16, 16, 3)
        fg = torch.rand(1, 16, 16, 3)  # single fg, broadcast
        result, mask = image_blend_advanced(bg, fg, "normal", 1.0)
        assert result.shape == (3, 16, 16, 3)
        assert mask.shape == (3, 16, 16)

    def test_single_pixel(self):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        bg = torch.full((1, 1, 1, 3), 0.3)
        fg = torch.full((1, 1, 1, 3), 0.7)
        result, _ = image_blend_advanced(bg, fg, "normal", 1.0)
        assert result.shape == (1, 1, 1, 3)

    def test_zero_opacity_is_identity(self, img_32):
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        fg = torch.ones(1, 32, 32, 3)
        result, _ = image_blend_advanced(img_32, fg, "normal", 0.0)
        assert torch.allclose(result, img_32, atol=1e-5)


# ---------------------------------------------------------------------------
# Color Adjustments
# ---------------------------------------------------------------------------

class TestExposure:
    def test_ev_zero_identity(self, img_32):
        from serenityflow.nodes.layer_compositing import exposure
        result = exposure(img_32, 0.0)
        assert torch.allclose(result[0], img_32, atol=1e-6)

    def test_ev_positive_brightens(self, img_4x4):
        from serenityflow.nodes.layer_compositing import exposure
        result = exposure(img_4x4, 1.0)
        assert result[0].mean() > img_4x4.mean()

    def test_ev_negative_darkens(self, img_4x4):
        from serenityflow.nodes.layer_compositing import exposure
        result = exposure(img_4x4, -1.0)
        assert result[0].mean() < img_4x4.mean()

    def test_output_clamped(self, white_4x4):
        from serenityflow.nodes.layer_compositing import exposure
        result = exposure(white_4x4, 3.0)
        assert result[0].max() <= 1.0

    def test_batch(self):
        from serenityflow.nodes.layer_compositing import exposure
        img = torch.rand(4, 8, 8, 3)
        result = exposure(img, 0.5)
        assert result[0].shape == (4, 8, 8, 3)


class TestColorTemperature:
    def test_neutral_6500(self, img_32):
        from serenityflow.nodes.layer_compositing import color_temperature
        result = color_temperature(img_32, 6500.0)
        assert torch.allclose(result[0], img_32, atol=1e-5)

    def test_warm_boosts_red(self, img_4x4):
        from serenityflow.nodes.layer_compositing import color_temperature
        result = color_temperature(img_4x4, 3000.0)
        # Warmer should boost red relative to blue
        assert result[0][..., 0].mean() > result[0][..., 2].mean()

    def test_cool_boosts_blue(self, img_4x4):
        from serenityflow.nodes.layer_compositing import color_temperature
        result = color_temperature(img_4x4, 10000.0)
        assert result[0][..., 2].mean() > result[0][..., 0].mean()

    def test_output_clamped(self, white_4x4):
        from serenityflow.nodes.layer_compositing import color_temperature
        result = color_temperature(white_4x4, 2000.0)
        assert result[0].min() >= 0.0
        assert result[0].max() <= 1.0


class TestLUTApply:
    def test_identity_lut(self, img_4x4):
        from serenityflow.nodes.layer_compositing import lut_apply
        # Create a minimal identity 2x2x2 LUT
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cube", delete=False) as f:
            f.write("LUT_3D_SIZE 2\n")
            for r in (0.0, 1.0):
                for g in (0.0, 1.0):
                    for b in (0.0, 1.0):
                        f.write(f"{r} {g} {b}\n")
            path = f.name
        try:
            result = lut_apply(img_4x4, path)
            assert result[0].shape == img_4x4.shape
            assert torch.allclose(result[0], img_4x4, atol=0.05)
        finally:
            os.unlink(path)

    def test_invalid_path_passthrough(self, img_4x4):
        from serenityflow.nodes.layer_compositing import lut_apply
        result = lut_apply(img_4x4, "/nonexistent/path.cube")
        assert torch.allclose(result[0], img_4x4)

    def test_output_range(self, img_32):
        from serenityflow.nodes.layer_compositing import lut_apply
        # Inversion LUT
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cube", delete=False) as f:
            f.write("LUT_3D_SIZE 2\n")
            for r in (0.0, 1.0):
                for g in (0.0, 1.0):
                    for b in (0.0, 1.0):
                        f.write(f"{1-r} {1-g} {1-b}\n")
            path = f.name
        try:
            result = lut_apply(img_32, path)
            assert result[0].min() >= 0.0
            assert result[0].max() <= 1.0
        finally:
            os.unlink(path)


class TestAutoAdjust:
    def test_already_full_range(self):
        from serenityflow.nodes.layer_compositing import auto_adjust
        img = torch.zeros(1, 4, 4, 3)
        img[0, 0, 0, :] = 1.0  # one white pixel
        result = auto_adjust(img)
        assert result[0].min() >= 0.0
        assert result[0].max() <= 1.0

    def test_stretches_range(self):
        from serenityflow.nodes.layer_compositing import auto_adjust
        img = torch.full((1, 4, 4, 3), 0.4)
        img[0, 0, 0, :] = 0.6
        result = auto_adjust(img)
        # After stretch, min should be 0, max should be 1
        assert result[0].min() < 0.01
        assert result[0].max() > 0.99

    def test_single_value_no_crash(self):
        from serenityflow.nodes.layer_compositing import auto_adjust
        img = torch.full((1, 4, 4, 3), 0.5)
        result = auto_adjust(img)
        assert result[0].shape == img.shape


class TestNegative:
    def test_invert(self, black_4x4, white_4x4):
        from serenityflow.nodes.layer_compositing import negative
        result = negative(black_4x4)
        assert torch.allclose(result[0], white_4x4)

    def test_double_invert_identity(self, img_32):
        from serenityflow.nodes.layer_compositing import negative
        result = negative(negative(img_32)[0])
        assert torch.allclose(result[0], img_32, atol=1e-6)

    def test_mid_gray(self, img_4x4):
        from serenityflow.nodes.layer_compositing import negative
        result = negative(img_4x4)
        assert torch.allclose(result[0], img_4x4, atol=1e-6)  # 1-0.5 = 0.5


# ---------------------------------------------------------------------------
# Image Filters
# ---------------------------------------------------------------------------

class TestAddGrain:
    def test_output_shape(self, img_32):
        from serenityflow.nodes.layer_compositing import add_grain
        result = add_grain(img_32, 0.1, 1, 42)
        assert result[0].shape == img_32.shape

    def test_zero_amount_identity(self, img_32):
        from serenityflow.nodes.layer_compositing import add_grain
        result = add_grain(img_32, 0.0, 1, 42)
        assert torch.allclose(result[0], img_32)

    def test_deterministic_with_seed(self, img_32):
        from serenityflow.nodes.layer_compositing import add_grain
        r1 = add_grain(img_32, 0.1, 1, 42)
        r2 = add_grain(img_32, 0.1, 1, 42)
        assert torch.allclose(r1[0], r2[0])

    def test_different_seeds_differ(self, img_32):
        from serenityflow.nodes.layer_compositing import add_grain
        r1 = add_grain(img_32, 0.5, 1, 42)
        r2 = add_grain(img_32, 0.5, 1, 99)
        assert not torch.allclose(r1[0], r2[0])

    def test_output_clamped(self, white_4x4):
        from serenityflow.nodes.layer_compositing import add_grain
        result = add_grain(white_4x4, 1.0, 1, 42)
        assert result[0].min() >= 0.0
        assert result[0].max() <= 1.0

    def test_grain_size(self, img_32):
        from serenityflow.nodes.layer_compositing import add_grain
        result = add_grain(img_32, 0.1, 4, 42)
        assert result[0].shape == img_32.shape


class TestChannelShake:
    def test_zero_offsets_identity(self, img_32):
        from serenityflow.nodes.layer_compositing import channel_shake
        result = channel_shake(img_32, 0, 0, 0, 0, 0, 0)
        assert torch.allclose(result[0], img_32)

    def test_output_shape(self, img_32):
        from serenityflow.nodes.layer_compositing import channel_shake
        result = channel_shake(img_32, 2, 0, -2, 0, 0, 3)
        assert result[0].shape == img_32.shape

    def test_single_pixel(self):
        from serenityflow.nodes.layer_compositing import channel_shake
        img = torch.rand(1, 1, 1, 3)
        result = channel_shake(img, 1, 1, 0, 0, -1, -1)
        assert result[0].shape == (1, 1, 1, 3)

    def test_output_clamped(self, img_32):
        from serenityflow.nodes.layer_compositing import channel_shake
        result = channel_shake(img_32, 5, 5, -5, -5, 3, -3)
        assert result[0].min() >= 0.0
        assert result[0].max() <= 1.0


# ---------------------------------------------------------------------------
# Layer Style Effects
# ---------------------------------------------------------------------------

class TestDropShadow:
    def test_output_shape(self, img_32, mask_32):
        from serenityflow.nodes.layer_compositing import drop_shadow
        result = drop_shadow(img_32, mask_32, 135.0, 10, 5, 0.0, 0.0, 0.0, 0.5)
        assert result[0].shape == img_32.shape

    def test_output_clamped(self, img_32, mask_32):
        from serenityflow.nodes.layer_compositing import drop_shadow
        result = drop_shadow(img_32, mask_32, 135.0, 10, 5, 0.0, 0.0, 0.0, 1.0)
        assert result[0].min() >= 0.0
        assert result[0].max() <= 1.0

    def test_zero_opacity_no_shadow(self, img_32, mask_32):
        from serenityflow.nodes.layer_compositing import drop_shadow
        result = drop_shadow(img_32, mask_32, 135.0, 10, 5, 0.0, 0.0, 0.0, 0.0)
        # With zero opacity shadow, result should be close to original (composited through mask)
        assert result[0].shape == img_32.shape

    def test_batch(self):
        from serenityflow.nodes.layer_compositing import drop_shadow
        img = torch.rand(3, 16, 16, 3)
        mask = torch.ones(3, 16, 16)
        result = drop_shadow(img, mask, 45.0, 5, 3, 0.0, 0.0, 0.0, 0.5)
        assert result[0].shape == (3, 16, 16, 3)


class TestInnerGlow:
    def test_output_shape(self, img_32, mask_32):
        from serenityflow.nodes.layer_compositing import inner_glow
        result = inner_glow(img_32, mask_32, 1.0, 1.0, 1.0, 5, 0.5)
        assert result[0].shape == img_32.shape

    def test_output_clamped(self, img_32, mask_32):
        from serenityflow.nodes.layer_compositing import inner_glow
        result = inner_glow(img_32, mask_32, 1.0, 1.0, 1.0, 10, 1.0)
        assert result[0].min() >= 0.0
        assert result[0].max() <= 1.0

    def test_zero_opacity(self, img_32, mask_32):
        from serenityflow.nodes.layer_compositing import inner_glow
        result = inner_glow(img_32, mask_32, 1.0, 0.0, 0.0, 5, 0.0)
        assert torch.allclose(result[0], img_32, atol=1e-5)

    def test_zero_size(self, img_32, mask_32):
        from serenityflow.nodes.layer_compositing import inner_glow
        result = inner_glow(img_32, mask_32, 1.0, 1.0, 1.0, 0, 0.5)
        # Size 0 means no edge detected, so no glow
        assert torch.allclose(result[0], img_32, atol=1e-5)


class TestInnerShadow:
    def test_output_shape(self, img_32, mask_32):
        from serenityflow.nodes.layer_compositing import inner_shadow
        result = inner_shadow(img_32, mask_32, 135.0, 5, 5, 0.0, 0.0, 0.0, 0.5)
        assert result[0].shape == img_32.shape

    def test_output_clamped(self, img_32, mask_32):
        from serenityflow.nodes.layer_compositing import inner_shadow
        result = inner_shadow(img_32, mask_32, 135.0, 5, 5, 0.0, 0.0, 0.0, 1.0)
        assert result[0].min() >= 0.0
        assert result[0].max() <= 1.0

    def test_batch(self):
        from serenityflow.nodes.layer_compositing import inner_shadow
        img = torch.rand(2, 16, 16, 3)
        mask = torch.ones(2, 16, 16)
        result = inner_shadow(img, mask, 90.0, 3, 3, 0.0, 0.0, 0.0, 0.5)
        assert result[0].shape == (2, 16, 16, 3)


class TestGradientOverlay:
    def test_output_shape(self, img_32):
        from serenityflow.nodes.layer_compositing import gradient_overlay
        result = gradient_overlay(img_32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, "normal", 0.5)
        assert result[0].shape == img_32.shape

    def test_zero_opacity_identity(self, img_32):
        from serenityflow.nodes.layer_compositing import gradient_overlay
        result = gradient_overlay(img_32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, "normal", 0.0)
        assert torch.allclose(result[0], img_32, atol=1e-5)

    def test_output_clamped(self, img_32):
        from serenityflow.nodes.layer_compositing import gradient_overlay
        result = gradient_overlay(img_32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 45.0, "add", 1.0)
        assert result[0].min() >= 0.0
        assert result[0].max() <= 1.0


class TestColorOverlay:
    def test_output_shape(self, img_32):
        from serenityflow.nodes.layer_compositing import color_overlay
        result = color_overlay(img_32, 1.0, 0.0, 0.0, "normal", 0.5)
        assert result[0].shape == img_32.shape

    def test_full_opacity_normal(self, img_4x4):
        from serenityflow.nodes.layer_compositing import color_overlay
        result = color_overlay(img_4x4, 1.0, 0.0, 0.0, "normal", 1.0)
        assert torch.allclose(result[0][..., 0], torch.ones(1, 4, 4), atol=1e-5)
        assert torch.allclose(result[0][..., 1], torch.zeros(1, 4, 4), atol=1e-5)

    def test_zero_opacity_identity(self, img_32):
        from serenityflow.nodes.layer_compositing import color_overlay
        result = color_overlay(img_32, 1.0, 0.0, 0.0, "normal", 0.0)
        assert torch.allclose(result[0], img_32, atol=1e-5)


# ---------------------------------------------------------------------------
# Utility Nodes
# ---------------------------------------------------------------------------

class TestColorImage:
    def test_shape(self):
        from serenityflow.nodes.layer_compositing import color_image
        result = color_image(64, 32, 1.0, 0.5, 0.0)
        assert result[0].shape == (1, 32, 64, 3)

    def test_color_values(self):
        from serenityflow.nodes.layer_compositing import color_image
        result = color_image(4, 4, 0.2, 0.4, 0.6)
        assert torch.allclose(result[0][..., 0], torch.full((1, 4, 4), 0.2))
        assert torch.allclose(result[0][..., 1], torch.full((1, 4, 4), 0.4))
        assert torch.allclose(result[0][..., 2], torch.full((1, 4, 4), 0.6))


class TestGradientImage:
    def test_linear_shape(self):
        from serenityflow.nodes.layer_compositing import gradient_image
        result = gradient_image(128, 64, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, "linear", 0.0)
        assert result[0].shape == (1, 64, 128, 3)

    def test_radial_shape(self):
        from serenityflow.nodes.layer_compositing import gradient_image
        result = gradient_image(64, 64, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, "radial", 0.0, 0.5, 0.5)
        assert result[0].shape == (1, 64, 64, 3)

    def test_range(self):
        from serenityflow.nodes.layer_compositing import gradient_image
        result = gradient_image(64, 64, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, "linear", 0.0)
        assert result[0].min() >= 0.0
        assert result[0].max() <= 1.0

    def test_single_pixel(self):
        from serenityflow.nodes.layer_compositing import gradient_image
        result = gradient_image(1, 1, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, "linear", 0.0)
        assert result[0].shape == (1, 1, 1, 3)


class TestSimpleTextImage:
    def test_output_shape(self):
        from serenityflow.nodes.layer_compositing import simple_text_image
        result = simple_text_image("Hello", 256, 64, 24, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        assert result[0].shape == (1, 64, 256, 3)

    def test_dtype_float32(self):
        from serenityflow.nodes.layer_compositing import simple_text_image
        result = simple_text_image("Test", 64, 32, 16, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        assert result[0].dtype == torch.float32

    def test_output_range(self):
        from serenityflow.nodes.layer_compositing import simple_text_image
        result = simple_text_image("X", 64, 64, 32, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5)
        assert result[0].min() >= 0.0
        assert result[0].max() <= 1.0


class TestExtendCanvas:
    def test_shape(self, img_4x4):
        from serenityflow.nodes.layer_compositing import extend_canvas
        result = extend_canvas(img_4x4, top=2, bottom=3, left=1, right=4, fill_r=0.0, fill_g=0.0, fill_b=0.0)
        assert result[0].shape == (1, 9, 9, 3)  # 4+2+3=9, 4+1+4=9

    def test_original_preserved(self, img_4x4):
        from serenityflow.nodes.layer_compositing import extend_canvas
        result = extend_canvas(img_4x4, top=2, bottom=2, left=2, right=2, fill_r=0.0, fill_g=0.0, fill_b=0.0)
        inner = result[0][:, 2:6, 2:6, :]
        assert torch.allclose(inner, img_4x4, atol=1e-6)

    def test_fill_color(self):
        from serenityflow.nodes.layer_compositing import extend_canvas
        img = torch.zeros(1, 2, 2, 3)
        result = extend_canvas(img, top=1, bottom=1, left=1, right=1, fill_r=1.0, fill_g=0.0, fill_b=0.0)
        # Top-left corner should be the fill color
        assert result[0][0, 0, 0, 0] == 1.0
        assert result[0][0, 0, 0, 1] == 0.0

    def test_zero_padding_identity(self, img_32):
        from serenityflow.nodes.layer_compositing import extend_canvas
        result = extend_canvas(img_32, 0, 0, 0, 0, 0.0, 0.0, 0.0)
        assert torch.allclose(result[0], img_32)


class TestImageReel:
    def test_shape_2x2(self):
        from serenityflow.nodes.layer_compositing import image_reel
        imgs = torch.rand(4, 16, 16, 3)
        result = image_reel(imgs, columns=2, rows=2, spacing=0)
        assert result[0].shape == (1, 32, 32, 3)

    def test_shape_with_spacing(self):
        from serenityflow.nodes.layer_compositing import image_reel
        imgs = torch.rand(4, 16, 16, 3)
        result = image_reel(imgs, columns=2, rows=2, spacing=4)
        assert result[0].shape == (1, 36, 36, 3)

    def test_auto_rows(self):
        from serenityflow.nodes.layer_compositing import image_reel
        imgs = torch.rand(6, 10, 10, 3)
        result = image_reel(imgs, columns=3, rows=0, spacing=0)
        # 6 images / 3 columns = 2 rows
        assert result[0].shape == (1, 20, 30, 3)

    def test_single_image(self):
        from serenityflow.nodes.layer_compositing import image_reel
        imgs = torch.rand(1, 8, 8, 3)
        result = image_reel(imgs, columns=1, rows=1, spacing=0)
        assert result[0].shape == (1, 8, 8, 3)


# ---------------------------------------------------------------------------
# Mask Utilities
# ---------------------------------------------------------------------------

class TestMaskBlur:
    def test_output_shape(self, mask_32):
        from serenityflow.nodes.layer_compositing import mask_blur
        result = mask_blur(mask_32, 3)
        assert result[0].shape == mask_32.shape

    def test_zero_radius_identity(self, mask_32):
        from serenityflow.nodes.layer_compositing import mask_blur
        result = mask_blur(mask_32, 0)
        assert torch.allclose(result[0], mask_32)

    def test_output_clamped(self):
        from serenityflow.nodes.layer_compositing import mask_blur
        mask = torch.rand(1, 32, 32)
        result = mask_blur(mask, 5)
        assert result[0].min() >= 0.0
        assert result[0].max() <= 1.0

    def test_2d_mask(self):
        from serenityflow.nodes.layer_compositing import mask_blur
        mask = torch.rand(32, 32)
        result = mask_blur(mask, 3)
        assert result[0].ndim == 3  # promoted to BHW


class TestMaskExpand:
    def test_output_shape(self, mask_32):
        from serenityflow.nodes.layer_compositing import mask_expand
        result = mask_expand(mask_32, 5)
        assert result[0].shape == mask_32.shape

    def test_zero_expand_identity(self, mask_32):
        from serenityflow.nodes.layer_compositing import mask_expand
        result = mask_expand(mask_32, 0)
        assert torch.allclose(result[0], mask_32)

    def test_expansion(self):
        from serenityflow.nodes.layer_compositing import mask_expand
        mask = torch.zeros(1, 32, 32)
        mask[:, 15:17, 15:17] = 1.0
        result = mask_expand(mask, 2)
        # Expanded mask should have more nonzero pixels
        assert result[0].sum() > mask.sum()

    def test_2d_input(self):
        from serenityflow.nodes.layer_compositing import mask_expand
        mask = torch.zeros(32, 32)
        mask[15:17, 15:17] = 1.0
        result = mask_expand(mask, 2)
        assert result[0].ndim == 3


class TestMaskErode:
    def test_output_shape(self, mask_32):
        from serenityflow.nodes.layer_compositing import mask_erode
        result = mask_erode(mask_32, 5)
        assert result[0].shape == mask_32.shape

    def test_zero_erode_identity(self, mask_32):
        from serenityflow.nodes.layer_compositing import mask_erode
        result = mask_erode(mask_32, 0)
        assert torch.allclose(result[0], mask_32)

    def test_erosion(self):
        from serenityflow.nodes.layer_compositing import mask_erode
        mask = torch.ones(1, 32, 32)
        result = mask_erode(mask, 3)
        # Eroded full mask should have zero edges
        assert result[0][:, 0, :].sum() == 0.0
        assert result[0][:, -1, :].sum() == 0.0

    def test_small_mask_fully_eroded(self):
        from serenityflow.nodes.layer_compositing import mask_erode
        mask = torch.ones(1, 4, 4)
        result = mask_erode(mask, 3)
        # 4x4 mask eroded by 3 should be all zero
        assert result[0].sum() == 0.0


class TestCropByMask:
    def test_basic_crop(self):
        from serenityflow.nodes.layer_compositing import crop_by_mask
        img = torch.rand(1, 64, 64, 3)
        mask = torch.zeros(1, 64, 64)
        mask[:, 20:40, 10:50] = 1.0
        result, stitch = crop_by_mask(img, mask, padding=0)
        assert result.shape[1] == 20  # 40-20
        assert result.shape[2] == 40  # 50-10
        assert stitch["x"] == 10
        assert stitch["y"] == 20

    def test_with_padding(self):
        from serenityflow.nodes.layer_compositing import crop_by_mask
        img = torch.rand(1, 64, 64, 3)
        mask = torch.zeros(1, 64, 64)
        mask[:, 20:40, 20:40] = 1.0
        result, stitch = crop_by_mask(img, mask, padding=5)
        assert result.shape[1] == 30  # 20 + 5*2
        assert result.shape[2] == 30
        assert stitch["y"] == 15
        assert stitch["x"] == 15

    def test_empty_mask_returns_full(self):
        from serenityflow.nodes.layer_compositing import crop_by_mask
        img = torch.rand(1, 32, 32, 3)
        mask = torch.zeros(1, 32, 32)
        result, stitch = crop_by_mask(img, mask, padding=0)
        assert result.shape == img.shape
        assert stitch["x"] == 0 and stitch["y"] == 0

    def test_2d_mask(self):
        from serenityflow.nodes.layer_compositing import crop_by_mask
        img = torch.rand(1, 32, 32, 3)
        mask = torch.zeros(32, 32)
        mask[10:20, 10:20] = 1.0
        result, stitch = crop_by_mask(img, mask, padding=0)
        assert result.shape[1] == 10


class TestRestoreCropBox:
    def test_roundtrip(self):
        from serenityflow.nodes.layer_compositing import crop_by_mask, restore_crop_box
        img = torch.rand(1, 64, 64, 3)
        mask = torch.zeros(1, 64, 64)
        mask[:, 20:40, 20:40] = 1.0
        cropped, stitch = crop_by_mask(img, mask, padding=0)

        # Modify cropped region
        modified = torch.ones_like(cropped)
        bg = img.clone()
        result = restore_crop_box(modified, stitch, bg)
        # restore_crop_box returns (IMAGE,)
        assert result[0].shape == img.shape
        # The cropped region should now be all ones
        assert result[0][:, 20:40, 20:40, :].mean() > 0.99

    def test_output_clamped(self):
        from serenityflow.nodes.layer_compositing import restore_crop_box
        cropped = torch.ones(1, 10, 10, 3) * 2.0  # intentionally >1
        stitch = {"x": 0, "y": 0, "orig_h": 32, "orig_w": 32}
        bg = torch.zeros(1, 32, 32, 3)
        result = restore_crop_box(cropped, stitch, bg)
        # result is (IMAGE,) tuple
        assert result[0].max() <= 1.0


# ---------------------------------------------------------------------------
# Edge cases across nodes
# ---------------------------------------------------------------------------

class TestEdgeCasesCompositing:
    def test_blend_advanced_different_sizes(self):
        """Foreground smaller than background."""
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        bg = torch.rand(1, 64, 64, 3)
        fg = torch.rand(1, 16, 16, 3)
        result, mask = image_blend_advanced(bg, fg, "normal", 1.0)
        assert result.shape == bg.shape
        assert mask.shape == (1, 64, 64)

    def test_blend_advanced_fg_larger_than_bg(self):
        """Foreground larger than background -- should clip."""
        from serenityflow.nodes.layer_compositing import image_blend_advanced
        bg = torch.rand(1, 16, 16, 3)
        fg = torch.rand(1, 64, 64, 3)
        result, mask = image_blend_advanced(bg, fg, "normal", 1.0)
        assert result.shape == bg.shape

    def test_exposure_batch_1_pixel(self):
        from serenityflow.nodes.layer_compositing import exposure
        img = torch.full((1, 1, 1, 3), 0.5)
        result = exposure(img, 1.0)
        assert result[0].shape == (1, 1, 1, 3)

    def test_extend_canvas_batch(self):
        from serenityflow.nodes.layer_compositing import extend_canvas
        img = torch.rand(3, 8, 8, 3)
        result = extend_canvas(img, 2, 2, 2, 2, 0.0, 0.0, 0.0)
        assert result[0].shape == (3, 12, 12, 3)

    def test_mask_blur_all_zeros(self):
        from serenityflow.nodes.layer_compositing import mask_blur
        mask = torch.zeros(1, 16, 16)
        result = mask_blur(mask, 3)
        assert result[0].sum() == 0.0

    def test_image_reel_fewer_images_than_slots(self):
        from serenityflow.nodes.layer_compositing import image_reel
        imgs = torch.rand(2, 8, 8, 3)
        result = image_reel(imgs, columns=3, rows=2, spacing=0)
        # 3x2 grid with only 2 images; remaining slots should be black (zeros)
        assert result[0].shape == (1, 16, 24, 3)
