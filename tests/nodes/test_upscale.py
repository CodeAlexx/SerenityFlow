"""Tests for upscale & hires fix nodes."""
from __future__ import annotations

import math

import torch
import pytest

from serenityflow.bridge.types import unwrap_latent, wrap_latent
from serenityflow.nodes.upscale import (
    _slerp_element,
    bislerp,
    _upscale_latent,
    latent_upscale_bislerp,
    latent_upscale_by_bislerp,
    latent_composite_masked,
    _generate_tiles,
    _generate_seam_tiles,
    _create_feather_mask,
)


# ---------------------------------------------------------------------------
# bislerp unit tests
# ---------------------------------------------------------------------------

class TestBislerp:
    def test_identity_same_size(self):
        """bislerp with same target size returns identical tensor."""
        x = torch.randn(1, 4, 8, 8)
        result = bislerp(x, width=8, height=8)
        assert torch.allclose(result, x), "Same-size bislerp should be identity"

    def test_output_shape(self):
        """bislerp produces correct output dimensions."""
        x = torch.randn(1, 4, 8, 8)
        result = bislerp(x, width=16, height=16)
        assert result.shape == (1, 4, 16, 16)

    def test_output_shape_nonsquare(self):
        """bislerp handles non-square targets."""
        x = torch.randn(2, 4, 8, 8)
        result = bislerp(x, width=12, height=20)
        assert result.shape == (2, 4, 20, 12)

    def test_batch_preserved(self):
        """bislerp preserves batch dimension."""
        x = torch.randn(3, 4, 8, 8)
        result = bislerp(x, width=16, height=16)
        assert result.shape[0] == 3

    def test_downscale(self):
        """bislerp can downscale."""
        x = torch.randn(1, 4, 16, 16)
        result = bislerp(x, width=8, height=8)
        assert result.shape == (1, 4, 8, 8)

    def test_finite_output(self):
        """bislerp output is finite (no NaN/Inf)."""
        x = torch.randn(1, 4, 8, 8)
        result = bislerp(x, width=16, height=16)
        assert torch.isfinite(result).all(), "bislerp output contains non-finite values"


class TestSlerpElement:
    def test_t_zero_returns_a(self):
        """slerp at t=0 should return a."""
        a = torch.randn(10, 4)
        b = torch.randn(10, 4)
        t = torch.zeros(10)
        result = _slerp_element(a, b, t)
        assert torch.allclose(result, a, atol=1e-5)

    def test_t_one_returns_b(self):
        """slerp at t=1 should return b."""
        a = torch.randn(10, 4)
        b = torch.randn(10, 4)
        t = torch.ones(10)
        result = _slerp_element(a, b, t)
        assert torch.allclose(result, b, atol=1e-5)

    def test_parallel_vectors_fallback(self):
        """slerp with parallel vectors should fall back to lerp."""
        a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        b = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # same direction
        t = torch.tensor([0.5])
        result = _slerp_element(a, b, t)
        expected = a * 0.5 + b * 0.5
        assert torch.allclose(result, expected, atol=1e-5)

    def test_known_orthogonal(self):
        """slerp between orthogonal unit vectors at t=0.5 should bisect."""
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[0.0, 1.0]])
        t = torch.tensor([0.5])
        result = _slerp_element(a, b, t)
        # At t=0.5, slerp of orthogonal unit vectors = [cos(pi/4), sin(pi/4)]
        expected = torch.tensor([[math.cos(math.pi / 4), math.sin(math.pi / 4)]])
        assert torch.allclose(result, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# LatentUpscaleBislerp
# ---------------------------------------------------------------------------

class TestLatentUpscaleBislerp:
    def test_bilinear_doubles_size(self):
        """LatentUpscaleBislerp with bilinear doubles latent dimensions."""
        latent = wrap_latent(torch.randn(1, 4, 8, 8))
        (result,) = latent_upscale_bislerp(latent, upscale_method="bilinear", width=128, height=128)
        out = unwrap_latent(result)
        # 128 // 8 = 16
        assert out.shape == (1, 4, 16, 16)

    def test_bislerp_method(self):
        """LatentUpscaleBislerp with bislerp produces correct shape."""
        latent = wrap_latent(torch.randn(1, 4, 8, 8))
        (result,) = latent_upscale_bislerp(latent, upscale_method="bislerp", width=128, height=128)
        out = unwrap_latent(result)
        assert out.shape == (1, 4, 16, 16)

    def test_nearest_method(self):
        """LatentUpscaleBislerp with nearest."""
        latent = wrap_latent(torch.randn(1, 4, 8, 8))
        (result,) = latent_upscale_bislerp(latent, upscale_method="nearest", width=128, height=128)
        out = unwrap_latent(result)
        assert out.shape == (1, 4, 16, 16)

    def test_output_dtype(self):
        """Output dtype matches input."""
        latent = wrap_latent(torch.randn(1, 4, 8, 8))
        (result,) = latent_upscale_bislerp(latent, upscale_method="bislerp", width=128, height=128)
        assert unwrap_latent(result).dtype == torch.float32


# ---------------------------------------------------------------------------
# LatentUpscaleByBislerp
# ---------------------------------------------------------------------------

class TestLatentUpscaleByBislerp:
    def test_2x_scale(self):
        """LatentUpscaleByBislerp with 2x scale factor."""
        latent = wrap_latent(torch.randn(1, 4, 8, 8))
        (result,) = latent_upscale_by_bislerp(latent, upscale_method="bilinear", scale_by=2.0)
        out = unwrap_latent(result)
        assert out.shape == (1, 4, 16, 16)

    def test_bislerp_2x(self):
        """LatentUpscaleByBislerp with bislerp 2x."""
        latent = wrap_latent(torch.randn(1, 4, 8, 8))
        (result,) = latent_upscale_by_bislerp(latent, upscale_method="bislerp", scale_by=2.0)
        out = unwrap_latent(result)
        assert out.shape == (1, 4, 16, 16)

    def test_half_scale(self):
        """LatentUpscaleByBislerp with 0.5x scale factor (downscale)."""
        latent = wrap_latent(torch.randn(1, 4, 16, 16))
        (result,) = latent_upscale_by_bislerp(latent, upscale_method="bislerp", scale_by=0.5)
        out = unwrap_latent(result)
        assert out.shape == (1, 4, 8, 8)

    def test_fractional_scale(self):
        """LatentUpscaleByBislerp with 1.5x scale."""
        latent = wrap_latent(torch.randn(1, 4, 8, 8))
        (result,) = latent_upscale_by_bislerp(latent, upscale_method="bislerp", scale_by=1.5)
        out = unwrap_latent(result)
        assert out.shape == (1, 4, 12, 12)


# ---------------------------------------------------------------------------
# LatentCompositeMasked
# ---------------------------------------------------------------------------

class TestLatentCompositeMasked:
    def test_no_mask_overwrites(self):
        """Without mask, source region fully overwrites destination."""
        dest = wrap_latent(torch.zeros(1, 4, 8, 8))
        src = wrap_latent(torch.ones(1, 4, 4, 4))
        (result,) = latent_composite_masked(dest, src, x=0, y=0, mask=None)
        out = unwrap_latent(result)
        # Source occupies top-left 4x4 in latent (coords / 8)
        # x=0, y=0 => lx=0, ly=0, src is 4x4 latent
        assert out[:, :, :4, :4].sum() == pytest.approx(1 * 4 * 4 * 4, abs=1e-5)
        assert out[:, :, 4:, :].sum() == pytest.approx(0.0, abs=1e-5)

    def test_with_mask(self):
        """With mask=0.5, result is blend of dest and src."""
        dest = wrap_latent(torch.zeros(1, 4, 8, 8))
        src = wrap_latent(torch.ones(1, 4, 8, 8))
        # Mask in pixel space: [1, 64, 64] all 0.5
        mask = torch.full((1, 64, 64), 0.5)
        (result,) = latent_composite_masked(dest, src, x=0, y=0, mask=mask)
        out = unwrap_latent(result)
        # Should be approximately 0.5 everywhere
        assert torch.allclose(out, torch.full_like(out, 0.5), atol=0.15)

    def test_mask_added_to_dict(self):
        """Verify output is a proper latent dict."""
        dest = wrap_latent(torch.zeros(1, 4, 8, 8))
        src = wrap_latent(torch.ones(1, 4, 4, 4))
        (result,) = latent_composite_masked(dest, src, x=0, y=0)
        assert "samples" in result

    def test_offset_position(self):
        """Source placed at offset (32, 32) in pixel space = (4, 4) latent."""
        dest = wrap_latent(torch.zeros(1, 4, 8, 8))
        src = wrap_latent(torch.ones(1, 4, 2, 2))
        (result,) = latent_composite_masked(dest, src, x=32, y=32)
        out = unwrap_latent(result)
        assert out[:, :, 4:6, 4:6].sum() == pytest.approx(1 * 4 * 2 * 2, abs=1e-5)
        # Rest should be zero
        out[:, :, 4:6, 4:6] = 0
        assert out.sum() == pytest.approx(0.0, abs=1e-5)

    def test_out_of_bounds_no_crash(self):
        """Source placed completely outside destination doesn't crash."""
        dest = wrap_latent(torch.zeros(1, 4, 8, 8))
        src = wrap_latent(torch.ones(1, 4, 4, 4))
        (result,) = latent_composite_masked(dest, src, x=1000, y=1000)
        out = unwrap_latent(result)
        assert out.sum() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# SetLatentNoiseMask (already in latent.py, verify it works)
# ---------------------------------------------------------------------------

class TestSetLatentNoiseMask:
    def test_mask_added(self):
        """Verify noise_mask key is added to latent dict."""
        from serenityflow.nodes.latent import set_latent_noise_mask
        latent = wrap_latent(torch.randn(1, 4, 8, 8))
        mask = torch.ones(1, 64, 64)
        (result,) = set_latent_noise_mask(latent, mask)
        assert "noise_mask" in result
        assert result["noise_mask"] is mask

    def test_samples_preserved(self):
        """Original samples tensor is preserved."""
        from serenityflow.nodes.latent import set_latent_noise_mask
        samples = torch.randn(1, 4, 8, 8)
        latent = wrap_latent(samples)
        mask = torch.ones(1, 64, 64)
        (result,) = set_latent_noise_mask(latent, mask)
        assert torch.equal(result["samples"], samples)


# ---------------------------------------------------------------------------
# Tile generation
# ---------------------------------------------------------------------------

class TestTileGeneration:
    def test_single_tile_small_image(self):
        """Image smaller than tile produces one tile."""
        tiles = _generate_tiles(256, 256, 512, 512, 64, True)
        assert len(tiles) >= 1
        # Should cover the image
        y, x, h, w = tiles[0]
        assert y == 0 and x == 0

    def test_covers_full_image(self):
        """All pixels are covered by at least one tile."""
        img_h, img_w = 1024, 1024
        tile_h, tile_w = 512, 512
        overlap = 64
        tiles = _generate_tiles(img_h, img_w, tile_h, tile_w, overlap, True)

        coverage = torch.zeros(img_h, img_w, dtype=torch.int32)
        for (y, x, h, w) in tiles:
            coverage[y:y + h, x:x + w] += 1
        assert (coverage > 0).all(), "Not all pixels covered by tiles"

    def test_force_uniform_vs_not(self):
        """Both uniform and non-uniform modes produce valid tiles."""
        for uniform in (True, False):
            tiles = _generate_tiles(1024, 1024, 512, 512, 64, uniform)
            assert len(tiles) > 0
            for (y, x, h, w) in tiles:
                assert y >= 0 and x >= 0
                assert y + h <= 1024 and x + w <= 1024


class TestFeatherMask:
    def test_shape(self):
        mask = _create_feather_mask(64, 64, 8, torch.device("cpu"), torch.float32)
        assert mask.shape == (1, 1, 64, 64)

    def test_center_is_one(self):
        mask = _create_feather_mask(64, 64, 8, torch.device("cpu"), torch.float32)
        assert mask[0, 0, 32, 32] == 1.0

    def test_edges_are_faded(self):
        mask = _create_feather_mask(64, 64, 8, torch.device("cpu"), torch.float32)
        # Corner should be much less than 1
        assert mask[0, 0, 0, 0] < 0.2

    def test_no_overlap_all_ones(self):
        mask = _create_feather_mask(64, 64, 0, torch.device("cpu"), torch.float32)
        assert torch.allclose(mask, torch.ones_like(mask))


class TestSeamTiles:
    def test_none_mode_empty(self):
        tiles = [(0, 0, 512, 512), (0, 448, 512, 512)]
        seam = _generate_seam_tiles(tiles, 1024, 1024, 64, "None")
        assert len(seam) == 0

    def test_band_mode_produces_seams(self):
        tiles = [(0, 0, 512, 512), (0, 448, 512, 512), (448, 0, 512, 512)]
        seam = _generate_seam_tiles(tiles, 1024, 1024, 64, "Band")
        assert len(seam) > 0


# ---------------------------------------------------------------------------
# UltimateSDUpscale (test tiling logic, mock sampling)
# ---------------------------------------------------------------------------

class TestUltimateSDUpscale:
    def test_without_bridge_returns_upscaled(self):
        """Without bridge, UltimateSDUpscale returns the model-upscaled image."""
        from serenityflow.nodes.upscale import ultimate_sd_upscale

        image = torch.rand(1, 64, 64, 3)
        # Mock upscale_model: just a dict (will trigger bicubic fallback)
        upscale_model = {"state_dict": {}, "path": "fake.safetensors"}
        # These won't be used without bridge
        model = None
        positive = None
        negative = None
        vae = None

        (result,) = ultimate_sd_upscale(
            image=image,
            model=model,
            positive=positive,
            negative=negative,
            vae=vae,
            upscale_model=upscale_model,
            tile_width=512,
            tile_height=512,
            seam_fix_mode="None",
        )
        # Should be 4x upscaled (bicubic fallback)
        assert result.shape[0] == 1
        assert result.shape[3] == 3
        assert result.shape[1] == 256  # 64 * 4
        assert result.shape[2] == 256


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_nodes_registered(self):
        """All upscale nodes are registered in the global registry."""
        from serenityflow.nodes.registry import registry
        for name in [
            "LatentUpscaleBislerp",
            "LatentUpscaleByBislerp",
            "LatentCompositeMasked",
            "UltimateSDUpscale",
        ]:
            assert registry.has(name), f"Node {name} not registered"

    def test_existing_nodes_still_registered(self):
        """Existing latent nodes (from latent.py) remain registered."""
        from serenityflow.nodes.registry import registry
        for name in [
            "LatentUpscale",
            "LatentUpscaleBy",
            "SetLatentNoiseMask",
            "LatentComposite",
        ]:
            assert registry.has(name), f"Node {name} not registered"
