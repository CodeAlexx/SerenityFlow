"""Tests for RES4LYF / ClownSampler advanced sampling nodes."""
from __future__ import annotations

import torch
import pytest


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------

class TestRegistration:
    """All res4lyf nodes are registered in the global registry."""

    _EXPECTED = [
        "ClownSampler", "ClownSamplerAdvanced", "ChainSampler",
        "ClownGuide", "ClownGuides", "ClownGuide_Style",
        "ClownGuide_FrequencySeparation", "ClownGuide_AdaIN_MMDiT",
        "ClownOptions_Combine", "ClownOptions_SDE", "ClownOptions_Momentum",
        "ClownOptions_Tile", "ClownOptions_DetailBoost",
        "ClownRegionalConditioning", "ClownRegionalConditioning2",
        "ClownRegionalConditioning3",
        "ClownOptions_FrameWeights",
        "CLIPTextEncodeFluxUnguided",
        "AdvancedNoise",
    ]

    @pytest.mark.parametrize("name", _EXPECTED)
    def test_registered(self, name):
        from serenityflow.nodes.registry import registry
        assert registry.has(name), f"{name} not registered"

    @pytest.mark.parametrize("name", _EXPECTED)
    def test_has_input_types(self, name):
        from serenityflow.nodes.registry import registry
        node = registry.get(name)
        assert "required" in node.input_types


# ---------------------------------------------------------------------------
# ClownSampler / ClownSamplerAdvanced
# ---------------------------------------------------------------------------

class TestClownSampler:
    def test_default_config(self):
        from serenityflow.nodes.res4lyf import clown_sampler
        model, pos, neg, lat = "m", [{}], [{}], {"samples": torch.zeros(1)}
        (result,) = clown_sampler(model, pos, neg, lat)
        assert result["type"] == "clown_sampler"
        assert result["model"] == "m"
        assert result["sampler_name"] == "euler"
        assert result["steps"] == 20
        assert result["cfg"] == 7.5
        assert result["denoise"] == 1.0
        assert result["denoise_alt"] == 1.0
        assert result["eta"] == 0.0
        assert result["s_noise"] == 1.0
        assert result["d_noise"] == 1.0
        assert result["shift"] == 1.0
        assert result["base_shift"] == 0.5
        assert result["shift_scaling"] == "exponential"
        assert result["sampler_mode"] == "standard"
        assert "guides" not in result

    def test_custom_params(self):
        from serenityflow.nodes.res4lyf import clown_sampler
        (result,) = clown_sampler(
            "m", [{}], [{}], {"samples": torch.zeros(1)},
            sampler_name="dpmpp_2m", steps=30, cfg=12.0, eta=0.5,
            sampler_mode="unsample",
        )
        assert result["sampler_name"] == "dpmpp_2m"
        assert result["steps"] == 30
        assert result["cfg"] == 12.0
        assert result["eta"] == 0.5
        assert result["sampler_mode"] == "unsample"

    def test_with_guides(self):
        from serenityflow.nodes.res4lyf import clown_sampler
        guides = [{"type": "velocity", "weight": 0.8}]
        (result,) = clown_sampler("m", [{}], [{}], {"samples": torch.zeros(1)},
                                  guides=guides)
        assert result["guides"] is guides

    def test_with_options(self):
        from serenityflow.nodes.res4lyf import clown_sampler
        options = {"momentum": 0.5, "momentum_sign": "positive"}
        (result,) = clown_sampler("m", [{}], [{}], {"samples": torch.zeros(1)},
                                  options=options)
        assert result["momentum"] == 0.5
        assert result["momentum_sign"] == "positive"

    def test_return_type(self):
        from serenityflow.nodes.registry import registry
        node = registry.get("ClownSampler")
        assert node.return_types == ("LATENT",)


class TestClownSamplerAdvanced:
    def test_has_step_control(self):
        from serenityflow.nodes.res4lyf import clown_sampler_advanced
        (result,) = clown_sampler_advanced(
            "m", [{}], [{}], {"samples": torch.zeros(1)},
            start_step=5, end_step=15, add_noise=False,
            return_leftover_noise=True,
        )
        assert result["type"] == "clown_sampler_advanced"
        assert result["start_step"] == 5
        assert result["end_step"] == 15
        assert result["add_noise"] is False
        assert result["return_leftover_noise"] is True

    def test_defaults(self):
        from serenityflow.nodes.res4lyf import clown_sampler_advanced
        (result,) = clown_sampler_advanced("m", [{}], [{}], {"samples": torch.zeros(1)})
        assert result["start_step"] == 0
        assert result["end_step"] == 20
        assert result["add_noise"] is True
        assert result["return_leftover_noise"] is False


# ---------------------------------------------------------------------------
# ChainSampler
# ---------------------------------------------------------------------------

class TestChainSampler:
    def test_two_stage(self):
        from serenityflow.nodes.res4lyf import chain_sampler
        (result,) = chain_sampler(
            "m", [{}], [{}], {"samples": torch.zeros(1)},
            stage_1_sampler="euler", stage_1_steps=10,
            stage_1_denoise=1.0, stage_1_cfg=7.5,
            stage_2_sampler="dpmpp_2m", stage_2_steps=10,
            stage_2_denoise=0.5, stage_2_cfg=5.0,
            seed=42,
        )
        assert result["type"] == "chain_sampler"
        assert len(result["stages"]) == 2
        assert result["stages"][0]["sampler"] == "euler"
        assert result["stages"][1]["sampler"] == "dpmpp_2m"
        assert result["seed"] == 42

    def test_three_stage(self):
        from serenityflow.nodes.res4lyf import chain_sampler
        (result,) = chain_sampler(
            "m", [{}], [{}], {"samples": torch.zeros(1)},
            stage_3_sampler="heun", stage_3_steps=5,
            stage_3_denoise=0.3, stage_3_cfg=3.0,
        )
        assert len(result["stages"]) == 3
        assert result["stages"][2]["sampler"] == "heun"
        assert result["stages"][2]["steps"] == 5

    def test_two_stage_no_optional(self):
        from serenityflow.nodes.res4lyf import chain_sampler
        (result,) = chain_sampler("m", [{}], [{}], {"samples": torch.zeros(1)})
        assert len(result["stages"]) == 2

    def test_stage_configs(self):
        from serenityflow.nodes.res4lyf import chain_sampler
        (result,) = chain_sampler(
            "m", [{}], [{}], {"samples": torch.zeros(1)},
            stage_1_cfg=12.0, stage_2_denoise=0.8,
        )
        assert result["stages"][0]["cfg"] == 12.0
        assert result["stages"][1]["denoise"] == 0.8


# ---------------------------------------------------------------------------
# Guide system
# ---------------------------------------------------------------------------

class TestClownGuide:
    def test_basic_guide(self):
        from serenityflow.nodes.res4lyf import clown_guide
        (result,) = clown_guide("epsilon_projection", weight=0.8)
        assert result["type"] == "epsilon_projection"
        assert result["weight"] == 0.8

    def test_default_weight(self):
        from serenityflow.nodes.res4lyf import clown_guide
        (result,) = clown_guide("velocity")
        assert result["weight"] == 1.0


class TestClownGuides:
    def test_single(self):
        from serenityflow.nodes.res4lyf import clown_guides
        g1 = {"type": "velocity", "weight": 1.0}
        (result,) = clown_guides(g1)
        assert result == [g1]

    def test_multiple(self):
        from serenityflow.nodes.res4lyf import clown_guides
        g1 = {"type": "velocity", "weight": 1.0}
        g2 = {"type": "epsilon_projection", "weight": 0.5}
        (result,) = clown_guides(g1, guide_2=g2)
        assert len(result) == 2
        assert result[0] is g1
        assert result[1] is g2

    def test_five_guides(self):
        from serenityflow.nodes.res4lyf import clown_guides
        guides = [{"type": f"g{i}"} for i in range(5)]
        (result,) = clown_guides(guides[0], guide_2=guides[1], guide_3=guides[2],
                                 guide_4=guides[3], guide_5=guides[4])
        assert len(result) == 5

    def test_skip_none(self):
        from serenityflow.nodes.res4lyf import clown_guides
        g1 = {"type": "a"}
        g3 = {"type": "c"}
        (result,) = clown_guides(g1, guide_3=g3)
        assert len(result) == 2


class TestClownGuideStyle:
    def test_config(self):
        from serenityflow.nodes.res4lyf import clown_guide_style
        img = torch.zeros(1, 3, 64, 64)
        (result,) = clown_guide_style(img, weight=0.7, start_at=0.1, end_at=0.9)
        assert result["type"] == "style"
        assert result["weight"] == 0.7
        assert result["start_at"] == 0.1
        assert result["end_at"] == 0.9
        assert result["style_image"] is img


class TestClownGuideFrequencySeparation:
    def test_config(self):
        from serenityflow.nodes.res4lyf import clown_guide_frequency_separation
        (result,) = clown_guide_frequency_separation(low_weight=0.8, high_weight=1.2, cutoff=0.3)
        assert result["type"] == "frequency_separation"
        assert result["low_weight"] == 0.8
        assert result["high_weight"] == 1.2
        assert result["cutoff"] == 0.3

    def test_defaults(self):
        from serenityflow.nodes.res4lyf import clown_guide_frequency_separation
        (result,) = clown_guide_frequency_separation()
        assert result["cutoff"] == 0.5


class TestClownGuideAdaINMMDiT:
    def test_config(self):
        from serenityflow.nodes.res4lyf import clown_guide_adain_mmdit
        (result,) = clown_guide_adain_mmdit(weight=0.6, start_at=0.2, end_at=0.8)
        assert result["type"] == "adain_mmdit"
        assert result["weight"] == 0.6
        assert result["start_at"] == 0.2
        assert result["end_at"] == 0.8


# ---------------------------------------------------------------------------
# Options system
# ---------------------------------------------------------------------------

class TestClownOptionsCombine:
    def test_single(self):
        from serenityflow.nodes.res4lyf import clown_options_combine
        opt = {"momentum": 0.5}
        (result,) = clown_options_combine(opt)
        assert result == {"momentum": 0.5}

    def test_merge_two(self):
        from serenityflow.nodes.res4lyf import clown_options_combine
        o1 = {"momentum": 0.5}
        o2 = {"tile_width": 512}
        (result,) = clown_options_combine(o1, options_2=o2)
        assert result["momentum"] == 0.5
        assert result["tile_width"] == 512

    def test_later_overrides(self):
        from serenityflow.nodes.res4lyf import clown_options_combine
        o1 = {"momentum": 0.5}
        o2 = {"momentum": 0.8}
        (result,) = clown_options_combine(o1, options_2=o2)
        assert result["momentum"] == 0.8


class TestClownOptionsSDE:
    def test_config(self):
        from serenityflow.nodes.res4lyf import clown_options_sde
        (result,) = clown_options_sde(noise_type="brownian", eta=0.5, s_noise=1.2)
        assert result["sde_noise_type"] == "brownian"
        assert result["sde_eta"] == 0.5
        assert result["sde_s_noise"] == 1.2


class TestClownOptionsMomentum:
    def test_config(self):
        from serenityflow.nodes.res4lyf import clown_options_momentum
        (result,) = clown_options_momentum(momentum=0.9, momentum_sign="alternate")
        assert result["momentum"] == 0.9
        assert result["momentum_sign"] == "alternate"

    def test_defaults(self):
        from serenityflow.nodes.res4lyf import clown_options_momentum
        (result,) = clown_options_momentum()
        assert result["momentum"] == 0.0
        assert result["momentum_sign"] == "positive"


class TestClownOptionsTile:
    def test_config(self):
        from serenityflow.nodes.res4lyf import clown_options_tile
        (result,) = clown_options_tile(tile_width=1024, tile_height=768, overlap=128)
        assert result["tile_width"] == 1024
        assert result["tile_height"] == 768
        assert result["tile_overlap"] == 128

    def test_defaults(self):
        from serenityflow.nodes.res4lyf import clown_options_tile
        (result,) = clown_options_tile()
        assert result["tile_width"] == 512
        assert result["tile_height"] == 512
        assert result["tile_overlap"] == 64


class TestClownOptionsDetailBoost:
    def test_config(self):
        from serenityflow.nodes.res4lyf import clown_options_detail_boost
        (result,) = clown_options_detail_boost(boost_strength=0.8, start_at=0.1, end_at=0.9)
        assert result["detail_boost_strength"] == 0.8
        assert result["detail_boost_start_at"] == 0.1
        assert result["detail_boost_end_at"] == 0.9

    def test_defaults(self):
        from serenityflow.nodes.res4lyf import clown_options_detail_boost
        (result,) = clown_options_detail_boost()
        assert result["detail_boost_strength"] == 0.5
        assert result["detail_boost_start_at"] == 0.3
        assert result["detail_boost_end_at"] == 0.7


# ---------------------------------------------------------------------------
# Regional conditioning
# ---------------------------------------------------------------------------

class TestClownRegionalConditioning:
    def test_applies_mask(self):
        from serenityflow.nodes.res4lyf import clown_regional_conditioning
        cond = [{"cross_attn": torch.zeros(1, 77, 768)}]
        mask = torch.ones(1, 64, 64)
        (result,) = clown_regional_conditioning(cond, mask, strength=0.8)
        assert len(result) == 1
        assert result[0]["mask"] is mask
        assert result[0]["strength"] == 0.8
        assert result[0]["set_area_to_bounds"] is False
        # Original data preserved
        assert "cross_attn" in result[0]

    def test_multiple_cond_entries(self):
        from serenityflow.nodes.res4lyf import clown_regional_conditioning
        cond = [{"a": 1}, {"b": 2}]
        mask = torch.ones(1, 64, 64)
        (result,) = clown_regional_conditioning(cond, mask)
        assert len(result) == 2
        assert all(r["mask"] is mask for r in result)


class TestClownRegionalConditioning2:
    def test_two_zones(self):
        from serenityflow.nodes.res4lyf import clown_regional_conditioning_2
        c1 = [{"text": "sky"}]
        c2 = [{"text": "ground"}]
        m1 = torch.ones(1, 64, 64)
        m2 = torch.zeros(1, 64, 64)
        (result,) = clown_regional_conditioning_2(c1, m1, c2, m2)
        assert len(result) == 2
        assert result[0]["mask"] is m1
        assert result[1]["mask"] is m2
        assert result[0]["text"] == "sky"
        assert result[1]["text"] == "ground"


class TestClownRegionalConditioning3:
    def test_three_zones(self):
        from serenityflow.nodes.res4lyf import clown_regional_conditioning_3
        conds = [{"t": f"zone{i}"} for i in range(3)]
        masks = [torch.rand(1, 64, 64) for _ in range(3)]
        (result,) = clown_regional_conditioning_3(
            [conds[0]], masks[0],
            [conds[1]], masks[1],
            [conds[2]], masks[2],
        )
        assert len(result) == 3
        for i in range(3):
            assert result[i]["mask"] is masks[i]
            assert result[i]["t"] == f"zone{i}"


# ---------------------------------------------------------------------------
# FrameWeights
# ---------------------------------------------------------------------------

class TestClownOptionsFrameWeights:
    def test_parse_basic(self):
        from serenityflow.nodes.res4lyf import _parse_frame_weights
        kf = _parse_frame_weights("0:1.0, 10:0.5, 20:0.8")
        assert kf == [(0, 1.0), (10, 0.5), (20, 0.8)]

    def test_parse_unordered(self):
        from serenityflow.nodes.res4lyf import _parse_frame_weights
        kf = _parse_frame_weights("20:0.8, 0:1.0, 10:0.5")
        assert kf == [(0, 1.0), (10, 0.5), (20, 0.8)]

    def test_interpolation_endpoints(self):
        from serenityflow.nodes.res4lyf import _interpolate_frame_weights
        weights = _interpolate_frame_weights([(0, 1.0), (10, 0.0)], total_frames=11)
        assert len(weights) == 11
        assert weights[0] == pytest.approx(1.0)
        assert weights[10] == pytest.approx(0.0)
        assert weights[5] == pytest.approx(0.5)

    def test_interpolation_before_first(self):
        from serenityflow.nodes.res4lyf import _interpolate_frame_weights
        weights = _interpolate_frame_weights([(5, 0.5), (10, 1.0)], total_frames=15)
        # Frames before first keyframe get first keyframe's value
        assert weights[0] == pytest.approx(0.5)
        assert weights[4] == pytest.approx(0.5)

    def test_interpolation_after_last(self):
        from serenityflow.nodes.res4lyf import _interpolate_frame_weights
        weights = _interpolate_frame_weights([(0, 1.0), (5, 0.5)], total_frames=10)
        # Frames after last keyframe get last keyframe's value
        assert weights[9] == pytest.approx(0.5)

    def test_empty_keyframes(self):
        from serenityflow.nodes.res4lyf import _interpolate_frame_weights
        weights = _interpolate_frame_weights([], total_frames=5)
        assert weights == [1.0] * 5

    def test_node_output(self):
        from serenityflow.nodes.res4lyf import clown_options_frame_weights
        (result,) = clown_options_frame_weights("0:1.0, 10:0.5", total_frames=11)
        assert "frame_weights" in result
        assert result["total_frames"] == 11
        assert len(result["frame_weights"]) == 11


# ---------------------------------------------------------------------------
# CLIPTextEncodeFluxUnguided
# ---------------------------------------------------------------------------

class TestCLIPTextEncodeFluxUnguided:
    def test_strips_guidance(self):
        from unittest.mock import patch
        cond_with_guidance = [{"cross_attn": torch.zeros(1, 77, 768), "guidance": 3.5}]
        with patch("serenityflow.bridge.serenity_api.encode_text", return_value=cond_with_guidance):
            from serenityflow.nodes.res4lyf import clip_text_encode_flux_unguided
            (result,) = clip_text_encode_flux_unguided("clip", "hello")
            assert len(result) == 1
            assert "guidance" not in result[0]
            assert "cross_attn" in result[0]

    def test_no_guidance_present(self):
        from unittest.mock import patch
        cond = [{"cross_attn": torch.zeros(1, 77, 768)}]
        with patch("serenityflow.bridge.serenity_api.encode_text", return_value=cond):
            from serenityflow.nodes.res4lyf import clip_text_encode_flux_unguided
            (result,) = clip_text_encode_flux_unguided("clip", "test")
            assert len(result) == 1
            assert "guidance" not in result[0]


# ---------------------------------------------------------------------------
# AdvancedNoise
# ---------------------------------------------------------------------------

class TestAdvancedNoise:
    def test_gaussian(self):
        from serenityflow.nodes.res4lyf import advanced_noise
        (result,) = advanced_noise(seed=42, noise_type="gaussian", scale=1.0)
        assert result["type"] == "gaussian"
        assert result["seed"] == 42
        assert result["scale"] == 1.0

    def test_simplex(self):
        from serenityflow.nodes.res4lyf import advanced_noise
        (result,) = advanced_noise(seed=0, noise_type="simplex", scale=2.0)
        assert result["type"] == "simplex"
        assert result["scale"] == 2.0

    def test_fractal(self):
        from serenityflow.nodes.res4lyf import advanced_noise
        (result,) = advanced_noise(seed=1, noise_type="fractal")
        assert result["type"] == "fractal"

    def test_uniform(self):
        from serenityflow.nodes.res4lyf import advanced_noise
        (result,) = advanced_noise(seed=7, noise_type="uniform", scale=0.5)
        assert result["type"] == "uniform"
        assert result["scale"] == 0.5

    def test_return_type(self):
        from serenityflow.nodes.registry import registry
        node = registry.get("AdvancedNoise")
        assert node.return_types == ("NOISE",)
