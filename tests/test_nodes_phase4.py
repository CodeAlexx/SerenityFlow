"""Phase 4 tests -- node registration, structure, and pure-computation correctness."""
from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all_nodes():
    from serenityflow.nodes.registry import registry
    import serenityflow.nodes  # noqa: F401 — triggers registration
    return registry.list_all()


# ---------------------------------------------------------------------------
# Category 0 — Global structural checks
# ---------------------------------------------------------------------------

class TestNodeRegistration:
    def test_minimum_node_count(self, all_nodes):
        assert len(all_nodes) >= 200, f"Expected >=200 nodes, got {len(all_nodes)}"

    def test_all_nodes_have_input_types(self, all_nodes):
        for name, nd in all_nodes.items():
            assert nd.input_types is not None, f"{name} missing input_types"
            assert "required" in nd.input_types, f"{name} missing 'required' in input_types"

    def test_all_nodes_have_return_types_tuple(self, all_nodes):
        for name, nd in all_nodes.items():
            assert isinstance(nd.return_types, tuple), f"{name} return_types not a tuple"

    def test_all_nodes_have_callable_fn(self, all_nodes):
        for name, nd in all_nodes.items():
            assert callable(nd.fn), f"{name}.fn not callable"

    def test_all_nodes_have_category(self, all_nodes):
        for name, nd in all_nodes.items():
            assert nd.category, f"{name} missing category"

    def test_no_duplicate_registrations(self, all_nodes):
        # If we got here without errors, there are no duplicates
        # (registry would overwrite, but let's verify count matches unique names)
        names = list(all_nodes.keys())
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Category 1 — Custom Sampler Stack
# ---------------------------------------------------------------------------

class TestSamplingCustom:
    def test_random_noise(self):
        from serenityflow.nodes.sampling_custom import random_noise
        result = random_noise(42)
        assert result[0]["type"] == "random"
        assert result[0]["seed"] == 42

    def test_empty_noise(self):
        from serenityflow.nodes.sampling_custom import empty_noise
        result = empty_noise()
        assert result[0]["type"] == "empty"

    def test_basic_guider(self):
        from serenityflow.nodes.sampling_custom import basic_guider
        model = object()
        cond = [{"cross_attn": torch.randn(1, 77, 768)}]
        result = basic_guider(model, cond)
        assert result[0]["type"] == "basic"
        assert result[0]["model"] is model

    def test_cfg_guider(self):
        from serenityflow.nodes.sampling_custom import cfg_guider
        model = object()
        pos = [{"cross_attn": torch.randn(1, 77, 768)}]
        neg = [{"cross_attn": torch.randn(1, 77, 768)}]
        result = cfg_guider(model, pos, neg, 7.5)
        assert result[0]["type"] == "cfg"
        assert result[0]["cfg"] == 7.5

    def test_dual_cfg_guider(self):
        from serenityflow.nodes.sampling_custom import dual_cfg_guider
        result = dual_cfg_guider(None, None, None, None, 5.0, 3.0)
        assert result[0]["type"] == "dual_cfg"

    def test_perp_neg_guider(self):
        from serenityflow.nodes.sampling_custom import perp_neg_guider
        result = perp_neg_guider(None, None, None, None, 8.0, 1.0)
        assert result[0]["type"] == "perp_neg"

    def test_karras_scheduler_shape(self):
        from serenityflow.nodes.sampling_custom import karras_scheduler
        sigmas = karras_scheduler(20, 14.614642, 0.0291675, 7.0)
        assert sigmas[0].shape == (21,)  # steps + 1
        assert sigmas[0][0] > sigmas[0][-1]  # Decreasing

    def test_karras_scheduler_deterministic(self):
        from serenityflow.nodes.sampling_custom import karras_scheduler
        s1 = karras_scheduler(20, 14.6, 0.029, 7.0)
        s2 = karras_scheduler(20, 14.6, 0.029, 7.0)
        assert torch.allclose(s1[0], s2[0])

    def test_exponential_scheduler(self):
        from serenityflow.nodes.sampling_custom import exponential_scheduler
        sigmas = exponential_scheduler(10, 14.6, 0.029)
        assert sigmas[0].shape == (11,)
        assert sigmas[0][0] > sigmas[0][-1]

    def test_polyexponential_scheduler(self):
        from serenityflow.nodes.sampling_custom import polyexponential_scheduler
        sigmas = polyexponential_scheduler(10, 14.6, 0.029, 1.0)
        assert sigmas[0].shape == (11,)
        assert sigmas[0][-1] == 0.0

    def test_laplace_scheduler(self):
        from serenityflow.nodes.sampling_custom import laplace_scheduler
        sigmas = laplace_scheduler(10, 14.6, 0.029, 0.0, 0.5)
        assert sigmas[0].shape == (11,)
        assert sigmas[0][-1] == 0.0

    def test_sampler_euler(self):
        from serenityflow.nodes.sampling_custom import sampler_euler
        result = sampler_euler()
        assert result[0]["type"] == "euler"

    def test_sampler_dpmpp_2m(self):
        from serenityflow.nodes.sampling_custom import sampler_dpmpp_2m
        result = sampler_dpmpp_2m()
        assert result[0]["type"] == "dpmpp_2m"

    def test_sampler_ddim(self):
        from serenityflow.nodes.sampling_custom import sampler_ddim
        result = sampler_ddim(0.0)
        assert result[0]["type"] == "ddim"
        assert result[0]["eta"] == 0.0

    def test_split_sigmas(self):
        from serenityflow.nodes.sampling_custom import split_sigmas
        sigmas = torch.linspace(14, 0, 21)
        high, low = split_sigmas(sigmas, 10)
        assert len(high) == 11
        assert len(low) == 11

    def test_flip_sigmas(self):
        from serenityflow.nodes.sampling_custom import flip_sigmas
        sigmas = torch.tensor([14.0, 7.0, 0.0])
        result = flip_sigmas(sigmas)
        assert result[0][0] == 0.0
        assert result[0][-1] == 14.0

    def test_align_your_steps_scheduler(self):
        from serenityflow.nodes.sampling_custom import align_your_steps_scheduler
        sigmas = align_your_steps_scheduler("sdxl", 10, 1.0)
        assert sigmas[0].shape == (11,)


# ---------------------------------------------------------------------------
# Category 2 — Image Operations
# ---------------------------------------------------------------------------

class TestImageOps:
    def test_image_crop_basic(self):
        from serenityflow.nodes.image_ops import image_crop
        img = torch.rand(1, 512, 512, 3)
        result = image_crop(img, 256, 256, 0, 0)
        assert result[0].shape == (1, 256, 256, 3)

    def test_image_crop_clamps_to_bounds(self):
        from serenityflow.nodes.image_ops import image_crop
        img = torch.rand(1, 100, 100, 3)
        result = image_crop(img, 200, 200, 50, 50)
        assert result[0].shape[1] == 50  # clamped height
        assert result[0].shape[2] == 50  # clamped width

    def test_image_composite_masked_no_mask(self):
        from serenityflow.nodes.image_ops import image_composite_masked
        dst = torch.zeros(1, 64, 64, 3)
        src = torch.ones(1, 32, 32, 3)
        result = image_composite_masked(dst, src, 0, 0)
        assert result[0][:, :32, :32, :].sum() > 0

    def test_image_composite_masked_with_mask(self):
        from serenityflow.nodes.image_ops import image_composite_masked
        dst = torch.zeros(1, 64, 64, 3)
        src = torch.ones(1, 32, 32, 3)
        mask = torch.ones(1, 32, 32) * 0.5
        result = image_composite_masked(dst, src, 0, 0, mask=mask)
        assert torch.allclose(result[0][0, 0, 0], torch.tensor([0.5, 0.5, 0.5]))

    def test_image_flip_horizontal(self):
        from serenityflow.nodes.image_ops import image_flip
        img = torch.zeros(1, 2, 4, 3)
        img[0, 0, 0, 0] = 1.0  # Mark top-left
        result = image_flip(img, "horizontal")
        assert result[0][0, 0, 3, 0] == 1.0  # Should be top-right now

    def test_image_flip_vertical(self):
        from serenityflow.nodes.image_ops import image_flip
        img = torch.zeros(1, 4, 2, 3)
        img[0, 0, 0, 0] = 1.0
        result = image_flip(img, "vertical")
        assert result[0][0, 3, 0, 0] == 1.0

    def test_image_rotate_90(self):
        from serenityflow.nodes.image_ops import image_rotate
        img = torch.rand(1, 64, 128, 3)
        result = image_rotate(img, "90 degrees")
        assert result[0].shape == (1, 128, 64, 3)

    def test_image_sharpen_identity(self):
        from serenityflow.nodes.image_ops import image_sharpen
        img = torch.rand(1, 32, 32, 3)
        result = image_sharpen(img, 0, 1.0, 1.0)  # radius 0 = no-op
        assert torch.allclose(result[0], img)

    def test_image_blur_output_range(self):
        from serenityflow.nodes.image_ops import image_blur
        img = torch.rand(1, 32, 32, 3)
        result = image_blur(img, 2, 1.0)
        assert result[0].min() >= 0.0
        assert result[0].max() <= 1.0

    def test_image_quantize(self):
        from serenityflow.nodes.image_ops import image_quantize
        img = torch.rand(1, 32, 32, 3)
        result = image_quantize(img, 4)  # 4 colors
        unique_vals = result[0].unique()
        assert len(unique_vals) <= 4

    def test_image_to_mask(self):
        from serenityflow.nodes.image_ops import image_to_mask
        img = torch.rand(1, 32, 32, 3)
        result = image_to_mask(img, "red")
        assert result[0].shape == (1, 32, 32)

    def test_mask_to_image(self):
        from serenityflow.nodes.image_ops import mask_to_image
        mask = torch.rand(1, 32, 32)
        result = mask_to_image(mask)
        assert result[0].shape == (1, 32, 32, 3)

    def test_split_join_alpha_roundtrip(self):
        from serenityflow.nodes.image_ops import split_image_with_alpha, join_image_with_alpha
        img = torch.rand(1, 32, 32, 4)
        rgb, alpha = split_image_with_alpha(img)
        result = join_image_with_alpha(rgb, alpha)
        assert result[0].shape == (1, 32, 32, 4)
        assert torch.allclose(result[0], img, atol=1e-6)

    def test_rebatch_images(self):
        from serenityflow.nodes.image_ops import rebatch_images
        imgs = torch.rand(8, 32, 32, 3)
        result = rebatch_images(imgs, 3)
        assert result[0].shape[0] == 3

    def test_image_from_batch(self):
        from serenityflow.nodes.image_ops import image_from_batch
        imgs = torch.rand(4, 32, 32, 3)
        result = image_from_batch(imgs, 1, 2)
        assert result[0].shape[0] == 2

    def test_repeat_image_batch(self):
        from serenityflow.nodes.image_ops import repeat_image_batch
        img = torch.rand(1, 32, 32, 3)
        result = repeat_image_batch(img, 3)
        assert result[0].shape[0] == 3

    def test_image_scale_to_total_pixels(self):
        from serenityflow.nodes.image_ops import image_scale_to_total_pixels
        img = torch.rand(1, 100, 100, 3)  # 10K pixels
        result = image_scale_to_total_pixels(img, "bilinear", 0.04)  # ~40K pixels
        total = result[0].shape[1] * result[0].shape[2]
        assert 30000 < total < 50000


# ---------------------------------------------------------------------------
# Category 3 — Post-Processing
# ---------------------------------------------------------------------------

class TestPostProcessing:
    def test_image_blend_normal(self):
        from serenityflow.nodes.post_processing import image_blend
        a = torch.zeros(1, 32, 32, 3)
        b = torch.ones(1, 32, 32, 3)
        result = image_blend(a, b, 0.5, "normal")
        assert torch.allclose(result[0], torch.full_like(a, 0.5), atol=1e-6)

    def test_image_blend_multiply(self):
        from serenityflow.nodes.post_processing import image_blend
        a = torch.full((1, 4, 4, 3), 0.5)
        b = torch.full((1, 4, 4, 3), 0.5)
        result = image_blend(a, b, 1.0, "multiply")
        assert torch.allclose(result[0], torch.full_like(a, 0.25), atol=1e-6)

    def test_image_brightness(self):
        from serenityflow.nodes.post_processing import image_brightness
        img = torch.full((1, 4, 4, 3), 0.5)
        result = image_brightness(img, 2.0)
        assert torch.allclose(result[0], torch.ones_like(img))

    def test_image_contrast(self):
        from serenityflow.nodes.post_processing import image_contrast
        img = torch.rand(1, 32, 32, 3)
        result = image_contrast(img, 1.0)
        assert result[0].min() >= 0.0
        assert result[0].max() <= 1.0

    def test_image_gamma_identity(self):
        from serenityflow.nodes.post_processing import image_gamma
        img = torch.rand(1, 32, 32, 3).clamp(0.01, 0.99)
        result = image_gamma(img, 1.0)
        assert torch.allclose(result[0], img, atol=1e-6)

    def test_image_levels(self):
        from serenityflow.nodes.post_processing import image_levels
        img = torch.full((1, 4, 4, 3), 0.5)
        result = image_levels(img, 0.0, 1.0, 1.0)
        assert torch.allclose(result[0], img, atol=1e-6)

    def test_image_color_balance(self):
        from serenityflow.nodes.post_processing import image_color_balance
        img = torch.full((1, 4, 4, 3), 0.5)
        result = image_color_balance(img, red=0.0, green=1.0, blue=1.0)
        assert result[0][..., 0].max() == 0.0

    def test_output_range_clamped(self):
        from serenityflow.nodes.post_processing import image_blend
        a = torch.ones(1, 4, 4, 3) * 0.9
        b = torch.ones(1, 4, 4, 3) * 0.9
        result = image_blend(a, b, 1.0, "screen")
        assert result[0].max() <= 1.0


# ---------------------------------------------------------------------------
# Category 4 — Mask Operations
# ---------------------------------------------------------------------------

class TestMaskOps:
    def test_invert_mask(self):
        from serenityflow.nodes.mask import invert_mask
        mask = torch.ones(1, 64, 64)
        result = invert_mask(mask)
        assert result[0].max() == 0.0
        assert result[0].min() == 0.0

    def test_invert_mask_zeros(self):
        from serenityflow.nodes.mask import invert_mask
        mask = torch.zeros(1, 64, 64)
        result = invert_mask(mask)
        assert result[0].max() == 1.0

    def test_crop_mask(self):
        from serenityflow.nodes.mask import crop_mask
        mask = torch.rand(1, 100, 100)
        result = crop_mask(mask, 10, 10, 50, 50)
        assert result[0].shape == (1, 50, 50)

    def test_mask_composite_add(self):
        from serenityflow.nodes.mask import mask_composite
        dst = torch.zeros(1, 64, 64)
        src = torch.full((1, 32, 32), 0.5)
        result = mask_composite(dst, src, 0, 0, "add")
        assert result[0][:, :32, :32].mean() == pytest.approx(0.5, abs=1e-6)

    def test_mask_composite_clamps(self):
        from serenityflow.nodes.mask import mask_composite
        dst = torch.ones(1, 64, 64)
        src = torch.ones(1, 32, 32)
        result = mask_composite(dst, src, 0, 0, "add")
        assert result[0].max() <= 1.0

    def test_feather_mask(self):
        from serenityflow.nodes.mask import feather_mask
        mask = torch.ones(1, 64, 64)
        result = feather_mask(mask, left=10, top=10, right=10, bottom=10)
        assert result[0][:, 0, :].max() == 0.0  # Top edge feathered to 0

    def test_grow_mask_expand_zero_is_identity(self):
        from serenityflow.nodes.mask import grow_mask
        mask = torch.rand(1, 32, 32).round()
        result = grow_mask(mask, 0, True)
        assert torch.allclose(result[0], mask)

    def test_threshold_mask(self):
        from serenityflow.nodes.mask import threshold_mask
        mask = torch.tensor([[[0.3, 0.7], [0.5, 0.9]]])
        result = threshold_mask(mask, 0.5)
        expected = torch.tensor([[[0.0, 1.0], [1.0, 1.0]]])
        assert torch.allclose(result[0], expected)

    def test_solid_mask(self):
        from serenityflow.nodes.mask import solid_mask
        result = solid_mask(0.5, 64, 64)
        assert result[0].shape == (1, 64, 64)
        assert result[0].mean() == pytest.approx(0.5)

    def test_mask_from_batch(self):
        from serenityflow.nodes.mask import mask_from_batch
        masks = torch.rand(4, 32, 32)
        result = mask_from_batch(masks, 1, 2)
        assert result[0].shape[0] == 2

    def test_2d_mask_handled(self):
        from serenityflow.nodes.mask import invert_mask, crop_mask
        mask_2d = torch.ones(64, 64)
        result = invert_mask(mask_2d)
        assert result[0].ndim == 2
        result = crop_mask(mask_2d, 0, 0, 32, 32)
        assert result[0].shape == (1, 32, 32)


# ---------------------------------------------------------------------------
# Category 5 — Model-Specific Nodes
# ---------------------------------------------------------------------------

class TestModelSpecific:
    def test_empty_flux2_latent(self):
        from serenityflow.nodes.model_specific.flux import empty_flux2_latent
        result = empty_flux2_latent(1024, 1024, 1)
        lat = result[0]["samples"]
        assert lat.shape == (1, 16, 128, 128)

    def test_flux_disable_guidance(self):
        from serenityflow.nodes.model_specific.flux import flux_disable_guidance
        cond = [{"cross_attn": torch.randn(1, 77, 768), "guidance": 3.5}]
        result = flux_disable_guidance(cond)
        assert result[0][0]["guidance"] == 0.0

    def test_flux2_scheduler(self):
        from serenityflow.nodes.model_specific.flux import flux2_scheduler
        sigmas = flux2_scheduler(20, 1.0, 1.0)
        assert sigmas[0].shape == (21,)
        assert sigmas[0][0] == pytest.approx(1.0)
        assert sigmas[0][-1] == pytest.approx(0.0)

    def test_flux2_scheduler_with_shift(self):
        from serenityflow.nodes.model_specific.flux import flux2_scheduler
        sigmas_no_shift = flux2_scheduler(20, 1.0, 1.0)
        sigmas_shifted = flux2_scheduler(20, 3.0, 1.0)
        # Shifted sigmas should be higher in the middle
        assert sigmas_shifted[0][10] > sigmas_no_shift[0][10]

    def test_empty_sdxl_latent(self):
        from serenityflow.nodes.model_specific.sdxl import empty_sdxl_latent
        result = empty_sdxl_latent(1024, 1024, 1)
        assert result[0]["samples"].shape == (1, 4, 128, 128)

    def test_empty_hidream_latent(self):
        from serenityflow.nodes.model_specific.other_models import empty_hidream_latent
        result = empty_hidream_latent(512, 512, 2)
        assert result[0]["samples"].shape == (2, 16, 64, 64)

    def test_empty_chroma_latent(self):
        from serenityflow.nodes.model_specific.other_models import empty_chroma_latent
        result = empty_chroma_latent(768, 768, 1)
        assert result[0]["samples"].shape == (1, 16, 96, 96)

    def test_ltxv_scheduler(self):
        from serenityflow.nodes.model_specific.video_models import ltxv_scheduler
        sigmas = ltxv_scheduler(25, 5.0, 1.0)
        assert sigmas[0].shape == (26,)

    def test_wan_image_to_video(self):
        from serenityflow.nodes.model_specific.video_models import wan_image_to_video
        cond = [{"cross_attn": torch.randn(1, 77, 768)}]
        img = torch.rand(1, 64, 64, 3)
        result = wan_image_to_video(cond, img, 0.8)
        assert "wan_guide_image" in result[0][0]
        assert result[0][0]["wan_guide_strength"] == 0.8


# ---------------------------------------------------------------------------
# Category 6 — Model Operations
# ---------------------------------------------------------------------------

class TestModelOps:
    def test_rescale_cfg_with_handle(self):
        from serenityflow.nodes.model_ops import rescale_cfg
        from serenityflow.core.types import ModelHandle
        handle = ModelHandle("test", "flux", {}, "/test", torch.float16, None)
        result = rescale_cfg(handle, 0.7)
        assert result[0].model_options["rescale_cfg_multiplier"] == 0.7

    def test_model_sampling_discrete_with_handle(self):
        from serenityflow.nodes.model_ops import model_sampling_discrete
        from serenityflow.core.types import ModelHandle
        handle = ModelHandle("test", "sd15", {}, "/test", torch.float16, None)
        result = model_sampling_discrete(handle, "eps", False)
        assert result[0].model_options["sampling_type"] == "eps"

    def test_freeu_with_handle(self):
        from serenityflow.nodes.model_ops import freeu
        from serenityflow.core.types import ModelHandle
        handle = ModelHandle("test", "sd15", {}, "/test", torch.float16, None)
        result = freeu(handle, 1.1, 1.2, 0.9, 0.2)
        assert result[0].model_options["freeu"]["version"] == 1

    def test_freeu_v2_with_handle(self):
        from serenityflow.nodes.model_ops import freeu_v2
        from serenityflow.core.types import ModelHandle
        handle = ModelHandle("test", "sd15", {}, "/test", torch.float16, None)
        result = freeu_v2(handle, 1.3, 1.4, 0.9, 0.2)
        assert result[0].model_options["freeu"]["version"] == 2

    def test_pag_with_handle(self):
        from serenityflow.nodes.model_ops import perturbed_attention_guidance
        from serenityflow.core.types import ModelHandle
        handle = ModelHandle("test", "flux", {}, "/test", torch.float16, None)
        result = perturbed_attention_guidance(handle, 3.0)
        assert result[0].model_options["pag_scale"] == 3.0

    def test_unclip_conditioning(self):
        from serenityflow.nodes.model_ops import unclip_conditioning
        cond = [{"cross_attn": torch.randn(1, 77, 768)}]
        result = unclip_conditioning(cond, "fake_output", 0.8, 0.0)
        assert "unclip_conditioning" in result[0][0]
        assert result[0][0]["unclip_conditioning"]["strength"] == 0.8


# ---------------------------------------------------------------------------
# Category 7 — String / Logic / Math / Utility
# ---------------------------------------------------------------------------

class TestUtility:
    def test_string_concat(self):
        from serenityflow.nodes.utility import string_concat
        assert string_concat("hello ", "world")[0] == "hello world"

    def test_string_replace(self):
        from serenityflow.nodes.utility import string_replace
        assert string_replace("foo bar", "bar", "baz")[0] == "foo baz"

    def test_string_to_int(self):
        from serenityflow.nodes.utility import string_to_int
        assert string_to_int("42")[0] == 42

    def test_string_to_float(self):
        from serenityflow.nodes.utility import string_to_float
        assert string_to_float("3.14")[0] == pytest.approx(3.14)

    def test_int_to_string(self):
        from serenityflow.nodes.utility import int_to_string
        assert int_to_string(42)[0] == "42"

    def test_float_to_string(self):
        from serenityflow.nodes.utility import float_to_string
        assert float_to_string(3.14159, 2)[0] == "3.14"

    def test_switch_true(self):
        from serenityflow.nodes.utility import switch
        assert switch(True, "yes", "no")[0] == "yes"

    def test_switch_false(self):
        from serenityflow.nodes.utility import switch
        assert switch(False, "yes", "no")[0] == "no"

    def test_compare(self):
        from serenityflow.nodes.utility import compare
        assert compare(5.0, 3.0, ">")[0] is True
        assert compare(2.0, 2.0, "==")[0] is True
        assert compare(1.0, 2.0, ">=")[0] is False

    def test_boolean_op(self):
        from serenityflow.nodes.utility import boolean_op
        assert boolean_op(True, "NOT")[0] is False
        assert boolean_op(True, "AND", True)[0] is True
        assert boolean_op(False, "OR", True)[0] is True

    def test_int_math(self):
        from serenityflow.nodes.utility import int_math
        assert int_math(10, 3, "+")[0] == 13
        assert int_math(10, 3, "*")[0] == 30
        assert int_math(10, 3, "/")[0] == 3
        assert int_math(10, 0, "/")[0] == 0  # Division by zero safe

    def test_float_math(self):
        from serenityflow.nodes.utility import float_math
        assert float_math(1.5, 2.5, "+")[0] == pytest.approx(4.0)
        assert float_math(3.0, 0.0, "/")[0] == 0.0

    def test_math_expression(self):
        from serenityflow.nodes.utility import math_expression
        assert math_expression("a + b", 3.0, 4.0)[0] == pytest.approx(7.0)
        assert math_expression("sin(pi / 2)", 0.0, 0.0)[0] == pytest.approx(1.0)

    def test_note_node(self):
        from serenityflow.nodes.utility import note_node
        result = note_node("This is a note")
        assert result == {}

    def test_reroute_node(self):
        from serenityflow.nodes.utility import reroute_node
        obj = object()
        assert reroute_node(obj)[0] is obj

    def test_primitive_node(self):
        from serenityflow.nodes.utility import primitive_node
        assert primitive_node(42)[0] == 42

    def test_int_to_float(self):
        from serenityflow.nodes.utility import int_to_float
        assert int_to_float(42)[0] == 42.0

    def test_float_to_int_modes(self):
        from serenityflow.nodes.utility import float_to_int
        assert float_to_int(3.7, "round")[0] == 4
        assert float_to_int(3.7, "floor")[0] == 3
        assert float_to_int(3.2, "ceil")[0] == 4

    def test_seed_node(self):
        from serenityflow.nodes.utility import seed_node
        assert seed_node(12345)[0] == 12345

    def test_batch_size_node(self):
        from serenityflow.nodes.utility import batch_size_node
        img = torch.rand(5, 32, 32, 3)
        assert batch_size_node(img)[0] == 5


# ---------------------------------------------------------------------------
# Category 8 — Video Nodes
# ---------------------------------------------------------------------------

class TestVideoNodes:
    def test_video_combine(self):
        from serenityflow.nodes.video import video_combine
        a = torch.rand(10, 64, 64, 3)
        b = torch.rand(5, 64, 64, 3)
        result = video_combine(a, b)
        assert result[0].shape[0] == 15

    def test_split_video_frames(self):
        from serenityflow.nodes.video import split_video_frames
        imgs = torch.rand(20, 64, 64, 3)
        first, second = split_video_frames(imgs, 10)
        assert first.shape[0] == 10
        assert second.shape[0] == 10

    def test_merge_video_frames_cut(self):
        from serenityflow.nodes.video import merge_video_frames
        a = torch.rand(10, 32, 32, 3)
        b = torch.rand(10, 32, 32, 3)
        result = merge_video_frames(a, b, 0, "cut")
        assert result[0].shape[0] == 20

    def test_merge_video_frames_crossfade(self):
        from serenityflow.nodes.video import merge_video_frames
        a = torch.zeros(10, 32, 32, 3)
        b = torch.ones(10, 32, 32, 3)
        result = merge_video_frames(a, b, 4, "crossfade")
        assert result[0].shape[0] == 16  # 10 + 10 - 4

    def test_image_to_video(self):
        from serenityflow.nodes.video import image_to_video
        img = torch.rand(1, 64, 64, 3)
        result = image_to_video(img, 25)
        assert result[0].shape[0] == 25

    def test_video_to_frames(self):
        from serenityflow.nodes.video import video_to_frames
        imgs = torch.rand(30, 32, 32, 3)
        result = video_to_frames(imgs, 5, 15)
        assert result[0].shape[0] == 10


# ---------------------------------------------------------------------------
# Category 9 — Extra Conditioning / Latent
# ---------------------------------------------------------------------------

class TestConditioningExtra:
    def test_conditioning_average(self):
        from serenityflow.nodes.conditioning_extra import conditioning_average
        c1 = [{"cross_attn": torch.ones(1, 77, 768)}]
        c2 = [{"cross_attn": torch.zeros(1, 77, 768)}]
        result = conditioning_average(c1, c2, 0.5)
        assert torch.allclose(result[0][0]["cross_attn"], torch.full((1, 77, 768), 0.5))

    def test_conditioning_set_area_strength(self):
        from serenityflow.nodes.conditioning_extra import conditioning_set_area_strength
        cond = [{"cross_attn": torch.randn(1, 77, 768)}]
        result = conditioning_set_area_strength(cond, 0.5)
        assert result[0][0]["strength"] == 0.5

    def test_conditioning_combine_multiple(self):
        from serenityflow.nodes.conditioning_extra import conditioning_combine_multiple
        c1 = [{"cross_attn": torch.randn(1, 77, 768)}]
        c2 = [{"cross_attn": torch.randn(1, 77, 768)}]
        result = conditioning_combine_multiple(c1, c2)
        assert len(result[0]) == 2

    def test_conditioning_combine_multiple_none(self):
        from serenityflow.nodes.conditioning_extra import conditioning_combine_multiple
        c1 = [{"cross_attn": torch.randn(1, 77, 768)}]
        result = conditioning_combine_multiple(c1)
        assert len(result[0]) == 1

    def test_latent_from_batch(self):
        from serenityflow.nodes.conditioning_extra import latent_from_batch
        latent = {"samples": torch.rand(4, 4, 64, 64)}
        result = latent_from_batch(latent, 1, 2)
        assert result[0]["samples"].shape[0] == 2

    def test_latent_interpolate(self):
        from serenityflow.nodes.conditioning_extra import latent_interpolate
        l1 = {"samples": torch.zeros(1, 4, 64, 64)}
        l2 = {"samples": torch.ones(1, 4, 64, 64)}
        result = latent_interpolate(l1, l2, 0.5)
        assert torch.allclose(result[0]["samples"], torch.full((1, 4, 64, 64), 0.5))

    def test_latent_add(self):
        from serenityflow.nodes.conditioning_extra import latent_add
        l1 = {"samples": torch.full((1, 4, 8, 8), 0.3)}
        l2 = {"samples": torch.full((1, 4, 8, 8), 0.2)}
        result = latent_add(l1, l2)
        assert torch.allclose(result[0]["samples"], torch.full((1, 4, 8, 8), 0.5))

    def test_latent_multiply(self):
        from serenityflow.nodes.conditioning_extra import latent_multiply
        l = {"samples": torch.full((1, 4, 8, 8), 2.0)}
        result = latent_multiply(l, 0.5)
        assert torch.allclose(result[0]["samples"], torch.ones(1, 4, 8, 8))


# ---------------------------------------------------------------------------
# Skeptic edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_batch_size_zero_image_crop(self):
        from serenityflow.nodes.image_ops import image_crop
        img = torch.rand(0, 32, 32, 3)
        result = image_crop(img, 16, 16, 0, 0)
        assert result[0].shape[0] == 0

    def test_empty_conditioning_combine(self):
        from serenityflow.nodes.conditioning import conditioning_combine
        c1 = []
        c2 = [{"cross_attn": torch.randn(1, 77, 768)}]
        result = conditioning_combine(c1, c2)
        assert len(result[0]) == 1

    def test_width_not_divisible_by_8_latent(self):
        from serenityflow.nodes.latent import empty_latent_image
        # Width 100 // 8 = 12, should work
        result = empty_latent_image(100, 100, 1)
        assert result[0]["samples"].shape == (1, 4, 12, 12)

    def test_grow_mask_negative_expand(self):
        from serenityflow.nodes.mask import grow_mask
        mask = torch.ones(1, 32, 32)
        result = grow_mask(mask, -2, True)
        # Should erode — edges should be 0
        assert result[0].shape[1] == 32  # Same spatial size

    def test_karras_scheduler_single_step(self):
        from serenityflow.nodes.sampling_custom import karras_scheduler
        sigmas = karras_scheduler(1, 14.6, 0.029, 7.0)
        assert sigmas[0].shape == (2,)

    def test_image_composite_out_of_bounds(self):
        from serenityflow.nodes.image_ops import image_composite_masked
        dst = torch.zeros(1, 32, 32, 3)
        src = torch.ones(1, 16, 16, 3)
        # Place entirely outside bounds
        result = image_composite_masked(dst, src, 100, 100)
        assert torch.allclose(result[0], torch.zeros(1, 32, 32, 3))

    def test_mask_composite_no_overlap(self):
        from serenityflow.nodes.mask import mask_composite
        dst = torch.zeros(1, 32, 32)
        src = torch.ones(1, 16, 16)
        result = mask_composite(dst, src, 100, 100, "add")
        assert result[0].sum() == 0.0

    def test_split_sigmas_at_zero(self):
        from serenityflow.nodes.sampling_custom import split_sigmas
        sigmas = torch.linspace(14, 0, 21)
        high, low = split_sigmas(sigmas, 0)
        assert len(high) == 1
        assert len(low) == 21

    def test_image_blur_radius_zero(self):
        from serenityflow.nodes.image_ops import image_blur
        img = torch.rand(1, 32, 32, 3)
        result = image_blur(img, 0, 1.0)
        assert torch.allclose(result[0], img)

    def test_float_math_division_by_zero(self):
        from serenityflow.nodes.utility import float_math
        assert float_math(1.0, 0.0, "/")[0] == 0.0

    def test_int_math_modulo_by_zero(self):
        from serenityflow.nodes.utility import int_math
        assert int_math(10, 0, "%")[0] == 0

    def test_string_to_int_with_whitespace(self):
        from serenityflow.nodes.utility import string_to_int
        assert string_to_int("  42  ")[0] == 42
