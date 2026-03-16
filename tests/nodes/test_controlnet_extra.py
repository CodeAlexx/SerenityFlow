"""Tests for ControlNet extra nodes -- ControlNetApply, stacking, preprocessors, T2I."""
from __future__ import annotations

import torch
import pytest


def _make_cond(batch=1, seq=77, dim=768):
    """Create test conditioning in ComfyUI format: list[dict]."""
    return [{"cross_attn": torch.randn(batch, seq, dim)}]


def _make_image(batch=1, h=64, w=64, c=3):
    """Create test image tensor BHWC float32 [0,1]."""
    return torch.rand(batch, h, w, c)


class TestControlNetApply:
    def test_adds_control_hint(self):
        from serenityflow.nodes.controlnet import controlnet_apply
        cond = _make_cond()
        image = _make_image()
        cn = {"type": "controlnet", "model": "test"}
        result, = controlnet_apply(cond, cn, image, strength=0.8)
        assert len(result) == 1
        assert "control_hints" in result[0]
        assert len(result[0]["control_hints"]) == 1
        hint = result[0]["control_hints"][0]
        assert hint["control_net"] is cn
        assert hint["strength"] == 0.8

    def test_stacks_multiple_hints(self):
        from serenityflow.nodes.controlnet import controlnet_apply
        cond = _make_cond()
        cn1 = {"type": "cn1"}
        cn2 = {"type": "cn2"}
        image = _make_image()
        result, = controlnet_apply(cond, cn1, image)
        result, = controlnet_apply(result, cn2, image)
        assert len(result[0]["control_hints"]) == 2

    def test_does_not_mutate_original(self):
        from serenityflow.nodes.controlnet import controlnet_apply
        cond = _make_cond()
        original_keys = set(cond[0].keys())
        controlnet_apply(cond, {"type": "cn"}, _make_image())
        assert set(cond[0].keys()) == original_keys


class TestControlNetStack:
    def test_single_entry(self):
        from serenityflow.nodes.controlnet_extra import controlnet_stack
        cn = {"type": "cn1"}
        image = _make_image()
        stack, = controlnet_stack(cn, image, strength_1=0.5, start_percent_1=0.1, end_percent_1=0.9)
        assert len(stack) == 1
        assert stack[0]["control_net"] is cn
        assert stack[0]["strength"] == 0.5
        assert stack[0]["start_percent"] == 0.1
        assert stack[0]["end_percent"] == 0.9

    def test_three_entries(self):
        from serenityflow.nodes.controlnet_extra import controlnet_stack
        cns = [{"type": f"cn{i}"} for i in range(3)]
        imgs = [_make_image() for _ in range(3)]
        stack, = controlnet_stack(
            cns[0], imgs[0],
            control_net_2=cns[1], image_2=imgs[1],
            control_net_3=cns[2], image_3=imgs[2],
        )
        assert len(stack) == 3

    def test_five_entries(self):
        from serenityflow.nodes.controlnet_extra import controlnet_stack
        cns = [{"type": f"cn{i}"} for i in range(5)]
        imgs = [_make_image() for _ in range(5)]
        stack, = controlnet_stack(
            cns[0], imgs[0],
            control_net_2=cns[1], image_2=imgs[1],
            control_net_3=cns[2], image_3=imgs[2],
            control_net_4=cns[3], image_4=imgs[3],
            control_net_5=cns[4], image_5=imgs[4],
        )
        assert len(stack) == 5

    def test_skips_none_entries(self):
        from serenityflow.nodes.controlnet_extra import controlnet_stack
        cn = {"type": "cn1"}
        image = _make_image()
        stack, = controlnet_stack(cn, image, control_net_3={"type": "cn3"}, image_3=_make_image())
        # Entry 2 is None, should be skipped; entry 3 present
        assert len(stack) == 2


class TestApplyControlNetStack:
    def test_applies_stack_to_conditioning(self):
        from serenityflow.nodes.controlnet_extra import apply_controlnet_stack
        pos = _make_cond()
        neg = _make_cond()
        stack = [
            {"control_net": "cn1", "image": _make_image(), "strength": 1.0,
             "start_percent": 0.0, "end_percent": 1.0},
            {"control_net": "cn2", "image": _make_image(), "strength": 0.5,
             "start_percent": 0.2, "end_percent": 0.8},
        ]
        pos_out, neg_out = apply_controlnet_stack(pos, neg, stack)
        assert len(pos_out[0]["control_hints"]) == 2
        assert len(neg_out[0]["control_hints"]) == 2
        assert pos_out[0]["control_hints"][1]["strength"] == 0.5

    def test_empty_stack_passthrough(self):
        from serenityflow.nodes.controlnet_extra import apply_controlnet_stack
        pos = _make_cond()
        neg = _make_cond()
        pos_out, neg_out = apply_controlnet_stack(pos, neg, [])
        assert "control_hints" not in pos_out[0]


class TestTilePreprocessor:
    def test_output_shape_matches_input(self):
        from serenityflow.nodes.controlnet_extra import tile_preprocessor
        image = _make_image(2, 128, 128)
        result, = tile_preprocessor(image, pyrUp_iters=3)
        assert result.shape == image.shape

    def test_values_in_range(self):
        from serenityflow.nodes.controlnet_extra import tile_preprocessor
        image = _make_image()
        result, = tile_preprocessor(image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_batch_dimension(self):
        from serenityflow.nodes.controlnet_extra import tile_preprocessor
        image = _make_image(4, 32, 32)
        result, = tile_preprocessor(image, pyrUp_iters=2)
        assert result.shape[0] == 4


class TestSoftEdgePreprocessor:
    def test_produces_edge_map(self):
        from serenityflow.nodes.controlnet_extra import soft_edge_preprocessor
        image = _make_image(1, 64, 64)
        result, = soft_edge_preprocessor(image)
        assert result.shape == image.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_safe_mode(self):
        from serenityflow.nodes.controlnet_extra import soft_edge_preprocessor
        image = _make_image(1, 64, 64)
        result, = soft_edge_preprocessor(image, safe=True)
        assert result.shape == image.shape

    def test_batch_dimension(self):
        from serenityflow.nodes.controlnet_extra import soft_edge_preprocessor
        image = _make_image(3, 32, 32)
        result, = soft_edge_preprocessor(image)
        assert result.shape[0] == 3


class TestLineartPreprocessor:
    def test_produces_lineart(self):
        from serenityflow.nodes.controlnet_extra import lineart_preprocessor
        image = _make_image(1, 64, 64)
        result, = lineart_preprocessor(image, coarse=False)
        assert result.shape == image.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_coarse_mode(self):
        from serenityflow.nodes.controlnet_extra import lineart_preprocessor
        image = _make_image(1, 64, 64)
        result, = lineart_preprocessor(image, coarse=True)
        assert result.shape == image.shape

    def test_batch_dimension(self):
        from serenityflow.nodes.controlnet_extra import lineart_preprocessor
        image = _make_image(2, 48, 48)
        result, = lineart_preprocessor(image)
        assert result.shape[0] == 2


class TestDepthAnythingPreprocessor:
    def test_fallback_returns_grayscale(self):
        from serenityflow.nodes.controlnet_extra import depth_anything_preprocessor
        image = _make_image(1, 64, 64)
        result, = depth_anything_preprocessor(image)
        assert result.shape == image.shape

    def test_batch_dimension(self):
        from serenityflow.nodes.controlnet_extra import depth_anything_preprocessor
        image = _make_image(3, 32, 32)
        result, = depth_anything_preprocessor(image)
        assert result.shape[0] == 3


class TestOpenPosePreprocessor:
    def test_fallback_returns_black(self):
        from serenityflow.nodes.controlnet_extra import openpose_preprocessor
        image = _make_image(1, 64, 64)
        result, = openpose_preprocessor(image)
        assert result.shape == image.shape
        assert result.max() == 0.0  # black image


class TestT2IAdapterApply:
    def test_adds_adapter_to_conditioning(self):
        from serenityflow.nodes.controlnet_extra import t2i_adapter_apply
        cond = _make_cond()
        adapter = {"model_name": "test", "type": "t2i_adapter"}
        image = _make_image()
        result, = t2i_adapter_apply(cond, adapter, image, strength=0.7)
        assert "t2i_adapters" in result[0]
        assert len(result[0]["t2i_adapters"]) == 1
        assert result[0]["t2i_adapters"][0]["strength"] == 0.7

    def test_stacks_multiple_adapters(self):
        from serenityflow.nodes.controlnet_extra import t2i_adapter_apply
        cond = _make_cond()
        a1 = {"model_name": "a1", "type": "t2i_adapter"}
        a2 = {"model_name": "a2", "type": "t2i_adapter"}
        image = _make_image()
        result, = t2i_adapter_apply(cond, a1, image)
        result, = t2i_adapter_apply(result, a2, image)
        assert len(result[0]["t2i_adapters"]) == 2


class TestT2IAdapterLoader:
    def test_registration(self):
        from serenityflow.nodes.registry import registry
        assert registry.has("T2IAdapterLoader")
        node = registry.get("T2IAdapterLoader")
        assert node.return_types == ("T2I_ADAPTER",)
