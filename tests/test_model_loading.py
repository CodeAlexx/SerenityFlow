"""Tests for Multi-LoRA and advanced model loading nodes."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all_nodes():
    from serenityflow.nodes.registry import registry
    import serenityflow.nodes  # noqa: F401
    return registry.list_all()


def _make_linear_model(in_f=4, out_f=4):
    """Create a simple nn.Module for testing merge operations."""
    model = nn.Sequential(nn.Linear(in_f, out_f, bias=False))
    return model


class _MockModelHandle:
    """Minimal ModelHandle mock with with_options support."""
    def __init__(self, options=None):
        self.model_options = options or {}

    def with_options(self, opts):
        merged = dict(self.model_options)
        merged.update(opts)
        return _MockModelHandle(merged)


# ---------------------------------------------------------------------------
# Task 1: LoraLoaderStack with 5 slots
# ---------------------------------------------------------------------------

class TestLoraLoaderStack:
    def test_registration_has_5_slots(self, all_nodes):
        nd = all_nodes["LoraLoaderStack"]
        optional = nd.input_types.get("optional", {})
        assert "lora_name_4" in optional, "Missing lora_name_4 optional input"
        assert "strength_4" in optional, "Missing strength_4 optional input"
        assert "lora_name_5" in optional, "Missing lora_name_5 optional input"
        assert "strength_5" in optional, "Missing strength_5 optional input"

    @patch("serenityflow.bridge.model_paths.get_model_paths")
    @patch("serenityflow.bridge.serenity_api.apply_lora_clip")
    @patch("serenityflow.bridge.serenity_api.apply_lora")
    def test_applies_all_5_loras(self, mock_apply, mock_apply_clip, mock_paths):
        from serenityflow.nodes.conditioning_extra import lora_loader_stack

        mock_paths.return_value.find = lambda name, folder: f"/fake/{name}"
        mock_apply.side_effect = lambda m, p, strength: m
        mock_apply_clip.side_effect = lambda c, p, strength: c

        model = MagicMock()
        clip = MagicMock()

        result = lora_loader_stack(
            model, clip,
            lora_name_1="a.safetensors", strength_1=1.0,
            lora_name_2="b.safetensors", strength_2=0.8,
            lora_name_3="c.safetensors", strength_3=0.6,
            lora_name_4="d.safetensors", strength_4=0.4,
            lora_name_5="e.safetensors", strength_5=0.2,
        )
        assert mock_apply.call_count == 5
        assert mock_apply_clip.call_count == 5
        assert len(result) == 2

    @patch("serenityflow.bridge.model_paths.get_model_paths")
    @patch("serenityflow.bridge.serenity_api.apply_lora_clip")
    @patch("serenityflow.bridge.serenity_api.apply_lora")
    def test_skips_none_slots(self, mock_apply, mock_apply_clip, mock_paths):
        from serenityflow.nodes.conditioning_extra import lora_loader_stack

        mock_paths.return_value.find = lambda name, folder: f"/fake/{name}"
        mock_apply.side_effect = lambda m, p, strength: m
        mock_apply_clip.side_effect = lambda c, p, strength: c

        model = MagicMock()
        clip = MagicMock()

        lora_loader_stack(
            model, clip,
            lora_name_1="a.safetensors", strength_1=1.0,
            lora_name_3="c.safetensors", strength_3=0.5,
        )
        # Only slots 1 and 3 have names, so 2 calls
        assert mock_apply.call_count == 2


# ---------------------------------------------------------------------------
# Task 2: ModelMergeSimple
# ---------------------------------------------------------------------------

class TestModelMergeSimple:
    def test_nn_module_merge(self):
        from serenityflow.nodes.model_ops import model_merge_simple

        m1 = _make_linear_model()
        m2 = _make_linear_model()
        # Set known weights
        with torch.no_grad():
            m1[0].weight.fill_(0.0)
            m2[0].weight.fill_(1.0)

        (merged,) = model_merge_simple(m1, m2, ratio=0.5)
        # Should be 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        assert torch.allclose(merged[0].weight, torch.full_like(m1[0].weight, 0.5), atol=1e-6)

    def test_nn_module_merge_ratio_zero(self):
        from serenityflow.nodes.model_ops import model_merge_simple

        m1 = _make_linear_model()
        m2 = _make_linear_model()
        with torch.no_grad():
            m1[0].weight.fill_(2.0)
            m2[0].weight.fill_(8.0)

        (merged,) = model_merge_simple(m1, m2, ratio=0.0)
        assert torch.allclose(merged[0].weight, torch.full_like(m1[0].weight, 2.0))

    def test_model_handle_merge(self):
        from serenityflow.nodes.model_ops import model_merge_simple

        h1 = _MockModelHandle()
        h2 = _MockModelHandle()
        (result,) = model_merge_simple(h1, h2, ratio=0.7)
        assert result.model_options["merge"]["ratio"] == 0.7
        assert result.model_options["merge"]["type"] == "simple"


# ---------------------------------------------------------------------------
# Task 3: ModelMergeBlocks
# ---------------------------------------------------------------------------

class TestModelMergeBlocks:
    def test_model_handle_blocks(self):
        from serenityflow.nodes.model_ops import model_merge_blocks

        h1 = _MockModelHandle()
        h2 = _MockModelHandle()
        # Note: 'input' is a builtin, but it's used as a kwarg name in the node
        (result,) = model_merge_blocks(h1, h2, **{"input": 0.3, "middle": 0.5, "out": 0.7})
        merge = result.model_options["merge"]
        assert merge["type"] == "blocks"
        assert merge["input"] == 0.3
        assert merge["middle"] == 0.5
        assert merge["out"] == 0.7

    def test_nn_module_block_merge(self):
        from serenityflow.nodes.model_ops import model_merge_blocks

        # Simple models -- all keys classify as "middle" (default)
        m1 = _make_linear_model()
        m2 = _make_linear_model()
        with torch.no_grad():
            m1[0].weight.fill_(0.0)
            m2[0].weight.fill_(1.0)

        (merged,) = model_merge_blocks(m1, m2, **{"input": 0.0, "middle": 0.75, "out": 0.0})
        # Keys classify as "middle" -> ratio 0.75
        expected = 0.25 * 0.0 + 0.75 * 1.0  # (1-r)*m1 + r*m2
        assert torch.allclose(merged[0].weight, torch.full_like(m1[0].weight, expected), atol=1e-6)


# ---------------------------------------------------------------------------
# Task 4: CLIPMergeSimple
# ---------------------------------------------------------------------------

class TestCLIPMergeSimple:
    def test_nn_module_clip_merge(self):
        from serenityflow.nodes.model_ops import clip_merge_simple

        c1 = nn.Linear(4, 4, bias=False)
        c2 = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            c1.weight.fill_(0.0)
            c2.weight.fill_(1.0)

        (merged,) = clip_merge_simple(c1, c2, ratio=0.3)
        expected = 0.7 * 0.0 + 0.3 * 1.0
        assert torch.allclose(merged.weight, torch.full_like(c1.weight, expected), atol=1e-6)


# ---------------------------------------------------------------------------
# Task 5: CLIPLoaderGGUF registration
# ---------------------------------------------------------------------------

class TestCLIPLoaderGGUF:
    def test_registered(self, all_nodes):
        assert "CLIPLoaderGGUF" in all_nodes
        nd = all_nodes["CLIPLoaderGGUF"]
        assert nd.return_types == ("CLIP",)
        req = nd.input_types["required"]
        assert "clip_name" in req
        assert "clip_type" in req


# ---------------------------------------------------------------------------
# Task 6: PatchModelAddDownscale
# ---------------------------------------------------------------------------

class TestPatchModelAddDownscale:
    def test_with_options(self):
        from serenityflow.nodes.model_ops import patch_model_add_downscale

        h = _MockModelHandle()
        (result,) = patch_model_add_downscale(
            h, block_number=3, downscale_factor=2.0,
            start_percent=0.0, end_percent=0.35, downscale_after_skip=True
        )
        patch = result.model_options["downscale_patch"]
        assert patch["block_number"] == 3
        assert patch["downscale_factor"] == 2.0
        assert patch["start_percent"] == 0.0
        assert patch["end_percent"] == 0.35
        assert patch["downscale_after_skip"] is True

    def test_registered(self, all_nodes):
        assert "PatchModelAddDownscale" in all_nodes
        nd = all_nodes["PatchModelAddDownscale"]
        assert nd.return_types == ("MODEL",)


# ---------------------------------------------------------------------------
# Task 7: UpscaleModelLoader registration
# ---------------------------------------------------------------------------

class TestUpscaleModelLoader:
    def test_registered(self, all_nodes):
        assert "UpscaleModelLoader" in all_nodes
        nd = all_nodes["UpscaleModelLoader"]
        assert nd.return_types == ("UPSCALE_MODEL",)
        assert "model_name" in nd.input_types["required"]


# ---------------------------------------------------------------------------
# Task 8: ImageUpscaleWithModel tile processing
# ---------------------------------------------------------------------------

class TestImageUpscaleWithModel:
    def test_tile_upscale_with_mock_model(self):
        from serenityflow.nodes.model_ops import image_upscale_with_model

        # Create a simple 2x upscale mock model
        class Mock2xUpscale(nn.Module):
            def forward(self, x):
                return torch.nn.functional.interpolate(
                    x, scale_factor=2, mode="nearest"
                )

        model = Mock2xUpscale()
        # Create a small test image: BHWC float32
        image = torch.rand(1, 32, 32, 3)
        (result,) = image_upscale_with_model(model, image)

        assert result.shape == (1, 64, 64, 3), f"Expected (1,64,64,3), got {result.shape}"
        assert result.dtype == torch.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_dict_model_bicubic_fallback(self):
        from serenityflow.nodes.model_ops import image_upscale_with_model

        # Dict without bridge -> falls through to bicubic 4x
        fake_model = {"state_dict": {}, "path": "/fake/model.pth"}
        image = torch.rand(1, 16, 16, 3)

        with patch("serenityflow.nodes.model_ops.image_upscale_with_model.__module__", "test"):
            # The function should fall through to bicubic since bridge isn't available
            (result,) = image_upscale_with_model(fake_model, image)
            assert result.shape[1] == 64  # 4x upscale
            assert result.shape[2] == 64


# ---------------------------------------------------------------------------
# Task 9: CLIPVisionLoader and CLIPVisionEncode registration
# ---------------------------------------------------------------------------

class TestCLIPVision:
    def test_loader_registered(self, all_nodes):
        assert "CLIPVisionLoader" in all_nodes
        nd = all_nodes["CLIPVisionLoader"]
        assert nd.return_types == ("CLIP_VISION",)

    def test_encode_registered(self, all_nodes):
        assert "CLIPVisionEncode" in all_nodes
        nd = all_nodes["CLIPVisionEncode"]
        assert nd.return_types == ("CLIP_VISION_OUTPUT",)

    def test_encode_with_nn_module(self):
        from serenityflow.nodes.model_ops import clip_vision_encode

        # Create a simple mock vision model
        class MockVision(nn.Module):
            def forward(self, x):
                B = x.shape[0]
                return torch.randn(B, 257, 768)

        model = MockVision()
        image = torch.rand(1, 224, 224, 3)
        (result,) = clip_vision_encode(model, image)
        assert "last_hidden_state" in result
        assert result["last_hidden_state"].shape == (1, 257, 768)
