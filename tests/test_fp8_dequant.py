"""Tests for serenityflow.bridge.fp8_dequant — universal scaled FP8 dequantization."""
from __future__ import annotations

import torch
import pytest

from serenityflow.bridge.fp8_dequant import (
    dequantize_fp8,
    dequant_scaled_fp8,
    dequant_scaled_fp8_in_model,
    has_fp8_scales,
    _find_scale_pairs,
)


# Skip entire module if FP8 dtypes not available
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="FP8 dtypes not available in this PyTorch build",
)


def _make_fp8_weight(shape=(64, 64), scale_value=0.002):
    """Create a fake FP8 weight + scale pair."""
    # Random bf16 → clamp to FP8 range → cast to FP8
    raw = torch.randn(shape, dtype=torch.bfloat16).clamp(-448, 448)
    fp8 = raw.to(torch.float8_e4m3fn)
    scale = torch.tensor(scale_value, dtype=torch.float32)
    return fp8, scale


class TestFindScalePairs:
    def test_ltx_pattern(self):
        keys = [
            "blocks.0.attn.to_q.weight",
            "blocks.0.attn.to_q.weight_scale",
            "blocks.0.attn.to_k.weight",
            "blocks.0.attn.to_k.weight_scale",
        ]
        pairs = _find_scale_pairs(keys)
        assert len(pairs) == 2
        assert pairs["blocks.0.attn.to_q.weight"] == "blocks.0.attn.to_q.weight_scale"
        assert pairs["blocks.0.attn.to_k.weight"] == "blocks.0.attn.to_k.weight_scale"

    def test_flux_pattern(self):
        keys = [
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_q.scale_weight",
            "transformer_blocks.0.attn.to_k.weight",
            "transformer_blocks.0.attn.to_k.scale_weight",
        ]
        pairs = _find_scale_pairs(keys)
        assert len(pairs) == 2
        assert pairs["transformer_blocks.0.attn.to_q.weight"] == "transformer_blocks.0.attn.to_q.scale_weight"

    def test_no_scale_keys(self):
        keys = ["blocks.0.weight", "blocks.0.bias"]
        pairs = _find_scale_pairs(keys)
        assert len(pairs) == 0

    def test_orphan_scale_no_match(self):
        keys = ["blocks.0.attn.to_q.weight_scale"]  # No matching weight key
        pairs = _find_scale_pairs(keys)
        assert len(pairs) == 0


class TestHasFp8Scales:
    def test_detects_ltx_scales(self):
        sd = {
            "blocks.0.weight": torch.zeros(1),
            "blocks.0.weight_scale": torch.zeros(1),
        }
        assert has_fp8_scales(sd) is True

    def test_detects_flux_scales(self):
        sd = {
            "blocks.0.weight": torch.zeros(1),
            "blocks.0.scale_weight": torch.zeros(1),
        }
        assert has_fp8_scales(sd) is True

    def test_no_scales(self):
        sd = {"blocks.0.weight": torch.zeros(1), "blocks.0.bias": torch.zeros(1)}
        assert has_fp8_scales(sd) is False


class TestDequantScaledFp8:
    def test_ltx_dequant(self):
        fp8_w, scale = _make_fp8_weight((32, 32), scale_value=0.5)
        sd = {
            "blocks.0.weight": fp8_w,
            "blocks.0.weight_scale": scale,
            "blocks.0.bias": torch.randn(32, dtype=torch.bfloat16),
        }
        result = dequant_scaled_fp8(sd)

        # Scale key removed
        assert "blocks.0.weight_scale" not in result
        # Weight is now bf16
        assert result["blocks.0.weight"].dtype == torch.bfloat16
        # Bias unchanged
        assert result["blocks.0.bias"].dtype == torch.bfloat16
        expected = dequantize_fp8(fp8_w, scale)
        assert torch.allclose(result["blocks.0.weight"], expected)

    def test_flux_dequant(self):
        fp8_w, scale = _make_fp8_weight((16, 16), scale_value=0.1)
        sd = {
            "blocks.0.attn.to_q.weight": fp8_w,
            "blocks.0.attn.to_q.scale_weight": scale,
        }
        result = dequant_scaled_fp8(sd)

        assert "blocks.0.attn.to_q.scale_weight" not in result
        assert result["blocks.0.attn.to_q.weight"].dtype == torch.bfloat16
        expected = dequantize_fp8(fp8_w, scale)
        assert torch.allclose(result["blocks.0.attn.to_q.weight"], expected)

    def test_dequantize_fp8_handles_rowwise_scale(self):
        fp8_w, _ = _make_fp8_weight((8, 4), scale_value=1.0)
        row_scale = torch.linspace(0.1, 0.8, 8, dtype=torch.float32).unsqueeze(-1)

        expected = (fp8_w.to(torch.float32) * row_scale).to(torch.bfloat16)

        assert torch.allclose(dequantize_fp8(fp8_w, row_scale), expected)

    def test_dequantize_fp8_repairs_flat_row_scale(self):
        fp8_w, _ = _make_fp8_weight((8, 4), scale_value=1.0)
        row_scale = torch.linspace(0.1, 0.8, 8, dtype=torch.float32)

        expected = (fp8_w.to(torch.float32) * row_scale.unsqueeze(-1)).to(torch.bfloat16)

        assert torch.allclose(dequantize_fp8(fp8_w, row_scale), expected)

    def test_dequantize_fp8_expands_blockwise_scale(self):
        fp8_w, _ = _make_fp8_weight((4, 4), scale_value=1.0)
        block_scale = torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=torch.float32)
        expanded = block_scale.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)

        expected = (fp8_w.to(torch.float32) * expanded).to(torch.bfloat16)

        assert torch.allclose(dequantize_fp8(fp8_w, block_scale), expected)

    def test_removes_input_scale_keys(self):
        fp8_w, scale = _make_fp8_weight()
        sd = {
            "blocks.0.weight": fp8_w,
            "blocks.0.weight_scale": scale,
            "blocks.0.input_scale": torch.tensor(1.0),
            "blocks.1.scale_input": torch.tensor(1.0),
        }
        result = dequant_scaled_fp8(sd)
        assert "blocks.0.input_scale" not in result
        assert "blocks.1.scale_input" not in result

    def test_no_fp8_passthrough(self):
        sd = {
            "blocks.0.weight": torch.randn(16, 16, dtype=torch.bfloat16),
            "blocks.0.bias": torch.randn(16, dtype=torch.bfloat16),
        }
        original_weight = sd["blocks.0.weight"].clone()
        result = dequant_scaled_fp8(sd)
        assert torch.equal(result["blocks.0.weight"], original_weight)

    def test_mixed_patterns(self):
        fp8_w1, scale1 = _make_fp8_weight((8, 8), 0.3)
        fp8_w2, scale2 = _make_fp8_weight((8, 8), 0.7)
        sd = {
            "ltx.blocks.0.weight": fp8_w1,
            "ltx.blocks.0.weight_scale": scale1,
            "flux.blocks.0.weight": fp8_w2,
            "flux.blocks.0.scale_weight": scale2,
        }
        result = dequant_scaled_fp8(sd)
        assert "ltx.blocks.0.weight_scale" not in result
        assert "flux.blocks.0.scale_weight" not in result
        assert result["ltx.blocks.0.weight"].dtype == torch.bfloat16
        assert result["flux.blocks.0.weight"].dtype == torch.bfloat16


class TestDequantScaledFp8InModel:
    def test_dequant_fixes_model_weights(self, tmp_path):
        """Write a fake FP8 safetensors, load model, verify dequant fixes weights."""
        from safetensors.torch import save_file

        fp8_w, scale = _make_fp8_weight((16, 16), scale_value=0.25)
        expected = dequantize_fp8(fp8_w, scale)

        # Save FP8 checkpoint
        checkpoint = {
            "linear.weight": fp8_w,
            "linear.weight_scale": scale,
        }
        ckpt_path = str(tmp_path / "test_fp8.safetensors")
        save_file(checkpoint, ckpt_path)

        # Create a model with wrong bf16 values (simulating naive cast)
        model = torch.nn.Module()
        model.linear = torch.nn.Linear(16, 16, bias=False)
        model.linear.weight = torch.nn.Parameter(fp8_w.to(torch.bfloat16))  # Wrong: cast without scale

        fixed = dequant_scaled_fp8_in_model(model, ckpt_path)
        assert fixed == 1
        assert torch.allclose(model.linear.weight.data, expected)

    def test_no_fp8_returns_zero(self, tmp_path):
        from safetensors.torch import save_file

        checkpoint = {"linear.weight": torch.randn(8, 8, dtype=torch.bfloat16)}
        ckpt_path = str(tmp_path / "test_bf16.safetensors")
        save_file(checkpoint, ckpt_path)

        model = torch.nn.Module()
        model.linear = torch.nn.Linear(8, 8, bias=False)

        fixed = dequant_scaled_fp8_in_model(model, ckpt_path)
        assert fixed == 0
