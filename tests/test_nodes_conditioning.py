"""Tests for conditioning node implementations."""
from __future__ import annotations

import torch
import pytest

from serenityflow.bridge.types import find_cross_attn_key


def _make_cond(batch=1, seq=77, dim=768, pooled_dim=None):
    """Create test conditioning in ComfyUI format: list[dict]."""
    entry = {"cross_attn": torch.randn(batch, seq, dim)}
    if pooled_dim is not None:
        entry["pooled_output"] = torch.randn(batch, pooled_dim)
    return [entry]


class TestConditioningSetArea:
    def test_area_added(self):
        from serenityflow.nodes.conditioning import conditioning_set_area
        cond = _make_cond()
        result, = conditioning_set_area(cond, width=512, height=512, x=0, y=0, strength=1.0)
        assert len(result) == 1
        assert "area" in result[0]
        assert result[0]["area"] == (64, 64, 0, 0)  # 512/8 = 64

    def test_strength_set(self):
        from serenityflow.nodes.conditioning import conditioning_set_area
        cond = _make_cond()
        result, = conditioning_set_area(cond, 256, 256, 64, 64, strength=0.5)
        assert result[0]["strength"] == 0.5

    def test_original_unchanged(self):
        from serenityflow.nodes.conditioning import conditioning_set_area
        cond = _make_cond()
        original_keys = set(cond[0].keys())
        conditioning_set_area(cond, 512, 512, 0, 0, 1.0)
        assert set(cond[0].keys()) == original_keys  # Not mutated


class TestConditioningSetMask:
    def test_mask_added(self):
        from serenityflow.nodes.conditioning import conditioning_set_mask
        cond = _make_cond()
        mask = torch.ones(1, 64, 64)
        result, = conditioning_set_mask(cond, mask, strength=0.8)
        assert "mask" in result[0]
        assert result[0]["strength"] == 0.8

    def test_set_area_to_bounds(self):
        from serenityflow.nodes.conditioning import conditioning_set_mask
        cond = _make_cond()
        mask = torch.ones(1, 64, 64)
        result, = conditioning_set_mask(cond, mask, 1.0, set_cond_area="mask bounds")
        assert result[0]["set_area_to_bounds"] is True


class TestConditioningCombine:
    def test_combines_lists(self):
        from serenityflow.nodes.conditioning import conditioning_combine
        c1 = _make_cond()
        c2 = _make_cond()
        result, = conditioning_combine(c1, c2)
        assert len(result) == 2

    def test_preserves_entries(self):
        from serenityflow.nodes.conditioning import conditioning_combine
        c1 = _make_cond()
        c2 = _make_cond(dim=1024)
        result, = conditioning_combine(c1, c2)
        assert result[0]["cross_attn"].shape[-1] == 768
        assert result[1]["cross_attn"].shape[-1] == 1024


class TestConditioningZeroOut:
    def test_zeros_cross_attn(self):
        from serenityflow.nodes.conditioning import conditioning_zero_out
        cond = _make_cond()
        result, = conditioning_zero_out(cond)
        assert torch.all(result[0]["cross_attn"] == 0)

    def test_zeros_pooled_output(self):
        from serenityflow.nodes.conditioning import conditioning_zero_out
        cond = _make_cond(pooled_dim=768)
        result, = conditioning_zero_out(cond)
        assert torch.all(result[0]["pooled_output"] == 0)


class TestConditioningSetTimestepRange:
    def test_range_set(self):
        from serenityflow.nodes.conditioning import conditioning_set_timestep_range
        cond = _make_cond()
        result, = conditioning_set_timestep_range(cond, start=0.2, end=0.8)
        assert result[0]["timestep_start"] == 0.2
        assert result[0]["timestep_end"] == 0.8


class TestConditioningSetAreaPercentage:
    def test_percentage_area(self):
        from serenityflow.nodes.conditioning import conditioning_set_area_percentage
        cond = _make_cond()
        result, = conditioning_set_area_percentage(cond, width=0.5, height=0.5, x=0.25, y=0.25, strength=1.0)
        assert result[0]["area"] == ("percentage", 0.5, 0.5, 0.25, 0.25)


class TestFluxGuidance:
    def test_guidance_set(self):
        from serenityflow.nodes.conditioning import flux_guidance
        cond = _make_cond()
        result, = flux_guidance(cond, guidance=7.5)
        assert result[0]["guidance"] == 7.5
