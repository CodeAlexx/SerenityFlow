from __future__ import annotations
import torch
import pytest
from serenityflow.core.conditioning import (
    validate, normalize, merge, set_area, set_mask, set_timestep_range, zero_out,
)


def _cond(n=1, seq_len=77, dim=768, **extra_meta):
    entries = []
    for _ in range(n):
        t = torch.randn(1, seq_len, dim)
        meta = dict(extra_meta)
        entries.append((t, meta))
    return entries


class TestValidate:
    def test_valid(self):
        cond = _cond(2)
        assert validate(cond) == []

    def test_empty_list(self):
        assert validate([]) == []

    def test_not_list(self):
        errs = validate("nope")
        assert any("list" in e for e in errs)

    def test_entry_not_tuple(self):
        errs = validate(["bad"])
        assert len(errs) > 0

    def test_entry_wrong_length(self):
        errs = validate([(torch.randn(1, 77, 768),)])
        assert len(errs) > 0

    def test_first_not_tensor(self):
        errs = validate([("not_tensor", {})])
        assert any("Tensor" in e for e in errs)

    def test_second_not_dict(self):
        errs = validate([(torch.randn(1, 77, 768), "not_dict")])
        assert any("dict" in e for e in errs)

    def test_unknown_key_warning(self):
        cond = [(torch.randn(1, 77, 768), {"unknown_key": 42})]
        errs = validate(cond)
        assert any("unknown" in e.lower() for e in errs)

    def test_known_key_wrong_type(self):
        cond = [(torch.randn(1, 77, 768), {"strength": "not_a_number"})]
        errs = validate(cond)
        assert any("strength" in e for e in errs)

    def test_valid_with_known_keys(self):
        cond = [(torch.randn(1, 77, 768), {
            "pooled_output": torch.randn(1, 768),
            "strength": 0.8,
            "area": (0, 0, 64, 64),
        })]
        assert validate(cond) == []


class TestNormalize:
    def test_strips_unknown(self):
        cond = [(torch.randn(1, 77, 768), {"strength": 1.0, "garbage": True})]
        result = normalize(cond)
        assert "garbage" not in result[0][1]
        assert result[0][1]["strength"] == 1.0

    def test_preserves_tensor(self):
        t = torch.randn(1, 77, 768)
        cond = [(t, {"strength": 1.0})]
        result = normalize(cond)
        assert result[0][0] is t


class TestMerge:
    def test_concatenation(self):
        a = _cond(2)
        b = _cond(3)
        result = merge(a, b)
        assert len(result) == 5


class TestSetArea:
    def test_applies_to_all(self):
        cond = _cond(3)
        result = set_area(cond, (0, 0, 32, 32), strength=0.5)
        assert len(result) == 3
        for _, meta in result:
            assert meta["area"] == (0, 0, 32, 32)
            assert meta["strength"] == 0.5


class TestSetMask:
    def test_applies_to_all(self):
        mask = torch.rand(64, 64)
        cond = _cond(2)
        result = set_mask(cond, mask, strength=0.7)
        for _, meta in result:
            assert meta["mask"] is mask
            assert meta["mask_strength"] == 0.7


class TestSetTimestepRange:
    def test_applies_to_all(self):
        cond = _cond(2)
        result = set_timestep_range(cond, 0.2, 0.8)
        for _, meta in result:
            assert meta["timestep_start"] == 0.2
            assert meta["timestep_end"] == 0.8


class TestZeroOut:
    def test_zeros_tensor_preserves_meta(self):
        cond = [(torch.ones(1, 77, 768), {"strength": 1.0})]
        result = zero_out(cond)
        assert torch.all(result[0][0] == 0)
        assert result[0][1]["strength"] == 1.0
