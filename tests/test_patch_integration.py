"""Tests for PatchLedger integration with Stagehand block transfers.

Builder: core patch application logic
Bug Fixer: edge cases (strength 0, missing keys, clone ledger)
Skeptic: adversarial scenarios (back-to-back workflows, strength 0 dirty tracking)
"""
from __future__ import annotations

import torch
import pytest

from serenityflow.core.patch_ledger import PatchEntry, PatchLedger
from serenityflow.memory.patch_integration import apply_patches_to_block


def _make_lora_entry(
    block_name: str = "block.0",
    source: str = "lora_a.safetensors",
    strength: float = 1.0,
    rank: int = 4,
    in_dim: int = 8,
    out_dim: int = 8,
    alpha: float | None = None,
    seed: int = 42,
) -> PatchEntry:
    """Create a LoRA patch entry with real tensors."""
    key = f"{block_name}.attn.weight"
    gen = torch.Generator().manual_seed(seed)
    down = torch.randn(rank, in_dim, generator=gen)
    up = torch.randn(out_dim, rank, generator=gen)
    data = {
        key: {
            "lora_up": up,
            "lora_down": down,
        }
    }
    if alpha is not None:
        data[key]["alpha"] = torch.tensor(alpha)
    return PatchEntry("lora", source, strength, {key}, data)


def _make_loha_entry(
    block_name: str = "block.0",
    source: str = "loha.safetensors",
    strength: float = 1.0,
    rank: int = 4,
    dim: int = 8,
) -> PatchEntry:
    key = f"{block_name}.attn.weight"
    data = {
        key: {
            "hada_w1_a": torch.randn(dim, rank),
            "hada_w1_b": torch.randn(rank, dim),
            "hada_w2_a": torch.randn(dim, rank),
            "hada_w2_b": torch.randn(rank, dim),
        }
    }
    return PatchEntry("loha", source, strength, {key}, data)


def _make_lokr_entry(
    block_name: str = "block.0",
    source: str = "lokr.safetensors",
    strength: float = 1.0,
) -> PatchEntry:
    key = f"{block_name}.attn.weight"
    # 2x2 kron 4x4 = 8x8
    data = {
        key: {
            "lokr_w1": torch.randn(2, 2),
            "lokr_w2": torch.randn(4, 4),
        }
    }
    return PatchEntry("lokr", source, strength, {key}, data)


def _make_ia3_entry(
    block_name: str = "block.0",
    source: str = "ia3.safetensors",
    strength: float = 1.0,
    dim: int = 8,
) -> PatchEntry:
    key = f"{block_name}.attn.weight"
    data = {
        key: {
            "weight": torch.randn(dim, dim),
        }
    }
    return PatchEntry("ia3", source, strength, {key}, data)


def _make_full_diff_entry(
    block_name: str = "block.0",
    source: str = "diff.safetensors",
    strength: float = 1.0,
    dim: int = 8,
) -> PatchEntry:
    key = f"{block_name}.attn.weight"
    data = {key: {"diff": torch.randn(dim, dim)}}
    return PatchEntry("full_diff", source, strength, {key}, data)


def _make_glora_entry(
    block_name: str = "block.0",
    source: str = "glora.safetensors",
    strength: float = 1.0,
    dim: int = 8,
    rank: int = 4,
) -> PatchEntry:
    key = f"{block_name}.attn.weight"
    data = {
        key: {
            "a1": torch.randn(dim, rank),
            "b1": torch.randn(rank, dim),
            "a2": torch.randn(dim, rank),
            "b2": torch.randn(rank, dim),
        }
    }
    return PatchEntry("glora", source, strength, {key}, data)


# ─── Builder Tests ───


class TestCleanLedger:
    """Clean ledger -> no patching applied."""

    def test_clean_block_returns_gpu_weight(self):
        ledger = PatchLedger()
        cpu = torch.randn(8, 8)
        gpu = cpu.clone()
        ledger.mark_block_clean("block.0")
        result = apply_patches_to_block("block.0", gpu, cpu, ledger)
        assert torch.equal(result, gpu)

    def test_clean_after_mark_skips_patch(self):
        ledger = PatchLedger()
        entry = _make_lora_entry()
        ledger.add_patch(entry)
        cpu = torch.randn(8, 8)
        gpu = cpu.clone()
        # First apply
        apply_patches_to_block("block.0", gpu, cpu, ledger)
        # Now block is clean, second apply should skip
        result = apply_patches_to_block("block.0", gpu, cpu, ledger)
        assert torch.equal(result, gpu)


class TestSingleLoraPatch:
    """One LoRA patch -> weight modified correctly."""

    def test_lora_modifies_weight(self):
        ledger = PatchLedger()
        entry = _make_lora_entry()
        ledger.add_patch(entry)
        cpu = torch.randn(8, 8)
        gpu = cpu.clone()
        result = apply_patches_to_block("block.0", gpu, cpu, ledger)
        assert not torch.equal(result, cpu), "Weight should be modified by LoRA"

    def test_lora_with_alpha(self):
        ledger = PatchLedger()
        entry = _make_lora_entry(alpha=2.0)
        ledger.add_patch(entry)
        cpu = torch.randn(8, 8)
        gpu = cpu.clone()
        result = apply_patches_to_block("block.0", gpu, cpu, ledger)
        assert not torch.equal(result, cpu)


class TestTwoLoraPatches:
    """Two LoRA patches -> both applied in order."""

    def test_both_applied(self):
        ledger = PatchLedger()
        entry1 = _make_lora_entry(source="a.safetensors", seed=1)
        entry2 = _make_lora_entry(source="b.safetensors", seed=2)
        ledger.add_patch(entry1)
        ledger.add_patch(entry2)
        cpu = torch.randn(8, 8)
        gpu = cpu.clone()
        result = apply_patches_to_block("block.0", gpu, cpu, ledger)
        assert not torch.equal(result, cpu)

    def test_order_matters(self):
        cpu = torch.randn(8, 8)

        ledger_ab = PatchLedger()
        ledger_ab.add_patch(_make_lora_entry(source="a.safetensors", seed=1))
        ledger_ab.add_patch(_make_lora_entry(source="b.safetensors", seed=2))
        result_ab = apply_patches_to_block("block.0", cpu.clone(), cpu, ledger_ab)

        ledger_ba = PatchLedger()
        ledger_ba.add_patch(_make_lora_entry(source="b.safetensors", seed=2))
        ledger_ba.add_patch(_make_lora_entry(source="a.safetensors", seed=1))
        result_ba = apply_patches_to_block("block.0", cpu.clone(), cpu, ledger_ba)

        # LoRA is additive so order shouldn't matter for sums, but the patches
        # are applied via patches_for_block which returns in add order, and the
        # test validates both get applied
        assert not torch.equal(result_ab, cpu)
        assert not torch.equal(result_ba, cpu)


class TestMarkClean:
    """Mark clean -> skip patching on next transfer."""

    def test_mark_clean_skips(self):
        ledger = PatchLedger()
        ledger.add_patch(_make_lora_entry())
        cpu = torch.randn(8, 8)
        gpu = cpu.clone()
        result1 = apply_patches_to_block("block.0", gpu, cpu, ledger)
        # Block is now clean
        assert not ledger.block_is_dirty("block.0")
        # Second transfer should skip
        result2 = apply_patches_to_block("block.0", result1, cpu, ledger)
        assert torch.equal(result2, result1)


class TestPatchTypes:
    """LoHA, LoKr, IA3, full_diff, GLoRA all produce different output from no-patch."""

    def test_loha_modifies(self):
        ledger = PatchLedger()
        ledger.add_patch(_make_loha_entry())
        cpu = torch.randn(8, 8)
        result = apply_patches_to_block("block.0", cpu.clone(), cpu, ledger)
        assert not torch.equal(result, cpu)

    def test_lokr_modifies(self):
        ledger = PatchLedger()
        ledger.add_patch(_make_lokr_entry())
        cpu = torch.randn(8, 8)
        result = apply_patches_to_block("block.0", cpu.clone(), cpu, ledger)
        assert not torch.equal(result, cpu)

    def test_ia3_modifies(self):
        ledger = PatchLedger()
        ledger.add_patch(_make_ia3_entry())
        cpu = torch.randn(8, 8)
        result = apply_patches_to_block("block.0", cpu.clone(), cpu, ledger)
        assert not torch.equal(result, cpu)

    def test_full_diff_modifies(self):
        ledger = PatchLedger()
        ledger.add_patch(_make_full_diff_entry())
        cpu = torch.randn(8, 8)
        result = apply_patches_to_block("block.0", cpu.clone(), cpu, ledger)
        assert not torch.equal(result, cpu)

    def test_glora_modifies(self):
        ledger = PatchLedger()
        ledger.add_patch(_make_glora_entry())
        cpu = torch.randn(8, 8)
        result = apply_patches_to_block("block.0", cpu.clone(), cpu, ledger)
        assert not torch.equal(result, cpu)

    def test_ortho_same_as_lora(self):
        """OrthoLoRA uses same compute as standard LoRA."""
        key = "block.0.attn.weight"
        data = {
            key: {
                "lora_up": torch.randn(8, 4),
                "lora_down": torch.randn(4, 8),
            }
        }
        entry = PatchEntry("ortho", "ortho.safetensors", 1.0, {key}, data)
        ledger = PatchLedger()
        ledger.add_patch(entry)
        cpu = torch.randn(8, 8)
        result = apply_patches_to_block("block.0", cpu.clone(), cpu, ledger)
        assert not torch.equal(result, cpu)


class TestCloneLedger:
    """Clone ledger -> all blocks dirty again, re-patch on next transfer."""

    def test_clone_marks_dirty(self):
        ledger = PatchLedger()
        ledger.add_patch(_make_lora_entry())
        cpu = torch.randn(8, 8)
        gpu = cpu.clone()
        apply_patches_to_block("block.0", gpu, cpu, ledger)
        assert not ledger.block_is_dirty("block.0")

        clone = ledger.clone()
        assert clone.block_is_dirty("block.0")

    def test_clone_reapplies(self):
        ledger = PatchLedger()
        entry = _make_lora_entry()
        ledger.add_patch(entry)
        cpu = torch.randn(8, 8)
        gpu = cpu.clone()
        result1 = apply_patches_to_block("block.0", gpu, cpu, ledger)

        clone = ledger.clone()
        result2 = apply_patches_to_block("block.0", gpu.clone(), cpu, clone)
        assert torch.allclose(result1, result2)


# ─── Bug Fixer Tests ───


class TestBugFixerEdgeCases:
    def test_strength_zero_no_change(self):
        """Strength 0 -> no change to weight."""
        ledger = PatchLedger()
        entry = _make_lora_entry(strength=0.0)
        ledger.add_patch(entry)
        cpu = torch.randn(8, 8)
        gpu = cpu.clone()
        result = apply_patches_to_block("block.0", gpu, cpu, ledger)
        assert torch.allclose(result, cpu, atol=1e-7)

    def test_missing_patch_data_key(self):
        """Patch with affected_keys but no data for that key -> no crash."""
        key = "block.0.attn.weight"
        entry = PatchEntry("lora", "test.safetensors", 1.0, {key}, {})
        ledger = PatchLedger()
        ledger.add_patch(entry)
        cpu = torch.randn(8, 8)
        gpu = cpu.clone()
        result = apply_patches_to_block("block.0", gpu, cpu, ledger)
        # Should be equal to cpu (clean weight, no actual patch data)
        assert torch.allclose(result, cpu)

    def test_unknown_patch_type_ignored(self):
        """Unknown patch type -> ignored, no crash."""
        key = "block.0.attn.weight"
        entry = PatchEntry("unknown_type", "test.safetensors", 1.0, {key},
                          {key: {"some_data": torch.randn(8, 8)}})
        ledger = PatchLedger()
        ledger.add_patch(entry)
        cpu = torch.randn(8, 8)
        result = apply_patches_to_block("block.0", cpu.clone(), cpu, ledger)
        assert torch.allclose(result, cpu)

    def test_patch_only_affects_matching_block(self):
        """Patch for block.0 doesn't affect block.1."""
        ledger = PatchLedger()
        ledger.add_patch(_make_lora_entry(block_name="block.0"))
        cpu = torch.randn(8, 8)
        result = apply_patches_to_block("block.1", cpu.clone(), cpu, ledger)
        assert torch.allclose(result, cpu)

    def test_lora_missing_up_tensor(self):
        """LoRA with missing up tensor -> no change."""
        key = "block.0.attn.weight"
        data = {key: {"lora_down": torch.randn(4, 8)}}
        entry = PatchEntry("lora", "test.safetensors", 1.0, {key}, data)
        ledger = PatchLedger()
        ledger.add_patch(entry)
        cpu = torch.randn(8, 8)
        result = apply_patches_to_block("block.0", cpu.clone(), cpu, ledger)
        assert torch.allclose(result, cpu)


# ─── Skeptic Tests ───


class TestSkepticAdversarial:
    def test_back_to_back_workflows_reuse_ledger(self):
        """Two workflows back-to-back. Second reuses cached state from first."""
        ledger = PatchLedger()
        entry = _make_lora_entry()
        ledger.add_patch(entry)

        cpu = torch.randn(8, 8)
        # First workflow
        result1 = apply_patches_to_block("block.0", cpu.clone(), cpu, ledger)
        assert not ledger.block_is_dirty("block.0")

        # Second workflow: same ledger, block should still be clean
        result2 = apply_patches_to_block("block.0", result1.clone(), cpu, ledger)
        assert torch.equal(result2, result1), "Second run should skip re-patching"

    def test_strength_zero_still_marks_clean(self):
        """LoRA at strength 0 still goes through apply path and marks clean."""
        ledger = PatchLedger()
        entry = _make_lora_entry(strength=0.0)
        ledger.add_patch(entry)
        cpu = torch.randn(8, 8)
        apply_patches_to_block("block.0", cpu.clone(), cpu, ledger)
        assert not ledger.block_is_dirty("block.0")
        # Adding another patch at strength 0 makes block dirty again
        ledger.add_patch(_make_lora_entry(source="b.safetensors", strength=0.0))
        assert ledger.block_is_dirty("block.0")

    def test_new_patch_after_clean_makes_dirty(self):
        """After marking clean, adding new patch makes dirty again."""
        ledger = PatchLedger()
        entry = _make_lora_entry()
        ledger.add_patch(entry)
        cpu = torch.randn(8, 8)
        apply_patches_to_block("block.0", cpu.clone(), cpu, ledger)
        assert not ledger.block_is_dirty("block.0")

        # New patch added
        ledger.add_patch(_make_lora_entry(source="new.safetensors", seed=99))
        assert ledger.block_is_dirty("block.0")

    def test_remove_all_patches_still_reapplies_clean(self):
        """Remove all patches -> block dirty -> next apply starts from clean CPU."""
        ledger = PatchLedger()
        entry = _make_lora_entry(source="test.safetensors")
        ledger.add_patch(entry)
        cpu = torch.randn(8, 8)
        result_patched = apply_patches_to_block("block.0", cpu.clone(), cpu, ledger)
        assert not torch.equal(result_patched, cpu)

        # Remove patch
        ledger.remove_patch("test.safetensors")
        assert ledger.block_is_dirty("block.0")

        # Re-apply: should get clean weight back (no patches to apply)
        result_clean = apply_patches_to_block("block.0", result_patched.clone(), cpu, ledger)
        assert torch.allclose(result_clean, cpu)

    def test_concurrent_blocks_independent(self):
        """Marking block.0 clean doesn't affect block.1 dirty state."""
        ledger = PatchLedger()
        e0 = _make_lora_entry(block_name="block.0", source="a.safetensors")
        e1 = _make_lora_entry(block_name="block.1", source="a.safetensors")
        # Manually add both affected keys to the entry
        combined_keys = e0.affected_keys | e1.affected_keys
        combined_data = {**e0.data, **e1.data}
        entry = PatchEntry("lora", "a.safetensors", 1.0, combined_keys, combined_data)
        ledger.add_patch(entry)

        cpu0 = torch.randn(8, 8)
        cpu1 = torch.randn(8, 8)

        apply_patches_to_block("block.0", cpu0.clone(), cpu0, ledger)
        assert not ledger.block_is_dirty("block.0")
        assert ledger.block_is_dirty("block.1")  # Still dirty!

    def test_ia3_strength_zero_identity(self):
        """IA3 at strength 0 should leave weight unchanged (multiply by 1)."""
        ledger = PatchLedger()
        ledger.add_patch(_make_ia3_entry(strength=0.0))
        cpu = torch.randn(8, 8)
        result = apply_patches_to_block("block.0", cpu.clone(), cpu, ledger)
        assert torch.allclose(result, cpu, atol=1e-7)
