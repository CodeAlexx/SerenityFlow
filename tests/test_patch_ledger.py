from __future__ import annotations
import pytest
from serenityflow.core.patch_ledger import PatchLedger, PatchEntry


def _entry(source="lora_a.safetensors", keys=None, strength=1.0, ptype="lora"):
    return PatchEntry(ptype, source, strength, keys or {"block.0.attn.weight"}, {})


class TestPatchLedgerBasics:
    def test_new_ledger(self):
        pl = PatchLedger()
        assert pl.epoch == 0
        assert pl.patches == []
        fp = pl.fingerprint()
        assert isinstance(fp, str) and len(fp) == 16

    def test_add_patch_increments_epoch(self):
        pl = PatchLedger()
        fp_before = pl.fingerprint()
        pl.add_patch(_entry())
        assert pl.epoch == 1
        assert pl.fingerprint() != fp_before

    def test_order_matters_for_fingerprint(self):
        pl1 = PatchLedger()
        pl1.add_patch(_entry("a.safetensors"))
        pl1.add_patch(_entry("b.safetensors"))

        pl2 = PatchLedger()
        pl2.add_patch(_entry("b.safetensors"))
        pl2.add_patch(_entry("a.safetensors"))

        assert pl1.fingerprint() != pl2.fingerprint()

    def test_remove_patch(self):
        pl = PatchLedger()
        pl.add_patch(_entry("a.safetensors"))
        fp_before = pl.fingerprint()
        epoch_before = pl.epoch
        pl.remove_patch("a.safetensors")
        assert pl.epoch == epoch_before + 1
        assert pl.fingerprint() != fp_before
        assert len(pl.patches) == 0

    def test_remove_nonexistent(self):
        pl = PatchLedger()
        pl.add_patch(_entry())
        epoch_before = pl.epoch
        fp_before = pl.fingerprint()
        pl.remove_patch("nonexistent.safetensors")
        assert pl.epoch == epoch_before
        assert pl.fingerprint() == fp_before

    def test_clear_nonempty(self):
        pl = PatchLedger()
        pl.add_patch(_entry())
        epoch_before = pl.epoch
        pl.clear()
        assert pl.epoch == epoch_before + 1
        assert len(pl.patches) == 0

    def test_clear_empty(self):
        pl = PatchLedger()
        pl.clear()
        assert pl.epoch == 0


class TestDirtyTracking:
    def test_new_ledger_all_dirty(self):
        pl = PatchLedger()
        assert pl.block_is_dirty("block.0")
        assert pl.block_is_dirty("block.99")

    def test_mark_clean(self):
        pl = PatchLedger()
        pl.mark_block_clean("block.0")
        assert not pl.block_is_dirty("block.0")

    def test_add_patch_makes_dirty_again(self):
        pl = PatchLedger()
        pl.mark_block_clean("block.0")
        assert not pl.block_is_dirty("block.0")
        pl.add_patch(_entry())
        assert pl.block_is_dirty("block.0")


class TestPatchesForBlock:
    def test_matching(self):
        pl = PatchLedger()
        pl.add_patch(_entry(keys={"block.0.attn.weight", "block.0.mlp.weight"}))
        pl.add_patch(_entry(source="b.safetensors", keys={"block.1.attn.weight"}))
        result = pl.patches_for_block("block.0")
        assert len(result) == 1
        assert result[0].source == "lora_a.safetensors"

    def test_prefix_matching(self):
        pl = PatchLedger()
        pl.add_patch(_entry(keys={"transformer.blocks.0.attn.weight"}))
        result = pl.patches_for_block("transformer.blocks.0")
        assert len(result) == 1


class TestClone:
    def test_independent_epoch(self):
        pl = PatchLedger()
        pl.add_patch(_entry())
        clone = pl.clone()
        clone.add_patch(_entry("extra.safetensors"))
        assert clone.epoch == pl.epoch + 1

    def test_all_blocks_dirty(self):
        pl = PatchLedger()
        pl.mark_block_clean("block.0")
        clone = pl.clone()
        assert clone.block_is_dirty("block.0")

    def test_shared_entries(self):
        pl = PatchLedger()
        e = _entry()
        pl.add_patch(e)
        clone = pl.clone()
        assert clone.patches[0] is e

    def test_mutation_independence(self):
        pl = PatchLedger()
        pl.add_patch(_entry())
        clone = pl.clone()
        clone.add_patch(_entry("extra.safetensors"))
        assert len(pl.patches) == 1
        assert len(clone.patches) == 2


class TestFingerprint:
    def test_deterministic(self):
        def build():
            pl = PatchLedger()
            pl.add_patch(_entry("a.safetensors", strength=0.8))
            pl.add_patch(_entry("b.safetensors", strength=1.0))
            return pl.fingerprint()
        assert build() == build()

    def test_cache_invalidation(self):
        pl = PatchLedger()
        pl.add_patch(_entry())
        fp1 = pl.fingerprint()
        assert pl._fingerprint_cache == fp1
        pl.add_patch(_entry("extra.safetensors"))
        assert pl._fingerprint_cache is None
        fp2 = pl.fingerprint()
        assert fp1 != fp2
