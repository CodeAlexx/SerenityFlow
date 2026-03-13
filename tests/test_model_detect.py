"""Tests for model architecture detection."""
from __future__ import annotations

import pytest

from serenityflow.bridge.model_detect import (
    ArchSignature,
    SIGNATURES,
    detect_architecture,
)


# ─── Synthetic key sets for each architecture ───

FLUX_KEYS = {
    "double_blocks.0.img_attn.qkv.weight",
    "double_blocks.0.img_attn.proj.weight",
    "double_blocks.1.img_attn.qkv.weight",
    "single_blocks.0.linear1.weight",
    "single_blocks.0.linear2.weight",
    "single_blocks.1.linear1.weight",
    "img_in.weight",
    "time_in.in_layer.weight",
    "txt_in.weight",
}

SD3_KEYS = {
    "joint_blocks.0.context_block.attn.qkv.weight",
    "joint_blocks.0.x_block.attn.qkv.weight",
    "joint_blocks.1.context_block.attn.qkv.weight",
    "x_embedder.proj.weight",
    "t_embedder.mlp.0.weight",
    "context_embedder.weight",
}

SDXL_KEYS = {
    "input_blocks.0.0.weight",
    "input_blocks.0.0.bias",
    "input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight",
    "input_blocks.4.1.transformer_blocks.0.attn1.to_k.weight",
    "middle_block.1.transformer_blocks.0.attn1.to_q.weight",
    "output_blocks.3.1.conv.weight",
}

SD15_KEYS = {
    "input_blocks.0.0.weight",
    "input_blocks.0.0.bias",
    "input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
    "middle_block.1.transformer_blocks.0.attn1.to_q.weight",
    "output_blocks.3.1.conv.weight",
}

LTXV_KEYS = {
    "transformer_blocks.0.attn1.to_q.weight",
    "transformer_blocks.0.attn1.to_k.weight",
    "transformer_blocks.1.attn1.to_q.weight",
    "patchify_proj.weight",
    "patchify_proj.bias",
}

WAN_KEYS = {
    "blocks.0.self_attn.q.weight",
    "blocks.0.self_attn.k.weight",
    "blocks.1.self_attn.q.weight",
    "patch_embedding.proj.weight",
    "patch_embedding.proj.bias",
}

KLEIN_KEYS = {
    "double_blocks.0.img_attn.qkv.weight",
    "double_blocks.0.img_attn.proj.weight",
    "double_blocks.1.img_attn.qkv.weight",
    "img_in.weight",
    "time_in.in_layer.weight",
}

ZIMAGE_KEYS = {
    "layers.0.attention.q.weight",
    "layers.0.attention.k.weight",
    "layers.1.attention.q.weight",
    "cap_embedder.1.weight",
    "cap_embedder.1.bias",
}


class TestArchSignature:
    def test_match_all_required(self):
        sig = ArchSignature(
            name="test",
            required_keys=["a.weight", "b.weight"],
            serenity_pipeline="test",
        )
        matched, conf, _ = sig.match({"a.weight", "b.weight", "c.weight"})
        assert matched
        assert conf == 1.0

    def test_no_match_missing_key(self):
        sig = ArchSignature(
            name="test",
            required_keys=["a.weight", "b.weight"],
            serenity_pipeline="test",
        )
        matched, _, _ = sig.match({"a.weight", "c.weight"})
        assert not matched

    def test_forbidden_key_blocks(self):
        sig = ArchSignature(
            name="test",
            required_keys=["a.weight"],
            forbidden_keys=["forbidden.weight"],
            serenity_pipeline="test",
        )
        matched, _, explanation = sig.match({"a.weight", "forbidden.weight"})
        assert not matched
        assert "forbidden" in explanation

    def test_partial_key_matching(self):
        """Required keys use 'in' matching, not exact match."""
        sig = ArchSignature(
            name="test",
            required_keys=["attn.qkv.weight"],
            serenity_pipeline="test",
        )
        matched, _, _ = sig.match({"blocks.0.attn.qkv.weight"})
        assert matched


class TestSignatureTable:
    def test_flux_detected(self):
        sig = next(s for s in SIGNATURES if s.name == "flux")
        matched, conf, _ = sig.match(FLUX_KEYS)
        assert matched
        assert conf > 0

    def test_sd3_detected(self):
        sig = next(s for s in SIGNATURES if s.name == "sd3")
        matched, _, _ = sig.match(SD3_KEYS)
        assert matched

    def test_sdxl_detected(self):
        sig = next(s for s in SIGNATURES if s.name == "sdxl")
        matched, _, _ = sig.match(SDXL_KEYS)
        assert matched

    def test_sd15_detected(self):
        sig = next(s for s in SIGNATURES if s.name == "sd15")
        matched, _, _ = sig.match(SD15_KEYS)
        assert matched

    def test_ltxv_detected(self):
        sig = next(s for s in SIGNATURES if s.name == "ltxv")
        matched, _, _ = sig.match(LTXV_KEYS)
        assert matched

    def test_wan_detected(self):
        sig = next(s for s in SIGNATURES if s.name == "wan")
        matched, _, _ = sig.match(WAN_KEYS)
        assert matched

    def test_klein_detected(self):
        sig = next(s for s in SIGNATURES if s.name == "klein")
        matched, _, _ = sig.match(KLEIN_KEYS)
        assert matched

    def test_zimage_detected(self):
        sig = next(s for s in SIGNATURES if s.name == "zimage")
        matched, _, _ = sig.match(ZIMAGE_KEYS)
        assert matched

    def test_flux_not_sd3(self):
        """FLUX keys must NOT match SD3 signature."""
        sig = next(s for s in SIGNATURES if s.name == "sd3")
        matched, _, _ = sig.match(FLUX_KEYS)
        assert not matched

    def test_sd3_not_flux(self):
        """SD3 keys must NOT match FLUX signature."""
        sig = next(s for s in SIGNATURES if s.name == "flux")
        matched, _, _ = sig.match(SD3_KEYS)
        assert not matched

    def test_sdxl_not_sd15(self):
        """SDXL keys must NOT match SD15 (forbidden key blocks it)."""
        sig = next(s for s in SIGNATURES if s.name == "sd15")
        matched, _, _ = sig.match(SDXL_KEYS)
        assert not matched

    def test_sd15_not_sdxl(self):
        """SD15 keys must NOT match SDXL (missing transformer blocks key)."""
        sig = next(s for s in SIGNATURES if s.name == "sdxl")
        matched, _, _ = sig.match(SD15_KEYS)
        assert not matched


class TestConfigExtract:
    def test_flux_config(self):
        sig = next(s for s in SIGNATURES if s.name == "flux")
        config = sig.config_extract(list(FLUX_KEYS))
        assert "double_blocks" in config
        assert "single_blocks" in config
        assert config["double_blocks"] == 2
        assert config["single_blocks"] == 2

    def test_sd3_config(self):
        sig = next(s for s in SIGNATURES if s.name == "sd3")
        config = sig.config_extract(list(SD3_KEYS))
        assert "joint_blocks" in config

    def test_wan_config(self):
        sig = next(s for s in SIGNATURES if s.name == "wan")
        config = sig.config_extract(list(WAN_KEYS))
        assert "blocks" in config
        assert config["blocks"] == 2
