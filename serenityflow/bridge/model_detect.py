"""Model architecture detection from checkpoint file headers.

Reads tensor names from safetensors headers (or torch state dicts) and matches
against known architecture signatures. No tensor data loaded.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

log = logging.getLogger(__name__)


@dataclass
class ArchSignature:
    """Signature for identifying a model architecture from tensor key names."""
    name: str
    required_keys: list[str]
    forbidden_keys: list[str] = field(default_factory=list)
    config_extract: Optional[Callable] = None
    serenity_pipeline: str = ""

    def match(self, tensor_names: set[str]) -> tuple[bool, float, str]:
        """Check if tensor names match this signature.
        Returns (matched, confidence, explanation).
        """
        # Check forbidden keys first
        for fk in self.forbidden_keys:
            if any(fk in name for name in tensor_names):
                return False, 0.0, f"forbidden key '{fk}' present"

        # Check required keys
        found = []
        for rk in self.required_keys:
            if any(rk in name for name in tensor_names):
                found.append(rk)

        if len(found) == len(self.required_keys):
            confidence = len(found) / max(len(self.required_keys), 1)
            return True, confidence, f"matched {self.name}: found {found}"

        missing = set(self.required_keys) - set(found)
        return False, 0.0, f"missing keys: {missing}"


# ─── Config extractors ───

def _flux_config(tensor_names: list[str]) -> dict:
    double_count = sum(1 for k in tensor_names
                       if k.startswith("double_blocks.") and k.endswith(".img_attn.qkv.weight"))
    single_count = sum(1 for k in tensor_names
                       if k.startswith("single_blocks.") and k.endswith(".linear1.weight"))
    return {"double_blocks": double_count, "single_blocks": single_count}


def _sd3_config(tensor_names: list[str]) -> dict:
    joint_count = sum(1 for k in tensor_names
                      if k.startswith("joint_blocks.") and k.endswith(".attn.qkv.weight"))
    return {"joint_blocks": joint_count}


def _sdxl_config(tensor_names: list[str]) -> dict:
    return {"variant": "sdxl"}


def _sd15_config(tensor_names: list[str]) -> dict:
    return {"variant": "sd15"}


def _ltxv_config(tensor_names: list[str]) -> dict:
    block_count = sum(1 for k in tensor_names
                      if k.startswith("transformer_blocks.") and k.endswith(".attn1.to_q.weight"))
    return {"transformer_blocks": block_count}


def _wan_config(tensor_names: list[str]) -> dict:
    block_count = sum(1 for k in tensor_names
                      if k.startswith("blocks.") and k.endswith(".self_attn.q.weight"))
    return {"blocks": block_count}


def _klein_config(tensor_names: list[str]) -> dict:
    double_count = sum(1 for k in tensor_names
                       if k.startswith("double_blocks.") and k.endswith(".img_attn.qkv.weight"))
    single_count = sum(1 for k in tensor_names
                       if k.startswith("single_blocks.") and k.endswith(".linear1.weight"))
    return {"double_blocks": double_count, "single_blocks": single_count}


def _zimage_config(tensor_names: list[str]) -> dict:
    layer_count = sum(1 for k in tensor_names
                      if k.startswith("layers.") and k.endswith(".attention.q.weight"))
    return {"layers": layer_count}


# ─── Signature table ───
# Order matters: more specific signatures first to avoid false positives.

SIGNATURES = [
    ArchSignature(
        name="flux",
        required_keys=[
            "double_blocks.0.img_attn.qkv.weight",
            "single_blocks.0.linear1.weight",
        ],
        forbidden_keys=["joint_blocks.0.context_block.attn.qkv.weight"],
        config_extract=_flux_config,
        serenity_pipeline="flux",
    ),
    ArchSignature(
        name="sd3",
        required_keys=[
            "joint_blocks.0.context_block.attn.qkv.weight",
            "x_embedder.proj.weight",
        ],
        forbidden_keys=["double_blocks.0.img_attn.qkv.weight"],
        config_extract=_sd3_config,
        serenity_pipeline="sd3",
    ),
    ArchSignature(
        name="sdxl",
        required_keys=[
            "input_blocks.0.0.weight",
            "input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight",
        ],
        config_extract=_sdxl_config,
        serenity_pipeline="sdxl",
    ),
    ArchSignature(
        name="sd15",
        required_keys=[
            "input_blocks.0.0.weight",
        ],
        forbidden_keys=[
            "input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight",
        ],
        config_extract=_sd15_config,
        serenity_pipeline="sd15",
    ),
    ArchSignature(
        name="ltxv",
        required_keys=[
            "transformer_blocks.0.attn1.to_q.weight",
            "patchify_proj.weight",
        ],
        config_extract=_ltxv_config,
        serenity_pipeline="ltxv",
    ),
    ArchSignature(
        name="wan",
        required_keys=[
            "blocks.0.self_attn.q.weight",
            "patch_embedding.proj.weight",
        ],
        config_extract=_wan_config,
        serenity_pipeline="wan",
    ),
    ArchSignature(
        name="zimage",
        required_keys=[
            "layers.0.attention.q.weight",
            "cap_embedder.1.weight",
        ],
        config_extract=_zimage_config,
        serenity_pipeline="zimage",
    ),
    # Klein last — shares double_blocks with Flux but lacks single_blocks constraint
    ArchSignature(
        name="klein",
        required_keys=[
            "double_blocks.0.img_attn.qkv.weight",
        ],
        config_extract=_klein_config,
        serenity_pipeline="klein",
    ),
]


def detect_architecture(path: str) -> tuple[str, dict, float, str]:
    """Detect model architecture from file header. No tensor data loaded.

    Returns (pipeline_name, config_dict, confidence, explanation).
    Raises RuntimeError if no signature matches.
    """
    tensor_names = read_tensor_names(path)
    name_set = set(tensor_names)

    best = None
    best_confidence = 0.0

    for sig in SIGNATURES:
        matched, confidence, explanation = sig.match(name_set)
        if matched and confidence > best_confidence:
            config = sig.config_extract(tensor_names) if sig.config_extract else {}
            best = (sig.serenity_pipeline, config, confidence, explanation)
            best_confidence = confidence

    if best is None:
        sample_keys = sorted(tensor_names)[:20]
        raise RuntimeError(
            f"Unknown model architecture at {path}. "
            f"Sample keys: {sample_keys}. "
            f"No signature matched."
        )
    return best


def read_tensor_names(path: str) -> list[str]:
    """Read tensor names from file header without loading tensor data."""
    ext = os.path.splitext(path)[1].lower()

    if ext in (".safetensors",):
        # Try serenity-safetensors first
        try:
            from serenity_safetensors import probe_model
            info = probe_model(path)
            return info.tensor_names
        except (ImportError, AttributeError):
            pass

        # Fallback: read safetensors header directly
        with open(path, "rb") as f:
            header_size = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(header_size))
            return [k for k in header.keys() if k != "__metadata__"]

    elif ext in (".gguf",):
        try:
            from serenity_safetensors import load_gguf_index
            index = load_gguf_index(path)
            return [t.name for t in index.tensors]
        except ImportError:
            raise RuntimeError(
                f"GGUF support requires serenity-safetensors: {path}"
            )

    else:
        # .pt, .ckpt, .bin — must load to inspect
        import torch
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict):
            if "state_dict" in sd:
                sd = sd["state_dict"]
            elif "model" in sd:
                sd = sd["model"]
        return list(sd.keys())


__all__ = [
    "ArchSignature",
    "SIGNATURES",
    "detect_architecture",
    "read_tensor_names",
]
