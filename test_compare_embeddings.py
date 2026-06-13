#!/usr/bin/env python3
"""Compare Stagehand vs official CPU text encoder output."""
import os, gc, torch, logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

from serenityflow.bridge.ltxv import (
    _patch_ltx_gemma_transformers_compat,
    _resolve_ltxv_gemma_root,
    _load_ltx_text_encoder_with_fallback,
    _encode_ltx_prompts_with_stagehand,
    load_ltxv_model,
)
from ltx_pipelines.utils import ModelLedger
from ltx_core.text_encoders.gemma import encode_text

CHECKPOINT = os.path.expanduser("~/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors")
gemma_root = _resolve_ltxv_gemma_root("", is_fp8=False)
device = torch.device("cuda")
prompt = "A red sports car driving fast"

# Method 1: Stagehand path (what we use)
_patch_ltx_gemma_transformers_compat()
model = load_ltxv_model(checkpoint_path=CHECKPOINT, gemma_path="", dtype="bfloat16", backend="auto")
ledger = model.model_ledger

text_encoder, _ = _load_ltx_text_encoder_with_fallback(ledger, bind_text_only_precompute=True)
stg_outputs = _encode_ltx_prompts_with_stagehand(text_encoder, (prompt,), device=device, gemma_root=gemma_root)
v_stg = stg_outputs[0].video_encoding
a_stg = stg_outputs[0].audio_encoding

print(f"Stagehand: video={v_stg.shape}, mean={v_stg.float().mean():.4f}, std={v_stg.float().std():.4f}")

# Method 2: Official direct path (on CPU, same text encoder instance)
# The text encoder is still loaded - use it directly via the official forward()
print("\nRunning official text_encoder(prompt) on CPU...")
with torch.no_grad():
    official_out = text_encoder(prompt)
v_off = official_out.video_encoding
a_off = official_out.audio_encoding

print(f"Official:  video={v_off.shape}, mean={v_off.float().mean():.4f}, std={v_off.float().std():.4f}")

# Compare
print(f"\n=== COMPARISON ===")
print(f"Shape match: {v_stg.shape == v_off.shape}")
if v_stg.shape == v_off.shape:
    diff = (v_stg.cpu().float() - v_off.cpu().float()).abs()
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    cosine = torch.nn.functional.cosine_similarity(v_stg.cpu().float().flatten(), v_off.cpu().float().flatten(), dim=0)
    print(f"Cosine similarity: {cosine:.6f}")
else:
    print(f"SHAPE MISMATCH: stagehand={v_stg.shape} vs official={v_off.shape}")
