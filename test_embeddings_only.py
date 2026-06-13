#!/usr/bin/env python3
"""Just check the text embeddings - are they sane or garbage?"""
import os, torch, logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

from serenityflow.bridge.ltxv import (
    load_ltxv_model, _patch_ltx_gemma_transformers_compat,
    _load_ltx_text_encoder_with_fallback, _encode_ltx_prompts_with_stagehand,
)

CHECKPOINT = os.path.expanduser("~/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors")

model = load_ltxv_model(checkpoint_path=CHECKPOINT, gemma_path="", dtype="bfloat16", backend="auto")
ledger = model.model_ledger

text_encoder, te_device = _load_ltx_text_encoder_with_fallback(ledger, bind_text_only_precompute=True)

prompt = "A busy Tokyo street at night with neon signs"
device = torch.device("cuda")

outputs = _encode_ltx_prompts_with_stagehand(text_encoder, (prompt,), device=device, gemma_root=model.gemma_root_path)
out = outputs[0]

v = out.video_encoding
a = out.audio_encoding
m = out.attention_mask

print(f"video_encoding: shape={v.shape}, dtype={v.dtype}")
print(f"  mean={v.float().mean():.6f}, std={v.float().std():.6f}")
print(f"  min={v.float().min():.6f}, max={v.float().max():.6f}")
print(f"  has_nan={torch.isnan(v).any()}, has_inf={torch.isinf(v).any()}")
print(f"  zeros={(v == 0).sum().item()}/{v.numel()} ({100*(v==0).sum().item()/v.numel():.1f}%)")

if a is not None:
    print(f"audio_encoding: shape={a.shape}, dtype={a.dtype}")
    print(f"  mean={a.float().mean():.6f}, std={a.float().std():.6f}")
else:
    print("audio_encoding: None")

print(f"attention_mask: shape={m.shape}, sum={m.sum().item()}")
