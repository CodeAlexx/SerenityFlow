#!/usr/bin/env python3
"""Compare actual embedding VALUES between Stagehand and official precompute."""
import os, gc, torch, logging
logging.basicConfig(level=logging.WARNING)

from serenityflow.bridge.ltxv import (
    _patch_ltx_gemma_transformers_compat,
    _resolve_ltxv_gemma_root,
    _load_ltx_text_encoder_with_fallback,
    _encode_ltx_prompts_with_stagehand,
    _ltx_finalize_gemma_text_encoder_output,
    load_ltxv_model,
)

CHECKPOINT = os.path.expanduser("~/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors")
device = torch.device("cuda")
prompt = "A red car"

_patch_ltx_gemma_transformers_compat()
model = load_ltxv_model(checkpoint_path=CHECKPOINT, gemma_path="", dtype="bfloat16", backend="auto")
ledger = model.model_ledger

text_encoder, _ = _load_ltx_text_encoder_with_fallback(ledger, bind_text_only_precompute=True)

# Method 1: Stagehand
stg_out = _encode_ltx_prompts_with_stagehand(text_encoder, (prompt,), device=device, gemma_root=model.gemma_root_path)
v_stg = stg_out[0].video_encoding.cpu().float()

# Method 2: Official precompute on CPU (use ORIGINAL precompute, not patched)
# The patched precompute calls language_model - let's call the ORIGINAL base_encoder precompute
from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
orig_precompute = GemmaTextEncoder.precompute  # class method (may be patched)

# Actually just call the original precompute logic manually
token_pairs = text_encoder.tokenizer.tokenize_with_weights(prompt)["gemma"]
input_ids = torch.tensor([[t[0] for t in token_pairs]], device=text_encoder.model.device)
attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=text_encoder.model.device)

print(f"Tokens: {input_ids.shape[1]}, device={input_ids.device}")
print(f"Model device: {text_encoder.model.device}")

with torch.no_grad():
    # Call self.model() like official precompute
    outputs = text_encoder.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hs_official = outputs.hidden_states
    print(f"Official hidden_states: {len(hs_official)} layers, shape={hs_official[0].shape}")

    # Feature extraction + embeddings processor
    video_feats, audio_feats = text_encoder.feature_extractor(hs_official, attention_mask, "left")
    additive_mask = text_encoder._convert_to_additive_mask(attention_mask, video_feats.dtype)
    v_off, a_off, mask_off = text_encoder.embeddings_processor.create_embeddings(video_feats, audio_feats, additive_mask)
    v_off = v_off.cpu().float()

print(f"\nStagehand:  shape={v_stg.shape}, first 5 values: {v_stg[0, 0, :5].tolist()}")
print(f"Official:   shape={v_off.shape}, first 5 values: {v_off[0, 0, :5].tolist()}")

if v_stg.shape == v_off.shape:
    diff = (v_stg - v_off).abs()
    cos = torch.nn.functional.cosine_similarity(v_stg.flatten(), v_off.flatten(), dim=0)
    print(f"\nMax diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    print(f"Cosine similarity: {cos:.6f}")
    print(f"Are they identical? {torch.allclose(v_stg, v_off, atol=1e-3)}")
else:
    print(f"\nSHAPE MISMATCH! Stagehand={v_stg.shape} vs Official={v_off.shape}")
