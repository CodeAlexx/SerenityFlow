#!/usr/bin/env python3
"""Compare text encoder outputs between patched and original precompute paths."""
import os, sys, torch, logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

# Load model
from serenityflow.bridge.ltxv import load_ltxv_model, _patch_ltx_gemma_transformers_compat
from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder

CHECKPOINT = os.path.expanduser("~/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors")

# First, let's check what the feature extractor expects
_patch_ltx_gemma_transformers_compat()

model = load_ltxv_model(
    checkpoint_path=CHECKPOINT,
    gemma_path="",
    dtype="bfloat16",
    backend="auto",
)

# Access the text encoder from the ledger
ledger = model.model_ledger
print(f"Ledger type: {type(ledger)}")

# Let's check what version of feature extractor is used
from ltx_core.text_encoders.gemma.feature_extractor import FeatureExtractorV1, FeatureExtractorV2

# Build the text encoder to inspect it
text_encoder = ledger.text_encoder()
print(f"Text encoder type: {type(text_encoder)}")
print(f"Feature extractor type: {type(text_encoder.feature_extractor)}")

if isinstance(text_encoder.feature_extractor, FeatureExtractorV1):
    print("Using FeatureExtractorV1 (19B)")
    print(f"  aggregate_embed: {text_encoder.feature_extractor.aggregate_embed}")
elif isinstance(text_encoder.feature_extractor, FeatureExtractorV2):
    print("Using FeatureExtractorV2 (20B)")
    print(f"  video_aggregate_embed: {text_encoder.feature_extractor.video_aggregate_embed}")
    print(f"  embedding_dim: {text_encoder.feature_extractor.embedding_dim}")

print(f"\nEmbeddings processor type: {type(text_encoder.embeddings_processor)}")
print(f"Model type: {type(text_encoder.model)}")

# Check what model.device is
print(f"Model device: {text_encoder.model.device if hasattr(text_encoder.model, 'device') else 'N/A'}")

# Now test both paths
prompt = "A busy Tokyo street at night"

# Path 1: Original precompute (self.model)
print("\n=== Testing ORIGINAL precompute (self.model) ===")
try:
    token_pairs = text_encoder.tokenizer.tokenize_with_weights(prompt)["gemma"]
    input_ids = torch.tensor([[t[0] for t in token_pairs]], device=text_encoder.model.device)
    attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=text_encoder.model.device)

    print(f"input_ids shape: {input_ids.shape}, device: {input_ids.device}")
    print(f"attention_mask shape: {attention_mask.shape}")
    print(f"attention_mask values: min={attention_mask.min()}, max={attention_mask.max()}, dtype={attention_mask.dtype}")

    with torch.no_grad():
        outputs_orig = text_encoder.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    hs_orig = outputs_orig.hidden_states
    print(f"Original hidden states: {len(hs_orig)} layers, shape={hs_orig[0].shape}, dtype={hs_orig[0].dtype}")
    print(f"  Layer 0 stats: mean={hs_orig[0].float().mean():.6f}, std={hs_orig[0].float().std():.6f}")
    print(f"  Layer -1 stats: mean={hs_orig[-1].float().mean():.6f}, std={hs_orig[-1].float().std():.6f}")
except Exception as e:
    print(f"Original path failed: {e}")
    hs_orig = None

# Path 2: Patched precompute (language_model)
print("\n=== Testing PATCHED precompute (language_model) ===")
try:
    language_model = getattr(getattr(text_encoder.model, "model", None), "language_model", None)
    print(f"language_model type: {type(language_model)}")

    with torch.no_grad():
        outputs_patched = language_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    hs_patched = outputs_patched.hidden_states
    print(f"Patched hidden states: {len(hs_patched)} layers, shape={hs_patched[0].shape}, dtype={hs_patched[0].dtype}")
    print(f"  Layer 0 stats: mean={hs_patched[0].float().mean():.6f}, std={hs_patched[0].float().std():.6f}")
    print(f"  Layer -1 stats: mean={hs_patched[-1].float().mean():.6f}, std={hs_patched[-1].float().std():.6f}")
except Exception as e:
    print(f"Patched path failed: {e}")
    hs_patched = None

# Compare if both succeeded
if hs_orig is not None and hs_patched is not None:
    print("\n=== COMPARISON ===")
    print(f"Original layers: {len(hs_orig)}, Patched layers: {len(hs_patched)}")

    if len(hs_orig) != len(hs_patched):
        print("!!! DIFFERENT NUMBER OF HIDDEN STATES !!!")

    for i in [0, len(hs_orig)//2, -1]:
        diff = (hs_orig[i].float() - hs_patched[i].float()).abs()
        print(f"  Layer {i}: max_diff={diff.max():.8f}, mean_diff={diff.mean():.8f}")

    # Test feature extraction on both
    print("\n=== Feature extraction comparison ===")
    fe = text_encoder.feature_extractor
    with torch.no_grad():
        v_orig, a_orig = fe(hs_orig, attention_mask, "left")
        v_patched, a_patched = fe(hs_patched, attention_mask, "left")

    v_diff = (v_orig.float() - v_patched.float()).abs()
    print(f"Video features diff: max={v_diff.max():.8f}, mean={v_diff.mean():.8f}")
    print(f"Video features orig: mean={v_orig.float().mean():.6f}, std={v_orig.float().std():.6f}")
    print(f"Video features patched: mean={v_patched.float().mean():.6f}, std={v_patched.float().std():.6f}")
