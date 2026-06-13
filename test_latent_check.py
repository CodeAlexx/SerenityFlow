#!/usr/bin/env python3
"""Check stage 1 latent stats to see if denoiser output is sane."""
import os, torch, logging, gc
logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

from serenityflow.bridge.ltxv import load_ltxv_model, sample_ltxv

# Monkey-patch to capture stage 1 latent
import serenityflow.bridge.ltxv as _ltxv_mod
_orig_info = _ltxv_mod.logger.info
_s1_latent = [None]

def _capturing_info(msg, *args):
    _orig_info(msg, *args)
    if "Stage 1 complete" in str(msg):
        # We can't capture the latent this way, but we can check the output video
        pass

CHECKPOINT = os.path.expanduser("~/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors")
model = load_ltxv_model(checkpoint_path=CHECKPOINT, gemma_path="", dtype="bfloat16", backend="auto")

# Run with fewer frames for speed
result = sample_ltxv(
    model=model,
    prompt="A red sports car driving on a highway",
    width=512,
    height=320,
    num_frames=41,
    steps=8,
    guidance_scale=1.0,
    stg_scale=0.0,
    seed=123,
    frame_rate=25.0,
)

video = result["video"]
print(f"\nVideo: shape={video.shape}, dtype={video.dtype}")
print(f"  mean={video.float().mean():.2f}, std={video.float().std():.2f}")
print(f"  min={video.min()}, max={video.max()}")

# Check if all frames are similar (would indicate denoiser not working)
if video.dim() == 4:  # [F, H, W, C]
    frame_means = video.float().mean(dim=(1,2,3))
    print(f"  per-frame means: {frame_means[:5].tolist()}...")
    print(f"  frame variance: {frame_means.var():.4f}")

    # Check color distribution
    r = video[:,:,:,0].float().mean()
    g = video[:,:,:,1].float().mean()
    b = video[:,:,:,2].float().mean()
    print(f"  avg RGB: ({r:.1f}, {g:.1f}, {b:.1f})")
