#!/usr/bin/env python3
"""Minimal bf16 generation test."""
import os, torch, logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

from serenityflow.bridge.ltxv import load_ltxv_model, sample_ltxv

CHECKPOINT = os.path.expanduser("~/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors")

model = load_ltxv_model(
    checkpoint_path=CHECKPOINT,
    gemma_path="",
    dtype="bfloat16",
    backend="auto",
)

result = sample_ltxv(
    model=model,
    prompt="A busy Tokyo street at night with neon signs and people walking",
    width=512,
    height=320,
    num_frames=81,
    steps=25,
    guidance_scale=3.5,
    stg_scale=0.0,
    seed=42,
    frame_rate=25.0,
)

video = result["video"]
print(f"Video shape: {video.shape}, dtype: {video.dtype}")

import imageio.v3 as iio
outpath = "/home/alex/serenityflow-v2/output/test_bf16_verify.mp4"
os.makedirs(os.path.dirname(outpath), exist_ok=True)
if video.dim() == 4:
    # [F, H, W, C] uint8
    frames = video.numpy()
elif video.dim() == 5:
    # [B, C, F, H, W] float
    frames = video.squeeze(0).permute(1, 2, 3, 0).clamp(0, 1).mul(255).byte().numpy()
else:
    raise ValueError(f"Unexpected video shape: {video.shape}")
iio.imwrite(outpath, frames, fps=25, codec="libx264", plugin="pyav")
print(f"Saved to {outpath}")
