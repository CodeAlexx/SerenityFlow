#!/usr/bin/env python3
"""Generate a 10-second video with FP8 LTX + GPTQ Gemma."""
import os
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

from serenityflow.bridge.serenity_api import load_ltxv_model, sample_ltxv

CHECKPOINT = os.path.expanduser("~/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors")

model = load_ltxv_model(
    checkpoint_path=CHECKPOINT,
    gemma_path="",  # let resolver pick GPTQ-4b automatically
    backend="auto",
)

result = sample_ltxv(
    model=model,
    prompt="A golden retriever running through a sunlit meadow with wildflowers, slow motion, cinematic, golden hour lighting",
    width=512,
    height=320,
    num_frames=249,  # ~10s at 25fps
    steps=8,
    guidance_scale=1.0,
    stg_scale=0.0,
    seed=42,
    frame_rate=25.0,
    mode="distilled",
)

video = result["video"]
print(f"Video shape: {video.shape}, dtype: {video.dtype}")

import imageio.v3 as iio
outdir = os.path.expanduser("~/serenity/output")
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, "golden_retriever_10s.mp4")
frames = video.squeeze(0).permute(1, 2, 3, 0).clamp(0, 1).mul(255).byte().numpy()
iio.imwrite(outpath, frames, fps=25, codec="libx264", plugin="pyav")
print(f"Saved to {outpath}")

counters = result.get("counters")
if counters:
    print(f"Steps: {counters['steps']}, Total: {counters['total_ms']:.0f}ms, "
          f"Avg: {counters['avg_step_ms']:.0f}ms/step, Peak VRAM: {counters['peak_vram_mb']:.0f}MB")
