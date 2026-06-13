#!/usr/bin/env python3
"""Generate a video. Distilled mode, 8 steps, fast."""
import os
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

from serenityflow.bridge.serenity_api import load_ltxv_model, sample_ltxv

CHECKPOINT = os.path.expanduser("~/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors")
GEMMA = os.path.expanduser("~/.serenity/models/text_encoders/gemma-3-12b-ltx")
UPSCALER = os.path.expanduser("~/.serenity/models/upscalers/ltx-2-spatial-upscaler-x2-1.0.safetensors")
DISTILLED_LORA = os.path.expanduser("~/.serenity/models/loras/ltx-2-19b-distilled-lora-384.safetensors")

model = load_ltxv_model(
    checkpoint_path=CHECKPOINT,
    gemma_path=GEMMA,
    spatial_upsampler_path=UPSCALER,
    distilled_lora_path=DISTILLED_LORA,
)

result = sample_ltxv(
    model=model,
    prompt="A golden retriever running through a sunlit meadow, slow motion, cinematic",
    width=512,
    height=320,
    num_frames=25,
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
outpath = os.path.join(outdir, "test_stg_video.mp4")
iio.imwrite(outpath, video.numpy(), fps=25, codec="libx264", plugin="pyav")
print(f"Saved to {outpath}")

counters = result.get("counters")
if counters:
    print(f"Steps: {counters['steps']}, Total: {counters['total_ms']:.0f}ms, "
          f"Avg: {counters['avg_step_ms']:.0f}ms/step, Peak VRAM: {counters['peak_vram_mb']:.0f}MB")
