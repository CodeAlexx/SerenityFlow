#!/usr/bin/env python3
"""Debug Stagehand block loading - check if params are correctly staged."""
import os, torch, logging
logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")

from ltx_pipelines.utils import ModelLedger
from stagehand import StagehandRuntime, StagehandConfig

CHECKPOINT = os.path.expanduser("~/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors")
GEMMA = "/home/alex/.serenity/models/text_encoders/gemma-3-12b-it-standalone"

device = torch.device("cuda")

ledger = ModelLedger(dtype=torch.bfloat16, device=torch.device("cpu"), checkpoint_path=CHECKPOINT, gemma_root_path=GEMMA, spatial_upsampler_path=None, loras=(), quantization=None)
transformer = ledger.transformer()

def _unwrap(m):
    for a in ("velocity_model", "model", "module"):
        i = getattr(m, a, None)
        if i and hasattr(i, "transformer_blocks"): return i
        if i:
            for a2 in ("velocity_model", "model", "module"):
                i2 = getattr(i, a2, None)
                if i2 and hasattr(i2, "transformer_blocks"): return i2
    raise RuntimeError("No blocks")

xfm_inner = _unwrap(transformer)

# Move non-blocks to GPU
block_param_ids = set(id(p) for p in xfm_inner.transformer_blocks.parameters())
block_buf_ids = set(id(b) for b in xfm_inner.transformer_blocks.buffers())
with torch.no_grad():
    for p in transformer.parameters():
        if id(p) not in block_param_ids:
            p.data = p.data.to(device, dtype=torch.bfloat16, non_blocking=True)
    for name, buf in transformer.named_buffers():
        if id(buf) not in block_buf_ids and buf.device != device:
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = transformer.get_submodule(parts[0])
                parent._buffers[parts[1]] = buf.to(device, non_blocking=True)
transformer.requires_grad_(False)

# Check block 0 params BEFORE stagehand
block0 = xfm_inner.transformer_blocks[0]
print("\n=== Block 0 BEFORE Stagehand ===")
for name, p in list(block0.named_parameters())[:5]:
    print(f"  {name}: shape={p.shape}, dtype={p.dtype}, device={p.device}, mean={p.float().mean():.6f}")

xfm_runtime = StagehandRuntime(
    model=xfm_inner,
    config=StagehandConfig(pinned_pool_mb=6400, pinned_slab_mb=800, vram_high_watermark_mb=18000, vram_low_watermark_mb=14000, prefetch_window_blocks=1, max_inflight_transfers=1, telemetry_enabled=False),
    block_pattern=r"^transformer_blocks\.\d+$", group="transformer", dtype=torch.bfloat16, inference_mode=True,
)

# Check block 0 params AFTER stagehand registration (before any forward)
print("\n=== Block 0 AFTER Stagehand registration ===")
for name, p in list(block0.named_parameters())[:5]:
    print(f"  {name}: shape={p.shape}, dtype={p.dtype}, device={p.device}, numel={p.numel()}")

# Simulate begin_step + before_block for block 0
print("\n=== Staging block 0 to GPU ===")
xfm_runtime.begin_step(0)
xfm_runtime._scheduler.before_block("transformer_blocks.0")

print("\n=== Block 0 AFTER staging to GPU ===")
for name, p in list(block0.named_parameters())[:5]:
    print(f"  {name}: shape={p.shape}, dtype={p.dtype}, device={p.device}, mean={p.float().mean():.6f}")

# Check if it matches original
print("\nDone")
xfm_runtime._scheduler.after_block("transformer_blocks.0")
xfm_runtime.end_step()
xfm_runtime.shutdown()
