#!/usr/bin/env python3
"""Run ONE denoising step and check the output velocity magnitude."""
import os, gc, torch, logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

from serenityflow.bridge.ltxv import load_ltxv_model, _patch_ltx_gemma_transformers_compat
from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.types import VideoPixelShape
from ltx_pipelines.utils.helpers import simple_denoising_func
from ltx_pipelines.utils.types import PipelineComponents
from stagehand import StagehandRuntime, StagehandConfig

CHECKPOINT = os.path.expanduser("~/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors")

_patch_ltx_gemma_transformers_compat()
model = load_ltxv_model(checkpoint_path=CHECKPOINT, gemma_path="", dtype="bfloat16", backend="auto")

# Get embeddings (using the FP8 GPU path)
from serenityflow.bridge.ltxv import _load_ltx_text_encoder_with_fallback
text_encoder, te_dev = _load_ltx_text_encoder_with_fallback(model.model_ledger)
from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaEncoderOutput

device = torch.device("cuda")
context_p = text_encoder("A red sports car")
v_ctx = context_p.video_encoding.to(device).clone()
a_ctx = context_p.audio_encoding
if a_ctx is not None: a_ctx = a_ctx.to(device).clone()

del text_encoder; gc.collect(); torch.cuda.empty_cache()
print(f"v_ctx: {v_ctx.shape}, a_ctx: {a_ctx.shape if a_ctx is not None else None}")

# Load transformer
ledger = model.model_ledger
ledger.device = torch.device("cpu")
transformer = ledger.transformer()
ledger.device = device

def _unwrap(m):
    for a in ("velocity_model", "model", "module"):
        i = getattr(m, a, None)
        if i and hasattr(i, "transformer_blocks"): return i
        if i:
            for a2 in ("velocity_model", "model", "module"):
                i2 = getattr(i, a2, None)
                if i2 and hasattr(i2, "transformer_blocks"): return i2
    raise RuntimeError("No blocks")

from serenityflow.bridge.ltxv import _unwrap_to_blocks, _move_non_blocks_to_device
xfm_inner = _unwrap_to_blocks(transformer)
_move_non_blocks_to_device(transformer, xfm_inner, device)
transformer.requires_grad_(False)

xfm_runtime = StagehandRuntime(
    model=xfm_inner,
    config=StagehandConfig(pinned_pool_mb=6400, pinned_slab_mb=800, vram_high_watermark_mb=18000, vram_low_watermark_mb=14000, prefetch_window_blocks=1, max_inflight_transfers=1, telemetry_enabled=False),
    block_pattern=r"^transformer_blocks\.\d+$", group="transformer", dtype=torch.bfloat16, inference_mode=True,
)

# Create initial noise
shape = VideoPixelShape(batch=1, frames=9, width=64, height=64, fps=25.0)
components = PipelineComponents(dtype=torch.bfloat16, device=device)

generator = torch.Generator(device=device).manual_seed(42)
noise = torch.randn(1, 128, 2, 4, 4, device=device, dtype=torch.bfloat16, generator=generator)

# Run transformer directly with Stagehand
from ltx_core.types import Modality, LatentState
sigma = torch.tensor(DISTILLED_SIGMA_VALUES[0], device=device)
timesteps = sigma.expand(1, 2, 4, 4)

video_mod = Modality(
    enabled=True,
    latent=noise,
    sigma=sigma,
    timesteps=timesteps,
    positions=None,
    context=v_ctx,
    context_mask=None,
    attention_mask=None,
)
audio_mod = Modality(
    enabled=a_ctx is not None,
    latent=torch.zeros(1, 128, 2, 1, 1, device=device, dtype=torch.bfloat16) if a_ctx is not None else None,
    sigma=sigma,
    timesteps=timesteps[:,:,:1,:1] if a_ctx is not None else None,
    positions=None,
    context=a_ctx,
    context_mask=None,
    attention_mask=None,
)

print(f"\nRunning ONE transformer forward through Stagehand...")
xfm_runtime.begin_step(0)
with xfm_runtime.managed_forward():
    denoised_v, denoised_a = transformer(video=video_mod, audio=audio_mod, perturbations=None)
xfm_runtime.end_step()

print(f"\nInput noise: mean={noise.float().mean():.4f}, std={noise.float().std():.4f}")
print(f"Output denoised: mean={denoised_v.float().mean():.4f}, std={denoised_v.float().std():.4f}")
print(f"  min={denoised_v.float().min():.4f}, max={denoised_v.float().max():.4f}")
print(f"  has_nan={torch.isnan(denoised_v).any()}, has_inf={torch.isinf(denoised_v).any()}")

# The denoised output should be different from input noise if text conditioning works
diff = (denoised_v.float() - noise.float()).abs()
print(f"  diff from noise: mean={diff.mean():.4f}, max={diff.max():.4f}")

xfm_runtime.shutdown()
