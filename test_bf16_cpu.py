#!/usr/bin/env python3
"""BF16 generation using CPU text encoding (no Stagehand), to isolate the problem."""
import os, gc, torch, logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

from serenityflow.bridge.ltxv import (
    _patch_ltx_gemma_transformers_compat,
    _resolve_ltxv_gemma_root,
    _resolve_ltxv_asset,
    _default_ltxv_spatial_upsampler_candidates,
)

CHECKPOINT = os.path.expanduser("~/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors")

_patch_ltx_gemma_transformers_compat()

from ltx_pipelines.utils import ModelLedger
gemma_root = _resolve_ltxv_gemma_root("", is_fp8=False)
device = torch.device("cuda")

ledger = ModelLedger(
    dtype=torch.bfloat16,
    device=device,
    checkpoint_path=CHECKPOINT,
    gemma_root_path=gemma_root,
    spatial_upsampler_path=_resolve_ltxv_asset(None, "upscale_models", *_default_ltxv_spatial_upsampler_candidates(CHECKPOINT)),
    loras=(),
    quantization=None,
)

# Load text encoder on CPU
print("Loading text encoder to CPU...")
ledger.device = torch.device("cpu")
text_encoder = ledger.text_encoder()
ledger.device = device
print(f"Text encoder loaded on CPU")
print(f"  model type: {type(text_encoder.model)}")
print(f"  model device: {text_encoder.model.device}")

# Use the ORIGINAL precompute (not patched), on CPU
prompt = "A busy Tokyo street at night with neon signs and people walking"
neg_prompt = "worst quality, blurry, distorted"

print("Running precompute on CPU (this will be slow)...")
with torch.no_grad():
    out_pos = text_encoder(prompt)
    out_neg = text_encoder(neg_prompt)

v_ctx = out_pos.video_encoding
a_ctx = out_pos.audio_encoding
v_ctx_neg = out_neg.video_encoding
a_ctx_neg = out_neg.audio_encoding

print(f"video_context: shape={v_ctx.shape}, dtype={v_ctx.dtype}")
print(f"  mean={v_ctx.float().mean():.6f}, std={v_ctx.float().std():.6f}")
print(f"  min={v_ctx.float().min():.6f}, max={v_ctx.float().max():.6f}")

# Check for NaN/Inf
print(f"  has_nan={torch.isnan(v_ctx).any()}, has_inf={torch.isinf(v_ctx).any()}")

print("\nText encoding done. If embeddings look sane, running denoising...")

# Now run stage 1 denoising with these embeddings
from stagehand import StagehandRuntime, StagehandConfig
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.components.guiders import MultiModalGuiderParams, create_multimodal_guider_factory
from ltx_core.denoiser import EulerDiffusionStep, euler_denoising_loop
from ltx_core.noise.noiser import GaussianNoiser
from ltx_core.denoiser import multi_modal_guider_factory_denoising_func, denoise_audio_video
from ltx_core.components.pipeline_components import PipelineComponents
from ltx_core.shapes import VideoPixelShape

v_ctx = v_ctx.to(device)
a_ctx = a_ctx.to(device) if a_ctx is not None else None
v_ctx_neg = v_ctx_neg.to(device)
a_ctx_neg = a_ctx_neg.to(device) if a_ctx_neg is not None else None

del text_encoder
gc.collect()

print("Loading transformer to CPU...")
ledger.device = torch.device("cpu")
transformer = ledger.transformer()
ledger.device = device

def _unwrap(model):
    for attr in ("velocity_model", "model", "module"):
        inner = getattr(model, attr, None)
        if inner is not None:
            if hasattr(inner, "transformer_blocks"):
                return inner
            for a2 in ("velocity_model", "model", "module"):
                i2 = getattr(inner, a2, None)
                if i2 is not None and hasattr(i2, "transformer_blocks"):
                    return i2
    raise RuntimeError("Cannot find transformer_blocks")

xfm_inner = _unwrap(transformer)

for name, child in xfm_inner.named_children():
    if name != "transformer_blocks":
        child.to(device)
for name, mod in transformer.named_modules():
    if mod is not xfm_inner and not any(mod is b for b in xfm_inner.transformer_blocks):
        try:
            mod.to(device)
        except Exception:
            pass

transformer.requires_grad_(False)

stg_config = StagehandConfig(
    pinned_pool_mb=6400, pinned_slab_mb=800,
    vram_high_watermark_mb=18000, vram_low_watermark_mb=14000,
    prefetch_window_blocks=1, max_inflight_transfers=1, telemetry_enabled=False,
)
xfm_runtime = StagehandRuntime(
    model=xfm_inner, config=stg_config,
    block_pattern=r"^transformer_blocks\.\d+$", group="transformer",
    dtype=torch.bfloat16, inference_mode=True,
)
print(f"Stagehand transformer ready ({len(xfm_runtime._registry)} blocks)")

generator = torch.Generator(device=device).manual_seed(42)
noiser = GaussianNoiser(generator=generator)
stepper = EulerDiffusionStep()
components = PipelineComponents(dtype=torch.bfloat16, device=device)

sigmas = LTX2Scheduler().execute(steps=25).to(dtype=torch.float32, device=device)

video_gp = MultiModalGuiderParams(cfg_scale=3.5, stg_scale=0.0, rescale_scale=0.7, modality_scale=3.0, skip_step=0, stg_blocks=[28])
audio_gp = MultiModalGuiderParams(cfg_scale=7.0, stg_scale=0.0, rescale_scale=0.7, modality_scale=3.0, skip_step=0, stg_blocks=[28])

video_gf = create_multimodal_guider_factory(params=video_gp, negative_context=v_ctx_neg)
audio_gf = create_multimodal_guider_factory(params=audio_gp, negative_context=a_ctx_neg)

base_fn = multi_modal_guider_factory_denoising_func(
    video_guider_factory=video_gf, audio_guider_factory=audio_gf,
    v_context=v_ctx, a_context=a_ctx, transformer=transformer,
)

_call = [0]
n_steps = len(sigmas) - 1

def wrapped_fn(video_state, audio_state, sigmas_arg, step_index):
    xfm_runtime.begin_step(_call[0])
    with xfm_runtime.managed_forward():
        result = base_fn(video_state, audio_state, sigmas_arg, step_index)
    xfm_runtime.end_step()
    _call[0] += 1
    vram = torch.cuda.memory_allocated() / (1024**2)
    print(f"Step {_call[0]}/{n_steps} (VRAM {vram:.0f}MB)")
    return result

def denoising_loop(sigmas_arg, video_state, audio_state, stepper_arg):
    return euler_denoising_loop(sigmas=sigmas_arg, video_state=video_state, audio_state=audio_state, stepper=stepper_arg, denoise_fn=wrapped_fn)

s1_shape = VideoPixelShape(batch=1, frames=81, width=256, height=160, fps=25.0)
print(f"Stage 1: {n_steps} steps at 256x160...")

video_state, audio_state = denoise_audio_video(
    output_shape=s1_shape, conditionings=[], noiser=noiser, sigmas=sigmas,
    stepper=stepper, denoising_loop_fn=denoising_loop, components=components,
    dtype=torch.bfloat16, device=device,
)
print("Stage 1 complete")

s1_latent = video_state.latent.cpu()
xfm_runtime.shutdown()
del xfm_runtime, transformer, xfm_inner, base_fn
gc.collect()
torch.cuda.empty_cache()

print("Decoding...")
video_decoder = ledger.video_decoder()
from ltx_core.model.vae import vae_decode
decoded = vae_decode(s1_latent.to(device), video_decoder, overlap=4)

frames = decoded.squeeze(0).permute(1, 2, 3, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
import imageio.v3 as iio
outpath = "/home/alex/serenityflow-v2/output/test_bf16_cpu_s1.mp4"
os.makedirs(os.path.dirname(outpath), exist_ok=True)
iio.imwrite(outpath, frames, fps=25, codec="libx264", plugin="pyav")
print(f"Saved to {outpath} ({frames.shape})")
