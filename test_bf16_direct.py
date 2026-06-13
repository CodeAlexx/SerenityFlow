#!/usr/bin/env python3
"""BF16 generation using DIRECT text encoding (no Stagehand), matching LTX-Desktop."""
import os, gc, torch, logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

from serenityflow.bridge.ltxv import (
    _patch_ltx_gemma_transformers_compat,
    _resolve_ltxv_gemma_root,
    _resolve_ltxv_asset,
    _default_ltxv_spatial_upsampler_candidates,
    _default_ltxv_distilled_lora_candidates,
    _build_ltxv_loras,
    _coerce_ltxv_quantization_policy,
)

CHECKPOINT = os.path.expanduser("~/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors")

# Step 1: Apply compat patch and create ledger — exactly like Desktop
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

# Step 2: Text encoding — DIRECT, like Desktop, no Stagehand
print("Loading text encoder to GPU directly...")
text_encoder = ledger.text_encoder()
print(f"Text encoder loaded: {type(text_encoder)}")
print(f"  model device: {text_encoder.model.device}")
print(f"  feature_extractor type: {type(text_encoder.feature_extractor)}")

from ltx_core.text_encoders.gemma import encode_text
prompt = "A busy Tokyo street at night with neon signs and people walking"
neg_prompt = "worst quality, blurry, distorted"

conditioning_list = encode_text(text_encoder, [prompt, neg_prompt])
v_ctx, a_ctx = conditioning_list[0]
v_ctx_neg, a_ctx_neg = conditioning_list[1]

print(f"video_context: shape={v_ctx.shape}, dtype={v_ctx.dtype}, mean={v_ctx.float().mean():.4f}, std={v_ctx.float().std():.4f}")
print(f"neg_context: shape={v_ctx_neg.shape}")

del text_encoder
gc.collect()
torch.cuda.empty_cache()
print("Text encoder freed")

# Step 3: Load transformer and run stage 1 denoising
from stagehand import StagehandRuntime, StagehandConfig
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.components.guiders import MultiModalGuiderParams, create_multimodal_guider_factory
from ltx_core.denoiser import EulerDiffusionStep, euler_denoising_loop
from ltx_core.noise.noiser import GaussianNoiser
from ltx_core.denoiser import multi_modal_guider_factory_denoising_func
from ltx_core.denoiser import denoise_audio_video
from ltx_core.components.pipeline_components import PipelineComponents
from ltx_core.shapes import VideoPixelShape

v_ctx = v_ctx.to(device)
a_ctx = a_ctx.to(device) if a_ctx is not None else None
v_ctx_neg = v_ctx_neg.to(device)
a_ctx_neg = a_ctx_neg.to(device) if a_ctx_neg is not None else None

print("Loading transformer to CPU...")
ledger.device = torch.device("cpu")
transformer = ledger.transformer()
ledger.device = device

# Find block container
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

# Move non-block params to GPU
for name, child in transformer.named_modules():
    if child is xfm_inner:
        continue
    if any(child is b for b in xfm_inner.transformer_blocks):
        continue
    try:
        child.to(device)
    except Exception:
        pass

# Move non-block params within xfm_inner
for name, child in xfm_inner.named_children():
    if name != "transformer_blocks":
        child.to(device)

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

steps = 25
sigmas = LTX2Scheduler().execute(steps=steps).to(dtype=torch.float32, device=device)

video_guider_params = MultiModalGuiderParams(
    cfg_scale=3.5, stg_scale=0.0, rescale_scale=0.7, modality_scale=3.0, skip_step=0, stg_blocks=[28],
)
audio_guider_params = MultiModalGuiderParams(
    cfg_scale=7.0, stg_scale=0.0, rescale_scale=0.7, modality_scale=3.0, skip_step=0, stg_blocks=[28],
)

video_guider_factory = create_multimodal_guider_factory(params=video_guider_params, negative_context=v_ctx_neg)
audio_guider_factory = create_multimodal_guider_factory(params=audio_guider_params, negative_context=a_ctx_neg)

base_fn = multi_modal_guider_factory_denoising_func(
    video_guider_factory=video_guider_factory,
    audio_guider_factory=audio_guider_factory,
    v_context=v_ctx,
    a_context=a_ctx,
    transformer=transformer,
)

_call = [0]
n_steps = len(sigmas) - 1

def wrapped_fn(video_state, audio_state, sigmas_arg, step_index):
    xfm_runtime.begin_step(_call[0])
    with xfm_runtime.managed_forward():
        result = base_fn(video_state, audio_state, sigmas_arg, step_index)
    xfm_runtime.end_step()
    _call[0] += 1
    vram = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    print(f"Step {_call[0]}/{n_steps} (VRAM {vram:.0f}MB)")
    return result

def denoising_loop(sigmas_arg, video_state, audio_state, stepper_arg):
    return euler_denoising_loop(sigmas=sigmas_arg, video_state=video_state, audio_state=audio_state, stepper=stepper_arg, denoise_fn=wrapped_fn)

s1_shape = VideoPixelShape(batch=1, frames=81, width=256, height=160, fps=25.0)
print(f"Stage 1: Denoising {n_steps} steps at 256x160, 81 frames...")

video_state, audio_state = denoise_audio_video(
    output_shape=s1_shape, conditionings=[], noiser=noiser, sigmas=sigmas,
    stepper=stepper, denoising_loop_fn=denoising_loop, components=components,
    dtype=torch.bfloat16, device=device,
)
print("Stage 1 complete")

# Save stage 1 latent as a quick check frame
s1_latent = video_state.latent.cpu()
xfm_runtime.shutdown()
del xfm_runtime, transformer, xfm_inner, base_fn
gc.collect()
torch.cuda.empty_cache()

# Decode stage 1 directly (skip stage 2 for speed)
print("Decoding stage 1 video...")
video_decoder = ledger.video_decoder()
from ltx_core.model.vae import vae_decode
decoded = vae_decode(s1_latent.to(device), video_decoder, overlap=4)
del video_decoder
gc.collect()
torch.cuda.empty_cache()

frames = decoded.squeeze(0).permute(1, 2, 3, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
import imageio.v3 as iio
outpath = "/home/alex/serenityflow-v2/output/test_bf16_direct_s1.mp4"
os.makedirs(os.path.dirname(outpath), exist_ok=True)
iio.imwrite(outpath, frames, fps=25, codec="libx264", plugin="pyav")
print(f"Saved to {outpath} ({frames.shape})")
