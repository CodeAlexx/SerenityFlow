#!/usr/bin/env python3
"""Use OFFICIAL ltx_core encode_text on CPU - bypass ALL serenityflow code for text encoding."""
import os, gc, torch, logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

CHECKPOINT = os.path.expanduser("~/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors")
device = torch.device("cuda")

# Build ledger WITHOUT any serenityflow patches
from ltx_pipelines.utils import ModelLedger
from ltx_core.text_encoders.gemma import encode_text

gemma_root = "/home/alex/.serenity/models/text_encoders/gemma-3-12b-it-standalone"
upsampler = os.path.expanduser("~/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/5a9c1c680bc66c159f708143bf274739961ecd08/ltx-2.3-spatial-upscaler-x2-1.0.safetensors")

ledger = ModelLedger(
    dtype=torch.bfloat16,
    device=torch.device("cpu"),  # Load text encoder on CPU
    checkpoint_path=CHECKPOINT,
    gemma_root_path=gemma_root,
    spatial_upsampler_path=upsampler,
    loras=(),
    quantization=None,
)

print("Loading text encoder on CPU (official ltx_core path, no patches)...")
text_encoder = ledger.text_encoder()
print(f"Text encoder type: {type(text_encoder)}")
print(f"Model device: {text_encoder.model.device}")

print("Encoding text on CPU (slow but correct)...")
prompt = "A red sports car driving on a highway"
conditioning_list = encode_text(text_encoder, [prompt])
v_ctx, a_ctx = conditioning_list[0]

print(f"video_ctx: shape={v_ctx.shape}, mean={v_ctx.float().mean():.4f}, std={v_ctx.float().std():.4f}")

del text_encoder
gc.collect()
torch.cuda.empty_cache()
import ctypes
ctypes.CDLL("libc.so.6").malloc_trim(0)

# Now denoise with Stagehand
v_ctx = v_ctx.to(device)
a_ctx = a_ctx.to(device) if a_ctx is not None else None

ledger.device = torch.device("cpu")
transformer = ledger.transformer()
ledger.device = device

from stagehand import StagehandRuntime, StagehandConfig
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.types import VideoPixelShape
from ltx_pipelines.utils.helpers import denoise_audio_video, simple_denoising_func
from ltx_pipelines.utils.samplers import euler_denoising_loop
from ltx_pipelines.utils.types import PipelineComponents
from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES

def _unwrap(model):
    for attr in ("velocity_model", "model", "module"):
        inner = getattr(model, attr, None)
        if inner and hasattr(inner, "transformer_blocks"):
            return inner
        if inner:
            for a2 in ("velocity_model", "model", "module"):
                i2 = getattr(inner, a2, None)
                if i2 and hasattr(i2, "transformer_blocks"):
                    return i2
    raise RuntimeError("Cannot find blocks")

xfm_inner = _unwrap(transformer)
for name, child in xfm_inner.named_children():
    if name != "transformer_blocks":
        child.to(device)
for name, mod in transformer.named_modules():
    if mod is not xfm_inner and not any(mod is b for b in xfm_inner.transformer_blocks):
        try: mod.to(device)
        except: pass
transformer.requires_grad_(False)

xfm_runtime = StagehandRuntime(
    model=xfm_inner,
    config=StagehandConfig(pinned_pool_mb=6400, pinned_slab_mb=800, vram_high_watermark_mb=18000, vram_low_watermark_mb=14000, prefetch_window_blocks=1, max_inflight_transfers=1, telemetry_enabled=False),
    block_pattern=r"^transformer_blocks\.\d+$", group="transformer", dtype=torch.bfloat16, inference_mode=True,
)

sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=device)
base_fn = simple_denoising_func(video_context=v_ctx, audio_context=a_ctx, transformer=transformer)

_call = [0]
n = len(sigmas) - 1
def wrapped(vs, as_, sig, si):
    xfm_runtime.begin_step(_call[0])
    with xfm_runtime.managed_forward():
        r = base_fn(vs, as_, sig, si)
    xfm_runtime.end_step()
    _call[0] += 1
    print(f"Step {_call[0]}/{n}")
    return r

shape = VideoPixelShape(batch=1, frames=41, width=256, height=160, fps=25.0)
print(f"Denoising {n} steps...")
vs, aus = denoise_audio_video(
    output_shape=shape, conditionings=[], noiser=GaussianNoiser(torch.Generator(device=device).manual_seed(42)),
    sigmas=sigmas, stepper=EulerDiffusionStep(),
    denoising_loop_fn=lambda s,v,a,st: euler_denoising_loop(sigmas=s, video_state=v, audio_state=a, stepper=st, denoise_fn=wrapped),
    components=PipelineComponents(dtype=torch.bfloat16, device=device), dtype=torch.bfloat16, device=device,
)

latent = vs.latent.cpu()
xfm_runtime.shutdown()
del xfm_runtime, transformer
gc.collect(); torch.cuda.empty_cache()

# Decode
from ltx_core.model.video_vae import decode_video as vae_decode
vd = ledger.video_decoder()
decoded = vae_decode(latent.to(device), vd)
frames = decoded.squeeze(0).permute(1, 2, 3, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
import imageio.v3 as iio
out = "/home/alex/serenityflow-v2/output/test_official_s1.mp4"
iio.imwrite(out, frames, fps=25, codec="libx264", plugin="pyav")
print(f"Saved {out} ({frames.shape})")
