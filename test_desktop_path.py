#!/usr/bin/env python3
"""Minimal test using Desktop's EXACT approach: FP8-quantized TE on GPU + Stagehand denoiser."""
import os, gc, torch, logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

# Use Desktop's exact approach - no serenityflow imports for the critical path
from ltx_pipelines.utils import ModelLedger
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.types import VideoPixelShape
from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.helpers import denoise_audio_video, simple_denoising_func
from ltx_pipelines.utils.samplers import euler_denoising_loop
from ltx_pipelines.utils.types import PipelineComponents
from stagehand import StagehandRuntime, StagehandConfig

CHECKPOINT = os.path.expanduser("~/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors")
GEMMA = "/home/alex/.serenity/models/text_encoders/gemma-3-12b-it-standalone"
UPSAMPLER = os.path.expanduser("~/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/5a9c1c680bc66c159f708143bf274739961ecd08/ltx-2.3-spatial-upscaler-x2-1.0.safetensors")

device = torch.device("cuda")
dtype = torch.bfloat16

# 1. Build ledger (Desktop way)
ledger = ModelLedger(dtype=dtype, device=device, checkpoint_path=CHECKPOINT, gemma_root_path=GEMMA, spatial_upsampler_path=UPSAMPLER, loras=(), quantization=None)

# 2. Text encoding (Desktop way: FP8 quantize on CPU, move to GPU)
print("Loading text encoder...")
ledger.device = torch.device("cpu")
text_encoder = ledger.text_encoder()
ledger.device = device

# FP8 quantize (Desktop's _quantize_linear_weights_fp8)
for child in text_encoder.modules():
    if isinstance(child, torch.nn.Linear):
        child.weight.data = child.weight.data.to(torch.float8_e4m3fn)
        if child.bias is not None:
            child.bias.data = child.bias.data.to(torch.float8_e4m3fn)
        def _make_fwd(lin):
            def _fwd(x, **kw):
                w = lin.weight.to(x.dtype)
                b = lin.bias.to(x.dtype) if lin.bias is not None else None
                return torch.nn.functional.linear(x, w, b)
            return _fwd
        child.forward = _make_fwd(child)

text_encoder.to(device)
print("Text encoder FP8 on GPU")

conditioning_list = encode_text(text_encoder, ["A busy Tokyo street at night with neon signs"])
v_ctx, a_ctx = conditioning_list[0]
print(f"Embeddings: v={v_ctx.shape} a={a_ctx.shape if a_ctx is not None else None}")

del text_encoder; gc.collect(); torch.cuda.empty_cache()

# 3. Load transformer with Stagehand (Desktop way)
print("Loading transformer...")
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

xfm_inner = _unwrap(transformer)

# Desktop's _move_non_blocks_to_device
block_param_ids = set(id(p) for p in xfm_inner.transformer_blocks.parameters())
block_buf_ids = set(id(b) for b in xfm_inner.transformer_blocks.buffers())
with torch.no_grad():
    for p in transformer.parameters():
        if id(p) not in block_param_ids:
            p.data = p.data.to(device, dtype=dtype, non_blocking=True)
    for name, buf in transformer.named_buffers():
        if id(buf) not in block_buf_ids and buf.device != device:
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = transformer.get_submodule(parts[0])
                parent._buffers[parts[1]] = buf.to(device, non_blocking=True)
transformer.requires_grad_(False)

xfm_runtime = StagehandRuntime(
    model=xfm_inner,
    config=StagehandConfig(pinned_pool_mb=6400, pinned_slab_mb=800, vram_high_watermark_mb=18000, vram_low_watermark_mb=14000, prefetch_window_blocks=1, max_inflight_transfers=1, telemetry_enabled=False),
    block_pattern=r"^transformer_blocks\.\d+$", group="transformer", dtype=dtype, inference_mode=True,
)
print(f"Stagehand ready ({len(xfm_runtime._registry)} blocks)")

# 4. Denoise (Desktop distilled mode)
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

shape = VideoPixelShape(batch=1, frames=41, width=192, height=128, fps=25.0)
print(f"Denoising {n} steps...")
vs, aus = denoise_audio_video(
    output_shape=shape, conditionings=[], noiser=GaussianNoiser(torch.Generator(device=device).manual_seed(42)),
    sigmas=sigmas, stepper=EulerDiffusionStep(),
    denoising_loop_fn=lambda s,v,a,st: euler_denoising_loop(sigmas=s, video_state=v, audio_state=a, stepper=st, denoise_fn=wrapped),
    components=PipelineComponents(dtype=dtype, device=device), dtype=dtype, device=device,
)

lat = vs.latent.cpu()
xfm_runtime.shutdown()
del xfm_runtime, transformer; gc.collect(); torch.cuda.empty_cache()

# Decode
from ltx_core.model.video_vae import decode_video
vd = ledger.video_decoder()
decoded = decode_video(lat.to(device), vd)
frames = decoded.squeeze(0).permute(1, 2, 3, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
import imageio.v3 as iio
out = "/home/alex/serenityflow-v2/output/test_desktop_path.mp4"
iio.imwrite(out, frames, fps=25, codec="libx264", plugin="pyav")
print(f"Saved {out} shape={frames.shape}")
print(f"RGB mean: ({frames[:,:,:,0].mean():.1f}, {frames[:,:,:,1].mean():.1f}, {frames[:,:,:,2].mean():.1f})")
