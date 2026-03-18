"""Clean API surface for Serenity inference.

Wraps Serenity's internal functions into stable callables.
If Serenity's API changes, only this file needs updating.
SerenityFlow nodes never import from serenity directly -- only from here.
"""
from __future__ import annotations

import copy
import functools
import gc
import io
import logging
import os
import sys
import threading
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_LTX_GEMMA_TRANSFORMERS_COMPAT_PATCHED = False


# ---------------------------------------------------------------------------
# Live preview infrastructure — thread-local WS sender for step callbacks
# ---------------------------------------------------------------------------

_preview_local = threading.local()


def set_preview_sender(send_progress_fn, send_binary_fn):
    """Set the WS sender functions for the current thread.

    Called from execution.py before sampling starts.
    send_progress_fn(step, total): sends JSON progress event
    send_binary_fn(data: bytes): sends binary preview image
    """
    _preview_local.send_progress = send_progress_fn
    _preview_local.send_binary = send_binary_fn


def clear_preview_sender():
    """Clear the WS sender for the current thread."""
    _preview_local.send_progress = None
    _preview_local.send_binary = None


def _latent_to_preview_jpeg(latent, max_size=512):
    """Convert a latent tensor to a small JPEG preview.

    Uses a cheap approximation: scale latent channels to RGB directly
    (no full VAE decode -- too slow per step).

    Handles both spatial (B,C,H,W) and packed/sequence (B,seq,C) formats,
    as well as video latents (B,C,T,H,W) where we take the middle frame.
    """
    from PIL import Image

    with torch.no_grad():
        lat = latent[0].float().cpu()

        # Handle video latents (C, T, H, W) -- take middle frame
        if lat.ndim == 4:
            mid = lat.shape[1] // 2
            lat = lat[:, mid, :, :]  # -> (C, H, W)

        # Handle packed/sequence format (seq, C) -- reshape to approximate spatial
        if lat.ndim == 2:
            seq, channels = lat.shape
            # Guess spatial dims from sequence length
            side = int(seq ** 0.5)
            if side * side < seq:
                side += 1
            # Pad if needed
            if side * side > seq:
                pad = torch.zeros(side * side - seq, channels, dtype=lat.dtype)
                lat = torch.cat([lat, pad], dim=0)
            lat = lat[:side * side].view(side, side, channels).permute(2, 0, 1)

        # lat is now (C, H, W)
        if lat.shape[0] >= 3:
            rgb = lat[:3]
        else:
            rgb = lat.repeat(3, 1, 1)[:3]

        # Normalize to 0-255
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        rgb = (rgb * 255).clamp(0, 255).byte()

        # Convert to PIL, resize to max_size
        img = Image.fromarray(rgb.permute(1, 2, 0).numpy(), 'RGB')
        # Latent is typically 1/8 resolution -- upscale for preview
        w, h = img.size
        scale = min(max_size / max(w, 1), max_size / max(h, 1))
        if scale > 1:
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

        # Encode as JPEG
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=50)
        return buf.getvalue()


def _make_step_callback(preview_interval=3):
    """Create a sampling step callback that sends WS progress + preview.

    Returns None if no preview sender is registered for this thread.
    """
    send_progress = getattr(_preview_local, 'send_progress', None)
    send_binary = getattr(_preview_local, 'send_binary', None)
    if send_progress is None:
        return None

    def callback(step, total, sigma, denoised):
        # Always send progress
        send_progress(step + 1, total)

        # Send preview every Nth step + final step
        if send_binary is not None and denoised is not None:
            if step % preview_interval == 0 or step == total - 1:
                try:
                    preview_bytes = _latent_to_preview_jpeg(denoised)
                    send_binary(preview_bytes)
                except Exception:
                    pass  # Don't break sampling for preview failures

    return callback


# ---------------------------------------------------------------------------
# Lazy imports from Serenity -- defer to avoid hard failure at import time
# ---------------------------------------------------------------------------

def _import_inference():
    """Import Serenity inference subsystems. Raises ImportError with a
    clear message if Serenity is not installed."""
    try:
        from serenity.inference.models.loader import (
            load_model as _load_model,
            load_state_dict as _load_sd,
            extract_vae_state_dict as _extract_vae,
        )
        from serenity.inference.models.detection import (
            detect_from_file,
            detect_model_type,
            ModelArchitecture,
        )
        from serenity.inference.vae.decoder import VAEDecoder
        from serenity.inference.vae.encoder import VAEEncoder
        from serenity.inference.text.encoders import TextEncoderManager, get_required_encoders
        from serenity.inference.sampling.sampler import sample as _sample, create_model_fn
        from serenity.inference.sampling.schedulers import compute_sigmas
        from serenity.inference.sampling.conditioning import Conditioning, create_noise
        from serenity.inference.sampling.prediction import get_prediction
        from serenity.inference.sampling.cfg import apply_cfg
        from serenity.inference.lora.loader import load_lora
        from serenity.inference.lora.merge import merge_lora_into_model

        return {
            "load_model": _load_model,
            "load_state_dict": _load_sd,
            "extract_vae": _extract_vae,
            "detect_from_file": detect_from_file,
            "detect_model_type": detect_model_type,
            "ModelArchitecture": ModelArchitecture,
            "VAEDecoder": VAEDecoder,
            "VAEEncoder": VAEEncoder,
            "TextEncoderManager": TextEncoderManager,
            "get_required_encoders": get_required_encoders,
            "sample": _sample,
            "create_model_fn": create_model_fn,
            "compute_sigmas": compute_sigmas,
            "Conditioning": Conditioning,
            "create_noise": create_noise,
            "get_prediction": get_prediction,
            "apply_cfg": apply_cfg,
            "load_lora": load_lora,
            "merge_lora_into_model": merge_lora_into_model,
        }
    except ImportError as e:
        raise ImportError(
            f"Serenity inference engine not found: {e}. "
            "Install serenity or ensure it is on PYTHONPATH."
        ) from e


_SERENITY = None


def _get():
    """Get cached Serenity imports."""
    global _SERENITY
    if _SERENITY is None:
        _SERENITY = _import_inference()
    return _SERENITY


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------


class LoadedCheckpoint:
    """Container for a loaded checkpoint's components."""
    __slots__ = ("model", "vae_decoder", "vae_encoder", "model_config", "adapter")

    def __init__(self, model, vae_decoder, vae_encoder, model_config, adapter):
        self.model = model
        self.vae_decoder = vae_decoder
        self.vae_encoder = vae_encoder
        self.model_config = model_config
        self.adapter = adapter


class CLIPWrapper:
    """Wraps Serenity's TextEncoderManager for use as a CLIP object in nodes."""
    __slots__ = ("_manager", "_arch", "_dtype", "_device", "_clip_skip")

    def __init__(self, manager, arch, dtype, device):
        self._manager = manager
        self._arch = arch
        self._dtype = dtype
        self._device = device
        self._clip_skip = 0

    def set_last_layer(self, stop_at_layer: int):
        self._clip_skip = abs(stop_at_layer)

    @property
    def clip_skip(self):
        return self._clip_skip


class VAEWrapper:
    """Wraps Serenity's VAEDecoder + VAEEncoder for node use."""
    __slots__ = ("decoder", "encoder")

    def __init__(self, decoder, encoder=None):
        self.decoder = decoder
        self.encoder = encoder


def load_checkpoint(path: str, dtype: str = "float16") -> tuple:
    """Load model + clip + vae from a single checkpoint file.

    Returns (model, clip_wrapper, vae_wrapper).
    """
    s = _get()
    dtype_map = {
        "float16": torch.float16, "fp16": torch.float16,
        "float32": torch.float32, "fp32": torch.float32,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)
    gpu_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Detect architecture
    model_config = s["detect_from_file"](path)
    if model_config is None:
        raise RuntimeError(f"Cannot detect model architecture from {path}")

    arch = model_config.architecture
    logger.info("Detected architecture: %s", arch.value)

    # Load model to CPU first to avoid OOM on large models (e.g. FLUX 22GB)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = s["load_model"](path, config=model_config, device="cpu", dtype=torch_dtype)

    # Fix scaled FP8: generic loader may cast FP8→bf16 without applying scales.
    try:
        from serenityflow.bridge.fp8_dequant import dequant_scaled_fp8_in_model
        fixed = dequant_scaled_fp8_in_model(model, path)
        if fixed > 0:
            logger.info("Checkpoint FP8 dequant: fixed %d weights", fixed)
    except Exception as exc:
        logger.debug("FP8 dequant check skipped for checkpoint: %s", exc)

    # Attach prediction metadata (same as load_diffusion_model does)
    from serenity.inference.models.loader import _get_adapter
    adapter = _get_adapter(model_config)
    if adapter:
        model._serenity_prediction_type = adapter.get_prediction_type()
        model._serenity_arch = model_config.architecture
        model._serenity_model_config = model_config
        logger.info("Checkpoint prediction type: %s (arch=%s)", model._serenity_prediction_type, arch.value)

    # VRAM-aware partial GPU loading with block-level CPU↔GPU streaming
    if torch.cuda.is_available():
        model_size = sum(p.data.nbytes for p in model.parameters())
        free_vram = torch.cuda.mem_get_info()[0]
        headroom = 4 * (1024**3)  # 4GB for CLIP + T5 + VAE + activations
        gpu_budget = max(0, free_vram - headroom)
        logger.info(
            "Checkpoint model: %.1f GB, VRAM free: %.1f GB, GPU budget: %.1f GB",
            model_size / (1024**3), free_vram / (1024**3), gpu_budget / (1024**3),
        )
        if model_size > gpu_budget:
            # Model doesn't fit — use block-level offloading
            _enable_layer_offloading(model)
            from serenity.inference.memory.manager import ModelManager
            manager = _get_model_manager()
            lm = manager.load(model, budget=gpu_budget)
            if lm.offloaded_size > 0:
                logger.info(
                    "Loaded %.1f GB to GPU, %.1f GB offloaded to CPU (per-layer streaming)",
                    lm.loaded_size / (1024**3), lm.offloaded_size / (1024**3),
                )
            else:
                logger.info("Fully loaded to GPU (%.1f GB)", lm.loaded_size / (1024**3))
        else:
            # Model fits — load fully to GPU
            model = model.to(gpu_device)

    # Load VAE from checkpoint
    sd = s["load_state_dict"](path)
    vae_sd = s["extract_vae"](sd)
    vae_wrapper = None
    if vae_sd:
        try:
            from serenity.inference.models.convert import convert_ldm_vae_to_diffusers, safe_load_state_dict
            from diffusers.models import AutoencoderKL

            vae_sd = convert_ldm_vae_to_diffusers(vae_sd)
            # Squeeze conv1x1 attention weights to linear format (512,512,1,1) → (512,512)
            for k in list(vae_sd.keys()):
                if vae_sd[k].ndim == 4 and vae_sd[k].shape[2:] == (1, 1) and 'attentions' in k:
                    vae_sd[k] = vae_sd[k].squeeze(-1).squeeze(-1)
            latent_ch = 4
            if "decoder.conv_in.weight" in vae_sd:
                latent_ch = vae_sd["decoder.conv_in.weight"].shape[1]

            # Use canonical SD1.5/SDXL VAE config
            vae = AutoencoderKL(
                in_channels=3,
                out_channels=3,
                latent_channels=latent_ch,
                block_out_channels=(128, 256, 512, 512),
                layers_per_block=2,
                down_block_types=("DownEncoderBlock2D",) * 4,
                up_block_types=("UpDecoderBlock2D",) * 4,
                norm_num_groups=32,
                act_fn="silu",
                sample_size=512,
            )
            safe_load_state_dict(vae, vae_sd)
            # VAE MUST run in fp32 — fp16 causes green/purple artifacts (known diffusers issue)
            vae = vae.to(device=gpu_device, dtype=torch.float32).eval()

            from serenity.inference.models.base import BaseModelAdapter
            # Get adapter for scaling factor
            from serenity.inference.models.loader import _get_adapter
            adapter = _get_adapter(model_config)
            scaling = adapter.get_vae_scaling_factor() if adapter else 0.18215

            decoder = s["VAEDecoder"](vae_model=vae, dtype=torch.float32, device=gpu_device, scaling_factor=scaling)
            encoder = s["VAEEncoder"](vae_model=vae, dtype=torch.float32, device=gpu_device, scaling_factor=scaling)
            vae_wrapper = VAEWrapper(decoder, encoder)
        except Exception as e:
            logger.warning("Failed to load VAE from checkpoint: %s", e)

    # Load text encoders
    text_mgr = s["TextEncoderManager"]()
    try:
        text_mgr.load_for_model(arch, dtype=torch_dtype, device=gpu_device)
    except Exception as e:
        logger.warning("Failed to load text encoders for %s: %s", arch.value, e)

    clip_wrapper = CLIPWrapper(text_mgr, arch, torch_dtype, gpu_device)

    return (model, clip_wrapper, vae_wrapper)


def load_diffusion_model(path: str, dtype: str = "default") -> nn.Module:
    """Load standalone diffusion model (UNet/DiT) with VRAM-aware offloading.

    Loads model to CPU first, then uses ModelManager for budget-aware partial
    GPU loading.  Layers that don't fit in VRAM are kept on CPU and streamed
    to GPU on-demand during forward passes (per-layer offloading).

    Attaches _serenity_prediction_type and _serenity_arch to the model
    so the sampler knows how to handle it.
    """
    s = _get()
    torch_dtype = torch.bfloat16 if dtype == "default" else _parse_dtype(dtype)

    # Always load to CPU first to avoid OOM
    model_config = s["detect_from_file"](path)
    if _is_flux2_model_config(model_config, s["ModelArchitecture"]):
        model = _load_flux2_transformer(path, torch_dtype, model_config)
    elif model_config is not None and model_config.architecture == s["ModelArchitecture"].ZIMAGE:
        model = _load_zimage_transformer(path, torch_dtype, model_config)
    elif model_config is not None and model_config.architecture == s["ModelArchitecture"].QWEN:
        model = _load_qwen_transformer(path, torch_dtype, model_config)
    else:
        model = s["load_model"](path, device="cpu", dtype=torch_dtype)

    if model_config is not None and not hasattr(model, "_serenity_prediction_type"):
        from serenity.inference.models.loader import _get_adapter
        adapter = _get_adapter(model_config)
        if adapter:
            model._serenity_prediction_type = adapter.get_prediction_type()
            model._serenity_arch = model_config.architecture
            model._serenity_model_config = model_config
            logger.info("Model prediction type: %s", model._serenity_prediction_type)
    if _is_flux2_model_config(model_config, s["ModelArchitecture"]):
        model._serenity_flux2 = True
        _attach_flux2_vae_stats(model, path)

    # Fix scaled FP8 for non-Flux models (Flux handled inside _load_flux2_transformer).
    # Loaders like from_single_file may cast FP8→bf16 without applying scale factors.
    if not _is_flux2_model_config(model_config, s["ModelArchitecture"]):
        try:
            from serenityflow.bridge.fp8_dequant import dequant_scaled_fp8_in_model
            fixed = dequant_scaled_fp8_in_model(model, path)
            if fixed > 0:
                logger.info("FP8 dequant: fixed %d weights for %s", fixed, path)
        except Exception as exc:
            logger.debug("FP8 dequant check skipped: %s", exc)

    # Partially load to GPU, keeping headroom for text encoders + VAE (~4GB).
    # Blocks that don't fit get CPU↔GPU streaming hooks.
    if torch.cuda.is_available():
        model_size = sum(p.data.nbytes for p in model.parameters())
        free_vram = torch.cuda.mem_get_info()[0]
        headroom = 4 * (1024**3)  # 4GB for CLIP + T5 + VAE + activations
        gpu_budget = max(0, free_vram - headroom)
        logger.info(
            "Model size: %.1f GB, VRAM free: %.1f GB, GPU budget: %.1f GB (%.1f GB headroom)",
            model_size / (1024**3), free_vram / (1024**3),
            gpu_budget / (1024**3), headroom / (1024**3),
        )
        _enable_layer_offloading(model)
        # Move as much as fits within budget
        from serenity.inference.memory.manager import ModelManager
        manager = _get_model_manager()
        lm = manager.load(model, budget=gpu_budget)
        if lm.offloaded_size > 0:
            logger.info(
                "Loaded %.1f GB to GPU, %.1f GB offloaded to CPU (per-layer streaming)",
                lm.loaded_size / (1024**3), lm.offloaded_size / (1024**3),
            )
        else:
            logger.info("Fully loaded to GPU (%.1f GB)", lm.loaded_size / (1024**3))

    return model


# Singleton model manager for the SerenityFlow process
_model_manager = None


def _get_model_manager():
    """Get or create the singleton ModelManager."""
    global _model_manager
    if _model_manager is None:
        from serenity.inference.memory.manager import ModelManager
        _model_manager = ModelManager()
    return _model_manager


def _enable_layer_offloading(model: nn.Module):
    """Enable offload_enabled on all leaf modules that support the OffloadMixin.

    For models NOT built with OffloadMixin layers, install forward hooks that
    stream weights GPU↔CPU per-layer (ComfyUI-style block offloading).
    """
    has_offload_layers = False
    for mod in model.modules():
        if hasattr(mod, 'offload_enabled') and hasattr(mod, '_init_offload'):
            has_offload_layers = True
            break

    if has_offload_layers:
        # Model uses OffloadMixin layers — just enable them
        for mod in model.modules():
            if hasattr(mod, 'offload_enabled'):
                mod.offload_enabled = True
        return

    # Fallback: install forward hooks on transformer blocks for GPU↔CPU streaming.
    # This handles standard nn.Linear/nn.LayerNorm models (e.g. diffusers-style).
    blocks = _find_transformer_blocks(model)
    if not blocks:
        logger.warning("No transformer blocks found for offloading — model may OOM")
        return

    device = torch.device("cuda")
    logger.info("Installing block-level CPU offload hooks on %d blocks", len(blocks))

    def make_pre_hook(block):
        def hook(module, args):
            module.to(device, non_blocking=True)
            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()
        return hook

    def make_post_hook(block):
        def hook(module, args, output):
            module.to("cpu", non_blocking=True)
        return hook

    # Collect all parameters belonging to blocks so we can identify non-block params
    block_param_ids = set()
    for block in blocks:
        for p in block.parameters():
            block_param_ids.add(id(p))

    # Move non-block params (embeddings, projections, norms) to GPU — they're small
    non_block_bytes = 0
    for p in model.parameters():
        if id(p) not in block_param_ids:
            non_block_bytes += p.data.nbytes
            p.data = p.data.to(device, non_blocking=True)

    logger.info(
        "Moved %.1f MB of non-block params (embeddings/projections/norms) to GPU",
        non_block_bytes / (1024**2),
    )

    for block in blocks:
        block.register_forward_pre_hook(make_pre_hook(block))
        block.register_forward_hook(make_post_hook(block))


def _find_transformer_blocks(model: nn.Module) -> list[nn.Module]:
    """Find repeated transformer/dit blocks suitable for block-level offloading."""
    # Common patterns: .transformer_blocks, .joint_transformer_blocks,
    # .single_transformer_blocks, .layers, .blocks
    block_attr_names = [
        "transformer_blocks", "joint_transformer_blocks",
        "single_transformer_blocks", "layers", "blocks",
    ]

    found = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 4:
            # A ModuleList with 4+ children is likely a repeated block stack
            parent_name = name.split(".")[-1] if "." in name else name
            if parent_name in block_attr_names or len(mod) >= 10:
                found.extend(list(mod))

    # Deduplicate (in case nested finds overlap)
    seen = set()
    unique = []
    for block in found:
        if id(block) not in seen:
            seen.add(id(block))
            unique.append(block)

    return unique


def _is_flux2_model_config(model_config, model_architecture) -> bool:
    """Return True for FLUX.2-style single-file transformers."""
    if model_config is None:
        return False
    if model_config.architecture in (
        model_architecture.FLUX_2_KLEIN_4B,
        model_architecture.FLUX_2_KLEIN_9B,
    ):
        return True
    return model_config.unet_config.get("image_model") == "flux2"


def _load_flux2_transformer(path: str, dtype: torch.dtype, model_config) -> nn.Module:
    """Load a FLUX.2 Dev/Klein transformer from a single safetensors file."""
    try:
        from diffusers import Flux2Transformer2DModel
        from serenity.models.flux2_probe import flux2_config_dir, probe_flux2_variant
        from serenity.models.single_file_utils import (
            check_not_lora,
            format_load_diagnostic,
            strip_accelerate_hooks,
        )
    except ImportError as exc:
        raise RuntimeError(
            "FLUX.2 loading requires diffusers and Serenity's flux2 helpers."
        ) from exc

    check_not_lora(path)
    variant, config = probe_flux2_variant(path)
    logger.info("Loading FLUX.2 transformer from %s (detected=%s)", os.path.basename(path), variant)

    try:
        with flux2_config_dir(config) as config_path:
            try:
                model = Flux2Transformer2DModel.from_single_file(
                    path,
                    config=config_path,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=False,
                )
            except TypeError:
                # Older diffusers builds do not expose low_cpu_mem_usage.
                model = Flux2Transformer2DModel.from_single_file(
                    path,
                    config=config_path,
                    torch_dtype=dtype,
                )
    except Exception as exc:
        diag = format_load_diagnostic(path, f"flux2/{variant}", exc)
        raise RuntimeError(diag) from exc

    strip_accelerate_hooks(model)
    model.eval()
    model._serenity_prediction_type = "flow_flux"
    model._serenity_arch = model_config.architecture
    model._serenity_model_config = model_config
    model._serenity_flux2 = True
    model._serenity_flux2_variant = variant

    # Fix scaled FP8: from_single_file casts FP8→bf16 without applying scales.
    # Re-read original FP8 values + scales and write correct bf16 in-place.
    from serenityflow.bridge.fp8_dequant import has_fp8_scales as _has_fp8_scales
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            raw_sd = {k: f.get_tensor(k) for k in f.keys()}
        if _has_fp8_scales(raw_sd):
            from serenityflow.bridge.fp8_dequant import dequant_scaled_fp8_in_model
            fixed = dequant_scaled_fp8_in_model(model, path)
            logger.info("FLUX.2 FP8: fixed %d weights from scaled checkpoint", fixed)
        del raw_sd
    except Exception as exc:
        logger.warning("FP8 dequant check failed for FLUX.2: %s", exc)

    return model


def _load_zimage_transformer(path: str, dtype: torch.dtype, model_config) -> nn.Module:
    """Load a Z-Image transformer from a single file or diffusers directory."""
    try:
        from diffusers import ZImageTransformer2DModel
    except ImportError as exc:
        raise RuntimeError("Z-Image loading requires diffusers.") from exc

    candidate = Path(path).expanduser()
    if candidate.is_file() and candidate.suffix.lower() == ".safetensors":
        logger.info("Loading Z-Image transformer from %s", os.path.basename(path))
        model = ZImageTransformer2DModel.from_single_file(
            str(candidate),
            torch_dtype=dtype,
        )
    else:
        load_path = candidate
        subfolder = "transformer" if (load_path / "transformer").is_dir() else None
        logger.info(
            "Loading Z-Image transformer from %s (subfolder=%s)",
            load_path,
            subfolder,
        )
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "local_files_only": True,
        }
        if subfolder is not None:
            load_kwargs["subfolder"] = subfolder
        model = ZImageTransformer2DModel.from_pretrained(str(load_path), **load_kwargs)

    model.eval()
    model._serenity_prediction_type = "flow"
    model._serenity_arch = model_config.architecture
    model._serenity_model_config = model_config
    return model


def _resolve_flux2_vae_path(checkpoint_path: str) -> str | None:
    """Best-effort lookup for the companion FLUX.2 VAE."""
    checkpoint = Path(checkpoint_path).expanduser()
    candidates = [
        checkpoint.parent / "flux2-vae.safetensors",
        checkpoint.parent / "vae" / "flux2-vae.safetensors",
        checkpoint.parent / "vae" / "diffusion_pytorch_model.safetensors",
        checkpoint.parent.parent / "flux2-vae.safetensors",
        checkpoint.parent.parent / "vae" / "flux2-vae.safetensors",
        checkpoint.parent.parent / "vae" / "diffusion_pytorch_model.safetensors",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    try:
        from serenityflow.bridge.model_paths import get_model_paths

        return get_model_paths().find("flux2-vae.safetensors", "vae")
    except Exception:
        return None


def _resolve_flux_like_vae_config_dir(path: str) -> str | None:
    """Best-effort lookup for a diffusers VAE config directory for 16-channel Flux/Z-Image VAEs."""
    candidate = Path(path).expanduser()
    for local_dir in (candidate.parent, candidate.parent / "vae", candidate.parent.parent / "vae"):
        if (local_dir / "config.json").is_file():
            return str(local_dir)

    for repo_id in (
        "Tongyi-MAI/Z-Image-Turbo",
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-schnell",
    ):
        snapshot = _resolve_hf_snapshot_path(repo_id)
        if not snapshot:
            continue
        vae_dir = Path(snapshot) / "vae"
        if (vae_dir / "config.json").is_file():
            return str(vae_dir)
    return None


def _attach_flux2_vae_stats(model: nn.Module, checkpoint_path: str) -> None:
    """Attach BN stats from the FLUX.2 VAE so decode can denormalize latents."""
    vae_path = _resolve_flux2_vae_path(checkpoint_path)
    if vae_path is None:
        logger.debug("No companion FLUX.2 VAE found for %s", checkpoint_path)
        return

    try:
        from safetensors import safe_open

        with safe_open(vae_path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            if "bn.running_mean" not in keys or "bn.running_var" not in keys:
                return
            model._serenity_flux2_bn_mean = f.get_tensor("bn.running_mean")
            model._serenity_flux2_bn_var = f.get_tensor("bn.running_var")
            model._serenity_flux2_bn_eps = 1e-4
            logger.info("Attached FLUX.2 VAE BN stats from %s", os.path.basename(vae_path))
    except Exception:
        logger.debug("Failed to attach FLUX.2 VAE BN stats", exc_info=True)


def load_vae(path: str) -> VAEWrapper:
    """Load standalone VAE model."""
    s = _get()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_native_flux2_encoder = False

    try:
        from diffusers import AutoencoderKLFlux2 as DiffusersAutoencoderKLFlux2
        from diffusers.models import AutoencoderKL

        # Detect Flux-style VAE (16ch latent) by peeking at decoder.conv_in shape
        sd_peek = s["load_state_dict"](path)
        if _is_wan_vae_state_dict(sd_peek):
            return _load_wan_vae(sd_peek, device)

        is_flux_vae = False
        is_flux2_vae = False
        if "decoder.conv_in.weight" in sd_peek:
            latent_ch = sd_peek["decoder.conv_in.weight"].shape[1]
            is_flux_vae = (latent_ch == 16)
            is_flux2_vae = (latent_ch == 32)
        if "bn.running_mean" in sd_peek and "bn.running_var" in sd_peek:
            is_flux2_vae = True
        del sd_peek  # free memory

        if is_flux2_vae:
            try:
                vae = DiffusersAutoencoderKLFlux2.from_single_file(path, torch_dtype=torch.float32)
            except Exception:
                _ensure_serenity_inference_path()
                from models.vae_flux2 import AutoencoderKLFlux2 as NativeAutoencoderKLFlux2

                logger.info("Falling back to Serenity native FLUX.2 VAE loader for %s", os.path.basename(path))
                vae_sd = s["load_state_dict"](path)
                vae = NativeAutoencoderKLFlux2.from_state_dict(vae_sd)
                use_native_flux2_encoder = True
            vae = vae.to(device=device, dtype=torch.float32).eval()
            scaling = 1.0
        elif is_flux_vae:
            config_dir = _resolve_flux_like_vae_config_dir(path)
            if config_dir is not None:
                logger.info("Loading Flux-like VAE from %s with config %s", os.path.basename(path), config_dir)
                vae = AutoencoderKL.from_single_file(
                    path,
                    config=config_dir,
                    torch_dtype=torch.float32,
                )
                vae = vae.to(device=device, dtype=torch.float32).eval()
                scaling = float(getattr(vae.config, "scaling_factor", 0.3611) or 0.3611)
            else:
                vae = AutoencoderKL(
                    in_channels=3,
                    out_channels=3,
                    latent_channels=16,
                    block_out_channels=[128, 256, 512, 512],
                    layers_per_block=2,
                    act_fn="silu",
                    norm_num_groups=32,
                    scaling_factor=0.3611,
                )
                from safetensors.torch import load_file

                vae_sd = load_file(path)
                try:
                    from serenity.inference.models.convert import convert_ldm_vae_to_diffusers

                    vae_sd = convert_ldm_vae_to_diffusers(vae_sd)
                except Exception:
                    pass
                for k in list(vae_sd.keys()):
                    if (
                        hasattr(vae_sd[k], "ndim")
                        and vae_sd[k].ndim == 4
                        and vae_sd[k].shape[2:] == (1, 1)
                        and "attentions" in k
                    ):
                        vae_sd[k] = vae_sd[k].squeeze(-1).squeeze(-1)
                vae.load_state_dict(vae_sd, strict=False, assign=True)
                vae = vae.to(device=device, dtype=torch.float32).eval()
                scaling = 0.3611
        else:
            # Standard VAE (SD 1.5 / SDXL / etc.)
            from serenity.inference.models.convert import convert_ldm_vae_to_diffusers, safe_load_state_dict

            sd = s["load_state_dict"](path)
            if any(k.startswith("first_stage_model.") or k.startswith("vae.") for k in sd):
                sd = s["extract_vae"](sd)
                sd = convert_ldm_vae_to_diffusers(sd)
            elif any(k.startswith("encoder.") or k.startswith("decoder.") for k in sd):
                try:
                    sd = convert_ldm_vae_to_diffusers(sd)
                except Exception:
                    pass

            latent_ch = 4
            if "decoder.conv_in.weight" in sd:
                latent_ch = sd["decoder.conv_in.weight"].shape[1]

            vae = AutoencoderKL(latent_channels=latent_ch)
            safe_load_state_dict(vae, sd)
            vae = vae.to(device=device, dtype=torch.float32).eval()
            scaling = 0.18215  # SD default

        decoder = s["VAEDecoder"](vae_model=vae, dtype=torch.float32, device=device, scaling_factor=scaling)
        if use_native_flux2_encoder:
            encoder = _Flux2NativeVAEEncoder(vae, device=device, dtype=torch.float32, scaling_factor=scaling)
        else:
            encoder = s["VAEEncoder"](vae_model=vae, dtype=torch.float32, device=device, scaling_factor=scaling)
        return VAEWrapper(decoder, encoder)
    except Exception as e:
        raise RuntimeError(f"Failed to load VAE from {path}: {e}") from e


class _Flux2NativeVAEEncoder:
    """Encoder adapter for Serenity's native FLUX.2 VAE implementation."""

    __slots__ = ("_model", "_device", "_dtype", "scaling_factor")

    def __init__(self, vae_model: nn.Module, device: str, dtype: torch.dtype, scaling_factor: float) -> None:
        self._model = vae_model
        self._device = device
        self._dtype = dtype
        self.scaling_factor = scaling_factor

    @torch.inference_mode()
    def encode(self, image: torch.Tensor, tiling: bool = False, tile_size: int = 512) -> torch.Tensor:
        del tiling, tile_size

        image = image.to(device=self._device, dtype=self._dtype)
        encoded = self._model.encode((image * 2.0) - 1.0)
        if hasattr(encoded, "latent_dist"):
            encoded = encoded.latent_dist.sample()
        elif hasattr(encoded, "sample"):
            encoded = encoded.sample

        if isinstance(encoded, torch.Tensor) and encoded.ndim == 4 and encoded.shape[1] == 64:
            encoded = encoded[:, :32]

        return encoded * self.scaling_factor


class _WanVAEDecoder:
    """Adapter that exposes Wan-style VAEs through the bridge decode API."""

    __slots__ = ("_model", "_device", "_dtype", "_latents_mean", "_latents_std")

    def __init__(self, model: nn.Module, device: str, dtype: torch.dtype) -> None:
        self._model = model
        self._device = device
        self._dtype = dtype
        self._latents_mean = torch.tensor(
            model.config.latents_mean,
            device=device,
            dtype=dtype,
        ).view(1, -1, 1, 1, 1)
        self._latents_std = torch.tensor(
            model.config.latents_std,
            device=device,
            dtype=dtype,
        ).view(1, -1, 1, 1, 1)

    def decode(self, latents: torch.Tensor, tiling: bool = False, tile_size: int = 512) -> torch.Tensor:
        del tiling, tile_size

        z = latents
        squeeze_time = False
        if z.ndim == 4:
            z = z.unsqueeze(2)
            squeeze_time = True

        z = z.to(device=self._device, dtype=self._dtype)
        if z.shape[1] == self._latents_mean.shape[1]:
            z = z * self._latents_std + self._latents_mean

        decoded = self._model.decode(z)
        if squeeze_time and decoded.ndim == 5 and decoded.shape[2] == 1:
            decoded = decoded.squeeze(2)
        return ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)


class _WanVAEEncoder:
    """Adapter that exposes Wan-style VAEs through the bridge encode API."""

    __slots__ = ("_model", "_device", "_dtype", "_latents_mean", "_latents_std")

    def __init__(self, model: nn.Module, device: str, dtype: torch.dtype) -> None:
        self._model = model
        self._device = device
        self._dtype = dtype
        self._latents_mean = torch.tensor(
            model.config.latents_mean,
            device=device,
            dtype=dtype,
        ).view(1, -1, 1, 1, 1)
        self._latents_std = torch.tensor(
            model.config.latents_std,
            device=device,
            dtype=dtype,
        ).view(1, -1, 1, 1, 1)

    def encode(self, image: torch.Tensor, tiling: bool = False, tile_size: int = 512) -> torch.Tensor:
        del tiling, tile_size

        x = image
        squeeze_time = False
        if x.ndim == 4:
            x = x.unsqueeze(2)
            squeeze_time = True

        x = (x.to(device=self._device, dtype=self._dtype) * 2.0) - 1.0
        latents = self._model.encode(x)
        if latents.shape[1] == self._latents_mean.shape[1]:
            latents = (latents - self._latents_mean) / self._latents_std
        if squeeze_time and latents.ndim == 5 and latents.shape[2] == 1:
            latents = latents.squeeze(2)
        return latents


def _is_wan_vae_state_dict(state_dict: dict[str, Any]) -> bool:
    """Detect Wan/Qwen image VAE checkpoints by their 3D convolution weights."""
    for key in (
        "post_quant_conv.weight",
        "decoder.conv_in.weight",
        "conv1.weight",
        "decoder.conv1.weight",
    ):
        weight = state_dict.get(key)
        if hasattr(weight, "ndim") and weight.ndim == 5:
            return True
    return False


def _load_wan_vae(state_dict: dict[str, Any], device: str) -> VAEWrapper:
    """Load Wan/Qwen-style VAE checkpoints with latent mean/std normalization."""
    _ensure_serenity_inference_path()
    from models.vae_wan import WanVAE

    compute_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    vae = WanVAE.from_state_dict(state_dict)
    vae = vae.to(device=device, dtype=compute_dtype).eval()
    decoder = _WanVAEDecoder(vae, device, compute_dtype)
    encoder = _WanVAEEncoder(vae, device, compute_dtype) if vae.encoder is not None else None
    return VAEWrapper(decoder, encoder)


def load_clip(path: str, clip_type: str = "stable_diffusion") -> CLIPWrapper:
    """Load CLIP text encoder from file."""
    s = _get()
    # Text encoders stay on CPU — moved to GPU only during encode_text()
    device = "cpu"
    arch = _resolve_single_clip_architecture(path, clip_type, s["ModelArchitecture"])
    external_mode = _resolve_external_text_mode(path, clip_type)
    if external_mode is not None:
        external_dtype = torch.bfloat16 if external_mode in {"flux2", "qwen"} else torch.float16
        return _load_external_clip(path, external_mode, arch, external_dtype, device)

    from serenity.inference.text.encoders import TextEncoderType, get_required_encoders, get_default_encoder_path
    required = get_required_encoders(arch)

    text_mgr = s["TextEncoderManager"]()
    for enc_type in required:
        local_file = _match_encoder_file(enc_type, [path])
        if local_file:
            local_dir = _resolve_text_encoder_dir(local_file)
            if enc_type in (TextEncoderType.QWEN, TextEncoderType.GEMMA) and local_dir:
                text_mgr.load_encoder(
                    enc_type,
                    model_path=local_dir,
                    dtype=torch.float16,
                    device=device,
                )
            elif enc_type not in (TextEncoderType.QWEN, TextEncoderType.GEMMA):
                _load_encoder_from_safetensors(
                    text_mgr,
                    enc_type,
                    local_file,
                    dtype=torch.float16,
                    device=device,
                )
            else:
                enc_path = get_default_encoder_path(arch, enc_type)
                if enc_path:
                    text_mgr.load_encoder(enc_type, model_path=enc_path.repo,
                                          dtype=torch.float16, device=device,
                                          subfolder=enc_path.subfolder,
                                          tokenizer_subfolder=enc_path.tokenizer_subfolder)
        else:
            enc_path = get_default_encoder_path(arch, enc_type)
            if enc_path:
                text_mgr.load_encoder(enc_type, model_path=enc_path.repo,
                                      dtype=torch.float16, device=device,
                                      subfolder=enc_path.subfolder,
                                      tokenizer_subfolder=enc_path.tokenizer_subfolder)

    return CLIPWrapper(text_mgr, arch, torch.float16, device)


class _ExternalTextManager:
    """Minimal text-manager facade for modern local encoder implementations."""

    def __init__(self, encoder, mode: str):
        self._encoder = encoder
        self._mode = mode

    def encode_for_model(self, _arch, prompt: str, negative: str = "", clip_skip: int = 0) -> dict[str, Any]:
        del negative, clip_skip

        attention_mask = None
        if self._mode == "qwen":
            out, attention_mask = self._encoder.encode(prompt, task_type="image")
        else:
            out = self._encoder.encode(prompt)

        return {
            "cond": out.hidden_states,
            "pooled": getattr(out, "pooled_output", None),
            "attention_mask": attention_mask,
        }


def _ensure_serenity_inference_path() -> Path:
    """Make the local serenity-inference repo importable."""
    root = Path.home() / "serenity-inference"
    if not root.exists():
        raise FileNotFoundError(
            f"Required local encoder repo not found: {root}"
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _resolve_hf_snapshot_path(repo_id: str) -> str | None:
    """Resolve a HuggingFace repo ID to a local snapshot path if cached."""
    repo_dir = Path.home() / ".cache" / "huggingface" / "hub" / (
        "models--" + repo_id.replace("/", "--")
    )
    if not repo_dir.exists():
        return None

    refs_main = repo_dir / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text().strip()
        snapshot = repo_dir / "snapshots" / revision
        if snapshot.exists():
            return str(snapshot)

    snapshots_dir = repo_dir / "snapshots"
    snapshots = sorted(snapshots_dir.glob("*")) if snapshots_dir.exists() else []
    if snapshots:
        return str(snapshots[-1])
    return None


def _resolve_external_text_mode(path: str, clip_type: str) -> str | None:
    """Choose an external text encoder implementation when required."""
    lowered = (clip_type or "").lower()
    basename = os.path.basename(path).lower()
    if lowered in {"klein", "flux2_klein"}:
        return "klein"
    if lowered in {"flux2", "mistral"}:
        return "flux2"
    if lowered == "zimage":
        return "zimage"
    if lowered == "qwen":
        return "qwen"
    if lowered == "lumina2" and "qwen" in basename:
        return "qwen"
    return None


def _resolve_external_text_encoder_path(path: str, mode: str) -> str:
    """Resolve local encoder assets for the external text backends."""
    candidate = Path(path).expanduser()
    if candidate.exists():
        if mode == "qwen" and candidate.is_file():
            return str(candidate)
        if mode == "flux2":
            if candidate.is_dir() and (candidate / "text_encoder").is_dir():
                return str(candidate / "text_encoder")
            if candidate.is_dir():
                return str(candidate)
        elif mode == "qwen":
            if candidate.is_dir():
                return str(candidate)
        else:
            return str(candidate)

    lowered = path.lower()
    repo_id = None
    if mode == "klein":
        repo_id = "Qwen/Qwen3-8B" if "8b" in lowered or "9b" in lowered else "Qwen/Qwen3-4B"
    elif mode == "zimage":
        repo_id = "Qwen/Qwen3-4B"
    elif mode == "qwen":
        repo_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    elif mode == "flux2":
        repo_id = "black-forest-labs/FLUX.2-dev"

    if repo_id is None:
        # Try local Qwen encoder directory before falling back to HF snapshot
        local_qwen = _resolve_local_qwen_encoder_path(os.path.basename(path))
        if local_qwen is not None:
            return local_qwen
        return path

    snapshot = _resolve_hf_snapshot_path(repo_id)
    if snapshot is None:
        return path

    snapshot_path = Path(snapshot)
    if mode == "flux2":
        text_dir = snapshot_path / "text_encoder"
        if text_dir.is_dir():
            return str(text_dir)
    return str(snapshot_path)


def _load_external_clip(path: str, mode: str, arch, dtype: torch.dtype, device: str) -> CLIPWrapper:
    """Load a corrected local text encoder for modern model families."""
    _ensure_serenity_inference_path()
    resolved_path = _resolve_external_text_encoder_path(path, mode)

    if mode == "klein":
        from text.qwen3 import Qwen3Encoder

        encoder = Qwen3Encoder(mode="klein", dtype=dtype, device=device)
    elif mode == "zimage":
        from text.qwen3 import Qwen3Encoder

        encoder = Qwen3Encoder(mode="zimage", dtype=dtype, device=device)
    elif mode == "qwen":
        from text.qwen25vl import Qwen25VLEncoder

        encoder = Qwen25VLEncoder(dtype=torch.bfloat16, device=device)
    elif mode == "flux2":
        from text.mistral import MistralEncoder

        encoder = MistralEncoder(dtype=dtype, device=device)
    else:
        raise ValueError(f"Unsupported external text mode: {mode}")

    logger.info("Loading external %s text encoder from %s", mode, resolved_path)
    encoder.load(resolved_path)
    return CLIPWrapper(_ExternalTextManager(encoder, mode), arch, dtype, device)


def _resolve_local_qwen_encoder_path(filename: str) -> str | None:
    """Return a local clip path for the named Qwen encoder when available."""
    try:
        from serenityflow.bridge.model_paths import get_model_paths
    except ImportError:
        return None

    try:
        return get_model_paths().find(filename, "clip")
    except Exception:
        return None


def _resolve_single_clip_architecture(path: str, clip_type: str, model_architecture) -> Any:
    """Map CLIP loader type strings to Serenity architectures.

    The workflow surface uses broader labels like `flux`, `wan`, and
    `lumina2`; resolve them here so shipped workflows hit the right encoder
    stack.
    """
    lowered = (clip_type or "").lower()
    basename = os.path.basename(path).lower()
    arch_map = {
        "stable_diffusion": model_architecture.SD15,
        "sdxl": model_architecture.SDXL,
        "sd3": model_architecture.SD3,
        "flux": model_architecture.FLUX_DEV,
        "flux2": model_architecture.FLUX_DEV,
        "klein": model_architecture.FLUX_2_KLEIN_4B,
        "wan": model_architecture.WAN,
        "qwen": model_architecture.QWEN,
        "zimage": model_architecture.ZIMAGE,
        "lumina": model_architecture.LUMINA,
    }
    if lowered == "stable_diffusion":
        if "qwen" in basename:
            return model_architecture.QWEN
        if "zimage" in basename or "z_image" in basename:
            return model_architecture.ZIMAGE
    if lowered == "lumina2":
        if "qwen" in basename:
            return model_architecture.QWEN
        if "zimage" in basename:
            return model_architecture.ZIMAGE
        return model_architecture.LUMINA
    return arch_map.get(lowered, model_architecture.SD15)


def _resolve_text_encoder_dir(path: str) -> str | None:
    """Return a directory suitable for HF/transformers loading if one exists."""
    if os.path.isdir(path):
        return path
    stem = os.path.splitext(path)[0]
    if os.path.isdir(stem):
        return stem
    return None


def _load_qwen_transformer(path: str, dtype: torch.dtype, model_config) -> nn.Module:
    """Load a Qwen transformer from a single safetensors file.

    Serenity's low-level inference loader still marks Qwen as unsupported.
    Use the native model helper when a local Qwen-Image snapshot is present.
    """
    try:
        from diffusers import QwenImageTransformer2DModel
        from serenity.models.qwen import _find_qwen_cache_snapshot
    except ImportError as exc:
        raise RuntimeError(
            "Qwen loading requires diffusers and Serenity's native qwen helpers."
        ) from exc

    cache_snapshot = _find_qwen_cache_snapshot()
    if cache_snapshot is None:
        raise RuntimeError(
            f"Cannot load Qwen model from {path}: no local Qwen-Image pipeline snapshot found."
        )

    model = QwenImageTransformer2DModel.from_single_file(
        path,
        config=str(cache_snapshot),
        subfolder="transformer",
        torch_dtype=dtype,
    )
    model.eval()
    model._serenity_prediction_type = "flow"
    model._serenity_arch = model_config.architecture
    model._serenity_model_config = model_config
    model._serenity_edit_mode = "edit" in os.path.basename(path).lower()
    return model


def _match_encoder_file(enc_type, files: list[str]) -> str | None:
    """Match an encoder type to a local file by naming convention."""
    import os
    from serenity.inference.text.encoders import TextEncoderType

    patterns = {
        TextEncoderType.CLIP_L: ["clip_l", "clip-l"],
        TextEncoderType.CLIP_G: ["clip_g", "clip-g"],
        TextEncoderType.T5_XXL: ["t5xxl", "t5_xxl", "t5-xxl"],
        TextEncoderType.QWEN: ["qwen"],
        TextEncoderType.GEMMA: ["gemma"],
    }
    for f in files:
        basename = os.path.basename(f).lower()
        for pattern in patterns.get(enc_type, []):
            if pattern in basename:
                return f
    return None


def _load_encoder_from_safetensors(text_mgr, enc_type, filepath: str,
                                    dtype, device: str) -> None:
    """Load a text encoder from a local safetensors file.

    Uses config from HF cache + state dict from local file.
    """
    from safetensors.torch import load_file
    from serenity.inference.text.encoders import TextEncoderType

    sd = load_file(filepath)

    if enc_type in (TextEncoderType.CLIP_L, TextEncoderType.CLIP_G):
        _load_clip_from_sd(text_mgr, enc_type, sd, dtype, device)
    elif enc_type == TextEncoderType.T5_XXL:
        _load_t5_from_sd(text_mgr, sd, dtype, device)
    else:
        logger.warning("No local-file loader for %s, skipping", enc_type)


def _load_clip_from_sd(text_mgr, enc_type, sd: dict, dtype, device: str) -> None:
    """Load CLIP encoder from a state dict."""
    from transformers import CLIPTextModel, CLIPTokenizer
    from serenity.inference.text.encoders import TextEncoderType

    encoder = text_mgr.get_encoder(enc_type)
    if encoder.is_loaded:
        return

    logger.info("Loading %s from local safetensors (%d keys)", enc_type.value, len(sd))

    # Tokenizer from LOCAL HF cache — local_files_only=True prevents any download
    if enc_type == TextEncoderType.CLIP_L:
        tok_repo = "openai/clip-vit-large-patch14"
        tok_kwargs = {}
    else:
        tok_repo = "stabilityai/stable-diffusion-xl-base-1.0"
        tok_kwargs = {"subfolder": "tokenizer_2"}
    try:
        encoder._tokenizer = CLIPTokenizer.from_pretrained(tok_repo, local_files_only=True, **tok_kwargs)
    except Exception as e:
        raise RuntimeError(f"CLIP tokenizer not in local cache ({tok_repo}). Run once with network to cache, or copy vocab.json+merges.txt manually. Error: {e}")

    # Model config inferred from state dict shapes — NO from_pretrained
    if enc_type == TextEncoderType.CLIP_G:
        from transformers import CLIPTextModelWithProjection, CLIPTextConfig
        # CLIP-G: 1280 hidden, 32 heads, 20 layers, 77 max_position
        config = CLIPTextConfig(
            hidden_size=1280, intermediate_size=5120, num_hidden_layers=32,
            num_attention_heads=20, projection_dim=1280,
            max_position_embeddings=77, vocab_size=49408,
        )
        model = CLIPTextModelWithProjection(config)
    else:
        from transformers import CLIPTextConfig
        # CLIP-L: 768 hidden, 12 heads, 12 layers, 77 max_position
        config = CLIPTextConfig(
            hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
            num_attention_heads=12, projection_dim=768,
            max_position_embeddings=77, vocab_size=49408,
        )
        model = CLIPTextModel(config)

    model.load_state_dict(sd, strict=False, assign=True)
    model = model.to(device=device, dtype=dtype).eval()
    encoder._model = model
    encoder._dtype = dtype
    encoder._device = device


def _load_t5_from_sd(text_mgr, sd: dict, dtype, device: str) -> None:
    """Load T5 encoder from a state dict."""
    from transformers import AutoTokenizer, T5EncoderModel, AutoConfig
    from serenity.inference.text.encoders import TextEncoderType

    encoder = text_mgr.get_encoder(TextEncoderType.T5_XXL)
    if encoder.is_loaded:
        return

    logger.info("Loading T5-XXL from local safetensors (%d keys)", len(sd))

    # Tokenizer from LOCAL HF cache — local_files_only=True prevents any download
    try:
        encoder._tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xxl", local_files_only=True)
    except Exception as e:
        raise RuntimeError(f"T5 tokenizer not in local cache. Run once with network to cache, or copy spiece.model manually. Error: {e}")

    # Model config from known T5-XXL architecture — NO from_pretrained
    from transformers import T5Config
    config = T5Config(
        d_model=4096, d_ff=10240, d_kv=64,
        num_heads=64, num_layers=24,
        vocab_size=32128, dense_act_fn="gelu_new",
        feed_forward_proj="gated-gelu",
        is_encoder_decoder=False,
    )
    model = T5EncoderModel(config)
    model.load_state_dict(sd, strict=False, assign=True)
    model = model.to(device=device, dtype=dtype).eval()
    encoder._model = model
    encoder._dtype = dtype
    encoder._device = device


def load_dual_clip(path1: str, path2: str, clip_type: str = "flux") -> CLIPWrapper:
    """Load dual CLIP (e.g., CLIP-L + T5-XXL for FLUX).

    Uses local safetensors files when provided, falling back to HF repos.
    Text encoders load on CPU to avoid VRAM pressure from the diffusion model.
    """
    s = _get()
    # Text encoders stay on CPU — moved to GPU only during encode_text()
    device = "cpu"
    arch_map = {
        "flux": s["ModelArchitecture"].FLUX_DEV,
        "sd3": s["ModelArchitecture"].SD3,
        "sdxl": s["ModelArchitecture"].SDXL,
    }
    arch = arch_map.get(clip_type, s["ModelArchitecture"].FLUX_DEV)

    from serenity.inference.text.encoders import TextEncoderType, get_required_encoders
    required = get_required_encoders(arch)

    text_mgr = s["TextEncoderManager"]()

    # Map file paths to encoder types based on naming conventions
    files = [path1, path2]
    for enc_type in required:
        local_file = _match_encoder_file(enc_type, files)
        if local_file:
            _load_encoder_from_safetensors(text_mgr, enc_type, local_file,
                                           dtype=torch.float16, device=device)
        else:
            # Fall back to HF repo
            from serenity.inference.text.encoders import get_default_encoder_path
            enc_path = get_default_encoder_path(arch, enc_type)
            if enc_path:
                text_mgr.load_encoder(enc_type, model_path=enc_path.repo,
                                      dtype=torch.float16, device=device,
                                      subfolder=enc_path.subfolder,
                                      tokenizer_subfolder=enc_path.tokenizer_subfolder)

    return CLIPWrapper(text_mgr, arch, torch.float16, device)


def load_controlnet(path: str):
    """Load ControlNet model. Returns the state dict + metadata."""
    s = _get()
    sd = s["load_state_dict"](path)
    return {"state_dict": sd, "path": path}


def load_clip_vision(path: str):
    """Load CLIP vision encoder."""
    s = _get()
    sd = s["load_state_dict"](path)
    return {"state_dict": sd, "path": path}


# ---------------------------------------------------------------------------
# Text Encoding
# ---------------------------------------------------------------------------


def encode_text(clip: CLIPWrapper, text: str) -> list[dict]:
    """Encode text using CLIP. Returns conditioning in ComfyUI format.

    Each entry is a dict with 'cross_attn' and optionally 'pooled_output'.
    """
    try:
        result = clip._manager.encode_for_model(
            clip._arch, text, "",
            clip_skip=clip.clip_skip,
        )
    except Exception as e:
        logger.warning("Text encoding failed: %s. Using zeros.", e, exc_info=True)
        return [{"cross_attn": torch.zeros(1, 77, 768), "pooled_output": torch.zeros(1, 768)}]

    cond = result.get("cond")
    pooled = result.get("pooled")
    if pooled is None:
        pooled = result.get("pooled_output")
    attention_mask = result.get("attention_mask")

    entry = {}
    if cond is not None:
        entry["cross_attn"] = cond
    else:
        entry["cross_attn"] = torch.zeros(1, 77, 768)

    if pooled is not None:
        entry["pooled_output"] = pooled
    if attention_mask is not None:
        entry["attention_mask"] = attention_mask

    # Qwen/Gemma-based image models need the prompt attention mask to recover
    # effective text sequence lengths during sampling.
    try:
        if "attention_mask" in entry:
            raise StopIteration
        from serenity.inference.text.encoders import TextEncoderType

        encoder_type = None
        if getattr(clip, "_arch", None) == clip._arch.__class__.QWEN:
            encoder_type = TextEncoderType.QWEN
        elif getattr(clip, "_arch", None) in (clip._arch.__class__.LUMINA, clip._arch.__class__.ZIMAGE):
            encoder_type = TextEncoderType.GEMMA

        if encoder_type is not None:
            encoder = clip._manager.get_encoder(encoder_type)
            tokenizer = getattr(encoder, "_tokenizer", None)
            if tokenizer is not None:
                tokens = tokenizer(
                    text,
                    max_length=256,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                if "attention_mask" in tokens:
                    entry["attention_mask"] = tokens["attention_mask"]
    except StopIteration:
        pass
    except Exception:
        logger.debug("Prompt attention mask inference skipped", exc_info=True)

    # Extra conditioning (FLUX clip_cond, etc.)
    for key in ("clip_cond", "neg_pooled"):
        if key in result:
            entry[key] = result[key]

    return [entry]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _extract_model_output(raw_out):
    """Extract tensor from model output (may be tuple or ModelOutput)."""
    if isinstance(raw_out, torch.Tensor):
        return raw_out
    if hasattr(raw_out, "sample"):
        return raw_out.sample
    if isinstance(raw_out, tuple):
        return raw_out[0]
    return raw_out


def _is_qwen_transformer(model: nn.Module) -> bool:
    """Best-effort detection for diffusers' Qwen image transformer."""
    return "qwen" in model.__class__.__name__.lower()


def _is_zimage_transformer(model: nn.Module) -> bool:
    """Best-effort detection for diffusers' Z-Image transformer."""
    class_name = model.__class__.__name__.lower()
    if "zimage" in class_name or "z_image" in class_name:
        return True
    arch = getattr(model, "_serenity_arch", None)
    return arch is not None and "zimage" in str(arch).lower()


def _pack_flux_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack FLUX-style BCHW latents to sequence tokens."""
    batch, channels, height, width = latents.shape
    packed = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    packed = packed.permute(0, 2, 4, 1, 3, 5)
    return packed.reshape(batch, (height // 2) * (width // 2), channels * 4)


def _unpack_flux_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Unpack FLUX-style sequence tokens to BCHW latents."""
    batch, _seq, channels = latents.shape
    latent_channels = channels // 4
    result = latents.view(batch, height // 2, width // 2, latent_channels, 2, 2)
    result = result.permute(0, 3, 1, 4, 2, 5)
    return result.reshape(batch, latent_channels, height, width)


def _patchify_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    """Patchify FLUX.2 latents from 32 channels to 128 channels."""
    batch, channels, height, width = latents.shape
    if channels == 128:
        return latents
    if channels != 32:
        raise ValueError(f"FLUX.2 latents must have 32 or 128 channels, got {channels}")
    patched = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    patched = patched.permute(0, 1, 3, 5, 2, 4)
    return patched.reshape(batch, channels * 4, height // 2, width // 2)


def _unpatchify_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    """Reverse FLUX.2 patchification back to 32-channel VAE latents."""
    batch, channels, height, width = latents.shape
    if channels == 32:
        return latents
    if channels != 128:
        raise ValueError(f"FLUX.2 patchified latents must have 32 or 128 channels, got {channels}")
    unpacked = latents.view(batch, 32, 2, 2, height, width)
    unpacked = unpacked.permute(0, 1, 4, 2, 5, 3)
    return unpacked.reshape(batch, 32, height * 2, width * 2)


def _pack_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    """Flatten patchified FLUX.2 spatial latents to a token sequence."""
    batch, channels, height, width = latents.shape
    return latents.permute(0, 2, 3, 1).reshape(batch, height * width, channels)


def _unpack_flux2_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Restore FLUX.2 patchified spatial layout from a token sequence."""
    batch, _seq, channels = latents.shape
    return latents.view(batch, height, width, channels).permute(0, 3, 1, 2)


def _prepare_flux_image_ids(height: int, width: int, device, dtype, t_offset: float = 0.0) -> torch.Tensor:
    """Create FLUX RoPE image ids for a packed latent grid."""
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 0] = t_offset
    latent_image_ids[..., 1] = torch.arange(height)[:, None]
    latent_image_ids[..., 2] = torch.arange(width)[None, :]
    return latent_image_ids.reshape(height * width, 3).to(device=device, dtype=dtype)


def _prepare_flux2_image_ids(height: int, width: int, device, dtype, t_offset: float = 0.0) -> torch.Tensor:
    """Create FLUX.2 / Klein RoPE ids for a patchified latent grid."""
    h_pos = torch.arange(height, device=device, dtype=dtype)
    w_pos = torch.arange(width, device=device, dtype=dtype)
    h_grid, w_grid = torch.meshgrid(h_pos, w_pos, indexing="ij")
    t_coord = torch.full_like(h_grid, t_offset)
    l_coord = torch.zeros_like(h_grid)
    return torch.stack([t_coord, h_grid, w_grid, l_coord], dim=-1).reshape(height * width, 4)


def _prepare_flux2_text_ids(text_seq_len: int, device, dtype) -> torch.Tensor:
    """Create FLUX.2 text ids following ComfyUI's txt_ids_dims=[3] convention."""
    txt_ids = torch.zeros((text_seq_len, 4), device=device, dtype=dtype)
    if text_seq_len > 0:
        txt_ids[:, 3] = torch.linspace(
            0,
            text_seq_len - 1,
            steps=text_seq_len,
            device=device,
            dtype=dtype,
        )
    return txt_ids


def _pack_qwen_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack Qwen latents `[B,C,1,H,W]` to sequence tokens."""
    batch, channels, frames, height, width = latents.shape
    if frames != 1:
        raise ValueError(f"Qwen latents must have exactly one frame, got {frames}")
    packed = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    packed = packed.permute(0, 2, 4, 1, 3, 5)
    return packed.reshape(batch, (height // 2) * (width // 2), channels * 4)


def _unpack_qwen_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Unpack Qwen sequence tokens to `[B,C,1,H,W]` latents."""
    batch, _seq, channels = latents.shape
    result = latents.view(batch, height // 2, width // 2, channels // 4, 2, 2)
    result = result.permute(0, 3, 1, 4, 2, 5)
    return result.reshape(batch, channels // 4, 1, height, width)


def _run_sampling(
    model: nn.Module,
    latent: torch.Tensor,
    positive: list[dict],
    negative: list[dict],
    cfg: float,
    sampler_name: str,
    sigmas: torch.Tensor,
    *,
    seed: int = 0,
    add_noise: bool = True,
    noise: torch.Tensor | None = None,
    denoise: float = 1.0,
) -> torch.Tensor:
    """Shared sampler implementation for KSampler and custom sampler nodes."""
    s = _get()
    device = latent.device if latent.is_cuda else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    cond_entry = positive[0] if positive else {}
    uncond_entry = negative[0] if negative else {}

    cond_tensor = cond_entry.get("cross_attn")
    uncond_tensor = uncond_entry.get("cross_attn")
    pooled_cond = cond_entry.get("pooled_output")
    pooled_uncond = uncond_entry.get("pooled_output")
    cond_mask = cond_entry.get("attention_mask")
    uncond_mask = uncond_entry.get("attention_mask")
    reference_latent = cond_entry.get("reference_latent")
    if reference_latent is None:
        reference_latent = uncond_entry.get("reference_latent")

    if cond_tensor is not None:
        cond_tensor = cond_tensor.to(device)
    if uncond_tensor is not None:
        uncond_tensor = uncond_tensor.to(device)
    if pooled_cond is not None:
        pooled_cond = pooled_cond.to(device)
    if pooled_uncond is not None:
        pooled_uncond = pooled_uncond.to(device)
    if cond_mask is not None:
        cond_mask = cond_mask.to(device)
    if uncond_mask is not None:
        uncond_mask = uncond_mask.to(device)

    logger.info(
        "Conditioning shapes: cond=%s, uncond=%s, pooled_cond=%s, pooled_uncond=%s",
        cond_tensor.shape if cond_tensor is not None else None,
        uncond_tensor.shape if uncond_tensor is not None else None,
        pooled_cond.shape if pooled_cond is not None else None,
        pooled_uncond.shape if pooled_uncond is not None else None,
    )

    prediction_type = "eps"
    prediction_kwargs = {}
    if hasattr(model, "_serenity_prediction_type"):
        prediction_type = model._serenity_prediction_type
    elif _is_qwen_transformer(model) or _is_zimage_transformer(model):
        prediction_type = "flow"
    if hasattr(model, "_serenity_model_config"):
        from serenity.inference.models.loader import _get_adapter
        try:
            adapter = _get_adapter(model._serenity_model_config)
        except Exception:
            adapter = None
        if adapter:
            prediction_kwargs = adapter.get_prediction_kwargs()

    is_qwen = _is_qwen_transformer(model)
    is_zimage = _is_zimage_transformer(model)
    is_flux = not is_qwen and not is_zimage and hasattr(model, "config") and hasattr(model.config, "joint_attention_dim")
    is_flux2 = bool(getattr(model, "_serenity_flux2", False))
    is_flux1 = is_flux and not is_flux2

    base_latent = latent.float()
    qwen_input_was_4d = False
    if is_qwen and base_latent.ndim == 4:
        qwen_input_was_4d = True
        base_latent = base_latent.unsqueeze(2)

    if prediction_type == "flow_flux" and base_latent.ndim == 4:
        if is_flux2:
            patched = _patchify_flux2_latents(base_latent)
            actual_seq_len = patched.shape[-2] * patched.shape[-1]
            lat_h, lat_w = patched.shape[-2:]
        else:
            _, _, lat_h, lat_w = base_latent.shape
            actual_seq_len = (lat_h // 2) * (lat_w // 2)
        prediction_kwargs = dict(prediction_kwargs)
        prediction_kwargs["seq_len"] = actual_seq_len
        logger.info("Flux seq_len=%d (from %dx%d latent)", actual_seq_len, lat_h, lat_w)

    prediction = s["get_prediction"](prediction_type, **prediction_kwargs)
    logger.info(
        "Sample: prediction=%s, is_flux=%s, is_qwen=%s, is_zimage=%s, latent=%s, cfg=%.1f",
        prediction_type,
        is_flux,
        is_qwen,
        is_zimage,
        tuple(base_latent.shape),
        cfg,
    )

    sigmas = sigmas.to(device)
    if prediction_type == "flow_flux" and hasattr(prediction, "apply_sigma_shift"):
        sigma_body = prediction.apply_sigma_shift(sigmas[:-1])
        sigmas = torch.cat([sigma_body, sigmas[-1:]])

    if noise is None:
        noise = s["create_noise"](
            seed=seed,
            shape=base_latent.shape,
            device="cpu",
            dtype=torch.float32,
        )
    if is_qwen and noise.ndim == 4:
        noise = noise.unsqueeze(2)

    if add_noise:
        if prediction_type in ("flow", "flow_flux"):
            noisy_latent = prediction.noise_scaling(
                sigmas[0].cpu(),
                noise,
                base_latent.cpu(),
                max_denoise=(denoise >= 1.0),
            ).to(device)
        else:
            noisy_latent = (noise * sigmas[0].cpu()).to(device)
    else:
        noisy_latent = base_latent.to(device).float()

    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32

    log_sigmas = None
    if not is_flux and not is_qwen and not is_zimage:
        betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        log_sigmas = ((1 - alphas_cumprod) / alphas_cumprod).sqrt().log().to(device)

    flux_h = flux_w = None
    flux_patch_h = flux_patch_w = None
    flux_img_ids = flux_txt_ids = None
    flux_condition_tokens = None
    if is_flux:
        flux_h = int(base_latent.shape[-2])
        flux_w = int(base_latent.shape[-1])
        txt_len = cond_tensor.shape[1] if cond_tensor is not None else 512
        if is_flux2:
            patched_latent = _patchify_flux2_latents(noisy_latent)
            flux_patch_h = int(patched_latent.shape[-2])
            flux_patch_w = int(patched_latent.shape[-1])
            noisy_latent = _pack_flux2_latents(patched_latent)
            flux_img_ids = _prepare_flux2_image_ids(flux_patch_h, flux_patch_w, device, model_dtype)
            flux_txt_ids = _prepare_flux2_text_ids(txt_len, device, model_dtype)
        else:
            noisy_latent = _pack_flux_latents(noisy_latent)
            flux_patch_h = flux_h // 2
            flux_patch_w = flux_w // 2
            flux_img_ids = _prepare_flux_image_ids(flux_patch_h, flux_patch_w, device, model_dtype)
            flux_txt_ids = torch.zeros(txt_len, 3, device=device, dtype=model_dtype)

        if isinstance(reference_latent, torch.Tensor):
            ref_latent = reference_latent.float()
            if ref_latent.ndim == 5 and ref_latent.shape[2] == 1:
                ref_latent = ref_latent.squeeze(2)
            if ref_latent.ndim == 4 and ref_latent.shape[-2:] == (flux_h, flux_w):
                if is_flux2:
                    ref_latent = _patchify_flux2_latents(ref_latent)
                    ref_patch_h = int(ref_latent.shape[-2])
                    ref_patch_w = int(ref_latent.shape[-1])
                    flux_condition_tokens = _pack_flux2_latents(ref_latent.to(device))
                    ref_img_ids = _prepare_flux2_image_ids(
                        ref_patch_h,
                        ref_patch_w,
                        device,
                        model_dtype,
                        t_offset=10.0,
                    )
                else:
                    flux_condition_tokens = _pack_flux_latents(ref_latent.to(device))
                    ref_img_ids = _prepare_flux_image_ids(
                        flux_patch_h,
                        flux_patch_w,
                        device,
                        model_dtype,
                        t_offset=10.0,
                    )
                flux_img_ids = torch.cat([flux_img_ids, ref_img_ids], dim=0)

    qwen_h = qwen_w = None
    qwen_img_shapes = None
    qwen_condition_tokens = None
    if is_qwen:
        qwen_h = int(base_latent.shape[-2])
        qwen_w = int(base_latent.shape[-1])
        noisy_latent = _pack_qwen_latents(noisy_latent)
        qwen_img_shapes = [[(1, qwen_h // 2, qwen_w // 2)] for _ in range(noisy_latent.shape[0])]
        if cond_entry.get("edit_image") is not None or getattr(model, "_serenity_edit_mode", False):
            qwen_condition_tokens = _pack_qwen_latents(base_latent.to(device))
            qwen_img_shapes = [
                [(1, qwen_h // 2, qwen_w // 2), (1, qwen_h // 2, qwen_w // 2)]
                for _ in range(noisy_latent.shape[0])
            ]

    sdxl_time_ids = None
    if not is_flux and not is_qwen and not is_zimage and base_latent.ndim == 4:
        _, _, lh, lw = base_latent.shape
        target_h, target_w = lh * 8, lw * 8
        sdxl_time_ids = torch.tensor(
            [[target_h, target_w, 0, 0, target_h, target_w]],
            device=device,
            dtype=model_dtype,
        )

    def sigma_to_discrete_timestep(sigma):
        """Convert continuous sigma to discrete diffusers timestep (0-999)."""
        log_sigma = sigma.log()
        dists = log_sigma - log_sigmas
        low_idx = dists.ge(0).cumsum(dim=0).argmax().clamp(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)

    def build_model_kwargs(enc_hidden, pooled, enc_mask):
        """Build model kwargs for this model family."""
        kwargs: dict[str, Any] = {}
        if enc_hidden is not None:
            if is_zimage:
                cap_feats: list[torch.Tensor] = []
                hidden = enc_hidden.to(dtype=model_dtype, device=device)
                mask = enc_mask.to(device) if enc_mask is not None else None
                for batch_idx in range(hidden.shape[0]):
                    tokens = hidden[batch_idx]
                    if mask is not None and batch_idx < mask.shape[0] and mask[batch_idx].shape[0] == tokens.shape[0]:
                        active = mask[batch_idx].bool()
                        if torch.any(active):
                            tokens = tokens[active]
                    cap_feats.append(tokens)
                kwargs["cap_feats"] = cap_feats
            else:
                kwargs["encoder_hidden_states"] = enc_hidden.to(dtype=model_dtype, device=device)
        if is_flux:
            if pooled is not None:
                kwargs["pooled_projections"] = pooled.to(dtype=model_dtype, device=device)
            else:
                kwargs["pooled_projections"] = torch.zeros(1, 768, device=device, dtype=model_dtype)
            flux_guidance = float(cond_entry.get("guidance", 3.5))
            if hasattr(model, "config") and getattr(model.config, "guidance_embeds", True) is False:
                flux_guidance = 0.0
            kwargs["guidance"] = torch.tensor([flux_guidance], device=device, dtype=model_dtype)
            kwargs["img_ids"] = flux_img_ids
            kwargs["txt_ids"] = flux_txt_ids
        elif is_qwen:
            if enc_hidden is not None:
                qwen_hidden = enc_hidden.to(dtype=model_dtype, device=device)
            else:
                qwen_hidden = None
            if enc_mask is not None:
                lengths = enc_mask.sum(dim=1).to(dtype=torch.int64).tolist()
                max_len = max(max(lengths), 1)
                if qwen_hidden is not None:
                    packed_hidden = qwen_hidden.new_zeros((qwen_hidden.shape[0], max_len, qwen_hidden.shape[-1]))
                    packed_mask = enc_mask.new_zeros((enc_mask.shape[0], max_len))
                    for batch_idx, length in enumerate(lengths):
                        active = enc_mask[batch_idx].bool()
                        tokens = qwen_hidden[batch_idx][active]
                        if tokens.shape[0] == 0:
                            tokens = qwen_hidden[batch_idx, :1]
                        token_count = int(tokens.shape[0])
                        packed_hidden[batch_idx, :token_count] = tokens
                        packed_mask[batch_idx, :token_count] = 1
                        lengths[batch_idx] = token_count
                    kwargs["encoder_hidden_states"] = packed_hidden
                    kwargs["encoder_hidden_states_mask"] = packed_mask
                else:
                    kwargs["encoder_hidden_states_mask"] = enc_mask[:, :max_len]
                kwargs["txt_seq_lens"] = lengths
            else:
                if qwen_hidden is not None:
                    kwargs["encoder_hidden_states"] = qwen_hidden
                seq_len = int(enc_hidden.shape[1]) if enc_hidden is not None else 0
                kwargs["txt_seq_lens"] = [seq_len] * noisy_latent.shape[0]
            kwargs["img_shapes"] = qwen_img_shapes
            if bool(getattr(model.config, "guidance_embeds", False)):
                kwargs["guidance"] = torch.full(
                    (noisy_latent.shape[0],),
                    max(cfg, 1.0),
                    device=device,
                    dtype=torch.float32,
                )
        elif pooled is not None:
            pooled_t = pooled.to(dtype=model_dtype, device=device)
            kwargs["added_cond_kwargs"] = {
                "text_embeds": pooled_t,
                "time_ids": sdxl_time_ids,
            }
        return kwargs

    def _invoke_with_retry(fn, *args, **kwargs):
        """Retry model calls after dropping unsupported keyword arguments."""
        attempted: set[str] = set()
        while True:
            try:
                return fn(*args, **kwargs)
            except TypeError as exc:
                msg = str(exc)
                if "unexpected keyword argument" not in msg:
                    raise
                parts = msg.split("'")
                bad_kwarg = parts[1] if len(parts) >= 2 else None
                if bad_kwarg is None or bad_kwarg not in kwargs or bad_kwarg in attempted:
                    raise
                attempted.add(bad_kwarg)
                logger.info(
                    "Dropping unsupported model kwarg '%s' for %s",
                    bad_kwarg,
                    model.__class__.__name__,
                )
                kwargs = dict(kwargs)
                kwargs.pop(bad_kwarg, None)

    def call_model(inp, timestep, **kwargs):
        """Call the underlying model with family-specific timestep handling."""
        if is_flux:
            return _invoke_with_retry(model, inp, timestep=timestep, **kwargs)
        if is_qwen:
            return _invoke_with_retry(model, inp, timestep=timestep / 1000.0, **kwargs)
        if is_zimage:
            if inp.ndim == 4:
                inp = inp.unsqueeze(2)
            latents = [sample.to(dtype=model_dtype, device=device) for sample in inp]
            cap_feats = kwargs.pop("cap_feats", None)
            if cap_feats is None:
                cap_feats = [torch.zeros(1, 2560, device=device, dtype=model_dtype) for _ in latents]
            z_timestep = 1.0 - (timestep.to(device=device, dtype=torch.float32).reshape(-1) / 1000.0)
            if z_timestep.numel() == 1 and len(latents) != 1:
                z_timestep = z_timestep.expand(len(latents))
            return _invoke_with_retry(model, latents, z_timestep, cap_feats, return_dict=True)
        discrete_t = sigma_to_discrete_timestep(timestep) if log_sigmas is not None else timestep
        return _invoke_with_retry(model, inp, discrete_t, **kwargs)

    def prepare_model_output(raw_out, dtype: torch.dtype, channels: int) -> torch.Tensor:
        """Normalize family-specific model outputs to a BCHW/BCHWT tensor."""
        out = _extract_model_output(raw_out)
        if is_zimage:
            out = torch.stack([item.to(device=device, dtype=dtype) for item in out], dim=0)
            if out.ndim == 5 and out.shape[2] == 1:
                out = out.squeeze(2)
            out = -out
        else:
            out = out.to(device=device, dtype=dtype)
        return out[:, :channels, ...]

    use_cfg = (not is_flux1) and uncond_tensor is not None and cfg > 1.0

    def denoise_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        model_input = prediction.calculate_input(sigma, x)
        timestep = prediction.sigma_to_timestep(sigma)
        inp = model_input.to(dtype=model_dtype)

        if flux_condition_tokens is not None:
            inp_for_model = torch.cat([inp, flux_condition_tokens.to(dtype=inp.dtype)], dim=1)
        elif qwen_condition_tokens is not None:
            inp_for_model = torch.cat([inp, qwen_condition_tokens.to(dtype=inp.dtype)], dim=1)
        else:
            inp_for_model = inp

        cond_kwargs = build_model_kwargs(cond_tensor, pooled_cond, cond_mask)
        with torch.no_grad():
            raw_out = call_model(inp_for_model, timestep, **cond_kwargs)
            cond_out = prepare_model_output(raw_out, x.dtype, x.shape[1])

        if not use_cfg:
            return prediction.calculate_denoised(sigma, cond_out, x)

        cond_denoised = prediction.calculate_denoised(sigma, cond_out, x)
        uncond_kwargs = build_model_kwargs(uncond_tensor, pooled_uncond, uncond_mask)
        with torch.no_grad():
            raw_out = call_model(inp_for_model, timestep, **uncond_kwargs)
            uncond_out = prepare_model_output(raw_out, x.dtype, x.shape[1])
        uncond_denoised = prediction.calculate_denoised(sigma, uncond_out, x)
        return s["apply_cfg"](cond_denoised, uncond_denoised, cfg)

    step_callback = _make_step_callback(preview_interval=3)

    # Wrap with pipeline counters if available.
    _sampling_counters = None
    try:
        from serenity.inference.sampling.counters import PipelineCounters
        _sampling_counters = PipelineCounters()
        _sampling_counters.start()
        _instrumented_fn = _sampling_counters.wrap_model_fn(denoise_fn)
        _instrumented_cb = _sampling_counters.make_callback(step_callback)
    except ImportError:
        _instrumented_fn = denoise_fn
        _instrumented_cb = step_callback

    result = s["sample"](
        model_fn=_instrumented_fn,
        noise=noisy_latent,
        sigmas=sigmas,
        sampler_type=sampler_name,
        callback=_instrumented_cb,
    )

    if _sampling_counters is not None:
        _sampling_counters.finalize()
        logger.info("Sampling perf:\n%s", _sampling_counters.report())

    if is_flux and result.ndim == 3 and flux_h is not None and flux_w is not None:
        if is_flux2 and flux_patch_h is not None and flux_patch_w is not None:
            result = _unpack_flux2_latents(result, flux_patch_h, flux_patch_w)
            bn_mean = getattr(model, "_serenity_flux2_bn_mean", None)
            bn_var = getattr(model, "_serenity_flux2_bn_var", None)
            bn_eps = float(getattr(model, "_serenity_flux2_bn_eps", 1e-4))
            if isinstance(bn_mean, torch.Tensor) and isinstance(bn_var, torch.Tensor):
                mean = bn_mean.view(1, -1, 1, 1).to(device=result.device, dtype=result.dtype)
                std = torch.sqrt(
                    bn_var.view(1, -1, 1, 1).to(device=result.device, dtype=result.dtype) + bn_eps
                )
                if mean.shape[1] == result.shape[1]:
                    result = result * std + mean
            result = _unpatchify_flux2_latents(result)
        else:
            result = _unpack_flux_latents(result, flux_h, flux_w)
            result = result + 0.1159
    elif is_zimage and result.ndim == 4:
        result = result + 0.1159
    elif is_qwen and result.ndim == 3 and qwen_h is not None and qwen_w is not None:
        result = _unpack_qwen_latents(result, qwen_h, qwen_w)
        if qwen_input_was_4d:
            result = result.squeeze(2)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result.cpu()


def sample(
    model: nn.Module,
    latent: torch.Tensor,
    positive: list[dict],
    negative: list[dict],
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    denoise: float = 1.0,
    start_step: int | None = None,
    end_step: int | None = None,
    add_noise: bool = True,
    return_with_leftover_noise: bool = False,
) -> torch.Tensor:
    """Run sampling. Returns denoised latent tensor."""
    s = _get()
    prediction_type = getattr(model, "_serenity_prediction_type", None)
    if prediction_type in ("flow", "flow_flux") or _is_qwen_transformer(model):
        sigma_min, sigma_max = 1e-4, 1.0
    else:
        sigma_min, sigma_max = 0.0292, 14.6146

    sigmas = s["compute_sigmas"](
        scheduler=scheduler,
        num_steps=steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )

    # Handle denoise < 1.0 (img2img): skip early sigmas
    if denoise < 1.0:
        total = len(sigmas) - 1
        skip = int(total * (1.0 - denoise))
        sigmas = sigmas[skip:]

    # Handle start_step / end_step
    if start_step is not None or end_step is not None:
        total = len(sigmas) - 1
        s_start = start_step or 0
        s_end = end_step or total
        sigmas = sigmas[s_start:s_end + 1]

    return _run_sampling(
        model=model,
        latent=latent,
        positive=positive,
        negative=negative,
        cfg=cfg,
        sampler_name=sampler_name,
        sigmas=sigmas,
        seed=seed,
        add_noise=add_noise,
        denoise=denoise,
    )


def sample_custom(
    model: nn.Module,
    latent: torch.Tensor,
    positive: list[dict],
    negative: list[dict],
    cfg: float,
    sampler_name: str,
    sigmas: torch.Tensor,
    *,
    seed: int = 0,
    add_noise: bool = True,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run sampling with an explicit sigma schedule and optional custom noise."""
    if not isinstance(sigmas, torch.Tensor):
        sigmas = torch.tensor(sigmas, dtype=torch.float32)
    return _run_sampling(
        model=model,
        latent=latent,
        positive=positive,
        negative=negative,
        cfg=cfg,
        sampler_name=sampler_name,
        sigmas=sigmas,
        seed=seed,
        add_noise=add_noise,
        noise=noise,
    )


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------


def vae_decode(vae: VAEWrapper, latent: torch.Tensor) -> torch.Tensor:
    """Decode latent to image. Returns BCHW float32 [0, 1].

    For LTXVModelWrapper, delegates to the video VAE decoder.
    """
    if isinstance(vae, LTXVModelWrapper):
        return vae.vae_decode(latent)
    if vae is None or vae.decoder is None:
        raise RuntimeError("No VAE decoder available")
    return vae.decoder.decode(latent)


def vae_encode(vae: VAEWrapper, image: torch.Tensor) -> torch.Tensor:
    """Encode image to latent. Expects BCHW float32 [0, 1]."""
    if vae is None or vae.encoder is None:
        raise RuntimeError("No VAE encoder available")
    return vae.encoder.encode(image)


def vae_decode_tiled(vae: VAEWrapper, latent: torch.Tensor, tile_size: int = 512) -> torch.Tensor:
    """Tiled VAE decode for large images."""
    if vae is None or vae.decoder is None:
        raise RuntimeError("No VAE decoder available")
    return vae.decoder.decode(latent, tiling=True, tile_size=tile_size)


def vae_encode_tiled(vae: VAEWrapper, image: torch.Tensor, tile_size: int = 512) -> torch.Tensor:
    """Tiled VAE encode."""
    if vae is None or vae.encoder is None:
        raise RuntimeError("No VAE encoder available")
    return vae.encoder.encode(image, tiling=True, tile_size=tile_size)


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------


def apply_lora(model: nn.Module, lora_path: str, strength: float = 1.0) -> nn.Module:
    """Apply LoRA to model. Returns the modified model (same object, mutated)."""
    s = _get()
    lora_sd = s["load_lora"](lora_path)
    s["merge_lora_into_model"](model, lora_sd, strength=strength)
    return model


def apply_lora_clip(clip: CLIPWrapper, lora_path: str, strength: float = 1.0) -> CLIPWrapper:
    """Apply LoRA to CLIP. Currently a no-op pass-through since Serenity's
    text encoders don't support runtime LoRA merge in the inference path."""
    if strength != 0.0:
        logger.debug("CLIP LoRA application not yet supported, skipping (strength=%.2f)", strength)
    return clip


# ---------------------------------------------------------------------------
# ControlNet
# ---------------------------------------------------------------------------


def apply_controlnet(positive, negative, controlnet, image,
                     strength=1.0, start_percent=0.0, end_percent=1.0):
    """Apply ControlNet conditioning. Returns (positive, negative).

    Attaches controlnet info to conditioning dicts for the sampler to use.
    """
    pos_out = []
    for c in positive:
        n = dict(c)
        n["control"] = controlnet
        n["control_image"] = image
        n["control_strength"] = strength
        n["control_start"] = start_percent
        n["control_end"] = end_percent
        pos_out.append(n)

    neg_out = []
    for c in negative:
        n = dict(c)
        n["control"] = controlnet
        n["control_image"] = image
        n["control_strength"] = strength
        n["control_start"] = start_percent
        n["control_end"] = end_percent
        neg_out.append(n)

    return pos_out, neg_out


# ---------------------------------------------------------------------------
# Latent Utils
# ---------------------------------------------------------------------------

# Latent channel/downscale mapping per model family
_LATENT_PARAMS: dict[str, tuple[int, int]] = {
    "sd15": (4, 8),
    "sdxl": (4, 8),
    "sd3": (16, 8),
    "flux": (16, 8),
    "wan": (16, 8),
    "ltxv": (128, 32),  # LTX Video: 128ch, 32x downscale
    "hunyuan_video": (16, 8),
}


def empty_latent(width: int, height: int, batch_size: int = 1,
                 model_type: str = "sd15") -> torch.Tensor:
    """Create empty latent for the given model type."""
    channels, downscale = _LATENT_PARAMS.get(model_type, (4, 8))
    return torch.zeros(batch_size, channels, height // downscale, width // downscale)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16, "fp16": torch.float16,
        "float32": torch.float32, "fp32": torch.float32,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "default": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float16)


# ---------------------------------------------------------------------------
# LTX-V (Video) Support
# ---------------------------------------------------------------------------


class LTXVModelWrapper:
    """Wraps LTX-2 pipeline components for SerenityFlow nodes.

    Uses ltx_pipelines ModelLedger directly (same path as LTX2-Desktop app).
    No diffusers — loads ComfyUI-format single-file safetensors via ltx_core.
    """
    __slots__ = (
        "model_ledger",
        "device",
        "dtype",
        "_arch",
        "checkpoint_path",
        "gemma_root_path",
        "spatial_upsampler_path",
        "distilled_lora_path",
        "lora_paths",
        "lora_strengths",
        "quantization",
        "backend",
        "is_scaled_fp8",
        "_cached_text_encoder",
    )

    def __init__(
        self,
        model_ledger: Any,
        device: torch.device,
        dtype: torch.dtype,
        checkpoint_path: str = "",
        gemma_root_path: str = "",
        spatial_upsampler_path: str | None = None,
        distilled_lora_path: str | None = None,
        lora_paths: tuple[str, ...] = (),
        lora_strengths: tuple[float, ...] = (),
        quantization: str = "auto",
        backend: str = "auto",
    ):
        self.model_ledger = model_ledger
        self.device = device
        self.dtype = dtype
        self._arch = "ltxv"
        self.checkpoint_path = checkpoint_path
        self.gemma_root_path = gemma_root_path
        self.spatial_upsampler_path = spatial_upsampler_path
        self.distilled_lora_path = distilled_lora_path
        self.lora_paths = lora_paths
        self.lora_strengths = lora_strengths
        self.quantization = quantization
        self.backend = backend
        self.is_scaled_fp8 = _checkpoint_has_weight_scales(checkpoint_path)
        self._cached_text_encoder = None

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode 5D video latent [B,C,T,H,W] to pixel frames."""
        import gc
        from ltx_core.model.video_vae import decode_video as vae_decode_video

        video_decoder = self.model_ledger.video_decoder()
        decoded = vae_decode_video(latent.to(device=self.device, dtype=self.dtype), video_decoder)
        # decode_video may return Iterator or Tensor — materialise
        if not isinstance(decoded, torch.Tensor):
            decoded = torch.cat(list(decoded), dim=0)
        del video_decoder
        gc.collect()
        torch.cuda.empty_cache()
        return decoded.cpu()


def _ltx_checkpoint_looks_23(checkpoint_path: str) -> bool:
    name = Path(checkpoint_path).name.lower()
    return "2.3" in name or "22b" in name


def _ltx_gemma_rope_profiles(config: Any) -> tuple[int, dict[str, Any], int]:
    """Normalize Gemma 3 rope settings across older and newer transformers configs."""
    rope_config = getattr(config, "rope_parameters", None)
    if not isinstance(rope_config, dict) or not rope_config:
        rope_config = getattr(config, "rope_scaling", None)

    local_cfg: dict[str, Any] | None = None
    full_cfg: dict[str, Any] | None = None
    if isinstance(rope_config, dict):
        if "rope_type" in rope_config:
            local_cfg = rope_config
            full_cfg = rope_config
        else:
            maybe_local = rope_config.get("sliding_attention")
            if isinstance(maybe_local, dict):
                local_cfg = maybe_local
            maybe_full = rope_config.get("full_attention")
            if isinstance(maybe_full, dict):
                full_cfg = maybe_full
            if full_cfg is None:
                for value in rope_config.values():
                    if isinstance(value, dict) and "rope_type" in value:
                        full_cfg = value
                        break

    local_base = int(
        (local_cfg or {}).get("rope_theta")
        or getattr(config, "rope_local_base_freq", 0)
        or getattr(config, "rope_theta", 0)
        or 10000
    )
    normalized_full = dict(full_cfg or {})
    if "rope_type" not in normalized_full:
        normalized_full["rope_type"] = "default"
    full_theta = int(normalized_full.get("rope_theta") or getattr(config, "rope_theta", 0) or local_base)
    normalized_full.setdefault("rope_theta", full_theta)
    return local_base, normalized_full, full_theta


def _ltx_gemma_rope_parameters(config: Any) -> dict[str, dict[str, Any]]:
    """Return Gemma rope parameters in the modern per-layer format."""
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and rope_parameters:
        normalized = {
            key: dict(value) for key, value in rope_parameters.items() if isinstance(value, dict)
        }
    else:
        normalized = {}

    local_base, full_cfg, _ = _ltx_gemma_rope_profiles(config)
    normalized.setdefault("sliding_attention", {"rope_type": "default", "rope_theta": local_base})
    normalized.setdefault("full_attention", dict(full_cfg))
    return normalized


def _ltx_call_rope_init(fn, config: Any, layer_type: str | None = None) -> tuple[torch.Tensor, float]:
    """Call rope init helpers across transformers API variants."""
    try:
        return fn(config, device="cpu", layer_type=layer_type)
    except TypeError:
        try:
            return fn(config, "cpu", layer_type=layer_type)
        except TypeError:
            return fn(config)


def _patch_ltx_gemma_transformers_compat() -> None:
    """Adapt LTX Gemma setup to modern transformers Gemma3 rope config layout."""
    global _LTX_GEMMA_TRANSFORMERS_COMPAT_PATCHED
    if _LTX_GEMMA_TRANSFORMERS_COMPAT_PATCHED:
        return

    try:
        import ltx_core.text_encoders.gemma as gemma_mod
        from ltx_core.loader.module_ops import ModuleOps
        from ltx_core.text_encoders.gemma.encoders import base_encoder as base_encoder_mod
        from ltx_core.text_encoders.gemma.encoders import encoder_configurator as cfg_mod
    except Exception:
        return

    original_op = cfg_mod.GEMMA_MODEL_OPS
    original_precompute = base_encoder_mod.GemmaTextEncoder.precompute

    def patched_create_and_populate(module):
        model = module.model
        v_model = model.model.vision_tower.vision_model
        l_model = model.model.language_model

        config = model.config.text_config
        rope_parameters = _ltx_gemma_rope_parameters(config)
        config.rope_parameters = rope_parameters

        positions_length = len(v_model.embeddings.position_ids[0])
        position_ids = torch.arange(positions_length, dtype=torch.long, device="cpu").unsqueeze(0)
        v_model.embeddings.register_buffer("position_ids", position_ids)
        embed_scale = torch.tensor(model.config.text_config.hidden_size**0.5, device="cpu")
        l_model.embed_tokens.register_buffer("embed_scale", embed_scale)

        rotary_emb_local = getattr(l_model, "rotary_emb_local", None)
        if rotary_emb_local is not None:
            dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            local_base, full_rope_cfg, full_rope_theta = _ltx_gemma_rope_profiles(config)
            local_rope_freqs = 1.0 / (
                local_base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim)
            )

            original_rope_scaling = getattr(config, "rope_scaling", None)
            had_rope_theta = hasattr(config, "rope_theta")
            original_rope_theta = getattr(config, "rope_theta", None)
            try:
                config.rope_scaling = full_rope_cfg
                config.rope_theta = full_rope_theta
                inv_freqs, _ = _ltx_call_rope_init(
                    cfg_mod.ROPE_INIT_FUNCTIONS[full_rope_cfg["rope_type"]],
                    config,
                )
            finally:
                config.rope_scaling = original_rope_scaling
                if had_rope_theta:
                    config.rope_theta = original_rope_theta
                else:
                    try:
                        delattr(config, "rope_theta")
                    except Exception:
                        pass

            rotary_emb_local.register_buffer("inv_freq", local_rope_freqs)
            l_model.rotary_emb.register_buffer("inv_freq", inv_freqs)
            return module

        sliding_cfg = rope_parameters["sliding_attention"]
        full_cfg = rope_parameters["full_attention"]
        sliding_fn = getattr(l_model.rotary_emb, "compute_default_rope_parameters")
        full_fn = sliding_fn if full_cfg.get("rope_type") == "default" else cfg_mod.ROPE_INIT_FUNCTIONS[full_cfg["rope_type"]]
        if sliding_cfg.get("rope_type") != "default":
            sliding_fn = cfg_mod.ROPE_INIT_FUNCTIONS[sliding_cfg["rope_type"]]

        sliding_inv_freq, _ = _ltx_call_rope_init(sliding_fn, config, layer_type="sliding_attention")
        full_inv_freq, _ = _ltx_call_rope_init(full_fn, config, layer_type="full_attention")
        l_model.rotary_emb.register_buffer("sliding_attention_inv_freq", sliding_inv_freq)
        l_model.rotary_emb.register_buffer("sliding_attention_original_inv_freq", sliding_inv_freq.clone())
        l_model.rotary_emb.register_buffer("full_attention_inv_freq", full_inv_freq)
        l_model.rotary_emb.register_buffer("full_attention_original_inv_freq", full_inv_freq.clone())

        return module

    def patched_precompute(self, text: str, padding_side: str = "left"):
        token_pairs = self.tokenizer.tokenize_with_weights(text)["gemma"]
        input_ids = torch.tensor([[t[0] for t in token_pairs]], device=self.model.device)
        attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=self.model.device)

        language_model = getattr(getattr(self.model, "model", None), "language_model", None)
        if language_model is None:
            return original_precompute(self, text, padding_side)

        outputs = language_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        video_feats, audio_feats = self.feature_extractor(outputs.hidden_states, attention_mask, padding_side)
        return video_feats, audio_feats, attention_mask

    patched_op = ModuleOps(
        name=original_op.name,
        matcher=original_op.matcher,
        mutator=patched_create_and_populate,
    )
    cfg_mod.create_and_populate = patched_create_and_populate
    cfg_mod.GEMMA_MODEL_OPS = patched_op
    gemma_mod.GEMMA_MODEL_OPS = patched_op
    base_encoder_mod.GemmaTextEncoder.precompute = patched_precompute

    ledger_mod = sys.modules.get("ltx_pipelines.utils.model_ledger")
    if ledger_mod is not None:
        ledger_mod.GEMMA_MODEL_OPS = patched_op

    _LTX_GEMMA_TRANSFORMERS_COMPAT_PATCHED = True
    logger.info("Applied LTX Gemma transformers compatibility patch")


def _default_ltxv_spatial_upsampler_candidates(checkpoint_path: str) -> tuple[str, ...]:
    if _ltx_checkpoint_looks_23(checkpoint_path):
        return (
            "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
            "ltx-2-spatial-upscaler-x2-1.0.safetensors",
        )
    return ("ltx-2-spatial-upscaler-x2-1.0.safetensors",)


def _default_ltxv_distilled_lora_candidates(checkpoint_path: str) -> tuple[str, ...]:
    if _ltx_checkpoint_looks_23(checkpoint_path):
        return ("ltx-2.3-22b-distilled-lora-384.safetensors",)
    return ("ltx-2-19b-distilled-lora-384.safetensors",)


def _desktop_fast_ltx_checkpoint_candidates(checkpoint_path: str) -> tuple[str, ...]:
    """Map experimental scaled-FP8 requests to the desktop-style 22B checkpoints."""
    name = Path(checkpoint_path).name.lower()
    if not _ltx_checkpoint_looks_23(name) or "fp8" not in name:
        return ()
    return ("ltx-2.3-22b-distilled.safetensors",)


def _should_prefer_desktop_fast_ltx_checkpoint(checkpoint_path: str, backend: str) -> bool:
    if backend == "official":
        return False
    if os.environ.get("SERENITY_LTX_SCALED_FP8_EXPERIMENTAL", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if not _desktop_fast_ltx_checkpoint_candidates(checkpoint_path):
        return False
    if not _checkpoint_has_weight_scales(checkpoint_path):
        return False
    if not torch.cuda.is_available():
        return False
    try:
        total_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    except Exception:
        return False
    return total_memory <= 26 * (1024**3)


def _maybe_prefer_desktop_fast_ltx_checkpoint(checkpoint_path: str, backend: str) -> str:
    """Prefer the desktop-style 22B non-FP8 path on 24GB-class GPUs."""
    if not _should_prefer_desktop_fast_ltx_checkpoint(checkpoint_path, backend):
        return checkpoint_path

    candidates = _desktop_fast_ltx_checkpoint_candidates(checkpoint_path)
    sibling_root = Path(checkpoint_path).expanduser().resolve().parent

    def _exact_named_candidate(candidate_name: str) -> str | None:
        sibling = sibling_root / candidate_name
        if sibling.is_file():
            return str(sibling)

        try:
            from serenityflow.bridge.model_paths import get_model_paths

            paths = get_model_paths()
            for base_dir in paths.dirs.get("diffusion_models", []):
                candidate = Path(base_dir) / candidate_name
                if candidate.is_file():
                    return str(candidate)
        except Exception:
            pass

        return _resolve_ltxv_asset_from_hf_cache(candidate_name)

    for candidate_name in candidates:
        resolved = _exact_named_candidate(candidate_name)
        if resolved is None or _checkpoint_has_weight_scales(resolved):
            continue
        logger.info(
            "24GB fast path: rerouting experimental scaled-FP8 checkpoint %s -> %s",
            Path(checkpoint_path).name,
            Path(resolved).name,
        )
        logger.info("Set SERENITY_LTX_SCALED_FP8_EXPERIMENTAL=1 to force the scaled-FP8 runtime")
        return resolved

    logger.warning(
        "24GB fast path wanted to reroute %s, but no non-FP8 22B checkpoint was found; keeping the scaled-FP8 path",
        Path(checkpoint_path).name,
    )
    return checkpoint_path


def _hf_hub_root() -> Path:
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


@functools.lru_cache(maxsize=64)
def _resolve_ltxv_asset_from_hf_cache(name: str) -> str | None:
    if not name:
        return None

    hub_root = _hf_hub_root()
    repo_patterns = (
        "models--Lightricks--LTX-2.3",
        "models--Lightricks--LTX-2",
        "models--Lightricks--LTX-Video*",
    )
    matches: list[Path] = []
    for repo_pattern in repo_patterns:
        matches.extend(hub_root.glob(f"{repo_pattern}/snapshots/*/{name}"))

    if not matches:
        return None

    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for match in matches:
        if match.is_file():
            return str(match)
    return None


def _resolve_ltxv_asset(path: str | None, folder: str, *fallback_names: str) -> str | None:
    requested = (path or "").strip()
    search_names: list[str] = []
    if requested:
        search_names.append(requested)
    for candidate in fallback_names:
        if candidate and candidate not in search_names:
            search_names.append(candidate)

    for name in search_names:
        try:
            from serenityflow.bridge.model_paths import get_model_paths

            paths = get_model_paths()
            try:
                return paths.find(name, folder)
            except FileNotFoundError:
                pass
        except Exception:
            pass

        resolved = _resolve_ltxv_asset_from_hf_cache(name)
        if resolved is not None:
            return resolved

    if requested:
        candidate = Path(requested).expanduser()
        if candidate.exists():
            return str(candidate)
        if "/" in requested:
            return requested
    return None


def _resolve_ltxv_gemma_root(gemma_root_path: str | None) -> str:
    requested = (gemma_root_path or "").strip()
    fallback_names: list[str] = []
    if requested:
        requested_path = Path(requested).expanduser()
        if requested_path.is_dir() and (requested_path / "tokenizer.model").is_file():
            return str(requested_path)
        fallback_names.append(requested)
    if not requested or "gemma-3-12b-it" in requested:
        # Fallback order favors standalone (bf16) when no explicit Gemma root
        # resolves first. GPTQ-4b had builder compatibility issues here, but an
        # explicit request should still win so users can force it when needed.
        for candidate in (
            "gemma-3-12b-it-standalone",
            "gemma-3-12b-it",
            "gemma-3-12b-it-qat-q4_0-unquantized",
            "gemma-3-12b-it-GPTQ-4b",
        ):
            if candidate not in fallback_names:
                fallback_names.append(candidate)

    try:
        from serenityflow.bridge.model_paths import get_model_paths

        paths = get_model_paths()
        for candidate_name in fallback_names:
            candidate_path = Path(candidate_name).expanduser()
            if candidate_path.is_dir() and (candidate_path / "tokenizer.model").is_file():
                return str(candidate_path)
            for base_dir in paths.dirs.get("clip", []):
                candidate_dir = Path(base_dir) / candidate_name
                if candidate_dir.is_dir() and (candidate_dir / "tokenizer.model").is_file():
                    return str(candidate_dir)
    except Exception:
        pass

    resolved = _resolve_ltxv_asset(None, "clip", *fallback_names)
    if resolved is None:
        raise FileNotFoundError(
            "Unable to resolve a Gemma root for LTX. Provide gemma_path or install "
            "gemma-3-12b-it-GPTQ-4b / gemma-3-12b-it-standalone."
        )

    path = Path(resolved).expanduser()
    if path.is_dir():
        direct_tokenizer = path / "tokenizer.model"
        if not direct_tokenizer.is_file():
            for candidate_name in fallback_names:
                child = path / candidate_name
                if child.is_dir() and (child / "tokenizer.model").is_file():
                    return str(child)
            for child in sorted(path.iterdir()):
                if child.is_dir() and child.name in fallback_names and (child / "tokenizer.model").is_file():
                    return str(child)
    if path.is_file():
        for candidate in (path.parent, path.parent.parent):
            if candidate.is_dir() and list(candidate.rglob("tokenizer.model")):
                return str(candidate)
    return str(path)


def _patch_ltx_gemma_transformers_compat() -> None:
    """Normalize newer transformers Gemma configs for the LTX builder."""
    global _LTX_GEMMA_TRANSFORMERS_COMPAT_PATCHED
    if _LTX_GEMMA_TRANSFORMERS_COMPAT_PATCHED:
        return

    from ltx_core.loader.module_ops import ModuleOps
    from ltx_core.text_encoders.gemma.encoders import encoder_configurator as gemma_encoder_configurator
    from ltx_pipelines.utils import model_ledger as ltx_model_ledger

    if getattr(gemma_encoder_configurator.create_and_populate, "_serenity_compat_patch", False):
        _LTX_GEMMA_TRANSFORMERS_COMPAT_PATCHED = True
        return

    def _build_global_inv_freqs(config) -> torch.Tensor:
        rope_parameters = getattr(config, "rope_parameters", None)
        if not isinstance(rope_parameters, dict):
            rope_parameters = {}
        full_attention = dict(rope_parameters.get("full_attention") or {})
        rope_type = str(full_attention.get("rope_type") or "linear")
        rope_theta = float(
            full_attention.get("rope_theta")
            or getattr(config, "rope_theta", None)
            or 1_000_000.0
        )
        partial_rotary_factor = float(full_attention.get("partial_rotary_factor", 1.0))
        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)

        if rope_type == "default":
            return 1.0 / (
                rope_theta
                ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim)
            )

        rope_fn = gemma_encoder_configurator.ROPE_INIT_FUNCTIONS.get(rope_type)
        if rope_fn is None:
            rope_type = "linear"
            rope_fn = gemma_encoder_configurator.ROPE_INIT_FUNCTIONS[rope_type]
            full_attention.setdefault("factor", 1.0)
        full_attention.setdefault("rope_theta", rope_theta)

        # Build rope_scaling dict for transformers' rope init functions.
        # _compute_linear_scaling_rope_parameters reads config.rope_scaling["factor"].
        _rope_scaling = {"rope_type": rope_type, "factor": float(full_attention.get("factor", 1.0))}
        _rope_scaling.update(full_attention)

        rope_config = type(
            "_SerenityLTXGemmaRopeConfig",
            (),
            {
                "hidden_size": config.hidden_size,
                "num_attention_heads": config.num_attention_heads,
                "head_dim": getattr(config, "head_dim", None),
                "rope_theta": rope_theta,
                "rope_parameters": full_attention,
                "rope_scaling": _rope_scaling,
                "partial_rotary_factor": partial_rotary_factor,
                "standardize_rope_params": lambda self: None,
            },
        )()

        inv_freqs, _ = rope_fn(rope_config)
        return inv_freqs

    def _patched_create_and_populate(module):
        model = module.model
        v_model = model.model.vision_tower.vision_model
        l_model = model.model.language_model

        positions_length = len(v_model.embeddings.position_ids[0])
        position_ids = torch.arange(positions_length, dtype=torch.long, device="cpu").unsqueeze(0)
        v_model.embeddings.register_buffer("position_ids", position_ids)
        embed_scale = torch.tensor(model.config.text_config.hidden_size**0.5, device="cpu")
        l_model.embed_tokens.register_buffer("embed_scale", embed_scale)

        if hasattr(l_model, "rotary_emb_local"):
            config = model.config.text_config
            dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            rope_parameters = getattr(config, "rope_parameters", None)
            if not isinstance(rope_parameters, dict):
                rope_parameters = {}
            sliding_attention = dict(rope_parameters.get("sliding_attention") or {})
            local_base = float(
                getattr(config, "rope_local_base_freq", None)
                or sliding_attention.get("rope_theta")
                or 10_000.0
            )
            local_rope_freqs = 1.0 / (
                local_base
                ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim)
            )
            inv_freqs = _build_global_inv_freqs(config)
            l_model.rotary_emb_local.register_buffer("inv_freq", local_rope_freqs)
            l_model.rotary_emb.register_buffer("inv_freq", inv_freqs)
        elif hasattr(l_model, "rotary_emb"):
            config = model.config.text_config
            dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            rope_parameters = getattr(config, "rope_parameters", None)
            if not isinstance(rope_parameters, dict):
                rope_parameters = {}
            sliding_attention = dict(rope_parameters.get("sliding_attention") or {})
            local_base = float(
                getattr(config, "rope_local_base_freq", None)
                or sliding_attention.get("rope_theta")
                or 10_000.0
            )
            local_rope_freqs = 1.0 / (
                local_base
                ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim)
            )
            inv_freqs = _build_global_inv_freqs(config)
            l_model.rotary_emb.register_buffer("sliding_attention_inv_freq", local_rope_freqs)
            l_model.rotary_emb.register_buffer(
                "sliding_attention_original_inv_freq",
                local_rope_freqs.clone(),
            )
            l_model.rotary_emb.register_buffer("full_attention_inv_freq", inv_freqs)
            l_model.rotary_emb.register_buffer(
                "full_attention_original_inv_freq",
                inv_freqs.clone(),
            )
        else:
            raise AttributeError(
                f"Unsupported Gemma language model layout: missing rotary embedding modules on {type(l_model).__name__}"
            )
        return module

    _patched_create_and_populate._serenity_compat_patch = True

    patched_module_ops = ModuleOps(
        name=gemma_encoder_configurator.GEMMA_MODEL_OPS.name,
        matcher=gemma_encoder_configurator.GEMMA_MODEL_OPS.matcher,
        mutator=_patched_create_and_populate,
    )
    gemma_encoder_configurator.create_and_populate = _patched_create_and_populate
    gemma_encoder_configurator.GEMMA_MODEL_OPS = patched_module_ops
    ltx_model_ledger.GEMMA_MODEL_OPS = patched_module_ops
    _LTX_GEMMA_TRANSFORMERS_COMPAT_PATCHED = True


def _checkpoint_has_weight_scales(checkpoint_path: str) -> bool:
    """Check if a safetensors checkpoint contains weight_scale or scale_weight tensors (scaled FP8)."""
    try:
        from safetensors import safe_open
        with safe_open(checkpoint_path, framework="pt") as f:
            for key in f.keys():
                if key.endswith(".weight_scale") or key.endswith(".scale_weight"):
                    return True
    except Exception:
        pass
    return False


def _build_scaled_fp8_raw_load_policy():
    """Build a QuantizationPolicy that preserves raw scaled-FP8 checkpoint tensors.

    ModelLedger's quantized branch skips the blanket dtype cast that would otherwise
    turn FP8 weights into incorrect bf16 values during load. Runtime preparation is
    handled later by _prepare_ltx_scaled_fp8_transformer_for_runtime().
    """
    from ltx_core.quantization.policy import QuantizationPolicy
    from ltx_core.loader.module_ops import ModuleOps
    from ltx_core.model.transformer.model import LTXModel

    logger.info("Detected scaled FP8 checkpoint — preserving raw FP8 tensors for runtime preparation")
    return QuantizationPolicy(
        sd_ops=None,
        module_ops=(
            ModuleOps(
                name="scaled_fp8_raw_load",
                matcher=lambda model: isinstance(model, LTXModel),
                mutator=lambda model: model,
            ),
        ),
    )


def _inject_fp8_weight_scales(model: torch.nn.Module, checkpoint_path: str) -> int:
    """Inject weight_scale tensors as plain attributes on Linear modules.

    Called AFTER model loading. Scales are tiny CPU scalars — the forward hook
    moves them to GPU on the fly (4 bytes, negligible cost).
    """
    from safetensors import safe_open

    try:
        inner = _unwrap_to_blocks(model)
    except AttributeError:
        inner = model
        for attr in ("velocity_model", "model"):
            if hasattr(inner, attr):
                inner = getattr(inner, attr)
                break

    scale_map: dict[str, torch.Tensor] = {}
    with safe_open(checkpoint_path, framework="pt") as f:
        for key in f.keys():
            if key.endswith(".weight_scale"):
                module_path = key.replace("model.diffusion_model.", "").removesuffix(".weight_scale")
                scale_map[module_path] = f.get_tensor(key)

    injected = 0
    for name, m in inner.named_modules():
        if name in scale_map and isinstance(m, torch.nn.Linear):
            m.weight_scale = scale_map[name]
            injected += 1

    logger.info("Injected %d/%d FP8 weight scales", injected, len(scale_map))
    return injected


def _dequant_scaled_fp8_weights(model: torch.nn.Module, checkpoint_path: str) -> int:
    """Dequantize scaled FP8 weights in-place from the raw checkpoint tensors.

    Re-reads the original FP8 values and scale factors from the safetensors file
    and writes correctly dequantized bf16 values into the model parameters.
    """
    from serenityflow.bridge.fp8_dequant import dequantize_fp8
    from safetensors import safe_open

    try:
        inner = _unwrap_to_blocks(model)
    except AttributeError:
        inner = model
        for attr in ("velocity_model", "model"):
            if hasattr(inner, attr):
                inner = getattr(inner, attr)
                break

    fixed = 0
    with safe_open(checkpoint_path, framework="pt") as f:
        for key in f.keys():
            if not key.endswith(".weight_scale"):
                continue
            weight_key = key.removesuffix("_scale")
            if weight_key not in f.keys():
                continue

            module_path = weight_key.replace("model.diffusion_model.", "")
            parts = module_path.rsplit(".", 1)
            if len(parts) != 2:
                continue
            parent_path, attr_name = parts

            try:
                parent = inner.get_submodule(parent_path)
            except (AttributeError, torch.nn.modules.module.ModuleAttributeError):
                continue

            param = getattr(parent, attr_name, None)
            if param is None or not isinstance(param, (torch.Tensor, nn.Parameter)):
                continue

            fp8_weight = f.get_tensor(weight_key)
            scale = f.get_tensor(key)
            dequanted = dequantize_fp8(fp8_weight, scale, out_dtype=torch.bfloat16)
            param.data = dequanted
            fixed += 1

    logger.info("Dequantized %d scaled FP8 weights in-place", fixed)
    return fixed


_LTX_ERIQUANT_FP8_EXCLUDES = (
    "*patchify_proj*",
    "*adaln_single*",
    "*av_ca_video_scale_shift_adaln_single*",
    "*av_ca_a2v_gate_adaln_single*",
    "*caption_projection*",
    "*proj_out*",
    "*audio_patchify_proj*",
    "*audio_adaln_single*",
    "*av_ca_audio_scale_shift_adaln_single*",
    "*av_ca_v2a_gate_adaln_single*",
    "*audio_caption_projection*",
    "*audio_proj_out*",
    "transformer_blocks.0.*",
    "transformer_blocks.43.*",
    "transformer_blocks.44.*",
    "transformer_blocks.45.*",
    "transformer_blocks.46.*",
    "transformer_blocks.47.*",
)


def _ltx_scaled_fp8_runtime_backend() -> str:
    """Choose the runtime path for scaled FP8 LTX checkpoints."""
    requested = (os.getenv("SERENITY_LTX_SCALED_FP8_BACKEND") or "").strip().lower()
    if requested in {"dequant_bf16", "bf16", "dequant"}:
        return "dequant_bf16"
    if requested == "eriquant_fp8":
        return "eriquant_fp8"
    if requested:
        logger.warning(
            "Unknown SERENITY_LTX_SCALED_FP8_BACKEND=%r; falling back to dequant_bf16",
            requested,
        )

    try:
        from serenity.inference.quantization.eriquant_backend import is_available as eriquant_available
    except Exception:
        eriquant_available = None

    # Default to plain bf16 runtime for scaled checkpoints. It is the most
    # faithful fallback on current hardware, and eriquant remains opt-in.
    if eriquant_available is not None and eriquant_available():
        logger.info("Scaled FP8 runtime defaulting to dequant_bf16; set SERENITY_LTX_SCALED_FP8_BACKEND=eriquant_fp8 to opt in")
    return "dequant_bf16"


def _prepare_ltx_scaled_fp8_transformer_for_runtime(
    transformer: torch.nn.Module,
    checkpoint_path: str,
    *,
    stage_label: str,
) -> str:
    """Prepare a raw scaled-FP8 LTX transformer for inference runtime."""
    backend = getattr(transformer, "_serenity_scaled_fp8_backend", None)
    if backend:
        return backend

    fixed = _dequant_scaled_fp8_weights(transformer, checkpoint_path)
    if fixed <= 0:
        backend = "unmodified"
        transformer._serenity_scaled_fp8_backend = backend
        return backend

    backend = _ltx_scaled_fp8_runtime_backend()
    if backend == "eriquant_fp8":
        from serenity.inference.quantization.eriquant_backend import quantize_model_eriquant

        logger.info("%s scaled FP8: dequantized %d weights, converting to eriquant_fp8 runtime", stage_label, fixed)
        quantize_model_eriquant(
            transformer,
            mode="eriquant_fp8",
            arch="default",
            exclude=list(_LTX_ERIQUANT_FP8_EXCLUDES),
        )
    else:
        logger.info("%s scaled FP8: dequantized %d weights to bf16 runtime", stage_label, fixed)

    transformer._serenity_scaled_fp8_backend = backend
    return backend


def _coerce_ltxv_quantization_policy(checkpoint_path: str, quantization: str = "auto"):
    quantization_name = (quantization or "auto").strip().lower()
    if quantization_name in {"", "auto"}:
        quantization_name = "fp8-cast" if "fp8" in checkpoint_path.lower() else "none"
    if quantization_name in {"none", "off"}:
        return None

    # Scaled FP8 checkpoints carry pre-quantized weights and scale tensors.
    # Preserve the raw tensors during load, then prepare an inference runtime
    # explicitly once the transformer object exists.
    if _checkpoint_has_weight_scales(checkpoint_path):
        return _build_scaled_fp8_raw_load_policy()

    # Legacy naive FP8 — fall back to ltx_core's upcast-during-inference.
    from ltx_core.quantization.policy import QuantizationPolicy

    if quantization_name == "fp8-cast":
        return QuantizationPolicy.fp8_cast()
    if quantization_name == "fp8-scaled-mm":
        return QuantizationPolicy.fp8_scaled_mm()
    raise ValueError(f"Unsupported LTX quantization mode: {quantization}")


def _build_ltxv_loras(paths: tuple[str, ...], strengths: tuple[float, ...]) -> list[Any]:
    if not paths:
        return []

    from ltx_core.loader import LoraPathStrengthAndSDOps
    from ltx_core.model import transformer as ltx_transformer

    renaming_map = getattr(
        ltx_transformer,
        "LTXV_LORA_COMFY_RENAMING_MAP",
        getattr(ltx_transformer, "LTXV_MODEL_COMFY_RENAMING_MAP"),
    )

    loras = []
    for idx, path in enumerate(paths):
        if not path:
            continue
        strength = strengths[idx] if idx < len(strengths) else 1.0
        loras.append(LoraPathStrengthAndSDOps(path, float(strength), renaming_map))
    return loras


def _save_ltx_conditioning_image(guide_image: torch.Tensor) -> str:
    import tempfile
    import numpy as np
    from PIL import Image

    frame = guide_image
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim != 3:
        raise ValueError(f"Expected IMAGE tensor with 3 or 4 dims, got {list(guide_image.shape)}")

    image_np = frame.detach().cpu().float().clamp(0, 1).mul(255).to(torch.uint8).numpy()
    temp_dir = Path(os.path.realpath("temp"))
    temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="ltxv_conditioning_", suffix=".png", dir=temp_dir, delete=False) as handle:
        Image.fromarray(np.ascontiguousarray(image_np)).save(handle.name)
        return handle.name


def _materialize_ltxv_image_conditionings(
    guide_image: torch.Tensor | None,
    guide_frame_idx: int,
    guide_strength: float,
) -> list[Any]:
    if guide_image is None:
        return []

    from ltx_pipelines.utils.args import ImageConditioningInput

    return [
        ImageConditioningInput(
            path=_save_ltx_conditioning_image(guide_image),
            frame_idx=max(int(guide_frame_idx), 0),
            strength=float(guide_strength),
        )
    ]


def _write_audio_waveform_to_wav(waveform: torch.Tensor, sample_rate: int) -> str:
    import tempfile
    import wave

    samples = waveform.detach().cpu().float()
    if samples.ndim == 1:
        samples = samples.unsqueeze(0)
    if samples.ndim == 2 and samples.shape[0] > samples.shape[1]:
        samples = samples.transpose(0, 1)
    if samples.ndim != 2:
        raise ValueError(f"Unsupported audio waveform shape: {list(samples.shape)}")
    if samples.shape[0] > 8:
        samples = samples.transpose(0, 1)

    interleaved = samples.transpose(0, 1).contiguous()
    pcm16 = interleaved.clamp(-1.0, 1.0).mul(32767.0).to(torch.int16).numpy()

    temp_dir = Path(os.path.realpath("temp"))
    temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="ltxv_audio_", suffix=".wav", dir=temp_dir, delete=False) as handle:
        with wave.open(handle.name, "wb") as wav_file:
            wav_file.setnchannels(int(interleaved.shape[1]))
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sample_rate))
            wav_file.writeframes(pcm16.tobytes())
        return handle.name


def _materialize_ltxv_audio_path(audio: Any) -> str | None:
    if audio is None:
        return None

    if isinstance(audio, dict):
        path = audio.get("path")
        if path and os.path.exists(path):
            return str(path)
        waveform = audio.get("waveform")
        sample_rate = audio.get("sampling_rate") or audio.get("sample_rate")
    else:
        path = getattr(audio, "path", None)
        if path and os.path.exists(path):
            return str(path)
        waveform = getattr(audio, "waveform", None)
        sample_rate = getattr(audio, "sampling_rate", None) or getattr(audio, "sample_rate", None)

    if waveform is None or sample_rate is None:
        return None
    return _write_audio_waveform_to_wav(waveform, int(sample_rate))


def _serialize_ltxv_audio(audio: Any) -> dict[str, Any] | None:
    if audio is None:
        return None
    if isinstance(audio, dict):
        return audio

    waveform = getattr(audio, "waveform", None)
    sample_rate = getattr(audio, "sampling_rate", None) or getattr(audio, "sample_rate", None)
    if waveform is None or sample_rate is None:
        return None
    return {
        "path": None,
        "waveform": waveform.detach().cpu(),
        "sample_rate": int(sample_rate),
        "sampling_rate": int(sample_rate),
    }


def _decode_ltxv_video_iterator(video_iter: Any) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for chunk in video_iter:
        tensor = chunk if torch.is_tensor(chunk) else torch.as_tensor(chunk)
        tensor = tensor.detach().cpu()
        if tensor.dtype == torch.uint8 or tensor.float().max().item() > 1.5:
            tensor = tensor.float() / 255.0
        else:
            tensor = tensor.float()
        chunks.append(tensor)

    if not chunks:
        raise RuntimeError("LTX pipeline returned no decoded video frames")
    return torch.cat(chunks, dim=0).clamp(0, 1)


def _ltx_text_encoder_device_candidates(device: torch.device | str | None) -> tuple[torch.device, ...]:
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if torch.cuda.is_available() and target.type == "cuda":
        return (target, torch.device("cpu"))
    return (torch.device("cpu"),)


@functools.lru_cache(maxsize=8)
def _ltx_gemma_weight_bytes(gemma_root: str | None) -> int:
    if not gemma_root:
        return 0
    root = Path(gemma_root).expanduser()
    if not root.is_dir():
        return 0
    return sum(path.stat().st_size for path in root.rglob("model*.safetensors"))


def _should_try_cuda_for_full_ltx_text_encoder(ledger: Any, device: torch.device) -> bool:
    if not torch.cuda.is_available() or device.type != "cuda":
        return False

    weight_bytes = _ltx_gemma_weight_bytes(getattr(ledger, "gemma_root_path", None))
    if weight_bytes <= 0:
        return True

    free_vram = torch.cuda.mem_get_info()[0]
    load_headroom = 1 * (1024**3)
    fits = weight_bytes < (free_vram - load_headroom)
    if not fits:
        logger.info(
            "Skipping full Gemma CUDA load: weights %.1f GB, VRAM free %.1f GB, headroom %.1f GB",
            weight_bytes / (1024**3),
            free_vram / (1024**3),
            load_headroom / (1024**3),
        )
    return fits


def _load_ltx_text_encoder_with_fallback(
    ledger: Any,
    *,
    bind_text_only_precompute: bool = False,
) -> tuple[Any, torch.device]:
    previous_device = torch.device(getattr(ledger, "device", "cpu"))
    last_exc: Exception | None = None
    candidates = _ltx_text_encoder_device_candidates(previous_device)
    original_text_encoder = getattr(ledger, "_serenity_original_text_encoder", None)
    text_encoder_factory = original_text_encoder or ledger.text_encoder

    if candidates and candidates[0].type == "cuda" and not _should_try_cuda_for_full_ltx_text_encoder(ledger, candidates[0]):
        candidates = (torch.device("cpu"),)

    for candidate in candidates:
        try:
            ledger.device = candidate
            logger.info("Loading Gemma 3 text encoder on %s...", candidate)
            text_encoder = text_encoder_factory()
            if bind_text_only_precompute:
                _bind_gemma_text_encoder_text_only_precompute(text_encoder)
            return text_encoder, candidate
        except Exception as exc:
            last_exc = exc
            if candidate.type == "cuda":
                logger.warning("Gemma 3 text encoder load on %s failed: %s. Falling back to CPU.", candidate, exc)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
        finally:
            ledger.device = previous_device

    assert last_exc is not None
    raise last_exc


class _OfficialLTXTextEncoderAdapter:
    """Adapter that lets official pipelines reuse Serenity's text-encoding path."""

    def __init__(
        self,
        text_encoder: Any,
        *,
        text_encoder_device: torch.device,
        device: torch.device,
        gemma_root: str,
    ):
        self._text_encoder = text_encoder
        self._text_encoder_device = torch.device(text_encoder_device)
        self._device = torch.device(device)
        self._gemma_root = gemma_root

    def _move_output_to_device(self, output: Any):
        video_encoding = output.video_encoding.to(device=self._device).clone()
        audio_encoding = output.audio_encoding
        if audio_encoding is not None:
            audio_encoding = audio_encoding.to(device=self._device).clone()
        attention_mask = output.attention_mask.to(device=self._device).clone()
        return type(output)(video_encoding, audio_encoding, attention_mask)

    def __call__(self, text: str, padding_side: str = "left"):
        if self._text_encoder_device.type == "cpu" and self._device.type == "cuda" and torch.cuda.is_available():
            output = _encode_ltx_prompts_with_stagehand(
                self._text_encoder,
                (text,),
                device=self._device,
                gemma_root=self._gemma_root,
            )[0]
            return self._move_output_to_device(output)
        return self._move_output_to_device(self._text_encoder(text, padding_side))

    def __getattr__(self, name: str):
        return getattr(self._text_encoder, name)


def _wrap_ltx_ledger_text_encoder_cpu(ledger: Any, model: LTXVModelWrapper) -> None:
    """Make official LTX pipelines use Serenity's text-encoder loading/encoding path."""
    if ledger is None or getattr(ledger, "_serenity_text_encoder_cpu_wrap", False):
        return

    ledger._serenity_original_text_encoder = ledger.text_encoder

    def _text_encoder_with_fallback():
        if model._cached_text_encoder is not None:
            logger.info("Using cached Gemma 3 text encoder on CPU for official LTX pipeline")
            text_encoder = model._cached_text_encoder
            text_encoder_device = torch.device("cpu")
        else:
            text_encoder, text_encoder_device = _load_ltx_text_encoder_with_fallback(
                ledger,
                bind_text_only_precompute=True,
            )
            if text_encoder_device.type == "cpu":
                logger.info("Caching Gemma 3 text encoder on CPU for official LTX pipeline")
                model._cached_text_encoder = text_encoder

        return _OfficialLTXTextEncoderAdapter(
            text_encoder,
            text_encoder_device=text_encoder_device,
            device=model.device,
            gemma_root=getattr(ledger, "gemma_root_path", "") or "",
        )

    ledger.text_encoder = _text_encoder_with_fallback
    ledger._serenity_text_encoder_cpu_wrap = True


def _patch_official_ltx_pipeline_text_encoder_cpu(pipeline: Any, model: LTXVModelWrapper) -> None:
    for attr in ("model_ledger", "stage_1_model_ledger", "stage_2_model_ledger"):
        _wrap_ltx_ledger_text_encoder_cpu(getattr(pipeline, attr, None), model)


def _wrap_official_ltx_ledger_transformer_cpu_stage(
    ledger: Any,
    model: LTXVModelWrapper,
    *,
    stage_label: str,
) -> None:
    """Load official-pipeline transformers on CPU first, then move to GPU."""
    if ledger is None or getattr(ledger, "_serenity_transformer_cpu_stage_wrap", False):
        return

    ledger._serenity_original_transformer = ledger.transformer

    def _transformer_with_cpu_stage():
        target_device = torch.device(getattr(ledger, "device", model.device))
        previous_device = target_device
        try:
            ledger.device = torch.device("cpu")
            logger.info("%s transformer: loading on CPU first for official LTX pipeline", stage_label)
            transformer = ledger._serenity_original_transformer()
        finally:
            ledger.device = previous_device

        if model.is_scaled_fp8:
            _prepare_ltx_scaled_fp8_transformer_for_runtime(
                transformer,
                model.checkpoint_path,
                stage_label=stage_label,
            )

        if target_device.type == "cuda":
            model_bytes = sum(p.data.nbytes for p in transformer.parameters())
            use_direct_gpu = _try_load_ltx_transformer_direct_gpu(
                transformer,
                device=target_device,
                model_bytes=model_bytes,
                is_scaled_fp8=model.is_scaled_fp8,
                stage_label=stage_label,
            )
            if not use_direct_gpu:
                raise RuntimeError(f"{stage_label} transformer does not fit on GPU in official backend")

        return transformer

    ledger.transformer = _transformer_with_cpu_stage
    ledger._serenity_transformer_cpu_stage_wrap = True


def _patch_official_ltx_pipeline_transformers(pipeline: Any, model: LTXVModelWrapper) -> None:
    _wrap_official_ltx_ledger_transformer_cpu_stage(
        getattr(pipeline, "model_ledger", None),
        model,
        stage_label="Official",
    )
    _wrap_official_ltx_ledger_transformer_cpu_stage(
        getattr(pipeline, "stage_1_model_ledger", None),
        model,
        stage_label="Official Stage 1",
    )
    _wrap_official_ltx_ledger_transformer_cpu_stage(
        getattr(pipeline, "stage_2_model_ledger", None),
        model,
        stage_label="Official Stage 2",
    )


def _patch_official_ltx_pipeline_runtime(pipeline: Any, model: LTXVModelWrapper) -> None:
    _patch_official_ltx_pipeline_text_encoder_cpu(pipeline, model)
    _patch_official_ltx_pipeline_transformers(pipeline, model)


def _should_use_official_ltx_backend(
    model: LTXVModelWrapper,
    *,
    guide_image: torch.Tensor | None = None,
    audio: Any = None,
) -> bool:
    del guide_image, audio

    if model.backend == "official":
        return True
    if model.backend in {"legacy_stagehand", "stagehand", "auto"}:
        return False
    return False


def _build_ltxv_stage2_ledger(model: LTXVModelWrapper) -> Any:
    """Build a stage-2 ledger that adds the distilled LoRA when available."""
    if not model.distilled_lora_path:
        return model.model_ledger

    distilled_loras = tuple(_build_ltxv_loras((model.distilled_lora_path,), (1.0,)))
    if not distilled_loras:
        return model.model_ledger
    return model.model_ledger.with_loras(distilled_loras)


def _sample_ltxv_official(
    model: LTXVModelWrapper,
    prompt: str,
    *,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 512,
    num_frames: int = 25,
    steps: int = 8,
    guidance_scale: float = 3.0,
    stg_scale: float = 1.0,
    stg_blocks: list[int] | None = None,
    stg_rescale: float = 0.7,
    seed: int = 42,
    frame_rate: float = 25.0,
    mode: str = "auto",
    guide_image: torch.Tensor | None = None,
    guide_strength: float = 1.0,
    guide_frame_idx: int = 0,
    audio: Any = None,
    audio_start_time: float = 0.0,
    audio_duration: float | None = None,
) -> dict[str, torch.Tensor | dict[str, Any] | None]:
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_pipelines import (
        A2VidPipelineTwoStage,
        DistilledPipeline,
        TI2VidOneStagePipeline,
        TI2VidTwoStagesPipeline,
    )

    checkpoint_path = model.checkpoint_path
    resolved_mode = _detect_ltxv_mode(checkpoint_path, mode)
    gemma_root = _resolve_ltxv_gemma_root(model.gemma_root_path)
    spatial_upsampler_path = (
        model.spatial_upsampler_path
        or _resolve_ltxv_asset(None, "upscale_models", *_default_ltxv_spatial_upsampler_candidates(checkpoint_path))
    )
    distilled_lora_path = (
        model.distilled_lora_path
        or _resolve_ltxv_asset(None, "loras", *_default_ltxv_distilled_lora_candidates(checkpoint_path))
    )
    quantization_policy = _coerce_ltxv_quantization_policy(checkpoint_path, model.quantization)
    user_loras = _build_ltxv_loras(model.lora_paths, model.lora_strengths)
    distilled_loras = _build_ltxv_loras(
        (distilled_lora_path,) if distilled_lora_path else (),
        (1.0,) if distilled_lora_path else (),
    )
    conditioning_images = _materialize_ltxv_image_conditionings(
        guide_image=guide_image,
        guide_frame_idx=guide_frame_idx,
        guide_strength=guide_strength,
    )

    if stg_blocks is None:
        stg_blocks = [28] if _ltx_checkpoint_looks_23(checkpoint_path) else [29]

    video_guider_params = MultiModalGuiderParams(
        cfg_scale=float(guidance_scale),
        stg_scale=float(stg_scale),
        stg_blocks=stg_blocks,
        rescale_scale=float(stg_rescale),
        modality_scale=3.0,
        skip_step=0,
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=float(guidance_scale),
        stg_scale=float(stg_scale),
        stg_blocks=stg_blocks,
        rescale_scale=float(stg_rescale),
        modality_scale=3.0,
        skip_step=0,
    )

    audio_path = _materialize_ltxv_audio_path(audio)

    if audio_path:
        if spatial_upsampler_path is None or not distilled_loras:
            raise RuntimeError(
                "LTX image+audio-to-video requires both a spatial upscaler and distilled LoRA. "
                "Install the 2.3 x2 upscaler and distilled lora, or provide their paths explicitly."
            )
        pipeline = A2VidPipelineTwoStage(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_loras,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=gemma_root,
            loras=user_loras,
            device=model.device,
            quantization=quantization_policy,
        )
        _patch_official_ltx_pipeline_runtime(pipeline, model)
        video_iter, decoded_audio = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or _DEFAULT_NEGATIVE_PROMPT,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=max(int(steps), 1),
            video_guider_params=video_guider_params,
            images=conditioning_images,
            audio_path=audio_path,
            audio_start_time=float(audio_start_time),
            audio_max_duration=audio_duration,
        )
    elif resolved_mode == "distilled":
        if spatial_upsampler_path is None:
            raise RuntimeError("Distilled LTX generation requires a spatial upscaler model")
        pipeline = DistilledPipeline(
            distilled_checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=user_loras,
            device=model.device,
            quantization=quantization_policy,
        )
        _patch_official_ltx_pipeline_runtime(pipeline, model)
        video_iter, decoded_audio = pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=conditioning_images,
        )
    elif spatial_upsampler_path and distilled_loras:
        pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_loras,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=gemma_root,
            loras=user_loras,
            device=model.device,
            quantization=quantization_policy,
        )
        _patch_official_ltx_pipeline_runtime(pipeline, model)
        video_iter, decoded_audio = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or _DEFAULT_NEGATIVE_PROMPT,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=max(int(steps), 1),
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            images=conditioning_images,
        )
    else:
        pipeline = TI2VidOneStagePipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            loras=user_loras,
            device=model.device,
            quantization=quantization_policy,
        )
        _patch_official_ltx_pipeline_runtime(pipeline, model)
        video_iter, decoded_audio = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or _DEFAULT_NEGATIVE_PROMPT,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=max(int(steps), 1),
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            images=conditioning_images,
        )

    return {
        "video": _decode_ltxv_video_iterator(video_iter),
        "audio": _serialize_ltxv_audio(decoded_audio),
    }


def load_ltxv_model(
    checkpoint_path: str,
    gemma_path: str,
    dtype: str = "bfloat16",
    spatial_upsampler_path: str | None = None,
    distilled_lora_path: str | None = None,
    lora_paths: tuple[str, ...] = (),
    lora_strengths: tuple[float, ...] = (),
    quantization: str = "auto",
    backend: str = "auto",
) -> LTXVModelWrapper:
    """Load LTX-V model using ltx_pipelines ModelLedger.

    Uses the same proven path as LTX2-Desktop: ComfyUI-format safetensors
    loaded via ltx_core's SingleGPUModelBuilder. Nothing is loaded into GPU
    until sample_ltxv() is called — ModelLedger only stores builder configs.

    Args:
        checkpoint_path: Path to ComfyUI-format .safetensors checkpoint.
        gemma_path: Path to Gemma 3 12B text encoder directory.
        dtype: Weight dtype (bfloat16, float16, float32).
    """
    _patch_ltx_gemma_transformers_compat()
    from ltx_pipelines.utils import ModelLedger

    torch_dtype = _parse_dtype(dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved_checkpoint = _resolve_ltxv_asset(checkpoint_path, "diffusion_models") or checkpoint_path
    resolved_checkpoint = _maybe_prefer_desktop_fast_ltx_checkpoint(resolved_checkpoint, backend)
    resolved_gemma = _resolve_ltxv_gemma_root(gemma_path)
    resolved_spatial_upsampler = (
        _resolve_ltxv_asset(spatial_upsampler_path, "upscale_models", *_default_ltxv_spatial_upsampler_candidates(resolved_checkpoint))
        if spatial_upsampler_path or checkpoint_path
        else None
    )
    resolved_distilled_lora = (
        _resolve_ltxv_asset(distilled_lora_path, "loras", *_default_ltxv_distilled_lora_candidates(resolved_checkpoint))
        if distilled_lora_path or checkpoint_path
        else None
    )
    resolved_lora_paths = tuple(
        _resolve_ltxv_asset(path, "loras") or path for path in lora_paths if path
    )
    resolved_quantization = quantization or "auto"
    quantization_policy = _coerce_ltxv_quantization_policy(resolved_checkpoint, resolved_quantization)
    user_loras = tuple(_build_ltxv_loras(resolved_lora_paths, lora_strengths))

    if quantization_policy is not None:
        logger.info("LTX quantization enabled: %s", resolved_quantization)
    if resolved_spatial_upsampler:
        logger.info("LTX spatial upsampler: %s", resolved_spatial_upsampler)
    if resolved_distilled_lora:
        logger.info("LTX distilled lora: %s", resolved_distilled_lora)

    logger.info("Creating ModelLedger: ckpt=%s, gemma=%s", resolved_checkpoint, resolved_gemma)
    ledger = ModelLedger(
        dtype=torch_dtype,
        device=device,
        checkpoint_path=resolved_checkpoint,
        gemma_root_path=resolved_gemma,
        spatial_upsampler_path=resolved_spatial_upsampler,
        loras=user_loras,
        quantization=quantization_policy,
    )

    logger.info("LTX-V ModelLedger created (components load on demand)")
    return LTXVModelWrapper(
        ledger,
        device,
        torch_dtype,
        checkpoint_path=resolved_checkpoint,
        gemma_root_path=resolved_gemma,
        spatial_upsampler_path=resolved_spatial_upsampler,
        distilled_lora_path=resolved_distilled_lora,
        lora_paths=resolved_lora_paths,
        lora_strengths=tuple(float(x) for x in lora_strengths),
        quantization=resolved_quantization,
        backend=backend,
    )


def _stagehand_config_te(gemma_root: str = ""):
    """Stagehand config for Gemma 3 12B text encoder."""
    from stagehand import StagehandConfig
    is_fp4 = "fp4" in gemma_root.lower()
    return StagehandConfig(
        pinned_pool_mb=3072 if is_fp4 else 4096,
        pinned_slab_mb=256 if is_fp4 else 512,
        vram_high_watermark_mb=3400,
        vram_low_watermark_mb=2600,
        prefetch_window_blocks=0,
        eviction_cooldown_steps=0,
        max_inflight_transfers=1,
        telemetry_enabled=False,
    )


def _stagehand_config_xfm():
    """Stagehand config for 22B transformer (48 blocks, ~800MB each in bf16)."""
    from stagehand import StagehandConfig
    return StagehandConfig(
        pinned_pool_mb=6400,  # 8 slabs × 800MB (fits both FP8 ~400MB and bf16 ~800MB blocks)
        pinned_slab_mb=800,
        vram_high_watermark_mb=18000,
        vram_low_watermark_mb=14000,
        prefetch_window_blocks=1,
        max_inflight_transfers=1,
        telemetry_enabled=False,
    )


def _get_gemma_block_module(text_encoder):
    """Navigate Gemma3 model to the blockable layers."""
    import torch.nn as nn
    model = getattr(text_encoder, "model", text_encoder)
    # Path 1: model.language_model.model (has .layers)
    lm = getattr(model, "language_model", None)
    if lm is not None:
        inner = getattr(lm, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner
    # Path 2: model.model.language_model
    model_attr = getattr(model, "model", None)
    if model_attr is not None:
        lm2 = getattr(model_attr, "language_model", None)
        if lm2 is not None and hasattr(lm2, "layers"):
            return lm2
    raise AttributeError(f"Cannot find Gemma decoder layers on {type(text_encoder).__name__}")


def _unwrap_to_blocks(model):
    """Navigate X0Model -> velocity_model (LTXModel with transformer_blocks)."""
    for attr in ("velocity_model", "model", "module"):
        inner = getattr(model, attr, None)
        if inner is not None:
            if hasattr(inner, "transformer_blocks"):
                return inner
            for attr2 in ("velocity_model", "model", "module"):
                inner2 = getattr(inner, attr2, None)
                if inner2 is not None and hasattr(inner2, "transformer_blocks"):
                    return inner2
    if hasattr(model, "transformer_blocks"):
        return model
    raise AttributeError(f"Cannot find transformer_blocks on {type(model).__name__}")


def _move_non_blocks_to_device(root_module, block_container, device, *, preserve_fp8=False):
    """Move all params/buffers to device EXCEPT those inside block_container's layers.

    When preserve_fp8=True, FP8 params are moved to device WITHOUT dtype cast.
    This keeps them in float8_e4m3fn/e5m2 so the forward hooks can dequant correctly
    (fp8→bf16 * weight_scale). Without this, FP8 non-block params (proj_in, proj_out)
    get cast to bf16 with wrong values, causing grid artifacts.
    """
    _fp8_dtypes = set()
    try:
        _fp8_dtypes.add(torch.float8_e4m3fn)
        _fp8_dtypes.add(torch.float8_e5m2)
    except AttributeError:
        pass

    layers = getattr(block_container, "layers", None) or getattr(block_container, "transformer_blocks", None)
    if layers is None:
        raise AttributeError("block_container has no .layers or .transformer_blocks")

    block_param_ids = set(id(p) for p in layers.parameters())
    block_buf_ids = set(id(b) for b in layers.buffers())

    with torch.no_grad():
        for p in root_module.parameters():
            if id(p) not in block_param_ids:
                if preserve_fp8 and p.dtype in _fp8_dtypes:
                    # FP8 param: move to GPU preserving dtype — forward hook handles dequant.
                    if p.device != device:
                        p.data = p.data.to(device, non_blocking=True)
                elif p.device != device or p.dtype != torch.bfloat16:
                    p.data = p.data.to(device, dtype=torch.bfloat16, non_blocking=True)
        for name, buf in root_module.named_buffers():
            if id(buf) not in block_buf_ids and buf.device != device:
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = root_module.get_submodule(parts[0])
                    parent._buffers[parts[1]] = buf.to(device, non_blocking=True)
                else:
                    root_module._buffers[name] = buf.to(device, non_blocking=True)


def _module_has_fp8_params(module: torch.nn.Module) -> bool:
    """Return True when a module still carries raw FP8 parameters."""
    fp8_dtypes = set()
    try:
        fp8_dtypes.add(torch.float8_e4m3fn)
        fp8_dtypes.add(torch.float8_e5m2)
    except AttributeError:
        pass
    return any(param.dtype in fp8_dtypes for param in module.parameters())


def _move_module_to_device(module, device, dtype=torch.bfloat16):
    """Move a single module subtree to device without touching unrelated model branches."""
    with torch.no_grad():
        for p in module.parameters(recurse=True):
            if p.device != device or (torch.is_floating_point(p) and p.dtype != dtype):
                target_dtype = dtype if torch.is_floating_point(p) else None
                p.data = p.data.to(device=device, dtype=target_dtype, non_blocking=True)
        for name, buf in module.named_buffers(recurse=True):
            if buf.device == device and (not torch.is_floating_point(buf) or buf.dtype == dtype):
                continue
            target_dtype = dtype if torch.is_floating_point(buf) else None
            converted = buf.to(device=device, dtype=target_dtype, non_blocking=True)
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = module.get_submodule(parts[0])
                parent._buffers[parts[1]] = converted
            else:
                module._buffers[name] = converted


def _move_gemma_text_encoder_non_blocks_to_device(text_encoder, block_container, device):
    """Move only the text-only Gemma components required for prompt encoding."""
    modules = [
        getattr(block_container, "embed_tokens", None),
        getattr(block_container, "norm", None),
        getattr(block_container, "rotary_emb", None),
        getattr(block_container, "rotary_emb_local", None),
    ]
    for module in modules:
        if module is not None:
            _move_module_to_device(module, device)


def _ltx_trim_gemma_token_pairs(token_pairs):
    active_pairs = [(token_id, weight) for token_id, weight in token_pairs if int(weight) != 0]
    return active_pairs or token_pairs[:1]


def _ltx_prepare_gemma_token_pairs(text_encoder, text: str):
    token_pairs = _ltx_trim_gemma_token_pairs(text_encoder.tokenizer.tokenize_with_weights(text)["gemma"])
    register_multiple = getattr(
        getattr(text_encoder.embeddings_processor, "video_connector", None),
        "num_learnable_registers",
        None,
    )
    if register_multiple:
        target_len = ((len(token_pairs) + register_multiple - 1) // register_multiple) * register_multiple
        pad_len = target_len - len(token_pairs)
        if pad_len > 0:
            pad_token_id = getattr(text_encoder.tokenizer.tokenizer, "pad_token_id", 0) or 0
            token_pairs = [(pad_token_id, 0)] * pad_len + token_pairs
    return token_pairs


def _bind_gemma_text_encoder_text_only_precompute(text_encoder) -> None:
    """Override Gemma precompute on this instance to use the text LM directly."""
    original_precompute = getattr(text_encoder, "precompute")

    def patched_precompute(self, text: str, padding_side: str = "left"):
        language_model = getattr(getattr(self.model, "model", None), "language_model", None)
        if language_model is None:
            return original_precompute(text, padding_side)

        embed_tokens = getattr(language_model, "embed_tokens", None)
        device = embed_tokens.weight.device if embed_tokens is not None else self.model.device
        hf_tokenizer = getattr(getattr(self, "tokenizer", None), "tokenizer", None)
        if hf_tokenizer is not None:
            encoded = hf_tokenizer(
                text.strip(),
                padding=False,
                truncation=True,
                max_length=getattr(self.tokenizer, "max_length", None),
                return_tensors="pt",
            )
            input_ids = encoded.input_ids
            attention_mask = encoded.attention_mask
            register_multiple = getattr(
                getattr(self.embeddings_processor, "video_connector", None),
                "num_learnable_registers",
                None,
            )
            if register_multiple:
                seq_len = input_ids.shape[1]
                target_len = ((seq_len + register_multiple - 1) // register_multiple) * register_multiple
                pad_len = target_len - seq_len
                if pad_len > 0:
                    pad_token_id = getattr(hf_tokenizer, "pad_token_id", 0) or 0
                    input_ids = torch.nn.functional.pad(input_ids, (pad_len, 0), value=pad_token_id)
                    attention_mask = torch.nn.functional.pad(attention_mask, (pad_len, 0), value=0)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        else:
            token_pairs = _ltx_prepare_gemma_token_pairs(self, text)
            input_ids = torch.tensor([[t[0] for t in token_pairs]], device=device)
            attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=device)
        logger.info("LTX Gemma precompute: LM forward start (tokens=%d)", input_ids.shape[1])
        outputs = language_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        logger.info("LTX Gemma precompute: LM forward complete")
        hidden_states = outputs.hidden_states
        if isinstance(hidden_states, (list, tuple)):
            logger.info("LTX Gemma precompute: moving %d hidden states to CPU", len(hidden_states))
            hidden_states = tuple(state.to("cpu") for state in hidden_states)
        else:
            logger.info("LTX Gemma precompute: moving hidden states to CPU")
            hidden_states = hidden_states.to("cpu")
        attention_mask_cpu = attention_mask.to("cpu")
        logger.info("LTX Gemma precompute: feature extraction start")
        video_feats, audio_feats = self.feature_extractor(hidden_states, attention_mask_cpu, padding_side)
        logger.info("LTX Gemma precompute: feature extraction complete")
        return video_feats, audio_feats, attention_mask_cpu

    text_encoder.precompute = MethodType(patched_precompute, text_encoder)


def _ltx_gemma_text_encoder_lm_forward(text_encoder, text: str):
    token_pairs = _ltx_prepare_gemma_token_pairs(text_encoder, text)
    language_model = getattr(getattr(text_encoder.model, "model", None), "language_model", None)
    if language_model is None:
        raise AttributeError("Gemma language model is missing from the LTX text encoder")

    embed_tokens = getattr(language_model, "embed_tokens", None)
    device = embed_tokens.weight.device if embed_tokens is not None else text_encoder.model.device
    input_ids = torch.tensor([[token_id for token_id, _ in token_pairs]], device=device)
    attention_mask = torch.tensor([[weight for _, weight in token_pairs]], device=device)
    logger.info("LTX Gemma LM forward start (tokens=%d)", input_ids.shape[1])
    outputs = language_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    logger.info("LTX Gemma LM forward complete")
    return outputs.hidden_states, attention_mask


def _ltx_move_gemma_hidden_states_to_cpu(hidden_states, attention_mask):
    if isinstance(hidden_states, (list, tuple)):
        logger.info("LTX Gemma TE: moving %d hidden states to CPU", len(hidden_states))
        hidden_states = tuple(state.to("cpu") for state in hidden_states)
    else:
        logger.info("LTX Gemma TE: moving hidden states to CPU")
        hidden_states = hidden_states.to("cpu")
    return hidden_states, attention_mask.to("cpu")


def _ltx_finalize_gemma_text_encoder_output(text_encoder, hidden_states, attention_mask, padding_side: str = "left"):
    logger.info("LTX Gemma TE: feature extraction start")
    video_feats, audio_feats = text_encoder.feature_extractor(hidden_states, attention_mask, padding_side)
    logger.info("LTX Gemma TE: feature extraction complete")
    additive_mask = text_encoder._convert_to_additive_mask(attention_mask, video_feats.dtype)
    logger.info("LTX Gemma TE: embeddings projection start")
    video_enc, audio_enc, binary_mask = text_encoder.embeddings_processor.create_embeddings(
        video_feats,
        audio_feats,
        additive_mask,
    )
    logger.info("LTX Gemma TE: embeddings projection complete")
    return video_enc, audio_enc, binary_mask


def _encode_ltx_prompts_with_stagehand(
    text_encoder,
    prompts: tuple[str, ...],
    *,
    device: torch.device,
    gemma_root: str = "",
):
    """Run Gemma LM prompt encoding on GPU via Stagehand while keeping weights resident on CPU."""
    from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaEncoderOutput
    from stagehand import StagehandRuntime

    block_container = _get_gemma_block_module(text_encoder)
    _move_gemma_text_encoder_non_blocks_to_device(text_encoder, block_container, device)
    block_container.requires_grad_(False)

    runtime = StagehandRuntime(
        model=block_container,
        config=_stagehand_config_te(gemma_root),
        block_pattern=r"^layers\.\d+$",
        group="text_encoder",
        dtype=torch.bfloat16,
        inference_mode=True,
    )
    logger.info("Stagehand Gemma ready (%d blocks)", len(runtime._registry))

    outputs = []
    try:
        for step, prompt in enumerate(prompts):
            runtime.begin_step(step)
            with runtime.managed_forward():
                hidden_states, attention_mask = _ltx_gemma_text_encoder_lm_forward(text_encoder, prompt)
            runtime.end_step()
            hidden_states, attention_mask = _ltx_move_gemma_hidden_states_to_cpu(hidden_states, attention_mask)
            video_enc, audio_enc, binary_mask = _ltx_finalize_gemma_text_encoder_output(
                text_encoder,
                hidden_states,
                attention_mask,
            )
            outputs.append(GemmaEncoderOutput(video_enc, audio_enc, binary_mask))
    finally:
        runtime.shutdown()
        _move_gemma_text_encoder_non_blocks_to_device(text_encoder, block_container, torch.device("cpu"))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Stagehand Gemma prompt encoding complete")

    return outputs


def _try_load_ltx_transformer_direct_gpu(
    transformer,
    *,
    device: torch.device,
    model_bytes: int,
    is_scaled_fp8: bool,
    stage_label: str,
) -> bool:
    if not torch.cuda.is_available() or device.type != "cuda":
        return False

    torch.cuda.empty_cache()
    free_vram = torch.cuda.mem_get_info()[0]
    direct_margin = 256 * (1024**2) if is_scaled_fp8 else 4 * (1024**3)
    should_try = is_scaled_fp8 or model_bytes < (free_vram - direct_margin)
    logger.info(
        "%s transformer: %.1f GB, VRAM free: %.1f GB, margin: %.1f GB → %s",
        stage_label,
        model_bytes / (1024**3),
        free_vram / (1024**3),
        direct_margin / (1024**3),
        "try direct GPU" if should_try else "Stagehand",
    )
    if not should_try:
        return False

    try:
        transformer.to(device)
        post_free = torch.cuda.mem_get_info()[0]
        logger.info(
            "%s direct GPU: loaded entire transformer (%.1f GB, free after load %.1f GB)",
            stage_label,
            model_bytes / (1024**3),
            post_free / (1024**3),
        )
        return True
    except torch.OutOfMemoryError as exc:
        logger.warning("%s direct GPU load failed: %s. Falling back to Stagehand.", stage_label, exc)
        transformer.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        return False


def _detect_ltxv_mode(checkpoint_path: str, mode: str) -> str:
    """Resolve 'auto' mode to 'distilled' or 'dev' based on checkpoint filename."""
    if mode != "auto":
        return mode
    name = checkpoint_path.lower()
    if "distilled" in name:
        return "distilled"
    if "dev" in name:
        return "dev"
    # Default to distilled if unclear (safer -- works without CFG)
    return "distilled"


_DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
    "excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted "
    "proportions, deformed facial features, extra limbs, disfigured hands, artifacts, "
    "inconsistent perspective, camera shake, cartoonish rendering, 3D CGI look, "
    "unrealistic materials, uncanny valley effect"
)


@torch.inference_mode()
def sample_ltxv(
    model: LTXVModelWrapper,
    prompt: str,
    *,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 512,
    num_frames: int = 25,
    steps: int = 8,
    guidance_scale: float = 3.0,
    stg_scale: float = 1.0,
    stg_blocks: list[int] | None = None,
    stg_rescale: float = 0.7,
    seed: int = 42,
    frame_rate: float = 25.0,
    dtype: str = "bfloat16",
    mode: str = "auto",
    guide_image: torch.Tensor | None = None,
    guide_strength: float = 1.0,
    guide_frame_idx: int = 0,
    audio: Any = None,
    audio_start_time: float = 0.0,
    audio_duration: float | None = None,
) -> dict[str, torch.Tensor | dict[str, Any] | None]:
    """Generate video using LTX-V via ltx_pipelines + Stagehand block-swap.

    Same pipeline as LTX2-Desktop: load each component to CPU, use Stagehand
    to stream transformer blocks through GPU one at a time. Fits in 24GB VRAM.

    Supports two modes:
      - "distilled": Fixed 8-step sigma schedule, no CFG (for distilled checkpoints)
      - "dev": LTX2Scheduler sigma schedule with CFG/STG guidance (for dev checkpoints)
      - "auto": Auto-detect from checkpoint filename (default)

    Returns {"video": [B,C,T,H,W] tensor, "audio": tensor or None}.
    """
    import gc
    from stagehand import StagehandRuntime
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.model.audio_vae import decode_audio as vae_decode_audio, encode_audio as vae_encode_audio
    from ltx_core.model.video_vae import decode_video as vae_decode_video
    from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
    from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
    from ltx_pipelines.utils import image_conditionings_by_replacing_latent
    from ltx_pipelines.utils.helpers import (
        cleanup_memory,
        denoise_audio_video,
        denoise_video_only,
        simple_denoising_func,
    )
    from ltx_pipelines.utils.media_io import decode_audio_from_file
    from ltx_pipelines.utils.samplers import euler_denoising_loop
    from ltx_pipelines.utils.types import PipelineComponents

    logging.getLogger("stagehand").setLevel(logging.INFO)

    ledger = model.model_ledger
    stage_2_ledger = _build_ltxv_stage2_ledger(model)
    device = model.device

    should_use_official = _should_use_official_ltx_backend(
        model,
        guide_image=guide_image,
        audio=audio,
    )
    if should_use_official:
        return _sample_ltxv_official(
            model,
            prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            guidance_scale=guidance_scale,
            stg_scale=stg_scale,
            stg_blocks=stg_blocks,
            stg_rescale=stg_rescale,
            seed=seed,
            frame_rate=frame_rate,
            mode=mode,
            guide_image=guide_image,
            guide_strength=guide_strength,
            guide_frame_idx=guide_frame_idx,
            audio=audio,
            audio_start_time=audio_start_time,
            audio_duration=audio_duration,
        )

    # Resolve mode from checkpoint name
    resolved_mode = _detect_ltxv_mode(model.checkpoint_path, mode)
    is_dev = resolved_mode == "dev"

    # Safety cap: limit frames to avoid OOM from attention activation memory.
    # At full resolution, ~39K attention tokens (241 frames) exceeds 24GB VRAM.
    max_frames = 129  # Safe limit for 24GB GPU at 768x512
    if num_frames > max_frames:
        logger.warning("Clamping num_frames from %d to %d to avoid OOM", num_frames, max_frames)
        num_frames = max_frames

    if is_dev:
        # Dev mode: honor the requested step count; higher counts remain user-controlled.
        if not negative_prompt:
            negative_prompt = _DEFAULT_NEGATIVE_PROMPT
        logger.info("LTX-V dev mode: %d steps, cfg=%.1f, stg=%.1f", steps, guidance_scale, stg_scale)
    else:
        logger.info("LTX-V distilled mode: 8 steps (fixed sigma schedule)")

    conditioning_images = _materialize_ltxv_image_conditionings(
        guide_image=guide_image,
        guide_frame_idx=guide_frame_idx,
        guide_strength=guide_strength,
    )
    audio_path = _materialize_ltxv_audio_path(audio)

    # ---------------------------------------------------------------
    # 1. Text encoding (prefer GPU, fall back to cached CPU encoder)
    # ---------------------------------------------------------------
    text_encoder_device = torch.device("cpu")
    keep_text_encoder_cached = False
    if model._cached_text_encoder is not None:
        logger.info("Using cached Gemma 3 text encoder on CPU")
        text_encoder = model._cached_text_encoder
        keep_text_encoder_cached = True
    else:
        text_encoder, text_encoder_device = _load_ltx_text_encoder_with_fallback(
            ledger,
            bind_text_only_precompute=True,
        )
        if text_encoder_device.type == "cpu":
            logger.info("Caching Gemma 3 text encoder on CPU for reuse")
            model._cached_text_encoder = text_encoder
            keep_text_encoder_cached = True

    use_stagehand_text = text_encoder_device.type == "cpu" and torch.cuda.is_available()
    neg_context_p = None
    if use_stagehand_text:
        prompts = [prompt]
        if is_dev and negative_prompt:
            prompts.append(negative_prompt)
        try:
            logger.info("Using Stagehand Gemma prompt encoding from CPU weights")
            stagehand_outputs = _encode_ltx_prompts_with_stagehand(
                text_encoder,
                tuple(prompts),
                device=device,
                gemma_root=getattr(ledger, "gemma_root_path", "") or "",
            )
            context_p = stagehand_outputs[0]
            if len(stagehand_outputs) > 1:
                neg_context_p = stagehand_outputs[1]
        except Exception as exc:
            logger.warning("Stagehand Gemma prompt encoding failed: %s. Falling back to CPU.", exc)
            cleanup_memory()
            context_p = text_encoder(prompt)
            if is_dev and negative_prompt:
                neg_context_p = text_encoder(negative_prompt)
    else:
        # Encode positive prompt directly
        context_p = text_encoder(prompt)

        # Encode negative prompt for dev mode (CFG requires it)
        if is_dev and negative_prompt:
            neg_context_p = text_encoder(negative_prompt)

    video_context = context_p.video_encoding.to(device=device).clone()
    audio_context = context_p.audio_encoding
    if audio_context is not None:
        audio_context = audio_context.to(device=device).clone()

    neg_video_context = None
    neg_audio_context = None
    if neg_context_p is not None:
        neg_video_context = neg_context_p.video_encoding.to(device=device).clone()
        neg_audio_context = neg_context_p.audio_encoding
        if neg_audio_context is not None:
            neg_audio_context = neg_audio_context.to(device=device).clone()

    if keep_text_encoder_cached:
        logger.info("Text encoder retained on CPU cache")
    else:
        del text_encoder
        gc.collect()
        cleanup_memory()
        logger.info("Text encoder freed from %s", text_encoder_device)

    del context_p
    if neg_context_p is not None:
        del neg_context_p
    cleanup_memory()
    logger.info("Text encoding complete")

    # ---------------------------------------------------------------
    # 1b. Encode image/audio conditionings before loading transformer
    # ---------------------------------------------------------------
    stage_1_conditionings = []
    encoded_audio_latent = None
    preserved_audio = None

    if conditioning_images:
        logger.info("Preparing %d image conditioning input(s) for stage 1...", len(conditioning_images))
        video_encoder = ledger.video_encoder()
        stage_1_conditionings = image_conditionings_by_replacing_latent(
            images=conditioning_images,
            height=height // 2,
            width=width // 2,
            video_encoder=video_encoder,
            dtype=model.dtype,
            device=device,
        )
        del video_encoder
        gc.collect()
        cleanup_memory()
        logger.info("Stage 1 image conditioning ready")

    if audio_path:
        logger.info("Preparing audio conditioning from %s...", audio_path)
        decoded_input_audio = decode_audio_from_file(audio_path, device, audio_start_time, audio_duration)
        if decoded_input_audio is None:
            raise RuntimeError(f"LTX audio conditioning could not decode audio from {audio_path}")

        audio_encoder = ledger.audio_encoder()
        encoded_audio_latent = vae_encode_audio(decoded_input_audio, audio_encoder)
        audio_shape = AudioLatentShape.from_duration(
            batch=1,
            duration=num_frames / frame_rate,
            channels=8,
            mel_bins=16,
        )
        encoded_audio_latent = encoded_audio_latent[:, :, : audio_shape.frames]
        preserved_audio = Audio(
            waveform=decoded_input_audio.waveform.squeeze(0).detach().cpu(),
            sampling_rate=decoded_input_audio.sampling_rate,
        )
        del audio_encoder, decoded_input_audio
        gc.collect()
        cleanup_memory()
        logger.info("Audio conditioning ready: %s", list(encoded_audio_latent.shape))

    # ---------------------------------------------------------------
    # 2. Denoise with Stagehand (22B transformer)
    # ---------------------------------------------------------------
    logger.info("Loading transformer to CPU...")
    ledger.device = torch.device("cpu")
    transformer = ledger.transformer()
    ledger.device = device

    if model.is_scaled_fp8:
        _prepare_ltx_scaled_fp8_transformer_for_runtime(
            transformer,
            model.checkpoint_path,
            stage_label="Stage 1",
        )

    xfm_inner = _unwrap_to_blocks(transformer)

    preserve_fp8_non_blocks = _module_has_fp8_params(transformer)
    _move_non_blocks_to_device(transformer, xfm_inner, device, preserve_fp8=preserve_fp8_non_blocks)
    transformer.requires_grad_(False)

    # Decide: direct GPU load vs Stagehand.
    # FP8 models (~19GB) fit in 24GB VRAM — load directly, skip block-swap.
    # bf16 models (~35GB+) need Stagehand to stream blocks.
    xfm_runtime = None
    model_bytes = sum(p.data.nbytes for p in transformer.parameters())
    use_direct_gpu = _try_load_ltx_transformer_direct_gpu(
        transformer,
        device=device,
        model_bytes=model_bytes,
        is_scaled_fp8=model.is_scaled_fp8,
        stage_label="Stage 1",
    )

    if not use_direct_gpu:
        _move_non_blocks_to_device(transformer, xfm_inner, device, preserve_fp8=preserve_fp8_non_blocks)
        xfm_runtime = StagehandRuntime(
            model=xfm_inner,
            config=_stagehand_config_xfm(),
            block_pattern=r"^transformer_blocks\.\d+$",
            group="transformer",
            dtype=model.dtype,
            inference_mode=True,
        )
        logger.info("Stagehand transformer ready (%d blocks)", len(xfm_runtime._registry))

    generator = torch.Generator(device=device).manual_seed(seed)
    noiser = GaussianNoiser(generator=generator)
    stepper = EulerDiffusionStep()
    components = PipelineComponents(dtype=model.dtype, device=device)

    # Build sigmas and denoising function based on mode
    if is_dev:
        # Dev mode: LTX2Scheduler for sigma schedule + CFG/STG guidance
        from ltx_core.components.schedulers import LTX2Scheduler
        from ltx_core.components.guiders import (
            MultiModalGuiderParams,
            create_multimodal_guider_factory,
        )
        from ltx_pipelines.utils.helpers import multi_modal_guider_factory_denoising_func

        sigmas = LTX2Scheduler().execute(steps=steps).to(
            dtype=torch.float32, device=device,
        )

        if stg_blocks is None:
            stg_blocks = [28]  # LTX 2.3 default

        video_guider_params = MultiModalGuiderParams(
            cfg_scale=guidance_scale,
            stg_scale=stg_scale,
            rescale_scale=stg_rescale,
            modality_scale=3.0,
            skip_step=0,
            stg_blocks=stg_blocks,
        )
        audio_guider_params = MultiModalGuiderParams(
            cfg_scale=7.0,
            stg_scale=stg_scale,
            rescale_scale=stg_rescale,
            modality_scale=3.0,
            skip_step=0,
            stg_blocks=stg_blocks,
        )

        video_guider_factory = create_multimodal_guider_factory(
            params=video_guider_params,
            negative_context=neg_video_context,
        )
        audio_guider_factory = create_multimodal_guider_factory(
            params=audio_guider_params,
            negative_context=neg_audio_context,
        )

        base_fn = multi_modal_guider_factory_denoising_func(
            video_guider_factory=video_guider_factory,
            audio_guider_factory=audio_guider_factory,
            v_context=video_context,
            a_context=audio_context,
            transformer=transformer,
        )
    else:
        # Distilled mode: fixed sigma schedule, simple denoising (no CFG)
        sigmas = torch.tensor(
            DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=device,
        )

        base_fn = simple_denoising_func(
            video_context=video_context,
            audio_context=audio_context,
            transformer=transformer,
        )

    _call = [0]
    n_steps = len(sigmas) - 1
    _ltxv_send_progress = getattr(_preview_local, 'send_progress', None)
    _ltxv_send_binary = getattr(_preview_local, 'send_binary', None)

    # Per-step timing counters (nanoseconds).
    import time as _time
    _step_times_ns: list[dict] = []

    def wrapped_fn(video_state, audio_state, sigmas_arg, step_index):
        t0 = _time.perf_counter_ns()
        if xfm_runtime is not None:
            xfm_runtime.begin_step(_call[0])
            with xfm_runtime.managed_forward():
                result = base_fn(video_state, audio_state, sigmas_arg, step_index)
            xfm_runtime.end_step()
        else:
            result = base_fn(video_state, audio_state, sigmas_arg, step_index)
        elapsed_ns = _time.perf_counter_ns() - t0
        _call[0] += 1

        vram_mb = 0.0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        _step_times_ns.append({
            "step": _call[0],
            "elapsed_ms": elapsed_ns / 1_000_000,
            "vram_mb": round(vram_mb, 1),
        })
        logger.info(
            "Step %d/%d complete (%.0fms, VRAM %.0fMB)",
            _call[0], n_steps, elapsed_ns / 1_000_000, vram_mb,
        )
        # Send progress + preview via WS
        if _ltxv_send_progress is not None:
            _ltxv_send_progress(_call[0], n_steps)
        if _ltxv_send_binary is not None and video_state.latent is not None:
            if _call[0] % 3 == 0 or _call[0] == n_steps:
                try:
                    preview_bytes = _latent_to_preview_jpeg(video_state.latent)
                    _ltxv_send_binary(preview_bytes)
                except Exception:
                    pass
        return result

    def denoising_loop(sigmas_arg, video_state, audio_state, stepper_arg):
        return euler_denoising_loop(
            sigmas=sigmas_arg,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper_arg,
            denoise_fn=wrapped_fn,
        )

    # ---------------------------------------------------------------
    # Stage 1: Denoise at HALF resolution
    # ---------------------------------------------------------------
    s1_w, s1_h = width // 2, height // 2
    s1_shape = VideoPixelShape(
        batch=1, frames=num_frames, width=s1_w, height=s1_h, fps=frame_rate,
    )

    logger.info(
        "Stage 1: Denoising %d steps at %dx%d, %d frames (mode=%s)...",
        n_steps, s1_w, s1_h, num_frames, resolved_mode,
    )
    if encoded_audio_latent is not None:
        video_state = denoise_video_only(
            output_shape=s1_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=components,
            dtype=model.dtype,
            device=device,
            initial_audio_latent=encoded_audio_latent,
        )
        audio_state = None
    else:
        video_state, audio_state = denoise_audio_video(
            output_shape=s1_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=components,
            dtype=model.dtype,
            device=device,
        )
    logger.info("Stage 1 complete")

    s1_video_latent = video_state.latent.cpu()
    if encoded_audio_latent is not None:
        s1_audio_latent = encoded_audio_latent.detach().cpu()
    else:
        s1_audio_latent = audio_state.latent.cpu() if audio_state is not None and audio_state.latent is not None else None
    if xfm_runtime is not None:
        xfm_runtime.shutdown()
    del xfm_runtime, transformer, xfm_inner, base_fn, encoded_audio_latent
    gc.collect()
    cleanup_memory()
    logger.info("Stage 1 transformer freed")

    # ---------------------------------------------------------------
    # Stage 2: Spatial upsample + refine at full resolution
    # ---------------------------------------------------------------
    from ltx_core.model.upsampler import upsample_video
    from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES

    logger.info("Stage 2: Spatial upsampling...")
    video_encoder = ledger.video_encoder()
    stage2_conditionings = image_conditionings_by_replacing_latent(
        images=conditioning_images,
        height=height,
        width=width,
        video_encoder=video_encoder,
        dtype=model.dtype,
        device=device,
    ) if conditioning_images else []
    if not model.distilled_lora_path:
        logger.warning(
            "No compatible distilled LTX stage-2 LoRA resolved for %s — skipping stage 2 and using stage 1 output directly",
            model.checkpoint_path,
        )
        upsampler = None
    else:
        try:
            upsampler = stage_2_ledger.spatial_upsampler()
        except Exception:
            logger.warning("No spatial upsampler available — skipping stage 2, using stage 1 output directly")
            upsampler = None

    if upsampler is not None:
        upscaled = upsample_video(
            latent=s1_video_latent.to(device)[:1],
            video_encoder=video_encoder,
            upsampler=upsampler,
        )
        del video_encoder, upsampler, s1_video_latent
        gc.collect()
        cleanup_memory()
        logger.info("Upscaled latent: %s", list(upscaled.shape))

        # Reload transformer for stage 2 refinement
        logger.info("Stage 2: Loading transformer for refinement...")
        stage_2_ledger.device = torch.device("cpu")
        transformer2 = stage_2_ledger.transformer()
        stage_2_ledger.device = device

        if model.is_scaled_fp8:
            _prepare_ltx_scaled_fp8_transformer_for_runtime(
                transformer2,
                model.checkpoint_path,
                stage_label="Stage 2",
            )

        xfm_inner2 = _unwrap_to_blocks(transformer2)
        preserve_fp8_non_blocks2 = _module_has_fp8_params(transformer2)
        _move_non_blocks_to_device(transformer2, xfm_inner2, device, preserve_fp8=preserve_fp8_non_blocks2)
        transformer2.requires_grad_(False)

        # Direct GPU vs Stagehand for stage 2
        xfm_runtime2 = None
        s2_model_bytes = sum(p.data.nbytes for p in transformer2.parameters())
        use_direct_gpu2 = _try_load_ltx_transformer_direct_gpu(
            transformer2,
            device=device,
            model_bytes=s2_model_bytes,
            is_scaled_fp8=model.is_scaled_fp8,
            stage_label="Stage 2",
        )
        if not use_direct_gpu2:
            _move_non_blocks_to_device(transformer2, xfm_inner2, device, preserve_fp8=preserve_fp8_non_blocks2)
            xfm_runtime2 = StagehandRuntime(
                model=xfm_inner2,
                config=_stagehand_config_xfm(),
                block_pattern=r"^transformer_blocks\.\d+$",
                group="transformer",
                dtype=model.dtype,
                inference_mode=True,
            )
            logger.info("Stage 2 Stagehand ready (%d blocks)", len(xfm_runtime2._registry))

        # Stage 2 always uses simple denoising (distilled sigmas, no CFG)
        s2_sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=device)
        s2_base_fn = simple_denoising_func(
            video_context=video_context,
            audio_context=audio_context,
            transformer=transformer2,
        )

        _call2 = [0]
        s2_n_steps = len(s2_sigmas) - 1

        def s2_wrapped_fn(video_state, audio_state, sigmas_arg, step_index):
            if xfm_runtime2 is not None:
                xfm_runtime2.begin_step(_call2[0])
                with xfm_runtime2.managed_forward():
                    result = s2_base_fn(video_state, audio_state, sigmas_arg, step_index)
                xfm_runtime2.end_step()
            else:
                result = s2_base_fn(video_state, audio_state, sigmas_arg, step_index)
            _call2[0] += 1
            logger.info("Stage 2 step %d/%d complete", _call2[0], s2_n_steps)
            # Send progress + preview via WS (stage 2)
            if _ltxv_send_progress is not None:
                _ltxv_send_progress(n_steps + _call2[0], n_steps + s2_n_steps)
            if _ltxv_send_binary is not None and video_state.latent is not None:
                if _call2[0] % 3 == 0 or _call2[0] == s2_n_steps:
                    try:
                        preview_bytes = _latent_to_preview_jpeg(video_state.latent)
                        _ltxv_send_binary(preview_bytes)
                    except Exception:
                        pass
            return result

        def s2_loop(sigmas_arg, video_state, audio_state, stepper_arg):
            return euler_denoising_loop(
                sigmas=sigmas_arg,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper_arg,
                denoise_fn=s2_wrapped_fn,
            )

        s2_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width, height=height, fps=frame_rate,
        )

        logger.info("Stage 2: Refining %d steps at %dx%d...", s2_n_steps, width, height)
        if preserved_audio is not None:
            video_state = denoise_video_only(
                output_shape=s2_shape,
                conditionings=stage2_conditionings,
                noiser=noiser,
                sigmas=s2_sigmas,
                stepper=EulerDiffusionStep(),
                denoising_loop_fn=s2_loop,
                components=components,
                dtype=model.dtype,
                device=device,
                noise_scale=s2_sigmas[0].item(),
                initial_video_latent=upscaled,
                initial_audio_latent=s1_audio_latent.to(device) if s1_audio_latent is not None else None,
            )
            audio_state = None
        else:
            video_state, audio_state = denoise_audio_video(
                output_shape=s2_shape,
                conditionings=stage2_conditionings,
                noiser=noiser,
                sigmas=s2_sigmas,
                stepper=EulerDiffusionStep(),
                denoising_loop_fn=s2_loop,
                components=components,
                dtype=model.dtype,
                device=device,
                noise_scale=s2_sigmas[0].item(),
                initial_video_latent=upscaled,
                initial_audio_latent=s1_audio_latent.to(device) if s1_audio_latent is not None else None,
            )
        logger.info("Stage 2 complete")

        if xfm_runtime2 is not None:
            xfm_runtime2.shutdown()
        del xfm_runtime2, transformer2, xfm_inner2, s2_base_fn, upscaled
    else:
        # No upsampler — use stage 1 output directly
        video_state.latent = s1_video_latent.to(device)
        del video_encoder

    gc.collect()
    cleanup_memory()
    logger.info("Transformer freed")

    # ---------------------------------------------------------------
    # 3. Decode video
    # ---------------------------------------------------------------
    logger.info("Decoding video...")
    video_decoder = stage_2_ledger.video_decoder()
    decoded_video = vae_decode_video(video_state.latent, video_decoder)
    if not isinstance(decoded_video, torch.Tensor):
        decoded_video = torch.cat(list(decoded_video), dim=0)

    # ---------------------------------------------------------------
    # 4. Decode audio (optional)
    # ---------------------------------------------------------------
    decoded_audio = None
    if preserved_audio is not None:
        decoded_audio = preserved_audio
    elif audio_state is not None:
        try:
            audio_decoder = stage_2_ledger.audio_decoder()
            vocoder = stage_2_ledger.vocoder()
            decoded_audio = vae_decode_audio(audio_state.latent, audio_decoder, vocoder)
            del audio_decoder, vocoder
        except Exception:
            logger.debug("Audio decode skipped (no audio components)")

    del video_decoder
    gc.collect()
    cleanup_memory()

    # Build performance counters summary.
    perf_counters = None
    if _step_times_ns:
        step_ms = [s["elapsed_ms"] for s in _step_times_ns]
        perf_counters = {
            "steps": len(_step_times_ns),
            "total_ms": sum(step_ms),
            "avg_step_ms": sum(step_ms) / len(step_ms),
            "min_step_ms": min(step_ms),
            "max_step_ms": max(step_ms),
            "peak_vram_mb": max(s["vram_mb"] for s in _step_times_ns),
            "per_step": _step_times_ns,
        }

    return {"video": decoded_video.cpu(), "audio": decoded_audio, "counters": perf_counters}


__all__ = [
    "CLIPWrapper",
    "LTXVModelWrapper",
    "VAEWrapper",
    "LoadedCheckpoint",
    "apply_controlnet",
    "apply_lora",
    "apply_lora_clip",
    "empty_latent",
    "encode_text",
    "load_checkpoint",
    "load_clip",
    "load_clip_vision",
    "load_controlnet",
    "load_diffusion_model",
    "load_dual_clip",
    "load_ltxv_model",
    "load_vae",
    "sample",
    "sample_custom",
    "sample_ltxv",
    "vae_decode",
    "vae_decode_tiled",
    "vae_encode",
    "vae_encode_tiled",
]
