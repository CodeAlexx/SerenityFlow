"""Model, VAE, and CLIP loading + wrapper classes."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# --- Facade proxies (avoid circular import at module level) ---
def _get():
    from serenityflow.bridge.serenity_api import _get as _impl
    return _impl()


def _parse_dtype(dtype_str):
    from serenityflow.bridge.serenity_api import _parse_dtype as _impl
    return _impl(dtype_str)


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

        # Try Stagehand coordinator first (async prefetch with pinned pool).
        # Stagehand and legacy hooks are mutually exclusive — never both.
        stagehand_runtime = None
        if model_size > gpu_budget:
            try:
                from serenityflow.memory.coordinator import get_coordinator
                coord = get_coordinator()
                if coord is not None:
                    stagehand_runtime = coord.get_or_create_runtime(model)
            except Exception:
                logger.debug("Stagehand coordinator not available", exc_info=True)

        if stagehand_runtime is not None:
            # Stagehand manages this model — attach runtime for sampling to find.
            # Blocks stay on CPU; managed_forward() handles per-block GPU streaming.
            # But non-block params (embeddings, projections, norms) must be on GPU.
            blocks = _find_transformer_blocks(model)
            block_param_ids = set()
            for block in blocks:
                for p in block.parameters():
                    block_param_ids.add(id(p))
            non_block_bytes = 0
            device = torch.device("cuda")
            for p in model.parameters():
                if id(p) not in block_param_ids:
                    non_block_bytes += p.data.nbytes
                    p.data = p.data.to(device, non_blocking=True)
            logger.info("Moved %.1f MB of non-block params to GPU for Stagehand",
                        non_block_bytes / (1024**2))

            model._stagehand_runtime = stagehand_runtime
            model._stagehand_checkpoint_path = path
            logger.info("Model managed by Stagehand (async prefetch, %d-block lookahead)",
                        stagehand_runtime._config.prefetch_window_blocks)
        else:
            # Legacy fallback: synchronous per-block hooks (no prefetch)
            _enable_layer_offloading(model)
            from serenity.inference.memory.manager import ModelManager
            manager = _get_model_manager()
            lm = manager.load(model, budget=gpu_budget)
            if lm.offloaded_size > 0:
                logger.info(
                    "Loaded %.1f GB to GPU, %.1f GB offloaded to CPU (legacy per-layer streaming)",
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
