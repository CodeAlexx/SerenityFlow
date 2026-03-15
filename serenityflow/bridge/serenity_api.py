"""Clean API surface for Serenity inference.

Wraps Serenity's internal functions into stable callables.
If Serenity's API changes, only this file needs updating.
SerenityFlow nodes never import from serenity directly -- only from here.
"""
from __future__ import annotations

import copy
import logging
import os
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


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
    model = s["load_model"](path, device="cpu", dtype=torch_dtype)

    if model_config is not None:
        from serenity.inference.models.loader import _get_adapter
        adapter = _get_adapter(model_config)
        if adapter:
            model._serenity_prediction_type = adapter.get_prediction_type()
            model._serenity_arch = model_config.architecture
            model._serenity_model_config = model_config
            logger.info("Model prediction type: %s", model._serenity_prediction_type)

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


def load_vae(path: str) -> VAEWrapper:
    """Load standalone VAE model."""
    s = _get()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from diffusers.models import AutoencoderKL

        # Detect Flux-style VAE (16ch latent) by peeking at decoder.conv_in shape
        sd_peek = s["load_state_dict"](path)
        is_flux_vae = False
        if "decoder.conv_in.weight" in sd_peek:
            latent_ch = sd_peek["decoder.conv_in.weight"].shape[1]
            is_flux_vae = (latent_ch == 16)
        del sd_peek  # free memory

        if is_flux_vae:
            # Flux VAE: LDM-format keys with 4D conv attention weights,
            # custom block_out_channels. Use from_single_file which handles
            # all conversion internally.
            # Flux VAE config from known architecture — NO HF download
            from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL as _AEConfig
            vae = AutoencoderKL(
                in_channels=3, out_channels=3, latent_channels=16,
                block_out_channels=[128, 256, 512, 512],
                layers_per_block=2, act_fn="silu", norm_num_groups=32,
                scaling_factor=0.3611,
            )
            vae_sd = load_file(path)
            # Convert LDM keys if needed
            from serenity.inference.models.convert import convert_vae_keys
            try:
                vae_sd = convert_vae_keys(vae_sd)
            except Exception:
                pass  # Already in diffusers format
            vae.load_state_dict(vae_sd, strict=False, assign=True)
            # VAE goes on available device (transformer offloaded after sampling)
            vae = vae.to(device=device, dtype=torch.float32).eval()
            scaling = 0.3611  # Flux VAE scaling factor
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
        encoder = s["VAEEncoder"](vae_model=vae, dtype=torch.float32, device=device, scaling_factor=scaling)
        return VAEWrapper(decoder, encoder)
    except Exception as e:
        raise RuntimeError(f"Failed to load VAE from {path}: {e}") from e


def load_clip(path: str, clip_type: str = "stable_diffusion") -> CLIPWrapper:
    """Load CLIP text encoder from file."""
    s = _get()
    # Text encoders stay on CPU — moved to GPU only during encode_text()
    device = "cpu"
    arch_map = {
        "stable_diffusion": s["ModelArchitecture"].SD15,
        "sdxl": s["ModelArchitecture"].SDXL,
        "sd3": s["ModelArchitecture"].SD3,
    }
    arch = arch_map.get(clip_type, s["ModelArchitecture"].SD15)

    from serenity.inference.text.encoders import TextEncoderType, get_required_encoders, get_default_encoder_path
    required = get_required_encoders(arch)

    text_mgr = s["TextEncoderManager"]()
    for enc_type in required:
        local_file = _match_encoder_file(enc_type, [path])
        if local_file:
            _load_encoder_from_safetensors(text_mgr, enc_type, local_file,
                                           dtype=torch.float16, device=device)
        else:
            enc_path = get_default_encoder_path(arch, enc_type)
            if enc_path:
                text_mgr.load_encoder(enc_type, model_path=enc_path.repo,
                                      dtype=torch.float16, device=device,
                                      subfolder=enc_path.subfolder,
                                      tokenizer_subfolder=enc_path.tokenizer_subfolder)

    return CLIPWrapper(text_mgr, arch, torch.float16, device)


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

    entry = {}
    if cond is not None:
        entry["cross_attn"] = cond
    else:
        entry["cross_attn"] = torch.zeros(1, 77, 768)

    if pooled is not None:
        entry["pooled_output"] = pooled

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
    device = latent.device if latent.is_cuda else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Extract conditioning tensors
    cond_tensor = positive[0].get("cross_attn") if positive else None
    uncond_tensor = negative[0].get("cross_attn") if negative else None
    pooled_cond = positive[0].get("pooled_output") if positive else None
    pooled_uncond = negative[0].get("pooled_output") if negative else None

    if cond_tensor is not None:
        cond_tensor = cond_tensor.to(device)
    if uncond_tensor is not None:
        uncond_tensor = uncond_tensor.to(device)
    if pooled_cond is not None:
        pooled_cond = pooled_cond.to(device)
    if pooled_uncond is not None:
        pooled_uncond = pooled_uncond.to(device)

    logger.info("Conditioning shapes: cond=%s, uncond=%s, pooled_cond=%s, pooled_uncond=%s",
                 cond_tensor.shape if cond_tensor is not None else None,
                 uncond_tensor.shape if uncond_tensor is not None else None,
                 pooled_cond.shape if pooled_cond is not None else None,
                 pooled_uncond.shape if pooled_uncond is not None else None)

    # Determine prediction type from model
    prediction_type = "eps"
    prediction_kwargs = {}
    if hasattr(model, "_serenity_prediction_type"):
        prediction_type = model._serenity_prediction_type
    if hasattr(model, "_serenity_model_config"):
        from serenity.inference.models.loader import _get_adapter
        adapter = _get_adapter(model._serenity_model_config)
        if adapter:
            prediction_kwargs = adapter.get_prediction_kwargs()

    # For Flux, override seq_len based on actual latent dimensions
    if prediction_type == "flow_flux" and latent.ndim == 4:
        _, _, lat_h, lat_w = latent.shape
        actual_seq_len = (lat_h // 2) * (lat_w // 2)  # packed patch count
        prediction_kwargs = dict(prediction_kwargs)  # copy to avoid mutating
        prediction_kwargs["seq_len"] = actual_seq_len
        logger.info("Flux seq_len=%d (from %dx%d latent)", actual_seq_len, lat_h, lat_w)

    prediction = s["get_prediction"](prediction_type, **prediction_kwargs)
    _is_flux_model = hasattr(model, "config") and hasattr(model.config, "joint_attention_dim")
    logger.info("Sample: prediction=%s, is_flux=%s, latent=%s, steps=%d, cfg=%.1f, scheduler=%s",
                prediction_type, _is_flux_model, latent.shape, steps, cfg, scheduler)

    # Determine sigma range
    if prediction_type in ("flow", "flow_flux"):
        sigma_min, sigma_max = 1e-4, 1.0
    else:
        sigma_min, sigma_max = 0.0292, 14.6146

    # Compute sigmas
    sigmas = s["compute_sigmas"](
        scheduler=scheduler,
        num_steps=steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    ).to(device)

    # Flux: apply exponential time shift to sigmas
    if prediction_type == "flow_flux" and hasattr(prediction, "apply_sigma_shift"):
        # apply_sigma_shift expects sigmas without terminal 0
        sigma_body = sigmas[:-1]
        sigma_body = prediction.apply_sigma_shift(sigma_body)
        sigmas = torch.cat([sigma_body, sigmas[-1:]])

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

    # Create noise on CPU then move — avoids VRAM pressure during allocation
    noise = s["create_noise"](seed=seed, shape=latent.shape, device="cpu", dtype=torch.float32)

    # Initialize noisy latent
    if add_noise:
        if prediction_type in ("flow", "flow_flux"):
            # Flow matching: sigma * noise + (1 - sigma) * latent
            # For full denoise (latent=zeros), this is just noise
            noisy_latent = prediction.noise_scaling(
                sigmas[0].cpu(), noise, latent, max_denoise=(denoise >= 1.0)
            ).to(device)
        else:
            # Diffusion: noise * sigma_max
            noisy_latent = (noise * sigmas[0].cpu()).to(device)
    else:
        noisy_latent = latent.to(device).float()

    # Build denoise function — use diffusers-compatible wrapper
    model_dtype = next(model.parameters()).dtype

    # Detect if model is a Flux DiT (uses keyword-only args) vs UNet (positional timestep)
    _is_flux = hasattr(model, "config") and hasattr(model.config, "joint_attention_dim")

    # Build sigma→timestep lookup for diffusers UNet models (SD1.5, SDXL)
    # These models expect discrete timesteps (0-999), not continuous sigmas.
    _log_sigmas = None
    if not _is_flux:
        # Standard diffusion schedule: beta_start=0.00085, beta_end=0.012, 1000 steps
        betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        _log_sigmas = ((1 - alphas_cumprod) / alphas_cumprod).sqrt().log().to(device)

    # Flux latent packing: (B, C, H, W) → (B, H/2*W/2, C*4) and position IDs
    _flux_img_ids = None
    _flux_txt_ids = None
    _flux_h = None
    _flux_w = None
    if _is_flux:
        _flux_h = latent.shape[2] if latent.ndim == 4 else latent.shape[-2]
        _flux_w = latent.shape[3] if latent.ndim == 4 else latent.shape[-1]
        txt_len = cond_tensor.shape[1] if cond_tensor is not None else 512

        # img_ids: (H/2 * W/2, 3) — packed patch coordinates
        ph, pw = _flux_h // 2, _flux_w // 2
        img_ids_list = []
        for y in range(ph):
            for x in range(pw):
                img_ids_list.append([0, y, x])
        _flux_img_ids = torch.tensor(img_ids_list, device=device, dtype=model_dtype)

        # txt_ids: (txt_len, 3) — zeros for text positions
        _flux_txt_ids = torch.zeros(txt_len, 3, device=device, dtype=model_dtype)

    # SDXL time_ids: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
    _sdxl_time_ids = None
    if not _is_flux and latent.ndim == 4:
        _, _, lh, lw = latent.shape
        # VAE scale factor is 8 for SDXL
        target_h, target_w = lh * 8, lw * 8
        _sdxl_time_ids = torch.tensor(
            [[target_h, target_w, 0, 0, target_h, target_w]],
            device=device, dtype=model_dtype,
        )

    def _build_model_kwargs(enc_hidden, pooled):
        """Build model forward kwargs for this model type."""
        kwargs: dict[str, Any] = {}
        if enc_hidden is not None:
            kwargs["encoder_hidden_states"] = enc_hidden.to(dtype=model_dtype, device=device)
        if _is_flux:
            # Flux requires pooled_projections — provide zeros if missing
            if pooled is not None:
                kwargs["pooled_projections"] = pooled.to(dtype=model_dtype, device=device)
            else:
                kwargs["pooled_projections"] = torch.zeros(1, 768, device=device, dtype=model_dtype)
            # Flux Dev guidance embedding (separate from CFG — CFG should be 1.0).
            # Default 3.5 for Dev, 0.0 for Schnell.
            _flux_guidance = 3.5
            if hasattr(model, "config") and getattr(model.config, "guidance_embeds", True) is False:
                _flux_guidance = 0.0
            kwargs["guidance"] = torch.tensor([_flux_guidance], device=device, dtype=model_dtype)
            # Position IDs for RoPE
            kwargs["img_ids"] = _flux_img_ids
            kwargs["txt_ids"] = _flux_txt_ids
        elif pooled is not None:
            # UNet2DConditionModel expects pooled in added_cond_kwargs, not as direct kwarg
            pooled_t = pooled.to(dtype=model_dtype, device=device)
            # SDXL needs text_embeds + time_ids in added_cond_kwargs
            kwargs["added_cond_kwargs"] = {
                "text_embeds": pooled_t,
                "time_ids": _sdxl_time_ids,
            }
        return kwargs

    def _sigma_to_discrete_timestep(sigma):
        """Convert continuous sigma to discrete diffusers timestep (0-999)."""
        log_sigma = sigma.log()
        # Find closest index in the log_sigmas table
        dists = log_sigma - _log_sigmas
        low_idx = dists.ge(0).cumsum(dim=0).argmax().clamp(max=_log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low = _log_sigmas[low_idx]
        high = _log_sigmas[high_idx]
        # Interpolate between discrete timesteps
        w = (low - log_sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)

    def _call_model(inp, timestep, **kwargs):
        """Call model with the right arg convention."""
        if _is_flux:
            # FluxTransformer2DModel: forward(hidden_states, encoder_hidden_states, ..., timestep=, ...)
            return model(inp, timestep=timestep, **kwargs)
        else:
            # UNet2DConditionModel: forward(sample, timestep, encoder_hidden_states=, ...)
            # Convert continuous sigma to discrete timestep (0-999)
            discrete_t = _sigma_to_discrete_timestep(timestep) if _log_sigmas is not None else timestep
            return model(inp, discrete_t, **kwargs)

    # Flux Dev: guidance is an embedding, NOT CFG. Never run uncond pass.
    _use_cfg = not _is_flux and uncond_tensor is not None and cfg > 1.0

    def denoise_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        model_input = prediction.calculate_input(sigma, x)
        timestep = prediction.sigma_to_timestep(sigma)
        inp = model_input.to(dtype=model_dtype)

        cond_kwargs = _build_model_kwargs(cond_tensor, pooled_cond)

        with torch.no_grad():
            raw_out = _call_model(inp, timestep, **cond_kwargs)
            cond_out = _extract_model_output(raw_out).to(x.dtype)

        if not _use_cfg:
            return prediction.calculate_denoised(sigma, cond_out, x)

        # CFG path: run uncond pass (non-Flux models only)
        cond_denoised = prediction.calculate_denoised(sigma, cond_out, x)

        uncond_kwargs = _build_model_kwargs(uncond_tensor, pooled_uncond)
        with torch.no_grad():
            raw_out = _call_model(inp, timestep, **uncond_kwargs)
            uncond_out = _extract_model_output(raw_out).to(x.dtype)
        uncond_denoised = prediction.calculate_denoised(sigma, uncond_out, x)

        return s["apply_cfg"](cond_denoised, uncond_denoised, cfg)

    # Pack latent for Flux (B, C, H, W) → (B, H/2*W/2, C*4)
    if _is_flux and noisy_latent.ndim == 4:
        B, C, H, W = noisy_latent.shape
        noisy_latent = noisy_latent.view(B, C, H // 2, 2, W // 2, 2)
        noisy_latent = noisy_latent.permute(0, 2, 4, 1, 3, 5)
        noisy_latent = noisy_latent.reshape(B, (H // 2) * (W // 2), C * 4)

    # Sample
    result = s["sample"](
        model_fn=denoise_fn,
        noise=noisy_latent,
        sigmas=sigmas,
        sampler_type=sampler_name,
    )

    # Unpack Flux result (B, seq, C*4) → (B, C, H, W)
    if _is_flux and result.ndim == 3 and _flux_h is not None:
        B, seq, channels = result.shape
        C = channels // 4
        H, W = _flux_h, _flux_w
        result = result.view(B, H // 2, W // 2, C, 2, 2)
        result = result.permute(0, 3, 1, 4, 2, 5)
        result = result.reshape(B, C, H, W)

    # Flux: apply shift factor (VAE decoder only divides by scale, doesn't add shift)
    if _is_flux:
        FLUX1_SHIFT_FACTOR = 0.1159
        result = result + FLUX1_SHIFT_FACTOR

    # Free GPU for VAE decode. Don't model.to("cpu") — that breaks
    # block offload hooks on subsequent runs. Just clear the cache.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result.cpu()


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
    __slots__ = ("model_ledger", "device", "dtype", "_arch", "checkpoint_path")

    def __init__(self, model_ledger: Any, device: torch.device, dtype: torch.dtype,
                 checkpoint_path: str = ""):
        self.model_ledger = model_ledger
        self.device = device
        self.dtype = dtype
        self._arch = "ltxv"
        self.checkpoint_path = checkpoint_path

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


def load_ltxv_model(
    checkpoint_path: str,
    gemma_path: str,
    dtype: str = "bfloat16",
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
    from ltx_pipelines.utils import ModelLedger

    torch_dtype = _parse_dtype(dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect FP8 checkpoint for quantization policy
    quantization = None
    if "fp8" in checkpoint_path.lower():
        from ltx_core.quantization import QuantizationPolicy
        quantization = QuantizationPolicy.fp8_cast()
        logger.info("FP8 checkpoint detected — using fp8_cast quantization policy")

    # Auto-find spatial upsampler (needed for two-stage pipeline)
    import os
    spatial_upsampler_path = None
    for base in [
        os.path.dirname(checkpoint_path),
        os.path.expanduser("~/SwarmUI/Models/ltx2"),
        os.path.expanduser("~/models/LTX-2"),
    ]:
        candidate = os.path.join(base, "ltx-2.3-spatial-upscaler-x2-1.0.safetensors")
        if os.path.exists(candidate):
            spatial_upsampler_path = candidate
            break
    if spatial_upsampler_path:
        logger.info("Spatial upsampler found: %s", spatial_upsampler_path)

    logger.info("Creating ModelLedger: ckpt=%s, gemma=%s", checkpoint_path, gemma_path)
    ledger = ModelLedger(
        dtype=torch_dtype,
        device=device,
        checkpoint_path=checkpoint_path,
        gemma_root_path=gemma_path,
        spatial_upsampler_path=spatial_upsampler_path,
        loras=(),
        quantization=quantization,
    )

    logger.info("LTX-V ModelLedger created (components load on demand)")
    return LTXVModelWrapper(ledger, device, torch_dtype, checkpoint_path=checkpoint_path)


def _stagehand_config_te(gemma_root: str = ""):
    """Stagehand config for Gemma 3 12B text encoder."""
    from stagehand import StagehandConfig
    is_fp4 = "fp4" in gemma_root.lower()
    return StagehandConfig(
        pinned_pool_mb=4096 if is_fp4 else 6144,
        pinned_slab_mb=256 if is_fp4 else 512,
        vram_high_watermark_mb=18000,
        vram_low_watermark_mb=14000,
        prefetch_window_blocks=2,
        max_inflight_transfers=2,
        telemetry_enabled=False,
    )


def _stagehand_config_xfm():
    """Stagehand config for 22B transformer (48 blocks, ~800MB each in bf16)."""
    from stagehand import StagehandConfig
    return StagehandConfig(
        pinned_pool_mb=6400,
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


def _move_non_blocks_to_device(root_module, block_container, device):
    """Move all params/buffers to device EXCEPT those inside block_container's layers."""
    layers = getattr(block_container, "layers", None) or getattr(block_container, "transformer_blocks", None)
    if layers is None:
        raise AttributeError("block_container has no .layers or .transformer_blocks")

    block_param_ids = set(id(p) for p in layers.parameters())
    block_buf_ids = set(id(b) for b in layers.buffers())

    with torch.no_grad():
        for p in root_module.parameters():
            if id(p) not in block_param_ids and (p.device != device or p.dtype != torch.bfloat16):
                p.data = p.data.to(device, dtype=torch.bfloat16, non_blocking=True)
        for name, buf in root_module.named_buffers():
            if id(buf) not in block_buf_ids and buf.device != device:
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = root_module.get_submodule(parts[0])
                    parent._buffers[parts[1]] = buf.to(device, non_blocking=True)
                else:
                    root_module._buffers[name] = buf.to(device, non_blocking=True)


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
) -> dict[str, torch.Tensor | None]:
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
    from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
    from ltx_core.model.video_vae import decode_video as vae_decode_video
    from ltx_core.types import VideoPixelShape
    from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
    from ltx_pipelines.utils.helpers import (
        cleanup_memory,
        denoise_audio_video,
        simple_denoising_func,
    )
    from ltx_pipelines.utils.samplers import euler_denoising_loop
    from ltx_pipelines.utils.types import PipelineComponents

    ledger = model.model_ledger
    device = model.device

    # Resolve mode from checkpoint name
    resolved_mode = _detect_ltxv_mode(model.checkpoint_path, mode)
    is_dev = resolved_mode == "dev"

    if is_dev:
        # Dev mode: use configured steps (default 30), need negative prompt for CFG
        if steps <= 8:
            steps = 30  # override low step counts that were meant for distilled
        if not negative_prompt:
            negative_prompt = _DEFAULT_NEGATIVE_PROMPT
        logger.info("LTX-V dev mode: %d steps, cfg=%.1f, stg=%.1f", steps, guidance_scale, stg_scale)
    else:
        logger.info("LTX-V distilled mode: 8 steps (fixed sigma schedule)")

    # ---------------------------------------------------------------
    # 1. Text encoding with Stagehand (Gemma 3 12B)
    # ---------------------------------------------------------------
    logger.info("Loading Gemma 3 text encoder to CPU...")
    ledger.device = torch.device("cpu")
    text_encoder = ledger.text_encoder()
    ledger.device = device

    block_module = _get_gemma_block_module(text_encoder)
    _move_non_blocks_to_device(text_encoder, block_module, device)

    te_runtime = StagehandRuntime(
        model=block_module,
        config=_stagehand_config_te(),
        block_pattern=r"^layers\.\d+$",
        group="text_encoder",
        dtype=model.dtype,
        inference_mode=True,
    )
    logger.info("Stagehand TE ready (%d blocks)", len(te_runtime._registry))

    # Encode positive prompt
    te_runtime.begin_step(0)
    with te_runtime.managed_forward():
        raw_hs, raw_mask = text_encoder.encode(prompt)
    te_runtime.end_step()

    # Encode negative prompt for dev mode (CFG requires it)
    neg_raw_hs = None
    neg_raw_mask = None
    if is_dev and negative_prompt:
        te_runtime.begin_step(1)
        with te_runtime.managed_forward():
            neg_raw_hs, neg_raw_mask = text_encoder.encode(negative_prompt)
        te_runtime.end_step()

    # Shut down TE Stagehand, free text encoder
    te_runtime.shutdown()
    del te_runtime, text_encoder, block_module
    gc.collect()
    cleanup_memory()
    logger.info("Text encoder freed")

    # Process hidden states through embeddings processor (small, fits on GPU)
    raw_hs = tuple(h.to(device=device) for h in raw_hs)
    raw_mask = raw_mask.to(device=device)
    emb_proc = ledger.gemma_embeddings_processor()
    context_p = emb_proc.process_hidden_states(raw_hs, raw_mask)
    video_context = context_p.video_encoding.clone()
    audio_context = context_p.audio_encoding
    if audio_context is not None:
        audio_context = audio_context.clone()

    # Process negative embeddings for dev mode
    neg_video_context = None
    neg_audio_context = None
    if neg_raw_hs is not None:
        neg_raw_hs = tuple(h.to(device=device) for h in neg_raw_hs)
        neg_raw_mask = neg_raw_mask.to(device=device)
        neg_context_p = emb_proc.process_hidden_states(neg_raw_hs, neg_raw_mask)
        neg_video_context = neg_context_p.video_encoding.clone()
        neg_audio_context = neg_context_p.audio_encoding
        if neg_audio_context is not None:
            neg_audio_context = neg_audio_context.clone()
        del neg_context_p, neg_raw_hs, neg_raw_mask

    del emb_proc, raw_hs, raw_mask, context_p
    gc.collect()
    cleanup_memory()
    logger.info("Text encoding complete")

    # ---------------------------------------------------------------
    # 2. Denoise with Stagehand (22B transformer)
    # ---------------------------------------------------------------
    logger.info("Loading transformer to CPU...")
    ledger.device = torch.device("cpu")
    transformer = ledger.transformer()
    ledger.device = device

    xfm_inner = _unwrap_to_blocks(transformer)
    _move_non_blocks_to_device(transformer, xfm_inner, device)
    transformer.requires_grad_(False)

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

    def wrapped_fn(video_state, audio_state, sigmas_arg, step_index):
        xfm_runtime.begin_step(_call[0])
        with xfm_runtime.managed_forward():
            result = base_fn(video_state, audio_state, sigmas_arg, step_index)
        xfm_runtime.end_step()
        _call[0] += 1
        logger.info("Step %d/%d complete", _call[0], n_steps)
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
    video_state, audio_state = denoise_audio_video(
        output_shape=s1_shape,
        conditionings=[],
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
    s1_audio_latent = audio_state.latent.cpu() if audio_state.latent is not None else None
    xfm_runtime.shutdown()
    del xfm_runtime, transformer, xfm_inner, base_fn
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
    try:
        upsampler = ledger.spatial_upsampler()
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
        ledger.device = torch.device("cpu")
        transformer2 = ledger.transformer()
        ledger.device = device

        xfm_inner2 = _unwrap_to_blocks(transformer2)
        _move_non_blocks_to_device(transformer2, xfm_inner2, device)
        transformer2.requires_grad_(False)

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
            xfm_runtime2.begin_step(_call2[0])
            with xfm_runtime2.managed_forward():
                result = s2_base_fn(video_state, audio_state, sigmas_arg, step_index)
            xfm_runtime2.end_step()
            _call2[0] += 1
            logger.info("Stage 2 step %d/%d complete", _call2[0], s2_n_steps)
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
        video_state, audio_state = denoise_audio_video(
            output_shape=s2_shape,
            conditionings=[],
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
    video_decoder = ledger.video_decoder()
    decoded_video = vae_decode_video(video_state.latent, video_decoder)
    if not isinstance(decoded_video, torch.Tensor):
        decoded_video = torch.cat(list(decoded_video), dim=0)

    # ---------------------------------------------------------------
    # 4. Decode audio (optional)
    # ---------------------------------------------------------------
    decoded_audio = None
    try:
        audio_decoder = ledger.audio_decoder()
        vocoder = ledger.vocoder()
        decoded_audio = vae_decode_audio(audio_state.latent, audio_decoder, vocoder)
        del audio_decoder, vocoder
    except Exception:
        logger.debug("Audio decode skipped (no audio components)")

    del video_decoder
    gc.collect()
    cleanup_memory()

    return {"video": decoded_video.cpu(), "audio": decoded_audio}


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
    "sample_ltxv",
    "vae_decode",
    "vae_decode_tiled",
    "vae_encode",
    "vae_encode_tiled",
]
