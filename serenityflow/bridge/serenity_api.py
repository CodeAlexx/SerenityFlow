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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Detect architecture
    model_config = s["detect_from_file"](path)
    if model_config is None:
        raise RuntimeError(f"Cannot detect model architecture from {path}")

    arch = model_config.architecture
    logger.info("Detected architecture: %s", arch.value)

    # Load model
    model = s["load_model"](path, config=model_config, device=device, dtype=torch_dtype)

    # Load VAE from checkpoint
    sd = s["load_state_dict"](path)
    vae_sd = s["extract_vae"](sd)
    vae_wrapper = None
    if vae_sd:
        try:
            from serenity.inference.models.convert import convert_ldm_vae_to_diffusers, safe_load_state_dict
            from diffusers.models import AutoencoderKL

            vae_sd = convert_ldm_vae_to_diffusers(vae_sd)
            latent_ch = 4
            if "decoder.conv_in.weight" in vae_sd:
                latent_ch = vae_sd["decoder.conv_in.weight"].shape[1]

            vae = AutoencoderKL(latent_channels=latent_ch)
            safe_load_state_dict(vae, vae_sd)
            vae = vae.to(device=device, dtype=torch_dtype).eval()

            from serenity.inference.models.base import BaseModelAdapter
            # Get adapter for scaling factor
            from serenity.inference.models.loader import _get_adapter
            adapter = _get_adapter(model_config)
            scaling = adapter.get_vae_scaling_factor() if adapter else 0.18215

            decoder = s["VAEDecoder"](vae_model=vae, dtype=torch_dtype, device=device, scaling_factor=scaling)
            encoder = s["VAEEncoder"](vae_model=vae, dtype=torch_dtype, device=device, scaling_factor=scaling)
            vae_wrapper = VAEWrapper(decoder, encoder)
        except Exception as e:
            logger.warning("Failed to load VAE from checkpoint: %s", e)

    # Load text encoders
    text_mgr = s["TextEncoderManager"]()
    try:
        text_mgr.load_for_model(arch, dtype=torch_dtype, device=device)
    except Exception as e:
        logger.warning("Failed to load text encoders for %s: %s", arch.value, e)

    clip_wrapper = CLIPWrapper(text_mgr, arch, torch_dtype, device)

    return (model, clip_wrapper, vae_wrapper)


def load_diffusion_model(path: str, dtype: str = "default") -> nn.Module:
    """Load standalone diffusion model (UNet/DiT).

    Attaches _serenity_prediction_type and _serenity_arch to the model
    so the sampler knows how to handle it.
    """
    s = _get()
    torch_dtype = torch.bfloat16 if dtype == "default" else _parse_dtype(dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Detect architecture for prediction type
    model_config = s["detect_from_file"](path)
    model = s["load_model"](path, device=device, dtype=torch_dtype)

    if model_config is not None:
        from serenity.inference.models.loader import _get_adapter
        adapter = _get_adapter(model_config)
        if adapter:
            model._serenity_prediction_type = adapter.get_prediction_type()
            model._serenity_arch = model_config.architecture
            model._serenity_model_config = model_config
            logger.info("Model prediction type: %s", model._serenity_prediction_type)

    return model


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
            flux_vae_config = os.path.join(
                os.path.expanduser("~"),
                ".cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/"
                "snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/vae",
            )
            if os.path.isdir(flux_vae_config):
                vae = AutoencoderKL.from_single_file(path, config=flux_vae_config)
            else:
                # Fallback: use HF repo config (requires network)
                vae = AutoencoderKL.from_single_file(
                    path, config="black-forest-labs/FLUX.1-dev", subfolder="vae",
                )
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

    # Determine HF repo for config + tokenizer
    if enc_type == TextEncoderType.CLIP_L:
        config_repo = "openai/clip-vit-large-patch14"
    else:
        config_repo = "stabilityai/stable-diffusion-xl-base-1.0"

    encoder = text_mgr.get_encoder(enc_type)
    if encoder.is_loaded:
        return

    logger.info("Loading %s from local safetensors (%d keys)", enc_type.value, len(sd))

    # Tokenizer from HF cache
    tok_kwargs = {}
    if enc_type == TextEncoderType.CLIP_G:
        tok_kwargs["subfolder"] = "tokenizer_2"
    encoder._tokenizer = CLIPTokenizer.from_pretrained(config_repo, **tok_kwargs)

    # Model from config + local weights
    if enc_type == TextEncoderType.CLIP_G:
        from transformers import CLIPTextModelWithProjection, CLIPTextConfig
        config = CLIPTextConfig.from_pretrained(config_repo, subfolder="text_encoder_2")
        model = CLIPTextModelWithProjection(config)
    else:
        from transformers import CLIPTextConfig
        config = CLIPTextConfig.from_pretrained(config_repo)
        model = CLIPTextModel(config)

    model.load_state_dict(sd, strict=False)
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

    config_repo = "google/t5-v1_1-xxl"
    logger.info("Loading T5-XXL from local safetensors (%d keys)", len(sd))

    # Tokenizer from HF cache
    encoder._tokenizer = AutoTokenizer.from_pretrained(config_repo)

    # Model from config + local weights
    config = AutoConfig.from_pretrained(config_repo)
    model = T5EncoderModel(config)
    model.load_state_dict(sd, strict=False)
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
        logger.warning("Text encoding failed: %s. Using zeros.", e)
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

    logger.debug("Conditioning shapes: cond=%s, uncond=%s, pooled_cond=%s, pooled_uncond=%s",
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
            # Flux Dev requires guidance input (guidance scale as tensor)
            kwargs["guidance"] = torch.tensor([cfg], device=device, dtype=model_dtype)
            # Position IDs for RoPE
            kwargs["img_ids"] = _flux_img_ids
            kwargs["txt_ids"] = _flux_txt_ids
        elif pooled is not None:
            kwargs["pooled_projections"] = pooled.to(dtype=model_dtype, device=device)
        return kwargs

    def _call_model(inp, timestep, **kwargs):
        """Call model with the right arg convention."""
        if _is_flux:
            # FluxTransformer2DModel: forward(hidden_states, encoder_hidden_states, ..., timestep=, ...)
            return model(inp, timestep=timestep, **kwargs)
        else:
            # UNet2DConditionModel: forward(sample, timestep, encoder_hidden_states=, ...)
            return model(inp, timestep, **kwargs)

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

    # Offload model to CPU after sampling to free GPU for VAE
    model.to("cpu")
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
    __slots__ = ("model_ledger", "device", "dtype", "_arch")

    def __init__(self, model_ledger: Any, device: torch.device, dtype: torch.dtype):
        self.model_ledger = model_ledger
        self.device = device
        self.dtype = dtype
        self._arch = "ltxv"

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode 5D video latent [B,C,T,H,W] to pixel frames."""
        import gc
        from ltx_core.model.video_vae import decode_video as vae_decode_video

        video_decoder = self.model_ledger.video_decoder()
        decoded = vae_decode_video(latent.to(device=self.device, dtype=self.dtype), video_decoder)
        # decode_video may return Iterator or Tensor — materialise
        if not isinstance(decoded, torch.Tensor):
            decoded = torch.cat(list(decoded), dim=2)
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

    logger.info("Creating ModelLedger: ckpt=%s, gemma=%s", checkpoint_path, gemma_path)
    ledger = ModelLedger(
        dtype=torch_dtype,
        device=device,
        checkpoint_path=checkpoint_path,
        gemma_root_path=gemma_path,
        loras=(),
        quantization=quantization,
    )

    logger.info("LTX-V ModelLedger created (components load on demand)")
    return LTXVModelWrapper(ledger, device, torch_dtype)


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


@torch.inference_mode()
def sample_ltxv(
    model: LTXVModelWrapper,
    prompt: str,
    *,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 512,
    num_frames: int = 25,
    steps: int = 40,
    guidance_scale: float = 3.0,
    stg_scale: float = 1.0,
    stg_blocks: list[int] | None = None,
    stg_rescale: float = 0.7,
    seed: int = 42,
    frame_rate: float = 25.0,
    dtype: str = "bfloat16",
) -> dict[str, torch.Tensor | None]:
    """Generate video using LTX-V via ltx_pipelines + Stagehand block-swap.

    Same pipeline as LTX2-Desktop: load each component to CPU, use Stagehand
    to stream transformer blocks through GPU one at a time. Fits in 24GB VRAM.
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

    # Encode prompt
    te_runtime.begin_step(0)
    with te_runtime.managed_forward():
        raw_hs, raw_mask = text_encoder.encode(prompt)
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
    sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, device=device)
    components = PipelineComponents(dtype=model.dtype, device=device)

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

    output_shape = VideoPixelShape(
        batch=1, frames=num_frames, width=width, height=height, fps=frame_rate,
    )

    logger.info("Denoising %d steps at %dx%d, %d frames...", n_steps, width, height, num_frames)
    video_state, audio_state = denoise_audio_video(
        output_shape=output_shape,
        conditionings=[],
        noiser=noiser,
        sigmas=sigmas,
        stepper=stepper,
        denoising_loop_fn=denoising_loop,
        components=components,
        dtype=model.dtype,
        device=device,
    )

    xfm_runtime.shutdown()
    del xfm_runtime, transformer, xfm_inner, base_fn
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
        decoded_video = torch.cat(list(decoded_video), dim=2)

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
