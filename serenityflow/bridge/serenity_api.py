"""Clean API surface for Serenity inference.

Wraps Serenity's internal functions into stable callables.
If Serenity's API changes, only this file needs updating.
SerenityFlow nodes never import from serenity directly -- only from here.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_LTX_GEMMA_TRANSFORMERS_COMPAT_PATCHED = False


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
# LoRA
# ---------------------------------------------------------------------------


def apply_lora(model: nn.Module, lora_path: str, strength: float = 1.0) -> nn.Module:
    """Apply LoRA to model. Returns the modified model (same object, mutated)."""
    s = _get()
    lora_sd = s["load_lora"](lora_path)
    s["merge_lora_into_model"](model, lora_sd, strength=strength)
    return model


def apply_lora_clip(clip, lora_path: str, strength: float = 1.0):
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
# Re-exports from domain modules
# ---------------------------------------------------------------------------

from serenityflow.bridge.preview import (  # noqa: E402
    set_preview_sender,
    clear_preview_sender,
    _make_step_callback,
    _latent_to_preview_jpeg,
)

from serenityflow.bridge.loading import (  # noqa: E402
    LoadedCheckpoint,
    CLIPWrapper,
    VAEWrapper,
    load_checkpoint,
    load_diffusion_model,
    load_vae,
    load_clip,
    load_dual_clip,
    load_triple_clip,
    load_controlnet,
    load_clip_vision,
)

from serenityflow.bridge.sampling import (  # noqa: E402
    encode_text,
    _extract_model_output,
    sample,
    sample_custom,
    vae_decode,
    vae_encode,
    vae_decode_tiled,
    vae_encode_tiled,
)

from serenityflow.bridge.ltxv import (  # noqa: E402
    LTXVModelWrapper,
    load_ltxv_model,
    sample_ltxv,
    # Private symbols re-exported for tests and internal use
    _build_ltxv_stage2_ledger,
    _checkpoint_has_weight_scales,
    _coerce_ltxv_quantization_policy,
    _decode_ltxv_video_iterator,
    _default_ltxv_distilled_lora_candidates,
    _default_ltxv_spatial_upsampler_candidates,
    _desktop_fast_ltx_checkpoint_candidates,
    _detect_ltxv_mode,
    _ltx_checkpoint_looks_23,
    _ltx_gemma_rope_parameters,
    _ltx_gemma_rope_profiles,
    _ltx_gemma_weight_bytes,
    _ltx_prepare_gemma_token_pairs,
    _ltx_scaled_fp8_runtime_backend,
    _ltx_text_encoder_device_candidates,
    _materialize_ltxv_audio_path,
    _materialize_ltxv_image_conditionings,
    _maybe_prefer_desktop_fast_ltx_checkpoint,
    _patch_ltx_gemma_transformers_compat,
    _prepare_ltx_scaled_fp8_transformer_for_runtime,
    _resolve_ltxv_asset,
    _resolve_ltxv_asset_from_hf_cache,
    _resolve_ltxv_gemma_root,
    _sample_ltxv_official,
    _should_try_cuda_for_full_ltx_text_encoder,
    _should_use_official_ltx_backend,
    _wrap_ltx_ledger_text_encoder_cpu,
    _wrap_official_ltx_ledger_transformer_cpu_stage,
)


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
    "load_triple_clip",
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
