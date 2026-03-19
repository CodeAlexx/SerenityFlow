"""LTX-V (Video) model loading, sampling, and all LTX-specific helpers."""
from __future__ import annotations

import copy
import functools
import gc
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


# --- Facade proxies (avoid circular import at module level) ---
def _parse_dtype(dtype_str):
    from serenityflow.bridge.serenity_api import _parse_dtype as _impl
    return _impl(dtype_str)


from serenityflow.bridge.preview import _latent_to_preview_jpeg, _preview_local


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
    import serenityflow.bridge.serenity_api as _facade
    if _facade._LTX_GEMMA_TRANSFORMERS_COMPAT_PATCHED:
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

    _facade._LTX_GEMMA_TRANSFORMERS_COMPAT_PATCHED = True
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
    # Disabled: FP8 forward hooks keep weights at FP8 size (~14GB) on GPU.
    # No need to reroute to distilled bf16 anymore.
    return False


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


def _resolve_ltxv_gemma_root(gemma_root_path: str | None, *, is_fp8: bool = False) -> str:
    requested = (gemma_root_path or "").strip()
    fallback_names: list[str] = []
    if requested:
        requested_path = Path(requested).expanduser()
        if requested_path.is_dir() and (requested_path / "tokenizer.model").is_file():
            return str(requested_path)
        fallback_names.append(requested)
    if not requested or "gemma-3-12b-it" in requested:
        if is_fp8:
            # FP8 checkpoint: prefer FP8 Gemma (~13GB) — fits on GPU without Stagehand
            candidates = (
                "gemma-3-12b-it-fp8",
                "gemma-3-12b-it-GPTQ-4b",
                "gemma-3-12b-it-qat-q4_0-unquantized",
                "gemma-3-12b-it-standalone",
                "gemma-3-12b-it",
            )
        else:
            candidates = (
                "gemma-3-12b-it-standalone",
                "gemma-3-12b-it",
                "gemma-3-12b-it-qat-q4_0-unquantized",
                "gemma-3-12b-it-GPTQ-4b",
            )
        for candidate in candidates:
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


def _materialise_meta_params(module: torch.nn.Module) -> None:
    """Replace any meta-device parameters/buffers with zeroed CPU tensors.

    This is needed when an FP8 checkpoint omits weights for a sub-module
    (e.g. vision tower) — the meta tensors left behind would crash on
    ``.to(device)``.  Materialising them to zeros is safe because the
    sub-module is never used at inference time.

    NOTE: We iterate ``named_modules()`` and inspect each module's
    ``_parameters`` / ``_buffers`` dicts directly, because
    ``named_parameters()`` deduplicates by ``data_ptr()`` — and ALL
    meta tensors share ``data_ptr() == 0``, causing most of them to be
    silently skipped.
    """
    meta_device = torch.device("meta")
    for mod_name, mod in module.named_modules():
        for pname, param in list(mod._parameters.items()):
            if param is not None and param.device == meta_device:
                new = torch.zeros(param.shape, dtype=param.dtype, device="cpu")
                mod._parameters[pname] = torch.nn.Parameter(new, requires_grad=False)
        for bname, buf in list(mod._buffers.items()):
            if buf is not None and buf.device == meta_device:
                new = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu")
                mod._buffers[bname] = new


def _patch_ltx_gemma_transformers_compat() -> None:
    """Normalize newer transformers Gemma configs for the LTX builder."""
    import serenityflow.bridge.serenity_api as _facade
    if _facade._LTX_GEMMA_TRANSFORMERS_COMPAT_PATCHED:
        return

    from ltx_core.loader.module_ops import ModuleOps
    from ltx_core.text_encoders.gemma.encoders import encoder_configurator as gemma_encoder_configurator
    from ltx_pipelines.utils import model_ledger as ltx_model_ledger

    if getattr(gemma_encoder_configurator.create_and_populate, "_serenity_compat_patch", False):
        _facade._LTX_GEMMA_TRANSFORMERS_COMPAT_PATCHED = True
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

    # Patch SingleGPUModelBuilder._return_model so that meta-device params
    # (e.g. vision tower weights missing from FP8 checkpoints) are
    # materialised to CPU zeros instead of left on meta.  Without this,
    # model_ledger's `.to(device)` call after `.build()` crashes.
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder

    _orig_return_model = SingleGPUModelBuilder._return_model

    def _safe_return_model(self, meta_model, device):
        _materialise_meta_params(meta_model)
        return _orig_return_model(self, meta_model, device)

    SingleGPUModelBuilder._return_model = _safe_return_model

    _facade._LTX_GEMMA_TRANSFORMERS_COMPAT_PATCHED = True


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
    if requested in {"fp8_hooks", "hooks", "fp8"}:
        return "fp8_hooks"
    if requested:
        logger.warning(
            "Unknown SERENITY_LTX_SCALED_FP8_BACKEND=%r; falling back to fp8_hooks",
            requested,
        )

    # Default: full dequant to bf16. FP8 hooks have grid artifacts — need debugging.
    # TODO: fix FP8 hooks for ComfyUI-style per-layer dequant
    return "dequant_bf16"


def _install_fp8_forward_hooks(model: torch.nn.Module, checkpoint_path: str) -> int:
    """Install per-layer FP8 dequant hooks. Keeps weights as FP8 (~14GB) on GPU.

    For each Linear layer with a scaled FP8 weight:
    - Stores the original FP8 weight and scale as attributes
    - Installs a forward pre-hook that dequants weight → bf16 before the forward
    - Installs a forward hook that restores the FP8 weight after the forward

    This way the model stays at FP8 size in VRAM and dequants per-layer on-the-fly,
    matching ComfyUI's approach.
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

    hooked = 0
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

            if not isinstance(parent, nn.Linear):
                # The attr_name is "weight", so parent_path is the Linear module
                pass

            param = getattr(parent, attr_name, None)
            if param is None or not isinstance(param, (torch.Tensor, nn.Parameter)):
                continue

            fp8_weight = f.get_tensor(weight_key)
            scale = f.get_tensor(key)

            # Store FP8 weight and scale as plain attributes
            parent._fp8_weight = fp8_weight
            parent._fp8_scale = scale

            # Replace param data with FP8 weight (keeps model at ~14GB FP8 size)
            param.data = fp8_weight

            def _make_pre_hook(mod):
                def _pre_hook(module, args):
                    w = module._fp8_weight
                    s = module._fp8_scale
                    # Move to weight's device on first use
                    if w.device != module.weight.device:
                        w = w.to(module.weight.device)
                        s = s.to(module.weight.device)
                        module._fp8_weight = w
                        module._fp8_scale = s
                    module.weight.data = dequantize_fp8(w, s, out_dtype=torch.bfloat16)
                return _pre_hook

            def _make_post_hook(mod):
                def _post_hook(module, args, output):
                    # Restore FP8 weight to keep VRAM at FP8 size
                    module.weight.data = module._fp8_weight
                    return output
                return _post_hook

            parent.register_forward_pre_hook(_make_pre_hook(parent))
            parent.register_forward_hook(_make_post_hook(parent))
            hooked += 1

    logger.info("Installed FP8 forward hooks on %d layers (weights stay FP8 on GPU)", hooked)
    return hooked


def _prepare_ltx_scaled_fp8_transformer_for_runtime(
    transformer: torch.nn.Module,
    checkpoint_path: str,
    *,
    stage_label: str,
) -> str:
    """Prepare a raw scaled-FP8 LTX transformer for inference runtime.

    Default: install per-layer FP8 dequant hooks (keeps ~14GB FP8 on GPU).
    Fallback: full dequant to bf16 (needs ~35GB, requires Stagehand).
    """
    backend = getattr(transformer, "_serenity_scaled_fp8_backend", None)
    if backend:
        return backend

    requested = _ltx_scaled_fp8_runtime_backend()

    if requested == "dequant_bf16":
        # Legacy full-dequant path
        fixed = _dequant_scaled_fp8_weights(transformer, checkpoint_path)
        if fixed <= 0:
            backend = "unmodified"
        else:
            backend = "dequant_bf16"
            logger.info("%s scaled FP8: dequantized %d weights to bf16 runtime", stage_label, fixed)
    elif requested == "eriquant_fp8":
        fixed = _dequant_scaled_fp8_weights(transformer, checkpoint_path)
        if fixed <= 0:
            backend = "unmodified"
        else:
            try:
                from serenity.inference.quantization.eriquant_backend import quantize_model_eriquant
                logger.info("%s scaled FP8: dequantized %d weights, converting to eriquant_fp8 runtime", stage_label, fixed)
                quantize_model_eriquant(
                    transformer,
                    mode="eriquant_fp8",
                    arch="default",
                    exclude=list(_LTX_ERIQUANT_FP8_EXCLUDES),
                )
                backend = "eriquant_fp8"
            except Exception as exc:
                logger.warning("%s eriquant_fp8 failed: %s, falling back to fp8_hooks", stage_label, exc)
                backend = None
    else:
        backend = None

    # Default: full dequant to bf16 (correct values, fits on GPU after dequant)
    if backend is None:
        fixed = _dequant_scaled_fp8_weights(transformer, checkpoint_path)
        backend = "dequant_bf16" if fixed > 0 else "unmodified"
        if fixed > 0:
            logger.info("%s scaled FP8: dequantized %d weights to bf16", stage_label, fixed)

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
    is_fp8 = "fp8" in (checkpoint_path or "").lower()
    gemma_root = _resolve_ltxv_gemma_root(model.gemma_root_path, is_fp8=is_fp8)
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


class _FP8GemmaKeyRemapLoader:
    """Wrapper around StateDictLoader that remaps FP8 Gemma key prefixes.

    HuggingFace FP8 Gemma files use ``model.layers.*`` while the standard
    Gemma3-for-LTX format uses ``language_model.model.layers.*``.  This
    loader intercepts the raw safetensors load and renames keys BEFORE the
    SD ops see them, so the standard AV_GEMMA_TEXT_ENCODER_KEY_OPS can
    match and remap them correctly.
    """

    _PREFIX_MAP = (
        ("model.embed_tokens.", "language_model.model.embed_tokens."),
        ("model.layers.",       "language_model.model.layers."),
        ("model.norm.",         "language_model.model.norm."),
        ("vision_model.",       "vision_tower.vision_model."),
    )

    def __init__(self, inner_loader):
        self._inner = inner_loader

    def metadata(self, path):
        return self._inner.metadata(path)

    def load(self, path, sd_ops=None, device=None):
        import safetensors
        from ltx_core.loader.primitives import StateDict

        sd = {}
        size = 0
        dtype_set = set()
        device = device or torch.device("cpu")
        model_paths = path if isinstance(path, list) else [path]

        for shard_path in model_paths:
            with safetensors.safe_open(shard_path, framework="pt", device=str(device)) as f:
                for name in f.keys():
                    # Remap FP8 Gemma keys before SD ops see them
                    remapped = self._remap_key(name)
                    expected_name = remapped if sd_ops is None else sd_ops.apply_to_key(remapped)
                    if expected_name is None:
                        continue
                    value = f.get_tensor(name).to(device=device, non_blocking=True, copy=False)
                    key_value_pairs = ((expected_name, value),)
                    if sd_ops is not None:
                        key_value_pairs = sd_ops.apply_to_key_value(expected_name, value)
                    for key, val in key_value_pairs:
                        size += val.nbytes
                        dtype_set.add(val.dtype)
                        sd[key] = val

        return StateDict(sd=sd, device=device, size=size, dtype=dtype_set)

    def _remap_key(self, key: str) -> str:
        for prefix, replacement in self._PREFIX_MAP:
            if key.startswith(prefix):
                return replacement + key[len(prefix):]
        return key


def _patch_ledger_for_fp8_gemma(ledger: Any, gemma_root: str) -> None:
    """Patch the text_encoder_builder SD ops for FP8 Gemma checkpoints.

    HuggingFace FP8 Gemma files use key prefix ``model.*`` (e.g.
    ``model.layers.0.*``), while the standard Gemma3-for-LTX format uses
    ``language_model.model.*``.  The default SD ops only handle the latter.

    This function detects the key format by reading the first key from the
    safetensors file and, if needed, rebuilds the builder with extended SD ops
    that pre-map ``model.*`` → ``language_model.model.*`` before the standard
    remapping.
    """
    if not gemma_root or not hasattr(ledger, "text_encoder_builder"):
        return

    from pathlib import Path
    from safetensors import safe_open

    root = Path(gemma_root).expanduser()
    sft_files = sorted(root.rglob("*.safetensors"))
    if not sft_files:
        return

    # Check if any weight file uses the short key format
    needs_remap = False
    for sft in sft_files:
        try:
            with safe_open(str(sft), framework="pt") as f:
                for key in f.keys():
                    # Short format: model.layers.0.* or model.embed_tokens.*
                    if key.startswith("model.layers.") or key.startswith("model.embed_tokens."):
                        needs_remap = True
                        break
                    # Standard format: language_model.model.*
                    if key.startswith("language_model.model."):
                        return  # Already in expected format
        except Exception:
            continue
        if needs_remap:
            break

    if not needs_remap:
        return

    logger.info("FP8 Gemma uses short key format — installing key remap loader")

    from dataclasses import replace as dc_replace

    builder = ledger.text_encoder_builder
    ledger.text_encoder_builder = dc_replace(
        builder,
        model_loader=_FP8GemmaKeyRemapLoader(builder.model_loader),
    )


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
    is_fp8 = "fp8" in (resolved_checkpoint or "").lower()
    resolved_gemma = _resolve_ltxv_gemma_root(gemma_path, is_fp8=is_fp8)
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
    _patch_ledger_for_fp8_gemma(ledger, resolved_gemma)

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
        # Cast hidden states to match feature extractor weight dtype (bf16).
        # On CPU the language model may output float32 hidden states.
        fe_dtype = next(self.feature_extractor.parameters()).dtype
        if isinstance(hidden_states, (list, tuple)):
            hidden_states = tuple(s.to(dtype=fe_dtype) for s in hidden_states)
        else:
            hidden_states = hidden_states.to(dtype=fe_dtype)
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
    max_shift: float = 2.05,
    base_shift: float = 0.95,
    decode_timestep: float = 0.05,
    decode_noise_scale: float = 0.025,
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
        multi_modal_guider_factory_denoising_func,
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

        sigmas = LTX2Scheduler().execute(
            steps=steps, max_shift=max_shift, base_shift=base_shift,
        ).to(dtype=torch.float32, device=device)

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

        if stg_scale > 0:
            # STG in distilled mode: use guider pipeline with cfg_scale=1 (no CFG)
            from ltx_core.components.guiders import (
                MultiModalGuiderParams,
                create_multimodal_guider_factory,
            )
            if stg_blocks is None:
                stg_blocks = [28]  # LTX 2.3 default

            distilled_guider_params = MultiModalGuiderParams(
                cfg_scale=1.0,
                stg_scale=stg_scale,
                stg_blocks=stg_blocks,
                rescale_scale=stg_rescale,
            )
            video_guider_factory = create_multimodal_guider_factory(
                params=distilled_guider_params,
            )
            audio_guider_factory = create_multimodal_guider_factory(
                params=distilled_guider_params,
            )
            base_fn = multi_modal_guider_factory_denoising_func(
                video_guider_factory=video_guider_factory,
                audio_guider_factory=audio_guider_factory,
                v_context=video_context,
                a_context=audio_context,
                transformer=transformer,
            )
            logger.info("Distilled mode with STG: scale=%.2f, blocks=%s, rescale=%.2f",
                        stg_scale, stg_blocks, stg_rescale)
        else:
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
    # Override VAE decode params if caller specified non-default values
    if hasattr(video_decoder, 'decode_noise_scale'):
        video_decoder.decode_noise_scale = decode_noise_scale
    if hasattr(video_decoder, 'decode_timestep'):
        video_decoder.decode_timestep = decode_timestep
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
