"""Debug introspection router for SerenityFlow.

Exposes ``/debug/*`` endpoints for runtime inspection of the inference
pipeline, VRAM state, LoRA compatibility, tensor statistics, and more.

Register with::

    from serenityflow.debug.router import register_debug_routes
    register_debug_routes(app)
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class LoRACheckRequest(BaseModel):
    lora_path: str
    verbose: bool = True


class TensorProbeRequest(BaseModel):
    tensor_path: str
    component: str | None = None
    histogram_bins: int = 0


class ArchDiffRequest(BaseModel):
    lora_path: str
    model_path: str | None = None


class DebugGenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 4
    guidance_scale: float = 3.5
    seed: int = 42
    lora_path: str | None = None
    return_intermediates: bool = False
    trace_level: str = "full"


class ModelLoadRequest(BaseModel):
    model_path: str
    pipeline_type: str
    quant: str | None = None
    keep_fp8: bool = True


class ModelUnloadRequest(BaseModel):
    component: str = "all"


# ---------------------------------------------------------------------------
# Architecture detection heuristics for LoRA key prefixes
# ---------------------------------------------------------------------------

ARCH_PREFIX_PATTERNS: dict[str, list[str]] = {
    "flux": ["transformer.single_transformer_blocks.", "transformer.transformer_blocks.",
             "single_blocks.", "double_blocks."],
    "sd15": ["model.diffusion_model.input_blocks.", "model.diffusion_model.middle_block.",
             "model.diffusion_model.output_blocks."],
    "sdxl": ["model.diffusion_model.input_blocks.", "model.diffusion_model.middle_block.",
             "model.diffusion_model.output_blocks.", "conditioner."],
    "sd3": ["model.diffusion_model.joint_blocks.", "joint_blocks."],
    "ltxv": ["transformer.transformer_blocks.", "transformer_blocks."],
    "dit": ["blocks.", "transformer_blocks."],
    "unet": ["model.diffusion_model."],
}

# Unique discriminators (checked first → most specific match)
# Includes both native (dotted) and Kohya/ComfyUI (underscored, lora_unet_ prefix) formats.
_ARCH_DISCRIMINATORS: list[tuple[str, list[str]]] = [
    ("flux", [
        "single_transformer_blocks.", "transformer.single_transformer_blocks.",
        "single_blocks.", "lora_unet_single_blocks_",
        # FLUX has both double_blocks AND single_blocks — that's the discriminator
    ]),
    ("sd3", ["joint_blocks.", "model.diffusion_model.joint_blocks.",
             "lora_unet_joint_blocks_"]),
    ("sdxl", ["conditioner."]),
    # sd15 and sdxl both have input_blocks; sdxl distinguished by conditioner above
]


def _has_both_flux_block_types(key_str: str) -> bool:
    """FLUX models have BOTH double_blocks and single_blocks."""
    has_double = ("double_blocks." in key_str or "double_blocks_" in key_str
                  or "lora_unet_double_blocks_" in key_str)
    has_single = ("single_blocks." in key_str or "single_blocks_" in key_str
                  or "lora_unet_single_blocks_" in key_str
                  or "single_transformer_blocks" in key_str)
    return has_double and has_single


def _guess_architecture_from_keys(keys: list[str]) -> str | None:
    """Guess model architecture from state-dict key prefixes."""
    key_str = "\n".join(keys[:500])  # Sample first 500 keys

    # FLUX has a unique dual-block pattern — check first
    if _has_both_flux_block_types(key_str):
        return "flux"

    for arch, discriminators in _ARCH_DISCRIMINATORS:
        if any(d in key_str for d in discriminators):
            return arch

    # Fallback: check generic patterns (native + Kohya naming)
    if ("model.diffusion_model.input_blocks." in key_str
            or "lora_unet_input_blocks_" in key_str
            or "lora_unet_down_blocks_" in key_str):
        # Could be SD15 or SDXL — check for SDXL discriminators
        if ("lora_te1_" in key_str and "lora_te2_" in key_str) or "conditioner." in key_str:
            return "sdxl"
        # Kohya SDXL uses lora_unet_down_blocks (not input_blocks)
        if "lora_unet_down_blocks_" in key_str:
            return "sdxl"
        return "sd15"
    if "blocks." in key_str or "transformer_blocks." in key_str:
        return "dit"

    return None


def _strip_lora_suffix(key: str) -> str:
    """Strip LoRA adapter suffixes to get the base model key."""
    for suffix in (
        ".lora_down.weight", ".lora_up.weight",
        ".lora_A.weight", ".lora_B.weight",
        ".lora_A.default.weight", ".lora_B.default.weight",
        ".lora_down", ".lora_up",
        ".lora_A", ".lora_B",
    ):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_server_state():
    """Import server state lazily to avoid circular imports."""
    from serenityflow.server.app import state
    return state


def _get_coordinator():
    """Get StagehandCoordinator if available."""
    try:
        from serenityflow.memory.coordinator import get_coordinator
        return get_coordinator()
    except ImportError:
        return None


def _iter_cache_outputs(cache: Any):
    """Yield (node_id, output_item) from the runner cache, unwrapping CachedOutput."""
    store = getattr(cache, "cache", {})
    for node_id, cached in store.items():
        # CachedOutput has .outputs attribute; raw tuples don't
        outputs = getattr(cached, "outputs", cached)
        if not isinstance(outputs, (tuple, list)):
            continue
        for item in outputs:
            if item is not None:
                yield node_id, item


def _component_info(component: Any, name: str) -> dict:
    """Extract dtype/device/param info from a pipeline component."""
    if component is None:
        return {"loaded": False}

    info: dict[str, Any] = {"loaded": True}

    # Get the actual module (unwrap wrappers)
    module = component
    if hasattr(component, "_manager"):
        # CLIPWrapper
        module = component._manager
    elif hasattr(component, "decoder"):
        # VAEWrapper
        module = component.decoder

    try:
        import torch
        if isinstance(module, torch.nn.Module):
            params = list(module.parameters())
            if params:
                info["dtype"] = str(params[0].dtype)
                info["device"] = str(params[0].device)
            info["params"] = sum(p.numel() for p in params)
    except Exception:
        pass

    return info


def _safe_read_safetensors_keys(path: str) -> list[str]:
    """Read tensor keys from a safetensors file without loading tensors."""
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            return list(f.keys())
    except Exception as exc:
        log.warning("Failed to read safetensors keys from %s: %s", path, exc)
        return []


def _safe_read_safetensors_metadata(path: str) -> dict:
    """Read metadata from a safetensors file header."""
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            return dict(f.metadata()) if f.metadata() else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def register_debug_routes(app: FastAPI) -> None:  # noqa: C901 — intentionally flat
    """Register all ``/debug/*`` endpoints on the FastAPI app."""

    # === 1. Pipeline Status ===

    @app.get("/debug/pipeline/status")
    async def pipeline_status():
        state = _get_server_state()
        runner = state.runner
        if runner is None:
            return JSONResponse({"pipeline_type": None, "model_loaded": False})

        # Walk the runner's cache for loaded model components
        result: dict[str, Any] = {
            "pipeline_type": None,
            "model_loaded": False,
            "components": {},
            "active_loras": [],
        }

        # Check if there's a loaded checkpoint in the cache
        cache = getattr(runner, "cache", None)
        if cache is None:
            return JSONResponse(result)

        # Scan cache entries for model components
        loaded_model = None
        loaded_clip = None
        loaded_vae = None

        for _nid, out in _iter_cache_outputs(cache):
            type_name = type(out).__name__
            if type_name == "LoadedCheckpoint":
                loaded_model = out
            elif hasattr(out, "_manager") and hasattr(out, "_arch"):
                loaded_clip = out
            elif hasattr(out, "decoder") and hasattr(out, "encoder"):
                loaded_vae = out

        if loaded_model is not None:
            result["model_loaded"] = True
            model = getattr(loaded_model, "model", None)
            config = getattr(loaded_model, "model_config", None)
            adapter = getattr(loaded_model, "adapter", None)

            if config is not None:
                arch = getattr(config, "architecture", None)
                result["pipeline_type"] = arch.value if hasattr(arch, "value") else str(arch)

            if model is not None:
                result["components"]["transformer"] = _component_info(model, "transformer")

        if loaded_clip is not None:
            result["components"]["text_encoder"] = _component_info(loaded_clip, "text_encoder")

        if loaded_vae is not None:
            result["components"]["vae"] = _component_info(loaded_vae, "vae")

        return JSONResponse(result)

    # === 2. VRAM Status ===

    @app.get("/debug/vram/status")
    async def vram_status():
        result: dict[str, Any] = {"gpu": None, "stagehand": None, "system_ram": None}

        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                allocated = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                total = props.total_memory
                result["gpu"] = {
                    "name": props.name,
                    "total_vram_mb": total // (1024 * 1024),
                    "allocated_mb": allocated // (1024 * 1024),
                    "reserved_mb": reserved // (1024 * 1024),
                    "free_mb": (total - allocated) // (1024 * 1024),
                }
        except Exception:
            pass

        coordinator = _get_coordinator()
        if coordinator is not None:
            sh_info: dict[str, Any] = {
                "vram_budget_mb": getattr(coordinator, "_vram_budget_mb", None),
                "pool_mb": getattr(coordinator, "_pool_mb", None),
            }
            pool = getattr(coordinator, "_pool", None)
            if pool is not None:
                sh_info["pool_allocated"] = True
            else:
                sh_info["pool_allocated"] = False
            runtimes = getattr(coordinator, "_runtimes", {})
            sh_info["active_runtimes"] = len(runtimes)
            result["stagehand"] = sh_info

        try:
            import psutil
            mem = psutil.virtual_memory()
            result["system_ram"] = {
                "total_mb": mem.total // (1024 * 1024),
                "available_mb": mem.available // (1024 * 1024),
                "used_mb": mem.used // (1024 * 1024),
                "percent": mem.percent,
            }
        except ImportError:
            pass

        return JSONResponse(result)

    # === 3. Engine Logs ===

    @app.get("/debug/logs")
    async def engine_logs(lines: int = 100, level: str | None = None, component: str | None = None):
        from serenityflow.debug.log_buffer import get_handler

        handler = get_handler()
        if handler is None:
            return JSONResponse({
                "lines": [],
                "total_available": 0,
                "handler_installed": False,
            })

        entries = handler.get_entries(n=lines, level=level, component=component)
        return JSONResponse({
            "lines": entries,
            "total_available": handler.total,
            "truncated": handler.total > lines,
        })

    # === 4. Models Available ===

    @app.get("/debug/models/available")
    async def models_available():
        result: dict[str, list[dict]] = {"models": [], "loras": []}

        try:
            from serenityflow.bridge.model_paths import get_model_paths
            mp = get_model_paths()

            # Checkpoints / diffusion models
            for folder in ("checkpoints", "diffusion_models"):
                for name in mp.list_models(folder):
                    path = mp._find_exact(name, folder)
                    if path is None:
                        continue
                    entry: dict[str, Any] = {
                        "name": name,
                        "path": path,
                        "folder": folder,
                    }
                    try:
                        entry["size_gb"] = round(os.path.getsize(path) / (1024**3), 2)
                    except OSError:
                        entry["size_gb"] = None

                    # Read safetensors metadata for quant info
                    if path.endswith(".safetensors"):
                        meta = _safe_read_safetensors_metadata(path)
                        entry["quant"] = meta.get("quantization") or meta.get("format")
                    result["models"].append(entry)

            # LoRAs
            for name in mp.list_models("loras"):
                path = mp._find_exact(name, "loras")
                if path is None:
                    continue
                lora_entry: dict[str, Any] = {
                    "name": name,
                    "path": path,
                }
                try:
                    lora_entry["size_mb"] = round(os.path.getsize(path) / (1024**2), 1)
                except OSError:
                    lora_entry["size_mb"] = None

                if path.endswith(".safetensors"):
                    lora_entry["metadata"] = _safe_read_safetensors_metadata(path)
                result["loras"].append(lora_entry)

        except Exception as exc:
            log.warning("Failed to enumerate models: %s", exc)
            result["error"] = str(exc)

        return JSONResponse(result)

    # === 5. Config Dump ===

    @app.get("/debug/config")
    async def config_dump():
        state = _get_server_state()
        result: dict[str, Any] = {}

        # Stagehand config
        sh_cfg = getattr(state, "stagehand_config", {})
        result["stagehand_config"] = dict(sh_cfg) if sh_cfg else {}

        coordinator = _get_coordinator()
        if coordinator is not None:
            result["stagehand_runtime"] = {
                "vram_budget_mb": getattr(coordinator, "_vram_budget_mb", None),
                "pool_mb": getattr(coordinator, "_pool_mb", None),
                "prefetch_window": getattr(coordinator, "_prefetch_window", None),
                "telemetry": getattr(coordinator, "_telemetry", None),
                "block_threshold_mb": getattr(coordinator, "_block_threshold_mb", None),
            }

        # Engine info
        try:
            import torch
            result["engine"] = {
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda or "N/A",
            }
            if torch.cuda.is_available():
                result["engine"]["device_name"] = torch.cuda.get_device_properties(0).name
        except Exception:
            pass

        # Output dirs
        result["directories"] = {
            "output": state.output_dir,
            "input": state.input_dir,
            "temp": state.temp_dir,
        }

        return JSONResponse(result)

    # === 6. Tensor Probe ===

    @app.post("/debug/tensor/probe")
    async def tensor_probe(req: TensorProbeRequest):
        state = _get_server_state()
        runner = state.runner
        if runner is None:
            return JSONResponse({"error": "No runner available"}, status_code=503)

        # Find the target module in cached components
        target_module = None
        cache = getattr(runner, "cache", None)
        if cache is not None:
            for _nid, out in _iter_cache_outputs(cache):
                model = None
                if type(out).__name__ == "LoadedCheckpoint":
                    model = getattr(out, "model", None)
                elif req.component == "vae" and hasattr(out, "decoder"):
                    model = out.decoder
                elif req.component == "text_encoder" and hasattr(out, "_manager"):
                    model = out._manager

                if model is None:
                    continue

                try:
                    import torch
                    if isinstance(model, torch.nn.Module):
                        target_module = model
                        break
                except Exception:
                    pass

        if target_module is None:
            return JSONResponse({"error": "No model loaded or component not found"}, status_code=404)

        # Navigate to the tensor
        try:
            import torch

            # Try get_submodule + .weight first
            parts = req.tensor_path.rsplit(".", 1)
            tensor = None

            if len(parts) == 2:
                module_path, param_name = parts
                try:
                    sub = target_module.get_submodule(module_path)
                    tensor = getattr(sub, param_name, None)
                except (AttributeError, ModuleNotFoundError):
                    pass

            if tensor is None:
                # Try as a full parameter path
                for name, param in target_module.named_parameters():
                    if name == req.tensor_path:
                        tensor = param
                        break

            if tensor is None:
                return JSONResponse({"error": f"Tensor '{req.tensor_path}' not found"}, status_code=404)

            # Compute stats (dequant FP8 if needed)
            t = tensor.detach()
            if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                t = t.float()

            stats = {
                "mean": float(t.float().mean()),
                "std": float(t.float().std()),
                "min": float(t.float().min()),
                "max": float(t.float().max()),
                "abs_mean": float(t.float().abs().mean()),
                "nan_count": int(t.float().isnan().sum()),
                "inf_count": int(t.float().isinf().sum()),
                "zero_count": int((t == 0).sum()),
                "numel": t.numel(),
            }

            result: dict[str, Any] = {
                "path": req.tensor_path,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "stats": stats,
            }

            # Optional histogram
            if req.histogram_bins > 0:
                hist = torch.histogram(t.float().cpu(), bins=req.histogram_bins)
                result["histogram"] = {
                    "counts": hist.hist.tolist(),
                    "bin_edges": hist.bin_edges.tolist(),
                }

            return JSONResponse(result)

        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    # === 7. LoRA Compatibility Check ===

    @app.post("/debug/lora/check")
    async def lora_check(req: LoRACheckRequest):
        path = os.path.expanduser(req.lora_path)
        if not os.path.isfile(path):
            return JSONResponse({"error": f"File not found: {path}"}, status_code=404)

        # Read LoRA keys
        lora_keys = _safe_read_safetensors_keys(path)
        if not lora_keys:
            return JSONResponse({"error": "Could not read LoRA keys"}, status_code=400)

        metadata = _safe_read_safetensors_metadata(path)

        # Get base keys from lora (strip lora suffixes)
        lora_base_keys: dict[str, str] = {}  # base_key -> original lora key
        for key in lora_keys:
            base = _strip_lora_suffix(key)
            if base != key:  # Only include actual LoRA keys
                lora_base_keys[base] = key

        # Get model keys from currently loaded model
        model_keys: set[str] = set()
        state = _get_server_state()
        runner = state.runner
        model = None

        if runner is not None:
            cache = getattr(runner, "cache", None)
            if cache is not None:
                for _nid, out in _iter_cache_outputs(cache):
                    if type(out).__name__ == "LoadedCheckpoint":
                        model = getattr(out, "model", None)
                        break

        if model is not None:
            try:
                import torch
                if isinstance(model, torch.nn.Module):
                    model_keys = set(model.state_dict().keys())
            except Exception:
                pass

        if not model_keys:
            return JSONResponse({
                "error": "No model currently loaded. Load a model first to check LoRA compatibility.",
                "lora_keys_count": len(lora_base_keys),
                "metadata": metadata,
            }, status_code=503)

        # Match keys
        matched = []
        missed = []

        for base_key, lora_key in lora_base_keys.items():
            # Try direct match
            if base_key + ".weight" in model_keys:
                match_info = {"lora_key": lora_key, "model_key": base_key + ".weight"}
                matched.append(match_info)
            elif base_key in model_keys:
                match_info = {"lora_key": lora_key, "model_key": base_key}
                matched.append(match_info)
            else:
                missed.append({
                    "lora_key": lora_key,
                    "reason": "no matching key in model",
                })

        total = len(lora_base_keys)
        match_ratio = len(matched) / total if total > 0 else 0.0

        result: dict[str, Any] = {
            "compatible": match_ratio > 0.5,
            "match_ratio": round(match_ratio, 4),
            "matched_count": len(matched),
            "missed_count": len(missed),
            "total_lora_keys": total,
            "metadata": metadata,
        }

        if req.verbose:
            result["matched_keys"] = matched[:50]  # Cap to avoid huge responses
            result["missed_keys"] = missed[:50]

        return JSONResponse(result)

    # === 8. Architecture Diff ===

    @app.post("/debug/architecture/diff")
    async def architecture_diff(req: ArchDiffRequest):
        lora_path = os.path.expanduser(req.lora_path)
        if not os.path.isfile(lora_path):
            return JSONResponse({"error": f"LoRA file not found: {lora_path}"}, status_code=404)

        # Read LoRA keys
        lora_keys = _safe_read_safetensors_keys(lora_path)
        if not lora_keys:
            return JSONResponse({"error": "Could not read LoRA keys"}, status_code=400)

        lora_base_keys = [_strip_lora_suffix(k) for k in lora_keys if _strip_lora_suffix(k) != k]

        # Get model keys
        model_keys: list[str] = []
        model_source = "none"

        if req.model_path:
            model_path = os.path.expanduser(req.model_path)
            if not os.path.isfile(model_path):
                return JSONResponse({"error": f"Model file not found: {model_path}"}, status_code=404)
            model_keys = _safe_read_safetensors_keys(model_path)
            model_source = model_path
        else:
            # Use currently loaded model
            state = _get_server_state()
            runner = state.runner
            if runner is not None:
                cache = getattr(runner, "cache", None)
                if cache is not None:
                    for _nid, out in _iter_cache_outputs(cache):
                        if type(out).__name__ == "LoadedCheckpoint":
                            m = getattr(out, "model", None)
                            if m is not None:
                                try:
                                    import torch
                                    if isinstance(m, torch.nn.Module):
                                        model_keys = list(m.state_dict().keys())
                                        model_source = "loaded_model"
                                except Exception:
                                    pass
                            break

        lora_arch = _guess_architecture_from_keys(lora_base_keys)
        model_arch = _guess_architecture_from_keys(model_keys) if model_keys else None

        # Extract unique prefixes (up to 3 levels deep)
        def _extract_prefixes(keys: list[str], depth: int = 3) -> list[str]:
            prefixes: set[str] = set()
            for k in keys:
                parts = k.split(".")
                for d in range(1, min(depth + 1, len(parts) + 1)):
                    prefixes.add(".".join(parts[:d]) + ".")
            return sorted(prefixes)[:20]

        lora_prefixes = _extract_prefixes(lora_base_keys)
        model_prefixes = _extract_prefixes(model_keys) if model_keys else []

        architectures_match = (
            lora_arch is not None
            and model_arch is not None
            and lora_arch == model_arch
        )

        # Generate diagnosis
        if not model_keys:
            diagnosis = "No model provided or loaded. Cannot compare architectures."
            suggestion = "Load a model or provide a model_path to compare against."
        elif architectures_match:
            diagnosis = f"Both LoRA and model appear to be {lora_arch} architecture. Keys should be compatible."
            suggestion = "Use /debug/lora/check for detailed key-level matching."
        elif lora_arch and model_arch:
            diagnosis = (
                f"LoRA appears to be {lora_arch} architecture, "
                f"model appears to be {model_arch}. These are likely incompatible."
            )
            suggestion = f"Load a {lora_arch} model to use this LoRA."
        else:
            diagnosis = "Could not determine architecture from key patterns."
            suggestion = "Check key prefixes manually for compatibility."

        return JSONResponse({
            "lora_architecture_guess": lora_arch,
            "model_architecture": model_arch,
            "architectures_match": architectures_match,
            "lora_key_prefixes": lora_prefixes,
            "model_key_prefixes": model_prefixes,
            "model_source": model_source,
            "diagnosis": diagnosis,
            "suggestion": suggestion,
        })

    # === 9. Debug Generate ===

    @app.post("/debug/generate")
    async def debug_generate(req: DebugGenerateRequest):
        import asyncio

        def _run_generate() -> dict:
            import time
            import torch
            from serenityflow.debug.log_buffer import get_handler

            state = _get_server_state()
            runner = state.runner
            if runner is None:
                return {"error": "No runner available"}

            # Find loaded components from cache
            cache = getattr(runner, "cache", None)
            if cache is None:
                return {"error": "No cache available"}

            model = None
            clip = None
            vae = None

            for _nid, out in _iter_cache_outputs(cache):
                type_name = type(out).__name__
                if type_name == "LoadedCheckpoint":
                    model = getattr(out, "model", None)
                elif hasattr(out, "_manager") and hasattr(out, "_arch"):
                    clip = out
                elif hasattr(out, "decoder") and hasattr(out, "encoder"):
                    vae = out

            if model is None:
                return {"error": "No model loaded. Submit a workflow that loads a model first."}

            warnings: list[str] = []
            timing: dict[str, Any] = {}
            result: dict[str, Any] = {}

            # Capture log buffer position for warnings
            handler = get_handler()
            log_start = handler.total if handler else 0

            # --- Text encoding ---
            text_embed_info: dict[str, Any] = {}
            positive = None
            negative = None

            if clip is not None:
                t0 = time.perf_counter()
                try:
                    from serenityflow.bridge.sampling import encode_text
                    positive = encode_text(clip, req.prompt)
                    if req.negative_prompt:
                        negative = encode_text(clip, req.negative_prompt)
                except Exception as exc:
                    warnings.append(f"Text encoding failed: {exc}")
                timing["text_encode_ms"] = round((time.perf_counter() - t0) * 1000, 1)

                # Record embedding info
                if positive:
                    for i, cond in enumerate(positive):
                        ca = cond.get("cross_attn")
                        if ca is not None:
                            text_embed_info[f"encoder_{i}"] = {
                                "shape": list(ca.shape),
                                "norm": round(float(ca.float().norm()), 4),
                            }
            else:
                warnings.append("No text encoder loaded — generating with zeros.")
                # Detect correct text embedding dim from the model's txt_in layer
                embed_dim = 768  # default for SD 1.5
                seq_len = 77
                pooled_dim = 0
                for name, param in model.named_parameters():
                    if "txt_in.weight" in name or "context_embedder" in name:
                        # txt_in shape: [hidden, embed_dim]
                        embed_dim = param.shape[-1]
                        seq_len = 256  # transformer models use longer sequences
                        break
                    if "y_embedder" in name or "pooled_text_proj" in name:
                        pooled_dim = param.shape[-1]

                cond: dict[str, Any] = {"cross_attn": torch.zeros(1, seq_len, embed_dim)}
                if pooled_dim > 0:
                    cond["pooled_output"] = torch.zeros(1, pooled_dim)
                elif embed_dim > 768:
                    # FLUX/SD3 usually need pooled output
                    cond["pooled_output"] = torch.zeros(1, 768)
                positive = [cond]

            # --- Create empty latent ---
            latent_channels = 4
            arch = getattr(model, "_serenity_arch", None)
            if arch is not None:
                arch_name = arch.value if hasattr(arch, "value") else str(arch)
                arch_lower = arch_name.lower()
                if "flux_2" in arch_lower or "klein" in arch_lower:
                    latent_channels = 32  # FLUX.2 / Klein uses 32ch
                elif "flux" in arch_lower or "sd3" in arch_lower:
                    latent_channels = 16
            # Check model attribute as fallback
            if hasattr(model, "in_channels"):
                latent_channels = model.in_channels
            latent_h = req.height // 8
            latent_w = req.width // 8
            latent = torch.zeros(1, latent_channels, latent_h, latent_w,
                                 device="cuda" if torch.cuda.is_available() else "cpu",
                                 dtype=torch.float32)

            # --- VRAM snapshot before ---
            vram_before = 0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                vram_before = torch.cuda.memory_allocated() // (1024 * 1024)

            # --- Sampling ---
            t0 = time.perf_counter()
            denoised = None
            try:
                from serenityflow.bridge.sampling import sample
                torch.manual_seed(req.seed)
                denoised = sample(
                    model=model,
                    latent=latent,
                    positive=positive,
                    negative=negative,
                    seed=req.seed,
                    steps=req.steps,
                    cfg=req.guidance_scale,
                    sampler_name="euler",
                    scheduler="simple",
                    denoise=1.0,
                )
            except Exception as exc:
                warnings.append(f"Sampling failed: {exc}")
                return {
                    "error": f"Sampling failed: {exc}",
                    "timing": timing,
                    "warnings": warnings,
                }
            timing["denoise_ms"] = round((time.perf_counter() - t0) * 1000, 1)

            # --- Latent stats ---
            latent_stats: dict[str, Any] = {}
            if denoised is not None:
                d = denoised.float()
                latent_stats["final"] = {
                    "mean": round(float(d.mean()), 4),
                    "std": round(float(d.std()), 4),
                    "min": round(float(d.min()), 4),
                    "max": round(float(d.max()), 4),
                }

            # --- VRAM snapshot peak ---
            vram_peak = 0
            vram_after = 0
            if torch.cuda.is_available():
                vram_peak = torch.cuda.max_memory_allocated() // (1024 * 1024)
                vram_after = torch.cuda.memory_allocated() // (1024 * 1024)

            # --- VAE decode ---
            image_base64 = None
            if denoised is not None and vae is not None:
                t0 = time.perf_counter()
                try:
                    from serenityflow.bridge.sampling import vae_decode
                    pixels = vae_decode(vae, denoised)
                    timing["vae_decode_ms"] = round((time.perf_counter() - t0) * 1000, 1)

                    # Convert to PNG base64
                    import base64
                    import io
                    from PIL import Image

                    img = pixels.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    img = (img * 255).astype("uint8")
                    pil_img = Image.fromarray(img)
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    image_base64 = base64.b64encode(buf.getvalue()).decode()
                except Exception as exc:
                    warnings.append(f"VAE decode failed: {exc}")
                    timing["vae_decode_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            elif vae is None:
                warnings.append("No VAE loaded — returning latent stats only.")

            # Compute total
            timing["total_ms"] = round(
                sum(v for v in timing.values() if isinstance(v, (int, float))), 1
            )

            # Collect warnings from log buffer during generation
            if handler:
                new_entries = handler.get_entries(
                    n=50, level="WARNING",
                )
                for entry in new_entries:
                    if entry["message"] not in [w for w in warnings]:
                        warnings.append(entry["message"])

            result = {
                "seed_used": req.seed,
                "timing": timing,
                "vram_trace": {
                    "before_mb": vram_before,
                    "peak_mb": vram_peak,
                    "after_mb": vram_after,
                },
                "latent_stats": latent_stats,
                "text_embeddings": text_embed_info,
                "warnings": warnings[:20],
                "errors": [],
            }
            if image_base64:
                result["image_base64"] = image_base64

            return result

        # Run in thread to not block the event loop
        try:
            import asyncio
            result = await asyncio.get_event_loop().run_in_executor(None, _run_generate)
            status = 200 if "error" not in result else 500
            return JSONResponse(result, status_code=status)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    # === 10. Model Load ===

    @app.post("/debug/model/load")
    async def model_load(req: ModelLoadRequest):
        import asyncio

        def _run_load() -> dict:
            import time
            import torch

            path = os.path.expanduser(req.model_path)
            if not os.path.isfile(path):
                return {"error": f"File not found: {path}"}

            t0 = time.perf_counter()

            try:
                from serenityflow.bridge.loading import load_checkpoint, load_diffusion_model

                state = _get_server_state()

                if req.pipeline_type == "checkpoint":
                    # Full checkpoint (model + clip + vae)
                    dtype = req.quant or "bfloat16"
                    model, clip, vae = load_checkpoint(path, dtype=dtype)

                    # Store in runner cache so other endpoints can find them
                    if state.runner is not None:
                        runner = state.runner
                        cache = getattr(runner, "cache", None)
                        if cache is not None:
                            from serenityflow.bridge.loading import LoadedCheckpoint
                            ckpt = LoadedCheckpoint(
                                model=model,
                                vae_decoder=vae.decoder if vae else None,
                                vae_encoder=vae.encoder if vae else None,
                                model_config=getattr(model, "_serenity_model_config", None),
                                adapter=None,
                            )
                            cache.set("debug_model", (ckpt,), {}, "debug_load")
                            if clip is not None:
                                cache.set("debug_clip", (clip,), {}, "debug_load")
                            if vae is not None:
                                cache.set("debug_vae", (vae,), {}, "debug_load")

                else:
                    # Standalone diffusion model (UNet/DiT)
                    dtype = req.quant or "default"
                    model = load_diffusion_model(path, dtype=dtype)

                    if state.runner is not None:
                        runner = state.runner
                        cache = getattr(runner, "cache", None)
                        if cache is not None:
                            from serenityflow.bridge.loading import LoadedCheckpoint
                            config = getattr(model, "_serenity_model_config", None)
                            ckpt = LoadedCheckpoint(
                                model=model,
                                vae_decoder=None,
                                vae_encoder=None,
                                model_config=config,
                                adapter=None,
                            )
                            cache.set("debug_model", (ckpt,), {}, "debug_load")

                load_time = round((time.perf_counter() - t0) * 1000, 1)

                vram_info = {}
                if torch.cuda.is_available():
                    vram_info = {
                        "allocated_mb": torch.cuda.memory_allocated() // (1024 * 1024),
                        "reserved_mb": torch.cuda.memory_reserved() // (1024 * 1024),
                    }

                return {
                    "status": "loaded",
                    "path": path,
                    "pipeline_type": req.pipeline_type,
                    "load_time_ms": load_time,
                    "vram": vram_info,
                }

            except Exception as exc:
                return {"error": str(exc)}

        try:
            result = await asyncio.get_event_loop().run_in_executor(None, _run_load)
            status = 200 if "error" not in result else 500
            return JSONResponse(result, status_code=status)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    # === 11. Model Unload ===

    @app.post("/debug/model/unload")
    async def model_unload(req: ModelUnloadRequest):
        import asyncio

        def _run_unload() -> dict:
            import gc
            import torch

            state = _get_server_state()
            runner = state.runner
            freed_components: list[str] = []

            if runner is not None:
                cache = getattr(runner, "cache", None)
                if cache is not None:
                    store = getattr(cache, "cache", {})
                    if req.component == "all":
                        keys_to_remove = list(store.keys())
                    else:
                        keys_to_remove = [k for k in store.keys() if req.component in k]

                    for key in keys_to_remove:
                        del store[key]
                        freed_components.append(key)

            # Release Stagehand runtimes
            coordinator = _get_coordinator()
            if coordinator is not None and req.component == "all":
                try:
                    coordinator.shutdown()
                except Exception:
                    pass

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            vram_info = {}
            if torch.cuda.is_available():
                vram_info = {
                    "allocated_mb": torch.cuda.memory_allocated() // (1024 * 1024),
                    "reserved_mb": torch.cuda.memory_reserved() // (1024 * 1024),
                }

            return {
                "status": "unloaded",
                "freed_components": freed_components,
                "vram_after": vram_info,
            }

        try:
            result = await asyncio.get_event_loop().run_in_executor(None, _run_unload)
            return JSONResponse(result)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)


__all__ = ["register_debug_routes"]
