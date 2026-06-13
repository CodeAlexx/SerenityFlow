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


class ABCompareRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 4
    guidance_scale: float = 3.5
    seed: int = 42
    lora_path: str = ""
    lora_strength: float = 1.0


class BreakpointGenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 4
    guidance_scale: float = 3.5
    seed: int = 42
    break_at_step: int = 2
    resume_token: str | None = None


class ModelLoadRequest(BaseModel):
    model_path: str
    pipeline_type: str
    quant: str | None = None
    keep_fp8: bool = True


class ModelUnloadRequest(BaseModel):
    component: str = "all"


class PipelineDiffRequest(BaseModel):
    snapshot_a: str | None = None
    snapshot_b: str | None = None
    save_snapshot: str | None = None


class DiagnoseRequest(BaseModel):
    mode: str = "full"
    lora_path: str | None = None
    tensor_paths: list[str] | None = None
    training_log_dir: str | None = None


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


def _get_active_loras():
    """Get currently active LoRAs from the registry."""
    try:
        from serenityflow.bridge.lora_utils import get_lora_registry
        return get_lora_registry().to_dicts()
    except Exception:
        return []


def register_debug_routes(app: FastAPI) -> None:  # noqa: C901 — intentionally flat
    """Register all ``/debug/*`` endpoints on the FastAPI app."""

    # === 1. Pipeline Status ===

    @app.get("/debug/pipeline/status")
    async def pipeline_status():
        state = _get_server_state()
        runner = state.runner
        if runner is None:
            return JSONResponse({
                "pipeline_type": None,
                "model_loaded": False,
                "components": {},
                "active_loras": _get_active_loras(),
            })

        # Walk the runner's cache for loaded model components
        result: dict[str, Any] = {
            "pipeline_type": None,
            "model_loaded": False,
            "components": {},
            "active_loras": _get_active_loras(),
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
                try:
                    pool_stats = pool.stats()
                    sh_info["pinned_pool"] = pool_stats
                except Exception:
                    sh_info["pinned_pool"] = None
            else:
                sh_info["pool_allocated"] = False
                sh_info["pinned_pool"] = None

            runtimes = getattr(coordinator, "_runtimes", {})
            sh_info["active_runtimes"] = len(runtimes)

            # Per-runtime details
            runtime_details: list[dict[str, Any]] = []
            for rt_name, runtime in runtimes.items():
                rt_info: dict[str, Any] = {"name": str(rt_name)}

                # ResidencyMap
                try:
                    residency = getattr(runtime, "_residency", None)
                    if residency is not None:
                        gpu_blocks = residency.gpu_resident_blocks()
                        resident_count = len(list(gpu_blocks)) if gpu_blocks is not None else 0
                        # Total blocks via iteration
                        total_blocks = 0
                        try:
                            for _ in residency:
                                total_blocks += 1
                        except TypeError:
                            total_blocks = resident_count  # fallback
                        evicted_count = total_blocks - resident_count
                        evictable = 0
                        try:
                            evictable = len(list(residency.eviction_candidates(999999, 0)))
                        except Exception:
                            pass
                        rt_info["residency"] = {
                            "resident_blocks": resident_count,
                            "total_blocks": total_blocks,
                            "evicted_blocks": evicted_count,
                            "evictable_blocks": evictable,
                        }
                except Exception:
                    rt_info["residency"] = None

                # BudgetManager
                try:
                    budget = getattr(runtime, "_budget", None)
                    if budget is not None:
                        rt_info["budget"] = {
                            "vram_used_mb": budget.vram_used_mb(),
                            "headroom_mb": budget.headroom_mb(),
                            "above_high_watermark": budget.above_high_watermark(),
                        }
                except Exception:
                    rt_info["budget"] = None

                # StagehandTelemetry
                try:
                    telemetry = getattr(runtime, "_telemetry", None)
                    if telemetry is not None:
                        rt_info["telemetry"] = {
                            "hit_rate": telemetry.hit_rate(),
                            "mean_stall_ms": telemetry.mean_stall_ms(),
                            "vram_trend": telemetry.vram_trend(),
                        }
                except Exception:
                    rt_info["telemetry"] = None

                # BlockRegistry — total model size
                try:
                    registry = getattr(runtime, "_registry", None)
                    if registry is not None:
                        blocks = registry.blocks_in_order()
                        total_bytes = sum(getattr(b, "size_bytes", 0) for b in blocks)
                        rt_info["total_model_size_mb"] = round(total_bytes / (1024 * 1024), 1)
                except Exception:
                    rt_info["total_model_size_mb"] = None

                runtime_details.append(rt_info)

            sh_info["runtime_details"] = runtime_details
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

    # === 10. Breakpoint Generate ===

    _breakpoint_states: dict[str, dict] = {}

    def _cleanup_stale_breakpoints() -> None:
        """Remove breakpoint states older than 5 minutes."""
        now = time.monotonic()
        stale = [k for k, v in _breakpoint_states.items() if now - v["timestamp"] > 300]
        for k in stale:
            del _breakpoint_states[k]

    @app.post("/debug/generate/breakpoint")
    async def debug_generate_breakpoint(req: BreakpointGenerateRequest):
        import asyncio

        def _run_breakpoint_generate() -> dict:
            import time as _time
            import uuid
            import torch
            from serenityflow.debug.log_buffer import get_handler

            _cleanup_stale_breakpoints()

            state = _get_server_state()
            runner = state.runner
            if runner is None:
                return {"_status": 503, "error": "No runner available. Load a model first."}

            cache = getattr(runner, "cache", None)
            if cache is None:
                return {"_status": 503, "error": "No cache available"}

            model = None
            clip = None

            for _nid, out in _iter_cache_outputs(cache):
                type_name = type(out).__name__
                if type_name == "LoadedCheckpoint":
                    model = getattr(out, "model", None)
                elif hasattr(out, "_manager") and hasattr(out, "_arch"):
                    clip = out

            if model is None:
                return {"_status": 503, "error": "No model loaded. Submit a workflow that loads a model first."}

            warnings: list[str] = []
            timing: dict[str, Any] = {}

            # Validate break_at_step
            if req.break_at_step < 1:
                return {"_status": 400, "error": "break_at_step must be >= 1"}
            if req.break_at_step > req.steps:
                return {"_status": 400, "error": f"break_at_step ({req.break_at_step}) exceeds total steps ({req.steps})"}

            # Handle resume
            start_step = None
            resume_latent = None
            positive = None
            negative = None
            if req.resume_token is not None:
                bp_state = _breakpoint_states.get(req.resume_token)
                if bp_state is None:
                    return {"_status": 400, "error": f"Invalid or expired resume_token: {req.resume_token}"}
                start_step = bp_state["completed_steps"]
                resume_latent = bp_state["latent"]
                positive = bp_state["positive"]
                negative = bp_state["negative"]
                warnings.append(f"Resuming from step {start_step}")

            # --- Text encoding ---
            if positive is None:
                if clip is not None:
                    t0 = _time.perf_counter()
                    try:
                        from serenityflow.bridge.sampling import encode_text
                        positive = encode_text(clip, req.prompt)
                        if req.negative_prompt:
                            negative = encode_text(clip, req.negative_prompt)
                    except Exception as exc:
                        warnings.append(f"Text encoding failed: {exc}")
                    timing["text_encode_ms"] = round((_time.perf_counter() - t0) * 1000, 1)
                else:
                    warnings.append("No text encoder loaded — generating with zeros.")
                    embed_dim = 768
                    seq_len = 77
                    pooled_dim = 0
                    for name, param in model.named_parameters():
                        if "txt_in.weight" in name or "context_embedder" in name:
                            embed_dim = param.shape[-1]
                            seq_len = 256
                            break
                        if "y_embedder" in name or "pooled_text_proj" in name:
                            pooled_dim = param.shape[-1]
                    cond: dict[str, Any] = {"cross_attn": torch.zeros(1, seq_len, embed_dim)}
                    if pooled_dim > 0:
                        cond["pooled_output"] = torch.zeros(1, pooled_dim)
                    elif embed_dim > 768:
                        cond["pooled_output"] = torch.zeros(1, 768)
                    positive = [cond]

            # --- Create or restore latent ---
            if resume_latent is not None:
                latent = resume_latent.to(
                    device="cuda" if torch.cuda.is_available() else "cpu")
            else:
                latent_channels = 4
                arch = getattr(model, "_serenity_arch", None)
                if arch is not None:
                    arch_name = arch.value if hasattr(arch, "value") else str(arch)
                    arch_lower = arch_name.lower()
                    if "flux_2" in arch_lower or "klein" in arch_lower:
                        latent_channels = 32
                    elif "flux" in arch_lower or "sd3" in arch_lower:
                        latent_channels = 16
                if hasattr(model, "in_channels"):
                    latent_channels = model.in_channels
                latent_h = req.height // 8
                latent_w = req.width // 8
                latent = torch.zeros(1, latent_channels, latent_h, latent_w,
                                     device="cuda" if torch.cuda.is_available() else "cpu",
                                     dtype=torch.float32)

            # --- Step callback for collecting per-step data ---
            step_data: list[dict] = []
            step_times: list[float] = []

            def on_step(step, total, sigma, denoised):
                step_times.append(_time.perf_counter())
                stats = {}
                if denoised is not None:
                    with torch.no_grad():
                        d = denoised.float()
                        stats = {
                            "mean": round(float(d.mean()), 6),
                            "std": round(float(d.std()), 6),
                            "min": round(float(d.min()), 6),
                            "max": round(float(d.max()), 6),
                            "nan_count": int(d.isnan().sum()),
                        }
                step_data.append({
                    "step": step,
                    "sigma": round(float(sigma), 6) if sigma is not None else None,
                    "latent_stats": stats,
                    "vram_allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 1) if torch.cuda.is_available() else 0,
                })

            # --- Sampling ---
            t0 = _time.perf_counter()
            denoised = None
            try:
                from serenityflow.bridge.sampling import sample
                torch.manual_seed(req.seed)
                is_resuming = resume_latent is not None
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
                    start_step=start_step,
                    end_step=req.break_at_step,
                    add_noise=not is_resuming,
                    step_callback=on_step,
                )
            except Exception as exc:
                warnings.append(f"Sampling failed: {exc}")
                return {
                    "error": f"Sampling failed: {exc}",
                    "timing": timing,
                    "warnings": warnings,
                }
            timing["denoise_ms"] = round((_time.perf_counter() - t0) * 1000, 1)

            # --- Compute per-step elapsed_ms ---
            for idx, sd in enumerate(step_data):
                if idx == 0:
                    sd["elapsed_ms"] = round((step_times[0] - t0) * 1000, 1)
                else:
                    sd["elapsed_ms"] = round((step_times[idx] - step_times[idx - 1]) * 1000, 1)

            # --- Store resume state ---
            token = f"bp_{uuid.uuid4().hex[:12]}"
            completed = req.break_at_step
            if denoised is not None:
                _breakpoint_states[token] = {
                    "latent": denoised.cpu(),
                    "positive": positive,
                    "negative": negative,
                    "completed_steps": completed,
                    "total_steps": req.steps,
                    "timestamp": _time.monotonic(),
                }
            else:
                token = None

            timing["total_ms"] = round(
                sum(v for v in timing.values() if isinstance(v, (int, float))), 1
            )

            # Collect warnings from log buffer
            handler = get_handler()
            if handler:
                new_entries = handler.get_entries(n=50, level="WARNING")
                for entry in new_entries:
                    if entry["message"] not in warnings:
                        warnings.append(entry["message"])

            return {
                "seed_used": req.seed,
                "total_steps": req.steps,
                "completed_steps": completed,
                "per_step": step_data,
                "resume_token": token,
                "timing": timing,
                "warnings": warnings[:20],
            }

        try:
            result = await asyncio.get_event_loop().run_in_executor(None, _run_breakpoint_generate)
            status_code = result.pop("_status", 200)
            if "error" in result and status_code == 200:
                status_code = 500
            return JSONResponse(result, status_code=status_code)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    # === 11. Model Load ===

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

            # Clear LoRA registry
            try:
                from serenityflow.bridge.lora_utils import get_lora_registry
                get_lora_registry().clear()
            except Exception:
                pass

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


    # === 12. A/B Compare ===

    @app.post("/debug/generate/ab_compare")
    async def debug_ab_compare(req: ABCompareRequest):
        import asyncio

        def _run_ab_compare() -> dict:
            import time as _time
            import torch
            import torch.nn.functional as F

            state = _get_server_state()
            runner = state.runner
            if runner is None:
                return {"_status": 503, "error": "No runner available. Load a model first."}

            cache = getattr(runner, "cache", None)
            if cache is None:
                return {"_status": 503, "error": "No cache available"}

            model = None
            clip = None

            for _nid, out in _iter_cache_outputs(cache):
                type_name = type(out).__name__
                if type_name == "LoadedCheckpoint":
                    model = getattr(out, "model", None)
                elif hasattr(out, "_manager") and hasattr(out, "_arch"):
                    clip = out

            if model is None:
                return {"_status": 503, "error": "No model loaded. Submit a workflow that loads a model first."}

            # Validate LoRA path
            lora_path = os.path.expanduser(req.lora_path)
            if not lora_path or not os.path.isfile(lora_path):
                return {"_status": 404, "error": f"LoRA file not found: {lora_path}"}

            warnings: list[str] = []

            # --- Shared text encoding ---
            positive = None
            negative = None

            if clip is not None:
                try:
                    from serenityflow.bridge.sampling import encode_text
                    positive = encode_text(clip, req.prompt)
                    if req.negative_prompt:
                        negative = encode_text(clip, req.negative_prompt)
                except Exception as exc:
                    warnings.append(f"Text encoding failed: {exc}")
            else:
                warnings.append("No text encoder loaded — generating with zeros.")
                embed_dim = 768
                seq_len = 77
                pooled_dim = 0
                for name, param in model.named_parameters():
                    if "txt_in.weight" in name or "context_embedder" in name:
                        embed_dim = param.shape[-1]
                        seq_len = 256
                        break
                    if "y_embedder" in name or "pooled_text_proj" in name:
                        pooled_dim = param.shape[-1]
                cond: dict[str, Any] = {"cross_attn": torch.zeros(1, seq_len, embed_dim)}
                if pooled_dim > 0:
                    cond["pooled_output"] = torch.zeros(1, pooled_dim)
                elif embed_dim > 768:
                    cond["pooled_output"] = torch.zeros(1, 768)
                positive = [cond]

            # --- Shared latent shape ---
            latent_channels = 4
            arch = getattr(model, "_serenity_arch", None)
            if arch is not None:
                arch_name = arch.value if hasattr(arch, "value") else str(arch)
                arch_lower = arch_name.lower()
                if "flux_2" in arch_lower or "klein" in arch_lower:
                    latent_channels = 32
                elif "flux" in arch_lower or "sd3" in arch_lower:
                    latent_channels = 16
            if hasattr(model, "in_channels"):
                latent_channels = model.in_channels
            latent_h = req.height // 8
            latent_w = req.width // 8
            device = "cuda" if torch.cuda.is_available() else "cpu"

            def _make_latent():
                torch.manual_seed(req.seed)
                return torch.randn(1, latent_channels, latent_h, latent_w,
                                   device=device, dtype=torch.float32)

            def _latent_stats(t: torch.Tensor) -> dict:
                d = t.float()
                return {
                    "mean": round(float(d.mean()), 6),
                    "std": round(float(d.std()), 6),
                    "min": round(float(d.min()), 6),
                    "max": round(float(d.max()), 6),
                }

            from serenityflow.bridge.sampling import sample

            # =============== Run A (baseline, no LoRA) ===============
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            vram_before_a = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            latent_a_input = _make_latent()
            t0_a = _time.perf_counter()
            try:
                denoised_a = sample(
                    model=model,
                    latent=latent_a_input,
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
                return {"error": f"Run A (baseline) sampling failed: {exc}", "warnings": warnings}
            time_a = round((_time.perf_counter() - t0_a) * 1000, 1)
            vram_peak_a = torch.cuda.max_memory_allocated() // (1024 * 1024) if torch.cuda.is_available() else 0
            stats_a = _latent_stats(denoised_a)

            # =============== Save model state for restoration ===============
            total_params = sum(p.numel() for p in model.parameters())
            large_model = total_params > 6_000_000_000
            saved_sd = None

            if not large_model:
                saved_sd = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            else:
                warnings.append(
                    f"Model has {total_params / 1e9:.1f}B parameters — too large to "
                    "snapshot. Model will remain LoRA-modified after this call."
                )

            # =============== Apply LoRA ===============
            try:
                from serenityflow.bridge.lora_utils import load_lora, merge_lora_into_model
                lora_sd = load_lora(lora_path)
                merge_lora_into_model(model, lora_sd, strength=req.lora_strength, lora_path=lora_path)
            except Exception as exc:
                # Restore model before returning error
                if saved_sd is not None:
                    model.load_state_dict(saved_sd)
                return {"error": f"LoRA merge failed: {exc}", "warnings": warnings}

            # =============== Run B (with LoRA) ===============
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            latent_b_input = _make_latent()
            t0_b = _time.perf_counter()
            try:
                denoised_b = sample(
                    model=model,
                    latent=latent_b_input,
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
                # Attempt restore even on failure
                if saved_sd is not None:
                    model.load_state_dict(saved_sd)
                return {"error": f"Run B (LoRA) sampling failed: {exc}", "warnings": warnings}
            time_b = round((_time.perf_counter() - t0_b) * 1000, 1)
            vram_peak_b = torch.cuda.max_memory_allocated() // (1024 * 1024) if torch.cuda.is_available() else 0
            stats_b = _latent_stats(denoised_b)

            # =============== Restore model state ===============
            if saved_sd is not None:
                model.load_state_dict(saved_sd)
                del saved_sd

            # Clear LoRA registry entry
            try:
                from serenityflow.bridge.lora_utils import get_lora_registry
                get_lora_registry().clear()
            except Exception:
                pass

            # =============== Comparison metrics ===============
            latent_a_flat = denoised_a.flatten().float()
            latent_b_flat = denoised_b.flatten().float()
            mse = float((latent_a_flat - latent_b_flat).pow(2).mean())
            cosine_sim = float(F.cosine_similarity(latent_a_flat.unsqueeze(0), latent_b_flat.unsqueeze(0)))
            mean_diff = float(denoised_b.float().mean() - denoised_a.float().mean())
            std_diff = float(denoised_b.float().std() - denoised_a.float().std())

            return {
                "seed_used": req.seed,
                "run_a": {
                    "label": "without_lora",
                    "timing": {"denoise_ms": time_a, "total_ms": time_a},
                    "latent_stats": stats_a,
                    "vram_peak_mb": vram_peak_a,
                },
                "run_b": {
                    "label": "with_lora",
                    "lora_path": req.lora_path,
                    "lora_strength": req.lora_strength,
                    "timing": {"denoise_ms": time_b, "total_ms": time_b},
                    "latent_stats": stats_b,
                    "vram_peak_mb": vram_peak_b,
                },
                "comparison": {
                    "latent_mse": round(mse, 6),
                    "latent_cosine_similarity": round(cosine_sim, 6),
                    "latent_mean_diff": round(mean_diff, 6),
                    "latent_std_diff": round(std_diff, 6),
                    "timing_diff_ms": round(time_b - time_a, 1),
                    "model_restored": not large_model,
                },
                "warnings": warnings[:20],
            }

        try:
            result = await asyncio.get_event_loop().run_in_executor(None, _run_ab_compare)
            status_code = result.pop("_status", 200)
            if "error" in result and status_code == 200:
                status_code = 500
            return JSONResponse(result, status_code=status_code)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    # === Training Metrics ===

    @app.get("/debug/training/metrics")
    async def training_metrics(
        log_dir: str | None = None,
        run_name: str | None = None,
        last_n_steps: int = 50,
        tags: str | None = None,
    ):
        """Query training metrics from SerenityBoard's SQLite database."""
        if log_dir is None:
            return JSONResponse(
                {"error": "log_dir parameter is required. Point it at your training output's log directory."},
                status_code=400,
            )

        tag_list = tags.split(",") if tags else None

        try:
            from serenityflow.debug.training_reader import read_training_metrics as _read

            result = _read(
                log_dir=log_dir,
                run_name=run_name,
                last_n_steps=last_n_steps,
                tags=tag_list,
            )
            return JSONResponse(result)
        except FileNotFoundError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        except Exception as exc:
            return JSONResponse({"error": f"Failed to read training metrics: {exc}"}, status_code=500)

    # === 13. Pipeline Diff ===

    _config_snapshots: dict[str, dict] = {}

    def _capture_pipeline_snapshot() -> dict:
        """Capture current pipeline state as a comparable snapshot."""
        import time as _time

        state = _get_server_state()
        snap: dict[str, Any] = {"_timestamp": _time.time()}

        # Pipeline info (reuse pipeline_status logic)
        runner = state.runner
        if runner is None:
            snap["model"] = {"loaded": False}
            snap["text_encoder"] = {"loaded": False}
            snap["vae"] = {"loaded": False}
            snap["active_loras"] = _get_active_loras()
            snap["stagehand"] = None
            try:
                import torch
                snap["engine"] = {
                    "pytorch_version": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                }
            except Exception:
                snap["engine"] = None
            return snap

        cache = getattr(runner, "cache", None)
        loaded_model = loaded_clip = loaded_vae = None
        if cache:
            for _nid, out in _iter_cache_outputs(cache):
                type_name = type(out).__name__
                if type_name == "LoadedCheckpoint":
                    loaded_model = out
                elif hasattr(out, "_manager") and hasattr(out, "_arch"):
                    loaded_clip = out
                elif hasattr(out, "decoder") and hasattr(out, "encoder"):
                    loaded_vae = out

        if loaded_model is not None:
            model = getattr(loaded_model, "model", None)
            config = getattr(loaded_model, "model_config", None)
            snap["model"] = {
                "loaded": True,
                "architecture": (
                    config.architecture.value
                    if hasattr(getattr(config, "architecture", None), "value")
                    else str(getattr(config, "architecture", None))
                ) if config else None,
                "dtype": str(next(model.parameters()).dtype) if model else None,
                "device": str(next(model.parameters()).device) if model else None,
                "param_count": sum(p.numel() for p in model.parameters()) if model else 0,
            }
        else:
            snap["model"] = {"loaded": False}

        snap["text_encoder"] = _component_info(loaded_clip, "text_encoder") if loaded_clip else {"loaded": False}
        snap["vae"] = _component_info(loaded_vae, "vae") if loaded_vae else {"loaded": False}
        snap["active_loras"] = _get_active_loras()

        # Stagehand config
        coordinator = _get_coordinator()
        if coordinator is not None:
            snap["stagehand"] = {
                "vram_budget_mb": getattr(coordinator, "_vram_budget_mb", None),
                "pool_mb": getattr(coordinator, "_pool_mb", None),
            }
        else:
            snap["stagehand"] = None

        # Engine info
        try:
            import torch
            snap["engine"] = {
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            }
        except Exception:
            snap["engine"] = None

        return snap

    def _diff_snapshots(a: dict, b: dict, prefix: str = "") -> tuple[list[dict], list[str]]:
        """Compare two snapshot dicts recursively.

        Returns (differences, identical_keys).
        """
        differences: list[dict] = []
        identical: list[str] = []
        all_keys = sorted(set(list(a.keys()) + list(b.keys())))

        for key in all_keys:
            if key.startswith("_"):  # skip metadata keys like _timestamp
                continue
            full_key = f"{prefix}.{key}" if prefix else key
            val_a = a.get(key)
            val_b = b.get(key)

            if isinstance(val_a, dict) and isinstance(val_b, dict):
                sub_diffs, sub_identical = _diff_snapshots(val_a, val_b, full_key)
                differences.extend(sub_diffs)
                identical.extend(sub_identical)
            elif val_a == val_b:
                identical.append(full_key)
            else:
                # Determine category from top-level key
                category = key if not prefix else prefix.split(".")[0]
                differences.append({
                    "key": full_key,
                    "a": val_a,
                    "b": val_b,
                    "category": category,
                })

        return differences, identical

    @app.post("/debug/pipeline/diff")
    async def pipeline_diff(req: PipelineDiffRequest):
        # Save snapshot mode
        if req.save_snapshot is not None:
            snap = _capture_pipeline_snapshot()
            _config_snapshots[req.save_snapshot] = snap
            return JSONResponse({
                "action": "saved",
                "name": req.save_snapshot,
                "keys_captured": [k for k in snap if not k.startswith("_")],
                "timestamp": snap["_timestamp"],
            })

        # List snapshots mode (all None)
        if req.snapshot_a is None and req.snapshot_b is None:
            listing = {}
            for name, snap in _config_snapshots.items():
                listing[name] = {
                    "timestamp": snap.get("_timestamp"),
                    "keys": [k for k in snap if not k.startswith("_")],
                }
            return JSONResponse({
                "action": "list",
                "snapshots": listing,
                "count": len(listing),
            })

        # Diff mode: resolve snapshots
        def _resolve(name: str | None) -> dict | None:
            if name is None:
                return _capture_pipeline_snapshot()
            return _config_snapshots.get(name)

        snap_a = _resolve(req.snapshot_a)
        snap_b = _resolve(req.snapshot_b)

        if snap_a is None:
            return JSONResponse(
                {"error": f"Snapshot '{req.snapshot_a}' not found. Available: {list(_config_snapshots.keys())}"},
                status_code=404,
            )
        if snap_b is None:
            return JSONResponse(
                {"error": f"Snapshot '{req.snapshot_b}' not found. Available: {list(_config_snapshots.keys())}"},
                status_code=404,
            )

        diffs, identical = _diff_snapshots(snap_a, snap_b)
        return JSONResponse({
            "action": "diff",
            "snapshot_a": req.snapshot_a or "(current)",
            "snapshot_b": req.snapshot_b or "(current)",
            "differences": diffs,
            "identical_count": len(identical),
            "diff_count": len(diffs),
        })

    # === Diagnose (meta-tool) ===

    @app.post("/debug/diagnose")
    async def diagnose(req: DiagnoseRequest):
        from serenityflow.debug.diagnose import DiagnosticRunner
        import asyncio

        runner = DiagnosticRunner()

        def _run():
            return runner.run(
                mode=req.mode,
                lora_path=req.lora_path,
                tensor_paths=req.tensor_paths,
                training_log_dir=req.training_log_dir,
            )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run)
        return JSONResponse(result)


__all__ = ["register_debug_routes"]
