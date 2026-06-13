"""Automatic diagnostic meta-tool for SerenityFlow debug system."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["DiagnosticRunner"]


@dataclass
class Section:
    name: str
    status: str  # "ok", "warning", "error"
    data: dict[str, Any]
    diagnosis: str


class DiagnosticRunner:
    """Chains multiple debug checks into a structured diagnostic report."""

    def run(
        self,
        mode: str = "full",
        lora_path: str | None = None,
        tensor_paths: list[str] | None = None,
        training_log_dir: str | None = None,
    ) -> dict:
        sections: dict[str, Section] = {}

        # Mode-to-checks mapping
        checks = {
            "full": ["pipeline", "vram", "lora", "weight_health", "logs", "training"],
            "lora": ["pipeline", "lora", "weight_health"],
            "performance": ["pipeline", "vram", "logs"],
            "health": ["pipeline", "vram", "weight_health", "logs", "training"],
        }

        active_checks = checks.get(mode, checks["full"])

        if "pipeline" in active_checks:
            sections["pipeline"] = self._check_pipeline()

        if "vram" in active_checks:
            sections["vram"] = self._check_vram()

        if "lora" in active_checks and lora_path:
            sections["lora_compatibility"] = self._check_lora(lora_path)
        elif "lora" in active_checks and mode == "lora" and not lora_path:
            sections["lora_compatibility"] = Section(
                name="lora_compatibility",
                status="error",
                data={},
                diagnosis="lora_path is required for lora diagnostic mode.",
            )

        if "weight_health" in active_checks:
            sections["weight_health"] = self._check_weight_health(tensor_paths)

        if "logs" in active_checks:
            sections["logs"] = self._check_logs()

        if "training" in active_checks and training_log_dir:
            sections["training"] = self._check_training(training_log_dir)

        # Compute overall status
        statuses = [s.status for s in sections.values()]
        if "error" in statuses:
            overall = "error"
        elif "warning" in statuses:
            overall = "warning"
        else:
            overall = "ok"

        # Generate summary
        summary_parts = [s.diagnosis for s in sections.values()]

        return {
            "mode": mode,
            "overall_status": overall,
            "sections": {k: asdict(v) for k, v in sections.items()},
            "summary": " ".join(summary_parts),
        }

    def _check_pipeline(self) -> Section:
        """Check pipeline state."""
        from serenityflow.debug.router import (
            _get_active_loras,
            _get_server_state,
            _iter_cache_outputs,
            _component_info,
        )

        state = _get_server_state()
        runner = state.runner

        if runner is None:
            return Section("pipeline", "error", {"model_loaded": False},
                          "No pipeline runner active.")

        cache = getattr(runner, "cache", None)
        if cache is None:
            return Section("pipeline", "error", {"model_loaded": False},
                          "Pipeline has no cache.")

        loaded_model = None
        components: dict[str, Any] = {}
        for _nid, out in _iter_cache_outputs(cache):
            type_name = type(out).__name__
            if type_name == "LoadedCheckpoint":
                loaded_model = out
            elif hasattr(out, "_manager") and hasattr(out, "_arch"):
                components["text_encoder"] = _component_info(out, "text_encoder")
            elif hasattr(out, "decoder") and hasattr(out, "encoder"):
                components["vae"] = _component_info(out, "vae")

        if loaded_model is None:
            return Section("pipeline", "warning",
                          {"model_loaded": False, "components": components},
                          "No model loaded in pipeline.")

        model = getattr(loaded_model, "model", None)
        config = getattr(loaded_model, "model_config", None)
        arch = None
        if config:
            arch_val = getattr(config, "architecture", None)
            arch = arch_val.value if hasattr(arch_val, "value") else str(arch_val)

        dtype = str(next(model.parameters()).dtype) if model else "unknown"
        loras = _get_active_loras()

        data = {
            "model_loaded": True,
            "architecture": arch,
            "dtype": dtype,
            "components": components,
            "active_loras": loras,
        }

        lora_info = f" {len(loras)} LoRA(s) active." if loras else ""
        return Section("pipeline", "ok", data,
                       f"{arch or 'Unknown'} model loaded ({dtype}).{lora_info}")

    def _check_vram(self) -> Section:
        """Check VRAM status."""
        try:
            import torch
            if not torch.cuda.is_available():
                return Section("vram", "ok", {"cuda_available": False},
                              "No CUDA GPU detected.")

            allocated = torch.cuda.memory_allocated() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            usage_pct = (allocated / total * 100) if total > 0 else 0

            data = {
                "allocated_gb": round(allocated, 2),
                "total_gb": round(total, 2),
                "usage_pct": round(usage_pct, 1),
            }

            if usage_pct > 98:
                return Section("vram", "error", data,
                              f"VRAM critically full: {usage_pct:.0f}% ({allocated:.1f}/{total:.1f} GB).")
            elif usage_pct > 90:
                return Section("vram", "warning", data,
                              f"VRAM high: {usage_pct:.0f}% ({allocated:.1f}/{total:.1f} GB).")
            else:
                return Section("vram", "ok", data,
                              f"VRAM {usage_pct:.0f}% ({allocated:.1f}/{total:.1f} GB).")
        except Exception as exc:
            return Section("vram", "warning", {"error": str(exc)},
                          f"Could not read VRAM: {exc}")

    def _check_lora(self, lora_path: str) -> Section:
        """Check LoRA compatibility."""
        import os
        if not os.path.isfile(lora_path):
            return Section("lora_compatibility", "error",
                          {"lora_path": lora_path, "exists": False},
                          f"LoRA file not found: {lora_path}")

        try:
            from safetensors import safe_open
            with safe_open(lora_path, framework="pt") as f:
                lora_keys = list(f.keys())
        except Exception as exc:
            return Section("lora_compatibility", "error",
                          {"error": str(exc)},
                          f"Failed to read LoRA file: {exc}")

        from serenityflow.debug.router import (
            _get_server_state, _iter_cache_outputs,
            _strip_lora_suffix, _guess_architecture_from_keys,
        )

        # Guess LoRA architecture
        lora_arch = _guess_architecture_from_keys(lora_keys)

        # Get model keys
        state = _get_server_state()
        runner = state.runner
        model_keys: set[str] = set()
        model_arch = None

        if runner and getattr(runner, "cache", None):
            for _nid, out in _iter_cache_outputs(runner.cache):
                if type(out).__name__ == "LoadedCheckpoint":
                    model = getattr(out, "model", None)
                    config = getattr(out, "model_config", None)
                    if model:
                        model_keys = set(model.state_dict().keys())
                    if config:
                        arch_val = getattr(config, "architecture", None)
                        model_arch = arch_val.value if hasattr(arch_val, "value") else str(arch_val)
                    break

        if not model_keys:
            return Section("lora_compatibility", "warning",
                          {"lora_arch": lora_arch, "model_loaded": False},
                          f"LoRA appears to be {lora_arch or 'unknown'} architecture."
                          " No model loaded to check against.")

        # Key matching
        lora_base_keys = [
            _strip_lora_suffix(k) for k in lora_keys
            if _strip_lora_suffix(k) != k
        ]
        matched = sum(
            1 for k in lora_base_keys
            if k + ".weight" in model_keys or k in model_keys
        )
        total = len(lora_base_keys) if lora_base_keys else 1
        match_ratio = matched / total

        data = {
            "lora_arch": lora_arch,
            "model_arch": model_arch,
            "architectures_match": (
                lora_arch == model_arch if (lora_arch and model_arch) else None
            ),
            "match_ratio": round(match_ratio, 3),
            "matched": matched,
            "total_lora_keys": total,
        }

        if match_ratio < 0.5:
            status = "error"
            diag = (f"LoRA incompatible: {matched}/{total} keys match "
                    f"({match_ratio:.0%}). LoRA={lora_arch}, Model={model_arch}.")
        elif match_ratio < 0.9:
            status = "warning"
            diag = f"LoRA partial match: {matched}/{total} keys ({match_ratio:.0%})."
        else:
            status = "ok"
            diag = f"LoRA compatible: {matched}/{total} keys match ({match_ratio:.0%})."

        return Section("lora_compatibility", status, data, diag)

    def _check_weight_health(self, tensor_paths: list[str] | None = None) -> Section:
        """Check weight health by probing key tensors."""
        from serenityflow.debug.router import _get_server_state, _iter_cache_outputs

        state = _get_server_state()
        runner = state.runner
        if not runner or not getattr(runner, "cache", None):
            return Section("weight_health", "warning", {},
                          "No model loaded to inspect.")

        # Find model
        model = None
        for _nid, out in _iter_cache_outputs(runner.cache):
            if type(out).__name__ == "LoadedCheckpoint":
                model = getattr(out, "model", None)
                break

        if model is None:
            return Section("weight_health", "warning", {},
                          "No model found in cache.")

        # Auto-select tensors if none provided
        if not tensor_paths:
            tensor_paths = []
            param_names = [n for n, _ in model.named_parameters()]
            # Pick representative layers
            attn_q = [
                n for n in param_names
                if "attn" in n and "to_q" in n and "weight" in n
            ]
            norms = [
                n for n in param_names
                if "norm" in n.lower() and "weight" in n
            ]
            if attn_q:
                tensor_paths.append(attn_q[0])
                if len(attn_q) > 1:
                    tensor_paths.append(attn_q[-1])
            if norms:
                tensor_paths.append(norms[0])
            if not tensor_paths and param_names:
                tensor_paths = param_names[:3]

        # Probe each tensor
        import torch
        probes = []
        has_nan = False
        has_inf = False
        has_dead = False

        for path in tensor_paths[:5]:  # cap at 5
            try:
                param = dict(model.named_parameters()).get(path)

                if param is None:
                    probes.append({"path": path, "error": "not found"})
                    continue

                with torch.no_grad():
                    t = param.float()
                    nan_count = int(t.isnan().sum())
                    inf_count = int(t.isinf().sum())
                    zero_pct = float((t == 0).sum() / t.numel() * 100)

                    probes.append({
                        "path": path,
                        "shape": list(param.shape),
                        "dtype": str(param.dtype),
                        "mean": round(float(t.mean()), 6),
                        "std": round(float(t.std()), 6),
                        "nan_count": nan_count,
                        "inf_count": inf_count,
                        "zero_pct": round(zero_pct, 1),
                    })

                    if nan_count > 0:
                        has_nan = True
                    if inf_count > 0:
                        has_inf = True
                    if zero_pct > 99:
                        has_dead = True
            except Exception as exc:
                probes.append({"path": path, "error": str(exc)})

        data = {"probes": probes, "tensors_checked": len(probes)}

        if has_nan or has_inf:
            nan_inf = "/".join(
                s for s in ["NaN" if has_nan else "", "Inf" if has_inf else ""] if s
            )
            return Section("weight_health", "error", data,
                          f"Found {nan_inf} in {len(probes)} probed tensors.")
        elif has_dead:
            return Section("weight_health", "warning", data,
                          f"Dead weights detected (>99% zeros) in {len(probes)} probed tensors.")
        else:
            return Section("weight_health", "ok", data,
                          f"Sampled {len(probes)} tensors. No NaN/Inf/dead weights detected.")

    def _check_logs(self) -> Section:
        """Check recent engine logs for errors/warnings."""
        try:
            from serenityflow.debug.log_buffer import get_handler
            handler = get_handler()
            if handler is None:
                return Section("logs", "ok", {"available": False},
                              "Log buffer not installed.")

            entries = handler.get_entries(n=100)
            errors = [e for e in entries if e["level"] == "ERROR"]
            warnings = [e for e in entries if e["level"] == "WARNING"]

            data = {
                "total_entries": len(entries),
                "error_count": len(errors),
                "warning_count": len(warnings),
                "recent_errors": [
                    {"message": e["message"], "component": e["component"]}
                    for e in errors[-5:]
                ],
                "recent_warnings": [
                    {"message": e["message"], "component": e["component"]}
                    for e in warnings[-5:]
                ],
            }

            if errors:
                return Section("logs", "error", data,
                              f"{len(errors)} errors in last 100 log entries.")
            elif warnings:
                return Section("logs", "warning", data,
                              f"{len(warnings)} warnings in last 100 log entries. No errors.")
            else:
                return Section("logs", "ok", data,
                              "No errors or warnings in recent logs.")
        except Exception as exc:
            return Section("logs", "ok", {"error": str(exc)},
                          f"Could not read logs: {exc}")

    def _check_training(self, log_dir: str) -> Section:
        """Check training metrics."""
        try:
            from serenityflow.debug.training_reader import read_training_metrics
            metrics = read_training_metrics(log_dir=log_dir)

            status = metrics.get("session_status", "unknown")
            summary = metrics.get("summary", {})
            loss_trend = summary.get("loss_trend", "unknown")
            loss_mean = summary.get("loss_last_n_mean")
            step = metrics.get("current_step", 0)

            data = {
                "session_status": status,
                "current_step": step,
                "loss_trend": loss_trend,
                "loss_mean": loss_mean,
                "summary": summary,
            }

            diag_parts = [f"Training {status} at step {step}."]
            if loss_mean is not None:
                diag_parts.append(f"Loss {loss_trend} (avg {loss_mean:.4f}).")

            sec_status = "ok"
            if loss_trend == "increasing":
                sec_status = "warning"
            if status == "crashed":
                sec_status = "error"

            return Section("training", sec_status, data, " ".join(diag_parts))
        except FileNotFoundError:
            return Section("training", "warning",
                          {"log_dir": log_dir},
                          f"No training data found at {log_dir}.")
        except Exception as exc:
            return Section("training", "warning",
                          {"error": str(exc)},
                          f"Could not read training metrics: {exc}")
