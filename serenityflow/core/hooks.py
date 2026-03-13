from __future__ import annotations

"""
Tier A: Patch hooks — weight modification during Stagehand H2D (managed by PatchLedger)
Tier B: Inference hooks — activation modification during forward (timestep-aware)
Tier C: Runtime hooks — observation only (telemetry, preview, debug)
"""
from dataclasses import dataclass, field
from typing import Callable, Any
import torch

# ─── Tier B: Inference Hooks ───

@dataclass
class InferenceHook:
    hook_id: str
    target_layers: list[str]        # Layer name patterns to intercept
    timestep_range: tuple[float, float] = (0.0, 1.0)  # sigma range
    strength: float = 1.0
    priority: int = 0               # Lower = runs first

    def should_activate(self, sigma: float) -> bool:
        return self.timestep_range[0] <= sigma <= self.timestep_range[1]

    def apply(self, output: torch.Tensor, sigma: float, **context) -> torch.Tensor:
        raise NotImplementedError

    def matches_layer(self, layer_name: str) -> bool:
        for pattern in self.target_layers:
            if pattern in layer_name or layer_name.startswith(pattern):
                return True
        return False

# ─── Tier C: Runtime Hooks ───

class RuntimeHook:
    """Base class for observation hooks. Never modify model state."""
    def on_block_load(self, block_name: str, size_bytes: int) -> None: pass
    def on_block_evict(self, block_name: str) -> None: pass
    def on_step(self, step: int, total: int, latent: torch.Tensor, sigma: float) -> None: pass
    def on_node_start(self, node_id: str, class_type: str) -> None: pass
    def on_node_end(self, node_id: str, elapsed_ms: float, cache_hit: bool) -> None: pass
    def on_model_load(self, model_id: str, size_bytes: int, elapsed_ms: float) -> None: pass
    def on_model_evict(self, model_id: str) -> None: pass

# ─── Hook Registry ───

class HookRegistry:
    def __init__(self):
        self.inference_hooks: list[InferenceHook] = []
        self.runtime_hooks: list[RuntimeHook] = []

    # Inference hooks
    def register_inference(self, hook: InferenceHook) -> None:
        self.inference_hooks.append(hook)
        self.inference_hooks.sort(key=lambda h: h.priority)

    def unregister_inference(self, hook_id: str) -> None:
        self.inference_hooks = [h for h in self.inference_hooks if h.hook_id != hook_id]

    def get_active_hooks(self, layer_name: str, sigma: float) -> list[InferenceHook]:
        return [h for h in self.inference_hooks
                if h.matches_layer(layer_name) and h.should_activate(sigma)]

    def apply_hooks(self, layer_name: str, output: torch.Tensor,
                    sigma: float, **context) -> torch.Tensor:
        for hook in self.get_active_hooks(layer_name, sigma):
            output = hook.apply(output, sigma, **context)
        return output

    # Runtime hooks
    def register_runtime(self, hook: RuntimeHook) -> None:
        self.runtime_hooks.append(hook)

    def unregister_runtime(self, hook: RuntimeHook) -> None:
        self.runtime_hooks = [h for h in self.runtime_hooks if h is not hook]

    def fire_block_load(self, block_name: str, size_bytes: int) -> None:
        for h in self.runtime_hooks:
            h.on_block_load(block_name, size_bytes)

    def fire_block_evict(self, block_name: str) -> None:
        for h in self.runtime_hooks:
            h.on_block_evict(block_name)

    def fire_step(self, step: int, total: int, latent: torch.Tensor, sigma: float) -> None:
        for h in self.runtime_hooks:
            h.on_step(step, total, latent, sigma)

    def fire_node_start(self, node_id: str, class_type: str) -> None:
        for h in self.runtime_hooks:
            h.on_node_start(node_id, class_type)

    def fire_node_end(self, node_id: str, elapsed_ms: float, cache_hit: bool) -> None:
        for h in self.runtime_hooks:
            h.on_node_end(node_id, elapsed_ms, cache_hit)

    def fire_model_load(self, model_id: str, size_bytes: int, elapsed_ms: float) -> None:
        for h in self.runtime_hooks:
            h.on_model_load(model_id, size_bytes, elapsed_ms)

    def fire_model_evict(self, model_id: str) -> None:
        for h in self.runtime_hooks:
            h.on_model_evict(model_id)
