from __future__ import annotations
import torch
import pytest
from serenityflow.core.hooks import InferenceHook, RuntimeHook, HookRegistry


class ScaleHook(InferenceHook):
    """Test hook that scales output by strength."""
    def apply(self, output: torch.Tensor, sigma: float, **context) -> torch.Tensor:
        return output * self.strength


class RecordingRuntimeHook(RuntimeHook):
    def __init__(self):
        self.events = []

    def on_block_load(self, block_name, size_bytes):
        self.events.append(("block_load", block_name, size_bytes))

    def on_block_evict(self, block_name):
        self.events.append(("block_evict", block_name))

    def on_step(self, step, total, latent, sigma):
        self.events.append(("step", step, total))

    def on_node_start(self, node_id, class_type):
        self.events.append(("node_start", node_id, class_type))

    def on_node_end(self, node_id, elapsed_ms, cache_hit):
        self.events.append(("node_end", node_id, elapsed_ms, cache_hit))

    def on_model_load(self, model_id, size_bytes, elapsed_ms):
        self.events.append(("model_load", model_id))

    def on_model_evict(self, model_id):
        self.events.append(("model_evict", model_id))


class TestInferenceHook:
    def test_should_activate_in_range(self):
        h = ScaleHook("h1", ["attn"], timestep_range=(0.2, 0.8))
        assert h.should_activate(0.5)
        assert h.should_activate(0.2)
        assert h.should_activate(0.8)

    def test_should_activate_out_of_range(self):
        h = ScaleHook("h1", ["attn"], timestep_range=(0.2, 0.8))
        assert not h.should_activate(0.1)
        assert not h.should_activate(0.9)

    def test_matches_layer_contains(self):
        h = ScaleHook("h1", ["attn"])
        assert h.matches_layer("block.0.attn.proj")
        assert not h.matches_layer("block.0.mlp.fc")

    def test_matches_layer_startswith(self):
        h = ScaleHook("h1", ["block.0"])
        assert h.matches_layer("block.0.attn")
        assert not h.matches_layer("block.1.attn")


class TestHookRegistry:
    def test_register_sorted_by_priority(self):
        reg = HookRegistry()
        reg.register_inference(ScaleHook("h2", ["attn"], priority=10))
        reg.register_inference(ScaleHook("h1", ["attn"], priority=1))
        reg.register_inference(ScaleHook("h3", ["attn"], priority=5))
        ids = [h.hook_id for h in reg.inference_hooks]
        assert ids == ["h1", "h3", "h2"]

    def test_get_active_hooks_filters(self):
        reg = HookRegistry()
        reg.register_inference(ScaleHook("h1", ["attn"], timestep_range=(0.0, 0.5)))
        reg.register_inference(ScaleHook("h2", ["mlp"], timestep_range=(0.0, 1.0)))
        reg.register_inference(ScaleHook("h3", ["attn"], timestep_range=(0.5, 1.0)))
        active = reg.get_active_hooks("block.0.attn", sigma=0.3)
        assert len(active) == 1
        assert active[0].hook_id == "h1"

    def test_apply_hooks_priority_order(self):
        reg = HookRegistry()
        reg.register_inference(ScaleHook("h1", ["attn"], strength=2.0, priority=0))
        reg.register_inference(ScaleHook("h2", ["attn"], strength=3.0, priority=1))
        t = torch.ones(1, 4)
        result = reg.apply_hooks("block.attn", t, sigma=0.5)
        # h1 runs first: 1*2=2, then h2: 2*3=6
        assert torch.allclose(result, torch.ones(1, 4) * 6.0)

    def test_unregister_inference(self):
        reg = HookRegistry()
        reg.register_inference(ScaleHook("h1", ["attn"]))
        reg.register_inference(ScaleHook("h2", ["attn"]))
        reg.unregister_inference("h1")
        assert len(reg.inference_hooks) == 1
        assert reg.inference_hooks[0].hook_id == "h2"


class TestRuntimeHook:
    def test_default_methods_noop(self):
        h = RuntimeHook()
        h.on_block_load("b", 100)
        h.on_block_evict("b")
        h.on_step(0, 10, torch.zeros(1), 1.0)
        h.on_node_start("n", "KSampler")
        h.on_node_end("n", 10.0, False)
        h.on_model_load("m", 1000, 5.0)
        h.on_model_evict("m")

    def test_fire_methods(self):
        reg = HookRegistry()
        rec = RecordingRuntimeHook()
        reg.register_runtime(rec)

        reg.fire_block_load("block.0", 1024)
        reg.fire_block_evict("block.0")
        reg.fire_step(0, 10, torch.zeros(1), 1.0)
        reg.fire_node_start("n1", "KSampler")
        reg.fire_node_end("n1", 50.0, True)
        reg.fire_model_load("m1", 5000, 10.0)
        reg.fire_model_evict("m1")

        assert len(rec.events) == 7
        assert rec.events[0] == ("block_load", "block.0", 1024)
        assert rec.events[1] == ("block_evict", "block.0")
        assert rec.events[2] == ("step", 0, 10)
        assert rec.events[3] == ("node_start", "n1", "KSampler")
        assert rec.events[6] == ("model_evict", "m1")

    def test_unregister_runtime(self):
        reg = HookRegistry()
        rec = RecordingRuntimeHook()
        reg.register_runtime(rec)
        reg.unregister_runtime(rec)
        reg.fire_block_load("b", 100)
        assert len(rec.events) == 0
