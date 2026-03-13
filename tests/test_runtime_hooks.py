"""Tests for concrete RuntimeHook implementations.

Builder: TimelineHook and StepCounterHook basics
Bug Fixer: edge cases (no active node, multiple hooks)
Skeptic: adversarial scenarios (all cache hits, interleaved events)
"""
from __future__ import annotations

import time
import torch
import pytest

from serenityflow.core.hooks import HookRegistry, RuntimeHook
from serenityflow.core.timeline import ExecutionTimeline
from serenityflow.core.runtime_hooks import TimelineHook, StepCounterHook


# ─── Builder Tests ───


class TestTimelineHookBasics:
    def test_records_node_start_end(self):
        """TimelineHook records node start/end."""
        timeline = ExecutionTimeline()
        timeline.start_execution()
        hook = TimelineHook(timeline)

        hook.on_node_start("node1", "KSampler")
        hook.on_node_end("node1", 100.0, cache_hit=False)

        assert len(timeline.records) == 1
        assert timeline.records[0].node_id == "node1"
        assert timeline.records[0].class_type == "KSampler"
        assert timeline.records[0].cache_hit is False

    def test_records_model_events(self):
        """TimelineHook records model load/evict events."""
        timeline = ExecutionTimeline()
        timeline.start_execution()
        hook = TimelineHook(timeline)

        hook.on_node_start("node1", "KSampler")
        hook.on_model_load("flux_model", 4_000_000_000, 500.0)
        hook.on_model_evict("old_model")
        hook.on_node_end("node1", 600.0, cache_hit=False)

        record = timeline.records[0]
        assert len(record.model_events) == 2
        assert record.model_events[0].model_id == "flux_model"
        assert record.model_events[0].event_type == "load"
        assert record.model_events[0].size_bytes == 4_000_000_000
        assert record.model_events[0].elapsed_ms == 500.0
        assert record.model_events[1].model_id == "old_model"
        assert record.model_events[1].event_type == "evict"

    def test_cache_hit_recorded(self):
        """TimelineHook records cache hit status."""
        timeline = ExecutionTimeline()
        timeline.start_execution()
        hook = TimelineHook(timeline)

        hook.on_node_start("node1", "KSampler")
        hook.on_node_end("node1", 0.1, cache_hit=True)

        assert timeline.records[0].cache_hit is True


class TestStepCounterHook:
    def test_tracks_step_total(self):
        """StepCounterHook tracks step/total."""
        hook = StepCounterHook()
        dummy_latent = torch.randn(1, 4, 64, 64)
        hook.on_step(5, 20, dummy_latent, 0.5)
        assert hook.current_step == 5
        assert hook.total_steps == 20

    def test_fires_callbacks(self):
        """StepCounterHook fires callbacks."""
        hook = StepCounterHook()
        received = []
        hook.add_callback(lambda step, total, sigma: received.append((step, total, sigma)))

        hook.on_step(3, 10, None, 0.7)
        assert received == [(3, 10, 0.7)]

    def test_multiple_callbacks(self):
        """Multiple callbacks all fire."""
        hook = StepCounterHook()
        a, b = [], []
        hook.add_callback(lambda s, t, sig: a.append(s))
        hook.add_callback(lambda s, t, sig: b.append(s))
        hook.on_step(1, 5, None, 0.9)
        assert a == [1]
        assert b == [1]


class TestTimelineSummary:
    def test_summary_reflects_hook_data(self):
        """Timeline summary reflects hook-recorded data."""
        timeline = ExecutionTimeline()
        timeline.start_execution()
        hook = TimelineHook(timeline)

        hook.on_node_start("n1", "LoadCheckpoint")
        hook.on_model_load("model_A", 1_000_000, 100.0)
        hook.on_node_end("n1", 100.0, cache_hit=False)

        hook.on_node_start("n2", "KSampler")
        hook.on_node_end("n2", 50.0, cache_hit=True)

        summary = timeline.summary()
        assert "2 nodes" in summary
        assert "1 executed" in summary
        assert "1 cached" in summary
        assert "1 model events" in summary


# ─── Bug Fixer Tests ───


class TestHookRegistration:
    def test_multiple_hooks_fire_in_order(self):
        """Multiple runtime hooks fire in registration order."""
        registry = HookRegistry()
        order = []

        class HookA(RuntimeHook):
            def on_node_start(self, node_id, class_type):
                order.append("A")

        class HookB(RuntimeHook):
            def on_node_start(self, node_id, class_type):
                order.append("B")

        registry.register_runtime(HookA())
        registry.register_runtime(HookB())
        registry.fire_node_start("n1", "Test")
        assert order == ["A", "B"]

    def test_unregister_hook(self):
        """Unregistered hook doesn't fire."""
        registry = HookRegistry()
        fired = []

        class MyHook(RuntimeHook):
            def on_node_start(self, node_id, class_type):
                fired.append(True)

        hook = MyHook()
        registry.register_runtime(hook)
        registry.fire_node_start("n1", "Test")
        assert len(fired) == 1

        registry.unregister_runtime(hook)
        registry.fire_node_start("n2", "Test")
        assert len(fired) == 1  # Not fired again


class TestModelEventWithoutActiveNode:
    def test_model_event_without_active_node(self):
        """Model event without active node -> silently ignored."""
        timeline = ExecutionTimeline()
        timeline.start_execution()
        hook = TimelineHook(timeline)

        # No start_node called
        hook.on_model_load("orphan_model", 1000, 10.0)
        # Should not crash
        assert len(timeline.records) == 0

    def test_end_node_without_start(self):
        """end_node without start_node -> no crash."""
        timeline = ExecutionTimeline()
        timeline.start_execution()
        hook = TimelineHook(timeline)
        hook.on_node_end("n1", 10.0, cache_hit=False)
        assert len(timeline.records) == 0


class TestStepCounterEdgeCases:
    def test_no_callbacks(self):
        """on_step with no callbacks -> no crash."""
        hook = StepCounterHook()
        hook.on_step(1, 10, None, 0.5)
        assert hook.current_step == 1

    def test_step_updates_overwrite(self):
        """Multiple step calls overwrite current_step."""
        hook = StepCounterHook()
        hook.on_step(1, 10, None, 0.9)
        hook.on_step(5, 10, None, 0.5)
        hook.on_step(10, 10, None, 0.0)
        assert hook.current_step == 10
        assert hook.total_steps == 10


# ─── Skeptic Tests ───


class TestSkepticAdversarial:
    def test_all_cache_hits_timeline(self):
        """Does the timeline still record correctly when every node is a cache hit?"""
        timeline = ExecutionTimeline()
        timeline.start_execution()
        hook = TimelineHook(timeline)

        for i in range(5):
            hook.on_node_start(f"n{i}", f"Type{i}")
            hook.on_node_end(f"n{i}", 0.01, cache_hit=True)

        assert len(timeline.records) == 5
        assert all(r.cache_hit for r in timeline.records)
        summary = timeline.summary()
        assert "5 nodes" in summary
        assert "0 executed" in summary
        assert "5 cached" in summary

    def test_model_events_during_cache_hits(self):
        """Model events during cached nodes are still recorded."""
        timeline = ExecutionTimeline()
        timeline.start_execution()
        hook = TimelineHook(timeline)

        hook.on_node_start("n1", "Loader")
        hook.on_model_load("model_X", 500_000, 50.0)
        hook.on_node_end("n1", 0.5, cache_hit=True)

        assert len(timeline.records[0].model_events) == 1

    def test_interleaved_hooks(self):
        """Timeline and StepCounter hooks interleaved on same registry."""
        timeline = ExecutionTimeline()
        timeline.start_execution()
        registry = HookRegistry()
        t_hook = TimelineHook(timeline)
        s_hook = StepCounterHook()
        steps = []
        s_hook.add_callback(lambda s, t, sig: steps.append(s))

        registry.register_runtime(t_hook)
        registry.register_runtime(s_hook)

        registry.fire_node_start("n1", "KSampler")
        registry.fire_step(1, 20, torch.randn(1, 4, 8, 8), 0.95)
        registry.fire_step(2, 20, torch.randn(1, 4, 8, 8), 0.90)
        registry.fire_node_end("n1", 500.0, cache_hit=False)

        assert len(timeline.records) == 1
        assert s_hook.current_step == 2
        assert steps == [1, 2]

    def test_timeline_to_dict_with_hooks(self):
        """to_dict works correctly with hook-recorded data."""
        timeline = ExecutionTimeline()
        timeline.start_execution()
        hook = TimelineHook(timeline)

        hook.on_node_start("n1", "Loader")
        hook.on_model_load("model_A", 2_000_000_000, 300.0)
        hook.on_node_end("n1", 300.0, cache_hit=False)

        hook.on_node_start("n2", "KSampler")
        hook.on_node_end("n2", 5000.0, cache_hit=False)

        d = timeline.to_dict()
        assert len(d["nodes"]) == 2
        assert d["nodes"][0]["class_type"] == "Loader"
        assert len(d["nodes"][0]["model_events"]) == 1
        assert d["nodes"][0]["model_events"][0]["model_id"] == "model_A"
        assert d["nodes"][0]["model_events"][0]["type"] == "load"

    def test_back_to_back_executions(self):
        """Two complete execution runs through same timeline/hooks."""
        timeline = ExecutionTimeline()
        hook = TimelineHook(timeline)

        # First run
        timeline.start_execution()
        hook.on_node_start("n1", "Loader")
        hook.on_node_end("n1", 100.0, cache_hit=False)
        assert len(timeline.records) == 1

        # Second run: start_execution clears records
        timeline.start_execution()
        assert len(timeline.records) == 0
        hook.on_node_start("n1", "Loader")
        hook.on_node_end("n1", 50.0, cache_hit=True)
        assert len(timeline.records) == 1
        assert timeline.records[0].cache_hit is True

    def test_step_counter_reset_between_nodes(self):
        """StepCounter tracks latest node's steps, not cumulative."""
        hook = StepCounterHook()
        # First sampling node
        hook.on_step(1, 20, None, 0.9)
        hook.on_step(20, 20, None, 0.0)
        assert hook.current_step == 20

        # Second sampling node starts fresh
        hook.on_step(1, 10, None, 0.9)
        assert hook.current_step == 1
        assert hook.total_steps == 10
