"""Concrete RuntimeHook implementations.

TimelineHook bridges the RuntimeHook event system to ExecutionTimeline.
StepCounterHook tracks sampling step progress for UI reporting.
"""
from __future__ import annotations

from serenityflow.core.hooks import RuntimeHook
from serenityflow.core.timeline import ExecutionTimeline

__all__ = ["TimelineHook", "StepCounterHook"]


class TimelineHook(RuntimeHook):
    """Records execution timeline as a RuntimeHook."""

    def __init__(self, timeline: ExecutionTimeline):
        self.timeline = timeline

    def on_node_start(self, node_id: str, class_type: str) -> None:
        self.timeline.start_node(node_id, class_type)

    def on_node_end(self, node_id: str, elapsed_ms: float, cache_hit: bool) -> None:
        self.timeline.end_node(cache_hit=cache_hit)

    def on_model_load(self, model_id: str, size_bytes: int, elapsed_ms: float) -> None:
        self.timeline.record_model_event(model_id, "load", size_bytes, elapsed_ms)

    def on_model_evict(self, model_id: str) -> None:
        self.timeline.record_model_event(model_id, "evict", 0)


class StepCounterHook(RuntimeHook):
    """Tracks sampling step progress for UI reporting."""

    def __init__(self):
        self.current_step: int = 0
        self.total_steps: int = 0
        self.callbacks: list = []

    def on_step(self, step: int, total: int, latent, sigma: float) -> None:
        self.current_step = step
        self.total_steps = total
        for cb in self.callbacks:
            cb(step, total, sigma)

    def add_callback(self, cb) -> None:
        self.callbacks.append(cb)
