from __future__ import annotations

"""
Records per-node execution data: timing, cache hits, model loads, VRAM snapshots.
Queryable after execution completes.
"""
import time
import torch
from dataclasses import dataclass, field

@dataclass
class ModelEvent:
    model_id: str
    event_type: str  # "load" or "evict"
    size_bytes: int
    elapsed_ms: float = 0.0

@dataclass
class NodeRecord:
    node_id: str
    class_type: str
    start_ms: float = 0.0
    end_ms: float = 0.0
    cache_hit: bool = False
    model_events: list[ModelEvent] = field(default_factory=list)
    vram_before: int = 0
    vram_after: int = 0

    @property
    def elapsed_ms(self) -> float:
        return self.end_ms - self.start_ms

class ExecutionTimeline:
    def __init__(self):
        self.records: list[NodeRecord] = []
        self._active: NodeRecord | None = None
        self._start_time: float = 0.0

    def start_execution(self) -> None:
        self.records.clear()
        self._start_time = time.perf_counter()

    def start_node(self, node_id: str, class_type: str) -> None:
        self._active = NodeRecord(
            node_id=node_id,
            class_type=class_type,
            start_ms=(time.perf_counter() - self._start_time) * 1000,
            vram_before=self._get_vram(),
        )

    def end_node(self, cache_hit: bool = False) -> None:
        if self._active is None:
            return
        self._active.end_ms = (time.perf_counter() - self._start_time) * 1000
        self._active.cache_hit = cache_hit
        self._active.vram_after = self._get_vram()
        self.records.append(self._active)
        self._active = None

    def record_model_event(self, model_id: str, event_type: str,
                           size_bytes: int, elapsed_ms: float = 0.0) -> None:
        event = ModelEvent(model_id, event_type, size_bytes, elapsed_ms)
        if self._active is not None:
            self._active.model_events.append(event)

    def to_dict(self) -> dict:
        return {
            "total_ms": (time.perf_counter() - self._start_time) * 1000 if self._start_time else 0,
            "nodes": [
                {
                    "node_id": r.node_id,
                    "class_type": r.class_type,
                    "elapsed_ms": round(r.elapsed_ms, 2),
                    "cache_hit": r.cache_hit,
                    "model_events": [
                        {"model_id": e.model_id, "type": e.event_type,
                         "size_mb": round(e.size_bytes / 1024 / 1024, 1),
                         "elapsed_ms": round(e.elapsed_ms, 2)}
                        for e in r.model_events
                    ],
                    "vram_before_mb": round(r.vram_before / 1024 / 1024, 1),
                    "vram_after_mb": round(r.vram_after / 1024 / 1024, 1),
                }
                for r in self.records
            ],
        }

    def summary(self) -> str:
        total = sum(r.elapsed_ms for r in self.records)
        cached = sum(1 for r in self.records if r.cache_hit)
        executed = len(self.records) - cached
        loads = sum(len(r.model_events) for r in self.records)
        return (f"{len(self.records)} nodes ({executed} executed, {cached} cached), "
                f"{total:.0f}ms total, {loads} model events")

    @staticmethod
    def _get_vram() -> int:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        return 0
