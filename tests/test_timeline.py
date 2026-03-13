from __future__ import annotations
import time
import json
import torch
import pytest
from serenityflow.core.timeline import ExecutionTimeline, NodeRecord, ModelEvent


class TestExecutionTimeline:
    def test_start_end_node_records_timing(self):
        tl = ExecutionTimeline()
        tl.start_execution()
        tl.start_node("n1", "KSampler")
        time.sleep(0.01)
        tl.end_node()
        assert len(tl.records) == 1
        assert tl.records[0].node_id == "n1"
        assert tl.records[0].class_type == "KSampler"

    def test_elapsed_ms_positive(self):
        tl = ExecutionTimeline()
        tl.start_execution()
        tl.start_node("n1", "KSampler")
        time.sleep(0.005)
        tl.end_node()
        assert tl.records[0].elapsed_ms > 0

    def test_cache_hit_recorded(self):
        tl = ExecutionTimeline()
        tl.start_execution()
        tl.start_node("n1", "KSampler")
        tl.end_node(cache_hit=True)
        assert tl.records[0].cache_hit is True

    def test_model_events_within_node(self):
        tl = ExecutionTimeline()
        tl.start_execution()
        tl.start_node("n1", "LoadModel")
        tl.record_model_event("flux_dev", "load", 24_000_000_000, elapsed_ms=500.0)
        tl.end_node()
        assert len(tl.records[0].model_events) == 1
        assert tl.records[0].model_events[0].model_id == "flux_dev"

    def test_to_dict_serializable(self):
        tl = ExecutionTimeline()
        tl.start_execution()
        tl.start_node("n1", "KSampler")
        tl.record_model_event("m1", "load", 1_000_000, 10.0)
        tl.end_node()
        d = tl.to_dict()
        json.dumps(d)  # Must not raise
        assert "nodes" in d
        assert d["nodes"][0]["node_id"] == "n1"
        assert len(d["nodes"][0]["model_events"]) == 1

    def test_summary_readable(self):
        tl = ExecutionTimeline()
        tl.start_execution()
        tl.start_node("n1", "A")
        tl.end_node()
        tl.start_node("n2", "B")
        tl.end_node(cache_hit=True)
        s = tl.summary()
        assert "2 nodes" in s
        assert "1 executed" in s
        assert "1 cached" in s

    def test_multiple_nodes_ordered(self):
        tl = ExecutionTimeline()
        tl.start_execution()
        for i in range(5):
            tl.start_node(f"n{i}", "Type")
            tl.end_node()
        assert [r.node_id for r in tl.records] == [f"n{i}" for i in range(5)]

    def test_vram_snapshot(self):
        tl = ExecutionTimeline()
        tl.start_execution()
        tl.start_node("n1", "A")
        tl.end_node()
        # On CPU-only machines, VRAM is 0
        assert tl.records[0].vram_before >= 0
        assert tl.records[0].vram_after >= 0

    def test_end_node_without_start_is_noop(self):
        tl = ExecutionTimeline()
        tl.start_execution()
        tl.end_node()  # Should not crash
        assert len(tl.records) == 0

    def test_model_event_without_active_node(self):
        tl = ExecutionTimeline()
        tl.start_execution()
        tl.record_model_event("m1", "load", 1000)  # No active node, should not crash
