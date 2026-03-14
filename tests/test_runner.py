"""Tests for WorkflowRunner: execution, linking, error handling."""
from __future__ import annotations

import json
import os
import pytest
import torch

from serenityflow.executor.graph import WorkflowGraph
from serenityflow.executor.runner import WorkflowRunner, ExecutionError
from serenityflow.nodes.registry import registry

# Import mock nodes to register them
import serenityflow.nodes.mock  # noqa: F401


FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def load_fixture(name):
    with open(os.path.join(FIXTURES, name)) as f:
        return json.load(f)


class TestBasicExecution:
    def test_flux_t2i_runs_to_completion(self):
        data = load_fixture("basic_flux_t2i.json")
        graph = WorkflowGraph(data)
        runner = WorkflowRunner(registry)
        results = runner.execute(graph)

        # SaveImage (node 9) should be in results
        assert "9" in results
        assert "ui" in results["9"]

    def test_flux_lora_runs_to_completion(self):
        data = load_fixture("flux_lora.json")
        graph = WorkflowGraph(data)
        runner = WorkflowRunner(registry)
        results = runner.execute(graph)

        assert "9" in results
        assert "ui" in results["9"]

    def test_all_nodes_executed(self):
        data = load_fixture("basic_flux_t2i.json")
        graph = WorkflowGraph(data)
        runner = WorkflowRunner(registry)
        runner.execute(graph)

        # All reachable nodes should have outputs
        for node_id in graph.topological_order():
            assert node_id in runner.outputs, f"Node {node_id} was not executed"

    def test_execution_times_recorded(self):
        data = load_fixture("basic_flux_t2i.json")
        graph = WorkflowGraph(data)
        runner = WorkflowRunner(registry)
        runner.execute(graph)

        for node_id in graph.topological_order():
            assert node_id in runner.execution_times
            assert runner.execution_times[node_id] >= 0


class TestLinkResolution:
    def test_scalar_inputs_pass_through(self):
        """Verify seed=42 arrives at KSampler correctly."""
        data = load_fixture("basic_flux_t2i.json")
        graph = WorkflowGraph(data)

        # Patch KSampler to capture its inputs
        captured = {}
        original_fn = registry.get("KSampler").fn

        def capturing_ksampler(**kwargs):
            captured.update(kwargs)
            return original_fn(**kwargs)

        registry.get("KSampler").fn = capturing_ksampler
        try:
            runner = WorkflowRunner(registry)
            runner.execute(graph)

            assert captured["seed"] == 42
            assert captured["steps"] == 20
            assert captured["cfg"] == 3.5
            assert captured["sampler_name"] == "euler"
        finally:
            registry.get("KSampler").fn = original_fn

    def test_link_resolution_correct_types(self):
        """VAEDecode receives KSampler's latent output, not garbage."""
        data = load_fixture("basic_flux_t2i.json")
        graph = WorkflowGraph(data)

        captured_vae = {}
        original_fn = registry.get("VAEDecode").fn

        def capturing_vae_decode(**kwargs):
            captured_vae.update(kwargs)
            return original_fn(**kwargs)

        registry.get("VAEDecode").fn = capturing_vae_decode
        try:
            runner = WorkflowRunner(registry)
            runner.execute(graph)

            # samples should be a dict with "samples" tensor
            assert isinstance(captured_vae["samples"], dict)
            assert "samples" in captured_vae["samples"]
            assert isinstance(captured_vae["samples"]["samples"], torch.Tensor)

            # vae should be a string from mock
            assert isinstance(captured_vae["vae"], str)
            assert "mock_vae" in captured_vae["vae"]
        finally:
            registry.get("VAEDecode").fn = original_fn

    def test_lora_model_propagation(self):
        """LoRA-modified model string propagates to KSampler."""
        data = load_fixture("flux_lora.json")
        graph = WorkflowGraph(data)

        captured = {}
        original_fn = registry.get("KSampler").fn

        def capturing_ksampler(**kwargs):
            captured.update(kwargs)
            return original_fn(**kwargs)

        registry.get("KSampler").fn = capturing_ksampler
        try:
            runner = WorkflowRunner(registry)
            runner.execute(graph)

            # Model should have lora applied
            assert "lora:my_lora.safetensors" in captured["model"]
        finally:
            registry.get("KSampler").fn = original_fn


class TestOutputNodeResults:
    def test_save_image_ui_dict(self):
        data = load_fixture("basic_flux_t2i.json")
        graph = WorkflowGraph(data)
        runner = WorkflowRunner(registry)
        results = runner.execute(graph)

        assert "9" in results
        result = results["9"]
        assert isinstance(result, dict)
        assert "ui" in result
        assert "images" in result["ui"]
        assert "test_mock.png" in result["ui"]["images"]


class TestErrorHandling:
    def test_missing_node_type_error(self):
        graph = WorkflowGraph({
            "1": {"class_type": "NonExistentNode", "inputs": {}},
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}},
        })
        runner = WorkflowRunner(registry)
        with pytest.raises(ExecutionError, match="NonExistentNode"):
            runner.execute(graph)

    def test_node_execution_error_wrapping(self):
        """Errors inside node functions are wrapped in ExecutionError."""
        error_registry = type(registry)()

        @error_registry.register("BrokenNode", return_types=("IMAGE",))
        def broken(**kwargs):
            raise RuntimeError("intentional test error")

        @error_registry.register("SaveImage", is_output=True)
        def save(**kwargs):
            return {"ui": {}}

        graph = WorkflowGraph({
            "1": {"class_type": "BrokenNode", "inputs": {}},
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}},
        })
        runner = WorkflowRunner(error_registry)
        with pytest.raises(ExecutionError, match="intentional test error"):
            runner.execute(graph)

    def test_interrupt_stops_execution(self):
        data = load_fixture("basic_flux_t2i.json")
        graph = WorkflowGraph(data)
        runner = WorkflowRunner(registry)

        executed = []
        def on_progress(node_id, class_type):
            executed.append(node_id)
            if len(executed) >= 3:
                runner.interrupt()

        runner.execute(graph, progress_callback=on_progress)
        # Should have stopped before completing all 9 nodes
        assert len(executed) < 9


class TestReset:
    def test_reset_clears_state(self):
        data = load_fixture("basic_flux_t2i.json")
        graph = WorkflowGraph(data)
        runner = WorkflowRunner(registry)

        runner.execute(graph)
        assert len(runner.outputs) > 0
        assert len(runner.execution_times) > 0

        runner.reset()
        assert len(runner.outputs) == 0
        assert len(runner.execution_times) == 0

    def test_double_execution(self):
        """Can execute same graph twice."""
        data = load_fixture("basic_flux_t2i.json")
        graph = WorkflowGraph(data)
        runner = WorkflowRunner(registry)

        results1 = runner.execute(graph)
        results2 = runner.execute(graph)

        assert "9" in results1
        assert "9" in results2


class TestProgressCallback:
    def test_callback_called_for_each_node(self):
        data = load_fixture("basic_flux_t2i.json")
        graph = WorkflowGraph(data)
        runner = WorkflowRunner(registry)

        progress = []
        def on_progress(node_id, class_type):
            progress.append((node_id, class_type))

        runner.execute(graph, progress_callback=on_progress)
        assert len(progress) == len(graph.topological_order())

    def test_callback_receives_correct_class_types(self):
        data = load_fixture("basic_flux_t2i.json")
        graph = WorkflowGraph(data)
        runner = WorkflowRunner(registry)

        progress = {}
        def on_progress(node_id, class_type):
            progress[node_id] = class_type

        runner.execute(graph, progress_callback=on_progress)
        assert progress["6"] == "KSampler"
        assert progress["9"] == "SaveImage"
