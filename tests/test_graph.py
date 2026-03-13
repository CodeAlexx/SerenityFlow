"""Tests for WorkflowGraph: parsing, topological sort, cycle detection."""
from __future__ import annotations

import json
import os
import pytest

from serenityflow.executor.graph import WorkflowGraph, NodeSpec


FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def load_fixture(name):
    with open(os.path.join(FIXTURES, name)) as f:
        return json.load(f)


class TestWorkflowGraphParsing:
    def test_basic_flux_parse(self):
        data = load_fixture("basic_flux_t2i.json")
        graph = WorkflowGraph(data)
        assert len(graph.all_node_ids()) == 9
        assert "9" in graph.get_output_nodes()

    def test_node_spec_fields(self):
        data = load_fixture("basic_flux_t2i.json")
        graph = WorkflowGraph(data)
        node = graph.get_node("6")
        assert node.class_type == "KSampler"
        assert node.node_id == "6"
        assert not node.is_output

    def test_output_node_detection(self):
        data = load_fixture("basic_flux_t2i.json")
        graph = WorkflowGraph(data)
        outputs = graph.get_output_nodes()
        assert "9" in outputs
        node = graph.get_node("9")
        assert node.is_output

    def test_missing_class_type_raises(self):
        with pytest.raises(ValueError, match="missing 'class_type'"):
            WorkflowGraph({"1": {"inputs": {}}})

    def test_broken_link_raises(self):
        with pytest.raises(ValueError, match="non-existent node"):
            WorkflowGraph({
                "1": {"class_type": "SaveImage", "inputs": {"images": ["999", 0], "filename_prefix": "x"}},
            })

    def test_no_output_nodes_raises(self):
        with pytest.raises(ValueError, match="No output nodes"):
            WorkflowGraph({
                "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "x", "weight_dtype": "y"}},
            })

    def test_is_link_detection(self):
        assert WorkflowGraph.is_link(["1", 0]) is True
        assert WorkflowGraph.is_link([1, 0]) is True
        assert WorkflowGraph.is_link(42) is False
        assert WorkflowGraph.is_link("hello") is False
        assert WorkflowGraph.is_link([1, 2, 3]) is False
        assert WorkflowGraph.is_link(["1", "0"]) is False


class TestTopologicalSort:
    def test_basic_flux_order(self):
        data = load_fixture("basic_flux_t2i.json")
        graph = WorkflowGraph(data)
        order = graph.topological_order()

        # Loaders must come before their consumers
        assert order.index("1") < order.index("6")   # UNET before KSampler
        assert order.index("2") < order.index("3")   # CLIP before encode
        assert order.index("2") < order.index("4")   # CLIP before encode
        assert order.index("5") < order.index("6")   # EmptyLatent before KSampler
        assert order.index("6") < order.index("8")   # KSampler before VAEDecode
        assert order.index("7") < order.index("8")   # VAE before VAEDecode
        assert order.index("8") < order.index("9")   # VAEDecode before SaveImage

    def test_lora_workflow_order(self):
        data = load_fixture("flux_lora.json")
        graph = WorkflowGraph(data)
        order = graph.topological_order()

        # LoRA loader must come after UNET and CLIP loaders
        assert order.index("1") < order.index("10")
        assert order.index("2") < order.index("10")
        # LoRA output feeds KSampler and CLIP encodes
        assert order.index("10") < order.index("6")
        assert order.index("10") < order.index("3")

    def test_deterministic_order(self):
        """Same graph always produces same order."""
        data = load_fixture("basic_flux_t2i.json")
        graph1 = WorkflowGraph(data)
        graph2 = WorkflowGraph(data)
        assert graph1.topological_order() == graph2.topological_order()

    def test_cycle_detection(self):
        with pytest.raises(ValueError, match="Cycle detected"):
            graph = WorkflowGraph({
                "1": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "a"}},
                "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 0], "text": "b"}},
                "3": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}},
            })
            graph.topological_order()


class TestDependencies:
    def test_get_dependencies(self):
        data = load_fixture("basic_flux_t2i.json")
        graph = WorkflowGraph(data)
        # KSampler depends on UNET, both CLIPEncodes, and EmptyLatent
        deps = graph.get_dependencies("6")
        assert "1" in deps
        assert "3" in deps
        assert "4" in deps
        assert "5" in deps

    def test_get_consumers(self):
        data = load_fixture("basic_flux_t2i.json")
        graph = WorkflowGraph(data)
        # DualCLIPLoader is consumed by both CLIPTextEncodes
        consumers = graph.get_consumers("2")
        assert "3" in consumers
        assert "4" in consumers

    def test_loader_has_no_dependencies(self):
        data = load_fixture("basic_flux_t2i.json")
        graph = WorkflowGraph(data)
        assert graph.get_dependencies("1") == []
        assert graph.get_dependencies("7") == []
