"""Tests for graph-aware execution planner.

Builder: core plan generation (spans, prefetch, evict)
Bug Fixer: edge cases (empty graph, single node, negative index)
Skeptic: adversarial scenarios (overlapping models, plan invalidation)
"""
from __future__ import annotations

import pytest

from serenityflow.executor.graph import WorkflowGraph
from serenityflow.executor.planner import ExecutionPlan, GraphPlanner


def _make_graph(prompt: dict) -> WorkflowGraph:
    """Build a WorkflowGraph from a ComfyUI-format dict."""
    return WorkflowGraph(prompt)


def _linear_graph() -> tuple[WorkflowGraph, list[str]]:
    """A -> B -> C -> SaveImage. Models: A uses M1, B uses M1+M2, C uses M2."""
    prompt = {
        "1": {"class_type": "LoadCheckpoint", "inputs": {}},
        "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
        "3": {"class_type": "VAEDecode", "inputs": {"samples": ["2", 0]}},
        "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
    }
    graph = _make_graph(prompt)
    order = graph.topological_order()
    return graph, order


# ─── Builder Tests ───


class TestPlannerBasics:
    def test_single_model_three_nodes(self):
        """Single model used by 3 nodes -> span covers all 3."""
        graph, order = _linear_graph()
        node_models = {
            order[1]: ["model_A"],
            order[2]: ["model_A"],
            order[3]: ["model_A"],
        }
        planner = GraphPlanner()
        plan = planner.plan(graph, order, node_models)

        assert "model_A" in plan.model_spans
        first, last = plan.model_spans["model_A"]
        assert first == order[1]
        assert last == order[3]

    def test_two_models_non_overlapping(self):
        """Two models, non-overlapping -> evict first before second's first use."""
        graph, order = _linear_graph()
        # M1 used by node 1, M2 used by node 3
        node_models = {
            order[0]: ["M1"],
            order[2]: ["M2"],
        }
        planner = GraphPlanner()
        plan = planner.plan(graph, order, node_models)

        # M1 span is (order[0], order[0])
        assert plan.model_spans["M1"] == (order[0], order[0])
        # M2 span is (order[2], order[2])
        assert plan.model_spans["M2"] == (order[2], order[2])

        # M1 evicts at order[0]
        assert "M1" in plan.evict.get(order[0], [])
        # M2 evicts at order[2]
        assert "M2" in plan.evict.get(order[2], [])

    def test_two_models_overlapping(self):
        """Two models, overlapping -> both resident during overlap."""
        graph, order = _linear_graph()
        # M1 used by nodes 0-2, M2 used by nodes 1-3
        node_models = {
            order[0]: ["M1"],
            order[1]: ["M1", "M2"],
            order[2]: ["M1", "M2"],
            order[3]: ["M2"],
        }
        planner = GraphPlanner()
        plan = planner.plan(graph, order, node_models)

        assert plan.model_spans["M1"] == (order[0], order[2])
        assert plan.model_spans["M2"] == (order[1], order[3])

        # M1 evicts at order[2], M2 evicts at order[3]
        assert "M1" in plan.evict.get(order[2], [])
        assert "M2" in plan.evict.get(order[3], [])

    def test_prefetch_one_node_before(self):
        """Prefetch happens one node before first use."""
        graph, order = _linear_graph()
        # Model used at node index 2
        node_models = {order[2]: ["M1"]}
        planner = GraphPlanner()
        plan = planner.plan(graph, order, node_models)

        # Prefetch at index 1
        assert "M1" in plan.prefetch.get(order[1], [])

    def test_eviction_at_last_use(self):
        """Eviction hint at the last-use node, not after."""
        graph, order = _linear_graph()
        node_models = {
            order[1]: ["M1"],
            order[2]: ["M1"],
        }
        planner = GraphPlanner()
        plan = planner.plan(graph, order, node_models)

        assert "M1" in plan.evict.get(order[2], [])
        # Not at order[3]
        assert "M1" not in plan.evict.get(order[3], [])

    def test_suggested_order_valid(self):
        """Suggested order contains all nodes."""
        graph, order = _linear_graph()
        node_models = {order[0]: ["M1"]}
        planner = GraphPlanner()
        plan = planner.plan(graph, order, node_models)
        assert set(plan.suggested_order) == set(order)


# ─── Bug Fixer Tests ───


class TestPlannerEdgeCases:
    def test_empty_graph(self):
        """Empty exec_order -> empty plan."""
        prompt = {
            "1": {"class_type": "SaveImage", "inputs": {}},
        }
        graph = _make_graph(prompt)
        planner = GraphPlanner()
        plan = planner.plan(graph, [], {})
        assert plan.model_spans == {}
        assert plan.prefetch == {}
        assert plan.evict == {}
        assert plan.suggested_order == []
        assert plan.valid is True

    def test_no_models(self):
        """Nodes exist but no models -> empty plan."""
        graph, order = _linear_graph()
        planner = GraphPlanner()
        plan = planner.plan(graph, order, {})
        assert plan.model_spans == {}
        assert plan.suggested_order == list(order)

    def test_prefetch_index_doesnt_go_negative(self):
        """Model used at first node -> prefetch at same node (index 0)."""
        graph, order = _linear_graph()
        node_models = {order[0]: ["M1"]}
        planner = GraphPlanner()
        plan = planner.plan(graph, order, node_models)

        # Prefetch at max(0, 0-1) = 0
        assert "M1" in plan.prefetch.get(order[0], [])

    def test_single_node_uses_model(self):
        """Model used by exactly one node -> span first==last."""
        graph, order = _linear_graph()
        node_models = {order[2]: ["M1"]}
        planner = GraphPlanner()
        plan = planner.plan(graph, order, node_models)
        assert plan.model_spans["M1"][0] == plan.model_spans["M1"][1] == order[2]

    def test_plan_starts_valid(self):
        """Plans start valid."""
        graph, order = _linear_graph()
        planner = GraphPlanner()
        plan = planner.plan(graph, order, {})
        assert plan.valid is True


# ─── Skeptic Tests ───


class TestPlannerAdversarial:
    def test_invalidation(self):
        """Plan invalidation after ephemeral node."""
        graph, order = _linear_graph()
        node_models = {order[0]: ["M1"]}
        planner = GraphPlanner()
        plan = planner.plan(graph, order, node_models)
        assert plan.valid is True

        planner.invalidate(plan)
        assert plan.valid is False

    def test_model_used_by_all_nodes(self):
        """Model used by every node -> span is entire graph."""
        graph, order = _linear_graph()
        node_models = {nid: ["M1"] for nid in order}
        planner = GraphPlanner()
        plan = planner.plan(graph, order, node_models)
        assert plan.model_spans["M1"] == (order[0], order[-1])
        # Prefetch at first node (max(0, 0-1) = 0)
        assert "M1" in plan.prefetch.get(order[0], [])
        # Evict at last node
        assert "M1" in plan.evict.get(order[-1], [])

    def test_many_models_same_node(self):
        """Node uses 5 models -> all prefetched/evicted correctly."""
        graph, order = _linear_graph()
        models = [f"M{i}" for i in range(5)]
        node_models = {order[2]: models}
        planner = GraphPlanner()
        plan = planner.plan(graph, order, node_models)

        for m in models:
            assert m in plan.model_spans
            # All prefetch at order[1], all evict at order[2]
            assert m in plan.prefetch.get(order[1], [])
            assert m in plan.evict.get(order[2], [])

    def test_replan_after_invalidation(self):
        """Can create a new plan after invalidating old one."""
        graph, order = _linear_graph()
        node_models = {order[0]: ["M1"]}
        planner = GraphPlanner()
        plan1 = planner.plan(graph, order, node_models)
        planner.invalidate(plan1)

        plan2 = planner.plan(graph, order, node_models)
        assert plan2.valid is True
        assert plan1.valid is False

    def test_locality_optimization_preserves_all_nodes(self):
        """Locality optimization doesn't drop any nodes."""
        prompt = {
            "1": {"class_type": "LoadA", "inputs": {}},
            "2": {"class_type": "LoadB", "inputs": {}},
            "3": {"class_type": "Process", "inputs": {"a": ["1", 0], "b": ["2", 0]}},
            "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
        }
        graph = _make_graph(prompt)
        order = graph.topological_order()
        node_models = {
            order[0]: ["M1"],
            order[1]: ["M2"],
            order[2]: ["M1", "M2"],
        }
        planner = GraphPlanner()
        plan = planner.plan(graph, order, node_models)
        assert set(plan.suggested_order) == set(order)
