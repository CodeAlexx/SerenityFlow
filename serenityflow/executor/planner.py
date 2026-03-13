"""Graph-aware execution planner.

Analyzes the workflow graph to compute model lifetimes, prefetch points,
and eviction points. Advises the runner on optimal node ordering.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from serenityflow.executor.graph import WorkflowGraph

__all__ = ["ExecutionPlan", "GraphPlanner"]


@dataclass
class ExecutionPlan:
    """Advisory plan. The runner uses this for ordering and Stagehand hints."""
    # node_id -> list of model handle_ids to prefetch before this node
    prefetch: dict[str, list[str]] = field(default_factory=dict)
    # node_id -> list of model handle_ids safe to evict after this node
    evict: dict[str, list[str]] = field(default_factory=dict)
    # model handle_id -> (first_node_id, last_node_id)
    model_spans: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Suggested execution order (topologically valid, locality-optimized)
    suggested_order: list[str] = field(default_factory=list)
    # Whether the plan is still valid (False after dynamic graph mutation)
    valid: bool = True


class GraphPlanner:
    """Build execution plans from workflow graphs and model usage maps."""

    def plan(
        self,
        graph: WorkflowGraph,
        exec_order: list[str],
        node_models: dict[str, list[str]],
    ) -> ExecutionPlan:
        """Build an execution plan.

        Args:
            graph: The workflow graph.
            exec_order: Topological order from WorkflowGraph.
            node_models: node_id -> list of model handle_ids used by that node.

        Returns:
            ExecutionPlan with prefetch/evict hints and locality-optimized order.
        """
        plan = ExecutionPlan()

        if not exec_order or not node_models:
            plan.suggested_order = list(exec_order) if exec_order else []
            return plan

        # 1. Compute model lifetimes
        first_use: dict[str, int] = {}   # model_id -> first node index
        last_use: dict[str, int] = {}    # model_id -> last node index

        for i, node_id in enumerate(exec_order):
            for model_id in node_models.get(node_id, []):
                if model_id not in first_use:
                    first_use[model_id] = i
                last_use[model_id] = i

        for model_id in first_use:
            first_node = exec_order[first_use[model_id]]
            last_node = exec_order[last_use[model_id]]
            plan.model_spans[model_id] = (first_node, last_node)

        # 2. Compute prefetch points (one node before first use, clamped to 0)
        for model_id, first_idx in first_use.items():
            prefetch_idx = max(0, first_idx - 1)
            prefetch_node = exec_order[prefetch_idx]
            plan.prefetch.setdefault(prefetch_node, []).append(model_id)

        # 3. Compute eviction points (at last use node)
        for model_id, last_idx in last_use.items():
            evict_node = exec_order[last_idx]
            plan.evict.setdefault(evict_node, []).append(model_id)

        # 4. Locality-optimized ordering
        plan.suggested_order = self._optimize_order(
            graph, exec_order, node_models
        )

        return plan

    def _optimize_order(
        self,
        graph: WorkflowGraph,
        exec_order: list[str],
        node_models: dict[str, list[str]],
    ) -> list[str]:
        """Reorder within topological levels to minimize model swaps.

        Nodes that share models with their predecessor are preferred.
        Preserves topological validity.
        """
        if len(exec_order) <= 2:
            return list(exec_order)

        # Build topological levels
        order_index = {nid: i for i, nid in enumerate(exec_order)}
        levels: dict[int, list[str]] = {}
        node_level: dict[str, int] = {}

        for node_id in exec_order:
            deps = graph.get_dependencies(node_id)
            if not deps:
                level = 0
            else:
                level = max(
                    (node_level.get(d, 0) for d in deps if d in order_index),
                    default=0,
                ) + 1
            node_level[node_id] = level
            levels.setdefault(level, []).append(node_id)

        # Within each level, sort by model overlap with previous node
        result = []
        prev_models: set[str] = set()

        for level_idx in sorted(levels.keys()):
            level_nodes = levels[level_idx]
            if len(level_nodes) <= 1:
                result.extend(level_nodes)
                if level_nodes:
                    prev_models = set(node_models.get(level_nodes[0], []))
                continue

            # Greedy: pick the node with most model overlap with prev
            remaining = list(level_nodes)
            while remaining:
                best = max(
                    remaining,
                    key=lambda n: len(
                        prev_models & set(node_models.get(n, []))
                    ),
                )
                remaining.remove(best)
                result.append(best)
                prev_models = set(node_models.get(best, []))

        return result

    def invalidate(self, plan: ExecutionPlan) -> None:
        """Called when dynamic graph mutation occurs."""
        plan.valid = False
