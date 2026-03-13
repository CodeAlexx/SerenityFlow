"""Stagehand graph planner -- advisory model-level scheduling.

Cleaned from v1 -- NO comfy imports. Works with SerenityFlow's WorkflowGraph.

Analyzes the execution graph BEFORE the loop runs. Produces model lifetime
map and per-node prefetch/evict hints. Advisory only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from serenityflow.executor.graph import WorkflowGraph
from serenityflow.executor.types import MODEL_TYPES

logger = logging.getLogger(__name__)


@dataclass
class ModelLifetime:
    """Tracks which nodes first and last use a model."""
    model_key: str          # f"{source_node_id}:{output_index}"
    loader_node_id: str     # Node that produced this model
    model_type: str         # "MODEL", "CLIP", "VAE", etc.
    first_node_id: str      # First node that needs this model
    last_node_id: str       # Last node that needs this model
    estimated_size: int = 0 # Bytes (0 if unknown)


@dataclass
class NodePlan:
    """What should happen when a specific node is about to execute."""
    node_id: str
    prefetch_before: list[str] = field(default_factory=list)
    evict_after: list[str] = field(default_factory=list)


class GraphSchedule:
    """Complete schedule for an execution run."""

    def __init__(
        self,
        node_plans: dict[str, NodePlan],
        model_lifetimes: dict[str, ModelLifetime],
    ):
        self.node_plans = node_plans
        self.model_lifetimes = model_lifetimes
        self.is_valid = True

    def get_plan_for_node(self, node_id: str) -> NodePlan | None:
        if not self.is_valid:
            return None
        return self.node_plans.get(node_id)

    def invalidate(self):
        self.is_valid = False

    def was_invalidated(self) -> bool:
        return not self.is_valid


class StagehandGraphPlanner:
    """Analyze execution graph to produce model-level prefetch/evict schedule."""

    def analyze(
        self,
        graph: WorkflowGraph,
        node_registry=None,
    ) -> GraphSchedule | None:
        """Build model lifetime map from graph.

        Uses WorkflowGraph instead of ComfyUI's DynamicPrompt.
        node_registry: optional NodeRegistry for INPUT_TYPES metadata.
        """
        all_node_ids = list(graph.all_node_ids())
        if not all_node_ids:
            return None

        # Discover which models each node uses
        node_models: dict[str, list[tuple[str, str]]] = {}
        for node_id in all_node_ids:
            model_inputs = self._get_model_inputs(node_id, graph, node_registry)
            for input_name, type_str in model_inputs:
                model_key = self._trace_model_source(node_id, input_name, graph)
                if model_key is not None:
                    node_models.setdefault(node_id, []).append((model_key, type_str))

        if not node_models:
            return None

        # Build ModelLifetime for each unique model_key
        model_users: dict[str, list[str]] = {}
        model_type_map: dict[str, str] = {}
        for node_id, models in node_models.items():
            for model_key, type_str in models:
                model_users.setdefault(model_key, []).append(node_id)
                model_type_map[model_key] = type_str

        # Use topological order for ranking
        topo_order = graph.topological_order()
        topo_rank = {nid: i for i, nid in enumerate(topo_order)}

        lifetimes: dict[str, ModelLifetime] = {}
        for model_key, users in model_users.items():
            sorted_users = sorted(users, key=lambda nid: (topo_rank.get(nid, 0), nid))
            first_node = sorted_users[0]
            last_node = sorted_users[-1]
            source_node_id = model_key.split(":")[0]
            lifetimes[model_key] = ModelLifetime(
                model_key=model_key,
                loader_node_id=source_node_id,
                model_type=model_type_map[model_key],
                first_node_id=first_node,
                last_node_id=last_node,
            )

        # Build per-node plans
        plans: dict[str, NodePlan] = {}
        for model_key, lt in lifetimes.items():
            if lt.first_node_id not in plans:
                plans[lt.first_node_id] = NodePlan(node_id=lt.first_node_id)
            plans[lt.first_node_id].prefetch_before.append(model_key)

            if lt.last_node_id not in plans:
                plans[lt.last_node_id] = NodePlan(node_id=lt.last_node_id)
            plans[lt.last_node_id].evict_after.append(model_key)

        schedule = GraphSchedule(node_plans=plans, model_lifetimes=lifetimes)

        logger.info(
            "Graph planner: %d models tracked, %d nodes with plans",
            len(lifetimes), len(plans),
        )
        return schedule

    def _trace_model_source(
        self, node_id: str, input_name: str, graph: WorkflowGraph
    ) -> str | None:
        """Follow the link for input_name to its immediate source."""
        spec = graph.get_node(node_id)
        value = spec.inputs.get(input_name)
        if value is None or not graph.is_link(value):
            return None
        source_node_id, output_index = str(value[0]), value[1]
        return f"{source_node_id}:{output_index}"

    def _get_model_inputs(
        self,
        node_id: str,
        graph: WorkflowGraph,
        node_registry=None,
    ) -> list[tuple[str, str]]:
        """Return [(input_name, type_string), ...] for model-carrying inputs."""
        if node_registry is None:
            return []

        spec = graph.get_node(node_id)
        if not node_registry.has(spec.class_type):
            return []

        node_def = node_registry.get(spec.class_type)
        results = []
        for category in ("required", "optional"):
            category_inputs = node_def.input_types.get(category, {})
            for input_name, input_spec in category_inputs.items():
                if not isinstance(input_spec, (tuple, list)) or len(input_spec) == 0:
                    continue
                type_str = input_spec[0]
                if isinstance(type_str, str) and type_str in MODEL_TYPES:
                    results.append((input_name, type_str))

        return results


__all__ = ["StagehandGraphPlanner", "GraphSchedule", "ModelLifetime", "NodePlan"]
