"""Workflow graph parser and topological sort.

Parses ComfyUI API-format JSON into an executable graph.
No ComfyUI imports. No external dependencies beyond stdlib.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class NodeSpec:
    """A single node in the workflow graph."""
    node_id: str
    class_type: str
    inputs: dict          # name -> scalar value or [source_node_id, output_index]
    is_output: bool       # True for terminal nodes (SaveImage, PreviewImage, etc.)


class WorkflowGraph:
    """Parsed workflow graph with topological ordering.

    Accepts ComfyUI API format:
    {
        "node_id": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],   # link to node 4, output 0
                "seed": 42,           # scalar
                ...
            }
        },
        ...
    }
    """

    OUTPUT_NODE_TYPES = {
        "SaveImage", "PreviewImage", "SaveLatent",
    }

    def __init__(self, prompt: dict):
        self._nodes: dict[str, NodeSpec] = {}
        self._consumers: dict[str, set[str]] = {}  # node_id -> set of consumer node_ids

        # Parse all nodes
        for node_id, node_data in prompt.items():
            node_id = str(node_id)
            class_type = node_data.get("class_type")
            if class_type is None:
                raise ValueError(f"Node {node_id} missing 'class_type'")
            inputs = node_data.get("inputs", {})
            is_output = class_type in self.OUTPUT_NODE_TYPES

            self._nodes[node_id] = NodeSpec(
                node_id=node_id,
                class_type=class_type,
                inputs=inputs,
                is_output=is_output,
            )

        # Validate all link targets exist
        for node_id, spec in self._nodes.items():
            for input_name, value in spec.inputs.items():
                if self.is_link(value):
                    source_id = str(value[0])
                    if source_id not in self._nodes:
                        raise ValueError(
                            f"Node {node_id} input '{input_name}' links to "
                            f"non-existent node {source_id}"
                        )

        # Build consumer map
        for node_id, spec in self._nodes.items():
            for input_name, value in spec.inputs.items():
                if self.is_link(value):
                    source_id = str(value[0])
                    self._consumers.setdefault(source_id, set()).add(node_id)

        # Validate at least one output node exists
        output_nodes = self.get_output_nodes()
        if not output_nodes:
            raise ValueError(
                "No output nodes found. Need at least one of: "
                f"{self.OUTPUT_NODE_TYPES}"
            )

        # Validate no orphan nodes (trace from outputs backward)
        reachable = self._trace_reachable_from_outputs()
        orphans = self.all_node_ids() - reachable
        if orphans:
            # Orphans are just warnings, not errors -- ComfyUI allows them
            pass

    def topological_order(self) -> list[str]:
        """Kahn's algorithm. Returns node_ids in execution order.

        Only includes nodes reachable from output nodes.
        Detects cycles. Breaks ties by node_id for determinism.
        """
        reachable = self._trace_reachable_from_outputs()

        # Build in-degree map for reachable nodes only
        in_degree: dict[str, int] = {nid: 0 for nid in reachable}
        dependents: dict[str, list[str]] = {nid: [] for nid in reachable}

        for nid in reachable:
            spec = self._nodes[nid]
            for value in spec.inputs.values():
                if self.is_link(value):
                    source_id = str(value[0])
                    if source_id in reachable:
                        in_degree[nid] += 1
                        dependents[source_id].append(nid)

        # Kahn's algorithm with sorted tie-breaking
        queue: list[str] = sorted(
            [nid for nid, deg in in_degree.items() if deg == 0]
        )
        order: list[str] = []

        while queue:
            node_id = queue.pop(0)
            order.append(node_id)
            for dep in sorted(dependents[node_id]):
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    # Insert sorted
                    queue.append(dep)
            queue.sort()

        if len(order) != len(reachable):
            # Cycle detected
            remaining = reachable - set(order)
            raise ValueError(
                f"Cycle detected in workflow graph involving nodes: {remaining}"
            )

        return order

    def get_node(self, node_id: str) -> NodeSpec:
        return self._nodes[node_id]

    def get_output_nodes(self) -> list[str]:
        """Nodes whose class_type is in OUTPUT_NODE_TYPES."""
        return [
            nid for nid, spec in self._nodes.items()
            if spec.is_output
        ]

    def get_dependencies(self, node_id: str) -> list[str]:
        """Node IDs that this node depends on (via input links)."""
        spec = self._nodes[node_id]
        deps = []
        for value in spec.inputs.values():
            if self.is_link(value):
                source_id = str(value[0])
                if source_id not in deps:
                    deps.append(source_id)
        return deps

    def get_consumers(self, node_id: str) -> list[str]:
        """Node IDs that consume this node's outputs."""
        return list(self._consumers.get(node_id, set()))

    @staticmethod
    def is_link(value) -> bool:
        """Links are [str, int]. Scalars are anything else."""
        return (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and isinstance(value[0], (str, int))
            and isinstance(value[1], int)
        )

    def all_node_ids(self) -> set[str]:
        return set(self._nodes.keys())

    def model_carrying_inputs(self, node_id: str) -> list[tuple[str, str]]:
        """Return [(input_name, source_model_key), ...] for inputs that carry models.

        Used by graph planner. Phase 6: returns empty.
        """
        return []

    def _trace_reachable_from_outputs(self) -> set[str]:
        """BFS backward from output nodes to find all reachable nodes."""
        output_nodes = [
            nid for nid, spec in self._nodes.items()
            if spec.is_output
        ]
        visited: set[str] = set()
        queue = deque(output_nodes)

        while queue:
            nid = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            spec = self._nodes[nid]
            for value in spec.inputs.values():
                if self.is_link(value):
                    source_id = str(value[0])
                    if source_id not in visited and source_id in self._nodes:
                        queue.append(source_id)

        return visited


__all__ = ["WorkflowGraph", "NodeSpec"]
