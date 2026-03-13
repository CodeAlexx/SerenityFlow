"""Compatibility shim for comfy_execution.graph_utils."""
from __future__ import annotations


class GraphBuilder:
    def __init__(self):
        self.nodes = {}
        self._id_counter = 0

    def node(self, class_type, **kwargs):
        self._id_counter += 1
        node_id = str(self._id_counter)
        self.nodes[node_id] = {"class_type": class_type, "inputs": kwargs}
        return node_id

    def finalize(self):
        return self.nodes


def is_link(val):
    return isinstance(val, (list, tuple)) and len(val) == 2
