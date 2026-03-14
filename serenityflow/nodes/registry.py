"""Node registration and lookup.

Nodes register with class_type, callable, input/return type metadata.
Registration via decorator or explicit call.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class NodeDef:
    """Metadata about a registered node."""

    def __init__(
        self,
        class_type: str,
        fn,
        input_types: dict | callable | None = None,
        return_types: tuple = (),
        return_names: list[str] | None = None,
        category: str = "uncategorized",
        is_output: bool = False,
        display_name: str | None = None,
    ):
        self.class_type = class_type
        self.fn = fn
        self._input_types = input_types or {"required": {}, "optional": {}}
        self.return_types = return_types
        self.return_names = return_names or [f"output_{i}" for i in range(len(return_types))]
        self.category = category
        self.is_output = is_output
        self.display_name = display_name or class_type

    @property
    def input_types(self) -> dict:
        """Resolve input_types — supports callables for dynamic values."""
        if callable(self._input_types):
            return self._input_types()
        return self._input_types


class NodeRegistry:
    def __init__(self):
        self._nodes: dict[str, NodeDef] = {}

    def register(
        self,
        class_type: str,
        return_types: tuple = (),
        category: str = "uncategorized",
        is_output: bool = False,
        input_types: dict | None = None,
        display_name: str | None = None,
        return_names: list[str] | None = None,
    ):
        """Decorator for node registration."""
        def decorator(fn):
            node_def = NodeDef(
                class_type=class_type,
                fn=fn,
                input_types=input_types,
                return_types=return_types,
                return_names=return_names,
                category=category,
                is_output=is_output,
                display_name=display_name,
            )
            self._nodes[class_type] = node_def
            log.debug("Registered node: %s", class_type)
            return fn
        return decorator

    def get(self, class_type: str) -> NodeDef:
        """Look up node by class_type. Raises KeyError if not found."""
        if class_type not in self._nodes:
            raise KeyError(
                f"Unknown node type: {class_type}. "
                f"Registered: {sorted(self._nodes.keys())}"
            )
        return self._nodes[class_type]

    def get_function(self, class_type: str):
        """Shortcut: return just the callable."""
        return self.get(class_type).fn

    def list_all(self) -> dict[str, NodeDef]:
        """All registered nodes."""
        return dict(self._nodes)

    def has(self, class_type: str) -> bool:
        return class_type in self._nodes


# Global registry instance
registry = NodeRegistry()

__all__ = ["NodeRegistry", "NodeDef", "registry"]
