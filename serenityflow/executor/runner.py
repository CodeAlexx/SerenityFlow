"""Workflow execution engine.

Executes nodes in topological order, resolving links to previous outputs.
Coordinates with Stagehand for memory management.
Integrates ExecutionTimeline, CacheStore, and HookRegistry when provided.
"""
from __future__ import annotations

import logging
import time

import torch

from serenityflow.core.hooks import HookRegistry
from serenityflow.core.timeline import ExecutionTimeline
from serenityflow.executor.cache import CacheStore, compute_signature
from serenityflow.executor.graph import WorkflowGraph

log = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Wraps node execution failures with context."""

    def __init__(self, node_id: str, class_type: str, original_error: Exception):
        self.node_id = node_id
        self.class_type = class_type
        self.original_error = original_error
        super().__init__(f"Node {node_id} ({class_type}): {original_error}")


class WorkflowRunner:
    def __init__(self, node_registry, coordinator=None, timeline=None,
                 cache=None, hooks=None):
        """
        node_registry: NodeRegistry with .get(class_type) -> NodeDef
        coordinator: StagehandCoordinator or None
        timeline: ExecutionTimeline or None (created if not provided)
        cache: CacheStore or None (created if not provided)
        hooks: HookRegistry or None (created if not provided)
        """
        self.registry = node_registry
        self.coordinator = coordinator
        self.timeline = timeline if timeline is not None else ExecutionTimeline()
        self.cache = cache if cache is not None else CacheStore()
        self.hooks = hooks if hooks is not None else HookRegistry()
        self.outputs: dict[str, tuple] = {}       # node_id -> tuple of outputs
        self.execution_times: dict[str, float] = {}  # node_id -> seconds
        self._signatures: dict[str, str] = {}     # node_id -> cache signature
        self._interrupted = False

    def execute(self, graph: WorkflowGraph, progress_callback=None) -> dict:
        """Execute all nodes in topological order.

        Returns dict of output node results.
        """
        self.reset()
        order = graph.topological_order()

        # Set schedule on coordinator if available
        if self.coordinator is not None:
            try:
                self.coordinator.set_schedule(None)  # Phase 6: no graph planner yet
            except Exception:
                pass

        output_results = {}

        for node_id in order:
            if self._interrupted:
                log.warning("Execution interrupted after node %s", node_id)
                break

            spec = graph.get_node(node_id)

            # Timeline + hooks: node starting
            self.timeline.start_node(node_id, spec.class_type)
            self.hooks.fire_node_start(node_id, spec.class_type)

            # Coordinator: prepare for node
            if self.coordinator is not None:
                try:
                    self.coordinator.prepare_for_node(node_id)
                except Exception:
                    log.debug("Coordinator prepare_for_node failed", exc_info=True)

            # Resolve inputs
            try:
                inputs = self._resolve_inputs(spec, graph)
            except Exception as e:
                raise ExecutionError(node_id, spec.class_type, e) from e

            # Compute cache signature from class_type, scalar inputs, and
            # upstream node signatures
            upstream_sigs = {}
            for value in spec.inputs.values():
                if graph.is_link(value):
                    source_id = str(value[0])
                    if source_id in self._signatures:
                        upstream_sigs[source_id] = self._signatures[source_id]

            sig = compute_signature(spec.class_type, spec.inputs, upstream_sigs)
            self._signatures[node_id] = sig

            # Check cache -- hit only if node_id is cached with matching signature
            cached = self.cache.get(node_id)
            cache_hit = cached is not None and cached.signature == sig

            if cache_hit:
                # Restore from cache: prefer ui dict if present, else outputs tuple
                result = cached.ui if cached.ui else cached.outputs
                log.debug("Node %s (%s) cache hit", node_id, spec.class_type)
            else:
                # Execute node
                result = self._execute_node(node_id, spec, inputs)

                # Store in cache (separate ui dict for dict results)
                if isinstance(result, dict):
                    self.cache.set(node_id, (), result, sig)
                else:
                    self.cache.set(node_id, result, {}, sig)

            # Store outputs
            self.outputs[node_id] = result

            # Timeline + hooks: node complete
            node_elapsed_ms = self.execution_times.get(node_id, 0.0) * 1000
            self.timeline.end_node(cache_hit=cache_hit)
            self.hooks.fire_node_end(node_id, node_elapsed_ms, cache_hit)

            # Collect output node results
            if spec.is_output:
                output_results[node_id] = result

            # Coordinator: complete node
            if self.coordinator is not None:
                try:
                    self.coordinator.complete_node(node_id)
                except Exception:
                    log.debug("Coordinator complete_node failed", exc_info=True)

            # Progress callback
            if progress_callback is not None:
                progress_callback(node_id, spec.class_type)

        return output_results

    def interrupt(self):
        """Set interrupt flag. Current node finishes, then execution stops."""
        self._interrupted = True

    def reset(self):
        """Clear outputs and state for next execution."""
        self.outputs.clear()
        self.execution_times.clear()
        self._signatures.clear()
        self._interrupted = False
        self.timeline.start_execution()

    def _resolve_inputs(self, spec, graph: WorkflowGraph) -> dict:
        """Replace link references with actual output values."""
        resolved = {}
        for input_name, value in spec.inputs.items():
            if graph.is_link(value):
                source_id = str(value[0])
                output_index = value[1]

                if source_id not in self.outputs:
                    raise RuntimeError(
                        f"Input '{input_name}' links to node {source_id} "
                        f"which has no outputs (not yet executed or failed)"
                    )

                source_outputs = self.outputs[source_id]

                # Handle dict results (UI responses from output nodes)
                if isinstance(source_outputs, dict):
                    raise RuntimeError(
                        f"Input '{input_name}' links to output node {source_id} "
                        f"which returned a UI dict, not indexable outputs"
                    )

                # Handle tuple/list outputs
                if isinstance(source_outputs, (tuple, list)):
                    if output_index >= len(source_outputs):
                        raise RuntimeError(
                            f"Input '{input_name}' requests output index "
                            f"{output_index} from node {source_id}, but it "
                            f"only has {len(source_outputs)} outputs"
                        )
                    resolved[input_name] = source_outputs[output_index]
                else:
                    # Single value, index must be 0
                    if output_index != 0:
                        raise RuntimeError(
                            f"Input '{input_name}' requests output index "
                            f"{output_index} from node {source_id}, but it "
                            f"returned a single value"
                        )
                    resolved[input_name] = source_outputs
            else:
                # Scalar value -- pass through
                resolved[input_name] = value

        return resolved

    def _execute_node(self, node_id: str, spec, inputs: dict):
        """Execute a single node. Returns normalized tuple of outputs."""
        # Look up node function
        try:
            node_def = self.registry.get(spec.class_type)
        except KeyError as e:
            raise ExecutionError(node_id, spec.class_type, e) from e

        fn = node_def.fn

        # Normalize input names: remap workflow keys to registered parameter
        # names when they differ (e.g. litegraph port "video" → param "images").
        inputs = self._normalize_input_names(node_def, inputs)

        start = time.perf_counter()
        try:
            with torch.inference_mode():
                result = fn(**inputs)
        except Exception as e:
            raise ExecutionError(node_id, spec.class_type, e) from e
        elapsed = time.perf_counter() - start
        self.execution_times[node_id] = elapsed

        log.debug("Node %s (%s) executed in %.3fs", node_id, spec.class_type, elapsed)

        # Normalize result
        if result is None:
            return ()
        if isinstance(result, dict):
            # UI response (SaveImage etc.)
            return result
        if isinstance(result, tuple):
            return result
        # Single value -> wrap in tuple
        return (result,)


    @staticmethod
    def _normalize_input_names(node_def, inputs: dict) -> dict:
        """Remap input keys to match registered parameter names.

        Handles mismatches between litegraph port names and registered
        input_types keys (e.g. port ``video`` → param ``images``).
        Unknown extras that cannot be mapped are dropped instead of being
        forwarded into node functions as unexpected kwargs.
        """
        it = node_def.input_types
        registered: list[str] = []
        for section in ("required", "optional"):
            s = it.get(section, {})
            if isinstance(s, dict):
                registered.extend(s.keys())

        if not registered:
            return inputs

        # Fast path: all input keys are known
        unknown = [k for k in inputs if k not in registered]
        if not unknown:
            return inputs

        normalized = {}
        unknown_set = set(unknown)

        # Prefer explicit aliases for known workflow/UI port names.
        alias_map = {
            "images": "video",
            "video": "images",
            "positive": "conditioning",
        }
        remap = {
            key: alias_map[key]
            for key in unknown
            if key in alias_map and alias_map[key] in registered and alias_map[key] not in inputs
        }

        # Fall back to a conservative positional remap only when it is
        # unambiguous. Broad positional remapping caused workflow extras like
        # `device` or `vae` to be mis-bound into unrelated parameters.
        remaining_unknown = [k for k in unknown if k not in remap and not k.startswith("_widget_")]
        missing = [k for k in registered if k not in inputs and k not in remap.values()]
        if len(remaining_unknown) == 1 and len(missing) == 1:
            remap[remaining_unknown[0]] = missing[0]

        for key, value in inputs.items():
            if key in unknown_set and key in remap:
                normalized[remap[key]] = value
            elif key.startswith("_widget_") or key == "device":
                # Skip UI-only placeholders and legacy device hints.
                continue
            elif key in unknown_set:
                # Drop unknown extras rather than forwarding unexpected kwargs.
                continue
            else:
                normalized[key] = value

        return normalized


__all__ = ["WorkflowRunner", "ExecutionError"]
