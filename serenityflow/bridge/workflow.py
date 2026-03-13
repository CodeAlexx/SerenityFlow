"""Parse ComfyUI workflow JSON in both API and litegraph formats.

API format: {"node_id": {"class_type": ..., "inputs": {...}}}
Litegraph format: {"nodes": [...], "links": [...], "groups": [...]}

Auto-detects format and converts litegraph to API format.
"""
from __future__ import annotations


def parse_api_format(data: dict) -> dict:
    """Validate and return API format prompt.

    Validation:
    - Each node has "class_type" and "inputs"
    - All link targets [node_id, index] reference existing nodes
    """
    prompt = {}
    for node_id, node_data in data.items():
        node_id = str(node_id)
        if not isinstance(node_data, dict):
            raise ValueError(f"Node {node_id}: expected dict, got {type(node_data).__name__}")
        if "class_type" not in node_data:
            raise ValueError(f"Node {node_id}: missing 'class_type'")
        # Copy to avoid mutating caller's dict; ensure "inputs" key exists
        entry = dict(node_data)
        if "inputs" not in entry:
            entry["inputs"] = {}
        else:
            # Normalize link source IDs to strings for consistency
            normalized = {}
            for input_name, value in entry["inputs"].items():
                if _is_link(value):
                    normalized[input_name] = [str(value[0]), value[1]]
                else:
                    normalized[input_name] = value
            entry["inputs"] = normalized
        prompt[node_id] = entry

    # Validate link targets
    node_ids = set(prompt.keys())
    for node_id, node_data in prompt.items():
        for input_name, value in node_data.get("inputs", {}).items():
            if _is_link(value):
                target_id = str(value[0])
                if target_id not in node_ids:
                    raise ValueError(
                        f"Node {node_id} input '{input_name}' links to "
                        f"non-existent node {target_id}"
                    )

    return prompt


def parse_litegraph_format(data: dict) -> dict:
    """Convert litegraph format to API format.

    Litegraph nodes have id, type, inputs (with link ids), outputs, widgets_values.
    Litegraph links are [link_id, origin_id, origin_slot, target_id, target_slot, type_string].
    """
    nodes = data.get("nodes", [])
    links = data.get("links", [])

    if not nodes:
        raise ValueError("Litegraph format: no nodes found")

    # Build node type lookup for reroute detection
    node_types: dict[str, str] = {}
    # Build per-node input link lookup: node_id -> [(input_slot, link_id), ...]
    node_input_links: dict[str, list[tuple[int, int]]] = {}
    for node in nodes:
        nid = str(node.get("id", ""))
        ntype = node.get("type", "")
        if nid and ntype:
            node_types[nid] = ntype
        # Collect input links for reroute tracing
        for slot_idx, inp in enumerate(node.get("inputs", []) or []):
            link_id = inp.get("link")
            if link_id is not None:
                node_input_links.setdefault(nid, []).append((slot_idx, link_id))

    # Build link lookup: link_id -> (origin_node_id, origin_slot)
    link_map: dict[int, tuple[str, int]] = {}
    for link in links:
        if len(link) >= 4:
            link_id = link[0]
            origin_id = str(link[1])
            origin_slot = link[2]
            link_map[link_id] = (origin_id, origin_slot)

    # Set of node IDs to skip (Reroute, Note, etc.)
    skip_types = {"Note"}
    reroute_ids: set[str] = set()
    for nid, ntype in node_types.items():
        if ntype.startswith("Reroute"):
            reroute_ids.add(nid)

    def _resolve_through_reroutes(origin_id: str, origin_slot: int,
                                  visited: set[str] | None = None) -> tuple[str, int]:
        """Trace through Reroute nodes to find the real origin."""
        if visited is None:
            visited = set()
        if origin_id not in reroute_ids:
            return (origin_id, origin_slot)
        if origin_id in visited:
            # Cycle in reroutes -- return as-is to avoid infinite loop
            return (origin_id, origin_slot)
        visited.add(origin_id)
        # Find what feeds into this reroute node (slot 0)
        for slot_idx, link_id in node_input_links.get(origin_id, []):
            if slot_idx == 0 and link_id in link_map:
                upstream_id, upstream_slot = link_map[link_id]
                return _resolve_through_reroutes(upstream_id, upstream_slot, visited)
        # Reroute has no input -- return as-is (will fail validation downstream)
        return (origin_id, origin_slot)

    prompt = {}
    for node in nodes:
        node_id = str(node.get("id", ""))
        class_type = node.get("type", "")

        if not node_id or not class_type:
            continue

        # Skip special litegraph nodes
        if class_type in skip_types or class_type.startswith("Reroute"):
            continue

        inputs_dict = {}

        # Process linked inputs
        node_inputs = node.get("inputs", []) or []
        for inp in node_inputs:
            input_name = inp.get("name", "")
            link_id = inp.get("link")
            if link_id is not None and link_id in link_map:
                origin_id, origin_slot = link_map[link_id]
                # Trace through any Reroute nodes to find the real source
                origin_id, origin_slot = _resolve_through_reroutes(origin_id, origin_slot)
                inputs_dict[input_name] = [origin_id, origin_slot]

        # Process widget values (scalars)
        # Widget values are positional -- map them to input names by position.
        # We skip inputs that are already linked.
        widgets = node.get("widgets_values", []) or []
        if widgets:
            # Get names of non-linked inputs from the node's input spec
            # Since we don't have INPUT_TYPES metadata in Phase 6,
            # we use a heuristic: widget values fill non-linked input names
            # that aren't already in inputs_dict.
            # For Phase 6 this is best-effort.
            # In Phase 7, we'll use actual INPUT_TYPES from the registry
            for i, val in enumerate(widgets):
                # Skip values that look like link references to avoid
                # false positives in WorkflowGraph.is_link()
                if _is_link(val):
                    continue
                # Store with a generated name if we can't determine the real one
                name = f"_widget_{i}"
                inputs_dict.setdefault(name, val)

        prompt[node_id] = {
            "class_type": class_type,
            "inputs": inputs_dict,
        }

    return prompt


def parse_workflow(data: dict) -> dict:
    """Auto-detect format and parse.

    Litegraph has "nodes" key with a list value.
    API format has string keys with dict values containing "class_type".
    """
    if "nodes" in data and isinstance(data["nodes"], list):
        return parse_litegraph_format(data)
    else:
        return parse_api_format(data)


def _is_link(value) -> bool:
    """Check if a value is a link reference."""
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


__all__ = ["parse_workflow", "parse_api_format", "parse_litegraph_format"]
