"""
For every template where all nodes are registered:
  1. Parse the workflow JSON (litegraph or API format)
  2. Build the node graph and verify links resolve
  3. Verify no dangling references
  4. Verify no cycles in the graph
Does NOT run inference — just validates the graph is structurally sound.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict, deque

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(TESTS_DIR, "template_audit_report.json")
TEMPLATE_DIR = os.path.join(TESTS_DIR, "workflow_templates")


def _parse_litegraph(data: dict) -> dict:
    """Parse litegraph format into node_id -> {class_type, inputs, outputs, links}."""
    nodes_list = data.get("nodes", [])
    links_list = data.get("links", [])

    # Build link table: link_id -> (source_node_id, source_slot, target_node_id, target_slot, type)
    link_table = {}
    for link in links_list:
        # link format: [link_id, source_node_id, source_slot, target_node_id, target_slot, type_str]
        if isinstance(link, list) and len(link) >= 5:
            link_id = link[0]
            link_table[link_id] = {
                "source_node": link[1],
                "source_slot": link[2],
                "target_node": link[3],
                "target_slot": link[4],
                "type": link[5] if len(link) > 5 else "*",
            }

    # Build node table
    parsed = {}
    for node in nodes_list:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", ""))
        class_type = node.get("type", "")
        if not node_id or not class_type:
            continue

        # Collect input links for this node
        input_links = {}
        for inp in node.get("inputs", []):
            if isinstance(inp, dict) and inp.get("link") is not None:
                link_id = inp["link"]
                inp_name = inp.get("name", f"input_{inp.get('slot_index', 0)}")
                if link_id in link_table:
                    source = link_table[link_id]
                    input_links[inp_name] = str(source["source_node"])

        parsed[node_id] = {
            "class_type": class_type,
            "input_links": input_links,
            "output_links": [],
        }

        # Collect output links
        for out in node.get("outputs", []):
            if isinstance(out, dict) and out.get("links"):
                for lk in out["links"]:
                    if lk in link_table:
                        parsed[node_id]["output_links"].append(str(link_table[lk]["target_node"]))

    return parsed


def _parse_api_format(data: dict) -> dict | None:
    """Parse API format {node_id: {class_type, inputs}} into normalized form."""
    nodes = {}
    for key, val in data.items():
        if isinstance(val, dict) and "class_type" in val:
            input_links = {}
            for inp_name, inp_val in val.get("inputs", {}).items():
                if (isinstance(inp_val, (list, tuple))
                        and len(inp_val) == 2
                        and isinstance(inp_val[0], (str, int))
                        and isinstance(inp_val[1], int)):
                    input_links[inp_name] = str(inp_val[0])
            nodes[str(key)] = {
                "class_type": val["class_type"],
                "input_links": input_links,
            }

    return nodes if nodes else None


def parse_template(data: dict) -> dict | None:
    """Parse a template in either format. Returns {node_id: {class_type, input_links}}."""
    # Try litegraph first
    if "nodes" in data and isinstance(data.get("nodes"), list):
        return _parse_litegraph(data)

    # Try API format
    result = _parse_api_format(data)
    if result:
        return result

    # Try wrapped formats
    for wrapper in ("prompt", "output", "workflow"):
        if wrapper in data and isinstance(data[wrapper], dict):
            result = _parse_api_format(data[wrapper])
            if result:
                return result

    return None


def get_runnable_templates() -> list[str]:
    """Return paths to templates where all nodes are registered."""
    if not os.path.exists(REPORT_PATH):
        return []

    with open(REPORT_PATH) as f:
        report = json.load(f)

    runnable = []
    for name, status in report["template_status"].items():
        if isinstance(status, dict) and status.get("runnable"):
            path = os.path.join(TEMPLATE_DIR, name)
            if os.path.exists(path):
                runnable.append(path)

    return runnable


_runnable = get_runnable_templates()


@pytest.mark.parametrize(
    "template_path",
    _runnable,
    ids=[os.path.relpath(p, TEMPLATE_DIR) for p in _runnable],
)
def test_template_parses(template_path):
    """Template parses without error and has at least one node."""
    with open(template_path) as f:
        data = json.load(f)

    nodes = parse_template(data)
    assert nodes is not None, "Could not parse template"
    assert len(nodes) > 0, "Template has no nodes"


@pytest.mark.parametrize(
    "template_path",
    _runnable,
    ids=[os.path.relpath(p, TEMPLATE_DIR) for p in _runnable],
)
def test_template_links_resolve(template_path):
    """All input links point to existing nodes (no dangling references)."""
    with open(template_path) as f:
        data = json.load(f)

    nodes = parse_template(data)
    if nodes is None:
        pytest.skip("Could not parse template")

    for node_id, node_data in nodes.items():
        for inp_name, source_id in node_data["input_links"].items():
            assert source_id in nodes, (
                f"Node {node_id} ({node_data['class_type']}) input "
                f"'{inp_name}' references non-existent node {source_id}"
            )


@pytest.mark.parametrize(
    "template_path",
    _runnable,
    ids=[os.path.relpath(p, TEMPLATE_DIR) for p in _runnable],
)
def test_template_no_cycles(template_path):
    """Graph has no cycles (topological sort succeeds)."""
    with open(template_path) as f:
        data = json.load(f)

    nodes = parse_template(data)
    if nodes is None:
        pytest.skip("Could not parse template")

    # Build in-degree map
    in_degree = {nid: 0 for nid in nodes}
    dependents = defaultdict(list)

    for nid, nd in nodes.items():
        for source_id in nd["input_links"].values():
            if source_id in nodes:
                in_degree[nid] += 1
                dependents[source_id].append(nid)

    # Kahn's algorithm
    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    visited = 0

    while queue:
        node_id = queue.popleft()
        visited += 1
        for dep in dependents[node_id]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    assert visited == len(nodes), (
        f"Cycle detected: {visited} of {len(nodes)} nodes visited. "
        f"Remaining: {[nid for nid, deg in in_degree.items() if deg > 0]}"
    )
