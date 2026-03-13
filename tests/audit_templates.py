"""
Audit workflow templates against SerenityFlow v2 node registry.
Produces:
  1. Total unique node types across all templates
  2. Which nodes v2 has
  3. Which nodes are missing
  4. Per-template status (runnable / missing N nodes)
  5. Missing nodes ranked by how many templates need them
"""
from __future__ import annotations

import json
import glob
import os
import sys
from collections import Counter, defaultdict


def extract_node_types(workflow) -> set[str]:
    """Extract class_type from workflow JSON. Handles both API and litegraph formats."""
    types = set()

    if not isinstance(workflow, dict):
        return types

    # API format: {node_id: {class_type: "...", inputs: {...}}}
    for key, val in workflow.items():
        if isinstance(val, dict) and "class_type" in val:
            types.add(val["class_type"])

    # Litegraph format: {"nodes": [...], "links": [...]}
    if "nodes" in workflow and isinstance(workflow["nodes"], list):
        for node in workflow["nodes"]:
            if isinstance(node, dict):
                node_type = node.get("type") or node.get("class_type")
                if node_type:
                    types.add(node_type)

    # Some templates wrap in {"prompt": {...}} or {"workflow": {...}}
    for wrapper_key in ("prompt", "workflow", "output"):
        if wrapper_key in workflow and isinstance(workflow[wrapper_key], dict):
            types |= extract_node_types(workflow[wrapper_key])

    return types


def load_registry() -> set[str]:
    """Load v2's registered node names."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    try:
        # Import triggers all @registry.register decorators
        from serenityflow.nodes.registry import registry
        import serenityflow.nodes  # noqa: F401 — triggers imports
        return set(registry.list_all().keys())
    except Exception as exc:
        print(f"WARNING: Could not import registry ({exc}), falling back to regex scan")
        import re
        registered = set()
        nodes_dir = os.path.join(os.path.dirname(__file__), "..", "serenityflow", "nodes")
        for f in glob.glob(os.path.join(nodes_dir, "**", "*.py"), recursive=True):
            with open(f) as fh:
                content = fh.read()
                # Match @registry.register("NodeName", ...) or registry.register(class_type="NodeName", ...)
                for match in re.finditer(
                    r'registry\.register\(\s*["\']([^"\']+)["\']', content
                ):
                    registered.add(match.group(1))
                for match in re.finditer(
                    r'registry\.register\(\s*class_type\s*=\s*["\']([^"\']+)["\']',
                    content,
                ):
                    registered.add(match.group(1))
        return registered


def audit():
    template_dir = os.path.join(os.path.dirname(__file__), "workflow_templates")

    if not os.path.exists(template_dir):
        print(
            f"ERROR: {template_dir} not found. Run:\n"
            f"  git clone --depth 1 https://github.com/Comfy-Org/workflow_templates {template_dir}"
        )
        sys.exit(1)

    registered = load_registry()
    print(f"Registry loaded: {len(registered)} node types\n")

    # Collect data
    all_types: set[str] = set()
    type_usage: Counter = Counter()
    template_status: dict = {}
    type_to_templates: dict[str, list[str]] = defaultdict(list)
    skipped = 0

    templates = glob.glob(os.path.join(template_dir, "**", "*.json"), recursive=True)

    for template_path in sorted(templates):
        try:
            with open(template_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            template_status[template_path] = {"error": str(e)}
            continue

        types = extract_node_types(data)
        if not types:
            skipped += 1
            continue

        rel_path = os.path.relpath(template_path, template_dir)
        all_types |= types

        have = types & registered
        missing = types - registered

        for t in types:
            type_usage[t] += 1
            type_to_templates[t].append(rel_path)

        template_status[rel_path] = {
            "total": len(types),
            "have": len(have),
            "missing": len(missing),
            "missing_list": sorted(missing),
            "runnable": len(missing) == 0,
        }

    # Sort missing by usage count (most needed first)
    missing_all = all_types - registered
    missing_ranked = sorted(missing_all, key=lambda t: -type_usage[t])

    # === Print Report ===
    print("=" * 70)
    print("SERENITYFLOW v2 — WORKFLOW TEMPLATE AUDIT")
    print("=" * 70)
    print()
    print(f"Templates scanned:    {len(template_status)}")
    print(f"Skipped (no nodes):   {skipped}")
    print(f"Unique node types:    {len(all_types)}")
    print(f"v2 has:               {len(all_types & registered)}")
    print(f"v2 missing:           {len(missing_all)}")
    print()

    runnable = sum(
        1 for s in template_status.values() if isinstance(s, dict) and s.get("runnable")
    )
    print(f"Templates runnable:   {runnable}/{len(template_status)}")
    print()

    # Missing nodes ranked by demand
    if missing_ranked:
        print("-" * 70)
        print("MISSING NODES (ranked by template usage)")
        print("-" * 70)
        for node_type in missing_ranked:
            count = type_usage[node_type]
            examples = type_to_templates[node_type][:3]
            suffix = (
                f" (+{len(type_to_templates[node_type]) - 3} more)"
                if len(type_to_templates[node_type]) > 3
                else ""
            )
            print(
                f"  {node_type:50s} used in {count:3d} templates  "
                f"[{', '.join(examples)}{suffix}]"
            )
        print()

    # Per-template status
    print("-" * 70)
    print("PER-TEMPLATE STATUS")
    print("-" * 70)
    for template, status in sorted(template_status.items()):
        if isinstance(status, dict) and "error" in status:
            print(f"  ERROR  {template}: {status['error']}")
            continue
        if not isinstance(status, dict):
            continue
        icon = "+" if status["runnable"] else "x"
        line = f"  {icon}  {template:55s}  {status['have']}/{status['total']} nodes"
        if status["missing_list"]:
            names = ", ".join(status["missing_list"][:5])
            extra = (
                f" (+{len(status['missing_list']) - 5} more)"
                if len(status["missing_list"]) > 5
                else ""
            )
            line += f"  missing: {names}{extra}"
        print(line)
    print()

    # Nodes v2 has that no template uses
    unused = registered - all_types
    if unused:
        print("-" * 70)
        print(f"v2 NODES NOT USED BY ANY TEMPLATE ({len(unused)})")
        print("-" * 70)
        for name in sorted(unused):
            print(f"  {name}")
        print()

    # === Write machine-readable output ===
    report = {
        "summary": {
            "templates_scanned": len(template_status),
            "unique_node_types": len(all_types),
            "v2_has": len(all_types & registered),
            "v2_missing": len(missing_all),
            "templates_runnable": runnable,
        },
        "missing_ranked": [
            {
                "node_type": t,
                "usage_count": type_usage[t],
                "templates": type_to_templates[t],
            }
            for t in missing_ranked
        ],
        "template_status": template_status,
        "unused_v2_nodes": sorted(unused) if unused else [],
    }

    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "template_audit_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Machine-readable report: {report_path}")


if __name__ == "__main__":
    audit()
