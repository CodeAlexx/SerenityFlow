"""SerenityFlow v2 entry point."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def _setup_compat_path():
    """Insert compat/ into sys.path so `import comfy.*` resolves to our shims."""
    compat_dir = os.path.join(os.path.dirname(__file__), "compat")
    if compat_dir not in sys.path:
        sys.path.insert(0, compat_dir)


# Wire compat layer before anything else can import comfy.*
_setup_compat_path()


def main():
    parser = argparse.ArgumentParser(description="SerenityFlow inference server")
    parser.add_argument("--workflow", type=str, help="Run a workflow JSON file directly")
    parser.add_argument("--listen", default="127.0.0.1", help="Server listen address")
    parser.add_argument("--port", type=int, default=8188, help="Server port")
    parser.add_argument("--model-dir", type=str, default=".", help="Base directory for models")
    parser.add_argument("--extra-model-paths", type=str, help="Path to extra_model_paths.yaml")

    # Stagehand args
    parser.add_argument("--stagehand-disable", action="store_true")
    parser.add_argument("--stagehand-pool-mb", type=int, default=0, help="0 = auto")
    parser.add_argument("--stagehand-vram-budget", type=int, default=0, help="0 = auto")
    parser.add_argument("--stagehand-prefetch", type=int, default=3)
    parser.add_argument("--stagehand-block-threshold", type=int, default=2048)
    parser.add_argument("--stagehand-telemetry", action="store_true")

    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.workflow:
        run_workflow(args)
    else:
        from serenityflow.server.__main__ import main as server_main
        sys.argv = [sys.argv[0],
                    "--host", args.listen,
                    "--port", str(args.port)]
        if args.model_dir and args.model_dir != ".":
            sys.argv += ["--model-dir", args.model_dir]
        if args.verbose:
            sys.argv.append("--verbose")
        if args.stagehand_disable:
            sys.argv.append("--stagehand-disable")
        if args.stagehand_pool_mb:
            sys.argv += ["--stagehand-pool-mb", str(args.stagehand_pool_mb)]
        if args.stagehand_vram_budget:
            sys.argv += ["--stagehand-vram-budget", str(args.stagehand_vram_budget)]
        if args.stagehand_prefetch != 3:
            sys.argv += ["--stagehand-prefetch", str(args.stagehand_prefetch)]
        if args.stagehand_block_threshold != 2048:
            sys.argv += ["--stagehand-block-threshold", str(args.stagehand_block_threshold)]
        if args.stagehand_telemetry:
            sys.argv.append("--stagehand-telemetry")
        server_main()


def run_workflow(args):
    """Load and execute a workflow JSON file."""
    from serenityflow.bridge.workflow import parse_workflow
    from serenityflow.executor.graph import WorkflowGraph
    from serenityflow.executor.runner import WorkflowRunner
    from serenityflow.nodes.registry import registry

    # Import real nodes to register them
    import serenityflow.nodes  # noqa: F401

    # Set up model paths
    from serenityflow.bridge.model_paths import get_model_paths
    get_model_paths(args.model_dir)

    log = logging.getLogger("serenityflow")

    # Parse workflow
    with open(args.workflow) as f:
        data = json.load(f)
    prompt = parse_workflow(data, registry=registry)

    # Build graph
    graph = WorkflowGraph(prompt, registry=registry)
    log.info("Graph: %d nodes, %d output nodes", len(graph.all_node_ids()), len(graph.get_output_nodes()))

    # Create Stagehand coordinator for VRAM-managed block-swap
    coordinator = None
    if not args.stagehand_disable:
        try:
            from serenityflow.memory.coordinator import StagehandCoordinator
            coordinator = StagehandCoordinator(
                pool_mb=args.stagehand_pool_mb or None,
                vram_budget_mb=args.stagehand_vram_budget or None,
                prefetch_window=args.stagehand_prefetch,
                telemetry=args.stagehand_telemetry,
                block_threshold_mb=args.stagehand_block_threshold,
            )
        except ImportError:
            log.info("Stagehand not installed, running without block-swap")
        except Exception as e:
            log.warning("Stagehand init failed: %s", e)

    # Create runner
    runner = WorkflowRunner(registry, coordinator=coordinator)

    # Execute
    def on_progress(node_id, class_type):
        log.info("Executed: %s (%s)", node_id, class_type)

    results = runner.execute(graph, progress_callback=on_progress)

    log.info("Execution complete.")
    for node_id, result in results.items():
        log.info("  Output %s: %s", node_id, result)

    # Print timing summary
    total = sum(runner.execution_times.values())
    log.info("Total execution time: %.3fs", total)
    for node_id in graph.topological_order():
        if node_id in runner.execution_times:
            spec = graph.get_node(node_id)
            log.info("  %s (%s): %.3fs", node_id, spec.class_type, runner.execution_times[node_id])

    # Shutdown coordinator
    if coordinator is not None:
        coordinator.shutdown()


if __name__ == "__main__":
    main()
