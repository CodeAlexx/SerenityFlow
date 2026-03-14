"""Start SerenityFlow server.

Usage: python -m serenityflow.server [--host 127.0.0.1] [--port 8188] [--model-dir ~/models]
"""
from __future__ import annotations

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="SerenityFlow v2 Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--input-dir", default="input")
    parser.add_argument("--custom-node-dir", default=None)
    parser.add_argument("--frontend-dir", default=None, help="Path to ComfyUI frontend")
    parser.add_argument("--serenity-path", default=None,
                        help="Path to Serenity trainer repo (for inference engine)")
    parser.add_argument("--listen", action="store_true", help="Listen on 0.0.0.0")
    parser.add_argument("--verbose", "-v", action="store_true")

    # Stagehand args
    parser.add_argument("--stagehand-disable", action="store_true",
                        help="Disable Stagehand block-swap memory management")
    parser.add_argument("--stagehand-vram-budget", type=int, default=0,
                        help="VRAM budget in MB (0 = auto, 90%% of GPU)")
    parser.add_argument("--stagehand-pool-mb", type=int, default=0,
                        help="Pinned memory pool size in MB (0 = auto)")
    parser.add_argument("--stagehand-prefetch", type=int, default=3,
                        help="Prefetch window (blocks)")
    parser.add_argument("--stagehand-block-threshold", type=int, default=2048,
                        help="Minimum model size (MB) to activate block-swap")
    parser.add_argument("--stagehand-telemetry", action="store_true",
                        help="Enable Stagehand telemetry logging")

    args = parser.parse_args()

    if args.listen:
        args.host = "0.0.0.0"

    # Add Serenity trainer to sys.path for inference engine access
    serenity_path = args.serenity_path
    if serenity_path is None:
        # Auto-detect: check common locations relative to this repo
        candidates = [
            os.path.expanduser("~/serenity"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "..", "serenity"),
        ]
        for cand in candidates:
            if os.path.isfile(os.path.join(cand, "serenity", "inference", "__init__.py")):
                serenity_path = cand
                break
    if serenity_path:
        serenity_path = os.path.realpath(serenity_path)
        if serenity_path not in sys.path:
            sys.path.insert(0, serenity_path)

    # Setup logging
    import logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Setup compat path
    from serenityflow.cli import _setup_compat_path
    _setup_compat_path()

    # Configure model paths
    if args.model_dir:
        model_dir = os.path.expanduser(args.model_dir)
        model_dir = os.path.realpath(model_dir)

        # Update compat folder_paths
        try:
            import folder_paths
            folder_paths.set_base_path(model_dir)
        except ImportError:
            pass

        # Update bridge model_paths
        from serenityflow.bridge.model_paths import ModelPaths, set_model_paths
        set_model_paths(ModelPaths(model_dir))

    # Set frontend dir if specified
    if args.frontend_dir:
        os.environ["SERENITYFLOW_FRONTEND_DIR"] = args.frontend_dir

    # Import app (triggers route registration)
    from serenityflow.server.app import app, state
    state.output_dir = os.path.realpath(args.output_dir)
    state.input_dir = os.path.realpath(args.input_dir)

    # Pass Stagehand config to app state for coordinator init
    state.stagehand_config = {
        "disable": args.stagehand_disable,
        "pool_mb": args.stagehand_pool_mb or None,
        "vram_budget_mb": args.stagehand_vram_budget or None,
        "prefetch_window": args.stagehand_prefetch,
        "block_threshold_mb": args.stagehand_block_threshold,
        "telemetry": args.stagehand_telemetry,
    }

    os.makedirs(state.output_dir, exist_ok=True)
    os.makedirs(state.input_dir, exist_ok=True)

    # Sync folder_paths output/input dirs with server state
    try:
        import folder_paths
        folder_paths.set_output_directory(state.output_dir)
        folder_paths.set_input_directory(state.input_dir)
    except ImportError:
        pass

    print(f"\nSerenityFlow v2 server starting")
    print(f"  Host: {args.host}:{args.port}")
    print(f"  Models: {args.model_dir or 'default'}")
    print(f"  Output: {args.output_dir}")
    if args.custom_node_dir:
        print(f"  Custom nodes: {args.custom_node_dir}")
    print()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
