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
    parser.add_argument("--listen", action="store_true", help="Listen on 0.0.0.0")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.listen:
        args.host = "0.0.0.0"

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
        try:
            import folder_paths
            folder_paths.base_path = args.model_dir
            folder_paths.models_dir = os.path.join(args.model_dir, "models")
        except ImportError:
            pass

    # Import app (triggers route registration)
    from serenityflow.server.app import app, state
    state.output_dir = args.output_dir
    state.input_dir = args.input_dir

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.input_dir, exist_ok=True)

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
