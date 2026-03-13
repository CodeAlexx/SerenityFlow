"""Bridges server prompt submission to WorkflowRunner execution.

Sends all required WebSocket events in exact ComfyUI order:
  1. execution_start
  2. execution_cached
  3. For each node: executing -> executed
  4. execution_success OR execution_error OR execution_interrupted
"""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from serenityflow.executor.graph import WorkflowGraph
from serenityflow.executor.runner import ExecutionError

log = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sf-exec")


async def execute_prompt(server_state, prompt_id: str, prompt: dict, extra_data: dict):
    """Execute a workflow prompt with full WebSocket event streaming."""
    from serenityflow.server.websocket import send_event

    runner = server_state.runner
    runner.interrupt()  # Clear any previous interrupt
    runner._interrupted = False

    # Reset comfy interrupt flag
    try:
        import comfy.model_management as mm
        mm.interrupt_current_processing(False)
    except Exception:
        pass

    # 1. execution_start
    await send_event(server_state, "execution_start", {"prompt_id": prompt_id})

    # Build graph
    try:
        print(f"[EXEC] Building graph from {len(prompt)} nodes", flush=True)
        graph = WorkflowGraph(prompt, registry=runner.registry)
        print(f"[EXEC] Graph built: {graph.topological_order()}", flush=True)
    except Exception as e:
        print(f"[EXEC] Graph build FAILED: {e}", flush=True)
        await send_event(server_state, "execution_error", {
            "prompt_id": prompt_id,
            "node_id": "",
            "node_type": "",
            "executed": [],
            "exception_message": str(e),
            "exception_type": type(e).__name__,
            "traceback": [],
            "current_inputs": [],
            "current_outputs": [],
        })
        return

    # 2. Find cached nodes
    cached_nodes = []
    order = graph.topological_order()
    for node_id in order:
        cached = runner.cache.get(node_id)
        if cached is not None:
            cached_nodes.append(node_id)

    await send_event(server_state, "execution_cached", {
        "nodes": cached_nodes,
        "prompt_id": prompt_id,
    })

    # 3. Execute with progress callback sending WS events
    executed = set()
    ui_outputs = {}
    loop = asyncio.get_running_loop()

    def progress_callback(node_id: str, class_type: str):
        """Called synchronously from runner after each node completes."""
        executed.add(node_id)
        result = runner.outputs.get(node_id)
        if isinstance(result, dict):
            ui_outputs[node_id] = result

    try:
        # Send executing events via a hook on the runner
        # We wrap execute in a thread since it's blocking
        def run_in_thread():
            print(f"[EXEC-THREAD] Starting runner.execute()", flush=True)
            try:
                r = runner.execute(graph, progress_callback=progress_callback)
                print(f"[EXEC-THREAD] runner.execute() returned: {type(r)}", flush=True)
                return r
            except Exception as e:
                print(f"[EXEC-THREAD] runner.execute() FAILED: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raise

        results = await loop.run_in_executor(_executor, run_in_thread)

        # Send executed events for each node after completion
        for node_id in order:
            if node_id in executed:
                spec = graph.get_node(node_id)
                output = ui_outputs.get(node_id, {})
                await send_event(server_state, "executing", {
                    "node": node_id,
                    "display_node": node_id,
                    "prompt_id": prompt_id,
                })
                await send_event(server_state, "executed", {
                    "node": node_id,
                    "display_node": node_id,
                    "output": output,
                    "prompt_id": prompt_id,
                })

        # Signal end of execution (executing null = done)
        await send_event(server_state, "executing", {
            "node": None,
            "display_node": None,
            "prompt_id": prompt_id,
        })

        # Check if interrupted
        if runner._interrupted:
            await send_event(server_state, "execution_interrupted", {
                "prompt_id": prompt_id,
                "node_id": server_state.last_node_id or "",
                "node_type": "",
                "executed": list(executed),
            })
        else:
            # 4. execution_success
            await send_event(server_state, "execution_success", {
                "prompt_id": prompt_id,
            })

        # Store in history
        server_state.history[prompt_id] = {
            "prompt": prompt,
            "outputs": ui_outputs,
            "timestamp": time.time(),
        }

        # Store timeline if available
        if runner.timeline:
            server_state.history[prompt_id]["timeline"] = runner.timeline.to_dict()

    except ExecutionError as e:
        await send_event(server_state, "execution_error", {
            "prompt_id": prompt_id,
            "node_id": e.node_id,
            "node_type": e.class_type,
            "executed": list(executed),
            "exception_message": str(e.original_error),
            "exception_type": type(e.original_error).__name__,
            "traceback": [],
            "current_inputs": [],
            "current_outputs": list(ui_outputs.keys()),
        })
    except Exception as e:
        await send_event(server_state, "execution_error", {
            "prompt_id": prompt_id,
            "node_id": server_state.last_node_id or "",
            "node_type": "",
            "executed": list(executed),
            "exception_message": str(e),
            "exception_type": type(e).__name__,
            "traceback": [],
            "current_inputs": [],
            "current_outputs": list(ui_outputs.keys()),
        })

    server_state.last_node_id = None


__all__ = ["execute_prompt"]
