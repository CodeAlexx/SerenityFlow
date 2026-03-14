"""Bridges server prompt submission to WorkflowRunner execution.

Sends all required WebSocket events in exact ComfyUI order:
  1. execution_start
  2. execution_cached
  3. For each node: executing -> executed (real-time from worker thread)
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


def _send_from_thread(loop, server_state, event_type: str, data: dict):
    """Send a WS event from a worker thread using the main event loop."""
    from serenityflow.server.websocket import send_event
    asyncio.run_coroutine_threadsafe(
        send_event(server_state, event_type, data),
        loop,
    )


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

    # 3. Execute with real-time progress via WS events from worker thread
    executed = set()
    ui_outputs = {}
    loop = asyncio.get_running_loop()

    def progress_callback(node_id: str, class_type: str):
        """Called synchronously from runner after each node completes.
        Sends WS events in real-time from the worker thread."""
        executed.add(node_id)

        # Extract UI output (unwrap the "ui" key for ComfyUI compat)
        result = runner.outputs.get(node_id)
        output = {}
        if isinstance(result, dict):
            if "ui" in result:
                output = result["ui"]
                ui_outputs[node_id] = result["ui"]
            else:
                output = result
                ui_outputs[node_id] = result

        # Send executing event (node is done, mark it)
        _send_from_thread(loop, server_state, "executed", {
            "node": node_id,
            "display_node": node_id,
            "output": output,
            "prompt_id": prompt_id,
        })

        server_state.last_node_id = node_id

    # Use RuntimeHook to send "executing" before each node starts
    from serenityflow.core.hooks import RuntimeHook

    class _ExecutingHook(RuntimeHook):
        def on_node_start(self, node_id: str, class_type: str) -> None:
            _send_from_thread(loop, server_state, "executing", {
                "node": node_id,
                "display_node": node_id,
                "prompt_id": prompt_id,
            })

    ws_hook = _ExecutingHook()
    runner.hooks.register_runtime(ws_hook)

    try:
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
    finally:
        # Remove the WS hook
        runner.hooks.unregister_runtime(ws_hook)

    server_state.last_node_id = None


__all__ = ["execute_prompt"]
