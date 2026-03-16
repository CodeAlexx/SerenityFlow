"""FastAPI application. ComfyUI-compatible REST + WebSocket server."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

log = logging.getLogger(__name__)


class ServerState:
    """Mutable server state shared across routes, websocket, and execution."""

    def __init__(self):
        self.prompt_queue: asyncio.Queue = asyncio.Queue()
        self.history: dict[str, dict] = {}
        self.executing: str | None = None
        self.last_node_id: str | None = None
        self.ws_connections: dict[str, object] = {}  # sid -> WebSocket
        self.runner = None  # WorkflowRunner
        self.output_dir = "output"
        self.input_dir = "input"
        self.temp_dir = "temp"


state = ServerState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    from serenityflow.executor.cache import CacheStore
    from serenityflow.executor.runner import WorkflowRunner
    from serenityflow.nodes.registry import registry

    # Import nodes to trigger registration
    import serenityflow.nodes  # noqa: F401

    # Initialize Stagehand coordinator for VRAM-managed block-swap
    coordinator = None
    sh_cfg = getattr(state, "stagehand_config", {})
    if not sh_cfg.get("disable", False):
        try:
            from serenityflow.memory.coordinator import StagehandCoordinator
            coordinator = StagehandCoordinator(
                pool_mb=sh_cfg.get("pool_mb"),
                vram_budget_mb=sh_cfg.get("vram_budget_mb"),
                prefetch_window=sh_cfg.get("prefetch_window", 3),
                telemetry=sh_cfg.get("telemetry", False),
                block_threshold_mb=sh_cfg.get("block_threshold_mb", 2048),
            )
        except ImportError:
            log.info("Stagehand not installed, running without block-swap")
        except Exception as e:
            log.warning("Stagehand init failed: %s", e)

    state.runner = WorkflowRunner(
        node_registry=registry,
        cache=CacheStore(),
        coordinator=coordinator,
    )

    for d in [state.output_dir, state.input_dir, state.temp_dir]:
        os.makedirs(d, exist_ok=True)

    # Wire compat PromptServer to real server
    from serenityflow.compat.server import PromptServer
    from serenityflow.server.websocket import send_sync

    PromptServer.instance._server_state = state
    PromptServer.instance._send_fn = send_sync

    # Start execution loop
    asyncio.create_task(_execution_loop())
    log.info("Server startup complete")

    yield  # Server runs

    log.info("Server shutting down")
    if coordinator is not None:
        coordinator.shutdown()


app = FastAPI(title="SerenityFlow v2", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _execution_loop():
    """Process queued prompts one at a time."""
    from serenityflow.server.execution import execute_prompt
    from serenityflow.server.websocket import send_event

    while True:
        item = await state.prompt_queue.get()
        prompt_id = item["prompt_id"]
        state.executing = prompt_id
        print(f"[EXEC] Starting execution for {prompt_id}", flush=True)

        try:
            await execute_prompt(
                state,
                prompt_id,
                item["prompt"],
                item.get("extra_data", {}),
            )
            print(f"[EXEC] Completed execution for {prompt_id}", flush=True)
        except Exception as e:
            print(f"[EXEC] FAILED: {e}", flush=True)
            log.exception("Execution failed for %s", prompt_id)
            await send_event(state, "execution_error", {
                "prompt_id": prompt_id,
                "node_id": "",
                "node_type": "",
                "executed": [],
                "exception_message": "Internal server error",
                "exception_type": "RuntimeError",
                "traceback": [],
                "current_inputs": [],
                "current_outputs": [],
            })
        finally:
            state.executing = None
            from serenityflow.server.websocket import get_queue_info
            await send_event(state, "status", {"status": get_queue_info(state)})


# Register routes and websocket endpoint
from serenityflow.server.routes import register_routes  # noqa: E402
from serenityflow.server.sam_routes import register_sam_routes  # noqa: E402
from serenityflow.server.preprocess_routes import register_preprocess_routes  # noqa: E402
from serenityflow.server.websocket import register_websocket  # noqa: E402

# SAM and preprocess routes MUST register before the catch-all static route in register_routes
register_sam_routes(app)
register_preprocess_routes(app)
register_routes(app)
register_websocket(app)

__all__ = ["app", "state"]
