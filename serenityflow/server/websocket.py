"""WebSocket handler. ComfyUI protocol exact match."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

log = logging.getLogger(__name__)


def register_websocket(app: FastAPI):
    """Register the /ws endpoint on the FastAPI app."""

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket, clientId: str = ""):
        from serenityflow.server.app import state

        await ws.accept()
        sid = clientId or str(uuid.uuid4())
        state.ws_connections[sid] = ws

        # Send initial status
        await ws.send_json({
            "type": "status",
            "data": {"status": get_queue_info(state), "sid": sid},
        })

        try:
            while True:
                data = await ws.receive_text()
                # Handle client messages (e.g. feature_flags negotiation)
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "feature_flags":
                        pass  # Store client capabilities if needed
                except json.JSONDecodeError:
                    pass
        except WebSocketDisconnect:
            state.ws_connections.pop(sid, None)
        except Exception:
            state.ws_connections.pop(sid, None)


async def send_event(server_state, event_type: str, data: dict, sid: str | None = None):
    """Send JSON event to WebSocket clients."""
    message = {"type": event_type, "data": data}
    dead = []

    if sid and sid in server_state.ws_connections:
        targets = {sid: server_state.ws_connections[sid]}
    else:
        targets = dict(server_state.ws_connections)

    for client_sid, ws in targets.items():
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(client_sid)

    for d in dead:
        server_state.ws_connections.pop(d, None)


def send_sync(event_type: str, data: dict, sid: str | None = None):
    """Synchronous wrapper for send_event (called from executor thread / compat layer).

    Uses run_coroutine_threadsafe to safely send from any thread.
    """
    from serenityflow.server.app import state

    try:
        # Try to get the running loop (works if called from async context)
        loop = asyncio.get_running_loop()
        loop.create_task(send_event(state, event_type, data, sid))
    except RuntimeError:
        # Called from a worker thread — find the main event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    send_event(state, event_type, data, sid), loop
                )
        except RuntimeError:
            pass


async def send_binary(server_state, data: bytes, sid: str | None = None):
    """Send binary preview data to WebSocket clients."""
    dead = []

    if sid and sid in server_state.ws_connections:
        targets = {sid: server_state.ws_connections[sid]}
    else:
        targets = dict(server_state.ws_connections)

    for client_sid, ws in targets.items():
        try:
            await ws.send_bytes(data)
        except Exception:
            dead.append(client_sid)

    for d in dead:
        server_state.ws_connections.pop(d, None)


def get_queue_info(server_state) -> dict:
    """Return queue status in ComfyUI format."""
    return {
        "exec_info": {
            "queue_remaining": server_state.prompt_queue.qsize() + (
                1 if server_state.executing else 0
            ),
        }
    }


__all__ = ["register_websocket", "send_event", "send_sync", "send_binary", "get_queue_info"]
