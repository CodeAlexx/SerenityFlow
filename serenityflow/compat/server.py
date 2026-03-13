"""Compatibility shim for server module.

PromptServer singleton with route registration and WebSocket send.
Created before custom nodes import so they can register routes.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class _RouteTableDef:
    """Minimal aiohttp-like route table for custom node route registration."""

    def __init__(self):
        self._routes: list[tuple[str, str, object]] = []

    def get(self, path):
        def decorator(fn):
            self._routes.append(("GET", path, fn))
            return fn
        return decorator

    def post(self, path):
        def decorator(fn):
            self._routes.append(("POST", path, fn))
            return fn
        return decorator

    def put(self, path):
        def decorator(fn):
            self._routes.append(("PUT", path, fn))
            return fn
        return decorator

    def delete(self, path):
        def decorator(fn):
            self._routes.append(("DELETE", path, fn))
            return fn
        return decorator

    def __iter__(self):
        return iter(self._routes)

    def __len__(self):
        return len(self._routes)


class PromptServer:
    instance = None

    def __init__(self):
        self.routes = _RouteTableDef()
        self.client_id = None
        self.last_node_id = None
        self.last_prompt_id = None
        self._send_queue: list[tuple[str, dict]] = []
        self.supports = {}
        self.loop = None
        self.app = None

    def send_sync(self, event: str, data: dict, sid=None):
        self._send_queue.append((event, data))
        logger.debug("PromptServer send_sync: %s (queued)", event)

    def send_progress_text(self, text: str, node_id=None):
        self.send_sync("progress_text", {"text": text, "node": node_id})

    def add_routes(self, routes):
        for method, path, handler in routes:
            self.routes._routes.append((method, path, handler))

    @classmethod
    def _create_instance(cls):
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance


# Create singleton on import — custom nodes expect this to exist
PromptServer._create_instance()
