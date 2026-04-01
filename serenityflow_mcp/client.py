"""HTTP client for SerenityFlow's /debug API."""
from __future__ import annotations

from typing import Any

import httpx

__all__ = ["SerenityFlowClient"]


class SerenityFlowClient:
    """Async HTTP client targeting SerenityFlow's ``/debug/*`` endpoints."""

    def __init__(self, base_url: str = "http://localhost:8188"):
        self.base_url = base_url
        self._client = httpx.AsyncClient(base_url=base_url, timeout=120.0)

    async def get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        r = await self._client.get(f"/debug{path}", params=params)
        r.raise_for_status()
        return r.json()

    async def post(self, path: str, json: dict[str, Any] | None = None) -> dict:
        r = await self._client.post(f"/debug{path}", json=json)
        r.raise_for_status()
        return r.json()

    async def close(self) -> None:
        await self._client.aclose()
