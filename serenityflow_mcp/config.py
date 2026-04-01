"""Configuration for the SerenityFlow MCP debug server."""
from __future__ import annotations

import os

__all__ = ["Config"]


class Config:
    """Connection settings for the SerenityFlow backend."""

    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or os.environ.get(
            "SF_API_URL", "http://localhost:8188"
        )
