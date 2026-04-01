"""SerenityFlow MCP debug server — stdio transport entry point."""
from __future__ import annotations

import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server

from .config import Config
from .tools import register_tools

__all__ = ["main"]


def main() -> None:
    config = Config()
    server = Server("serenityflow-debug")
    register_tools(server, config)

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
