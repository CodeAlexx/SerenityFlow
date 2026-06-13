# SerenityFlow v2

Standalone inference/workflow execution engine for diffusion models. NOT ComfyUI — only shares workflow JSON format and REST/WS protocol.

## Project Structure
- `serenityflow/` — main package (19K lines)
- `serenityflow/cli.py` — entry point
- `serenityflow/graph/` — workflow graph execution
- `serenityflow/nodes/` — node implementations
- `serenityflow/server/` — FastAPI REST/WebSocket server
- `serenityflow_mcp/` — MCP server integration
- `tests/` — 30+ test files (pytest)

## Commands
- **Run**: `./run.sh` or `python -m serenityflow.cli --port 8188 -v`
- **Tests**: `python -m pytest tests/ -x -q`

## Hard Rules
- **This is NOT ComfyUI.** Do not reference ComfyUI internals or assume ComfyUI patterns.
- **NEVER remove `serenity-safetensors`** from the venv — it's a Rust crate SerenityFlow depends on.
- **MUST test video quality before AND after any bridge/node changes.** No exceptions.
- **NEVER DOWNLOAD ANYTHING** without explicit user approval.

## Key Facts
- Branch: `master`
- Remote: `https://github.com/CodeAlexx/SerenityFlow.git`
- Python >= 3.10 | Deps: torch, fastapi, uvicorn, pyyaml
- Optional: stagehand, mcp
- 465 tests as of last count
- Uses Stagehand for GPU memory management during inference
