"""MCP tool definitions and handlers for SerenityFlow debug introspection."""
from __future__ import annotations

import json
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from .client import SerenityFlowClient
from .config import Config

__all__ = ["register_tools"]


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_TOOLS: list[Tool] = [
    Tool(
        name="pipeline_status",
        description=(
            "Get current pipeline state: loaded model, components, "
            "active LoRAs, dtypes, devices."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="lora_check",
        description=(
            "Dry-run LoRA compatibility check against the currently loaded model. "
            "Reports key matches, misses, shape mismatches, dtype info, and LoRA "
            "metadata. Does NOT apply the LoRA."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "lora_path": {
                    "type": "string",
                    "description": "Path to LoRA safetensors file",
                },
                "verbose": {"type": "boolean", "default": True},
            },
            "required": ["lora_path"],
        },
    ),
    Tool(
        name="tensor_probe",
        description=(
            "Inspect a named tensor in the loaded pipeline. Returns shape, dtype, "
            "stats (mean/std/min/max/nan_count), optional histogram."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "tensor_path": {
                    "type": "string",
                    "description": (
                        "Dotted path to tensor, e.g. "
                        "'transformer.single_blocks.0.attn.to_q.weight'"
                    ),
                },
                "component": {
                    "type": "string",
                    "description": (
                        "Pipeline component: transformer, text_encoder, vae"
                    ),
                },
                "histogram_bins": {"type": "integer", "default": 0},
            },
            "required": ["tensor_path"],
        },
    ),
    Tool(
        name="vram_status",
        description=(
            "GPU VRAM state: allocated, reserved, free, Stagehand budget/pool, "
            "system RAM."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="debug_generate",
        description=(
            "Run a test generation with full diagnostics: timing per phase, "
            "VRAM trace, latent stats, text embedding norms, warnings/errors. "
            "Use low res + low steps for fast iteration."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "width": {"type": "integer", "default": 512},
                "height": {"type": "integer", "default": 512},
                "steps": {"type": "integer", "default": 4},
                "guidance_scale": {"type": "number", "default": 3.5},
                "seed": {"type": "integer", "default": 42},
                "lora_path": {"type": "string"},
                "trace_level": {
                    "type": "string",
                    "enum": ["minimal", "full"],
                    "default": "full",
                },
            },
            "required": ["prompt"],
        },
    ),
    Tool(
        name="engine_logs",
        description=(
            "Retrieve last N lines of engine logs. Filter by level "
            "(DEBUG/INFO/WARNING/ERROR) and component name."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "lines": {"type": "integer", "default": 100},
                "level": {"type": "string", "default": "INFO"},
                "component": {"type": "string"},
            },
        },
    ),
    Tool(
        name="list_models",
        description=(
            "List all available models and LoRAs in the Serenity model directory "
            "with architecture, size, quant format, and loaded status."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="load_model",
        description="Load a model into the inference pipeline.",
        inputSchema={
            "type": "object",
            "properties": {
                "model_path": {"type": "string"},
                "pipeline_type": {"type": "string"},
                "quant": {"type": "string"},
                "keep_fp8": {"type": "boolean", "default": True},
            },
            "required": ["model_path", "pipeline_type"],
        },
    ),
    Tool(
        name="unload_model",
        description="Unload model and free VRAM.",
        inputSchema={
            "type": "object",
            "properties": {
                "component": {
                    "type": "string",
                    "default": "all",
                    "description": "Which component to unload, or 'all'",
                },
            },
        },
    ),
    Tool(
        name="architecture_diff",
        description=(
            "Compare a LoRA's architecture against a model. Answers: "
            "'was this LoRA trained for this model?' Reports key prefix "
            "patterns, compatibility, and suggests correct model."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "lora_path": {"type": "string"},
                "model_path": {
                    "type": "string",
                    "description": (
                        "Path to model safetensors. "
                        "If omitted, uses currently loaded model."
                    ),
                },
            },
            "required": ["lora_path"],
        },
    ),
    Tool(
        name="config_dump",
        description=(
            "Dump current active configuration: generation defaults, "
            "Stagehand config, engine config."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="pipeline_diff",
        description=(
            "Save pipeline config snapshots and compare them. Use save_snapshot "
            "to capture current state, then diff two snapshots to see what changed "
            "(model, dtype, LoRAs, Stagehand config)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "snapshot_a": {
                    "type": "string",
                    "description": "First snapshot name, or omit for current state",
                },
                "snapshot_b": {
                    "type": "string",
                    "description": "Second snapshot name, or omit for current state",
                },
                "save_snapshot": {
                    "type": "string",
                    "description": "Save current pipeline state under this name",
                },
            },
        },
    ),
    Tool(
        name="training_metrics",
        description=(
            "Query training metrics from an active or recent Serenity training run. "
            "Reads loss, gradient norms, learning rate, speed, and VRAM from "
            "SerenityBoard's training logs."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "log_dir": {
                    "type": "string",
                    "description": "Path to training logs directory containing board.db files",
                },
                "run_name": {
                    "type": "string",
                    "description": "Specific run name. Auto-detects latest if omitted.",
                },
                "last_n_steps": {"type": "integer", "default": 50},
                "tags": {
                    "type": "string",
                    "description": "Comma-separated scalar tags to query",
                },
            },
            "required": ["log_dir"],
        },
    ),
    Tool(
        name="debug_ab_compare",
        description=(
            "A/B comparison: run same prompt with and without LoRA using "
            "identical seed. Returns latent stats, timing, and comparison "
            "metrics (MSE, cosine similarity)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "width": {"type": "integer", "default": 512},
                "height": {"type": "integer", "default": 512},
                "steps": {"type": "integer", "default": 4},
                "guidance_scale": {"type": "number", "default": 3.5},
                "seed": {"type": "integer", "default": 42},
                "lora_path": {
                    "type": "string",
                    "description": "Path to LoRA safetensors file",
                },
                "lora_strength": {"type": "number", "default": 1.0},
            },
            "required": ["prompt", "lora_path"],
        },
    ),
    Tool(
        name="debug_generate_breakpoint",
        description=(
            "Run generation with breakpoint at step N. Returns per-step "
            "latent stats, timing, and a resume token to continue from "
            "the breakpoint."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "width": {"type": "integer", "default": 512},
                "height": {"type": "integer", "default": 512},
                "steps": {"type": "integer", "default": 4},
                "guidance_scale": {"type": "number", "default": 3.5},
                "seed": {"type": "integer", "default": 42},
                "break_at_step": {
                    "type": "integer",
                    "default": 2,
                    "description": "Pause after this step (1-indexed)",
                },
                "resume_token": {
                    "type": "string",
                    "description": "Token from previous breakpoint to resume",
                },
            },
            "required": ["prompt"],
        },
    ),
    Tool(
        name="diagnose",
        description=(
            "Automatic diagnostic meta-tool. Chains pipeline, VRAM, LoRA, "
            "weight health, logs, and training checks. Modes: 'full' "
            "(everything), 'lora' (compatibility), 'performance' "
            "(speed/VRAM), 'health' (weight integrity)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["full", "lora", "performance", "health"],
                    "default": "full",
                },
                "lora_path": {
                    "type": "string",
                    "description": "LoRA file path (required for 'lora' mode)",
                },
                "tensor_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific tensors to probe. Auto-selects if omitted.",
                },
                "training_log_dir": {
                    "type": "string",
                    "description": "Training logs directory for training metrics.",
                },
            },
        },
    ),
]


# ---------------------------------------------------------------------------
# Routing table: tool name -> (HTTP method, debug API path, body source)
# ---------------------------------------------------------------------------

_ROUTES: dict[str, tuple[str, str]] = {
    "pipeline_status": ("GET", "/pipeline/status"),
    "lora_check": ("POST", "/lora/check"),
    "tensor_probe": ("POST", "/tensor/probe"),
    "vram_status": ("GET", "/vram/status"),
    "debug_generate": ("POST", "/generate"),
    "engine_logs": ("GET", "/logs"),
    "list_models": ("GET", "/models/available"),
    "load_model": ("POST", "/model/load"),
    "unload_model": ("POST", "/model/unload"),
    "architecture_diff": ("POST", "/architecture/diff"),
    "config_dump": ("GET", "/config"),
    "pipeline_diff": ("POST", "/pipeline/diff"),
    "training_metrics": ("GET", "/training/metrics"),
    "debug_ab_compare": ("POST", "/generate/ab_compare"),
    "debug_generate_breakpoint": ("POST", "/generate/breakpoint"),
    "diagnose": ("POST", "/diagnose"),
}


def register_tools(server: Server, config: Config) -> None:
    """Register all debug tools on the MCP server."""
    client = SerenityFlowClient(config.base_url)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return list(_TOOLS)

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name not in _ROUTES:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

        method, path = _ROUTES[name]

        try:
            if method == "GET":
                # For GET requests, pass arguments as query params
                params = arguments if arguments else None
                result = await client.get(path, params=params)
            else:
                result = await client.post(path, json=arguments)
        except Exception as exc:
            result = {"error": str(exc), "tool": name}

        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
