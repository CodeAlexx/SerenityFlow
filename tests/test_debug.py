"""Tests for the debug introspection subsystem.

Tests log_buffer, debug router endpoints, architecture detection, and MCP tools.
All tests mock pipeline state — no GPU or model loading required.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Setup compat path before importing server
from serenityflow.cli import _setup_compat_path
_setup_compat_path()

from fastapi.testclient import TestClient
from serenityflow.server.app import app, state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _setup_dirs(tmp_path):
    """Use temp dirs for output/input/temp."""
    state.output_dir = str(tmp_path / "output")
    state.input_dir = str(tmp_path / "input")
    state.temp_dir = str(tmp_path / "temp")
    os.makedirs(state.output_dir, exist_ok=True)
    os.makedirs(state.input_dir, exist_ok=True)
    os.makedirs(state.temp_dir, exist_ok=True)
    state.history.clear()
    # Reset runner for clean state
    state.runner = None


@pytest.fixture
def client():
    return TestClient(app)


# Fake CachedOutput to match executor/cache.py
@dataclass
class FakeCachedOutput:
    outputs: tuple
    ui: dict
    signature: str


class FakeCache:
    """Minimal cache mock matching CacheStore interface."""
    def __init__(self):
        self.cache: dict[str, FakeCachedOutput] = {}

    def set_outputs(self, node_id: str, outputs: tuple):
        self.cache[node_id] = FakeCachedOutput(outputs=outputs, ui={}, signature="test")


class FakeRunner:
    """Minimal runner mock."""
    def __init__(self, cache=None):
        self.cache = cache or FakeCache()


# ---------------------------------------------------------------------------
# log_buffer tests
# ---------------------------------------------------------------------------


class TestLogBuffer:
    def test_install_creates_singleton(self):
        from serenityflow.debug.log_buffer import install, get_handler, _handler
        import serenityflow.debug.log_buffer as lb

        # Reset singleton
        lb._handler = None

        handler = install(capacity=100)
        assert handler is not None
        assert get_handler() is handler

        # Second call returns same instance
        handler2 = install(capacity=200)
        assert handler2 is handler

        # Cleanup
        logging.getLogger("serenityflow").removeHandler(handler)
        lb._handler = None

    def test_captures_log_entries(self):
        from serenityflow.debug.log_buffer import RingBufferHandler

        handler = RingBufferHandler(capacity=50)
        handler.setFormatter(logging.Formatter("%(message)s"))

        logger = logging.getLogger("serenityflow.test.capture")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("hello world")
        logger.warning("something bad")

        entries = handler.get_entries(n=10)
        assert len(entries) == 2
        assert entries[0]["message"] == "hello world"
        assert entries[0]["level"] == "INFO"
        assert entries[1]["message"] == "something bad"
        assert entries[1]["level"] == "WARNING"

        logger.removeHandler(handler)

    def test_filter_by_level(self):
        from serenityflow.debug.log_buffer import RingBufferHandler

        handler = RingBufferHandler(capacity=50)
        handler.setFormatter(logging.Formatter("%(message)s"))

        logger = logging.getLogger("serenityflow.test.level_filter")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.debug("debug msg")
        logger.info("info msg")
        logger.warning("warn msg")
        logger.error("error msg")

        # Filter WARNING and above
        entries = handler.get_entries(n=10, level="WARNING")
        assert len(entries) == 2
        assert entries[0]["message"] == "warn msg"
        assert entries[1]["message"] == "error msg"

        logger.removeHandler(handler)

    def test_filter_by_component(self):
        from serenityflow.debug.log_buffer import RingBufferHandler

        handler = RingBufferHandler(capacity=50)
        handler.setFormatter(logging.Formatter("%(message)s"))

        logger_a = logging.getLogger("serenityflow.bridge.loading")
        logger_b = logging.getLogger("serenityflow.memory.coordinator")
        for l in (logger_a, logger_b):
            l.addHandler(handler)
            l.setLevel(logging.DEBUG)

        logger_a.info("loading model")
        logger_b.info("coordinator started")

        entries = handler.get_entries(n=10, component="loading")
        assert len(entries) == 1
        assert entries[0]["message"] == "loading model"

        for l in (logger_a, logger_b):
            l.removeHandler(handler)

    def test_ring_buffer_eviction(self):
        from serenityflow.debug.log_buffer import RingBufferHandler

        handler = RingBufferHandler(capacity=3)
        handler.setFormatter(logging.Formatter("%(message)s"))

        logger = logging.getLogger("serenityflow.test.eviction")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        for i in range(5):
            logger.info(f"msg-{i}")

        entries = handler.get_entries(n=10)
        assert len(entries) == 3
        assert entries[0]["message"] == "msg-2"
        assert entries[2]["message"] == "msg-4"
        assert handler.total == 3

        logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Architecture detection tests
# ---------------------------------------------------------------------------


class TestArchDetection:
    def test_guess_flux(self):
        from serenityflow.debug.router import _guess_architecture_from_keys

        keys = [
            "transformer.single_transformer_blocks.0.attn.to_q.weight",
            "transformer.transformer_blocks.0.attn.to_q.weight",
        ]
        assert _guess_architecture_from_keys(keys) == "flux"

    def test_guess_sd3(self):
        from serenityflow.debug.router import _guess_architecture_from_keys

        keys = [
            "model.diffusion_model.joint_blocks.0.x_block.attn.to_q.weight",
            "model.diffusion_model.joint_blocks.1.x_block.attn.to_q.weight",
        ]
        assert _guess_architecture_from_keys(keys) == "sd3"

    def test_guess_sd15(self):
        from serenityflow.debug.router import _guess_architecture_from_keys

        keys = [
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.middle_block.0.in_layers.0.weight",
            "model.diffusion_model.output_blocks.0.0.weight",
        ]
        assert _guess_architecture_from_keys(keys) == "sd15"

    def test_guess_sdxl_with_conditioner(self):
        from serenityflow.debug.router import _guess_architecture_from_keys

        keys = [
            "conditioner.embedders.0.model.text_projection",
            "model.diffusion_model.input_blocks.0.0.weight",
        ]
        assert _guess_architecture_from_keys(keys) == "sdxl"

    def test_guess_unknown(self):
        from serenityflow.debug.router import _guess_architecture_from_keys

        keys = ["some.completely.unknown.key"]
        assert _guess_architecture_from_keys(keys) is None

    def test_strip_lora_suffix(self):
        from serenityflow.debug.router import _strip_lora_suffix

        assert _strip_lora_suffix("blocks.0.to_q.lora_down.weight") == "blocks.0.to_q"
        assert _strip_lora_suffix("blocks.0.to_q.lora_up.weight") == "blocks.0.to_q"
        assert _strip_lora_suffix("blocks.0.to_q.lora_A.weight") == "blocks.0.to_q"
        assert _strip_lora_suffix("blocks.0.to_q.lora_B.weight") == "blocks.0.to_q"
        assert _strip_lora_suffix("blocks.0.to_q.weight") == "blocks.0.to_q.weight"


# ---------------------------------------------------------------------------
# Debug router endpoint tests
# ---------------------------------------------------------------------------


class TestDebugEndpoints:
    def test_pipeline_status_no_runner(self, client):
        state.runner = None
        r = client.get("/debug/pipeline/status")
        assert r.status_code == 200
        data = r.json()
        assert data["pipeline_type"] is None
        assert data["model_loaded"] is False

    def test_pipeline_status_empty_cache(self, client):
        state.runner = FakeRunner()
        r = client.get("/debug/pipeline/status")
        assert r.status_code == 200
        data = r.json()
        assert data["model_loaded"] is False

    def test_vram_status(self, client):
        r = client.get("/debug/vram/status")
        assert r.status_code == 200
        data = r.json()
        # Should have at least the top-level keys
        assert "gpu" in data
        assert "stagehand" in data
        assert "system_ram" in data

    def test_logs_no_handler(self, client):
        """Logs endpoint works even if log handler isn't installed."""
        import serenityflow.debug.log_buffer as lb
        original = lb._handler
        lb._handler = None

        r = client.get("/debug/logs")
        assert r.status_code == 200
        data = r.json()
        assert data["handler_installed"] is False

        lb._handler = original

    def test_logs_with_handler(self, client):
        from serenityflow.debug.log_buffer import RingBufferHandler
        import serenityflow.debug.log_buffer as lb

        handler = RingBufferHandler(capacity=100)
        handler.setFormatter(logging.Formatter("%(message)s"))
        original = lb._handler
        lb._handler = handler

        # Add a log entry
        logger = logging.getLogger("serenityflow.test.endpoint")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info("test entry")

        r = client.get("/debug/logs?lines=50")
        assert r.status_code == 200
        data = r.json()
        assert len(data["lines"]) >= 1
        assert any("test entry" in line["message"] for line in data["lines"])

        logger.removeHandler(handler)
        lb._handler = original

    def test_config_dump(self, client):
        r = client.get("/debug/config")
        assert r.status_code == 200
        data = r.json()
        assert "directories" in data

    def test_models_available(self, client):
        """Models endpoint returns structure even if no models found."""
        r = client.get("/debug/models/available")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert "loras" in data

    def test_tensor_probe_no_runner(self, client):
        state.runner = None
        r = client.post("/debug/tensor/probe", json={"tensor_path": "foo.weight"})
        assert r.status_code == 503

    def test_lora_check_file_not_found(self, client):
        r = client.post("/debug/lora/check", json={
            "lora_path": "/nonexistent/path.safetensors",
        })
        assert r.status_code == 404

    def test_architecture_diff_file_not_found(self, client):
        r = client.post("/debug/architecture/diff", json={
            "lora_path": "/nonexistent/path.safetensors",
        })
        assert r.status_code == 404

    def test_debug_generate_no_model(self, client):
        """Generate with no model loaded returns error."""
        state.runner = FakeRunner()
        r = client.post("/debug/generate", json={"prompt": "test"})
        data = r.json()
        assert "error" in data

    def test_model_load_file_not_found(self, client):
        """Load with nonexistent file returns error."""
        r = client.post("/debug/model/load", json={
            "model_path": "/nonexistent/test.safetensors",
            "pipeline_type": "flux",
        })
        data = r.json()
        assert "error" in data

    def test_model_unload_empty(self, client):
        """Unload with empty cache succeeds."""
        state.runner = FakeRunner()
        r = client.post("/debug/model/unload", json={"component": "all"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "unloaded"


class TestArchDiffWithFiles:
    """Test architecture diff using real temporary safetensors files."""

    def _write_fake_safetensors(self, path: str, keys: list[str]):
        """Create a minimal safetensors file with given keys (zero tensors)."""
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            pytest.skip("safetensors or torch not available")

        tensors = {k: torch.zeros(2, 2) for k in keys}
        save_file(tensors, path)

    def test_matching_architectures(self, client, tmp_path):
        lora_path = str(tmp_path / "test_lora.safetensors")
        model_path = str(tmp_path / "test_model.safetensors")

        lora_keys = [
            "transformer.single_transformer_blocks.0.attn.to_q.lora_down.weight",
            "transformer.single_transformer_blocks.0.attn.to_q.lora_up.weight",
            "transformer.transformer_blocks.0.attn.to_q.lora_down.weight",
            "transformer.transformer_blocks.0.attn.to_q.lora_up.weight",
        ]
        model_keys = [
            "transformer.single_transformer_blocks.0.attn.to_q.weight",
            "transformer.transformer_blocks.0.attn.to_q.weight",
        ]

        self._write_fake_safetensors(lora_path, lora_keys)
        self._write_fake_safetensors(model_path, model_keys)

        r = client.post("/debug/architecture/diff", json={
            "lora_path": lora_path,
            "model_path": model_path,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["lora_architecture_guess"] == "flux"
        assert data["model_architecture"] == "flux"
        assert data["architectures_match"] is True

    def test_mismatched_architectures(self, client, tmp_path):
        lora_path = str(tmp_path / "flux_lora.safetensors")
        model_path = str(tmp_path / "sd15_model.safetensors")

        lora_keys = [
            "transformer.single_transformer_blocks.0.attn.to_q.lora_down.weight",
            "transformer.single_transformer_blocks.0.attn.to_q.lora_up.weight",
        ]
        model_keys = [
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.output_blocks.0.0.weight",
        ]

        self._write_fake_safetensors(lora_path, lora_keys)
        self._write_fake_safetensors(model_path, model_keys)

        r = client.post("/debug/architecture/diff", json={
            "lora_path": lora_path,
            "model_path": model_path,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["lora_architecture_guess"] == "flux"
        assert data["model_architecture"] == "sd15"
        assert data["architectures_match"] is False
        assert "flux" in data["suggestion"].lower()


class TestLoRACheckWithFiles:
    """Test LoRA compatibility check using real temporary safetensors + mocked model."""

    def _write_fake_safetensors(self, path: str, keys: list[str]):
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            pytest.skip("safetensors or torch not available")
        tensors = {k: torch.zeros(2, 2) for k in keys}
        save_file(tensors, path)

    def test_lora_check_no_model(self, client, tmp_path):
        """LoRA check returns 503 when no model is loaded."""
        lora_path = str(tmp_path / "lora.safetensors")
        self._write_fake_safetensors(lora_path, [
            "blocks.0.to_q.lora_down.weight",
            "blocks.0.to_q.lora_up.weight",
        ])

        state.runner = FakeRunner()

        r = client.post("/debug/lora/check", json={"lora_path": lora_path})
        assert r.status_code == 503
        data = r.json()
        assert data["lora_keys_count"] == 1  # 1 unique base key


# ---------------------------------------------------------------------------
# _iter_cache_outputs tests
# ---------------------------------------------------------------------------


class TestIterCacheOutputs:
    def test_empty_cache(self):
        from serenityflow.debug.router import _iter_cache_outputs

        cache = FakeCache()
        assert list(_iter_cache_outputs(cache)) == []

    def test_unwraps_cached_output(self):
        from serenityflow.debug.router import _iter_cache_outputs

        cache = FakeCache()
        sentinel = object()
        cache.set_outputs("node1", (sentinel, None))

        results = list(_iter_cache_outputs(cache))
        assert len(results) == 1
        assert results[0] == ("node1", sentinel)

    def test_multiple_nodes(self):
        from serenityflow.debug.router import _iter_cache_outputs

        cache = FakeCache()
        a, b = object(), object()
        cache.set_outputs("n1", (a,))
        cache.set_outputs("n2", (b,))

        results = list(_iter_cache_outputs(cache))
        assert len(results) == 2


# ---------------------------------------------------------------------------
# MCP tools module tests (no network needed)
# ---------------------------------------------------------------------------

_has_mcp = True
try:
    import mcp  # noqa: F401
except ImportError:
    _has_mcp = False


@pytest.mark.skipif(not _has_mcp, reason="mcp package not installed")
class TestMCPTools:
    def test_tool_list_complete(self):
        from serenityflow_mcp.tools import _TOOLS, _ROUTES

        tool_names = {t.name for t in _TOOLS}
        route_names = set(_ROUTES.keys())

        # Every tool must have a route
        assert tool_names == route_names

    def test_all_tools_have_descriptions(self):
        from serenityflow_mcp.tools import _TOOLS

        for tool in _TOOLS:
            assert tool.description, f"Tool {tool.name} has no description"
            assert tool.inputSchema, f"Tool {tool.name} has no schema"

    def test_config_defaults(self):
        from serenityflow_mcp.config import Config

        cfg = Config()
        assert cfg.base_url == "http://localhost:8188"

    def test_config_env_override(self):
        from serenityflow_mcp.config import Config

        with patch.dict(os.environ, {"SF_API_URL": "http://gpu-box:9999"}):
            cfg = Config()
            assert cfg.base_url == "http://gpu-box:9999"

    def test_config_explicit_override(self):
        from serenityflow_mcp.config import Config

        cfg = Config(base_url="http://custom:1234")
        assert cfg.base_url == "http://custom:1234"


# ---------------------------------------------------------------------------
# MCP client tests (mock HTTP)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_mcp, reason="mcp package not installed")
class TestMCPClient:
    @pytest.mark.asyncio
    async def test_get_request(self):
        from serenityflow_mcp.client import SerenityFlowClient
        import httpx

        client = SerenityFlowClient("http://fake:8188")

        # Mock the internal httpx client
        mock_response = httpx.Response(
            200,
            json={"status": "ok"},
            request=httpx.Request("GET", "http://fake:8188/debug/pipeline/status"),
        )

        async def mock_get(url, **kwargs):
            return mock_response

        client._client.get = mock_get  # type: ignore

        result = await client.get("/pipeline/status")
        assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_post_request(self):
        from serenityflow_mcp.client import SerenityFlowClient
        import httpx

        client = SerenityFlowClient("http://fake:8188")

        mock_response = httpx.Response(
            200,
            json={"compatible": True},
            request=httpx.Request("POST", "http://fake:8188/debug/lora/check"),
        )

        async def mock_post(url, **kwargs):
            return mock_response

        client._client.post = mock_post  # type: ignore

        result = await client.post("/lora/check", json={"lora_path": "/test.safetensors"})
        assert result == {"compatible": True}
