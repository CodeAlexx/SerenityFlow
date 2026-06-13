"""Tests for the debug introspection subsystem.

Tests log_buffer, debug router endpoints, architecture detection, and MCP tools.
All tests mock pipeline state — no GPU or model loading required.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
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
        assert "active_loras" in data
        assert "components" in data

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


# ---------------------------------------------------------------------------
# LoRA Registry tests
# ---------------------------------------------------------------------------


class TestLoraRegistry:
    def test_record_and_retrieve(self):
        from serenityflow.bridge.lora_utils import LoraRecord, LoraRegistry

        reg = LoraRegistry()
        rec = LoraRecord(
            path="/test/lora.safetensors",
            strength=0.8,
            rank=16,
            alpha=16.0,
            keys_matched=42,
            keys_missed=3,
            target_modules=["blocks.0.to_q.weight"],
            timestamp=1.0,
        )
        reg.record(rec)
        results = reg.get_all()
        assert len(results) == 1
        assert results[0].path == "/test/lora.safetensors"
        assert results[0].rank == 16

    def test_clear(self):
        from serenityflow.bridge.lora_utils import LoraRecord, LoraRegistry

        reg = LoraRegistry()
        rec = LoraRecord(
            path="/test.safetensors", strength=1.0, rank=8, alpha=None,
            keys_matched=10, keys_missed=0, target_modules=[], timestamp=0.0,
        )
        reg.record(rec)
        assert len(reg.get_all()) == 1
        reg.clear()
        assert len(reg.get_all()) == 0

    def test_to_dicts(self):
        from serenityflow.bridge.lora_utils import LoraRecord, LoraRegistry

        reg = LoraRegistry()
        rec = LoraRecord(
            path="/test.safetensors", strength=0.5, rank=32, alpha=16.0,
            keys_matched=5, keys_missed=2, target_modules=["a.weight", "b.weight"],
            timestamp=99.0,
        )
        reg.record(rec)
        dicts = reg.to_dicts()
        assert len(dicts) == 1
        d = dicts[0]
        assert isinstance(d, dict)
        assert d["path"] == "/test.safetensors"
        assert d["strength"] == 0.5
        assert d["rank"] == 32
        assert d["alpha"] == 16.0
        assert d["keys_matched"] == 5
        assert d["keys_missed"] == 2
        assert d["target_modules"] == ["a.weight", "b.weight"]
        assert d["timestamp"] == 99.0


class TestPipelineStatusLoras:
    def test_active_loras_populated(self, client):
        from serenityflow.bridge.lora_utils import LoraRecord, get_lora_registry

        registry = get_lora_registry()
        # Clean slate
        registry.clear()
        registry.record(LoraRecord(
            path="/my/lora.safetensors", strength=0.75, rank=16, alpha=16.0,
            keys_matched=100, keys_missed=5, target_modules=["x.weight"],
            timestamp=1.0,
        ))

        state.runner = FakeRunner()
        r = client.get("/debug/pipeline/status")
        assert r.status_code == 200
        data = r.json()
        assert len(data["active_loras"]) == 1
        assert data["active_loras"][0]["path"] == "/my/lora.safetensors"
        assert data["active_loras"][0]["strength"] == 0.75

        # Cleanup
        registry.clear()

    def test_active_loras_visible_without_runner(self, client):
        """LoRAs registered globally must appear even when runner is None."""
        from serenityflow.bridge.lora_utils import LoraRecord, get_lora_registry

        registry = get_lora_registry()
        registry.clear()
        registry.record(LoraRecord(
            path="/orphan/lora.safetensors", strength=1.0, rank=8, alpha=None,
            keys_matched=10, keys_missed=0, target_modules=[], timestamp=0.0,
        ))

        state.runner = None
        r = client.get("/debug/pipeline/status")
        assert r.status_code == 200
        data = r.json()
        assert len(data["active_loras"]) == 1
        assert data["active_loras"][0]["path"] == "/orphan/lora.safetensors"

        registry.clear()


class TestBreakpointGenerate:
    def test_no_model_returns_503(self, client):
        """Breakpoint generate with no runner returns 503."""
        state.runner = None
        r = client.post("/debug/generate/breakpoint", json={"prompt": "test"})
        assert r.status_code == 503
        data = r.json()
        assert "error" in data

    def test_no_model_loaded_returns_503(self, client):
        """Breakpoint generate with runner but no model returns 503."""
        state.runner = FakeRunner()
        r = client.post("/debug/generate/breakpoint", json={"prompt": "test"})
        assert r.status_code == 503
        data = r.json()
        assert "error" in data
        assert "No model loaded" in data["error"]

    def test_invalid_resume_token(self, client):
        """Breakpoint generate with invalid resume_token returns 400."""
        # Need a runner with a model to get past the early checks
        import torch

        fake_model = torch.nn.Linear(4, 4)
        fake_model._serenity_arch = None

        cache = FakeCache()
        # Create a fake LoadedCheckpoint
        class FakeLoadedCheckpoint:
            __name__ = "LoadedCheckpoint"
            def __init__(self):
                self.model = fake_model
        lc = FakeLoadedCheckpoint()
        type(lc).__name__ = "LoadedCheckpoint"
        cache.set_outputs("n1", (lc,))

        state.runner = FakeRunner(cache=cache)
        r = client.post("/debug/generate/breakpoint", json={
            "prompt": "test",
            "resume_token": "invalid_token_xyz",
        })
        assert r.status_code == 400
        data = r.json()
        assert "error" in data
        assert "invalid" in data["error"].lower() or "expired" in data["error"].lower()

    def test_breakpoint_request_model_defaults(self):
        """BreakpointGenerateRequest has correct defaults."""
        from serenityflow.debug.router import BreakpointGenerateRequest

        req = BreakpointGenerateRequest(prompt="hello")
        assert req.prompt == "hello"
        assert req.negative_prompt == ""
        assert req.width == 512
        assert req.height == 512
        assert req.steps == 4
        assert req.guidance_scale == 3.5
        assert req.seed == 42
        assert req.break_at_step == 2
        assert req.resume_token is None


class TestABCompare:
    def test_no_model_returns_503(self, client):
        """A/B compare with no runner returns 503."""
        state.runner = None
        r = client.post("/debug/generate/ab_compare", json={
            "prompt": "test",
            "lora_path": "/some/lora.safetensors",
        })
        assert r.status_code == 503
        data = r.json()
        assert "error" in data

    def test_no_model_loaded_returns_503(self, client):
        """A/B compare with runner but no model returns 503."""
        state.runner = FakeRunner()
        r = client.post("/debug/generate/ab_compare", json={
            "prompt": "test",
            "lora_path": "/some/lora.safetensors",
        })
        assert r.status_code == 503
        data = r.json()
        assert "error" in data
        assert "No model loaded" in data["error"]

    def test_missing_lora_file(self, client):
        """A/B compare with nonexistent lora_path returns 404."""
        import torch

        fake_model = torch.nn.Linear(4, 4)
        fake_model._serenity_arch = None

        cache = FakeCache()

        class FakeLoadedCheckpoint:
            def __init__(self):
                self.model = fake_model
        lc = FakeLoadedCheckpoint()
        type(lc).__name__ = "LoadedCheckpoint"
        cache.set_outputs("n1", (lc,))

        state.runner = FakeRunner(cache=cache)
        r = client.post("/debug/generate/ab_compare", json={
            "prompt": "test",
            "lora_path": "/nonexistent/lora.safetensors",
        })
        assert r.status_code == 404
        data = r.json()
        assert "error" in data
        assert "not found" in data["error"].lower()

    def test_ab_compare_request_model(self):
        """ABCompareRequest has correct defaults."""
        from serenityflow.debug.router import ABCompareRequest

        req = ABCompareRequest(prompt="hello")
        assert req.prompt == "hello"
        assert req.negative_prompt == ""
        assert req.width == 512
        assert req.height == 512
        assert req.steps == 4
        assert req.guidance_scale == 3.5
        assert req.seed == 42
        assert req.lora_path == ""
        assert req.lora_strength == 1.0


class TestVramStatusEnhanced:
    def test_vram_status_no_coordinator(self, client):
        """VRAM endpoint works with no stagehand coordinator."""
        r = client.get("/debug/vram/status")
        assert r.status_code == 200
        data = r.json()
        assert "gpu" in data
        assert "stagehand" in data
        assert "system_ram" in data


# ---------------------------------------------------------------------------
# Training metrics tests
# ---------------------------------------------------------------------------


def _create_board_db(db_path: str, scalars: list[tuple] | None = None, sessions: list[tuple] | None = None):
    """Create a minimal SerenityBoard board.db with schema and optional data."""
    import sqlite3
    from pathlib import Path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT NOT NULL, start_time REAL NOT NULL,
            resume_step INTEGER, status TEXT NOT NULL,
            PRIMARY KEY (session_id)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS scalars (
            tag TEXT NOT NULL, step INTEGER NOT NULL,
            wall_time REAL NOT NULL, value REAL NOT NULL,
            PRIMARY KEY (tag, step)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_scalars_tag_step ON scalars(tag, step DESC);
    ''')
    if sessions:
        conn.executemany("INSERT INTO sessions VALUES (?, ?, ?, ?)", sessions)
    if scalars:
        conn.executemany("INSERT INTO scalars VALUES (?, ?, ?, ?)", scalars)
    conn.commit()
    conn.close()


class TestTrainingMetrics:
    def test_find_latest_run(self, tmp_path):
        from serenityflow.debug.training_reader import find_latest_run

        # Create two run dirs
        run1 = tmp_path / "run1" / "board.db"
        run2 = tmp_path / "run2" / "board.db"
        _create_board_db(str(run1))

        import time
        time.sleep(0.05)  # ensure different mtime
        _create_board_db(str(run2))

        result = find_latest_run(str(tmp_path))
        assert result is not None
        name, path = result
        assert name == "run2"
        assert path.endswith("board.db")

    def test_find_latest_run_empty_dir(self, tmp_path):
        from serenityflow.debug.training_reader import find_latest_run

        assert find_latest_run(str(tmp_path)) is None

    def test_find_latest_run_nonexistent_dir(self):
        from serenityflow.debug.training_reader import find_latest_run

        assert find_latest_run("/nonexistent/path/xyz") is None

    def test_read_latest_scalars(self, tmp_path):
        from serenityflow.debug.training_reader import _open_readonly, _read_latest_scalars

        db_path = str(tmp_path / "run1" / "board.db")
        _create_board_db(db_path, scalars=[
            ("loss/train", 1, 1000.0, 0.5),
            ("loss/train", 2, 1001.0, 0.4),
            ("loss/train", 3, 1002.0, 0.35),
            ("grad_norm", 1, 1000.0, 1.2),
            ("grad_norm", 2, 1001.0, 0.9),
        ])

        conn = _open_readonly(db_path)
        try:
            result = _read_latest_scalars(conn, ["loss/train", "grad_norm", "lr/default"])
            assert "loss/train" in result
            assert result["loss/train"]["step"] == 3
            assert result["loss/train"]["value"] == 0.35
            assert "grad_norm" in result
            assert result["grad_norm"]["step"] == 2
            assert result["grad_norm"]["value"] == 0.9
            # lr/default not inserted, should not appear
            assert "lr/default" not in result
        finally:
            conn.close()

    def test_read_recent_series(self, tmp_path):
        from serenityflow.debug.training_reader import _open_readonly, _read_recent_series

        db_path = str(tmp_path / "run1" / "board.db")
        scalars = [("loss/train", i, 1000.0 + i, 0.5 - i * 0.01) for i in range(1, 11)]
        _create_board_db(db_path, scalars=scalars)

        conn = _open_readonly(db_path)
        try:
            series = _read_recent_series(conn, "loss/train", 5)
            assert len(series) == 5
            # Should be in chronological order (step 6..10)
            assert series[0][0] == 6
            assert series[4][0] == 10
            # Values should decrease
            assert series[0][1] > series[4][1]
        finally:
            conn.close()

    def test_compute_summary_decreasing_loss(self):
        from serenityflow.debug.training_reader import _compute_summary

        series = {
            "loss/train": [[i, 1.0 - i * 0.1] for i in range(10)],
        }
        summary = _compute_summary(series)
        assert summary["loss_trend"] == "decreasing"
        assert "loss_last_n_mean" in summary
        assert "loss_last_n_std" in summary

    def test_compute_summary_increasing_loss(self):
        from serenityflow.debug.training_reader import _compute_summary

        series = {
            "loss/train": [[i, 0.1 + i * 0.1] for i in range(10)],
        }
        summary = _compute_summary(series)
        assert summary["loss_trend"] == "increasing"

    def test_compute_summary_with_speed_and_grad(self):
        from serenityflow.debug.training_reader import _compute_summary

        series = {
            "perf/steps_per_sec": [[i, 2.5] for i in range(5)],
            "grad_norm": [[i, 1.0 + i * 0.1] for i in range(5)],
        }
        summary = _compute_summary(series)
        assert summary["training_speed_steps_per_sec"] == 2.5
        assert "grad_norm_mean" in summary
        assert "grad_norm_max" in summary

    def test_read_training_metrics_full(self, tmp_path):
        from serenityflow.debug.training_reader import read_training_metrics

        db_path = str(tmp_path / "myrun" / "board.db")
        scalars = [
            ("loss/train", i, 1000.0 + i, 0.5 - i * 0.01) for i in range(1, 21)
        ] + [
            ("grad_norm", i, 1000.0 + i, 1.0) for i in range(1, 21)
        ] + [
            ("lr/default", 20, 1020.0, 0.0001),
        ]
        _create_board_db(db_path, scalars=scalars, sessions=[
            ("sess1", 1000.0, None, "running"),
        ])

        result = read_training_metrics(str(tmp_path), run_name="myrun", last_n_steps=10)
        assert result["run_name"] == "myrun"
        assert result["session_status"] == "running"
        assert result["current_step"] == 20
        assert "loss/train" in result["latest"]
        assert "lr/default" in result["latest"]
        assert "loss/train" in result["recent_series"]
        assert len(result["recent_series"]["loss/train"]) == 10
        assert "loss_trend" in result["summary"]
        assert "available_tags" in result
        assert "myrun" in result["available_runs"]

    def test_read_training_metrics_auto_detect_run(self, tmp_path):
        from serenityflow.debug.training_reader import read_training_metrics

        db_path = str(tmp_path / "auto_run" / "board.db")
        _create_board_db(db_path, scalars=[
            ("loss/train", 1, 1000.0, 0.5),
        ], sessions=[
            ("s1", 1000.0, None, "complete"),
        ])

        result = read_training_metrics(str(tmp_path))
        assert result["run_name"] == "auto_run"

    def test_read_training_metrics_no_db(self, tmp_path):
        from serenityflow.debug.training_reader import read_training_metrics

        with pytest.raises(FileNotFoundError):
            read_training_metrics(str(tmp_path))

    def test_read_training_metrics_specific_run_not_found(self, tmp_path):
        from serenityflow.debug.training_reader import read_training_metrics

        with pytest.raises(FileNotFoundError):
            read_training_metrics(str(tmp_path), run_name="nonexistent")

    def test_no_board_db_returns_404(self, client, tmp_path):
        r = client.get("/debug/training/metrics", params={"log_dir": str(tmp_path)})
        assert r.status_code == 404
        assert "error" in r.json()

    def test_missing_log_dir_returns_400(self, client):
        r = client.get("/debug/training/metrics")
        assert r.status_code == 400
        assert "error" in r.json()
        assert "log_dir" in r.json()["error"]

    def test_endpoint_with_real_db(self, client, tmp_path):
        db_path = str(tmp_path / "test_run" / "board.db")
        _create_board_db(db_path, scalars=[
            ("loss/train", 1, 1000.0, 0.5),
            ("loss/train", 2, 1001.0, 0.45),
            ("loss/train", 3, 1002.0, 0.4),
            ("grad_norm", 1, 1000.0, 1.5),
            ("lr/default", 3, 1002.0, 0.0001),
        ], sessions=[
            ("s1", 1000.0, None, "running"),
        ])

        r = client.get("/debug/training/metrics", params={
            "log_dir": str(tmp_path),
            "run_name": "test_run",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["run_name"] == "test_run"
        assert data["session_status"] == "running"
        assert data["current_step"] == 3
        assert "loss/train" in data["latest"]
        assert data["latest"]["loss/train"]["value"] == 0.4
        assert "available_tags" in data

    def test_endpoint_with_custom_tags(self, client, tmp_path):
        db_path = str(tmp_path / "tag_run" / "board.db")
        _create_board_db(db_path, scalars=[
            ("custom/metric", 1, 1000.0, 42.0),
            ("loss/train", 1, 1000.0, 0.5),
        ])

        r = client.get("/debug/training/metrics", params={
            "log_dir": str(tmp_path),
            "run_name": "tag_run",
            "tags": "custom/metric,loss/train",
        })
        assert r.status_code == 200
        data = r.json()
        assert "custom/metric" in data["latest"]
        assert data["latest"]["custom/metric"]["value"] == 42.0


# ---------------------------------------------------------------------------
# Pipeline Diff tests
# ---------------------------------------------------------------------------


class TestPipelineDiff:
    def test_save_snapshot(self, client):
        state.runner = None
        r = client.post("/debug/pipeline/diff", json={"save_snapshot": "test1"})
        assert r.status_code == 200
        data = r.json()
        assert data["action"] == "saved"
        assert data["name"] == "test1"
        assert "model" in data["keys_captured"]
        assert "timestamp" in data

    def test_list_snapshots(self, client):
        state.runner = None
        # Save two snapshots
        client.post("/debug/pipeline/diff", json={"save_snapshot": "snap_a"})
        client.post("/debug/pipeline/diff", json={"save_snapshot": "snap_b"})

        r = client.post("/debug/pipeline/diff", json={})
        assert r.status_code == 200
        data = r.json()
        assert data["action"] == "list"
        assert data["count"] >= 2
        assert "snap_a" in data["snapshots"]
        assert "snap_b" in data["snapshots"]

    def test_diff_identical(self, client):
        state.runner = None
        # Save a snapshot, then diff it against current (same state)
        client.post("/debug/pipeline/diff", json={"save_snapshot": "baseline"})

        r = client.post("/debug/pipeline/diff", json={
            "snapshot_a": "baseline",
            "snapshot_b": None,  # current
        })
        assert r.status_code == 200
        data = r.json()
        assert data["action"] == "diff"
        assert data["diff_count"] == 0
        assert data["identical_count"] > 0

    def test_diff_unknown_snapshot(self, client):
        r = client.post("/debug/pipeline/diff", json={
            "snapshot_a": "nonexistent_snap",
            "snapshot_b": None,
        })
        assert r.status_code == 404
        data = r.json()
        assert "error" in data
        assert "nonexistent_snap" in data["error"]

    def test_diff_snapshots_with_differences(self, client):
        state.runner = None
        # Save a snapshot
        client.post("/debug/pipeline/diff", json={"save_snapshot": "before"})

        # Access the internal snapshot storage and modify it to simulate change
        from serenityflow.debug.router import register_debug_routes
        # We need to get at the _config_snapshots closure. Instead, save another
        # snapshot and then mutate the stored one via a second save with different state.
        # Simpler: save "before", change state, save "after", then diff.
        # Since runner is None both times, model.loaded will be the same.
        # So let's directly modify the snapshot through the endpoint by saving
        # a second snapshot with a different runner state.
        import torch
        fake_model = torch.nn.Linear(4, 4)

        class FakeLoadedCheckpoint:
            def __init__(self):
                self.model = fake_model
                self.model_config = None
        lc = FakeLoadedCheckpoint()
        type(lc).__name__ = "LoadedCheckpoint"

        cache = FakeCache()
        cache.set_outputs("n1", (lc,))
        state.runner = FakeRunner(cache=cache)

        client.post("/debug/pipeline/diff", json={"save_snapshot": "after"})

        r = client.post("/debug/pipeline/diff", json={
            "snapshot_a": "before",
            "snapshot_b": "after",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["action"] == "diff"
        assert data["diff_count"] > 0
        # model.loaded should differ (False vs True)
        model_diffs = [d for d in data["differences"] if d["key"] == "model.loaded"]
        assert len(model_diffs) == 1
        assert model_diffs[0]["a"] is False
        assert model_diffs[0]["b"] is True

    def test_pipeline_diff_request_model(self):
        from serenityflow.debug.router import PipelineDiffRequest

        req = PipelineDiffRequest()
        assert req.snapshot_a is None
        assert req.snapshot_b is None
        assert req.save_snapshot is None

        req2 = PipelineDiffRequest(save_snapshot="foo")
        assert req2.save_snapshot == "foo"


# ---------------------------------------------------------------------------
# Diagnose (meta-tool) tests
# ---------------------------------------------------------------------------


class TestDiagnose:
    def test_full_mode_no_model(self, client):
        """Diagnose full mode with no runner returns overall error."""
        state.runner = None
        r = client.post("/debug/diagnose", json={"mode": "full"})
        assert r.status_code == 200
        data = r.json()
        assert data["overall_status"] == "error"
        assert "pipeline" in data["sections"]
        assert data["sections"]["pipeline"]["status"] == "error"

    def test_full_mode_structure(self, client):
        """Response has required top-level keys."""
        state.runner = None
        r = client.post("/debug/diagnose", json={"mode": "full"})
        assert r.status_code == 200
        data = r.json()
        assert "mode" in data
        assert "overall_status" in data
        assert "sections" in data
        assert "summary" in data
        assert data["mode"] == "full"

    def test_lora_mode_without_path(self, client):
        """Lora mode without lora_path returns error section."""
        state.runner = None
        r = client.post("/debug/diagnose", json={"mode": "lora"})
        assert r.status_code == 200
        data = r.json()
        assert "lora_compatibility" in data["sections"]
        assert data["sections"]["lora_compatibility"]["status"] == "error"
        assert "required" in data["sections"]["lora_compatibility"]["diagnosis"].lower()

    def test_performance_mode(self, client):
        """Performance mode has pipeline, vram, logs but NOT lora or weight_health."""
        state.runner = None
        r = client.post("/debug/diagnose", json={"mode": "performance"})
        assert r.status_code == 200
        data = r.json()
        sections = data["sections"]
        assert "pipeline" in sections
        assert "vram" in sections
        assert "logs" in sections
        assert "lora_compatibility" not in sections
        assert "weight_health" not in sections

    def test_health_mode(self, client):
        """Health mode has pipeline, vram, weight_health, logs."""
        state.runner = None
        r = client.post("/debug/diagnose", json={"mode": "health"})
        assert r.status_code == 200
        data = r.json()
        sections = data["sections"]
        assert "pipeline" in sections
        assert "vram" in sections
        assert "weight_health" in sections
        assert "logs" in sections

    def test_diagnose_request_model(self):
        """DiagnoseRequest Pydantic model has correct defaults."""
        from serenityflow.debug.router import DiagnoseRequest

        req = DiagnoseRequest()
        assert req.mode == "full"
        assert req.lora_path is None
        assert req.tensor_paths is None
        assert req.training_log_dir is None

    def test_runner_pipeline_no_state(self):
        """DiagnosticRunner._check_pipeline returns error when no runner."""
        from serenityflow.debug.diagnose import DiagnosticRunner

        # state.runner is None from the autouse fixture
        runner = DiagnosticRunner()
        section = runner._check_pipeline()
        assert section.status == "error"
        assert section.data["model_loaded"] is False
        assert "No pipeline runner" in section.diagnosis

    def test_full_mode_with_model(self, client):
        """Diagnose with a fake model loaded returns ok for pipeline."""
        import torch

        fake_model = torch.nn.Linear(4, 4)

        class FakeLoadedCheckpoint:
            def __init__(self):
                self.model = fake_model
                self.model_config = None
        lc = FakeLoadedCheckpoint()
        type(lc).__name__ = "LoadedCheckpoint"

        cache = FakeCache()
        cache.set_outputs("n1", (lc,))
        state.runner = FakeRunner(cache=cache)

        r = client.post("/debug/diagnose", json={"mode": "full"})
        assert r.status_code == 200
        data = r.json()
        assert data["sections"]["pipeline"]["status"] == "ok"
        assert data["sections"]["pipeline"]["data"]["model_loaded"] is True

    def test_training_section_with_db(self, client, tmp_path):
        """Diagnose with training_log_dir containing a board.db."""
        state.runner = None
        db_path = str(tmp_path / "run1" / "board.db")
        _create_board_db(db_path, scalars=[
            ("loss/train", i, 1000.0 + i, 0.5 - i * 0.01)
            for i in range(1, 11)
        ], sessions=[
            ("s1", 1000.0, None, "running"),
        ])

        r = client.post("/debug/diagnose", json={
            "mode": "health",
            "training_log_dir": str(tmp_path),
        })
        assert r.status_code == 200
        data = r.json()
        assert "training" in data["sections"]
        assert data["sections"]["training"]["data"]["session_status"] == "running"
        assert data["sections"]["training"]["data"]["current_step"] == 10
