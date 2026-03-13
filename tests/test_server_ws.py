"""WebSocket protocol tests."""
from __future__ import annotations

import os

import pytest

from serenityflow.cli import _setup_compat_path
_setup_compat_path()

from fastapi.testclient import TestClient
from serenityflow.server.app import app, state


@pytest.fixture(autouse=True)
def _setup_dirs(tmp_path):
    state.output_dir = str(tmp_path / "output")
    state.input_dir = str(tmp_path / "input")
    state.temp_dir = str(tmp_path / "temp")
    os.makedirs(state.output_dir, exist_ok=True)
    os.makedirs(state.input_dir, exist_ok=True)
    os.makedirs(state.temp_dir, exist_ok=True)
    state.history.clear()
    # Clear WS connections from previous tests
    state.ws_connections.clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_ws_connect(client):
    """WebSocket should connect and receive initial status."""
    with client.websocket_connect("/ws") as ws:
        data = ws.receive_json()
        assert data["type"] == "status"
        assert "exec_info" in data["data"]["status"]
        assert "queue_remaining" in data["data"]["status"]["exec_info"]


def test_ws_connect_with_client_id(client):
    """WebSocket should accept clientId query parameter."""
    with client.websocket_connect("/ws?clientId=test123") as ws:
        data = ws.receive_json()
        assert data["type"] == "status"
        assert data["data"]["sid"] == "test123"


def test_ws_connect_no_client_id(client):
    """WebSocket without clientId should get a generated sid."""
    with client.websocket_connect("/ws") as ws:
        data = ws.receive_json()
        assert data["type"] == "status"
        assert "sid" in data["data"]
        assert len(data["data"]["sid"]) > 0


def test_ws_multiple_clients(client):
    """Two WebSocket clients should both connect successfully."""
    with client.websocket_connect("/ws?clientId=client1") as ws1:
        data1 = ws1.receive_json()
        assert data1["data"]["sid"] == "client1"

        with client.websocket_connect("/ws?clientId=client2") as ws2:
            data2 = ws2.receive_json()
            assert data2["data"]["sid"] == "client2"


def test_ws_feature_flags(client):
    """Client should be able to send feature_flags message."""
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # Consume initial status
        ws.send_json({"type": "feature_flags", "data": {"supports_preview": True}})
        # Should not crash — no response expected


def test_ws_invalid_json(client):
    """Sending invalid JSON should not crash the connection."""
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # Consume initial status
        ws.send_text("not json {{{")
        # Connection should stay alive — send another message
        ws.send_json({"type": "ping"})
