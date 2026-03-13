"""Server REST endpoint tests."""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

# Setup compat path before importing server
from serenityflow.cli import _setup_compat_path
_setup_compat_path()

from fastapi.testclient import TestClient
from serenityflow.server.app import app, state


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


@pytest.fixture
def client():
    return TestClient(app)


def test_system_stats(client):
    r = client.get("/system_stats")
    assert r.status_code == 200
    data = r.json()
    assert "system" in data
    assert "devices" in data
    assert "python_version" in data["system"]
    assert "pytorch_version" in data["system"]


def test_object_info(client):
    r = client.get("/object_info")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)
    assert len(data) > 0  # At least compat stubs registered


def test_object_info_single_exists(client):
    r = client.get("/object_info/KSampler")
    assert r.status_code == 200
    data = r.json()
    assert "KSampler" in data


def test_object_info_single_missing(client):
    r = client.get("/object_info/NonexistentNode99999")
    assert r.status_code == 404


def test_models_list(client):
    r = client.get("/models")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_queue_empty(client):
    r = client.get("/queue")
    assert r.status_code == 200
    data = r.json()
    assert "queue_running" in data
    assert "queue_pending" in data
    assert len(data["queue_running"]) == 0


def test_history_empty(client):
    r = client.get("/history")
    assert r.status_code == 200


def test_history_missing_item(client):
    r = client.get("/history/nonexistent-id")
    assert r.status_code == 404


def test_prompt_empty(client):
    r = client.post("/prompt", json={"prompt": {}})
    assert r.status_code == 400
    assert "error" in r.json()


def test_prompt_invalid_node(client):
    r = client.post("/prompt", json={
        "prompt": {
            "1": {"class_type": "TotallyFakeNode999", "inputs": {}}
        }
    })
    assert r.status_code == 400
    assert "node_errors" in r.json()


def test_prompt_valid_compat_node(client):
    """KSampler is a compat stub — should be accepted for queueing."""
    r = client.post("/prompt", json={
        "prompt": {
            "1": {"class_type": "KSampler", "inputs": {}}
        }
    })
    assert r.status_code == 200
    data = r.json()
    assert "prompt_id" in data


def test_interrupt(client):
    r = client.post("/interrupt")
    assert r.status_code == 200


def test_free(client):
    r = client.post("/free", json={"free_memory": True})
    assert r.status_code == 200


def test_features(client):
    r = client.get("/features")
    assert r.status_code == 200
    assert r.json() == []


def test_view_nonexistent(client):
    r = client.get("/view?filename=nonexistent.png")
    assert r.status_code == 404


def test_view_path_traversal(client):
    r = client.get("/view?filename=../../etc/passwd")
    assert r.status_code in (403, 404)


def test_view_path_traversal_subfolder(client):
    r = client.get("/view?filename=test.png&subfolder=../../etc")
    assert r.status_code in (403, 404)


def test_view_existing_file(client, tmp_path):
    """Create a file in output dir and serve it."""
    test_file = os.path.join(state.output_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("hello")
    r = client.get("/view?filename=test.txt&type=output")
    assert r.status_code == 200
    assert r.text == "hello"


def test_upload_no_file(client):
    r = client.post("/upload/image")
    assert r.status_code == 422  # Missing file


def test_upload_image(client):
    """Upload a small file."""
    import io
    content = b"fake image content"
    r = client.post(
        "/upload/image",
        files={"image": ("test.png", io.BytesIO(content), "image/png")},
        data={"type": "input", "overwrite": "false", "subfolder": ""},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "test.png"
    assert data["type"] == "input"
    # Verify file exists
    assert os.path.exists(os.path.join(state.input_dir, "test.png"))


def test_upload_no_overwrite(client):
    """Upload same filename twice without overwrite — should rename."""
    import io
    content = b"data"
    # First upload
    client.post(
        "/upload/image",
        files={"image": ("dupe.png", io.BytesIO(content), "image/png")},
        data={"type": "input", "overwrite": "false", "subfolder": ""},
    )
    # Second upload
    r = client.post(
        "/upload/image",
        files={"image": ("dupe.png", io.BytesIO(content), "image/png")},
        data={"type": "input", "overwrite": "false", "subfolder": ""},
    )
    assert r.status_code == 200
    assert r.json()["name"] == "dupe_1.png"


def test_upload_invalid_type(client):
    import io
    r = client.post(
        "/upload/image",
        files={"image": ("test.png", io.BytesIO(b"data"), "image/png")},
        data={"type": "output", "overwrite": "false", "subfolder": ""},
    )
    assert r.status_code == 400


def test_history_clear(client):
    state.history["test-1"] = {"prompt": {}, "outputs": {}}
    r = client.post("/history", json={"clear": True})
    assert r.status_code == 200
    assert len(state.history) == 0


def test_history_delete(client):
    state.history["test-1"] = {"prompt": {}, "outputs": {}}
    state.history["test-2"] = {"prompt": {}, "outputs": {}}
    r = client.post("/history", json={"delete": ["test-1"]})
    assert r.status_code == 200
    assert "test-1" not in state.history
    assert "test-2" in state.history


def test_queue_clear(client):
    r = client.post("/queue", json={"clear": True})
    assert r.status_code == 200


def test_timeline_missing(client):
    r = client.get("/api/sf/timeline/nonexistent")
    assert r.status_code == 404


def test_timeline_exists(client):
    state.history["test-1"] = {
        "prompt": {},
        "outputs": {},
        "timeline": {"total_ms": 100, "nodes": []},
    }
    r = client.get("/api/sf/timeline/test-1")
    assert r.status_code == 200
    assert r.json()["total_ms"] == 100


def test_embeddings(client):
    r = client.get("/embeddings")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
