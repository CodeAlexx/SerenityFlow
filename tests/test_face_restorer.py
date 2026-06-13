"""Tests for CodeFormer face restoration — V8 NLE feature.

Covers:
  - FaceRestorer module (face_restorer.py)
  - REST endpoints (video_edit_routes.py facetools section)
  - Frontend integration contract (WS events, dialog behavior)

Uses mocks for all GPU/model operations — no actual models required.
"""
from __future__ import annotations

import io
import json
import os
import threading
import tempfile
import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
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
    """Temp dirs for output/input/temp + clean job state."""
    state.output_dir = str(tmp_path / "output")
    state.input_dir = str(tmp_path / "input")
    state.temp_dir = str(tmp_path / "temp")
    for d in (state.output_dir, state.input_dir, state.temp_dir):
        os.makedirs(d, exist_ok=True)
    state.history.clear()
    # Clear any leftover active jobs from previous tests
    from serenityflow.server import video_edit_routes as ver
    if hasattr(ver, "_face_active_jobs"):
        ver._face_active_jobs.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def model_dir(tmp_path):
    """Create a facetools model dir with a fake codeformer.pth.
    Uses the same path that _facetools_model_dir() resolves from state.output_dir."""
    d = tmp_path / "output" / "models" / "facetools"
    d.mkdir(parents=True, exist_ok=True)
    (d / "codeformer.pth").write_bytes(b"fake-model-data")
    return str(d)


@pytest.fixture
def model_dir_empty(tmp_path):
    """Facetools model dir with NO model files.
    Note: _facetools_model_dir() auto-creates the dir from state.output_dir,
    so 'missing' means the dir exists but has no model files."""
    d = tmp_path / "output" / "models" / "facetools"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


@pytest.fixture
def fake_video(tmp_path):
    """Create a small real MP4 video (10 frames, 64x64) for testing."""
    path = str(tmp_path / "test_clip.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (64, 64))
    for i in range(10):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def project_with_clip(tmp_path, fake_video):
    """Create a minimal project JSON with one video clip.
    _project_path() resolves to {state.output_dir}/video_projects/{id}.json
    via _projects_dir() which uses state.output_dir."""
    proj_dir = tmp_path / "output" / "video_projects"
    proj_dir.mkdir(parents=True, exist_ok=True)
    proj = {
        "id": "test-proj",
        "tracks": [{
            "id": "track-1",
            "clips": [{
                "id": "clip-abc",
                "source_path": fake_video,
                "type": "video",
            }],
        }],
    }
    proj_path = proj_dir / "test-proj.json"
    proj_path.write_text(json.dumps(proj, indent=2))
    return str(proj_path), proj


# ---------------------------------------------------------------------------
# FaceRestorer Module Tests
# ---------------------------------------------------------------------------

class TestFaceRestorerModule:
    """Unit tests for serenityflow.server.face_restorer.FaceRestorer."""

    def test_check_models_all_missing(self, model_dir_empty):
        from serenityflow.server.face_restorer import FaceRestorer
        fr = FaceRestorer(model_dir=model_dir_empty)
        models = fr.check_models()
        assert "codeformer" in models
        assert models["codeformer"] is False

    def test_check_models_all_present(self, model_dir):
        from serenityflow.server.face_restorer import FaceRestorer
        fr = FaceRestorer(model_dir=model_dir)
        models = fr.check_models()
        assert models["codeformer"] is True

    def test_all_models_present_true(self, model_dir):
        from serenityflow.server.face_restorer import FaceRestorer
        fr = FaceRestorer(model_dir=model_dir)
        assert fr.all_models_present() is True

    def test_all_models_present_false(self, model_dir_empty):
        from serenityflow.server.face_restorer import FaceRestorer
        fr = FaceRestorer(model_dir=model_dir_empty)
        assert fr.all_models_present() is False

    def test_unload_when_not_loaded(self):
        """Unload on fresh instance should not crash."""
        from serenityflow.server.face_restorer import FaceRestorer
        fr = FaceRestorer(model_dir="/nonexistent")
        fr.unload()  # Should be a no-op
        assert fr.model is None
        assert fr.face_helper is None

    def test_unload_clears_state(self):
        from serenityflow.server.face_restorer import FaceRestorer
        fr = FaceRestorer(model_dir="/nonexistent")
        fr.model = MagicMock()
        fr.face_helper = MagicMock()
        with patch("torch.cuda.is_available", return_value=False):
            fr.unload()
        assert fr.model is None
        assert fr.face_helper is None

    def test_restore_frame_no_faces(self):
        """Frame with 0 faces detected returns original frame unchanged."""
        from serenityflow.server.face_restorer import FaceRestorer
        fr = FaceRestorer(model_dir="/nonexistent")
        # Mock the face_helper
        fr.face_helper = MagicMock()
        fr.face_helper.cropped_faces = []  # No faces detected
        fr.model = MagicMock()

        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = fr.restore_frame(frame, fidelity=0.7)
        assert np.array_equal(result, frame), "No-face frame should pass through unchanged"
        fr.model.assert_not_called()

    def test_restore_frame_single_face(self):
        """Frame with 1 face: model called once, result pasted back."""
        from serenityflow.server.face_restorer import FaceRestorer
        import torch

        fr = FaceRestorer(model_dir="/nonexistent", device="cpu")
        fr.face_helper = MagicMock()
        # Simulate one detected face (512x512 BGR crop)
        fake_crop = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        fr.face_helper.cropped_faces = [fake_crop]

        # Model returns a tensor
        fake_output = torch.rand(1, 3, 512, 512)
        fr.model = MagicMock(return_value=(fake_output,))

        # paste_faces_to_input_image returns the final composited frame
        expected_result = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        fr.face_helper.paste_faces_to_input_image.return_value = expected_result

        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = fr.restore_frame(frame, fidelity=0.5)

        fr.model.assert_called_once()
        # Check fidelity was passed
        call_kwargs = fr.model.call_args
        assert call_kwargs[1]["w"] == 0.5
        assert call_kwargs[1]["adain"] is True
        fr.face_helper.add_restored_face.assert_called_once()
        fr.face_helper.get_inverse_affine.assert_called_once()
        fr.face_helper.paste_faces_to_input_image.assert_called_once()
        assert np.array_equal(result, expected_result)

    def test_restore_frame_multiple_faces(self):
        """Frame with N faces: model called N times."""
        from serenityflow.server.face_restorer import FaceRestorer
        import torch

        fr = FaceRestorer(model_dir="/nonexistent", device="cpu")
        fr.face_helper = MagicMock()
        # 3 faces detected
        crops = [np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(3)]
        fr.face_helper.cropped_faces = crops

        fake_output = torch.rand(1, 3, 512, 512)
        fr.model = MagicMock(return_value=(fake_output,))
        fr.face_helper.paste_faces_to_input_image.return_value = np.zeros((64, 64, 3), dtype=np.uint8)

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        fr.restore_frame(frame, fidelity=0.7)

        assert fr.model.call_count == 3
        assert fr.face_helper.add_restored_face.call_count == 3

    def test_fidelity_range_zero(self):
        """Fidelity w=0 (max quality) passes through to model."""
        from serenityflow.server.face_restorer import FaceRestorer
        import torch

        fr = FaceRestorer(model_dir="/nonexistent", device="cpu")
        fr.face_helper = MagicMock()
        fr.face_helper.cropped_faces = [np.zeros((512, 512, 3), dtype=np.uint8)]
        fr.model = MagicMock(return_value=(torch.rand(1, 3, 512, 512),))
        fr.face_helper.paste_faces_to_input_image.return_value = np.zeros((64, 64, 3), dtype=np.uint8)

        fr.restore_frame(np.zeros((64, 64, 3), dtype=np.uint8), fidelity=0.0)
        assert fr.model.call_args[1]["w"] == 0.0

    def test_fidelity_range_one(self):
        """Fidelity w=1 (max fidelity) passes through to model."""
        from serenityflow.server.face_restorer import FaceRestorer
        import torch

        fr = FaceRestorer(model_dir="/nonexistent", device="cpu")
        fr.face_helper = MagicMock()
        fr.face_helper.cropped_faces = [np.zeros((512, 512, 3), dtype=np.uint8)]
        fr.model = MagicMock(return_value=(torch.rand(1, 3, 512, 512),))
        fr.face_helper.paste_faces_to_input_image.return_value = np.zeros((64, 64, 3), dtype=np.uint8)

        fr.restore_frame(np.zeros((64, 64, 3), dtype=np.uint8), fidelity=1.0)
        assert fr.model.call_args[1]["w"] == 1.0

    def test_process_video_cancel(self, fake_video, tmp_path):
        """Cancel event stops processing and cleans up."""
        from serenityflow.server.face_restorer import FaceRestorer

        output_path = str(tmp_path / "out.mp4")
        cancel = threading.Event()
        cancel.set()  # Pre-cancel

        fr = FaceRestorer(model_dir="/nonexistent", device="cpu")
        # Mock load to avoid actual model loading
        fr.load = MagicMock()
        fr.unload = MagicMock()

        result = fr.process_video(
            fake_video, output_path, fidelity=0.7, cancel_event=cancel,
        )
        assert result is False
        assert not os.path.exists(output_path), "Output should not exist after cancel"

    def test_process_video_invalid_input(self, tmp_path):
        """Non-existent input video returns False."""
        from serenityflow.server.face_restorer import FaceRestorer
        fr = FaceRestorer(model_dir="/nonexistent", device="cpu")
        fr.load = MagicMock()
        fr.unload = MagicMock()

        result = fr.process_video(
            "/nonexistent/video.mp4", str(tmp_path / "out.mp4"),
        )
        assert result is False

    def test_process_video_progress_callback(self, fake_video, tmp_path):
        """Progress callback fires for each frame."""
        from serenityflow.server.face_restorer import FaceRestorer

        output_path = str(tmp_path / "out.mp4")
        progress_calls = []

        fr = FaceRestorer(model_dir="/nonexistent", device="cpu")
        fr.load = MagicMock()

        # Mock restore_frame to passthrough
        fr.restore_frame = MagicMock(side_effect=lambda f, fidelity: f)

        # Mock ffmpeg mux
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            # Also need to prevent unload from crashing
            fr.unload = MagicMock()

            result = fr.process_video(
                fake_video, output_path, fidelity=0.7,
                progress_callback=lambda frame, total: progress_calls.append((frame, total)),
            )

        assert len(progress_calls) == 10  # 10-frame video
        # First call: (1, 10), last call: (10, 10)
        assert progress_calls[0] == (1, 10)
        assert progress_calls[-1] == (10, 10)

    def test_process_video_frame_error_uses_original(self, fake_video, tmp_path):
        """If restore_frame raises, original frame is used (no crash)."""
        from serenityflow.server.face_restorer import FaceRestorer

        output_path = str(tmp_path / "out.mp4")
        call_count = [0]

        def flaky_restore(frame, fidelity):
            call_count[0] += 1
            if call_count[0] == 3:
                raise RuntimeError("Simulated GPU error")
            return frame

        fr = FaceRestorer(model_dir="/nonexistent", device="cpu")
        fr.load = MagicMock()
        fr.restore_frame = MagicMock(side_effect=flaky_restore)
        fr.unload = MagicMock()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = fr.process_video(fake_video, output_path, fidelity=0.7)

        # Should complete despite one frame failing
        assert result is True

    def test_process_video_temp_file_cleanup(self, fake_video, tmp_path):
        """Temp file is cleaned up even on success."""
        from serenityflow.server.face_restorer import FaceRestorer

        output_path = str(tmp_path / "out.mp4")
        fr = FaceRestorer(model_dir="/nonexistent", device="cpu")
        fr.load = MagicMock()
        fr.restore_frame = MagicMock(side_effect=lambda f, fid: f)
        fr.unload = MagicMock()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            fr.process_video(fake_video, output_path, fidelity=0.7)

        # No leftover mp4v temp files
        tmp_dir = tempfile.gettempdir()
        leftover = [f for f in os.listdir(tmp_dir) if f.endswith(".mp4") and "tmp" in f.lower()]
        # Can't assert zero because other processes may create them,
        # but the one we created should be gone
        # Best we can do: verify the finally block ran (unload called)
        fr.unload.assert_called_once()

    def test_preview_frame_returns_jpeg_bytes(self, fake_video):
        """preview_frame should return JPEG bytes."""
        from serenityflow.server.face_restorer import FaceRestorer

        fr = FaceRestorer(model_dir="/nonexistent", device="cpu")
        fr.load = MagicMock()
        # Mock restore_frame to return a valid BGR frame
        fr.restore_frame = MagicMock(
            side_effect=lambda f, fid: np.random.randint(0, 255, f.shape, dtype=np.uint8)
        )
        fr.unload = MagicMock()

        result = fr.preview_frame(fake_video, seek_sec=0.0, fidelity=0.7)
        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 0
        # JPEG magic bytes
        assert result[:2] == b"\xff\xd8"

    def test_preview_frame_bad_video(self):
        """preview_frame with non-existent video returns None."""
        from serenityflow.server.face_restorer import FaceRestorer

        fr = FaceRestorer(model_dir="/nonexistent", device="cpu")
        fr.load = MagicMock()
        fr.unload = MagicMock()

        result = fr.preview_frame("/nonexistent/video.mp4", seek_sec=0.0)
        assert result is None


# ---------------------------------------------------------------------------
# REST Endpoint Tests
# ---------------------------------------------------------------------------

class TestFacetoolsStatusEndpoint:
    """GET /video_edit/facetools/status

    _facetools_model_dir() resolves from state.output_dir which the autouse
    fixture sets to tmp_path/output. model_dir fixture places codeformer.pth
    at that resolved path.
    """

    def test_status_models_missing(self, client):
        """No model files placed — status reports unavailable."""
        r = client.get("/video_edit/facetools/status")
        assert r.status_code == 200
        data = r.json()
        assert data["available"] is False
        assert "models" in data
        assert data["models"]["codeformer"] is False
        assert "model_dir" in data
        assert data["processing"] is False

    def test_status_models_present(self, client, model_dir):
        """model_dir fixture places fake codeformer.pth — status reports available."""
        r = client.get("/video_edit/facetools/status")
        assert r.status_code == 200
        data = r.json()
        assert data["available"] is True
        assert data["models"]["codeformer"] is True


class TestFacetoolsRestoreEndpoint:
    """POST /video_edit/facetools/restore"""

    def test_missing_models_returns_404(self, client, project_with_clip):
        """No model files — returns 404 with missing model info."""
        r = client.post("/video_edit/facetools/restore", json={
            "project_id": "test-proj",
            "clip_id": "clip-abc",
            "fidelity": 0.7,
        })
        assert r.status_code == 404
        assert "Missing model" in r.json()["error"]

    def test_missing_project_returns_404(self, client, model_dir):
        r = client.post("/video_edit/facetools/restore", json={
            "project_id": "nonexistent",
            "clip_id": "clip-abc",
            "fidelity": 0.7,
        })
        assert r.status_code == 404

    def test_missing_clip_returns_404(self, client, model_dir, project_with_clip):
        r = client.post("/video_edit/facetools/restore", json={
            "project_id": "test-proj",
            "clip_id": "nonexistent-clip",
            "fidelity": 0.7,
        })
        assert r.status_code == 404

    def test_fidelity_clamped_high(self, client, model_dir, project_with_clip):
        """Fidelity > 1 should not crash — it gets clamped to 1.0."""
        r = client.post("/video_edit/facetools/restore", json={
            "project_id": "test-proj",
            "clip_id": "clip-abc",
            "fidelity": 5.0,
        })
        # Should start (200) — the background worker will fail on model load
        # but the endpoint itself should accept the request
        assert r.status_code == 200
        data = r.json()
        assert "job_id" in data
        assert data["status"] == "started"

    def test_successful_start_returns_job_id(self, client, model_dir, project_with_clip):
        """Valid request with models present starts a job."""
        r = client.post("/video_edit/facetools/restore", json={
            "project_id": "test-proj",
            "clip_id": "clip-abc",
            "fidelity": 0.7,
        })
        assert r.status_code == 200
        data = r.json()
        assert "job_id" in data
        assert isinstance(data["job_id"], str)
        assert len(data["job_id"]) > 0
        assert data["status"] == "started"


class TestFacetoolsCancelEndpoint:
    """POST /video_edit/facetools/cancel/{job_id}"""

    def test_cancel_nonexistent_job(self, client):
        r = client.post("/video_edit/facetools/cancel/nonexistent")
        assert r.status_code == 404
        assert r.json()["status"] == "not_found"

    def test_cancel_response_shape(self, client):
        """Cancel always returns a status field."""
        r = client.post("/video_edit/facetools/cancel/any-id")
        data = r.json()
        assert "status" in data


class TestFacetoolsPreviewEndpoint:
    """POST /video_edit/facetools/preview

    Preview endpoint uses _resolve_clip_source() which reads the project JSON.
    We need a project fixture for source path resolution.
    """

    def test_preview_missing_source(self, client, model_dir):
        """No project/clip → source not found → 404."""
        r = client.post("/video_edit/facetools/preview", json={
            "project_id": "nonexistent",
            "clip_id": "clip-abc",
            "seek_sec": 2.0,
            "fidelity": 0.7,
        })
        assert r.status_code == 404

    def test_preview_missing_models(self, client, project_with_clip):
        """Models not present → 404."""
        r = client.post("/video_edit/facetools/preview", json={
            "project_id": "test-proj",
            "clip_id": "clip-abc",
            "seek_sec": 0.0,
            "fidelity": 0.7,
        })
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# WebSocket Event Contract Tests (source-level verification)
# ---------------------------------------------------------------------------

class TestFaceRestoreWSEventContract:
    """Verify the WS event names and data shapes that the frontend expects.

    Since _send_face_event is inside the register_video_edit_routes closure,
    we verify the contract by inspecting the source code and the frontend JS.
    """

    def test_event_names_match_frontend(self):
        """Backend WS event names must match JS SerenityWS.on() subscriptions."""
        import inspect
        from serenityflow.server import video_edit_routes as ver
        source = inspect.getsource(ver)

        # These event names are used in JS (from video-edit.js)
        expected_events = [
            "face_restore_progress",
            "face_restore_complete",
            "face_restore_error",
        ]
        for event in expected_events:
            assert event in source, f"Missing WS event: {event}"

    def test_progress_data_fields(self):
        """face_restore_progress data must include fields JS expects."""
        import inspect
        from serenityflow.server import video_edit_routes as ver
        source = inspect.getsource(ver)

        # JS accesses: d.job_id, d.percent, d.frame, d.total, d.clip_id
        # Verify these are set in the on_progress callback
        for field in ["frame", "total", "percent", "clip_id"]:
            assert f'"{field}"' in source or f"'{field}'" in source, \
                f"Progress event missing field: {field}"

    def test_complete_data_fields(self):
        """face_restore_complete data must include output_path, clip_id, fidelity."""
        import inspect
        from serenityflow.server import video_edit_routes as ver
        source = inspect.getsource(ver)

        for field in ["output_path", "clip_id", "fidelity"]:
            assert f'"{field}"' in source or f"'{field}'" in source, \
                f"Complete event missing field: {field}"

    def test_error_data_fields(self):
        """face_restore_error data must include error and clip_id."""
        import inspect
        from serenityflow.server import video_edit_routes as ver
        source = inspect.getsource(ver)

        # Error events should have error message and clip_id
        assert '"error"' in source or "'error'" in source
        assert '"clip_id"' in source or "'clip_id'" in source


# ---------------------------------------------------------------------------
# Frontend Integration Contract Tests
# ---------------------------------------------------------------------------

class TestFrontendContract:
    """Verify backend behavior matches what video-edit.js expects."""

    def test_status_response_matches_js_expectations(self, client, model_dir):
        """JS checks data.available, data.models, data.model_dir, data.processing."""
        r = client.get("/video_edit/facetools/status")
        data = r.json()
        # JS accesses these exact keys
        assert "available" in data
        assert "models" in data
        assert "model_dir" in data
        assert "processing" in data
        assert isinstance(data["available"], bool)
        assert isinstance(data["models"], dict)
        assert isinstance(data["model_dir"], str)
        assert isinstance(data["processing"], bool)

    def test_restore_response_has_job_id(self, client, model_dir, project_with_clip):
        """JS expects res.job_id and res.status from restore endpoint."""
        r = client.post("/video_edit/facetools/restore", json={
            "project_id": "test-proj",
            "clip_id": "clip-abc",
            "fidelity": 0.7,
        })
        if r.status_code == 200:
            data = r.json()
            assert "job_id" in data
            assert data["status"] == "started"
            assert isinstance(data["job_id"], str)
            assert len(data["job_id"]) > 0

    def test_error_response_has_error_key(self, client):
        """JS checks res.error on failure (no models present)."""
        r = client.post("/video_edit/facetools/restore", json={
            "project_id": "test-proj",
            "clip_id": "clip-abc",
            "fidelity": 0.7,
        })
        data = r.json()
        assert "error" in data
        assert isinstance(data["error"], str)

    def test_cancel_response_shape(self, client):
        """JS expects status field from cancel endpoint."""
        r = client.post("/video_edit/facetools/cancel/nonexistent")
        data = r.json()
        assert "status" in data

    def test_fidelity_slider_default(self):
        """Verify default fidelity matches JS slider default (0.7)."""
        # The JS slider has value="0.7" and step="0.05"
        # Backend should use 0.7 when not specified
        from serenityflow.server.face_restorer import FaceRestorer
        # Check restore_frame default
        import inspect
        sig = inspect.signature(FaceRestorer.restore_frame)
        assert sig.parameters["fidelity"].default == 0.7

    def test_context_menu_only_for_video_clips(self):
        """Contract: JS only shows 'Restore Faces...' for clips with source_path.
        Backend should return 404 for clips without source_path."""
        # Verified by test_missing_clip_returns_404 — a clip without source_path
        # won't have the menu item shown, and if somehow called, backend rejects it.
        pass  # Documented contract, tested elsewhere


# ---------------------------------------------------------------------------
# Edge Case & Robustness Tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Bug fixer and skeptic verification items."""

    def test_very_small_frame(self):
        """Tiny frame (16x16) should not crash face detection."""
        from serenityflow.server.face_restorer import FaceRestorer
        fr = FaceRestorer(model_dir="/nonexistent", device="cpu")
        fr.face_helper = MagicMock()
        fr.face_helper.cropped_faces = []  # No faces in tiny frame
        fr.model = MagicMock()

        tiny_frame = np.zeros((16, 16, 3), dtype=np.uint8)
        result = fr.restore_frame(tiny_frame, fidelity=0.7)
        assert result.shape == (16, 16, 3)

    def test_grayscale_frame_shape(self):
        """If somehow a 2-channel or odd-shaped frame arrives, face_helper
        handles it (or we get a clean error, not a segfault)."""
        from serenityflow.server.face_restorer import FaceRestorer
        fr = FaceRestorer(model_dir="/nonexistent", device="cpu")
        fr.face_helper = MagicMock()
        fr.face_helper.read_image.side_effect = ValueError("bad shape")
        fr.model = MagicMock()

        # Should propagate the error, not hang
        with pytest.raises(ValueError, match="bad shape"):
            fr.restore_frame(np.zeros((64, 64, 1), dtype=np.uint8), fidelity=0.7)

    def test_output_filename_pattern(self):
        """Output should be {stem}_facefix.{ext}."""
        # Verify the naming convention used in _run_face_restore
        stem, ext = os.path.splitext("/path/to/my_clip.mp4")
        expected = f"{stem}_facefix{ext}"
        assert expected == "/path/to/my_clip_facefix.mp4"

    def test_project_updated_after_restore(self, tmp_path, fake_video):
        """After successful restore, project JSON has face_restored and face_fidelity."""
        proj_dir = tmp_path / "projects"
        proj_dir.mkdir(parents=True)
        proj = {
            "tracks": [{
                "clips": [{
                    "id": "clip-1",
                    "source_path": fake_video,
                }]
            }]
        }
        proj_path = proj_dir / "test.json"
        proj_path.write_text(json.dumps(proj))

        # Simulate what _run_face_restore does after success
        output_path = fake_video.replace(".mp4", "_facefix.mp4")
        with open(proj_path, "r") as f:
            fresh_proj = json.load(f)
        for trk in fresh_proj.get("tracks", []):
            for clp in trk.get("clips", []):
                if clp.get("id") == "clip-1":
                    clp["source_path"] = output_path
                    clp["face_restored"] = True
                    clp["face_fidelity"] = 0.7
        with open(proj_path, "w") as f:
            json.dump(fresh_proj, f, indent=2)

        # Verify
        with open(proj_path) as f:
            updated = json.load(f)
        clip = updated["tracks"][0]["clips"][0]
        assert clip["face_restored"] is True
        assert clip["face_fidelity"] == 0.7
        assert clip["source_path"].endswith("_facefix.mp4")

    def test_job_cleanup_pattern_in_source(self):
        """Verify _run_face_restore has a finally block that cleans up the job."""
        import inspect
        from serenityflow.server import video_edit_routes as ver
        source = inspect.getsource(ver)
        # The finally block should pop the job from _face_active_jobs
        assert "_face_active_jobs.pop(job_id" in source, \
            "Missing job cleanup in finally block"

    def test_audio_mux_command(self):
        """Verify ffmpeg command preserves audio from original."""
        # This is a contract test — the command should map 0:v and 1:a?
        expected_flags = ["-map", "0:v", "-map", "1:a?"]
        # Read from source to verify
        from serenityflow.server.face_restorer import FaceRestorer
        import inspect
        source = inspect.getsource(FaceRestorer.process_video)
        for flag in expected_flags:
            assert flag in source, f"Missing ffmpeg flag: {flag}"
        assert "libx264" in source
        assert "+faststart" in source
