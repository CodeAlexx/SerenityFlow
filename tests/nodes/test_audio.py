from __future__ import annotations

from serenityflow.bridge import model_paths
from serenityflow.nodes.audio import load_audio


def test_load_audio_resolves_input_dir(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    audio_path = input_dir / "clip.wav"
    audio_path.write_bytes(b"RIFFtest")

    monkeypatch.setattr(model_paths, "_instance", model_paths.ModelPaths(str(tmp_path)))

    (audio,) = load_audio("clip.wav")

    assert audio["path"] == str(audio_path)
