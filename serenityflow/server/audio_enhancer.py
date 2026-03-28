"""Audio enhancement pipeline for video clips.

Two modes:
- ffmpeg presets: loudnorm, clean speech, podcast, music, de-hum (instant, no deps)
- DeepFilterNet: AI speech enhancement (optional pip install, CPU, ~5s per 10s audio)

All modes keep the video stream untouched (c:v copy).
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile

log = logging.getLogger(__name__)

__all__ = ["AudioEnhancer", "FFMPEG_PRESETS"]


FFMPEG_PRESETS = {
    "normalize": {
        "name": "Normalize",
        "description": "Standard broadcast loudness",
        "filter": "loudnorm=I=-16:TP=-1.5:LRA=11",
    },
    "clean_speech": {
        "name": "Clean Speech",
        "description": "Remove noise, rumble, hiss + normalize",
        "filter": "highpass=f=80,lowpass=f=8000,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",
    },
    "podcast": {
        "name": "Podcast",
        "description": "Compress + normalize for consistent voice",
        "filter": "highpass=f=100,acompressor=threshold=-20dB:ratio=4:attack=5:release=50,loudnorm=I=-16:TP=-1.5:LRA=11",
    },
    "music": {
        "name": "Music",
        "description": "Gentle normalization for music",
        "filter": "loudnorm=I=-14:TP=-1:LRA=7",
    },
    "dehum": {
        "name": "De-hum",
        "description": "Remove 50Hz electrical hum + harmonics",
        "filter": "bandreject=f=50:w=2,bandreject=f=100:w=2,bandreject=f=150:w=2,bandreject=f=200:w=2,loudnorm=I=-16:TP=-1.5:LRA=11",
    },
}


def deepfilter_available() -> bool:
    """Check if DeepFilterNet is installed."""
    try:
        import df.enhance  # noqa: F401
        return True
    except ImportError:
        return False


def _has_audio_stream(path: str) -> bool:
    """Probe whether a file has at least one audio stream."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10,
        )
        return "audio" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class AudioEnhancer:
    """Audio enhancement pipeline."""

    def process_ffmpeg(
        self,
        input_path: str,
        output_path: str,
        preset: str = "normalize",
        cancel_event=None,
    ) -> tuple[bool, str | None]:
        """Apply ffmpeg audio filter preset. Video stream copied untouched."""
        if preset not in FFMPEG_PRESETS:
            return False, f"Unknown preset: {preset}"

        if not _has_audio_stream(input_path):
            return False, "No audio stream found in clip"

        af = FFMPEG_PRESETS[preset]["filter"]

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-c:v", "copy",
            "-af", af,
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            output_path,
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except FileNotFoundError:
            return False, "ffmpeg not found — install ffmpeg"

        while proc.poll() is None:
            if cancel_event and cancel_event.is_set():
                proc.kill()
                proc.wait()
                if os.path.exists(output_path):
                    os.unlink(output_path)
                return False, "Cancelled"
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass

        if proc.returncode != 0:
            stderr = proc.stderr.read().decode(errors="replace")
            log.error("ffmpeg audio enhance failed: %s", stderr[-500:])
            return False, stderr[-200:] if stderr else "ffmpeg error"
        return True, None

    def process_deepfilter(
        self,
        input_path: str,
        output_path: str,
        progress_callback=None,
        cancel_event=None,
    ) -> tuple[bool, str | None]:
        """AI speech enhancement using DeepFilterNet."""
        try:
            from df.enhance import enhance, init_df, load_audio, save_audio
        except ImportError:
            return False, "DeepFilterNet not installed. Run: pip install deepfilternet"

        tmp_audio_in = None
        tmp_audio_out = None

        try:
            # Create temp files
            fd_in, tmp_audio_in = tempfile.mkstemp(suffix=".wav")
            os.close(fd_in)
            fd_out, tmp_audio_out = tempfile.mkstemp(suffix=".wav")
            os.close(fd_out)

            # Extract audio from video
            extract_cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1",
                tmp_audio_in,
            ]
            try:
                result = subprocess.run(extract_cmd, capture_output=True)
            except FileNotFoundError:
                return False, "ffmpeg not found — install ffmpeg"

            if result.returncode != 0 or not os.path.exists(tmp_audio_in) or os.path.getsize(tmp_audio_in) < 100:
                return False, "No audio stream found in clip"

            if cancel_event and cancel_event.is_set():
                return False, "Cancelled"

            # Load and enhance
            model, df_state, _ = init_df()
            audio, _ = load_audio(tmp_audio_in, sr=df_state.sr())

            if progress_callback:
                progress_callback(1, 3)

            if cancel_event and cancel_event.is_set():
                return False, "Cancelled"

            enhanced = enhance(model, df_state, audio)

            if progress_callback:
                progress_callback(2, 3)

            save_audio(tmp_audio_out, enhanced, sr=df_state.sr())

            if cancel_event and cancel_event.is_set():
                return False, "Cancelled"

            # Mux enhanced audio back with original video
            # DeepFilter outputs mono — duplicate to stereo to preserve
            # original channel layout (mono→mono is a no-op for -ac 2)
            mux_cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-i", tmp_audio_out,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy",
                "-ac", "2",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                output_path,
            ]
            try:
                result = subprocess.run(mux_cmd, capture_output=True)
            except FileNotFoundError:
                return False, "ffmpeg not found — install ffmpeg"

            if progress_callback:
                progress_callback(3, 3)

            if result.returncode != 0:
                return False, "Audio mux failed"
            return True, None

        finally:
            for f in [tmp_audio_in, tmp_audio_out]:
                if f and os.path.exists(f):
                    os.unlink(f)

    def preview_ffmpeg(
        self,
        input_path: str,
        preset: str = "normalize",
        duration: float = 5.0,
    ) -> bytes | None:
        """Apply filter to first N seconds, return MP3 bytes."""
        if preset not in FFMPEG_PRESETS:
            return None

        af = FFMPEG_PRESETS[preset]["filter"]

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-t", str(duration),
            "-vn",
            "-af", af,
            "-c:a", "libmp3lame", "-b:a", "128k",
            "-f", "mp3",
            "pipe:1",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

        if result.returncode != 0 or not result.stdout:
            return None
        return result.stdout
