"""Real-ESRGAN super-resolution with tiled inference.

Uses the ``realesrgan`` + ``basicsr`` packages for the RRDBNet architecture
and ``RealESRGANer`` wrapper with built-in tile processing.

Usage:
    upscaler = ESRGANUpscaler(model_dir="/path/to/models/esrgan")
    upscaler.process_video("input.mp4", "output.mp4", outscale=4)

Model file (place in model_dir):
    - RealESRGAN_x4plus.pth (~67MB)

License: BSD (unrestricted use).
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)

__all__ = ["ESRGANUpscaler"]


class ESRGANUpscaler:
    """Real-ESRGAN wrapper with tiled inference for VRAM safety."""

    MODEL_FILE = "RealESRGAN_x4plus.pth"

    def __init__(self, model_dir: str, device: str = "cuda"):
        self.model_dir = model_dir
        self.device = device
        self.upsampler = None

    def model_present(self) -> bool:
        return os.path.isfile(os.path.join(self.model_dir, self.MODEL_FILE))

    def load(self, tile_size: int = 512):
        if self.upsampler is not None:
            return

        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4,
        )
        model_path = os.path.join(self.model_dir, self.MODEL_FILE)

        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=tile_size,
            tile_pad=10,
            pre_pad=0,
            half=True,
            device=self.device,
        )
        log.info("ESRGANUpscaler loaded (device=%s, tile=%d)", self.device, tile_size)

    def unload(self):
        if self.upsampler is not None:
            del self.upsampler
            self.upsampler = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("ESRGANUpscaler unloaded")

    def upscale_frame(self, frame_bgr: np.ndarray, outscale: int = 4) -> np.ndarray:
        """Upscale a single BGR frame. outscale: 2 or 4."""
        output, _ = self.upsampler.enhance(frame_bgr, outscale=outscale)
        return output

    def process_video(
        self,
        input_path: str,
        output_path: str,
        outscale: int = 4,
        tile_size: int = 512,
        progress_callback=None,
        cancel_event=None,
    ) -> bool:
        """Upscale all frames in a video. Returns True on success."""
        self.load(tile_size=tile_size)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            log.error("Failed to open video: %s", input_path)
            self.unload()
            return False

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_w = width * outscale
        out_h = height * outscale

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (out_w, out_h))

        try:
            for i in range(total):
                if cancel_event and cancel_event.is_set():
                    writer.release()
                    cap.release()
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    return False

                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    upscaled = self.upscale_frame(frame, outscale)
                    # Ensure output matches expected dims (odd input can cause mismatch)
                    uh, uw = upscaled.shape[:2]
                    if uw != out_w or uh != out_h:
                        upscaled = cv2.resize(upscaled, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
                except Exception:
                    log.warning("Frame %d upscale failed, using bicubic fallback", i)
                    upscaled = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

                writer.write(upscaled)

                if progress_callback:
                    progress_callback(i + 1, total)

            writer.release()
            cap.release()

            # Mux audio from original via ffmpeg
            mux_cmd = [
                "ffmpeg", "-y",
                "-i", tmp_path,
                "-i", input_path,
                "-map", "0:v", "-map", "1:a?",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "copy", "-movflags", "+faststart",
                output_path,
            ]
            try:
                result = subprocess.run(mux_cmd, capture_output=True)
            except FileNotFoundError:
                log.error("ffmpeg not found — install ffmpeg to mux audio")
                return False
            if result.returncode != 0:
                log.error("ffmpeg mux failed: %s", result.stderr.decode(errors="replace"))
                return False
            return True

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            self.unload()

    def preview_frame(
        self, video_path: str, seek_sec: float, outscale: int = 4, tile_size: int = 512,
    ) -> bytes | None:
        """Extract one frame, upscale, return JPEG bytes (capped at 1920px wide)."""
        self.load(tile_size=tile_size)
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_MSEC, seek_sec * 1000)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None

            upscaled = self.upscale_frame(frame, outscale)

            # Cap preview width at 1920px to avoid huge JPEG responses
            h, w = upscaled.shape[:2]
            if w > 1920:
                scale = 1920 / w
                upscaled = cv2.resize(
                    upscaled, (1920, int(h * scale)), interpolation=cv2.INTER_AREA,
                )

            _, buf = cv2.imencode(".jpg", upscaled, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return buf.tobytes()
        finally:
            self.unload()
