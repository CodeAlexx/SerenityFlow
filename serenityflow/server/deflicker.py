"""Deflicker / temporal consistency for AI-generated video.

Three modes:
- Light: ffmpeg deflicker filter (instant, zero VRAM)
- Medium: histogram matching with EMA (fast, CPU)
- Heavy: optical-flow-guided frame blending (slow, best quality, CPU)

No AI models needed. Uses only ffmpeg and OpenCV.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile

import cv2
import numpy as np

log = logging.getLogger(__name__)

__all__ = ["Deflicker"]


class Deflicker:
    """Deflicker processing pipeline."""

    def process_light(
        self,
        input_path: str,
        output_path: str,
        window: int = 5,
        mode: str = "am",
        progress_callback=None,
        cancel_event=None,
    ) -> bool:
        """ffmpeg deflicker filter. Nearly instant."""
        window = max(3, min(15, window))
        if mode not in ("am", "gm", "hm", "median"):
            mode = "am"

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", f"deflicker=size={window}:mode={mode}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "copy", "-movflags", "+faststart",
            output_path,
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except FileNotFoundError:
            log.error("ffmpeg not found — install ffmpeg")
            return False

        # Poll for cancel while ffmpeg runs
        while proc.poll() is None:
            if cancel_event and cancel_event.is_set():
                proc.kill()
                proc.wait()
                if os.path.exists(output_path):
                    os.unlink(output_path)
                return False
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass

        if progress_callback:
            progress_callback(1, 1)
        if proc.returncode != 0:
            log.error("ffmpeg deflicker failed: %s", proc.stderr.read().decode(errors="replace"))
            return False
        return True

    def process_medium(
        self,
        input_path: str,
        output_path: str,
        strength: float = 0.7,
        ema_decay: float = 0.85,
        progress_callback=None,
        cancel_event=None,
    ) -> bool:
        """Histogram-matching deflicker in LAB L-channel."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            log.error("Failed to open video: %s", input_path)
            return False

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total < 1:
            cap.release()
            return False

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

        ema_hist = None
        cancelled = False

        try:
            for i in range(total):
                if cancel_event and cancel_event.is_set():
                    cancelled = True
                    return False

                ret, frame = cap.read()
                if not ret:
                    break

                # Work in LAB L-channel only (preserves color)
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l_channel = lab[:, :, 0]

                hist = cv2.calcHist([l_channel], [0], None, [256], [0, 256]).flatten()
                hist_sum = hist.sum()
                if hist_sum > 0:
                    hist = hist / hist_sum

                if ema_hist is None:
                    ema_hist = hist.copy()
                else:
                    ema_hist = ema_decay * ema_hist + (1 - ema_decay) * hist
                    ema_sum = ema_hist.sum()
                    if ema_sum > 0:
                        ema_hist = ema_hist / ema_sum

                matched_l = self._histogram_match(l_channel, hist, ema_hist)
                blended_l = np.clip(
                    strength * matched_l.astype(np.float32)
                    + (1 - strength) * l_channel.astype(np.float32),
                    0, 255,
                ).astype(np.uint8)
                lab[:, :, 0] = blended_l
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

                writer.write(result)
                if progress_callback:
                    progress_callback(i + 1, total)

            writer.release()
            cap.release()
            writer = None
            cap = None

            return self._mux_audio(tmp_path, input_path, output_path)
        finally:
            if writer is not None:
                writer.release()
            if cap is not None:
                cap.release()
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if cancelled and os.path.exists(output_path):
                os.unlink(output_path)

    def process_heavy(
        self,
        input_path: str,
        output_path: str,
        blend_alpha: float = 0.15,
        progress_callback=None,
        cancel_event=None,
    ) -> bool:
        """Optical-flow-guided frame blending."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            log.error("Failed to open video: %s", input_path)
            return False

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total < 1:
            cap.release()
            return False

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
        cancelled = False

        try:
            ret, prev_frame = cap.read()
            if not ret:
                return False
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

            # Write first frame as-is
            writer.write(prev_frame)
            if progress_callback:
                progress_callback(1, total)

            ret, curr_frame = cap.read()
            if not ret:
                # Single-frame video
                writer.release()
                cap.release()
                writer = None
                cap = None
                return self._mux_audio(tmp_path, input_path, output_path)

            for i in range(1, total):
                if cancel_event and cancel_event.is_set():
                    cancelled = True
                    return False

                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                ret, next_frame = cap.read()

                try:
                    if ret:
                        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

                        flow_prev = cv2.calcOpticalFlowFarneback(
                            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0,
                        )
                        flow_next = cv2.calcOpticalFlowFarneback(
                            next_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0,
                        )

                        warped_prev = self._warp_flow(prev_frame, flow_prev)
                        warped_next = self._warp_flow(next_frame, flow_next)

                        a = blend_alpha
                        blended = (
                            a * warped_prev.astype(np.float32)
                            + (1 - 2 * a) * curr_frame.astype(np.float32)
                            + a * warped_next.astype(np.float32)
                        ).clip(0, 255).astype(np.uint8)
                        writer.write(blended)
                    else:
                        # Last frame — only blend with prev
                        flow_prev = cv2.calcOpticalFlowFarneback(
                            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0,
                        )
                        warped_prev = self._warp_flow(prev_frame, flow_prev)
                        a = blend_alpha
                        blended = (
                            a * warped_prev.astype(np.float32)
                            + (1 - a) * curr_frame.astype(np.float32)
                        ).clip(0, 255).astype(np.uint8)
                        writer.write(blended)
                except Exception:
                    log.warning("Frame %d optical flow failed, writing unblended", i)
                    writer.write(curr_frame)

                if progress_callback:
                    progress_callback(i + 1, total)

                prev_frame = curr_frame
                prev_gray = curr_gray
                if ret:
                    curr_frame = next_frame
                else:
                    break

            writer.release()
            cap.release()
            writer = None
            cap = None
            return self._mux_audio(tmp_path, input_path, output_path)
        finally:
            if writer is not None:
                writer.release()
            if cap is not None:
                cap.release()
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if cancelled and os.path.exists(output_path):
                os.unlink(output_path)

    @staticmethod
    def _warp_flow(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp image using optical flow field."""
        h, w = flow.shape[:2]
        map_x = np.arange(w, dtype=np.float32)[np.newaxis, :] + flow[:, :, 0]
        map_y = np.arange(h, dtype=np.float32)[:, np.newaxis] + flow[:, :, 1]
        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def _histogram_match(
        source: np.ndarray, source_hist: np.ndarray, target_hist: np.ndarray,
    ) -> np.ndarray:
        """Match source image histogram to target histogram via CDF mapping."""
        src_cdf = np.cumsum(source_hist)
        tgt_cdf = np.cumsum(target_hist)
        lut = np.zeros(256, dtype=np.uint8)
        for s_val in range(256):
            lut[s_val] = np.argmin(np.abs(tgt_cdf - src_cdf[s_val]))
        return lut[source]

    @staticmethod
    def _mux_audio(video_path: str, audio_source: str, output_path: str) -> bool:
        """Mux audio from audio_source into video_path, write to output_path."""
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path, "-i", audio_source,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "copy", "-movflags", "+faststart",
            output_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True)
        except FileNotFoundError:
            log.error("ffmpeg not found — install ffmpeg")
            return False
        if result.returncode != 0:
            log.error("ffmpeg mux failed: %s", result.stderr.decode(errors="replace"))
            return False
        return True
