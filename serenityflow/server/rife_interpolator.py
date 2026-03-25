"""RIFE (Real-Time Intermediate Flow Estimation) frame interpolation.

Implements Practical-RIFE v4.25 IFNet architecture for AI-powered frame
interpolation. Loads flownet.pkl weights. No external RIFE dependencies.

Usage:
    interp = RIFEInterpolator(model_dir="/path/to/models/rife/v4.25")
    interp.interpolate_video("input.mp4", "output.mp4", multiplier=2)
"""
from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import subprocess
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

__all__ = ["RIFEInterpolator"]


# ---------------------------------------------------------------------------
# Warping utility
# ---------------------------------------------------------------------------

def _warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp *img* using pixel-displacement *flow* (B,2,H,W)."""
    B, _, H, W = img.shape
    xx = torch.linspace(-1, 1, W, device=img.device).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1, 1, H, device=img.device).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], dim=1)
    flow_norm = torch.cat([
        flow[:, 0:1] / ((W - 1) / 2),
        flow[:, 1:2] / ((H - 1) / 2),
    ], dim=1)
    grid = (grid + flow_norm).permute(0, 2, 3, 1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


# ---------------------------------------------------------------------------
# IFNet building blocks
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """Conv(stride=2) → PReLU → Conv(stride=1) → PReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv0 = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.act0 = nn.PReLU(out_ch)
        self.conv1 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.act1 = nn.PReLU(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act0(self.conv0(x))
        x = self.act1(self.conv1(x))
        return x


class _IFBlock(nn.Module):
    """One level of the coarse-to-fine flow decoder."""

    def __init__(self, in_ch: int, mid_ch: int = 64):
        super().__init__()
        self.conv0 = nn.Conv2d(in_ch, mid_ch, 3, padding=1)
        self.act0 = nn.PReLU(mid_ch)
        self.conv1 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1)
        self.act1 = nn.PReLU(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1)
        self.act2 = nn.PReLU(mid_ch)
        # Output: 4 flow channels + 1 mask channel = 5
        self.conv3 = nn.Conv2d(mid_ch, 5, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act0(self.conv0(x))
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return self.conv3(x)


class IFNet(nn.Module):
    """Intermediate Flow Network — predicts one interpolated frame.

    Input:  img0 (B,3,H,W), img1 (B,3,H,W), timestep (B,1,1,1)
    Output: interpolated frame (B,3,H,W)
    """

    def __init__(self):
        super().__init__()
        # Encoder: img0 + img1 + timestep = 3+3+1 = 7 input channels
        self.encoder = nn.ModuleList([
            _ConvBlock(7, 64),
            _ConvBlock(64, 128),
            _ConvBlock(128, 256),
        ])
        # Decoder: one IFBlock per pyramid level (coarse → fine)
        # Input to each block: features + 4 flow channels from coarser level
        self.decoder = nn.ModuleList([
            _IFBlock(256, 128),       # level 2 (coarsest)
            _IFBlock(128 + 5, 96),    # level 1 + prev output
            _IFBlock(64 + 5, 64),     # level 0 (finest)
        ])

    def forward(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = img0.shape

        # Expand timestep to spatial dims
        t_map = timestep.expand(B, 1, H, W)
        x = torch.cat([img0, img1, t_map], dim=1)  # B, 7, H, W

        # Encode — build feature pyramid
        feats = []
        for enc in self.encoder:
            x = enc(x)
            feats.append(x)

        # Decode — coarse to fine with residual refinement
        flow_mask = None
        for i, dec in enumerate(self.decoder):
            feat = feats[len(feats) - 1 - i]
            if flow_mask is not None:
                # Upsample previous flow+mask and concat with features
                flow_mask_up = F.interpolate(
                    flow_mask, scale_factor=2.0, mode="bilinear", align_corners=False
                )
                # Scale flow magnitudes by 2 (spatial upsampling)
                flow_mask_up[:, :4] *= 2.0
                feat = torch.cat([feat, flow_mask_up], dim=1)
                # Residual: each level refines the coarser prediction
                flow_mask = dec(feat) + flow_mask_up
            else:
                flow_mask = dec(feat)

        # Final flow_mask is at input resolution / 2 — upsample to full res
        flow_mask = F.interpolate(
            flow_mask, size=(H, W), mode="bilinear", align_corners=False
        )
        flow_mask[:, :4] *= 2.0  # scale flow to full resolution

        flow_0to1 = flow_mask[:, :2]   # flow from img0 → img1
        flow_1to0 = flow_mask[:, 2:4]  # flow from img1 → img0
        mask = torch.sigmoid(flow_mask[:, 4:5])

        # Compute flows at timestep t
        flow_t0 = -(1 - timestep) * timestep * flow_0to1 + timestep * timestep * flow_1to0
        flow_t1 = (1 - timestep) * (1 - timestep) * flow_0to1 - timestep * (1 - timestep) * flow_1to0

        # Warp both frames to timestep t
        warped0 = _warp(img0, flow_t0)
        warped1 = _warp(img1, flow_t1)

        # Blend using learned mask
        result = mask * warped0 + (1 - mask) * warped1
        return result


# ---------------------------------------------------------------------------
# High-level interpolator
# ---------------------------------------------------------------------------

class RIFEInterpolator:
    """Manages RIFE model loading, inference, and full video interpolation."""

    def __init__(self, model_dir: str, device: str = "cuda", fp16: bool = True):
        self.model_dir = model_dir
        self.device = device
        self.fp16 = fp16
        self.model: IFNet | None = None

    # -- Model lifecycle --

    def load_model(self):
        if self.model is not None:
            return
        weights = os.path.join(self.model_dir, "flownet.pkl")
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"RIFE weights not found: {weights}")
        self.model = IFNet()
        state_dict = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        if self.fp16:
            self.model.half()
        self.model.eval()
        log.info("RIFE model loaded from %s (fp16=%s)", weights, self.fp16)

    def unload_model(self):
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info("RIFE model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    # -- Single-pair inference --

    @torch.no_grad()
    def interpolate_pair(
        self, img0: torch.Tensor, img1: torch.Tensor, t: float = 0.5,
    ) -> torch.Tensor:
        """Interpolate between two frames.

        Args:
            img0, img1: [1, 3, H, W] float32 tensors in [0, 1].
            t: timestep in (0, 1). 0.5 = midpoint.

        Returns:
            [1, 3, H, W] float32 tensor in [0, 1].
        """
        _, _, h, w = img0.shape
        # Pad to multiple of 64 for pyramid alignment
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        pad = (0, pw - w, 0, ph - h)
        img0_p = F.pad(img0, pad)
        img1_p = F.pad(img1, pad)

        if self.fp16:
            img0_p = img0_p.half()
            img1_p = img1_p.half()

        timestep = torch.tensor([t], device=self.device).reshape(1, 1, 1, 1)
        if self.fp16:
            timestep = timestep.half()

        result = self.model(img0_p, img1_p, timestep)
        return result[:, :, :h, :w].float().clamp(0, 1)

    # -- Frame I/O helpers --

    def _load_frame(self, path: str) -> torch.Tensor:
        """Load PNG → [1, 3, H, W] float32 on device."""
        import numpy as np
        from PIL import Image

        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

    @staticmethod
    def _save_frame(tensor: torch.Tensor, path: str):
        """Save [1, 3, H, W] float32 tensor as PNG."""
        import numpy as np
        from PIL import Image

        arr = (tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(path)

    # -- Full video pipeline --

    def interpolate_video(
        self,
        input_path: str,
        output_path: str,
        multiplier: int = 2,
        progress_callback=None,
        cancel_event=None,
    ) -> bool:
        """Interpolate a video by *multiplier* (2 or 4).

        Pipeline: ffmpeg decode → RIFE per-pair → ffmpeg encode.
        Returns True on success.
        """
        if multiplier not in (2, 4):
            raise ValueError("multiplier must be 2 or 4")

        self.load_model()

        tmp_input = tempfile.mkdtemp(prefix="rife_in_")
        tmp_output = tempfile.mkdtemp(prefix="rife_out_")

        try:
            # --- Probe source (all streams for audio detection) ---
            probe_cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", input_path,
            ]
            try:
                probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
                probe_data = json.loads(probe.stdout)
                streams = probe_data.get("streams", [])
                if not streams:
                    log.error("ffprobe returned no streams for %s", input_path)
                    return False
            except (json.JSONDecodeError, KeyError, subprocess.TimeoutExpired) as exc:
                log.error("Failed to probe source video %s: %s", input_path, exc)
                return False

            # Find video stream for FPS
            stream = None
            for s in streams:
                if s.get("codec_type") == "video":
                    stream = s
                    break
            if not stream:
                log.error("No video stream found in %s", input_path)
                return False

            fps_parts = stream["r_frame_rate"].split("/")
            src_fps = (
                float(fps_parts[0]) / float(fps_parts[1])
                if len(fps_parts) == 2
                else float(fps_parts[0])
            )
            target_fps = src_fps * multiplier

            # --- Extract frames ---
            extract_cmd = [
                "ffmpeg", "-y", "-i", input_path,
                os.path.join(tmp_input, "%08d.png"),
            ]
            subprocess.run(extract_cmd, capture_output=True, timeout=600)

            frames = sorted(glob.glob(os.path.join(tmp_input, "*.png")))
            total_frames = len(frames)
            if total_frames < 2:
                log.warning("Source has < 2 frames, cannot interpolate")
                return False

            # --- Interpolate ---
            output_idx = 0
            for i in range(total_frames):
                if cancel_event and cancel_event.is_set():
                    log.info("RIFE interpolation cancelled at frame %d/%d", i, total_frames)
                    return False

                # Write original frame
                dst = os.path.join(tmp_output, f"{output_idx:08d}.png")
                shutil.copy2(frames[i], dst)
                output_idx += 1

                # Generate intermediate frames to next original
                if i < total_frames - 1:
                    img0 = self._load_frame(frames[i])
                    img1 = self._load_frame(frames[i + 1])

                    for j in range(1, multiplier):
                        t = j / multiplier
                        mid = self.interpolate_pair(img0, img1, t)
                        self._save_frame(mid, os.path.join(tmp_output, f"{output_idx:08d}.png"))
                        output_idx += 1

                if progress_callback:
                    progress_callback(i + 1, total_frames)

            # --- Reassemble with audio ---
            has_audio = any(s.get("codec_type") == "audio" for s in streams)

            encode_cmd = [
                "ffmpeg", "-y",
                "-framerate", str(target_fps),
                "-i", os.path.join(tmp_output, "%08d.png"),
            ]
            if has_audio:
                encode_cmd += ["-i", input_path, "-map", "0:v", "-map", "1:a", "-c:a", "copy"]
            encode_cmd += [
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                output_path,
            ]
            result = subprocess.run(encode_cmd, capture_output=True, timeout=600)
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace")[-500:]
                log.error("ffmpeg encode failed: %s", stderr)
                return False

            log.info(
                "RIFE %dx complete: %d → %d frames, %s → %s fps",
                multiplier, total_frames, output_idx, src_fps, target_fps,
            )
            return True

        finally:
            shutil.rmtree(tmp_input, ignore_errors=True)
            shutil.rmtree(tmp_output, ignore_errors=True)
            self.unload_model()
