"""CodeFormer face restoration with YOLOv5 face detection.

Uses basicsr + facexlib packages for the CodeFormer model class and
face detection/alignment/paste-back utilities.

Usage:
    restorer = FaceRestorer(model_dir="/path/to/models/facetools")
    restorer.process_video("input.mp4", "output.mp4", fidelity=0.7)

Model files (place in model_dir):
    - codeformer.pth (~376MB)
    - yolov5l-face.pth (~178MB)
    - parsing_parsenet.pth (~85MB)

License: CodeFormer uses S-Lab License 1.0 (non-commercial).
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

__all__ = ["FaceRestorer"]


class FaceRestorer:
    """CodeFormer face restoration wrapper."""

    # Only codeformer.pth must be placed manually; facexlib auto-downloads
    # detection (RetinaFace) and parsing (ParseNet) models on first use.
    REQUIRED_MODELS = {
        "codeformer": "codeformer.pth",
    }

    def __init__(self, model_dir: str, device: str = "cuda"):
        self.model_dir = model_dir
        self.device = device
        self.model = None
        self.face_helper = None

    def check_models(self) -> dict[str, bool]:
        """Return dict of model_name -> exists."""
        return {
            name: os.path.isfile(os.path.join(self.model_dir, fname))
            for name, fname in self.REQUIRED_MODELS.items()
        }

    def all_models_present(self) -> bool:
        return all(self.check_models().values())

    def load(self):
        if self.model is not None:
            return

        # Patch basicsr compatibility with torchvision >= 0.18
        # (functional_tensor was removed, moved to functional)
        import importlib
        import torchvision.transforms.functional as _tvf
        if not importlib.util.find_spec("torchvision.transforms.functional_tensor"):
            import sys
            sys.modules["torchvision.transforms.functional_tensor"] = _tvf

        from codeformer.basicsr.utils.registry import ARCH_REGISTRY
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper

        # Load CodeFormer (dim_embd=512 matches official checkpoint)
        self.model = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(self.device)
        ckpt_path = os.path.join(self.model_dir, "codeformer.pth")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["params_ema"])
        self.model.eval()

        # Face helper (detection + alignment + paste-back)
        # facexlib supports retinaface_resnet50 and retinaface_mobile0.25
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            device=self.device,
            model_rootpath=self.model_dir,
        )
        log.info("FaceRestorer loaded (device=%s)", self.device)

    def unload(self):
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
        if self.face_helper is not None:
            del self.face_helper
            self.face_helper = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("FaceRestorer unloaded")

    @torch.no_grad()
    def restore_frame(self, frame_bgr: np.ndarray, fidelity: float = 0.7) -> np.ndarray:
        """Restore faces in a single BGR frame. Returns BGR with restored faces."""
        self.face_helper.clean_all()
        self.face_helper.read_image(frame_bgr)
        self.face_helper.get_face_landmarks_5(only_center_face=False)
        self.face_helper.align_warp_face()

        if len(self.face_helper.cropped_faces) == 0:
            return frame_bgr

        for cropped_face in self.face_helper.cropped_faces:
            face_t = torch.from_numpy(cropped_face.astype(np.float32) / 255.0)
            face_t = face_t.permute(2, 0, 1).unsqueeze(0).to(self.device)

            output = self.model(face_t, w=fidelity, adain=True)[0]

            restored = output.squeeze(0).permute(1, 2, 0).clamp(0, 1)
            restored = (restored.cpu().numpy() * 255).astype(np.uint8)
            self.face_helper.add_restored_face(restored)

        self.face_helper.get_inverse_affine(None)
        result = self.face_helper.paste_faces_to_input_image(
            upsample_img=frame_bgr,
        )
        return result

    def process_video(
        self,
        input_path: str,
        output_path: str,
        fidelity: float = 0.7,
        progress_callback=None,
        cancel_event=None,
    ) -> bool:
        """Process all frames in a video. Returns True on success."""
        self.load()

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            log.error("Failed to open video: %s", input_path)
            return False

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

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
                    restored = self.restore_frame(frame, fidelity)
                except Exception:
                    log.warning("Frame %d restore failed, using original", i)
                    restored = frame

                writer.write(restored)

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
            result = subprocess.run(mux_cmd, capture_output=True)
            if result.returncode != 0:
                log.error("ffmpeg mux failed: %s", result.stderr.decode(errors="replace"))
                return False
            return True

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            self.unload()

    def preview_frame(self, video_path: str, seek_sec: float, fidelity: float = 0.7) -> bytes | None:
        """Extract one frame, restore faces, return JPEG bytes."""
        self.load()
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_MSEC, seek_sec * 1000)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None

            restored = self.restore_frame(frame, fidelity)
            _, buf = cv2.imencode(".jpg", restored, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return buf.tobytes()
        finally:
            self.unload()
