"""Live preview infrastructure — thread-local WS sender for step callbacks."""
from __future__ import annotations

import io
import threading

import torch

_preview_local = threading.local()


def set_preview_sender(send_progress_fn, send_binary_fn):
    """Set the WS sender functions for the current thread.

    Called from execution.py before sampling starts.
    send_progress_fn(step, total): sends JSON progress event
    send_binary_fn(data: bytes): sends binary preview image
    """
    _preview_local.send_progress = send_progress_fn
    _preview_local.send_binary = send_binary_fn


def clear_preview_sender():
    """Clear the WS sender for the current thread."""
    _preview_local.send_progress = None
    _preview_local.send_binary = None


def _latent_to_preview_jpeg(latent, max_size=512):
    """Convert a latent tensor to a small JPEG preview.

    Uses a cheap approximation: scale latent channels to RGB directly
    (no full VAE decode -- too slow per step).

    Handles both spatial (B,C,H,W) and packed/sequence (B,seq,C) formats,
    as well as video latents (B,C,T,H,W) where we take the middle frame.
    """
    from PIL import Image

    with torch.no_grad():
        lat = latent[0].float().cpu()

        # Handle video latents (C, T, H, W) -- take middle frame
        if lat.ndim == 4:
            mid = lat.shape[1] // 2
            lat = lat[:, mid, :, :]  # -> (C, H, W)

        # Handle packed/sequence format (seq, C) -- reshape to approximate spatial
        if lat.ndim == 2:
            seq, channels = lat.shape
            # Guess spatial dims from sequence length
            side = int(seq ** 0.5)
            if side * side < seq:
                side += 1
            # Pad if needed
            if side * side > seq:
                pad = torch.zeros(side * side - seq, channels, dtype=lat.dtype)
                lat = torch.cat([lat, pad], dim=0)
            lat = lat[:side * side].view(side, side, channels).permute(2, 0, 1)

        # lat is now (C, H, W)
        if lat.shape[0] >= 3:
            rgb = lat[:3]
        else:
            rgb = lat.repeat(3, 1, 1)[:3]

        # Normalize to 0-255
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        rgb = (rgb * 255).clamp(0, 255).byte()

        # Convert to PIL, resize to max_size
        img = Image.fromarray(rgb.permute(1, 2, 0).numpy(), 'RGB')
        # Latent is typically 1/8 resolution -- upscale for preview
        w, h = img.size
        scale = min(max_size / max(w, 1), max_size / max(h, 1))
        if scale > 1:
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

        # Encode as JPEG
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=50)
        return buf.getvalue()


def _make_step_callback(preview_interval=3):
    """Create a sampling step callback that sends WS progress + preview.

    Returns None if no preview sender is registered for this thread.
    """
    send_progress = getattr(_preview_local, 'send_progress', None)
    send_binary = getattr(_preview_local, 'send_binary', None)
    if send_progress is None:
        return None

    def callback(step, total, sigma, denoised):
        # Always send progress
        send_progress(step + 1, total)

        # Send preview every Nth step + final step
        if send_binary is not None and denoised is not None:
            if step % preview_interval == 0 or step == total - 1:
                try:
                    preview_bytes = _latent_to_preview_jpeg(denoised)
                    send_binary(preview_bytes)
                except Exception:
                    pass  # Don't break sampling for preview failures

    return callback
