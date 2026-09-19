"""The picture stages between the tap and the encoder.

Written on tensors without a device in them: the same code runs on CPU
tensors in CI and on the GPU on the host. Frames are HWC uint8 RGB and
overlays HWC uint8 RGBA; every stage returns a new tensor on the input's
device and leaves its inputs alone.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


def _even(n: int) -> int:
    return max(2, n - n % 2)


def _resize(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Area interpolation when shrinking, bilinear otherwise, rounded back to uint8."""
    src_h, src_w = int(img.shape[0]), int(img.shape[1])
    if (src_h, src_w) == (h, w):
        return img
    x = img.permute(2, 0, 1).unsqueeze(0).float()
    if h <= src_h and w <= src_w:
        y = F.interpolate(x, size=(h, w), mode="area")
    else:
        y = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return y.squeeze(0).permute(1, 2, 0).round().clamp_(0, 255).to(torch.uint8)


def resize_to_width(frame: torch.Tensor, width: int) -> torch.Tensor:
    """Scale a frame down to ``width``, keeping its aspect ratio and landing
    on even sizes for the encoder. A frame no wider than that is returned as
    it is: the profile never upscales.
    """
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == torch.uint8, frame.shape
    h, w = int(frame.shape[0]), int(frame.shape[1])
    if w <= width:
        return frame
    return _resize(frame, _even(round(h * width / w)), _even(width))


def blend_overlay(frame: torch.Tensor, overlay: torch.Tensor) -> torch.Tensor:
    """Draw an RGBA overlay onto a frame by its alpha.

    The overlay is brought to the frame's size first, so the blend happens at
    the profile's size rather than the source's.
    """
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == torch.uint8, frame.shape
    assert overlay.ndim == 3 and overlay.shape[2] == 4 and overlay.dtype == torch.uint8, overlay.shape
    assert overlay.device == frame.device, (overlay.device, frame.device)
    ov = _resize(overlay, int(frame.shape[0]), int(frame.shape[1]))
    a = ov[..., 3:4].float() / 255.0
    out = frame.float() * (1.0 - a) + ov[..., :3].float() * a
    return out.round().clamp_(0, 255).to(torch.uint8)
