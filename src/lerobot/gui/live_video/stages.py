"""The picture stages between the tap and the encoder.

Frames are HWC uint8 RGB and overlays HWC uint8 RGBA; every stage returns a
new tensor on the input's device and leaves its inputs alone. The same code
runs on CPU tensors on a host without a GPU and on the GPU on the rig, with
one exception: the resize. On the GPU it goes through torch; on the CPU
through OpenCV in uint8, because torch's route copies every full-size frame
to float32 first, and on a CPU that copy is what limits the frame rate once
several cameras convert at once. The two agree exactly at the whole-number
ratios the profile meets on the rig's cameras.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812


def _even(n: int) -> int:
    return max(2, n - n % 2)


def _resize_tensor(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Area interpolation when shrinking, bilinear otherwise, in float32 and
    rounded back to uint8: the GPU's route, and the reference the CPU's is
    held to."""
    x = img.permute(2, 0, 1).unsqueeze(0).float()
    if h <= int(img.shape[0]) and w <= int(img.shape[1]):
        y = F.interpolate(x, size=(h, w), mode="area")
    else:
        y = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return y.squeeze(0).permute(1, 2, 0).round().clamp_(0, 255).to(torch.uint8)


def _resize(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Area interpolation when shrinking, bilinear otherwise."""
    src_h, src_w = int(img.shape[0]), int(img.shape[1])
    if (src_h, src_w) == (h, w):
        return img
    if img.device.type != "cpu":
        return _resize_tensor(img, h, w)
    interpolation = cv2.INTER_AREA if h <= src_h and w <= src_w else cv2.INTER_LINEAR
    out = cv2.resize(np.ascontiguousarray(img.numpy()), (w, h), interpolation=interpolation)
    return torch.from_numpy(out)


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
