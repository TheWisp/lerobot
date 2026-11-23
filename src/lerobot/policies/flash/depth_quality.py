"""Pixel-level validity for metric observations from aligned depth frames."""

from __future__ import annotations

import cv2
import numpy as np


def depth_support_mask(depth: np.ndarray, *, radius_px: int) -> np.ndarray:
    """Require measured depth throughout the pixel-location uncertainty window.

    This does not fill holes or claim that a positive reading is accurate. A
    tracked RGB feature can survive a failed metric sample and be measured on
    a later frame. The radius describes uncertainty in the projected pixel,
    not a metric tolerance or an erosion of the object's segmentation mask.
    """
    if depth.ndim != 2 or isinstance(radius_px, bool) or not isinstance(radius_px, int) or radius_px < 0:
        raise ValueError("depth support needs an HxW image and a nonnegative pixel radius")
    valid = (np.isfinite(depth) & (depth > 0)).astype(np.uint8)
    if radius_px == 0:
        return valid.astype(bool)
    width = 2 * radius_px + 1
    return cv2.erode(
        valid,
        np.ones((width, width), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
