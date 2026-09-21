# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Dense patch features for a masked region of a frame.

The contract is deliberately tiny so tests can substitute a synthetic extractor:
``extract(image, mask) -> (coords (N,2) float64 in IMAGE pixels, feats (N,D)
L2-normalised)``. The DINOv2 implementation crops to the mask's padded bounding
box, runs one forward pass, and maps patch centres back to full-image pixels —
registration then never needs to know a crop happened.
"""

from __future__ import annotations

import numpy as np

# Patch features come from the mask's padded bbox resized to a fixed square. 518 is
# DINOv2's native grid (37x37 patches of 14 px) — finer localisation than the raw
# tokens is registration's job (RANSAC + least squares), not the extractor's.
_INPUT_SIZE = 518
_PATCH = 14


class DinoPatchExtractor:
    """DINOv2 patch features inside a mask. Loads once, ~50 MB VRAM for -small.

    Pre for :meth:`extract`: ``image`` HxWx3 uint8, ``mask`` HxW bool with >= 1
    true pixel. Post: coords in image pixels; every coord lies inside the dilated
    mask; features L2-normalised float32.
    """

    def __init__(self, model_id: str = "facebook/dinov2-small", device: str = "cuda"):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, dtype=torch.float16).to(device).eval()

    def extract(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8
        assert mask.shape == image.shape[:2] and mask.dtype == bool and mask.any()

        crop, (ox, oy, cw, ch) = _padded_crop(image, mask)
        crop_mask = np.zeros(image.shape[:2], bool)
        crop_mask[oy : oy + ch, ox : ox + cw] = True
        sub_mask = mask[oy : oy + ch, ox : ox + cw]

        import cv2

        resized = cv2.resize(crop, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        inputs = self.processor(
            images=resized, return_tensors="pt", do_resize=False, do_center_crop=False
        ).to(self.device)
        with torch.inference_mode():
            out = self.model(**inputs)
        g = _INPUT_SIZE // _PATCH
        feats = out.last_hidden_state[0, 1 : 1 + g * g].float().cpu().numpy()  # drop CLS

        # Patch centre (grid) -> crop pixel -> image pixel; keep patches the mask covers.
        gy, gx = np.mgrid[0:g, 0:g]
        cx = (gx.ravel() + 0.5) * _PATCH * (cw / _INPUT_SIZE) + ox
        cy = (gy.ravel() + 0.5) * _PATCH * (ch / _INPUT_SIZE) + oy
        m_small = cv2.resize(sub_mask.astype(np.uint8), (g, g), interpolation=cv2.INTER_AREA)
        keep = m_small.ravel() > 0.3
        assert keep.any(), "mask too small for the patch grid — nothing to register"
        coords = np.stack([cx, cy], axis=1)[keep]
        f = feats[keep]
        f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-9
        return coords.astype(np.float64), f.astype(np.float32)


def _padded_crop(image: np.ndarray, mask: np.ndarray, pad: float = 0.15):
    """Bbox of the mask grown by ``pad`` on each side, clamped to the frame.

    The pad keeps object boundary patches (half in, half out) — those carry the
    most orientation signal on weakly textured objects.
    """
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
    x0, y0 = max(0, x0 - px), max(0, y0 - py)
    x1, y1 = min(image.shape[1], x1 + px), min(image.shape[0], y1 + py)
    assert x1 > x0 and y1 > y0
    return image[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)
