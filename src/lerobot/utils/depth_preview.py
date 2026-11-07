# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Turn a depth map into something a person can look at.

Depth arrives as one channel of physical units (uint16 millimetres from a
camera, float millimetres or metres from a dataset). Every display path
needs the same rendering, so the rule lives here once: the palette runs
warm at the near end and cool at the far end, and ``0``, which means "no
measurement" throughout LeRobot, is black.

Which distances the ends of the palette stand for is a display choice, never
a property of the data. Callers either pass the range, or let the frame's own
1st and 99th percentiles pick it.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

#: The percentiles of the valid pixels that bound an automatic range: the
#: same 1st and 99th the dataset statistics record as q01 and q99.
AUTO_PERCENTILES = (1.0, 99.0)


def auto_range(depth: NDArray, subsample: int = 8, invalid_below: float = 0.0) -> tuple[float, float] | None:
    """The 1st and 99th percentile of the valid pixels, in the map's own unit.

    Estimated on every ``subsample``-th row and column so the call is cheap
    enough for a control loop. ``None`` when no pixel is valid. A pixel is
    valid above ``invalid_below``: ``0`` straight from a sensor, or the
    quantizer's floor for depth read back from a dataset, since the codec
    path maps "no measurement" onto that floor.
    """
    d = np.asarray(depth)
    if d.ndim == 3:
        d = d[..., 0]
    sample = d[::subsample, ::subsample]
    sample = sample[sample > invalid_below]
    if sample.size == 0:
        return None
    lo, hi = np.percentile(sample, AUTO_PERCENTILES)
    return float(lo), float(hi)


def nearness(depth: NDArray, *, lo: float, hi: float) -> NDArray[np.float32]:
    """Where each pixel sits between the far end (``0``) and the near end
    (``1``) of a range. Values outside the range saturate. Invalid pixels get
    a value too; callers mask them.
    """
    d = np.asarray(depth, dtype=np.float32)
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1.0
    return (1.0 - np.clip((d - lo) / (hi - lo), 0.0, 1.0)).astype(np.float32)


def depth_to_rgb_preview(
    depth: NDArray,
    subsample: int = 8,
    *,
    lo: float | None = None,
    hi: float | None = None,
    invalid_below: float = 0.0,
) -> NDArray[np.uint8]:
    """Render ``(H, W)`` or ``(H, W, 1)`` depth as ``(H, W, 3)`` uint8 RGB.

    ``lo`` and ``hi`` are the distances drawn at the warm and cool ends, in
    the map's own unit; leave them out for the frame's :func:`auto_range`.
    Values outside the range saturate. Pixels
    at or below ``invalid_below`` carry no measurement and are black, as is a
    frame without a single valid pixel.
    """
    d = np.asarray(depth)
    if d.ndim == 3:
        d = d[..., 0]
    valid = d > invalid_below
    if lo is None or hi is None:
        found = auto_range(d, subsample, invalid_below)
        if found is None:
            return np.zeros((*d.shape, 3), np.uint8)
        lo, hi = found
    # Turbo runs blue -> red, so feed it nearness to get warm-near / cool-far.
    codes = (nearness(d, lo=lo, hi=hi) * 255.0).astype(np.uint8)
    rgb = cv2.applyColorMap(codes, cv2.COLORMAP_TURBO)[..., ::-1]
    rgb[~valid] = 0
    return np.ascontiguousarray(rgb)
