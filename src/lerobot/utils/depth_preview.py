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
needs the same rendering, so the rule lives here once: valid pixels are
stretched between their 2nd and 98th percentiles, near is warm and far is
cool, and ``0``, which means "no measurement" throughout LeRobot, is black.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def depth_to_rgb_preview(depth: NDArray, subsample: int = 8) -> NDArray[np.uint8]:
    """Render ``(H, W)`` or ``(H, W, 1)`` depth as ``(H, W, 3)`` uint8 RGB.

    The stretch is estimated on every ``subsample``-th row and column so the
    call is cheap enough for a control loop; a frame without a single valid
    pixel comes back black.
    """
    d = np.asarray(depth)
    if d.ndim == 3:
        d = d[..., 0]
    valid = d > 0
    sample = d[::subsample, ::subsample]
    sample = sample[sample > 0]
    if sample.size == 0:
        return np.zeros((*d.shape, 3), np.uint8)
    lo, hi = np.percentile(sample, [2, 98])
    if hi <= lo:
        hi = lo + 1
    far = np.clip((d.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    # Turbo runs blue -> red, so feed it nearness to get warm-near / cool-far.
    codes = ((1.0 - far) * 255.0).astype(np.uint8)
    rgb = cv2.applyColorMap(codes, cv2.COLORMAP_TURBO)[..., ::-1]
    rgb[~valid] = 0
    return np.ascontiguousarray(rgb)
