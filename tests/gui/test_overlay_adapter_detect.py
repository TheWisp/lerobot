# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Regression test for the SAM3 adapter's detector seed selection.

Data editing must protect EVERY instance of a concept (e.g. both robot arms),
not just the largest — SAM3 returns them as separate instances and the old code
kept only the biggest. This drives ``_detect`` with a mocked detector (no GPU /
gated weights) and checks the single-vs-union selection directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from lerobot.overlays.adapters import Sam3TrackByDetectionAdapter


def _adapter_with_masks(masks):
    """A bare adapter (no model load) whose detector returns ``masks``."""
    a = object.__new__(Sam3TrackByDetectionAdapter)
    a._torch = torch
    a._Image = Image
    a.device = "cpu"
    a._det_threshold = 0.5
    a._seed_multi = False
    det_proc = MagicMock()
    det_proc.return_value.to.return_value = {}  # processor(...).to(device) -> **inp
    det_proc.post_process_instance_segmentation.return_value = [{"masks": masks}]
    a.det_proc = det_proc
    a.det = MagicMock(return_value=MagicMock())
    return a


def _two_arms(h=20, w=40):
    a1 = np.zeros((h, w), dtype=bool)
    a1[2:14, 2:12] = True  # bigger "arm" (120 px)
    a2 = np.zeros((h, w), dtype=bool)
    a2[4:12, 28:36] = True  # smaller "arm" (64 px)
    return a1, a2


def test_detect_single_instance_keeps_largest():
    a1, a2 = _two_arms()
    ad = _adapter_with_masks([a1, a2])
    ad._seed_multi = False
    out = ad._detect(np.zeros((20, 40, 3), np.uint8), "robot arm", 20, 40)
    assert out is not None
    assert (out == a1).all()  # only the largest instance
    assert not (out & a2).any()  # the second arm is dropped (debug-viz lock)


def test_detect_multi_instance_unions_all():
    a1, a2 = _two_arms()
    ad = _adapter_with_masks([a1, a2])
    ad._seed_multi = True
    out = ad._detect(np.zeros((20, 40, 3), np.uint8), "robot arm", 20, 40)
    assert out is not None
    assert (out == (a1 | a2)).all()  # BOTH arms protected
    assert int(out.sum()) == int(a1.sum()) + int(a2.sum())


def test_detect_drops_tiny_specks():
    a1, _ = _two_arms()
    speck = np.zeros((20, 40, 3), np.uint8)[:, :, 0].astype(bool)
    speck[0, 0] = True  # 1 px, below the >50 area gate
    ad = _adapter_with_masks([a1, speck])
    ad._seed_multi = True
    out = ad._detect(np.zeros((20, 40, 3), np.uint8), "robot arm", 20, 40)
    assert (out == a1).all()  # speck excluded


def test_detect_none_when_nothing_found():
    ad = _adapter_with_masks([])
    for multi in (False, True):
        ad._seed_multi = multi
        assert ad._detect(np.zeros((20, 40, 3), np.uint8), "robot arm", 20, 40) is None


def test_segment_and_infer_set_the_flag():
    # segment() must request multi-instance; infer() must not. Both call
    # _infer_masks — stub it to capture the flag at call time.
    a1, _ = _two_arms()
    ad = _adapter_with_masks([a1])
    seen = {}

    def fake_infer_masks(frame):
        seen["multi"] = ad._seed_multi
        return {}, frame.shape[0], frame.shape[1]

    ad._infer_masks = fake_infer_masks
    ad._concepts = []
    ad._signs = {}
    frame = np.zeros((20, 40, 3), np.uint8)

    ad.segment(frame)
    assert seen["multi"] is True
    # infer() also composites; give it what _composite_concepts needs via a stub.
    import lerobot.overlays.adapters as mod

    orig = mod._composite_concepts
    mod._composite_concepts = lambda *a, **k: np.zeros((20, 40, 4), np.uint8)
    try:
        ad._colors = {}
        ad._bg_color = None
        ad._cv2 = MagicMock()
        ad.infer(frame)
    finally:
        mod._composite_concepts = orig
    assert seen["multi"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
