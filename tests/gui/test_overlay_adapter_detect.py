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
    """A bare adapter (no model load) whose detector post-process returns ``masks``."""
    a = object.__new__(Sam3TrackByDetectionAdapter)
    a._torch = torch
    a._Image = Image
    a.device = "cpu"
    a._det_threshold = 0.5
    a._seed_multi = False
    a._proc_size = {"height": 672, "width": 672}  # processor size override (matches load-time res)
    a._text_cache = {}
    a._pv_cache = {}
    a._init_click_state()  # click-to-segment state _infer_masks reads on every frame
    det_proc = MagicMock()
    det_proc.return_value.to.return_value = {
        "pixel_values": MagicMock(),
        "input_ids": MagicMock(),
        "attention_mask": None,
    }
    # The batched decode post-processes ALL concepts of one call together, so
    # the mock returns one result per requested target size.
    det_proc.post_process_instance_segmentation.side_effect = (
        lambda fwd, threshold, target_sizes: [{"masks": masks}] * len(target_sizes)
    )
    a.det_proc = det_proc
    a.det = MagicMock(return_value=MagicMock())
    # Real tensors where the batching path concatenates/expands: text features
    # are stacked across concepts, and the encoder output's tensor fields are
    # broadcast to the concept batch.
    a.det.get_text_features.return_value.pooler_output = torch.zeros((1, 4, 8))
    a.det.vision_encoder.return_value = {"last_hidden_state": torch.zeros((1, 3, 8))}
    return a


def _two_arms(h=20, w=40):
    a1 = np.zeros((h, w), dtype=bool)
    a1[2:14, 2:12] = True  # bigger "arm" (120 px)
    a2 = np.zeros((h, w), dtype=bool)
    a2[4:12, 28:36] = True  # smaller "arm" (64 px)
    return a1, a2


def test_select_single_instance_keeps_largest():
    a1, a2 = _two_arms()
    ad = _adapter_with_masks([])
    ad._seed_multi = False
    out = ad._select_instances({"masks": [a1, a2]}, 20, 40)
    assert out is not None
    assert (out == a1).all()  # only the largest instance
    assert not (out & a2).any()  # the second arm is dropped (debug-viz lock)


def test_select_multi_instance_unions_all():
    a1, a2 = _two_arms()
    ad = _adapter_with_masks([])
    ad._seed_multi = True
    out = ad._select_instances({"masks": [a1, a2]}, 20, 40)
    assert out is not None
    assert (out == (a1 | a2)).all()  # BOTH arms protected
    assert int(out.sum()) == int(a1.sum()) + int(a2.sum())


def test_select_drops_tiny_specks():
    a1, _ = _two_arms()
    speck = np.zeros((20, 40, 3), np.uint8)[:, :, 0].astype(bool)
    speck[0, 0] = True  # 1 px, below the >50 area gate
    ad = _adapter_with_masks([])
    ad._seed_multi = True
    out = ad._select_instances({"masks": [a1, speck]}, 20, 40)
    assert (out == a1).all()  # speck excluded


def test_select_none_when_nothing_found():
    ad = _adapter_with_masks([])
    for multi in (False, True):
        ad._seed_multi = multi
        assert ad._select_instances({"masks": []}, 20, 40) is None


def test_detect_many_encodes_the_frame_once():
    # N concepts must cost ONE vision encode + N text-side decodes (the whole point
    # of _detect_many); text features are cached, so a second frame with the same
    # concepts must not touch the text encoder at all.
    a1, _ = _two_arms()
    ad = _adapter_with_masks([a1])
    ad._seed_multi = True
    frame = np.zeros((20, 40, 3), np.uint8)

    out = ad._detect_many(frame, ["ring", "dowel"], 20, 40)
    assert set(out) == {"ring", "dowel"}
    assert ad.det.vision_encoder.call_count == 1  # one encode for both concepts
    assert ad.det.get_text_features.call_count == 2  # one per concept, first time
    # Concepts are BATCHED through one fusion/decode: N serial passes measured
    # 175 ms -> 21 ms at six concepts, with per-concept masks equal to serial
    # (fp16 boundary noise only). One call, not one per concept.
    assert ad.det.call_count == 1

    ad._detect_many(frame, ["ring", "dowel"], 20, 40)
    assert ad.det.vision_encoder.call_count == 2  # new frame -> new encode
    assert ad.det.get_text_features.call_count == 2  # cached: text encoder untouched

    assert ad._detect_many(frame, [], 20, 40) == {}  # empty concept list is free


def test_sessionless_seed_probe_is_throttled():
    # With no objects in view (session never seeds), the probe must NOT run the
    # detector for every concept on every frame — that measured ~30 ms/concept/frame
    # and dominated a live run during empty-scene stretches. Contract: the FIRST
    # frame after a reset probes immediately (scrub pickup), then every
    # RECOVER_EVERY-th frame until something is found.
    ad = _adapter_with_masks([])  # detector finds nothing -> session stays None
    ad.prompt = "green ring . wooden dowel"
    ad._colors = {}
    ad._signs = {}
    ad._tracks = {}
    ad._cam = "cam"
    ad._pv = lambda frame: None  # tracker preprocessing not under test
    probes = []
    orig = ad._detect_many
    ad._detect_many = lambda frame, cs, h, w: (probes.extend(cs), orig(frame, cs, h, w))[1]

    frame = np.zeros((20, 40, 3), np.uint8)
    n_frames = 2 * ad.RECOVER_EVERY + 1
    for _ in range(n_frames):
        masks, h, w = ad._infer_masks(frame)
        assert masks == {c: [] for c in ad._concepts}  # nothing held while unseeded
    # 2 concepts x (frame 0, then every RECOVER_EVERY-th): 3 probe frames, not 11.
    assert len(probes) == 2 * 3, probes

    ad.reset()  # scrub/discontinuity -> the very next frame must probe again at once
    probes.clear()
    ad._infer_masks(frame)
    assert len(probes) == 2


def test_set_control_toggles_multi_instance_and_restarts_tracking():
    # The instance policy is a shared knob set via set_control (so overlay preview
    # and batch processing agree). A change restarts tracking so it takes effect now.
    a1, _ = _two_arms()
    ad = _adapter_with_masks([a1])
    ad._seed_multi = False
    ad.prompt = "robot arm"
    ad._colors = {}
    ad._signs = {}
    ad._tracks = {"cam": {"session": object()}}  # pretend a track exists

    ad.set_control({"multi_instance": True})
    assert ad._seed_multi is True
    assert ad._tracks == {}  # restarted so the next frame re-seeds under the new policy

    ad._tracks = {"cam": {"session": object()}}
    ad.set_control({"multi_instance": True})  # unchanged -> no restart
    assert ad._tracks != {}

    ad.set_control({"multi_instance": False})
    assert ad._seed_multi is False


def test_segment_does_not_override_the_flag():
    # The entry point respects whatever set_control chose — it never sets the policy.
    a1, _ = _two_arms()
    ad = _adapter_with_masks([a1])
    ad._concepts = []
    ad._signs = {}
    seen = []
    ad._infer_masks = lambda frame: (seen.append(ad._seed_multi), ({}, frame.shape[0], frame.shape[1]))[1]

    ad._seed_multi = True
    ad.segment(np.zeros((20, 40, 3), np.uint8))
    ad._seed_multi = False
    ad.segment(np.zeros((20, 40, 3), np.uint8))
    assert seen == [True, False]  # each call saw the current policy — neither changed it


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
