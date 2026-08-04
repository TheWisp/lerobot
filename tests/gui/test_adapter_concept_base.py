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
"""ConceptMaskAdapter base contract (no GPU / model load).

The base owns the shared segmenter behaviour both SAM3 adapters (two-tier and
unified video) inherit: the resolution presets (a load-time knob), the control
contract (name changes restart tracking, display-only changes don't), and the
``+``/``-`` carving in ``segment()``. A fake subclass exercises it directly, so
drift between the two adapters' shared semantics is caught without weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.overlays.adapters import (
    ADAPTERS,
    SEGMENTER_KEYS,
    ConceptMaskAdapter,
    PolicySaliencyAdapter,
    build_adapter,
)


class _FakeSegmenter(ConceptMaskAdapter):
    key = "fake"
    label = "fake"

    def __init__(self, device: str = "cpu", resolution: int | None = None):
        super().__init__(device, resolution)
        self.restarts = 0
        self.masks_by_concept: dict[str, list[np.ndarray]] = {}

    def _restart_tracking(self) -> None:
        self.restarts += 1

    def _infer_masks(self, frame_rgb):
        h, w = frame_rgb.shape[:2]
        self._concepts = self._parse_concepts()
        return self.masks_by_concept, h, w


def _mask(h, w, ys, xs):
    m = np.zeros((h, w), dtype=bool)
    m[ys, xs] = True
    return m


def test_registry_segmenters_and_presets():
    # All SAM adapters are segmenters; the overlay-only saliency step is not.
    assert set(SEGMENTER_KEYS) == {"sam3_track", "sam3_video", "sam3_1"}
    assert not issubclass(PolicySaliencyAdapter, ConceptMaskAdapter)
    for k in SEGMENTER_KEYS:
        assert issubclass(ADAPTERS[k], ConceptMaskAdapter)
    # Presets sane: patch-multiple, descending, default among them.
    assert all(r % 14 == 0 for r in ConceptMaskAdapter.RESOLUTIONS)
    assert ConceptMaskAdapter.DEFAULT_RESOLUTION in ConceptMaskAdapter.RESOLUTIONS


def test_models_endpoint_resolutions_match_adapter_presets():
    # The GUI listing is a labelled copy of the adapter presets — drift would offer
    # a preset the endpoints then 400.
    from lerobot.gui.api.overlays import _RESOLUTIONS

    assert [r["value"] for r in _RESOLUTIONS] == list(ConceptMaskAdapter.RESOLUTIONS)


def test_overlay_endpoint_resolution_validation():
    # The overlay endpoints 400 a resolution outside the presets (None = default is fine).
    from fastapi import HTTPException

    from lerobot.gui.api.overlays import _validate_resolution

    _validate_resolution(None)
    _validate_resolution(ConceptMaskAdapter.DEFAULT_RESOLUTION)
    with pytest.raises(HTTPException):
        _validate_resolution(999)


def test_resolution_default_and_validation():
    assert _FakeSegmenter().resolution == ConceptMaskAdapter.DEFAULT_RESOLUTION
    assert _FakeSegmenter(resolution=1008).resolution == 1008
    with pytest.raises(AssertionError):
        _FakeSegmenter(resolution=1000)  # not a patch multiple
    with pytest.raises(AssertionError):
        _FakeSegmenter(resolution=140)  # under the sane floor


def test_build_adapter_passes_resolution_only_to_segmenters():
    # Saliency ignores the knob entirely (loads no model of its own).
    sal = build_adapter("policy_saliency", device="cpu", resolution=672)
    assert not hasattr(sal, "resolution")
    with pytest.raises(ValueError):
        build_adapter("nope", device="cpu")


def test_name_change_restarts_display_change_does_not():
    ad = _FakeSegmenter()
    ad.set_control({"objects": [{"name": "ring", "sign": "+"}]})
    assert ad.restarts == 1  # prompt changed from the default
    ad.set_control({"objects": [{"name": "ring", "sign": "+", "color": [1, 2, 3]}]})
    assert ad.restarts == 1  # colour/sign are display-only
    ad.set_control({"objects": [{"name": "ring", "sign": "-"}]})
    assert ad.restarts == 1  # sign flip: same name set, no restart
    ad.set_control({"objects": [{"name": "dowel", "sign": "+"}]})
    assert ad.restarts == 2  # name change restarts
    ad.set_control({"multi_instance": True})
    assert ad.restarts == 3  # seed-policy change restarts (base default is False)
    ad.set_control({"multi_instance": True})
    assert ad.restarts == 3  # idempotent


def test_segment_unions_instances_and_carves_negatives():
    ad = _FakeSegmenter()
    ad.set_control(
        {
            "objects": [{"name": "arm", "sign": "+"}, {"name": "held object", "sign": "-"}],
            "multi_instance": True,
        }
    )
    h = w = 10
    arm1 = _mask(h, w, slice(0, 4), slice(0, 4))
    arm2 = _mask(h, w, slice(6, 9), slice(6, 9))
    held = _mask(h, w, slice(0, 2), slice(0, 2))  # overlaps arm1
    ad.masks_by_concept = {"arm": [arm1, arm2], "held object": [held]}
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    out = ad.segment(frame)
    assert set(out) == {"arm"}  # negatives are carved, never returned
    expected = (arm1 | arm2) & ~held
    assert (out["arm"] == expected).all()


def test_segment_many_serial_fallback_and_flag():
    # Base contract: segment_many == per-camera segment (order-preserved), _prime_batch
    # is only invoked when batching is on AND >1 camera. The flag is a runtime
    # set_control toggle that must NOT restart tracking.
    ad = _FakeSegmenter()
    ad.set_control({"objects": [{"name": "ring", "sign": "+"}]})
    m = _mask(6, 8, slice(0, 3), slice(0, 3))
    ad.masks_by_concept = {"ring": [m]}
    primed = []
    ad._prime_batch = lambda frames: primed.append(sorted(frames))

    frames = {"a": np.zeros((6, 8, 3), np.uint8), "b": np.zeros((6, 8, 3), np.uint8)}
    out = ad.segment_many(frames)
    assert set(out) == {"a", "b"} and (out["a"]["ring"] == m).all()
    # Batching defaults OFF (measured tracking-quality regression when batched:
    # holds collapsed on a real dataset) — never primed unless explicitly enabled.
    assert primed == []

    restarts = ad.restarts
    ad.set_control({"batch_cameras": True})
    assert ad.restarts == restarts  # runtime toggle, no tracking restart
    ad.segment_many(frames)
    assert primed == [["a", "b"]]  # opt-in + 2 cams -> primed

    ad.set_control({"batch_cameras": False})
    ad.segment_many(frames)
    assert primed == [["a", "b"]]  # off again -> not primed

    ad.set_control({"batch_cameras": True})
    ad.segment_many({"a": frames["a"]})
    assert primed == [["a", "b"]]  # single camera -> nothing to share


def test_sam31_registered_and_listed_in_gui_steps():
    # The SAM 3.1 sidecar adapter is a full citizen: in the registry AND offered by
    # both tabs' model pickers (the worker is tab-agnostic).
    from lerobot.gui.api.overlays import _STEPS

    assert "sam3_1" in SEGMENTER_KEYS
    assert any(s["key"] == "sam3_1" for s in _STEPS)


def test_python_for_model_selects_sidecar_only_for_sam31(monkeypatch):
    # Non-sidecar models run in the server's interpreter; sam3_1 uses the sidecar
    # (env-overridable) and fails with the setup recipe when it's missing.
    import sys

    from lerobot.overlays.adapters import python_for_model

    assert python_for_model("sam3_track") == sys.executable
    monkeypatch.setenv("LEROBOT_SAM31_PYTHON", sys.executable)  # any existing file
    assert python_for_model("sam3_1") == sys.executable
    monkeypatch.setenv("LEROBOT_SAM31_PYTHON", "/nonexistent/python")
    with pytest.raises(RuntimeError, match="sidecar"):
        python_for_model("sam3_1")


def test_sam31_missing_sidecar_gives_actionable_error():
    # In the main env (no Meta sam3 repo) the adapter must fail with the sidecar
    # hint, not a bare ImportError.
    import importlib.util

    if importlib.util.find_spec("sam3") is not None:
        pytest.skip("sam3 present in this env")
    with pytest.raises(RuntimeError, match="sidecar"):
        build_adapter("sam3_1", device="cpu")
