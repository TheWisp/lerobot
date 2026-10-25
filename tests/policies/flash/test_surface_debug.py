"""Consumer invariants for the perception-only surface debugger."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, observation
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.tracking_debug_worker import seed_frame


def frame(number):
    return RGBDFrame(
        number,
        np.zeros((8, 10, 3), np.uint8),
        np.ones((8, 10), np.uint16),
        np.eye(3),
        0.001,
        np.eye(4),
        float(number),
        float(number),
        "test",
    )


def test_click_is_bound_to_frozen_image_not_newest_camera_frame():
    old, latest = frame(8), frame(12)
    assert seed_frame({"point": [4, 5], "frame_id": 8}, old, latest) is old
    assert seed_frame({}, None, latest) is latest
    with pytest.raises(ValueError, match="Freeze"):
        seed_frame({"point": [4, 5], "frame_id": 12}, old, latest)
    with pytest.raises(ValueError, match="Freeze"):
        seed_frame({"point": [4, 5], "frame_id": 8}, None, latest)
    with pytest.raises(ValueError, match="outside"):
        seed_frame({"point": [11, 5], "frame_id": 8}, old, latest)


def test_missing_observation_never_exposes_old_pose_as_current():
    data = np.load(Path(__file__).with_name("fixtures") / "dowel_held_frame_315.npz")
    depth, mask, k = data["depth"].astype(np.float32) * 0.001, data["mask"].astype(bool), data["intrinsics"]
    rgb = np.zeros((*depth.shape, 3), np.uint8)
    session = SurfaceTrackingSession(None)
    result = session.update(rgb, depth, mask, k, 315)
    assert result["status"] == "reference"
    reference = session.reference.copy()
    origin = result["camera_from_object"]
    lost = session.update(rgb, depth, np.zeros_like(mask), k, 316)
    assert lost["status"] == "no_mask"
    assert "camera_from_object" not in lost
    np.testing.assert_array_equal(session.reference, reference)
    no_depth = session.update(rgb, np.zeros_like(depth), mask, k, 317)
    assert no_depth["status"] == "insufficient_depth"
    assert "camera_from_object" not in no_depth
    assert origin[3] == [0, 0, 0, 1]
    assert result["diagnostics"]["reference_frame_id"] == 315


def test_filters_keep_measured_depth_and_reject_boundary_and_missing_depth():
    depth = np.ones((30, 40), np.float32)
    mask = np.zeros(depth.shape, bool)
    mask[5:25, 5:35] = True
    depth[13:16, 18:21] = 0
    k = np.array([[80, 0, 20], [0, 80, 15], [0, 0, 1]], float)
    result = observation(depth, mask, k, asdict(SurfaceObservationConfig()))
    assert not result["valid_image"][5, 20]
    assert not result["valid_image"][14, 19]
    assert not result["valid_image"][12, 19]
    assert result["valid_image"][10, 10]
    pixels = result["pixels"]
    np.testing.assert_array_equal(result["points"][:, 2], depth[pixels[:, 1], pixels[:, 0]])


def test_reselecting_reuses_weights_but_not_old_clicked_object(monkeypatch):
    from lerobot.overlays.adapters import ConceptMaskAdapter, Sam3TrackByDetectionAdapter
    from lerobot.policies.flash.tracking_debug_worker import new_selection

    # Exercise the real adapter's control sequencing and reset policy, without
    # loading GPU weights. Only the mask inference itself is replaced.
    adapter = object.__new__(Sam3TrackByDetectionAdapter)
    ConceptMaskAdapter.__init__(adapter, device="cpu", resolution=504)
    adapter._tracks = {}
    seen = []

    def segment(rgb):
        pending = adapter._pending_clicks.pop("debug", None)
        if pending:
            seen.append(pending)
            label = adapter._pending_click_name["debug"]
            mask = np.zeros(rgb.shape[:2], bool)
            x, y, _ = pending[0]
            mask[int(y), int(x)] = True
            adapter._click_names["debug"] = [label]
            adapter._tracks["debug"] = {"masks": {label: mask}, "scores": {label: 1.0}}
        return adapter._tracks.get("debug", {}).get("masks", {})

    monkeypatch.setattr(adapter, "segment", segment)
    first, sequence = new_selection(adapter, "selected object", (2, 3), 0)
    assert first.masks(frame(1))["object"][3, 2]
    second, sequence = new_selection(adapter, "selected object", (6, 5), sequence)
    assert not adapter._tracks.get("debug")
    assert second.masks(frame(2))["object"][5, 6]
    assert seen == [[(2.0, 3.0, 1)], [(6.0, 5.0, 1)]]


def test_adding_clicked_object_keeps_existing_mask_and_advances_sam_once(monkeypatch):
    from lerobot.overlays.adapters import ConceptMaskAdapter, Sam3TrackByDetectionAdapter
    from lerobot.policies.flash.tracking_debug_worker import new_selection

    adapter = object.__new__(Sam3TrackByDetectionAdapter)
    ConceptMaskAdapter.__init__(adapter, device="cpu", resolution=504)
    adapter._tracks = {}
    calls = []

    def segment(rgb):
        calls.append(1)
        track = adapter._tracks.setdefault("debug", {"masks": {}, "scores": {}})
        pending = adapter._pending_clicks.pop("debug", None)
        if pending:
            label = adapter._pending_click_name["debug"]
            x, y, _ = pending[0]
            mask = np.zeros(rgb.shape[:2], bool)
            mask[int(y), int(x)] = True
            track["masks"][label] = mask
            track["scores"][label] = 1.0
        return track["masks"]

    monkeypatch.setattr(adapter, "segment", segment)
    source, sequence = new_selection(adapter, "ring", (2, 3), 0)
    source.masks(frame(1))
    before = len(calls)
    second = frame(2)
    added, _ = source.add_clicked_role("object_2", "dowel", (6, 5), second)
    masks = source.masks(second)
    assert added
    assert len(calls) == before + 1  # the pipeline reuses the selected frame's masks
    assert masks["object"][3, 2] and masks["object_2"][5, 6]
    with pytest.raises(ValueError, match="newer frame"):
        source.add_clicked_role("object_3", "block", (4, 4), second)
    replacement, _ = new_selection(adapter, "fresh", (1, 1), max(sequence, source.next_click_sequence))
    assert set(replacement.masks(frame(3))) == {"object"}
