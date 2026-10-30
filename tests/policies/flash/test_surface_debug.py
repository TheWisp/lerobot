"""Consumer invariants for the perception-only surface debugger."""

import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pytest

from lerobot.policies.flash.causal_service import CausalTrackingConfig
from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.perception_step import PerceptionResult
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, observation
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.tracking_debug_worker import (
    DebugGroupFallback,
    RGBDTrace,
    group_display,
    seed_frame,
)
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


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


def test_debugger_group_bridge_consumes_camera_frames_between_vision_results():
    import time

    image = np.random.default_rng(4).integers(0, 255, (96, 96), dtype=np.uint8)
    frames = []
    for index in range(3):
        rgb = cv2.warpAffine(image, np.float32([[1, 0, 2 * index], [0, 1, 0]]), (96, 96))
        frames.append(
            RGBDFrame(
                index,
                np.repeat(rgb[:, :, None], 3, axis=2),
                np.full((96, 96), 1000, np.uint16),
                np.array([[100, 0, 48], [0, 100, 48], [0, 0, 1]], float),
                0.001,
                np.eye(4),
                index / 30,
                index / 30,
                "camera-v1",
            )
        )
    mask = np.zeros((96, 96), bool)
    mask[20:76, 20:76] = True
    direct = PoseEstimate(
        "object",
        np.eye(4),
        0.0,
        0.0,
        Evidence.OBSERVED,
        "object-v1",
        (True,) * 3,
        (True,) * 3,
    )
    result = PerceptionResult(
        (direct,),
        {},
        0,
        0.0,
        "camera-v1",
        {"object": mask},
    )
    bridge = DebugGroupFallback(CausalTrackingConfig(8, 8, 12, 0.1, 0.005))
    try:
        bridge.submit_result(frames[0], result)
        for item in frames[1:]:
            bridge.submit_frame(item)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            pose = bridge.estimates().get("object")
            if pose is not None and pose.source_frame_id == 2:
                break
            time.sleep(0.01)
        assert bridge.service.error is None
        assert bridge.service.frames == 3
        assert pose.evidence == Evidence.PREDICTED
        np.testing.assert_allclose(pose.pose[0, 3], 0.04, atol=0.005)
    finally:
        bridge.close()


def test_debugger_group_bridge_keeps_moving_target_after_direct_loss(monkeypatch):
    import time

    import lerobot.policies.flash.tracking_debug_worker as debugger
    from lerobot.policies.flash.scene_groups import SceneTracks

    rng = np.random.default_rng(8)
    points = rng.uniform((-0.08, -0.08, 0.98), (0.08, 0.08, 1.02), (48, 3))
    pixels = np.column_stack((rng.integers(12, 84, len(points)), rng.integers(12, 84, len(points))))
    base_time = time.perf_counter()
    scene = [
        SceneTracks(
            index, base_time + index / 30, np.arange(len(points)), pixels, points + [index * 0.02, 0, 0]
        )
        for index in range(3)
    ]
    build = debugger.build_causal_pose_service

    def with_scene(*args, **kwargs):
        service = build(*args, **kwargs)
        service.scene.update = lambda frame: scene[frame.frame_id]
        return service

    monkeypatch.setattr(debugger, "build_causal_pose_service", with_scene)
    mask = np.zeros((96, 96), bool)
    mask[pixels[:4, 1], pixels[:4, 0]] = True
    bridge = DebugGroupFallback(CausalTrackingConfig(8, 8, 12, 0.1, 0.005))

    def camera_frame(index):
        return RGBDFrame(
            index,
            np.zeros((96, 96, 3), np.uint8),
            np.full((96, 96), 1000, np.uint16),
            np.array([[100, 0, 48], [0, 100, 48], [0, 0, 1]], float),
            0.001,
            np.eye(4),
            base_time + index / 30,
            base_time + index / 30,
            "camera-v1",
        )

    def result(index):
        pose = np.eye(4)
        pose[0, 3] = index * 0.02
        pose[2, 3] = 1.0
        direct = PoseEstimate(
            "target",
            pose,
            base_time + index / 30,
            base_time + index / 30,
            Evidence.OBSERVED,
            "target-v1",
            (True,) * 3,
            (True,) * 3,
        )
        return PerceptionResult(
            (direct,),
            {},
            index,
            base_time + index / 30,
            "camera-v1",
            {"target": mask.copy()},
        )

    try:
        bridge.submit_result(camera_frame(0), result(0))
        bridge.submit_frame(camera_frame(1))
        bridge.submit_result(camera_frame(1), result(1))
        deadline = time.monotonic() + 2
        while (
            bridge.service.anchor_outcomes_by_role.get("target", {}).get("direct_accepted", 0) < 2
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        bridge.submit_frame(camera_frame(2))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            pose = bridge.estimates().get("target")
            if pose is not None and pose.source_frame_id == 2:
                break
            time.sleep(0.01)
        assert bridge.service.error is None
        assert pose.estimator == "environment_group_motion"
        np.testing.assert_allclose(pose.pose[:3, 3], [0.04, 0, 1], atol=0.005)
    finally:
        bridge.close()


def test_group_projection_uses_saved_reference_when_direct_fit_is_lost():
    source = frame(8)
    source.intrinsics[:] = [[100, 0, 5], [0, 100, 4], [0, 0, 1]]
    accepted = {
        "accepted": True,
        "camera_from_reference": np.eye(4).tolist(),
        "camera_from_object": np.eye(4).tolist(),
        "reference_m": [[0, 0, 1]],
        "anchors_m": [[0, 0, 1]],
        "axes_m": [[0.01, 0, 1]],
    }
    predicted = np.eye(4)
    predicted[0, 3] = 0.02
    pose = PoseEstimate(
        "object",
        predicted,
        8.0,
        8.0,
        Evidence.PREDICTED,
        "object-v1",
        (True,) * 3,
        (True,) * 3,
        estimator="environment_group_motion",
        source_frame_id=8,
    )
    missing = {"accepted": False, "reference_version": "object-v1"}
    display = group_display(missing, pose, source, accepted)
    assert display["group_estimator"] == "environment_group_motion"
    np.testing.assert_allclose(display["group_reference_px"], [[7, 4]])
    assert group_display(missing, pose, frame(9), accepted)["group_current_frame"] is False


def test_diagnostic_trace_preserves_aligned_depth_mask_and_pose(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    selected = frame(8)
    selected.rgb[2, 3] = [201, 18, 71]
    selected.depth[2, 3] = 413
    mask = np.zeros(selected.depth.shape, bool)
    mask[2, 3] = True
    pose = np.eye(4).tolist()
    trace = RGBDTrace(1, {"vision_pipeline": {"backend": "surface"}})
    trace.save(
        selected,
        {"object": mask},
        {"object": {"status": "fitted", "accepted": False, "camera_from_reference": pose, "diagnostics": {}}},
    )
    trace.finish()
    stem = "000008.png"
    rgb = cv2.cvtColor(cv2.imread(str(trace.path / "rgb" / stem)), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(trace.path / "depth" / stem), -1)
    saved_mask = cv2.imread(str(trace.path / "masks" / "object" / stem), 0)
    np.testing.assert_array_equal(rgb, selected.rgb)
    np.testing.assert_array_equal(depth, selected.depth)
    np.testing.assert_array_equal(saved_mask > 0, mask)
    metadata = json.loads((trace.path / "frames.jsonl").read_text())
    assert metadata["objects"]["object"]["camera_from_reference"] == pose
    assert trace.status()["complete"] and trace.status()["frames"] == 1


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
