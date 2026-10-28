"""A delayed pose anchor must replay against its own RGB-D frame."""

import time
from threading import Event

import cv2
import numpy as np
import pytest

from lerobot.policies.flash.causal_points import CausalRigidPointTracker
from lerobot.policies.flash.causal_service import CausalPoseService, CausalTrackingConfig, _Anchor
from lerobot.policies.flash.perception import LatestPoseStore, RGBDFrame
from lerobot.policies.flash.scene_groups import ScenePointTracker
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


def frame(image, frame_id):
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    return RGBDFrame(
        frame_id,
        rgb,
        np.full(image.shape, 1000, dtype=np.uint16),
        np.array([[100.0, 0, 48.0], [0, 100.0, 48.0], [0, 0, 1]]),
        0.001,
        np.eye(4),
        frame_id / 30,
        frame_id / 30,
        "camera-v1",
    )


def test_causal_config_rejects_unbounded_or_invalid_tracking():
    valid = {
        "history_frames": 60,
        "queue_frames": 60,
        "min_inliers": 12,
        "max_frame_gap_s": 0.1,
        "fit_threshold_m": 0.005,
    }
    CausalTrackingConfig(**valid, image_scale=0.5)
    with pytest.raises(ValueError, match="invalid causal tracking configuration"):
        CausalTrackingConfig(**{**valid, "image_scale": 0.0})
    complete = {
        **valid,
        "image_scale": 0.5,
        "scene_max_points": 800,
        "environment_context_radius_m": 0.25,
        "environment_min_motion_m": 0.008,
        "environment_prediction_max_s": 1.0,
        "environment_min_inliers": 12,
        "static_prediction_max_s": None,
        "max_pose_disagreement_translation_m": 0.01,
        "max_pose_disagreement_rotation_rad": 0.15,
    }
    assert CausalTrackingConfig.from_config(complete).scene_max_points == 800
    assert CausalTrackingConfig.from_config(complete).depth_support_radius_px == 1
    assert (
        CausalTrackingConfig.from_config({**complete, "depth_support_radius_px": 0}).depth_support_radius_px
        == 0
    )
    with pytest.raises(ValueError, match="causal tracking keys differ"):
        CausalTrackingConfig.from_config(
            {key: value for key, value in complete.items() if key != "scene_max_points"}
        )
    with pytest.raises(ValueError, match="invalid causal tracking configuration"):
        CausalTrackingConfig.from_config({**complete, "max_pose_disagreement_translation_m": 0})


def test_delayed_anchor_uses_retained_frame_and_all_intervening_motion():
    image = np.random.default_rng(4).integers(0, 255, (96, 96), dtype=np.uint8)
    frames = [
        frame(cv2.warpAffine(image, np.float32([[1, 0, 2 * i], [0, 1, 0]]), (96, 96)), i) for i in range(3)
    ]
    mask = np.zeros(image.shape, dtype=bool)
    mask[20:76, 20:76] = True
    store = LatestPoseStore()
    service = CausalPoseService(store)
    service.start()
    try:
        for item in frames:
            service.submit_frame(item)
        anchor = PoseEstimate(
            "object", np.eye(4), 0.0, 0.0, Evidence.OBSERVED, "object-v1", (True,) * 3, (True,) * 3
        )
        service.submit_anchor(0, (anchor,), {"object": mask})
        deadline = time.monotonic() + 2
        while service.published < 2 and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.error is None
        assert service.anchors == 1
        pose = store.snapshot()[1]["object"]
        assert pose.source_time_s == frames[2].source_time_s
        assert pose.evidence == Evidence.PREDICTED
        center = pose.pose @ np.array([0.0, 0.0, 1.0, 1.0])
        np.testing.assert_allclose(center[:3], [0.04, 0.0, 1.0], atol=0.005)
    finally:
        service.close()


def test_later_observation_rebases_pending_track_without_restarting_replay():
    image = np.random.default_rng(14).integers(0, 255, (96, 96), dtype=np.uint8)
    frames = [
        frame(cv2.warpAffine(image, np.float32([[1, 0, 2 * i], [0, 1, 0]]), (96, 96)), i) for i in range(5)
    ]
    mask = np.zeros(image.shape, dtype=bool)
    mask[20:76, 20:76] = True
    shifted_mask = np.zeros_like(mask)
    shifted_mask[20:76, 24:80] = True
    store = LatestPoseStore()
    accepted_direct = []
    service = CausalPoseService(store, on_direct_observation=accepted_direct.append)
    service.start()
    try:
        for item in frames:
            service.submit_frame(item)
        first = PoseEstimate(
            "object", np.eye(4), 0.0, 0.0, Evidence.OBSERVED, "object-v1", (True,) * 3, (True,) * 3
        )
        service.submit_anchor(0, (first,), {"object": mask})
        second_pose = np.eye(4)
        second_pose[0, 3] = 0.05  # frame 2 plus a 1 cm absolute correction
        second = PoseEstimate(
            "object", second_pose, 2 / 30, 2 / 30, Evidence.OBSERVED, "object-v1", (True,) * 3, (True,) * 3
        )
        service.submit_anchor(2, (second,), {"object": shifted_mask})
        deadline = time.monotonic() + 2
        while (
            service.anchor_outcomes_by_role.get("object", {}).get("point_replay_caught_up", 0) < 1
            and service.error is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert service.error is None
        assert service.anchors == 1
        assert service.anchor_outcomes_by_role["object"]["pending_point_track"] == 1
        assert service.anchor_outcomes_by_role["object"]["point_replay_caught_up"] == 1
        result = store.snapshot()[1]["object"]
        assert result.source_time_s == frames[-1].source_time_s
        assert result.last_direct_observation_s == frames[2].source_time_s
        np.testing.assert_allclose(result.pose[0, 3], 0.09, atol=0.005)
        assert [estimate.source_time_s for estimate in accepted_direct] == [
            frames[0].source_time_s,
            frames[2].source_time_s,
        ]
    finally:
        service.close()


def test_old_point_replay_yields_to_a_new_camera_frame(monkeypatch):
    image = np.random.default_rng(24).integers(0, 255, (96, 96), dtype=np.uint8)
    frames = [frame(image, index) for index in range(22)]
    mask = np.ones(image.shape, dtype=bool)
    first_replay = Event()
    allow_replay = Event()
    new_scene_frame = Event()
    allow_scene = Event()
    replay_calls = []
    point_update = CausalRigidPointTracker.update
    scene_update = ScenePointTracker.update

    def tracked_update(self, item):
        replay_calls.append(item.frame_id)
        if len(replay_calls) == 1:
            first_replay.set()
            assert allow_replay.wait(2)
        return point_update(self, item)

    def scene_frame(self, item):
        if item.frame_id == 21:
            new_scene_frame.set()
            assert allow_scene.wait(2)
        return scene_update(self, item)

    monkeypatch.setattr(CausalRigidPointTracker, "update", tracked_update)
    monkeypatch.setattr(ScenePointTracker, "update", scene_frame)
    service = CausalPoseService(LatestPoseStore(), history_frames=60)
    service.start()
    try:
        for item in frames[:21]:
            service.submit_frame(item)
        anchor = PoseEstimate(
            "object", np.eye(4), 0.0, 0.0, Evidence.OBSERVED, "object-v1", (True,) * 3, (True,) * 3
        )
        service.submit_anchor(0, (anchor,), {"object": mask})
        assert first_replay.wait(2)
        service.submit_frame(frames[21])
        allow_replay.set()
        assert new_scene_frame.wait(2)
        assert replay_calls == [1]
    finally:
        allow_replay.set()
        allow_scene.set()
        service.close()
    assert service.error is None


def test_anchor_requires_the_recorded_capture_time_and_calibration():
    image = np.random.default_rng(9).integers(0, 255, (96, 96), dtype=np.uint8)
    captured = frame(image, 1)
    mask = np.ones(image.shape, dtype=bool)
    estimate = PoseEstimate(
        "object",
        np.eye(4),
        captured.source_time_s,
        captured.available_time_s,
        Evidence.OBSERVED,
        "object-v1",
        (True,) * 3,
        (True,) * 3,
    )
    store = LatestPoseStore()
    service = CausalPoseService(store)
    service.start()
    try:
        service.submit_frame(captured)
        service.submit_anchor(
            1,
            (estimate,),
            {"object": mask},
            source_time_s=captured.source_time_s,
            calibration_version="different-calibration",
        )
        service.submit_anchor(
            1,
            (estimate,),
            {"object": mask},
            source_time_s=captured.source_time_s + 0.01,
            calibration_version=captured.calibration_version,
        )
        deadline = time.monotonic() + 2
        while service.missed_anchors < 2 and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.error is None
        assert service.missed_anchors == 2
        assert store.snapshot() == (0, {})
    finally:
        service.close()


def test_accepted_live_pose_is_available_at_scene_decision_not_backend_completion():
    image = np.random.default_rng(51).integers(0, 255, (96, 96), dtype=np.uint8)
    captured = frame(image, 0)
    candidate = PoseEstimate(
        "object",
        np.eye(4),
        captured.source_time_s,
        captured.available_time_s,
        Evidence.OBSERVED,
        "object-v1",
        (True,) * 3,
        (True,) * 3,
    )
    decisions = []
    store = LatestPoseStore()
    service = CausalPoseService(store, on_anchor_decision=decisions.append)
    service.start()
    try:
        service.submit_frame(captured)
        service.submit_anchor(0, (candidate,), {"object": np.ones(image.shape, dtype=bool)})
        deadline = time.monotonic() + 2
        while not decisions and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.error is None
        assert len(decisions) == 1
        decision = decisions[0]
        assert decision.outcome == "accepted"
        assert decision.candidate.available_time_s == captured.available_time_s
        assert decision.accepted.available_time_s == decision.decided_at_s
        assert store.snapshot()[1]["object"].available_time_s >= decision.decided_at_s
    finally:
        service.close()


def test_offline_source_order_pose_keeps_recorded_availability_time():
    image = np.random.default_rng(52).integers(0, 255, (96, 96), dtype=np.uint8)
    captured = frame(image, 0)
    candidate = PoseEstimate(
        "object",
        np.eye(4),
        captured.source_time_s,
        captured.available_time_s,
        Evidence.OBSERVED,
        "object-v1",
        (True,) * 3,
        (True,) * 3,
    )
    store = LatestPoseStore()
    service = CausalPoseService(store)
    result = service.process_offline(captured, (candidate,), {"object": np.ones(image.shape, dtype=bool)})
    assert result["object"].available_time_s == captured.available_time_s


def test_delayed_symmetry_proposal_does_not_use_later_pose_as_its_reference():
    image = np.random.default_rng(53).integers(0, 255, (96, 96), dtype=np.uint8)
    frames = [frame(image, index) for index in range(2)]
    references = []

    class RecordingSymmetry:
        role = "object"

        def canonicalize(self, pose, *, reference_pose=None):
            references.append(reference_pose)
            return pose

    store = LatestPoseStore()
    service = CausalPoseService(store, axial_symmetries=(RecordingSymmetry(),))
    for item in frames:
        service._frame(item)
    newer = PoseEstimate(
        "object",
        np.eye(4),
        frames[1].source_time_s,
        frames[1].available_time_s,
        Evidence.OBSERVED,
        "object-v1",
        (True,) * 3,
        (True,) * 3,
    )
    service.active["object"] = (CausalRigidPointTracker(), newer)
    store.publish((newer,))
    older = PoseEstimate(
        "object",
        np.eye(4),
        frames[0].source_time_s,
        frames[0].available_time_s,
        Evidence.OBSERVED,
        "object-v1",
        (True,) * 3,
        (True,) * 3,
    )
    service._anchor(_Anchor(0, (older,), {"object": np.ones(image.shape, dtype=bool)}))
    assert references == [None]
    assert service.superseded_proposals == 1
