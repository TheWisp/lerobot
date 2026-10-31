"""Recorded extraction must use the same group bridge as a live estimator."""

from dataclasses import replace

import numpy as np

import lerobot.policies.flash.offline_perception as offline_perception
from lerobot.policies.flash.causal_service import CausalTrackingConfig
from lerobot.policies.flash.perception import ObjectReference, ReferenceSnippet, RGBDFrame
from lerobot.policies.flash.scene_groups import SceneTracks
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


def _pose(x):
    pose = np.eye(4)
    pose[:3, 3] = [x, 0, 1]
    return pose


def _frame(index):
    return RGBDFrame(
        index,
        np.zeros((96, 96, 3), dtype=np.uint8),
        np.full((96, 96), 1000, dtype=np.uint16),
        np.array([[100, 0, 48], [0, 100, 48], [0, 0, 1]], dtype=float),
        0.001,
        np.eye(4),
        index / 30,
        index / 30,
        "camera-v1",
    )


class _RecordedBackend:
    def __init__(self, mask):
        self.mask = mask
        self.last_masks = {}

    def initialize(self, references):
        assert [reference.role for reference in references] == ["target"]

    def update(self, frame):
        visible = frame.frame_id < 2
        self.last_masks = {"target": self.mask.copy() if visible else np.zeros_like(self.mask)}
        return (
            PoseEstimate(
                "target",
                _pose(frame.frame_id * 0.02) if visible else None,
                frame.source_time_s,
                frame.available_time_s,
                Evidence.OBSERVED if visible else Evidence.LOST,
                "v1",
                (visible,) * 3,
                (visible,) * 3,
                estimator="recorded_direct",
            ),
        )


def test_demo_group_bridge_uses_live_scene_estimator(monkeypatch):
    rng = np.random.default_rng(8)
    points = rng.uniform((-0.08, -0.08, 0.98), (0.08, 0.08, 1.02), (48, 3))
    pixels = np.column_stack((rng.integers(12, 84, len(points)), rng.integers(12, 84, len(points))))
    mask = np.zeros((96, 96), dtype=bool)
    mask[pixels[:4, 1], pixels[:4, 0]] = True
    tracks = [
        SceneTracks(index, index / 30, np.arange(len(points)), pixels, points + [index * 0.02, 0, 0])
        for index in range(4)
    ]
    original_builder = offline_perception.build_causal_pose_service

    def with_recorded_scene(*args, **kwargs):
        service = original_builder(*args, **kwargs)
        service.scene.update = lambda frame: tracks[frame.frame_id]
        return service

    monkeypatch.setattr(offline_perception, "build_causal_pose_service", with_recorded_scene)
    reference = ObjectReference(
        "target", "v1", (ReferenceSnippet("demo", 0, 0, "target", np.eye(4), "mask.png", "camera-v1"),)
    )
    frames = [_frame(index) for index in range(4)]
    direct_only = offline_perception.extract_pose_trace(
        frames, _RecordedBackend(mask), (reference,), causal_tracking=None
    )
    with_group = offline_perception.extract_pose_trace(
        frames,
        _RecordedBackend(mask),
        (reference,),
        causal_tracking=CausalTrackingConfig(8, 8, 12, 0.1, 0.005),
    )

    assert direct_only[2]["target"].evidence == Evidence.LOST
    assert with_group[2]["target"].evidence == Evidence.PREDICTED
    assert with_group[2]["target"].estimator == "environment_group_motion"
    np.testing.assert_allclose(with_group[2]["target"].pose[:3, 3], [0.04, 0, 1], atol=0.005)
    assert with_group[2]["target"].last_direct_observation_s == frames[1].source_time_s
    assert with_group[2]["target"].available_time_s == frames[2].available_time_s


def test_offline_extraction_does_not_publish_backend_wall_clock_as_recorded_availability():
    mask = np.zeros((96, 96), dtype=bool)
    mask[40:56, 40:56] = True
    reference = ObjectReference(
        "target", "v1", (ReferenceSnippet("demo", 0, 0, "target", np.eye(4), "mask.png", "camera-v1"),)
    )

    class WallClockBackend(_RecordedBackend):
        def update(self, frame):
            return tuple(
                replace(estimate, available_time_s=1_000_000.0) for estimate in super().update(frame)
            )

    vision_results = []
    scene_rows = []
    frames = [_frame(index) for index in range(2)]
    trace = offline_perception.extract_pose_trace(
        frames,
        WallClockBackend(mask),
        (reference,),
        causal_tracking=CausalTrackingConfig(8, 8, 12, 0.1, 0.005),
        on_vision_result=vision_results.append,
        on_frame_evidence=lambda frame, result, groups, estimates: scene_rows.append((result, estimates)),
    )
    for index, frame in enumerate(frames):
        assert vision_results[index].estimates[0].available_time_s == frame.available_time_s
        assert vision_results[index].proposals[0].estimate.available_time_s == frame.available_time_s
        assert scene_rows[index][0].proposals[0].estimate.available_time_s == frame.available_time_s
        assert scene_rows[index][1]["target"].available_time_s == frame.available_time_s
        assert trace[index]["target"].available_time_s == frame.available_time_s


def test_demo_extraction_uses_motor_event_time_between_camera_frames(monkeypatch):
    mask = np.ones((96, 96), dtype=bool)
    reference = ObjectReference(
        "target", "v1", (ReferenceSnippet("demo", 0, 0, "target", np.eye(4), "mask.png", "camera-v1"),)
    )
    services = []
    original_builder = offline_perception.build_causal_pose_service

    def capture_service(*args, **kwargs):
        service = original_builder(*args, **kwargs)
        services.append(service)
        return service

    monkeypatch.setattr(offline_perception, "build_causal_pose_service", capture_service)
    grasp_time_s = 0.4 / 30  # nearest camera is frame 0, captured before the grasp
    release_time_s = 1.6 / 30
    rows = offline_perception.extract_pose_trace(
        (_frame(index) for index in range(3)),
        _RecordedBackend(mask),
        (reference,),
        causal_tracking=CausalTrackingConfig(8, 8, 12, 0.1, 0.005),
        attachment_events=((grasp_time_s, "target"), (release_time_s, None)),
    )

    service = services[0]
    assert service.attachment_by_frame == {0: None, 1: "target", 2: None}
    assert [event.source_time_s for event in service._attachment_events] == [
        grasp_time_s,
        release_time_s,
    ]
    assert rows[0]["target"].evidence == Evidence.OBSERVED
    with np.testing.assert_raises_regex(ValueError, "event times or camera-frame roles"):
        offline_perception.extract_pose_trace(
            (_frame(index) for index in range(1)),
            _RecordedBackend(mask),
            (reference,),
            causal_tracking=None,
            attachment_events=((grasp_time_s, "target"),),
            attached_role_at_frame=lambda _: "target",
        )


def test_demo_attachment_disables_environment_bridge(monkeypatch):
    """A held object must not inherit motion from its former tray group."""
    rng = np.random.default_rng(9)
    points = rng.uniform((-0.08, -0.08, 0.98), (0.08, 0.08, 1.02), (48, 3))
    pixels = np.column_stack((rng.integers(12, 84, len(points)), rng.integers(12, 84, len(points))))
    mask = np.zeros((96, 96), dtype=bool)
    mask[pixels[:4, 1], pixels[:4, 0]] = True
    original_builder = offline_perception.build_causal_pose_service

    def with_recorded_scene(*args, **kwargs):
        service = original_builder(*args, **kwargs)
        service.scene.update = lambda frame: SceneTracks(
            frame.frame_id,
            frame.source_time_s,
            np.arange(len(points)),
            pixels,
            points + [frame.frame_id * 0.02, 0, 0],
        )
        return service

    monkeypatch.setattr(offline_perception, "build_causal_pose_service", with_recorded_scene)
    reference = ObjectReference(
        "target", "v1", (ReferenceSnippet("demo", 0, 0, "target", np.eye(4), "mask.png", "camera-v1"),)
    )
    rows = offline_perception.extract_pose_trace(
        (_frame(index) for index in range(4)),
        _RecordedBackend(mask),
        (reference,),
        causal_tracking=CausalTrackingConfig(8, 8, 12, 0.1, 0.005),
        attached_role_at_frame=lambda index: "target" if index >= 2 else None,
    )
    assert rows[2]["target"].evidence == Evidence.LOST


def test_demo_held_prediction_uses_the_same_scene_owner_as_live():
    mask = np.zeros((96, 96), dtype=bool)
    mask[40:56, 40:56] = True
    reference = ObjectReference(
        "target", "v1", (ReferenceSnippet("demo", 0, 0, "target", np.eye(4), "mask.png", "camera-v1"),)
    )
    frames = [_frame(index) for index in range(3)]

    def gripper_at_source(role, frame):
        return PoseEstimate(
            role,
            _pose(0.2 + frame.frame_id * 0.01),
            frame.source_time_s,
            frame.available_time_s,
            Evidence.PREDICTED,
            "v1",
            (True,) * 3,
            (True,) * 3,
            estimator="gripper_motion",
            source_frame_id=frame.frame_id,
        )

    rows = offline_perception.extract_pose_trace(
        frames,
        _RecordedBackend(mask),
        (reference,),
        causal_tracking=CausalTrackingConfig(8, 8, 12, 0.1, 0.005),
        attachment_events=((0.5 / 30, "target"),),
        kinematic_prediction_at=gripper_at_source,
    )

    assert rows[0]["target"].evidence == Evidence.OBSERVED
    assert rows[1]["target"].evidence == Evidence.OBSERVED
    assert rows[1]["target"].estimator == "recorded_direct"
    assert rows[2]["target"].evidence == Evidence.PREDICTED
    assert rows[2]["target"].estimator == "gripper_motion"
    np.testing.assert_allclose(rows[2]["target"].pose[:3, 3], [0.22, 0, 1])


def test_rejected_direct_fit_is_not_reintroduced_as_a_demo_observation(monkeypatch):
    mask = np.zeros((96, 96), dtype=bool)
    mask[46:48, 46:48] = True  # Too few pixels to seed point motion.
    reference = ObjectReference(
        "target", "v1", (ReferenceSnippet("demo", 0, 0, "target", np.eye(4), "mask.png", "camera-v1"),)
    )
    original_builder = offline_perception.build_causal_pose_service

    def reject_second_frame(*args, **kwargs):
        service = original_builder(*args, **kwargs)
        service._direct_conflicts_with_point_motion = (
            lambda estimate, frame, instance_mask: frame.frame_id == 1
        )
        return service

    monkeypatch.setattr(offline_perception, "build_causal_pose_service", reject_second_frame)
    raw, review = [], []
    rows = offline_perception.extract_pose_trace(
        (_frame(index) for index in range(2)),
        _RecordedBackend(mask),
        (reference,),
        causal_tracking=CausalTrackingConfig(8, 8, 12, 0.1, 0.005),
        on_vision_result=raw.append,
        on_frame_evidence=lambda frame, result, groups, estimates: review.append(estimates),
    )

    assert raw[1].estimates[0].evidence == Evidence.OBSERVED
    assert rows[0]["target"].evidence == Evidence.OBSERVED
    assert rows[1]["target"].evidence == Evidence.LOST
    assert rows[1]["target"].pose is None
    assert rows[1]["target"].estimator == "scene_estimator_rejected"
    assert review[1]["target"] == rows[1]["target"]
