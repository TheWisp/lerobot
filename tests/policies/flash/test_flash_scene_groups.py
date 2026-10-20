"""Environment motion must be measured, associated and allowed to abstain."""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from lerobot.policies.flash.causal_service import CausalPoseService
from lerobot.policies.flash.perception import LatestPoseStore, RGBDFrame
from lerobot.policies.flash.scene_groups import (
    ScenePointTracker,
    SceneTracks,
    establish_environment_relationship,
)
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate

RECORDED_MOTION = Path(__file__).parent / "fixtures/ring_tray_motion_20261016.npz"


def _tracks(frame_id: int, points: np.ndarray, pixels: np.ndarray) -> SceneTracks:
    return SceneTracks(frame_id, frame_id / 30, np.arange(len(points)), pixels, points)


def _pose(x: float) -> np.ndarray:
    result = np.eye(4)
    result[0, 3] = x
    result[2, 3] = 1.0
    return result


def test_environment_relation_needs_non_object_co_motion_and_tracks_occlusion():
    rng = np.random.default_rng(8)
    group = rng.uniform((-0.12, -0.12, 0.98), (0.12, 0.12, 1.02), (36, 3))
    static = rng.uniform((-0.12, -0.12, 0.98), (0.12, 0.12, 1.02), (20, 3))
    pixels = np.column_stack((rng.integers(20, 76, 56), rng.integers(20, 76, 56)))
    before = _tracks(0, np.concatenate((group, static)), pixels)
    after = _tracks(1, np.concatenate((group + [0.02, 0, 0], static)), pixels)
    hidden = _tracks(2, np.concatenate((group + [0.04, 0, 0], static)), pixels)
    mask = np.zeros((96, 96), dtype=bool)
    mask[pixels[:8, 1], pixels[:8, 0]] = True  # Object points cannot certify its environment.
    relation = establish_environment_relationship(
        before,
        after,
        _pose(0),
        _pose(0.02),
        (mask,),
        context_radius_m=0.25,
        fit_threshold_m=0.003,
        min_motion_m=0.008,
        min_inliers=12,
        reference_version="v1",
    )
    assert relation is not None
    assert len(relation.ids) >= 20
    np.testing.assert_allclose(
        relation.predict(hidden, fit_threshold_m=0.003, min_inliers=12)[:3, 3],
        [0.04, 0, 1.0],
        atol=1e-6,
    )
    too_few = SceneTracks(2, 2 / 30, hidden.ids[:5], hidden.xy[:5], hidden.base_xyz[:5])
    assert relation.predict(too_few, fit_threshold_m=0.003, min_inliers=12) is None

    # A moving object over a static support, or a scene with no measured
    # motion, must not acquire an environmental attachment by proximity.
    for new_points, new_pose in [(before.base_xyz, _pose(0.02)), (before.base_xyz, _pose(0))]:
        assert (
            establish_environment_relationship(
                before,
                _tracks(1, new_points, pixels),
                _pose(0),
                new_pose,
                (mask,),
                context_radius_m=0.25,
                fit_threshold_m=0.003,
                min_motion_m=0.008,
                min_inliers=12,
                reference_version="v1",
            )
            is None
        )


def _camera_frame(rgb: np.ndarray, frame_id: int) -> RGBDFrame:
    return RGBDFrame(
        frame_id,
        rgb,
        np.full(rgb.shape[:2], 1000, dtype=np.uint16),
        np.array([[100.0, 0, 48.0], [0, 100.0, 48.0], [0, 0, 1.0]]),
        0.001,
        np.eye(4),
        frame_id / 30,
        frame_id / 30,
        "cal",
    )


def test_causal_service_uses_verified_scene_motion_when_object_points_disappear():
    rng = np.random.default_rng(12)
    patch = rng.integers(0, 255, (56, 56, 3), dtype=np.uint8)
    frames = []
    for index in range(4):
        rgb = np.zeros((96, 96, 3), dtype=np.uint8)
        rgb[20:76, 20 + 2 * index : 76 + 2 * index] = patch
        frames.append(_camera_frame(rgb, index))
    mask = np.zeros((96, 96), dtype=bool)
    mask[46:50, 46:50] = True  # Too small to seed object-only point motion.
    store = LatestPoseStore()
    service = CausalPoseService(
        store,
        image_scale=1.0,
        environment_min_inliers=12,
        environment_context_radius_m=0.25,
    )
    service.start()
    try:
        for index, frame in enumerate(frames):
            service.submit_frame(frame)
            if index < 3:
                observed = PoseEstimate(
                    "object",
                    _pose(0.02 * index),
                    frame.source_time_s,
                    frame.available_time_s,
                    Evidence.OBSERVED,
                    "v1",
                    (True,) * 3,
                    (True,) * 3,
                )
                service.submit_anchor(index, (observed,), {"object": mask})
        deadline = time.monotonic() + 3
        while service.environment_published < 1 and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.error is None
        assert service.environment_relationships == 1
        assert service.environment_published >= 1
        predicted = store.snapshot()[1]["object"]
        assert predicted.evidence == Evidence.PREDICTED
        assert predicted.estimator == "environment_group_motion"
        assert predicted.last_direct_observation_s == frames[2].source_time_s
        assert predicted.pose[0, 3] == pytest.approx(0.06, abs=0.005)
    finally:
        service.close()


def test_causal_service_marks_bounded_static_occlusion_as_prediction():
    rgb = np.random.default_rng(13).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    frames = [_camera_frame(rgb, index) for index in range(4)]
    mask = np.zeros((96, 96), dtype=bool)
    mask[46:48, 46:48] = True
    observed = PoseEstimate(
        "object",
        _pose(0.03),
        frames[0].source_time_s,
        frames[0].available_time_s,
        Evidence.OBSERVED,
        "v1",
        (True,) * 3,
        (True,) * 3,
    )
    store = LatestPoseStore()
    service = CausalPoseService(store, image_scale=1.0, static_prediction_max_s=0.05)
    service.start()
    try:
        service.submit_frame(frames[0])
        service.submit_anchor(0, (observed,), {"object": mask})
        service.submit_frame(frames[1])
        deadline = time.monotonic() + 3
        while service.frames < 2 and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.error is None
        predicted = store.snapshot()[1]["object"]
        assert predicted.evidence == Evidence.PREDICTED
        assert predicted.estimator == "stationary_prior"
        assert predicted.source_time_s == frames[1].source_time_s
        assert predicted.last_direct_observation_s == frames[0].source_time_s
        np.testing.assert_allclose(predicted.pose, observed.pose)

        service.submit_frame(frames[3])  # beyond the configured blind interval
        deadline = time.monotonic() + 3
        while service.frames < 3 and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.error is None
        assert store.snapshot()[1]["object"].source_time_s == frames[1].source_time_s
    finally:
        service.close()


def test_delayed_absolute_proposal_is_replayed_by_the_only_pose_owner():
    rgb = np.random.default_rng(15).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    frames = [_camera_frame(rgb, index) for index in range(4)]
    mask = np.zeros((96, 96), dtype=bool)
    mask[46:48, 46:48] = True  # No object-point motion fit is possible.
    observed = PoseEstimate(
        "object", _pose(0.03), 0.0, 0.0, Evidence.OBSERVED, "v1", (True,) * 3, (True,) * 3
    )
    lost = PoseEstimate(
        "object",
        None,
        frames[3].source_time_s,
        frames[3].available_time_s,
        Evidence.LOST,
        "v1",
        (False,) * 3,
        (False,) * 3,
    )
    store = LatestPoseStore()
    service = CausalPoseService(store, image_scale=1.0, static_prediction_max_s=1.0)
    service.start()
    try:
        for frame in frames:
            service.submit_frame(frame)
        service.submit_anchor(0, (observed,), {"object": mask})
        service.submit_anchor(3, (lost,), {})
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            current = store.snapshot()[1].get("object")
            if service.error is not None or (
                current is not None and current.source_time_s == frames[-1].source_time_s
            ):
                break
            time.sleep(0.01)
        assert service.error is None
        estimate = store.snapshot()[1]["object"]
        assert estimate.evidence == Evidence.PREDICTED
        assert estimate.estimator == "stationary_prior"
        assert estimate.source_time_s == frames[-1].source_time_s
        assert estimate.last_direct_observation_s == frames[0].source_time_s
    finally:
        service.close()


def test_scene_tracks_keep_identities_on_every_frame():
    image = np.random.default_rng(4).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    tracker = ScenePointTracker(image_scale=1.0)
    first = tracker.update(_camera_frame(image, 0))
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 2], [0, 1, 0]]), (96, 96))
    second = tracker.update(_camera_frame(shifted, 1))
    shared = np.intersect1d(first.ids, second.ids)
    assert len(shared) >= 12
    before = first.base_xyz[np.searchsorted(first.ids, shared)]
    after = second.base_xyz[np.searchsorted(second.ids, shared)]
    assert np.median(after[:, 0] - before[:, 0]) == pytest.approx(0.02, abs=0.004)

    interrupted = _camera_frame(shifted, 2)
    interrupted = RGBDFrame(
        interrupted.frame_id,
        interrupted.rgb,
        interrupted.depth,
        interrupted.intrinsics,
        interrupted.depth_m_per_unit,
        interrupted.base_from_camera,
        2.0,
        2.0,
        interrupted.calibration_version,
    )
    after_gap = tracker.update(interrupted)
    assert not np.intersect1d(second.ids, after_gap.ids).size
    assert tracker.gaps == 1


def test_recorded_moving_tray_group_predicts_ring_but_not_independent_motion():
    """The old run supplies measured scene tracks, not synthetic pixel shifts."""
    with np.load(RECORDED_MOTION) as recorded:

        def snapshot(frame_id: int) -> SceneTracks:
            return SceneTracks(
                frame_id,
                float(recorded[f"time_{frame_id}"]),
                recorded[f"ids_{frame_id}"],
                recorded[f"xy_{frame_id}"],
                recorded[f"xyz_{frame_id}"],
            )

        before, after = snapshot(331), snapshot(360)
        relation = establish_environment_relationship(
            before,
            after,
            recorded["pose_331"],
            recorded["pose_360"],
            (recorded["mask_331"],),
            context_radius_m=0.25,
            fit_threshold_m=0.005,
            min_motion_m=0.008,
            min_inliers=12,
            reference_version="ring-v1",
        )
        assert relation is not None and len(relation.ids) >= 12
        for frame_id in (390, 422):
            predicted = relation.predict(snapshot(frame_id), fit_threshold_m=0.005, min_inliers=12)
            assert predicted is not None
            assert np.linalg.norm(predicted[:3, 3] - recorded[f"pose_{frame_id}"][:3, 3]) < 0.005

        independently_moved = recorded["pose_360"].copy()
        independently_moved[0, 3] += 0.04
        assert (
            establish_environment_relationship(
                before,
                after,
                recorded["pose_331"],
                independently_moved,
                (recorded["mask_331"],),
                context_radius_m=0.25,
                fit_threshold_m=0.005,
                min_motion_m=0.008,
                min_inliers=12,
                reference_version="ring-v1",
            )
            is None
        )
