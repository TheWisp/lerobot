"""Environment motion must be measured, associated and allowed to abstain."""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from lerobot.policies.flash.causal_service import CausalPoseService, _Anchor
from lerobot.policies.flash.perception import LatestPoseStore, RGBDFrame
from lerobot.policies.flash.scene_groups import (
    EnvironmentRelationship,
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


def test_scene_group_support_snapshot_uses_active_point_ids_only():
    service = CausalPoseService(LatestPoseStore())
    points = np.array([[0.1, 0, 1], [0.2, 0, 1], [0.3, 0, 1]])
    pixels = np.array([[10, 10], [20, 10], [30, 10]])
    service.scene_history.append(SceneTracks(4, 4 / 30, np.array([1, 2, 3]), pixels, points))
    service.relationships["object"] = EnvironmentRelationship(
        np.array([1, 3, 8]), points.copy(), _pose(0), 0, "v1", 4 / 30
    )

    support = service.scene_group_supports()["object"]
    assert support["verified"] is True
    assert support["measured_this_frame"] is True
    np.testing.assert_array_equal(support["ids"], [1, 3])
    np.testing.assert_array_equal(support["xy"], pixels[[0, 2]])
    np.testing.assert_array_equal(support["base_xyz"], points[[0, 2]])


def test_scene_support_callback_reports_the_same_group_used_on_that_camera_frame(monkeypatch):
    captured = []
    service = CausalPoseService(
        LatestPoseStore(), on_scene_support=lambda frame_id, groups: captured.append((frame_id, groups))
    )
    rng = np.random.default_rng(4)
    points = rng.uniform((0.0, 0.0, 0.9), (0.1, 0.1, 1.1), (12, 3))
    pixels = np.column_stack((np.arange(12) + 20, np.arange(12) + 30))
    tracks = SceneTracks(1, 1 / 30, np.arange(12), pixels, points)
    monkeypatch.setattr(service.scene, "update", lambda _frame: tracks)
    service.relationships["object"] = EnvironmentRelationship(
        tracks.ids.copy(), points.copy(), _pose(0), 0, "v1", 0
    )
    frame = RGBDFrame(
        1,
        np.zeros((64, 64, 3), dtype=np.uint8),
        np.ones((64, 64), dtype=np.uint16),
        np.eye(3),
        0.001,
        np.eye(4),
        1 / 30,
        1 / 30,
        "camera-v1",
    )

    service._frame(frame)

    assert captured[0][0] == frame.frame_id
    assert captured[0][1]["object"]["verified"] is True
    np.testing.assert_array_equal(captured[0][1]["object"]["ids"], tracks.ids)
    np.testing.assert_array_equal(captured[0][1]["object"]["xy"], tracks.xy)


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


def test_group_bridge_is_same_when_first_direct_pose_arrives_after_second():
    rng = np.random.default_rng(81)
    points = rng.uniform((-0.06, -0.06, 0.98), (0.06, 0.06, 1.02), (40, 3))
    pixels = np.column_stack((rng.integers(12, 84, 40), rng.integers(12, 84, 40)))
    scenes = [_tracks(index, points + [index * 0.02, 0, 0], pixels) for index in range(3)]
    rgb = np.zeros((96, 96, 3), dtype=np.uint8)
    frames = [_camera_frame(rgb, index) for index in range(3)]
    mask = np.zeros((96, 96), dtype=bool)
    mask[pixels[:4, 1], pixels[:4, 0]] = True

    def run(order):
        store = LatestPoseStore()
        decisions = []
        support_events = []
        service = CausalPoseService(
            store,
            image_scale=1.0,
            on_anchor_decision=decisions.append,
            on_scene_support=lambda frame_id, groups: support_events.append((frame_id, groups)),
        )
        service.scene.update = lambda item: scenes[item.frame_id]
        for kind, index in order:
            if kind == "frame":
                service._frame(frames[index])
            else:
                estimate = PoseEstimate(
                    "object",
                    _pose(0.02 * index),
                    frames[index].source_time_s,
                    frames[index].available_time_s,
                    Evidence.OBSERVED,
                    "v1",
                    (True,) * 3,
                    (True,) * 3,
                )
                service._anchor(_Anchor(index, (estimate,), {"object": mask}))
        return service, store.snapshot()[1]["object"], decisions, support_events

    ordered, ordered_pose, _, _ = run(
        [("frame", 0), ("direct", 0), ("frame", 1), ("direct", 1), ("frame", 2)]
    )
    reordered, reordered_pose, decisions, support_events = run(
        [("frame", 0), ("frame", 1), ("direct", 1), ("frame", 2), ("direct", 0)]
    )
    assert "object" in ordered.relationships
    assert "object" in reordered.relationships
    assert ordered_pose.estimator == reordered_pose.estimator == "environment_group_motion"
    assert ordered_pose.source_frame_id == reordered_pose.source_frame_id == 2
    np.testing.assert_allclose(ordered_pose.pose, reordered_pose.pose, atol=1e-6)
    assert decisions[-1].outcome == "accepted_historical_group"
    assert reordered_pose.available_time_s >= decisions[-1].decided_at_s
    assert support_events[-1][0] == frames[2].frame_id
    assert support_events[-1][1]["object"]["verified"]
    assert support_events[-1][1]["object"]["ids"].size >= reordered.environment_min_inliers


def test_late_group_bridge_rejects_intervening_direct_pose_contradiction():
    rng = np.random.default_rng(82)
    points = rng.uniform((-0.06, -0.06, 0.98), (0.06, 0.06, 1.02), (40, 3))
    pixels = np.column_stack((rng.integers(12, 84, 40), rng.integers(12, 84, 40)))
    scenes = [_tracks(index, points + [index * 0.02, 0, 0], pixels) for index in range(5)]
    rgb = np.zeros((96, 96, 3), dtype=np.uint8)
    frames = [_camera_frame(rgb, index) for index in range(5)]
    mask = np.zeros((96, 96), dtype=bool)
    mask[pixels[:4, 1], pixels[:4, 0]] = True
    store = LatestPoseStore()
    decisions = []
    service = CausalPoseService(store, image_scale=1.0, on_anchor_decision=decisions.append)
    service.scene.update = lambda item: scenes[item.frame_id]

    for frame in frames:
        service._frame(frame)
    for index, x in ((1, 0.02), (2, 0.055), (3, 0.06), (0, 0.0)):
        estimate = PoseEstimate(
            "object",
            _pose(x),
            frames[index].source_time_s,
            frames[index].available_time_s,
            Evidence.OBSERVED,
            "v1",
            (True,) * 3,
            (True,) * 3,
        )
        service._anchor(_Anchor(index, (estimate,), {"object": mask}))

    service._drain_pending()

    assert "object" not in service.relationships
    assert decisions[-1].outcome == "replayed_accepted"
    assert len(service.direct_history["object"]) == 4
    assert store.snapshot()[1]["object"].source_time_s == frames[4].source_time_s
    assert store.snapshot()[1]["object"].estimator == "environment_support_hypothesis"


def test_unresolved_support_rotation_does_not_rotate_object_frame():
    """Small stationary support patches cannot justify a rigid-body tilt."""
    rng = np.random.default_rng(44)
    points = rng.uniform((-0.01, -0.01, 1.0), (0.01, 0.01, 1.0), (40, 3))
    pixels = np.column_stack((rng.integers(20, 76, 40), rng.integers(20, 76, 40)))
    angle = np.deg2rad(15)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    center = points.mean(axis=0)
    changed = (points - center) @ rotation.T + center
    relation = EnvironmentRelationship(np.arange(len(points)), points, _pose(0), 0.0, "v1", 0.0)
    predicted = relation.predict(_tracks(1, changed, pixels), fit_threshold_m=0.005, min_inliers=12)
    assert predicted is not None
    np.testing.assert_allclose(predicted[:3, :3], np.eye(3), atol=1e-8)
    np.testing.assert_allclose(predicted[:3, 3], _pose(0)[:3, 3], atol=0.001)


def test_stationary_support_requires_direct_agreement_and_surviving_context_points():
    rng = np.random.default_rng(31)
    points = rng.uniform((-0.1, -0.1, 0.98), (0.1, 0.1, 1.02), (40, 3))
    pixels = np.column_stack((rng.integers(20, 76, 40), rng.integers(20, 76, 40)))
    before = _tracks(0, points, pixels)
    after = _tracks(1, points.copy(), pixels)
    mask = np.zeros((96, 96), dtype=bool)
    mask[pixels[:4, 1], pixels[:4, 0]] = True
    options = {
        "context_radius_m": 0.25,
        "fit_threshold_m": 0.005,
        "min_motion_m": 0.008,
        "min_inliers": 12,
        "reference_version": "v1",
        "allow_stationary": True,
    }
    relation = establish_environment_relationship(before, after, _pose(0), _pose(0), (mask,), **options)
    assert relation is not None
    assert len(relation.ids) >= 12
    assert relation.predict(after, fit_threshold_m=0.005, min_inliers=12) is not None
    assert (
        establish_environment_relationship(before, after, _pose(0), _pose(0.02), (mask,), **options) is None
    )

    # The object frame can be far from its visible surface. A small rotation
    # around that surface moves the frame origin without moving the support.
    remote_center = np.array([0.0, 0.0, 0.4])
    baseline = _pose(0)
    baseline[:3, 3] -= remote_center
    theta = np.deg2rad(4)
    turned = baseline.copy()
    turned[:3, :3] = np.array(
        [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    )
    turned[:3, 3] = _pose(0)[:3, 3] - turned[:3, :3] @ remote_center
    assert np.linalg.norm(turned[:3, 3] - baseline[:3, 3]) > 0.02
    assert (
        establish_environment_relationship(
            before, after, baseline, turned, (mask,), **options, object_center_object_m=remote_center
        )
        is not None
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


def test_unverified_group_loss_uses_only_configured_bounded_stationary_prior():
    rgb = np.random.default_rng(22).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    store = LatestPoseStore()
    service = CausalPoseService(store, image_scale=1.0, static_prediction_max_s=0.05)
    first = _camera_frame(rgb, 0)
    service._frame(first)
    observed = PoseEstimate("object", _pose(0.03), 0, 0, Evidence.OBSERVED, "v1", (True,) * 3, (True,) * 3)
    service.observed["object"] = (observed, service.scene_history[-1], ())
    store.publish((observed,))
    # The provisional group's IDs have disappeared. It must not suppress the
    # independently bounded prior; a confirmed group still has no such fallback.
    lost_group = EnvironmentRelationship(
        np.arange(10000, 10012), np.zeros((12, 3)), observed.pose, 0, "v1", 0
    )
    service.candidate_relationships["object"] = lost_group
    service._frame(_camera_frame(rgb, 1))
    estimate = store.snapshot()[1]["object"]
    assert estimate.estimator == "stationary_prior"
    assert estimate.source_time_s == pytest.approx(1 / 30)
    assert estimate.last_direct_observation_s == 0
    service._frame(_camera_frame(rgb, 2))  # beyond the configured blind interval
    assert store.snapshot()[1]["object"].source_time_s == pytest.approx(1 / 30)

    confirmed_store = LatestPoseStore()
    confirmed = CausalPoseService(
        confirmed_store,
        image_scale=1.0,
        static_prediction_max_s=1.0,
        environment_prediction_max_s=0.05,
    )
    confirmed._frame(first)
    confirmed.observed["object"] = (observed, confirmed.scene_history[-1], ())
    confirmed_store.publish((observed,))
    confirmed.relationships["object"] = lost_group
    confirmed._frame(_camera_frame(rgb, 1))
    confirmed._frame(_camera_frame(rgb, 2))  # verified group expires
    assert confirmed_store.snapshot()[1]["object"].source_time_s == 0


@pytest.mark.parametrize("old_surface_depth_mm,expected", [(1200, Evidence.LOST), (900, Evidence.PREDICTED)])
def test_visible_moving_object_invalidates_only_depth_contradicted_static_prior(
    old_surface_depth_mm, expected
):
    """A failed direct fit cannot leave an object at a measured empty location."""
    rgb = np.zeros((80, 80, 3), dtype=np.uint8)
    camera = np.array([[100.0, 0, 40.0], [0, 100.0, 40.0], [0, 0, 1.0]])
    first_depth = np.full((80, 80), 1200, dtype=np.uint16)
    first_depth[25:45, 15:35] = 1000
    first_mask = np.zeros((80, 80), dtype=bool)
    first_mask[25:45, 15:35] = True
    moved_depth = np.full((80, 80), old_surface_depth_mm, dtype=np.uint16)
    moved_depth[25:45, 45:65] = 1000
    moved_mask = np.zeros((80, 80), dtype=bool)
    moved_mask[25:45, 45:65] = True
    frames = [
        RGBDFrame(i, rgb, depth, camera, 0.001, np.eye(4), i / 30, i / 30, "camera")
        for i, depth in enumerate((first_depth, moved_depth))
    ]
    observed = PoseEstimate("object", np.eye(4), 0, 0, Evidence.OBSERVED, "v1", (True,) * 3, (True,) * 3)
    lost = PoseEstimate("object", None, 1 / 30, 1 / 30, Evidence.LOST, "v1", (False,) * 3, (False,) * 3)
    service = CausalPoseService(LatestPoseStore(), static_prediction_max_s=1.0)
    service.process_offline(frames[0], (observed,), {"object": first_mask})
    result = service.process_offline(frames[1], (lost,), {"object": moved_mask})["object"]
    assert result.evidence == expected
    if expected == Evidence.LOST:
        assert result.estimator == "stationary_prior_contradicted"
        assert "object" not in service.observed
    else:
        assert result.estimator == "stationary_prior"


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
        while (
            (
                service.environment_relationships < 1
                or store.snapshot()[1].get("object") is None
                or store.snapshot()[1]["object"].source_time_s < frames[3].source_time_s
            )
            and service.error is None
            and time.monotonic() < deadline
        ):
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


def test_verified_group_defers_duplicate_point_replay_until_grasp():
    rgb = np.random.default_rng(48).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=bool)
    mask[30:65, 30:65] = True
    service = CausalPoseService(LatestPoseStore(), image_scale=1.0)
    service.start()
    try:
        for index in range(3):
            frame = _camera_frame(rgb, index)
            service.submit_frame(frame)
            service.submit_anchor(
                index,
                (
                    PoseEstimate(
                        "object",
                        _pose(0),
                        frame.source_time_s,
                        frame.available_time_s,
                        Evidence.OBSERVED,
                        "v1",
                        (True,) * 3,
                        (True,) * 3,
                    ),
                ),
                {"object": mask},
            )
        deadline = time.monotonic() + 3
        while (
            service.anchor_outcomes_by_role.get("object", {}).get("standby_point_seeded", 0) == 0
            and service.error is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert service.error is None
        assert "object" in service.relationships
        assert "object" in service.pending
        replay_before = service.replay_compute_total_s
        for index in range(3, 8):
            service.submit_frame(_camera_frame(rgb, index))
        while service.frames < 8 and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.error is None
        assert service.replay_compute_total_s == replay_before
        assert "object" in service.pending

        service.set_attached_role("object", source_time_s=8 / 30)
        service.submit_frame(_camera_frame(rgb, 8))
        while (
            service.replay_compute_total_s == replay_before
            and service.error is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert service.error is None
        assert service.replay_compute_total_s > replay_before
    finally:
        service.close()


def test_stationary_support_stays_measured_until_grasp_then_is_withdrawn():
    rgb = np.random.default_rng(32).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=bool)
    mask[46:50, 46:50] = True
    store = LatestPoseStore()
    service = CausalPoseService(store, image_scale=1.0, environment_prediction_max_s=1.0)
    service.start()
    try:
        for index in range(3):
            frame = _camera_frame(rgb, index)
            service.submit_frame(frame)
            observed = PoseEstimate(
                "object",
                _pose(0),
                frame.source_time_s,
                frame.available_time_s,
                Evidence.OBSERVED,
                "v1",
                (True,) * 3,
                (True,) * 3,
            )
            service.submit_anchor(index, (observed,), {"object": mask})
        deadline = time.monotonic() + 5
        while service.environment_relationships < 1 and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.error is None
        assert service.environment_relationships == 1

        for index in range(3, 46):
            service.submit_frame(_camera_frame(rgb, index))
            while (
                store.snapshot()[1]["object"].source_time_s < index / 30
                and service.error is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.002)
        assert service.error is None
        assert service.frames == 46
        predicted = store.snapshot()[1]["object"]
        assert predicted.estimator == "environment_group_motion"
        assert predicted.source_time_s == pytest.approx(45 / 30)
        assert predicted.last_direct_observation_s == pytest.approx(2 / 30)

        published = service.environment_published
        service.set_attached_role("object", source_time_s=46 / 30)
        service.submit_frame(_camera_frame(rgb, 46))
        while service.frames < 47 and service.error is None and time.monotonic() < deadline:
            time.sleep(0.002)
        assert service.error is None
        assert "object" not in service.relationships
        assert service.environment_published == published
    finally:
        service.close()


def test_single_visible_pose_can_seed_a_labeled_support_hypothesis():
    rgb = np.random.default_rng(33).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=bool)
    mask[46:50, 46:50] = True
    store = LatestPoseStore()
    service = CausalPoseService(store, image_scale=1.0)
    service.start()
    try:
        first = _camera_frame(rgb, 0)
        service.submit_frame(first)
        service.submit_anchor(
            0,
            (PoseEstimate("object", _pose(0), 0, 0, Evidence.OBSERVED, "v1", (True,) * 3, (True,) * 3),),
            {"object": mask},
        )
        service.submit_frame(_camera_frame(rgb, 1))
        deadline = time.monotonic() + 3
        while (
            service.environment_candidate_published_by_role.get("object", 0) < 1
            and service.error is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert service.error is None
        estimate = store.snapshot()[1]["object"]
        assert estimate.evidence == Evidence.PREDICTED
        assert estimate.estimator == "environment_support_hypothesis"
        assert estimate.last_direct_observation_s == first.source_time_s
        service.set_attached_role("object", source_time_s=2 / 30)
        service.submit_frame(_camera_frame(rgb, 2))
        while service.frames < 3 and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "object" not in service.candidate_relationships
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
    service = CausalPoseService(
        store, image_scale=1.0, static_prediction_max_s=0.05, environment_min_inliers=1000
    )
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
    service = CausalPoseService(
        store, image_scale=1.0, static_prediction_max_s=1.0, environment_min_inliers=1000
    )
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


def test_delayed_direct_measurement_refreshes_a_newer_stationary_prediction():
    rgb = np.random.default_rng(16).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    frames = [_camera_frame(rgb, index) for index in range(4)]
    mask = np.zeros((96, 96), dtype=bool)
    mask[46:48, 46:48] = True  # No object-point seed; only a scene-time prediction.
    store = LatestPoseStore()
    service = CausalPoseService(
        store, image_scale=1.0, static_prediction_max_s=1.0, environment_min_inliers=1000
    )
    service.start()
    try:
        service.submit_frame(frames[0])
        first = PoseEstimate(
            "object",
            _pose(0.03),
            frames[0].source_time_s,
            frames[0].available_time_s,
            Evidence.OBSERVED,
            "v1",
            (True,) * 3,
            (True,) * 3,
        )
        service.submit_anchor(0, (first,), {"object": mask})
        for frame in frames[1:]:
            service.submit_frame(frame)
        deadline = time.monotonic() + 3
        while service.frames < len(frames) and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.error is None
        assert store.snapshot()[1]["object"].estimator == "stationary_prior"
        second = PoseEstimate(
            "object",
            _pose(0.04),
            frames[1].source_time_s,
            frames[1].available_time_s,
            Evidence.OBSERVED,
            "v1",
            (True,) * 3,
            (True,) * 3,
        )
        service.submit_anchor(1, (second,), {"object": mask})
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            current = store.snapshot()[1]["object"]
            if current.last_direct_observation_s == frames[1].source_time_s or service.error is not None:
                break
            time.sleep(0.01)
        assert service.error is None
        current = store.snapshot()[1]["object"]
        assert current.source_time_s == frames[-1].source_time_s
        assert current.last_direct_observation_s == frames[1].source_time_s
        assert current.pose[0, 3] == pytest.approx(0.04)
        assert service.anchor_outcomes_by_role["object"]["direct_rebased"] == 1
    finally:
        service.close()


def test_older_delayed_proposal_cannot_erase_newer_direct_observation():
    rgb = np.random.default_rng(18).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    frames = [_camera_frame(rgb, index) for index in range(3)]
    mask = np.zeros((96, 96), dtype=bool)
    mask[46:48, 46:48] = True  # No object-point motion fit is possible.
    newer = PoseEstimate(
        "object",
        _pose(0.04),
        frames[1].source_time_s,
        frames[1].available_time_s,
        Evidence.OBSERVED,
        "v1",
        (True,) * 3,
        (True,) * 3,
    )
    older = PoseEstimate(
        "object",
        _pose(0),
        frames[0].source_time_s,
        frames[0].available_time_s,
        Evidence.OBSERVED,
        "v1",
        (True,) * 3,
        (True,) * 3,
    )
    store = LatestPoseStore()
    service = CausalPoseService(store, image_scale=1.0, static_prediction_max_s=1.0)
    service.start()
    try:
        for frame in frames:
            service.submit_frame(frame)
        service.submit_anchor(1, (newer,), {"object": mask})
        deadline = time.monotonic() + 3
        while service.direct_published < 1 and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.error is None
        assert service.direct_published == 1
        service.submit_anchor(0, (older,), {"object": mask})
        deadline = time.monotonic() + 3
        while service.event_replays < 1 and service.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert service.error is None
        assert service.event_replays == 1
        estimate = store.snapshot()[1]["object"]
        assert estimate.last_direct_observation_s == frames[1].source_time_s
        assert estimate.pose[0, 3] == pytest.approx(0.04)
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

    missing = _camera_frame(shifted, 2)
    missing = RGBDFrame(
        missing.frame_id,
        missing.rgb,
        np.zeros_like(missing.depth),
        missing.intrinsics,
        missing.depth_m_per_unit,
        missing.base_from_camera,
        missing.source_time_s,
        missing.available_time_s,
        missing.calibration_version,
    )
    assert len(tracker.update(missing).ids) == 0
    restored = tracker.update(_camera_frame(shifted, 3))
    assert len(np.intersect1d(second.ids, restored.ids)) >= 12

    interrupted = _camera_frame(shifted, 4)
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
    assert len(np.intersect1d(second.ids, after_gap.ids)) >= 12
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
