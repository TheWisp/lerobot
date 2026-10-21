"""A delayed direct fit must compete with source-time causal point motion."""

from collections import deque
from pathlib import Path

import numpy as np
import pytest

from lerobot.policies.flash.causal_points import CausalRigidPointTracker, RigidPointMotion
from lerobot.policies.flash.causal_service import CausalPoseService, _Anchor
from lerobot.policies.flash.perception import LatestPoseStore, RGBDFrame
from lerobot.policies.flash.pose_disagreement import PoseComparisonVerdict
from lerobot.policies.flash.scene_groups import SceneTracks
from lerobot.policies.flash.symmetry import AxialSymmetry
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


def _recorded_service(role: str):
    """One real insertion RGB-D frame, with a previously anchored point track."""
    fixture = Path(__file__).parent / "fixtures" / "direct_point_contact_20261021.npz"
    with np.load(fixture) as recorded:
        frame_id = int(recorded["frame_id"])
        depth = recorded["depth"]
        intrinsics = recorded["intrinsics"]
        points = recorded[f"{role}_points"]
        mask = recorded[f"{role}_mask"].astype(bool)
        seed_pose = recorded[f"{role}_seed_pose"]
        direct_pose = recorded[f"{role}_direct_pose"]
        motion_pose = recorded[f"{role}_motion"]
        pixels = recorded[f"{role}_pixels"]
    source_time_s = frame_id / 30
    source = RGBDFrame(
        frame_id,
        np.zeros((*depth.shape, 3), dtype=np.uint8),
        depth,
        intrinsics,
        0.001,
        np.eye(4),
        source_time_s,
        source_time_s,
        "recorded-camera",
    )
    scene = SceneTracks(
        frame_id, source_time_s, np.array([], dtype=np.int64), np.empty((0, 2), dtype=int), np.empty((0, 3))
    )
    store = LatestPoseStore()
    seed = PoseEstimate(
        role, seed_pose, 430 / 30, 430 / 30, Evidence.OBSERVED, "object-v1", (True,) * 3, (True,) * 3
    )
    predicted = PoseEstimate(
        role,
        motion_pose @ seed_pose,
        source_time_s,
        source_time_s,
        Evidence.PREDICTED,
        "object-v1",
        (True,) * 3,
        (True,) * 3,
        estimator="object_point_motion",
        last_direct_observation_s=seed.source_time_s,
    )
    store.publish((predicted,))
    comparisons = []
    service = CausalPoseService(
        store,
        fit_threshold_m=0.005,
        object_points_by_role={role: points},
        on_point_comparison=lambda name, comparison: comparisons.append((name, comparison)),
    )
    service.history.append(source)
    service.scene_history.append(scene)
    service.active[role] = (CausalRigidPointTracker(min_inliers=12), seed)
    service.motion_history[role] = deque(
        [
            (
                frame_id,
                RigidPointMotion(
                    motion_pose, source_time_s, source_time_s, len(pixels), len(pixels), 18, 0.001
                ),
                pixels,
            )
        ]
    )
    direct = PoseEstimate(
        role,
        direct_pose,
        source_time_s,
        source_time_s + 0.02,
        Evidence.OBSERVED,
        "object-v1",
        (True,) * 3,
        (True,) * 3,
        estimator="recorded_direct_pose",
    )
    return service, store, source, mask, direct, predicted, comparisons


def test_real_contact_conflict_keeps_point_prediction_and_records_direct_candidate():
    service, store, frame, mask, direct, predicted, comparisons = _recorded_service("dowel")

    service._anchor(
        _Anchor(frame.frame_id, (direct,), {"dowel": mask}, frame.source_time_s, frame.calibration_version)
    )

    assert store.snapshot()[1]["dowel"] is predicted
    assert service.anchor_outcomes_by_role["dowel"]["direct_point_unresolved"] == 1
    assert comparisons[0][0] == "dowel"
    assert comparisons[0][1].verdict == PoseComparisonVerdict.AMBIGUOUS
    assert comparisons[0][1].object_center_difference_m > 0.01


def test_same_rule_accepts_recorded_ring_correction():
    service, store, frame, mask, direct, _, comparisons = _recorded_service("ring")

    service._anchor(
        _Anchor(frame.frame_id, (direct,), {"ring": mask}, frame.source_time_s, frame.calibration_version)
    )

    accepted = store.snapshot()[1]["ring"]
    assert accepted.evidence == Evidence.OBSERVED
    assert accepted.estimator == "recorded_direct_pose"
    assert service.anchor_outcomes_by_role["ring"]["direct_published"] == 1
    assert comparisons[0][1].verdict == PoseComparisonVerdict.AGREE


def test_unmatched_point_pixels_cannot_veto_direct_reacquisition():
    service, store, frame, mask, direct, _, comparisons = _recorded_service("dowel")
    frame_id, motion, pixels = service.motion_history["dowel"][0]
    service.motion_history["dowel"][0] = (frame_id, motion, np.zeros_like(pixels))

    service._anchor(
        _Anchor(frame.frame_id, (direct,), {"dowel": mask}, frame.source_time_s, frame.calibration_version)
    )

    assert store.snapshot()[1]["dowel"].estimator == "recorded_direct_pose"
    assert not comparisons


def test_axial_spin_is_ignored_in_published_point_pose_and_direct_comparison():
    identity = np.eye(4)
    turn = np.eye(4)
    turn[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    symmetry = AxialSymmetry("object", np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), identity)
    store = LatestPoseStore()
    service = CausalPoseService(store, axial_symmetries=(symmetry,))
    anchor = PoseEstimate("object", identity, 0.0, 0.0, Evidence.OBSERVED, "v1", (True,) * 3, (True,) * 3)
    motion = RigidPointMotion(turn, 1 / 30, 1 / 30, 20, 20, 20, 0.0)

    service._publish("object", anchor, motion)

    estimate = store.snapshot()[1]["object"]
    np.testing.assert_allclose(estimate.pose, identity, atol=1e-8)

    x, y = np.meshgrid(np.linspace(-0.04, 0.04, 10), np.linspace(-0.04, 0.04, 10))
    model = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))
    frame = RGBDFrame(
        1,
        np.zeros((96, 96, 3), dtype=np.uint8),
        np.full((96, 96), 1000, dtype=np.uint16),
        np.array([[100.0, 0.0, 48.0], [0.0, 100.0, 48.0], [0.0, 0.0, 1.0]]),
        0.001,
        identity,
        1 / 30,
        1 / 30,
        "camera-v1",
    )
    service.object_points_by_role["object"] = model
    service.active["object"] = (CausalRigidPointTracker(min_inliers=12), anchor)
    service.motion_history["object"] = deque([(frame.frame_id, motion, np.tile([48, 48], (20, 1)))])
    direct = PoseEstimate(
        "object",
        identity,
        frame.source_time_s,
        frame.available_time_s,
        Evidence.OBSERVED,
        "v1",
        (True,) * 3,
        (True,) * 3,
    )
    assert not service._direct_conflicts_with_point_motion(
        direct, frame, np.ones(frame.depth.shape, dtype=bool)
    )
    assert service.last_direct_point_comparison["object"].verdict == PoseComparisonVerdict.AGREE


def test_same_frame_observation_supersedes_prediction_and_stays_observed():
    store = LatestPoseStore()
    identity = np.eye(4)
    predicted = PoseEstimate("object", identity, 1.0, 1.0, Evidence.PREDICTED, "v1", (True,) * 3, (True,) * 3)
    observed = PoseEstimate("object", identity, 1.0, 1.02, Evidence.OBSERVED, "v1", (True,) * 3, (True,) * 3)
    late_prediction = PoseEstimate(
        "object", identity, 1.0, 1.03, Evidence.PREDICTED, "v1", (True,) * 3, (True,) * 3
    )

    assert store.publish((predicted,)) == 1
    assert store.publish((observed,)) == 2
    assert store.publish((late_prediction,)) == 2
    assert store.snapshot()[1]["object"] is observed


def test_duplicate_symmetry_role_is_rejected():
    symmetry = AxialSymmetry("object", np.array([0.0, 0.0, 1.0]), np.zeros(3), np.eye(4))
    with pytest.raises(ValueError, match="at most one axial symmetry"):
        CausalPoseService(LatestPoseStore(), axial_symmetries=(symmetry, symmetry))
