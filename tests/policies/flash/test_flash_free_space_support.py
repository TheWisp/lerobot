"""A candidate cannot occupy measured free space, even without a point prior."""

from pathlib import Path

import numpy as np

from lerobot.policies.flash.causal_service import CausalPoseService, _Anchor
from lerobot.policies.flash.perception import LatestPoseStore, RGBDFrame
from lerobot.policies.flash.pose_disagreement import FreeSpaceVerdict, free_space_support
from lerobot.policies.flash.scene_groups import SceneTracks
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


def _frame(depth: np.ndarray, intrinsics: np.ndarray, frame_id: int) -> RGBDFrame:
    return RGBDFrame(
        frame_id,
        np.zeros((*depth.shape, 3), dtype=np.uint8),
        depth,
        intrinsics,
        0.001,
        np.eye(4),
        frame_id / 30,
        frame_id / 30,
        "recorded-camera",
    )


def test_recorded_dowel_initialization_branch_is_contradicted_but_ring_is_not():
    fixture = Path(__file__).parent / "fixtures" / "free_space_insertion_20261021.npz"
    with np.load(fixture) as recorded:
        frame = _frame(recorded["depth"], recorded["intrinsics"], int(recorded["frame_id"]))
        dowel_good = free_space_support(
            frame,
            recorded["dowel_points"],
            recorded["dowel_sealed"],
            depth_tolerance_m=0.005,
            max_local_depth_span_m=0.01,
        )
        dowel_bad = free_space_support(
            frame,
            recorded["dowel_points"],
            recorded["dowel_sam3_initial"],
            depth_tolerance_m=0.005,
            max_local_depth_span_m=0.01,
        )
        ring_good = free_space_support(
            frame,
            recorded["ring_points"],
            recorded["ring_sealed"],
            depth_tolerance_m=0.005,
            max_local_depth_span_m=0.01,
        )
        ring_alternative = free_space_support(
            frame,
            recorded["ring_points"],
            recorded["ring_sam3_initial"],
            depth_tolerance_m=0.005,
            max_local_depth_span_m=0.01,
        )

    assert dowel_good.valid_depth_pixels > 100
    assert dowel_bad.valid_depth_pixels > 100
    assert dowel_good.median_signed_gap_m < 0.01
    assert dowel_bad.median_signed_gap_m > 0.05
    assert dowel_good.contradiction_fraction < 0.5
    assert dowel_bad.contradiction_fraction > 0.7
    assert ring_good.median_signed_gap_m < 0.01
    assert ring_alternative.median_signed_gap_m < 0.01


def test_nearer_occluder_is_not_a_free_space_contradiction():
    intrinsics = np.array([[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]])
    x, y = np.meshgrid(np.linspace(-0.01, 0.01, 5), np.linspace(-0.01, 0.01, 5))
    points = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))
    occluded = free_space_support(
        _frame(np.full((21, 21), 800, dtype=np.uint16), intrinsics, 1),
        points,
        np.eye(4),
        depth_tolerance_m=0.005,
        max_local_depth_span_m=0.01,
    )
    free = free_space_support(
        _frame(np.full((21, 21), 1200, dtype=np.uint16), intrinsics, 1),
        points,
        np.eye(4),
        depth_tolerance_m=0.005,
        max_local_depth_span_m=0.01,
    )

    assert occluded.median_signed_gap_m < -0.1
    assert occluded.contradiction_fraction == 0
    assert free.median_signed_gap_m > 0.1
    assert free.contradiction_fraction == 1


def test_causal_service_records_both_pose_branches_without_changing_pose_ownership():
    fixture = Path(__file__).parent / "fixtures" / "free_space_insertion_20261021.npz"
    with np.load(fixture) as recorded:
        frame_id = int(recorded["frame_id"])
        frame = _frame(recorded["depth"], recorded["intrinsics"], frame_id)
        points = recorded["dowel_points"]
        good_pose = recorded["dowel_sealed"]
        bad_pose = recorded["dowel_sam3_initial"]
    store = LatestPoseStore()
    comparisons = []
    service = CausalPoseService(
        store,
        fit_threshold_m=0.005,
        max_pose_disagreement_translation_m=0.01,
        object_points_by_role={"object": points},
        on_free_space_comparison=lambda role, result: comparisons.append((role, result)),
    )

    for pose in (good_pose, bad_pose):
        estimate = PoseEstimate(
            "object",
            pose,
            frame.source_time_s,
            frame.available_time_s,
            Evidence.OBSERVED,
            "reference-v1",
            (True,) * 3,
            (True,) * 3,
        )
        service._record_direct_free_space(estimate, frame)

    assert [result.verdict for _, result in comparisons] == [
        FreeSpaceVerdict.NOT_CONTRADICTED,
        FreeSpaceVerdict.CONTRADICTED,
    ]
    assert service.direct_free_space_counts_by_role["object"] == {
        "not_contradicted": 1,
        "contradicted": 1,
    }
    assert not store.snapshot()[1]


def test_recorded_pose_in_measured_empty_space_cannot_replace_group_prediction():
    fixture = Path(__file__).parent / "fixtures" / "free_space_insertion_20261021.npz"
    with np.load(fixture) as recorded:
        frame = _frame(recorded["depth"], recorded["intrinsics"], int(recorded["frame_id"]))
        points = recorded["dowel_points"]
        good_pose = recorded["dowel_sealed"]
        bad_pose = recorded["dowel_sam3_initial"]
    scene = SceneTracks(
        frame.frame_id,
        frame.source_time_s,
        np.array([], dtype=np.int64),
        np.empty((0, 2), dtype=int),
        np.empty((0, 3)),
    )
    predicted = PoseEstimate(
        "object",
        good_pose,
        frame.source_time_s,
        frame.source_time_s,
        Evidence.PREDICTED,
        "reference-v1",
        (True,) * 3,
        (True,) * 3,
        estimator="environment_group_motion",
        last_direct_observation_s=frame.source_time_s - 0.1,
    )
    store = LatestPoseStore()
    store.publish((predicted,))
    service = CausalPoseService(
        store,
        fit_threshold_m=0.005,
        max_pose_disagreement_translation_m=0.01,
        object_points_by_role={"object": points},
    )
    service.history.append(frame)
    service.scene_history.append(scene)
    mask = np.ones(frame.depth.shape, dtype=bool)
    bad = PoseEstimate(
        "object",
        bad_pose,
        frame.source_time_s,
        frame.source_time_s,
        Evidence.OBSERVED,
        "reference-v1",
        (True,) * 3,
        (True,) * 3,
        estimator="appearance_rgbd",
    )

    service._anchor(_Anchor(frame.frame_id, (bad,), {"object": mask}, frame.source_time_s, "recorded-camera"))

    assert store.snapshot()[1]["object"] is predicted
    assert service.anchor_outcomes_by_role["object"]["direct_free_space_contradicted"] == 1
    assert service.last_direct_free_space_comparison["object"].verdict == FreeSpaceVerdict.CONTRADICTED

    good = PoseEstimate(
        "object",
        good_pose,
        frame.source_time_s,
        frame.source_time_s,
        Evidence.OBSERVED,
        "reference-v1",
        (True,) * 3,
        (True,) * 3,
        estimator="appearance_rgbd",
    )
    service._anchor(
        _Anchor(frame.frame_id, (good,), {"object": mask}, frame.source_time_s, "recorded-camera")
    )
    assert store.snapshot()[1]["object"].evidence == Evidence.OBSERVED
    assert service.last_direct_free_space_comparison["object"].verdict == FreeSpaceVerdict.NOT_CONTRADICTED


def test_translation_observation_keeps_position_when_predicted_rotation_conflicts():
    intrinsics = np.array([[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]])
    x, y = np.meshgrid(np.linspace(-0.1, 0.1, 21), np.linspace(-0.1, 0.1, 21))
    points = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))
    frame = _frame(np.full((21, 21), 1200, dtype=np.uint16), intrinsics, 1)
    store = LatestPoseStore()
    service = CausalPoseService(
        store,
        fit_threshold_m=0.005,
        max_pose_disagreement_translation_m=0.01,
        object_points_by_role={"object": points},
    )
    service.history.append(frame)
    service.scene_history.append(
        SceneTracks(
            1,
            frame.source_time_s,
            np.array([], dtype=np.int64),
            np.empty((0, 2), dtype=int),
            np.empty((0, 3)),
        )
    )
    partial = PoseEstimate(
        "object",
        np.eye(4),
        frame.source_time_s,
        frame.source_time_s,
        Evidence.PREDICTED,
        "reference-v1",
        (True,) * 3,
        (True,) * 3,
        direct_translation_observed=True,
    )

    service._anchor(
        _Anchor(
            1,
            (partial,),
            {"object": np.ones(frame.depth.shape, bool)},
            frame.source_time_s,
            "recorded-camera",
        )
    )

    assert service.last_direct_free_space_comparison["object"].verdict == FreeSpaceVerdict.CONTRADICTED
    assert store.snapshot()[1]["object"].direct_translation_observed


def test_new_object_uses_first_accepted_depth_surface_to_reject_later_jump():
    intrinsics = np.array([[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]])
    depth = np.full((21, 21), 1200, dtype=np.uint16)
    depth[7:14, 7:14] = 1000
    mask = np.zeros(depth.shape, dtype=bool)
    mask[7:14, 7:14] = True
    frame = _frame(depth, intrinsics, 1)
    store = LatestPoseStore()
    service = CausalPoseService(
        store,
        fit_threshold_m=0.005,
        max_pose_disagreement_translation_m=0.01,
    )
    service.history.append(frame)
    service.scene_history.append(
        SceneTracks(
            1, frame.source_time_s, np.array([], dtype=np.int64), np.empty((0, 2), int), np.empty((0, 3))
        )
    )
    first = PoseEstimate(
        "object",
        np.eye(4),
        frame.source_time_s,
        frame.source_time_s,
        Evidence.OBSERVED,
        "local-reference",
        (True,) * 3,
        (True,) * 3,
        estimator="surface_rgbd",
    )
    service._anchor(_Anchor(1, (first,), {"object": mask}, frame.source_time_s, "recorded-camera"))
    assert "object" in service.static_support_points_by_role

    shifted_pose = np.eye(4)
    shifted_pose[0, 3] = 0.08
    shifted = PoseEstimate(
        "object",
        shifted_pose,
        frame.source_time_s,
        frame.source_time_s,
        Evidence.OBSERVED,
        "local-reference",
        (True,) * 3,
        (True,) * 3,
        estimator="surface_rgbd",
    )
    service._anchor(_Anchor(1, (shifted,), {"object": mask}, frame.source_time_s, "recorded-camera"))

    assert service.last_direct_free_space_comparison["object"].verdict == FreeSpaceVerdict.CONTRADICTED
    assert store.snapshot()[1]["object"].pose[0, 3] == 0
