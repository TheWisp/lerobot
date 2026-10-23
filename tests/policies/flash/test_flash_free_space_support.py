"""A candidate cannot occupy measured free space, even without a point prior."""

from pathlib import Path

import numpy as np

from lerobot.policies.flash.causal_service import CausalPoseService
from lerobot.policies.flash.perception import LatestPoseStore, RGBDFrame
from lerobot.policies.flash.pose_disagreement import FreeSpaceVerdict, free_space_support
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
