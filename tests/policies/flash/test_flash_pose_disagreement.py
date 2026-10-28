"""A held-view patch fit must not automatically override a conflicting group pose."""

from pathlib import Path

import numpy as np

from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.pose_disagreement import PoseComparisonVerdict, compare_pose_proposals


def test_equal_surface_support_does_not_claim_the_new_pose_is_better():
    frame = RGBDFrame(
        1,
        np.zeros((32, 32, 3), dtype=np.uint8),
        np.ones((32, 32), dtype=np.float32),
        np.array([[100.0, 0, 16], [0, 100.0, 16], [0, 0, 1]]),
        1.0,
        np.eye(4),
        1.0,
        1.0,
        "calibration",
    )
    points = np.array([(x, y, 1.0) for x in (-0.02, 0, 0.02) for y in (-0.02, 0, 0.02)])
    candidate = np.eye(4)
    candidate[0, 3] = 0.03
    result = compare_pose_proposals(
        frame,
        np.ones((32, 32), dtype=bool),
        points,
        candidate,
        np.eye(4),
        max_agreement_offset_m=0.01,
        max_agreement_angle_rad=0.1,
        min_masked_depth_points=3,
    )
    assert result.verdict == PoseComparisonVerdict.AMBIGUOUS


def test_recorded_held_dowel_proposals_conflict_across_depth_and_mask_support():
    fixture = Path(__file__).with_name("fixtures") / "dowel_held_frame_315.npz"
    with np.load(fixture) as data:
        points = data["model_points"]
        candidate, group = data["candidate"], data["group_pose"]
        mask, depth, intrinsics = data["mask"], data["depth"], data["intrinsics"]
    frame = RGBDFrame(
        315,
        np.zeros((*depth.shape, 3), dtype=np.uint8),
        depth,
        intrinsics,
        0.001,
        np.eye(4),
        10.5,
        10.5,
        "recorded",
    )
    result = compare_pose_proposals(
        frame,
        mask,
        points,
        candidate,
        group,
        max_agreement_offset_m=0.01,
        max_agreement_angle_rad=0.2,
        min_masked_depth_points=50,
    )
    assert result.verdict == PoseComparisonVerdict.AMBIGUOUS
    assert result.object_center_difference_m > 0.03
    assert result.candidate.depth_residual_p90_m < result.prior.depth_residual_p90_m
    assert result.candidate.projected_point_mask_fraction < result.prior.projected_point_mask_fraction
