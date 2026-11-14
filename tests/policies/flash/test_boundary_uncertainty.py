"""Recorded source-edge uncertainty, with independent wrong-pose controls.

This pins an experimental measurement policy, not a ground-truth pose label.
The stored candidate is the unchanged small_gicp proposal before Kalman.
"""

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from benchmarks.flash_boundary_uncertainty import boundary_band
from lerobot.policies.flash.surface_grounding import ShapeConfig, SurfaceFitView, WholeSurfaceFit
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession


@pytest.fixture(scope="module")
def boundary_toy():
    with np.load(Path(__file__).parent / "fixtures/green_toy_boundary_20261113.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    k = data["intrinsics"]

    def observation(prefix):
        result = {
            key: data[prefix + "_" + key]
            for key in ("rgb", "depth", "mask", "scene_depth", "all_surface_points")
        }
        result["obs"] = {
            key: data[prefix + "_obs_" + key]
            for key in (
                "points",
                "normals",
                "pixels",
                "fit",
                "valid_image",
                "depth_reliable",
                "depth_range_rejected",
            )
        }
        return {**result, "intrinsics": k, "frame_id": int(data[prefix + "_frame_id"])}

    return data, observation("current"), observation("retained")


def toy_geometry(fixture):
    data, current, retained = fixture
    return WholeSurfaceFit(
        data["map"],
        current,
        data["intrinsics"],
        ShapeConfig(**json.loads(str(data["shape_config"]))),
        reference_views=(SurfaceFitView(retained, data["intrinsics"], data["retained_pose"]),),
    )


def toy_verdict(fixture, pose=None):
    data = fixture[0]
    return toy_geometry(fixture).check_pose(
        data["candidate"] if pose is None else pose, **json.loads(str(data["limits"]))
    )


def test_saved_rejection_requires_both_source_reliability_and_outside_uncertainty(boundary_toy):
    data, current, _ = boundary_toy
    before = {key: value.copy() for key, value in current.items() if isinstance(value, np.ndarray)}
    baseline = toy_verdict(boundary_toy)
    assert not baseline["accepted"]
    assert all(view["coherent_protrusion"] for view in baseline["scene_depth_views"])
    with boundary_band(1, "outside", source_margin=2):
        outside_only = toy_verdict(boundary_toy)
        assert not outside_only["accepted"]
        assert outside_only["scene_depth_views"][1]["coherent_protrusion"]
    with boundary_band(0, "outside", source_margin=3):
        source_only = toy_verdict(boundary_toy)
        assert not source_only["accepted"]
        assert source_only["scene_depth_views"][0]["coherent_protrusion"]
    with boundary_band(1, "outside", source_margin=3):
        result = toy_verdict(boundary_toy)
        assert result["accepted"]
        assert result["surface_median_mm"] < 3
        assert not any(view["coherent_protrusion"] for view in result["scene_depth_views"])
    for key, old in before.items():
        np.testing.assert_array_equal(current[key], old)
    assert not toy_verdict(boundary_toy)["accepted"]


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("offset", [-0.01, 0.01])
@pytest.mark.parametrize("margin,outside", [(3, 1), (4, 2)])
def test_boundary_policy_still_rejects_centimetre_displacements(boundary_toy, axis, offset, margin, outside):
    pose = boundary_toy[0]["candidate"].copy()
    pose[axis, 3] += offset
    with boundary_band(outside, "outside", source_margin=margin):
        assert not toy_verdict(boundary_toy, pose)["accepted"]


@pytest.mark.parametrize("margin", [3, 4])
@pytest.mark.parametrize("outside", [1, 2])
@pytest.mark.parametrize("compiled", [False, True])
def test_same_policy_keeps_recorded_wrong_cube_rejected(margin, outside, compiled):
    from test_recorded_cube_publication_geometry import geometry, published_cube

    from lerobot.policies.flash.surface_acceleration import compiled_kernels

    fixture = published_cube.__wrapped__()
    kernels = compiled_kernels() if compiled else None
    if compiled and kernels is None:
        pytest.skip("optional compiled kernels unavailable")
    with boundary_band(outside, "outside", source_margin=margin):
        fit = geometry(fixture, kernels)
        assert not fit.check_pose(
            fixture[0]["pose"], max_outlier_fraction=0.25, agreement_m=0.005, min_overlap=0.25
        )["accepted"]


@pytest.mark.parametrize("margin", [3, 4])
@pytest.mark.parametrize("outside", [1, 2])
def test_static_controller_and_wrong_controls_under_same_policy(margin, outside):
    from test_retained_view_boundary_depth import clip

    fixture = clip.__wrapped__()
    old = fixture["two"]
    with boundary_band(outside, "outside", source_margin=margin):
        fit = WholeSurfaceFit(
            old.reference, old.observation, old.k, old.config, reference_views=old.reference_views
        )
        for delta in [(0, 0, 0), (0, 0, -0.01), (0, 0, 0.01), (0.008, 0, 0), (0, 0.008, 0)]:
            pose = fixture["arrays"]["prediction_pose"].copy()
            pose[:3, 3] += delta
            assert fixture["check"](fit, pose)["accepted"] == (delta == (0, 0, 0))


@pytest.mark.parametrize("margin", [3, 4])
@pytest.mark.parametrize("outside", [1, 2])
@pytest.mark.parametrize("step", [1, 3, 5])
def test_same_policy_rejects_rotated_square_at_different_densities(margin, outside, step):
    from test_distributed_free_space import test_rotated_square_face_cannot_hide_its_four_protruding_corners

    with boundary_band(outside, "outside", source_margin=margin):
        test_rotated_square_face_cannot_hide_its_four_protruding_corners(step)


@pytest.mark.parametrize("current_shift", [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)])
@pytest.mark.parametrize("retained_shift", [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)])
def test_one_pixel_mask_uncertainty_does_not_flip_frozen_pose_verdict(
    boundary_toy, current_shift, retained_shift
):
    def shifted(observation, shift):
        h, w = observation["mask"].shape
        mask = cv2.warpAffine(
            observation["mask"].astype(np.uint8),
            np.array([[1.0, 0, shift[0]], [0, 1.0, shift[1]]]),
            (w, h),
            flags=cv2.INTER_NEAREST,
        ).astype(bool)
        return {**observation, "mask": mask}

    data, current, retained = boundary_toy
    with boundary_band(2, "outside", source_margin=4):
        assert toy_verdict((data, shifted(current, current_shift), shifted(retained, retained_shift)))[
            "accepted"
        ]


@pytest.mark.parametrize("margin,outside", [(3, 1), (4, 2)])
def test_cleanup_is_shared_by_current_points_depth_and_retained_validation(margin, outside):
    k = np.array([[500.0, 0, 60], [0, 500.0, 50], [0, 0, 1.0]])
    mask = np.zeros((100, 120), bool)
    mask[20:80, 30:90] = True
    depth = np.where(mask, 0.4, 0.8).astype(np.float32)
    rgb = np.zeros((*mask.shape, 3), np.uint8)
    session = SurfaceTrackingSession(None)
    session.observation_config = replace(
        session.observation_config, mask_margin_px=margin, outer_mask_margin_px=outside
    )
    current = session.prepare(rgb, depth, mask, k, 0)["current"]
    interior = cv2.erode(mask.astype(np.uint8), np.ones((2 * margin + 1,) * 2, np.uint8)) > 0
    xy = current["obs"]["pixels"]
    assert interior[xy[:, 1], xy[:, 0]].all()
    assert not current["obs"]["valid_image"][mask & ~interior].any()
    assert not current["depth"][mask & ~interior].any()
    np.testing.assert_array_equal(current["mask"], mask)
    assert current["scene_depth"][50, 100] == pytest.approx(0.8)
    assert session.observation_config.mask_margin_px == margin
    assert current["scene_depth"][50, 29] == 0
    assert not current["obs"]["depth_reliable"][50, 29]
    fit = WholeSurfaceFit(
        current["all_surface_points"], current, k, ShapeConfig(), reference_observation=current
    )
    assert fit.check_pose(np.eye(4), max_outlier_fraction=0.25, min_overlap=0.25)["accepted"]
    wrong = np.eye(4)
    wrong[2, 3] = -0.01
    assert not fit.check_pose(wrong, max_outlier_fraction=0.25, min_overlap=0.25)["accepted"]
