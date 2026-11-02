"""Known rigid motion and deliberately wrong image matches, through real fitting."""

from dataclasses import asdict

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.experimental_surface_session import ImageConditionedFit
from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, move, observation


def measured_shape(dx_px=0, *, scale=1):
    # Asymmetric planar object: no hidden labels enter the fitting objective.
    mask = np.zeros((100, 140), bool)
    mask[25:75, 35 + dx_px : 85 + dx_px] = True
    mask[25:45, 85 + dx_px : 100 + dx_px] = True
    depth = np.full(mask.shape, 0.8 * scale, np.float32)
    k = np.array([[800.0, 0, 70], [0, 800.0, 50], [0, 0, 1]])
    measured = observation(
        depth,
        mask,
        k,
        asdict(SurfaceObservationConfig(component_gap_m=None, surface_spacing_m=0.001 * scale)),
    )
    return {"obs": measured, "depth": depth, "mask": mask}, k


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("actual_pixels", [-10, 0, 10])
def test_geometry_rejects_wrong_image_motion_without_freezing_actual_motion(scale, actual_pixels):
    source, k = measured_shape(scale=scale)
    current, _ = measured_shape(actual_pixels, scale=scale)
    reference = source["obs"]["points"]
    expected = np.eye(4)
    expected[0, 3] = actual_pixels * 0.001 * scale
    # Image matches claim a further 15-pixel slide inside/on the object.
    wrong = expected.copy()
    wrong[0, 3] += 0.015 * scale
    fitter = ImageConditionedFit(
        reference,
        current,
        k,
        ShapeConfig(),
        image_reference=reference,
        image_current=move(reference, wrong),
        image_scale_m=0.005,
        image_weight=1,
    )
    result, _ = fitter.correct_surface_first(wrong, np.eye(4))
    error = np.linalg.norm(move(reference, result) - move(reference, expected), axis=1)
    assert np.percentile(error, 95) < 0.001 * scale
    assert np.rad2deg(Rotation.from_matrix(result[:3, :3]).magnitude()) < 1


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("angle_deg", [-20, 20])
def test_geometry_tracks_real_rotation_despite_wrong_image_matches(scale, angle_deg):
    source, k = measured_shape(scale=scale)
    expected = np.eye(4)
    expected[:3, :3] = Rotation.from_euler("z", angle_deg, degrees=True).as_matrix()
    affine = np.column_stack((expected[:2, :2], k[:2, 2] - expected[:2, :2] @ k[:2, 2]))
    mask = cv2.warpAffine(source["mask"].astype(np.uint8), affine, (140, 100), flags=cv2.INTER_NEAREST) > 0
    measured = observation(
        source["depth"],
        mask,
        k,
        asdict(SurfaceObservationConfig(component_gap_m=None, surface_spacing_m=0.001 * scale)),
    )
    reference = source["obs"]["points"]
    wrong = expected.copy()
    wrong[:3, :3] = Rotation.from_euler("z", angle_deg + 15, degrees=True).as_matrix()
    fitter = ImageConditionedFit(
        reference,
        {"obs": measured, "depth": source["depth"], "mask": mask},
        k,
        ShapeConfig(),
        image_reference=reference,
        image_current=move(reference, wrong),
        image_scale_m=0.005,
        image_weight=1,
    )
    result, _ = fitter.correct_surface_first(wrong, np.eye(4))
    error = np.linalg.norm(move(reference, result) - move(reference, expected), axis=1)
    # Rasterizing the rotated mask introduces up to one pixel of edge error.
    assert np.percentile(error, 95) < 0.001 * scale
    assert np.rad2deg(Rotation.from_matrix(result[:3, :3] @ expected[:3, :3].T).magnitude()) < 1


def test_unknown_depth_outside_visible_mask_does_not_assert_object_is_absent():
    source, k = measured_shape()
    reference = source["obs"]["points"]
    visible = source["mask"].copy()
    visible[:, :65] = False
    costs = {}
    for name, left_depth in (("unknown", 0.0), ("occluder", 0.4), ("empty_space", 1.2)):
        depth = source["depth"].copy()
        depth[:, :65] = left_depth
        measured = observation(depth, visible, k, asdict(SurfaceObservationConfig(component_gap_m=None)))
        fit = WholeSurfaceFit(reference, {"obs": measured, "depth": depth, "mask": visible}, k, ShapeConfig())
        costs[name] = fit.score(np.eye(4))
    # Unknown rays and a nearer occluder both lack evidence against the hidden
    # object. Seeing a farther surface DOES rule out an object in front of it.
    assert costs["unknown"] == pytest.approx(costs["occluder"], abs=1e-12)
    assert costs["empty_space"] > costs["unknown"] + 0.1


def test_partial_reference_is_not_shifted_to_cover_mask_where_depth_was_missing():
    original, k = measured_shape()
    measured_interior = cv2.erode(original["mask"].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    depth = np.where(measured_interior, original["depth"], 0).astype(np.float32)
    config = asdict(SurfaceObservationConfig(component_gap_m=None))
    source = observation(depth, original["mask"], k, config)
    expanded = original["mask"].copy()
    expanded[75:95, 35:85] = True
    measured = observation(depth, expanded, k, config)
    np.testing.assert_array_equal(source["points"], measured["points"])
    # Same known XYZ, with a visible but unmeasured continuation of the object.
    # Fitting must not move the measured part to cover that missing geometry.
    reference = source["points"]
    fit = WholeSurfaceFit(reference, {"obs": measured, "depth": depth, "mask": expanded}, k, ShapeConfig())
    pose, _ = fit.correct(np.eye(4))
    assert np.max(np.linalg.norm(move(reference, pose) - reference, axis=1)) < 0.001
    assert np.rad2deg(Rotation.from_matrix(pose[:3, :3]).magnitude()) < 1


@pytest.mark.parametrize("object_really_moves", [False, True])
def test_observed_shape_decides_whether_object_matches_or_surrounding_group_are_wrong(object_really_moves):
    from benchmarks.flash_experiment_groups import GroupConfig, GroupHints
    from lerobot.policies.flash.perception import RGBDFrame
    from lerobot.policies.flash.scene_groups import SceneTracks

    source, k = measured_shape()
    y, x = np.mgrid[35:66:10, 45:76:10]
    object_uv = np.column_stack((x.ravel(), y.ravel()))
    y, x = np.mgrid[30:61:10, 105:136:10]
    outside_uv = np.column_stack((x.ravel(), y.ravel()))
    uv = np.concatenate((object_uv, outside_uv))
    xyz = np.column_stack(((uv[:, 0] - 70) * 0.001, (uv[:, 1] - 50) * 0.001, np.full(len(uv), 0.8)))
    groups = GroupHints(GroupConfig(support_confirmation_s=0), shared_image_points=True)
    reference = source["obs"]["points"]
    for index in range(3):
        current, _ = measured_shape(8 if index == 2 and object_really_moves else 0)
        pixels, points = uv.copy(), xyz.copy()
        if index == 2:
            # Same reported correspondence shift in both tests. Only the real
            # mask/depth observation differs; the object pose is never an input.
            pixels[: len(object_uv), 0] += 8
            points[: len(object_uv), 0] += 0.008
        frame = RGBDFrame(
            index,
            np.zeros((100, 140, 3), np.uint8),
            current["depth"],
            k,
            1.0,
            np.eye(4),
            index / 30,
            index / 30,
            "test",
        )
        scene = SceneTracks(index, index / 30, np.arange(len(uv)), pixels, points)
        masks = {"arbitrary object": current["mask"]}
        hint, details = groups.predict(frame, scene, masks, "arbitrary object", 0, observation=current)
        if index < 2:
            groups.observe(
                frame,
                scene,
                masks,
                "arbitrary object",
                {
                    "status": "observed_candidate",
                    "camera_from_reference": np.eye(4),
                    "diagnostics": {"reference_frame": 0},
                },
                reference,
            )
    assert details["environment_separated"] == object_really_moves
    expected = np.eye(4)
    expected[0, 3] = 0.008 if object_really_moves else 0
    np.testing.assert_allclose(hint.camera_from_reference, expected, atol=1e-10)
    if not object_really_moves:
        assert details["object_matches_rejected_by_observation"]
        rejected = [p for p in hint.points if p.kind == "measured"]
        assert rejected and all(p.weight == 0 for p in rejected)
