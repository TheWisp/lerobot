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
