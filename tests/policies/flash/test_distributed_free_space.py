"""Independent measured-air conflicts cannot disappear with sparse sampling."""

from dataclasses import replace

import numpy as np
import pytest

from lerobot.policies.flash.surface_acceleration import compiled_kernels
from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_observation import unproject
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession


def scene():
    k = np.array([[500.0, 0, 60], [0, 500.0, 50], [0, 0, 1]])
    mask = np.zeros((100, 120), bool)
    mask[25:75, 40:80] = True
    depth = np.full(mask.shape, 0.8, np.float32)
    depth[mask] = 0.4
    current = SurfaceTrackingSession(None).prepare(np.zeros((*mask.shape, 3), np.uint8), depth, mask, k, 0)[
        "current"
    ]
    return current, k


def patches(k, count=4, samples=1):
    # Disconnected small map patches over four distinct background regions.
    centers = np.array([[15, 15], [100, 15], [15, 85], [100, 85]])[:count]
    uv = (centers[:, None, :] + np.c_[np.linspace(0, 0.5, samples), np.zeros(samples)]).reshape(-1, 2)
    return np.c_[(uv - k[:2, 2]) / np.diag(k)[:2] * 0.4, np.full(len(uv), 0.4)]


def verdict(current, k, reference, *, kernels=None, config=None):
    return WholeSurfaceFit(reference, current, k, config or ShapeConfig(), fit_kernels=kernels).check_pose(
        np.eye(4), max_outlier_fraction=0.99, agreement_m=0.005, min_overlap=0.25
    )


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("count", [3, 4])
@pytest.mark.parametrize("samples", [1, 10])
def test_disconnected_protrusions_reject_with_sparse_or_dense_surface(compiled, count, samples):
    current, k = scene()
    kernels = compiled_kernels() if compiled else None
    if compiled and kernels is None:
        pytest.skip("optional compiled kernels unavailable")
    reference = np.concatenate((current["obs"]["points"], patches(k, count, samples)))
    result = verdict(current, k, reference, kernels=kernels)
    assert result["scene_depth_outlier_fraction"] < 0.25
    assert not result["accepted"]
    assert result["scene_depth_coherent_protrusion"]


@pytest.mark.parametrize("samples", [1, 10, 100])
def test_densifying_one_stray_patch_does_not_invent_independent_evidence(samples):
    current, k = scene()
    reference = np.concatenate((current["obs"]["points"], patches(k, 1, samples)))
    assert verdict(current, k, reference)["accepted"]


@pytest.mark.parametrize("depth_value", [0.0, 0.3, 0.404])
def test_holes_occluders_and_subtolerance_noise_do_not_become_conflicts(depth_value):
    current, k = scene()
    reference = np.concatenate((current["obs"]["points"], patches(k)))
    depth = current["depth"].copy()
    depth[~current["mask"]] = depth_value
    current = {**current, "depth": depth, "scene_depth": depth}
    result = verdict(current, k, reference)
    assert result["scene_depth_outlier_fraction"] == 0
    assert result["accepted"]


def test_isolated_bad_sensor_pixels_do_not_reject_a_correct_surface():
    current, k = scene()
    reference = current["obs"]["points"]
    depth = current["depth"].copy()
    depth[[35, 35, 65, 65], [50, 70, 50, 70]] = 0.44
    current = {**current, "depth": depth, "scene_depth": depth}
    assert verdict(current, k, reference)["accepted"]


def test_partial_interpolation_next_to_holes_is_not_independent_hard_evidence():
    current, k = scene()
    depth = current["depth"].copy()
    depth[~current["mask"]] = 0
    centers = np.array([[15, 15], [100, 15], [15, 85], [100, 85]])
    for x, y in centers:
        depth[y - 2 : y + 3, x - 2 : x + 3] = 0.8
    # The conservative neighborhood is known at one interpolation corner;
    # the other three corners are unknown. Four such slivers are not four
    # fully supported patches of empty space.
    uv = centers + 0.5
    extra = np.c_[(uv - k[:2, 2]) / np.diag(k)[:2] * 0.4, np.full(len(uv), 0.4)]
    reference = np.concatenate((current["obs"]["points"], extra))
    current = {**current, "depth": depth, "scene_depth": depth}
    result = verdict(current, k, reference)
    assert 0 < result["scene_depth_outlier_fraction"] < 0.25
    assert result["accepted"]


@pytest.mark.parametrize("step", [1, 3, 5])
def test_rotated_square_face_cannot_hide_its_four_protruding_corners(step):
    current, k = scene()
    mask = np.zeros_like(current["mask"])
    mask[30:70, 40:80] = True
    depth = np.full(mask.shape, 0.8, np.float32)
    depth[mask] = 0.4
    current = SurfaceTrackingSession(None).prepare(current["rgb"], depth, mask, k, 0)["current"]
    sampled = np.zeros_like(mask)
    # Retain the same physical extent at every density, including all corners.
    y = np.linspace(30, 69, 40 // step, dtype=int)
    x = np.linspace(40, 79, 40 // step, dtype=int)
    sampled[np.ix_(y, x)] = True
    reference = unproject(depth, k)[sampled]
    angle = np.deg2rad(30)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    assert verdict(current, k, reference)["accepted"]
    reference[:, :2] = reference[:, :2] @ rotation.T
    assert not verdict(current, k, reference)["accepted"]


def test_support_requirement_is_configurable_without_changing_depth_tolerance():
    current, k = scene()
    reference = np.concatenate((current["obs"]["points"], patches(k, 3)))
    assert not verdict(current, k, reference, config=ShapeConfig(free_space_min_supports=3))["accepted"]
    assert verdict(current, k, reference, config=ShapeConfig(free_space_min_supports=4))["accepted"]


@pytest.mark.parametrize("count", [0, -1, 2.5, True])
def test_support_requirement_rejects_invalid_config(count):
    with pytest.raises(ValueError, match="free_space_min_supports"):
        replace(ShapeConfig(), free_space_min_supports=count)
