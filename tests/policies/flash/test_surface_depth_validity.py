"""Withheld depth is unknown in fitting, validation and retained observations."""

from dataclasses import replace

import numpy as np
import pytest

from benchmarks.flash_partial_surface_control import box_observation
from lerobot.policies.flash.experimental_surface_session import (
    GrowingSurfaceConfig,
    GrowingSurfaceSession,
    SurfaceView,
)
from lerobot.policies.flash.surface_acceleration import compiled_kernels
from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, unproject
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig, check_surface_contradiction


@pytest.mark.parametrize("source_angle,target_angle", [(15, 20), (20, 15)])
def test_true_pose_survives_depth_cleanup_of_a_real_narrow_side(source_angle, target_angle):
    source, k, source_pose = box_observation(source_angle)
    current, _, target_pose = box_observation(target_angle)
    pose = target_pose @ np.linalg.inv(source_pose)
    fit = WholeSurfaceFit(source["obs"]["points"], current, k, ShapeConfig(), reference_observation=source)
    # Both exact, noiseless observations include a side that depth cleanup
    # withholds. Missing neighbours there cannot contradict the true pose.
    assert source["obs"]["edge_count"] + current["obs"]["edge_count"] > 0
    assert check_surface_contradiction(fit, pose, PoseUncertaintyConfig())["accepted"]
    for offset in (-0.04, 0.04):
        wrong = pose.copy()
        wrong[0, 3] += offset
        assert not check_surface_contradiction(fit, wrong, PoseUncertaintyConfig())["accepted"]


def filtered_scene():
    mask = np.zeros((100, 120), bool)
    mask[10:90, 10:110] = True
    depth = np.full(mask.shape, 1.5, np.float32)
    depth[mask] = 1.0
    depth[30:70, 20:50] = 3.0  # A coherent reading explicitly rejected by cleanup.
    k = np.array([[500.0, 0, 60], [0, 500.0, 50], [0, 0, 1]])
    builder = SurfaceTrackingSession(
        None,
        observation_config=SurfaceObservationConfig(component_gap_m=None, depth_outlier_sigma=3.0),
    )
    current = builder.prepare(np.zeros((*depth.shape, 3), np.uint8), depth, mask, k, 0)["current"]
    # Reconstitute the raw recording plus its cleanup metadata, as direct
    # fitting callers and retained-view builders may receive it.
    current = {**current, "depth": depth, "scene_depth": depth}
    clean = np.where(
        np.where(mask, current["obs"]["valid_image"], current["obs"]["depth_reliable"]), depth, 0
    )
    missing = {**current, "depth": clean, "scene_depth": clean}
    reference_depth = np.where(mask, 1.0, 0).astype(np.float32)
    reference = unproject(reference_depth, k)[mask]
    return reference, current, missing, k


@pytest.mark.parametrize("compiled", [False, True])
def test_excluded_readings_behave_like_missing_depth_in_all_checks(compiled):
    reference, current, missing, k = filtered_scene()
    original = current["depth"].copy()
    kernels = compiled_kernels() if compiled else None
    if compiled and kernels is None:
        pytest.skip("optional compiled CPU kernels unavailable")
    config = ShapeConfig(scene_depth_constraints=True, require_full_mask_coverage=True)
    excluded = WholeSurfaceFit(reference, current, k, config, fit_kernels=kernels)
    absent = WholeSurfaceFit(reference, missing, k, config, fit_kernels=kernels)
    for offset in (0.0, 0.04, -0.04):
        pose = np.eye(4)
        pose[0, 3] = offset
        np.testing.assert_array_equal(excluded.residual(pose), absent.residual(pose))
        assert excluded.check_pose(pose, max_outlier_fraction=0.25) == absent.check_pose(
            pose, max_outlier_fraction=0.25
        )
    assert excluded.check_pose(np.eye(4), max_outlier_fraction=0.25)["accepted"]
    # Background depth still proves the translated object would occupy air.
    assert not excluded.check_pose(pose, max_outlier_fraction=0.99)["accepted"]
    np.testing.assert_array_equal(current["depth"], original)
    # Unknown depth leaves the full outline available as a search preference.
    polygon = np.array([[20, 20], [40, 20], [40, 40], [20, 40]], dtype=float)
    assert excluded.uncovered_mask(polygon).max() > 20


def test_retained_views_do_not_restore_excluded_readings():
    reference, current, missing, k = filtered_scene()
    view = SurfaceView.from_observation(current, k, ShapeConfig(), retain_measurement=True)
    hole_view = SurfaceView.from_observation(missing, k, ShapeConfig(), retain_measurement=True)
    np.testing.assert_array_equal(view.contradicts(reference), hole_view.contradicts(reference))
    # Exercise the real store -> retained view -> reverse geometry consumer.
    tracker = GrowingSurfaceSession(ShapeConfig(), GrowingSurfaceConfig(partial_surface_matching=True))
    tracker.store.add(current["obs"]["points"], np.eye(4), 0, active=True, view=view)
    fit = WholeSurfaceFit(reference, missing, k, ShapeConfig(), reference_views=tracker._fit_views())
    assert check_surface_contradiction(fit, np.eye(4), PoseUncertaintyConfig())["accepted"]
    np.testing.assert_array_equal(view.depth, missing["depth"])
    np.testing.assert_array_equal(view.mask, current["mask"])


@pytest.mark.parametrize("in_unknown_area", [False, True])
def test_unknown_depth_stays_unknown_and_reliable_depth_still_rejects(in_unknown_area):
    depth = np.ones((80, 100), np.float32)
    mask = np.ones(depth.shape, bool)
    valid = mask.copy()
    valid[:, 40:60] = False
    k = np.array([[500.0, 0, 50], [0, 500.0, 40], [0, 0, 1]])
    current = {
        "depth": depth,
        "mask": mask,
        "obs": {"points": unproject(depth, k)[valid], "valid_image": valid},
    }
    # A long patch entirely in missing depth is unknown. Moving it to the
    # reliable pixels beyond the existing projection tolerance must reveal it.
    x = 50 if in_unknown_area else 63
    y = np.arange(20, 60)
    reference = np.c_[np.full(len(y), (x - 50) * 0.7 / 500), (y - 40) * 0.7 / 500, np.full(len(y), 0.7)]
    fit = WholeSurfaceFit(reference, current, k, ShapeConfig())
    assert fit.scene_depth_check(np.eye(4))["scene_depth_coherent_protrusion"] == (not in_unknown_area)


@pytest.mark.parametrize("legacy_option", [False, True])
def test_shared_frame_preparation_never_reintroduces_rejected_depth(legacy_option):
    _, current, missing, k = filtered_scene()
    config = SurfaceObservationConfig(component_gap_m=None, depth_outlier_sigma=3.0)
    session = SurfaceTrackingSession(
        None, observation_config=replace(config, filtered_depth_for_fitting=legacy_option)
    )
    prepared = session.prepare(current["rgb"], current["depth"], current["mask"], k, 0)["current"]
    np.testing.assert_array_equal(prepared["depth"], missing["depth"])
    np.testing.assert_array_equal(prepared["scene_depth"], missing["depth"])
    np.testing.assert_array_equal(prepared["mask"], current["mask"])
