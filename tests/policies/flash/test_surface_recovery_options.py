"""Generic controls for the opt-in recovery experiments, including false fits."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig
from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit, pca_candidates
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, move
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig, check_surface_contradiction
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig


def test_rejected_depth_cannot_reenter_pose_validation_with_either_legacy_setting():
    depth = np.ones((100, 120), np.float32)
    mask = np.zeros(depth.shape, bool)
    mask[10:90, 10:110] = True
    k = np.array([[500.0, 0, 60], [0, 500.0, 50], [0, 0, 1]])
    rgb = np.zeros((*depth.shape, 3), np.uint8)
    config = SurfaceObservationConfig(component_gap_m=None, depth_outlier_sigma=3.0)
    source = SurfaceTrackingSession(None, observation_config=config).prepare(rgb, depth, mask, k, 0)
    # A minority region has a coherent but very wrong depth measurement, as in
    # the drill recording. Local smoothness alone cannot identify this error.
    bad_depth = depth.copy()
    bad_depth[15:85, 15:48] = 3.0
    original = bad_depth.copy()
    results = []
    for enabled in (False, True):
        builder = SurfaceTrackingSession(
            None, observation_config=replace(config, filtered_depth_for_fitting=enabled)
        )
        current = builder.prepare(rgb, bad_depth, mask, k, 0)["current"]
        results.append(current)
        assert current["obs"]["depth_outlier_count"] > 1000
        assert np.max(current["obs"]["points"][:, 2]) < 1.01
        np.testing.assert_array_equal(current["mask"], mask)
        fit = WholeSurfaceFit(
            source["current"]["obs"]["points"],
            current,
            k,
            ShapeConfig(),
            reference_observation=source["current"],
        )
        assert check_surface_contradiction(fit, np.eye(4), PoseUncertaintyConfig())["accepted"]
        wrong = np.eye(4)
        wrong[0, 3] = 0.2
        assert not check_surface_contradiction(fit, wrong, PoseUncertaintyConfig())["accepted"]
    np.testing.assert_array_equal(bad_depth, original)
    np.testing.assert_array_equal(results[0]["obs"]["points"], results[1]["obs"]["points"])
    np.testing.assert_array_equal(results[0]["depth"], results[1]["depth"])
    assert results[0]["depth"][40, 30] == 0


def test_depth_range_filter_retains_disconnected_visible_parts():
    depth = np.ones((100, 120), np.float32)
    mask = np.zeros(depth.shape, bool)
    mask[10:90, 10:40] = True
    mask[10:90, 70:100] = True
    mask[40:50, 105:115] = True
    depth[40:50, 105:115] = 3.0
    builder = SurfaceTrackingSession(
        None, observation_config=SurfaceObservationConfig(component_gap_m=None, depth_outlier_sigma=3.0)
    )
    k = np.array([[500.0, 0, 60], [0, 500.0, 50], [0, 0, 1]])
    measured = builder.prepare(np.zeros((*depth.shape, 3), np.uint8), depth, mask, k, 0)["current"]["obs"]
    assert (measured["pixels"][:, 0] < 40).any()
    assert (measured["pixels"][:, 0] > 70).any()
    assert measured["points"][:, 2].max() < 1.01


def test_bounded_coarse_search_only_generates_proper_rigid_transforms():
    points = np.random.default_rng(4).normal(size=(200, 3)) * [0.03, 0.015, 0.006] + [0, 0, 0.8]
    truth = np.eye(4)
    truth[:3, :3] = Rotation.from_euler("xyz", [25, -40, 110], degrees=True).as_matrix()
    truth[:3, 3] = [0.03, 0.02, 0.05]
    target = move(points, truth)
    old = pca_candidates(points, target)
    expanded = pca_candidates(points, target, all_axis_orders=True)
    assert len(old) == 4 and len(expanded) == 24
    assert len({tuple(p.round(10).ravel()) for p in expanded}) == 24
    for p in expanded:
        np.testing.assert_allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=1e-12)
        assert np.linalg.det(p[:3, :3]) == pytest.approx(1.0)
        np.testing.assert_allclose(move(points, p).mean(0), target.mean(0), atol=1e-12)
    for p in old:
        assert any(np.allclose(p, q) for q in expanded)


@pytest.mark.parametrize("starts", [-1, 25, 1.5, True])
def test_recovery_search_budget_is_bounded(starts):
    with pytest.raises(ValueError):
        GrowingSurfaceConfig(recovery_starts=starts)


def test_experiments_are_opt_in_and_config_roundtrips():
    config = VisionPipelineConfig()
    assert config.observation.depth_outlier_sigma is None
    assert not config.observation.filtered_depth_for_fitting
    assert GrowingSurfaceConfig().recovery_starts == 0
    changed = VisionPipelineConfig.from_dict({"observation": {"filtered_depth_for_fitting": True}})
    assert changed.observation.filtered_depth_for_fitting
    with pytest.raises(ValueError):
        VisionPipelineConfig.from_dict({"observation": {"filtered_depth_for_fitting": "yes"}})
