"""Acceleration preserves measured geometry, state order and image ownership."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, observation


def test_prepared_frame_geometry_reuses_trees_without_changing_residual():
    from lerobot.policies.flash.experimental_surface_session import ImageConditionedFit

    with np.load(Path(__file__).with_name("fixtures") / "ring_visible_motion_20261016.npz") as data:
        points = data["source_surface"]
        depth = data["target_depth_mm"].astype(np.float32) * 0.001
        mask = data["target_mask"].astype(bool)
        k = data["intrinsics"]
    obs = observation(depth, mask, k, asdict(SurfaceObservationConfig()))
    current = {"obs": obs, "mask": mask, "depth": depth, "all_surface_points": obs["points"]}
    config = ShapeConfig()
    image = ImageConditionedFit(
        points,
        current,
        k,
        config,
        image_reference=np.empty((0, 3)),
        image_current=np.empty((0, 3)),
        image_scale_m=config.depth_tolerance_m,
        image_weight=0,
    )
    reused = WholeSurfaceFit(
        points,
        current,
        k,
        config,
        prepared_reference=image.prepared_reference,
        prepared_target=image.prepared_target,
    )
    independent = WholeSurfaceFit(points, current, k, config)
    assert reused.reference_tree is image.reference_tree
    assert reused.target_tree is image.target_tree
    np.testing.assert_array_equal(reused.residual(np.eye(4)), independent.residual(np.eye(4)))


@pytest.mark.parametrize("fixture", ["ring_visible_motion_20261016", "dowel_held_frame_315"])
def test_surface_priority_reuses_search_pool_without_changing_solution(fixture, monkeypatch):
    from threading import Thread

    from lerobot.policies.flash.experimental_surface_session import ImageConditionedFit
    from lerobot.policies.flash.surface_acceleration import SearchPool
    from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig

    with np.load(Path(__file__).with_name("fixtures") / f"{fixture}.npz") as data:
        ring = "source_surface" in data
        points = data["source_surface" if ring else "model_points"]
        depth = data["target_depth_mm" if ring else "depth"].astype(np.float32) * 0.001
        mask = data["target_mask" if ring else "mask"].astype(bool)
        k = data["intrinsics"]
        initial = np.eye(4) if ring else data["candidate"]
    obs = observation(depth, mask, k, asdict(SurfaceObservationConfig()))
    current = {"obs": obs, "mask": mask, "depth": depth, "all_surface_points": obs["points"]}
    shape = ShapeConfig(max_evaluations=5, batch_query_workers=2)
    kwargs = {
        "image_reference": np.empty((0, 3)),
        "image_current": np.empty((0, 3)),
        "image_scale_m": 0.005,
        "image_weight": 1,
    }
    uncertainty = PoseUncertaintyConfig(enabled=True)
    expected, old_detail = ImageConditionedFit(points, current, k, shape, **kwargs).correct_surface_first(
        initial,
        initial,
        uncertainty=uncertainty,
    )
    starts = []
    original = Thread.start

    def start(thread):
        starts.append(thread.name)
        return original(thread)

    monkeypatch.setattr(Thread, "start", start)
    with SearchPool(2) as pool:
        actual, detail = ImageConditionedFit(
            points, current, k, shape, search_pool=pool, **kwargs
        ).correct_surface_first(
            initial,
            initial,
            uncertainty=uncertainty,
        )
    np.testing.assert_array_equal(actual, expected)
    assert detail == old_detail
    assert starts and len(starts) <= 2
    assert all(name.startswith("flash-surface-search") for name in starts)


@pytest.mark.parametrize("fixture", ["ring_visible_motion_20261016", "dowel_held_frame_315"])
def test_batched_solver_matches_scalar_on_recorded_surfaces(fixture, monkeypatch):
    data = np.load(Path(__file__).with_name("fixtures") / f"{fixture}.npz")
    is_ring = "source_surface" in data
    points = data["source_surface" if is_ring else "model_points"]
    depth = data["target_depth_mm" if is_ring else "depth"].astype(np.float32) * 0.001
    mask = data["target_mask" if is_ring else "mask"].astype(bool)
    k = data["intrinsics"]
    obs = observation(depth, mask, k, asdict(SurfaceObservationConfig()))
    current = {"obs": obs, "mask": mask, "depth": depth, "all_surface_points": obs["points"]}
    scalar = WholeSurfaceFit(points, current, k, ShapeConfig())
    batched = WholeSurfaceFit(points, current, k, ShapeConfig())
    initial = np.eye(4) if is_ring else data["candidate"]
    poses = []
    for offset in (0, 0.003, -0.005):
        pose = initial.copy()
        pose[:3, 3] += offset
        poses.append(pose)
    for actual, expected in zip(batched.batch_residuals(poses), map(scalar.residual, poses), strict=True):
        np.testing.assert_array_equal(actual, expected)
    old, old_detail = scalar.correct(initial)
    new, new_detail = batched.correct(initial)
    np.testing.assert_array_equal(new, old)
    assert new_detail["cost"] == old_detail["cost"]
    assert new_detail["evaluations"] == old_detail["evaluations"]
    assert new_detail["used_coarse"] == old_detail["used_coarse"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compiled_ray_evidence_matches_numpy_for_partial_views(dtype):
    """Depth holes, occluders, empty space, edges and subpixel projections."""

    from benchmarks.flash_partial_surface_control import box_observation
    from lerobot.policies.flash.surface_acceleration import compiled_kernels
    from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit

    source, k, _ = box_observation(0)
    current, _, _ = box_observation(40)
    current["depth"] = current["depth"].astype(dtype)
    current["depth"][60:80, 100:130] = 0
    current["depth"][80:100, 100:130] = 0.3
    config = ShapeConfig(scene_depth_constraints=True, prefer_observed_overlap=True)
    numpy_fit = WholeSurfaceFit(source["obs"]["points"], current, k, config)
    native_fit = WholeSurfaceFit(source["obs"]["points"], current, k, config, fit_kernels=compiled_kernels())
    rng = np.random.default_rng(14)
    points = rng.uniform([-0.3, -0.2, 0.2], [0.3, 0.2, 1.8], size=(3000, 3))
    # Include exact and fractional rays around all four image borders.
    uv = np.array([[x, y] for x in (-0.1, 0, 0.5, 239, 239.1) for y in (-0.1, 0, 0.5, 179, 179.1)])
    rays = np.column_stack(((uv - k[:2, 2]) / np.diag(k)[:2], np.ones(len(uv)))) * 0.7
    points = np.concatenate((points, rays))
    for method in ("_projection_evidence", "_scene_depth_evidence"):
        expected = getattr(numpy_fit, method)(points)
        actual = getattr(native_fit, method)(points)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-14)


def test_image_and_geometry_objectives_share_prepared_views():
    from benchmarks.flash_partial_surface_control import box_observation
    from lerobot.policies.flash.surface_grounding import SurfaceFitView

    source, k, _ = box_observation(0)
    current, _, pose = box_observation(30)
    config = ShapeConfig(scene_depth_constraints=True)
    view = SurfaceFitView(source, k, np.eye(4))
    a = WholeSurfaceFit(source["obs"]["points"], current, k, config, reference_views=(view,))
    b = WholeSurfaceFit(
        a.reference,
        current,
        k,
        config,
        prepared_reference=a.prepared_reference,
        prepared_target=a.prepared_target,
        reference_views=(view,),
        prepared_reverse_fits=a.reverse_fits,
    )
    assert b.reverse_fits is a.reverse_fits
    np.testing.assert_array_equal(a.residual(pose), b.residual(pose))
    assert a.check_pose(pose, max_outlier_fraction=0.25) == b.check_pose(pose, max_outlier_fraction=0.25)
    other, _, _ = box_observation(20)
    with pytest.raises(ValueError, match="prepared views"):
        WholeSurfaceFit(
            a.reference, other, k, config, reference_views=(view,), prepared_reverse_fits=a.reverse_fits
        )


def test_reused_residuals_preserve_geometry_after_eviction_and_frame_change():
    """A workspace cannot change costs, acceptance, or the next frame's evidence."""
    from benchmarks.flash_partial_surface_control import box_observation
    from lerobot.policies.flash.surface_grounding import SurfaceFitView

    source, k, _ = box_observation(0)
    current, _, pose = box_observation(30)
    config = ShapeConfig(scene_depth_constraints=True)
    views = (SurfaceFitView(source, k, np.eye(4)),)
    fit = WholeSurfaceFit(source["obs"]["points"], current, k, config, reference_views=views)
    expected = fit.residual(pose).copy()
    expected_check = fit.check_pose(pose, max_outlier_fraction=0.25)
    # Exceed the workspace capacity with distinct optimizer probes. Re-evaluating
    # an evicted pose must agree with a fresh fitter, not a remembered score.
    for delta in np.linspace(-0.01, 0.01, 96):
        probe = pose.copy()
        probe[0, 3] += delta
        fit.residual(probe)
    assert len(fit._residual_cache) <= 64
    np.testing.assert_array_equal(fit.residual(pose), expected)
    assert fit.check_pose(pose, max_outlier_fraction=0.25) == expected_check

    # The same pose is not the same evidence after the object moves.
    later, _, _ = box_observation(60)
    next_fit = WholeSurfaceFit(source["obs"]["points"], later, k, config, reference_views=views)
    assert not np.array_equal(next_fit.residual(pose), expected)
    np.testing.assert_array_equal(fit.residual(pose), expected)
