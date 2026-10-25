"""Sampling must bound solver work without replacing measured grounding geometry."""

from dataclasses import asdict
from pathlib import Path

import numpy as np

from lerobot.policies.flash.surface_correspondence import SurfaceMotionConfig, crop_pair
from lerobot.policies.flash.surface_grounding import (
    PreparedReferenceSurface,
    ShapeConfig,
    WholeSurfaceFit,
)
from lerobot.policies.flash.surface_observation import (
    SurfaceObservationConfig,
    observation,
    prepare_surface_frame,
    spatial_sample,
    voxel_sample,
)


def test_minimum_budget_terminates_for_cloud_spanning_world_origin():
    points = np.array(np.meshgrid([-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0])).reshape(3, -1).T
    assert 0 < len(spatial_sample(points, 3)) <= 3


def test_fast_voxels_preserve_original_measured_points():
    rng = np.random.default_rng(4)
    points = rng.normal(size=(2000, 3)) * 0.03 + [-0.05, 0, 0.8]
    points = np.concatenate((points, points[:100]))
    for spacing in (0.001, 0.003, 0.02):
        _, original = np.unique(np.floor(points / spacing).astype(np.int32), axis=0, return_index=True)
        np.testing.assert_array_equal(voxel_sample(points, spacing), np.sort(original))


def test_tiny_objects_retain_more_actual_pixels_without_densifying_large_ones():
    depth = np.full((200, 250), 0.6, np.float32)
    k = np.array([[600, 0, 125], [0, 600, 100], [0, 0, 1]], float)
    config = asdict(SurfaceObservationConfig())
    small = np.zeros(depth.shape, bool)
    small[85:115, 110:140] = True
    old = observation(
        depth,
        small,
        k,
        {
            key: value
            for key, value in config.items()
            if key not in {"samples_across_object", "min_surface_spacing_m"}
        },
    )
    new = observation(depth, small, k, config)
    assert len(new["points"]) > 2 * len(old["points"])
    # Sampling selects measured pixels: there is no synthetic upsampling.
    px = new["pixels"]
    np.testing.assert_array_equal(new["points"][:, 2], depth[px[:, 1], px[:, 0]])
    large = observation(depth, np.ones(depth.shape, bool), k, config)
    assert large["spacing_m"] == config["surface_spacing_m"]
    assert new["spacing_m"] == config["min_surface_spacing_m"]


def test_shared_frame_filters_preserve_each_objects_measured_surface():
    depth = np.full((100, 120), 0.6, np.float32)
    depth[20:25, 25:30] = 0
    depth[70:75, 75:80] = 0.48
    k = np.array([[250, 0, 60], [0, 250, 50], [0, 0, 1]], float)
    config = asdict(SurfaceObservationConfig())
    filters = prepare_surface_frame(depth, config)
    for bounds in ((10, 45, 10, 55), (55, 90, 60, 105)):
        top, bottom, left, right = bounds
        mask = np.zeros(depth.shape, bool)
        mask[top:bottom, left:right] = True
        independent = observation(depth, mask, k, config)
        shared = observation(depth, mask, k, config, frame_filters=filters)
        for key in ("points", "normals", "pixels", "fit", "valid_image", "depth_reliable"):
            np.testing.assert_array_equal(shared[key], independent[key])
    with np.testing.assert_raises_regex(ValueError, "another depth frame"):
        observation(depth.copy(), mask, k, config, frame_filters=filters)


def test_fit_budget_keeps_dense_geometry_and_recovers_known_translation():
    depth = np.full((100, 120), 0.6, np.float32)
    k = np.array([[250, 0, 60], [0, 250, 50], [0, 0, 1]], float)
    mask = np.zeros(depth.shape, bool)
    mask[15:85, 15:105] = True
    obs = observation(depth, mask, k, asdict(SurfaceObservationConfig(mask_margin_px=0)))
    target = obs["points"]
    offset = np.array([0.01, -0.006, 0])
    config = ShapeConfig(max_fit_points=100, mask_boundary_tolerance_px=0)
    fit = WholeSurfaceFit(
        target - offset, {"obs": obs, "mask": mask, "depth": depth, "all_surface_points": target}, k, config
    )
    assert len(target) > 100
    assert len(fit.reference_queries) <= 100
    assert len(fit.target_queries) <= 100
    assert fit.reference_tree.n == len(target) == fit.target_tree.n
    prepared = PreparedReferenceSurface.build(target - offset, config)
    reused = WholeSurfaceFit(
        prepared.points,
        {"obs": obs, "mask": mask, "depth": depth, "all_surface_points": target},
        k,
        config,
        prepared_reference=prepared,
    )
    np.testing.assert_array_equal(reused.residual(np.eye(4)), fit.residual(np.eye(4)))
    # Fit a small known translation. A sampled-to-sampled fit has artificial gaps
    # even at the correct answer; dense trees must still give exact agreement.
    truth = np.eye(4)
    truth[:3, 3] = offset
    np.testing.assert_allclose(fit.residual(truth), 0, atol=1e-7)
    wrong_depth = truth.copy()
    wrong_depth[2, 3] += 0.005
    result, _ = fit.correct(wrong_depth)
    np.testing.assert_allclose(result, truth, atol=1e-6)


def test_rgb_crop_never_discards_large_object_extremities():
    mask = np.zeros((720, 1280), bool)
    mask[10:700, 100:1180] = True
    frame = {"rgb": np.zeros((*mask.shape, 3), np.uint8), "mask": mask}
    config = SurfaceMotionConfig()
    a, b, (x, y, size) = crop_pair(frame, frame, config)
    assert a.shape == b.shape == (config.input_size_px, config.input_size_px, 3)
    assert x <= 100 and x + size >= 1179.5
    assert y <= 10 and y + size >= 699.5


def test_prepared_reference_preserves_recorded_ring_fit():
    data = np.load(Path(__file__).with_name("fixtures") / "ring_visible_motion_20261016.npz")
    depth = data["target_depth_mm"].astype(np.float32) * 0.001
    mask, k = data["target_mask"], data["intrinsics"]
    observed = observation(depth, mask, k, asdict(SurfaceObservationConfig()))
    measured = {"obs": observed, "mask": mask, "depth": depth, "all_surface_points": observed["points"]}
    config = ShapeConfig(max_fit_points=512)
    reference = data["source_surface"]
    baseline = WholeSurfaceFit(reference, measured, k, config)
    accelerated = WholeSurfaceFit(
        reference, measured, k, config, prepared_reference=PreparedReferenceSurface.build(reference, config)
    )
    threaded_config = ShapeConfig(max_fit_points=512, nn_queries_per_worker=64)
    threaded = WholeSurfaceFit(reference, measured, k, threaded_config)
    for offset in (0, 0.003, -0.005):
        pose = np.eye(4)
        pose[0, 3] = offset
        np.testing.assert_array_equal(accelerated.residual(pose), baseline.residual(pose))
        np.testing.assert_array_equal(threaded.residual(pose), baseline.residual(pose))
    initial = np.eye(4)
    initial[0, 3] = 0.003
    previous_pose, previous = baseline.correct(initial)
    new_pose, current = accelerated.correct(initial)
    np.testing.assert_allclose(new_pose, previous_pose, atol=1e-9)
    assert current["cost"] == previous["cost"]
