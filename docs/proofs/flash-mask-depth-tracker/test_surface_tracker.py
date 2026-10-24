"""Known rigid motion and signed depth visibility checks, independent of the demo."""

import json
from pathlib import Path

import numpy as np
from run_surface_tracker import evidence, fit_surface, largest_connected_surface, move, observation
from scipy.spatial.transform import Rotation


def settings():
    return json.loads(Path(__file__).with_name("config.json").read_text())


def test_detached_depth_island_does_not_become_a_landmark():
    x, y = np.meshgrid(np.arange(10) * 0.003, np.arange(10) * 0.003)
    surface = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.5)))
    detached = surface[:5] + [0, 0, 0.1]
    keep = largest_connected_surface(np.concatenate((surface, detached)), 0.01)
    assert keep[: len(surface)].all()
    assert not keep[len(surface) :].any()


def test_recovers_known_rigid_motion_of_asymmetric_surface():
    rng = np.random.default_rng(7)
    sphere = rng.normal(size=(1500, 3))
    sphere /= np.linalg.norm(sphere, axis=1)[:, None]
    radii = np.array([0.06, 0.025, 0.015])
    model = sphere * radii
    normals = sphere / radii
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    expected = np.eye(4)
    expected[:3, :3] = Rotation.from_rotvec([0.08, -0.05, 0.06]).as_matrix()
    expected[:3, 3] = [0.01, -0.005, 0.7]
    observed = {
        "points": move(model, expected),
        "normals": normals @ expected[:3, :3].T,
        "fit": np.ones(len(model), dtype=bool),
    }
    initial = np.eye(4)
    initial[:3, 3] = [0, 0, 0.7]
    actual, _ = fit_surface(model, normals, observed, initial, settings())
    assert np.linalg.norm(initial[:3, 3] - expected[:3, 3]) > 0.01
    assert np.linalg.norm(actual[:3, 3] - expected[:3, 3]) < 0.001
    assert Rotation.from_matrix(actual[:3, :3] @ expected[:3, :3].T).magnitude() < 0.01


def test_occluder_is_unknown_but_observed_empty_space_contradicts():
    model = np.array([[0, 0, 0.7]])
    intrinsics = np.array([[100, 0, 4], [0, 100, 4], [0, 0, 1]])
    observed = {"points": model, "fit": np.array([False]), "depth_reliable": np.ones((9, 9), dtype=bool)}
    mask = np.ones((9, 9), dtype=bool)
    counts = []
    for depth in [0.6, 0.7, 0.8]:
        result = evidence(model, np.eye(4), observed, np.full((9, 9), depth), mask, intrinsics, settings())
        counts.append((result["supported"], result["contradicted"]))
    assert counts == [(0, 0), (1, 0), (0, 1)]


def test_excludes_boundary_bleed_holes_and_depth_steps_without_moving_good_points():
    depth = np.full((40, 40), 0.7, dtype=np.float32)
    mask = np.zeros_like(depth, dtype=bool)
    mask[5:35, 5:35] = True
    depth[:, 20:] = 0.65
    depth[12, 12] = 0
    intrinsics = np.array([[100, 0, 20], [0, 100, 20], [0, 0, 1]])
    result = observation(depth, mask, intrinsics, settings())
    assert not result["valid_image"][5:7].any()
    assert not result["valid_image"][:, 19:21].any()
    assert not result["valid_image"][11:14, 11:14].any()
    assert result["valid_image"][15, 15]
    assert result["valid_image"][25, 25]
    # Filtering is not hole filling, nor is the smoothed normal map used as XYZ.
    assert np.all(np.isin(result["points"][:, 2], np.array([0.65, 0.7], np.float32)))
