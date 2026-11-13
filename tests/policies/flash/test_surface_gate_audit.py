"""Guard against diagnosing one map/pose while actually evaluating another."""

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from benchmarks.flash_surface_gate_audit import audit_pose, place_measurements, relative_labels
from lerobot.policies.flash.surface_grounding import ShapeConfig
from lerobot.policies.flash.surface_observation import move


def test_label_placement_preserves_samples_and_uses_inverse_camera_transform():
    points = np.array([[0.02, 0.01, 0.5], [0.01, 0.04, 0.6], [-0.05, 0.02, 0.4]])
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [23, -47, 12], degrees=True).as_matrix()
    pose[:3, 3] = [0.1, -0.2, 0.03]
    measured = move(points, pose)
    patch = SimpleNamespace(frame=3, measured=measured.copy())
    placed = place_measurements([patch], {3: pose})
    np.testing.assert_allclose(placed, points, atol=1e-15)
    np.testing.assert_array_equal(patch.measured, measured)
    # A different placement must actually change the diagnostic map.
    wrong = place_measurements([patch], {3: np.eye(4)})
    assert np.linalg.norm(wrong - placed) > 0.1


def test_relative_labels_use_session_initialization_frame_not_dataset_frame_zero():
    poses = np.tile(np.eye(4), (4, 1, 1))
    poses[:, 0, 3] = [0.1, 0.2, 0.3, 0.4]
    result = relative_labels(poses, 2)
    np.testing.assert_allclose(result[2], np.eye(4), atol=1e-15)
    np.testing.assert_allclose(result[:, 0, 3], [-0.2, -0.1, 0, 0.1], atol=1e-15)


def test_same_true_pose_passes_correct_map_and_fails_misplaced_map_without_mutation():
    x, y = np.meshgrid(np.linspace(-0.04, 0.04, 9), np.linspace(-0.03, 0.03, 7))
    points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.5)))
    current = {
        "depth": np.full((80, 80), 0.5),
        "mask": np.ones((80, 80), bool),
        "obs": {"points": points.copy()},
    }
    k = np.array([[200.0, 0, 40], [0, 200.0, 40], [0, 0, 1]])
    maps = {"misplaced": points - [0, 0, 0.02], "correct": points.copy()}
    frozen = {key: value.copy() for key, value in maps.items()}
    pose = np.eye(4)
    result = audit_pose(
        maps,
        {"current_only": ()},
        current,
        k,
        ShapeConfig(),
        {"max_outlier_fraction": 0.25, "agreement_m": 0.005, "min_overlap": 0.75},
        pose,
    )
    assert result["correct/current_only"]["accepted"]
    assert not result["misplaced/current_only"]["accepted"]
    assert result["misplaced/current_only"]["scene_depth_coherent_protrusion"]
    for key in maps:
        np.testing.assert_array_equal(maps[key], frozen[key])
    np.testing.assert_array_equal(pose, np.eye(4))
    np.testing.assert_array_equal(current["obs"]["points"], points)
    np.testing.assert_array_equal(current["depth"], np.full((80, 80), 0.5))
