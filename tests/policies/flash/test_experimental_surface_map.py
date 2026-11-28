"""A candidate map must not manufacture evidence or move its original reference."""

import numpy as np
import pytest

from lerobot.policies.flash.experimental_surface_map import CandidateSurfaceMap, SurfaceMapConfig
from lerobot.policies.flash.surface_observation import move


def test_new_surface_needs_three_distinct_confirmed_views_and_keeps_provenance():
    reference = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
    extra = np.array([[0.01, 0.01, 0.005]])
    model = CandidateSurfaceMap(
        reference, SurfaceMapConfig(spacing_m=0.001, novelty_distance_m=0.002, max_points=4), source_frame=0
    )
    pose = np.eye(4)
    pose[:3, 3] = [0.1, 0.2, 0.3]
    measured = move(np.concatenate((reference, extra)), pose)
    assert model.observe(measured, pose, 1)["promoted"] == 0
    assert model.observe(measured, pose, 2)["promoted"] == 0
    with pytest.raises(ValueError, match="newer camera frame"):
        model.observe(measured, pose, 2)
    assert model.observe(measured, pose, 3)["promoted"] == 1
    np.testing.assert_allclose(model.points[:3], reference)
    np.testing.assert_allclose(model.points[3], extra[0])
    assert model.provenance[3] == (1, 2, 3)
    assert model.observe(measured, pose, 4)["map_points"] == 4


def test_initial_reference_cannot_be_silently_thinned_to_fit_capacity():
    reference = np.zeros((4, 3))
    with pytest.raises(ValueError, match="capacity"):
        CandidateSurfaceMap(
            reference,
            SurfaceMapConfig(spacing_m=0.001, novelty_distance_m=0.002, max_points=3),
            source_frame=0,
        )


def test_repeated_depth_noise_does_not_become_new_surface():
    reference = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
    model = CandidateSurfaceMap(
        reference,
        SurfaceMapConfig(spacing_m=0.001, novelty_distance_m=0.005, max_points=10),
        source_frame=0,
    )
    noisy = reference + np.array([0.0, 0.0, 0.003])
    for frame_id in range(1, 10):
        assert model.observe(noisy, np.eye(4), frame_id)["promoted"] == 0
    assert len(model.points) == len(reference)
    with pytest.raises(ValueError, match="positive spacing"):
        SurfaceMapConfig(spacing_m=0.003, novelty_distance_m=0.002)


def test_opposite_face_is_distinct_even_within_depth_noise_distance():
    reference = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
    front_normals = np.tile([0.0, 0.0, 1.0], (3, 1))
    back = np.array([[0.005, 0.005, 0.003]])
    back_normals = np.array([[0.0, 0.0, -1.0]])
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
    pose[:3, 3] = [0.1, 0.2, 0.5]
    measured_back = move(back, pose)
    measured_normals = back_normals @ pose[:3, :3].T
    model = CandidateSurfaceMap(
        reference,
        SurfaceMapConfig(spacing_m=0.001, novelty_distance_m=0.009, max_points=4),
        source_frame=0,
        reference_normals=front_normals,
    )
    for frame_id in (1, 2):
        assert (
            model.observe(measured_back, pose, frame_id, measured_normals_camera=measured_normals)["promoted"]
            == 0
        )
    assert model.observe(measured_back, pose, 3, measured_normals_camera=measured_normals)["promoted"] == 1
    assert model.observe(measured_back, pose, 4, measured_normals_camera=measured_normals)["promoted"] == 0
    np.testing.assert_allclose(model.points[-1], back[0])
    np.testing.assert_allclose(model.normals[-1], back_normals[0])
