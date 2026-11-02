"""Audit against analytic geometry, including deliberately wrong labels."""

import cv2
import numpy as np
import pytest

from benchmarks.flash_label_depth_audit import audit, render_depth, residuals, statistics


def test_ray_depth_matches_analytic_oblique_plane_and_frontmost_surface():
    k = np.array([[20.0, 0, 15], [0, 20, 15], [0, 0, 1]])
    vertices = np.array([[-1, -1, 0.8], [1, -1, 1.2], [1, 1, 1.2], [-1, 1, 0.8]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    rendered = render_depth(vertices, faces, k, 31, 31)
    y, x = np.nonzero(np.isfinite(rendered))
    # Independently solved ray/plane intersection: z = 1 + 0.2*x_camera.
    expected = 1 / (1 - 0.2 * (x - 15) / 20)
    np.testing.assert_allclose(rendered[y, x], expected, atol=1e-12)
    doubled_vertices = np.concatenate([vertices, vertices * 2])
    doubled_faces = np.concatenate([faces, faces + 4])
    np.testing.assert_allclose(render_depth(doubled_vertices, doubled_faces[::-1, ::-1], k, 31, 31), rendered)
    assert not np.isfinite(render_depth(vertices + [-10, 0, 0], faces, k, 31, 31)).any()
    with pytest.raises(ValueError, match="front"):
        render_depth(-vertices, faces, k, 31, 31)


def test_missing_depth_and_uncovered_pixels_are_counted_not_scored_as_matches():
    measured = np.array([[1.0, 0, np.nan, 1, 1, 1]])
    rendered = np.array([[1.02, 1, 1, np.inf, 1, 1]])
    mask = np.array([[True, True, True, True, False, True]])
    values, counts = residuals(rendered, measured, mask)
    np.testing.assert_allclose(values, [20, 0])
    assert counts == {
        "mask_pixels": 5,
        "depth_valid_pixels": 3,
        "mesh_and_depth_pixels": 2,
        "valid_depth_without_mesh_pixels": 1,
    }
    assert statistics(values)["within_10mm_pct"] == 50
    assert statistics([]) == {"pixels": 0}


def test_real_readers_render_labels_against_depth_without_fitting_them(tmp_path):
    import trimesh

    for folder in ("rgb", "depth", "masks/object"):
        (tmp_path / folder).mkdir(parents=True)
    for i in range(2):
        cv2.imwrite(str(tmp_path / "rgb" / f"{i:06d}.png"), np.zeros((20, 20, 3), np.uint8))
        cv2.imwrite(str(tmp_path / "depth" / f"{i:06d}.png"), np.full((20, 20), 1000, np.uint16))
        cv2.imwrite(str(tmp_path / "masks/object" / f"{i:06d}.png"), np.full((20, 20), 255, np.uint8))
    k = np.array([[20.0, 0, 10], [0, 20, 10], [0, 0, 1]])
    np.savez(
        tmp_path / "calibration.npz",
        times_s=[0, 1 / 30],
        intrinsics=[k, k],
        world_from_camera=[np.eye(4), np.eye(4)],
        depth_m_per_unit=[0.001, 0.001],
    )
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[:, 2, 3] = [1, 1.02]  # Second annotation deliberately wrong by 20 mm.
    np.save(tmp_path / "poses.npy", poses)
    mesh = trimesh.Trimesh(
        vertices=[[-1000, -1000, 0], [1000, -1000, 0], [1000, 1000, 0], [-1000, 1000, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    mesh.export(tmp_path / "plane.ply")
    case = {
        "kind": "rgbd_sequence",
        "source_root": str(tmp_path),
        "expected_frames": 2,
        "roles": ["object"],
        "evaluation": {"object": {"format": "poses_npy", "path": str(tmp_path / "poses.npy")}},
    }
    report = audit(case, tmp_path / "plane.ply", 0.001, "object", tmp_path / "result")
    assert report["frames"][0]["0"]["median_mm"] == pytest.approx(0)
    assert report["frames"][1]["0"]["median_mm"] == pytest.approx(20)
    assert report["aggregate"]["0"]["within_10mm_pct"] == 50
