"""Known geometry checks for the whole-shape correction and foreground handling."""

import cv2
import numpy as np
from compare import move
from scipy.spatial.transform import Rotation
from shape_grounding import ShapeConfig, WholeSurfaceFit, project


def fixture():
    x, y = np.meshgrid(np.linspace(-0.05, 0.05, 31), np.linspace(-0.025, 0.025, 17))
    keep = (x < 0.015) | (y < 0.005)
    points = np.column_stack((x[keep], y[keep], 0.55 + 0.2 * x[keep] ** 2 / 0.05))
    k = np.array([[600.0, 0, 160], [0, 600.0, 120], [0, 0, 1]])
    return points, k


def observation(points, k, foreground=None):
    depth = np.ones((240, 320), np.float32)
    mask = np.zeros(depth.shape, np.uint8)
    for point, pixel in zip(points, np.rint(project(points, k)).astype(int), strict=True):
        cv2.circle(depth, tuple(pixel), 3, float(point[2]), -1)
        cv2.circle(mask, tuple(pixel), 3, 1, -1)
    # The rasterized disks must not enlarge the known object's outer boundary.
    envelope = np.zeros_like(mask)
    cv2.fillConvexPoly(envelope, cv2.convexHull(np.rint(project(points, k)).astype(np.int32)), 1)
    mask &= envelope
    if foreground is not None:
        depth[:, :foreground] = 0.4
        mask[:, :foreground] = 0
    return {"depth": depth, "mask": mask.astype(bool), "obs": {"points": points}}


def test_whole_shape_corrects_known_tilt_without_point_correspondences():
    source, k = fixture()
    truth = np.eye(4)
    truth[:3, :3] = Rotation.from_euler("xyz", [0.05, -0.1, 0.12]).as_matrix()
    truth[:3, 3] = [0.012, -0.006, 0.018]
    target = move(source, truth)
    fit = WholeSurfaceFit(source, observation(target, k), k, ShapeConfig())
    wrong = truth.copy()
    delta = Rotation.from_euler("y", 23, degrees=True).as_matrix()
    center = target.mean(0)
    wrong[:3, :3] = delta @ truth[:3, :3]
    wrong[:3, 3] = delta @ (truth[:3, 3] - center) + center
    corrected, _ = fit.correct(wrong)
    error = np.mean(np.linalg.norm(move(source, corrected) - target, axis=1))
    # Disabling the new correction leaves about 10 mm of known surface error.
    assert np.mean(np.linalg.norm(move(source, wrong) - target, axis=1)) > 0.008
    assert error < 0.001


def test_nearer_foreground_does_not_pull_hidden_reference_points():
    source, k = fixture()
    visible = source[source[:, 0] > 0.005]
    fit = WholeSurfaceFit(source, observation(visible, k, foreground=166), k, ShapeConfig())
    pose, _ = fit.correct(np.eye(4))
    assert np.mean(np.linalg.norm(move(source, pose) - source, axis=1)) < 0.001


def test_pose_cannot_evade_shape_check_by_hiding_behind_depth():
    source, k = fixture()
    fit = WholeSurfaceFit(source, observation(source, k), k, ShapeConfig())
    behind = np.eye(4)
    behind[2, 3] = 0.05
    assert fit.score(np.eye(4)) < 1e-8
    assert fit.score(behind) > 10


def test_missing_depth_is_not_evidence_that_visible_shape_is_absent():
    source, k = fixture()
    data = observation(source, k)
    data["depth"][:, :166] = 0
    data["obs"]["points"] = source[source[:, 0] > 0.005]
    fit = WholeSurfaceFit(source, data, k, ShapeConfig())
    pose, _ = fit.correct(np.eye(4))
    assert np.mean(np.linalg.norm(move(source, pose) - source, axis=1)) < 0.001
