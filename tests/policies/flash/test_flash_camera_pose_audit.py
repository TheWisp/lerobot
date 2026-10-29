"""A camera-pose audit must expose source inconsistency without adopting a fit."""

import cv2
import numpy as np
import pytest

from benchmarks.flash_camera_pose_audit import compare_pose_to_background


def test_camera_pose_audit_checks_a_pnp_candidate_on_unseen_background_points():
    random = np.random.default_rng(42)
    world = random.uniform([-0.25, -0.2, 0.5], [0.25, 0.2, 1.2], (100, 3))
    ids = np.arange(len(world))
    intrinsics = np.array([[200.0, 0.0, 96.0], [0.0, 200.0, 128.0], [0.0, 0.0, 1.0]])
    true_camera_from_world = np.eye(4)
    true_camera_from_world[:3, :3], _ = cv2.Rodrigues(np.array([0.02, -0.03, 0.01]))
    true_camera_from_world[:3, 3] = [0.025, -0.01, 0.015]
    camera = world @ true_camera_from_world[:3, :3].T + true_camera_from_world[:3, 3]
    xy = np.column_stack((camera[:, 0] * 200 / camera[:, 2] + 96, camera[:, 1] * 200 / camera[:, 2] + 128))
    xy += random.normal(0, 0.1, xy.shape)
    # Moving foreground slipped into the unmasked point set. It should not
    # govern the fitted camera pose or its held-out verification.
    xy[1:10] += [50, -30]
    recorded_world_from_camera = np.eye(4)
    result = compare_pose_to_background(world, xy, ids, recorded_world_from_camera, intrinsics)
    assert result["source"]["median_px"] > 5
    assert result["pnp"]["training_inliers"] >= 60
    assert result["pnp"]["heldout_fit"]["median_px"] < 0.3
    assert result["pnp"]["heldout_source"]["median_px"] > 5
    assert result["pnp"]["source_delta_translation_m"] == pytest.approx(0.03, abs=0.005)
    candidate = np.asarray(result["pnp"]["candidate_world_from_camera"])
    assert np.linalg.norm(candidate[:3, 3] - np.linalg.inv(true_camera_from_world)[:3, 3]) < 0.005


def test_camera_pose_audit_reports_insufficient_tracks_without_inventing_a_pose():
    result = compare_pose_to_background(
        np.zeros((3, 3)),
        np.zeros((3, 2)),
        np.arange(3),
        np.eye(4),
        np.eye(3),
    )
    assert result["tracks"] == 3
    assert result["pnp"] is None


def test_camera_pose_audit_omits_points_behind_the_camera_from_pixel_error():
    result = compare_pose_to_background(
        np.array([[0.0, 0.0, -1.0]]),
        np.array([[5.0, 5.0]]),
        np.array([0]),
        np.eye(4),
        np.eye(3),
    )
    assert result["source"] is None
