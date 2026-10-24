"""Geometry observability and motion ownership, with known physical transforms."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.registration import transform_points
from lerobot.policies.flash.surface_motion import (
    PendingMotionCorrection,
    RigidMotionState,
    SurfaceMotionConfig,
    surface_support,
)


def planar_cloud():
    x, y = np.meshgrid(np.linspace(-0.06, 0.06, 41), np.linspace(-0.04, 0.04, 31))
    return np.c_[x.ravel(), y.ravel(), np.full(x.size, 0.6)]


def test_plane_observes_tilt_and_normal_translation_but_not_surface_sliding():
    points = planar_cloud()
    support = surface_support(points, np.eye(4), points, SurfaceMotionConfig())
    assert support.basis.shape[1] == 3
    # No x/y translation or spin around the plane normal, despite >1,000 points.
    assert np.max(np.abs(support.basis[[2, 3, 4]])) < 1e-8
    assert np.linalg.matrix_rank(support.basis[[0, 1, 5]]) == 3


def test_tiny_visible_patch_does_not_gain_full_confidence_by_having_dense_points():
    points = planar_cloud()
    patch = points[np.linalg.norm(points[:, :2], axis=1) < 0.01]
    config = SurfaceMotionConfig()
    sparse = surface_support(points, np.eye(4), patch, config)
    dense = surface_support(points, np.eye(4), np.repeat(patch, 20, axis=0), config)
    # One small patch can measure its normal displacement, not all six axes.
    assert sparse.basis.shape[1] == dense.basis.shape[1] == 1
    np.testing.assert_allclose(sparse.sensitivity, dense.sensitivity, atol=1e-12)


def test_observability_does_not_depend_on_object_coordinate_origin():
    points = planar_cloud()
    pose = np.eye(4)
    pose[:3, 3] = [2, -4, 1]
    first = surface_support(points, np.eye(4), points, SurfaceMotionConfig())
    second = surface_support(points - pose[:3, 3], pose, points, SurfaceMotionConfig())
    np.testing.assert_allclose(first.sensitivity, second.sensitivity, atol=1e-10)
    np.testing.assert_allclose(first.basis @ first.basis.T, second.basis @ second.basis.T, atol=1e-8)


def test_gripper_carries_accepted_state_and_spans_missing_sample_without_inventing_observation():
    points = planar_cloud()
    state = RigidMotionState()
    ee = np.eye(4)
    assert state.predict(0, "grasp-1", ee) is None
    state.accept(np.eye(4), measured=True, carrier=ee)
    moved = np.eye(4)
    moved[:3, :3] = Rotation.from_euler("x", 30, degrees=True).as_matrix()
    moved[:3, 3] = [0.01, -0.02, 0.04]
    prior = state.predict(1, "grasp-1", moved)
    np.testing.assert_allclose(transform_points(prior, points), transform_points(moved, points))
    assert state.last_measurement_s == 0
    assert state.predict(2, "grasp-1", None) is None
    moved_again = moved.copy()
    moved_again[:3, 3] += [0.03, 0, 0]
    np.testing.assert_allclose(state.predict(3, "grasp-1", moved_again), moved_again, atol=1e-12)


def test_release_withdraws_gripper_motion():
    state = RigidMotionState()
    state.predict(0, "grasp-1", np.eye(4))
    state.accept(np.eye(4), measured=True, carrier=np.eye(4))
    assert state.predict(1, None, None) is None
    assert state.carrier is None
    with pytest.raises(ValueError, match="increasing"):
        state.predict(1, None, None)


def test_reanchored_group_moves_accepted_state_not_its_own_visual_anchor():
    state = RigidMotionState()
    accepted = np.eye(4)
    accepted[:3, 3] = [0.1, 0.2, 0.6]
    state.predict(0, None, None)
    state.accept(accepted, measured=True)
    # The group's visual anchor can be wrong; only its independently measured
    # rigid motion is useful. The caller supplies motion @ accepted-at-anchor.
    group_motion = np.eye(4)
    group_motion[0, 3] = 0.02
    result = state.predict(1, "group-verified", group_motion, anchor_pose=accepted)
    np.testing.assert_allclose(result[:3, 3], [0.12, 0.2, 0.6])
    assert state.last_measurement_s == 0


def test_unobserved_carrier_after_visual_reacquisition_does_not_apply_motion_twice():
    state = RigidMotionState()
    state.predict(0, "grasp", np.eye(4))
    state.accept(np.eye(4), measured=True, carrier=np.eye(4))
    assert state.predict(1, "grasp", None) is None
    current = np.eye(4)
    current[0, 3] = 0.02
    state.accept(current, measured=True)  # image exists, synchronized EE does not
    assert state.predict(2, "grasp", current) is None


@pytest.mark.parametrize("next_relationship", [None, "different-grasp"])
def test_pending_visual_correction_does_not_survive_release_or_new_grasp(next_relationship):
    pending = PendingMotionCorrection()
    wrong, good = np.eye(4), np.eye(4)
    wrong[0, 3] = 0.02

    def score(pose):
        return (pose[0, 3] / 0.005) ** 2

    pending.propose("grasp", np.eye(4), wrong, good, score, 1)
    assert pending.check(next_relationship, np.eye(4), wrong, score, 1) is None


def test_candidate_must_improve_a_different_frame_and_follow_actual_carrier_motion():
    pending = PendingMotionCorrection()
    wrong, good = np.eye(4), np.eye(4)
    wrong[2, 3] = 0.02

    def score(pose):
        return (pose[2, 3] / 0.005) ** 2

    pending.propose("grasp", np.eye(4), wrong, good, score, 1)
    motion = np.eye(4)
    motion[2, 3] = 0.03

    # A different current frame agrees with the old prediction: reject the
    # earlier image's tempting but contradicted correction.
    def current_score(pose):
        return ((pose[2, 3] - 0.05) / 0.005) ** 2

    assert pending.check("grasp", motion, motion @ wrong, current_score, 1) is None
    pending.propose("grasp", np.eye(4), wrong, good, score, 1)

    # Now the next observation agrees with the proposed correction. A no-op
    # implementation, or one forgetting to transport it by EE, fails here.
    def current_score(pose):
        return ((pose[2, 3] - 0.03) / 0.005) ** 2

    result = pending.check("grasp", motion, motion @ wrong, current_score, 1)
    np.testing.assert_allclose(result, motion @ good)
