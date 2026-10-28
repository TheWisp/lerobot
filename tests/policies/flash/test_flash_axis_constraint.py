import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.axis_constraint import UndirectedAxisConstraint


def _constraint(measured: np.ndarray) -> UndirectedAxisConstraint:
    object_axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    return UndirectedAxisConstraint(
        "tool", "reference-v1", 12, 2.5, "gripper", tuple(object_axis), tuple(measured)
    )


def test_partial_axis_projection_preserves_pivot_and_unobserved_roll() -> None:
    prior = np.eye(4)
    prior[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.4, 0.3]).as_matrix()
    prior[:3, 3] = [0.1, -0.2, 0.3]
    object_axis = np.asarray(_constraint(np.array([1.0, 0.0, 0.0])).axis_object)
    predicted_axis = prior[:3, :3] @ object_axis
    measured = Rotation.from_rotvec(np.array([0.0, 0.0, 0.4])).apply(predicted_axis)
    constraint = _constraint(measured)
    pivot = np.array([0.02, 0.04, -0.03])

    projection = constraint.closest_pose(prior, pivot_object_m=pivot)
    corrected = projection.candidate_pose
    np.testing.assert_allclose(corrected[:3, :3] @ object_axis, measured, atol=1e-12)
    np.testing.assert_allclose(
        corrected[:3, :3] @ pivot + corrected[:3, 3], prior[:3, :3] @ pivot + prior[:3, 3]
    )
    np.testing.assert_allclose(constraint.observable_rotation_projector_object @ object_axis, 0, atol=1e-12)
    assert np.linalg.matrix_rank(constraint.observable_rotation_projector_object) == 2
    assert constraint.residual_rad(corrected) == pytest.approx(0, abs=1e-12)
    delta_axis_angle = Rotation.from_matrix(corrected[:3, :3] @ prior[:3, :3].T).as_rotvec()
    assert delta_axis_angle @ predicted_axis == pytest.approx(0, abs=1e-12)
    assert projection.turn_from_prior_rad > 0
    assert projection.branch_margin_rad > 0

    opposite = _constraint(-measured).closest_pose(prior, pivot_object_m=pivot)
    np.testing.assert_allclose(opposite.candidate_pose, corrected, atol=1e-12)


def test_orthogonal_line_has_no_preferred_end_assignment() -> None:
    object_axis = np.asarray(_constraint(np.array([1.0, 0.0, 0.0])).axis_object)
    measured = np.cross(object_axis, np.array([0.0, 0.0, 1.0]))
    projection = _constraint(measured).closest_pose(np.eye(4), pivot_object_m=np.zeros(3))
    assert projection.turn_from_prior_rad == pytest.approx(np.pi / 2)
    assert projection.branch_margin_rad == pytest.approx(0)


def test_axis_constraint_rejects_invalid_measurement() -> None:
    with pytest.raises(ValueError, match="unit 3D direction"):
        _constraint(np.array([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="finite object-frame pivot"):
        _constraint(np.array([1.0, 0.0, 0.0])).closest_pose(
            np.eye(4), pivot_object_m=np.array([np.nan, 0.0, 0.0])
        )
