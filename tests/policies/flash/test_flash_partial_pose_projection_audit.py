"""Physical invariants of the offline partial-pose projection ablation."""

import numpy as np
from scipy.spatial.transform import Rotation

from benchmarks.flash_partial_pose_projection_audit import partial_pose


def test_unsupported_motion_keeps_prior_and_supported_turn_preserves_pivot():
    prior = np.eye(4)
    candidate = np.eye(4)
    candidate[:3, :3] = Rotation.from_euler("z", 0.2).as_matrix()
    candidate[:3, 3] = [0.03, 0, 0]
    pivot_object = np.array([0.02, 0, 0])
    info = {
        "pivot_base_m": (candidate[:3, :3] @ pivot_object + candidate[:3, 3]).tolist(),
        "twist_basis_base": np.eye(6).tolist(),
        # The measured surface constrains only a turn about z. It says
        # nothing about the visually proposed translation.
        "strength": [0, 0, 1, 0, 0, 0],
    }

    projected, weights = partial_pose(prior, candidate, info, prior_rms_m=0.01, depth_error_m=0.005)

    assert np.allclose(weights, [0, 0, 0.8, 0, 0, 0])
    assert np.isclose(Rotation.from_matrix(projected[:3, :3]).magnitude(), 0.16)
    assert np.allclose(projected[:3, :3] @ pivot_object + projected[:3, 3], pivot_object)
    assert not np.allclose(projected[:3, 3], candidate[:3, 3])

    info["twist_basis_base"] = np.empty((6, 0)).tolist()
    info["strength"] = [0] * 6
    unobserved, weights = partial_pose(prior, candidate, info, prior_rms_m=0.01, depth_error_m=0.005)
    assert np.array_equal(unobserved, prior)
    assert np.array_equal(weights, np.zeros(6))
