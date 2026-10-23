"""Frame-consistent SE(3) operations used by FLASH.

``T_A_B`` maps B-frame coordinates into A. All translation units are meters.
"""

from __future__ import annotations

import numpy as np


def transform(value: np.ndarray | list[list[float]]) -> np.ndarray:
    """Validate and copy a metric rigid transform, avoiding aliased state."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("pose must be a finite 4x4 transform")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-5
    ):
        raise ValueError("pose rotation must be in SO(3)")
    return matrix.copy()


def inverse(pose: np.ndarray) -> np.ndarray:
    pose = transform(pose)
    result = np.eye(4)
    result[:3, :3] = pose[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ pose[:3, 3]
    return result


def pregrasp_goal(demo_ee: np.ndarray, demo_object: np.ndarray, live_object: np.ndarray) -> np.ndarray:
    """Keep the demonstrated gripper-to-object relation before grasp."""
    return transform(live_object) @ inverse(demo_object) @ transform(demo_ee)


def held_object_goal(
    demo_held: np.ndarray,
    demo_target: np.ndarray,
    live_target: np.ndarray,
    live_attachment: np.ndarray,
) -> np.ndarray:
    """Make the live held object follow its demo relation to the live target.

    ``live_attachment`` is ``T_E_H``, estimated from this rollout. It can be
    predicted through occlusion, but its evidence state belongs to the pose
    provider and must be checked by the caller.
    """
    desired_relation = inverse(demo_target) @ transform(demo_held)
    return transform(live_target) @ desired_relation @ inverse(live_attachment)


def interpolate(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    """Linear position and shortest-path rotation interpolation."""
    from scipy.spatial.transform import Rotation, Slerp

    a, b = transform(a), transform(b)
    u = float(np.clip(fraction, 0.0, 1.0))
    result = np.eye(4)
    result[:3, 3] = (1 - u) * a[:3, 3] + u * b[:3, 3]
    rotations = Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]]))
    result[:3, :3] = Slerp([0, 1], rotations)(u).as_matrix()
    return result
