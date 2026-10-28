"""Undirected, partial orientation evidence for a rigid object.

An observed line constrains two rotational dimensions. It says nothing about
translation, rotation about that line, or which physical end points forward.
Projection against a prior is a *candidate*, not an observed six-axis pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, isfinite, pi
from numbers import Integral

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import transform


def _unit_axis(value: tuple[float, float, float], name: str) -> np.ndarray:
    axis = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(axis))
    if axis.shape != (3,) or not np.isfinite(axis).all() or abs(norm - 1) > 1e-6:
        raise ValueError(f"{name} must be a unit 3D direction")
    return axis / norm


@dataclass(frozen=True)
class AxisProjection:
    """Nearest orientation satisfying one line measurement; position stays prior-based."""

    candidate_pose: np.ndarray
    turn_from_prior_rad: float
    branch_margin_rad: float


@dataclass(frozen=True)
class UndirectedAxisConstraint:
    """Relate a known object-frame line to a measured line in a parent frame."""

    role: str
    reference_version: str
    source_frame_id: int
    source_time_s: float
    parent_frame: str
    axis_object: tuple[float, float, float]
    axis_parent: tuple[float, float, float]

    def __post_init__(self) -> None:
        if (
            not self.role
            or not self.reference_version
            or not self.parent_frame
            or isinstance(self.source_frame_id, bool)
            or not isinstance(self.source_frame_id, Integral)
            or self.source_frame_id < 0
            or not isfinite(self.source_time_s)
        ):
            raise ValueError("axis constraint needs identity, frame and source time")
        object.__setattr__(self, "axis_object", tuple(_unit_axis(self.axis_object, "object axis")))
        object.__setattr__(self, "axis_parent", tuple(_unit_axis(self.axis_parent, "measured axis")))

    @property
    def observable_rotation_projector_object(self) -> np.ndarray:
        """Rank-two tangent projector; the object-axis rotation is unresolved."""
        axis = np.asarray(self.axis_object)
        return np.eye(3) - np.outer(axis, axis)

    def residual_rad(self, parent_from_object: np.ndarray) -> float:
        pose = transform(parent_from_object)
        predicted = pose[:3, :3] @ self.axis_object
        measured = np.asarray(self.axis_parent)
        return atan2(float(np.linalg.norm(np.cross(predicted, measured))), abs(float(predicted @ measured)))

    def closest_pose(
        self, prior_parent_from_object: np.ndarray, *, pivot_object_m: np.ndarray
    ) -> AxisProjection:
        """Rotate the prior minimally about its pivot to satisfy the measured line.

        The sign is selected by least rotation; ``branch_margin_rad`` records
        how much worse the opposite end assignment would be. A near-zero
        margin signals an unresolved end assignment, not high confidence.
        """
        prior = transform(prior_parent_from_object)
        pivot = np.asarray(pivot_object_m, dtype=float)
        if pivot.shape != (3,) or not np.isfinite(pivot).all():
            raise ValueError("axis projection needs a finite object-frame pivot")
        current = prior[:3, :3] @ self.axis_object
        measured = np.asarray(self.axis_parent)
        if current @ measured < 0:
            measured = -measured
        cross = np.cross(current, measured)
        sine = float(np.linalg.norm(cross))
        cosine = float(np.clip(current @ measured, 0, 1))
        turn = atan2(sine, cosine)
        delta = (
            np.eye(3)
            if sine <= np.finfo(float).eps
            else Rotation.from_rotvec(cross / sine * turn).as_matrix()
        )
        candidate = prior.copy()
        candidate[:3, :3] = delta @ prior[:3, :3]
        prior_pivot = prior[:3, :3] @ pivot + prior[:3, 3]
        candidate[:3, 3] = prior_pivot - candidate[:3, :3] @ pivot
        return AxisProjection(candidate, turn, pi - 2 * turn)
