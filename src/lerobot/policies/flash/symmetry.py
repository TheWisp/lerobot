"""Canonical object frames when an annotated axial rotation is unobservable.

An axis of continuous object symmetry defines physically equivalent poses.
This removes arbitrary fitted spin about that axis while preserving the
measured object centre and the measured direction of the axis. It does not
grant task-side freedom: an interaction may impose stricter constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import transform
from .perception import InstanceMaskPoseBackend, ObjectReference, PoseBackend, RGBDFrame
from .trajectory import Evidence, PoseEstimate


@dataclass(frozen=True)
class AxialSymmetry:
    role: str
    axis_object: np.ndarray
    center_object_m: np.ndarray
    baseline_base_from_object: np.ndarray
    lock_to_baseline: bool = False
    max_observed_tilt_rad: float | None = None

    def __post_init__(self) -> None:
        axis = np.asarray(self.axis_object, dtype=float)
        center = np.asarray(self.center_object_m, dtype=float)
        if not self.role or axis.shape != (3,) or center.shape != (3,):
            raise ValueError("axial symmetry needs a role, axis and object centre")
        if not np.isfinite(axis).all() or not np.isfinite(center).all() or np.linalg.norm(axis) < 1e-9:
            raise ValueError("axial symmetry geometry must be finite and nonzero")
        object.__setattr__(self, "axis_object", axis / np.linalg.norm(axis))
        object.__setattr__(self, "center_object_m", center)
        object.__setattr__(self, "baseline_base_from_object", transform(self.baseline_base_from_object))
        if self.lock_to_baseline and (
            self.max_observed_tilt_rad is None
            or not np.isfinite(self.max_observed_tilt_rad)
            or not 0 < self.max_observed_tilt_rad < np.pi
        ):
            raise ValueError("locked support orientation needs an observed-tilt limit")

    def canonicalize(self, base_from_object: np.ndarray) -> np.ndarray:
        observed = transform(base_from_object)
        baseline = self.baseline_base_from_object
        start_axis = baseline[:3, :3] @ self.axis_object
        observed_axis = observed[:3, :3] @ self.axis_object
        cross = np.cross(start_axis, observed_axis)
        dot = float(np.clip(np.dot(start_axis, observed_axis), -1.0, 1.0))
        if self.lock_to_baseline and np.arccos(dot) > self.max_observed_tilt_rad:
            raise ValueError("observed support axis contradicts its declared support plane")
        if dot <= -1 + 1e-8:
            raise ValueError("symmetry axis reversed; the equivalent normal is ambiguous")
        rotation = (
            Rotation.from_rotvec(
                cross / np.linalg.norm(cross) * np.arctan2(np.linalg.norm(cross), dot)
            ).as_matrix()
            if not self.lock_to_baseline and np.linalg.norm(cross) > 1e-12
            else np.eye(3)
        )
        canonical = np.eye(4)
        canonical[:3, :3] = rotation @ baseline[:3, :3]
        center_base = observed[:3, :3] @ self.center_object_m + observed[:3, 3]
        canonical[:3, 3] = center_base - canonical[:3, :3] @ self.center_object_m
        return transform(canonical)


class SymmetryCanonicalizingBackend:
    """Apply the same declared object-frame quotient to demo and live poses."""

    def __init__(self, inner: PoseBackend, symmetries: tuple[AxialSymmetry, ...]):
        self.inner = inner
        self.symmetries = {item.role: item for item in symmetries}
        self._last_failure: dict[str, str] = {}
        if len(self.symmetries) != len(symmetries):
            raise ValueError("at most one axial symmetry per role")

    @property
    def last_failure(self):
        return {**getattr(self.inner, "last_failure", {}), **self._last_failure}

    @property
    def last_masks(self) -> dict[str, np.ndarray]:
        return cast(InstanceMaskPoseBackend, self.inner).last_masks

    def initialize(self, references: tuple[ObjectReference, ...]) -> None:
        if not set(self.symmetries) <= {item.role for item in references}:
            raise ValueError("axial symmetry role lacks an object reference")
        self.inner.initialize(references)

    def update(self, frame: RGBDFrame) -> tuple[PoseEstimate, ...]:
        out = []
        self._last_failure.clear()
        for estimate in self.inner.update(frame):
            symmetry = self.symmetries.get(estimate.role)
            if symmetry is None or estimate.evidence == Evidence.LOST:
                out.append(estimate)
            else:
                try:
                    out.append(
                        replace(
                            estimate,
                            pose=symmetry.canonicalize(estimate.pose),
                            evidence=Evidence.PREDICTED if symmetry.lock_to_baseline else estimate.evidence,
                        )
                    )
                except ValueError as exc:
                    self._last_failure[estimate.role] = str(exc)
                    out.append(
                        replace(
                            estimate,
                            pose=None,
                            evidence=Evidence.LOST,
                            translation_supported=(False,) * 3,
                            rotation_supported=(False,) * 3,
                        )
                    )
        return tuple(out)
