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
    ends_interchangeable: bool = False

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

    def equivalent_transform(self, angle_rad: float = 0.0, *, exchange_ends: bool = False) -> np.ndarray:
        """Object-local rotation about its axis, optionally exchanging equal ends.

        The transform rotates about the annotated physical centre. It is a
        planning alternative, not a new pose measurement: the scene estimator
        retains its canonical object frame and the grasp attachment uses it.
        """
        if not np.isfinite(angle_rad):
            raise ValueError("symmetry angle must be finite")
        if exchange_ends and not self.ends_interchangeable:
            raise ValueError("object ends were not annotated as interchangeable")
        rotation = Rotation.from_rotvec(angle_rad * self.axis_object).as_matrix()
        if exchange_ends:
            basis = np.eye(3)[int(np.argmin(np.abs(self.axis_object)))]
            transverse = basis - self.axis_object * np.dot(basis, self.axis_object)
            transverse /= np.linalg.norm(transverse)
            rotation = rotation @ Rotation.from_rotvec(np.pi * transverse).as_matrix()
        result = np.eye(4)
        result[:3, :3] = rotation
        result[:3, 3] = self.center_object_m - rotation @ self.center_object_m
        return transform(result)

    def canonicalize(
        self, base_from_object: np.ndarray, *, reference_pose: np.ndarray | None = None
    ) -> np.ndarray:
        """Remove axial gauge using the previous pose when end identity is ambiguous.

        A fixed first-view reference has a branch cut at 90 degrees for an
        end-interchangeable axis. Following a prior pose lifts that unoriented
        axis continuously while adjacent observations support the motion.
        """
        observed = transform(base_from_object)
        baseline = self.baseline_base_from_object
        reference = (
            transform(reference_pose)
            if reference_pose is not None and not self.lock_to_baseline
            else baseline
        )
        start_axis = reference[:3, :3] @ self.axis_object
        observed_axis = observed[:3, :3] @ self.axis_object
        if self.ends_interchangeable and np.dot(start_axis, observed_axis) < 0:
            observed_axis = -observed_axis
        cross = np.cross(start_axis, observed_axis)
        dot = float(np.clip(np.dot(start_axis, observed_axis), -1.0, 1.0))
        if self.lock_to_baseline and np.arccos(dot) > self.max_observed_tilt_rad:
            raise ValueError("observed support axis contradicts its declared support plane")
        if dot <= -1 + 1e-8:
            raise ValueError("symmetry axis reversed; object ends were not declared interchangeable")
        if self.lock_to_baseline:
            rotation = np.eye(3)
        elif np.linalg.norm(cross) > 1e-12:
            rotation = Rotation.from_rotvec(
                cross / np.linalg.norm(cross) * np.arctan2(np.linalg.norm(cross), dot)
            ).as_matrix()
        else:
            rotation = np.eye(3)
        canonical = np.eye(4)
        canonical[:3, :3] = rotation @ reference[:3, :3]
        center_base = observed[:3, :3] @ self.center_object_m + observed[:3, 3]
        canonical[:3, 3] = center_base - canonical[:3, :3] @ self.center_object_m
        return transform(canonical)


class SymmetryCanonicalizingBackend:
    """Apply the same declared object-frame quotient to demo and live poses."""

    def __init__(self, inner: PoseBackend, symmetries: tuple[AxialSymmetry, ...]):
        self.inner = inner
        self.symmetries = {item.role: item for item in symmetries}
        self._last_failure: dict[str, str] = {}
        self._last_pose: dict[str, np.ndarray] = {}
        if len(self.symmetries) != len(symmetries):
            raise ValueError("at most one axial symmetry per role")

    @property
    def last_failure(self):
        return {**getattr(self.inner, "last_failure", {}), **self._last_failure}

    @property
    def last_details(self):
        return getattr(self.inner, "last_details", {})

    @property
    def last_masks(self) -> dict[str, np.ndarray]:
        return cast(InstanceMaskPoseBackend, self.inner).last_masks

    @property
    def last_candidate_poses(self) -> dict[str, np.ndarray]:
        """Keep rejected raw fits inspectable after symmetry canonicalization."""
        return getattr(self.inner, "last_candidate_poses", {})

    @property
    def last_visible_support(self):
        """Preserve measured partial evidence even when no full pose survives."""
        return getattr(self.inner, "last_visible_support", {})

    def initialize(self, references: tuple[ObjectReference, ...]) -> None:
        if not set(self.symmetries) <= {item.role for item in references}:
            raise ValueError("axial symmetry role lacks an object reference")
        self._last_pose.clear()
        self.inner.initialize(references)

    @property
    def can_pipeline(self):
        return getattr(self.inner, "can_pipeline", False)

    def prepare(self, frame):
        return self.inner.prepare(frame)

    def finish(self, prepared):
        return self._canonicalize(self.inner.finish(prepared))

    def close(self):
        close = getattr(self.inner, "close", None)
        if close is not None:
            close()

    def update(self, frame: RGBDFrame) -> tuple[PoseEstimate, ...]:
        return self._canonicalize(self.inner.update(frame))

    def _canonicalize(self, estimates):
        out = []
        self._last_failure.clear()
        for estimate in estimates:
            symmetry = self.symmetries.get(estimate.role)
            if symmetry is None or estimate.evidence == Evidence.LOST:
                out.append(estimate)
            else:
                try:
                    pose = symmetry.canonicalize(
                        estimate.pose, reference_pose=self._last_pose.get(estimate.role)
                    )
                    self._last_pose[estimate.role] = pose
                    out.append(
                        replace(
                            estimate,
                            pose=pose,
                            evidence=Evidence.PREDICTED if symmetry.lock_to_baseline else estimate.evidence,
                            symmetry_canonicalized=True,
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
