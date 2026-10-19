"""Bounded pose coasting through short, annotated visibility gaps.

The wrapper never presents a carried-forward pose as an observation. It is
intended for intervals where the scene object is expected to stay fixed, such
as an approach that obscures the object immediately before grasp.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .perception import ObjectReference, PoseBackend, RGBDFrame
from .trajectory import Evidence, PoseEstimate


@dataclass(frozen=True)
class PoseCoastLimit:
    role: str
    max_age_s: float
    max_translation_m: float
    max_rotation_rad: float
    center_object_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.role or any(
            not np.isfinite(value) or value <= 0
            for value in (self.max_age_s, self.max_translation_m, self.max_rotation_rad)
        ):
            raise ValueError("pose coasting needs a role and positive finite bounds")
        if self.center_object_m is not None:
            center = np.asarray(self.center_object_m, dtype=float)
            if center.shape != (3,) or not np.isfinite(center).all():
                raise ValueError("pose-coasting object centre must be a finite 3-vector")
            object.__setattr__(self, "center_object_m", center)


class PoseContinuityBackend:
    def __init__(self, inner: PoseBackend, limits: tuple[PoseCoastLimit, ...]) -> None:
        self.inner = inner
        self.limits = {item.role: item for item in limits}
        if len(self.limits) != len(limits):
            raise ValueError("duplicate pose-coasting role")
        self._anchor: dict[str, PoseEstimate] = {}

    @property
    def last_failure(self):
        return getattr(self.inner, "last_failure", {})

    def initialize(self, references: tuple[ObjectReference, ...]) -> None:
        if not set(self.limits) <= {item.role for item in references}:
            raise ValueError("pose-coasting role lacks an object reference")
        self._anchor.clear()
        self.inner.initialize(references)

    def update(self, frame: RGBDFrame) -> tuple[PoseEstimate, ...]:
        out = []
        for estimate in self.inner.update(frame):
            limit = self.limits.get(estimate.role)
            if limit is None:
                out.append(estimate)
                continue
            anchor = self._anchor.get(estimate.role)
            consistent = estimate.evidence == Evidence.OBSERVED and estimate.pose is not None
            if consistent and anchor is not None:
                center = limit.center_object_m
                if center is None:
                    center = np.zeros(3)
                observed_center = estimate.pose[:3, :3] @ center + estimate.pose[:3, 3]
                anchor_center = anchor.pose[:3, :3] @ center + anchor.pose[:3, 3]
                shift = np.linalg.norm(observed_center - anchor_center)
                turn = np.arccos(
                    np.clip((np.trace(anchor.pose[:3, :3].T @ estimate.pose[:3, :3]) - 1) / 2, -1, 1)
                )
                consistent = shift <= limit.max_translation_m and turn <= limit.max_rotation_rad
            if consistent:
                self._anchor[estimate.role] = estimate
                out.append(estimate)
            elif anchor is not None and 0 <= frame.source_time_s - anchor.source_time_s <= limit.max_age_s:
                out.append(
                    replace(
                        anchor,
                        source_time_s=frame.source_time_s,
                        available_time_s=frame.available_time_s,
                        evidence=Evidence.PREDICTED,
                    )
                )
            else:
                out.append(replace(estimate, pose=None, evidence=Evidence.LOST))
        return tuple(out)
