"""Source-frame pose proposals before scene-level arbitration."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .geometry import transform
from .trajectory import PoseEstimate


@dataclass(frozen=True)
class VisibleSurfaceSupport:
    """Measured support; missing fields mean the producer did not measure them."""

    mask_pixels: int
    depth_points: int | None = None
    motion_inliers: int | None = None
    depth_residual_p90_m: float | None = None
    surface_fit_converged: bool | None = None

    def __post_init__(self) -> None:
        if self.mask_pixels < 0 or any(
            value is not None and value < 0 for value in (self.depth_points, self.motion_inliers)
        ):
            raise ValueError("visible support counts must be nonnegative")
        if self.depth_residual_p90_m is not None and (
            not isfinite(self.depth_residual_p90_m) or self.depth_residual_p90_m < 0
        ):
            raise ValueError("visible depth residual must be finite and nonnegative")


@dataclass(frozen=True)
class PoseProposal:
    """A backend's raw fit and observed support, including fits it abstains from using."""

    frame_id: int
    calibration_version: str
    estimate: PoseEstimate
    raw_pose: np.ndarray | None
    visible_support: VisibleSurfaceSupport
    failure: str | None = None

    def __post_init__(self) -> None:
        if self.frame_id < 0 or not self.calibration_version:
            raise ValueError("pose proposal needs a source frame and calibration")
        if self.raw_pose is not None:
            object.__setattr__(self, "raw_pose", transform(self.raw_pose))

    @property
    def role(self) -> str:
        return self.estimate.role

    @property
    def reference_version(self) -> str:
        return self.estimate.reference_version

    @property
    def source_time_s(self) -> float:
        return self.estimate.source_time_s
