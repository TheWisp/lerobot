"""Source-frame pose proposals before scene-level arbitration."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .geometry import transform
from .trajectory import PoseEstimate


@dataclass(frozen=True)
class VisibleSurfaceSupport:
    """Measured support; missing fields mean the producer did not measure them.

    The principal axis belongs to the currently visible patch, not necessarily
    the complete object. Its sign is arbitrary; its spectrum describes shape,
    not calibrated orientation confidence or a recovered contact point. The
    spatial split tests stability within one image; its halves can still share
    the same segmentation or depth bias.
    """

    mask_pixels: int
    depth_points: int | None = None
    motion_inliers: int | None = None
    depth_residual_p90_m: float | None = None
    surface_fit_converged: bool | None = None
    reference_depth_pixels: int | None = None
    reference_depth_residual_p90_m: float | None = None
    visible_lateral_spread_m: float | None = None
    principal_surface_axis_base: tuple[float, float, float] | None = None
    principal_surface_spreads_m: tuple[float, float, float] | None = None
    principal_surface_points: int | None = None
    principal_surface_split_axes_base: (
        tuple[tuple[float, float, float], tuple[float, float, float]] | None
    ) = None
    principal_surface_split_gap_rad: float | None = None

    def __post_init__(self) -> None:
        if self.mask_pixels < 0 or any(
            value is not None and value < 0
            for value in (
                self.depth_points,
                self.motion_inliers,
                self.reference_depth_pixels,
                self.principal_surface_points,
            )
        ):
            raise ValueError("visible support counts must be nonnegative")
        for measurement in (
            self.depth_residual_p90_m,
            self.reference_depth_residual_p90_m,
            self.visible_lateral_spread_m,
        ):
            if measurement is not None and (not isfinite(measurement) or measurement < 0):
                raise ValueError("visible surface measurement must be finite and nonnegative")
        axis, spreads = self.principal_surface_axis_base, self.principal_surface_spreads_m
        if (axis is None) != (spreads is None) or (axis is None) != (self.principal_surface_points is None):
            raise ValueError("principal visible surface evidence must be complete")
        if axis is not None:
            axis_array = np.asarray(axis, dtype=float)
            spread_array = np.asarray(spreads, dtype=float)
            if (
                axis_array.shape != (3,)
                or spread_array.shape != (3,)
                or not np.isfinite(axis_array).all()
                or not np.isfinite(spread_array).all()
                or not np.isclose(np.linalg.norm(axis_array), 1.0, rtol=0, atol=1e-6)
                or np.any(spread_array < 0)
                or np.any(np.diff(spread_array) < 0)
                or self.principal_surface_points < 3
            ):
                raise ValueError("invalid principal visible surface evidence")
        split_axes = self.principal_surface_split_axes_base
        split_gap = self.principal_surface_split_gap_rad
        if (split_axes is None) != (split_gap is None):
            raise ValueError("split visible surface evidence must be complete")
        if split_axes is not None:
            split_array = np.asarray(split_axes, dtype=float)
            if (
                axis is None
                or split_array.shape != (2, 3)
                or not np.isfinite(split_array).all()
                or not np.allclose(np.linalg.norm(split_array, axis=1), 1.0, rtol=0, atol=1e-6)
                or not isfinite(split_gap)
                or not 0 <= split_gap <= np.pi / 2
            ):
                raise ValueError("invalid split visible surface evidence")


@dataclass(frozen=True)
class PoseProposal:
    """Raw fit and support, plus an optional independent motion-only prior.

    Neither optional pose is an accepted measurement. A missing motion prior
    means no complete measured frame-motion chain reaches this image.
    """

    frame_id: int
    calibration_version: str
    estimate: PoseEstimate
    raw_pose: np.ndarray | None
    visible_support: VisibleSurfaceSupport
    failure: str | None = None
    motion_prior_pose: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.frame_id < 0 or not self.calibration_version:
            raise ValueError("pose proposal needs a source frame and calibration")
        if self.raw_pose is not None:
            object.__setattr__(self, "raw_pose", transform(self.raw_pose))
        if self.motion_prior_pose is not None:
            object.__setattr__(self, "motion_prior_pose", transform(self.motion_prior_pose))

    @property
    def role(self) -> str:
        return self.estimate.role

    @property
    def reference_version(self) -> str:
        return self.estimate.reference_version

    @property
    def source_time_s(self) -> float:
        return self.estimate.source_time_s
