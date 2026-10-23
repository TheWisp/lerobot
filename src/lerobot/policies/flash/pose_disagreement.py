"""Explain disagreement between two metric object-pose proposals.

Depth consistency on a small visible patch is not enough to choose a full
rigid pose: one proposal can fit that patch better while another explains
more of the segmented surface. This diagnostic records both dimensions and
marks non-dominated proposals as ambiguous, leaving the control policy to
obtain more evidence or apply a declared fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .geometry import inverse, transform
from .perception import RGBDFrame
from .registration import transform_points


@dataclass(frozen=True)
class SurfaceSupport:
    projected_points: int
    masked_depth_points: int
    projected_point_mask_fraction: float
    depth_residual_p90_m: float


@dataclass(frozen=True)
class FreeSpaceSupport:
    """One-sided metric contradiction of a projected measured object surface.

    Positive signed gap means the camera measured a farther surface where the
    proposed object surface should have blocked the ray. Negative gaps may be
    ordinary occluders, so this is evidence against a pose, not a pose fit.
    """

    projected_pixels: int
    valid_depth_pixels: int
    median_signed_gap_m: float | None
    contradiction_fraction: float | None


class FreeSpaceVerdict(str, Enum):
    NOT_CONTRADICTED = "not_contradicted"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_DEPTH = "insufficient_depth"


@dataclass(frozen=True)
class FreeSpaceComparison:
    verdict: FreeSpaceVerdict
    support: FreeSpaceSupport
    tolerated_gap_m: float


class PoseComparisonVerdict(str, Enum):
    AGREE = "agree"
    CANDIDATE_FAVORED = "candidate_favored"
    PRIOR_FAVORED = "prior_favored"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class PoseComparison:
    verdict: PoseComparisonVerdict
    object_center_difference_m: float
    rotation_difference_rad: float
    candidate: SurfaceSupport
    prior: SurfaceSupport


def free_space_support(
    frame: RGBDFrame,
    points_object_m: np.ndarray,
    base_from_object: np.ndarray,
    *,
    depth_tolerance_m: float,
    max_local_depth_span_m: float,
) -> FreeSpaceSupport:
    """Z-buffer a measured object cloud against same-frame aligned depth.

    Each first-view model pixel votes once at its nearest projected depth.
    Depth at discontinuities or with missing neighbors is excluded; otherwise
    an occluder in front does not count as a free-space contradiction.
    """
    if (
        not np.isfinite(depth_tolerance_m)
        or depth_tolerance_m <= 0
        or not np.isfinite(max_local_depth_span_m)
        or max_local_depth_span_m <= 0
    ):
        raise ValueError("free-space comparison needs positive metric tolerances")
    points = np.asarray(points_object_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("object model needs finite Nx3 points")
    camera_from_object = inverse(frame.base_from_camera) @ transform(base_from_object)
    camera_points = transform_points(camera_from_object, points)
    camera_points = camera_points[camera_points[:, 2] > 0]
    height, width = frame.depth.shape
    if not len(camera_points):
        return FreeSpaceSupport(0, 0, None, None)
    x = np.rint(camera_points[:, 0] / camera_points[:, 2] * frame.intrinsics[0, 0] + frame.intrinsics[0, 2])
    y = np.rint(camera_points[:, 1] / camera_points[:, 2] * frame.intrinsics[1, 1] + frame.intrinsics[1, 2])
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    flat = y[inside].astype(int) * width + x[inside].astype(int)
    front = np.full(height * width, np.inf, dtype=float)
    np.minimum.at(front, flat, camera_points[inside, 2])
    projected = np.isfinite(front).reshape((height, width))
    measured = frame.metric_depth().astype(np.float32)
    measured = np.where(np.isfinite(measured), measured, 0)
    neighborhood = np.ones((3, 3), dtype=np.uint8)
    lower = cv2.erode(measured, neighborhood)
    upper = cv2.dilate(measured, neighborhood)
    valid = projected & (measured > 0) & (lower > 0) & (upper - lower <= max_local_depth_span_m)
    gaps = measured[valid] - front.reshape((height, width))[valid]
    if not len(gaps):
        return FreeSpaceSupport(int(projected.sum()), 0, None, None)
    return FreeSpaceSupport(
        int(projected.sum()),
        len(gaps),
        float(np.median(gaps)),
        float(np.mean(gaps > depth_tolerance_m)),
    )


def surface_support(
    frame: RGBDFrame, mask: np.ndarray, points_object_m: np.ndarray, base_from_object: np.ndarray
) -> SurfaceSupport:
    """Compare one proposed object's projected model points with aligned depth.

    These are diagnostics, not a full visibility likelihood: model points may
    be self-occluded, and another object may occlude them. The mask and depth
    must come from the same aligned RGB-D frame.
    """
    if mask.shape != frame.depth.shape or mask.dtype != bool:
        raise ValueError("object mask must align with RGB-D")
    points = np.asarray(points_object_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or not len(points):
        raise ValueError("object model needs finite Nx3 points")
    camera_from_object = inverse(frame.base_from_camera) @ transform(base_from_object)
    camera_points = transform_points(camera_from_object, points)
    positive = camera_points[:, 2] > 0
    camera_points = camera_points[positive]
    if not len(camera_points):
        return SurfaceSupport(0, 0, 0.0, float("inf"))
    x = np.rint(camera_points[:, 0] / camera_points[:, 2] * frame.intrinsics[0, 0] + frame.intrinsics[0, 2])
    y = np.rint(camera_points[:, 1] / camera_points[:, 2] * frame.intrinsics[1, 1] + frame.intrinsics[1, 2])
    inside = (x >= 0) & (x < frame.depth.shape[1]) & (y >= 0) & (y < frame.depth.shape[0])
    xy = np.column_stack((x[inside], y[inside])).astype(int)
    camera_points = camera_points[inside]
    if not len(xy):
        return SurfaceSupport(0, 0, 0.0, float("inf"))
    depth = frame.metric_depth()[xy[:, 1], xy[:, 0]]
    supported = mask[xy[:, 1], xy[:, 0]] & np.isfinite(depth) & (depth > 0)
    residual = depth[supported] - camera_points[supported, 2]
    return SurfaceSupport(
        len(xy),
        int(supported.sum()),
        float(supported.mean()),
        float(np.percentile(np.abs(residual), 90)) if len(residual) else float("inf"),
    )


def compare_pose_proposals(
    frame: RGBDFrame,
    mask: np.ndarray,
    points_object_m: np.ndarray,
    candidate_base_from_object: np.ndarray,
    prior_base_from_object: np.ndarray,
    *,
    max_agreement_offset_m: float,
    max_agreement_angle_rad: float,
    min_masked_depth_points: int,
) -> PoseComparison:
    """Pareto-compare surface fit and mask coverage after a task-sized pose change.

    The agreement limits are supplied by the caller's task tolerance. A
    non-dominated conflict is never silently resolved in favor of either
    source; the consumer must log it and decide how to continue safely.
    A caller with rotational symmetry must compare equivalent pose
    representatives before applying the angle limit.
    """
    if max_agreement_offset_m <= 0 or max_agreement_angle_rad <= 0 or min_masked_depth_points < 1:
        raise ValueError("comparison needs positive task tolerance and visible-point count")
    candidate_pose = transform(candidate_base_from_object)
    prior_pose = transform(prior_base_from_object)
    center = np.asarray(points_object_m, dtype=float).mean(axis=0, keepdims=True)
    difference = float(
        np.linalg.norm(transform_points(candidate_pose, center)[0] - transform_points(prior_pose, center)[0])
    )
    rotation_difference = float(
        np.arccos(np.clip((np.trace(candidate_pose[:3, :3].T @ prior_pose[:3, :3]) - 1) / 2, -1, 1))
    )
    candidate = surface_support(frame, mask, points_object_m, candidate_pose)
    prior = surface_support(frame, mask, points_object_m, prior_pose)
    if (
        candidate.masked_depth_points < min_masked_depth_points
        or prior.masked_depth_points < min_masked_depth_points
    ):
        verdict = PoseComparisonVerdict.INSUFFICIENT_EVIDENCE
    elif difference <= max_agreement_offset_m and rotation_difference <= max_agreement_angle_rad:
        verdict = PoseComparisonVerdict.AGREE
    elif (
        candidate.projected_point_mask_fraction >= prior.projected_point_mask_fraction
        and candidate.depth_residual_p90_m <= prior.depth_residual_p90_m
    ):
        verdict = PoseComparisonVerdict.CANDIDATE_FAVORED
    elif (
        prior.projected_point_mask_fraction >= candidate.projected_point_mask_fraction
        and prior.depth_residual_p90_m <= candidate.depth_residual_p90_m
    ):
        verdict = PoseComparisonVerdict.PRIOR_FAVORED
    else:
        verdict = PoseComparisonVerdict.AMBIGUOUS
    return PoseComparison(verdict, difference, rotation_difference, candidate, prior)
