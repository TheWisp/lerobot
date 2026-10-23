"""Causal rigid tracking of one visible object from depth-supported point IDs.

The local frame is chosen at the first supported observation. Later surface
points join that frame only after a pose has been measured; a tracking gap does
not silently create a new, supposedly equivalent object frame. This module
does not predict through occlusion or establish identity across episodes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .registration import DegenerateRegistrationError, register_object, rigid_fit, transform_points


@dataclass(frozen=True)
class VisibleRigidObservation:
    pose: np.ndarray | None
    supported_points: int
    matched_points: int
    rigid_inliers: int
    median_inlier_error_m: float | None
    new_points: int


@dataclass(frozen=True)
class VisibleSurfaceEvidence:
    surface_supported: int
    in_front_of_depth: int
    projected_points: int


@dataclass(frozen=True)
class VisibleSurfaceDecision:
    pose: np.ndarray
    accepted: bool
    point_inliers: int
    baseline: VisibleSurfaceEvidence
    proposal: VisibleSurfaceEvidence | None


@dataclass(frozen=True)
class VisibleRGBDFrame:
    surface_m: np.ndarray
    mask: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray


def visible_surface_cloud(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    *,
    edge_pixels: int,
    max_points: int,
) -> np.ndarray:
    """Back-project depth-backed mask interior without assigning point IDs."""
    depth = np.asarray(depth_m, dtype=float)
    inside = np.asarray(mask, dtype=bool)
    k = np.asarray(intrinsics, dtype=float)
    if (
        depth.shape != inside.shape
        or depth.ndim != 2
        or k.shape != (3, 3)
        or edge_pixels < 0
        or max_points < 3
    ):
        raise ValueError("invalid RGB-D visible surface")
    if edge_pixels:
        inside = cv2.erode(inside.astype(np.uint8), np.ones((2 * edge_pixels + 1,) * 2, np.uint8)) > 0
    y, x = np.nonzero(inside & np.isfinite(depth) & (depth > 0))
    if len(x) < 3:
        raise DegenerateRegistrationError("visible mask has too little depth-backed interior")
    if len(x) > max_points:
        indices = np.linspace(0, len(x) - 1, max_points, dtype=int)
        x, y = x[indices], y[indices]
    z = depth[y, x]
    return np.column_stack(((x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z))


class VisibleRigidTracker:
    """Keep a fixed material-point frame while supported tracks enter and leave.

    The caller must check tracker visibility against an instance mask and aligned
    depth before passing ``supported``. A lost frame retains the map, so the
    same old point IDs can relocalize it later. New IDs are never attached while
    the pose is unknown.
    """

    def __init__(self, point_count: int, *, tolerance_m: float, min_inliers: int, trials: int):
        if point_count < min_inliers or min_inliers < 3 or tolerance_m <= 0 or trials < 1:
            raise ValueError("invalid visible rigid tracker settings")
        self.canonical = np.full((point_count, 3), np.nan)
        self.tolerance_m = tolerance_m
        self.min_inliers = min_inliers
        self.trials = trials
        self.frame = 0

    def update(self, points: np.ndarray, supported: np.ndarray) -> VisibleRigidObservation:
        points = np.asarray(points, dtype=float)
        supported = np.asarray(supported, dtype=bool)
        if points.shape != self.canonical.shape or supported.shape != (len(points),):
            raise ValueError("visible tracks and support have inconsistent shapes")
        if np.any(supported & (~np.isfinite(points).all(axis=1) | (points[:, 2] <= 0))):
            raise ValueError("supported tracks must have finite positive depth")
        self.frame += 1
        count = int(supported.sum())
        known = np.isfinite(self.canonical).all(axis=1)
        if not known.any():
            if count < self.min_inliers:
                return VisibleRigidObservation(None, count, 0, 0, None, 0)
            self.canonical[supported] = points[supported]
            return VisibleRigidObservation(np.eye(4), count, count, count, 0.0, count)
        ids = np.flatnonzero(supported & known)
        if len(ids) < self.min_inliers:
            return VisibleRigidObservation(None, count, len(ids), 0, None, 0)
        try:
            fitted = register_object(
                self.canonical[ids],
                points[ids],
                threshold_m=self.tolerance_m,
                min_inliers=self.min_inliers,
                trials=self.trials,
                seed=self.frame,
            )
        except (DegenerateRegistrationError, ValueError):
            return VisibleRigidObservation(None, count, len(ids), 0, None, 0)
        new = supported & ~known
        pose = fitted.base_from_demo_object
        self.canonical[new] = (points[new] - pose[:3, 3]) @ pose[:3, :3]
        return VisibleRigidObservation(
            pose,
            count,
            len(ids),
            len(fitted.inlier_indices),
            fitted.median_error_m,
            int(new.sum()),
        )

    def update_with_surface(
        self,
        points: np.ndarray,
        supported: np.ndarray,
        *,
        source_surface: np.ndarray,
        target_surface: np.ndarray,
        target_mask: np.ndarray,
        target_depth: np.ndarray,
        intrinsics: np.ndarray,
        depth_m_per_unit: float,
        surface_tolerance_m: float,
        iterations: int,
        trim_fraction: float,
    ) -> tuple[VisibleRigidObservation, VisibleSurfaceDecision | None]:
        """Update a visible group, then test RGB-D correction before adding new IDs."""
        known_before = np.isfinite(self.canonical).all(axis=1)
        result = self.update(points, supported)
        if result.pose is None or not known_before.any():
            return result, None
        matched = np.asarray(supported, dtype=bool) & known_before
        decision = choose_visible_surface_pose(
            source_surface,
            target_surface,
            result.pose,
            self.canonical[matched],
            np.asarray(points)[matched],
            target_mask,
            target_depth,
            intrinsics,
            depth_m_per_unit=depth_m_per_unit,
            track_tolerance_m=self.tolerance_m,
            surface_tolerance_m=surface_tolerance_m,
            min_inliers=self.min_inliers,
            iterations=iterations,
            trim_fraction=trim_fraction,
        )
        if not decision.accepted:
            return result, decision
        new = np.asarray(supported, dtype=bool) & ~known_before
        pose = decision.pose
        self.canonical[new] = (np.asarray(points)[new] - pose[:3, 3]) @ pose[:3, :3]
        residual = np.linalg.norm(
            transform_points(pose, self.canonical[matched]) - np.asarray(points)[matched], axis=1
        )
        accepted = residual <= self.tolerance_m
        corrected = VisibleRigidObservation(
            pose,
            result.supported_points,
            result.matched_points,
            int(accepted.sum()),
            float(np.median(residual[accepted])),
            result.new_points,
        )
        return corrected, decision


def fit_visible_trace(
    points: np.ndarray,
    supported: np.ndarray,
    *,
    tolerance_m: float,
    min_inliers: int,
    trials: int,
    reference_surface: np.ndarray | None = None,
    rgbd_frames: Sequence[VisibleRGBDFrame | None] | None = None,
    depth_m_per_unit: float | None = None,
    surface_tolerance_m: float | None = None,
    icp_iterations: int | None = None,
    trim_fraction: float | None = None,
) -> dict[str, np.ndarray]:
    """Fit one causal local frame from an aligned sequence of visible tracks.

    A pose is written only when depth-supported material IDs determine it. The
    returned ``lost`` flag must be respected by downstream code; the identity
    matrix stored in a lost slot is only an array placeholder.
    """
    xyz = np.asarray(points, dtype=float)
    visible = np.asarray(supported, dtype=bool)
    if xyz.ndim != 3 or xyz.shape[2] != 3 or visible.shape != xyz.shape[:2]:
        raise ValueError("visible trace must be frames x points x 3 with matching support")
    use_surface = reference_surface is not None
    if use_surface != (rgbd_frames is not None) or (rgbd_frames is not None and len(rgbd_frames) != len(xyz)):
        raise ValueError("surface reference and aligned RGB-D frames must be supplied together")
    if use_surface and (
        depth_m_per_unit is None
        or surface_tolerance_m is None
        or icp_iterations is None
        or trim_fraction is None
    ):
        raise ValueError("visible-surface settings are incomplete")
    tracker = VisibleRigidTracker(
        xyz.shape[1], tolerance_m=tolerance_m, min_inliers=min_inliers, trials=trials
    )
    poses, lost, inliers, medians, new_points, conflicts = [], [], [], [], [], []
    surface_refined = []
    previous_pose = None
    previous_points = None
    previous_support = None
    for index, (frame_points, frame_support) in enumerate(zip(xyz, visible, strict=True)):
        rgbd = None if rgbd_frames is None else rgbd_frames[index]
        if use_surface and rgbd is not None:
            observation, decision = tracker.update_with_surface(
                frame_points,
                frame_support,
                source_surface=reference_surface,
                target_surface=rgbd.surface_m,
                target_mask=rgbd.mask,
                target_depth=rgbd.depth,
                intrinsics=rgbd.intrinsics,
                depth_m_per_unit=depth_m_per_unit,
                surface_tolerance_m=surface_tolerance_m,
                iterations=icp_iterations,
                trim_fraction=trim_fraction,
            )
        else:
            observation, decision = tracker.update(frame_points, frame_support), None
        surface_refined.append(decision is not None and decision.accepted)
        pose = observation.pose
        poses.append(np.eye(4) if pose is None else pose)
        lost.append(pose is None)
        inliers.append(observation.rigid_inliers)
        medians.append(
            float("nan") if observation.median_inlier_error_m is None else observation.median_inlier_error_m
        )
        new_points.append(observation.new_points)
        conflicting = False
        if pose is not None and previous_pose is not None:
            shared = frame_support & previous_support
            ids = np.flatnonzero(shared)
            if len(ids) >= min_inliers:
                expected = transform_points(pose, tracker.canonical[ids]) - transform_points(
                    previous_pose, tracker.canonical[ids]
                )
                measured = frame_points[ids] - previous_points[ids]
                conflicting = bool(np.median(np.linalg.norm(expected - measured, axis=1)) > 2 * tolerance_m)
        conflicts.append(conflicting)
        previous_pose, previous_points, previous_support = pose, frame_points, frame_support
    result = {
        "poses": np.stack(poses),
        "lost": np.asarray(lost, dtype=bool),
        "inliers": np.asarray(inliers, dtype=int),
        "median_fit_error_m": np.asarray(medians, dtype=float),
        "new_points": np.asarray(new_points, dtype=int),
        "temporal_conflicts": np.asarray(conflicts, dtype=bool),
    }
    if use_surface:
        result["surface_refined"] = np.asarray(surface_refined, dtype=bool)
    return result


def refine_visible_surface(
    source_surface: np.ndarray,
    target_surface: np.ndarray,
    initial_pose: np.ndarray,
    *,
    max_correspondence_m: float,
    iterations: int,
    trim_fraction: float,
) -> np.ndarray:
    """Local ICP proposal, initialized by material point correspondences.

    The caller must still validate this proposal against the tracked points,
    target mask and signed depth. ICP alone cannot resolve an unobservable
    rotation or distinguish a different but similar surface.
    """
    source = np.asarray(source_surface, dtype=float)
    target = np.asarray(target_surface, dtype=float)
    pose = np.asarray(initial_pose, dtype=float).copy()
    if (
        source.ndim != 2
        or target.ndim != 2
        or source.shape[1:] != (3,)
        or target.shape[1:] != (3,)
        or min(len(source), len(target)) < 3
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
        or pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or max_correspondence_m <= 0
        or iterations < 1
        or not 0 < trim_fraction <= 1
    ):
        raise ValueError("invalid visible-surface refinement inputs")
    tree = cKDTree(target)
    for _ in range(iterations):
        moved = transform_points(pose, source)
        distance, nearest = tree.query(moved)
        keep = distance <= min(max_correspondence_m, float(np.quantile(distance, trim_fraction)))
        if keep.sum() < 3:
            raise DegenerateRegistrationError("visible surfaces do not overlap at the proposed pose")
        updated = rigid_fit(source[keep], target[nearest[keep]])
        change = np.linalg.norm(transform_points(updated, source[keep]) - moved[keep], axis=1)
        pose = updated
        if float(np.median(change)) < 1e-6:
            break
    return pose


def _surface_evidence(
    source_surface: np.ndarray,
    pose: np.ndarray,
    target_mask: np.ndarray,
    target_depth: np.ndarray,
    intrinsics: np.ndarray,
    *,
    depth_m_per_unit: float,
    tolerance_m: float,
) -> VisibleSurfaceEvidence:
    moved = transform_points(pose, source_surface)
    z = moved[:, 2]
    visible = z > 0
    x = np.full(len(z), -1, dtype=int)
    y = np.full(len(z), -1, dtype=int)
    x[visible] = np.rint(intrinsics[0, 0] * moved[visible, 0] / z[visible] + intrinsics[0, 2]).astype(int)
    y[visible] = np.rint(intrinsics[1, 1] * moved[visible, 1] / z[visible] + intrinsics[1, 2]).astype(int)
    inside = visible & (x >= 0) & (x < target_mask.shape[1]) & (y >= 0) & (y < target_mask.shape[0])
    x = np.clip(x, 0, target_mask.shape[1] - 1)
    y = np.clip(y, 0, target_mask.shape[0] - 1)
    measured_z = target_depth[y, x] * depth_m_per_unit
    valid = inside & np.isfinite(measured_z) & (measured_z > 0)
    delta = z - measured_z
    supported = valid & target_mask[y, x] & (np.abs(delta) <= tolerance_m)
    front = valid & (delta < -tolerance_m)
    return VisibleSurfaceEvidence(int(supported.sum()), int(front.sum()), len(source_surface))


def choose_visible_surface_pose(
    source_surface: np.ndarray,
    target_surface: np.ndarray,
    initial_pose: np.ndarray,
    canonical_tracks: np.ndarray,
    observed_tracks: np.ndarray,
    target_mask: np.ndarray,
    target_depth: np.ndarray,
    intrinsics: np.ndarray,
    *,
    depth_m_per_unit: float,
    track_tolerance_m: float,
    surface_tolerance_m: float,
    min_inliers: int,
    iterations: int,
    trim_fraction: float,
) -> VisibleSurfaceDecision:
    """Accept an ICP proposal only when both point IDs and RGB-D support it.

    This is a Pareto check rather than a weighted score: the proposal must
    retain a measurable point consensus, increase mask/depth support, and
    reduce impossible foreground projections. A change beyond two point-fit
    tolerances is rejected because two individually acceptable fits cannot
    justify it. The surface can still be symmetric; this does not certify a
    fully observable 6D orientation.
    """
    source = np.asarray(canonical_tracks, dtype=float)
    observed = np.asarray(observed_tracks, dtype=float)
    mask = np.asarray(target_mask, dtype=bool)
    depth = np.asarray(target_depth)
    camera = np.asarray(intrinsics, dtype=float)
    if (
        source.shape != observed.shape
        or source.ndim != 2
        or source.shape[1] != 3
        or len(source) < min_inliers
        or mask.shape != depth.shape
        or camera.shape != (3, 3)
        or depth_m_per_unit <= 0
        or track_tolerance_m <= 0
        or surface_tolerance_m <= 0
        or min_inliers < 3
    ):
        raise ValueError("invalid visible pose selection inputs")
    initial = np.asarray(initial_pose, dtype=float)
    baseline_error = np.linalg.norm(transform_points(initial, source) - observed, axis=1)
    baseline_inlier = baseline_error <= track_tolerance_m
    baseline = _surface_evidence(
        source_surface,
        initial,
        mask,
        depth,
        camera,
        depth_m_per_unit=depth_m_per_unit,
        tolerance_m=surface_tolerance_m,
    )
    try:
        proposal = refine_visible_surface(
            source_surface,
            target_surface,
            initial,
            max_correspondence_m=surface_tolerance_m,
            iterations=iterations,
            trim_fraction=trim_fraction,
        )
    except DegenerateRegistrationError:
        return VisibleSurfaceDecision(initial, False, int(baseline_inlier.sum()), baseline, None)
    proposed_error = np.linalg.norm(transform_points(proposal, source) - observed, axis=1)
    proposed_inlier = proposed_error <= track_tolerance_m
    candidate = _surface_evidence(
        source_surface,
        proposal,
        mask,
        depth,
        camera,
        depth_m_per_unit=depth_m_per_unit,
        tolerance_m=surface_tolerance_m,
    )
    shift = np.linalg.norm(
        transform_points(proposal, source[baseline_inlier])
        - transform_points(initial, source[baseline_inlier]),
        axis=1,
    )
    accepted = bool(
        proposed_inlier.sum() >= min_inliers
        and baseline_inlier.sum() >= min_inliers
        and np.median(proposed_error[baseline_inlier])
        <= np.median(baseline_error[baseline_inlier]) + track_tolerance_m
        and np.quantile(shift, 0.9) <= 2 * track_tolerance_m
        and candidate.surface_supported > baseline.surface_supported
        and candidate.in_front_of_depth < baseline.in_front_of_depth
    )
    return VisibleSurfaceDecision(
        proposal if accepted else initial,
        accepted,
        int(proposed_inlier.sum() if accepted else baseline_inlier.sum()),
        baseline,
        candidate,
    )
