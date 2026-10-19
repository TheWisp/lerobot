"""Model-free registration between a demo object frame and a live tracker."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import inverse, transform


class DegenerateRegistrationError(RuntimeError):
    pass


def unproject_pixels(
    depth: np.ndarray, intrinsics: np.ndarray, xy: np.ndarray, depth_m_per_unit: float
) -> np.ndarray:
    """Return metric camera-frame points at aligned RGB/depth pixels."""
    pixels = np.asarray(xy, dtype=float)
    camera = np.asarray(intrinsics, dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or camera.shape != (3, 3):
        raise ValueError("pixels must be Nx2 and intrinsics 3x3")
    if not np.isfinite(camera).all() or camera[0, 0] <= 0 or camera[1, 1] <= 0:
        raise ValueError("invalid focal length")
    if depth_m_per_unit <= 0 or not np.isfinite(depth_m_per_unit):
        raise ValueError("invalid depth scale")
    indices = np.rint(pixels).astype(int)
    if (
        np.any(indices[:, 0] < 0)
        or np.any(indices[:, 0] >= depth.shape[1])
        or np.any(indices[:, 1] < 0)
        or np.any(indices[:, 1] >= depth.shape[0])
    ):
        raise ValueError("registration pixel lies outside aligned depth")
    z = depth[indices[:, 1], indices[:, 0]].astype(float) * depth_m_per_unit
    if np.any(~np.isfinite(z)) or np.any(z <= 0):
        raise ValueError("registration point has no valid depth")
    x = (pixels[:, 0] - camera[0, 2]) * z / camera[0, 0]
    y = (pixels[:, 1] - camera[1, 2]) * z / camera[1, 1]
    return np.column_stack((x, y, z))


def transform_points(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    pose = transform(pose)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite Nx3")
    return points @ pose[:3, :3].T + pose[:3, 3]


def rigid_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Kabsch fit for corresponding 3D points, with reflection rejection."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("registration needs at least three corresponding 3D points")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("registration points must be finite")
    src_center, dst_center = source.mean(axis=0), target.mean(axis=0)
    centered = source - src_center
    singular = np.linalg.svd(centered, compute_uv=False)
    # The second singular direction must have measurable extent; otherwise
    # rotation about a line is unconstrained by the selected correspondences.
    if singular[1] <= singular[0] * 1e-4:
        raise DegenerateRegistrationError("correspondences are nearly collinear")
    u, _, vt = np.linalg.svd(centered.T @ (target - dst_center))
    correction = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
    rotation = vt.T @ correction @ u.T
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = dst_center - rotation @ src_center
    return transform(pose)


@dataclass(frozen=True)
class RegistrationResult:
    base_from_demo_object: np.ndarray
    inlier_indices: tuple[int, ...]
    median_error_m: float
    max_error_m: float


def register_object(
    demo_object_points: np.ndarray,
    live_base_points: np.ndarray,
    *,
    threshold_m: float,
    min_inliers: int,
    trials: int,
    seed: int = 0,
) -> RegistrationResult:
    """Robustly fit a live pose from matched surface points across episodes.

    Source points may come from multiple demo snippets, all expressed in the
    same object frame. Live points are unprojected RGB-D points expressed in
    the robot base. ``threshold_m`` is a task/calibration tolerance supplied
    by the caller, not an object-name-dependent magic constant.
    """
    source = np.asarray(demo_object_points, dtype=float)
    target = np.asarray(live_base_points, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target correspondences must be matching Nx3 arrays")
    if len(source) < min_inliers or min_inliers < 3 or threshold_m <= 0 or trials <= 0:
        raise ValueError("invalid robust registration settings")
    generator = np.random.default_rng(seed)
    best_indices: np.ndarray | None = None
    best_median = float("inf")
    for _ in range(trials):
        indices = generator.choice(len(source), size=3, replace=False)
        try:
            candidate = rigid_fit(source[indices], target[indices])
        except DegenerateRegistrationError:
            continue
        residual = np.linalg.norm(transform_points(candidate, source) - target, axis=1)
        inliers = np.flatnonzero(residual <= threshold_m)
        median = float(np.median(residual[inliers])) if len(inliers) else float("inf")
        if len(inliers) > (0 if best_indices is None else len(best_indices)) or (
            best_indices is not None and len(inliers) == len(best_indices) and median < best_median
        ):
            best_indices, best_median = inliers, median
    if best_indices is None or len(best_indices) < min_inliers:
        raise DegenerateRegistrationError("too few geometrically consistent correspondences")
    refined = rigid_fit(source[best_indices], target[best_indices])
    all_residual = np.linalg.norm(transform_points(refined, source) - target, axis=1)
    final_indices = np.flatnonzero(all_residual <= threshold_m)
    if len(final_indices) < min_inliers:
        raise DegenerateRegistrationError("refit left too few consistent correspondences")
    refined = rigid_fit(source[final_indices], target[final_indices])
    residual = np.linalg.norm(
        transform_points(refined, source[final_indices]) - target[final_indices], axis=1
    )
    if float(np.max(residual)) > threshold_m:
        raise DegenerateRegistrationError("refit exceeds registration tolerance")
    return RegistrationResult(
        refined, tuple(int(i) for i in final_indices), float(np.median(residual)), float(np.max(residual))
    )


@dataclass(frozen=True)
class TrackerFrameBinding:
    """Fixed transform from a tracker's object frame into the demo frame."""

    tracker_from_demo_object: np.ndarray
    reference_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tracker_from_demo_object", transform(self.tracker_from_demo_object))
        if not self.reference_version:
            raise ValueError("tracker binding requires an object reference version")

    @classmethod
    def at_initialization(
        cls, base_from_tracker_object: np.ndarray, registration: RegistrationResult, reference_version: str
    ) -> TrackerFrameBinding:
        return cls(inverse(base_from_tracker_object) @ registration.base_from_demo_object, reference_version)

    def demo_object_pose(self, base_from_tracker_object: np.ndarray) -> np.ndarray:
        return transform(base_from_tracker_object) @ transform(self.tracker_from_demo_object)
