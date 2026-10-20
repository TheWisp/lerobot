"""Conservative metric shape proposal from a reference surface and live RGB-D mask.

This is an identity-preserving fallback: the instance mask supplies the role,
the prepared point cloud supplies its stable object frame, and measured depth
supplies the live surface. Several PCA orientations initialize ICP. A pose is
returned only if the visible surfaces agree within the caller's metric fit
tolerance and all passing object-frame orientations agree. A symmetric or
partly observed shape can therefore abstain instead of inventing its yaw or
which end is which.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np
from scipy.spatial import cKDTree

from .registration import DegenerateRegistrationError, rigid_fit, unproject_pixels


@dataclass(frozen=True)
class ShapeRegistration:
    camera_from_object: np.ndarray
    model_p90_m: float
    live_p90_m: float
    agreeing_seeds: int


def _principal_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centre = points.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov((points - centre).T))
    return centre, axes


def register_visible_shape(
    model_points_object_m: np.ndarray,
    live_depth: np.ndarray,
    live_mask: np.ndarray,
    intrinsics: np.ndarray,
    depth_m_per_unit: float,
    *,
    tolerance_m: float,
    min_points: int,
    max_live_points: int = 1000,
    icp_iterations: int = 20,
) -> ShapeRegistration:
    """Fit one fully supported pose or explain why the RGB-D surface cannot.

    The 90th-percentile *both ways* permits a minority of bad mask/depth pixels
    without letting a tiny visible patch explain an entire reference object.
    ``tolerance_m`` is the same configured metric surface tolerance as matched
    RGB-D registration. Equivalent ICP starts may differ by at most twice that
    tolerance at every stored model point; other passing starts mean the object
    frame is ambiguous, even if their unlabeled surfaces look identical.
    """
    model = np.asarray(model_points_object_m, dtype=float)
    mask = np.asarray(live_mask)
    if (
        model.ndim != 2
        or model.shape[1] != 3
        or not np.isfinite(model).all()
        or mask.shape != live_depth.shape
        or mask.dtype != bool
        or not np.isfinite(tolerance_m)
        or tolerance_m <= 0
        or min_points < 3
        or max_live_points < min_points
        or icp_iterations < 1
    ):
        raise ValueError("invalid metric shape registration inputs")
    valid = mask & np.isfinite(live_depth) & (live_depth > 0)
    y, x = np.nonzero(valid)
    if len(model) < min_points or len(x) < min_points:
        raise DegenerateRegistrationError("too few depth-backed surface points")
    if len(x) > max_live_points:
        keep = np.linspace(0, len(x) - 1, max_live_points, dtype=int)
        x, y = x[keep], y[keep]
    live = unproject_pixels(live_depth, intrinsics, np.column_stack((x, y)), depth_m_per_unit)
    model_centre, model_axes = _principal_frame(model)
    live_centre, live_axes = _principal_frame(live)
    tree = cKDTree(live)
    accepted: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    for order in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            rotation = live_axes[:, order] @ np.diag(signs) @ model_axes.T
            if np.linalg.det(rotation) < 0:
                continue
            translation = live_centre - rotation @ model_centre
            try:
                for _ in range(icp_iterations):
                    moved = model @ rotation.T + translation
                    distances, nearest = tree.query(moved)
                    # The broad gate lets a coarse PCA seed move toward the
                    # object. The final bidirectional check remains strict.
                    gate = max(2 * tolerance_m, float(np.quantile(distances, 0.8)))
                    support = distances <= gate
                    if int(support.sum()) < min_points:
                        raise DegenerateRegistrationError("shape seed lacks overlap")
                    pose = rigid_fit(model[support], live[nearest[support]])
                    rotation, translation = pose[:3, :3], pose[:3, 3]
            except DegenerateRegistrationError:
                continue
            moved = model @ rotation.T + translation
            model_p90 = float(np.quantile(tree.query(moved)[0], 0.9))
            live_p90 = float(np.quantile(cKDTree(moved).query(live)[0], 0.9))
            if max(model_p90, live_p90) <= tolerance_m:
                accepted.append((model_p90, live_p90, rotation.copy(), translation.copy()))
    if not accepted:
        raise DegenerateRegistrationError("reference and live depth surfaces do not agree")
    accepted.sort(key=lambda item: max(item[0], item[1]))
    model_p90, live_p90, rotation, translation = accepted[0]
    best_surface = model @ rotation.T + translation
    for _, _, candidate_rotation, candidate_translation in accepted[1:]:
        candidate_surface = model @ candidate_rotation.T + candidate_translation
        if np.max(np.linalg.norm(best_surface - candidate_surface, axis=1)) > 2 * tolerance_m:
            raise DegenerateRegistrationError("visible shape leaves multiple object-frame poses")
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, translation
    return ShapeRegistration(pose, model_p90, live_p90, len(accepted))
