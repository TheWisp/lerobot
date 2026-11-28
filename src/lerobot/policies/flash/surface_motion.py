"""Carry one rigid pose with measured motion; qualify visual corrections by geometry.

This is an experimental RGB-D fusion component, not the default live backend.
It does not infer an attachment or a support relationship. Callers must supply
those relationships and synchronized motion. Missing motion is not a stationary
measurement, and a segmentation mask alone does not observe six pose axes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .registration import transform_points


@dataclass(frozen=True)
class SurfaceMotionConfig:
    # Metres: inherited depth-fit error budget, and the acceptable uncertainty
    # of a correction expressed as RMS displacement of the reference surface.
    depth_error_m: float = 0.005
    correction_uncertainty_m: float = 0.01
    # Four 3 mm surface voxels; enough neighbourhood to estimate a local plane.
    normal_radius_m: float = 0.012
    correspondence_distance_m: float = 0.03

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in vars(self).values()):
            raise ValueError("surface-motion scales must be finite and positive")


@dataclass(frozen=True)
class SurfaceInformation:
    """Local point-to-plane sensitivity conditional on a proposed pose.

    ``basis`` columns are twists [rotation radians, centroid translation metres]
    that each move the whole reference by one metre RMS to first order.
    ``strength`` is the squared measured-normal motion for those unit twists,
    normalized by the *whole* reference, including unseen points. No strength
    is interpreted as a confidence or automatically accepted pose dimension.
    """

    pivot: np.ndarray
    basis: np.ndarray
    strength: np.ndarray
    matched_points: int
    reference_points: int


def measured_surface_information(
    reference: np.ndarray,
    pose: np.ndarray,
    observed: np.ndarray,
    observed_normals: np.ndarray,
    *,
    correspondence_distance_m: float,
) -> SurfaceInformation:
    """Measure constrained rigid motions using the RGB-D observation's normals.

    The image pipeline has already validated, sampled and differentiated the
    masked depth surface. Reusing its normals avoids a second local-plane fit.
    Nearest geometry is only a *conditional* association: a wrong pose can yield
    a plausible spectrum, so a downstream consumer must still check identity,
    fit residuals, free space and later frames before using this for control.
    """
    points = np.asarray(reference, dtype=float)
    observed = np.asarray(observed, dtype=float)
    normals = np.asarray(observed_normals, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or observed.ndim != 2
        or observed.shape[1] != 3
        or normals.shape != observed.shape
        or not np.isfinite(points).all()
        or not np.isfinite(observed).all()
        or not np.isfinite(normals).all()
        or not np.isfinite(correspondence_distance_m)
        or correspondence_distance_m <= 0
    ):
        raise ValueError("surface information needs finite 3D points, normals and distance")
    if np.asarray(pose).shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("surface information needs a finite 4x4 pose")
    predicted = transform_points(pose, points)
    center = predicted.mean(0) if len(points) else np.zeros(3)
    empty = SurfaceInformation(center, np.empty((6, 0)), np.zeros(6), 0, len(points))
    if len(points) < 3 or len(observed) < 3:
        return empty
    norm = np.linalg.norm(normals, axis=1)
    if not np.allclose(norm, 1, rtol=0, atol=1e-5):
        raise ValueError("observed surface normals must be unit length")
    jac = displacement_jacobian(predicted, center)
    metric = np.einsum("nij,nik->jk", jac, jac) / len(points)
    values, vectors = np.linalg.eigh(metric)
    if values[0] <= np.finfo(float).eps * values[-1]:
        return empty
    whitening = (vectors / np.sqrt(values)) @ vectors.T
    distance, indices = cKDTree(observed).query(predicted)
    reverse_distance, reverse_indices = cKDTree(predicted).query(observed)
    covered = np.zeros(len(points), bool)
    covered[reverse_indices[reverse_distance <= correspondence_distance_m]] = True
    matched = (distance <= correspondence_distance_m) & covered
    if not matched.any():
        return empty
    rows = np.einsum("ni,nij->nj", normals[indices[matched]], jac[matched]) @ whitening
    information = rows.T @ rows / len(points)
    strength, directions = np.linalg.eigh(information)
    return SurfaceInformation(
        center,
        whitening @ directions,
        np.maximum(strength, 0),
        int(matched.sum()),
        len(points),
    )


def displacement_jacobian(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    """d(R(p-c)+c+t)/d[rotation, translation], shape N x 3 x 6."""
    relative = points - center
    jac = np.zeros((len(points), 3, 6))
    jac[:, :, 3:] = np.eye(3)
    jac[:, 0, 1], jac[:, 0, 2] = relative[:, 2], -relative[:, 1]
    jac[:, 1, 0], jac[:, 1, 2] = -relative[:, 2], relative[:, 0]
    jac[:, 2, 0], jac[:, 2, 1] = relative[:, 1], -relative[:, 0]
    return jac


def pose_with_delta(pose: np.ndarray, center: np.ndarray, delta: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(delta[:3]).as_matrix()
    result = pose.copy()
    result[:3, :3] = rotation @ pose[:3, :3]
    result[:3, 3] = rotation @ (pose[:3, 3] - center) + center + delta[3:]
    return result
