"""Offline ablation: local 6D uncertainty of a partial surface measurement.

Not a calibrated probability and not a Kalman filter. Keep the existing geometric
candidate, but combine its local point-to-plane information with the existing
image/group point evidence. The sensor-noise assumptions are explicit. In
particular, a correlated depth error must not become hundreds of independent votes.
No recording identity, time interval, grasp annotation or stationary rule enters
the estimator. The normal and correspondence uncertainties remain unmodelled.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

from lerobot.policies.flash.experimental_surface_session import PointEvidence
from lerobot.policies.flash.surface_motion import displacement_jacobian, pose_with_delta
from lerobot.policies.flash.surface_observation import move, spatial_sample


@dataclass(frozen=True)
class CovarianceConfig:
    # Experimental standard deviations, NOT the acceptance tolerance and NOT
    # calibrated RealSense specifications. Sweep them; do not tune to an event.
    independent_error_m: float = 0.001
    correlated_error_m: float = 0.002
    correlation_length_m: float = 0.01
    max_measurements: int = 256
    # Optional local Gaussian consistency check. A clearly contradicted source
    # should be rejected, not averaged with the direct observation. This is an
    # experiment, not a calibrated false-rejection guarantee.
    compatibility_probability: float | None = None

    def __post_init__(self):
        if not np.isfinite(
            [self.independent_error_m, self.correlated_error_m, self.correlation_length_m]
        ).all():
            raise ValueError("noise settings must be finite")
        if self.independent_error_m <= 0 or self.correlated_error_m < 0 or self.correlation_length_m <= 0:
            raise ValueError("invalid noise settings")
        if type(self.max_measurements) is not int or self.max_measurements < 6:
            raise ValueError("at least six measurements are required")
        if self.compatibility_probability is not None and not 0 < self.compatibility_probability < 1:
            raise ValueError("compatibility probability must be between zero and one")


def plane_information(points, normals, pivot, config):
    """Generalized least-squares information, twist order [rx,ry,rz,tx,ty,tz].

    A smooth isotropic 3D error field plus independent error is projected onto
    each measured normal. R is positive definite; rotating normals or changing
    their arbitrary signs preserves the information. This is conditional on
    valid point associations and normals, not a global confidence measure.
    """
    jac = np.einsum("ni,nij->nj", normals, displacement_jacobian(points, pivot))
    covariance = np.eye(len(points)) * config.independent_error_m**2
    if config.correlated_error_m:
        kernel = np.exp(-cdist(points, points, "sqeuclidean") / (2 * config.correlation_length_m**2))
        covariance += config.correlated_error_m**2 * kernel * (normals @ normals.T)
    information = jac.T @ cho_solve(cho_factor(covariance), jac)
    return (information + information.T) / 2


def fuse_local(geometry_pose, prior_pose, reference, observed, normals, evidence, config, distance_m):
    """One local Gaussian update; evidence supplied by the unchanged group path.

    Predicted points from one source are correlated by construction, so their
    information is normalized per source. Their existing scale is provisionally
    interpreted as a standard deviation; this is another uncalibrated assumption.
    Without supportive evidence retain the original candidate (do not invent a
    stationary prior). Degenerate directions are reported via information rank;
    a pseudoinverse's zero is never reported as zero uncertainty.
    """
    if not evidence:
        return geometry_pose, {"applied": False, "reason": "no_supportive_evidence"}
    pivot = move(reference, prior_pose).mean(0)
    indices = spatial_sample(observed, config.max_measurements)
    observed, normals = observed[indices], normals[indices]
    predicted = move(reference, geometry_pose)
    distances, closest = cKDTree(predicted).query(observed)
    keep = distances <= distance_m
    points, normals = predicted[closest[keep]], normals[keep]
    if len(points) < 6:
        return geometry_pose, {"applied": False, "reason": "insufficient_associations"}
    geometry_info = plane_information(points, normals, pivot, config)
    rotation = Rotation.from_matrix(geometry_pose[:3, :3] @ prior_pose[:3, :3].T).as_rotvec()
    delta = np.r_[rotation, move(reference.mean(0, keepdims=True), geometry_pose)[0] - pivot]
    prior_info, prior_rhs = np.zeros((6, 6)), np.zeros(6)
    for source in evidence:
        if source.weight <= 0:
            continue
        p = move(source.reference, prior_pose)
        jac = displacement_jacobian(p, pivot)
        weight = source.weight / (len(p) * source.scale_m**2)
        prior_info += np.einsum("nij,nik->jk", jac, jac) * weight
        prior_rhs += np.einsum("nij,ni->j", jac, source.current - p) * weight
    # Condition by object size, keeping physical units in the returned covariance.
    jac = displacement_jacobian(move(reference, prior_pose), pivot)
    metric = np.einsum("nij,nik->jk", jac, jac) / len(reference)
    values, vectors = np.linalg.eigh(metric)
    if values[0] <= np.finfo(float).eps * values[-1]:
        return geometry_pose, {"applied": False, "reason": "degenerate_reference"}
    whitening = (vectors / np.sqrt(values)) @ vectors.T
    total = whitening.T @ (geometry_info + prior_info) @ whitening
    values, vectors = np.linalg.eigh(total)
    if values[0] <= np.finfo(float).eps * values[-1]:
        return geometry_pose, {"applied": False, "reason": "unconstrained_posterior"}
    inverse = (vectors / values) @ vectors.T
    covariance = whitening @ inverse @ whitening.T
    correction = covariance @ (geometry_info @ delta + prior_rhs)
    result = pose_with_delta(prior_pose, pivot, correction)
    strengths, directions = np.linalg.eigh(whitening.T @ geometry_info @ whitening)
    consistency = {"tested": False}
    if config.compatibility_probability is not None:
        prior = whitening.T @ prior_info @ whitening
        pv = np.linalg.eigvalsh(prior)
        constrained = strengths > np.finfo(float).eps * max(strengths[-1], 1)
        if pv[0] > np.finfo(float).eps * pv[-1] and constrained.any():
            prior_covariance = np.linalg.inv(prior)
            prior_delta = prior_covariance @ whitening.T @ prior_rhs
            axes = directions[:, constrained]
            innovation = axes.T @ (np.linalg.solve(whitening, delta) - prior_delta)
            innovation_covariance = np.diag(1 / strengths[constrained]) + axes.T @ prior_covariance @ axes
            squared = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
            threshold = float(chi2.ppf(config.compatibility_probability, int(constrained.sum())))
            consistency = {
                "tested": True,
                "squared_mahalanobis": squared,
                "threshold": threshold,
                "support_rejected": squared > threshold,
            }
            if squared > threshold:
                result = geometry_pose
    return result, {
        "applied": True,
        "measurement_count": len(points),
        "geometry_information_eigenvalues": strengths.tolist(),
        "posterior_covariance": covariance.tolist(),
        "posterior_pivot_m": pivot.tolist(),
        "twist_order": ["rx", "ry", "rz", "tx", "ty", "tz"],
        "geometry_delta": delta.tolist(),
        "fused_delta": correction.tolist(),
        "calibrated": False,
        "consistency": consistency,
        # The covariance above belongs to the fused candidate, not an accepted
        # geometry-only result after support rejection.
        "posterior_used": not consistency.get("support_rejected", False),
    }


def correct_with_covariance(fitter, proposal, previous, baseline_correct, config):
    """Opt-in experiment hook. Existing map updates and acceptance still run."""
    geometry_pose, detail = baseline_correct(fitter, proposal, previous)
    evidence = list(fitter.point_evidence)
    if len(fitter.image_reference) >= 3 and fitter.image_weight > 0:
        evidence.append(
            PointEvidence(
                fitter.image_reference,
                fitter.image_current,
                "image_matches",
                "measured",
                fitter.image_scale_m,
                fitter.image_weight,
            )
        )
    pose, uncertainty = fuse_local(
        geometry_pose,
        proposal,
        fitter.reference,
        fitter.observation["obs"]["points"],
        fitter.observation["obs"]["normals"],
        evidence,
        config,
        fitter.config.max_surface_distance_m,
    )
    return pose, {**detail, "covariance_experiment": uncertainty}
