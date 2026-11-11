"""Conditional uncertainty of a visible surface, separate from pose validity.

A candidate must pass direct surface checks before local covariance compares
it with another fit or supplies observation information to the temporal Kalman
filter. This is an experimental noise model, not a calibrated probability of
object identity or tracking correctness.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import chi2

from .surface_grounding import WholeSurfaceFit
from .surface_motion import displacement_jacobian
from .surface_observation import move, spatial_sample


@dataclass(frozen=True)
class PoseUncertaintyConfig:
    enabled: bool = False
    # selection preserves the earlier experiment; kalman carries pose covariance.
    mode: str = "selection"
    # Initial and random-walk process noise; assumptions, not calibrated accuracy.
    initial_position_std_m: float = 0.005
    initial_rotation_std_deg: float = 10.0
    position_process_std_m_sqrt_s: float = 0.002
    rotation_process_std_deg_sqrt_s: float = 2.0
    # Optional process-noise floor when no measured motion advanced the filter.
    # Chosen before fitting; independent of the fitted candidate.
    unmeasured_rotation_std_deg_sqrt_s: float | None = None
    # Legacy reset is retained for old recordings. False retains the prior
    # for ordinary covariance-weighted correction, even if geometry conflicts.
    reset_on_geometry_conflict: bool = True
    # Standard deviation of independent normal-direction error per measurement.
    independent_error_m: float = 0.001
    # Standard deviation of a spatially correlated 3D error field.
    correlated_error_m: float = 0.002
    # Gaussian spatial correlation length; neighbouring errors are not independent.
    correlation_length_m: float = 0.01
    # Cap the dense measurement covariance matrix (memory grows quadratically).
    max_measurements: int = 256
    # Nominal chi-square coverage under the assumed noise model, not accuracy.
    compatibility_probability: float = 0.95
    # Geometry limit, applied even when uncertainty.enabled is false.
    # Distance uses ShapeConfig.depth_tolerance_m; a small fraction of noisy
    # depth pixels may fail, but coherent empty-space protrusion is never allowed.
    max_visible_outlier_fraction: float = 0.1
    # Prefer compatible predicted point sources; otherwise least pose change.
    prefer_supported_prediction: bool = True

    def __post_init__(self):
        if self.mode not in {"selection", "kalman"}:
            raise ValueError("uncertainty mode must be selection or kalman")
        for name in (
            "initial_position_std_m",
            "initial_rotation_std_deg",
            "position_process_std_m_sqrt_s",
            "rotation_process_std_deg_sqrt_s",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        value = self.unmeasured_rotation_std_deg_sqrt_s
        if value is not None and (type(value) is bool or not np.isfinite(value) or value <= 0):
            raise ValueError("unmeasured_rotation_std_deg_sqrt_s must be finite and positive or null")
        if any(
            type(value) is not bool
            for value in (self.enabled, self.prefer_supported_prediction, self.reset_on_geometry_conflict)
        ):
            raise ValueError("uncertainty switches must be booleans")
        for name in ("independent_error_m", "correlated_error_m", "correlation_length_m"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or value == 0 and name != "correlated_error_m":
                raise ValueError(f"invalid {name}")
        if type(self.max_measurements) is not int or self.max_measurements < 6:
            raise ValueError("max_measurements must be an integer >= 6")
        if not 0 < self.compatibility_probability < 1:
            raise ValueError("compatibility_probability must be between zero and one")
        if not 0 <= self.max_visible_outlier_fraction < 1:
            raise ValueError("max_visible_outlier_fraction must be in [0, 1)")


class SurfacePoseUncertainty:
    """One frame's observation and the pre-update stored surface.

    Covariance is conditional on local nearest-surface associations and measured
    normals. Unknown depth and foreground occlusion are handled by WholeSurfaceFit;
    they are not counted as contradictions. No unseen geometry is invented.
    """

    def __init__(self, geometry: WholeSurfaceFit, best, config: PoseUncertaintyConfig):
        self.geometry, self.best, self.config = geometry, best, config
        self.pivot = move(geometry.reference, best).mean(0)
        obs = geometry.observation["obs"]
        chosen = spatial_sample(obs["points"], config.max_measurements)
        self.observed, self.normals = obs["points"][chosen], obs["normals"][chosen]
        predicted = move(geometry.reference, best)
        distances, indices = cKDTree(predicted).query(self.observed)
        keep = distances <= geometry.config.depth_tolerance_m
        self.observed, self.normals = self.observed[keep], self.normals[keep]
        self.local_points = geometry.reference[indices[keep]]
        points = predicted[indices[keep]]
        self.best_points = points
        self.rank = 0
        self.information = np.zeros((6, 6))
        self.covariance = None
        self.factor = None
        self.strengths = np.zeros(6)
        if not len(points):
            return
        covariance = np.eye(len(points)) * config.independent_error_m**2
        if config.correlated_error_m:
            kernel = np.exp(-cdist(points, points, "sqeuclidean") / (2 * config.correlation_length_m**2))
            covariance += config.correlated_error_m**2 * kernel * (self.normals @ self.normals.T)
        self.factor = cho_factor(covariance)
        jac = np.einsum("ni,nij->nj", self.normals, displacement_jacobian(points, self.pivot))
        self.information = jac.T @ cho_solve(self.factor, jac)
        self.information = (self.information + self.information.T) / 2
        # Rank is evaluated in units of whole-object RMS displacement, rather
        # than comparing metres and radians or depending on the world origin.
        full_jac = displacement_jacobian(predicted, self.pivot)
        metric = np.einsum("nij,nik->jk", full_jac, full_jac) / len(predicted)
        values, vectors = np.linalg.eigh(metric)
        if values[0] <= np.finfo(float).eps * values[-1]:
            return
        whiten = (vectors / np.sqrt(values)) @ vectors.T
        self.strengths = np.linalg.eigvalsh(whiten.T @ self.information @ whiten)
        self.rank = int(np.sum(self.strengths > np.finfo(float).eps * max(self.strengths[-1], 1)))
        if self.rank == 6:
            self.covariance = np.linalg.inv(self.information)

    def direct_check(self, pose):
        return check_surface_contradiction(self.geometry, pose, self.config)

    def check(self, pose, *, direct=None):
        direct = self.direct_check(pose) if direct is None else direct
        if not direct["accepted"]:
            return {**direct, "reason": "visible_surface_contradiction", "covariance_tested": False}
        if self.factor is None or not self.rank:
            return {**direct, "reason": "no_observable_direction", "covariance_tested": False}
        # Exact transformed points, local fixed correspondences. This avoids
        # linearizing a potentially large pose jump into an unchecked average.
        difference = np.einsum("ni,ni->n", move(self.local_points, pose) - self.best_points, self.normals)
        squared = float(difference @ cho_solve(self.factor, difference))
        threshold = float(chi2.ppf(self.config.compatibility_probability, self.rank))
        return {
            **direct,
            "accepted": squared <= threshold,
            "covariance_tested": True,
            "squared_mahalanobis": squared,
            "threshold": threshold,
            "reason": "compatible" if squared <= threshold else "resolved_by_geometry",
        }

    def diagnostics(self):
        return {
            "covariance": self.covariance.tolist() if self.covariance is not None else None,
            "information": self.information.tolist(),
            "camera_from_reference_at_linearization": self.best.tolist(),
            "perturbation_frame": "camera_axes_about_pivot",
            "pivot_m": self.pivot.tolist(),
            "observable_rank": self.rank,
            "information_strengths": self.strengths.tolist(),
            "measurement_count": len(self.local_points),
            "calibrated": False,
            "twist_order": ["rx", "ry", "rz", "tx", "ty", "tz"],
        }


def check_surface_contradiction(geometry, pose, config):
    """Compatibility entry point; validity belongs to the measured geometry."""
    return geometry.check_pose(pose, max_outlier_fraction=config.max_visible_outlier_fraction)
