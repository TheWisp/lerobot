"""Error-state pose filter: measured group motion predicts, visible depth corrects.

State is rotation plus the position of one FIXED point in object coordinates.
Covariances use camera-axis rotation (radians), then position (metres), about
that point. Neither map growth nor changes in visible surface move this origin.
The local Gaussian noise model is conditional on correct associations and group
membership. It does not turn optical-flow visibility into a probability.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

from .surface_correspondence import rigid_fit
from .surface_motion import displacement_jacobian, pose_with_delta
from .surface_observation import move, spatial_sample


def skew(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotate_covariance(covariance, rotation):
    axes = np.zeros((6, 6))
    axes[:3, :3] = axes[3:, 3:] = rotation
    return axes @ covariance @ axes.T


def left_jacobian(vector):
    angle = np.linalg.norm(vector)
    cross = skew(vector)
    if angle < np.sqrt(np.finfo(float).eps):
        return np.eye(3) + cross / 2 + cross @ cross / 6
    return (
        np.eye(3)
        + (1 - np.cos(angle)) / angle**2 * cross
        + (angle - np.sin(angle)) / angle**3 * cross @ cross
    )


@dataclass(frozen=True)
class MeasuredGroupMotion:
    """Actual visible point pairs in base coordinates, never predicted hidden points."""

    previous: np.ndarray
    current: np.ndarray
    from_time_s: float
    to_time_s: float
    source: str

    def __post_init__(self):
        a, b = np.asarray(self.previous, dtype=float), np.asarray(self.current, dtype=float)
        if a.ndim != 2 or a.shape[1:] != (3,) or a.shape != b.shape or len(a) < 3:
            raise ValueError("motion needs at least three measured point pairs")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("motion points must be finite")
        if not np.isfinite([self.from_time_s, self.to_time_s]).all() or self.to_time_s <= self.from_time_s:
            raise ValueError("motion requires increasing finite timestamps")
        if not self.source:
            raise ValueError("motion needs a source")
        object.__setattr__(self, "previous", a.copy())
        object.__setattr__(self, "current", b.copy())


class PoseKalmanFilter:
    def __init__(self, pose, center_object, time_s, config):
        self.pose = np.asarray(pose, dtype=float).copy()
        self.center_object = np.asarray(center_object, dtype=float).copy()
        self.time_s, self.config = time_s, config
        self.covariance = self.initial_covariance()
        self.last_prediction = {}

    def initial_covariance(self):
        c = self.config
        return np.diag([np.deg2rad(c.initial_rotation_std_deg) ** 2] * 3 + [c.initial_position_std_m**2] * 3)

    def center(self, pose=None):
        return move(self.center_object[None], self.pose if pose is None else pose)[0]

    def snapshot(self):
        """State that a downstream geometry rejection must roll back together."""
        return {"pose": self.pose.copy(), "covariance": self.covariance.copy()}

    def restore(self, state):
        self.pose = state["pose"].copy()
        self.covariance = state["covariance"].copy()

    def predict(self, time_s, camera_change, base_from_camera, motion=None):
        dt = time_s - self.time_s
        if dt <= 0:
            raise ValueError("Kalman time must strictly increase")
        if motion is not None and (motion.from_time_s != self.time_s or motion.to_time_s != time_s):
            raise ValueError("group motion must connect the filter's consecutive frames")
        self.pose = camera_change @ self.pose
        self.covariance = rotate_covariance(self.covariance, camera_change[:3, :3])
        detail = {"motion_source": None, "motion_covariance": None}
        if motion is not None:
            camera_from_base = np.linalg.inv(base_from_camera)
            a = move(motion.previous, camera_from_base)
            b = move(motion.current, camera_from_base)
            delta = rigid_fit(a, b)
            predicted_pose = delta @ self.pose
            chosen = spatial_sample(b, self.config.max_measurements)
            p = move(a[chosen], delta)
            jac = displacement_jacobian(p, self.center(predicted_pose))
            # Both RGB-D endpoints are noisy. Spatially correlated noise stops
            # nearby points from becoming hundreds of independent votes.
            noise = 2 * self.config.independent_error_m**2 * np.eye(len(p))
            kernel = np.exp(-cdist(p, p, "sqeuclidean") / (2 * self.config.correlation_length_m**2))
            noise += 2 * self.config.correlated_error_m**2 * kernel
            residual = b[chosen] - p
            noise += np.eye(len(p)) * float(np.mean(residual**2))
            solved = cho_solve(cho_factor(noise), jac.reshape(len(p), 18)).reshape(jac.shape)
            information = np.einsum("nij,nik->jk", jac, solved)
            values = np.linalg.eigvalsh(information)
            if values[0] > np.finfo(float).eps * values[-1]:
                motion_covariance = np.linalg.inv(information)
                self.pose = predicted_pose
                self.covariance = rotate_covariance(self.covariance, delta[:3, :3]) + motion_covariance
                detail.update(
                    motion_source=motion.source,
                    motion_covariance=motion_covariance.tolist(),
                    motion_points=len(p),
                    motion_residual_mm=1000 * float(np.sqrt(np.mean(residual**2))),
                )
            else:
                detail["motion_rejected"] = "unconstrained_group_motion"
        c = self.config
        process = (
            np.diag(
                [np.deg2rad(c.rotation_process_std_deg_sqrt_s) ** 2] * 3
                + [c.position_process_std_m_sqrt_s**2] * 3
            )
            * dt
        )
        self.covariance += process
        self.covariance = (self.covariance + self.covariance.T) / 2
        self.time_s = time_s
        self.last_prediction = {
            **detail,
            "process_covariance": process.tolist(),
            "prior_covariance": self.covariance.tolist(),
            "prior_pose": self.pose.tolist(),
        }

    def correct(self, observation_pose, information, observation_pivot, *, contradicted=False):
        """Apply the saved reset policy and retain the actual filter decision."""
        result = self._correct(
            observation_pose,
            information,
            observation_pivot,
            contradicted=contradicted and self.config.reset_on_geometry_conflict,
        )
        result["filter_reason"] = result["reason"]
        result["prediction_failed_geometry"] = bool(contradicted)
        result["prior_rejected_by_depth_mask"] = bool(contradicted)
        return result

    def _correct(self, observation_pose, information, observation_pivot, *, contradicted=False):
        """Information-form EKF update; zero measurement information means no correction.

        A contradicted motion relationship starts a new broad prior at the visible
        estimate. Otherwise the innovation test uses BOTH prior and observation
        covariance in the observation's constrained directions.
        """
        prior_pose, prior_covariance = self.pose.copy(), self.covariance.copy()
        # Convert observed information to the filter's fixed object point.
        shift = self.center(observation_pose) - observation_pivot
        inverse_change = np.eye(6)
        inverse_change[3:, :3] = skew(shift)
        info = inverse_change.T @ information @ inverse_change
        info = (info + info.T) / 2
        rotation = Rotation.from_matrix(observation_pose[:3, :3] @ self.pose[:3, :3].T).as_rotvec()
        innovation = np.r_[rotation, self.center(observation_pose) - self.center()]
        values, vectors = np.linalg.eigh(info)
        keep = values > np.finfo(float).eps * max(values[-1], 1.0)
        axes = vectors[:, keep]
        score, threshold = 0.0, 0.0
        if keep.any():
            delta = axes.T @ innovation
            combined = axes.T @ self.covariance @ axes + np.diag(1 / values[keep])
            score = float(delta @ np.linalg.solve(combined, delta))
            threshold = float(chi2.ppf(self.config.compatibility_probability, int(keep.sum())))
        detail = {
            **self.last_prediction,
            "observation_pose": observation_pose.tolist(),
            "observation_information": info.tolist(),
            "observable_rank": int(keep.sum()),
            "innovation": innovation.tolist(),
            "innovation_squared": score,
            "innovation_limit": threshold,
            "prior_rejected_by_depth_mask": bool(contradicted),
            "calibrated": False,
        }
        if not keep.any():
            return {
                **detail,
                "accepted": False,
                "reason": "no_observation_information",
                "posterior_covariance": self.covariance.tolist(),
            }
        if score > threshold and not contradicted:
            return {
                **detail,
                "accepted": False,
                "reason": "innovation_conflict",
                "posterior_covariance": self.covariance.tolist(),
            }
        if contradicted:
            self.pose = observation_pose.copy()
            self.covariance = self.initial_covariance()
            innovation = np.zeros(6)
        posterior = np.linalg.inv(np.linalg.inv(self.covariance) + info)
        gain = posterior @ info
        correction = gain @ innovation
        self.pose = pose_with_delta(self.pose, self.center(), correction)
        reset = np.eye(6)
        reset[:3, :3] = left_jacobian(correction[:3])
        self.covariance = reset @ posterior @ reset.T
        self.covariance = (self.covariance + self.covariance.T) / 2
        return {
            **detail,
            "accepted": True,
            "reason": "reinitialized_after_contradiction" if contradicted else "corrected",
            "gain": gain.tolist(),
            "correction": correction.tolist(),
            "posterior_covariance": self.covariance.tolist(),
            "previous_pose": prior_pose.tolist(),
            "previous_covariance": prior_covariance.tolist(),
        }
