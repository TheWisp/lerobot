"""Experimental constant-velocity error-state filter; no runtime default change.

State: rotation error, fixed-anchor position error, angular velocity, linear
velocity, all in current camera axes. Velocity describes object motion relative
to the fixed base, expressed in camera axes; camera motion is transported, not
learned as object motion. The observation gate/correction is the existing one.
Only measurements that survive the session's geometry checks may update state.

Measured group motion is a relative measurement between consecutive states.
Its correlation with the previous pose is retained in the update; it is not
added on top of a velocity displacement or counted as an independent absolute
pose. Group covariance retains the existing conditional noise assumptions.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.surface_kalman import PoseKalmanFilter, left_jacobian, skew
from lerobot.policies.flash.surface_motion import pose_with_delta


@dataclass(frozen=True)
class VelocityConfig:
    # Assumptions, not calibrated confidence. One setting for every scene.
    initial_linear_std: float = 0.1  # metres/second
    initial_angular_std_deg: float = 90.0  # degrees/second
    linear_acceleration_std: float = 0.5  # metres/second**1.5, white acceleration
    angular_acceleration_std_deg: float = 180.0  # degrees/second**1.5

    def __post_init__(self):
        for value in vars(self).values():
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Velocity uncertainty must be finite and positive")


class VelocityPoseFilter(PoseKalmanFilter):
    def __init__(self, *args, velocity_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.velocity_config = velocity_config or VelocityConfig()
        self.velocity = np.zeros(6)
        self.joint_covariance = np.zeros((12, 12))
        self.joint_covariance[:6, :6] = self.covariance
        c = self.velocity_config
        self.joint_covariance[6:, 6:] = np.diag(
            [np.deg2rad(c.initial_angular_std_deg) ** 2] * 3 + [c.initial_linear_std**2] * 3
        )

    def snapshot(self):
        return {
            **super().snapshot(),
            "velocity": self.velocity.copy(),
            "joint_covariance": self.joint_covariance.copy(),
        }

    def restore(self, state):
        super().restore(state)
        self.velocity = state["velocity"].copy()
        self.joint_covariance = state["joint_covariance"].copy()

    def discretize(self, dt):
        # Exact integration of the local linear error dynamics and continuous
        # white acceleration. Van Loan's block exponential also transports Q
        # during rotation, avoiding a frame-count-dependent approximation.
        a = np.zeros((12, 12))
        a[:3, :3] = skew(self.velocity[:3])
        a[:6, 6:] = np.eye(6)
        c, v = self.config, self.velocity_config
        density = np.diag(
            [np.deg2rad(c.rotation_process_std_deg_sqrt_s) ** 2] * 3
            + [c.position_process_std_m_sqrt_s**2] * 3
            + [np.deg2rad(v.angular_acceleration_std_deg) ** 2] * 3
            + [v.linear_acceleration_std**2] * 3
        )
        block = np.zeros((24, 24))
        block[:12, :12], block[:12, 12:], block[12:, 12:] = a, density, -a.T
        integrated = expm(block * dt)
        transition = integrated[:12, :12]
        noise = integrated[:12, 12:] @ transition.T
        return transition, (noise + noise.T) / 2

    def _apply_joint_correction(self, correction, covariance):
        self.pose = pose_with_delta(self.pose, self.center(), correction[:6])
        self.velocity += correction[6:]
        reset = np.eye(12)
        reset[:3, :3] = left_jacobian(correction[:3])
        self.joint_covariance = reset @ covariance @ reset.T
        self.joint_covariance = (self.joint_covariance + self.joint_covariance.T) / 2
        self.covariance = self.joint_covariance[:6, :6].copy()

    def predict(self, time_s, camera_change, base_from_camera, motion=None):
        # Reuse exactly the current group's rigid fit and covariance estimator.
        # This probe also validates timestamps before mutating any filter state.
        group = PoseKalmanFilter(self.pose, self.center_object, self.time_s, self.config)
        group.covariance = self.covariance.copy()
        group.predict(time_s, camera_change, base_from_camera, motion)
        dt = time_s - self.time_s
        self.joint_covariance[:6, :6] = self.covariance
        axes = np.kron(np.eye(4), camera_change[:3, :3])
        previous = axes @ self.joint_covariance @ axes.T
        self.velocity = np.kron(np.eye(2), camera_change[:3, :3]) @ self.velocity
        self.pose = camera_change @ self.pose
        previous_pose = self.pose.copy()
        transition, noise = self.discretize(dt)
        self.pose = pose_with_delta(self.pose, self.center(), self.velocity * dt)
        predicted = transition @ previous @ transition.T + noise
        self.joint_covariance = predicted
        detail = dict(group.last_prediction)
        if detail["motion_source"] is not None:
            # Observe pose_now = group_delta * pose_previous. Include the
            # previous/current cross covariance so pose uncertainty is not
            # counted twice as independent evidence.
            g = np.kron(np.eye(2), group.pose[:3, :3] @ previous_pose[:3, :3].T)
            cross = transition @ previous
            measurement_noise = np.asarray(detail["motion_covariance"])
            joint_residual = predicted[:, :6] - cross[:, :6] @ g.T
            residual_covariance = (
                predicted[:6, :6]
                + g @ previous[:6, :6] @ g.T
                - cross[:6, :6] @ g.T
                - g @ cross[:6, :6].T
                + measurement_noise
            )
            residual_covariance = (residual_covariance + residual_covariance.T) / 2
            gain = np.linalg.solve(residual_covariance, joint_residual.T).T
            innovation = np.r_[
                Rotation.from_matrix(group.pose[:3, :3] @ self.pose[:3, :3].T).as_rotvec(),
                group.center() - self.center(),
            ]
            self._apply_joint_correction(gain @ innovation, predicted - gain @ residual_covariance @ gain.T)
            detail["group_relative_innovation"] = innovation.tolist()
        else:
            self.covariance = self.joint_covariance[:6, :6].copy()
        self.time_s = time_s
        self.last_prediction = {
            **detail,
            "prediction_model": "constant_velocity",
            "process_covariance": noise[:6, :6].tolist(),
            "prior_covariance": self.covariance.tolist(),
            "prior_pose": self.pose.tolist(),
            "velocity": self.velocity.tolist(),
            "joint_prior_covariance": self.joint_covariance.tolist(),
        }

    def correct(self, observation_pose, information, observation_pivot, *, contradicted=False):
        self.joint_covariance[:6, :6] = self.covariance
        prior = self.joint_covariance.copy()
        # Same geometry-conditioned information and innovation gate as before;
        # a failed previous prediction never automatically resets this variant.
        detail = super().correct(observation_pose, information, observation_pivot, contradicted=False)
        if detail["accepted"]:
            correction = np.asarray(detail["correction"])
            reset = np.eye(6)
            reset[:3, :3] = left_jacobian(correction[:3])
            inverse = np.linalg.inv(reset)
            linear_posterior = inverse @ self.covariance @ inverse.T
            conditional = np.linalg.solve(prior[:6, :6], prior[:6, 6:]).T
            self.velocity += conditional @ correction
            self.joint_covariance[:6, :6] = self.covariance
            self.joint_covariance[:6, 6:] = reset @ linear_posterior @ conditional.T
            self.joint_covariance[6:, :6] = self.joint_covariance[:6, 6:].T
            self.joint_covariance[6:, 6:] = (
                prior[6:, 6:] + conditional @ (linear_posterior - prior[:6, :6]) @ conditional.T
            )
            self.joint_covariance = (self.joint_covariance + self.joint_covariance.T) / 2
        detail.update(
            filter_reason=detail["reason"],
            prediction_failed_geometry=bool(contradicted),
            prior_rejected_by_depth_mask=bool(contradicted),
            velocity=self.velocity.tolist(),
        )
        return detail
