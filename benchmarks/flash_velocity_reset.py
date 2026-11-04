"""Ablation: change motion prediction while retaining the CURRENT reset rule.

This intentionally retains the known wrong-mug-reset defect, for attribution;
it is not a proposed recovery fix. A reset clears velocity and its correlations.
"""

import numpy as np

from benchmarks.flash_velocity_filter import VelocityPoseFilter
from lerobot.policies.flash.surface_kalman import PoseKalmanFilter


class VelocityResetPoseFilter(VelocityPoseFilter):
    def correct(self, observation_pose, information, observation_pivot, *, contradicted=False):
        if not contradicted:
            return super().correct(observation_pose, information, observation_pivot)
        result = PoseKalmanFilter.correct(
            self, observation_pose, information, observation_pivot, contradicted=True
        )
        if result["accepted"]:
            self.velocity = np.zeros(6)
            self.joint_covariance = np.zeros((12, 12))
            self.joint_covariance[:6, :6] = self.covariance
            c = self.velocity_config
            self.joint_covariance[6:, 6:] = np.diag(
                [np.deg2rad(c.initial_angular_std_deg) ** 2] * 3 + [c.initial_linear_std**2] * 3
            )
        result.update(
            filter_reason=result["reason"], prediction_failed_geometry=True, velocity=self.velocity.tolist()
        )
        return result
