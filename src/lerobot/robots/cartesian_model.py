"""A robot's optional Cartesian capability, in its ordinary action units.

Building this model never connects hardware. FK and IK belong to the robot;
editors and policies consume this interface without selecting a robot family.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class CartesianModel:
    kinematics: object
    joint_names: tuple[str, ...]
    units_per_degree: tuple[float, ...]
    gripper: str
    lower_deg: tuple[float, ...]
    upper_deg: tuple[float, ...]
    model_id: str = ""

    def description(self):
        return {
            key: list(getattr(self, key))
            for key in ("joint_names", "units_per_degree", "lower_deg", "upper_deg")
        } | {"gripper": self.gripper, "model_id": self.model_id}

    def degrees(self, state):
        q = np.asarray([state[name] for name in self.joint_names], dtype=float)
        if not np.isfinite(q).all():
            raise ValueError("Robot state contains nonfinite joints")
        return q / np.asarray(self.units_per_degree)

    def forward(self, state):
        return self.kinematics.forward_kinematics(self.degrees(state))

    def inverse(self, state, target):
        q = np.asarray(self.kinematics.inverse_kinematics(self.degrees(state), target))
        if not np.isfinite(q).all() or np.any(q < self.lower_deg) or np.any(q > self.upper_deg):
            raise ValueError("Target exceeds the calibrated joint limits")
        reached = self.kinematics.forward_kinematics(q)
        distance = np.linalg.norm(reached[:3, 3] - target[:3, 3])
        angle = Rotation.from_matrix(reached[:3, :3].T @ target[:3, :3]).magnitude()
        # A failed IK solve is not a reachable target. These are execution
        # tolerances (3 mm / 3 degrees), independent of perception acceptance.
        if distance > 0.003 or angle > np.deg2rad(3):
            raise ValueError(f"IK did not reach target ({distance * 1000:.1f} mm, {np.rad2deg(angle):.1f}°)")
        return dict(zip(self.joint_names, q * self.units_per_degree, strict=True))
