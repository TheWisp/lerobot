"""A robot's optional Cartesian capability, in its ordinary action units.

Building this model never connects hardware. FK and IK belong to the robot;
editors and policies consume this interface without selecting a robot family.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CartesianModel:
    kinematics: object
    joint_names: tuple[str, ...]
    units_per_degree: tuple[float, ...]
    gripper: str
    lower_deg: tuple[float, ...]
    upper_deg: tuple[float, ...]
    model_id: str = ""

    @classmethod
    def from_description(cls, description: dict, geometry: dict) -> "CartesianModel":
        """Rebuild a recorded model without hardware or today's profile/calibration files."""
        model_id = description["model_id"]
        if model_id not in {"so107:left:measured-tool", "so107:right:measured-tool"}:
            raise ValueError(f"Recorded Cartesian model is not supported: {model_id}")
        from .kinematics import SelectedJointKinematics
        from .so107_description.calibrated_kinematics import make_calibrated_so107_kinematics
        from .so107_description.joint_alignment import MOTOR_NAMES

        names = tuple(name.removesuffix(".pos") for name in description["joint_names"])
        kinematics = make_calibrated_so107_kinematics(
            geometry,
            arm=model_id.split(":")[1],
            joint_limits_deg=dict(
                zip(names, zip(description["lower_deg"], description["upper_deg"], strict=True), strict=True)
            ),
        )
        return cls(
            SelectedJointKinematics(kinematics, MOTOR_NAMES, names, {"gripper": 0.0}),
            **{key: tuple(value) if isinstance(value, list) else value for key, value in description.items()},
        )

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
        if not np.isfinite(q).all():
            raise ValueError("Robot IK produced nonfinite joints")
        # The robot model defines its actual joint range. A Cartesian target
        # may be approximate at a limit; do not turn that into an editor veto.
        q = np.clip(q, self.lower_deg, self.upper_deg)
        return dict(zip(self.joint_names, q * self.units_per_degree, strict=True))
