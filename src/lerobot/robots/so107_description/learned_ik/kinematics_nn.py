"""
NN-based inference wrapper. Drop-in So107 kinematics that uses a trained MLP
for IK instead of placo / ikpy / DLS.

Interface mirrors So107Kinematics:
    fk_from_motors(motor_pos)                 -> 4x4 SE(3) (via the same placo FK)
    ik_to_motors(current_motor_pos, target_T) -> dict[motor_name, deg]

The hybrid mode (refine_with_dls=True) uses the NN's joint prediction as the
initial guess for a couple of DLS Newton steps that pin EE position exactly.
Best-of-both: NN-natural posture + analytical EE accuracy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin
import torch

from .. import get_urdf_path
from ..kinematics import (
    MOTOR_NAMES,
    RIGHT_ARM_MAP,
    JointMap,
    So107Kinematics,
    motor_pos_to_urdf_q,
    urdf_q_to_motor_pos,
)
from .model import IKMLP, IKModelConfig


class So107NNKinematics:
    """NN IK with optional DLS refinement. Drop-in for So107Kinematics."""

    def __init__(
        self,
        model_path: str | Path,
        joint_map: dict[str, JointMap] | None = None,
        device: str = "cpu",
        refine_with_dls: bool = True,
        dls_iters: int = 2,
        dls_damping: float = 0.05,
    ):
        self.joint_map = joint_map or RIGHT_ARM_MAP
        self.device = device

        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        cfg = IKModelConfig(**ckpt["config"])
        self.model = IKMLP(cfg).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        # FK (and optional DLS refinement) via placo-backed kinematics.
        self.fk = So107Kinematics(joint_map=self.joint_map)
        self.refine_with_dls = refine_with_dls
        self.dls_iters = dls_iters
        self.dls_damping = dls_damping

        # Pinocchio for DLS Jacobian (only loaded if refine_with_dls).
        if self.refine_with_dls:
            self._pin_model = pin.buildModelFromUrdf(str(get_urdf_path()))
            self._pin_data = self._pin_model.createData()
            self._pin_frame_id = self._pin_model.getFrameId("L7_1")

    def fk_from_motors(self, motor_pos: dict[str, float]) -> np.ndarray:
        return self.fk.fk_from_motors(motor_pos)

    def _nn_predict(self, current_motors: dict[str, float], target_ee: np.ndarray) -> dict[str, float]:
        """Run a single forward pass: predict next motors from current + ee delta."""
        current_ee = self.fk.fk_from_motors(current_motors)[:3, 3]
        ee_delta = target_ee - current_ee  # 3-vec in meters
        joints_vec = np.array([current_motors[n] for n in MOTOR_NAMES], dtype=np.float32)
        x = np.concatenate([joints_vec, ee_delta.astype(np.float32)])
        with torch.no_grad():
            x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)
            delta_joints = self.model(x_t).squeeze(0).cpu().numpy()
        new_joints = joints_vec + delta_joints
        return {n: float(new_joints[i]) for i, n in enumerate(MOTOR_NAMES)}

    def _dls_refine(self, motor_guess: dict[str, float], target_ee: np.ndarray) -> dict[str, float]:
        """A few DLS Newton steps to pin EE exactly to target. Starts from NN guess."""
        q = motor_pos_to_urdf_q(motor_guess, self.joint_map)
        for _ in range(self.dls_iters):
            pin.computeJointJacobians(self._pin_model, self._pin_data, q)
            pin.updateFramePlacements(self._pin_model, self._pin_data)
            J = pin.getFrameJacobian(
                self._pin_model, self._pin_data, self._pin_frame_id, pin.LOCAL_WORLD_ALIGNED
            )
            Jp = J[:3, :]
            err = target_ee - self._pin_data.oMf[self._pin_frame_id].translation
            if np.linalg.norm(err) < 1e-5:
                break
            A = Jp @ Jp.T + (self.dls_damping**2) * np.eye(3)
            q_delta = Jp.T @ np.linalg.solve(A, err)
            q = q + q_delta
        return urdf_q_to_motor_pos(q, self.joint_map)

    def ik_to_motors(
        self,
        current_motor_pos: dict[str, float],
        target_pose: np.ndarray,
    ) -> tuple[dict[str, float], float]:
        """NN IK (optionally refined). Returns (motor_pos_deg, position_error_mm)."""
        target_ee = target_pose[:3, 3]
        nn_motors = self._nn_predict(current_motor_pos, target_ee)
        final = self._dls_refine(nn_motors, target_ee) if self.refine_with_dls else nn_motors

        # Position error after solve, for monitoring.
        T_achieved = self.fk.fk_from_motors(final)
        err_mm = float(np.linalg.norm(T_achieved[:3, 3] - target_ee)) * 1000
        return final, err_mm
