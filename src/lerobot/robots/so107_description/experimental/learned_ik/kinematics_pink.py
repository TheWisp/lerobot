"""
Pink-based IK (QP, position+orientation) for SO-107.

Drop-in for So107NNKinematics — same `ik_to_motors(current_motor_pos, target_T)`
signature. Uses [pink](https://github.com/stephane-caron/pink) which solves a
weighted quadratic-program over a FrameTask (6-DOF EE tracking) and a
PostureTask (regularize toward current joints to avoid null-space drift in
this 7-DOF arm).

Compared to So107NNKinematics:
    + ~1000x more accurate on orientation (Pink solves position + rotation
      jointly via FrameTask; nn_dls's DLS step is position-only).
    + Built-in joint+velocity limits (read from URDF).
    - No human-posture bias. (You can layer one on by replacing the
      posture target with NN-predicted joints — left as a separate hybrid step.)
"""

from __future__ import annotations

import os

import numpy as np
import pink
import pinocchio as pin
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pinocchio.robot_wrapper import RobotWrapper

from ... import get_urdf_path
from ...kinematics import (
    RIGHT_ARM_MAP,
    JointMap,
    So107Kinematics,
    motor_pos_to_urdf_q,
    urdf_q_to_motor_pos,
)


class So107PinkKinematics:
    """Pink (QP-based) IK with posture regularization. Drop-in for So107NNKinematics."""

    def __init__(
        self,
        joint_map: dict[str, JointMap] | None = None,
        position_cost: float = 1.0,
        orientation_cost: float = 1.0,
        posture_cost: float = 0.05,
        max_iters: int = 10,
        converge_threshold: float = 1e-4,
        dt: float = 1.0 / 30,
        solver: str = "proxqp",
        ee_frame: str = "L7_1",
        nn_posture_model: str | os.PathLike | None = None,
    ):
        self.joint_map = joint_map or RIGHT_ARM_MAP
        self.position_cost = position_cost
        self.orientation_cost = orientation_cost
        self.posture_cost = posture_cost
        self.max_iters = max_iters
        self.converge_threshold = converge_threshold
        self.dt = dt
        self.solver = solver
        self.ee_frame = ee_frame

        # FK shared with placo-backed kinematics, so EE position/rotation reads
        # are consistent across the codebase (verified identical to pinocchio FK).
        self.fk = So107Kinematics(joint_map=self.joint_map)

        urdf_path = str(get_urdf_path())
        mesh_dir = os.path.dirname(urdf_path)
        self.robot = RobotWrapper.BuildFromURDF(urdf_path, [mesh_dir])
        # Seed values can land microscopically past the URDF's joint limits due to floating-point
        # error in our motor->radians conversion; pink's solve_ik strictly validates seeds, so
        # we clamp using a tiny safety inset.
        self.q_lo = self.robot.model.lowerPositionLimit.copy()
        self.q_hi = self.robot.model.upperPositionLimit.copy()

        # Optional NN to provide a "human-like" posture target each tick. When set,
        # PostureTask regularizes toward the NN's prediction instead of the seed.
        # This keeps pink inside the training-data manifold (no self-collisions, no
        # null-space wander into unusual poses).
        self.nn_posture = None
        if nn_posture_model is not None:
            from .kinematics_nn import So107NNKinematics

            # refine_with_dls=False -> bare NN prediction, no DLS step. Adds ~0.1ms.
            self.nn_posture = So107NNKinematics(
                model_path=nn_posture_model,
                joint_map=self.joint_map,
                refine_with_dls=False,
            )

    def fk_from_motors(self, motor_pos: dict[str, float]) -> np.ndarray:
        return self.fk.fk_from_motors(motor_pos)

    def ik_to_motors(
        self,
        current_motor_pos: dict[str, float],
        target_pose: np.ndarray,
    ) -> tuple[dict[str, float], float]:
        """6-DOF IK. Posture target = current_motor_pos (stays close to seed,
        suppresses null-space drift). Returns (motor_pos_deg, position_error_mm)."""
        seed_q = motor_pos_to_urdf_q(current_motor_pos, self.joint_map)
        # Clamp microscopic overshoot of URDF joint limits — pink validates strictly.
        seed_q = np.clip(seed_q, self.q_lo + 1e-4, self.q_hi - 1e-4)
        config = pink.Configuration(self.robot.model, self.robot.data, seed_q)
        target_se3 = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])

        # Posture target: NN prediction if configured, else seed itself.
        if self.nn_posture is not None:
            nn_motors, _ = self.nn_posture.ik_to_motors(current_motor_pos, target_pose)
            posture_q = motor_pos_to_urdf_q(nn_motors, self.joint_map)
            posture_q = np.clip(posture_q, self.q_lo + 1e-4, self.q_hi - 1e-4)
        else:
            posture_q = seed_q

        frame_task = FrameTask(
            self.ee_frame,
            position_cost=self.position_cost,
            orientation_cost=self.orientation_cost,
        )
        frame_task.set_target(target_se3)
        posture_task = PostureTask(cost=self.posture_cost)
        posture_task.set_target(posture_q)
        tasks = [frame_task, posture_task]

        for _ in range(self.max_iters):
            v = solve_ik(config, tasks, self.dt, solver=self.solver)
            if float(np.linalg.norm(v)) * self.dt < self.converge_threshold:
                break
            config.integrate_inplace(v, self.dt)
            # Pink's integrate may overshoot URDF limits by floating-point amounts;
            # next solve_ik call would raise NotWithinConfigurationLimits. Re-clamp.
            # config.q is read-only, so rebuild the Configuration with clamped q.
            clamped_q = np.clip(config.q, self.q_lo + 1e-4, self.q_hi - 1e-4)
            config = pink.Configuration(self.robot.model, self.robot.data, clamped_q)

        new_motors = urdf_q_to_motor_pos(config.q, self.joint_map)
        achieved_T = self.fk.fk_from_motors(new_motors)
        err_mm = float(np.linalg.norm(achieved_T[:3, 3] - target_pose[:3, 3])) * 1000
        return new_motors, err_mm
