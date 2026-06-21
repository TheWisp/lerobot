# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Closed-loop do-as-i-do pick controller: observation -> action, one step at a time.

Each call it (re)localizes the object, recomputes the EE target for the current phase
RELATIVE to the live object pose, solves IK seeded by the live joint state, and advances
the phase machine on reaching the target (or after the gripper has settled). Because the
target is recomputed from the live observation every step, errors and object motion are
corrected — that is the closed loop. The robot's IK and real gripper do the rest.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class Phase(IntEnum):
    PREGRASP = 0  # above the object, open
    APPROACH = 1  # at the grasp pose, open
    CLOSE = 2  # at the grasp pose, closing
    LIFT = 3  # above the grasp pose, closed
    DONE = 4


class DoAsIDoController:
    """Maps observations to SO-107 motor targets for a closed-loop pick.

    Preconditions on ``obs`` (the runner adapts the robot's raw observation to this):
      * ``obs["joint_deg"]`` — current motor angles S1..S7 (deg).
      * ``obs`` carries whatever ``localizer.object_pose`` needs (e.g. rgb/depth).

    ``predict`` returns a 7-vector motor target (S1..S6 from IK, S7 = gripper command).
    """

    def __init__(self, skill, kin, localizer, *, reach_tol_m: float = 0.012, close_steps: int = 25):
        self.skill = skill
        self.kin = kin
        self.localizer = localizer
        self.reach_tol = reach_tol_m
        self.close_steps = close_steps
        self.phase = Phase.PREGRASP
        self._close_count = 0

    @property
    def done(self) -> bool:
        return self.phase == Phase.DONE

    def predict(self, obs: dict) -> np.ndarray:
        q_now = np.asarray(obs["joint_deg"], dtype=float)
        grasp = self.skill.grasp_in_base(self.localizer.object_pose(obs))  # live, base frame
        target, gripper = self._phase_target(grasp)
        q = self.kin.inverse_kinematics(q_now, target).copy()
        q[6] = gripper
        self._maybe_advance(q_now, target)
        return q

    # --- phase machine -------------------------------------------------------
    def _phase_target(self, grasp: np.ndarray):
        up = np.array([0.0, 0.0, 1.0])
        T = grasp.copy()
        if self.phase == Phase.PREGRASP:
            T[:3, 3] = grasp[:3, 3] + up * self.skill.pregrasp_h
            return T, self.skill.gripper_open
        if self.phase == Phase.APPROACH:
            return grasp, self.skill.gripper_open
        if self.phase == Phase.CLOSE:
            return grasp, self.skill.gripper_closed
        # LIFT / DONE
        T[:3, 3] = grasp[:3, 3] + up * self.skill.lift_h
        return T, self.skill.gripper_closed

    def _maybe_advance(self, q_now: np.ndarray, target: np.ndarray) -> None:
        if self.phase == Phase.CLOSE:
            self._close_count += 1
            if self._close_count >= self.close_steps:
                self.phase = Phase.LIFT
            return
        ee = self.kin.forward_kinematics(q_now)
        if np.linalg.norm(ee[:3, 3] - target[:3, 3]) < self.reach_tol:
            self.phase = Phase(min(int(self.phase) + 1, int(Phase.DONE)))
