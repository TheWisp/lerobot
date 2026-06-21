# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Map a reconstructed human-hand demo onto SO-107 end-effector targets.

The ENTIRE hand->robot relationship is one rigid transform, calibrated from a single
correspondence where the human hand and the robot gripper grasp the same point the
same way. The robot's own IK and gripper handle everything else — no gripper geometry,
no jaw-side, no IK seeding to guess.
"""

from __future__ import annotations

import numpy as np


def calibrate(hand_ref: np.ndarray, ee_ref: np.ndarray) -> np.ndarray:
    """Fixed hand->gripper transform ``T`` (4x4) such that ``ee = T @ hand``.

    ``hand_ref`` and ``ee_ref`` are 4x4 SE(3) poses in the SAME base frame, captured
    once while the human hand and the robot gripper hold the SAME point in the SAME
    orientation (e.g. both grasping the cylinder at the same spot).
    """
    return ee_ref @ np.linalg.inv(hand_ref)


def remap(t_cal: np.ndarray, hand_pose: np.ndarray) -> np.ndarray:
    """Human hand pose (4x4, base frame) -> robot EE target (4x4, base frame)."""
    return t_cal @ hand_pose


def grip_cmd(pinch_m: float, closed_m: float = 0.02, open_m: float = 0.08) -> float:
    """Thumb<->index distance (m) -> gripper command in RANGE_0_100 (100 = open)."""
    return float(np.clip((pinch_m - closed_m) / (open_m - closed_m), 0.0, 1.0) * 100.0)
