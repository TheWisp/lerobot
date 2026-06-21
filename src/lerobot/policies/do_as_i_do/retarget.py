# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Retarget a grasp point into an SO-107 top-down pick. Dead simple: the gripper
points straight down, ONE spin angle sets the jaw direction, IK each waypoint."""

from __future__ import annotations

import numpy as np

from lerobot.robots.so107_description.joint_alignment import TIP_OFFSET

# Gripper approach axis in the IK-tip frame (the tip is offset from the wrist along
# the gripper's reach — this diagonal must point down for a top-down grasp).
A_TIP = TIP_OFFSET[:3, 3] / np.linalg.norm(TIP_OFFSET[:3, 3])
DOWN = np.array([0.0, 0.0, -1.0])


def _align(a, b):
    """Rotation R with R @ a parallel to b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def _rot(axis, th):
    """Rotation about `axis` by `th` radians."""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(th), np.sin(th)
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )


TOPDOWN = _align(A_TIP, DOWN)  # gripper pointing straight down (spin = 0)


def grasp_pose(point, spin_deg=0.0):
    """4x4 tip target: top-down at `point`, spun `spin_deg` about vertical."""
    T = np.eye(4)
    T[:3, :3] = _rot(DOWN, np.radians(spin_deg)) @ TOPDOWN
    T[:3, 3] = np.asarray(point, float)
    return T


def plan_pick(kin, grasp_point, seed, spin_deg=0.0, pregrasp_h=0.08, lift_h=0.12):
    """pregrasp -> approach -> close -> lift, IK each (warm-started)."""
    g = np.asarray(grasp_point, float)
    up = np.array([0.0, 0.0, 1.0])
    steps = [
        ("pregrasp", g + up * pregrasp_h, 100.0),
        ("approach", g, 100.0),
        ("close", g, 12.0),
        ("lift", g + up * lift_h, 12.0),
    ]
    out, q = [], np.asarray(seed, float)
    for tag, pos, grip in steps:
        T = grasp_pose(pos, spin_deg)
        for _ in range(4):
            q = kin.inverse_kinematics(q, T)
        perr = float(np.linalg.norm(kin.forward_kinematics(q)[:3, 3] - pos)) * 1000
        qq = q.copy()
        qq[6] = grip
        out.append({"tag": tag, "q": qq, "gripper": grip, "pos_err_mm": perr})
    return out
