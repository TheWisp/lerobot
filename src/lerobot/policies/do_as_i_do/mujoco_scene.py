# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MuJoCo pick scene for the SO-107: the URDF arm (via MjSpec) + a table + a
graspable cylinder + position actuators, all in the robot base frame so the
retargeted (base-frame) trajectory drops straight in.

Lets us go past the kinematic virtual pick to Level 1 (open-loop physics: does
the grasp actually hold?) and Level 2 (closed-loop feedback in sim).
"""

from __future__ import annotations

import mujoco
import numpy as np

from lerobot.robots.so107_description import get_urdf_path
from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES, READY_POSE_URDF_DEG

JOINTS = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")  # == MOTOR_NAMES order

# Gripper command is RANGE_0_100 (100=open). Route it through joint_alignment to
# a URDF S7 angle EXACTLY as the real robot's JointMappedKinematics does
# (urdf = sign*cmd + offset) — driving S7 with raw hand-picked angles instead
# reintroduces the calibrated direction/offset bug. Right arm gripper sign=-1, so
# open(100) -> rad(-100), closed(12) -> rad(-12) (opens toward NEGATIVE S7).
GRIPPER_OPEN_CMD = 100.0
GRIPPER_CLOSED_CMD = 12.0


def gripper_cmd_to_s7_rad(g_range_0_100: float, alignment: dict) -> float:
    """RANGE_0_100 gripper command -> URDF S7 angle (rad) via joint_alignment."""
    al = alignment["gripper"]
    return float(np.radians(al.sign * float(g_range_0_100) + al.offset_deg))


def motor_to_urdf_rad(q_motor_deg: np.ndarray, alignment: dict) -> np.ndarray:
    """Convert motor-space degrees (MOTOR_NAMES order) to URDF-space radians (S1..S7)."""
    sign = np.array([alignment[m].sign for m in MOTOR_NAMES], dtype=float)
    off = np.array([alignment[m].offset_deg for m in MOTOR_NAMES], dtype=float)
    return np.radians(sign * np.asarray(q_motor_deg, dtype=float) + off)


def build_pick_scene(
    *,
    obj_pos,
    obj_quat=(1.0, 0.0, 0.0, 0.0),
    obj_radius: float = 0.014,
    obj_halflen: float = 0.040,
    mesh_path: str | None = None,
    table_z: float = 0.075,
    table_center=(0.18, -0.22),
    table_half: float = 0.13,
):
    """Build the SO-107 + table + cylinder scene. Returns ``(model, home_rad)``.

    ``home_rad`` is the READY pose in URDF radians (S1..S7) for the arm to start at.
    The object is a cylinder at ``obj_pos`` (base-frame meters) with orientation
    ``obj_quat`` (w,x,y,z) — e.g. a horizontal dowel resting on the table. The
    cylinder's local +z is its long axis (the MuJoCo cylinder convention).
    """
    spec = mujoco.MjSpec.from_file(str(get_urdf_path()))
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    wb = spec.worldbody

    light = wb.add_light()
    light.pos = [table_center[0], table_center[1], 1.2]
    light.dir = [0, 0, -1]

    ground = wb.add_geom()
    ground.name = "ground"
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.size = [1, 1, 0.1]
    ground.rgba = [0.2, 0.2, 0.2, 1]

    table = wb.add_geom()
    table.name = "table"
    table.type = mujoco.mjtGeom.mjGEOM_BOX
    table.size = [table_half, table_half, table_z / 2]  # in front of the robot, clear of the base
    table.pos = [table_center[0], table_center[1], table_z / 2]
    table.rgba = [0.4, 0.28, 0.15, 1]
    table.friction = [1.0, 0.01, 0.001]
    table.solref = [0.005, 1.0]  # stiff contact (timeconst >= 2*dt) so the object rests on top, not sunk in
    table.solimp = [0.97, 0.99, 0.001, 0.5, 2.0]

    if mesh_path is not None:
        mesh = spec.add_mesh()
        mesh.name = "object_mesh"
        mesh.file = str(mesh_path)

    body = wb.add_body()
    body.name = "object"
    body.pos = [float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])]
    body.quat = [float(q) for q in obj_quat]
    body.add_freejoint()
    og = body.add_geom()
    og.name = "object_geom"
    if mesh_path is not None:
        # MuJoCo collides a mesh via its convex hull, so a SAM-3D reconstruction's
        # hollow interior is ignored for physics (effectively the solid dowel); the
        # mesh's local +z must be the object's long axis to match obj_quat.
        og.type = mujoco.mjtGeom.mjGEOM_MESH
        og.meshname = "object_mesh"
    else:
        og.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        og.size = [obj_radius, obj_halflen, 0]
    og.rgba = [0.85, 0.6, 0.3, 1]
    og.friction = [4.0, 0.2, 0.002]  # grippy so a clamped dowel doesn't slip out
    og.density = 400.0
    og.solref = [0.005, 1.0]
    og.solimp = [0.97, 0.99, 0.001, 0.5, 2.0]

    # PD position servos: kp stiff enough to drive the arm down against gravity,
    # plus a velocity term (biasprm[2] = -kv) for damping so it doesn't sag/oscillate.
    for j in JOINTS:
        kp = 120.0 if j == "S7" else 120.0  # strong gripper clamp so it holds the object on lift
        kv = 6.0 if j == "S7" else 12.0
        a = spec.add_actuator()
        a.name = j + "_p"
        a.target = j
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.gainprm[0] = kp
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.biasprm[1] = -kp
        a.biasprm[2] = -kv

    model = spec.compile()
    # Disable self-collision for the upper-arm links: the CAD URDF's link meshes
    # collide with each other in folded reach poses (a deep-penetration contact
    # whose constraint force fights the joint servos and jams the shoulder pan).
    # MuJoCo's URDF import doesn't exclude self-collision. Keep the gripper
    # (L6/L7) colliding so it can still grasp the object.
    for link in ("L1_1", "L2_1", "L3_1", "L4_1", "L5_1"):
        bid = model.body(link).id
        for gi in range(model.ngeom):
            if model.geom_bodyid[gi] == bid:
                model.geom_contype[gi] = 0
                model.geom_conaffinity[gi] = 0
    return model, np.radians(READY_POSE_URDF_DEG)


def set_arm(model, data, urdf_rad, gripper_rad: float | None = None) -> None:
    """Set arm joints S1..S6 (and optionally S7) from URDF radians."""
    for i, j in enumerate(JOINTS[:6]):
        data.qpos[model.joint(j).qposadr[0]] = urdf_rad[i]
    if gripper_rad is not None:
        data.qpos[model.joint("S7").qposadr[0]] = gripper_rad


def ctrl_for(model, urdf_rad, gripper_rad: float) -> np.ndarray:
    """Actuator targets (S1..S7) for a desired arm pose + gripper."""
    c = np.zeros(model.nu)
    for i in range(6):
        c[model.actuator(JOINTS[i] + "_p").id] = urdf_rad[i]
    c[model.actuator("S7_p").id] = gripper_rad
    return c
