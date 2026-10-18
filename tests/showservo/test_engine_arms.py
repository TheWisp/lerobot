# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""One engine drives the SO-107 and the OpenArm 2 (act_execution.md R10): the same act in the simulated world on
each arm's own kinematics (Pink), the demo laid out around the arm's working pose and kept at its fingertip's turn."""

from __future__ import annotations

import numpy as np
import pytest

from .enginesim import pick_place_demo, run_act

pink = pytest.mark.skipif(
    not __import__("lerobot.utils.import_utils", fromlist=["_pin_pink_available"])._pin_pink_available,
    reason="pin-pink (optional) not installed",
)


def _act_on(kin, q_ready, gripper, top_deg_s, offset=(-0.05, -0.06, 0.126)):
    """The act, with the demo laid out ``offset`` from where the arm stands ready. A demo recorded on the arm cannot
    turn a joint faster than ``top_deg_s``, what the arm does: one that does (its fingertip's turn held through a
    wrist near its singularity) has the engine follow motion the motor loop cannot, and freeze."""
    tip = np.asarray(kin.forward_kinematics(q_ready), dtype=float)
    work = tip[:3, 3] - np.array(offset)
    demo, kf, bodies = pick_place_demo(work=work, rot=tip[:3, :3], kin=kin, seed_q=q_ready, gripper=gripper)
    speed = np.abs(np.diff(demo.q, axis=0)) / np.diff(demo.t)[:, None]
    speed[:, gripper] = 0.0
    assert speed.max() <= top_deg_s, f"the demo turns a joint at {speed.max():.0f} deg/s"
    return run_act(demo, kf, bodies, kin=kin, gripper=gripper, seed_q=q_ready, work=work)


def _placed(sim):
    dowel = sim.world.bodies["dowel"].now()
    ring = sim.world.bodies["ring"].now()
    return np.linalg.norm(dowel.mean(axis=0)[:2] - ring.mean(axis=0)[:2]) <= 0.004


@pink
def test_the_engine_drives_the_so107():
    from lerobot.robots.so107_description.cartesian_ik import make_so107_arm_kinematics
    from lerobot.robots.so107_description.joint_alignment import LEFT_ARM_ALIGNMENT, MOTOR_NAMES

    kin = make_so107_arm_kinematics(LEFT_ARM_ALIGNMENT)
    ready = {  # hardcode-ok: the SO-107's working pose, as the jog's ready
        "shoulder_pan": 0.0,
        "shoulder_lift": -45.0,
        "elbow_flex": 74.0,
        "forearm_roll": 0.0,
        "wrist_flex": -41.0,
        "wrist_roll": 0.0,
        "gripper": 20.0,
    }
    # The fastest the rig's SO-107 joints turned in the jog's act records (2026-10-05 to 18, 106,545 ticks): 224 deg/s.
    sim = _act_on(
        kin,
        np.array([ready[m] for m in MOTOR_NAMES]),
        MOTOR_NAMES.index("gripper"),
        224.0,
        (-0.08, 0.0, 0.126),
    )
    assert sim.engine.result["ok"], sim.engine.result
    assert _placed(sim), sim.world.events
    assert sim.motor.clamped < 1e-6


@pink
def test_the_engine_drives_the_openarm2():
    from lerobot.robots.openarm_description.cartesian_ik import make_openarm_arm_kinematics

    kin = make_openarm_arm_kinematics("left")
    q_ready = np.array([10.0, 10.0, 0.0, 60.0, 0.0, 10.0, 0.0, 20.0])  # 7 arm joints, then the gripper's slot
    top = np.degrees(5.4454)  # the slowest joint limit openarm_right_viz.urdf declares, 312 deg/s
    sim = _act_on(kin, q_ready, 7, top)
    assert sim.engine.result["ok"], sim.engine.result
    assert _placed(sim), sim.world.events
    assert sim.motor.clamped < 1e-6
