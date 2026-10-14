#!/usr/bin/env python
"""Offline reachability sweep around recorded Act targets on one calibrated SO-107 arm.

This samples straight fingertip paths from the recorded starting joints, using the
same model, solver, tolerances and configured working joint limits as Act. It
never opens a motor port. Passing means a feasible path stays inside those
limits; it does not prove that a real arm cannot stall under load or contact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.gui.api import jog
from lerobot.gui.api._calib_core import (
    corrected_alignment,
    joint_zero_from_calibration,
    load_calibration,
    tip_offset_from_calibration,
)
from lerobot.robots.so107_description.cartesian_ik import make_so107_arm_kinematics
from lerobot.robots.so107_description.joint_alignment import (
    LEFT_ARM_ALIGNMENT,
    MOTOR_NAMES,
    RIGHT_ARM_ALIGNMENT,
)
from lerobot.showservo import motion as core


def model(profile: dict, calibration: dict, arm: str):
    alignment = LEFT_ARM_ALIGNMENT if arm == "left" else RIGHT_ARM_ALIGNMENT
    zeros = joint_zero_from_calibration(calibration)
    if zeros:
        alignment = corrected_alignment(alignment, zeros)
    kin = make_so107_arm_kinematics(alignment, tip_offset=tip_offset_from_calibration(calibration))
    # The SO-107 uses STS3215 motors, whose encoder has 4096 positions per turn.
    bus = SimpleNamespace(
        calibration={m: SimpleNamespace(**value) for m, value in calibration.items()},
        motors={m: SimpleNamespace(model="sts3215") for m in MOTOR_NAMES},
        model_resolution_table={"sts3215": 4096},
    )
    ranges = jog._profile_joint_ranges(profile.get("fields", {}), arm, *jog._servo_ranges_of(bus))
    jog._limit_to_servos(kin, alignment, *ranges)
    return kin, ranges


def sample_targets(targets: list[dict], offset_mm: float, turn_deg: float):
    poses = [np.asarray(t["pose"], dtype=float) for t in targets if t.get("pose") is not None]
    if not poses:
        return
    # Every observed target, plus a local action-space grid around the first,
    # middle and last. The grid has six signed translations and two signed turns.
    for i, pose in enumerate(poses):
        yield f"observed:{i}", pose
    for anchor, pose in (("first", poses[0]), ("middle", poses[len(poses) // 2]), ("last", poses[-1])):
        for axis in range(3):
            for sign in (-1, 1):
                changed = pose.copy()
                changed[axis, 3] += sign * offset_mm / 1000
                yield f"{anchor}:axis{axis}:{sign}", changed
        for sign in (-1, 1):
            changed = pose.copy()
            changed[:3, :3] = (
                Rotation.from_euler("z", sign * turn_deg, degrees=True).as_matrix() @ pose[:3, :3]
            )
            yield f"{anchor}:yaw:{sign}", changed


def check(kin, ranges, q0, goal, steps: int):
    gi = MOTOR_NAMES.index("gripper")
    start = kin.forward_kinematics(q0)
    distance, angle = core.pose_residual(start, goal)
    # Match the jog's default straight-line sampling; a coarse grid falsely
    # labels smooth paths as joint jumps.
    steps = max(
        steps,
        int(np.ceil(max(distance / jog.MAX_LINEAR_M_S, np.radians(angle) / jog.MAX_ANGULAR_RAD_S) * jog.HZ)),
    )
    poses = np.stack([core.interp_rigid(start, goal, i / steps) for i in range(steps + 1)])
    grips = np.full(steps + 1, q0[gi])
    hints = np.zeros((steps + 1, len(q0)))
    sol = core.solve_plan_joints(kin, poses, grips, hints, q0, gi, *ranges)
    arm = np.arange(len(q0)) != gi
    within = bool(
        np.all(sol["q"][:, arm] >= ranges[0][arm] - 1e-6)
        and np.all(sol["q"][:, arm] <= ranges[1][arm] + 1e-6)
    )
    feasible = bool(
        np.all(sol["residual_m"] <= core.ACT_REACH_TOL_M)
        and np.all(sol["residual_deg"] <= core.ACT_REACH_TOL_DEG)
        and np.all(sol["step_deg"] <= core.ACT_MAX_JOINT_STEP_DEG)
    )
    return {
        "feasible": feasible,
        "within_working_limits": within,
        "worst_residual_mm": round(float(sol["residual_m"].max() * 1000), 2),
        "worst_step_deg": round(float(sol["step_deg"].max()), 2),
        "max_wrist_roll_deg": round(float(sol["q"][:, MOTOR_NAMES.index("wrist_roll")].max()), 2),
        "steps": steps,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--arm", choices=("left", "right"), default="left")
    p.add_argument("--offset-mm", type=float, default=5.0)
    p.add_argument("--turn-deg", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("runs", type=Path, nargs="+")
    args = p.parse_args()
    if args.steps < 2 or args.offset_mm < 0 or args.turn_deg < 0:
        p.error("steps >= 2 and nonnegative offsets are required")
    profile, calibration = json.loads(args.profile.read_text()), load_calibration(args.calibration)
    kin, ranges = model(profile, calibration, args.arm)
    out = {
        "arm": args.arm,
        "working_limits_deg": {
            m: [float(ranges[0][i]), float(ranges[1][i])] for i, m in enumerate(MOTOR_NAMES) if m != "gripper"
        },
        "runs": [],
    }
    for run in args.runs:
        act = json.loads((run / "act.json").read_text())
        with np.load(run / "arm.npz") as arm:
            q0 = np.asarray(arm["q_obs"][0], dtype=float)
        cases = []
        for name, target in sample_targets(act["targets"], args.offset_mm, args.turn_deg):
            cases.append({"name": name, **check(kin, ranges, q0, target, args.steps)})
        out["runs"].append({"run": str(run), "cases": cases})
    out["summary"] = {
        "cases": sum(len(r["cases"]) for r in out["runs"]),
        "feasible": sum(c["feasible"] for r in out["runs"] for c in r["cases"]),
        "working_limit_violations": sum(
            not c["within_working_limits"] for r in out["runs"] for c in r["cases"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
