# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Far-field validation of the hand-eye rotation — R is global, so test it globally.

The proof calibrated and validated R with +-6..10-unit moves around ONE pose
(the hover). That leaves two things untested that only distance can test:
whether the FK chain (alignment, units) predicts correctly far from the
calibration pose, and which units interpretation is right — the two modes'
R matrices differ by just 6.8 deg locally, indistinguishable at small
amplitudes, while big elbow/pan excursions separate their predictions.

Protocol: load the persisted R and board normal (outputs/handeye_result.json
— NO refitting), sweep to substantially different poses, and at each compare
the ABSOLUTE predicted board normal R @ R_wrist(q) @ n_w against the
depth-plane measurement, plus a local wrist swing at that pose. Excursions
run as staged sub-sweeps with a designation between legs so the continuity
gate tracks legitimate travel instead of refusing it as a jump.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from showservo_handeye_proof import MOTOR_ORDER, BoardEye, enc_vector, sweep_to  # noqa: E402
from showservo_m1_policy import check_depth, joint_positions, make_robot  # noqa: E402


def ang(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(u @ v, -1.0, 1.0))))


def main() -> None:
    from lerobot.robots.so107_description.cartesian_ik import make_so107_arm_kinematics
    from lerobot.robots.so107_description.joint_alignment import LEFT_ARM_ALIGNMENT

    result = json.loads(pathlib.Path("outputs/handeye_result.json").read_text())
    modes = {
        mode: (np.array(d["r_cam_from_base"]), np.array(d["n_w"])) for mode, d in result["modes"].items()
    }

    robot = make_robot("m1_left")
    try:
        hover = {
            "shoulder_pan": 15.1,
            "shoulder_lift": -14.0,
            "elbow_flex": 56.0,
            "forearm_roll": 29.4,
            "wrist_flex": 3.9,
        }
        sweep_to(robot, hover)
        obs = robot.get_observation()
        check_depth(obs)
        eye = BoardEye(obs)
        home = joint_positions(obs)
        print("home:", {m: round(home[m], 2) for m in MOTOR_ORDER}, flush=True)

        kin = make_so107_arm_kinematics(LEFT_ARM_ALIGNMENT)
        cal = robot.calibration
        deg_per_norm = {
            m: (cal[m].range_max - cal[m].range_min) / 200.0 * (360.0 / 4096.0) for m in MOTOR_ORDER
        }

        def fk_rot(enc: np.ndarray, mode: str) -> np.ndarray:
            q = (
                enc.copy()
                if mode == "norm-as-deg"
                else np.array([enc[i] * deg_per_norm[m] for i, m in enumerate(MOTOR_ORDER)])
            )
            return kin.forward_kinematics(q)[:3, :3]

        def staged_sweep(targets: dict[str, float], legs: int = 2) -> None:
            cur = joint_positions(robot.get_observation())
            for i in range(1, legs + 1):
                frac = i / legs
                wp = {j: cur[j] + frac * (targets[j] - cur[j]) for j in targets}
                sweep_to(robot, wp)
                eye.measure(robot.get_observation())  # chain the continuity gate along the travel

        def report(tag: str) -> dict | None:
            o = robot.get_observation()
            snap = eye.measure(o)
            e = enc_vector(o)
            if snap is None:
                print(f"{tag}: REFUSED — designation did not survive here", flush=True)
                return None
            for mode, (r_cam, n_w) in modes.items():
                n_pred = r_cam @ (fk_rot(e, mode) @ n_w)
                err = ang(n_pred, snap["normal"])
                print(f"{tag} [{mode}]: ABSOLUTE normal err {err:5.1f} deg", flush=True)
            return {"e": e, "n": snap["normal"]}

        poses = [
            ("hover-baseline", {}),
            (
                "extend",
                {"elbow_flex": home["elbow_flex"] - 15.0, "shoulder_lift": home["shoulder_lift"] + 5.0},
            ),
            ("fold-up", {"elbow_flex": home["elbow_flex"] + 18.0}),
            (
                "pan-away",
                {"shoulder_pan": home["shoulder_pan"] - 12.0, "elbow_flex": home["elbow_flex"] - 8.0},
            ),
        ]
        for tag, deltas in poses:
            if deltas:
                staged_sweep(deltas)
            before = report(tag)
            if before is not None:
                start = joint_positions(robot.get_observation())
                sweep_to(robot, {"wrist_flex": start["wrist_flex"] + 6.0})
                after_obs = robot.get_observation()
                snap1 = eye.measure(after_obs)
                e1 = enc_vector(after_obs)
                sweep_to(robot, {"wrist_flex": start["wrist_flex"]})
                if snap1 is not None:
                    swing_meas = ang(before["n"], snap1["normal"])
                    for mode, (r_cam, n_w) in modes.items():
                        rw0 = fk_rot(before["e"], mode)
                        rw1 = fk_rot(e1, mode)
                        swing_pred = ang(rw0 @ n_w, rw1 @ n_w)
                        err_after = ang(r_cam @ (rw1 @ n_w), snap1["normal"])
                        print(
                            f"{tag} [{mode}]: local wrist swing meas {swing_meas:4.2f} vs model {swing_pred:4.2f} "
                            f"deg, after-normal err {err_after:5.1f} deg",
                            flush=True,
                        )
            if deltas:
                staged_sweep({j: home[j] for j in deltas})
    finally:
        robot.disconnect()
        print("disconnected (torque holds)", flush=True)


if __name__ == "__main__":
    main()
