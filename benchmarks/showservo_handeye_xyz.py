# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Hand-eye R from an IK translation sweep — the user's protocol, and the better one.

Drive the tip through a volume of Cartesian waypoints with the ORIENTATION
HELD (production IK, posture cost, coordinated multi-joint motion — the style
this arm demonstrably executes well), measuring the board centroid (the one
perception channel validated all campaign) at each stop. With orientation
held, the board's unknown offset from the tip is a CONSTANT vector, and a
constant folds into the translation of a point-set registration — the v1
point-mismatch confound vanishes by construction rather than by cleverness.
Signals are workspace-sized (tens of mm against a ~1 mm floor, versus the
rotation protocol's 2-5 deg against a 0.5 deg floor), and a 3D volume of
points pins ALL of R's rotational freedom — no narrow-cone soft axis.

Where orientation drifts slightly anyway, the offset enters linearly given
the FK wrist rotation, so an alternating fit recovers R, t, and the board's
wrist-frame offset together:

    cam_i = R @ (tip_i + W_i @ c_w) + t

Held-out waypoints report errors in millimetres — the servo's native
currency. The fitted R is also cross-checked against the rotation-protocol R
(two independent instruments agreeing is the strongest certificate we can
produce without ground truth).
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from showservo_handeye_proof import MOTOR_ORDER, BoardEye, enc_vector, kabsch_rotation, sweep_to  # noqa: E402
from showservo_m1_policy import check_depth, make_robot  # noqa: E402

HOVER = {
    "shoulder_pan.pos": 15.1,
    "shoulder_lift.pos": -14.0,
    "elbow_flex.pos": 56.0,
    "forearm_roll.pos": 29.4,
    "wrist_flex.pos": 3.9,
    "wrist_roll.pos": -17.3,
    "gripper.pos": 95.6,
}
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "forearm_roll", "wrist_flex", "wrist_roll")


def main() -> None:
    from lerobot.robots.rest_position import move_to_rest_position
    from lerobot.robots.so107_description.cartesian_ik import make_so107_arm_kinematics
    from lerobot.robots.so107_description.joint_alignment import LEFT_ARM_ALIGNMENT
    from lerobot.showservo.pose import rotation_vector

    robot = make_robot("m1_left")
    try:
        move_to_rest_position(robot, HOVER, duration_s=5.0)
        obs = robot.get_observation()
        check_depth(obs)
        eye = BoardEye(obs)
        # Production posture cost, production usage pattern: the solver is
        # built for small per-tick deltas (waypoint-scale jumps either fall
        # short under the posture cost or branch-flip without it), so each
        # waypoint is reached by seed-walking 10 mm sub-targets.
        kin = make_so107_arm_kinematics(LEFT_ARM_ALIGNMENT)
        cal = robot.calibration
        dpn = np.array(
            [(cal[m].range_max - cal[m].range_min) / 200.0 * (360.0 / 4096.0) for m in MOTOR_ORDER]
        )

        enc0 = enc_vector(obs)
        q0 = enc0 * dpn
        t0 = kin.forward_kinematics(q0)
        r_hold = t0[:3, :3].copy()
        print("hover tip (base frame):", np.round(t0[:3, 3] * 1000, 1), "mm", flush=True)

        # Grid: +-50 mm in x/y, two height levels (0 and 40 mm physically up;
        # base +z measured tonight to point opposite the physical rise).
        offsets = []
        for dz in (0.0, -0.040):
            for dy in (-0.04, 0.0, 0.04):
                for dx in (-0.04, 0.0, 0.04):
                    offsets.append(np.array([dx, dy, dz]))

        records = []
        q_seed = q0.copy()
        for k, off in enumerate(offsets):
            target = t0.copy()
            target[:3, 3] = t0[:3, 3] + off
            try:
                # Seed-walk: the solver is a per-tick-delta solver in
                # production; feed it 10 mm sub-targets from the previous
                # solution so each solve stays in the near branch.
                q_sol = q_seed.copy()
                p_start = kin.forward_kinematics(q_sol)[:3, 3]
                seg = target[:3, 3] - p_start
                n_sub = max(1, int(np.ceil(np.linalg.norm(seg) / 0.010)))
                for s in range(1, n_sub + 1):
                    t_sub = target.copy()
                    t_sub[:3, 3] = p_start + (s / n_sub) * seg
                    q_prev = q_sol.copy()
                    q_sol = kin.inverse_kinematics(q_sol, t_sub)
                    assert np.max(np.abs(q_sol - q_prev)) <= 20.0, "branch flip in seed-walk"
            except Exception as e:
                print(f"wp {k} {np.round(off * 1000)}: IK failed ({e}) — skip", flush=True)
                continue
            t_sol = kin.forward_kinematics(q_sol)
            tip_err = np.linalg.norm(t_sol[:3, 3] - target[:3, 3]) * 1000
            rot_drift = np.degrees(np.linalg.norm(rotation_vector(t_sol[:3, :3] @ r_hold.T)))
            enc_target = q_sol / dpn
            dev = np.max(np.abs(q_sol - q0))
            # Exact tip placement is NOT required (the fit uses executed
            # encoders wherever the arm lands); gate only on safety and on
            # the held orientation that makes the offset constant.
            if rot_drift > 15.0 or dev > 45.0 or np.max(np.abs(enc_target[:6])) > 95.0:
                print(
                    f"wp {k} {np.round(off * 1000)}: rejected (rot_drift {rot_drift:.1f} deg, "
                    f"joint_dev {dev:.1f})",
                    flush=True,
                )
                continue
            if tip_err > 6.0:
                print(
                    f"wp {k} {np.round(off * 1000)}: note — IK lands {tip_err:.1f} mm off-label", flush=True
                )
            sweep_to(
                robot,
                {j: float(enc_target[MOTOR_ORDER.index(j)]) for j in ARM_JOINTS},
            )
            o = robot.get_observation()
            snap = eye.measure(o)
            enc_meas = enc_vector(o)
            if snap is None:
                print(f"wp {k} {np.round(off * 1000)}: designation refused — skip", flush=True)
                q_seed = q_sol
                continue
            q_meas = enc_meas * dpn
            t_meas = kin.forward_kinematics(q_meas)
            records.append(
                {
                    "off": off,
                    "tip": t_meas[:3, 3],
                    "wrot": t_meas[:3, :3],
                    "cam": snap["pos"],
                    "enc": enc_meas,
                }
            )
            print(
                f"wp {k} {np.round(off * 1000)}: ok — cam {np.round(snap['pos'] * 1000, 1)}, "
                f"rot_drift {rot_drift:.1f} deg",
                flush=True,
            )
            q_seed = q_sol

        sweep_to(robot, {j: HOVER[f"{j}.pos"] for j in ARM_JOINTS})
        n = len(records)
        print(f"\n{n} accepted waypoints", flush=True)
        assert n >= 6, "too few accepted waypoints to fit"

        fit_idx = [i for i in range(n) if i % 3 != 2]
        hold_idx = [i for i in range(n) if i % 3 == 2]
        tips = np.stack([records[i]["tip"] for i in fit_idx])
        wrots = np.stack([records[i]["wrot"] for i in fit_idx])
        cams = np.stack([records[i]["cam"] for i in fit_idx])

        # With orientation provably held, the board offset is constant and
        # folds into t — solving for it would only add unconstrained slack.
        drifts = [
            np.degrees(np.linalg.norm(rotation_vector(records[i]["wrot"] @ r_hold.T))) for i in range(n)
        ]
        solve_offset = max(drifts) >= 2.0
        print(
            f"max orientation drift {max(drifts):.2f} deg -> offset solve {'ON' if solve_offset else 'OFF'}",
            flush=True,
        )

        c_w = np.zeros(3)
        r_fit = np.eye(3)
        t_off = np.zeros(3)
        for _ in range(6 if solve_offset else 1):
            pts = tips + np.einsum("nij,j->ni", wrots, c_w)
            pc, cc = pts.mean(0), cams.mean(0)
            r_fit = kabsch_rotation(pts - pc, cams - cc)
            if solve_offset:
                a_rows = np.concatenate([r_fit @ wrots[i] for i in range(len(fit_idx))], axis=0).reshape(
                    -1, 3
                )
                a_full = np.hstack([a_rows, np.tile(np.eye(3), (len(fit_idx), 1))])
                b_full = (cams - np.einsum("ij,nj->ni", r_fit, tips)).reshape(-1)
                sol, *_ = np.linalg.lstsq(a_full, b_full, rcond=None)
                c_w, t_off = sol[:3], sol[3:]
            else:
                t_off = cc - r_fit @ pc

        def predict(i: int) -> np.ndarray:
            return r_fit @ (records[i]["tip"] + records[i]["wrot"] @ c_w) + t_off

        fit_err = [np.linalg.norm(predict(i) - records[i]["cam"]) * 1000 for i in fit_idx]
        hold_err = [np.linalg.norm(predict(i) - records[i]["cam"]) * 1000 for i in hold_idx]
        print(f"board offset in wrist frame: {np.round(c_w * 1000, 1)} mm", flush=True)
        print(f"fit rms {np.sqrt(np.mean(np.square(fit_err))):.1f} mm", flush=True)
        print(
            f"HELD-OUT errors (mm): {np.round(hold_err, 1)} — mean {np.mean(hold_err):.1f}",
            flush=True,
        )

        prev = json.loads(pathlib.Path("outputs/handeye_result.json").read_text())
        r_rot = np.array(prev["modes"]["calibrated-deg"]["r_cam_from_base"])
        cross = np.degrees(np.linalg.norm(rotation_vector(r_fit @ r_rot.T)))
        print(f"R(xyz sweep) vs R(rotation protocol): {cross:.1f} deg apart", flush=True)

        prev["xyz_fit"] = {
            "r_cam_from_base": r_fit.tolist(),
            "t": t_off.tolist(),
            "c_w": c_w.tolist(),
            "fit_rms_mm": float(np.sqrt(np.mean(np.square(fit_err)))),
            "holdout_mm": [float(e) for e in hold_err],
            "n_points": n,
            "records": [
                {"enc": r["enc"].tolist(), "cam": r["cam"].tolist(), "off": r["off"].tolist()}
                for r in records
            ],
        }
        pathlib.Path("outputs/handeye_result.json").write_text(json.dumps(prev, indent=2))
        print("persisted xyz_fit to outputs/handeye_result.json", flush=True)
    finally:
        robot.disconnect()
        print("disconnected (torque holds)", flush=True)


if __name__ == "__main__":
    main()
