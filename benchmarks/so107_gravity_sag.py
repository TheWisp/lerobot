#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gravity sag on the real SO-107: static stations along horizontal paths, by the encoders.

Streams the gripper tip along a radial line away from the base and around a horizontal
circle at constant commanded height through the production Cartesian-IK stack, dwells at
stations, and records commanded vs present joints. FK of the present joints against FK of
the commanded ones is the sag — no camera, no model of the load, just the encoders. The
servos' own holding effort (Present_Load / Present_Current) is sampled at every dwell.

Results and the fix (P=32 + gravity feed-forward) are in
``src/lerobot/robots/so107_description/docs/gravity_sag.md``.

Pre: this script owns the arm's serial bus (disconnect it from the GUI first) and the
start pose lies inside the URDF joint limits — use ``--start-joints`` to leave the fold.
The start pose becomes station R0 and every path is relative to it.

Usage::

    PYTHONPATH=src python benchmarks/so107_gravity_sag.py --start-joints -45 74 -41
    PYTHONPATH=src python benchmarks/so107_gravity_sag.py --p 32 --ff-alpha 3.2
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np

from lerobot.robots.so107_description.cartesian_ik import (
    SO107_WORKSPACE_MAX,
    SO107_WORKSPACE_MIN,
    CartesianIKController,
    make_so107_arm_kinematics,
)
from lerobot.robots.so107_description.joint_alignment import LEFT_ARM_ALIGNMENT, MOTOR_NAMES
from lerobot.robots.so_follower import SO107Follower, SO107FollowerConfig

HZ = 30.0
SPEED_M_S = 0.025
DWELL_S = 1.5
DIVERGE_DEG = 20.0  # present vs commanded on any joint: something is stalled or blocked
HOLD_STREAK_ABORT = 8  # consecutive IK holds = unreachable; stop extending
LAG_HOLD_DEG = 6.0  # a gravity joint this far behind its command pauses the path
LAG_RESUME_DEG = 4.0
LAG_STALL_S = 6.0  # paused this long with no catch-up = the joint cannot get there
URDF_SEAM_MARGIN_DEG = 15.0
GRAVITY_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex")


def urdf_deg(q_motor: np.ndarray) -> np.ndarray:
    return np.array(
        [
            LEFT_ARM_ALIGNMENT[m].sign * q_motor[i] + LEFT_ARM_ALIGNMENT[m].offset_deg
            for i, m in enumerate(MOTOR_NAMES)
        ]
    )


def stream_joints(
    robot, q_from: np.ndarray, q_to: np.ndarray, deg_per_s: float = 20.0, hz: float = 50.0
) -> np.ndarray:
    """Linear joint-space move at a bounded rate — the shape the rest interpolator proved on this arm.

    Post: returns the present joints; prints the residual if the arm could not get there in 5 s.
    """
    dur = max(0.5, float(np.max(np.abs(q_to - q_from))) / deg_per_s)
    n = int(math.ceil(dur * hz))
    for i in range(1, n + 1):
        t_tick = time.perf_counter()
        q = q_from + (q_to - q_from) * (i / n)
        robot.send_action({f"{m}.pos": float(q[j]) for j, m in enumerate(MOTOR_NAMES)})
        time.sleep(max(0.0, 1.0 / hz - (time.perf_counter() - t_tick)))
    t_end = time.perf_counter()
    while True:
        obs = robot.get_observation()
        q = np.array([obs[f"{m}.pos"] for m in MOTOR_NAMES], dtype=float)
        if np.max(np.abs(q[:6] - q_to[:6])) < 1.0 or time.perf_counter() - t_end > 5.0:
            break
        robot.send_action({f"{m}.pos": float(q_to[j]) for j, m in enumerate(MOTOR_NAMES)})
        time.sleep(0.1)
    residual = ", ".join(
        f"{m} {q[j] - q_to[j]:+.1f}" for j, m in enumerate(MOTOR_NAMES) if abs(q[j] - q_to[j]) > 0.5
    )
    print(f"  start move residual after {time.perf_counter() - t_end:.1f} s: {residual or 'none'}")
    return q


class Run:
    """One connected arm, one IK controller, a tick log, and the station protocol."""

    def __init__(self, robot, ctrl, kin, p0: np.ndarray, grip: float, out: pathlib.Path):
        self.robot, self.ctrl, self.kin, self.p0, self.grip, self.out = robot, ctrl, kin, p0, grip, out
        self.rows: list[list[float]] = []
        self.stations: list[dict] = []
        self.pos_cmd = p0.copy()
        self.t0 = time.perf_counter()
        self.can_load = True
        self.ki, self.i_clamp = 0.0, 8.0
        self.i_state = np.zeros(len(MOTOR_NAMES))
        self.q_now: np.ndarray | None = None
        self.grav_idx = [MOTOR_NAMES.index(j) for j in GRAVITY_JOINTS]
        self.stall_s = 0.0

    def present(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return np.array([obs[f"{m}.pos"] for m in MOTOR_NAMES], dtype=float)

    def tick(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        d = pos - self.p0
        out = self.ctrl(
            {
                "enabled": 1.0,
                "target_x": float(d[0]),
                "target_y": float(d[1]),
                "target_z": float(d[2]),
                "target_wx": 0.0,
                "target_wy": 0.0,
                "target_wz": 0.0,
                "gripper_pos": self.grip,
            }
        )
        q_cmd = np.array([out[f"{m}.pos"] for m in MOTOR_NAMES], dtype=float)
        if self.ki > 0 and self.q_now is not None:
            # Outer integral action on the gravity joints, bounded so a stalled joint cannot wind up.
            err = q_cmd - self.q_now
            for j in self.grav_idx:
                self.i_state[j] = float(
                    np.clip(self.i_state[j] + self.ki * err[j], -self.i_clamp, self.i_clamp)
                )
        q_send = q_cmd + self.i_state
        self.robot.send_action({f"{m}.pos": float(q_send[j]) for j, m in enumerate(MOTOR_NAMES)})
        q_now = self.present()
        self.q_now = q_now
        holding = bool(self.ctrl.is_holding)
        self.rows.append([time.perf_counter() - self.t0, *q_cmd, *q_now, *pos, float(holding), *q_send])
        div = np.max(np.abs(q_now[:6] - q_cmd[:6]))
        if div > DIVERGE_DEG:
            raise RuntimeError(
                f"joint divergence {div:.1f} deg (cmd {np.round(q_cmd[:6], 1)} now {np.round(q_now[:6], 1)})"
            )
        if (self.out / "STOP").exists():
            raise RuntimeError("STOP file present")
        return q_cmd, q_now, holding

    def lag(self) -> float:
        if self.q_now is None or not self.rows:
            return 0.0
        cmd = np.array(self.rows[-1][1 : 1 + len(MOTOR_NAMES)])
        return float(max(abs(cmd[j] - self.q_now[j]) for j in self.grav_idx))

    def wait_for_arm(self, pos: np.ndarray, label: str) -> None:
        """Hold the current point while a loaded joint catches up; stall time is logged, not hidden."""
        if self.lag() <= LAG_HOLD_DEG:
            return
        t_start = time.perf_counter()
        while self.lag() > LAG_RESUME_DEG:
            t_tick = time.perf_counter()
            self.tick(pos)
            time.sleep(max(0.0, 1.0 / HZ - (time.perf_counter() - t_tick)))
            if time.perf_counter() - t_start > LAG_STALL_S:
                raise RuntimeError(
                    f"{label}: joint lag {self.lag():.1f} deg did not close in {LAG_STALL_S:.0f} s"
                )
        self.stall_s += time.perf_counter() - t_start

    def goto(self, target: np.ndarray, label: str) -> bool:
        """Stream straight to target at SPEED_M_S. Returns False if the IK kept holding."""
        start = self.pos_cmd.copy()
        dist = float(np.linalg.norm(target - start))
        n = max(1, int(math.ceil(dist / (SPEED_M_S / HZ))))
        streak = 0
        for i in range(1, n + 1):
            t_tick = time.perf_counter()
            pos = start + (target - start) * (i / n)
            self.wait_for_arm(pos, label)
            _, _, holding = self.tick(pos)
            streak = streak + 1 if holding else 0
            if streak >= HOLD_STREAK_ABORT:
                print(
                    f"  {label}: IK held {streak} ticks at {np.round(pos * 1000, 1)} mm — unreachable, stopping this leg"
                )
                return False
            time.sleep(max(0.0, 1.0 / HZ - (time.perf_counter() - t_tick)))
        self.pos_cmd = target.copy()
        return True

    def dwell(self, label: str) -> dict:
        n = int(DWELL_S * HZ)
        tail_q, loads, currents = [], [], []
        q_cmd = None
        for i in range(n):
            t_tick = time.perf_counter()
            q_cmd, q_now, _ = self.tick(self.pos_cmd)
            if i >= n // 2:
                tail_q.append(q_now)
                if self.can_load and i % 5 == 0:
                    try:
                        loads.append(
                            [self.robot.bus.read("Present_Load", m, normalize=False) for m in MOTOR_NAMES]
                        )
                        currents.append(
                            [self.robot.bus.read("Present_Current", m, normalize=False) for m in MOTOR_NAMES]
                        )
                    except Exception as e:  # effort readback is a bonus, never the experiment
                        print("  (effort readback unavailable:", e, ")")
                        self.can_load = False
            time.sleep(max(0.0, 1.0 / HZ - (time.perf_counter() - t_tick)))
        q_mean = np.mean(tail_q, axis=0)
        try:
            temps = [int(self.robot.bus.read("Present_Temperature", m, normalize=False)) for m in MOTOR_NAMES]
        except Exception:
            temps = None
        fk_cmd = self.kin.forward_kinematics(q_cmd)[:3, 3]
        fk_now = self.kin.forward_kinematics(q_mean)[:3, 3]
        st = {
            "label": label,
            "pos_des_m": self.pos_cmd.tolist(),
            "q_cmd_deg": q_cmd.tolist(),
            "q_present_deg": q_mean.tolist(),
            "fk_cmd_m": fk_cmd.tolist(),
            "fk_present_m": fk_now.tolist(),
            "reach_m": float(np.linalg.norm(self.pos_cmd[:2])),
            "load_raw": np.mean(loads, axis=0).tolist() if loads else None,
            "current_raw": np.mean(currents, axis=0).tolist() if currents else None,
            "i_state_deg": self.i_state.tolist(),
            "stall_s_so_far": self.stall_s,
            "temp_c": temps,
        }
        dq = q_mean - q_cmd
        print(
            f"  {label:5s} reach {st['reach_m'] * 1000:6.1f} mm  z_cmd {fk_cmd[2] * 1000:6.1f}  z_fk(present) {fk_now[2] * 1000:6.1f}"
            f"  dz {1000 * (fk_now[2] - fk_cmd[2]):+5.1f} mm  dq(lift {dq[1]:+.2f} elbow {dq[2]:+.2f} wrist {dq[4]:+.2f} deg)"
            f"  I(lift {self.i_state[1]:+.1f} elbow {self.i_state[2]:+.1f})  T {max(temps) if temps else '?'}C"
        )
        self.stations.append(st)
        return st

    def save(self) -> None:
        columns = (
            ["t"]
            + [f"cmd_{m}" for m in MOTOR_NAMES]
            + [f"now_{m}" for m in MOTOR_NAMES]
            + ["x", "y", "z", "holding"]
            + [f"send_{m}" for m in MOTOR_NAMES]
        )
        np.savez_compressed(self.out / "ticks.npz", rows=np.array(self.rows), columns=json.dumps(columns))
        (self.out / "stations.json").write_text(json.dumps(self.stations, indent=1))


def connect_arm(args) -> tuple[SO107Follower, np.ndarray]:
    """Connect the left arm, write the requested gains, optionally move to a start pose.

    Post: the returned joints are inside the URDF limits with a margin, else the arm is
    disconnected and the error says how far it sits from the seam.
    """
    cfg = SO107FollowerConfig(
        id=args.id,
        port=args.port,
        use_degrees=True,
        disable_torque_on_disconnect=False,
        max_relative_target=float(args.max_rel),
        cameras={},
        p_coefficient=int(args.p),
        i_coefficient=int(args.i),
        d_coefficient=int(args.d),
        gravity_ff_alpha=float(args.ff_alpha),
        gravity_ff_arm="left",
    )
    robot = SO107Follower(cfg)
    robot.connect(calibrate=False)
    try:
        assert robot.is_calibrated, "arm reports uncalibrated"
        dead = [m for m in MOTOR_NAMES if int(robot.bus.read("Torque_Enable", m)) != 1]
        assert not dead, f"torque OFF on {dead}"
        obs = robot.get_observation()
        q0 = np.array([obs[f"{m}.pos"] for m in MOTOR_NAMES], dtype=float)
        if args.start_joints is not None:
            q_to = q0.copy()
            q_to[1], q_to[2], q_to[4] = args.start_joints
            print("streaming to start pose (deg):", dict(zip(MOTOR_NAMES, np.round(q_to, 1), strict=True)))
            q0 = stream_joints(robot, q0, q_to)
        u = urdf_deg(q0)
        assert np.all(np.abs(u[:6]) < 180.0 - URDF_SEAM_MARGIN_DEG), (
            f"URDF-space joints {np.round(u[:6], 1)} sit within {URDF_SEAM_MARGIN_DEG:.0f} deg of +-180 — "
            "move the arm forward of its fold first (--start-joints)"
        )
        return robot, q0
    except Exception:
        robot.disconnect()
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--id", default="white_left")
    ap.add_argument("--radial-mm", type=float, nargs="*", default=[30, 60, 90])
    ap.add_argument("--circle-r-mm", type=float, default=40.0)
    ap.add_argument(
        "--start-joints",
        type=float,
        nargs=3,
        metavar=("LIFT", "ELBOW", "WRIST_FLEX"),
        help="stream in joint space to this (motor deg) pose first; other joints unchanged",
    )
    ap.add_argument("--only-start", action="store_true", help="stop after the start move")
    ap.add_argument("--p", type=int, default=16, help="servo P_Coefficient (follower default 16)")
    ap.add_argument("--i", type=int, default=0, help="servo I_Coefficient (follower default 0)")
    ap.add_argument("--d", type=int, default=32, help="servo D_Coefficient (follower default 32)")
    ap.add_argument(
        "--ff-alpha",
        type=float,
        default=0.0,
        help="gravity feed-forward in the follower, deg per N*m (0 = off)",
    )
    ap.add_argument(
        "--ki", type=float, default=0.0, help="outer integral gain per tick on the gravity joints (0 = off)"
    )
    ap.add_argument("--i-clamp", type=float, default=8.0, help="outer integrator bound, deg")
    ap.add_argument(
        "--max-rel", type=float, default=5.0, help="follower max_relative_target (deg goal-vs-present clamp)"
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = pathlib.Path(args.out or f"outputs/gravity_sag/{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)

    kin = make_so107_arm_kinematics(LEFT_ARM_ALIGNMENT)
    robot, q0 = connect_arm(args)
    run = None
    try:
        if args.only_start:
            print("arrived (deg):", dict(zip(MOTOR_NAMES, np.round(q0, 1), strict=True)))
            return
        t0 = kin.forward_kinematics(q0)
        p0 = t0[:3, 3].copy()
        lo, hi = np.array(SO107_WORKSPACE_MIN), np.array(SO107_WORKSPACE_MAX)
        print("start joints (deg):", dict(zip(MOTOR_NAMES, np.round(q0, 1), strict=True)))
        print("start EE (m):", np.round(p0, 4), " workspace box", lo, hi)
        assert np.all(p0 > lo + 0.005) and np.all(p0 < hi - 0.005), (
            "start pose is outside/at the IK workspace box"
        )
        q_chk = np.asarray(kin.inverse_kinematics(q0, t0))
        assert np.max(np.abs(q_chk[:6] - q0[:6])) < 2.0, "IK does not reproduce the start pose"

        flat = np.array([p0[0], p0[1], 0.0])
        inward = -flat / np.linalg.norm(flat)
        outward = -inward
        perp = np.cross(inward, [0.0, 0.0, 1.0])
        perp /= np.linalg.norm(perp)
        ctrl = CartesianIKController(
            kinematics=kin,
            motor_names=list(MOTOR_NAMES),
            q_init=q0,
            workspace_min=SO107_WORKSPACE_MIN,
            workspace_max=SO107_WORKSPACE_MAX,
            label="left",
        )
        run = Run(robot, ctrl, kin, p0, float(q0[MOTOR_NAMES.index("gripper")]), out)
        run.ki, run.i_clamp = args.ki, args.i_clamp
        (out / "meta.json").write_text(
            json.dumps(
                {
                    "p0_m": p0.tolist(),
                    "outward": outward.tolist(),
                    "perp": perp.tolist(),
                    "q0_deg": q0.tolist(),
                    "radial_mm": args.radial_mm,
                    "circle_r_mm": args.circle_r_mm,
                    "speed_m_s": SPEED_M_S,
                    "hz": HZ,
                    "servo_p": args.p,
                    "servo_i": args.i,
                    "servo_d": args.d,
                    "ff_alpha": args.ff_alpha,
                    "ki": args.ki,
                    "i_clamp": args.i_clamp,
                    "max_rel": args.max_rel,
                },
                indent=1,
            )
        )

        print("== radial line, outward from the base, constant z ==")
        run.dwell("R0")
        for d in args.radial_mm:
            if not run.goto(p0 + outward * (d / 1000.0), f"R{int(d)}"):
                break
            run.dwell(f"R{int(d)}")
        run.goto(p0, "back")
        run.dwell("R0b")

        r = args.circle_r_mm / 1000.0
        if r > 0:
            print(f"== horizontal circle r={args.circle_r_mm:.0f} mm, nearest point = start ==")
            c = p0 + outward * r

            def on_circle(theta):
                return c + r * (-outward * math.cos(theta) + perp * math.sin(theta))

            n_st = 8
            for k in range(1, n_st + 1):
                th0, th1 = 2 * math.pi * (k - 1) / n_st, 2 * math.pi * k / n_st
                n = max(1, int(math.ceil(r * (th1 - th0) / (SPEED_M_S / HZ))))
                for i in range(1, n + 1):
                    t_tick = time.perf_counter()
                    pos = on_circle(th0 + (th1 - th0) * i / n)
                    run.wait_for_arm(pos, f"C{k}")
                    run.tick(pos)
                    run.pos_cmd = pos.copy()
                    time.sleep(max(0.0, 1.0 / HZ - (time.perf_counter() - t_tick)))
                run.dwell(f"C{k % n_st}")
        run.save()
        print("saved", out)
    finally:
        if run is not None:
            run.save()
        robot.disconnect()
        print("arm disconnected (torque on, holding)")


if __name__ == "__main__":
    main()
