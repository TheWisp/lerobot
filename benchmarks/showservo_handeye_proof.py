# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Prove the hand-eye measurement before building the IK-based servo on it.

The claim under test: three probe moves, each recorded twice — by the CAMERA
(depth-lifted bind of the wrist board, camera frame) and by the MODEL (URDF
forward kinematics of the tip, base frame) — determine the fixed camera<-base
rotation R via Kabsch, and R @ model displacements then predict what the
camera sees for moves it has never fitted.

Protocol:
  1. Fit set: sweep each servo joint (decisive rate), record (encoder deltas,
     camera displacement m_i, model displacement b_i).
  2. The units question answers itself: FK is evaluated under BOTH readings of
     the alignment's inputs (normalized-as-degrees vs calibration-converted
     degrees); the reading whose |b_i| magnitudes match |m_i| is the truth
     (millimetres are millimetres in any frame).
  3. R = Kabsch(m set, b set), no translation (displacements cancel it).
  4. Validation: two held-out COMBINED-joint moves; report the angle between
     R @ b and the measured m, and their magnitude ratio. The premise is
     proven if held-out directions agree within ~15 degrees.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from showservo_m1_policy import M1_JOINTS, check_depth, joint_positions, make_robot  # noqa: E402

MOTOR_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def enc_vector(obs: dict) -> np.ndarray:
    pos = joint_positions(obs)
    return np.array([pos[m] for m in MOTOR_ORDER])


def sweep_to(robot, targets: dict[str, float], hz: float = 50.0, speed: float = 25.0) -> None:
    """Decisive linear sweep, rest-replay style."""
    cur = joint_positions(robot.get_observation())
    ref = {j: cur[j] for j in targets}
    step = speed / hz
    while any(abs(targets[j] - ref[j]) > 1e-9 for j in ref):
        for j in ref:
            ref[j] += float(np.clip(targets[j] - ref[j], -step, step))
        robot.send_action({f"{j}.pos": v for j, v in ref.items()})
        time.sleep(1.0 / hz)
    time.sleep(1.2)  # settle before measuring


class BoardEye:
    """Camera-frame board position via the exact held-bind pipeline."""

    def __init__(self, reference_obs: dict):
        from showservo_m0 import DinoTier
        from showservo_real import Card, Designator

        from lerobot.showservo.pose import CameraIntrinsics

        meta = json.loads((pathlib.Path("captures/session_20260814_074909") / "intrinsics.json").read_text())
        self.intr = CameraIntrinsics(fx=meta["fx"], fy=meta["fy"], cx=meta["cx"], cy=meta["cy"])
        self.designator = Designator("sam3", "circuit board", "cuda")
        self.tier = DinoTier("facebook/dinov3-vits16-pretrain-lvd1689m", device="cuda")
        frame = self._frame(reference_obs)
        mask = self.designator.mask(frame)
        assert mask is not None, "board not designated in the reference view"
        self.card = Card(frame, mask, self.tier, self.intr)
        self.ref_centroid = self.card.xyz.mean(axis=0)

    @staticmethod
    def _frame(obs: dict):
        class F:
            pass

        f = F()
        f.rgb = np.asarray(obs["top"])
        f.depth = np.asarray(obs["top_depth"], dtype=np.float32) / 1000.0
        return f

    def position(self, obs: dict) -> np.ndarray | None:
        """Board centroid in camera frame, metres; None if the bind refuses."""
        from showservo_real import bind_rigid3d

        frame = self._frame(obs)
        mask = self.designator.mask(frame)
        if mask is None:
            return None
        fit, _ = bind_rigid3d(self.card, frame, mask, self.tier, self.intr)
        if fit is None:
            return None
        return fit.transform.apply(self.ref_centroid.reshape(1, 3))[0]


def kabsch_rotation(b: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Rotation R minimizing ||m_i - R b_i||^2 (rows are vectors)."""
    h = m.T @ b
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt)) or 1.0
    return u @ np.diag([1.0, 1.0, d]) @ vt


def main() -> None:
    import json as _json  # noqa: F401 (top-level json used in BoardEye via global)

    robot = make_robot("m1_left")
    try:
        obs = robot.get_observation()
        check_depth(obs)
        eye = BoardEye(obs)
        home = joint_positions(obs)
        print("home:", {j: round(home[j], 2) for j in M1_JOINTS}, flush=True)

        from lerobot.robots.so107_description.cartesian_ik import make_so107_arm_kinematics
        from lerobot.robots.so107_description.joint_alignment import LEFT_ARM_ALIGNMENT

        kin = make_so107_arm_kinematics(LEFT_ARM_ALIGNMENT)

        # Two unit interpretations for FK input; the scale certificate arbitrates.
        cal = robot.calibration
        deg_per_norm = {
            m: (cal[m].range_max - cal[m].range_min) / 200.0 * (360.0 / 4096.0) for m in MOTOR_ORDER
        }

        def fk_tip(enc: np.ndarray, mode: str) -> np.ndarray:
            if mode == "norm-as-deg":
                q = enc.copy()
            else:
                q = np.array([enc[i] * deg_per_norm[m] for i, m in enumerate(MOTOR_ORDER)])
            t = kin.forward_kinematics(q)
            return t[:3, 3]

        moves = [
            {"shoulder_pan": +6.0},
            {"shoulder_lift": +6.0},
            {"elbow_flex": -6.0},
            {"shoulder_pan": -4.0, "elbow_flex": +4.0},  # held-out
            {"shoulder_lift": -4.0, "shoulder_pan": -3.0},  # held-out
        ]
        records = []
        for k, deltas in enumerate(moves):
            before_obs = robot.get_observation()
            p0 = eye.position(before_obs)
            e0 = enc_vector(before_obs)
            targets = {j: joint_positions(before_obs)[j] + d for j, d in deltas.items()}
            sweep_to(robot, targets)
            after_obs = robot.get_observation()
            p1 = eye.position(after_obs)
            e1 = enc_vector(after_obs)
            # return to start pose for independence of the next move
            sweep_to(robot, {j: joint_positions(before_obs)[j] for j in deltas})
            assert p0 is not None and p1 is not None, f"bind refused around move {k}"
            m = p1 - p0
            records.append({"deltas": deltas, "m": m, "e0": e0, "e1": e1})
            print(
                f"move {k} {deltas}: camera |m| {np.linalg.norm(m) * 1000:6.1f} mm, "
                f"executed {np.round(e1 - e0, 2)[:3]}",
                flush=True,
            )

        for mode in ("norm-as-deg", "calibrated-deg"):
            b = np.stack([fk_tip(r["e1"], mode) - fk_tip(r["e0"], mode) for r in records])
            m_all = np.stack([r["m"] for r in records])
            ratios = np.linalg.norm(m_all, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-9)
            print(f"\n[{mode}] |camera|/|model| per move: {np.round(ratios, 2)}", flush=True)
            r_fit = kabsch_rotation(b[:3], m_all[:3])  # fit on the 3 single-joint moves
            for k, rec in enumerate(records):
                pred = r_fit @ b[k]
                meas = rec["m"]
                cosang = float(pred @ meas / (np.linalg.norm(pred) * np.linalg.norm(meas) + 1e-12))
                ang = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
                tag = "FIT" if k < 3 else "HELD-OUT"
                print(
                    f"  [{mode}] move {k} ({tag}): angle(pred, meas) {ang:5.1f} deg, "
                    f"|pred| {np.linalg.norm(pred) * 1000:6.1f} mm vs |meas| "
                    f"{np.linalg.norm(meas) * 1000:6.1f} mm",
                    flush=True,
                )
    finally:
        robot.disconnect()
        print("disconnected (torque holds)", flush=True)


import json  # noqa: E402  (BoardEye reads intrinsics)

if __name__ == "__main__":
    main()
