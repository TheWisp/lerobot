# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Prove the hand-eye rotation before building the IK-based servo on it — v2.

v1 compared POSITION displacements and failed on two confounds: the camera
tracks the board centroid while the model reports the gripper tip (different
points on one wrist translate differently under wrist rotation), and
encoder-based FK overpredicts executed motion by the drivetrain's slack
fraction. Both are structural to positions.

v2 calibrates on ROTATIONS, which dodge both by construction. Every point on
the board experiences the SAME rotation (point identity cannot matter), and
slack shrinks how far a joint turns but not which axis it turns about (the
axis is machined into the joint). The camera side comes from a channel the
bind computes every frame and the servo has never consumed: the fitted
transform's rotation.

v3 note: the v2 run disqualified its own instrument. The bind's rotation
channel read the SAME physical wrist flex as 0.91 deg at one forearm-roll
configuration and 43.07 deg at another — rotation angle is basis-invariant,
so at least one reading lies, and the project record had already convicted
this channel on objects (a 90-deg roll certified as 4 deg). Only the bind's
CENTROID was ever validated. v3 measures orientation from the board's
depth-plane NORMAL instead: a plane fit over thousands of raw depth pixels,
no feature correspondences to slide. The old channel is still printed next
to the new one per move, as the record of the divergence.

Protocol (v3):
  1. Six moves, each measured as before/after (normal, centroid, encoders):
     pan +-10, lift +10, wrist_flex +15, two combined held-outs, and
     wrist_flex +15 again with the forearm roll zeroed — the same physical
     flex at two configurations must swing the normal by the same angle
     (instrument repeatability certificate).
  2. The board's fixed normal in the wrist frame (2 DOF) is fit R-free:
     angles BETWEEN camera normals across poses equal angles between
     model-rotated candidates, because a rigid R preserves angles.
  3. R by Kabsch on (model normal, camera normal) pairs across all poses,
     anchored by the pan move's translation direction (the one position
     channel both proofs agreed on) against the normals' narrow cone.
  4. Certificates: camera/model normal-swing ratio per move re-measures the
     per-joint slack with a trustworthy instrument and arbitrates units;
     held-out poses' predicted normals must land within ~10-15 deg.
  5. Stage 2 unchanged: with R fixed, the board's wrist-frame offset enters
     the POSITION records linearly; held-out direction errors are the final,
     servo-grade gate.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from showservo_m1_policy import check_depth, joint_positions, make_robot  # noqa: E402

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


# The parked right arm carries an IDENTICAL wrist PCB right of this column; a
# raw "circuit board" designation provably jumps to it when the left PCB tilts
# oblique (the v3 run's 170 mm / 50 deg "motion" was that jump, frame-verified).
ROI_X_MAX = 520  # hardcode-ok: rig-specific proof script, camera fixed
JUMP_LIMIT_M = 0.08


class BoardEye:
    """Camera-frame board POSE (rotation + centroid) via the held-bind pipeline.

    Designations pass two gates the servo pipeline already fields against
    imposters: a workspace ROI excluding the parked twin arm, and a continuity
    refusal when the masked centroid jumps farther than a real board could
    move between snapshots. Refusal (None) beats fiction.
    """

    def __init__(self, reference_obs: dict):
        from showservo_m0 import DinoTier
        from showservo_real import Card, Designator

        from lerobot.showservo.pose import CameraIntrinsics

        meta = json.loads((pathlib.Path("captures/session_20260814_074909") / "intrinsics.json").read_text())
        self.intr = CameraIntrinsics(fx=meta["fx"], fy=meta["fy"], cx=meta["cx"], cy=meta["cy"])
        self.designator = Designator("sam3", "circuit board", "cuda")
        self.tier = DinoTier("facebook/dinov3-vits16-pretrain-lvd1689m", device="cuda")
        self._last_centroid: np.ndarray | None = None
        frame = self._frame(reference_obs)
        mask = self._gated_mask(frame)
        assert mask is not None, "board not designated in the reference view"
        self.card = Card(frame, mask, self.tier, self.intr)
        self.ref_centroid = self.card.xyz.mean(axis=0)

    def _lift_masked(self, frame, mask: np.ndarray) -> np.ndarray | None:
        ys, xs = np.nonzero(mask)
        z = frame.depth[ys, xs]
        ok = z > 0.05
        ys, xs, z = ys[ok], xs[ok], z[ok]
        if len(z) < 200:
            return None
        lo, hi = np.percentile(z, [2.0, 98.0])
        ok = (z >= lo) & (z <= hi)
        ys, xs, z = ys[ok], xs[ok], z[ok]
        x = (xs - self.intr.cx) * z / self.intr.fx
        y = (ys - self.intr.cy) * z / self.intr.fy
        return np.stack([x, y, z], axis=1)

    def _gated_mask(self, frame) -> np.ndarray | None:
        # Designate WITHIN the workspace crop: with the full frame, SAM3
        # provably grabs the parked twin arm's identical PCB whenever the
        # left one presents badly. Cropping first forces it to find the
        # left PCB or nothing — refusal stays honest, fiction stays out.
        sub = type(frame)()
        sub.rgb = frame.rgb[:, :ROI_X_MAX]
        sub.depth = frame.depth[:, :ROI_X_MAX]
        sub_mask = self.designator.mask(sub)
        if sub_mask is None:
            print("  [gate] designator found nothing inside the ROI — refused", flush=True)
            return None
        mask = np.zeros(frame.rgb.shape[:2], dtype=bool)
        mask[:, :ROI_X_MAX] = sub_mask
        pts = self._lift_masked(frame, mask)
        if pts is None:
            print(
                f"  [gate] {int(mask.sum())} px designated inside ROI, <200 with valid depth — refused",
                flush=True,
            )
            return None
        centroid = pts.mean(axis=0)
        if self._last_centroid is not None and np.linalg.norm(centroid - self._last_centroid) > JUMP_LIMIT_M:
            print(
                f"  [gate] designation jumped {np.linalg.norm(centroid - self._last_centroid) * 1000:.0f} mm"
                " — refused",
                flush=True,
            )
            return None
        self._last_centroid = centroid
        return mask

    def measure(self, obs: dict) -> dict | None:
        """One gated designation → the full snapshot: bind rotation + centroid,
        plane normal, flatness. None if any stage refuses (reason printed)."""
        from showservo_real import bind_rigid3d

        frame = self._frame(obs)
        mask = self._gated_mask(frame)
        if mask is None:
            return None
        pts = self._lift_masked(frame, mask)
        assert pts is not None  # gate already required valid depth
        q = pts - pts.mean(axis=0)
        w, v = np.linalg.eigh(q.T @ q)
        n = v[:, 0]
        if n[2] > 0:
            n = -n
        fit, _ = bind_rigid3d(self.card, frame, mask, self.tier, self.intr)
        if fit is None:
            print("  [gate] bind refused (low inliers) on a gated mask", flush=True)
            return None
        return {
            "rot": fit.transform.rot,
            "pos": fit.transform.apply(self.ref_centroid.reshape(1, 3))[0],
            "normal": n,
            "flat": float(w[0] / max(w[1], 1e-12)),
            "rgb": np.asarray(obs["top"]).copy(),
            "mask": mask,
        }

    @staticmethod
    def _frame(obs: dict):
        class F:
            pass

        f = F()
        f.rgb = np.asarray(obs["top"])
        f.depth = np.asarray(obs["top_depth"], dtype=np.float32) / 1000.0
        return f


def kabsch_rotation(b: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Rotation R minimizing ||m_i - R b_i||^2 (rows are vectors)."""
    h = m.T @ b
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt)) or 1.0
    return u @ np.diag([1.0, 1.0, d]) @ vt


def main() -> None:
    from lerobot.robots.so107_description.cartesian_ik import make_so107_arm_kinematics
    from lerobot.robots.so107_description.joint_alignment import LEFT_ARM_ALIGNMENT
    from lerobot.showservo.pose import rotation_vector

    robot = make_robot("m1_left")
    try:
        # The wrist sags while torque is dropped between runs, and an oblique
        # PCB is exactly what invites the designator to wander. Normalize to
        # the hover every successful designation this campaign used.
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

        refusal_dir = pathlib.Path("outputs/handeye_frames")
        refusal_dir.mkdir(parents=True, exist_ok=True)

        def save_snap(tag: str, obs_s: dict, snap: dict | None) -> None:
            from PIL import Image

            rgb = np.asarray(obs_s["top"]).copy()
            if snap is not None:
                rgb[snap["mask"]] = (0.5 * rgb[snap["mask"]] + np.array([127.0, 0.0, 0.0])).astype(np.uint8)
            Image.fromarray(rgb).save(refusal_dir / f"{tag}.png")

        kin = make_so107_arm_kinematics(LEFT_ARM_ALIGNMENT)
        cal = robot.calibration
        deg_per_norm = {
            m: (cal[m].range_max - cal[m].range_min) / 200.0 * (360.0 / 4096.0) for m in MOTOR_ORDER
        }

        def fk_pose(enc: np.ndarray, mode: str) -> np.ndarray:
            q = (
                enc.copy()
                if mode == "norm-as-deg"
                else np.array([enc[i] * deg_per_norm[m] for i, m in enumerate(MOTOR_ORDER)])
            )
            return kin.forward_kinematics(q)

        moves = [
            {"shoulder_pan": +10.0},
            {"shoulder_lift": +10.0},
            {"wrist_flex": +10.0},
            {"shoulder_pan": -6.0, "wrist_flex": -8.0},  # held-out
            {"shoulder_lift": -6.0, "wrist_flex": +6.0},  # held-out
            {"_pre": {"forearm_roll": 0.0}, "wrist_flex": +10.0},  # repeatability probe
        ]
        records = []
        for k, spec in enumerate(moves):
            deltas = {j: d for j, d in spec.items() if not j.startswith("_")}
            pre = spec.get("_pre")
            pre_home = None
            if pre:
                pre_home = {j: joint_positions(robot.get_observation())[j] for j in pre}
                sweep_to(robot, pre)
            before_obs = robot.get_observation()
            snap0 = eye.measure(before_obs)
            e0 = enc_vector(before_obs)
            start = joint_positions(before_obs)
            sweep_to(robot, {j: start[j] + d for j, d in deltas.items()})
            after_obs = robot.get_observation()
            snap1 = eye.measure(after_obs)
            e1 = enc_vector(after_obs)
            sweep_to(robot, {j: start[j] for j in deltas})  # home again for independence
            if pre_home:
                sweep_to(robot, pre_home)
            save_snap(f"m{k}_before", before_obs, snap0)
            save_snap(f"m{k}_after", after_obs, snap1)
            if snap0 is None or snap1 is None:
                print(
                    f"move {k} {deltas}: REFUSED — skipping (frames saved), refusal over fiction", flush=True
                )
                records.append(None)
                continue
            n0, n1 = snap0["normal"], snap1["normal"]
            m_vec = snap1["pos"] - snap0["pos"]
            swing = float(np.degrees(np.arccos(np.clip(n0 @ n1, -1, 1))))
            old_rot = float(np.degrees(np.linalg.norm(rotation_vector(snap1["rot"] @ snap0["rot"].T))))
            records.append(
                {"deltas": deltas, "n0": n0, "n1": n1, "m": m_vec, "e0": e0, "e1": e1, "swing": swing}
            )
            print(
                f"move {k} {deltas}: NORMAL swing {swing:5.2f} deg (bind-rot said {old_rot:5.1f}), "
                f"|m| {np.linalg.norm(m_vec) * 1000:5.1f} mm, flatness {max(snap0['flat'], snap1['flat']):.4f}, "
                f"executed {np.round(e1 - e0, 1)}",
                flush=True,
            )

        assert all(r is not None for r in records[:3]), "a FIT move was refused — cannot calibrate this run"
        wf = MOTOR_ORDER.index("wrist_flex")
        for a, b in ((2, 5),):
            if records[a] is None or records[b] is None:
                continue
            sa = records[a]["swing"] / max(abs(records[a]["e1"][wf] - records[a]["e0"][wf]), 1e-9)
            sb = records[b]["swing"] / max(abs(records[b]["e1"][wf] - records[b]["e0"][wf]), 1e-9)
            print(
                f"\nREPEATABILITY: wrist swing per executed unit — roll@28.7 {sa:.3f} vs roll@0 {sb:.3f} "
                f"deg/u (must match for the instrument to be trusted)",
                flush=True,
            )

        def ang(u: np.ndarray, v: np.ndarray) -> float:
            return float(np.degrees(np.arccos(np.clip(u @ v, -1.0, 1.0))))

        results: dict = {
            "home": {m: float(home[m]) for m in MOTOR_ORDER},
            "records": [
                None
                if rec is None
                else {
                    "deltas": rec["deltas"],
                    "e0": rec["e0"].tolist(),
                    "e1": rec["e1"].tolist(),
                    "n0": rec["n0"].tolist(),
                    "n1": rec["n1"].tolist(),
                    "m": rec["m"].tolist(),
                    "swing_deg": rec["swing"],
                }
                for rec in records
            ],
            "modes": {},
        }

        def fib_sphere(n: int) -> np.ndarray:
            i = np.arange(n) + 0.5
            phi = np.arccos(1.0 - 2.0 * i / n)
            th = np.pi * (1.0 + 5.0**0.5) * i
            return np.stack([np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th), np.cos(phi)], axis=1)

        for mode in ("norm-as-deg", "calibrated-deg"):
            # Snapshots (encoder pose, camera normal). Fit uses moves 0-2 only;
            # moves 3-5 stay untouched validation.
            fit_snaps = [(rec[e], rec[n]) for rec in records[:3] for e, n in (("e0", "n0"), ("e1", "n1"))]
            r_stack = np.stack([fk_pose(e, mode)[:3, :3] for e, _ in fit_snaps])
            c_stack = np.stack([n for _, n in fit_snaps])

            # Board normal in the wrist frame, R-free: a rigid R preserves the
            # angles between normals, so pair angles alone pin n_w.
            cands = fib_sphere(4000)
            m_all = np.einsum("pij,cj->cpi", r_stack, cands)
            g_model = np.arccos(np.clip(m_all @ m_all.transpose(0, 2, 1), -1, 1))
            g_cam = np.arccos(np.clip(c_stack @ c_stack.T, -1, 1))
            upper = np.triu(np.ones_like(g_cam, dtype=bool), 1)
            cost = (((g_model - g_cam[None]) ** 2) * upper[None]).sum(axis=(1, 2))
            n_w = cands[int(np.argmin(cost))]
            rms = float(np.degrees(np.sqrt(cost.min() / upper.sum())))
            print(
                f"\n[{mode}] board normal in wrist frame {np.round(n_w, 3)}, pair-angle rms {rms:.2f} deg",
                flush=True,
            )

            # R by Kabsch on (model normal, camera normal) pairs + the pan
            # translation anchor (the channel both proofs agreed on). n_w sign
            # is free; keep whichever signs in better.
            t_cam_dir = records[0]["m"] / np.linalg.norm(records[0]["m"])
            tb0 = fk_pose(records[0]["e0"], mode)[:3, 3]
            tb1 = fk_pose(records[0]["e1"], mode)[:3, 3]
            t_base_dir = (tb1 - tb0) / max(np.linalg.norm(tb1 - tb0), 1e-12)
            best = None
            for sgn in (1.0, -1.0):
                m_rows = np.concatenate([(r_stack @ (sgn * n_w)), t_base_dir[None]], axis=0)
                c_rows = np.concatenate([c_stack, t_cam_dir[None]], axis=0)
                r_c = kabsch_rotation(m_rows, c_rows)
                resid = np.mean([ang(r_c @ m_rows[i], c_rows[i]) for i in range(len(m_rows))])
                if best is None or resid < best[0]:
                    best = (resid, r_c, sgn)
            fit_resid, r_fit, sgn = best
            n_w = sgn * n_w
            print(f"  [{mode}] R fit residual {fit_resid:.1f} deg (normals + pan anchor)", flush=True)
            results["modes"][mode] = {
                "r_cam_from_base": r_fit.tolist(),
                "n_w": n_w.tolist(),
                "fit_resid_deg": fit_resid,
                "nw_pair_rms_deg": rms,
            }

            for k, rec in enumerate(records):
                if rec is None:
                    continue
                rw0 = fk_pose(rec["e0"], mode)[:3, :3]
                rw1 = fk_pose(rec["e1"], mode)[:3, :3]
                model_swing = ang(rw0 @ n_w, rw1 @ n_w)
                pred_after = r_fit @ (rw1 @ n_w)
                err = ang(pred_after, rec["n1"])
                tag = "FIT" if k < 3 else "HELD-OUT"
                print(
                    f"  [{mode}] move {k} ({tag}): swing cam {rec['swing']:5.2f} vs model {model_swing:5.2f} deg "
                    f"(ratio {rec['swing'] / max(model_swing, 1e-9):.2f}), predicted-normal err {err:5.1f} deg",
                    flush=True,
                )

            # Stage 2: with R fixed, the board's wrist-frame offset is linear in the
            # position records: m_i = R (Ra_i - Rb_i) o + R (ta_i - tb_i).
            a_rows, b_rows = [], []
            for rec in records[:3]:
                t0 = fk_pose(rec["e0"], mode)
                t1 = fk_pose(rec["e1"], mode)
                a_rows.append(r_fit @ (t1[:3, :3] - t0[:3, :3]))
                b_rows.append(rec["m"] - r_fit @ (t1[:3, 3] - t0[:3, 3]))
            a = np.concatenate(a_rows, axis=0)
            b_vec = np.concatenate(b_rows, axis=0)
            offset, *_ = np.linalg.lstsq(a, b_vec, rcond=None)
            print(f"  [{mode}] board offset in wrist frame: {np.round(offset * 1000, 1)} mm", flush=True)
            for k, rec in enumerate(records):
                if rec is None:
                    continue
                t0 = fk_pose(rec["e0"], mode)
                t1 = fk_pose(rec["e1"], mode)
                pred_m = r_fit @ ((t1[:3, :3] - t0[:3, :3]) @ offset + (t1[:3, 3] - t0[:3, 3]))
                meas = rec["m"]
                cosang = float(
                    np.clip(pred_m @ meas / (np.linalg.norm(pred_m) * np.linalg.norm(meas) + 1e-12), -1, 1)
                )
                pos_ang = float(np.degrees(np.arccos(cosang)))
                tag = "FIT" if k < 3 else "HELD-OUT"
                print(
                    f"  [{mode}] move {k} ({tag}): POSITION angle {pos_ang:5.1f} deg, "
                    f"|pred| {np.linalg.norm(pred_m) * 1000:5.1f} vs |meas| "
                    f"{np.linalg.norm(meas) * 1000:5.1f} mm",
                    flush=True,
                )

        out_path = pathlib.Path("outputs/handeye_result.json")
        out_path.write_text(json.dumps(results, indent=2))
        print(f"result persisted to {out_path}", flush=True)
    finally:
        robot.disconnect()
        print("disconnected (torque holds)", flush=True)


if __name__ == "__main__":
    main()
