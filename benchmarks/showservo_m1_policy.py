# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""M1 v2 — the servo as a policy in the STANDARD robot control loop.

v1 drove the arm through a bespoke HTTP command path and rebuilt — badly — what
the robot stack already does: continuous references, read-act cycles, proven
gains, a single camera owner. v2 is the corrective: a single-arm robot profile
(one SO-107 + the RealSense with RAW aligned depth exposed, no overlay
processor), and a control loop of the standard shape — get_observation ->
act -> send_action at a fixed rate — with the servo living where a policy
lives. Perception (SAM3 + DINOv3) runs slower than the loop; the loop streams
smoothly toward the latest goal the whole time, exactly as teleop streams.

Staged bring-up, on the real robot at every step (user's directive):

  --mode sweep   Stage 1: command each servo joint back and forward through a
                 gentle triangle wave via the true control loop, logging
                 reference-vs-encoder tracking and saving camera frames at the
                 extremes. This is the decisive test that continuous references
                 cross the backlash flanks that burst commands could not.

  --mode servo   Stages 2+3: the full servo. A spawn-context subprocess owns
                 SAM3 + DINOv3 (teach from the capture session's photos, then
                 measure every frame it is handed: target fit with the sliver
                 gate and recruit fallback, held fit, 3D error, annotated
                 frame). The control loop ticks at a fixed rate, streams the
                 joint reference a small step toward the current goal every
                 tick, and refreshes the goal whenever the worker returns —
                 perception at its own ~2.5 Hz, actuation continuous. The
                 probe is a per-joint triangle measured by the camera, its
                 Jacobian column a central difference across both backlash
                 flanks. DONE under 4 mm sustained; honest halts otherwise.

Usage:
    PYTHONPATH=src python benchmarks/showservo_m1_policy.py \\
        --profile m1_left --mode servo --captures captures/<session> \\
        --concept "green ring" --held-concept "circuit board" --teach 0 1
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

M1_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex")


def make_robot(profile_name: str):
    """Build the single-arm robot from a GUI robot profile. Pre: profile type is
    so107_follower; the arm is calibrated (interactive calibration would hang)."""
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    from lerobot.robots.so_follower import SO107Follower, SO107FollowerConfig

    path = pathlib.Path.home() / ".config" / "lerobot" / "robots" / f"{profile_name}.json"
    profile = json.loads(path.read_text())
    assert profile.get("type") == "so107_follower", f"profile {profile_name!r} is not a single SO-107"
    fields = profile.get("fields", {})
    cameras = {}
    for key, cam in profile.get("cameras", {}).items():
        assert cam.get("type") == "intelrealsense", "M1 v2 expects the RealSense"
        cameras[key] = RealSenseCameraConfig(
            serial_number_or_name=str(cam["serial_number_or_name"]),
            width=int(cam["width"]),
            height=int(cam["height"]),
            fps=int(cam["fps"]),
            use_depth=bool(cam.get("use_depth", True)),
            enable_decimation=bool(cam.get("enable_decimation", False)),
            enable_hole_filling=bool(cam.get("enable_hole_filling", False)),
        )
    cfg = SO107FollowerConfig(
        id=str(fields["id"]),
        port=str(fields["port"]),
        use_degrees=bool(fields.get("use_degrees", False)),
        disable_torque_on_disconnect=bool(fields.get("disable_torque_on_disconnect", False)),
        max_relative_target=float(fields.get("max_relative_target", 4.0)),
        cameras=cameras,
    )
    robot = SO107Follower(cfg)
    robot.connect(calibrate=False)
    assert robot.is_calibrated, "arm uncalibrated — calibrate from the robot flow first"
    return robot


def joint_positions(obs: dict) -> dict[str, float]:
    return {k.removesuffix(".pos"): float(v) for k, v in obs.items() if k.endswith(".pos")}


def check_depth(obs: dict, cam_key: str = "top") -> None:
    """Print what the observation actually carries — the v1 gap made this a
    load-bearing check, not a formality."""
    rgb = obs.get(cam_key)
    depth = obs.get(f"{cam_key}_depth")
    assert rgb is not None, f"no {cam_key!r} image in the observation"
    assert depth is not None, f"no {cam_key}_depth in the observation — the depth path is broken again"
    d = np.asarray(depth)
    valid = d > 0
    print(
        f"observation: {cam_key} {np.asarray(rgb).shape}, depth {d.shape} dtype {d.dtype}, "
        f"valid {float(valid.mean()):.2f}, median {float(np.median(d[valid])) if valid.any() else 0:.0f}",
        flush=True,
    )


def sweep(robot, hz: float, amp: float, out_dir: pathlib.Path) -> None:
    """Back and forward on each servo joint through the true control loop.

    A triangle reference at ~3 units/s — the quasi-continuous shape teleop
    presents. Logs reference-vs-encoder tracking; saves a camera frame at each
    extreme so the CAMERA (not the encoder) certifies that the link moved.
    """
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    obs = robot.get_observation()
    check_depth(obs)
    home = joint_positions(obs)
    print("home:", {j: round(home[j], 2) for j in M1_JOINTS}, flush=True)

    dt = 1.0 / hz
    step = 3.0 / hz  # units per tick ~= 3 units/s
    for joint in M1_JOINTS:
        # Triangle: home -> +amp -> home -> -amp -> home, twice.
        waypoints = [home[joint] + amp, home[joint], home[joint] - amp, home[joint]] * 2
        ref = home[joint]
        worst = 0.0
        seen_lo, seen_hi = home[joint], home[joint]
        for w_i, target in enumerate(waypoints):
            while abs(ref - target) > 1e-9:
                ref += np.clip(target - ref, -step, step)
                action = {f"{j}.pos": home[j] for j in M1_JOINTS}
                action[f"{joint}.pos"] = ref
                robot.send_action(action)
                time.sleep(dt)
            time.sleep(0.4)  # settle at the extreme before judging tracking
            obs = robot.get_observation()
            enc = joint_positions(obs)[joint]
            worst = max(worst, abs(enc - ref))
            seen_lo, seen_hi = min(seen_lo, enc), max(seen_hi, enc)
            if w_i < 4:  # first cycle: save the camera's view at each waypoint
                frame = cv2.cvtColor(np.asarray(obs["top"]), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_dir / f"{joint}_w{w_i}.jpg"), frame)
            print(
                f"{joint}: waypoint {target - home[joint]:+.1f}u  ref {ref:+7.2f}  "
                f"encoder {enc:+7.2f}  err {abs(enc - ref):4.2f}",
                flush=True,
            )
        print(
            f"{joint}: RESULT worst tracking err {worst:.2f}u, encoder swept "
            f"[{seen_lo:+.2f} .. {seen_hi:+.2f}] (commanded +/-{amp}u)",
            flush=True,
        )


# --- stages 2+3: perception worker + streaming servo policy ---------------------


def _perception_main(conn, captures: str, concept: str, held_concept: str, teach: list[int]) -> None:
    """Subprocess entry (spawn context — CUDA cannot survive a fork). Teaches from
    the capture session, then answers every (rgb, depth) with a measurement dict
    and an annotated frame. The parent enforces backpressure by sending one frame
    at a time."""
    import pathlib as _pathlib
    import sys as _sys

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
    import cv2
    from showservo_m0 import DinoTier
    from showservo_m1 import annotate, held_centroid, teach_pairs
    from showservo_real import Designator, Recruits, _LiveFrame, bind_rigid3d, load_captures

    from lerobot.showservo.servo import servo_error_3d

    intr, scenes = load_captures(_pathlib.Path(captures))
    target_designator = Designator("sam3", concept, "cuda")
    held_designator = Designator("sam3", held_concept, "cuda")
    tier = DinoTier("facebook/dinov3-vits16-pretrain-lvd1689m", device="cuda")
    pairs = teach_pairs(scenes, teach, target_designator, held_designator, tier, intr)
    recruits = Recruits(tier, intr)
    t_inl_recent: list[int] = []
    conn.send({"ready": True, "demos": len(pairs)})

    while True:
        msg = conn.recv()
        if msg is None:
            return
        frame = _LiveFrame(msg["rgb"], msg["depth"])
        mask_t = target_designator.mask(frame)
        t_fit, t_uv, demo = None, None, 0
        if mask_t is not None:
            for d, pair in enumerate(pairs):
                fit, uv = bind_rigid3d(pair.target, frame, mask_t, tier, intr)
                if fit is not None and (t_fit is None or fit.n_inliers >= t_fit.n_inliers):
                    t_fit, t_uv, demo = fit, uv, d
        if t_fit is not None and t_inl_recent:
            floor = max(12.0, 0.25 * float(np.median(t_inl_recent)))
            if t_fit.n_inliers < floor:
                t_fit, t_uv = None, None  # sliver imposter: the recruits carry
        if t_fit is not None:
            t_inl_recent.append(t_fit.n_inliers)
            del t_inl_recent[:-8]
            recruits.refresh(frame, mask_t, t_fit, demo)
        else:
            rfit, _uv, _in = recruits.fallback(frame)
            if rfit is not None:
                t_fit, demo = rfit, recruits.anchor_demo
        pair = pairs[demo]
        mask_h = held_designator.mask(frame)
        h_fit = bind_rigid3d(pair.held, frame, mask_h, tier, intr)[0] if mask_h is not None else None
        err = servo_error_3d(pair.held.xyz, t_fit, h_fit) if (t_fit and h_fit) else None
        measured = err is not None and err.ok

        vis = frame.rgb.copy()
        extra = f"inl t{t_fit.n_inliers}/h{h_fit.n_inliers} demo {demo}" if (t_fit and h_fit) else ""
        annotate(vis, mask_t, mask_h, t_fit, t_uv, h_fit, pair, intr, msg.get("state", "?"), err, extra)
        _ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
        conn.send(
            {
                "measured": measured,
                "e_t": err.e_t.tolist() if measured else None,
                "centroid": held_centroid(h_fit, pair.held).tolist() if h_fit is not None else None,
                "t_inliers": int(t_fit.n_inliers) if t_fit is not None else 0,
                "h_inliers": int(h_fit.n_inliers) if h_fit is not None else 0,
                "jpg": jpg.tobytes(),
            }
        )


class Perception:
    """The slow half, at arm's length. ``submit`` hands the worker one frame when
    idle; ``latest`` is the newest completed measurement. One frame in flight ever."""

    def __init__(self, captures: str, concept: str, held_concept: str, teach: list[int]):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(
            target=_perception_main, args=(child, captures, concept, held_concept, teach), daemon=True
        )
        self.proc.start()
        self.busy = False
        self.latest: dict | None = None
        self.updates = 0

    def wait_ready(self, timeout: float = 300.0) -> dict:
        assert self.conn.poll(timeout), "perception worker did not come up"
        msg = self.conn.recv()
        assert msg.get("ready"), f"perception worker failed: {msg}"
        return msg

    def submit(self, rgb: np.ndarray, depth_m: np.ndarray, state: str) -> None:
        if not self.busy:
            self.conn.send({"rgb": rgb, "depth": depth_m, "state": state})
            self.busy = True

    def poll(self) -> dict | None:
        """Post: the newest measurement if one just arrived, else None."""
        if self.busy and self.conn.poll(0):
            self.latest = self.conn.recv()
            self.busy = False
            self.updates += 1
            return self.latest
        return None

    def stop(self) -> None:
        import contextlib

        with contextlib.suppress(BrokenPipeError, OSError):
            self.conn.send(None)
        self.proc.join(timeout=5)
        if self.proc.is_alive():
            self.proc.terminate()


def servo(robot, perception: Perception, hz: float, out_dir: pathlib.Path) -> None:
    """The policy loop: stream a small step toward the current joint reference
    every tick; update the goal whenever perception returns. States as in v1
    (WAIT -> PROBE -> SERVO -> DONE/HALTED) but actuation is continuous — the
    property the stage-1 sweep proved this arm needs."""
    from lerobot.showservo.servo import ConvergenceCertificate, JacobianEstimator, PIController

    out_dir.mkdir(parents=True, exist_ok=True)
    est = JacobianEstimator(n_joints=len(M1_JOINTS), m=3, damping=1e-3)
    pi = PIController(kp=0.6, ki=0.2, v_max=0.012, integral_limit=0.03)
    cert = ConvergenceCertificate(window=25, min_improvement=0.05)

    obs = robot.get_observation()
    check_depth(obs)
    home = joint_positions(obs)
    ref = {j: home[j] for j in M1_JOINTS}
    print("home:", {j: round(home[j], 2) for j in M1_JOINTS}, flush=True)

    # Reference-lead bursts: the gravity-loaded shoulder needs teleop-transient
    # torque (P x tens of units of tracking error) to break away at extended
    # poses; a polite 1-2 unit lead sits below breakaway forever. The reference
    # may lead the ENCODER by up to lead_max; a press that produces no encoder
    # progress within press_ticks_max releases (no stall-dwell — that is what cooked
    # the motor in v1), and three failed presses on a joint is an honest halt.
    # 12, not 7: two presses at 7 units moved the loaded lift < 0.3u — its
    # breakaway threshold at full extension sits above P x 7. Teleop's transient
    # errors reach tens of units; 12 approaches that scale while the 3-second
    # progress-gated release keeps each press a burst, never a dwell.
    lead_max = 12.0
    press_lead = 6.0
    press_ticks_max = int(3.0 * hz)
    press_ticks = dict.fromkeys(M1_JOINTS, 0)
    press_start_enc = dict.fromkeys(M1_JOINTS, 0.0)
    press_fail = dict.fromkeys(M1_JOINTS, 0)

    dt = 1.0 / hz
    tick_step = 3.0 / hz  # units per tick toward the target reference (~3 u/s)
    state = "WAIT"
    ready_streak = 0
    done_streak = 0
    stale = 0
    # Probe bookkeeping: per joint, a +amp then -amp triangle, measured by the
    # camera at each extreme; the column is the central difference across both
    # backlash flanks.
    probe_joint = 0
    probe_phase = 0  # 0: to +amp, 1: measure; 2: to -amp, 3: measure; 4: home
    # +/-5, not +/-2.4: the lift's ~4-unit backlash swallowed most of a 2.4-unit
    # triangle (0.9-1.2 mm responses), seeding columns too weak and too noisy to
    # aim by — the first v2 run then crawled flat at 66 mm until the certificate
    # called it. A 5-unit swing crosses the slack and measures the true gain.
    probe_amp = 5.0
    probe_target: float | None = None
    probe_settle_ticks = 0
    probe_plus: np.ndarray | None = None
    probe_minus: np.ndarray | None = None
    probe_enc_plus = 0.0
    probe_cols: list[np.ndarray] = []
    enc_at_last: np.ndarray | None = None
    prev_centroid: np.ndarray | None = None
    e_t = np.zeros(3)
    frame_i = 0
    halt = ""

    while True:
        t0 = time.perf_counter()
        obs = robot.get_observation()
        enc = joint_positions(obs)
        depth_m = np.asarray(obs["top_depth"], dtype=np.float32) / 1000.0
        perception.submit(np.asarray(obs["top"]), depth_m, state)
        upd = perception.poll()
        if upd is not None:
            frame_i += 1
            (out_dir / f"frame_{frame_i:05d}.jpg").write_bytes(upd["jpg"])
            for old in sorted(out_dir.glob("frame_*.jpg"))[:-400]:
                old.unlink()
            measured = upd["measured"]
            stale = 0
            if measured:
                e_t = np.asarray(upd["e_t"])
                e_norm_mm = float(np.linalg.norm(e_t)) * 1000.0
                centroid = np.asarray(upd["centroid"])
                if state == "WAIT":
                    ready_streak += 1
                    if ready_streak >= 3:
                        state = "PROBE"
                        prev_centroid = centroid
                        probe_target = ref[M1_JOINTS[0]] + probe_amp
                        print(f"p{frame_i}: WAIT -> PROBE |e| {e_norm_mm:.1f} mm", flush=True)
                elif state == "PROBE" and probe_settle_ticks <= 0 and probe_target is None:
                    joint = M1_JOINTS[probe_joint]
                    de = centroid - prev_centroid if prev_centroid is not None else np.zeros(3)
                    if probe_phase == 1:
                        probe_plus = de
                        probe_enc_plus = enc[joint]
                        prev_centroid = centroid
                        probe_target = home[joint] - probe_amp
                        probe_phase = 2
                    elif probe_phase == 3:
                        probe_minus = de
                        # Denominator = EXECUTED swing (encoder), not commanded:
                        # partial execution under load must scale the column up,
                        # not silently shrink the measured gain.
                        swing = probe_enc_plus - enc[joint]
                        col = (probe_plus - probe_minus) / (swing if abs(swing) > 0.5 else 2 * probe_amp)
                        probe_cols.append(col)
                        print(
                            f"p{frame_i}: probe {joint} +/-{probe_amp}u -> "
                            f"|+| {np.linalg.norm(probe_plus) * 1000:.1f} mm, "
                            f"|-| {np.linalg.norm(probe_minus) * 1000:.1f} mm "
                            f"(executed swing {swing:.1f}u)",
                            flush=True,
                        )
                        prev_centroid = centroid
                        probe_target = home[joint]  # go home, then next joint
                        probe_phase = 4
                elif state == "SERVO":
                    cert.update(float(np.linalg.norm(e_t)))
                    # Broyden pairs the camera's motion with EXECUTED joint motion
                    # (encoder deltas), not commanded reference deltas: under load
                    # the two diverge, and pairing with commands taught the last
                    # run that "this joint does nothing" — shrinking its column
                    # and self-throttling the very commands that needed to grow.
                    enc_vec = np.array([enc[j] for j in M1_JOINTS])
                    dq_exec = enc_vec - enc_at_last if enc_at_last is not None else np.zeros(3)
                    if float(dq_exec @ dq_exec) > 0.01 and prev_centroid is not None:
                        est.update(dq_exec, centroid - prev_centroid)
                    enc_at_last = enc_vec
                    prev_centroid = centroid
                    if e_norm_mm < 4.0:
                        done_streak += 1
                        if done_streak >= 3:
                            state = "DONE"
                            print(f"p{frame_i}: SERVO -> DONE |e| {e_norm_mm:.1f} mm", flush=True)
                    else:
                        done_streak = 0
                    if state == "SERVO" and not cert.progressing:
                        state, halt = "HALTED", "no progress over the window"
                    if frame_i % 5 == 0 and state == "SERVO":
                        print(
                            f"p{frame_i}: SERVO |e| {e_norm_mm:6.1f} mm  "
                            f"inl t{upd['t_inliers']}/h{upd['h_inliers']}",
                            flush=True,
                        )
            else:
                ready_streak = 0
                stale += 1
                if state == "SERVO" and stale >= 20:
                    state, halt = "HALTED", "20 unmeasured perception frames"

        # --- per-tick actuation: stream toward the current reference ------------
        if state == "PROBE":
            joint = M1_JOINTS[probe_joint]
            if probe_target is not None:
                d = probe_target - ref[joint]
                ref[joint] += float(np.clip(d, -tick_step, tick_step))
                if abs(ref[joint] - probe_target) < 1e-9:
                    probe_target = None
                    probe_settle_ticks = int(hz * 0.8)  # let motion finish, then measure
            elif probe_settle_ticks > 0:
                probe_settle_ticks -= 1
                if probe_settle_ticks == 0 and probe_phase in (0, 2):
                    probe_phase += 1  # extreme reached + settled: next perception is the measure
                elif probe_settle_ticks == 0 and probe_phase == 4:
                    probe_joint += 1
                    probe_phase = 0
                    if probe_joint >= len(M1_JOINTS):
                        cols = np.stack(probe_cols, axis=1)
                        est.seed_from_probe(np.eye(len(M1_JOINTS)), cols.T)
                        pi.reset()
                        cert.reset()
                        state = "SERVO"
                        print("J seeded (mm/unit):", flush=True)
                        for row in est.matrix * 1000.0:
                            print("   [" + "  ".join(f"{v:+6.2f}" for v in row) + "]", flush=True)
                    else:
                        probe_target = home[M1_JOINTS[probe_joint]] + probe_amp
        elif state == "SERVO":
            u = pi.step(e_t, dt=dt)
            u = u * min(1.0, float(np.linalg.norm(e_t)) / 0.015)  # asymptotic final approach
            dq = est.solve(u)
            dq = np.clip(dq, -tick_step, tick_step)
            for k, j in enumerate(M1_JOINTS):
                proposed = ref[j] + float(dq[k])
                lead = proposed - enc[j]
                if abs(lead) > lead_max:
                    proposed = enc[j] + lead_max * (1.0 if lead > 0 else -1.0)
                if abs(proposed - home[j]) < 45.0:  # excursion rail on the reference
                    ref[j] = proposed
                # Press bookkeeping: a big lead is a deliberate torque burst, but a
                # burst that moves nothing is a stall-dwell heating the motor.
                if abs(ref[j] - enc[j]) > press_lead:
                    if press_ticks[j] == 0:
                        press_start_enc[j] = enc[j]
                    press_ticks[j] += 1
                    if press_ticks[j] > press_ticks_max:
                        if abs(enc[j] - press_start_enc[j]) < 0.3:
                            press_fail[j] += 1
                            ref[j] = enc[j] + 2.0 * (1.0 if ref[j] > enc[j] else -1.0)
                            print(
                                f"press on {j} released without breakaway ({press_fail[j]}/3)",
                                flush=True,
                            )
                            if press_fail[j] >= 3:
                                state, halt = "HALTED", f"{j}: three presses without breakaway"
                        press_ticks[j] = 0
                else:
                    press_ticks[j] = 0

        robot.send_action({f"{j}.pos": ref[j] for j in M1_JOINTS})

        if state in ("DONE", "HALTED"):
            print(f"final state {state} {halt}  (encoder {enc})", flush=True)
            return
        time.sleep(max(0.0, dt - (time.perf_counter() - t0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="m1_left")
    ap.add_argument("--mode", choices=("sweep", "servo"), default="sweep")
    ap.add_argument("--captures", default="captures/session_20260814_074909")
    ap.add_argument("--concept", default="green ring")
    ap.add_argument("--held-concept", default="circuit board")
    ap.add_argument("--teach", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--hz", type=float, default=15.0)
    ap.add_argument("--amp", type=float, default=4.0)
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(
            "/tmp/claude-1000/-home-feit-Documents-lerobot-flash-dagger/"
            "4e4a1164-701c-42ec-8d41-2db356087281/scratchpad/m1_sweep"
        ),
    )
    args = ap.parse_args()

    if args.mode == "servo":
        # The worker loads models for ~30 s; bring it up BEFORE touching the robot
        # so the arm never sits torqued waiting on a model download.
        perception = Perception(args.captures, args.concept, args.held_concept, args.teach)
        info = perception.wait_ready()
        print(f"perception ready: {info['demos']} demos", flush=True)

    robot = make_robot(args.profile)
    try:
        if args.mode == "sweep":
            sweep(robot, args.hz, args.amp, args.out)
        else:
            servo(robot, perception, args.hz, args.out.parent / "m1_servo")
    finally:
        if args.mode == "servo":
            perception.stop()
        robot.disconnect()
        print("disconnected (torque holds)", flush=True)


if __name__ == "__main__":
    main()
