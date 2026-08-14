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

  --mode servo   The full v3 servo. A spawn-context subprocess owns SAM3 +
                 DINOv3 (teach from the capture session's photos, then measure
                 every frame it is handed: target fit with the sliver gate and
                 recruit fallback, crop-first held fit, 3D error, annotated
                 frame). The control loop ticks at a fixed rate and streams
                 toward a COORDINATED multi-joint IK goal: each perception
                 update becomes a small camera-space step, rotated into the
                 base by the calibrated hand-eye map (outputs/
                 handeye_result.json, the xyz-sweep R) and solved by the
                 production IK, seed-walked at held orientation — no probes,
                 no per-joint pushes. A decaying Broyden residual on the
                 camera-from-base map absorbs model error. DONE under 4 mm
                 sustained; honest halts otherwise. Runs start by placing the
                 arm at the campaign hover via the rest interpolator.

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
# v3 servo: IK decides the joints — the full arm participates (gripper excluded).
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "forearm_roll", "wrist_flex", "wrist_roll")
MOTOR_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
# The parked right arm carries an IDENTICAL wrist PCB right of this column; the
# held ("arm end") designation searches only left of it — with the full frame,
# SAM3 provably jumps to the twin when the left PCB tilts oblique.
HELD_ROI_X_MAX = 520  # hardcode-ok: rig-specific single-camera bench
# The campaign hover: every validated designation and the hand-eye calibration
# lived here; servo runs start from it via the rest interpolator.
HOVER = {  # hardcode-ok: rig-specific single-camera bench
    "shoulder_pan.pos": 15.1,
    "shoulder_lift.pos": -14.0,
    "elbow_flex.pos": 56.0,
    "forearm_roll.pos": 29.4,
    "wrist_flex.pos": 3.9,
    "wrist_roll.pos": -17.3,
    "gripper.pos": 95.6,
}


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

    intr, scenes = load_captures(_pathlib.Path(captures))
    target_designator = Designator("sam3", concept, "cuda")
    held_designator = Designator("sam3", held_concept, "cuda")
    tier = DinoTier("facebook/dinov3-vits16-pretrain-lvd1689m", device="cuda")
    pairs = teach_pairs(scenes, teach, target_designator, held_designator, tier, intr)
    recruits = Recruits(tier, intr)
    t_inl_recent: list[int] = []
    demo_locked: int | None = None
    conn.send({"ready": True, "demos": len(pairs)})

    while True:
        msg = conn.recv()
        if msg is None:
            return
        frame = _LiveFrame(msg["rgb"], msg["depth"])
        mask_t = target_designator.mask(frame)
        t_fit, t_uv, demo = None, None, 0
        # The demos are alternative VIEWS of one goal, not alternating goals:
        # re-picking per frame strobed the goal point +-20 mm and the servo
        # chased it. Lock the demo on SERVO entry; release outside SERVO.
        if msg.get("state") != "SERVO":
            demo_locked = None
        candidates = [demo_locked] if demo_locked is not None else list(range(len(pairs)))
        if mask_t is not None:
            for d in candidates:
                pair = pairs[d]
                fit, uv = bind_rigid3d(pair.target, frame, mask_t, tier, intr)
                if fit is not None and (t_fit is None or fit.n_inliers >= t_fit.n_inliers):
                    t_fit, t_uv, demo = fit, uv, d
        if msg.get("state") == "SERVO" and demo_locked is None and t_fit is not None:
            demo_locked = demo
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
        # Crop-first held designation: SAM3 sees only the left workspace, so it
        # finds the left wrist PCB or nothing — the twin-arm jump is impossible
        # by construction, and refusal stays honest.
        sub = _LiveFrame(frame.rgb[:, :HELD_ROI_X_MAX], frame.depth[:, :HELD_ROI_X_MAX])
        sub_mask = held_designator.mask(sub)
        mask_h = None
        if sub_mask is not None:
            mask_h = np.zeros(frame.rgb.shape[:2], dtype=bool)
            mask_h[:, :HELD_ROI_X_MAX] = sub_mask
        h_fit = bind_rigid3d(pair.held, frame, mask_h, tier, intr)[0] if mask_h is not None else None
        # Translation-only goal composition: the ring is rotationally
        # symmetric, so its fitted rotation is arbitrary and ALTERNATES
        # between registration basins (t-inliers 90<->20) — composing the
        # taught offset through it strobed the goal +-40 mm. Centroids are
        # basin-stable (every basin covers the same physical ring): the ring
        # moved by delta-centroid since teaching, so the goal moves by
        # delta-centroid, no target rotation consulted.
        e_t_vec = None
        if t_fit is not None and h_fit is not None:
            t_now_c = t_fit.transform.apply(pair.target.xyz).mean(axis=0)
            goal_c = pair.held.xyz.mean(axis=0) + (t_now_c - pair.target.xyz.mean(axis=0))
            e_t_vec = goal_c - held_centroid(h_fit, pair.held)
        measured = e_t_vec is not None

        vis = frame.rgb.copy()
        extra = ""
        if t_fit is not None and h_fit is not None:
            extra = f"inl t{t_fit.n_inliers}/h{h_fit.n_inliers} demo {demo}"
            if e_t_vec is not None:
                extra += f" |e| {float(np.linalg.norm(e_t_vec)) * 1000:.0f}mm"
        annotate(vis, mask_t, mask_h, t_fit, t_uv, h_fit, pair, intr, msg.get("state", "?"), None, extra)
        _ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
        conn.send(
            {
                "measured": measured,
                "e_t": e_t_vec.tolist() if measured else None,
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


def servo(
    robot, perception: Perception, hz: float, out_dir: pathlib.Path, goal_shift_m: np.ndarray | None = None
) -> None:
    """v3 — the IK+residual servo. Both ends stay camera-measured; each
    perception update becomes a small camera-space step, rotated into the base
    frame by the calibrated hand-eye map (the xyz-sweep R) and turned into a
    COORDINATED multi-joint reference by the production IK, seed-walked with
    the orientation held. A Broyden residual on the camera-from-base map
    absorbs what R and the model get wrong; the decisive streaming ticks and
    the press/lead safety machinery are unchanged from the proven loop. No
    probes: the model supplies the direction field, the camera the truth.
    States: WAIT -> SERVO -> DONE/HALTED."""
    from lerobot.robots.so107_description.cartesian_ik import make_so107_arm_kinematics
    from lerobot.robots.so107_description.joint_alignment import LEFT_ARM_ALIGNMENT
    from lerobot.showservo.servo import ConvergenceCertificate

    out_dir.mkdir(parents=True, exist_ok=True)
    handeye = json.loads(pathlib.Path("outputs/handeye_result.json").read_text())["xyz_fit"]
    r_cam = np.array(handeye["r_cam_from_base"])  # cam ~= R @ base + t
    d_res = np.zeros((3, 3))  # Broyden residual on the cam-from-base map
    cert = ConvergenceCertificate(window=25, min_improvement=0.05)
    kin = make_so107_arm_kinematics(LEFT_ARM_ALIGNMENT)
    cal = robot.calibration
    dpn = np.array([(cal[m].range_max - cal[m].range_min) / 200.0 * (360.0 / 4096.0) for m in MOTOR_ORDER])

    obs = robot.get_observation()
    check_depth(obs)
    home = joint_positions(obs)
    ref = {j: home[j] for j in ARM_JOINTS}
    jref = dict(ref)  # the IK-produced goal the streaming chases
    q_home = np.array([home[m] for m in MOTOR_ORDER]) * dpn
    r_hold = kin.forward_kinematics(q_home)[:3, :3].copy()
    print("home:", {j: round(home[j], 2) for j in ARM_JOINTS}, flush=True)

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
    press_ticks = dict.fromkeys(ARM_JOINTS, 0)
    press_start_enc = dict.fromkeys(ARM_JOINTS, 0.0)
    press_fail = dict.fromkeys(ARM_JOINTS, 0)

    dt = 1.0 / hz
    # 20 u/s, not 3: the rest-position replay lifts this arm ~147 units through
    # the same motors and gains with nothing but a DECISIVE 25-50 u/s reference
    # sweep — break static friction once, ride kinetic friction with momentum.
    # A 3 u/s creep re-grips static friction at every micro-step; presses built
    # error while stationary, the worst possible regime. The asymptotic error
    # scaling still slows the final approach.
    tick_step = 20.0 / hz
    state = "WAIT"
    ready_streak = 0
    done_streak = 0
    stale = 0
    tip_last: np.ndarray | None = None
    prev_centroid: np.ndarray | None = None
    e_t = np.zeros(3)
    frame_i = 0
    halt = ""

    def ik_goal(q_now: np.ndarray, b_step: np.ndarray) -> tuple[np.ndarray | None, str]:
        """Seed-walk the production solver to tip+b_step at held orientation;
        (None, reason) when the solution leaves the trusted envelope (skip the
        update rather than chase a wild branch)."""
        target = np.eye(4)
        target[:3, :3] = r_hold
        p_start = kin.forward_kinematics(q_now)[:3, 3]
        target[:3, 3] = p_start + b_step
        q_sol = q_now.copy()
        n_sub = max(1, int(np.ceil(np.linalg.norm(b_step) / 0.010)))
        for s in range(1, n_sub + 1):
            t_sub = target.copy()
            t_sub[:3, 3] = p_start + (s / n_sub) * b_step
            q_prev = q_sol.copy()
            try:
                q_sol = kin.inverse_kinematics(q_sol, t_sub)
            except Exception as e:
                return None, f"IK exception {type(e).__name__}: {e}"
            step = float(np.max(np.abs(q_sol - q_prev)))
            if step > 20.0:
                j = MOTOR_ORDER[int(np.argmax(np.abs(q_sol - q_prev)))]
                return None, f"branch flip {step:.1f} deg on {j}"
        enc_sol = q_sol / dpn
        # Wild-branch protection is about the STEP, not the journey: three
        # runs died at identical encoders because a 45-deg home rail fenced
        # off the task's legitimate excursion. Per-update step stays tight;
        # the absolute rail is a hard-safety backstop only.
        step_dev = float(np.max(np.abs(q_sol - q_now)))
        if step_dev > 25.0:
            j = MOTOR_ORDER[int(np.argmax(np.abs(q_sol - q_now)))]
            return None, f"step {step_dev:.1f} deg on {j}"
        dev = float(np.max(np.abs(q_sol - q_home)))
        if dev > 75.0:
            j = MOTOR_ORDER[int(np.argmax(np.abs(q_sol - q_home)))]
            return None, f"deviation {dev:.1f} deg on {j}"
        if float(np.max(np.abs(enc_sol[:6]))) > 95.0:
            return None, f"enc range {float(np.max(np.abs(enc_sol[:6]))):.1f}"
        return enc_sol, "ok"

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
                if goal_shift_m is not None:
                    # VALIDATION MODE: the goal is displaced by a fixed
                    # camera-space shift (e.g. into the reach sphere when the
                    # taught target sits beyond it). The relation direction is
                    # preserved; the run validates transit + final approach +
                    # certificate, NOT the taught task itself.
                    e_t = e_t + goal_shift_m
                e_norm_mm = float(np.linalg.norm(e_t)) * 1000.0
                centroid = np.asarray(upd["centroid"])
                if state == "WAIT":
                    ready_streak += 1
                    if ready_streak >= 3:
                        state = "SERVO"
                        prev_centroid = centroid
                        cert.reset()
                        print(f"p{frame_i}: WAIT -> SERVO |e| {e_norm_mm:.1f} mm", flush=True)
                elif state == "SERVO":
                    cert.update(float(np.linalg.norm(e_t)))
                    q_now = np.array([enc[m] for m in MOTOR_ORDER]) * dpn
                    tip_now = kin.forward_kinematics(q_now)[:3, 3]
                    # Broyden residual pairs EXECUTED base motion (FK of
                    # encoders) with the camera's measured motion. The anchors
                    # ACCUMULATE across cycles and only a summed displacement
                    # past the floor teaches: per-frame pairs sat below any
                    # safe floor at this arm's ~3 mm/cycle (residual never
                    # learned), while a tiny denominator once slammed
                    # ||d_res|| past 1 and the pinv commanded a 195 mm
                    # "10 mm" step. Decay keeps R the prior.
                    if tip_last is None or prev_centroid is None:
                        tip_last = tip_now
                        prev_centroid = centroid
                    else:
                        db = tip_now - tip_last
                        dc = centroid - prev_centroid
                        if float(np.linalg.norm(db)) > 0.012:
                            m_eff = r_cam + d_res
                            b_upd = np.outer(dc - m_eff @ db, db) / float(db @ db)
                            b_upd_norm = float(np.linalg.norm(b_upd))
                            if b_upd_norm > 0.2:
                                b_upd *= 0.2 / b_upd_norm
                            d_res += b_upd
                            d_res *= 0.95
                            res_norm = float(np.linalg.norm(d_res))
                            if res_norm > 0.5:  # a correction, never a replacement
                                d_res *= 0.5 / res_norm
                            tip_last = tip_now  # re-anchor only on an accepted lesson
                            prev_centroid = centroid
                    # Camera-space step -> base step -> coordinated IK goal.
                    u = e_t * 0.6
                    u *= min(1.0, e_norm_mm / 15.0)  # asymptotic final approach
                    nu = float(np.linalg.norm(u))
                    if nu > 0.010:
                        u *= 0.010 / nu
                    b_step = np.linalg.pinv(r_cam + d_res) @ u
                    nb = float(np.linalg.norm(b_step))
                    if nb > 0.012:  # no map conditioning may command more than intended
                        b_step *= 0.012 / nb
                    enc_sol, why = ik_goal(q_now, b_step)
                    if enc_sol is not None:
                        for i, m in enumerate(MOTOR_ORDER):
                            if m in jref:
                                jref[m] = float(enc_sol[i])
                    else:
                        print(
                            f"p{frame_i}: IK goal refused ({why}) — holding; "
                            f"e_t {np.round(e_t * 1000, 1)} mm, b_step {np.round(b_step * 1000, 1)} mm",
                            flush=True,
                        )
                    if e_norm_mm < 4.0:
                        done_streak += 1
                        if done_streak >= 3:
                            state = "DONE"
                            print(f"p{frame_i}: SERVO -> DONE |e| {e_norm_mm:.1f} mm", flush=True)
                    else:
                        done_streak = 0
                    if state == "SERVO" and not cert.progressing:
                        state, halt = "HALTED", "no progress over the window"
                    if state == "SERVO":
                        # The goal point in camera space: if it wanders while the
                        # target sits still, the goal itself is unstable (e.g.
                        # symmetric-target rotation ambiguity), not the control.
                        goal_cam = centroid + e_t
                        print(
                            f"p{frame_i}: SERVO |e| {e_norm_mm:6.1f} mm  "
                            f"goal {np.round(goal_cam * 1000, 1)}  "
                            f"inl t{upd['t_inliers']}/h{upd['h_inliers']}  "
                            f"|d_res| {float(np.linalg.norm(d_res)):.3f}",
                            flush=True,
                        )
            else:
                ready_streak = 0
                stale += 1
                if state == "SERVO" and stale >= 20:
                    state, halt = "HALTED", "20 unmeasured perception frames"

        # --- per-tick actuation: stream toward the IK goal ---------------------
        if state == "SERVO":
            for j in ARM_JOINTS:
                d = jref[j] - ref[j]
                proposed = ref[j] + float(np.clip(d, -tick_step, tick_step))
                lead = proposed - enc[j]
                if abs(lead) > lead_max:
                    proposed = enc[j] + lead_max * (1.0 if lead > 0 else -1.0)
                if abs(proposed - home[j]) < 75.0:  # hard-safety excursion rail
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
                            jref[j] = ref[j]
                            print(
                                f"press on {j} released without breakaway ({press_fail[j]}/3)",
                                flush=True,
                            )
                            if press_fail[j] >= 3:
                                state, halt = "HALTED", f"{j}: three presses without breakaway"
                        press_ticks[j] = 0
                else:
                    press_ticks[j] = 0

        robot.send_action({f"{j}.pos": ref[j] for j in ARM_JOINTS})

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
        "--goal-shift-mm",
        type=float,
        nargs=3,
        default=None,
        help="VALIDATION ONLY: displace the goal by this camera-frame shift (mm); "
        "the run then validates the loop, not the taught task.",
    )
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
            from lerobot.robots.rest_position import move_to_rest_position

            move_to_rest_position(robot, HOVER, duration_s=5.0)
            shift = None
            if args.goal_shift_mm is not None:
                shift = np.array(args.goal_shift_mm) / 1000.0
                print(f"VALIDATION GOAL: shifted by {args.goal_shift_mm} mm (camera frame)", flush=True)
            servo(robot, perception, args.hz, args.out.parent / "m1_servo", goal_shift_m=shift)
    finally:
        if args.mode == "servo":
            perception.stop()
        robot.disconnect()
        print("disconnected (torque holds)", flush=True)


if __name__ == "__main__":
    main()
