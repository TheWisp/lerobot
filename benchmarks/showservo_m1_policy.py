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

Later stages add the perception worker and the WAIT/PROBE/SERVO policy.

Usage:
    PYTHONPATH=src python benchmarks/showservo_m1_policy.py \\
        --profile m1_left --mode sweep
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="m1_left")
    ap.add_argument("--mode", choices=("sweep",), default="sweep")
    ap.add_argument("--hz", type=float, default=20.0)
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

    robot = make_robot(args.profile)
    try:
        if args.mode == "sweep":
            sweep(robot, args.hz, args.amp, args.out)
    finally:
        robot.disconnect()
        print("disconnected (torque holds)", flush=True)


if __name__ == "__main__":
    main()
