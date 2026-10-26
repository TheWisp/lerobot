"""Offline test of gripper-conditioned demo-to-live surface registration.

The optimizer starts from the observed arm state and masked RGB-D surfaces. It
does not receive registration hypotheses from the comparison report. Its search
variable is the grasp configuration: the demonstrated object-to-gripper relation
turns each configuration into one demo-to-live surface transform. This is a
single local solve, so failure to find a mode is evidence against that seed,
not proof that no valid registration exists.

No robot command or collision check is made here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from flash_cross_episode_trackeverything import _image, _live_masks
from flash_rgbd_registration import apply, cloud, score
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.geometry import interpolate
from lerobot.policies.flash.kinematics import SelectedJointKinematics
from lerobot.robots.so107_description.calibrated_kinematics import make_calibrated_so107_kinematics
from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES


def _downsample(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) <= count:
        return points
    return points[np.linspace(0, len(points) - 1, count).astype(int)]


def _camera_transform(
    grasp_joints: np.ndarray,
    kinematics: SelectedJointKinematics,
    demo_grasp: np.ndarray,
    base_from_camera: np.ndarray,
) -> np.ndarray:
    # M_base maps demo object points to live object points. The gripper at
    # grasp must satisfy FK(q) = M_base @ demo_grasp.
    base_remap = kinematics.forward_kinematics(grasp_joints) @ np.linalg.inv(demo_grasp)
    return np.linalg.inv(base_from_camera) @ base_remap @ base_from_camera


def _motor_bounds(kinematics: SelectedJointKinematics) -> tuple[np.ndarray, np.ndarray]:
    """Map the calibrated URDF limits into controlled motor degrees.

    This benchmark reaches through the existing adapters because they do not
    expose limits yet. A runtime implementation needs this as a public robot
    model contract, rather than duplicating joint ranges in the policy.
    """
    mapped = kinematics.inner
    pink = mapped._inner._inner
    urdf_lo = np.rad2deg(pink.q_lo)
    urdf_hi = np.rad2deg(pink.q_hi)
    a = (urdf_lo - mapped._offset) / mapped._sign
    b = (urdf_hi - mapped._offset) / mapped._sign
    return np.minimum(a, b)[kinematics._controlled_indices], np.maximum(a, b)[kinematics._controlled_indices]


def _path_check(
    grasp_joints: np.ndarray,
    start_joints: np.ndarray,
    remap_base: np.ndarray,
    intervals: list[dict],
    kinematics: SelectedJointKinematics,
    *,
    grasp_interval: int,
    connection_samples: int,
    refinement_attempts: int,
    position_tolerance_m: float,
    rotation_tolerance_deg: float,
) -> dict:
    joints = start_joints.copy()
    achieved = kinematics.forward_kinematics(joints)
    travel = np.zeros_like(joints)
    solved = 0
    grasp_achieved_joints = None
    for interval_index, interval in enumerate(intervals):
        if interval["mode"] == "connection":
            end = remap_base @ np.asarray(interval["samples"][-1]["demo_ee"], dtype=float)
            goals = [
                interpolate(achieved, end, j / connection_samples) for j in range(1, connection_samples + 1)
            ]
        elif interval["mode"] == "demonstrated":
            goals = [remap_base @ np.asarray(s["demo_ee"], dtype=float) for s in interval["samples"]]
        else:
            raise ValueError(f"unsupported mode: {interval['mode']}")
        for goal_index, goal in enumerate(goals):
            previous = joints.copy()
            try:
                for _ in range(refinement_attempts):
                    joints = kinematics.inverse_kinematics(joints, goal)
                    achieved = kinematics.forward_kinematics(joints)
                    position_error = float(np.linalg.norm(achieved[:3, 3] - goal[:3, 3]))
                    rotation_error = float(
                        np.degrees(Rotation.from_matrix(achieved[:3, :3] @ goal[:3, :3].T).magnitude())
                    )
                    if position_error <= position_tolerance_m and rotation_error <= rotation_tolerance_deg:
                        break
            except Exception as exc:
                return {
                    "solved_goals": solved,
                    "failure": {
                        "interval": interval_index,
                        "goal": goal_index + 1,
                        "reason": f"{type(exc).__name__}: {exc}",
                    },
                }
            if position_error > position_tolerance_m or rotation_error > rotation_tolerance_deg:
                return {
                    "solved_goals": solved,
                    "failure": {
                        "interval": interval_index,
                        "goal": goal_index + 1,
                        "position_error_mm": position_error * 1000,
                        "rotation_error_deg": rotation_error,
                    },
                }
            travel += np.abs(joints - previous)
            if interval_index == grasp_interval and goal_index == 0:
                grasp_achieved_joints = joints.copy()
            solved += 1
    return {
        "solved_goals": solved,
        "failure": None,
        "max_joint_travel_deg": float(travel.max()),
        "joint_travel_deg": travel.tolist(),
        "grasp_achieved_joints_deg": grasp_achieved_joints.tolist()
        if grasp_achieved_joints is not None
        else None,
        "grasp_joint_error_deg": float(np.linalg.norm(grasp_achieved_joints - grasp_joints))
        if grasp_achieved_joints is not None
        else None,
    }


def run(args: argparse.Namespace) -> dict:
    if args.grasp_interval < 0 or args.path_intervals <= args.grasp_interval:
        raise ValueError("path intervals must include the grasp interval")
    if args.sample_points < 3 or args.connection_samples < 2:
        raise ValueError("each surface needs at least three points and each connection two samples")
    if any(w < 0 for w in (*args.joint_weights, *args.observation_weights)):
        raise ValueError("cost weights must be nonnegative")
    started = time.perf_counter()
    skill = json.loads(args.skill.read_text())
    calibration = json.loads(args.calibration.read_text())
    trial_config_path = args.trial / "flash/v1/trial_config.json"
    trial_config = json.loads(trial_config_path.read_text()) if trial_config_path.exists() else {}
    with (args.trial / "flash/v1/episode_000000.jsonl").open() as telemetry:
        first_row = json.loads(telemetry.readline())
    if first_row["action_source"] != "observed_hold":
        raise ValueError("trial does not start from observed motor state")
    full_joints = np.array([first_row["effective_action"][f"{n}.pos"] for n in MOTOR_NAMES])
    arm_names = tuple(name for name in MOTOR_NAMES if name != "gripper")
    start_joints = full_joints[: len(arm_names)]
    kinematics = SelectedJointKinematics(
        make_calibrated_so107_kinematics(calibration, arm=args.arm),
        tuple(MOTOR_NAMES),
        arm_names,
        {"gripper": float(full_joints[-1])},
    )
    joint_lo, joint_hi = _motor_bounds(kinematics)
    base_from_camera = np.asarray(calibration["camera"]["T_base_cam"], dtype=float)
    demo_grasp = np.asarray(skill["intervals"][args.grasp_interval]["samples"][0]["demo_ee"], dtype=float)
    source_depth = _image(
        (args.demo_recording / "depth" / f"{args.demo_frame:06d}.png").read_bytes(), cv2.IMREAD_UNCHANGED
    )
    source_mask = _image(args.demo_mask.read_bytes(), cv2.IMREAD_GRAYSCALE) > 0
    table = pq.read_table(
        args.trial / "data/chunk-000/file-000.parquet",
        columns=["observation.images.overhead", "observation.images.overhead_depth"],
    )
    target_depth = _image(
        table["observation.images.overhead_depth"][args.live_frame].as_py()["bytes"], cv2.IMREAD_UNCHANGED
    )
    live_masks = _live_masks(args.trial, [args.live_frame], target_depth.shape, args.role)
    if args.live_frame not in live_masks:
        raise ValueError(f"no recorded mask for {args.role} in live frame {args.live_frame}")
    target_mask = live_masks[args.live_frame]
    intrinsics = np.loadtxt(args.demo_recording / "cam_K.txt").reshape(3, 3)
    source = cloud(source_depth, source_mask, intrinsics, args.edge_pixels)
    target = cloud(target_depth, target_mask, intrinsics, args.edge_pixels)
    source_small = _downsample(source, args.sample_points)
    target_small = _downsample(target, args.sample_points)
    target_tree = cKDTree(target_small)
    mask_outside_pixels = cv2.distanceTransform((~target_mask).astype(np.uint8), cv2.DIST_L2, 3)
    target_depth_m = target_depth.astype(np.float32) * 0.001

    # No visual registration is used for the seed. Put the demo object center
    # at the live observed center while keeping the current gripper orientation.
    initial_camera = _camera_transform(start_joints, kinematics, demo_grasp, base_from_camera)
    initial_center = apply(np.median(source, axis=0)[None], initial_camera[:3, :3], initial_camera[:3, 3])[0]
    center_shift_camera = np.median(target, axis=0) - initial_center
    seeded_grasp = kinematics.forward_kinematics(start_joints)
    seeded_grasp[:3, 3] += base_from_camera[:3, :3] @ center_shift_camera
    seed = start_joints.copy()
    seed_ik_success = True
    try:
        for _ in range(args.seed_ik_attempts):
            seed = kinematics.inverse_kinematics(seed, seeded_grasp)
    except Exception:
        seed = start_joints.copy()
        seed_ik_success = False
    seed = np.clip(seed, joint_lo + args.bound_inset_deg, joint_hi - args.bound_inset_deg)

    refinement_attempts = int(trial_config.get("ik_refinement_attempts", args.ik_refinement_attempts))
    position_tolerance_m = float(trial_config.get("position_error_m", args.position_tolerance_m))
    rotation_tolerance_deg = float(trial_config.get("rotation_error_deg", args.rotation_tolerance_deg))
    result = {
        "inputs": {
            "skill": str(args.skill),
            "calibration": str(args.calibration),
            "demo_recording": str(args.demo_recording),
            "demo_mask": str(args.demo_mask),
            "demo_frame": args.demo_frame,
            "trial": str(args.trial),
            "role": args.role,
            "arm": args.arm,
            "live_frame": args.live_frame,
            "grasp_interval": args.grasp_interval,
            "path_intervals": args.path_intervals,
            "connection_samples": args.connection_samples,
        },
        "experiment_settings": {
            "joint_weights": args.joint_weights,
            "observation_weights": args.observation_weights,
            "visual_scale_m": args.visual_scale_m,
            "depth_band_m": args.depth_band_m,
            "max_evaluations": args.max_evaluations,
            "path_corrections": args.path_corrections,
            "path_seed_convergence_deg": args.path_seed_convergence_deg,
            "proximity_m": args.proximity_m,
            "edge_pixels": args.edge_pixels,
        },
        "source_points": len(source),
        "target_points": len(target),
        "optimization_points_per_surface": args.sample_points,
        "path_check_settings": {
            "ik_refinement_attempts": refinement_attempts,
            "position_tolerance_m": position_tolerance_m,
            "rotation_tolerance_deg": rotation_tolerance_deg,
            "source": str(trial_config_path) if trial_config_path.exists() else "explicit CLI defaults",
        },
        "seed": "observed arm configuration, then masked-surface centroid translation",
        "seed_ik_success": seed_ik_success,
        "grasp_demo_frame": skill["intervals"][args.grasp_interval]["samples"][0]["source_frame_index"],
        "motor_joint_bounds_deg": {
            name: [float(lo), float(hi)] for name, lo, hi in zip(arm_names, joint_lo, joint_hi, strict=True)
        },
        "runs": [],
    }
    for observation_weight in args.observation_weights:
        for joint_weight in args.joint_weights:
            working_seed = seed.copy()
            for correction in range(args.path_corrections + 1):
                calls = 0

                def residual(
                    q: np.ndarray,
                    *,
                    joint_weight: float = joint_weight,
                    observation_weight: float = observation_weight,
                ) -> np.ndarray:
                    nonlocal calls
                    calls += 1
                    transform = _camera_transform(q, kinematics, demo_grasp, base_from_camera)
                    moved = apply(source_small, transform[:3, :3], transform[:3, 3])
                    forward = target_tree.query(moved)[0]
                    backward = cKDTree(moved).query(target_small)[0]
                    visual = np.r_[forward / np.sqrt(len(forward)), backward / np.sqrt(len(backward))]
                    regularizer = np.sqrt(joint_weight) * (q - start_joints) / 90.0 * args.visual_scale_m
                    if observation_weight:
                        u = intrinsics[0, 0] * moved[:, 0] / moved[:, 2] + intrinsics[0, 2]
                        v = intrinsics[1, 1] * moved[:, 1] / moved[:, 2] + intrinsics[1, 2]
                        coords_u = u.astype(np.float32).reshape(-1, 1)
                        coords_v = v.astype(np.float32).reshape(-1, 1)
                        outside_px = cv2.remap(
                            mask_outside_pixels,
                            coords_u,
                            coords_v,
                            cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=max(target_mask.shape),
                        )[:, 0]
                        observed_depth = cv2.remap(
                            target_depth_m,
                            coords_u,
                            coords_v,
                            cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )[:, 0]
                        pixel_to_meter = moved[:, 2] / np.sqrt(intrinsics[0, 0] * intrinsics[1, 1])
                        outside_m = outside_px * pixel_to_meter
                        front_m = np.where(
                            observed_depth > 0,
                            np.maximum(observed_depth - moved[:, 2] - args.depth_band_m, 0),
                            0,
                        )
                        observation = np.sqrt(observation_weight / len(moved)) * np.r_[outside_m, front_m]
                    else:
                        observation = np.zeros(2 * len(moved))
                    return np.r_[visual, observation, regularizer]

                solve_started = time.perf_counter()
                solve = least_squares(
                    residual,
                    working_seed,
                    bounds=(joint_lo + args.bound_inset_deg, joint_hi - args.bound_inset_deg),
                    max_nfev=args.max_evaluations,
                    diff_step=args.diff_step,
                    xtol=args.xtol,
                    ftol=args.ftol,
                    gtol=args.gtol,
                )
                optimizer_seconds = time.perf_counter() - solve_started
                transform = _camera_transform(solve.x, kinematics, demo_grasp, base_from_camera)
                full_score = score(
                    source,
                    target,
                    target_mask,
                    target_depth,
                    intrinsics,
                    transform[:3, :3],
                    transform[:3, 3],
                    args.proximity_m,
                )
                remap_base = base_from_camera @ transform @ np.linalg.inv(base_from_camera)
                path = _path_check(
                    solve.x,
                    start_joints,
                    remap_base,
                    skill["intervals"][: args.path_intervals],
                    kinematics,
                    grasp_interval=args.grasp_interval,
                    connection_samples=args.connection_samples,
                    refinement_attempts=refinement_attempts,
                    position_tolerance_m=position_tolerance_m,
                    rotation_tolerance_deg=rotation_tolerance_deg,
                )
                result["runs"].append(
                    {
                        "joint_weight": joint_weight,
                        "observation_weight": observation_weight,
                        "path_correction": correction,
                        "optimizer_success": bool(solve.success),
                        "optimizer_status": int(solve.status),
                        "function_evaluations": calls,
                        "optimizer_seconds": optimizer_seconds,
                        "evaluate_and_path_seconds": time.perf_counter() - solve_started - optimizer_seconds,
                        "grasp_joints_deg": solve.x.tolist(),
                        "surface_score": full_score,
                        "path": path,
                        "rotation": transform[:3, :3].tolist(),
                        "translation_m": transform[:3, 3].tolist(),
                    }
                )
                if path["failure"] is not None or path["grasp_achieved_joints_deg"] is None:
                    break
                reached = np.asarray(path["grasp_achieved_joints_deg"], dtype=float)
                if np.linalg.norm(reached - solve.x) < args.path_seed_convergence_deg:
                    break
                working_seed = np.clip(
                    reached, joint_lo + args.bound_inset_deg, joint_hi - args.bound_inset_deg
                )
    result["seconds"] = time.perf_counter() - started
    result["limits"] = (
        "Single local seed; mask reconstructed from telemetry polygon; no collisions or motor execution."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--demo-recording", type=Path, required=True)
    parser.add_argument("--demo-mask", type=Path, required=True)
    parser.add_argument("--demo-frame", type=int, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--arm", choices=("left", "right"), required=True)
    parser.add_argument("--live-frame", type=int, default=0)
    parser.add_argument("--grasp-interval", type=int, required=True)
    parser.add_argument("--path-intervals", type=int, required=True)
    parser.add_argument("--connection-samples", type=int, required=True)
    parser.add_argument("--sample-points", type=int, default=400)
    parser.add_argument("--edge-pixels", type=int, default=1)
    parser.add_argument("--proximity-m", type=float, default=0.006)
    parser.add_argument("--joint-weights", type=float, nargs="+", default=(0.0, 0.01, 0.1, 1.0))
    parser.add_argument("--observation-weights", type=float, nargs="+", default=(0.0,))
    parser.add_argument("--depth-band-m", type=float, default=0.003)
    parser.add_argument("--visual-scale-m", type=float, default=0.006)
    parser.add_argument("--max-evaluations", type=int, default=80)
    parser.add_argument("--seed-ik-attempts", type=int, default=10)
    parser.add_argument("--bound-inset-deg", type=float, default=0.01)
    parser.add_argument("--path-corrections", type=int, default=0)
    parser.add_argument("--path-seed-convergence-deg", type=float, default=0.1)
    parser.add_argument("--ik-refinement-attempts", type=int, default=10)
    parser.add_argument("--position-tolerance-m", type=float, default=0.003)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--diff-step", type=float, default=0.001)
    parser.add_argument("--xtol", type=float, default=1e-6)
    parser.add_argument("--ftol", type=float, default=1e-6)
    parser.add_argument("--gtol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for row in output["runs"]:
        print(
            f"joint_weight={row['joint_weight']:g} "
            f"observation_weight={row['observation_weight']:g} "
            f"correction={row['path_correction']} "
            f"coverage={row['surface_score']['source_supported_fraction']:.1%} "
            f"solved={row['path']['solved_goals']} "
            f"branch_gap_deg={row['path'].get('grasp_joint_error_deg')} "
            f"max_travel_deg={row['path'].get('max_joint_travel_deg')} "
            f"solve_ms={row['optimizer_seconds'] * 1000:.0f}"
        )


if __name__ == "__main__":
    main()
