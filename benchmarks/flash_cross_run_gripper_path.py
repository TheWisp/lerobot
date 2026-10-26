"""Compare demo-to-live registrations by the resulting initial gripper path.

This offline audit solves the first object-relative approach/grasp intervals
from a recorded motor state. It does not check collisions, command a robot, or
prove that a failed local IK search has no other solution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.geometry import interpolate
from lerobot.policies.flash.kinematics import SelectedJointKinematics
from lerobot.robots.so107_description.calibrated_kinematics import make_calibrated_so107_kinematics
from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES


def _rotation_error_deg(actual: np.ndarray, target: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(actual[:3, :3] @ target[:3, :3].T).magnitude()))


def audit(
    skill_path: Path,
    calibration_path: Path,
    report_path: Path,
    trial_root: Path,
    *,
    arm: str,
    methods: tuple[str, ...] | None,
    interval_count: int,
    connection_samples: int,
) -> dict:
    if interval_count < 1 or connection_samples < 2:
        raise ValueError("interval_count must be positive and connection_samples must be at least two")
    skill = json.loads(skill_path.read_text())
    intervals = skill["intervals"][:interval_count]
    if len(intervals) != interval_count or any(item["relation"] != "ee_to_object" for item in intervals):
        raise ValueError("this audit only supports initial object-relative intervals")
    calibration = json.loads(calibration_path.read_text())
    settings = json.loads((trial_root / "flash/v1/trial_config.json").read_text())
    report = json.loads(report_path.read_text())
    with (trial_root / "flash/v1/episode_000000.jsonl").open() as telemetry:
        first_row = json.loads(telemetry.readline())
    if first_row["action_source"] != "observed_hold":
        raise ValueError("trial does not begin with an observed motor state")
    full_joints = np.array(
        [first_row["effective_action"][f"{name}.pos"] for name in MOTOR_NAMES], dtype=float
    )
    arm_names = tuple(name for name in MOTOR_NAMES if name != "gripper")
    kinematics = SelectedJointKinematics(
        make_calibrated_so107_kinematics(calibration, arm=arm),
        tuple(MOTOR_NAMES),
        arm_names,
        {"gripper": float(full_joints[-1])},
    )
    starting_joints = full_joints[: len(arm_names)]
    starting_pose = kinematics.forward_kinematics(starting_joints)
    base_from_camera = np.asarray(calibration["camera"]["T_base_cam"], dtype=float)
    camera_from_base = np.linalg.inv(base_from_camera)
    candidates = {item["method"]: item for item in report["candidates"]}
    if methods is None:
        methods = tuple(candidates)
    missing = set(methods) - set(candidates)
    if missing:
        raise ValueError(f"registration methods not present: {sorted(missing)}")

    rows = []
    for method in methods:
        candidate = candidates[method]
        demo_camera_to_live_camera = np.eye(4)
        demo_camera_to_live_camera[:3, :3] = candidate["rotation"]
        demo_camera_to_live_camera[:3, 3] = candidate["translation_m"]
        remap = base_from_camera @ demo_camera_to_live_camera @ camera_from_base
        first_goal = remap @ np.asarray(intervals[0]["samples"][-1]["demo_ee"], dtype=float)
        row = {
            "method": method,
            "two_way_cloud_coverage": candidate["quality"],
            "first_goal_translation_mm": float(
                np.linalg.norm(first_goal[:3, 3] - starting_pose[:3, 3]) * 1000
            ),
            "first_goal_rotation_deg": _rotation_error_deg(first_goal, starting_pose),
        }
        joints = starting_joints.copy()
        achieved = starting_pose
        joint_travel_deg = np.zeros(len(arm_names), dtype=float)
        max_joint_step_deg = 0.0
        solved_goals = 0
        failure = None
        for interval_index, interval in enumerate(intervals):
            if interval["mode"] == "connection":
                endpoint = remap @ np.asarray(interval["samples"][-1]["demo_ee"], dtype=float)
                goals = [
                    interpolate(achieved, endpoint, step / connection_samples)
                    for step in range(1, connection_samples + 1)
                ]
            elif interval["mode"] == "demonstrated":
                goals = [remap @ np.asarray(sample["demo_ee"], dtype=float) for sample in interval["samples"]]
            else:
                raise ValueError(f"unsupported interval mode: {interval['mode']}")
            for goal_index, goal in enumerate(goals):
                before = joints.copy()
                try:
                    for _ in range(int(settings["ik_refinement_attempts"])):
                        joints = kinematics.inverse_kinematics(joints, goal)
                        achieved = kinematics.forward_kinematics(joints)
                        position_error_mm = float(np.linalg.norm(achieved[:3, 3] - goal[:3, 3]) * 1000)
                        rotation_error_deg = _rotation_error_deg(achieved, goal)
                        if position_error_mm <= float(
                            settings["position_error_m"]
                        ) * 1000 and rotation_error_deg <= float(settings["rotation_error_deg"]):
                            break
                except Exception as exc:  # An IK failure is a result for this candidate.
                    failure = {
                        "interval_index": interval_index,
                        "interval": interval["name"],
                        "goal_index": goal_index + 1,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                    break
                if position_error_mm > float(
                    settings["position_error_m"]
                ) * 1000 or rotation_error_deg > float(settings["rotation_error_deg"]):
                    failure = {
                        "interval_index": interval_index,
                        "interval": interval["name"],
                        "goal_index": goal_index + 1,
                        "position_error_mm": position_error_mm,
                        "rotation_error_deg": rotation_error_deg,
                    }
                    break
                step_deg = np.abs(joints - before)
                joint_travel_deg += step_deg
                max_joint_step_deg = max(max_joint_step_deg, float(np.max(step_deg)))
                solved_goals += 1
            if failure is not None:
                break
        row.update(
            {
                "solved_goals": solved_goals,
                "failure": failure,
                "joint_travel_deg": dict(zip(arm_names, joint_travel_deg.tolist(), strict=True)),
                "max_joint_travel_deg": float(np.max(joint_travel_deg)),
                "max_sample_joint_step_deg": max_joint_step_deg,
            }
        )
        rows.append(row)
    return {
        "basis": "recorded motor state and saved cross-run registration candidates",
        "skill": str(skill_path),
        "trial": str(trial_root),
        "registration_report": str(report_path),
        "arm": arm,
        "interval_count": interval_count,
        "connection_samples": connection_samples,
        "ik_refinement_attempts": settings["ik_refinement_attempts"],
        "position_tolerance_m": settings["position_error_m"],
        "rotation_tolerance_deg": settings["rotation_error_deg"],
        "limits": "No collision, motor-tracking, or alternative-path search; IK failure is not a proof of global unreachability.",
        "candidates": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--registration-report", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--arm", choices=("left", "right"), required=True)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--interval-count", type=int, required=True)
    parser.add_argument("--connection-samples", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.skill,
        args.calibration,
        args.registration_report,
        args.trial,
        arm=args.arm,
        methods=tuple(args.methods) if args.methods else None,
        interval_count=args.interval_count,
        connection_samples=args.connection_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["candidates"]:
        print(row["method"], row["solved_goals"], row["failure"])


if __name__ == "__main__":
    main()
