"""Audit how cross-run surface matches change the next FLASH keyframe goal.

This reads saved registration candidates and recorded motor state. It does not
move a robot, test collision avoidance, or claim that a surface fit is a valid
material-point correspondence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.robots.so107_description.calibrated_kinematics import make_calibrated_so107_kinematics
from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES


def run(
    skill_path: Path,
    calibration_path: Path,
    cases: list[tuple[Path, Path]],
    interval_index: int,
    arm: str,
) -> dict:
    skill = json.loads(skill_path.read_text())
    interval = skill["intervals"][interval_index]
    endpoint = np.asarray(interval["samples"][-1]["demo_ee"], dtype=float)
    calibration = json.loads(calibration_path.read_text())
    base_from_camera = np.asarray(calibration["camera"]["T_base_cam"], dtype=float)
    camera_from_base = np.linalg.inv(base_from_camera)
    kinematics = make_calibrated_so107_kinematics(calibration, arm=arm)
    rows = []
    for report_path, trial_path in cases:
        report = json.loads(report_path.read_text())
        with (trial_path / "flash/v1/episode_000000.jsonl").open() as telemetry:
            first_line = telemetry.readline()
        trial_start = json.loads(first_line)
        if trial_start["action_source"] != "observed_hold":
            raise ValueError(f"{trial_path} does not begin with an observed motor state")
        joints = np.asarray(
            [trial_start["effective_action"][f"{name}.pos"] for name in MOTOR_NAMES], dtype=float
        )
        current_ee = kinematics.forward_kinematics(joints)
        candidates = []
        for item in report["candidates"]:
            demo_camera_to_live_camera = np.eye(4)
            demo_camera_to_live_camera[:3, :3] = item["rotation"]
            demo_camera_to_live_camera[:3, 3] = item["translation_m"]
            live_goal = base_from_camera @ demo_camera_to_live_camera @ camera_from_base @ endpoint
            distance_mm = float(np.linalg.norm(live_goal[:3, 3] - current_ee[:3, 3]) * 1000)
            turn_deg = float(
                np.degrees(Rotation.from_matrix(live_goal[:3, :3] @ current_ee[:3, :3].T).magnitude())
            )
            candidates.append(
                {
                    "method": item["method"],
                    "surface_coverage": item["quality"],
                    "projection_mask_fraction": item["projection_mask_fraction"],
                    "straight_line_position_change_mm": distance_mm,
                    "orientation_change_deg": turn_deg,
                }
            )
        rows.append(
            {
                "trial": str(trial_path),
                "registration_report": str(report_path),
                "demo_reference_frame": report["demo_frame"],
                "live_frame": report["live_frame"],
                "next_keyframe": interval["samples"][-1]["source_frame_index"],
                "best_surface_fit": candidates[0],
                "least_position_change": min(
                    candidates, key=lambda candidate: candidate["straight_line_position_change_mm"]
                ),
                "least_orientation_change": min(
                    candidates, key=lambda candidate: candidate["orientation_change_deg"]
                ),
                "candidates": candidates,
            }
        )
    return {
        "purpose": "offline cross-run match to the next ordered keyframe; no robot command",
        "arm": arm,
        "limits": (
            "These are Cartesian differences, not collision-free paths or IK solutions. "
            "Saved registration quality is nearest-surface coverage, not a complete "
            "mask/depth acceptance decision."
        ),
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--arm", choices=("left", "right"), required=True)
    parser.add_argument("--case", nargs=2, action="append", metavar=("REPORT", "TRIAL"), required=True)
    parser.add_argument("--interval-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.skill,
        args.calibration,
        [(Path(report), Path(trial)) for report, trial in args.case],
        args.interval_index,
        args.arm,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for case in result["cases"]:
        print(
            Path(case["trial"]).name,
            "best fit:",
            case["best_surface_fit"],
            "shortest translation:",
            case["least_position_change"],
        )


if __name__ == "__main__":
    main()
