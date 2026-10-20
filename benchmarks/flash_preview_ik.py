"""Preview a prepared skill's joint paths without commanding the robot.

This isolates geometric IK feasibility and interval timing. It does not test
perception freshness, collision clearance, motor tracking or task success.
The optional trial replay begins after its last completed interval and uses
the recorded grasp-time pose as an attachment approximation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot.policies.flash.geometry import inverse
from lerobot.policies.flash.kinematics import SelectedJointKinematics
from lerobot.policies.flash.planner import JointPlanner, PlannerLimits
from lerobot.policies.flash.prepared import PreparedFlashSkill
from lerobot.policies.flash.skill import MotionMode, Relation
from lerobot.policies.flash.trajectory import AttachmentState, ConnectionTrajectory, Evidence, PoseEstimate
from lerobot.robots.so107_description.calibrated_kinematics import make_calibrated_so107_kinematics
from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES


def preview(
    source_root: Path,
    bundle_path: Path,
    config_path: Path,
    *,
    trial_root: Path | None = None,
    max_ticks_per_interval: int = 1000,
) -> dict:
    if max_ticks_per_interval < 2:
        raise ValueError("IK preview needs at least two ticks per interval")
    settings = json.loads(config_path.read_text())
    calibration = json.loads(Path(settings["calibration_path"]).expanduser().read_text())
    skill = PreparedFlashSkill.load(bundle_path).skill
    names = tuple(MOTOR_NAMES)
    arm_names = tuple(name for name in names if name != "gripper")
    trace = None
    first_interval = 0
    if trial_root is None:
        with np.load(source_root / "showservo_demo.npz", allow_pickle=False) as demo:
            current_joints = np.asarray(demo["q_obs"][0], dtype=float)
    else:
        import pyarrow.parquet as pq

        trace = [json.loads(line) for line in (trial_root / "flash/v1/episode_000000.jsonl").open()]
        observations = np.asarray(
            pq.read_table(trial_root / "data/chunk-000/file-000.parquet", columns=["observation.state"])
            .column(0)
            .to_pylist(),
            dtype=float,
        )
        if len(trace) != len(observations):
            raise ValueError("trial trace and observations have different frame counts")
        current_joints = observations[-1]
        completed = [
            event["interval"]
            for row in trace
            for event in row["progress_events"]
            if event["type"] == "interval_complete"
        ]
        if completed:
            first_interval = next(
                index + 1 for index, interval in enumerate(skill.intervals) if interval.name == completed[-1]
            )
    kinematics = SelectedJointKinematics(
        make_calibrated_so107_kinematics(calibration, arm=settings["robot_arm"]),
        names,
        arm_names,
        {"gripper": float(current_joints[-1])},
    )
    limits = PlannerLimits(
        tuple(settings["joint_speed_deg_s"]),
        settings["position_error_m"],
        np.deg2rad(settings["rotation_error_deg"]),
        1e6,  # Offline geometry only; no camera is running to refresh pose timestamps.
        allow_predicted_reference=True,
        ik_refinement_attempts=settings["ik_refinement_attempts"],
        max_connection_ik_samples=settings["max_connection_ik_samples"],
        max_connection_retimings=settings["max_connection_retimings"],
    )
    planner = JointPlanner(kinematics, arm_names, "gripper", limits)
    seed = current_joints[: len(arm_names)].copy()
    attachment = None
    if trace is not None:
        attach_row = next((row for row in trace if row.get("event_id") == "attach-held"), None)
        if attach_row is not None:
            held_role = next(
                transition.role
                for transition in skill.attachment_transitions
                if transition.event_id == "attach-held"
            )
            frame_index = attach_row["frame_index"]
            held_pose = np.asarray(attach_row["poses"][held_role]["pose"], dtype=float)
            attachment = AttachmentState(
                inverse(kinematics.forward_kinematics(observations[frame_index, : len(arm_names)]))
                @ held_pose,
                0.0,
                0.0,
                next(
                    transition.reference_version
                    for transition in skill.attachment_transitions
                    if transition.event_id == "attach-held"
                ),
            )
    rows = []
    for interval in skill.intervals[first_interval:]:
        if interval.relation == Relation.EE_TO_START:
            live_reference = kinematics.forward_kinematics(seed)
        elif trace is not None:
            live_reference = np.asarray(trace[-1]["poses"][interval.reference_role]["pose"], dtype=float)
        else:
            live_reference = interval.samples[0].demo_reference
        reference = PoseEstimate(
            interval.reference_role,
            live_reference,
            0.0,
            0.0,
            Evidence.OBSERVED,
            interval.reference_version,
            (True,) * 3,
            (True,) * 3,
        )
        if interval.relation == Relation.HELD_TO_TARGET and attachment is None:
            attachment = AttachmentState(
                inverse(interval.samples[0].demo_ee) @ interval.samples[0].demo_held,
                0.0,
                0.0,
                interval.held_reference_version,
            )
        connection = (
            ConnectionTrajectory.begin(
                interval,
                kinematics.forward_kinematics(seed),
                live_reference,
                attachment.gripper_from_object
                if attachment is not None and interval.relation == Relation.HELD_TO_TARGET
                else None,
                0.0,
            )
            if interval.mode == MotionMode.CONNECTION
            else None
        )
        actions = planner.plan(
            interval,
            start_tick=0,
            count=max_ticks_per_interval,
            epoch_s=0.0,
            period_s=1 / settings["fps"],
            interval_start_s=0.0,
            seed_deg=seed,
            reference=reference,
            attachment=attachment,
            connection=connection,
        )
        if not actions[-1].interval_complete:
            raise RuntimeError(f"{interval.name}: preview did not complete within the configured tick budget")
        seed = np.array([actions[-1].action[f"{name}.pos"] for name in arm_names])
        rows.append(
            {
                "interval": interval.name,
                "ticks": len(actions),
                "duration_s": len(actions) / settings["fps"],
                "event": actions[-1].event_id,
            }
        )
    return {
        "basis": "recorded_trial" if trial_root is not None else "demonstration",
        "first_interval": first_interval,
        "intervals": rows,
        "limits": {
            "position_error_m": limits.position_error_m,
            "rotation_error_deg": settings["rotation_error_deg"],
            "joint_speed_deg_s": limits.joint_speed_deg_s,
        },
        "limitations": "IK geometry only; no perception, collision, motor or outcome verification",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--trial-root", type=Path)
    parser.add_argument("--max-ticks-per-interval", type=int, default=1000)
    args = parser.parse_args()
    print(
        json.dumps(
            preview(
                args.source_root,
                args.bundle,
                args.config,
                trial_root=args.trial_root,
                max_ticks_per_interval=args.max_ticks_per_interval,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
