"""Motor-free preview of a prepared FLASH skill against one live RGB-D frame.

This does not claim collision safety or execute a robot. It tests registration,
calibrated FK/IK, object-relative goals, and solution continuity before a
physical attempt is considered.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.policies.flash.appearance import AppearancePixelMatcher, Sam3MaskSource
from lerobot.policies.flash.dino_features import DinoPatchExtractor
from lerobot.policies.flash.geometry import interpolate, inverse
from lerobot.policies.flash.kinematics import SelectedJointKinematics
from lerobot.policies.flash.matched_rgbd import MatchedRGBDPoseBackend
from lerobot.policies.flash.perception import RGBDFrame
from lerobot.policies.flash.prepared import PreparedFlashSkill
from lerobot.policies.flash.skill import AttachmentAction, MotionMode, Relation
from lerobot.policies.flash.symmetry import SymmetryCanonicalizingBackend
from lerobot.policies.flash.trajectory import Evidence, spatial_goal
from lerobot.robots.so107_description.calibrated_kinematics import make_calibrated_so107_kinematics
from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES


def preview(
    bundle_path: Path,
    rgb_path: Path,
    depth_path: Path,
    intrinsics_path: Path,
    calibration_path: Path,
    seed_path: Path,
    settings_path: Path,
) -> dict:
    bundle = PreparedFlashSkill.load(bundle_path)
    settings = json.loads(settings_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    seed = json.loads(seed_path.read_text())
    full = make_calibrated_so107_kinematics(calibration, arm=settings["robot_arm"])
    arm_names = tuple(name for name in MOTOR_NAMES if name != "gripper")
    kinematics = SelectedJointKinematics(
        full, tuple(MOTOR_NAMES), arm_names, {"gripper": seed["gripper.pos"]}
    )
    joints = np.array([seed[f"{name}.pos"] for name in arm_names], dtype=float)
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None or depth is None:
        raise FileNotFoundError("preview needs an aligned live RGB-D frame")
    frame = RGBDFrame(
        0,
        cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB),
        depth,
        np.loadtxt(intrinsics_path).reshape(3, 3),
        0.001,
        np.asarray(calibration["camera"]["T_base_cam"], dtype=float),
        0.0,
        0.0,
        bundle.references[0].snippets[0].calibration_version,
    )
    source = Sam3MaskSource(
        bundle.prompts_by_role,
        camera_id="flash-motion-preview",
        resolution=settings["sam_resolution"],
        seed_points_by_role=settings.get("seed_points_by_role"),
    )
    matcher = AppearancePixelMatcher(
        bundle.appearance_models,
        DinoPatchExtractor(bundle.descriptor_model_id),
        source,
        min_similarity=settings["min_similarity"],
    )
    backend = MatchedRGBDPoseBackend(
        tuple(model.points for model in bundle.appearance_models),
        matcher,
        threshold_m=settings["registration_threshold_m"],
        min_inliers=settings["registration_min_inliers"],
        trials=settings["registration_trials"],
        clock=lambda: 0.0,
    )
    if bundle.axial_symmetries:
        backend = SymmetryCanonicalizingBackend(backend, bundle.axial_symmetries)
    backend.initialize(bundle.references)
    found = {estimate.role: estimate for estimate in backend.update(frame)}
    locked_roles = {item.role for item in bundle.axial_symmetries if item.lock_to_baseline}
    missing = [
        role
        for role, estimate in found.items()
        if estimate.evidence == Evidence.LOST
        or (estimate.evidence == Evidence.PREDICTED and role not in locked_roles)
    ]
    if missing:
        raise ValueError(f"live frame lacks observed object poses: {missing}; {backend.last_failure}")
    live_poses = {role: estimate.pose for role, estimate in found.items()}
    attachment = None
    transitions = {item.event_id: item for item in bundle.skill.attachment_transitions}
    max_translation_error_m = 0.0
    max_rotation_error_deg = 0.0
    max_joint_step_deg = 0.0
    worst_fk = {}
    worst_step = {}
    solved_goals = 0
    interval_reports = []
    previous = kinematics.forward_kinematics(joints)
    for interval in bundle.skill.intervals:
        reference = (
            previous if interval.relation == Relation.EE_TO_START else live_poses[interval.reference_role]
        )
        if interval.relation == Relation.HELD_TO_TARGET and attachment is None:
            raise ValueError(f"{interval.name}: no live held-object attachment")
        if interval.mode == MotionMode.CONNECTION:
            endpoint = spatial_goal(interval, interval.samples[-1], reference, attachment)
            distance_m = float(np.linalg.norm(endpoint[:3, 3] - previous[:3, 3]))
            max_step_m = float(settings["preview_cartesian_step_m"])
            steps = max(2, int(np.ceil(distance_m / max_step_m)))
            goals = [interpolate(previous, endpoint, fraction / steps) for fraction in range(1, steps + 1)]
        else:
            goals = [spatial_goal(interval, sample, reference, attachment) for sample in interval.samples]
        before = joints.copy()
        for goal_index, target in enumerate(goals):
            refinement_seed = joints
            for _ in range(int(settings["ik_refinement_attempts"])):
                solved = np.asarray(kinematics.inverse_kinematics(refinement_seed, target), dtype=float)
                if solved.shape != joints.shape or not np.isfinite(solved).all():
                    raise ValueError(f"{interval.name}: invalid IK result")
                achieved = kinematics.forward_kinematics(solved)
                translation_error_m = float(np.linalg.norm(achieved[:3, 3] - target[:3, 3]))
                rotation_error_deg = float(
                    np.degrees(
                        np.arccos(np.clip((np.trace(achieved[:3, :3].T @ target[:3, :3]) - 1) / 2, -1, 1))
                    )
                )
                if (
                    translation_error_m <= settings["preview_position_tolerance_m"]
                    and rotation_error_deg <= settings["preview_rotation_tolerance_deg"]
                ):
                    break
                refinement_seed = solved
            if translation_error_m > max_translation_error_m:
                worst_fk = {
                    "interval": interval.name,
                    "goal_index": goal_index,
                    "target_xyz_m": target[:3, 3].round(4).tolist(),
                    "target_pose": target.tolist(),
                    "achieved_xyz_m": achieved[:3, 3].round(4).tolist(),
                    "before_joints_deg": joints.round(2).tolist(),
                    "joints_deg": solved.round(2).tolist(),
                }
            max_translation_error_m = max(max_translation_error_m, translation_error_m)
            max_rotation_error_deg = max(max_rotation_error_deg, rotation_error_deg)
            step_deg = float(np.max(np.abs(solved - joints)))
            if step_deg > max_joint_step_deg:
                worst_step = {
                    "interval": interval.name,
                    "goal_index": goal_index,
                    "before_deg": joints.round(2).tolist(),
                    "after_deg": solved.round(2).tolist(),
                }
            max_joint_step_deg = max(max_joint_step_deg, step_deg)
            joints, previous = solved, achieved
            solved_goals += 1
        event = transitions.get(interval.samples[-1].event_id)
        if event is not None:
            if event.action == AttachmentAction.ATTACH:
                attachment = inverse(previous) @ live_poses[event.role]
            else:
                attachment = None
        interval_reports.append(
            {
                "interval": interval.name,
                "goals": len(goals),
                "end_xyz_m": previous[:3, 3].round(4).tolist(),
                "joint_change_deg": float(np.max(np.abs(joints - before))),
            }
        )
    return {
        "pose_evidence": {role: estimate.evidence.value for role, estimate in found.items()},
        "live_object_poses": {role: pose.tolist() for role, pose in live_poses.items()},
        "solved_goals": solved_goals,
        "max_fk_translation_error_m": max_translation_error_m,
        "max_fk_rotation_error_deg": max_rotation_error_deg,
        "max_joint_step_deg": max_joint_step_deg,
        "worst_fk": worst_fk,
        "worst_step": worst_step,
        "intervals": interval_reports,
        "collision_checked": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "rgb", "depth", "intrinsics", "calibration", "seed", "settings"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            preview(
                args.bundle, args.rgb, args.depth, args.intrinsics, args.calibration, args.seed, args.settings
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
