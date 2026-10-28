"""Replay alternative held-correction gates on a sealed-format prepared demo.

This is a research harness, not a runtime policy. The counterfactual updater
rewinds an accepted attachment when a proposed gate would withhold it; its
baseline must reproduce the recorded trace exactly before any comparison is
reported. Run with PYTHONPATH=src and one or more --bundle paths.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from lerobot.policies.flash.geometry import transform
from lerobot.policies.flash.held_pose_diagnostics import _arm_pose_at
from lerobot.policies.flash.perception import HeldAttachmentUpdater
from lerobot.policies.flash.pose_disagreement import surface_support
from lerobot.policies.flash.runtime import AttachmentRuntimeLimits
from lerobot.policies.flash.sealed_demo_import import _frame, _nearest_index
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


class RecordedGraspKinematics:
    def __init__(self, grasp_tip):
        self.grasp_tip = grasp_tip

    def forward_kinematics(self, _joints):
        return self.grasp_tip


class AblatedUpdater(HeldAttachmentUpdater):
    def __init__(self, *args, rule, **kwargs):
        super().__init__(*args, **kwargs)
        self.rule = rule
        self.withheld = []

    def on_pose_update(self, version, poses, *, frame=None, mask=None, ee_at_capture=None):
        old_attachment = self._estimate.gripper_from_object
        old_time = self._estimate._source_time_s
        old_reference = self._estimate._reference_version
        old_postgrasp = self.has_postgrasp_observation
        old_conflicted = self._conflicted
        old_ambiguous = self._visual_ambiguous
        old_pending = self._pending_visual
        accepted = super().on_pose_update(version, poses, frame=frame, mask=mask, ee_at_capture=ee_at_capture)
        if not accepted or old_attachment is None or self.rule == "baseline":
            return accepted
        evidence = self.latest_visual_evidence()
        if evidence is None or evidence["gripper"] is None:
            return accepted
        visual, gripper = evidence["visual"], evidence["gripper"]
        v_depth, g_depth = visual["depth_residual_p90_m"], gripper["depth_residual_p90_m"]
        prior_dominates = (
            v_depth is not None
            and g_depth is not None
            and gripper["mask_fraction"] >= visual["mask_fraction"]
            and g_depth <= v_depth
            and (gripper["mask_fraction"] > visual["mask_fraction"] or g_depth < v_depth)
        )
        if self.rule == "prior_dominates" and prior_dominates:
            reject = True
        elif self.rule == "supported_prior_dominates":
            reject = (
                prior_dominates
                and gripper["supported_pixels"] >= evidence["min_supported_pixels"]
                and g_depth <= evidence["depth_tolerance_m"]
            )
        elif self.rule == "visual_over_tolerance":
            reject = v_depth is None or v_depth > evidence["depth_tolerance_m"]
        else:
            reject = False
        if reject:
            self._estimate._attachment = old_attachment
            self._estimate._source_time_s = old_time
            self._estimate._reference_version = old_reference
            self.has_postgrasp_observation = old_postgrasp
            self._conflicted = old_conflicted
            self._visual_ambiguous = old_ambiguous
            self._pending_visual = old_pending
            self._last_visual_correction_accepted = False
            self.last_rejection = "offline ablation withheld correction"
            self.withheld.append(evidence["frame_id"])
            return False
        return True


def evaluate(bundle: Path, rule: str) -> dict:
    prepared = bundle.resolve().parent
    root = prepared.parents[3]
    name = prepared.name
    annotation = json.loads((prepared / "annotation.json").read_text())
    report = json.loads((prepared / "preparation-report.json").read_text())
    role = annotation["held_role"]
    reference = f"{name}:analysis"
    with np.load(root / "showservo_demo.npz") as arm:
        arm_times = np.asarray(arm["t"], float)
        joints = np.asarray(arm["q_obs"], float)
        tips = np.asarray(arm["tips"], float)
    with np.load(root / report["pose_trace_cache"]) as data:
        poses = data[f"{role}_poses"].copy()
        observed = data[f"{role}_observed"].copy()
        translation_supported = data[f"{role}_translation_supported"].copy()
        rotation_supported = data[f"{role}_rotation_supported"].copy()
        masks = {
            int(i): mask.copy() for i, mask in zip(data["held_mask_indices"], data["held_masks"], strict=True)
        }
    with np.load(prepared / "models.npz") as models:
        model = models[f"points_{list(annotation['prompts_by_role']).index(role)}"].copy()
    with np.load(prepared / "held-attachment-trace.npz") as trace:
        camera_times = trace["camera_times_s"].copy()
        recorded_attachment = trace["gripper_from_object"].copy()
        recorded_conflicted = trace["conflicted"].copy()
        recorded_corrected = trace["corrected"].copy()
    calib = json.loads(Path(annotation["calibration_path"]).expanduser().read_text())
    base_from_camera = np.asarray(calib["camera"]["T_base_cam"], float)
    intrinsics = np.loadtxt(root / "recording/cam_K.txt").reshape(3, 3)
    grasp = _nearest_index(arm_times, annotation["grasp_time_s"])
    release = _nearest_index(arm_times, annotation["release_time_s"])
    limits = AttachmentRuntimeLimits.from_config(annotation["held_attachment"])
    config = annotation.get("causal_tracking")
    min_points = config["min_inliers"] if config else annotation["registration_min_inliers"]
    tolerance = config["fit_threshold_m"] if config else annotation["registration_threshold_m"]
    updater = AblatedUpdater(
        role,
        reference,
        RecordedGraspKinematics(tips[grasp]),
        tuple(f"joint-{i}" for i in range(joints.shape[1])),
        history_capacity=limits.history_capacity,
        max_motor_sample_gap_s=limits.max_motor_sample_gap_s,
        max_attachment_age_s=limits.max_attachment_age_s,
        max_update_shift_m=limits.max_update_shift_m,
        max_update_turn_rad=limits.max_update_turn_rad,
        object_center_object_m=model.mean(axis=0),
        object_points_object_m=model,
        min_visual_support_points=min_points,
        depth_occlusion_tolerance_m=tolerance,
        rule=rule,
    )
    selected_idx = int(annotation["grasp_reference_frame"])
    selected = PoseEstimate(
        role,
        poses[selected_idx],
        camera_times[selected_idx],
        camera_times[selected_idx],
        Evidence.OBSERVED,
        reference,
        (True,) * 3,
        (True,) * 3,
    )
    assert updater.activate(float(arm_times[grasp]), selected, joints[grasp])
    attachments = []
    conflicted = []
    corrected = []
    scores = []
    decisions = []
    accepted_frames = []
    last_attachment = updater.last_accepted_gripper_from_object
    has_correction = False
    for i, source_time in enumerate(camera_times):
        if arm_times[grasp] <= source_time < arm_times[release] and observed[i]:
            pose = PoseEstimate(
                role,
                poses[i],
                source_time,
                source_time,
                Evidence.OBSERVED,
                reference,
                tuple(bool(x) for x in translation_supported[i]),
                tuple(bool(x) for x in rotation_supported[i]),
            )
            mask = masks.get(i)
            ee = _arm_pose_at(arm_times, tips, float(source_time))
            if mask is not None and ee is not None:
                frame = _frame(root, i, intrinsics, base_from_camera, "analysis", float(source_time))
                accepted = updater.on_pose_update(i, {role: pose}, frame=frame, mask=mask, ee_at_capture=ee)
                evidence = updater.latest_visual_evidence()
                if evidence is not None:
                    decisions.append(
                        {
                            "frame": i,
                            "accepted": accepted,
                            "verdict": evidence["verdict"],
                            "difference_mm": None
                            if evidence["pose_difference_m"] is None
                            else evidence["pose_difference_m"] * 1000,
                            "visual_depth_mm": None
                            if evidence["visual"]["depth_residual_p90_m"] is None
                            else evidence["visual"]["depth_residual_p90_m"] * 1000,
                            "gripper_depth_mm": None
                            if evidence["gripper"] is None
                            or evidence["gripper"]["depth_residual_p90_m"] is None
                            else evidence["gripper"]["depth_residual_p90_m"] * 1000,
                            "reason": updater.last_rejection,
                        }
                    )
                if accepted:
                    last_attachment = updater.last_accepted_gripper_from_object
                    accepted_frames.append(i)
                    has_correction = True
                score = surface_support(
                    frame,
                    mask,
                    model,
                    transform(ee @ last_attachment),
                    depth_occlusion_tolerance_m=tolerance,
                )
                scores.append((i, score.projected_point_mask_fraction, score.depth_residual_p90_m))
        attachments.append(last_attachment.copy())
        conflicted.append(
            updater.state(float(source_time)) is None
            if arm_times[grasp] <= source_time < arm_times[release]
            else False
        )
        corrected.append(has_correction if arm_times[grasp] <= source_time < arm_times[release] else False)
    attachments = np.stack(attachments)
    baseline_max_abs = float(np.max(np.abs(attachments - recorded_attachment)))
    baseline_conflicted_mismatch = int(np.count_nonzero(np.asarray(conflicted) != recorded_conflicted))
    baseline_corrected_mismatch = int(np.count_nonzero(np.asarray(corrected) != recorded_corrected))
    if rule == "baseline" and (
        baseline_max_abs > 1e-12 or baseline_conflicted_mismatch or baseline_corrected_mismatch
    ):
        raise AssertionError("baseline did not reproduce the recorded attachment trace")
    finite = [r for _, _, r in scores if np.isfinite(r)]
    return {
        "dataset": root.name,
        "prepared_bundle": name,
        "rule": rule,
        "accepted_observation_frames": accepted_frames,
        "withheld_observation_frames": updater.withheld,
        "conflicted_camera_frames": int(sum(conflicted)),
        "median_visible_depth_p90_mm": float(np.median(finite) * 1000),
        "median_projected_mask_fraction": float(np.median([f for _, f, _ in scores])),
        "final_gripper_from_object_translation_delta_vs_recorded_mm": float(
            1000 * np.linalg.norm(attachments[-1, :3, 3] - recorded_attachment[-1, :3, 3])
        ),
        "source_observations": decisions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rules = ("baseline", "prior_dominates", "supported_prior_dominates", "visual_over_tolerance")
    results = []
    for bundle in args.bundle:
        if not bundle.is_file():
            parser.error(f"missing prepared bundle: {bundle}")
        for rule in rules:
            results.append(evaluate(bundle, rule))
    frame_rows = []
    for result in results:
        for decision in result.pop("source_observations"):
            frame_rows.append(
                {
                    "dataset": result["dataset"],
                    "prepared_bundle": result["prepared_bundle"],
                    "rule": result["rule"],
                    **decision,
                }
            )
    args.output.write_text(json.dumps({"results": results}, indent=2) + "\n")
    details_path = args.output.with_name(f"{args.output.stem}_frames.csv")
    with details_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=frame_rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(frame_rows)
