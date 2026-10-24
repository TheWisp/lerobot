"""Offline test of existing motion bridges with whole-surface fitting.

The saved visual-only proposals independently anchor the production group
service. A fixed attachment, seeded at the first evaluated held frame, predicts
gripper motion. Optional motion regularization and filtered-state propagation
are separate ablations. No prediction or refit is fed back as an independent
attachment observation. None of these modes validates the initial grasp.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from compare import RECORDING, frame_data, move
from full_demo_shape import available_geometry, diagnostics
from run_surface_tracker import unproject, voxel_sample
from shape_grounding import ShapeConfig, WholeSurfaceFit

from lerobot.policies.flash.causal_service import CausalPoseService, CausalTrackingConfig
from lerobot.policies.flash.perception import (
    AttachmentEstimate,
    LatestPoseStore,
    RGBDFrame,
    TimedEndEffectorHistory,
)
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


class MotionSupportedSurfaceFit(WholeSurfaceFit):
    """Add an explicit isotropic displacement prior on the saved surface.

    A single configurable metre scale gives the motion term the same units as
    the image/depth terms. This is a sensitivity experiment, not a calibrated
    noise model. No object axis, symmetry or landmark-specific rule is used.
    """

    def __init__(self, reference, observation, k, config, prior, motion_tolerance_m):
        super().__init__(reference, observation, k, config)
        if not np.isfinite(motion_tolerance_m) or motion_tolerance_m <= 0:
            raise ValueError("motion tolerance must be finite and positive")
        self.expected = move(reference, prior)
        self.motion_tolerance_m = motion_tolerance_m

    def residual(self, pose):
        motion = (move(self.reference, pose) - self.expected) / self.motion_tolerance_m
        return np.concatenate((super().residual(pose), motion.ravel() / np.sqrt(len(self.reference))))


def attachment_at(skill: dict, role: str, time_s: float) -> bool:
    transitions = {e["event_id"]: e for e in skill["attachment_transitions"] if e["role"] == role}
    events = {
        (s["time_s"], s["event_id"])
        for interval in skill["intervals"]
        for s in interval["samples"]
        if s.get("event_id") in transitions
    }
    attached = False
    for event_time, event in sorted(events):
        if event_time <= time_s:
            attached = transitions[event]["action"] == "attach"
    return attached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--trial-config", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--motion-tolerance-m", type=float)
    parser.add_argument("--propagate-filtered-held", action="store_true")
    args = parser.parse_args()
    baseline = np.load(args.baseline / "trace.npz")
    summary = json.loads((args.baseline / "summary.json").read_text())
    config = ShapeConfig(**summary["config"])
    trial = json.loads(args.trial_config.read_text())
    causal_config = CausalTrackingConfig.from_config(trial["causal_tracking"])
    skill = json.loads(args.skill.read_text())
    frames, source = baseline["frames"], baseline["reference_points"]
    times = np.loadtxt(RECORDING / "times.txt")
    k = np.loadtxt(RECORDING / "cam_K.txt").reshape(3, 3)
    recording = np.load(RECORDING.parent / "showservo_demo.npz")
    calibration = np.load(Path(__file__).parent.parent / "flash-tracker-comparison/pose_traces.npz")
    camera_from_base = np.linalg.inv(calibration["base_from_camera"])
    history = TimedEndEffectorHistory(
        capacity=len(recording["t"]), max_sample_gap_s=trial["attachment"]["max_motor_sample_gap_s"]
    )
    # Use actual capture/motor timestamps, not equal frame indices: the archive
    # has 713 camera frames but 711 robot samples.
    for timestamp, ee in zip(recording["t"], recording["tips"], strict=True):
        history.append(float(recording["t0"] + timestamp - times[0]), camera_from_base @ ee)
    layers = []
    service = CausalPoseService(
        LatestPoseStore(),
        **asdict(causal_config),
        object_centers_by_role={args.role: source.mean(0)},
        object_points_by_role={args.role: source},
        on_layer=layers.append,
    )
    attachment = AttachmentEstimate()
    poses, priors, runtime_poses, rows = [], [], [], []
    previous_held = False
    previous_ee = None
    for index, frame_id in enumerate(frames):
        observed = frame_data(int(frame_id), k)
        all_points = unproject(observed["depth"], k)[observed["obs"]["valid_image"]]
        observed["all_surface_points"] = all_points[voxel_sample(all_points, config.surface_spacing_m)]
        status = available_geometry(observed)
        now = float(times[frame_id] - times[0])
        held = attachment_at(skill, args.role, float(times[frame_id] - recording["t0"]))
        ee = history.at(now) if held else None
        frame = RGBDFrame(
            int(frame_id),
            observed["rgb"],
            observed["depth"].astype(np.float32),
            k,
            1.0,
            np.eye(4),
            now,
            now,
            "archived-fixed-camera",
        )
        direct = PoseEstimate(
            args.role,
            baseline["poses"][index],
            now,
            now,
            Evidence.OBSERVED,
            f"shape-reference-{summary['reference_frame']}",
            (True,) * 3,
            (True,) * 3,
            estimator="cached_visual_only_shape_fit",
        )
        started = time.perf_counter()
        layers.clear()
        accepted = service.process_offline(
            frame,
            (direct,) if status == "observed" else (),
            {args.role: observed["mask"]},
            attached_role=args.role if held else None,
        )
        service_ms = (time.perf_counter() - started) * 1000
        runtime = accepted.get(args.role)
        runtime_poses.append(runtime.pose if runtime is not None else np.full((4, 4), np.nan))
        prior, prior_kind = None, "none"
        if held and ee is not None:
            if not previous_held and status == "observed":
                attachment.seed_prediction(ee, direct, now)
            state = attachment.state(now, max_age_s=None)
            if state is not None:
                prior = ee @ state.gripper_from_object
                prior_kind = "gripper_attachment_prediction"
                if args.propagate_filtered_held and previous_held and previous_ee is not None:
                    # Predict the previous *filtered state*, not an independent
                    # observation. The attachment's measurement time is never
                    # refreshed from this estimate and no confidence is added.
                    prior = ee @ np.linalg.inv(previous_ee) @ poses[-1]
                    prior_kind = "gripper_delta_from_filtered_state"
        elif not held:
            groups = [p for p in layers if p.estimator == "environment_group_motion"]
            if groups:
                prior, prior_kind = groups[-1].pose, groups[-1].estimator
        previous_held = held
        previous_ee = ee
        fitter = WholeSurfaceFit(source, observed, k, config) if status == "observed" else None
        started = time.perf_counter()
        pose = direct.pose.copy()
        detail = {"evaluations": 0, "converged": True, "used_coarse": False}
        if prior is not None:
            constrained = (
                MotionSupportedSurfaceFit(source, observed, k, config, prior, args.motion_tolerance_m)
                if fitter is not None and args.motion_tolerance_m is not None
                else fitter
            )
            pose, detail = constrained.correct(prior) if constrained is not None else (prior.copy(), detail)
        correction_ms = (time.perf_counter() - started) * 1000
        poses.append(pose)
        priors.append(prior if prior is not None else np.full((4, 4), np.nan))
        row = {
            "frame": int(frame_id),
            "seconds": now,
            "held": int(held),
            "status": status,
            "prior_source": prior_kind,
            "runtime_source": runtime.estimator if runtime is not None else "missing",
            "visual_shape_cost": fitter.score(direct.pose) if fitter else np.nan,
            "prior_shape_cost": fitter.score(prior) if fitter and prior is not None else np.nan,
            "refit_shape_cost": fitter.score(pose) if fitter else np.nan,
            "service_ms": service_ms,
            "correction_ms": correction_ms,
            "converged": int(detail["converged"]),
            "used_coarse": int(detail["used_coarse"]),
            **(diagnostics(source, pose, observed, k) if fitter else {}),
        }
        rows.append(row)
        if frame_id % 30 == 0:
            print(frame_id, prior_kind, "shape cost", round(row["refit_shape_cost"], 3), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "trace.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.output / "trace.npz",
        frames=frames,
        poses=poses,
        prior_poses=priors,
        runtime_poses=runtime_poses,
        baseline_poses=baseline["poses"],
        reference_points=source,
        landmarks_reference=baseline["landmarks_reference"],
        statuses=[r["status"] for r in rows],
    )
    summary.update(
        {
            "method": "whole_surface_fit_with_recorded_motion_bridges",
            "motion_tolerance_m": args.motion_tolerance_m,
            "propagate_filtered_held": args.propagate_filtered_held,
            "statuses": dict(Counter(r["status"] for r in rows)),
            "optimizer_converged": sum(r["converged"] for r in rows),
            "coarse_reinitializations": sum(r["used_coarse"] for r in rows),
            "causal_config": asdict(causal_config),
            "prior_sources": dict(Counter(r["prior_source"] for r in rows)),
            "runtime_sources": dict(Counter(r["runtime_source"] for r in rows)),
            "anchor_outcomes": service.anchor_outcomes_by_role,
            "median_service_ms": float(np.median([r["service_ms"] for r in rows])),
            "median_correction_ms": float(np.median([r["correction_ms"] for r in rows])),
            "p95_correction_ms": float(np.percentile([r["correction_ms"] for r in rows], 95)),
            "warning": "Motion prediction assumes no slip over its interval. Fixed mode retains initial pose error; filtered mode may accumulate drift. This is an offline experiment, not a calibrated fused belief or a live integration.",
        }
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
