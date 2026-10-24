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
from collections import Counter, OrderedDict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from compare import RECORDING, frame_data, move
from full_demo_shape import available_geometry, diagnostics
from run_surface_tracker import unproject, voxel_sample
from scipy.optimize import least_squares
from shape_grounding import ShapeConfig, WholeSurfaceFit

from lerobot.policies.flash.causal_service import CausalPoseService, CausalTrackingConfig
from lerobot.policies.flash.perception import (
    AttachmentEstimate,
    LatestPoseStore,
    RGBDFrame,
    TimedEndEffectorHistory,
)
from lerobot.policies.flash.surface_motion import (
    PendingMotionCorrection,
    RigidMotionState,
    SurfaceMotionConfig,
    displacement_jacobian,
    pose_with_delta,
    surface_support,
)
from lerobot.policies.flash.trajectory import Evidence, PoseEstimate


class SupportedSurfaceFit(WholeSurfaceFit):
    """Correct only motion directions the current depth patch constrains.

    The inherited noise scales define an explicit dead band: sub-noise residuals
    do not repeatedly rewrite the attachment. No coarse six-axis reinitialization
    runs while a motion relationship is available.
    """

    def __init__(self, reference, observation, k, config, motion_config):
        super().__init__(reference, observation, k, config)
        self.motion_config = motion_config

    def bounded_residual(self, pose):
        # The four parent blocks are distances / (noise * sqrt(block size)).
        sizes = (len(self.reference), len(self.target), len(self.reference), len(self.mask_outline))
        floors = np.concatenate([np.full(size, 1 / np.sqrt(size)) for size in sizes])
        return np.maximum(super().residual(pose) - floors, 0)

    def correct(self, prior):
        if not np.any(self.bounded_residual(prior)):
            return prior.copy(), {
                "converged": False,
                "used_coarse": False,
                "evaluations": 0,
                "supported_modes": 0,
                "surface_matches": 0,
                "visual_changed": False,
            }
        # A bad grasp seed can lie outside the local registration basin. The
        # global fit is a proposal only: its unsupported components never enter
        # the authoritative state. Estimate observability at the fitted surface.
        proposal, proposal_detail = super().correct(prior)
        support = surface_support(self.reference, proposal, self.target, self.motion_config)
        center = move(self.reference, prior).mean(0)
        expected = move(self.reference, prior)

        def objective(parameters):
            pose = pose_with_delta(prior, center, support.basis @ parameters)
            motion = (move(self.reference, pose) - expected) / self.motion_config.correction_uncertainty_m
            return np.r_[self.bounded_residual(pose), motion.ravel() / np.sqrt(len(expected))]

        rank = support.basis.shape[1]
        detail = {
            "converged": False,
            "used_coarse": False,
            "evaluations": 0,
            "supported_modes": rank,
            "surface_matches": support.matched_points,
            "sensitivity": support.sensitivity.tolist(),
            "visual_changed": False,
        }
        if not rank or not np.any(self.bounded_residual(prior)):
            return prior.copy(), detail
        from scipy.spatial.transform import Rotation

        proposal_center = move(self.reference, proposal).mean(0)
        innovation = np.r_[
            Rotation.from_matrix(proposal[:3, :3] @ prior[:3, :3].T).as_rotvec(),
            proposal_center - center,
        ]
        jac = displacement_jacobian(move(self.reference, proposal), proposal_center)
        metric = np.einsum("nij,nik->jk", jac, jac) / len(self.reference)
        initial = support.basis.T @ metric @ innovation
        solution = least_squares(
            objective,
            initial,
            max_nfev=self.config.max_evaluations,
            ftol=self.config.solver_tolerance,
            xtol=self.config.solver_tolerance,
            gtol=self.config.solver_tolerance,
        )
        candidate = pose_with_delta(prior, center, support.basis @ solution.x)
        before, after = self.bounded_residual(prior), self.bounded_residual(candidate)
        accepted = bool(solution.success and np.dot(after, after) < np.dot(before, before))
        detail.update(
            converged=bool(solution.success),
            evaluations=solution.nfev + proposal_detail["evaluations"],
            visual_changed=accepted,
        )
        return (candidate if accepted else prior.copy()), detail


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


class CompleteSurfaceFit(SupportedSurfaceFit):
    """Conservative repair: require the visible extent of the saved surface.

    Test all reference principal directions, not a named object axis. Noise can
    move each end of an extent by the depth error budget. This does not prove
    full surface identity or resolve symmetry; it excludes the truncated views
    that caused this experiment's blind attachment updates.
    """

    def covers_reference(self, pose):
        _, _, axes = np.linalg.svd(self.reference - self.reference.mean(0), full_matrices=False)
        # Detached depth patches must not manufacture complete-object coverage.
        # The input cleaner already identifies the largest connected surface;
        # use it for admission even when fitting considers several visible parts.
        local = (self.observation["obs"]["points"] - pose[:3, 3]) @ pose[:3, :3]
        expected = self.reference @ axes.T
        observed = local @ axes.T
        tolerance = self.config.depth_tolerance_m
        return bool(np.all(np.ptp(observed, axis=0) + 2 * tolerance >= np.ptp(expected, axis=0)))

    def correct(self, prior):
        # Re-localization must leave a bad seed. The next camera observation
        # validates this proposal before it can change the attachment state.
        candidate, detail = WholeSurfaceFit.correct(self, prior)
        complete = self.covers_reference(candidate)
        detail.update(visual_changed=complete, complete_surface=complete, supported_modes=0)
        return (candidate if complete else prior.copy()), detail


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
    parser.add_argument(
        "--guarded", action="store_true", help="Use authoritative motion state and qualified correction"
    )
    parser.add_argument("--fusion-config", type=Path)
    parser.add_argument("--correction-mode", choices=("observable", "complete"), default="complete")
    args = parser.parse_args()
    baseline = np.load(args.baseline / "trace.npz")
    summary = json.loads((args.baseline / "summary.json").read_text())
    config = ShapeConfig(**summary["config"])
    fusion_config = (
        SurfaceMotionConfig(**json.loads(args.fusion_config.read_text()))
        if args.fusion_config
        else SurfaceMotionConfig()
    )
    state = RigidMotionState()
    pending_correction = PendingMotionCorrection()
    state_history = OrderedDict()
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
    poses, priors, runtime_poses, rows, motor_poses = [], [], [], [], []
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
        motor_poses.append(ee if ee is not None else np.full((4, 4), np.nan))
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
        direct_qualified = status == "observed"
        if args.guarded and args.correction_mode == "complete" and direct_qualified:
            direct_qualified = CompleteSurfaceFit(
                source, observed, k, config, fusion_config
            ).covers_reference(direct.pose)
        started = time.perf_counter()
        layers.clear()
        group_anchor = service.relationships.get(args.role)
        point_anchor = service.active.get(args.role, (None, None))[1]
        accepted = service.process_offline(
            frame,
            (direct,) if direct_qualified else (),
            {args.role: observed["mask"]},
            attached_role=args.role if held else None,
        )
        service_ms = (time.perf_counter() - started) * 1000
        runtime = accepted.get(args.role)
        tracked_pixels = next(
            (
                pixels
                for tracked_frame, _, pixels in service.motion_history.get(args.role, ())
                if tracked_frame == frame.frame_id
            ),
            np.empty((0, 2), dtype=int),
        )
        point_motion_supported = (
            int(observed["usable"][tracked_pixels[:, 1], tracked_pixels[:, 0]].sum())
            >= causal_config.min_inliers
        )
        runtime_poses.append(runtime.pose if runtime is not None else np.full((4, 4), np.nan))
        prior, prior_kind = None, "none"
        carrier, bridge_id, group_state = None, None, None
        if args.guarded:
            if held:
                bridge_id, carrier = "grasp", ee
                prior_kind = "gripper_attachment_prediction" if ee is not None else "none"
            elif group_anchor is not None and (
                groups := [p for p in layers if p.estimator == "environment_group_motion"]
            ):
                # A verified relationship owns the coordinate frame. Its motion
                # transports our accepted pose; its absolute visual anchor does
                # not replace it. An unverified support hypothesis is excluded.
                bridge_id = "group-verified"
                group_state = state_history.get(group_anchor.anchor_time_s)
                carrier = (
                    groups[-1].pose @ np.linalg.inv(group_anchor.anchor_pose)
                    if group_state is not None
                    else None
                )
                prior_kind = "environment_group_motion"
            elif (
                point_anchor is not None
                and point_motion_supported
                and (points := [p for p in layers if p.estimator == "object_point_motion"])
            ):
                bridge_id = "object-points"
                group_state = state_history.get(point_anchor.source_time_s)
                carrier = (
                    points[-1].pose @ np.linalg.inv(point_anchor.pose) if group_state is not None else None
                )
                prior_kind = "object_point_motion"
            elif not held and state.bridge_id is not None and state.bridge_id.startswith("group-"):
                # Retain the relationship for resumption, but publish no current
                # prediction when its measured motion is absent.
                bridge_id = state.bridge_id
            prior = state.predict(now, bridge_id, carrier, anchor_pose=group_state)
        elif held and ee is not None:
            if not previous_held and status == "observed":
                attachment.seed_prediction(ee, direct, now)
            attachment_state = attachment.state(now, max_age_s=None)
            if attachment_state is not None:
                prior = ee @ attachment_state.gripper_from_object
                prior_kind = "gripper_attachment_prediction"
                if args.propagate_filtered_held and previous_held and previous_ee is not None:
                    # Predict the previous *filtered state*, not an independent
                    # observation. The attachment's measurement time is never
                    # refreshed from this estimate and no confidence is added.
                    prior = ee @ np.linalg.inv(previous_ee) @ poses[-1]
                    prior_kind = "gripper_delta_from_filtered_state"
        elif not args.guarded and not held:
            groups = [p for p in layers if p.estimator == "environment_group_motion"]
            if groups:
                prior, prior_kind = groups[-1].pose, groups[-1].estimator
        previous_held = held
        previous_ee = ee
        fitter = WholeSurfaceFit(source, observed, k, config) if status == "observed" else None
        started = time.perf_counter()
        pose = direct.pose.copy()
        detail = {"evaluations": 0, "converged": True, "used_coarse": False}
        if args.guarded:
            if prior is not None:
                if fitter is not None:
                    fit_type = (
                        CompleteSurfaceFit if args.correction_mode == "complete" else SupportedSurfaceFit
                    )
                    guarded_fit = fit_type(source, observed, k, config, fusion_config)

                    def score(candidate, guarded_fit=guarded_fit):
                        residual = guarded_fit.bounded_residual(candidate)
                        return float(residual @ residual)

                    minimum_improvement = (fusion_config.depth_error_m / config.depth_tolerance_m) ** 2
                    # Gripper carrier coordinates persist throughout a grasp.
                    # Group motion is already independently verified by the
                    # causal service; do not validate across its changing anchor.
                    if held:
                        confirmed = pending_correction.check(
                            bridge_id, carrier, prior, score, minimum_improvement
                        )
                        if (
                            confirmed is not None
                            and isinstance(guarded_fit, CompleteSurfaceFit)
                            and not guarded_fit.covers_reference(confirmed)
                        ):
                            confirmed = None
                        pose = confirmed if confirmed is not None else prior.copy()
                        candidate, detail = guarded_fit.correct(pose)
                        pending_correction.propose(
                            bridge_id, carrier, pose, candidate, score, minimum_improvement
                        )
                        detail["visual_changed"] = confirmed is not None
                        detail["pending_correction"] = pending_correction.pending is not None
                    else:
                        pose, detail = guarded_fit.correct(prior)
                else:
                    pose = prior.copy()
                    pending_correction.pending = None
                state.accept(pose, measured=detail.get("visual_changed", False), carrier=carrier)
            elif (
                runtime is not None
                and runtime.pose is not None
                and runtime.source_time_s == now
                and direct_qualified
            ):
                # Reacquisition is explicit. Never bypass the causal service's
                # accepted result by substituting the raw cached fit.
                pose = runtime.pose.copy()
                state.accept(pose, measured=runtime.evidence == Evidence.OBSERVED, carrier=carrier)
            else:
                pose = np.full((4, 4), np.nan)
                pending_correction.pending = None
        elif prior is not None:
            constrained = (
                MotionSupportedSurfaceFit(source, observed, k, config, prior, args.motion_tolerance_m)
                if fitter is not None and args.motion_tolerance_m is not None
                else fitter
            )
            pose, detail = constrained.correct(prior) if constrained is not None else (prior.copy(), detail)
        correction_ms = (time.perf_counter() - started) * 1000
        fit_attempted = prior is not None and fitter is not None
        if args.guarded:
            fit_attempted = detail["evaluations"] > 0
        poses.append(pose)
        if args.guarded and np.isfinite(pose).all():
            state_history[now] = pose.copy()
            active_anchor = service.active.get(args.role, (None, None))[1]
            protected = {active_anchor.source_time_s} if active_anchor is not None else set()
            for stamp in list(state_history):
                if len(state_history) <= causal_config.history_frames + len(protected):
                    break
                if stamp not in protected:
                    del state_history[stamp]
        priors.append(prior if prior is not None else np.full((4, 4), np.nan))
        row = {
            "frame": int(frame_id),
            "seconds": now,
            "held": int(held),
            "status": status,
            "direct_qualified": int(direct_qualified),
            "point_motion_supported": int(point_motion_supported),
            "prior_source": prior_kind,
            "runtime_source": runtime.estimator if runtime is not None else "missing",
            "visual_shape_cost": fitter.score(direct.pose) if fitter else np.nan,
            "prior_shape_cost": fitter.score(prior) if fitter and prior is not None else np.nan,
            "refit_shape_cost": fitter.score(pose) if fitter and np.isfinite(pose).all() else np.nan,
            "service_ms": service_ms,
            "correction_ms": correction_ms,
            "fit_attempted": int(fit_attempted),
            "converged": int(fit_attempted and detail["converged"]),
            "used_coarse": int(detail["used_coarse"]),
            "supported_modes": detail.get("supported_modes", 0),
            "visual_changed": int(detail.get("visual_changed", False)),
            "pending_correction": int(detail.get("pending_correction", False)),
            "surface_matches": detail.get("surface_matches", 0),
            "output_available": int(np.isfinite(pose).all()),
            "output_source": (
                ("visual_corrected_" + prior_kind if detail.get("visual_changed") else prior_kind)
                if prior is not None
                else (
                    "reacquired_" + runtime.estimator
                    if np.isfinite(pose).all() and runtime is not None
                    else "lost"
                )
            )
            if args.guarded
            else "legacy_unqualified_fit",
            "last_visual_update_s": state.last_measurement_s if args.guarded else None,
            **(diagnostics(source, pose, observed, k) if fitter and np.isfinite(pose).all() else {}),
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
        end_effector_poses=motor_poses,
        reference_points=source,
        landmarks_reference=baseline["landmarks_reference"],
        statuses=[r["status"] for r in rows],
    )
    summary.update(
        {
            "method": "whole_surface_fit_with_recorded_motion_bridges",
            "motion_tolerance_m": args.motion_tolerance_m,
            "propagate_filtered_held": args.propagate_filtered_held,
            "guarded": args.guarded,
            "fusion_config": asdict(fusion_config) if args.guarded else None,
            "correction_mode": args.correction_mode if args.guarded else None,
            "statuses": dict(Counter(r["status"] for r in rows)),
            "optimizer_converged": sum(r["converged"] for r in rows),
            "optimizer_attempted": sum(r["fit_attempted"] for r in rows),
            "coarse_reinitializations": sum(r["used_coarse"] for r in rows),
            "causal_config": asdict(causal_config),
            "prior_sources": dict(Counter(r["prior_source"] for r in rows)),
            "runtime_sources": dict(Counter(r["runtime_source"] for r in rows)),
            "anchor_outcomes": service.anchor_outcomes_by_role,
            "median_service_ms": float(np.median([r["service_ms"] for r in rows])),
            "median_correction_ms": float(np.median([r["correction_ms"] for r in rows])),
            "p95_correction_ms": float(np.percentile([r["correction_ms"] for r in rows], 95)),
            "warning": "Motion prediction assumes no slip. Extent coverage and a second observation do not prove landmark identity or full pose observability. This offline experiment does not change the live backend. Failed observable-subspace and legacy ablations are retained for comparison.",
        }
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
