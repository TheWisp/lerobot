"""Small offline ablation: identical UFM motion, with and without surface correction.

Uses the archived core directly. These are candidate poses, without publication
gates or gripper prediction. Optional scene groups supply recorded motion hints.
Fit residuals are not accuracy.
Run with python -m benchmarks.flash_tracker_experiment --help.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from benchmarks.flash_experiment_groups import GroupConfig, GroupHints
from benchmarks.flash_experiment_inputs import samples
from benchmarks.flash_experiment_labels import load_pose_labels
from benchmarks.flash_flow_paths import DenseFlow, FlowPaths
from benchmarks.flash_full_pose_score import score_transforms, summarize
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig, GrowingSurfaceSession
from lerobot.policies.flash.scene_groups import ScenePointTracker
from lerobot.policies.flash.shared_image_tracking import SharedFrameTracker, SharedMatchingConfig
from lerobot.policies.flash.surface_acceleration import SearchPool, compiled_kernels
from lerobot.policies.flash.surface_correspondence import crop_pair, measured_pairs, robust_fit, ufm_pairs
from lerobot.policies.flash.surface_tracking import SurfaceTrackingSession, load_ufm
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig


class MotionChain:
    """A failed link cannot silently become a stationary measurement."""

    def __init__(self):
        self.pose = None
        self.broken = False

    def update(self, prepared):
        if prepared["status"] is not None:
            # prepare() keeps the last usable image in this case. A future
            # correspondence can still bridge to it, but this frame is unknown.
            return None
        if self.broken:
            return None
        if self.pose is None:
            self.pose = np.eye(4)
        elif prepared["step"] is None:
            # prepare() advanced its image even though this pair did not fit.
            self.broken = True
            return None
        else:
            self.pose = prepared["step"] @ self.pose
        return self.pose.copy()


class RetainedMotionReference:
    """Experiment: retry from the last connected image after a missed match.

    This is a recovery hypothesis, not a pose acceptance rule. Long-gap UFM
    correspondences can still be wrong; the labelled cases measure that risk.
    """

    def __init__(self, matcher, config, *, record_correspondences=False, shared_matching=None):
        self.matcher, self.config = matcher, config
        self.shared_matching = shared_matching
        self.record_correspondences = record_correspondences
        self.evidence = None
        self.paths = FlowPaths()
        self.reference = None
        self.reference_frame = None
        self.pose = np.eye(4)

    def update(self, prepared, previous):
        self.evidence = None
        diagnostics = {"reference_frame": self.reference_frame, "extra_match_ms": 0.0}
        if prepared["status"] is not None:
            return None, diagnostics
        current = prepared["current"]
        if self.reference is not None:
            step = prepared["step"]
            evidence = prepared.get("correspondences")
            if self.reference is not previous:
                started = time.perf_counter()
                a, b, crop = crop_pair(self.reference, current, self.config)
                evidence = {} if self.record_correspondences else None
                kwargs = {"evidence": evidence} if evidence is not None else {}
                if self.shared_matching is not None:
                    source, target = self.shared_matching.pairs(
                        self.reference, current, prepared["intrinsics"], evidence=evidence
                    )
                else:
                    source, target = ufm_pairs(
                        self.matcher,
                        a,
                        b,
                        crop,
                        self.reference,
                        current,
                        prepared["intrinsics"],
                        self.config,
                        **kwargs,
                    )
                step, inliers, residual = robust_fit(source, target, self.config)
                diagnostics.update(
                    extra_match_ms=(time.perf_counter() - started) * 1000,
                    matches=len(source),
                    inliers=inliers,
                    residual_mm=float(residual * 1000) if np.isfinite(residual) else None,
                )
            if evidence is not None:
                self.evidence = match_record(
                    evidence,
                    self.reference,
                    current,
                    prepared["intrinsics"],
                    step,
                    self.config,
                    paths=self.paths,
                )
            if step is None:
                return None, diagnostics
            self.pose = step @ self.pose
        self.reference = current
        self.reference_frame = prepared["frame_id"]
        return self.pose.copy(), diagnostics


def match_record(evidence, source, target, intrinsics, step, config, *, paths=None):
    """Save measured matches and fit agreement; never synthesize matches from a pose."""
    stages = evidence["stage"].copy()
    rows = np.flatnonzero(stages == 3)
    if step is not None and len(rows):
        a, b = measured_pairs(
            source, target, evidence["source_uv"][rows], evidence["target_uv"][rows], intrinsics
        )
        if len(a) != len(rows):
            raise ValueError("captured correspondence filter disagrees with the estimator")
        errors = np.linalg.norm(a @ step[:3, :3].T + step[:3, 3] - b, axis=1)
        stages[rows[errors < config.ransac_distance_m]] = 4
    record = {
        "source_frame": source["frame_id"],
        "target_frame": target["frame_id"],
        "source_uv": np.round(evidence["source_uv"], 3).tolist(),
        "target_uv": np.round(evidence["target_uv"], 3).tolist(),
        "stage": stages.tolist(),
        "fit_available": step is not None,
    }
    if "track_ids" in evidence:
        record["backend"] = evidence["backend"]
        record["track_ids"] = evidence["track_ids"].tolist()
        record["paths"] = [
            [int(i), int((i * 137) % 280), *np.round(a, 3).tolist(), *np.round(b, 3).tolist(), int(s)]
            for i, a, b, s in zip(
                evidence["track_ids"], evidence["source_uv"], evidence["target_uv"], stages, strict=True
            )
        ]
        record["path_sampling"] = "shared_point_identities"
    elif paths is not None and "flow" in evidence:
        crop = evidence["crop"]
        field = (
            DenseFlow(record, evidence["flow"], crop[:2], crop[2] / config.input_size_px)
            if len(stages)
            else None
        )
        record["paths"] = paths.update(record, field=field)
        record["path_sampling"] = "full_ufm_output"
    return record


def quantiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return (
        {"median": float(np.median(values)), "p90": float(np.quantile(values, 0.9))} if len(values) else None
    )


def score_case(case, rows, references, reference_frames):
    """Only this function reads pose annotations, after all inference finishes."""
    labels, provenance = load_pose_labels(case, rows)
    report = {
        "case": case["id"],
        "frames": len(rows),
        "accuracy_labels": bool(labels),
        "evaluation": provenance,
        "roles": {},
    }
    for role in case["roles"]:
        truth = labels.get(role)
        details = [row["objects"][role] for row in rows]
        first = reference_frames.get(role)
        result = {"reference_frame": first, "accuracy_labels": truth is not None, "methods": {}}
        if first is None:
            report["roles"][role] = result
            continue
        available_by_method = {}
        errors = {}
        methods = ["ufm_chain", "surface_corrected"]
        if "ufm_retained_reference" in details[0]:
            methods.append("ufm_retained_reference")
        methods += [
            name
            for name in ("conditioned_fixed", "conditioned_growing", "group_fixed", "group_growing")
            if name in details[0]
        ]
        for name in methods:
            poses = [item[name] for item in details[first:]]
            available = np.asarray([p is not None for p in poses])
            available_by_method[name] = available
            metrics = {
                "available_after_initialization": int(available[1:].sum()),
                "total_after_initialization": len(poses) - 1,
            }
            if truth is not None:
                transforms = np.stack([np.eye(4) if p is None else p for p in poses])
                relative_truth = truth[first:] @ np.linalg.inv(truth[first])
                error = score_transforms(transforms, available, relative_truth, references[role])
                metrics["pose_error"] = summarize(available, *error)
                errors[name] = error[0]
            result["methods"][name] = metrics
        if truth is not None:
            paired = available_by_method["ufm_chain"] & available_by_method["surface_corrected"]
            paired[0] = False
            result["paired_frames"] = int(paired.sum())
            result["paired_point_error_mm"] = {
                name: quantiles(error[paired]) for name, error in errors.items()
            }
            if "conditioned_growing" in errors:
                compared = (
                    available_by_method["conditioned_fixed"] & available_by_method["conditioned_growing"]
                )
                compared[0] = False
                result["growth_comparison"] = {
                    "paired_frames": int(compared.sum()),
                    "point_error_mm": {
                        name: quantiles(errors[name][compared])
                        for name in ("conditioned_fixed", "conditioned_growing")
                    },
                }
        result["timing_ms"] = {
            key: quantiles([item[key] for item in details[first + 1 :]])
            for key in ("prepare_ms", "finish_ms")
        }
        for name in ("conditioned_fixed", "conditioned_growing", "group_fixed", "group_growing"):
            if name in methods:
                result["timing_ms"][name] = quantiles([item[name + "_ms"] for item in details[first + 1 :]])
                result["methods"][name]["max_map_points"] = max(
                    item[name + "_details"]["map_points"] for item in details
                )
                result["methods"][name]["pose_sources"] = {
                    status: sum(item[name + "_status"] == status for item in details[first + 1 :])
                    for status in ("observed_candidate", "predicted", "lost")
                }
        if "group_fixed" in methods:
            result["timing_ms"]["scene_point_tracking_ms"] = quantiles(
                [row["scene_point_tracking_ms"] for row in rows[first + 1 :]]
            )
        report["roles"][role] = result
    return report


def code_snapshot(output):
    """Preserve the uncommitted source used by an experiment, without a commit."""
    repo = Path(__file__).resolve().parents[1]
    source_files = sorted((repo / "src/lerobot/policies/flash").glob("*.py"))
    source_files += [
        Path(__file__).resolve(),
        repo / "benchmarks/flash_experiment_inputs.py",
        repo / "benchmarks/flash_experiment_labels.py",
        repo / "benchmarks/flash_full_pose_score.py",
        repo / "benchmarks/flash_flow_paths.py",
        repo / "benchmarks/flash_experiment_browser.py",
        repo / "benchmarks/flash_rebuild_examples.py",
        repo / "benchmarks/flash_experiment_cache.py",
        repo / "benchmarks/flash_experiment_groups.py",
        repo / "src/lerobot/gui/flash_group_overlay.py",
        repo / "src/lerobot/gui/group_wireframe.py",
    ]
    hashes = {}
    for file in source_files:
        data = file.read_bytes()
        relative = file.relative_to(repo)
        dest = output / "code" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        hashes[str(relative)] = hashlib.sha256(data).hexdigest()
    (output / "worktree.patch").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=repo)
    )
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "source_sha256": hashes,
    }


def run_case(
    case,
    output,
    model,
    config,
    kernels,
    search_pool,
    *,
    retain_motion_reference=False,
    record_correspondences=False,
    surface_experiment=None,
    group_experiment=None,
    progress=None,
    shared_matching_config=None,
):
    output.mkdir(parents=True, exist_ok=False)
    shared = SharedFrameTracker(shared_matching_config, model) if shared_matching_config else None
    sessions = {
        role: SurfaceTrackingSession(
            model,
            observation_config=config.observation,
            shape_config=config.shape,
            motion_config=config.motion,
            fit_kernels=kernels,
            search_pool=search_pool,
            record_correspondences=record_correspondences,
            shared_matching=shared,
        )
        for role in case["roles"]
    }
    chains = {role: MotionChain() for role in sessions}
    paths = {role: FlowPaths() for role in sessions}
    retained = {
        role: RetainedMotionReference(
            model, config.motion, record_correspondences=record_correspondences, shared_matching=shared
        )
        for role in sessions
    }
    if group_experiment is not None and surface_experiment is None:
        raise ValueError("group variants require the surface experiment")
    variants = [("conditioned_fixed", False), ("conditioned_growing", True)]
    if group_experiment is not None:
        variants += [("group_fixed", False), ("group_growing", True)]
    candidates = (
        {
            name: {
                role: GrowingSurfaceSession(
                    config.shape,
                    GrowingSurfaceConfig(
                        **{
                            **asdict(surface_experiment),
                            "grow": grow,
                            "retain_unobserved_pose": name.startswith("group_"),
                        }
                    ),
                    fit_kernels=kernels,
                    search_pool=search_pool,
                )
                for role in sessions
            }
            for name, grow in variants
        }
        if surface_experiment is not None
        else {}
    )
    scene_tracker = ScenePointTracker() if group_experiment is not None and shared is None else None
    groups = {
        name: {
            role: GroupHints(group_experiment, shared_image_points=shared is not None) for role in sessions
        }
        for name in candidates
        if name.startswith("group_")
    }
    digest, rows, references, reference_frames = hashlib.sha256(), [], {}, {}
    started = time.perf_counter()
    (output / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    # A JSONL survives a stopped process. summary.json exists only after completion.
    with (
        (output / "frames.jsonl").open("w", buffering=1) as journal,
        (
            gzip.open(output / "correspondences.jsonl.gz", "wt") if record_correspondences else nullcontext()
        ) as match_journal,
    ):
        for frame, masks in samples(case):
            for value in (
                frame.rgb,
                frame.depth,
                frame.intrinsics,
                np.asarray(frame.source_time_s),
                *masks.values(),
            ):
                a = np.asarray(value)
                digest.update(str((a.shape, a.dtype)).encode() + a.tobytes())
            row = {"frame": frame.frame_id, "time_s": frame.source_time_s, "objects": {}}
            if shared is not None:
                scene = shared.update(frame, masks)
                row["shared_matching"] = dict(shared.diagnostics)
                row["scene_point_tracking_ms"] = shared.diagnostics["total_ms"]
            if scene_tracker is not None:
                begin = time.perf_counter()
                scene = scene_tracker.update(frame)
                row["scene_point_tracking_ms"] = (time.perf_counter() - begin) * 1000
            match_row = {"frame": frame.frame_id, "objects": {}}
            for role, session in sessions.items():
                begin = time.perf_counter()
                previous = session.previous
                prepared = session.prepare(
                    frame.rgb,
                    frame.depth.astype(np.float32) * frame.depth_m_per_unit,
                    masks[role],
                    frame.intrinsics,
                    frame.frame_id,
                )
                prepare_ms = (time.perf_counter() - begin) * 1000
                pair = None
                if prepared.get("correspondences") is not None:
                    pair = match_record(
                        prepared["correspondences"],
                        previous,
                        prepared["current"],
                        frame.intrinsics,
                        prepared["step"],
                        config.motion,
                        paths=paths[role],
                    )
                match_row["objects"][role] = {"pair": pair}
                chain = chains[role].update(prepared)
                begin = time.perf_counter()
                fitted = session.finish(prepared)
                finish_ms = (time.perf_counter() - begin) * 1000
                if role not in references and session.reference is not None:
                    references[role] = session.reference.copy()
                    reference_frames[role] = frame.frame_id
                row["objects"][role] = {
                    "ufm_chain": chain.tolist() if chain is not None else None,
                    "surface_corrected": fitted.get("camera_from_reference"),
                    "status": fitted["status"],
                    "motion_chain_broken": chains[role].broken,
                    "prepare_ms": prepare_ms,
                    "finish_ms": finish_ms,
                    "diagnostics": fitted["diagnostics"],
                }
                if retain_motion_reference:
                    retry_pose, retry_diagnostics = retained[role].update(prepared, previous)
                    match_row["objects"][role]["retry"] = retained[role].evidence
                    row["objects"][role].update(
                        ufm_retained_reference=retry_pose.tolist() if retry_pose is not None else None,
                        retained_reference_diagnostics=retry_diagnostics,
                    )
                for name, tracks in candidates.items():
                    begin = time.perf_counter()
                    track = tracks[role]
                    candidate_input = prepared
                    # A missed frame cannot silently become the source of the next
                    # motion estimate. Retry from this track's connected image.
                    if (
                        prepared["status"] is None
                        and track.connected_observation is not None
                        and prepared.get("match_source_frame") != track.connected_frame
                    ):
                        candidate_input = dict(prepared)
                        old = track.connected_observation
                        current = prepared["current"]
                        if shared is not None:
                            source, target = shared.pairs(old, current, prepared["intrinsics"])
                        else:
                            a, b, crop = crop_pair(old, current, config.motion)
                            source, target = ufm_pairs(
                                model, a, b, crop, old, current, prepared["intrinsics"], config.motion
                            )
                        step, _, _ = robust_fit(source, target, config.motion)
                        candidate_input.update(
                            step=step, point_pairs=(source, target), match_source_frame=track.connected_frame
                        )
                    hint = None
                    if name in groups:
                        hint, group_details = groups[name][role].predict(
                            frame,
                            scene,
                            masks,
                            role,
                            track.reference_frame,
                            **({"object_usable": prepared["current"]["usable"]} if shared else {}),
                            observation=prepared["current"] if prepared["status"] is None else None,
                            shape_config=config.shape,
                        )
                    candidate_check = (
                        partial(
                            groups[name][role].compare_candidate,
                            frame,
                            masks[role],
                            hint,
                            observation=prepared["current"] if track.config.surface_priority else None,
                            shape_config=config.shape,
                        )
                        if hint is not None
                        else None
                    )
                    result = track.update(
                        candidate_input,
                        frame.source_time_s,
                        hint=hint,
                        candidate_check=candidate_check,
                        base_from_camera=frame.base_from_camera,
                    )
                    details = result["diagnostics"]
                    if name in groups:
                        groups[name][role].observe(
                            frame,
                            scene,
                            masks,
                            role,
                            result,
                            track.reference,
                            **({"object_usable": prepared["current"]["usable"]} if shared else {}),
                        )
                        details["group"] = group_details
                    row["objects"][role].update(
                        {
                            name: result["camera_from_reference"],
                            name + "_ms": (time.perf_counter() - begin) * 1000,
                            name + "_status": result["status"],
                            name + "_details": details,
                        }
                    )
                    # Original sampled camera measurements and their placement are
                    # separate from derived clouds, so an addition can be undone.
                    archive = output / "surface_patches" / name / role
                    for event in details["map_events"]:
                        if event["event"] not in ("pending", "added"):
                            continue
                        patch = track.store.patches[event["frame"]]
                        archive.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            archive / f"{patch.frame:06d}.npz",
                            measured=patch.measured,
                            camera_from_reference=patch.camera_from_reference,
                            fitted_map_version=patch.fitted_map_version,
                        )
                    if details["map_version_after"] != details["map_version_used"]:
                        models = output / "surface_models" / name / role
                        models.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            models / f"{track.store.version:06d}.npz", points=track.store.points
                        )
            if match_journal is not None:
                match_journal.write(json.dumps(match_row, allow_nan=False, separators=(",", ":")) + "\n")
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            rows.append(row)
            if progress is not None:
                progress(frame.frame_id + 1, case["expected_frames"])
            if frame.frame_id % 50 == 0:
                print(f"{case['id']}: {frame.frame_id + 1}/{case['expected_frames']}", flush=True)
    report = score_case(case, rows, references, reference_frames)
    report.update(input_sha256=digest.hexdigest(), offline_wall_s=time.perf_counter() - started)
    if shared is not None:
        report["shared_matching"] = {
            "config": asdict(shared.config),
            "calls": shared.calls,
            "timing_ms": quantiles([r["shared_matching"]["total_ms"] for r in rows[1:]]),
        }
    np.savez_compressed(output / "reference_points.npz", **references)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--case", action="append", help="Repeat to select cases; omit for the whole suite")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("flash_tracking_config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--surface-experiment-config",
        type=Path,
        help="Opt in to fixed/growing surfaces with retained image evidence; JSON GrowingSurfaceConfig",
    )
    parser.add_argument(
        "--group-experiment-config",
        type=Path,
        help="Also compare fixed/growing surfaces with scene groups; JSON GroupConfig",
    )
    parser.add_argument(
        "--retain-motion-reference",
        action="store_true",
        help="Also test retrying from the last successfully connected image",
    )
    parser.add_argument(
        "--record-correspondences",
        action="store_true",
        help="Save sampled raw UFM matches and filtering evidence for playback",
    )
    parser.add_argument(
        "--shared-matching-config",
        type=Path,
        help="Opt in to one LK/UFM matching pass for objects and groups; JSON SharedMatchingConfig",
    )
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    cases = [c for c in suite["cases"] if args.case is None or c["id"] in args.case]
    if not cases or (args.case and len(cases) != len(set(args.case))):
        parser.error("select existing, distinct case IDs")
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    config = VisionPipelineConfig.from_dict(json.loads(args.config.read_text())["vision_pipeline"])
    surface_experiment = (
        GrowingSurfaceConfig(**json.loads(args.surface_experiment_config.read_text()))
        if args.surface_experiment_config
        else None
    )
    group_experiment = (
        GroupConfig.from_dict(json.loads(args.group_experiment_config.read_text()))
        if args.group_experiment_config
        else None
    )
    shared_config = (
        SharedMatchingConfig.from_dict(json.loads(args.shared_matching_config.read_text()))
        if args.shared_matching_config
        else None
    )
    provenance = code_snapshot(args.output)
    provenance.update(
        config=asdict(config),
        cases=cases,
        experiment="candidate poses: raw UFM chain vs archived surface correction",
        masks="recorded SAM for rig; supplied dataset masks for external",
        optional_motion_sources=["scene_points"] if group_experiment else [],
        certainty="uncalibrated diagnostics only",
        retain_motion_reference=args.retain_motion_reference,
        record_correspondences=args.record_correspondences,
        surface_experiment=asdict(surface_experiment) if surface_experiment else None,
        group_experiment=asdict(group_experiment) if group_experiment else None,
        shared_matching=asdict(shared_config) if shared_config else None,
        packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    )
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    cv2.setNumThreads(1)
    started = time.perf_counter()
    model = load_ufm() if shared_config is None or shared_config.backend == "ufm" else None
    kernels = compiled_kernels() if config.compiled_surface_fit else None
    pool = SearchPool(config.shape.batch_query_workers) if config.reuse_surface_search else None
    startup = time.perf_counter() - started
    results = []
    try:
        for case in cases:
            results.append(
                run_case(
                    case,
                    args.output / case["id"],
                    model,
                    config,
                    kernels,
                    pool,
                    retain_motion_reference=args.retain_motion_reference,
                    record_correspondences=args.record_correspondences,
                    surface_experiment=surface_experiment,
                    group_experiment=group_experiment,
                    shared_matching_config=shared_config,
                )
            )
    finally:
        if pool is not None:
            pool.close()
    (args.output / "summary.json").write_text(
        json.dumps({"startup_s": startup, "cases": results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
