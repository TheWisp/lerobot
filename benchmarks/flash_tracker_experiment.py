"""Small offline ablation: identical UFM motion, with and without surface correction.

Uses the archived core directly. These are candidate poses, without publication
gates, group fallback or gripper prediction. Fit residuals are not accuracy.
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
from pathlib import Path

import cv2
import numpy as np

from benchmarks.flash_experiment_inputs import path, samples
from benchmarks.flash_flow_paths import DenseFlow, FlowPaths
from benchmarks.flash_full_pose_score import score_transforms, summarize
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

    def __init__(self, matcher, config, *, record_correspondences=False):
        self.matcher, self.config = matcher, config
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
    if paths is not None and "flow" in evidence:
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
    truth = None
    if case["kind"] == "ycb_annotated_pose":
        files = sorted((path(case["source_root"]) / "annotated_poses").glob("*.txt"))
        start = case["start_frame"]
        truth = np.stack([np.loadtxt(f).reshape(4, 4) for f in files[start : start + len(rows)]])
        if len(truth) != len(rows):
            raise ValueError("missing evaluation pose annotations")
    report = {"case": case["id"], "frames": len(rows), "accuracy_labels": truth is not None, "roles": {}}
    for role in case["roles"]:
        details = [row["objects"][role] for row in rows]
        first = reference_frames.get(role)
        result = {"reference_frame": first, "methods": {}}
        if first is None:
            report["roles"][role] = result
            continue
        available_by_method = {}
        errors = {}
        methods = ["ufm_chain", "surface_corrected"]
        if "ufm_retained_reference" in details[0]:
            methods.append("ufm_retained_reference")
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
        result["timing_ms"] = {
            key: quantiles([item[key] for item in details[first + 1 :]])
            for key in ("prepare_ms", "finish_ms")
        }
        report["roles"][role] = result
    return report


def code_snapshot(output):
    """Preserve the uncommitted source used by an experiment, without a commit."""
    repo = Path(__file__).resolve().parents[1]
    source_files = sorted((repo / "src/lerobot/policies/flash").glob("*.py"))
    source_files += [
        Path(__file__).resolve(),
        repo / "benchmarks/flash_experiment_inputs.py",
        repo / "benchmarks/flash_full_pose_score.py",
        repo / "benchmarks/flash_flow_paths.py",
        repo / "benchmarks/flash_experiment_browser.py",
        repo / "benchmarks/flash_rebuild_examples.py",
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
    progress=None,
):
    output.mkdir(parents=True, exist_ok=False)
    sessions = {
        role: SurfaceTrackingSession(
            model,
            observation_config=config.observation,
            shape_config=config.shape,
            motion_config=config.motion,
            fit_kernels=kernels,
            search_pool=search_pool,
            record_correspondences=record_correspondences,
        )
        for role in case["roles"]
    }
    chains = {role: MotionChain() for role in sessions}
    paths = {role: FlowPaths() for role in sessions}
    retained = {
        role: RetainedMotionReference(model, config.motion, record_correspondences=record_correspondences)
        for role in sessions
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
        "--retain-motion-reference",
        action="store_true",
        help="Also test retrying from the last successfully connected image",
    )
    parser.add_argument(
        "--record-correspondences",
        action="store_true",
        help="Save sampled raw UFM matches and filtering evidence for playback",
    )
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    cases = [c for c in suite["cases"] if args.case is None or c["id"] in args.case]
    if not cases or (args.case and len(cases) != len(set(args.case))):
        parser.error("select existing, distinct case IDs")
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    config = VisionPipelineConfig.from_dict(json.loads(args.config.read_text())["vision_pipeline"])
    provenance = code_snapshot(args.output)
    provenance.update(
        config=asdict(config),
        cases=cases,
        experiment="candidate poses: raw UFM chain vs archived surface correction",
        masks="recorded SAM for rig; supplied dataset masks for external",
        optional_motion_sources=[],
        certainty="uncalibrated diagnostics only",
        retain_motion_reference=args.retain_motion_reference,
        record_correspondences=args.record_correspondences,
        packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    )
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    cv2.setNumThreads(1)
    started = time.perf_counter()
    model = load_ufm()
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
