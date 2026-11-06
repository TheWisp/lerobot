"""Recorded RGB-D input, archival and scoring around the shared frame session.

Comparison methods select image-only or surface/group estimation. Live camera
debugging uses the same core; annotations are read only after inference for scoring.
Fit residuals are not accuracy. This runner does not control a robot.
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
from itertools import islice
from pathlib import Path

import cv2
import numpy as np

from benchmarks.flash_experiment_groups import GroupConfig
from benchmarks.flash_experiment_inputs import samples
from benchmarks.flash_experiment_labels import load_pose_labels
from benchmarks.flash_full_pose_score import score_transforms, summarize
from benchmarks.flash_surface_score import save_surface_version, score_surface_trace
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig
from lerobot.policies.flash.shared_image_tracking import SharedMatchingConfig
from lerobot.policies.flash.surface_acceleration import SearchPool, compiled_kernels
from lerobot.policies.flash.surface_frame_session import (
    MotionChain as MotionChain,
    RetainedMotionReference as RetainedMotionReference,
    SurfaceFrameSession,
    match_record as match_record,
)
from lerobot.policies.flash.surface_tracking import load_ufm
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig


def quantiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return (
        {"median": float(np.median(values)), "p90": float(np.quantile(values, 0.9))} if len(values) else None
    )


def score_case(case, rows, references, reference_frames, *, run=None):
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
        methods = [name for name in ("ufm_chain", "surface_corrected") if name in details[0]]
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
            metrics["surface_error"] = score_surface_trace(
                case.get("evaluation_surfaces", {}).get(role),
                truth[first:] if truth is not None else None,
                details[first:],
                name,
                references[role],
                run=run,
                role=role,
            )
            result["methods"][name] = metrics
        if truth is not None and "ufm_chain" in errors and "surface_corrected" in errors:
            paired = available_by_method["ufm_chain"] & available_by_method["surface_corrected"]
            paired[0] = False
            result["paired_frames"] = int(paired.sum())
            result["paired_point_error_mm"] = {
                name: quantiles(error[paired]) for name, error in errors.items()
            }
            if "conditioned_growing" in errors and "conditioned_fixed" in errors:
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
            if key in details[0]
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
        if "group_fixed" in methods or "group_growing" in methods:
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
        repo / "benchmarks/flash_surface_score.py",
        repo / "src/lerobot/gui/flash_pose_overlay.py",
        repo / "benchmarks/flash_flow_paths.py",
        repo / "benchmarks/flash_experiment_browser.py",
        repo / "benchmarks/flash_rebuild_examples.py",
        repo / "benchmarks/flash_experiment_cache.py",
        repo / "benchmarks/flash_experiment_groups.py",
        repo / "benchmarks/flash_pipeline_settings.py",
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
    surface_methods=None,
    single_method=None,
    record_surfaces=True,
    mask_source=None,
    max_frames=None,
    parallel_objects=True,
):
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError("max_frames must be a positive integer")
    output.mkdir(parents=True, exist_ok=False)
    engine = SurfaceFrameSession(
        case["roles"],
        model,
        config,
        kernels,
        search_pool,
        retain_motion_reference=retain_motion_reference,
        record_correspondences=record_correspondences,
        surface_experiment=surface_experiment,
        group_experiment=group_experiment,
        shared_matching_config=shared_matching_config,
        surface_methods=surface_methods,
        single_method=single_method,
        parallel_objects=parallel_objects,
    )
    shared = engine.shared
    digest, rows = hashlib.sha256(), []
    references, reference_frames = engine.references, engine.reference_frames
    started = time.perf_counter()
    (output / "case.json").write_text(json.dumps(case, indent=2) + "\n")

    def archive_surface(name, role, track, details):
        # Performance runs can omit patch history, but must retain the actual
        # versioned surfaces needed by the same accuracy scorer as full runs.
        if not record_surfaces and role not in case.get("evaluation_surfaces", {}):
            return
        # Original sampled camera measurements and their placement are
        # separate from derived clouds, so an addition can be undone.
        archive = output / "surface_patches" / name / role
        for event in details["map_events"] if record_surfaces else ():
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
        save_surface_version(output, name, role, track, details)

    # A JSONL survives a stopped process. summary.json exists only after completion.
    with (
        (output / "frames.jsonl").open("w", buffering=1) as journal,
        (
            gzip.open(output / "correspondences.jsonl.gz", "wt") if record_correspondences else nullcontext()
        ) as match_journal,
        engine,
    ):
        for frame, masks in islice(samples(case), max_frames):
            # Disk decoding and output serialization are outside this timer.
            # A supplied live mask source is inside it; saved masks cost zero.
            mask_started = time.perf_counter()
            if mask_source is not None:
                measured_masks = mask_source.masks(frame)
                masks = {
                    role: measured_masks.get(role, np.zeros(frame.depth.shape, bool))
                    for role in engine.sessions
                }
            mask_ms = (time.perf_counter() - mask_started) * 1000 if mask_source else 0.0
            for value in (
                frame.rgb,
                frame.depth,
                frame.intrinsics,
                np.asarray(frame.source_time_s),
                *masks.values(),
            ):
                a = np.asarray(value)
                digest.update(str((a.shape, a.dtype)).encode() + a.tobytes())
            row, match_row = engine.step(
                frame,
                masks,
                mask_ms=mask_ms,
                archive_callback=archive_surface,
            )
            if match_journal is not None:
                match_journal.write(json.dumps(match_row, allow_nan=False, separators=(",", ":")) + "\n")
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            rows.append(row)
            if progress is not None:
                progress(frame.frame_id + 1, case["expected_frames"])
            if frame.frame_id % 50 == 0:
                print(f"{case['id']}: {frame.frame_id + 1}/{case['expected_frames']}", flush=True)
    report = score_case(case, rows, references, reference_frames, run=output)
    report.update(input_sha256=digest.hexdigest(), offline_wall_s=time.perf_counter() - started)
    report["processing"] = {
        "method": single_method or "all_comparisons",
        "masks": "computed" if mask_source is not None else "recorded",
        "includes": "segmentation if computed, shared matching, observation preparation, pose/group/map updates",
        "excludes": "disk input/output, scoring, playback, camera capture and display",
        "startup": "models warmed before run except lazy SAM loading in first_frame_ms; steady excludes first frame",
        "first_frame_ms": rows[0]["processing_ms"],
        "steady_ms": quantiles([r["processing_ms"] for r in rows[1:]]),
        "max_ms": max(r["processing_ms"] for r in rows[1:]) if len(rows) > 1 else None,
        "slower_than_source_interval": sum(
            b["processing_ms"] > 1000 * (b["time_s"] - a["time_s"])
            for a, b in zip(rows, rows[1:], strict=False)
        ),
        "intervals": len(rows) - 1,
    }
    report["surface_experiment"] = asdict(surface_experiment) if surface_experiment else None
    report["surface_registration"] = asdict(config.shape.registration)
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
