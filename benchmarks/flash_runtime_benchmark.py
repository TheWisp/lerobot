"""Replay one tracker or the camera-debugger backend on identical recorded inputs.

Saved masks isolate fitting changes. --sam-prompts runs the shared live SAM3
source instead; it changes the input masks and is a separate end-to-end trial.
Neither mode measures camera capture, controller latency, or GUI rendering.
--settings pins one existing pipeline recipe across scenes; --profile attributes
experimental fit costs separately from uninstrumented performance comparisons.
"""

import argparse
import hashlib
import json
import time
from contextlib import ExitStack
from dataclasses import replace
from functools import wraps
from itertools import islice
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from benchmarks.flash_experiment_groups import GroupConfig
from benchmarks.flash_experiment_inputs import samples
from benchmarks.flash_pipeline_settings import from_recording, validate
from benchmarks.flash_tracker_experiment import code_snapshot, quantiles, run_case, score_case
from lerobot.policies.flash.appearance import Sam3MaskSource
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig
from lerobot.policies.flash.shared_image_tracking import SharedMatchingConfig
from lerobot.policies.flash.surface_acceleration import SearchPool, compiled_kernels
from lerobot.policies.flash.surface_tracking import load_ufm
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig, build_vision_backend


class FitProfile(ExitStack):
    """Opt-in wall-time attribution for the serial experiment, without changing results.

    Exclusive times subtract instrumented children and can be summed. Timed
    production comparisons must run separately with this instrumentation off.
    """

    def __init__(self):
        super().__init__()
        self.stack, self.stats = [], {}

    def instrument(self, owner, name, label):
        original = getattr(owner, name)

        @wraps(original)
        def measured(*args, **kwargs):
            active = [time.perf_counter(), 0.0]
            self.stack.append(active)
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - active[0]
                self.stack.pop()
                if self.stack:
                    self.stack[-1][1] += elapsed
                row = self.stats.setdefault(label, {"calls": 0, "inclusive_ms": 0.0, "exclusive_ms": 0.0})
                row["calls"] += 1
                row["inclusive_ms"] += 1000 * elapsed
                row["exclusive_ms"] += 1000 * (elapsed - active[1])

        self.enter_context(patch.object(owner, name, measured))

    def __enter__(self):
        from benchmarks.flash_experiment_groups import GroupHints
        from lerobot.policies.flash import (
            experimental_surface_session as session,
            surface_uncertainty as uncertainty,
        )
        from lerobot.policies.flash.surface_grounding import WholeSurfaceFit
        from lerobot.policies.flash.surface_kalman import PoseKalmanFilter

        super().__enter__()
        for name in (
            "__init__",
            "refine",
            "residual",
            "batch_residuals",
            "score",
            "_legacy_view_residual",
            "_projection_evidence",
            "_scene_depth_evidence",
        ):
            self.instrument(WholeSurfaceFit, name, "fitter." + name)
        for owner in (session, uncertainty):
            self.instrument(owner, "check_surface_contradiction", "geometry_check")
        self.instrument(uncertainty.SurfacePoseUncertainty, "__init__", "covariance")
        self.instrument(PoseKalmanFilter, "correct", "kalman_update")
        self.instrument(session.SurfaceStore, "extend", "surface_growth")
        for name in ("predict", "compare_candidate", "observe"):
            self.instrument(GroupHints, name, "group." + name)
        return self


def run_debugger(case, output, model, config, *, max_frames=None):
    """Replay the real debugger backend/runner; score only its published estimates.

    All recorded frames are delivered in order, including with frame overlap.
    Camera capture, optional asynchronous group fallback and SAM are excluded.
    """
    from lerobot.policies.flash.perception_step import PerceptionPipeline, PerceptionRunner
    from lerobot.policies.flash.tracking_debug_worker import debug_reference

    output.mkdir(parents=True, exist_ok=False)
    sequence = list(islice(samples(case), max_frames))
    masks = {frame.frame_id: values for frame, values in sequence}
    digest = hashlib.sha256()
    for frame, values in sequence:
        for value in (
            frame.rgb,
            frame.depth,
            frame.intrinsics,
            np.asarray(frame.source_time_s),
            *values.values(),
        ):
            a = np.asarray(value)
            digest.update(str((a.shape, a.dtype)).encode() + a.tobytes())

    class Masks:
        def masks(self, frame):
            return masks[frame.frame_id]

    backend = build_vision_backend(config, Masks(), matcher=model, local_reference=True)
    references = tuple(debug_reference(role, sequence[0][0]) for role in case["roles"])
    pipeline = PerceptionPipeline(backend, references, require_masks=True)
    source, acquired = iter(sequence), {}

    def take_frame():
        item = next(source, None)
        if item is None:
            return None
        frame = item[0]
        acquired[frame.frame_id] = time.perf_counter()
        return frame

    rows, reference_frames, reference_points = [], {}, {}
    previous_end, started = None, time.perf_counter()
    with PerceptionRunner(pipeline) as runner:
        while (item := runner.process_next(take_frame)) is not None:
            frame, result = item
            ended = time.perf_counter()
            row = {
                "frame": frame.frame_id,
                "time_s": frame.source_time_s,
                "latency_ms": 1000 * (ended - acquired[frame.frame_id]),
                "completion_interval_ms": 1000 * (ended - previous_end) if previous_end is not None else None,
                "objects": {},
            }
            previous_end = ended
            for estimate in result.estimates:
                role = estimate.role
                session = backend.sessions[role]
                if role not in reference_points and session.reference is not None:
                    reference_points[role] = session.reference.copy()
                    reference_frames[role] = len(rows)
                pose = None
                if estimate.pose is not None:
                    pose = (
                        np.linalg.inv(frame.base_from_camera)
                        @ estimate.pose
                        @ np.linalg.inv(backend.reference_camera_from_object[role])
                    ).tolist()
                detail = result.details[role]
                row["objects"][role] = {
                    "surface_corrected": pose,
                    "surface_corrected_status": estimate.evidence.value,
                    "candidate": detail.get("camera_from_reference"),
                    "accepted": detail["accepted"],
                    "failure": result.failures.get(role),
                    "diagnostics": detail["diagnostics"],
                }
            rows.append(row)
            if len(rows) % 100 == 0:
                print(f"debugger {case['id']}: {len(rows)}/{len(sequence)}", flush=True)
    elapsed = time.perf_counter() - started
    report = score_case(case, rows, reference_points, reference_frames)
    intervals = [r["completion_interval_ms"] for r in rows[1:]]
    report.update(
        input_sha256=digest.hexdigest(),
        config=config.to_dict(),
        scope="Real debugger backend, published poses only; saved masks, every frame, no async group fallback",
        processing={
            "completion_interval_ms": quantiles(intervals),
            "mean_completion_interval_ms": float(np.mean(intervals)) if intervals else None,
            "latency_ms": quantiles([r["latency_ms"] for r in rows[1:]]),
            "max_latency_ms": max(r["latency_ms"] for r in rows[1:]) if intervals else None,
            "elapsed_s": elapsed,
        },
    )
    (output / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "frames.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return report


def compare_runs(before: Path, after: Path, method: str) -> dict:
    """Exact-output regression check; it does not establish absolute pose accuracy."""
    old_summary = json.loads((before / "summary.json").read_text())
    new_summary = json.loads((after / "summary.json").read_text())
    if (old_summary["input_sha256"], old_summary["frames"]) != (
        new_summary["input_sha256"],
        new_summary["frames"],
    ):
        raise ValueError("Correctness comparison needs identical frames, depth and masks")
    changes = {"pose": [], "status": [], "surface_events": []}
    with (before / "frames.jsonl").open() as old, (after / "frames.jsonl").open() as new:
        for a, b in zip(old, new, strict=True):
            a, b = json.loads(a), json.loads(b)
            if a["frame"] != b["frame"] or set(a["objects"]) != set(b["objects"]):
                raise ValueError("Comparison frame IDs or object roles differ")
            for role in a["objects"]:
                x, y = a["objects"][role], b["objects"][role]
                for label, key in (("pose", method), ("status", method + "_status")):
                    if x[key] != y[key]:
                        changes[label].append([a["frame"], role])
                fields = ("map_events", "map_points", "map_version_used", "map_version_after")
                dx, dy = x.get(method + "_details", {}), y.get(method + "_details", {})
                if any(dx.get(k) != dy.get(k) for k in fields):
                    changes["surface_events"].append([a["frame"], role])
    return {
        "identical": not any(changes.values()),
        "changed_frame_objects": {key: len(values) for key, values in changes.items()},
        "first_changes": {key: values[:10] for key, values in changes.items()},
        "before_processing": old_summary.get("processing"),
        "after_processing": new_summary.get("processing"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settings", type=Path, help="One complete pipeline-settings JSON for every scene")
    parser.add_argument("--ufm-revision", help="Pin the UFM checkpoint for repeatable comparisons")
    parser.add_argument("--compare-to", type=Path, help="Previous single-method benchmark output")
    parser.add_argument("--frames", type=int, help="Profile only the first N consecutive frames")
    parser.add_argument("--engine", choices=("experiment", "debugger"), default="experiment")
    parser.add_argument(
        "--debugger-defaults",
        action="store_true",
        help="Use debugger defaults instead of recorded fit settings",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Disable object concurrency (and debugger frame overlap), preserving the estimator",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Attribute experimental fitting time; separate from uninstrumented timing",
    )
    parser.add_argument(
        "--method",
        choices=("conditioned_fixed", "conditioned_growing", "group_fixed", "group_growing"),
        default="group_growing",
    )
    parser.add_argument(
        "--sam-prompts",
        type=Path,
        help="JSON keyed by case ID: prompts_by_role and optional seed_points_by_role",
    )
    args = parser.parse_args()
    if args.frames is not None and args.frames < 1:
        parser.error("--frames must be positive")
    if args.engine == "debugger" and (args.profile or args.sam_prompts):
        parser.error("Debugger replay uses saved masks without profiling")
    if args.engine == "experiment" and args.debugger_defaults:
        parser.error("--debugger-defaults applies only to debugger replay")
    args.output.mkdir(parents=True, exist_ok=False)
    fixed_settings = validate(json.loads(args.settings.read_text())) if args.settings else None
    prompts = json.loads(args.sam_prompts.read_text()) if args.sam_prompts else None
    snapshot = code_snapshot(args.output)
    script = Path(__file__)
    relative = Path("benchmarks") / script.name
    (args.output / "code" / relative).write_bytes(script.read_bytes())
    snapshot["source_sha256"][str(relative)] = hashlib.sha256(script.read_bytes()).hexdigest()
    (args.output / "provenance.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    cv2.setNumThreads(1)
    model, results = None, []
    for index, source in enumerate(args.source_run):
        case = json.loads((source / "case.json").read_text())
        settings = fixed_settings or from_recording(
            json.loads((source.parent / "provenance.json").read_text()), [args.method]
        )
        config = VisionPipelineConfig.from_dict(settings["config"])
        if args.debugger_defaults:
            config = VisionPipelineConfig()
        if args.serial and args.engine == "debugger":
            config = replace(config, parallel_surface_objects=False, pipeline_surface_frames=False)
        if args.engine == "experiment" and settings["surface_experiment"] is None:
            parser.error(f"{source} has no surface experiment")
        started = time.perf_counter()
        shared = settings["shared_matching"]
        if model is None and (args.engine == "debugger" or shared is None or shared["backend"] == "ufm"):
            model = load_ufm(revision=args.ufm_revision)
        kernels = compiled_kernels() if config.compiled_surface_fit else None
        pool = (
            SearchPool(config.shape.batch_query_workers)
            if args.engine == "experiment" and config.reuse_surface_search
            else None
        )
        source_masks = None
        if prompts is not None:
            spec = prompts[case["id"]]
            if set(spec["prompts_by_role"]) != set(case["roles"]):
                parser.error("SAM prompts must cover exactly the scene's roles")
            source_masks = Sam3MaskSource(
                spec["prompts_by_role"],
                camera_id="recorded-rgbd",
                resolution=config.sam_resolution,
                seed_points_by_role=spec.get("seed_points_by_role"),
                detection_correction=config.detection_correction.control(offline=False),
            )
        startup_s = time.perf_counter() - started
        from threadpoolctl import threadpool_info

        destination = args.output / f"{index:03d}"
        destination.mkdir()
        (destination / "provenance.json").write_text(
            json.dumps(
                {
                    **settings,
                    "source_run": str(source),
                    "sam_prompts": prompts,
                    "engine": args.engine,
                    "ufm_revision": args.ufm_revision,
                    "effective_vision_config": config.to_dict(),
                    "compute_thread_pools": threadpool_info(),
                },
                indent=2,
            )
            + "\n"
        )
        try:
            if args.engine == "debugger":
                result = run_debugger(case, destination / case["id"], model, config, max_frames=args.frames)
            else:
                with FitProfile() if args.profile else ExitStack() as profile:
                    result = run_case(
                        case,
                        destination / case["id"],
                        model,
                        config,
                        kernels,
                        pool,
                        surface_experiment=GrowingSurfaceConfig(**settings["surface_experiment"]),
                        group_experiment=GroupConfig.from_dict(settings["group_experiment"])
                        if args.method.startswith("group_")
                        else None,
                        shared_matching_config=SharedMatchingConfig.from_dict(shared) if shared else None,
                        single_method=args.method,
                        record_surfaces=False,
                        mask_source=source_masks,
                        max_frames=args.frames,
                        parallel_objects=not args.serial and not args.profile,
                    )
                    if args.profile:
                        result["fit_profile"] = profile.stats
                        (destination / case["id"] / "profile.json").write_text(
                            json.dumps(profile.stats, indent=2) + "\n"
                        )
        finally:
            if pool is not None:
                pool.close()
        result["startup_s"] = startup_s
        if args.compare_to is not None:
            comparison = compare_runs(
                args.compare_to / f"{index:03d}" / case["id"],
                destination / case["id"],
                "surface_corrected" if args.engine == "debugger" else args.method,
            )
            result["comparison"] = comparison
        results.append(result)
        (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({"case": case["id"], "processing": result["processing"]}), flush=True)
    if any(not result.get("comparison", {}).get("identical", True) for result in results):
        raise SystemExit("Correctness regression: inspect summary.json before accepting the speedup")


if __name__ == "__main__":
    main()
