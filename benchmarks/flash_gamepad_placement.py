"""Recorded clear-view correctness checks for optional registration libraries.

The five visible white controls are evaluation annotations, never fitter inputs.
Their 2D/3D distances do not pretend to be a full 6D ground-truth pose. The four
round buttons are interchangeable when scored; the directional pad is unique.
Outline overlap is useful here because the same broad face is fully visible in
both frames. It is NOT a general full-model coverage requirement.
"""

import argparse
import hashlib
import json
import sys
from contextlib import ExitStack
from dataclasses import asdict, replace
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from benchmarks.flash_library_registration import LibraryConfig
from lerobot.policies.flash.surface_grounding import SurfaceFitView, WholeSurfaceFit, project
from lerobot.policies.flash.surface_observation import move, observation
from lerobot.policies.flash.surface_uncertainty import check_surface_contradiction
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

FIXTURE = Path(__file__).resolve().parents[1] / "tests/policies/flash/fixtures/gamepad_placement"
METHODS = ("baseline", "small_gicp", "open3d_plane", "open3d_color")


def load():
    metadata = json.loads((FIXTURE / "manifest.json").read_text())
    assert hashlib.sha256((FIXTURE / "data.npz").read_bytes()).hexdigest() == metadata["sha256"]
    with np.load(FIXTURE / "data.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    data.update({key: data[source] for key, source in metadata.get("array_aliases", {}).items()})
    return data, metadata


def prepare(data, metadata, frame):
    config = VisionPipelineConfig.from_dict(metadata["settings"]["config"])
    result = {key: data[f"{key}_{frame}"] for key in ("rgb", "depth", "mask")}
    result["obs"] = observation(
        result["depth"], result["mask"], data[f"intrinsics_{frame}"], asdict(config.observation)
    )
    return result


def feature_error(source, target, pose):
    """No penalty for permuting identical round buttons; never permutation of D-pad."""
    predicted = move(source, pose)
    distance = np.linalg.norm(predicted[1:, None] - target[None, 1:], axis=-1)
    a, b = linear_sum_assignment(distance)
    return np.r_[np.linalg.norm(predicted[0] - target[0]), distance[a, b]]


def evaluate(data, source_frame, frame, pose):
    if pose is None:
        return {"available": False}
    source = data[f"features_xyz_{source_frame}"]
    target = data[f"features_xyz_{frame}"]
    predicted_uv = project(move(source, pose), data[f"intrinsics_{frame}"])
    target_uv = data[f"features_uv_{frame}"]
    # Reuse assignment logic with a dummy Z for independent 2D error.
    error_px = feature_error(np.c_[predicted_uv, np.zeros(5)], np.c_[target_uv, np.zeros(5)], np.eye(4))
    return {
        "available": True,
        "mean_feature_error_mm": float(feature_error(source, target, pose).mean() * 1000),
        "mean_feature_error_px": float(error_px.mean()),
        "dpad_error_px": float(error_px[0]),
    }


def outline_score(reference, current, k, pose):
    predicted = move(reference, pose)
    if not np.isfinite(predicted).all() or np.any(predicted[:, 2] <= 0):
        return {"outline_iou": 0.0, "inside_mask_fraction": 0.0, "surface_p90_mm": None}
    uv = np.rint(project(predicted, k)).astype(np.int32)
    polygon = np.zeros(current["mask"].shape, np.uint8)
    cv2.fillConvexPoly(polygon, cv2.convexHull(uv), 1)
    overlap = np.count_nonzero(current["mask"] & (polygon > 0))
    return {
        "outline_iou": float(overlap / max(1, np.count_nonzero(current["mask"] | (polygon > 0)))),
        "inside_mask_fraction": float(overlap / max(1, np.count_nonzero(polygon))),
        "surface_p90_mm": float(
            np.quantile(cKDTree(current["obs"]["points"]).query(predicted)[0], 0.9) * 1000
        ),
    }


def replay(method, frame=540, *, clean_start=False, iterations=20, matcher=None):
    """Run actual observation cleanup, local solver and common geometry check.

    history: recorded pre-frame map and seed, without rerunning the occlusion.
    clean: start at frame 510, after placement, with identity as the initial guess.
    Neither uses the evaluation points to initialize or guide the solver. This
    isolates the fitting stage; full pipeline outcomes are scored separately.
    """
    data, metadata = load()
    config = VisionPipelineConfig.from_dict(metadata["settings"]["config"])
    from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig

    uncertainty = PoseUncertaintyConfig(**metadata["settings"]["surface_experiment"]["uncertainty"])
    current = prepare(data, metadata, frame)
    if clean_start:
        source_frame = 510
        source = prepare(data, metadata, source_frame)
        reference = source["obs"]["points"]
        initial = np.eye(4)
        view = SurfaceFitView(source, data[f"intrinsics_{source_frame}"], np.eye(4))
    else:
        source_frame = 0
        reference = data[f"map_{method}_{frame}"]
        initial = data[f"seed_{method}_{frame}"]
        source = {key: data[f"previous_{method}_{frame}_{key}"] for key in ("rgb", "depth", "mask")}
        view = SurfaceFitView(source, data[f"intrinsics_{frame}"], data[f"view_pose_{method}_{frame}"])
    shape = replace(
        config.shape,
        registration=LibraryConfig(
            matcher or ("visible_icp" if method == "baseline" else method), iterations_per_level=iterations
        ),
    )
    # Match the real session: its first measured RGB-D view remains an appearance
    # anchor. Later colors fill unseen areas; they cannot rewrite that anchor.
    # None of the independently annotated controls enter the fitter.
    original = SurfaceFitView(
        prepare(data, metadata, source_frame), data[f"intrinsics_{source_frame}"], np.eye(4)
    )
    fit = WholeSurfaceFit(
        reference, current, data[f"intrinsics_{frame}"], shape, appearance_views=(original, view)
    )
    pose, detail = fit.correct(initial)
    return {
        "method": method,
        "frame": frame,
        "clean_start": clean_start,
        "result": evaluate(data, source_frame, frame, pose),
        "geometry": check_surface_contradiction(fit, pose, uncertainty)
        if pose is not None
        else {"accepted": False},
        "solver": detail,
        "pose": pose.tolist() if pose is not None else None,
    }


def score_run(run):
    """Evaluate the actual published pipeline poses over all 118 clear frames."""
    run = Path(run)
    data, metadata = load()
    records = [json.loads(line) for line in (run / "frames.jsonl").read_text().splitlines()]
    source_frame = records[0]["objects"]["gamepad"]["group_growing_details"]["reference_frame"]
    rows = []
    for record in records:
        frame = record["frame"]
        if frame not in metadata["evaluation_frames"]:
            continue
        pose = record["objects"]["gamepad"]["group_growing"]
        rows.append(
            {"frame": frame, **evaluate(data, source_frame, frame, np.asarray(pose) if pose else None)}
        )
    if len(rows) != len(metadata["evaluation_frames"]):
        raise ValueError("replay does not contain the full post-placement interval")
    errors = [row["mean_feature_error_mm"] for row in rows if row["available"]]
    return {
        "source_run": str(run),
        "reference_frame": source_frame,
        "frames": len(rows),
        "available": len(errors),
        "median_feature_error_mm": float(np.median(errors)) if errors else None,
        "p95_feature_error_mm": float(np.quantile(errors, 0.95)) if errors else None,
        "within_5mm": int(np.count_nonzero(np.asarray(errors) <= 5)),
        "rows": rows,
    }


def rerun_clear_interval(source_run, output, method):
    """Full shared-UFM/group/Kalman/map consumer, initialized after placement.

    This is a diagnostic reset chosen for the experiment, not a proposed timed
    reset in production. No labelled feature is passed into the tracker.
    """
    from benchmarks import flash_runtime_benchmark, flash_tracker_experiment

    _, metadata = load()
    case = json.loads((source_run / "case.json").read_text())
    if case["id"] != metadata["case"]["id"]:
        raise ValueError("the placement interval belongs to a different recording")
    original = flash_tracker_experiment.samples
    first, last = metadata["evaluation_frames"][0], metadata["evaluation_frames"][-1]
    with TemporaryDirectory() as directory, ExitStack() as stack:
        settings = Path(directory) / "settings.json"
        recipe = metadata["settings"]
        recipe["config"]["shape"]["registration"] = asdict(
            LibraryConfig("visible_icp" if method == "baseline" else method)
        )
        settings.write_text(json.dumps(recipe))
        stack.enter_context(
            patch.object(
                flash_tracker_experiment, "samples", lambda item: islice(original(item), first, last + 1)
            )
        )
        argv = [
            "placement-replay",
            "--source-run",
            str(source_run),
            "--settings",
            str(settings),
            "--ufm-revision",
            "980825b59331413bb51e5773f9272bfce525bce2",
            "--output",
            str(output),
        ]
        stack.enter_context(patch.object(sys, "argv", argv))
        flash_runtime_benchmark.main()
    report = score_run(output / "000" / case["id"])
    (output / "placement-score.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-run", type=Path, action="append", default=[])
    parser.add_argument("--fresh-replay", choices=METHODS)
    parser.add_argument("--source-run", type=Path)
    args = parser.parse_args()
    if args.fresh_replay:
        if args.source_run is None or args.score_run:
            parser.error("--fresh-replay needs --source-run and cannot combine with --score-run")
        rerun_clear_interval(args.source_run, args.output, args.fresh_replay)
        return
    if args.score_run:
        reports = [score_run(run) for run in args.score_run]
        args.output.write_text(json.dumps(reports, indent=2) + "\n")
        for report in reports:
            print({key: value for key, value in report.items() if key != "rows"})
        return
    results = [
        replay(method, frame, clean_start=clean)
        for method in METHODS
        for frame in (540, 627)
        for clean in (False, True)
    ]
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for row in results:
        print(
            row["method"],
            row["frame"],
            "clean" if row["clean_start"] else "history",
            row["result"],
            row["geometry"],
        )


if __name__ == "__main__":
    main()
