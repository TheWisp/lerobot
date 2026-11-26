"""Export compact group inputs and compare real frame processing with/without depth rejection."""

import argparse
import hashlib
import json
import time
from collections import Counter
from contextlib import nullcontext
from datetime import date
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from benchmarks.flash_motion_recovery import full_recording, read_recorded_frame
from lerobot.policies.flash.live_surface_session import LiveSurfaceSession
from lerobot.policies.flash.motion_groups import SceneMotionGroups
from lerobot.policies.flash.scene_depth import SceneDepth
from lerobot.policies.flash.shared_image_tracking import SharedFrameTracker, SharedMatchingConfig
from lerobot.policies.flash.surface_acceleration import SearchPool, compiled_kernels

FIXTURE = Path(__file__).resolve().parents[1] / "tests/policies/flash/fixtures/group_depth_20261119"
OLD_IDS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 591, 592, 593, 814, 889]


def export(recording, destination=FIXTURE):
    meta, _ = full_recording(recording)
    numbers = [n for n in meta["cases"]["cube_again"] if n <= 5381]
    tracker = SharedFrameTracker(SharedMatchingConfig.from_dict(meta["settings"]["shared_matching"]))
    arrays, times = {}, []
    for n in numbers:
        frame, masks = read_recorded_frame(recording, n)
        scene = tracker.update(frame, masks)
        arrays.update({f"ids_{n}": scene.ids, f"xy_{n}": scene.xy, f"xyz_{n}": scene.base_xyz})
        times.append(scene.source_time_s)
    arrays.update(depth=frame.depth, intrinsics=frame.intrinsics, base_from_camera=frame.base_from_camera)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "measurements.npz"
    np.savez_compressed(path, **arrays)
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "source_recording": Path(recording).name,
                "source_code_commit": "44fc18138",
                "frames": numbers,
                "times_s": times,
                "old_ids": OLD_IDS,
                "settings": meta["settings"],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "scope": "Exact shared-tracker measurements replayed from RGB and Foundation Stereo depth. "
                "Only endpoint depth is retained. Earlier group updates reconstruct the stale history "
                "without depth rejection; the last update tests removal through the frame owner. "
                "No video, object pose or serialized internal group state is stored.",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Exported {len(numbers)} measured frames and one depth image: {path.stat().st_size} bytes",
        flush=True,
    )


def benchmark(recording, repeats, *, interleave=False):
    meta, source_frames = full_recording(recording)
    numbers = [n for n in meta["cases"]["cube_again"] if n <= 5381]
    # File decoding is outside processing measurements and identical for both runs.
    frames = {n: source_frames[n] for n in numbers}
    settings = meta["settings"]
    source = type("SavedMasks", (), {"prompts": {"object": "selected object"}})()
    runs = []
    original = SceneMotionGroups._reject_depth
    check_ms = []

    def timed(self, *args, **kwargs):
        begin = time.perf_counter()
        result = original(self, *args, **kwargs)
        check_ms.append((time.perf_counter() - begin) * 1000)
        return result

    def measure(live, n, enabled):
        frame, masks = frames[n]
        with nullcontext() if enabled else patch.object(SceneDepth, "from_frame", return_value=None):
            begin = time.perf_counter()
            row, _, _ = live.step(frame, measured_masks=(n, masks, 0.0))
            elapsed = (time.perf_counter() - begin) * 1000
        obj = row["objects"]["object"]
        groups = live.engine.scene_groups.frame
        details = obj["group_growing_details"]
        group = details.get("group", {})
        return {
            "frame": n,
            "elapsed_ms": elapsed,
            "group_update_ms": row["group_update_ms"],
            "depth_prepare_ms": row["group_depth_prepare_ms"],
            "depth_check_ms": check_ms[-1] if enabled else 0,
            "image_tracking_ms": row["scene_point_tracking_ms"],
            "object_prepare_ms": obj["prepare_ms"],
            "object_update_ms": obj["group_growing_ms"],
            "fit_evaluations": details.get("fit", {}).get("evaluations", 0),
            "rejected_ids": sorted(groups.depth_rejected_ids),
            "old_members": sorted({int(i) for g in groups.groups.values() for i in g.ids} & set(OLD_IDS)),
            "status": obj["group_growing_status"],
            "group": {k: v for k, v in group.items() if k not in {"camera_from_reference", "kalman_motion"}},
        }

    # Interleaving compares each frame in independent sessions under nearby load.
    # A discarded complete pair warms both paths before collecting timings.
    for repeat in range(-1 if interleave else 0, repeats):
        rows_by_variant = {False: [], True: []}
        order = [False, True] if repeat % 2 == 0 else [True, False]
        schedule = (
            [
                (enabled, n)
                for j, n in enumerate(numbers)
                for enabled in (order if j % 2 == 0 else order[::-1])
            ]
            if interleave
            else [(enabled, n) for enabled in order for n in numbers]
        )
        with (
            SearchPool(settings["config"]["shape"]["batch_query_workers"]) as pool,
            patch.object(SceneMotionGroups, "_reject_depth", timed),
        ):
            kernels = compiled_kernels()
            sessions = {
                enabled: LiveSurfaceSession(settings, source, kernels=kernels, search_pool=pool)
                for enabled in order
            }
            try:
                for enabled, n in schedule:
                    rows_by_variant[enabled].append(measure(sessions[enabled], n, enabled))
            finally:
                for live in sessions.values():
                    live.close()
        if repeat < 0:
            continue
        for enabled in order:
            rows = rows_by_variant[enabled]
            for row in rows:
                row["added_depth_ms"] = row["depth_prepare_ms"] + row["depth_check_ms"]
            summary = {
                key: {
                    "median": float(np.median([r[key] for r in rows[1:]])),
                    "p95": float(np.percentile([r[key] for r in rows[1:]], 95)),
                }
                for key in ("elapsed_ms", "group_update_ms", "depth_prepare_ms", "depth_check_ms")
            }
            run = {
                "repeat": repeat,
                "enabled": enabled,
                "summary_ms": summary,
                "statuses": dict(Counter(r["status"] for r in rows)),
                "frames": rows,
            }
            runs.append(run)
            print(json.dumps({k: v for k, v in run.items() if k != "frames"}), flush=True)
    keys = (
        "elapsed_ms",
        "group_update_ms",
        "depth_prepare_ms",
        "depth_check_ms",
        "added_depth_ms",
        "image_tracking_ms",
        "object_prepare_ms",
        "object_update_ms",
    )
    aggregate = {}
    for enabled in (False, True):
        samples = [row for run in runs if run["enabled"] == enabled for row in run["frames"][1:]]
        aggregate["enabled" if enabled else "disabled"] = {
            k: dict(
                zip(
                    ("median", "p95"),
                    map(float, np.quantile([row[k] for row in samples], [0.5, 0.95])),
                    strict=True,
                )
            )
            for k in keys
        }
    by_run = {(r["repeat"], r["enabled"]): r["frames"][1:] for r in runs}
    differences = [
        new["elapsed_ms"] - old["elapsed_ms"]
        for repeat in range(repeats)
        for old, new in zip(by_run[repeat, False], by_run[repeat, True], strict=True)
    ]
    source = Path(__file__).resolve().parents[1] / "src/lerobot/policies/flash"
    return {
        "date": date.today().isoformat(),
        "schedule": "alternate variants per frame" if interleave else "alternate complete runs",
        "warmup_pairs_discarded": int(interleave),
        "paired_frame_delta_ms": {
            "median": float(np.median(differences)),
            "mean": float(np.mean(differences)),
            "p05": float(np.quantile(differences, 0.05)),
            "p95": float(np.quantile(differences, 0.95)),
        },
        "baseline_commit": "44fc18138",
        "comparison": "Disable only SceneDepth preparation/rejection; keep all other tracking settings identical.",
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in (
                "scene_depth.py",
                "motion_groups.py",
                "object_group_association.py",
                "surface_frame_session.py",
                "surface_grounding.py",
            )
        },
        "opencv_threads": cv2.getNumThreads(),
        "repetitions": repeats,
        "aggregate_ms": aggregate,
        "recording": Path(recording).name,
        "scope": "Full LiveSurfaceSession with frozen SAM and "
        "Foundation Stereo outputs; decoding/inference excluded; first frame excluded from timing summaries.",
        "runs": runs,
    }


def save_report(path, result):
    trace = path.with_suffix(".frames.jsonl")
    with trace.open("w") as output:
        for run in result["runs"]:
            for row in run["frames"]:
                output.write(json.dumps({"repeat": run["repeat"], "enabled": run["enabled"], **row}) + "\n")
    summary = {
        **result,
        "frame_trace": trace.name,
        "runs": [{k: v for k, v in run.items() if k != "frames"} for run in result["runs"]],
    }
    path.write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--interleave", action="store_true", help="Warm both variants, then alternate per frame"
    )
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.export:
        export(args.recording)
    if args.output:
        save_report(args.output, benchmark(args.recording, args.repeats, interleave=args.interleave))


if __name__ == "__main__":
    main()
