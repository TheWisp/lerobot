"""Compare Point2Pose's causal point frontends before its pose estimator.

Run in the Point2Pose environment. This adapter uses the upstream tracker API
and feeds source-frame queries at their actual query frame. Each backend's
per-frame prediction is depth-lifted identically after tracking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from benchmarks.flash_surface_benchmark import lift_measured_depth
from benchmarks.flash_surface_pacing import PacedConsumer

TRACKERS = {
    "bootstapir": ("tapir_tracker", "TapirTracker"),
    "tapnext": ("tapnext_tracker", "TapnextTracker"),
    "litetracker": ("litetracker_tracker", "LiteTrackerTracker"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    experiment: Path,
    repository: Path,
    checkpoint: Path,
    backend: str,
    output: Path,
    paced: bool = False,
) -> dict:
    manifest = json.loads((experiment / "manifest.json").read_text())
    if manifest["schema"] != "flash-visible-points-v1":
        raise ValueError("unsupported benchmark schema")
    if backend not in TRACKERS:
        raise ValueError(f"unknown frontend {backend}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    queries_path = experiment / "queries.npz"
    if _sha256(queries_path) != manifest["queries_sha256"]:
        raise ValueError("benchmark queries changed")
    with np.load(queries_path, allow_pickle=False) as saved:
        frames, queries, k, capture_s = (
            saved["frames"],
            saved["queries"],
            saved["intrinsics"],
            saved["capture_s"],
        )
    if not np.array_equal(frames, np.arange(frames[0], frames[-1] + 1)):
        raise ValueError("causal adapter requires consecutive source frames")
    if paced and np.any(queries[:, 0] != 0):
        raise ValueError("paced latest-frame replay cannot preserve queries born on skipped frames")
    recording = Path(manifest["recording"])
    if _sha256(recording / "rgb" / f"{frames[0]:06d}.jpg") != manifest["first_rgb_sha256"]:
        raise ValueError("initial RGB changed")
    sys.path.insert(0, str(repository))
    module_name, class_name = TRACKERS[backend]
    import importlib

    tracker_type = getattr(importlib.import_module(f"point2pose.modules.tracker.{module_name}"), class_name)
    # BootsTAPIR scales queries in add_query_points *before* initialize(frame)
    # sets the actual source size. Its defaults are 640x480, which silently
    # mis-scales an 848-pixel camera unless the source size is passed here.
    height, width = manifest["source_hw"]
    config = {
        "checkpoint_path": str(checkpoint),
        "device": "cuda",
        "img_height": height,
        "img_width": width,
    }
    if backend == "tapnext":
        config["input_resolution"] = 256
    started = time.perf_counter()
    tracker = tracker_type(config)
    torch.cuda.synchronize()
    model_load_s = time.perf_counter() - started
    t, n = len(frames), len(queries)
    xy = np.full((t, n, 2), np.nan, dtype=np.float32)
    xyz = np.full((t, n, 3), np.nan, dtype=np.float32)
    visible = np.zeros((t, n), bool)
    decode_s = np.zeros(t)
    tracker_s = np.zeros(t)
    lift_s = np.zeros(t)
    query_ids = np.arange(n)
    clock = PacedConsumer(capture_s) if paced else None
    index = 0
    while index < t:
        if clock is not None:
            index = clock.next_source(index)
        frame_id = frames[index]
        start = time.perf_counter()
        bgr = cv2.imread(str(recording / "rgb" / f"{frame_id:06d}.jpg"), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(recording / "depth" / f"{frame_id:06d}.png"), cv2.IMREAD_UNCHANGED)
        if bgr is None or raw is None or raw.shape != bgr.shape[:2] or raw.dtype != np.uint16:
            raise ValueError(f"bad RGB-D at source frame {frame_id}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = raw.astype(np.float32) * 0.001
        decode_s[index] = time.perf_counter() - start
        frame = SimpleNamespace(id=index, rgb=rgb)
        born = query_ids[queries[:, 0].astype(int) == index]
        start = time.perf_counter()
        if index == 0 and backend != "bootstapir":
            tracker.initialize(frame)
        if len(born):
            assigned = tracker.add_query_points(frame, queries[born, 1:3])
            if not np.array_equal(assigned, born):
                raise ValueError("upstream tracker reordered query IDs")
            if backend == "bootstapir":
                encoded = tracker.query_points[assigned].detach().cpu().numpy()
                requested = queries[born, 1:3]
                expected = np.column_stack(
                    (
                        np.full(len(born), index),
                        requested[:, 1] * tracker._resize_height / height,
                        requested[:, 0] * tracker._resize_width / width,
                    )
                )
                if not np.allclose(encoded, expected, atol=1e-4):
                    raise ValueError("BootsTAPIR query coordinates do not match source pixels")
        if index == 0 and backend == "bootstapir":
            tracker.initialize(frame)
        coords, _, is_visible = tracker.track_once(frame)
        torch.cuda.synchronize()
        tracker_s[index] = time.perf_counter() - start
        known = np.flatnonzero(queries[:, 0] <= index)
        if coords.shape != (len(known), 2) or len(is_visible) != len(known):
            raise ValueError(f"{backend} returned wrong points on frame {frame_id}")
        xy[index, known] = coords
        visible[index, known] = is_visible
        start = time.perf_counter()
        xyz[index, known] = lift_measured_depth(coords, depth, k)
        lift_s[index] = time.perf_counter() - start
        if clock is not None:
            clock.publish(index)
        index += 1
    pacing = clock.finish() if clock is not None else None
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frames=frames,
        queries=queries,
        xy=xy,
        xyz=xyz,
        visible=visible,
        decode_s=decode_s,
        tracker_s=tracker_s,
        lift_s=lift_s,
        geometry_source=np.asarray("sensor_at_prediction"),
        **(
            {
                "publication_source_index": np.asarray(pacing["publication_source_index"]),
                "publication_s": np.asarray(pacing["publication_s"]),
                "consumer_schedule_s": np.asarray(pacing["consumer_schedule_s"]),
                "consumer_actual_s": np.asarray(pacing["consumer_actual_s"]),
                "consumer_source_index": np.asarray(pacing["consumer_source_index"]),
            }
            if pacing is not None
            else {}
        ),
    )
    report = {
        "model": backend,
        "mode": (
            "paced causal latest-frame upstream Point2Pose frontend, no pose estimator"
            if paced
            else "causal per-frame upstream Point2Pose frontend, no pose estimator"
        ),
        "frames": t,
        "queries": n,
        "checkpoint_sha256": _sha256(checkpoint),
        "repository": str(repository.resolve()),
        "hardware": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "model_load_s": model_load_s,
        "decode_s_total": float(decode_s.sum()),
        "tracker_s_total": float(tracker_s.sum()),
        "lift_s_total": float(lift_s.sum()),
        "tracker_frame_s_median": float(np.median(tracker_s)),
        "tracker_frame_s_p95": float(np.quantile(tracker_s, 0.95)),
        "model_only_fps": float(np.count_nonzero(tracker_s) / tracker_s.sum()),
        "decode_track_lift_fps": float(
            np.count_nonzero(tracker_s) / (decode_s.sum() + tracker_s.sum() + lift_s.sum())
        ),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "pacing": pacing,
    }
    output.with_suffix(".timing.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", choices=TRACKERS, required=True)
    parser.add_argument(
        "--paced", action="store_true", help="release frames at recorded times, drop stale source frames"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.experiment, args.repository, args.checkpoint, args.backend, args.output, args.paced),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
