"""Measured-depth Lucas-Kanade control for the visible-surface benchmark.

This deliberately does not use object masks after query selection. Lost points
remain lost, and a plausible point elsewhere on the object is still scored as
the wrong material point against independent labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from benchmarks.flash_surface_benchmark import lift_measured_depth
from benchmarks.flash_surface_pacing import PacedConsumer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(experiment: Path, output: Path, window: int = 21, levels: int = 3, paced: bool = False) -> dict:
    if window < 3 or window % 2 != 1 or levels < 0:
        raise ValueError("window must be odd and at least 3; levels nonnegative")
    manifest = json.loads((experiment / "manifest.json").read_text())
    source = experiment / "queries.npz"
    if manifest["schema"] != "flash-visible-points-v1" or _sha256(source) != manifest["queries_sha256"]:
        raise ValueError("benchmark source/query manifest changed")
    with np.load(source, allow_pickle=False) as saved:
        frames, queries, k, capture_s = (
            saved["frames"],
            saved["queries"],
            saved["intrinsics"],
            saved["capture_s"],
        )
    t, n = len(frames), len(queries)
    if not np.array_equal(frames, np.arange(frames[0], frames[0] + t)):
        raise ValueError("LK requires consecutive source frames")
    if paced and np.any(queries[:, 0] != 0):
        raise ValueError("paced latest-frame replay cannot preserve queries born on skipped frames")
    recording = Path(manifest["recording"])
    xy = np.full((t, n, 2), np.nan, np.float32)
    xyz = np.full((t, n, 3), np.nan, np.float32)
    visible = np.zeros((t, n), bool)
    decode_s, tracker_s, lift_s = (np.zeros(t) for _ in range(3))
    previous = None
    active = np.zeros(n, bool)
    last_xy = np.full((n, 2), np.nan, np.float32)
    clock = PacedConsumer(capture_s) if paced else None
    index = 0
    while index < t:
        if clock is not None:
            index = clock.next_source(index)
        frame = frames[index]
        started = time.perf_counter()
        bgr = cv2.imread(str(recording / "rgb" / f"{frame:06d}.jpg"), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(recording / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
        if bgr is None or raw is None or bgr.shape[:2] != raw.shape or raw.dtype != np.uint16:
            raise ValueError(f"unaligned RGB-D at frame {frame}")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        decode_s[index] = time.perf_counter() - started
        started = time.perf_counter()
        if previous is not None and np.any(active):
            ids = np.flatnonzero(active)
            moved, status, _ = cv2.calcOpticalFlowPyrLK(
                previous,
                gray,
                last_xy[ids, None, :],
                None,
                winSize=(window, window),
                maxLevel=levels,
            )
            if moved is None or status is None:
                active[ids] = False
            else:
                moved = moved[:, 0, :]
                good = status[:, 0].astype(bool) & np.isfinite(moved).all(axis=1)
                active[ids[~good]] = False
                last_xy[ids[good]] = moved[good]
        born = np.flatnonzero(queries[:, 0] == index)
        if len(born):
            active[born] = True
            last_xy[born] = queries[born, 1:3]
        xy[index, active] = last_xy[active]
        visible[index, active] = True
        tracker_s[index] = time.perf_counter() - started
        started = time.perf_counter()
        xyz[index] = lift_measured_depth(xy[index], raw.astype(np.float32) * 0.001, k)
        lift_s[index] = time.perf_counter() - started
        previous = gray
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
        "model": "OpenCV pyramidal Lucas-Kanade",
        "mode": (
            "paced causal latest-frame optical flow; no object mask after seeding"
            if paced
            else "causal per-frame optical flow; no object mask after seeding"
        ),
        "frames": t,
        "queries": n,
        "window": window,
        "levels": levels,
        "decode_s_total": float(decode_s.sum()),
        "tracker_s_total": float(tracker_s.sum()),
        "lift_s_total": float(lift_s.sum()),
        "model_only_fps": float(np.count_nonzero(tracker_s) / tracker_s.sum()),
        "decode_track_lift_fps": float(
            np.count_nonzero(tracker_s) / (decode_s.sum() + tracker_s.sum() + lift_s.sum())
        ),
        "pacing": pacing,
    }
    output.with_suffix(".timing.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--paced", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.experiment, args.output, args.window, args.levels, args.paced), indent=2))


if __name__ == "__main__":
    main()
