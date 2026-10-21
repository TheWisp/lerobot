"""Replay TAPNext++ as a fixed-seed causal point frontend, without Point2Pose.

Run in the isolated Point2Pose environment with that checkout and its tapnet
source on PYTHONPATH. The output is raw 2D tracks; a separate scorer uses the
same RGB-D registration as FLASH's LK point-track replay. No future frame or
later mask is given to TAPNext++.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


def _frame(dataset: Path, frame_id: int) -> SimpleNamespace:
    bgr = cv2.imread(str(dataset / "rgb" / f"{frame_id:06d}.jpg"))
    if bgr is None:
        raise ValueError(f"missing RGB frame {frame_id}")
    return SimpleNamespace(id=frame_id, rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--role", required=True)
    parser.add_argument("--seed-frame", required=True, type=int)
    parser.add_argument("--last-frame", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.last_frame <= args.seed_frame or not args.checkpoint.is_file():
        parser.error("need a later final frame and downloaded checkpoint")
    from point2pose.modules.tracker.tapnext_tracker import TapnextTracker

    seed = _frame(args.dataset, args.seed_frame)
    mask = cv2.imread(str(args.masks / args.role / f"{args.seed_frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != seed.rgb.shape[:2]:
        raise ValueError("seed mask must align with the RGB frame")
    corners = cv2.goodFeaturesToTrack(
        cv2.cvtColor(seed.rgb, cv2.COLOR_RGB2GRAY),
        maxCorners=120,
        qualityLevel=0.005,
        minDistance=5,
        mask=mask,
    )
    query_uv = corners.reshape(-1, 2) if corners is not None else np.empty((0, 2), dtype=np.float32)
    seed_depth = cv2.imread(str(args.dataset / "depth" / f"{args.seed_frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    if seed_depth is None or seed_depth.shape != mask.shape:
        raise ValueError("seed depth must align with the RGB frame")
    if len(query_uv):
        pixels = np.rint(query_uv).astype(int)
        query_uv = query_uv[seed_depth[pixels[:, 1], pixels[:, 0]] > 0]
    if len(query_uv) < 3:
        raise ValueError("seed mask has fewer than three point queries")
    tracker = TapnextTracker(
        {
            "device": "cuda",
            "input_resolution": 256,
            "visible_threshold": 0.5,
            "half_precision": True,
            "checkpoint_path": str(args.checkpoint),
        }
    )
    tracker.initialize(seed)
    tracker.add_query_points(seed, query_uv)
    rows = []
    for frame_id in range(args.seed_frame, args.last_frame + 1):
        frame = seed if frame_id == args.seed_frame else _frame(args.dataset, frame_id)
        started = time.perf_counter()
        uv, uncertainty, visible = tracker.track_once(frame)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if len(uv) != len(query_uv):
            raise ValueError("TAPNext++ changed query count during fixed-seed replay")
        rows.append((frame_id, uv, uncertainty, visible, elapsed_ms))
        if frame_id % 25 == 0:
            print(frame_id, "visible", int(visible.sum()), "/", len(query_uv), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        role=np.asarray(args.role),
        seed_frame=np.asarray(args.seed_frame),
        seed_uv=query_uv,
        frames=np.asarray([row[0] for row in rows]),
        uv=np.stack([row[1] for row in rows]),
        uncertainty=np.stack([row[2] for row in rows]),
        visible=np.stack([row[3] for row in rows]),
        step_ms=np.asarray([row[4] for row in rows]),
    )
    print(args.output)


if __name__ == "__main__":
    main()
