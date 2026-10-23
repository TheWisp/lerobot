"""Replay-only Point2Pose/TAPNext++ ablation of late point-query locations.

Point2Pose admits a point after processing its source frame. The current
TAPNext++ adapter commits it on the next frame, still at the source pixel.
Move only those pending query locations by one-step optical flow before the
adapter commits them. The source depth, map construction, masks, fitting and
every other Point2Pose setting remain unchanged. This is a diagnostic, not a
candidate production implementation of point admission.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-bridge", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--diagnostic", required=True, type=Path)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("sealed_p2p_bridge", args.sealed_bridge)
    if spec is None or spec.loader is None:
        raise ValueError("could not load sealed Point2Pose bridge")
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    sys.path.insert(0, str(args.repo))
    from point2pose.modules.tracker.tapnext_tracker import TapnextTracker

    original_initialize = TapnextTracker.initialize
    original_add = TapnextTracker.add_query_points
    original_track = TapnextTracker.track_once
    args.diagnostic.parent.mkdir(parents=True, exist_ok=True)
    diagnostic = args.diagnostic.open("w", newline="")
    writer = csv.writer(diagnostic)
    writer.writerow(
        ("source_frame", "target_frame", "points", "lk_valid", "median_lk_shift_px", "p90_lk_shift_px")
    )

    def initialize(self, frame):
        self._query_source_frames = []
        return original_initialize(self, frame)

    def add_query_points(self, frame, new_points):
        indices = original_add(self, frame, new_points)
        self._query_source_frames.append((int(frame.id), frame.rgb.copy()))
        return indices

    def track_once(self, frame):
        if self._state is not None and self._pending_points:
            if len(self._query_source_frames) != len(self._pending_points):
                raise ValueError("pending TAPNext++ point batches lost their source frames")
            current_gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
            for index, ((source_id, source_rgb), old_uv) in enumerate(
                zip(self._query_source_frames, self._pending_points, strict=True)
            ):
                if source_id == int(frame.id):
                    continue
                source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
                new_uv, valid, _ = cv2.calcOpticalFlowPyrLK(
                    source_gray,
                    current_gray,
                    old_uv.astype(np.float32).reshape(-1, 1, 2),
                    None,
                    winSize=(21, 21),
                    maxLevel=2,
                )
                valid = valid.reshape(-1).astype(bool)
                shift = np.linalg.norm(new_uv.reshape(-1, 2) - old_uv, axis=1)[valid]
                corrected = old_uv.copy()
                corrected[valid] = new_uv.reshape(-1, 2)[valid]
                self._pending_points[index] = corrected
                writer.writerow(
                    (
                        source_id,
                        int(frame.id),
                        len(old_uv),
                        int(valid.sum()),
                        float(np.median(shift)) if len(shift) else float("nan"),
                        float(np.percentile(shift, 90)) if len(shift) else float("nan"),
                    )
                )
        self._query_source_frames = []
        return original_track(self, frame)

    TapnextTracker.initialize = initialize
    TapnextTracker.add_query_points = add_query_points
    TapnextTracker.track_once = track_once
    sys.argv = [str(args.sealed_bridge), "--repo", str(args.repo), "--config", str(args.config)]
    try:
        bridge.main()
    finally:
        diagnostic.close()


if __name__ == "__main__":
    main()
