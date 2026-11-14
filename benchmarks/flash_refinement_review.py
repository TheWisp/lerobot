"""Publish a saved surface-refinement trial for frame-by-frame browser inspection.

The export reuses the exact RGB, selected depth and mask from live history. It
does not run the tracker or alter its result. Each packet contains the trial's
before/after local maps and the pose calculated for that same captured frame.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from benchmarks.flash_clip_replay import recorded_tracker_input
from lerobot.gui.flash_live_scene import encode_rgbd_scene
from lerobot.gui.flash_pose_overlay import reference_axes
from lerobot.policies.flash.tracking_history import read_index


def export(
    clip: Path,
    trial: Path,
    surfaces: Path,
    output: Path,
    *,
    conflict_frame: int | None = None,
    conflict_points: Path | None = None,
) -> dict:
    evidence = json.loads(trial.read_text())
    rows = evidence["rows"]
    audits = evidence["audit"]
    if len(rows) != len(audits) or not rows:
        raise ValueError("Trial rows and surface audits do not align")
    history = {row["frame"]: row for row in read_index(clip)["frames"]}
    output.mkdir(parents=True, exist_ok=True)
    first_time = rows[0]["time_s"]
    previous_pose = None
    timeline = []
    conflicts = None
    if conflict_points is not None:
        if conflict_frame is None:
            raise ValueError("A conflict frame is required with conflict points")
        with np.load(conflict_points, allow_pickle=False) as saved:
            conflicts = {key: saved[key].copy() for key in ("uv", "predicted", "reference", "scene_nearest")}
    with np.load(surfaces, allow_pickle=False) as maps:
        initial = maps[f"{rows[0]['frame']}_after"]
        axes = reference_axes(initial).tolist()
        for i, (row, audit) in enumerate(zip(rows, audits, strict=True)):
            frame_id = row["frame"]
            if audit["frame"] != frame_id or frame_id not in history:
                raise ValueError("Trial frame and recorded camera frame disagree")
            frame, masks, _ = recorded_tracker_input(clip, history[frame_id])
            if not np.isclose(frame.source_time_s, row["time_s"], atol=1e-6, rtol=0):
                raise ValueError("Trial pose does not belong to this RGB-D frame")
            before = maps[f"{frame_id}_before"]
            after = maps[f"{frame_id}_after"]
            if len(before) != audit["map_points_before"] or len(after) != audit["map_points_after"]:
                raise ValueError("Trial surface snapshot does not match audit")
            if row["pose"] is not None:
                previous_pose = row["pose"]
            contours, _ = cv2.findContours(
                masks["object"].astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = [
                cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).tolist() for c in contours if len(c) >= 3
            ]
            growth = audit["growth"] or {}
            conflict_evidence = None
            if frame_id == conflict_frame and conflicts is not None:
                uv = conflicts["uv"]
                x = np.floor(uv[:, 0]).astype(int)
                y = np.floor(uv[:, 1]).astype(int)
                nearest_depth = conflicts["scene_nearest"][y, x]
                k = frame.intrinsics
                measured = np.column_stack(
                    (
                        (uv[:, 0] - k[0, 2]) * nearest_depth / k[0, 0],
                        (uv[:, 1] - k[1, 2]) * nearest_depth / k[1, 1],
                        nearest_depth,
                    )
                )
                conflict_evidence = {
                    "pixels": uv.tolist(),
                    "map_camera_points": conflicts["predicted"].tolist(),
                    "measured_camera_points": measured.tolist(),
                    "map_local_points": conflicts["reference"].tolist(),
                    "depth_gap_mm": ((nearest_depth - conflicts["predicted"][:, 2]) * 1000).tolist(),
                }
            header = {
                "review_frame": i,
                "frame": frame_id,
                "time_s": round(row["time_s"] - first_time, 6),
                "status": row["status"],
                "reason": row["reason"],
                "pose": row["pose"],
                "display_pose": previous_pose,
                "rejected_pose": audit["rejected_pose"],
                "before": before.tolist(),
                "after": after.tolist(),
                "axes": axes,
                "mask_contours": contours,
                "growth": growth,
                "map_points_before": len(before),
                "map_points_after": len(after),
                "depth_source": history[frame_id].get("processed_depth_source", "sensor"),
                "conflicts": conflict_evidence,
            }
            packet = encode_rgbd_scene(frame, header, rgb_format="jpeg")
            with gzip.open(output / f"{i:04d}.bin.gz", "wb", compresslevel=3) as stream:
                stream.write(packet)
            timeline.append(
                {
                    "frame": frame_id,
                    "t": header["time_s"],
                    "status": row["status"],
                    "added": growth.get("admitted_points", 0),
                    "refined": growth.get("refined_points", 0),
                    "map_points": len(after),
                }
            )
    index = {
        "schema_version": 1,
        "name": "Green toy — surface refinement replay",
        "clip": clip.name,
        "trial": trial.name,
        "description": "Recorded selected RGB-D and mask; offline refinement trial poses and local maps. No pose labels.",
        "frames": timeline,
    }
    (output / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--surfaces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conflict-frame", type=int)
    parser.add_argument("--conflict-points", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "frames": len(
                    export(
                        args.clip,
                        args.trial,
                        args.surfaces,
                        args.output,
                        conflict_frame=args.conflict_frame,
                        conflict_points=args.conflict_points,
                    )["frames"]
                )
            }
        )
    )
