"""Replay a saved FLASH live-history clip through the real frame session.

The clip's recorded masks, RGB-D frames and pipeline settings are the inputs;
nothing from the recorded tracking result is consumed. Each frame's new status
is printed beside the recorded one, so a code change can be compared against the
live run that produced the clip frame by frame. Settings may be overridden with
``--set dotted.key=json`` to test one knob in isolation.
"""

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np

from lerobot.policies.flash.live_surface_session import LIVE_METHOD, LiveSurfaceSession
from lerobot.policies.flash.surface_acceleration import SearchPool, compiled_kernels
from lerobot.policies.flash.tracking_history import decode_record, history_root, read_index


class RecordedMasks:
    """Mask source for LiveSurfaceSession fed from the clip's own packets."""

    def __init__(self, name, role="object"):
        self.prompts = {role: name}
        self.role = role
        self.current = None

    def masks(self, frame):
        return {self.role: self.current}


def replay(directory, *, start=None, stop=None, overrides=(), quiet=False):
    directory = Path(directory)
    settings = json.loads((directory / "config.json").read_text())["pipeline_settings"]
    for item in overrides:
        key, value = item.split("=", 1)
        node = settings
        *parents, leaf = key.split(".")
        for part in parents:
            node = node[part]
        node[leaf] = json.loads(value)
    index = read_index(directory)
    frames = [
        row["frame"]
        for row in index["frames"]
        if (directory / f"{row['frame']:010d}.bin.z").exists()
        and (start is None or row["frame"] >= start)
        and (stop is None or row["frame"] <= stop)
    ]
    role, first = next(iter(index["frames"][0]["objects"].items()))
    source = RecordedMasks(first["name"], role)
    kernels = compiled_kernels() if settings["config"].get("compiled_surface_fit") else None
    pool = (
        SearchPool(settings["config"]["shape"]["batch_query_workers"])
        if settings["config"].get("reuse_surface_search")
        else None
    )
    live = LiveSurfaceSession(settings, source, None, kernels, pool)
    rows = []
    started = time.perf_counter()
    try:
        for frame_id in frames:
            packet = zlib.decompress((directory / f"{frame_id:010d}.bin.z").read_bytes())
            rgbd, masks, recorded, _ = decode_record(packet)
            source.current = masks[role]
            row, _, _ = live.step(rgbd)
            result = row["objects"][role]
            details = result[LIVE_METHOD + "_details"]
            before = recorded["objects"][role]
            check = (
                details.get("prediction_geometry") or (details.get("fit") or {}).get("candidate_check") or {}
            )
            rows.append(
                {
                    "frame": frame_id,
                    "time_s": rgbd.source_time_s,
                    "status": result[LIVE_METHOD + "_status"],
                    "recorded_status": before[LIVE_METHOD + "_status"],
                    "reason": details.get("reason"),
                    "recorded_reason": before[LIVE_METHOD + "_details"].get("reason"),
                    "pose": result[LIVE_METHOD],
                    "recorded_pose": before[LIVE_METHOD],
                    "scene_depth_views": check.get("scene_depth_views"),
                    "observed_overlap": details.get("observed_overlap"),
                    "surface_median_mm": details.get("surface_median_mm"),
                    "processing_ms": row["processing_ms"],
                }
            )
            if not quiet:
                last = rows[-1]
                mark = "=" if last["status"] == last["recorded_status"] else "!"
                views = [
                    (v["direction"], round(v["outlier_fraction"], 4), v["coherent_protrusion"])
                    for v in last["scene_depth_views"] or []
                ]
                print(
                    f"{frame_id:6d} {mark} {last['status']:18s} recorded={last['recorded_status']:18s} "
                    f"{last['reason'] or '':38s} {views} {row['processing_ms']:.0f}ms",
                    flush=True,
                )
    finally:
        live.close()
        if pool is not None:
            pool.close()
    times = [r["processing_ms"] for r in rows]
    summary = {
        "clip": directory.name,
        "frames": len(rows),
        "observed": sum(r["status"] in ("observed_candidate", "initialized") for r in rows),
        "recorded_observed": sum(r["recorded_status"] in ("observed_candidate", "initialized") for r in rows),
        "predicted": sum(r["status"] == "predicted" for r in rows),
        "lost": sum(r["status"] == "lost" for r in rows),
        "status_matches_recorded": sum(r["status"] == r["recorded_status"] for r in rows),
        "processing_ms_mean": float(np.mean(times)) if times else None,
        "processing_ms_p99": float(np.percentile(times, 99)) if times else None,
        "wall_s": time.perf_counter() - started,
        "overrides": list(overrides),
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", help="recording id under the history root, or a directory")
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    parser.add_argument("--out", type=Path, help="write rows and summary as JSON")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    directory = Path(args.clip) if Path(args.clip).is_dir() else history_root() / args.clip
    rows, summary = replay(directory, start=args.start, stop=args.stop, overrides=args.set, quiet=args.quiet)
    print(json.dumps(summary))
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "rows": rows}))


if __name__ == "__main__":
    main()
