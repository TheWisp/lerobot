"""Measure existing depth-query implementations on inputs captured from the live replay.

This is an isolated query benchmark, not a changed tracking implementation or an
end-to-end speed claim. Model inference, depth preparation and group bookkeeping
are excluded. All variants must preserve the current rejection flags.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from benchmarks.flash_motion_recovery import full_recording
from lerobot.policies.flash.live_surface_session import LiveSurfaceSession
from lerobot.policies.flash.motion_groups import SceneMotionGroups
from lerobot.policies.flash.scene_depth import SceneDepth, project
from lerobot.policies.flash.surface_acceleration import SearchPool, compiled_kernels
from lerobot.policies.flash.surface_observation import move


def benchmark(recording, repeats=101):
    meta, frames = full_recording(recording)
    numbers = [n for n in meta["cases"]["cube_again"] if n <= 5381]
    settings = meta["settings"]
    masks_source = type("SavedMasks", (), {"prompts": {"object": "selected object"}})()
    kernels = compiled_kernels()
    queries = []
    original = SceneDepth.contradicted
    reject_depth = SceneMotionGroups._reject_depth
    captured = []

    def capture(tracker, depth, current, waiting=frozenset()):
        # Preserve group boundaries even when production submits a single batch.
        for group in tracker._groups.values():
            ids = sorted(group.anchors.keys() - group.inliers - waiting)
            if ids:
                points = move(np.array([group.anchors[i] for i in ids]), group.pose)
                queries.append((depth, points))
        return reject_depth(tracker, depth, current, waiting)

    with (
        SearchPool(settings["config"]["shape"]["batch_query_workers"]) as pool,
        patch.object(SceneMotionGroups, "_reject_depth", capture),
    ):
        live = LiveSurfaceSession(settings, masks_source, kernels=kernels, search_pool=pool)
        try:
            for n in numbers:
                frame, masks = frames[n]
                queries = []
                live.step(frame, measured_masks=(n, masks, 0.0))
                captured.append((n, frame.depth.shape, queries))
        finally:
            live.close()

    def compiled(depth, points):
        camera = move(points, depth.camera_from_base)
        _, _, bad = kernels[3](
            camera, project(camera, depth.intrinsics), depth.nearest_depth, depth.tolerance_m
        )
        return bad >= 1 - 1e-6

    def per_group(queries, evaluate):
        return np.concatenate([evaluate(depth, points) for depth, points in queries])

    def batch(queries, evaluate):
        depth = queries[0][0]
        return evaluate(depth, np.concatenate([points for _, points in queries]))

    variants = {
        "numpy_per_group": lambda q: per_group(q, original),
        "numpy_one_batch": lambda q: batch(q, original),
    }
    if kernels is not None:
        variants.update(
            existing_compiled_per_group=lambda q: per_group(q, compiled),
            existing_compiled_one_batch=lambda q: batch(q, compiled),
        )
    rows = []
    for n, shape, queries in captured:
        row = {
            "frame": n,
            "image_shape": list(shape),
            "query_groups": len(queries),
            "query_points": sum(len(points) for _, points in queries),
            "median_ms": dict.fromkeys(variants, 0.0),
        }
        if queries:
            assert all(depth is queries[0][0] for depth, _ in queries)
            expected = variants["numpy_per_group"](queries)
            # Equivalence and warmup happen outside the timing loop.
            for evaluate in variants.values():
                np.testing.assert_array_equal(evaluate(queries), expected)
            measurements = {name: [] for name in variants}
            names = list(variants)
            for repeat in range(repeats):
                # Rotate order to distribute cache/load advantages.
                order = names[repeat % len(names) :] + names[: repeat % len(names)]
                for name in order:
                    begin = time.perf_counter()
                    variants[name](queries)
                    measurements[name].append((time.perf_counter() - begin) * 1000)
            row["median_ms"] = {name: float(np.median(v)) for name, v in measurements.items()}
        rows.append(row)
    source = Path(__file__).resolve().parents[1] / "src/lerobot/policies/flash"
    return {
        "recording": Path(recording).name,
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("scene_depth.py", "motion_groups.py", "surface_acceleration.py")
        },
        "scope": "Captured group query inputs; camera transform, projection and depth lookup only. "
        "Excludes depth preparation, group bookkeeping and other tracking work. "
        "Repeated warm inputs; not an end-to-end performance prediction.",
        "opencv_threads": cv2.getNumThreads(),
        "compiled_kernels_available": kernels is not None,
        "repetitions_per_frame": repeats,
        "rejection_flags_identical": True,
        "median_over_post_initialization_frames_ms": {
            name: float(np.median([r["median_ms"][name] for r in rows[1:]])) for name in variants
        },
        "frames": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=101)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    result = benchmark(args.recording, args.repeats)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
