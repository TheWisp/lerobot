"""Where a live frame's time goes in the point groups' view, stage by stage, on recorded frames: reading the frame,
TAPIR (wall time and its GPU time by CUDA events, so the CPU's share around the kernels shows), 3D lookup, the
groups, the surfaces and the drawing. Runs in Point2Pose's environment, as the view does.

    python benchmarks/group_budget.py RECORDING_DIR [--points 200] [--frames 150]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import group_live as gl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--points", type=int, default=200)
    ap.add_argument("--frames", type=int, default=150)
    args = ap.parse_args()
    groups, scene = gl.load_by_path()
    import torch

    src = gl.RecordingSource(pathlib.Path(args.recording), 0, args.frames, 1)
    k = src.k
    tracker = groups.GroupTracker()
    world, memory = scene.World(), scene.SurfaceMemory()
    tapir = None
    t = {
        s: []
        for s in (
            "read",
            "points_3d",
            "stillness",
            "tapir_wall",
            "lookup",
            "groups",
            "surfaces",
            "draw",
            "jpeg",
        )
    }
    prev_small = prev_depth = still_for = None
    it = iter(src)
    while True:
        t0 = time.perf_counter()
        try:
            n, stamp, rgb, depth = next(it)
        except StopIteration:
            break
        t["read"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        pts = scene.points_3d(depth, k)
        t["points_3d"].append(time.perf_counter() - t0)
        t0 = (
            time.perf_counter()
        )  # the live loop's bookkeeping: did the picture move, how long each surface held
        small = cv2.GaussianBlur(cv2.resize(rgb[:, :, 1], (rgb.shape[1] // 4, rgb.shape[0] // 4)), (5, 5), 0)
        small = small.astype(np.float32)
        if prev_small is not None:
            float(np.mean(np.abs(small - prev_small) > 12))
        prev_small = small
        d_half = depth[::2, ::2]
        held = (np.abs(d_half - prev_depth) < 0.005) & (d_half > 0) if prev_depth is not None else d_half > 0
        still_for = np.where(held, (0 if still_for is None else still_for) + 1, 0)
        prev_depth = d_half
        t["stillness"].append(time.perf_counter() - t0)
        if tapir is None:
            h, w = depth.shape
            tapir = gl.Tapir(h, w, resize=256, pips_iters=2)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            tapir.add(rgb, scene.stable_points(gray, np.isfinite(pts).all(axis=2), args.points, 48))
            continue
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if len(t["tapir_wall"]) == 60:  # one profiled stretch: GPU kernel time against the step's wall time
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(20):
                    uv, vis = tapir.step(rgb)
                torch.cuda.synchronize()
            gpu_us = sum(e.self_device_time_total for e in prof.key_averages())
            print(f"  TAPIR profiled over 20 steps: GPU kernels {gpu_us / 20 / 1000:.1f} ms a step")
            t0 = time.perf_counter()
        uv, vis = tapir.step(rgb)
        torch.cuda.synchronize()
        t["tapir_wall"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        xyz, seen = scene.lookup_3d(uv, vis, pts)
        t["lookup"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        tracker.update(xyz, seen)
        base = world.update(tracker)
        t["groups"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        surfaces = memory.update(
            scene.group_surfaces(depth, uv, seen, tracker.group_of, base, quiet=world.quiet)
        )
        t["surfaces"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        img = scene.draw(
            rgb,
            k,
            tracker,
            xyz,
            seen,
            {},
            "header",
            "footer",
            surfaces=surfaces,
            base=base,
            quiet=world.quiet,
        )
        t["draw"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        t["jpeg"].append(time.perf_counter() - t0)
    warm = {s: np.array(v[10:]) * 1000 for s, v in t.items()}  # the first frames pay for warm-up
    total = sum(np.median(v) for v in warm.values())
    print(f"{args.points} points, {len(warm['groups'])} frames after warm-up (ms, median / p95):")
    for s, v in warm.items():
        print(f"  {s:11s} {np.median(v):6.1f} / {np.percentile(v, 95):6.1f}")
    print(f"  sum of medians, one frame end to end (TAPIR wall): {total:.1f} ms")


if __name__ == "__main__":
    main()
