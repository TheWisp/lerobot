"""Real checkpoint: scalar vs batched image pairs, through the metric fitter.

Includes real three-object crops and moving/held-object frame pairs. Different
times in one batch are a numerical stress test, not a causal throughput claim.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch
from benchmark_fits import compare_poses
from candidates import predict_pairs

from lerobot.policies.flash.surface_correspondence import (
    SurfaceMotionConfig,
    crop_pair,
    full_pixels,
    measured_pairs,
    robust_fit,
    ufm_pairs,
)
from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, observation
from lerobot.policies.flash.surface_tracking import load_ufm


def current(rgb, depth, mask, k):
    obs = observation(depth, mask, k, asdict(SurfaceObservationConfig()))
    connected = np.zeros(mask.shape, np.uint8)
    px = obs["pixels"]
    connected[px[:, 1], px[:, 0]] = 1
    connected = cv2.dilate(connected, np.ones((7, 7), np.uint8)) > 0
    return {
        "rgb": rgb,
        "depth": depth,
        "mask": mask,
        "obs": obs,
        "usable": obs["valid_image"] & connected,
        "all_surface_points": obs["points"],
    }


def jobs():
    traces = Path.home() / ".cache/huggingface/lerobot/flash_debug_traces"
    for trace_name, start in (
        ("trace-20261025-192743-253063024", 0),
        ("trace-20261025-181728-660815692", 90),
    ):
        trace = traces / trace_name
        rows = [json.loads(line) for line in (trace / "frames.jsonl").read_text().splitlines()]
        for offset in (0, 5, 11, 19):
            source, target = rows[start + offset], rows[start + offset + 1]
            k = np.asarray(source["intrinsics"])
            for role in source["objects"]:
                frames = []
                for row in (source, target):
                    stem = f"{row['frame_id']:06d}.png"
                    frames.append(
                        current(
                            cv2.cvtColor(cv2.imread(str(trace / "rgb" / stem)), cv2.COLOR_BGR2RGB),
                            cv2.imread(str(trace / "depth" / stem), -1).astype(np.float32)
                            * row["depth_m_per_unit"],
                            cv2.imread(str(trace / "masks" / role / stem), 0) > 0,
                            k,
                        )
                    )
                yield f"{trace_name}:{role}:{source['frame_id']}->{target['frame_id']}", *frames, k
    recording = Path.home() / ".cache/huggingface/lerobot/demos/insertion/recording"
    masks = Path.home() / ".cache/flash-mask-depth-experiment/sam3-full"
    k = np.loadtxt(recording / "cam_K.txt").reshape(3, 3)
    for first, last in (
        (100, 101),
        (225, 226),
        (234, 235),
        (267, 268),
        (279, 280),
        (300, 301),
        (315, 316),
        (360, 361),
        (435, 436),
        (500, 501),
        (514, 515),
    ):
        frames = []
        for n in (first, last):
            frames.append(
                current(
                    cv2.cvtColor(cv2.imread(str(recording / "rgb" / f"{n:06d}.jpg")), cv2.COLOR_BGR2RGB),
                    cv2.imread(str(recording / "depth" / f"{n:06d}.png"), -1).astype(np.float32) * 0.001,
                    cv2.imread(str(masks / f"{n:06d}.png"), 0) > 0,
                    k,
                )
            )
        yield f"insertion:{first}->{last}", *frames, k


def metric_pairs(flow, visible, a, b, crop, k):
    config = SurfaceMotionConfig()
    step, end = config.grid_step_px, config.input_size_px - config.grid_step_px
    ys, xs = np.mgrid[step:end:step, step:end:step]
    ys, xs = ys.ravel(), xs.ravel()
    keep = visible[ys, xs] >= config.covisibility_min
    uv_a = np.column_stack((xs[keep], ys[keep])).astype(float)
    uv_b = uv_a + flow[:, ys[keep], xs[keep]].T
    return measured_pairs(a, b, full_pixels(uv_a, crop), full_pixels(uv_b, crop), k)


def run(output, repeats, fixture=None):
    cv2.setNumThreads(1)
    work = list(jobs())
    crops = [crop_pair(a, b) for _, a, b, _ in work]
    aa, bb = [c[0] for c in crops], [c[1] for c in crops]
    model = load_ufm()
    predict_pairs(model, aa[:3], bb[:3])
    results, timings = {}, {}
    for batch_size in (1, 2, 3, 1):
        label = f"batch_{batch_size}" if batch_size != 1 or "batch_1" not in results else "scalar_repeat"
        times = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            outputs = []
            for start in range(0, len(work), batch_size):
                outputs.extend(
                    predict_pairs(model, aa[start : start + batch_size], bb[start : start + batch_size])
                )
            torch.cuda.synchronize()
            times.append((time.perf_counter() - begin) * 1000)
        results[label] = outputs
        timings[label] = times
        print(label, times, flush=True)
    report = {"timings_ms_all_pairs": timings, "pairs": []}
    for i, (name, a, b, k) in enumerate(work):
        row = {"pair": name, "modes": {}}
        baseline_flow, baseline_vis = results["batch_1"][i]
        base_pose = None
        base_pairs = None
        for label, outputs in results.items():
            flow, vis = outputs[i]
            flow_error = np.linalg.norm(flow - baseline_flow, axis=0)
            x, y = metric_pairs(flow, vis, a, b, crops[i][2], k)
            if label == "batch_1":
                expected_x, expected_y = ufm_pairs(model, aa[i], bb[i], crops[i][2], a, b, k)
                np.testing.assert_array_equal(x, expected_x)
                np.testing.assert_array_equal(y, expected_y)
            cv2.setRNGSeed(17)
            step, inliers, _ = robust_fit(x, y)
            entry = {
                "flow_error_median_px": float(np.median(flow_error)),
                "flow_error_p99_px": float(np.percentile(flow_error, 99)),
                "flow_error_max_px": float(flow_error.max()),
                "visibility_threshold_changes": int(np.count_nonzero((vis >= 0.5) != (baseline_vis >= 0.5))),
                "matches": len(x),
                "inliers": inliers,
                "step_available": step is not None,
            }
            if len(a["obs"]["points"]) >= 3 and len(b["obs"]["points"]) >= 3 and step is not None:
                fit = WholeSurfaceFit(a["obs"]["points"], b, k, ShapeConfig())
                pose, detail = fit.correct(step)
                if label == "batch_1":
                    base_pose = pose
                    base_pairs = (x.copy(), y.copy())
                entry.update(
                    converged=detail["converged"],
                    cost=detail["cost"],
                    pose_difference=compare_poses(pose, base_pose) if base_pose is not None else None,
                )
                if base_pose is not None:
                    predicted = a["obs"]["points"] @ pose[:3, :3].T + pose[:3, 3]
                    expected = a["obs"]["points"] @ base_pose[:3, :3].T + base_pose[:3, 3]
                    displacement = np.linalg.norm(predicted - expected, axis=1) * 1000
                    entry["surface_displacement_median_mm"] = float(np.median(displacement))
                    entry["surface_displacement_max_mm"] = float(displacement.max())
                    from scipy.spatial import cKDTree

                    gap = cKDTree(predicted).query(b["obs"]["points"])[0]
                    entry["surface_p90_mm"] = float(np.percentile(gap, 90) * 1000)
                if fixture is not None and name == "insertion:500->501" and label == "batch_3":
                    config = SurfaceMotionConfig()
                    step_px, end = config.grid_step_px, config.input_size_px - config.grid_step_px
                    gy, gx = np.mgrid[step_px:end:step_px, step_px:end:step_px]
                    gy, gx = gy.ravel(), gx.ravel()
                    grid = np.column_stack((gx, gy)).astype(float)
                    np.savez_compressed(
                        fixture,
                        reference=a["obs"]["points"],
                        target=b["obs"]["points"],
                        mask=b["mask"],
                        depth=b["depth"],
                        intrinsics=k,
                        scalar_source=base_pairs[0],
                        scalar_target=base_pairs[1],
                        batch_source=x,
                        batch_target=y,
                        source_depth=a["depth"],
                        source_usable=a["usable"],
                        target_usable=b["usable"],
                        source_uv=full_pixels(grid, crops[i][2]),
                        scalar_target_uv=full_pixels(grid + baseline_flow[:, gy, gx].T, crops[i][2]),
                        batch_target_uv=full_pixels(grid + flow[:, gy, gx].T, crops[i][2]),
                        scalar_keep=baseline_vis[gy, gx] >= config.covisibility_min,
                        batch_keep=vis[gy, gx] >= config.covisibility_min,
                    )
            row["modes"][label] = entry
        report["pairs"].append(row)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(name, json.dumps(row["modes"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    run(args.output, args.repeats, args.fixture)
