"""Break down the same-pose CPU/CUDA surface residual without changing FLASH."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from compare import cases, fit_input
from compare_same_poses import candidate_poses
from torch.profiler import ProfilerActivity, profile

from lerobot.policies.flash.experimental_gpu_fit import GpuSolverConfig, GpuWholeSurfaceFit
from lerobot.policies.flash.surface_observation import move


def median_ms(work, repeats: int) -> float:
    durations = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        work()
        torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) * 1000)
    return float(np.median(durations))


def event_count(events, key: str) -> int:
    return sum(event.count for event in events if event.key == key)


def measure(case, repeats: int, pose_count: int) -> dict:
    name, reference, depth, mask, intrinsics, proposal = case
    cpu = fit_input(reference, depth, mask, intrinsics)
    poses = candidate_poses(cpu, proposal, pose_count)
    gpu = GpuWholeSurfaceFit(cpu, GpuSolverConfig(nearest_method="tree"))
    gpu_poses = [gpu.tensor(pose)[None] for pose in poses]
    hulls = [gpu._hull_indices(pose) for pose in poses]
    for pose, hull in list(zip(gpu_poses, hulls, strict=True))[:2]:
        gpu.residuals(pose, hull)
    torch.cuda.synchronize()

    timings = {
        "cpu_full_residual_ms": median_ms(lambda: [cpu.residual(pose) for pose in poses], repeats),
        "gpu_resident_residual_ms": median_ms(
            lambda: [gpu.residuals(pose, hull) for pose, hull in zip(gpu_poses, hulls, strict=True)],
            repeats,
        ),
        "gpu_residual_with_download_ms": median_ms(
            lambda: [
                gpu.residuals(pose, hull)[0].cpu().numpy()
                for pose, hull in zip(gpu_poses, hulls, strict=True)
            ],
            repeats,
        ),
        "gpu_residual_with_upload_hull_download_ms": median_ms(
            lambda: [
                gpu.residuals(gpu.tensor(pose)[None], gpu._hull_indices(pose))[0].cpu().numpy()
                for pose in poses
            ],
            repeats,
        ),
    }

    pose = poses[0]
    source_queries = move(cpu.reference_queries, pose)
    target_queries = (cpu.target_queries - pose[:3, 3]) @ pose[:3, :3]
    gpu_source_queries = gpu.tensor(source_queries)[None]
    gpu_target_queries = gpu.tensor(target_queries)[None]
    timings["cpu_nearest_only_ms"] = median_ms(
        lambda: [
            (
                cpu.target_tree.query(source_queries, workers=cpu.reference_query_workers)[0],
                cpu.reference_tree.query(target_queries, workers=cpu.target_query_workers)[0],
            )
            for _ in poses
        ],
        repeats,
    )
    timings["gpu_nearest_only_ms"] = median_ms(
        lambda: [
            (
                gpu._nearest(gpu_source_queries, gpu.target),
                gpu._nearest(gpu_target_queries, gpu.reference),
            )
            for _ in poses
        ],
        repeats,
    )

    profiled = min(10, len(poses))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        for pose, hull in list(zip(gpu_poses, hulls, strict=True))[:profiled]:
            gpu.residuals(pose, hull)[0].cpu().numpy()
        torch.cuda.synchronize()
    events = trace.key_averages()
    return {
        "pose_count": len(poses),
        "reference_queries": len(cpu.reference_queries),
        "target_queries": len(cpu.target_queries),
        "timings": timings,
        "profile": {
            "pose_count": profiled,
            "cuda_kernel_launches": event_count(events, "cudaLaunchKernel"),
            "cuda_stream_synchronizations": event_count(events, "cudaStreamSynchronize"),
            "gpu_tree_queries": event_count(events, "_query_kernel"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--poses", type=int, default=80)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.poses < 1:
        parser.error("repeats and poses must be positive")
    report = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "cases": {}}
    for case in cases():
        if case[0] not in ("ring_visible_motion", "dowel_held_315"):
            continue
        report["cases"][case[0]] = measure(case, args.repeats, args.poses)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
