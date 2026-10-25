"""Compare the experimental CUDA fitter with the production CPU fitter.

Run from the repository root with PYTHONPATH=src. Both fitters receive the
same RGB-D observation, reference surface, initial pose, and ShapeConfig.
This does not change the live FLASH backend.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.experimental_gpu_fit import GpuSolverConfig, GpuWholeSurfaceFit
from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, observation

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/policies/flash/fixtures"


def cases():
    ring = np.load(FIXTURES / "ring_visible_motion_20261016.npz")
    yield (
        "ring_visible_motion",
        ring["source_surface"],
        ring["target_depth_mm"],
        ring["target_mask"],
        ring["intrinsics"],
        np.eye(4),
    )
    held = np.load(FIXTURES / "dowel_held_frame_315.npz")
    yield (
        "dowel_held_315",
        held["model_points"],
        held["depth"],
        held["mask"],
        held["intrinsics"],
        held["candidate"],
    )
    drift = np.load(FIXTURES / "dowel_static_fit_drift_20261020.npz")
    for frame in (0, 13):
        yield (
            f"dowel_static_{frame}",
            drift["model"],
            drift[f"depth_{frame}"],
            drift[f"mask_{frame}"],
            drift["intrinsics"],
            np.eye(4),
        )


def fit_input(reference, depth_mm, mask, intrinsics):
    depth = depth_mm.astype(np.float32) * 0.001
    measured = observation(depth, mask, intrinsics, asdict(SurfaceObservationConfig()))
    frame = {"obs": measured, "mask": mask, "depth": depth, "all_surface_points": measured["points"]}
    return WholeSurfaceFit(reference, frame, intrinsics, ShapeConfig())


def synchronize(torch):
    torch.cuda.synchronize()


def timed(function, *, repeats, torch=None):
    values = []
    result = None
    for _ in range(repeats):
        if torch is not None:
            synchronize(torch)
        start = time.perf_counter()
        result = function()
        if torch is not None:
            synchronize(torch)
        values.append((time.perf_counter() - start) * 1000)
    return result, values


def evaluate(cpu, gpu, proposal, *, repeats):
    import torch

    start = time.perf_counter()
    gpu_fit = GpuWholeSurfaceFit(cpu, gpu)
    synchronize(torch)
    upload_ms = (time.perf_counter() - start) * 1000
    source_residual = cpu.residual(proposal)
    gpu_residual = gpu_fit.residuals(gpu_fit.tensor(proposal)[None], gpu_fit._hull_indices(proposal))[0]
    synchronize(torch)
    residual_difference = np.abs(source_residual - gpu_residual.cpu().numpy())

    # Warm the CUDA path before timed runs. Each run starts from the same pose.
    gpu_fit.correct(proposal)
    synchronize(torch)
    allocated_before_mib = torch.cuda.memory_allocated() / 2**20
    torch.cuda.reset_peak_memory_stats()
    (cpu_pose, cpu_detail), cpu_ms = timed(lambda: cpu.correct(proposal), repeats=repeats)
    (gpu_pose, gpu_detail), gpu_ms = timed(lambda: gpu_fit.correct(proposal), repeats=repeats, torch=torch)
    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    rotation_delta = Rotation.from_matrix(gpu_pose[:3, :3] @ cpu_pose[:3, :3].T).magnitude()
    return {
        "reference_points": len(cpu.reference),
        "observed_points": len(cpu.target),
        "residual_max_abs_difference": float(residual_difference.max()),
        "residual_rms_difference": float(np.sqrt(np.mean(residual_difference**2))),
        "gpu_upload_ms": upload_ms,
        "cpu_fit_ms": cpu_ms,
        "gpu_fit_ms": gpu_ms,
        "gpu_peak_allocated_mib": peak_mib,
        "gpu_incremental_peak_mib": peak_mib - allocated_before_mib,
        "cpu_cost": cpu_detail["cost"],
        "cpu_solver_evaluations": cpu_detail["evaluations"],
        "cpu_solver_converged": cpu_detail["converged"],
        "gpu_solver_iterations": gpu_detail["iterations"],
        "gpu_cost_measured_by_cpu": cpu.score(gpu_pose),
        "cpu_used_coarse": cpu_detail["used_coarse"],
        "gpu_used_coarse": gpu_detail["used_coarse"],
        "pose_translation_disagreement_mm": float(np.linalg.norm(cpu_pose[:3, 3] - gpu_pose[:3, 3]) * 1000),
        "pose_rotation_disagreement_deg": float(np.degrees(rotation_delta)),
        "cpu_pose": cpu_pose.tolist(),
        "gpu_pose": gpu_pose.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device unavailable")
    config = GpuSolverConfig(iterations=arguments.iterations)
    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "shape_config": asdict(ShapeConfig()),
        "gpu_config": asdict(config),
        "cases": {},
    }
    for name, reference, depth, mask, intrinsics, proposal in cases():
        start = time.perf_counter()
        cpu = fit_input(reference, depth, mask, intrinsics)
        preparation_ms = (time.perf_counter() - start) * 1000
        report["cases"][name] = evaluate(cpu, config, proposal, repeats=arguments.repeats)
        report["cases"][name]["shared_frame_preparation_ms"] = preparation_ms
        case = report["cases"][name]
        print(
            f"{name}: CPU {np.median(case['cpu_fit_ms']):.0f} ms; "
            f"GPU {np.median(case['gpu_fit_ms']):.0f} ms + "
            f"{case['gpu_upload_ms']:.0f} ms upload; "
            f"CPU/GPU cost {case['cpu_cost']:.4f}/{case['gpu_cost_measured_by_cpu']:.4f}; "
            f"pose delta {case['pose_translation_disagreement_mm']:.1f} mm, "
            f"{case['pose_rotation_disagreement_deg']:.1f} deg",
            flush=True,
        )
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
