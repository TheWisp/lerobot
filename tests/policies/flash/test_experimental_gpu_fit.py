"""Numerical checks for the opt-in CUDA surface-fit research path."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig, observation

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("name", ["ring", "dowel"])
def test_cuda_residual_matches_cpu_for_multiple_poses(name):
    from lerobot.policies.flash.experimental_gpu_fit import GpuResidualScipyFit, GpuWholeSurfaceFit

    if name == "ring":
        recorded = np.load(FIXTURES / "ring_visible_motion_20261016.npz")
        reference, depth_mm, mask = (
            recorded["source_surface"],
            recorded["target_depth_mm"],
            recorded["target_mask"],
        )
    else:
        recorded = np.load(FIXTURES / "dowel_held_frame_315.npz")
        reference, depth_mm, mask = recorded["model_points"], recorded["depth"], recorded["mask"]
    k = recorded["intrinsics"]
    depth = depth_mm.astype(np.float32) * 0.001
    observed = observation(depth, mask, k, asdict(SurfaceObservationConfig()))
    cpu = WholeSurfaceFit(
        reference,
        {"obs": observed, "mask": mask, "depth": depth, "all_surface_points": observed["points"]},
        k,
        ShapeConfig(),
    )
    gpu = GpuWholeSurfaceFit(cpu)
    same_solver_gpu = GpuResidualScipyFit(cpu)
    initial = recorded["candidate"] if name == "dowel" else np.eye(4)
    for rotvec, offset in [([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), ([0.03, -0.02, 0.01], [0.002, 0.0, 0.0])]:
        pose = initial.copy()
        pose[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix() @ pose[:3, :3]
        pose[:3, 3] += offset
        actual = gpu.residuals(gpu.tensor(pose)[None], gpu._hull_indices(pose))[0].cpu().numpy()
        np.testing.assert_allclose(actual, cpu.residual(pose), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(same_solver_gpu.residual(pose), cpu.residual(pose), rtol=1e-5, atol=1e-6)
