"""Research-only GPU implementation of the FLASH whole-surface optimizer.

SAM and UFM already run on the GPU. This module keeps *all iterative surface
residuals, nearest distances, finite differences, and solver state* on CUDA.
CPU work is limited to once-per-frame preparation and choosing initial poses.
It is deliberately not imported by the live backend until matched-input
benchmarks establish both pose quality and wall-clock improvement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

from lerobot.policies.flash.surface_grounding import WholeSurfaceFit, pca_candidates, project
from lerobot.policies.flash.surface_observation import move


@dataclass(frozen=True)
class GpuSolverConfig:
    iterations: int = 50
    derivative_step: float = 1e-5
    initial_damping: float = 1e-3
    pair_budget_mib: int = 64
    max_step_rotation_rad: float = 0.3
    max_step_translation_m: float = 0.03


class GpuWholeSurfaceFit:
    """Keep one frame's surface fit on CUDA; never replace the CPU result here."""

    def __init__(self, cpu: WholeSurfaceFit, config: GpuSolverConfig = GpuSolverConfig()):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the experimental GPU fitter")
        self.torch = torch
        self.cpu = cpu
        self.config = config
        self.device = torch.device("cuda")
        self.dtype = torch.float64

        def tensor(value):
            return torch.as_tensor(np.ascontiguousarray(value), dtype=self.dtype, device=self.device)

        self.tensor = tensor
        self.reference = tensor(cpu.reference)
        self.reference_queries = tensor(cpu.reference_queries)
        self.target = tensor(cpu.target)
        self.target_queries = tensor(cpu.target_queries)
        self.envelope = tensor(cpu.envelope)
        self.outline = tensor(cpu.mask_outline)
        self.k = tensor(cpu.k)
        self.depth = tensor(cpu.observation["depth"])
        self.outside_mask = tensor(cpu.outside_mask)
        self.height, self.width = cpu.observation["depth"].shape
        self.identity = torch.eye(6, dtype=self.dtype, device=self.device)

    def _nearest(self, queries, points):
        """Batched exact nearest distance with a bounded pairwise workspace."""
        torch = self.torch
        batch, count, _ = queries.shape
        total = len(points)
        pair_limit = max(1, self.config.pair_budget_mib * 1024 * 1024 // queries.element_size())
        target_block = min(total, max(1, pair_limit // batch))
        query_block = max(1, pair_limit // (batch * target_block))
        blocks = []
        for q0 in range(0, count, query_block):
            query = queries[:, q0 : q0 + query_block]
            best = torch.full(query.shape[:2], float("inf"), dtype=self.dtype, device=self.device)
            for p0 in range(0, total, target_block):
                candidate = torch.cdist(query, points[p0 : p0 + target_block][None].expand(batch, -1, -1))
                best = torch.minimum(best, candidate.amin(dim=-1))
            blocks.append(best)
        return torch.cat(blocks, dim=1)

    def _project(self, points):
        return (points[..., :2] / points[..., 2, None]) @ self.k[:2, :2].T + self.k[:2, 2]

    def _image_values(self, image, x0, y0, x1, y1):
        return self.torch.stack((image[y0, x0], image[y0, x1], image[y1, x0], image[y1, x1]), dim=-1)

    def _uncovered_outline(self, envelope_uv, hull_indices):
        torch = self.torch
        polygon = envelope_uv[:, hull_indices]
        start, end = polygon, polygon.roll(-1, dims=1)
        edge = end - start
        delta = self.outline[None, :, None, :] - start[:, None]
        along = (delta * edge[:, None]).sum(-1) / (edge.square().sum(-1)[:, None] + 1e-30)
        along = along.clamp(0, 1)
        distances = torch.linalg.vector_norm(delta - along[..., None] * edge[:, None], dim=-1).amin(dim=-1)
        cross = edge[:, None, :, 0] * delta[..., 1] - edge[:, None, :, 1] * delta[..., 0]
        inside = (cross >= 0).all(-1) | (cross <= 0).all(-1)
        return torch.where(inside, 0, distances - self.cpu.config.mask_boundary_tolerance_px).clamp_min(0)

    def residuals(self, poses, hull_indices):
        """All residual blocks match the CPU objective, except fixed hull topology."""
        torch = self.torch
        rotation, translation = poses[:, :3, :3], poses[:, :3, 3]
        predicted = self.reference_queries[None] @ rotation.transpose(1, 2) + translation[:, None]
        uv = self._project(predicted)
        x, y = uv[..., 0], uv[..., 1]
        inside = (x >= 0) & (x <= self.width - 1) & (y >= 0) & (y <= self.height - 1)
        cx, cy = x.clamp(0, self.width - 1), y.clamp(0, self.height - 1)
        x0, y0 = cx.floor().long(), cy.floor().long()
        x1, y1 = (x0 + 1).clamp(max=self.width - 1), (y0 + 1).clamp(max=self.height - 1)
        fx, fy = cx - x0, cy - y0
        weights = torch.stack(((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy), dim=-1)
        mask_distance = (weights * self._image_values(self.outside_mask, x0, y0, x1, y1)).sum(-1)
        rays = self._image_values(self.depth, x0, y0, x1, y1)
        known = (rays > 0) & inside[..., None]
        tolerance = self.cpu.config.depth_tolerance_m
        ray_visibility = ((rays - predicted[..., 2, None] + tolerance) / tolerance).clamp(0, 1)
        depth_visibility = (weights * known * ray_visibility).sum(-1)
        visibility = depth_visibility + (weights * ~known).sum(-1)

        model_distance = self._nearest(predicted, self.target).clamp(
            max=self.cpu.config.max_surface_distance_m
        )
        target_local = (self.target_queries[None] - translation[:, None]) @ rotation
        target_distance = self._nearest(target_local, self.reference).clamp(
            max=self.cpu.config.max_surface_distance_m
        )
        silhouette = (mask_distance - self.cpu.config.mask_boundary_tolerance_px).clamp_min(0)
        silhouette = silhouette * predicted[..., 2] / self.k[0, 0]

        projected_envelope = self._project(
            self.envelope[None] @ rotation.transpose(1, 2) + translation[:, None]
        )
        uncovered = self._uncovered_outline(projected_envelope, hull_indices)
        z_sorted = predicted[..., 2].sort(dim=-1).values
        count = z_sorted.shape[-1]
        median_z = (z_sorted[:, (count - 1) // 2] + z_sorted[:, count // 2]) / 2
        return torch.cat(
            (
                (depth_visibility / predicted.shape[1]).sqrt() * model_distance / tolerance,
                target_distance / (tolerance * np.sqrt(len(self.target_queries))),
                (visibility / predicted.shape[1]).sqrt() * silhouette / tolerance,
                uncovered * median_z[:, None] / (self.k[0, 0] * tolerance * np.sqrt(len(self.outline))),
            ),
            dim=1,
        )

    def _poses(self, initial, center, deltas):
        torch = self.torch
        rotvec = deltas[:, :3]
        theta = torch.linalg.vector_norm(rotvec, dim=-1)
        safe_theta = theta.clamp_min(1e-12)
        sinc = torch.where(theta < 1e-4, 1 - theta.square() / 6, torch.sin(theta) / safe_theta)
        cosc = torch.where(
            theta < 1e-4, 0.5 - theta.square() / 24, (1 - torch.cos(theta)) / safe_theta.square()
        )
        x, y, z = rotvec.unbind(-1)
        zero = torch.zeros_like(x)
        skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(-1, 3, 3)
        rotation = torch.eye(3, dtype=self.dtype, device=self.device)[None] + sinc[:, None, None] * skew
        rotation = rotation + cosc[:, None, None] * (skew @ skew)
        poses = torch.eye(4, dtype=self.dtype, device=self.device)[None].repeat(len(deltas), 1, 1)
        poses[:, :3, :3] = rotation @ initial[:3, :3]
        poses[:, :3, 3] = (
            (rotation @ (initial[:3, 3] - center)[None, :, None]).squeeze(-1) + center + deltas[:, 3:]
        )
        return poses

    def _hull_indices(self, initial):
        uv = project(move(self.cpu.envelope, initial), self.cpu.k)
        return ConvexHull(uv).vertices.copy()

    def score(self, pose):
        hull = self._hull_indices(pose)
        residual = self.residuals(self.tensor(pose)[None], hull)[0]
        return float((residual @ residual).cpu())

    def refine(self, initial):
        torch = self.torch
        initial_gpu = self.tensor(initial)
        center = (self.reference @ initial_gpu[:3, :3].T + initial_gpu[:3, 3]).mean(0)
        hull = self._hull_indices(initial)
        delta = torch.zeros(6, dtype=self.dtype, device=self.device)
        damping = torch.tensor(self.config.initial_damping, dtype=self.dtype, device=self.device)
        step_size = self.config.derivative_step
        used_iterations = 0
        for _ in range(self.config.iterations):
            used_iterations += 1
            offsets = torch.cat(
                (torch.zeros((1, 6), dtype=self.dtype, device=self.device), self.identity * step_size)
            )
            candidates = delta[None] + offsets
            residuals = self.residuals(self._poses(initial_gpu, center, candidates), hull)
            base = residuals[0]
            jacobian = (residuals[1:] - base) / step_size
            hessian = jacobian @ jacobian.T
            gradient = jacobian @ base
            scale = hessian.diag().amax().clamp_min(1.0)
            step = torch.linalg.solve(hessian + self.identity * damping * scale, -gradient)
            rotation_norm = torch.linalg.vector_norm(step[:3]).clamp_min(1e-12)
            translation_norm = torch.linalg.vector_norm(step[3:]).clamp_min(1e-12)
            step = torch.cat(
                (
                    step[:3] * (self.config.max_step_rotation_rad / rotation_norm).clamp(max=1),
                    step[3:] * (self.config.max_step_translation_m / translation_norm).clamp(max=1),
                )
            )
            proposal = delta + step
            candidate_residual = self.residuals(self._poses(initial_gpu, center, proposal[None]), hull)[0]
            base_cost = base @ base
            candidate_cost = candidate_residual @ candidate_residual
            accept = candidate_cost < base_cost
            delta = torch.where(accept, proposal, delta)
            damping = torch.where(accept, (damping * 0.5).clamp_min(1e-8), (damping * 2).clamp_max(1e8))
            # One scalar synchronization can avoid dozens of needless residual
            # evaluations once the same objective tolerance as the CPU solver
            # has been reached. Everything else in the iteration stays on CUDA.
            converged = accept & (
                (base_cost - candidate_cost) <= self.cpu.config.solver_tolerance * (1 + base_cost)
            )
            if bool(converged.item()):
                break
        pose = self._poses(initial_gpu, center, delta[None])[0].cpu().numpy()
        return pose, self.score(pose), used_iterations

    def correct(self, proposal):
        local, local_score, local_iterations = self.refine(proposal)
        predicted = move(self.cpu.reference, proposal)
        coarse = min(
            pca_candidates(self.cpu.reference, self.cpu.observation["obs"]["points"]),
            key=lambda pose: np.mean(np.sum((move(self.cpu.reference, pose) - predicted) ** 2, axis=1)),
        )
        if self.score(coarse) >= local_score:
            return local, {"cost": local_score, "used_coarse": False, "iterations": local_iterations}
        alternative, alternative_score, alternative_iterations = self.refine(coarse)
        if alternative_score < local_score:
            return alternative, {
                "cost": alternative_score,
                "used_coarse": True,
                "iterations": local_iterations + alternative_iterations,
            }
        return local, {
            "cost": local_score,
            "used_coarse": False,
            "iterations": local_iterations + alternative_iterations,
        }


class GpuResidualScipyFit(WholeSurfaceFit):
    """Use the production SciPy solver with only its residual evaluated on CUDA.

    This is a controlled device comparison. Each candidate pose keeps the
    production solver, dynamic outline, convergence rule and branch choice.
    It is intentionally synchronous and remains outside the live backend.
    """

    def __init__(self, cpu: WholeSurfaceFit):
        self.__dict__.update(cpu.__dict__)
        self.gpu_residual = GpuWholeSurfaceFit(cpu)

    def residual(self, pose: np.ndarray) -> np.ndarray:
        gpu = self.gpu_residual
        hull = gpu._hull_indices(pose)
        return gpu.residuals(gpu.tensor(pose)[None], hull)[0].cpu().numpy()
