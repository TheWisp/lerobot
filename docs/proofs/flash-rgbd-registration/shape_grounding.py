"""Rigid whole-surface correction using measured RGB-D and an object mask.

The reference is a fixed observed surface, not a CAD model or a cylinder.
Foreground occlusion may hide reference samples. Target-to-reference coverage
prevents an incorrect candidate from evading all checks by moving behind an
occluder. Unseen reference geometry remains a limitation, not invented geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from compare import move
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ShapeConfig:
    # Observation tolerances inherited from the existing depth experiment.
    depth_tolerance_m: float = 0.005
    surface_spacing_m: float = 0.003
    mask_boundary_tolerance_px: float = 2.0
    max_surface_distance_m: float = 0.03
    max_evaluations: int = 50
    solver_tolerance: float = 1e-5


def project(points: np.ndarray, k: np.ndarray) -> np.ndarray:
    return points[:, :2] / points[:, 2, None] @ k[:2, :2].T + k[:2, 2]


def pca_candidates(source: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
    """Generic coarse initializations; never used as a pose or axis measurement."""
    source_center, target_center = source.mean(0), target.mean(0)
    _, _, source_basis = np.linalg.svd(source - source_center, full_matrices=False)
    _, _, target_basis = np.linalg.svd(target - target_center, full_matrices=False)
    candidates = []
    for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        rotation = target_basis.T @ np.diag(signs) @ source_basis
        if np.linalg.det(rotation) < 0:
            rotation = target_basis.T @ np.diag(np.array(signs) * [1, 1, -1]) @ source_basis
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = target_center - rotation @ source_center
        candidates.append(pose)
    return candidates


class WholeSurfaceFit:
    def __init__(self, reference: np.ndarray, observation: dict, k: np.ndarray, config: ShapeConfig):
        self.reference = reference
        self.observation = observation
        self.k = k
        self.config = config
        self.target = observation.get("all_surface_points", observation["obs"]["points"])
        self.reference_tree = cKDTree(reference)
        self.target_tree = cKDTree(self.target)
        self.outside_mask = cv2.distanceTransform((~observation["mask"]).astype(np.uint8), cv2.DIST_L2, 5)
        y, x = np.nonzero(observation["mask"])
        mask_pixels = np.column_stack((x, y)).astype(float)
        self.mask_outline = mask_pixels[ConvexHull(mask_pixels).vertices]

    def uncovered_mask(self, uv: np.ndarray) -> np.ndarray:
        # Containment in the projected convex envelope is a conservative check:
        # it enforces observed outer extent without inventing an unseen mesh or
        # requiring an occluder's cut-out to occur in the reference shape.
        polygon = uv[ConvexHull(uv).vertices]
        start, end = polygon, np.roll(polygon, -1, axis=0)
        edge = end - start
        delta = self.mask_outline[:, None] - start[None]
        along = np.clip(np.sum(delta * edge[None], axis=2) / np.sum(edge**2, axis=1), 0, 1)
        distance = np.linalg.norm(delta - along[..., None] * edge[None], axis=2).min(axis=1)
        cross = edge[None, :, 0] * delta[..., 1] - edge[None, :, 1] * delta[..., 0]
        inside = np.all(cross >= 0, axis=1) | np.all(cross <= 0, axis=1)
        distance[inside] = 0
        return np.maximum(distance - self.config.mask_boundary_tolerance_px, 0)

    def residual(self, pose: np.ndarray) -> np.ndarray:
        predicted = move(self.reference, pose)
        uv = project(predicted, self.k)
        height, width = self.observation["depth"].shape
        inside = (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
        coordinates = (np.clip(uv[:, 1], 0, height - 1), np.clip(uv[:, 0], 0, width - 1))
        mask_distance = map_coordinates(self.outside_mask, coordinates, order=1, mode="nearest")
        tolerance = self.config.depth_tolerance_m
        # A nearer measured surface may hide a model point. Smoothly suppress
        # its one-sided residual; do not suppress current observed object data.
        # Interpolate visibility evidence from four rays, never their depths:
        # blending a foreground edge or a zero reading invents false geometry.
        cy, cx = coordinates
        x0, y0 = np.floor(cx).astype(int), np.floor(cy).astype(int)
        x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
        fx, fy = cx - x0, cy - y0
        weights = np.column_stack(((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy))
        depth = self.observation["depth"]
        rays = np.column_stack((depth[y0, x0], depth[y0, x1], depth[y1, x0], depth[y1, x1]))
        known = (rays > 0) & inside[:, None]
        ray_visibility = np.clip((rays - predicted[:, 2, None] + tolerance) / tolerance, 0, 1)
        depth_visibility = np.sum(weights * known * ray_visibility, axis=1)
        visibility = depth_visibility + np.sum(weights * ~known, axis=1)
        model_distance = np.minimum(self.target_tree.query(predicted)[0], self.config.max_surface_distance_m)
        target_local = (self.target - pose[:3, 3]) @ pose[:3, :3]
        target_distance = np.minimum(
            self.reference_tree.query(target_local)[0], self.config.max_surface_distance_m
        )
        silhouette_m = (
            np.maximum(mask_distance - self.config.mask_boundary_tolerance_px, 0)
            * predicted[:, 2]
            / self.k[0, 0]
        )
        # Match the full observed surface in both directions. Each block gets
        # equal total weight regardless of cloud density or visible point count.
        return np.concatenate(
            (
                np.sqrt(depth_visibility / len(predicted)) * model_distance / tolerance,
                target_distance / (tolerance * np.sqrt(len(self.target))),
                np.sqrt(visibility / len(predicted)) * silhouette_m / tolerance,
                self.uncovered_mask(uv)
                * np.median(predicted[:, 2])
                / (self.k[0, 0] * tolerance * np.sqrt(len(self.mask_outline))),
            )
        )

    def score(self, pose: np.ndarray) -> float:
        residual = self.residual(pose)
        return float(np.dot(residual, residual))

    def refine(self, initial: np.ndarray) -> tuple[np.ndarray, dict]:
        center = move(self.reference, initial).mean(0)

        def pose_from_delta(delta):
            rotation = Rotation.from_rotvec(delta[:3]).as_matrix()
            pose = initial.copy()
            pose[:3, :3] = rotation @ initial[:3, :3]
            pose[:3, 3] = rotation @ (initial[:3, 3] - center) + center + delta[3:]
            return pose

        solution = least_squares(
            lambda delta: self.residual(pose_from_delta(delta)),
            np.zeros(6),
            x_scale="jac",
            max_nfev=self.config.max_evaluations,
            ftol=self.config.solver_tolerance,
            xtol=self.config.solver_tolerance,
            gtol=self.config.solver_tolerance,
        )
        pose = pose_from_delta(solution.x)
        return pose, {
            "initial_cost": self.score(initial),
            "cost": self.score(pose),
            "evaluations": solution.nfev,
            "converged": bool(solution.success),
        }

    def correct(self, proposal: np.ndarray) -> tuple[np.ndarray, dict]:
        """Compare local refinement with a coarse whole-surface initialization.

        Choose the coarse branch closest to the proposed motion, so this does
        not freely exchange visually similar ends at every frame. Both results
        are scored against the same current mask and depth, not by PCA agreement.
        """
        local, local_detail = self.refine(proposal)
        predicted = move(self.reference, proposal)
        coarse = min(
            pca_candidates(self.reference, self.observation["obs"]["points"]),
            key=lambda pose: np.mean(np.sum((move(self.reference, pose) - predicted) ** 2, axis=1)),
        )
        if self.score(coarse) >= local_detail["cost"]:
            return local, {**local_detail, "used_coarse": False}
        alternative, alternative_detail = self.refine(coarse)
        chosen, detail = (
            (alternative, alternative_detail)
            if alternative_detail["cost"] < local_detail["cost"]
            else (local, local_detail)
        )
        detail = dict(detail)
        detail["initial_cost"] = self.score(proposal)
        detail["used_coarse"] = chosen is alternative
        detail["evaluations"] = local_detail["evaluations"] + alternative_detail["evaluations"]
        return chosen, detail
