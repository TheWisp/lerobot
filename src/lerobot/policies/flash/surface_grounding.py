"""Rigid whole-surface correction using measured RGB-D and an object mask.

The reference is an observed surface, not a CAD model or a cylinder. When its
source depth and mask are supplied, both observations are treated as partial:
only mutually visible surface requires a correspondence. Either image can still
rule out a placement in measured empty space. Callers without the source view
retain the older, unconditional current-to-reference coverage objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from os import cpu_count

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation

from .surface_observation import move, spatial_sample

# SciPy < 1.16 remains supported: use its original scalar derivative path.
_HAS_SOLVER_WORKERS = "workers" in signature(least_squares).parameters


@dataclass(frozen=True)
class ShapeConfig:
    # Observation tolerances inherited from the existing depth experiment.
    depth_tolerance_m: float = 0.005
    surface_spacing_m: float = 0.003
    mask_boundary_tolerance_px: float = 2.0
    # A surface reconstructed from partial observations cannot be required to
    # cover the whole current silhouette. Enable only for a complete reference.
    require_full_mask_coverage: bool = False
    max_surface_distance_m: float = 0.03
    max_evaluations: int = 50
    solver_tolerance: float = 1e-5
    # Per residual direction. Dense clouds remain the distance/outline ground truth.
    max_fit_points: int = 2048
    # Amortize cKDTree's per-query thread startup over this many residual points.
    nn_queries_per_worker: int = 512
    # Exact finite-difference candidates share nearest-neighbour tree queries.
    batch_jacobian: bool = True
    batch_query_workers: int = 8


@dataclass(frozen=True)
class PreparedReferenceSurface:
    """Fixed search geometry; build it once when a surface reference is chosen."""

    points: np.ndarray
    config: ShapeConfig
    tree: cKDTree
    queries: np.ndarray
    envelope: np.ndarray

    @classmethod
    def build(cls, reference: np.ndarray, config: ShapeConfig):
        from scipy.spatial import QhullError

        try:
            envelope = reference[ConvexHull(reference).vertices]
        except QhullError:  # Exactly planar references are valid observations.
            envelope = reference
        return cls(
            reference,
            config,
            cKDTree(reference),
            reference[spatial_sample(reference, config.max_fit_points)],
            envelope,
        )


@dataclass(frozen=True)
class SurfaceFitView:
    """An actual earlier depth/mask and its camera relative to the stored map."""

    observation: dict
    intrinsics: np.ndarray
    camera_from_reference: np.ndarray


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
    def __init__(
        self,
        reference: np.ndarray,
        observation: dict,
        k: np.ndarray,
        config: ShapeConfig,
        *,
        prepared_reference: PreparedReferenceSurface | None = None,
        fit_kernels=None,
        search_pool=None,
        reference_observation: dict | None = None,
        reference_intrinsics: np.ndarray | None = None,
        reference_views: tuple[SurfaceFitView, ...] = (),
    ):
        """Fit reference-camera points into the current camera.

        ``reference_observation`` must be the actual depth/mask measurement in
        the reference points' camera coordinates, not a rendering of the fitted
        map. A growing map in arbitrary object coordinates cannot use this
        two-view contract without carrying the source camera transforms too.
        """
        prepared_reference = prepared_reference or PreparedReferenceSurface.build(reference, config)
        if prepared_reference.points is not reference or prepared_reference.config != config:
            raise ValueError("prepared reference belongs to another surface or fit configuration")
        self.reference = reference
        self.observation = observation
        self.k = k
        self.config = config
        self.fit_kernels = fit_kernels
        self.target = observation.get("all_surface_points", observation["obs"]["points"])
        self.reference_tree = prepared_reference.tree
        self.target_tree = cKDTree(self.target)
        if search_pool is not None:
            from .surface_acceleration import PooledTree

            self.reference_tree = PooledTree(self.reference_tree, search_pool)
            self.target_tree = PooledTree(self.target_tree, search_pool)
        self.reference_queries = prepared_reference.queries
        self.target_queries = self.target[spatial_sample(self.target, config.max_fit_points)]
        cores = cpu_count() or 1
        self.reference_query_workers = min(
            cores,
            max(
                1,
                (len(self.reference_queries) + config.nn_queries_per_worker - 1)
                // config.nn_queries_per_worker,
            ),
        )
        self.target_query_workers = min(
            cores,
            max(
                1,
                (len(self.target_queries) + config.nn_queries_per_worker - 1) // config.nn_queries_per_worker,
            ),
        )
        # Linear projection preserves the convex envelope of the 3D hull for
        # points in front of the camera. Retain its vertices independently of
        # residual sampling, including small but distinctive extremities.
        self.envelope = prepared_reference.envelope
        self.outside_mask = cv2.distanceTransform((~observation["mask"]).astype(np.uint8), cv2.DIST_L2, 5)
        contours, _ = cv2.findContours(
            observation["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        mask_pixels = np.concatenate(contours)[:, 0].astype(float)
        self.mask_outline = mask_pixels[ConvexHull(mask_pixels).vertices]
        if reference_observation is not None and reference_views:
            raise ValueError("supply one source observation or retained source views, not both")
        if reference_observation is not None:
            reference_views = (
                SurfaceFitView(
                    reference_observation,
                    k if reference_intrinsics is None else reference_intrinsics,
                    np.eye(4),
                ),
            )
        self.reference_views = tuple(reference_views)
        self.reverse_fits = []
        for view in self.reference_views:
            reverse = WholeSurfaceFit(
                self.target,
                view.observation,
                view.intrinsics,
                config,
                fit_kernels=fit_kernels,
                search_pool=search_pool,
            )
            self.reverse_fits.append((view.camera_from_reference, reverse))

    def uncovered_mask(self, uv: np.ndarray) -> np.ndarray:
        if not self.config.require_full_mask_coverage:
            return np.zeros(len(self.mask_outline))
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
        predicted = move(self.reference_queries, pose)
        target_local = (self.target_queries - pose[:3, 3]) @ pose[:3, :3]
        model_distance = self.target_tree.query(predicted, workers=self.reference_query_workers)[0]
        target_distance = self.reference_tree.query(target_local, workers=self.target_query_workers)[0]
        return self._residual_with_distances(pose, predicted, model_distance, target_distance)

    def batch_residuals(self, poses: list[np.ndarray]) -> list[np.ndarray]:
        """Same residuals, with two exact tree searches for all candidate poses."""
        if not poses:
            return []
        predicted = [move(self.reference_queries, pose) for pose in poses]
        local = [(self.target_queries - pose[:3, 3]) @ pose[:3, :3] for pose in poses]
        workers = min(cpu_count() or 1, self.config.batch_query_workers)
        model_distance = self.target_tree.query(np.concatenate(predicted), workers=workers)[0]
        target_distance = self.reference_tree.query(np.concatenate(local), workers=workers)[0]
        return [
            self._residual_with_distances(pose, points, model, target)
            for pose, points, model, target in zip(
                poses,
                predicted,
                model_distance.reshape(len(poses), -1),
                target_distance.reshape(len(poses), -1),
                strict=True,
            )
        ]

    def _residual_with_distances(self, pose, predicted, model_distance, target_distance):
        forward = self._one_view_residual(pose, predicted, model_distance, target_distance)
        if not self.reverse_fits:
            return forward
        n, m = len(self.reference_queries), len(self.target_queries)
        inverse = np.linalg.inv(pose)
        reverse = [
            (fit.residual(camera @ inverse), len(fit.target_queries)) for camera, fit in self.reverse_fits
        ]
        # Keep the visibility-tested distance and silhouette terms from BOTH
        # cameras. Drop both unconditional coverage terms: a newly exposed side
        # need not coincide with the previously measured side of the object.
        # Seeing the same point in many retained views must not multiply its
        # weight. Any view that measured it can require a correspondence; any
        # measured empty space can contradict it. Unknown/hidden views add none.
        matched = np.maximum.reduce([r[:m] for r, _ in reverse])
        free = np.maximum.reduce([r[m + count : m + count + m] for r, count in reverse])
        outlines = [r[m + count + m :] / np.sqrt(len(reverse)) for r, count in reverse]
        return np.concatenate((forward[:n], matched, forward[n + m :], free, *outlines))

    def _one_view_residual(self, pose, predicted, model_distance, target_distance):
        if self.fit_kernels is not None:
            uv = project(predicted, self.k)
            height, width = self.observation["depth"].shape
            coordinates = (np.clip(uv[:, 1], 0, height - 1), np.clip(uv[:, 0], 0, width - 1))
            mask_distance = map_coordinates(self.outside_mask, coordinates, order=1, mode="nearest")
            measured = self.fit_kernels[0](
                predicted,
                model_distance,
                target_distance,
                mask_distance,
                self.observation["depth"],
                self.observation["mask"],
                uv,
                self.config.depth_tolerance_m,
                self.config.max_surface_distance_m,
                self.config.mask_boundary_tolerance_px,
                self.k[0, 0],
            )
            if self.config.require_full_mask_coverage:
                envelope = project(move(self.envelope, pose), self.k)
                polygon = envelope[ConvexHull(envelope).vertices]
                outline = self.fit_kernels[1](
                    self.mask_outline, polygon, self.config.mask_boundary_tolerance_px
                )
            else:
                outline = np.zeros(len(self.mask_outline))
            outline = (
                outline
                * np.median(predicted[:, 2])
                / (self.k[0, 0] * self.config.depth_tolerance_m * np.sqrt(len(self.mask_outline)))
            )
            return np.concatenate((measured, outline))
        tolerance = self.config.depth_tolerance_m
        depth_visibility, visibility, silhouette_m = self._projection_evidence(predicted)
        model_distance = np.minimum(model_distance, self.config.max_surface_distance_m)
        target_distance = np.minimum(target_distance, self.config.max_surface_distance_m)
        return np.concatenate(
            (
                np.sqrt(depth_visibility / len(predicted)) * model_distance / tolerance,
                target_distance / (tolerance * np.sqrt(len(self.target_queries))),
                np.sqrt(visibility / len(predicted)) * silhouette_m / tolerance,
                self.uncovered_mask(project(move(self.envelope, pose), self.k))
                * np.median(predicted[:, 2])
                / (self.k[0, 0] * tolerance * np.sqrt(len(self.mask_outline))),
            )
        )

    def _projection_evidence(self, predicted):
        """Visibility and empty-space evidence, without blending measured depths."""
        uv = project(predicted, self.k)
        height, width = self.observation["depth"].shape
        coordinates = (np.clip(uv[:, 1], 0, height - 1), np.clip(uv[:, 0], 0, width - 1))
        mask_distance = map_coordinates(self.outside_mask, coordinates, order=1, mode="nearest")
        inside = (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
        tolerance = self.config.depth_tolerance_m
        # A nearer measured surface may hide a model point. Smoothly suppress
        # its one-sided residual. The reverse view applies the same test to
        # current data when the earlier measurement is available.
        # Interpolate visibility evidence from four rays, never their depths:
        # blending a foreground edge or a zero reading invents false geometry.
        cy, cx = coordinates
        x0, y0 = np.floor(cx).astype(int), np.floor(cy).astype(int)
        x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
        fx, fy = cx - x0, cy - y0
        weights = np.column_stack(((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy))
        depth = self.observation["depth"]
        rays = np.column_stack((depth[y0, x0], depth[y0, x1], depth[y1, x0], depth[y1, x1]))
        mask = self.observation["mask"]
        object_rays = np.column_stack((mask[y0, x0], mask[y0, x1], mask[y1, x0], mask[y1, x1]))
        known = (rays > 0) & inside[:, None]
        ray_visibility = np.clip((rays - predicted[:, 2, None] + tolerance) / tolerance, 0, 1)
        # Only rays belonging to the observed object require a nearest-surface
        # match. A reference point behind an occluder has no visible counterpart.
        depth_visibility = np.sum(weights * known * object_rays * ray_visibility, axis=1)
        # Outside the visible mask, reject a projected reference only when
        # measured depth proves space in front of a FARTHER surface is empty.
        # Nearer/unknown depth, or a gap within depth uncertainty, cannot do so.
        free_space = np.clip((rays - predicted[:, 2, None] - tolerance) / tolerance, 0, 1)
        visibility = np.sum(weights * known * free_space, axis=1)
        silhouette_m = (
            np.maximum(mask_distance - self.config.mask_boundary_tolerance_px, 0)
            * predicted[:, 2]
            / self.k[0, 0]
        )
        return depth_visibility, visibility, silhouette_m

    def partial_surface_check(self, pose):
        """Direct agreement for a pair of measured views; no covariance involved.

        Unseen samples neither pass nor fail a surface correspondence. If neither
        view has any visible overlap, this check cannot validate a pose.
        """
        if not self.reverse_fits:
            raise ValueError("partial surface checks require the source depth and mask")
        outliers, free_space, visible = [], [], []
        predicted = move(self.reference_queries, pose)
        weight, empty, silhouette = self._projection_evidence(predicted)
        inverse = np.linalg.inv(pose)
        previous = []
        for camera, fit in self.reverse_fits:
            old_points = move(fit.reference_queries, camera @ inverse)
            old_weight, old_empty, old_silhouette = fit._projection_evidence(old_points)
            distances = fit.target_tree.query(old_points)[0]
            previous.append((old_weight, old_empty, old_silhouette, distances))
        directions = [
            (weight, np.sqrt(empty) * silhouette, self.target_tree.query(predicted)[0]),
            (
                np.maximum.reduce([p[0] for p in previous]),
                np.maximum.reduce([np.sqrt(p[1]) * p[2] for p in previous]),
                None,
            ),
        ]
        for direction, (weight, empty_distance, distance) in enumerate(directions):
            count = float(weight.sum())
            visible.append(count)
            bad_weight = (
                weight * (distance > self.config.depth_tolerance_m)
                if direction == 0
                else np.maximum.reduce([p[0] * (p[3] > self.config.depth_tolerance_m) for p in previous])
            )
            outliers.append(float(bad_weight.sum() / count) if count else 0.0)
            free_space.append(float(np.mean(empty_distance > self.config.depth_tolerance_m)))
        return {
            "observed_outlier_fraction": max(outliers),
            "free_space_outlier_fraction": max(free_space),
            "visible_reference_weight": visible[0],
            "visible_current_weight": visible[1],
            "has_visible_overlap": any(count > 0 for count in visible),
        }

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

        def batch_map(_fun, deltas):
            return self.batch_residuals([pose_from_delta(delta) for delta in deltas])

        batched = self.config.batch_jacobian and _HAS_SOLVER_WORKERS
        solution = least_squares(
            lambda delta: self.residual(pose_from_delta(delta)),
            np.zeros(6),
            x_scale="jac",
            max_nfev=self.config.max_evaluations,
            ftol=self.config.solver_tolerance,
            xtol=self.config.solver_tolerance,
            gtol=self.config.solver_tolerance,
            **({"workers": batch_map} if batched else {}),
        )
        pose = pose_from_delta(solution.x)
        return pose, {
            "initial_cost": self.score(initial),
            "cost": self.score(pose),
            "evaluations": solution.nfev,
            "converged": bool(solution.success),
            "batched_jacobian": batched,
            "fit_reference_points": len(self.reference_queries),
            "fit_target_points": len(self.target_queries),
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
