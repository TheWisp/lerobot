"""Rigid whole-surface correction using measured RGB-D and an object mask.

The reference is an observed surface, not a CAD model or a cylinder. When its
source depth and mask are supplied, both observations are treated as partial:
only mutually visible surface requires a correspondence. Either image can still
rule out a placement in measured empty space, except that one retained image
cannot veto current surface lying on the saved surface: every view that built
the surface measured it there. Callers without the source view retain the
older, unconditional current-to-reference coverage objective.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from itertools import permutations
from numbers import Integral
from os import cpu_count

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree

from .surface_observation import move, reliable_observation_depth, spatial_sample
from .surface_registration import LibraryConfig


@dataclass(frozen=True)
class ShapeConfig:
    registration: LibraryConfig = field(default_factory=LibraryConfig)
    # Observation tolerances inherited from the existing depth experiment.
    depth_tolerance_m: float = 0.005
    surface_spacing_m: float = 0.003
    mask_boundary_tolerance_px: float = 2.0
    # Legacy name: enables a soft cost for uncovered current outline. Missing
    # coverage is not a contradiction: measured RGB-D surfaces are partial.
    # This experiment remains opt-in because a strong cost can bias a partial fit.
    require_full_mask_coverage: bool = False
    # Multiplier of that squared-error cost: 0 disables the preference; 1 keeps
    # the previous strength. This does not weaken depth/free-space checks.
    mask_coverage_weight: float = 1.0
    # Add scene depth to the search objective. Final empty-space validation is
    # mandatory regardless of this flag; holes/nearer occluders are not conflicts.
    scene_depth_constraints: bool = False
    # Experimental search preference: unknown surface is not a perfect match.
    prefer_observed_overlap: bool = False
    max_surface_distance_m: float = 0.03
    max_evaluations: int = 50
    solver_tolerance: float = 1e-5
    # Per residual direction. Dense clouds remain the distance/outline ground truth.
    max_fit_points: int = 2048
    # Amortize cKDTree's per-query thread startup over this many residual points.
    nn_queries_per_worker: int = 512
    batch_query_workers: int = 8

    def __post_init__(self):
        if isinstance(self.registration, dict):
            object.__setattr__(self, "registration", LibraryConfig(**self.registration))
        value = self.max_evaluations
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
            raise ValueError("max_evaluations must be an integer >= 1")


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

        envelope = reference
        if config.require_full_mask_coverage:
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
class PreparedTargetSurface:
    """One frame's measured geometry, shared by independent fit objectives."""

    observation: dict
    config: ShapeConfig
    points: np.ndarray
    tree: cKDTree
    queries: np.ndarray
    depth: np.ndarray
    outside_mask: np.ndarray
    mask_outline: np.ndarray
    scene_nearest_depth: np.ndarray

    @classmethod
    def build(cls, observation: dict, config: ShapeConfig):
        points = observation.get("all_surface_points", observation["obs"]["points"])
        depth = reliable_observation_depth(observation)
        raw = (
            reliable_observation_depth(observation, observation["scene_depth"])
            if "scene_depth" in observation
            else depth
        )
        radius = int(np.ceil(config.mask_boundary_tolerance_px))
        scene_nearest_depth = cv2.erode(
            raw,
            np.ones((2 * radius + 1,) * 2, np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask = observation["mask"]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_pixels = np.concatenate(contours)[:, 0].astype(float)
        return cls(
            observation,
            config,
            points,
            cKDTree(points),
            points[spatial_sample(points, config.max_fit_points)],
            depth,
            cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5),
            mask_pixels[ConvexHull(mask_pixels).vertices],
            scene_nearest_depth,
        )


@dataclass(frozen=True)
class SurfaceFitView:
    """An actual earlier depth/mask and its camera relative to the stored map."""

    observation: dict
    intrinsics: np.ndarray
    camera_from_reference: np.ndarray


def project(points: np.ndarray, k: np.ndarray) -> np.ndarray:
    return points[:, :2] / points[:, 2, None] @ k[:2, :2].T + k[:2, 2]


def pca_candidates(source: np.ndarray, target: np.ndarray, *, all_axis_orders=False) -> list[np.ndarray]:
    """Generic coarse initializations; never used as a pose or axis measurement."""
    source_center, target_center = source.mean(0), target.mean(0)
    _, _, source_basis = np.linalg.svd(source - source_center, full_matrices=False)
    _, _, target_basis = np.linalg.svd(target - target_center, full_matrices=False)
    candidates = []
    orders = permutations(range(3)) if all_axis_orders else [(0, 1, 2)]
    for order in orders:
        basis = target_basis[list(order)]
        for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
            rotation = basis.T @ np.diag(signs) @ source_basis
            if np.linalg.det(rotation) < 0:
                rotation = basis.T @ np.diag(np.array(signs) * [1, 1, -1]) @ source_basis
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
        prepared_target: PreparedTargetSurface | None = None,
        fit_kernels=None,
        search_pool=None,
        reference_observation: dict | None = None,
        reference_intrinsics: np.ndarray | None = None,
        reference_views: tuple[SurfaceFitView, ...] = (),
        appearance_views: tuple[SurfaceFitView, ...] = (),
        prepared_reverse_fits=None,
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
        prepared_target = prepared_target or PreparedTargetSurface.build(observation, config)
        if prepared_target.observation is not observation or prepared_target.config != config:
            raise ValueError("prepared target belongs to another observation or fit configuration")
        self.prepared_reference = prepared_reference
        self.prepared_target = prepared_target
        self.reference = reference
        self.observation = observation
        self.k = k
        self.config = config
        self.fit_kernels = fit_kernels
        self.search_pool = search_pool
        self.target = prepared_target.points
        self.reference_tree = prepared_reference.tree
        self.target_tree = prepared_target.tree
        if search_pool is not None:
            from .surface_acceleration import PooledTree

            self.reference_tree = PooledTree(self.reference_tree, search_pool)
            self.target_tree = PooledTree(self.target_tree, search_pool)
        self.reference_queries = prepared_reference.queries
        self.target_queries = prepared_target.queries
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
        self.outside_mask = prepared_target.outside_mask
        self.depth = prepared_target.depth
        self.scene_nearest_depth = prepared_target.scene_nearest_depth
        self.mask_outline = prepared_target.mask_outline
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
        self.appearance_views = tuple(appearance_views) or self.reference_views
        self._registration_solver = None
        self._residual_cache = OrderedDict()
        # Image-conditioned and geometry-only objectives use identical measured
        # views. Share their preparation, rather than rebuilding depth images,
        # trees and hulls for the second objective in the same frame.
        if prepared_reverse_fits is not None:
            if len(prepared_reverse_fits) != len(self.reference_views) or any(
                fit.reference is not self.target
                or fit.config != replace(config, require_full_mask_coverage=False)
                or not np.array_equal(fit.k, view.intrinsics)
                or fit.observation is not view.observation
                or not np.array_equal(camera, view.camera_from_reference)
                for (camera, fit), view in zip(prepared_reverse_fits, self.reference_views, strict=True)
            ):
                raise ValueError("prepared views belong to another fit")
            self.reverse_fits = prepared_reverse_fits
            return
        self.reverse_fits = []
        for view in self.reference_views:
            reverse = WholeSurfaceFit(
                self.target,
                view.observation,
                view.intrinsics,
                replace(config, require_full_mask_coverage=False),
                fit_kernels=fit_kernels,
                search_pool=search_pool,
            )
            self.reverse_fits.append((view.camera_from_reference, reverse))

    @property
    def coverage_enabled(self) -> bool:
        return self.config.require_full_mask_coverage

    @property
    def has_image_evidence(self) -> bool:
        return False

    def uncovered_mask(self, uv: np.ndarray) -> np.ndarray:
        if not self.coverage_enabled:
            return np.zeros(len(self.mask_outline))
        # Distance outside the projected outer envelope, not a complete mesh
        # silhouette. This is a search preference, not proof of a contradiction.
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

    def residual(self, pose: np.ndarray, *, include_coverage=True) -> np.ndarray:
        key = self.config, include_coverage, pose.tobytes()
        if key in self._residual_cache:
            self._residual_cache.move_to_end(key)
            return self._residual_cache[key]
        predicted = move(self.reference_queries, pose)
        model_distance = self.target_tree.query(predicted, workers=self.reference_query_workers)[0]
        # Partial-view composition discards unconditional target coverage.
        # Do not compute a nearest-neighbour search whose result is discarded.
        target_distance = (
            self.reference_tree.query(
                (self.target_queries - pose[:3, 3]) @ pose[:3, :3], workers=self.target_query_workers
            )[0]
            if include_coverage and not self.reverse_fits
            else np.zeros(len(self.target_queries))
        )
        return self._remember_residual(
            key, self._residual_with_distances(pose, predicted, model_distance, target_distance)
        )

    def _remember_residual(self, key, residual):
        # Bounded, frame-local workspace. 64 entries keep derivative batches and
        # shortlisted poses without retaining every optimizer evaluation.
        self._residual_cache[key] = residual
        self._residual_cache.move_to_end(key)
        if len(self._residual_cache) > 64:
            self._residual_cache.popitem(last=False)
        return residual

    def batch_residuals(self, poses: list[np.ndarray], *, include_coverage=True) -> list[np.ndarray]:
        """Same residuals, with two exact tree searches for all candidate poses."""
        if not poses:
            return []
        predicted = [move(self.reference_queries, pose) for pose in poses]
        workers = min(cpu_count() or 1, self.config.batch_query_workers)
        model_distance = self.target_tree.query(np.concatenate(predicted), workers=workers)[0]
        target_distance = (
            self.reference_tree.query(
                np.concatenate([(self.target_queries - pose[:3, 3]) @ pose[:3, :3] for pose in poses]),
                workers=workers,
            )[0]
            if include_coverage and not self.reverse_fits
            else np.zeros(len(poses) * len(self.target_queries))
        )
        # Prepare the reverse observations in a batch too. Otherwise every
        # forward candidate immediately starts another scalar tree query.
        if self.reverse_fits:
            inverses = np.linalg.inv(np.asarray(poses))
            for camera, fit in self.reverse_fits:
                fit.batch_residuals(list(camera @ inverses), include_coverage=False)
        return [
            self._remember_residual(
                (self.config, include_coverage, pose.tobytes()),
                self._residual_with_distances(pose, points, model, target),
            )
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
            (fit.residual(camera @ inverse, include_coverage=False), len(fit.target_queries))
            for camera, fit in self.reverse_fits
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
        residual = self._legacy_view_residual(pose, predicted, model_distance, target_distance)
        if self.config.prefer_observed_overlap:
            visible, _, _ = self._projection_evidence(predicted)
            agreement = visible * np.exp(-((model_distance / self.config.depth_tolerance_m) ** 2))
            residual[: len(predicted)] = np.sqrt(np.clip(1 - agreement, 0, 1) / len(predicted))
        if self.config.scene_depth_constraints:
            free, exposed, _ = self._scene_depth_evidence(predicted)
            n, m = len(predicted), len(self.target_queries)
            # Keep a fixed residual layout for the optimizer and reverse-view
            # composition. The same augmentation runs with native/Python fits.
            residual[n + m : n + m + n] = np.maximum(
                residual[n + m : n + m + n],
                free / (self.config.depth_tolerance_m * np.sqrt(max(exposed.sum(), 1.0))),
            )
        return residual

    def _legacy_view_residual(self, pose, predicted, model_distance, target_distance):
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
                self.depth,
                self.observation["mask"],
                uv,
                self.config.depth_tolerance_m,
                self.config.max_surface_distance_m,
                self.config.mask_boundary_tolerance_px,
                self.k[0, 0],
            )
            if self.coverage_enabled:
                envelope = project(move(self.envelope, pose), self.k)
                polygon = envelope[ConvexHull(envelope).vertices]
                outline = self.fit_kernels[1](
                    self.mask_outline, polygon, self.config.mask_boundary_tolerance_px
                )
            else:
                outline = np.zeros(len(self.mask_outline))
            outline = (
                outline
                * np.sqrt(self.config.mask_coverage_weight)
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
                (
                    self.uncovered_mask(project(move(self.envelope, pose), self.k))
                    if self.coverage_enabled
                    else np.zeros(len(self.mask_outline))
                )
                * np.sqrt(self.config.mask_coverage_weight)
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
        if self.fit_kernels is not None:
            return self.fit_kernels[2](
                predicted,
                uv,
                mask_distance,
                self.depth,
                self.observation["mask"],
                self.config.depth_tolerance_m,
                self.config.mask_boundary_tolerance_px,
                self.k[0, 0],
            )
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
        depth = self.depth
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

    def check_pose(self, pose, *, max_outlier_fraction=None, agreement_m=None, min_overlap=0.0):
        """Measured geometry only: independent of solver residuals and Kalman.

        The standalone fitter requires no coherent protrusion. Tracking callers
        also supply their existing outlier/overlap limits. The same policy is
        used during search and after temporal fusion; growth has separate rules.
        """
        scene = self.scene_depth_check(pose)
        if max_outlier_fraction is None and min_overlap == 0:
            return {**scene, "accepted": not scene["scene_depth_coherent_protrusion"]}
        if scene["scene_depth_coherent_protrusion"]:
            # Measured empty space is a hard contradiction. No amount of
            # object-surface overlap can accept this pose, so do not run the
            # two-way nearest-neighbour check on a known rejected candidate.
            return {
                **scene,
                "accepted": False,
                "geometry_accepted": False,
                "observed_overlap": 0.0,
                "observed_overlap_basis": "not_tested_after_protrusion",
                "surface_median_mm": None,
            }
        distances = self.prepared_reference.tree.query(
            (self.observation["obs"]["points"] - pose[:3, 3]) @ pose[:3, :3]
        )[0]
        if self.reverse_fits:
            direct = self.partial_surface_check(pose)
            overlap = 1.0 - direct["observed_outlier_fraction"] if direct["has_visible_overlap"] else 0.0
        else:
            predicted = move(self.reference_queries, pose)
            _, visibility, silhouette = self._projection_evidence(predicted)
            direct = {
                "observed_outlier_fraction": float(np.mean(distances > self.config.depth_tolerance_m)),
                "free_space_outlier_fraction": float(
                    np.mean(np.sqrt(visibility) * silhouette > self.config.depth_tolerance_m)
                ),
                "has_visible_overlap": bool(np.any(distances <= self.config.depth_tolerance_m)),
            }
            overlap = float(np.mean(distances <= (agreement_m or self.config.depth_tolerance_m)))
        accepted = not scene["scene_depth_coherent_protrusion"]
        if max_outlier_fraction is not None:
            accepted &= (
                direct["has_visible_overlap"]
                and max(
                    direct["observed_outlier_fraction"],
                    direct["free_space_outlier_fraction"],
                    scene["scene_depth_outlier_fraction"],
                )
                <= max_outlier_fraction
            )
        geometry_accepted = bool(accepted)
        accepted &= overlap >= min_overlap
        return {
            **direct,
            **scene,
            "accepted": bool(accepted),
            "geometry_accepted": geometry_accepted,
            "observed_overlap": overlap,
            "observed_overlap_basis": "mutually_visible_points"
            if self.reverse_fits
            else "all_current_points",
            "surface_median_mm": float(np.median(distances) * 1000),
        }

    def _scene_depth_evidence(self, predicted):
        """Depth bounds along rays, independent of object point-cloud trimming.

        Returns metric free-space violation, exposed weight and contradicted
        weight. Hidden/unknown samples contribute to neither denominator nor
        numerator. Interpolation combines evidence, never depth across a cliff.
        """
        uv = project(predicted, self.k)
        if self.fit_kernels is not None:
            return self.fit_kernels[3](predicted, uv, self.scene_nearest_depth, self.config.depth_tolerance_m)
        h, w = self.scene_nearest_depth.shape
        valid = (
            np.isfinite(uv).all(1)
            & (predicted[:, 2] > 0)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] <= w - 1)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] <= h - 1)
        )
        x, y = np.nan_to_num(uv[:, 0]).clip(0, w - 1), np.nan_to_num(uv[:, 1]).clip(0, h - 1)
        x0, y0 = x.astype(int), y.astype(int)
        x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
        fx, fy = x - x0, y - y0
        weights = np.column_stack(((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy))
        depth = self.scene_nearest_depth
        near = np.column_stack((depth[y0, x0], depth[y0, x1], depth[y1, x0], depth[y1, x1]))
        known = (near > 0) & valid[:, None]
        gap = near - predicted[:, 2, None]
        tolerance = self.config.depth_tolerance_m
        exposed = np.sum(weights * known * np.clip((gap + tolerance) / tolerance, 0, 1), axis=1)
        contradicted = np.sum(weights * known * (gap > tolerance), axis=1)
        free = np.sqrt(np.sum(weights * known * np.maximum(gap - tolerance, 0) ** 2, axis=1))
        return free, exposed, contradicted

    def scene_depth_check(self, pose):
        # A solver's subsampling budget must not hide a contradictory patch.
        directions = [(self, move(self.reference, pose), False)]
        inverse = np.linalg.inv(pose)
        directions.extend(
            (fit, move(fit.reference, camera @ inverse), True) for camera, fit in self.reverse_fits
        )
        fractions, exposed_counts, coherent_protrusions = [], [], []
        for fit, predicted, retained in directions:
            _, exposed, bad = fit._scene_depth_evidence(predicted)
            if retained and bad.any():
                # One retained frame measured this surface once; the saved
                # surface aggregates every accepted view. A current sample on
                # the saved surface under this pose therefore outvotes the one
                # older reading that puts it in air (rim depth read millimetres
                # farther in a single frame). Current surface beyond the saved
                # surface, and the current image's own air, keep the full veto.
                contradicted = np.flatnonzero(bad)
                distance = self.reference_tree.query(move(fit.reference[contradicted], inverse))[0]
                bad = bad.copy()
                bad[contradicted[distance <= self.config.depth_tolerance_m]] = 0.0
            count = float(exposed.sum())
            exposed_counts.append(count)
            fractions.append(float(bad.sum() / count) if count else 0.0)
            coherent_protrusions.append(fit._coherent_protrusion(predicted, bad))
        return {
            "scene_depth_outlier_fraction": max(fractions),
            "scene_depth_exposed_weights": exposed_counts,
            "scene_depth_coherent_protrusion": any(coherent_protrusions),
            # The first direction projects the map into the current image;
            # subsequent directions project current measurements into each
            # retained image. Keep these distinct for live failure diagnosis.
            "scene_depth_views": [
                {
                    "frame_id": fit.observation.get("frame_id"),
                    "direction": "map_to_current" if index == 0 else "current_to_retained",
                    "outlier_fraction": fractions[index],
                    "coherent_protrusion": coherent_protrusions[index],
                }
                for index, (fit, _, _) in enumerate(directions)
            ],
        }

    def _coherent_protrusion(self, predicted, contradicted):
        """Find a connected patch protruding into measured empty space.

        A single depth disagreement can be a bad map point or sensor reading.
        A patch wider than one surface-sample neighborhood is pose evidence.
        This also applies inside the mask: a surface floating in front of the
        measured object is impossible too. Depth already uses the conservative
        neighborhood minimum, so holes and closer occluders cannot reject it.
        """
        indices = np.flatnonzero(contradicted > 0)
        if len(indices) < 2:
            return False
        uv = project(predicted[indices], self.k)
        h, w = self.outside_mask.shape
        valid = np.isfinite(uv).all(axis=1) & (predicted[indices, 2] > 0)
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        uv = uv[valid]
        depth = predicted[indices[valid], 2]
        if len(uv) < 2:
            return False
        # One projected surface-sample neighborhood sets both connectivity
        # and the minimum span. Round to image pixels so float32 depth rounding
        # cannot disconnect adjacent samples. No extra threshold is introduced.
        neighborhood = max(
            self.config.mask_boundary_tolerance_px,
            np.ceil(self.k[0, 0] * self.config.surface_spacing_m / np.median(depth)),
        )
        pairs = cKDTree(uv).query_pairs(neighborhood, output_type="ndarray")
        if not len(pairs):
            return False
        edges = np.concatenate((pairs, pairs[:, ::-1]))
        graph = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(uv), len(uv)))
        count, labels = connected_components(graph, directed=False)
        # Reduce each component in one array pass. Scanning all pixels once
        # per component made scattered depth disagreements quadratic in count.
        low = np.full((count, 2), np.inf)
        high = np.full((count, 2), -np.inf)
        np.minimum.at(low, labels, uv)
        np.maximum.at(high, labels, uv)
        return bool(np.any(np.linalg.norm(high - low, axis=1) > 2 * neighborhood))

    def score(self, pose: np.ndarray) -> float:
        residual = self.residual(pose)
        cost = float(np.dot(residual, residual))
        if self.config.registration.appearance_weight:
            from .surface_registration import LibraryMatcher

            cost += self.config.registration.appearance_weight * LibraryMatcher.appearance_cost(self, pose)
        return cost

    def _solve(self, initial, max_evaluations):
        """Run the configured local registration solver from one starting pose."""
        if self.config.registration.backend == "visible_icp":
            from .surface_rigid_fit import solve

            return solve(self, initial, max_evaluations)
        from .surface_registration import LibraryMatcher

        if self._registration_solver is None:
            self._registration_solver = LibraryMatcher(self.config.registration)
        solver = self._registration_solver
        pose, detail = solver.refine(self, initial, max_evaluations=max_evaluations)
        return pose, detail, [(self.score(p), p) for p in solver._local.proposals]

    def fit_once(
        self,
        proposal,
        *,
        geometry=None,
        check=None,
    ):
        """Run one bounded registration; publish only an observed-compatible iterate."""
        geometry = self if geometry is None else geometry
        check = geometry.check_pose if check is None else check
        objective = (
            self
            if self.has_image_evidence and self.config.registration.backend == "visible_icp"
            else geometry
        )
        final, detail, iterates = objective._solve(proposal, self.config.max_evaluations)
        poses = list({pose.tobytes(): pose for _, pose in iterates}.values())
        if final.tobytes() not in {pose.tobytes() for pose in poses}:
            poses.append(final)
        geometry.batch_residuals(poses)
        ranked = sorted(poses, key=geometry.score)
        checks = {}
        chosen = None
        for pose in ranked:
            verdict = check(pose)
            checks[pose.tobytes()] = verdict
            if verdict["accepted"]:
                chosen = pose
                break
        selected = chosen if chosen is not None else final
        verdict = checks.get(selected.tobytes())
        if verdict is None:
            verdict = check(selected)
            checks[selected.tobytes()] = verdict
        detail["retained_admissible_iterate"] = chosen is not None
        return [
            {
                "pose": selected.copy(),
                "name": "local",
                "check": verdict,
                "cost": geometry.score(selected),
                "detail": detail,
            }
        ], {
            "evaluations": detail["evaluations"],
            "evaluation_budget": self.config.max_evaluations,
            "validation_count": len(checks),
            "numerical_solves": 1,
        }

    def refine(self, initial, *, check=None):
        candidates, detail = self.fit_once(initial, check=check)
        return self._fit_result(candidates, detail, initial)

    def _fit_result(self, candidates, detail, initial):
        valid = [c for c in candidates if c["check"]["accepted"]]
        if not valid:
            rejected = min(candidates, key=lambda c: c["cost"]) if candidates else None
            return None, {
                **(rejected["detail"] if rejected else {}),
                **detail,
                **(rejected["check"] if rejected else {}),
                "cost": rejected["cost"] if rejected else None,
                "used_coarse": False,
                "converged": False,
                "reason": "no_observation_compatible_candidate",
            }
        best = min(valid, key=lambda c: c["cost"])
        return best["pose"], {
            **best["detail"],
            **detail,
            "initial_cost": self.score(initial),
            "cost": best["cost"],
            "geometry_check": best["check"],
            "used_coarse": False,
        }

    def correct(self, proposal, *, check=None):
        candidates, detail = self.fit_once(proposal, check=check)
        return self._fit_result(candidates, detail, proposal)
