"""Optional CPU registration solvers; the fitter checks every published pose.

Imports of Open3D/small_gicp are lazy. Each fitter owns its solver and caches;
selecting a backend never patches another object's running fitter.
"""

import time
from dataclasses import dataclass
from importlib.metadata import version
from numbers import Integral
from threading import local

import numpy as np

from .surface_observation import move

MATCHERS = {
    "visible_icp": "Surface fit",
    "small_gicp": "small_gicp — GICP",
    "open3d_plane": "Open3D — point-to-plane",
    "open3d_color": "Open3D — geometry + color",
}


@dataclass(frozen=True)
class LibraryConfig:
    backend: str = "visible_icp"
    voxel_sizes_m: tuple[float, ...] = (0.006, 0.003)
    iterations_per_level: int = 20
    correspondence_voxels: float = 3.0
    threads: int = 2
    translation_epsilon_m: float = 0.0001
    rotation_epsilon_rad: float = 0.001
    colored_geometry_weight: float = 0.968  # Open3D's published default.
    appearance_weight: float = 1.0

    def __post_init__(self):
        if self.backend not in MATCHERS:
            raise ValueError("unknown registration backend")
        object.__setattr__(self, "voxel_sizes_m", tuple(self.voxel_sizes_m))
        if not self.voxel_sizes_m or any(not np.isfinite(v) or v <= 0 for v in self.voxel_sizes_m):
            raise ValueError("voxel sizes must be positive")
        if any(
            not isinstance(v, Integral) or isinstance(v, bool) or v < 1
            for v in (self.iterations_per_level, self.threads)
        ):
            raise ValueError("positive iteration, thread and correspondence budgets required")
        if any(
            not np.isfinite(v) or v <= 0
            for v in (self.correspondence_voxels, self.translation_epsilon_m, self.rotation_epsilon_rad)
        ):
            raise ValueError("registration tolerances must be positive and finite")
        if not np.isfinite(self.appearance_weight) or self.appearance_weight < 0:
            raise ValueError("appearance weight must be nonnegative and finite")
        if not np.isfinite(self.colored_geometry_weight) or not 0 <= self.colored_geometry_weight <= 1:
            raise ValueError("colored geometry weight must be in [0, 1]")


def measured_colors(points, observation, intrinsics, camera_from_reference, tolerance):
    """Only actual visible RGB-D samples supply color; hidden points have none."""
    from .surface_grounding import project

    colors = np.zeros_like(points, dtype=float)
    valid = np.zeros(len(points), dtype=bool)
    if "rgb" not in observation:
        return colors, valid
    camera = move(points, camera_from_reference)
    uv = project(camera, intrinsics)
    h, w = observation["mask"].shape
    inside = np.isfinite(uv).all(1) & (camera[:, 2] > 0)
    inside &= (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    indices = np.flatnonzero(inside)
    x, y = np.rint(uv[indices]).astype(int).T
    z = observation["depth"][y, x]
    usable = observation["mask"][y, x] & np.isfinite(z) & (z > 0)
    usable &= np.abs(camera[indices, 2] - z) <= tolerance
    indices, x, y = indices[usable], x[usable], y[usable]
    valid[indices] = True
    colors[indices] = observation["rgb"][y, x] / 255.0
    return colors, valid


class LibraryMatcher:
    def __init__(self, config):
        self.config = config
        if config.backend not in {"small_gicp", "open3d_plane", "open3d_color"}:
            raise ValueError("LibraryMatcher requires an optional library backend")
        if config.backend == "small_gicp":
            import small_gicp

            self.library = small_gicp
            self.version = version("small_gicp")
        else:
            import open3d

            self.library = open3d
            self.version = version("open3d")
        self.calls = []
        self._local = local()

    @staticmethod
    def appearance_cost(fitter, pose):
        """Compare measured colors only where both views agree on surface depth."""
        cached = getattr(fitter, "_registration_colors", None)
        if cached is None:
            colors = np.zeros_like(fitter.reference)
            valid = np.zeros(len(colors), bool)
            for view in fitter.appearance_views:
                values, seen = measured_colors(
                    fitter.reference,
                    view.observation,
                    view.intrinsics,
                    view.camera_from_reference,
                    fitter.config.depth_tolerance_m,
                )
                colors[seen & ~valid] = values[seen & ~valid]
                valid |= seen
            cached = fitter._registration_colors = colors, valid
        colors, valid = cached
        current, seen = measured_colors(
            fitter.reference, fitter.observation, fitter.k, pose, fitter.config.depth_tolerance_m
        )
        matched = seen & valid
        if not matched.any():
            return 0.0
        a, b = colors[matched], current[matched]
        a, b = a - a.mean(0), b - b.mean(0)
        scale = float(np.sqrt(np.mean(a * a) * np.mean(b * b)))
        return float(1 - np.mean(a * b) / scale) if scale > np.finfo(float).eps else 0.0

    def prepare(self, fitter):
        """Cache only for this immutable pair of surfaces and RGB-D views."""
        c = self.config
        cached = getattr(fitter, "_library_trial_levels", None)
        if cached is not None and cached[0] == c:
            return cached[1]
        source_points, target_points = fitter.reference, fitter.target
        if c.backend == "open3d_color":
            source_colors = np.zeros_like(source_points)
            source_valid = np.zeros(len(source_points), bool)
            views = fitter.appearance_views
            # The frozen baseline disables retained-view geometry checks. Color
            # still needs the last accepted measured RGB-D view, without enabling
            # any different geometry acceptance rules.
            if not views and getattr(self._local, "color_view", None) is not None:
                views = (self._local.color_view,)
            if not any("rgb" in view.observation for view in views):
                raise ValueError("Colored ICP requires a measured reference RGB-D view")
            for view in views:
                colors, valid = measured_colors(
                    source_points,
                    view.observation,
                    view.intrinsics,
                    view.camera_from_reference,
                    fitter.config.depth_tolerance_m,
                )
                source_colors[valid & ~source_valid] = colors[valid & ~source_valid]
                source_valid |= valid
            target_colors, target_valid = measured_colors(
                target_points, fitter.observation, fitter.k, np.eye(4), fitter.config.depth_tolerance_m
            )
            source_points, source_colors = source_points[source_valid], source_colors[source_valid]
            target_points, target_colors = target_points[target_valid], target_colors[target_valid]
        levels = []
        if len(source_points) >= 6 and len(target_points) >= 6:
            source_center = source_points.mean(0)
            target_center = target_points.mean(0)
            source_points = source_points - source_center
            target_points = target_points - target_center
            for voxel in c.voxel_sizes_m:
                if c.backend == "small_gicp":
                    source, _ = self.library.preprocess_points(
                        source_points, downsampling_resolution=voxel, num_threads=c.threads
                    )
                    target, tree = self.library.preprocess_points(
                        target_points, downsampling_resolution=voxel, num_threads=c.threads
                    )
                    if min(source.size(), target.size()) < 6:
                        continue
                    levels.append((voxel, source, target, tree, source_center, target_center))
                else:
                    o3d = self.library
                    clouds = []
                    for points, colors in (
                        (source_points, source_colors if c.backend == "open3d_color" else None),
                        (target_points, target_colors if c.backend == "open3d_color" else None),
                    ):
                        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
                        if colors is not None:
                            cloud.colors = o3d.utility.Vector3dVector(colors)
                        cloud = cloud.voxel_down_sample(voxel)
                        cloud.estimate_normals(
                            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30)
                        )
                        clouds.append(cloud)
                    if min(len(cloud.points) for cloud in clouds) < 6:
                        continue
                    levels.append((voxel, *clouds, None, source_center, target_center))
        fitter._library_trial_levels = c, levels
        return levels

    def refine(self, fitter, initial, *, admissible=None, max_evaluations=None):
        started = time.perf_counter()
        levels = self.prepare(fitter)
        prepared = time.perf_counter()
        c = self.config
        pose = np.array(initial, copy=True)
        candidates = [pose]
        inliers = 0
        attempted_levels, solved_levels = 0, 0
        failure = None
        converged = False
        budget_used = 0
        for voxel, source, target, tree, source_center, target_center in levels:
            limit = (
                c.iterations_per_level
                if max_evaluations is None
                else min(c.iterations_per_level, max_evaluations - budget_used)
            )
            if limit <= 0:
                break
            attempted_levels += 1
            if hasattr(fitter, "_projection_evidence"):
                points = np.asarray(source.points() if c.backend == "small_gicp" else source.points)[:, :3]
                predicted = move(points + source_center, pose)
                visible = fitter._projection_evidence(predicted)[0]
                contradicted = fitter._scene_depth_evidence(predicted)[2]
                selected = np.flatnonzero(np.maximum(visible, contradicted) > 0)
                if len(selected) < 6:
                    continue
                if c.backend == "small_gicp":
                    source, _ = self.library.preprocess_points(
                        points[selected], downsampling_resolution=voxel, num_threads=c.threads
                    )
                    if source.size() < 6:
                        continue
                else:
                    source = source.select_by_index(selected.tolist())
            local_pose = pose.copy()
            local_pose[:3, 3] += local_pose[:3, :3] @ source_center - target_center
            # Point-to-plane has unobservable tangent directions on a plane.
            # A point-to-point warm start supplies a bounded geometric proposal;
            # keep it even if the subsequent plane/GICP solve slides away.
            warm_limit = max(1, limit // 2)
            if c.backend == "small_gicp":
                warm = self.library.align(
                    target,
                    source,
                    tree,
                    local_pose,
                    registration_type="ICP",
                    num_threads=c.threads,
                    max_correspondence_distance=voxel * c.correspondence_voxels,
                    max_iterations=warm_limit,
                    translation_epsilon=c.translation_epsilon_m,
                    rotation_epsilon=c.rotation_epsilon_rad,
                )
                local_pose = np.asarray(warm.T_target_source).copy()
            else:
                reg = self.library.pipelines.registration
                warm = reg.registration_icp(
                    source,
                    target,
                    voxel * c.correspondence_voxels,
                    local_pose,
                    reg.TransformationEstimationPointToPoint(),
                    reg.ICPConvergenceCriteria(max_iteration=warm_limit),
                )
                local_pose = np.asarray(warm.transformation).copy()
            budget_used += warm_limit
            if not np.isfinite(local_pose).all():
                failure = "nonfinite_transform"
                break
            warmed = local_pose.copy()
            warmed[:3, 3] += target_center - warmed[:3, :3] @ source_center
            candidates.append(warmed)
            limit -= warm_limit
            if limit <= 0:
                pose = warmed
                break
            if c.backend == "small_gicp":
                result = self.library.align(
                    target,
                    source,
                    tree,
                    local_pose,
                    registration_type="GICP",
                    num_threads=c.threads,
                    max_correspondence_distance=voxel * c.correspondence_voxels,
                    max_iterations=limit,
                    translation_epsilon=c.translation_epsilon_m,
                    rotation_epsilon=c.rotation_epsilon_rad,
                )
                pose = np.asarray(result.T_target_source).copy()
                inliers = int(result.num_inliers)
                converged = bool(result.converged)
            else:
                reg = self.library.pipelines.registration
                if c.backend == "open3d_color":
                    solve = reg.registration_colored_icp
                    estimator = reg.TransformationEstimationForColoredICP(c.colored_geometry_weight)
                else:
                    solve = reg.registration_icp
                    estimator = reg.TransformationEstimationPointToPlane()
                try:
                    result = solve(
                        source,
                        target,
                        voxel * c.correspondence_voxels,
                        local_pose,
                        estimator,
                        reg.ICPConvergenceCriteria(max_iteration=limit),
                    )
                except RuntimeError as error:
                    if "No correspondences found" not in str(error):
                        raise
                    failure = "no_correspondences"
                    break
                pose = np.asarray(result.transformation).copy()
                inliers = len(result.correspondence_set)
                # Open3D does not expose an iteration count or convergence flag.
            pose[:3, 3] += target_center - pose[:3, :3] @ source_center
            # Open3D exposes no actual count: charge its requested upper bound.
            budget_used += min(limit, int(result.iterations)) if c.backend == "small_gicp" else limit
            if np.isfinite(pose).all():
                candidates.append(pose)
                solved_levels += int(inliers > 0)
            else:
                failure = "nonfinite_transform"
                pose = candidates[-1]
                break
        self._local.proposals = candidates
        solved = time.perf_counter()
        # Evaluate native iterates together. The shared fitter caches these exact
        # residuals, so diagnostics and selection do not dispatch a spatial query
        # for each pose. Standalone point-cloud callers may supply only score().
        if hasattr(fitter, "batch_residuals"):
            fitter.batch_residuals(candidates)
        valid = [p for p in candidates if admissible(p)] if admissible is not None else []
        if valid:
            pose = min(valid, key=fitter.score)
        detail = {
            "initial_cost": fitter.score(initial),
            "cost": fitter.score(pose),
            # Includes the warm start; Open3D exposes only the requested budget.
            "evaluations": budget_used,
            "converged": converged,
            "fit_reference_points": len(fitter.reference_queries),
            "fit_target_points": len(fitter.target_queries),
            "library": c.backend,
            "library_inliers": inliers,
            "library_levels": len(levels),
            "library_attempted_levels": attempted_levels,
            "library_solved_levels": solved_levels,
            "library_failure": failure,
            "retained_admissible_iterate": bool(valid),
        }
        self.calls.append(
            {
                "prepare_ms": (prepared - started) * 1000,
                "solve_ms": (solved - prepared) * 1000,
                "validation_score_ms": (time.perf_counter() - solved) * 1000,
                "inliers": inliers,
                "levels": len(levels),
                "attempted_levels": attempted_levels,
                "solved_levels": solved_levels,
            }
        )
        return pose, detail
