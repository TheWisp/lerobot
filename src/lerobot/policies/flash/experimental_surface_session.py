"""Shared live/video pose estimation and confirmed surface growth.

Frame flow: gui/static/flash_architecture.html. Predict, fit the confirmed map
once, check geometry, fuse with Kalman and recheck, then remember measurements.
Every rejected observation takes the same prediction-publication path. Pose
uncertainty and acceptance thresholds are not calibrated correctness probabilities.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, replace

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .geometry import transform
from .surface_grounding import (
    PreparedReferenceSurface,
    SurfaceFitView,
    WholeSurfaceFit,
    project,
)
from .surface_kalman import MeasuredGroupMotion, PoseKalmanFilter
from .surface_motion import measured_surface_information
from .surface_observation import (
    move,
    reliable_observation_depth,
    spatial_sample,
    voxel_sample,
)
from .surface_uncertainty import PoseUncertaintyConfig, SurfacePoseUncertainty, check_surface_contradiction


@dataclass(frozen=True)
class GrowingSurfaceConfig:
    # Match the baseline's 5 mm depth budget and 3 mm cloud sampling by default.
    spacing_m: float = 0.003
    agreement_m: float = 0.005
    # Require an overlapping portion, not a single accidental nearest neighbour.
    # These fractions are hypotheses to benchmark, not object-specific rules.
    min_overlap: float = 0.25
    # Growth needs a tighter match to more of the already stored surface.
    # 3 mm matches the storage spacing; 75% requires a substantial overlap.
    # These are conservative experiment defaults, not calibrated certainty.
    growth_agreement_m: float = 0.003
    growth_min_overlap: float = 0.75
    # The local-growth trial changes later cube fits, so leave it opt-in until
    # patch confirmation and map influence are validated on full replays.
    local_growth_enabled: bool = False
    # At the edge of an existing patch, only about half a local neighborhood
    # can be previously mapped. This threshold applies to that neighborhood.
    growth_local_overlap: float = 0.5
    # Fraction of a proposed addition not contradicted by earlier depth/masks.
    patch_consistency: float = 0.95
    # Fraction of a pending patch measured again within growth_agreement_m.
    patch_overlap: float = 0.8
    image_weight: float = 1.0
    group_point_evidence_weight: float = 1.0
    group_association_min_points: int = 12
    # Geometry is the primary objective; image/group evidence breaks ties.
    # False retains the earlier weighted-sum experiment for paired comparison.
    surface_priority: bool = True
    # Experiment: use retained measured views to distinguish newly visible
    # surface from contradictions. False preserves the published baseline.
    partial_surface_matching: bool = False
    # Accept old recording settings without writing the retired search option
    # back into new settings. Registration always performs one solve.
    recovery_starts: InitVar[int] = 0
    # Experimental cost/consistency ablation: use the newest measured views
    # for visibility checks, while retaining the whole accumulated point map.
    # None preserves all retained views. This can forget old empty-space evidence.
    fit_view_limit: int | None = None
    # One later view checks an addition before it can influence fitting.
    confirmations: int = 2
    max_patches: int = 32
    max_pending: int = 8
    points_per_patch: int = 1024
    max_points: int = 8192
    grow: bool = True
    retain_unobserved_pose: bool = False
    uncertainty: PoseUncertaintyConfig = field(
        default_factory=lambda: PoseUncertaintyConfig(
            enabled=True, mode="kalman", max_visible_outlier_fraction=0.25
        )
    )

    def __post_init__(self, recovery_starts):
        if isinstance(self.uncertainty, dict):
            object.__setattr__(self, "uncertainty", PoseUncertaintyConfig(**self.uncertainty))
        if not isinstance(self.uncertainty, PoseUncertaintyConfig):
            raise ValueError("uncertainty must be a PoseUncertaintyConfig or its JSON object")
        if self.uncertainty.enabled and not self.surface_priority:
            raise ValueError("uncertainty requires surface_priority")
        if self.fit_view_limit is not None and (
            type(self.fit_view_limit) is not int or self.fit_view_limit < 1
        ):
            raise ValueError("fit_view_limit must be a positive integer or None")
        for name in ("spacing_m", "agreement_m", "growth_agreement_m"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "min_overlap",
            "growth_min_overlap",
            "growth_local_overlap",
            "patch_consistency",
            "patch_overlap",
        ):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.growth_agreement_m > self.agreement_m or self.growth_min_overlap < self.min_overlap:
            raise ValueError(
                "growth must require at least as much overlap within at most the tracking distance"
            )
        for name in ("image_weight", "group_point_evidence_weight"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(self.group_association_min_points) is not int or self.group_association_min_points < 3:
            raise ValueError("group_association_min_points must be an integer >= 3")
        for name in ("confirmations", "max_patches", "max_pending", "points_per_patch", "max_points"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        if min(self.max_points, self.points_per_patch) < 3:
            raise ValueError("a surface needs at least three points")
        if any(
            type(getattr(self, key)) is not bool
            for key in (
                "grow",
                "retain_unobserved_pose",
                "surface_priority",
                "partial_surface_matching",
                "local_growth_enabled",
            )
        ):
            raise ValueError("surface experiment switches must be booleans")


def checked_pose(value):
    pose = np.asarray(value, dtype=float)
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0, 0, 0, 1])
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(pose[:3, :3]), 1)
    ):
        raise ValueError("pose must be a rigid 4x4 transform")
    return pose.copy()


@dataclass(frozen=True)
class PointEvidence:
    """One source's corresponding object coordinates and current camera points.

    Predicted points constrain motion but never enter the measured surface map.
    Weight is per source, not per point: densifying a prediction adds no votes.
    scale_m is an experimental residual scale, not a calibrated covariance.
    """

    reference: np.ndarray
    current: np.ndarray
    source: str
    kind: str
    scale_m: float
    weight: float = 1.0
    reuses_image_matches: bool = False

    def __post_init__(self):
        a, b = np.asarray(self.reference, dtype=float), np.asarray(self.current, dtype=float)
        if a.ndim != 2 or a.shape[1:] != (3,) or b.shape != a.shape or len(a) < 3:
            raise ValueError("point evidence needs matching Nx3 arrays with at least three points")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("point evidence must be finite")
        if not self.source or self.kind not in {"measured", "predicted"}:
            raise ValueError("point evidence needs its source and measured/predicted kind")
        if not np.isfinite([self.scale_m, self.weight]).all() or self.scale_m <= 0 or self.weight < 0:
            raise ValueError("invalid point evidence scale or weight")
        object.__setattr__(self, "reference", a.copy())
        object.__setattr__(self, "current", b.copy())


@dataclass(frozen=True)
class PoseHint:
    """Caller-validated co-movement prediction in this session's fixed coordinates.

    time_s is the time for which the pose was predicted (must equal frame time).
    observed_time_s identifies the actual measurement backing that prediction.
    This interface does not infer or verify a grasp or a group relationship.
    """

    camera_from_reference: np.ndarray
    source: str
    time_s: float
    observed_time_s: float
    reference_frame: int
    points: tuple[PointEvidence, ...] = ()
    motion: MeasuredGroupMotion | None = None

    def __post_init__(self):
        object.__setattr__(self, "camera_from_reference", checked_pose(self.camera_from_reference))
        if (
            not self.source
            or not np.isfinite([self.time_s, self.observed_time_s]).all()
            or self.observed_time_s > self.time_s
        ):
            raise ValueError("hint needs a source and causal timestamps")


@dataclass
class SurfaceView:
    """Depth rays from a retained measurement, used only to veto new geometry.

    Take the nearest depth within the existing projection tolerance. A point
    behind that depth may have been hidden; it is not a contradiction. Missing
    depth is also unknown. This is not an object-completion model.
    """

    nearest_depth: np.ndarray
    outside_mask: np.ndarray
    intrinsics: np.ndarray
    tolerance_m: float
    depth: np.ndarray | None = None
    mask: np.ndarray | None = None
    measured_points: np.ndarray | None = None
    scene_depth: np.ndarray | None = None

    @classmethod
    def from_observation(cls, observation, intrinsics, shape_config, *, retain_measurement=False):
        radius = int(np.ceil(shape_config.mask_boundary_tolerance_px))
        kernel = np.ones((2 * radius + 1,) * 2, np.uint8)
        depth = reliable_observation_depth(observation)
        scene_depth = (
            reliable_observation_depth(observation, observation["scene_depth"])
            if "scene_depth" in observation
            else depth
        )
        nearest = cv2.erode(scene_depth, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        outside = cv2.distanceTransform((~observation["mask"]).astype(np.uint8), cv2.DIST_L2, 5)
        return cls(
            nearest,
            outside > shape_config.mask_boundary_tolerance_px,
            intrinsics.copy(),
            shape_config.depth_tolerance_m,
            depth.copy() if retain_measurement else None,
            observation["mask"].copy() if retain_measurement else None,
            observation.get("all_surface_points", observation["obs"]["points"]).copy()
            if retain_measurement
            else None,
            scene_depth.copy() if retain_measurement else None,
        )

    def contradicts(self, camera_points):
        uv = project(camera_points, self.intrinsics)
        h, w = self.nearest_depth.shape
        valid = (
            np.isfinite(uv).all(1)
            & (camera_points[:, 2] > 0)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < w - 1)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < h - 1)
        )
        result = np.zeros(len(camera_points), dtype=bool)
        indices = np.flatnonzero(valid)
        xy = np.rint(uv[indices]).astype(int)
        depth = self.nearest_depth[xy[:, 1], xy[:, 0]]
        z = camera_points[indices, 2]
        result[indices] = (depth > 0) & (
            (z < depth - self.tolerance_m)
            | (self.outside_mask[xy[:, 1], xy[:, 0]] & (z <= depth + self.tolerance_m))
        )
        return result


@dataclass
class SurfacePatch:
    frame: int
    measured: np.ndarray
    camera_from_reference: np.ndarray
    fitted_map_version: int
    support_frames: set[int]
    active: bool = False
    view: SurfaceView | None = None

    def points(self):
        return move(self.measured, np.linalg.inv(self.camera_from_reference))


class SurfaceStore:
    """Bounded raw patches; exclude/reposition rebuilds rather than averages away history.

    A caller may persist evicted patches externally. RAM-only undo is limited to
    retained patches; excluded IDs never silently affect the combined surface.
    """

    def __init__(self, config: GrowingSurfaceConfig):
        self.config = config
        self.patches: dict[int, SurfacePatch] = {}
        self.version = 0
        self.points = np.empty((0, 3))
        self.events = []  # Drained by the session each frame; not a growing log.

    def _rebuild(self):
        active = [p.points() for p in self.patches.values() if p.active]
        points = np.concatenate(active) if active else np.empty((0, 3))
        if len(points):
            points = points[voxel_sample(points, self.config.spacing_m)]
            points = points[spatial_sample(points, self.config.max_points)]
        self.points = points
        self.version += 1

    def add(self, points, pose, frame, *, active=False, fitted_map_version=None, view=None):
        if frame in self.patches:
            raise ValueError("one patch per source frame")
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError("patch needs finite Nx3 camera measurements")
        # Use the same object-coordinate grid for the first view and additions.
        # Sampling a dense initial view by count first made its resolution depend
        # on the area of that view, while small later patches kept finer detail.
        local = move(points, np.linalg.inv(pose))
        selected = voxel_sample(local, self.config.spacing_m)
        selected = selected[spatial_sample(local[selected], self.config.points_per_patch)]
        points = points[selected].copy()
        version = self.version if fitted_map_version is None else fitted_map_version
        self.patches[frame] = SurfacePatch(frame, points, checked_pose(pose), version, {frame}, active, view)
        self.events.append({"event": "added" if active else "pending", "frame": frame})
        self._bound()
        if active:
            self._rebuild()

    def _bound(self):
        pending = [i for i, p in self.patches.items() if not p.active]
        for i in pending[: max(0, len(pending) - self.config.max_pending)]:
            del self.patches[i]
            self.events.append({"event": "evicted", "frame": i, "reason": "pending_budget"})
        # An age-only window discards the first broad view in favour of tiny
        # recent additions and progressively forgets the shape being tracked.
        # Remove the patch losing the fewest uniquely covered surface voxels.
        # This is a memory policy, not a claim that unique geometry is correct.
        while sum(p.active for p in self.patches.values()) > self.config.max_patches:
            cells = {
                i: {tuple(v) for v in np.floor(p.points() / self.config.spacing_m).astype(np.int64)}
                for i, p in self.patches.items()
                if p.active
            }
            counts = {}
            for patch_cells in cells.values():
                for cell in patch_cells:
                    counts[cell] = counts.get(cell, 0) + 1
            victim = min(cells, key=lambda i: (sum(counts[c] == 1 for c in cells[i]), len(cells[i]), i))
            del self.patches[victim]
            self.events.append({"event": "evicted", "frame": victim, "reason": "surface_coverage_budget"})

    def exclude(self, frame):
        del self.patches[frame]  # Retain original measurements in the run's input archive.
        self.events.append({"event": "excluded", "frame": frame})
        self._rebuild()

    def reposition(self, frame, pose):
        self.patches[frame].camera_from_reference = checked_pose(pose)
        self.events.append(
            {
                "event": "repositioned",
                "frame": frame,
                "camera_from_reference": self.patches[frame].camera_from_reference.tolist(),
            }
        )
        self._rebuild()

    def compatible_additions(self, measured, pose):
        """Check placement against earlier retained views, not just the next fit.

        Filtering a point means its proposed placement conflicts with a retained
        observation; it does not prove which pose/depth/mask was wrong. Keep the
        raw input recording for reprocessing. This cannot resolve unseen shape.
        """
        local = move(measured, np.linalg.inv(pose))
        keep = np.ones(len(local), dtype=bool)
        for patch in self.patches.values():
            if patch.active and patch.view is not None:
                keep &= ~patch.view.contradicts(move(local, patch.camera_from_reference))
        return keep

    def confirm(self, measured, pose, frame):
        """Later aligned measurements support a pending patch without fitting to it.

        Correlated pose errors can still pass. This tests repeatability, not truth.
        Missing/occluded patches remain pending and expire through the count budget.
        """
        tree = cKDTree(measured)
        changed = False
        for patch in self.patches.values():
            if patch.active or frame in patch.support_frames:
                continue
            distances = tree.query(move(patch.points(), pose))[0]
            agreement = np.mean(distances <= self.config.growth_agreement_m) if len(distances) else 0
            if agreement >= self.config.patch_overlap:
                patch.support_frames.add(frame)
                if len(patch.support_frames) >= self.config.confirmations:
                    patch.active = True
                    changed = True
                    self.events.append({"event": "confirmed", "frame": patch.frame, "by_frame": frame})
        if changed:
            self._bound()
            self._rebuild()

    def extend(self, measured, pose, frame, *, view):
        """Grow from well-supported parts of an accepted measured pose.

        A broadly agreeing frame may propose every non-contradicted new cell.
        Otherwise only cells next to a locally agreeing part of the current
        measurement may enter a pending patch. Both paths use the active map
        before confirmation, so a pending patch cannot support itself.
        """
        before = self.version
        distances = cKDTree(move(self.points, pose)).query(measured)[0]
        agreeing = distances <= self.config.growth_agreement_m
        overlap = float(np.mean(agreeing)) if len(distances) else 0.0
        broad_support = overlap >= self.config.growth_min_overlap
        # A nearby agreeing observation anchors a local extension. The radius
        # is four storage cells: far-away geometry cannot join a weak fit in
        # one step, but can grow after a neighboring patch is confirmed.
        radius = 4 * self.config.spacing_m

        def locally_supported(points):
            neighborhoods = cKDTree(measured).query_ball_point(points, radius)
            return np.fromiter(
                (
                    len(ids) >= 3
                    and np.count_nonzero(agreeing[ids]) >= 3
                    and np.mean(agreeing[ids]) >= self.config.growth_local_overlap
                    for ids in neighborhoods
                ),
                dtype=bool,
                count=len(points),
            )

        def new_cells():
            local = move(measured, np.linalg.inv(pose))
            selected = voxel_sample(np.concatenate((self.points, local)), self.config.spacing_m)
            return measured[selected[selected >= len(self.points)] - len(self.points)]

        local_support = (
            locally_supported(new_cells()) if not broad_support and self.config.local_growth_enabled else None
        )
        accepted = broad_support or (local_support is not None and bool(np.any(local_support)))
        details = {
            "alignment_accepted": accepted,
            "observed_overlap": overlap,
            "required_overlap": self.config.growth_min_overlap,
            "required_local_overlap": self.config.growth_local_overlap,
            "agreement_m": self.config.growth_agreement_m,
            "reason": (
                "sufficient_growth_overlap"
                if broad_support
                else "local_growth_support"
                if accepted
                else "insufficient_growth_overlap"
            ),
            "growth_mode": "broad"
            if broad_support
            else "local"
            if self.config.local_growth_enabled
            else "off",
            "local_supported_points": int(np.count_nonzero(local_support))
            if local_support is not None
            else 0,
            "new_measured_points": 0,
            "contradicted_by_retained_views": 0,
            "patch_consistent": None,
            "admitted_points": 0,
        }
        if not accepted:
            return details
        self.confirm(measured, pose, frame)
        # A new occupied grid cell extends the surface. A distance exclusion
        # around old samples leaves holes in adjacent cells and gives growth a
        # different density from initialization. Keep actual measured points.
        novel = new_cells()
        keep = self.compatible_additions(novel, pose)
        if broad_support:
            consistent = not len(novel) or float(np.mean(keep)) >= self.config.patch_consistency
            admitted = keep if consistent else np.zeros(len(novel), dtype=bool)
        else:
            # A contradiction elsewhere must not veto this local patch. Its
            # points must extend beyond the map in both 3D and the current
            # image. A neighboring voxel on an already projected face is
            # resampling/noise, not evidence of newly visible geometry.
            distance_to_map = cKDTree(move(self.points, pose)).query(novel)[0]
            admitted = keep & locally_supported(novel) & (distance_to_map >= self.config.growth_agreement_m)
            if view is not None and len(novel):
                old_camera = move(self.points, pose)
                visible_old = (old_camera[:, 2] > 0) & np.isfinite(old_camera).all(axis=1)
                visible_new = (novel[:, 2] > 0) & np.isfinite(novel).all(axis=1)
                if np.any(visible_old):
                    pixel_distance = np.full(len(novel), np.inf)
                    old_pixels = project(old_camera[visible_old], view.intrinsics)
                    new_pixels = project(novel[visible_new], view.intrinsics)
                    pixel_distance[visible_new] = cKDTree(old_pixels).query(new_pixels)[0]
                    one_voxel_px = (
                        self.config.spacing_m
                        * max(view.intrinsics[0, 0], view.intrinsics[1, 1])
                        / np.maximum(novel[:, 2], np.finfo(float).tiny)
                    )
                    admitted &= visible_new & (pixel_distance > np.maximum(2.0, one_voxel_px))
                else:
                    admitted[:] = False
            consistent = bool(np.any(admitted))
        details.update(
            new_measured_points=len(novel),
            contradicted_by_retained_views=int((~keep).sum()),
            patch_consistent=consistent,
            admitted_points=int(admitted.sum()),
        )
        novel = novel[admitted]
        if len(novel):
            self.add(novel, pose, frame, fitted_map_version=before, view=view)
        return details


class ImageConditionedFit(WholeSurfaceFit):
    """Keep image correspondences in the objective, not only in its initial guess."""

    def __init__(
        self,
        *args,
        image_reference,
        image_current,
        image_scale_m,
        image_weight,
        point_evidence=(),
        geometry=None,
        **kwargs,
    ):
        if geometry is not None:
            kwargs.update(
                prepared_reference=geometry.prepared_reference,
                prepared_target=geometry.prepared_target,
                prepared_reverse_fits=geometry.reverse_fits,
            )
        super().__init__(*args, **kwargs)
        self.image_reference, self.image_current = image_reference, image_current
        self.image_scale_m, self.image_weight = image_scale_m, image_weight
        self.point_evidence = point_evidence
        self.geometry = (
            geometry
            if geometry is not None
            else WholeSurfaceFit(
                self.reference,
                self.observation,
                self.k,
                self.config,
                prepared_reference=self.prepared_reference,
                prepared_target=self.prepared_target,
                fit_kernels=self.fit_kernels,
                search_pool=self.search_pool,
                reference_views=self.reference_views,
                appearance_views=self.appearance_views,
                prepared_reverse_fits=self.reverse_fits,
            )
        )

    @property
    def has_image_evidence(self):
        return bool(len(self.image_reference) and self.image_weight) or any(
            len(p.reference) and p.weight for p in self.point_evidence
        )

    def _image_residual(self, pose):
        blocks = []
        sources = [(self.image_reference, self.image_current, self.image_scale_m, self.image_weight)]
        sources.extend((p.reference, p.current, p.scale_m, p.weight) for p in self.point_evidence)
        for reference, current, scale, weight in sources:
            if not len(reference) or weight == 0:
                continue
            difference = (move(reference, pose) - current) / scale
            squared = np.where(abs(difference) <= 1, difference**2, 2 * abs(difference) - 1)
            residual = np.sign(difference) * np.sqrt(squared)
            blocks.append(residual.ravel() * np.sqrt(weight / len(residual)))
        return np.concatenate(blocks) if blocks else np.empty(0)

    def residual(self, pose, *, include_coverage=True):
        return np.concatenate(
            (self.geometry.residual(pose, include_coverage=include_coverage), self._image_residual(pose))
        )

    def batch_residuals(self, poses):
        return [
            np.concatenate((surface, self._image_residual(pose)))
            for surface, pose in zip(self.geometry.batch_residuals(poses), poses, strict=True)
        ]

    def correct_surface_first(self, proposal, previous, *, uncertainty=None, check=None):
        """Validate one fit, then optionally measure uncertainty."""
        uncertainty = uncertainty or PoseUncertaintyConfig()
        geometry = self.geometry
        if check is None:

            def check(pose):
                return geometry.check_pose(
                    pose, max_outlier_fraction=uncertainty.max_visible_outlier_fraction
                )

        candidates, detail = self.fit_once(
            proposal,
            geometry=geometry,
            check=check,
        )
        valid = [c for c in candidates if c["check"]["accepted"]]
        detail.update(surface_priority=True, geometry_costs=[c["cost"] for c in candidates])
        if not valid:
            # Keep the measured reason the one local solve failed. The caller
            # must not have to infer it from a generic missing-pose status.
            rejected = min(candidates, key=lambda c: c["cost"]) if candidates else None
            return None, {
                **detail,
                "candidate_check": rejected["check"] if rejected else None,
                "rejected_camera_from_reference": rejected["pose"].tolist() if rejected else None,
                "reason": "no_observation_compatible_candidate",
            }
        best = min(valid, key=lambda c: c["cost"])
        if uncertainty.enabled:
            estimator = SurfacePoseUncertainty(geometry, best["pose"], uncertainty)
            if uncertainty.mode == "selection":
                compatible = [(c, estimator.check(c["pose"], direct=c["check"])) for c in candidates]
                eligible = [c for c, verdict in compatible if verdict["accepted"]]
                supported = {p.source for p in self.point_evidence if p.kind == "predicted" and p.weight > 0}
                preferred = (
                    [c for c in eligible if c["name"] in supported]
                    if uncertainty.prefer_supported_prediction
                    else []
                )
                best = (
                    min(
                        preferred or eligible,
                        key=lambda c: float(
                            np.mean(
                                np.sum(
                                    (move(self.reference, c["pose"]) - move(self.reference, previous)) ** 2,
                                    axis=1,
                                )
                            )
                        ),
                    )
                    if eligible
                    else None
                )
                detail["uncertainty"] = {
                    **estimator.diagnostics(),
                    "selected": best["name"] if best else None,
                    "candidates": [{"name": c["name"], **verdict} for c, verdict in compatible],
                }
            else:
                detail["uncertainty"] = {
                    **estimator.diagnostics(),
                    "mode": "kalman",
                    "selected": best["name"],
                }
        else:
            tolerance = self.config.solver_tolerance * max(1.0, abs(best["cost"]))
            best = min(
                (c for c in valid if c["cost"] <= best["cost"] + tolerance),
                key=lambda c: self.score(c["pose"]),
            )
        if best is None:
            return None, detail
        return best["pose"], {
            **best["detail"],
            **detail,
            "geometry_check": best["check"],
            "geometry_cost": best["cost"],
            "selected_candidate": next(i for i, candidate in enumerate(candidates) if candidate is best),
            "selected_candidate_name": best["name"],
        }


class GrowingSurfaceSession:
    """One sequential object track shared by live and recorded frame adapters."""

    def __init__(self, shape_config, config=None, *, fit_kernels=None, search_pool=None):
        self.config = config or GrowingSurfaceConfig()
        self.shape_config = shape_config
        self.growth_shape_config = replace(
            shape_config,
            depth_tolerance_m=min(shape_config.depth_tolerance_m, self.config.growth_agreement_m),
        )
        self.fit_kernels, self.search_pool = fit_kernels, search_pool
        self.store = SurfaceStore(self.config)
        self.pose = None
        self.reference = None
        self.reference_frame = None
        self.accepted_frame = None
        self.accepted_pose = None
        self.accepted_observation = None
        self.accepted_intrinsics = None
        self.initial_appearance_view = None
        self.last_time = self.last_frame = None
        self.base_from_camera = np.eye(4)
        self.last_observed_time = None
        self.last_supported_time = None
        self.kalman = None
        self._prepared_reference = None

    def prepare_geometry(self, prepared):
        """One measured geometry context for this frame's group checks and fit."""
        if prepared["status"] is not None or self.pose is None or len(self.store.points) < 3:
            return None
        if self._prepared_reference is None or self._prepared_reference.points is not self.store.points:
            self._prepared_reference = PreparedReferenceSurface.build(self.store.points, self.shape_config)
        return WholeSurfaceFit(
            self.store.points,
            prepared["current"],
            prepared["intrinsics"],
            self.shape_config,
            prepared_reference=self._prepared_reference,
            fit_kernels=self.fit_kernels,
            search_pool=self.search_pool,
            reference_views=self._fit_views(),
            appearance_views=self._appearance_views(),
        )

    def update(
        self, prepared, time_s, *, hint=None, candidate_check=None, base_from_camera=None, geometry=None
    ):
        """Execute the maintained frame diagram without another registration search.

        Only an accepted observation changes the image reference or surface.
        candidate_check is an optional independent veto used by comparison runs;
        it follows the same rejection path as geometry and Kalman.
        """
        frame = prepared["frame_id"]
        if (
            not np.isfinite(time_s)
            or self.last_frame is not None
            and (frame <= self.last_frame or time_s <= self.last_time)
        ):
            raise ValueError("source frames and times must strictly increase")
        if hint is not None and (hint.time_s != time_s or hint.reference_frame != self.reference_frame):
            raise ValueError("hint belongs to another time or coordinate reference")
        camera = transform(np.eye(4) if base_from_camera is None else base_from_camera)
        # Recorded extrinsics may be rounded. Reject actual scale/shear above,
        # then remove roundoff before composing a persistent world-space state.
        camera[:3, :3] = Rotation.from_matrix(camera[:3, :3]).as_matrix()
        camera_change = np.linalg.inv(camera) @ self.base_from_camera
        # 1. Predict every frame, including empty observations. This updates
        # pose uncertainty, never the measured map or accepted image reference.
        if self.kalman is not None:
            self.kalman.predict(time_s, camera_change, camera, hint.motion if hint else None)
            self.pose = self.kalman.pose.copy()
        elif self.pose is not None:
            self.pose = camera_change @ self.pose
        self.base_from_camera = camera
        self.last_frame, self.last_time = frame, time_s
        before = self.store.version
        details = {
            "map_version_used": before,
            "surface_frames_used": sorted(p.frame for p in self.store.patches.values() if p.active),
            "accepted_frame_before": self.accepted_frame,
            "image_source_frame": prepared.get("match_source_frame"),
            "observation": {
                key: prepared["diagnostics"][key]
                for key in ("clean_count", "visible_lateral_spread_mm", "principal_surface_spreads_m")
                if key in prepared.get("diagnostics", {})
            },
            "reference_frame": self.reference_frame,
            "covariance": None,
        }
        # 2. Initialization needs a current usable measurement. Missing data
        # never reseeds the fixed object coordinates or pretends to measure shape.
        valid = prepared["status"] is None
        current = prepared["current"]
        points = current["obs"]["points"]
        if self.pose is None and valid:
            self.reference = points.copy()
            self.pose = np.eye(4)
            self.reference_frame = frame
            self.initial_appearance_view = SurfaceFitView(
                current, prepared["intrinsics"].copy(), self.pose.copy()
            )
            if self.config.uncertainty.enabled and self.config.uncertainty.mode == "kalman":
                self.kalman = PoseKalmanFilter(
                    self.pose, self.reference.mean(0), time_s, self.config.uncertainty
                )
            self.store.add(
                points,
                self.pose,
                frame,
                active=True,
                view=SurfaceView.from_observation(
                    current,
                    prepared["intrinsics"],
                    self.growth_shape_config,
                    retain_measurement=self.config.partial_surface_matching,
                ),
            )
            self.last_observed_time = self.last_supported_time = time_s
            self._remember_observation(prepared)
            details["reference_frame"] = frame
            return self._result("initialized", time_s, details)
        if self.pose is None or not valid:
            return self._publish_prediction(hint, time_s, details, prepared["status"] or "not_initialized")
        if len(self.store.points) < 3:
            return self._publish_prediction(hint, time_s, details, "surface_removed")
        geometry = geometry if geometry is not None else self.prepare_geometry(prepared)
        source = prepared.get("match_source_frame")
        step = prepared["step"] if source == self.accepted_frame else None
        # Measured motion of this object's points from the last accepted view
        # starts the one local fit. A scene group can carry the
        # object when those points are unavailable, and remains secondary fit
        # evidence even when the measured object motion supplies the seed.
        seed = (
            step @ self.accepted_pose
            if step is not None
            else (hint.camera_from_reference if hint is not None else self.pose)
        )
        details["hint_source"] = hint.source if hint else None
        details["fit_seed_source"] = (
            "object_matches" if step is not None else "group" if hint is not None else "previous_pose"
        )
        a, b = np.empty((0, 3)), np.empty((0, 3))
        if step is not None and "point_pairs" in prepared:
            a, b = prepared["point_pairs"]
            keep = np.linalg.norm(move(a, step) - b, axis=1) <= self.config.agreement_m
            a = move(a[keep], np.linalg.inv(self.accepted_pose))
            b = b[keep]
        fitter = ImageConditionedFit(
            self.store.points,
            current,
            prepared["intrinsics"],
            self.shape_config,
            geometry=geometry,
            image_reference=a,
            image_current=b,
            image_scale_m=self.config.agreement_m,
            # The shared object group reuses these same image identities. Its
            # measured pairs already carry their weight in point_evidence.
            image_weight=0.0
            if prepared.get("shared_matches")
            and hint
            and any(p.kind == "measured" and p.reuses_image_matches for p in hint.points)
            else self.config.image_weight,
            point_evidence=hint.points if hint else (),
            fit_kernels=self.fit_kernels,
            search_pool=self.search_pool,
            reference_views=geometry.reference_views,
            appearance_views=geometry.appearance_views,
        )
        details["image_pairs"] = len(a)
        details["image_weight"] = fitter.image_weight
        details["point_sources"] = (
            [
                {"source": p.source, "kind": p.kind, "count": len(p.reference), "weight": p.weight}
                for p in hint.points
            ]
            if hint
            else []
        )
        geometry_checks = {}

        def check_pose(pose):
            key = pose.tobytes()
            if key not in geometry_checks:
                geometry_checks[key] = fitter.check_pose(
                    pose,
                    max_outlier_fraction=self.config.uncertainty.max_visible_outlier_fraction,
                    agreement_m=self.config.agreement_m,
                    min_overlap=self.config.min_overlap,
                )
            return geometry_checks[key]

        # 3. One bounded solve against all active (confirmed) surface patches.
        # Image/group motion supplies the seed and secondary fitting evidence.
        candidate, fit = (
            fitter.correct_surface_first(
                seed,
                self.pose,
                uncertainty=self.config.uncertainty,
                check=check_pose,
            )
            if self.config.surface_priority
            else fitter.correct(seed, check=check_pose)
        )
        if "uncertainty" in fit:
            details["covariance"] = fit["uncertainty"]["covariance"]
        # Also covers weighted fitting and replacement matchers without Kalman.
        # No candidate may update the pose or map while occupying measured air.
        if candidate is not None and not check_pose(candidate)["accepted"]:
            fit["rejected_camera_from_reference"] = candidate.tolist()
            candidate = None
            fit["reason"] = "measured_empty_space_conflict"
        if candidate is None:
            details["fit"] = fit
            prior_check = check_pose(self.pose)
            if prior_check["geometry_accepted"] and not prior_check["accepted"]:
                # Too little overlap for a new measurement does not contradict
                # an existing prediction. It cannot update the filter or map.
                return self._publish_prediction(
                    hint, time_s, details, "insufficient_surface_overlap", geometry=fitter
                )
            # A failed local solve is not evidence that the independent motion
            # prediction failed. Check that prediction against this observation
            # before withholding the pose. It remains a prediction and cannot
            # update the stored surface or connected image.
            return self._publish_prediction(
                hint,
                time_s,
                details,
                "no_observation_compatible_candidate",
                geometry=fitter,
                require_overlap=True,
            )
        # 4. Fuse only a geometry-compatible fit. A fused pose can differ from
        # both inputs, so it must pass the same geometry check before publication.
        kalman_prior = None
        if self.kalman is not None:
            prior_check = check_pose(self.pose)
            kalman_prior = self.kalman.snapshot()
            measured = fit["uncertainty"]
            correction = self.kalman.correct(
                candidate,
                np.asarray(measured["information"]),
                np.asarray(measured["pivot_m"]),
                # A lack of overlap is not evidence that the motion prior is
                # wrong. Reset only when current measured geometry contradicts
                # it; the candidate has already passed both checks above.
                contradicted=not prior_check["geometry_accepted"],
            )
            details["kalman"] = correction
            candidate = self.kalman.pose.copy()
            if correction["accepted"] and not check_pose(candidate)["accepted"]:
                self._rollback_correction(kalman_prior, details, "fused_pose_contradicts_observation")
            self.pose = self.kalman.pose.copy()
            if not details["kalman"]["accepted"]:
                details["fit"] = fit
                return self._publish_prediction(
                    hint,
                    time_s,
                    details,
                    details["kalman"]["reason"],
                    prediction_check=prior_check,
                )
        accepted_geometry = check_pose(candidate)
        details.update(
            fit=fit,
            observed_overlap=accepted_geometry["observed_overlap"],
            observed_overlap_basis=accepted_geometry["observed_overlap_basis"],
            surface_median_mm=accepted_geometry["surface_median_mm"],
            candidate_camera_from_reference=candidate.tolist(),
        )
        if hint is not None:
            details["hint_disagreement_mm"] = float(
                1000
                * np.median(
                    np.linalg.norm(
                        move(self.reference, candidate) - move(self.reference, hint.camera_from_reference),
                        axis=1,
                    )
                )
            )
            if candidate_check is not None:
                accepted, comparison = candidate_check(self.store.points, candidate)
                details["group_comparison"] = comparison
                if not accepted:
                    if kalman_prior is not None:
                        self._rollback_correction(kalman_prior, details, "surface_group_disagreement")
                    return self._publish_prediction(
                        hint, time_s, details, "surface_group_disagreement", geometry=fitter
                    )
        # 5. Commit only the accepted measured result. Predicted/lost paths
        # return above and cannot advance this reference or confirm map patches.
        self.pose = candidate
        self.last_observed_time = self.last_supported_time = time_s
        self._remember_observation(prepared)
        info = measured_surface_information(
            self.store.points,
            self.pose,
            points,
            current["obs"]["normals"],
            correspondence_distance_m=self.config.agreement_m,
        )
        details["surface_information"] = {
            "strength": info.strength.tolist(),
            "basis": info.basis.tolist(),
            "pivot_m": info.pivot.tolist(),
            "matched_points": info.matched_points,
        }
        if self.config.grow:
            details["growth"] = self.store.extend(
                points,
                self.pose,
                frame,
                view=SurfaceView.from_observation(
                    current,
                    prepared["intrinsics"],
                    self.growth_shape_config,
                    retain_measurement=self.config.partial_surface_matching,
                ),
            )
        # Deliberately named candidate: overlap alone is not calibrated reliability.
        return self._result("observed_candidate", time_s, details)

    def _remember_observation(self, prepared):
        """Save actual RGB-D/mask and its accepted pose for later image matching."""
        self.accepted_frame = prepared["frame_id"]
        self.accepted_pose = self.pose.copy()
        self.accepted_observation = prepared["current"]
        self.accepted_intrinsics = prepared["intrinsics"].copy()

    def _fit_views(self):
        if not self.config.partial_surface_matching:
            return ()
        views = {
            patch.frame: SurfaceFitView(
                {
                    "depth": patch.view.depth,
                    "mask": patch.view.mask,
                    "obs": {"points": patch.view.measured_points},
                    **({"scene_depth": patch.view.scene_depth} if patch.view.scene_depth is not None else {}),
                },
                patch.view.intrinsics,
                patch.camera_from_reference,
            )
            for patch in self.store.patches.values()
            if patch.active and patch.view is not None
        }
        if self.accepted_observation is not None:
            # Last accepted measurement may reveal a useful view without
            # qualifying for surface growth. Never retain a rejected candidate.
            views[self.accepted_frame] = SurfaceFitView(
                self.accepted_observation, self.accepted_intrinsics, self.accepted_pose
            )
        if self.config.fit_view_limit is not None:
            return tuple(views[f] for f in sorted(views, reverse=True)[: self.config.fit_view_limit])
        return tuple(views.values())

    def _appearance_views(self):
        # Keep the first actual RGB-D measurement as an identity anchor. Colors
        # from a later accepted view fill unseen areas but never overwrite it.
        views = []
        if self.initial_appearance_view is not None:
            views.append(self.initial_appearance_view)
        if self.accepted_observation is not None:
            views.append(
                SurfaceFitView(self.accepted_observation, self.accepted_intrinsics, self.accepted_pose)
            )
        return tuple(views)

    def _rollback_correction(self, prior, details, reason):
        self.kalman.restore(prior)
        self.pose = self.kalman.pose.copy()
        # Preserve filter_reason for diagnosis, but don't report an update
        # discarded by a later acceptance check as an accepted correction.
        details["kalman"].update(accepted=False, reason=reason)

    def _publish_prediction(
        self, hint, time_s, details, reason, *, geometry=None, require_overlap=False, prediction_check=None
    ):
        """One fallback decision after missing input, failed fitting or rejected fusion.

        prediction_check reuses the current-frame check already performed before
        a Kalman correction. It validates the restored prediction, not the
        rejected fused pose. It never triggers another fit.
        """
        details["reason"] = reason
        if self.kalman is not None:
            # Prediction already propagated the posterior and group-motion
            # noise. Never replace it with an old absolute group anchor.
            self.pose = self.kalman.pose.copy()
            details.update(
                prediction_source=self.kalman.last_prediction.get("motion_source") or "last_world_pose",
                prediction_age_s=time_s - self.last_observed_time,
            )
            available = hint is not None or self.config.retain_unobserved_pose
        elif hint is not None and self.pose is not None:
            self.pose = hint.camera_from_reference.copy()
            self.last_supported_time = time_s
            details.update(prediction_source=hint.source, prediction_age_s=time_s - hint.observed_time_s)
            available = True
        elif self.config.retain_unobserved_pose and self.pose is not None:
            details.update(
                prediction_source="last_world_pose", prediction_age_s=time_s - self.last_supported_time
            )
            available = True
        else:
            return self._result("lost", time_s, details, available=False)
        if prediction_check is not None:
            # A current compatible observation can support retaining the prior
            # even when no external group supports it. An incompatible prior
            # must not be published merely because a group prediction exists.
            details["prediction_geometry"] = prediction_check
            available = prediction_check["accepted"]
        elif available and geometry is not None:
            # A failed alignment is not an empty observation. The fallback
            # still has to satisfy the current measured geometry. With no
            # usable observation, callers omit geometry and retain prediction.
            check = (
                geometry.check_pose(
                    self.pose,
                    max_outlier_fraction=self.config.uncertainty.max_visible_outlier_fraction,
                    agreement_m=self.config.agreement_m,
                    min_overlap=self.config.min_overlap,
                )
                if require_overlap
                else check_surface_contradiction(geometry, self.pose, self.config.uncertainty)
            )
            details["prediction_geometry"] = check
            if not check["accepted"]:
                return self._result("lost", time_s, details, available=False)
        return self._result("predicted", time_s, details, available=available)

    def _result(self, status, time_s, details, *, available=True):
        published_pose = self.pose.tolist() if available and self.pose is not None else None
        if published_pose is None:
            status = "lost"
        if self.kalman is not None:
            details["covariance"] = self.kalman.covariance.tolist()
            details.setdefault("kalman", {**self.kalman.last_prediction, "accepted": False, "reason": status})
            details["kalman"].update(
                posterior_covariance=self.kalman.covariance.tolist(),
                pivot_m=self.kalman.center().tolist(),
                calibrated=False,
            )
        details.update(
            unpublished_camera_from_reference=self.pose.tolist()
            if published_pose is None and self.pose is not None
            else None,
            map_version_after=self.store.version,
            map_points=len(self.store.points),
            retained_patches=len(self.store.patches),
            observation_age_s=time_s - self.last_observed_time
            if self.last_observed_time is not None
            else None,
            map_events=self.store.events,
        )
        self.store.events = []
        return {
            "status": status,
            "source_time_s": time_s,
            "camera_from_reference": published_pose,
            "diagnostics": details,
        }
