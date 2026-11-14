"""Accepted-observation surface replacement, shared by live and replay sessions.

All confirmed points participate in fitting. Only samples refreshed by the
latest accepted observation contribute negative evidence to the next check.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .experimental_surface_session import SurfacePatch, SurfaceStore
from .surface_grounding import SurfaceFitView, WholeSurfaceFit, project, scene_depth_evidence
from .surface_observation import move, spatial_sample


@dataclass
class RefreshedPatch(SurfacePatch):
    # Original measurements and retained views stay immutable; the working
    # geometry is a set of latest samples, whose size may differ from the raw patch.
    current_points: np.ndarray | None = None

    def points(self):
        return self.current_points if self.current_points is not None else super().points()


class RefreshingSurfaceStore(SurfaceStore):
    """Replace visible map samples with current measurements, without averaging.

    The refreshed map is resampled at the existing storage spacing. Association is projective:
    one storage-cell footprint finds the current measured sample at an old ray.
    Existing pose-depth tolerance distinguishes overlap from occlusion. There is
    no 3-D nearest-neighbour growth-distance cutoff on replacement. Unseen areas
    stay unchanged; contradicted points are retired by the existing shared rule.
    The inherited caller only invokes this after acceptance against the old map.
    """

    def __init__(self, config):
        super().__init__(config)
        self.last_refinement_frame = None

    def add(self, points, pose, frame, **kwargs):
        super().add(points, pose, frame, **kwargs)
        if frame in self.patches:
            self.patches[frame] = RefreshedPatch(**vars(self.patches[frame]))

    def reposition(self, frame, pose):
        self.patches[frame].current_points = None
        super().reposition(frame, pose)

    @staticmethod
    def _retirement_flags(patches, pose, view, tolerance_m):
        """Find old points contradicted by the accepted current observation.

        A reliable farther depth or a reliably non-object pixel at the same
        depth cannot back a visible stored point. Nearer depth hides it; missing
        depth, an uncertain mask boundary, and offscreen points say nothing.
        """
        lengths = [len(p.points()) for p in patches]
        if not lengths:
            return []
        local = np.concatenate([p.points() for p in patches])
        camera = move(local, pose)
        _, exposed, bad = scene_depth_evidence(camera, view.intrinsics, view.nearest_depth, tolerance_m)
        contradicted = bad >= 1.0 - 1e-6
        # A point can also be unsupported without a depth gap: the current
        # depth is at its distance, but the mask confidently identifies that
        # pixel as non-object. SurfaceView.contradicts applies the same
        # visibility, depth and mask-boundary rules used for patch additions.
        contradicted |= (exposed >= 1.0 - 1e-6) & view.contradicts(camera, tolerance_m=tolerance_m)
        offsets = np.cumsum([0, *lengths])
        return [contradicted[start:stop] for start, stop in zip(offsets[:-1], offsets[1:], strict=True)]

    def _refresh(self, measured, pose, frame, view, tolerance_m):
        used = np.zeros(len(measured), dtype=bool)
        report = {"refined_points": 0, "refinement_mean_shift_mm": 0.0, "update_mode": "replace"}
        if view is None or (self.last_refinement_frame is not None and frame <= self.last_refinement_frame):
            return used, report
        self.last_refinement_frame = frame
        used, report = self._replace_visible(measured, pose, frame, view, tolerance_m, used, report)
        patches = [p for p in self.patches.values() if len(p.points())]
        flags = self._retirement_flags(patches, pose, view, tolerance_m)
        report["retired_points"] = sum(int(bad.sum()) for bad in flags)
        for item, bad in zip(patches, flags, strict=True):
            if bad.any():
                item.current_points = item.points()[~bad].copy()
        if report["retired_points"]:
            self._rebuild()
        return used, report

    def _replace_visible(self, measured, pose, frame, view, tolerance_m, used, report):
        patches = [p for p in self.patches.values() if p.active and p.frame < frame and len(p.points())]
        if not patches or not len(measured):
            return used, report
        local = np.concatenate([p.points() for p in patches])
        camera = move(local, pose)
        uv = project(camera, view.intrinsics)
        height, width = view.nearest_depth.shape
        valid = (
            np.isfinite(uv).all(axis=1)
            & (camera[:, 2] > 0)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < width - 1)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < height - 1)
        )
        indices = np.flatnonzero(valid)
        xy = np.rint(uv[indices]).astype(int)
        depth = view.nearest_depth[xy[:, 1], xy[:, 0]]
        tolerance = tolerance_m
        supported = (
            (depth > 0) & ~view.outside_mask[xy[:, 1], xy[:, 0]] & (camera[indices, 2] <= depth + tolerance)
        )
        if view.mask is not None:
            supported &= view.mask[xy[:, 1], xy[:, 0]]
        indices = indices[supported]
        if not len(indices):
            return used, report
        measured_uv = project(measured, view.intrinsics)
        measured_valid = np.isfinite(measured_uv).all(axis=1) & (measured[:, 2] > 0)
        measured_ids = np.flatnonzero(measured_valid)
        if not len(measured_ids):
            return used, report
        pixel_distance, nearest = cKDTree(measured_uv[measured_ids]).query(uv[indices])
        nearest = measured_ids[nearest]
        footprint_px = (
            max(view.intrinsics[0, 0], view.intrinsics[1, 1]) * self.config.spacing_m / camera[indices, 2]
        )
        supported = (pixel_distance <= footprint_px) & (
            camera[indices, 2] <= measured[nearest, 2] + tolerance
        )
        indices, nearest = indices[supported], nearest[supported]
        if not len(indices):
            return used, report
        # Replace a region with the new point SET, not one new sample per old
        # sample. This retains all observed detail at the normal storage spacing.
        lengths = [len(p.points()) for p in patches]
        owners = np.repeat(np.arange(len(patches)), lengths)
        distance, old_nearest = cKDTree(uv[indices]).query(measured_uv[measured_ids])
        old_nearest = indices[old_nearest]
        current_overlap = (
            distance
            <= (
                max(view.intrinsics[0, 0], view.intrinsics[1, 1])
                * self.config.spacing_m
                / camera[old_nearest, 2]
            )
        ) & (camera[old_nearest, 2] <= measured[measured_ids, 2] + tolerance)
        fresh_ids = measured_ids[current_overlap]
        fresh_owners = owners[old_nearest[current_overlap]]
        fresh = move(measured[fresh_ids], np.linalg.inv(pose))
        removed = np.zeros(len(local), dtype=bool)
        removed[indices] = True
        offset = 0
        for owner, item in enumerate(patches):
            stop = offset + lengths[owner]
            kept = local[offset:stop][~removed[offset:stop]]
            current = np.concatenate((fresh[fresh_owners == owner], kept))
            # The combined map is already bounded, but retained working patches
            # also need a bound when observations repeatedly densify a region.
            # Keep actual samples and the original raw patch separately.
            item.current_points = current[spatial_sample(current, self.config.max_points)]
            offset = stop
        used[fresh_ids] = True
        report.update(
            refined_points=len(indices),
            refreshed_measured_points=len(fresh_ids),
            refinement_mean_shift_mm=float(
                1000 * np.linalg.norm(measured[nearest] - camera[indices], axis=1).mean()
            ),
        )
        self._rebuild()
        return used, report


def refreshed_membership(reference, observation, accepted_pose):
    """Which stored coordinates are actual reliable measurements of this view?

    Refresh stores measured points in object coordinates without averaging.
    The micrometre bound accommodates coordinate roundoff, not sensor noise or
    a correspondence tolerance. A merely nearby old point cannot gain a veto.
    No candidate pose is an input to this decision.
    """
    measured = move(observation["obs"]["points"], np.linalg.inv(accepted_pose))
    if not len(measured):
        return np.zeros(len(reference), dtype=bool)
    distance = cKDTree(measured).query(reference, distance_upper_bound=1e-6)[0]
    return distance < 1e-6


def rejection_geometry(full, accepted_view):
    eligible = refreshed_membership(
        full.reference, accepted_view.observation, accepted_view.camera_from_reference
    )
    # No reliable samples means no stored-point veto. Positive surface support
    # is measured separately against the full map, including unrefreshed parts.
    if not eligible.any():
        return eligible, None
    validator = WholeSurfaceFit(
        full.reference[eligible],
        full.observation,
        full.k,
        full.config,
        prepared_target=full.prepared_target,
        fit_kernels=full.fit_kernels,
        search_pool=full.search_pool,
        # Only the latest accepted measurement defines rejection reliability.
        # Older retained views still participate in the original fit objective.
        reference_views=(accepted_view,),
    )
    return eligible, validator


def attach_rejection_geometry(full, accepted_view):
    eligible, validator = rejection_geometry(full, accepted_view)
    full._latest_rejection = (eligible, validator, accepted_view.observation["frame_id"])
    return eligible


class LatestObservationFit(WholeSurfaceFit):
    """Full-map positive agreement with frozen latest-measurement vetoes."""

    def __init__(self, *args, accepted_view: SurfaceFitView, **kwargs):
        super().__init__(*args, **kwargs)
        attach_rejection_geometry(self, accepted_view)

    def check_pose(self, pose, *, max_outlier_fraction=None, agreement_m=None, min_overlap=0.0):
        full = self
        eligible, validator, frame = full._latest_rejection
        distances = full.prepared_reference.tree.query(
            (full.observation["obs"]["points"] - pose[:3, 3]) @ pose[:3, :3]
        )[0]
        overlap = (
            float(np.mean(distances <= (agreement_m or full.config.depth_tolerance_m)))
            if len(distances)
            else 0.0
        )
        if validator is None:
            result = {
                "observed_outlier_fraction": 0.0,
                "free_space_outlier_fraction": 0.0,
                "scene_depth_coherent_protrusion": False,
                "scene_depth_outlier_fraction": 0.0,
                "scene_depth_views": [],
            }
        else:
            result = {**validator.partial_surface_check(pose), **validator.scene_depth_check(pose)}
        geometry_accepted = not result["scene_depth_coherent_protrusion"]
        if max_outlier_fraction is not None:
            geometry_accepted &= (
                max(
                    result["observed_outlier_fraction"],
                    result["free_space_outlier_fraction"],
                    result["scene_depth_outlier_fraction"],
                )
                <= max_outlier_fraction
            )
        needs_overlap = max_outlier_fraction is not None or min_overlap > 0
        return {
            **result,
            "accepted": bool(
                geometry_accepted and (not needs_overlap or overlap > 0 and overlap >= min_overlap)
            ),
            "geometry_accepted": bool(geometry_accepted),
            "observed_overlap": overlap,
            "observed_overlap_basis": "current_points_to_full_map",
            "surface_median_mm": float(np.median(distances) * 1000) if len(distances) else None,
            "has_visible_overlap": overlap > 0,
            "rejection_reference_frame": frame,
            "reliable_reference_points": int(eligible.sum()),
            "full_reference_points": len(eligible),
        }
