"""Offline growing-surface experiment. Not a replacement for the live backend.

Reuse the baseline's observation cleaning and surface objective, but retain image
matches during fitting and keep reversible measured patches. Diagnostics and
acceptance thresholds are experimental, not calibrated pose confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .surface_grounding import WholeSurfaceFit
from .surface_motion import measured_surface_information
from .surface_observation import move, spatial_sample, voxel_sample


@dataclass(frozen=True)
class GrowingSurfaceConfig:
    # Match the baseline's 5 mm depth budget and 3 mm cloud sampling by default.
    spacing_m: float = 0.003
    agreement_m: float = 0.005
    # Require an overlapping portion, not a single accidental nearest neighbour.
    # These fractions are hypotheses to benchmark, not object-specific rules.
    min_overlap: float = 0.25
    patch_overlap: float = 0.5
    image_weight: float = 1.0
    # One later view checks an addition before it can influence fitting.
    confirmations: int = 2
    max_patches: int = 32
    max_pending: int = 8
    points_per_patch: int = 1024
    max_points: int = 8192
    grow: bool = True

    def __post_init__(self):
        for name in ("spacing_m", "agreement_m"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("min_overlap", "patch_overlap"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not np.isfinite(self.image_weight) or self.image_weight < 0:
            raise ValueError("image_weight must be finite and nonnegative")
        for name in ("confirmations", "max_patches", "max_pending", "points_per_patch", "max_points"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        if min(self.max_points, self.points_per_patch) < 3:
            raise ValueError("a surface needs at least three points")
        if not isinstance(self.grow, bool):
            raise ValueError("grow must be a boolean")


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

    def __post_init__(self):
        object.__setattr__(self, "camera_from_reference", checked_pose(self.camera_from_reference))
        if (
            not self.source
            or not np.isfinite([self.time_s, self.observed_time_s]).all()
            or self.observed_time_s > self.time_s
        ):
            raise ValueError("hint needs a source and causal timestamps")


@dataclass
class SurfacePatch:
    frame: int
    measured: np.ndarray
    camera_from_reference: np.ndarray
    fitted_map_version: int
    support_frames: set[int]
    active: bool = False

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

    def add(self, points, pose, frame, *, active=False, fitted_map_version=None):
        if frame in self.patches:
            raise ValueError("one patch per source frame")
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError("patch needs finite Nx3 camera measurements")
        points = points[spatial_sample(points, self.config.points_per_patch)].copy()
        version = self.version if fitted_map_version is None else fitted_map_version
        self.patches[frame] = SurfacePatch(frame, points, checked_pose(pose), version, {frame}, active)
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
            agreement = np.mean(distances <= self.config.agreement_m) if len(distances) else 0
            if agreement >= self.config.patch_overlap:
                patch.support_frames.add(frame)
                if len(patch.support_frames) >= self.config.confirmations:
                    patch.active = True
                    changed = True
                    self.events.append({"event": "confirmed", "frame": patch.frame, "by_frame": frame})
        if changed:
            self._bound()
            self._rebuild()


class ImageConditionedFit(WholeSurfaceFit):
    """Keep image correspondences in the objective, not only in its initial guess."""

    def __init__(self, *args, image_reference, image_current, image_scale_m, image_weight, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_reference, self.image_current = image_reference, image_current
        self.image_scale_m, self.image_weight = image_scale_m, image_weight

    def _residual_with_distances(self, pose, predicted, model_distance, target_distance):
        surface = super()._residual_with_distances(pose, predicted, model_distance, target_distance)
        if not len(self.image_reference) or self.image_weight == 0:
            return surface
        difference = (move(self.image_reference, pose) - self.image_current) / self.image_scale_m
        # Huber residual, normalized so increasing sampled image points does not
        # silently increase their total weight relative to geometry.
        squared = np.where(abs(difference) <= 1, difference**2, 2 * abs(difference) - 1)
        image = np.sign(difference) * np.sqrt(squared)
        return np.concatenate((surface, image.ravel() * np.sqrt(self.image_weight / len(image))))


class GrowingSurfaceSession:
    """One sequential experimental object track, with no robot/controller state."""

    def __init__(self, shape_config, config=None, *, fit_kernels=None, search_pool=None):
        self.config = config or GrowingSurfaceConfig()
        self.shape_config = shape_config
        self.fit_kernels, self.search_pool = fit_kernels, search_pool
        self.store = SurfaceStore(self.config)
        self.pose = None
        self.reference = None
        self.reference_frame = None
        self.connected_frame = None
        self.connected_pose = None
        self.connected_observation = None
        self.last_time = self.last_frame = None

    def update(self, prepared, time_s, *, hint=None):
        frame = prepared["frame_id"]
        if (
            not np.isfinite(time_s)
            or self.last_frame is not None
            and (frame <= self.last_frame or time_s <= self.last_time)
        ):
            raise ValueError("source frames and times must strictly increase")
        if hint is not None and (hint.time_s != time_s or hint.reference_frame != self.reference_frame):
            raise ValueError("hint belongs to another time or coordinate reference")
        self.last_frame, self.last_time = frame, time_s
        before = self.store.version
        details = {
            "map_version_used": before,
            "image_source_frame": prepared.get("match_source_frame"),
            "reference_frame": self.reference_frame,
            "covariance": None,
        }
        valid = prepared["status"] is None
        current = prepared["current"]
        points = current["obs"]["points"]
        if self.pose is None and valid:
            self.reference = points.copy()
            self.pose = np.eye(4)
            self.reference_frame = frame
            self.store.add(points, self.pose, frame, active=True)
            self._connect(prepared)
            details["reference_frame"] = frame
            return self._result("initialized", time_s, details)
        if self.pose is None or not valid:
            return self._missing(hint, time_s, details, prepared["status"] or "not_initialized")
        if len(self.store.points) < 3:
            return self._missing(hint, time_s, details, "surface_removed")
        source = prepared.get("match_source_frame")
        step = prepared["step"] if source == self.connected_frame else None
        seed = (
            step @ self.connected_pose
            if step is not None
            else (hint.camera_from_reference if hint else self.pose)
        )
        a, b = np.empty((0, 3)), np.empty((0, 3))
        if step is not None and "point_pairs" in prepared:
            a, b = prepared["point_pairs"]
            keep = np.linalg.norm(move(a, step) - b, axis=1) <= self.config.agreement_m
            a = move(a[keep], np.linalg.inv(self.connected_pose))
            b = b[keep]
        fitter = ImageConditionedFit(
            self.store.points,
            current,
            prepared["intrinsics"],
            self.shape_config,
            image_reference=a,
            image_current=b,
            image_scale_m=self.config.agreement_m,
            image_weight=self.config.image_weight,
            fit_kernels=self.fit_kernels,
            search_pool=self.search_pool,
        )
        candidate, fit = fitter.correct(seed)
        distances = cKDTree(move(self.store.points, candidate)).query(points)[0]
        overlap = float(np.mean(distances <= self.config.agreement_m))
        details.update(
            fit=fit,
            image_pairs=len(a),
            observed_overlap=overlap,
            surface_median_mm=float(np.median(distances) * 1000),
            candidate_camera_from_reference=candidate.tolist(),
        )
        if overlap < self.config.min_overlap:
            return self._missing(hint, time_s, details, "insufficient_surface_overlap")
        self.pose = candidate
        self._connect(prepared)
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
            self.store.confirm(points, self.pose, frame)
            distances = cKDTree(move(self.store.points, self.pose)).query(points)[0]
            novel = points[distances > self.config.spacing_m]
            if len(novel):
                self.store.add(novel, self.pose, frame, fitted_map_version=before)
        # Deliberately named candidate: overlap alone is not calibrated reliability.
        return self._result("observed_candidate", time_s, details)

    def _connect(self, prepared):
        self.connected_frame = prepared["frame_id"]
        self.connected_pose = self.pose.copy()
        self.connected_observation = prepared["current"]

    def _missing(self, hint, time_s, details, reason):
        details["reason"] = reason
        if hint is not None and self.pose is not None:
            self.pose = hint.camera_from_reference.copy()
            details.update(prediction_source=hint.source, prediction_age_s=time_s - hint.observed_time_s)
            return self._result("predicted", time_s, details)
        return self._result("lost", time_s, details, available=False)

    def _result(self, status, time_s, details, *, available=True):
        details.update(
            map_version_after=self.store.version,
            map_points=len(self.store.points),
            retained_patches=len(self.store.patches),
            map_events=self.store.events,
        )
        self.store.events = []
        return {
            "status": status,
            "source_time_s": time_s,
            "camera_from_reference": self.pose.tolist() if available and self.pose is not None else None,
            "diagnostics": details,
        }
