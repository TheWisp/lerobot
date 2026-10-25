"""Experimental, reversible multiview surface admission in a fixed object frame.

This does not decide whether a pose is trustworthy. The caller must supply a
pose accepted independently of this map; otherwise adding geometry would let
an incorrect pose manufacture its own supporting evidence. Each admitted point
retains its source frames so a future hypothesis manager can roll it back.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .surface_observation import move


@dataclass(frozen=True)
class SurfaceMapConfig:
    spacing_m: float
    novelty_distance_m: float
    confirmations: int = 3
    pending_lifetime_updates: int = 15
    max_points: int = 5000

    def __post_init__(self):
        if (
            not np.isfinite(self.spacing_m)
            or self.spacing_m <= 0
            or not np.isfinite(self.novelty_distance_m)
            or self.novelty_distance_m < self.spacing_m
            or self.confirmations < 2
            or self.pending_lifetime_updates < self.confirmations
        ):
            raise ValueError("surface admission needs positive spacing and repeated observations")
        if self.max_points < 3:
            raise ValueError("surface map needs capacity for at least three points")


class CandidateSurfaceMap:
    def __init__(
        self,
        reference: np.ndarray,
        config: SurfaceMapConfig,
        *,
        source_frame: int,
        reference_normals: np.ndarray | None = None,
    ):
        if reference.ndim != 2 or reference.shape[1] != 3 or len(reference) < 3:
            raise ValueError("surface map needs a measured 3D reference")
        if len(reference) > config.max_points:
            raise ValueError("map capacity must retain the complete initial reference")
        self.points = reference.copy()
        if reference_normals is not None and (
            reference_normals.shape != reference.shape or not np.isfinite(reference_normals).all()
        ):
            raise ValueError("surface normals must align with the measured reference")
        self.normals = reference_normals.copy() if reference_normals is not None else None
        self.config = config
        self.provenance = [(source_frame,)] * len(reference)
        self._tree = cKDTree(self.points)
        self._pending: dict[tuple[int, int, int], tuple[int, np.ndarray, np.ndarray, list[int], int]] = {}
        self._update = 0
        self._last_frame = source_frame

    def observe(
        self,
        measured_camera: np.ndarray,
        camera_from_reference: np.ndarray,
        frame_id: int,
        *,
        measured_normals_camera: np.ndarray | None = None,
    ) -> dict:
        """Admit repeatable *new* measured samples after the caller accepted this pose."""
        if frame_id <= self._last_frame:
            raise ValueError("surface admission requires a newer camera frame")
        if self.normals is not None and (
            measured_normals_camera is None or measured_normals_camera.shape != measured_camera.shape
        ):
            raise ValueError("oriented surface admission needs aligned measured normals")
        self._last_frame = frame_id
        self._update += 1
        local = move(measured_camera, np.linalg.inv(camera_from_reference))
        local_normals = (
            measured_normals_camera @ camera_from_reference[:3, :3]
            if measured_normals_camera is not None
            else None
        )
        if self.normals is None:
            distances = self._tree.query(local)[0]
            novel_mask = distances > self.config.novelty_distance_m
        else:
            # Opposite faces of a thin object can be close in XYZ. A nearby
            # point only explains the observation if its normal faces the
            # same hemisphere in the fixed object frame.
            nearby = self._tree.query_ball_point(local, self.config.novelty_distance_m)
            novel_mask = np.array(
                [
                    not neighbors or not np.any(self.normals[neighbors] @ normal >= 0)
                    for neighbors, normal in zip(nearby, local_normals, strict=True)
                ],
                dtype=bool,
            )
        novel = local[novel_mask]
        novel_normals = local_normals[novel_mask] if local_normals is not None else None
        keys = np.floor(novel / self.config.spacing_m).astype(np.int64)
        _, indices = np.unique(keys, axis=0, return_index=True)
        promoted, promoted_normals, samples = [], [], []
        for index in indices:
            key = tuple(keys[index])
            point = novel[index]
            normal = novel_normals[index] if novel_normals is not None else np.zeros(3)
            count, total, normal_total, frames, last = self._pending.get(
                key, (0, np.zeros(3), np.zeros(3), [], self._update)
            )
            if self._update - last > self.config.pending_lifetime_updates:
                count, total, normal_total, frames = 0, np.zeros(3), np.zeros(3), []
            count += 1
            total = total + point
            normal_total = normal_total + normal
            frames = [*frames, frame_id]
            if (
                count >= self.config.confirmations
                and len(self.points) + len(promoted) < self.config.max_points
            ):
                promoted.append(total / count)
                promoted_normals.append(normal_total / max(np.linalg.norm(normal_total), np.finfo(float).eps))
                samples.append(tuple(frames))
                self._pending.pop(key, None)
            else:
                self._pending[key] = (count, total, normal_total, frames, self._update)
        if promoted:
            self.points = np.concatenate((self.points, promoted))
            if self.normals is not None:
                self.normals = np.concatenate((self.normals, promoted_normals))
            self.provenance.extend(samples)
            self._tree = cKDTree(self.points)
        self._pending = {
            key: value
            for key, value in self._pending.items()
            if self._update - value[4] <= self.config.pending_lifetime_updates
        }
        return {
            "novel_samples": len(indices),
            "promoted": len(promoted),
            "pending": len(self._pending),
            "map_points": len(self.points),
        }
