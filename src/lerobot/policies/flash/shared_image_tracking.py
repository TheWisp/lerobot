"""One causal image-matching pass, shared by object fitting and scene groups.

Point IDs belong to this session, never to a fitted pose. A bounded history can
join surviving identities across a missed object observation without invoking
another matcher. Image matching continues when every object mask is empty.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from .depth_quality import depth_support_mask
from .image_matching import LKMatchConfig, UFMMatchConfig, prepare_image_pair
from .registration import transform_points, unproject_pixels
from .scene_groups import SceneTracks
from .surface_correspondence import measured_pairs


@dataclass(frozen=True)
class SharedMatchingConfig:
    backend: Literal["ufm", "lk"] = "ufm"
    # Full-scene coverage; a long-edge budget, not an object-dependent crop.
    max_image_size_px: int = 640
    scene_points: int = 800
    object_points: int = 256
    spacing_px: int = 5  # In the matching image; both backends use identical seeds.
    history_frames: int = 64  # Pixel/ID snapshots only, no RGB-D history copies.
    depth_support_radius_px: int = 1
    lk: LKMatchConfig = field(default_factory=LKMatchConfig)
    ufm: UFMMatchConfig = field(default_factory=UFMMatchConfig)

    def __post_init__(self):
        if self.backend not in ("ufm", "lk"):
            raise ValueError("choose ufm or lk for shared image matching")
        for key in ("max_image_size_px", "scene_points", "object_points", "spacing_px", "history_frames"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.depth_support_radius_px) is not int or self.depth_support_radius_px < 0:
            raise ValueError("depth support radius must be nonnegative")
        if self.history_frames < 2:
            raise ValueError("point history must retain both frames of a match")

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        values["lk"] = LKMatchConfig(**values.get("lk", {}))
        values["ufm"] = UFMMatchConfig(**values.get("ufm", {}))
        return cls(**values)


@dataclass(frozen=True)
class PointSnapshot:
    ids: np.ndarray
    xy: np.ndarray  # Original sensor pixels, not resized-image coordinates.
    metric_valid: np.ndarray


def pixels_inside(xy, shape):
    finite = np.isfinite(xy).all(axis=1)
    px = np.zeros(xy.shape, dtype=int)
    px[finite] = np.rint(xy[finite]).astype(int)
    inside = finite & (px[:, 0] >= 0) & (px[:, 0] < shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < shape[0])
    return px, inside


class SharedFrameTracker:
    def __init__(self, config: SharedMatchingConfig, model=None):
        self.config, self.model = config, model
        self.previous = None
        self.image = None
        self.scale = None
        self.ids = np.empty(0, dtype=np.int64)
        self.xy = np.empty((0, 2), dtype=float)
        self.owners = np.empty(0, dtype=object)  # Budget pools, not inferred object ownership.
        self.next_id = 0
        self.history = OrderedDict()
        self.raw = None
        self.calls = 0
        self.diagnostics = {}

    def _image(self, rgb):
        scale = min(1.0, self.config.max_image_size_px / max(rgb.shape[:2]))
        size = (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale)))
        image = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA) if size != rgb.shape[1::-1] else rgb
        return image, np.asarray(size) / np.asarray(rgb.shape[1::-1])

    def _seed(self, frame, masks, image, scale, depth_supported):
        # Reserve capacity where the object is visible NOW. A slipped point
        # outside its mask can still support the scene, but cannot consume that
        # object's entire budget and prevent replenishment after it reappears.
        px, inside = pixels_inside(self.xy, frame.depth.shape)
        self.owners = np.full(len(self.ids), None, dtype=object)
        rows = np.flatnonzero(inside)
        for role, mask in masks.items():
            self.owners[rows[mask[px[rows, 1], px[rows, 0]]]] = role
        keep = np.zeros(len(self.ids), bool)
        for owner, budget in [(None, self.config.scene_points)] + [
            (role, self.config.object_points) for role in masks
        ]:
            keep[np.flatnonzero(self.owners == owner)[:budget]] = True
        self.ids, self.xy, self.owners = self.ids[keep], self.xy[keep], self.owners[keep]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        occupied = np.zeros(gray.shape, np.uint8)
        for x, y in (self.xy + 0.5) * scale - 0.5:
            cv2.circle(occupied, (round(float(x)), round(float(y))), self.config.spacing_px, 1, -1)
        # The per-object reserve prevents the background from exhausting the
        # point budget. New seeds get new IDs; they never extend old identities.
        pools = [(role, mask & depth_supported, self.config.object_points) for role, mask in masks.items()]
        background = depth_supported.copy()
        for mask in masks.values():
            background &= ~mask
        pools.append((None, background, self.config.scene_points))
        for owner, allowed, budget in pools:
            needed = budget - int(np.count_nonzero(self.owners == owner))
            if needed <= 0:
                continue
            allowed = cv2.resize(allowed.astype(np.uint8), gray.shape[::-1], interpolation=cv2.INTER_NEAREST)
            allowed[occupied > 0] = 0
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=needed,
                qualityLevel=0.005,
                minDistance=self.config.spacing_px,
                mask=allowed,
            )
            candidates = corners.reshape(-1, 2).tolist() if corners is not None else []
            # UFM and LK receive the same geometry-distributed seeds on low-
            # texture objects. This supplies queries, not a claim of reliability.
            if owner is not None:
                y, x = np.nonzero(allowed)
                cells = np.column_stack((x, y)) // self.config.spacing_px
                if len(cells):
                    _, rows = np.unique(cells, axis=0, return_index=True)

                    candidates.extend(np.column_stack((x[rows], y[rows])).tolist())
            fresh = []
            for uv in candidates:
                x, y = np.rint(uv).astype(int)
                if occupied[y, x]:
                    continue
                full = (np.asarray(uv) + 0.5) / scale - 0.5
                px, valid = pixels_inside(full[None], frame.depth.shape)
                if not valid[0] or not depth_supported[px[0, 1], px[0, 0]]:
                    continue
                if owner is not None and not masks[owner][px[0, 1], px[0, 0]]:
                    continue
                fresh.append(full)
                cv2.circle(occupied, (x, y), self.config.spacing_px, 1, -1)
                if len(fresh) == needed:
                    break
            if fresh:
                self.xy = np.concatenate((self.xy, fresh))
                self.ids = np.concatenate((self.ids, np.arange(self.next_id, self.next_id + len(fresh))))
                self.owners = np.concatenate((self.owners, np.full(len(fresh), owner, dtype=object)))
                self.next_id += len(fresh)

    def update(self, frame, masks) -> SceneTracks:
        start = time.perf_counter()
        if any(m.dtype != bool or m.shape != frame.depth.shape for m in masks.values()):
            raise ValueError("object masks must align with the current RGB-D frame")
        if self.previous is not None:
            if frame.frame_id <= self.previous.frame_id or frame.source_time_s <= self.previous.source_time_s:
                raise ValueError("shared matching requires increasing frame IDs and camera timestamps")
            if frame.rgb.shape != self.previous.rgb.shape:
                raise ValueError("reset shared matching after camera resolution changes")
        image, scale = self._image(frame.rgb)
        self.raw = None
        match_ms = 0.0
        if self.previous is not None:
            begin = time.perf_counter()
            pair = prepare_image_pair(
                self.image,
                image,
                self.config.ufm if self.config.backend == "ufm" else self.config.lk,
                model=self.model,
            )
            matches = pair.match((self.xy + 0.5) * scale - 0.5)
            self.calls += 1
            target = (matches.target_xy + 0.5) / scale - 0.5
            _, inside = pixels_inside(target, frame.depth.shape)
            good = matches.accepted & inside
            self.raw = (self.previous.frame_id, self.ids.copy(), self.xy.copy(), target, good)
            self.ids, self.xy, self.owners = self.ids[good], target[good], self.owners[good]
            match_ms = (time.perf_counter() - begin) * 1000
        depth_supported = depth_support_mask(frame.depth, radius_px=self.config.depth_support_radius_px)
        self._seed(frame, masks, image, scale, depth_supported)
        px, good = pixels_inside(self.xy, frame.depth.shape)
        rows = np.flatnonzero(good)
        rows = rows[depth_supported[px[rows, 1], px[rows, 0]]]
        metric_valid = np.zeros(len(self.ids), bool)
        metric_valid[rows] = True
        self.history[frame.frame_id] = PointSnapshot(self.ids.copy(), self.xy.copy(), metric_valid)
        while len(self.history) > self.config.history_frames:
            self.history.popitem(last=False)
        camera_xyz = unproject_pixels(frame.depth, frame.intrinsics, px[rows], frame.depth_m_per_unit)
        scene = SceneTracks(
            frame.frame_id,
            frame.source_time_s,
            self.ids[rows].copy(),
            px[rows].copy(),
            transform_points(frame.base_from_camera, camera_xyz),
        )
        self.diagnostics = {
            "backend": self.config.backend,
            "source_frame": self.previous.frame_id if self.previous else None,
            "target_frame": frame.frame_id,
            "matching_ms": match_ms,
            "calls_total": self.calls,
            "tracked_points": len(self.ids),
            "measured_points": len(rows),
            "image_size": list(image.shape[:2]),
            "total_ms": (time.perf_counter() - start) * 1000,
        }
        self.previous, self.image, self.scale = frame, image, scale
        return scene

    def pairs(self, a, b, intrinsics, *, evidence=None):
        """Filter the SAME shared matches for this object's depth/mask fit.

        Retained references use only surviving IDs in the bounded point history.
        No fallback matcher is invoked when identities or history are absent.
        """
        if self.previous is None or b["frame_id"] != self.previous.frame_id:
            raise ValueError("matching evidence belongs to a different current frame")
        if self.raw is not None and a["frame_id"] == self.raw[0]:
            _, ids, source, target, accepted = self.raw
        elif a["frame_id"] in self.history:
            old = self.history[a["frame_id"]]
            ids, left, right = np.intersect1d(old.ids, self.ids, return_indices=True)
            source, target, accepted = old.xy[left], self.xy[right], np.ones(len(ids), bool)
        else:
            ids, source, target, accepted = (
                np.empty(0, int),
                np.empty((0, 2)),
                np.empty((0, 2)),
                np.zeros(0, bool),
            )
        pa, inside_a = pixels_inside(source, a["depth"].shape)
        pb, inside_b = pixels_inside(target, b["depth"].shape)
        rows = np.flatnonzero(inside_a & np.isfinite(target).all(axis=1))
        rows = rows[a["mask"][pa[rows, 1], pa[rows, 0]]]
        pa, pb = pa[rows], pb[rows]
        stage = np.zeros(len(rows), np.uint8)
        stage[accepted[rows]] = 1
        use = np.flatnonzero(accepted[rows] & inside_b[rows])
        use = use[b["mask"][pb[use, 1], pb[use, 0]]]
        stage[use] = 2
        clean = a["usable"][pa[use, 1], pa[use, 0]] & b["usable"][pb[use, 1], pb[use, 0]]
        old = self.history.get(a["frame_id"])
        now = self.history[b["frame_id"]]
        metric = (
            np.isin(ids[rows], old.ids[old.metric_valid]) if old is not None else np.zeros(len(rows), bool)
        )
        metric &= np.isin(ids[rows], now.ids[now.metric_valid])
        stage[use[clean & metric[use]]] = 3
        source, target, ids = source[rows], target[rows], ids[rows]
        if evidence is not None:
            evidence.update(
                source_uv=source, target_uv=target, stage=stage, track_ids=ids, backend=self.config.backend
            )
        return measured_pairs(a, b, source[stage == 3], target[stage == 3], intrinsics)
