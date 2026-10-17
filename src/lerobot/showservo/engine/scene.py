# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""What perception reports to the act engine, each with the time its camera frame was read.

Every pose here is a *motion since the demo*, base frame, the convention the act composes with: an object's motion
maps its pose in the demo to its pose now, so the identity means "where the demo had it". The camera's pose in the
base is ``t_bc`` (a point in the camera frame times ``t_bc`` is the point in the base frame).
"""

from __future__ import annotations

import bisect
import dataclasses
import threading
from collections import deque

import numpy as np


@dataclasses.dataclass(frozen=True)
class Camera:
    K: np.ndarray  # (3, 3) intrinsics
    t_bc: np.ndarray  # (4, 4) the camera's pose in the base
    size: tuple[int, int]  # (height, width) of its frames

    def project(self, points_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixels (N, 2) and camera depths (N,) of base-frame points (N, 3); a point behind the camera has z <= 0."""
        p = np.asarray(points_base, dtype=float).reshape(-1, 3)
        t_cb = np.linalg.inv(self.t_bc)
        cam = p @ t_cb[:3, :3].T + t_cb[:3, 3]
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (cam[:, :2] / z[:, None]) * np.diag(self.K)[:2] + self.K[:2, 2]
        return uv, z


@dataclasses.dataclass(frozen=True)
class Tracks:
    """One frame's 2D tracks of one object: stable ids (a re-seeded track gets a new id), pixels, and whether each
    was seen in this frame."""

    t: float
    ids: np.ndarray  # (N,) int
    uv: np.ndarray  # (N, 2)
    visible: np.ndarray  # (N,) bool


@dataclasses.dataclass(frozen=True)
class Depth:
    t: float
    metres: np.ndarray  # (H, W) camera z; 0 where the camera read nothing


@dataclasses.dataclass(frozen=True)
class View:
    """A view the pose rule placed: the object's motion since the demo, base frame."""

    t: float
    motion: np.ndarray


class SceneLog:
    """The scene as perception reported it, kept for ``keep_s``: views, tracks and depth frames by time, and the arm's
    fingertip (from its observed joints). Writers append in time order, from any thread; readers ask by time."""

    def __init__(self, camera: Camera | None = None, keep_s: float = 30.0):
        self.camera = camera
        self.keep_s = keep_s
        self.surfaces: dict[str, np.ndarray] = {}  # each object's surface points at its demo pose, base frame
        self.symmetry_axes: dict[
            str, np.ndarray
        ] = {}  # a continuous symmetry axis, base frame at the demo pose
        self._views: dict[str, deque[View]] = {}
        self._tracks: dict[str, deque[Tracks]] = {}
        self._depth: deque[Depth] = deque()
        self._tip_t: deque[float] = deque()
        self._tips: deque[np.ndarray] = deque()
        self._lock = threading.Lock()

    def _trim(self, q: deque, now: float, stamp=lambda x: x.t) -> None:
        while q and stamp(q[0]) < now - self.keep_s:
            q.popleft()

    def add_view(self, name: str, view: View) -> None:
        with self._lock:
            q = self._views.setdefault(name, deque())
            q.append(view)
            self._trim(q, view.t)

    def add_tracks(self, name: str, tracks: Tracks) -> None:
        with self._lock:
            q = self._tracks.setdefault(name, deque())
            q.append(tracks)
            self._trim(q, tracks.t)

    def add_depth(self, depth: Depth) -> None:
        with self._lock:
            self._depth.append(depth)
            self._trim(self._depth, depth.t)

    def add_tip(self, t: float, tip: np.ndarray) -> None:
        with self._lock:
            self._tip_t.append(float(t))
            self._tips.append(np.asarray(tip, dtype=float))
            while self._tip_t and self._tip_t[0] < t - self.keep_s:
                self._tip_t.popleft()
                self._tips.popleft()

    def views(self, name: str, since: float = -np.inf) -> list[View]:
        with self._lock:
            return [v for v in self._views.get(name, ()) if v.t >= since]

    def tracks(self, name: str, since: float = -np.inf) -> list[Tracks]:
        with self._lock:
            return [v for v in self._tracks.get(name, ()) if v.t >= since]

    def depths(self, since: float = -np.inf) -> list[Depth]:
        with self._lock:
            return [d for d in self._depth if d.t >= since]

    def tip_at(self, t: float, max_gap_s: float = 0.2) -> np.ndarray | None:
        """The fingertip at ``t``, from the nearest reading within ``max_gap_s``; None without one."""
        with self._lock:
            if not self._tip_t:
                return None
            times = list(self._tip_t)
            k = bisect.bisect_left(times, t)
            best = min((j for j in (k - 1, k) if 0 <= j < len(times)), key=lambda j: abs(times[j] - t))
            if abs(times[best] - t) > max_gap_s:
                return None
            return self._tips[best]
