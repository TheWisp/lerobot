# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""What the act engine reads of perception and of the arm, each at its own time (act_execution.md, "The loops, and
who writes what"): each object's measured pose per camera frame, the times of the camera frames, and the fingertip
from the joint readings.

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
class Measurement:
    """An object's pose as the trackers measured it in one camera frame, which they were confident in: its motion
    since the demo, base frame, at the frame's capture time."""

    t: float
    motion: np.ndarray


class SceneLog:
    """Measurements, camera frame times and fingertip readings, kept for ``keep_s``. Writers append in time order,
    from any thread; readers ask by time."""

    def __init__(self, camera: Camera | None = None, keep_s: float = 30.0):
        self.camera = camera
        self.keep_s = keep_s
        self._measurements: dict[str, deque[Measurement]] = {}
        self._frames: deque[float] = deque()
        self._tip_t: deque[float] = deque()
        self._tips: deque[np.ndarray] = deque()
        self._lock = threading.Lock()

    def add_frame(self, t: float) -> None:
        """A camera frame was read at ``t``, whatever it measured."""
        with self._lock:
            self._frames.append(float(t))
            while self._frames and self._frames[0] < t - self.keep_s:
                self._frames.popleft()

    def add_measurement(self, name: str, m: Measurement) -> None:
        with self._lock:
            q = self._measurements.setdefault(name, deque())
            q.append(m)
            while q and q[0].t < m.t - self.keep_s:
                q.popleft()

    def add_tip(self, t: float, tip: np.ndarray) -> None:
        with self._lock:
            self._tip_t.append(float(t))
            self._tips.append(np.asarray(tip, dtype=float))
            while self._tip_t and self._tip_t[0] < t - self.keep_s:
                self._tip_t.popleft()
                self._tips.popleft()

    def frames(self, since: float = -np.inf) -> list[float]:
        with self._lock:
            return [t for t in self._frames if t >= since]

    def measurements(self, name: str, since: float = -np.inf) -> list[Measurement]:
        with self._lock:
            return [m for m in self._measurements.get(name, ()) if m.t >= since]

    def tip_now(self) -> np.ndarray | None:
        """The newest fingertip reading; None before any."""
        with self._lock:
            return self._tips[-1] if self._tips else None

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
