# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Where each object is, as the commands use it, and how much what the camera says of it wanders while it stands
untouched (act_execution.md, "Legs" and "How close counts").

An object's pose here is its motion since the demo, base frame. The commands use a copy that moves toward the newest
accepted view no faster than the walk speed, so a jumpy view cannot jerk the arm. Its noise is measured on the views,
tracked points and depth it got while untouched in this act, never set.
"""

from __future__ import annotations

import numpy as np

from lerobot.showservo import motion as core

from . import predicates
from .scene import SceneLog
from .timeline import BASE

PIXEL = 1.0  # what a tracked point is measured in: the noise of a perfectly steady one is one pixel, not zero


def _toward(a: np.ndarray, b: np.ndarray, centre: np.ndarray, max_m: float, max_rad: float) -> np.ndarray:
    """Pose ``a`` moved toward ``b`` by at most ``max_m`` at ``centre`` and ``max_rad`` of turn."""
    c = np.append(centre, 1.0)
    dist = float(np.linalg.norm((b @ c)[:3] - (a @ c)[:3]))
    _, deg = core.pose_residual(a, b)
    ang = np.radians(deg)
    f = 1.0
    if dist > max_m:
        f = min(f, max_m / dist)
    if ang > max_rad:
        f = min(f, max_rad / ang)
    return b.copy() if f >= 1.0 else core.interp_rigid(a, b, f)


class Objects:
    """Each named object's pose for the commands and its measured noise. ``start`` is each one's motion since the
    demo as the act found it; ``surfaces`` its surface points at its demo pose (base frame); ``depth_unit_m`` the
    depth image's own step, the least its depth or a view can be trusted to."""

    def __init__(
        self,
        scene: SceneLog,
        start: dict[str, np.ndarray],
        surfaces: dict[str, np.ndarray],
        walk_m_s: float,
        walk_rad_s: float,
        depth_unit_m: float = 0.001,
    ):
        self.scene = scene
        self.used = {n: np.asarray(m, dtype=float).copy() for n, m in start.items()}
        self.latest = {n: m.copy() for n, m in self.used.items()}
        self.latest_t = dict.fromkeys(self.used, -np.inf)
        self.surfaces = {n: np.asarray(s, dtype=float) for n, s in surfaces.items()}
        self.centres = {n: s.mean(axis=0) for n, s in self.surfaces.items()}
        self.walk_m_s, self.walk_rad_s = walk_m_s, walk_rad_s
        self.depth_unit_m = depth_unit_m
        self.rest = {n: [-np.inf, np.inf] for n in self.used}  # [since, until): what counts toward its noise
        self.camera = scene.camera
        self._at_touch: dict[
            str, dict[str, object]
        ] = {}  # each touched object's noise, measured as it was touched

    def update(self, dt: float, hold: set[str] = frozenset()) -> None:
        """Read the newest views; move each object's pose toward its newest by at most the walk speed over ``dt``. An
        object in ``hold`` (one the arm is in contact with) takes only corrections within its noise: a larger one is
        for the engine to decide, not to drift into."""
        for n in self.used:
            views = self.scene.views(n, since=self.latest_t[n] + 1e-9)
            if views:
                self.latest[n], self.latest_t[n] = views[-1].motion, views[-1].t
            if n in hold and self.off(n) > self.move_threshold(n):
                continue
            self.used[n] = _toward(
                self.used[n],
                self.latest[n],
                self.centres.get(n, np.zeros(3)),
                self.walk_m_s * dt,
                self.walk_rad_s * dt,
            )

    def off(self, name: str) -> float:
        """How far the newest view puts the object's centre from where the commands have it, metres."""
        c = np.append(self.centres.get(name, np.zeros(3)), 1.0)
        return float(np.linalg.norm((self.latest[name] @ c)[:3] - (self.used[name] @ c)[:3]))

    def move_threshold(self, name: str) -> float:
        """How far a view must put an object from where the commands have it to say it moved: twice its noise (the
        two poses each wander that much); infinite while the noise is unmeasured."""
        noise = self.view_noise(name)
        return np.inf if noise is None else 2.0 * noise[0]

    def snap(self, name: str) -> None:
        """Use the newest view at once: after a failure, when the arm is about to move anew toward the object."""
        self.used[name] = self.latest[name].copy()

    def motion(self, name: str) -> np.ndarray:
        return np.eye(4) if name == BASE or name not in self.used else self.used[name]

    def touched(self, name: str, t: float) -> None:
        """From ``t`` the object is in the act's hands: what the camera says of it no longer counts as noise. Its noise
        is measured now, while the frames of it standing untouched are still in the log."""
        if name in self.rest:
            self.rest[name][1] = min(self.rest[name][1], t)
            self._at_touch[name] = {
                "view": self._view_noise(name),
                "track": self._track_noise(name),
                "depth": self._depth_noise(name, self.camera),
            }

    def resting(self, name: str, t: float) -> None:
        """From ``t`` it stands untouched again, somewhere new: its noise is measured afresh."""
        if name in self.rest:
            self.rest[name] = [t, np.inf]
            self._at_touch.pop(name, None)

    def view_noise(self, name: str) -> tuple[float, float] | None:
        return self._at_touch[name]["view"] if name in self._at_touch else self._view_noise(name)

    def track_noise(self, name: str) -> float | None:
        return self._at_touch[name]["track"] if name in self._at_touch else self._track_noise(name)

    def depth_noise(self, name: str, camera=None) -> float | None:
        if name in self._at_touch:
            return self._at_touch[name]["depth"]
        return self._depth_noise(name, camera or self.camera)

    def _in_rest(self, name: str, t: float) -> bool:
        since, until = self.rest.get(name, (-np.inf, np.inf))
        return since <= t < until

    def _view_noise(self, name: str) -> tuple[float, float] | None:
        """How far its views wandered while it stood untouched: metres at its centre and degrees, or None with too few
        views; never under the depth image's step."""
        views = [v for v in self.scene.views(name) if self._in_rest(name, v.t)]
        c = np.append(self.centres.get(name, np.zeros(3)), 1.0)
        m = predicates.spread(np.array([(v.motion @ c)[:3] for v in views])) if views else None
        d = predicates.spread_deg(np.array([v.motion[:3, :3] for v in views])) if views else None
        if m is None or d is None:
            return None
        return max(m, self.depth_unit_m), d

    def _track_noise(self, name: str) -> float | None:
        """How far the median of its tracked points wandered while it stood untouched, in pixels, from the first such
        frame; None with too few frames; never under a pixel."""
        frames = [f for f in self.scene.tracks(name) if self._in_rest(name, f.t)]
        if len(frames) < predicates.MEDIAN_MIN:
            return None
        ref = frames[0]
        before = {int(i): xy for i, xy, seen in zip(ref.ids, ref.uv, ref.visible, strict=True) if seen}
        shifts = []
        for f in frames[1:]:
            d = [
                xy - before[int(i)]
                for i, xy, seen in zip(f.ids, f.uv, f.visible, strict=True)
                if seen and int(i) in before
            ]
            if len(d) >= predicates.MEDIAN_MIN:
                shifts.append(np.median(np.array(d), axis=0))
        noise = predicates.spread(np.array(shifts)) if len(shifts) >= predicates.MEDIAN_MIN else None
        return None if noise is None else max(noise, PIXEL)

    def _depth_noise(self, name: str, camera) -> float | None:
        """How far the depth on its surface wandered while it stood untouched: per surface pixel, its distances from
        that pixel's median over the frames, at the 95th percentile; never under the depth image's step."""
        frames = [d for d in self.scene.depths() if self._in_rest(name, d.t)]
        if len(frames) < predicates.MEDIAN_MIN or name not in self.surfaces or camera is None:
            return None
        m = self.used[name]
        pts = self.surfaces[name] @ m[:3, :3].T + m[:3, 3]
        uv, z = camera.project(pts)
        h, w = frames[0].metres.shape
        ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        ok = (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        if not ok.any():
            return None
        stack = np.stack([f.metres[vi[ok], ui[ok]] for f in frames])
        read = np.all(stack > 0, axis=0)
        if not read.any():
            return None
        dev = np.abs(stack[:, read] - np.median(stack[:, read], axis=0))
        return max(float(np.percentile(dev, 95)), self.depth_unit_m)
