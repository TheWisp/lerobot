# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The world: each object's pose (act_execution.md, "The loops, and who writes what").

One rule for every object: its pose is the median of its measurements since it last moved (:class:`Steady`); between
measurements it is where they put it. Nothing here knows which object is held, touched or moving; how a held object
sits in the gripper is the task's, estimated the same way. Each object's noise, how far its measurements wander while
it stands untouched, is measured here for the checks and for telling a move from noise, never set.
"""

from __future__ import annotations

import numpy as np

from . import predicates
from .scene import SceneLog
from .timeline import BASE

STEADY_S = (
    1.0  # a choice: the measurements of a still thing its estimate is the median of (about ten on the rig)
)


class Steady:
    """An estimate of a pose that stands still until it moves: the median of its measurements since it last moved
    (their centres' median, their turns' mean), over the last STEADY_S. A move is two measurements in a row further
    than the threshold from the estimate; the measurements before it are dropped."""

    def __init__(self, pose: np.ndarray, centre: np.ndarray):
        self.estimate = np.asarray(pose, dtype=float).copy()
        self.centre = np.append(np.asarray(centre, dtype=float), 1.0)
        self.recent: list[tuple[float, np.ndarray]] = []
        self.beyond = 0

    def add(self, t: float, pose: np.ndarray, threshold: float) -> bool:
        """Take one measurement; whether it completed a move."""
        far = float(np.linalg.norm((pose @ self.centre)[:3] - (self.estimate @ self.centre)[:3])) > threshold
        self.beyond = self.beyond + 1 if far else 0
        moved = self.beyond >= 2
        if moved:
            self.recent, self.beyond = self.recent[-1:], 0
        self.recent = [(tv, m) for tv, m in self.recent if tv >= t - STEADY_S] + [
            (t, np.asarray(pose, dtype=float))
        ]
        self.estimate = self._median()
        return moved

    def _median(self) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        c = self.centre[:3]
        ms = np.stack([m for _, m in self.recent])
        p = np.median(ms[:, :3, :3] @ c + ms[:, :3, 3], axis=0)
        rot = Rotation.from_matrix(ms[:, :3, :3]).mean().as_matrix()
        out = np.eye(4)
        out[:3, :3], out[:3, 3] = rot, p - rot @ c
        return out


class World:
    """Each named object's pose and its measured noise. ``start`` is each one's motion since the demo as the act found
    it; ``centres`` its centre at its demo pose (base frame), where its position is measured; ``floor_m`` the least a
    measurement can be trusted to (the depth image's own step)."""

    def __init__(
        self,
        scene: SceneLog,
        start: dict[str, np.ndarray],
        centres: dict[str, np.ndarray],
        floor_m: float = 0.001,
    ):
        self.scene = scene
        self.centres = {n: np.asarray(centres.get(n, np.zeros(3)), dtype=float) for n in start}
        self.steady = {n: Steady(m, self.centres[n]) for n, m in start.items()}
        self.t = dict.fromkeys(self.steady, -np.inf)  # when its newest measurement was taken
        self.moves: dict[
            str, tuple[float, np.ndarray]
        ] = {}  # each object's last move: when, and its pose before
        self.floor_m = floor_m
        self.rest = {n: [-np.inf, np.inf] for n in self.pose}  # [since, until): what counts toward its noise
        self._noise_at_touch: dict[str, tuple[float, float] | None] = {}

    @property
    def pose(self) -> dict[str, np.ndarray]:
        return {n: e.estimate for n, e in self.steady.items()}

    def update(self) -> None:
        """The world step: every object takes its new measurements."""
        for n, e in self.steady.items():
            for m in self.scene.measurements(n, since=self.t[n] + 1e-9):
                before = e.estimate
                if e.add(m.t, m.motion, self.move_threshold(n)):
                    self.moves[n] = (m.t, before)
                self.t[n] = m.t

    def move_threshold(self, name: str) -> float:
        """How far a measurement must be from an object's pose to say it moved: twice its noise (the two each wander
        that much); infinite while the noise is unmeasured."""
        noise = self.noise(name)
        return np.inf if noise is None else 2.0 * noise[0]

    def motion(self, name: str) -> np.ndarray:
        return np.eye(4) if name == BASE or name not in self.steady else self.steady[name].estimate

    def centre(self, name: str, motion: np.ndarray) -> np.ndarray:
        """Where ``name``'s centre is with ``motion``, base frame."""
        return (motion @ np.append(self.centres.get(name, np.zeros(3)), 1.0))[:3]

    def touched(self, name: str, t: float) -> None:
        """From ``t`` the act acts on the object: its measurements no longer count as noise. Its noise is taken now,
        while the ones of it standing untouched are still in the log."""
        if name in self.rest and name not in self._noise_at_touch:
            self.rest[name][1] = min(self.rest[name][1], t)
            self._noise_at_touch[name] = self._noise(name)

    def resting(self, name: str, t: float) -> None:
        """From ``t`` it stands untouched again, somewhere new: its noise is measured afresh."""
        if name in self.rest:
            self.rest[name] = [t, np.inf]
            self._noise_at_touch.pop(name, None)

    def noise(self, name: str) -> tuple[float, float] | None:
        """How far its measurements wandered while it stood untouched: metres at its centre and degrees, or None with
        too few; never under ``floor_m``."""
        return self._noise_at_touch[name] if name in self._noise_at_touch else self._noise(name)

    def _noise(self, name: str) -> tuple[float, float] | None:
        since, until = self.rest.get(name, (-np.inf, np.inf))
        ms = [m for m in self.scene.measurements(name) if since <= m.t < until]
        if not ms:
            return None
        m = predicates.spread(np.array([self.centre(name, x.motion) for x in ms]))
        d = predicates.spread_deg(np.array([x.motion[:3, :3] for x in ms]))
        if m is None or d is None:
            return None
        return max(m, self.floor_m), d
