# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The demo as the engine plays it: legs between key frames, the demo's clock, and the Milestones the demo shows
(act_execution.md, "Legs", "The demo's clock", "Milestones").

Everything here is read off the demo once, before the act moves; nothing depends on where things are now.
"""

from __future__ import annotations

import bisect
import dataclasses
from typing import Any

import numpy as np

from lerobot.showservo import motion as core

BASE = core.WORLD_FRAME  # the arm's base: what a key frame held in no object's frame is relative to
START = core.START_FRAME  # where the leg begins: the demo's motion since its key frame before, from there


@dataclasses.dataclass(frozen=True)
class Demo:
    """What the engine takes of a recorded demo: its clock, the fingertip's pose (base frame), the gripper's command
    and reading, and the arm's joints; and ``grips``, per object it held, how the object sat in the gripper at each
    sample, as the demo's own perception had it (engine.demo_beliefs). Without them, the object sat as it lay at the
    grasp, all the way."""

    t: np.ndarray
    tips: np.ndarray
    grip: np.ndarray
    grip_obs: np.ndarray
    q: np.ndarray
    grips: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    def grip_of(self, obj: str, td: float) -> np.ndarray | None:
        """How ``obj`` sat in the demo's gripper at demo time ``td`` (its motion since the demo, in the fingertip's
        frame); None without the demo's perception of it."""
        grips = self.grips.get(obj)
        if grips is None:
            return None
        k, u = self._bracket(td)
        return core.interp_rigid(grips[k], grips[k + 1], u)

    def _bracket(self, td: float) -> tuple[int, float]:
        k = int(np.clip(np.searchsorted(self.t, td, side="right") - 1, 0, len(self.t) - 2))
        span = float(self.t[k + 1] - self.t[k])
        return k, 0.0 if span <= 0 else float(np.clip((td - self.t[k]) / span, 0.0, 1.0))

    def pose_at(self, td: float) -> np.ndarray:
        k, u = self._bracket(td)
        return core.interp_rigid(self.tips[k], self.tips[k + 1], u)

    def grip_at(self, td: float) -> float:
        return float(np.interp(td, self.t, self.grip))

    def q_at(self, td: float) -> np.ndarray:
        k, u = self._bracket(td)
        return (1.0 - u) * self.q[k] + u * self.q[k + 1]

    def index(self, td: float) -> int:
        return int(np.argmin(np.abs(self.t - td)))


@dataclasses.dataclass(frozen=True)
class Leg:
    """The demo between key frame ``n - 1`` (or, for the first, wherever the arm is) and key frame ``n``."""

    n: int
    label: str
    move: str  # "line", "follow" or "lift"
    start_obj: str | None  # what the key frame it starts at is held in; None for the first leg
    obj: str  # what its key frame is held in: an object, or BASE
    t0: float  # demo time it starts at (its key frame's for the first leg, which has no span)
    t1: float  # demo time of its key frame
    grip: str  # a line's gripper: "first" (before the arm) or "along" (with it)
    constraint: str  # "pose", "orientation" or "best_effort"
    up_m: float  # a lift's height
    held: str | None  # the object the demo held as the leg began

    @property
    def span(self) -> float:
        return self.t1 - self.t0

    @property
    def same_object(self) -> bool:
        return self.start_obj == self.obj and self.obj != BASE


@dataclasses.dataclass(frozen=True)
class Event:
    """Where the demo shows an object start or stop moving with something: a grasp (it moves with the gripper from
    here) or a release (it stays with ``ref`` from here)."""

    kind: str  # "grasp" or "release"
    obj: str
    i: int  # the demo sample
    leg: int
    s: float
    ref: str  # what it moves with from here: "gripper", an object, or BASE
    grasp_i: int  # the grasp this release ends, the grasp itself for a grasp


class Timeline:
    """Legs, the gripper's turning points and the Milestones' events of one demo. ``session_object`` names what key
    frames held in "" (the object taught before the demo) are relative to."""

    def __init__(self, demo: Demo, key_frames: list[dict[str, Any]], session_object: str | None = None):
        assert key_frames, "a key frame"
        self.demo = demo
        self.frames = key_frames

        def resolve(frame: str) -> str:
            return frame if frame in (BASE, START) else (frame or session_object or "")

        self.holdings = core.demo_holdings(demo.t, demo.grip, demo.grip_obs, key_frames)
        legs = []
        for n, k in enumerate(key_frames):
            held = next(
                (resolve(h["frame"]) for h in self.holdings if core.held_at(h, demo.t, key_frames, n)), None
            )
            legs.append(
                Leg(
                    n=n,
                    label=k["label"],
                    move=k["move"],
                    start_obj=None if n == 0 else resolve(key_frames[n - 1]["frame"]),
                    obj=resolve(k["frame"]),
                    t0=float(key_frames[n - 1]["t"]) if n else float(k["t"]),
                    t1=float(k["t"]),
                    grip=k.get("grip", "first"),
                    constraint=k.get("constraint", "pose"),
                    up_m=float(k.get("up_mm") or 0.0) / 1000.0,
                    held=held,
                )
            )
        self.legs = legs
        self.events = self._events(resolve)
        cmd = np.asarray(demo.grip, dtype=float)
        way = np.array([0 if abs(d) <= 1e-6 else int(np.sign(d)) for d in np.diff(cmd)])
        self.turning_times = np.array([float(demo.t[i]) for i in range(1, len(way)) if way[i] != way[i - 1]])

    def _events(self, resolve) -> list[Event]:
        out = []
        for h in self.holdings:
            obj = resolve(h["frame"])
            leg, s = self.position(h["grip"])
            out.append(Event("grasp", obj, h["grip"], leg, s, "gripper", h["grip"]))
            if h["release"] is not None:
                leg_r, s_r = self.position(h["release"])
                ref = self.legs[leg_r].obj
                out.append(
                    Event("release", obj, h["release"], leg_r, s_r, BASE if ref == START else ref, h["grip"])
                )
        return sorted(out, key=lambda e: (e.leg, e.s))

    def position(self, i: int) -> tuple[int, float]:
        """The leg and the fraction of it at demo sample ``i``."""
        td = float(self.demo.t[i])
        ends = [leg.t1 for leg in self.legs]
        n = min(bisect.bisect_left(ends, td), len(self.legs) - 1)
        leg = self.legs[n]
        return n, 1.0 if leg.span <= 0 else float(np.clip((td - leg.t0) / leg.span, 0.0, 1.0))

    def time(self, leg: int, s: float) -> float:
        """The demo time at fraction ``s`` of leg ``leg``."""
        return self.legs[leg].t0 + s * self.legs[leg].span

    def turning_near(self, td: float, half_window: float) -> float | None:
        """The gripper turning point within ``half_window`` of demo time ``td`` (on the near side, so each point falls
        to exactly one tick), or None."""
        if not len(self.turning_times):
            return None
        k = int(np.argmin(np.abs(self.turning_times - td)))
        t_star = float(self.turning_times[k])
        return t_star if td - half_window < t_star <= td + half_window else None

    def straight_share(self, leg: int, lin_m_s: float, ang_rad_s: float) -> float:
        """How much shorter the straight line between a leg's two key frames is than the path the demo took between
        them, in walk time (translation at ``lin_m_s``, turn at ``ang_rad_s``, whichever takes longer per step): 1
        where the demo went straight or did not move. A line takes the demo's time between its key frames times this.
        """
        lg = self.legs[leg]
        if lg.n == 0 or lg.span <= 0:
            return 1.0
        i0, i1 = self.demo.index(lg.t0), self.demo.index(lg.t1)
        if i1 <= i0:
            return 1.0
        lin, ang = lin_m_s, ang_rad_s
        path = float(core._walk_seconds(self.demo.tips, i0, i1, lin, ang).sum())
        if path <= 0.0:
            return 1.0
        dist, deg = core.pose_residual(self.demo.tips[i0], self.demo.tips[i1])
        straight = max(dist / lin, np.radians(deg) / ang)
        return float(min(1.0, straight / path))
