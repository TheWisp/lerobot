# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The act engine's run loop (act_execution.md, "The two loops"): commands made from the demo's clock a few ticks
ahead, Milestones checked on every camera frame, and Progress moved back or ahead when a check fails or an object
moves.

``Engine.step(now)`` is called often, at least once per motor tick: it reads what the motor loop has sent, checks the
open Milestones against the camera frames that arrived, and tops the queue up. It never waits; the arm stands still
only when the queue runs out on purpose (the act ended, a final Milestone, an object to be seen again).
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from lerobot.showservo import motion as core

from . import predicates as pred
from .objects import Objects
from .predicates import Answer, Verdict
from .scene import SceneLog, Tracks
from .timeline import BASE, Event, Timeline

FAILURE_FRAMES = 2  # a choice: the fewest camera frames in a row that ignore a single bad one
TRIES = 2  # act_execution.md Q1's leaning: a Milestone may fail once and be tried again
SLOPE_MAX = 3.0  # a cubic from 0 to 1 stays monotonic with end slopes up to 3


@dataclasses.dataclass
class Arm:
    """What the engine needs of an arm (act_execution.md O10). ``kin`` maps joints in degrees, the gripper's column
    included: ``forward_kinematics(q)`` and ``inverse_kinematics(seed, pose)``. ``step_deg`` is the most an arm joint
    may move in one tick (the motor loop's own bound for live commands); ``walk_m_s`` and ``walk_rad_s`` the walk
    speed the operator set."""

    kin: Any
    gripper: int
    closing: float = 1.0
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    step_deg: float = 3.0
    workspace: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    hz: float = 30.0
    walk_m_s: float = 0.04
    walk_rad_s: float = math.radians(30.0)
    grip_units_s: float = 80.0
    lag_deg: float | None = (
        None  # the motor loop's catch-up budget: a joint further behind than this is waited for
    )

    def fk(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(self.kin.forward_kinematics(np.asarray(q, dtype=float)), dtype=float)

    def solve(
        self, seed: np.ndarray, pose: np.ndarray, grip: float, hint: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        out = core.solve_plan_joints(
            self.kin, pose[None], np.array([grip]), hint[None], seed, self.gripper, self.lo, self.hi
        )
        return out["q"][0], float(out["residual_m"][0]), float(out["residual_deg"][0])

    def joint_step(self, a: np.ndarray, b: np.ndarray) -> float:
        d = np.abs(np.asarray(b, dtype=float) - np.asarray(a, dtype=float))
        d[self.gripper] = 0.0
        return float(d.max())

    def walk_seconds(self, a: np.ndarray, b: np.ndarray) -> float:
        """How long the walk takes from pose ``a`` to ``b``: its translation or its turn, whichever is longer."""
        m, deg = core.pose_residual(a, b)
        return max(m / self.walk_m_s, math.radians(deg) / self.walk_rad_s)


class Playback(Protocol):
    """The motor loop's queue: index 0 is the command the act started from."""

    def sent(self) -> int: ...

    def clock(self) -> float: ...

    def append(self, expected_tail: int, samples: list[np.ndarray], due_s: list[float]) -> bool: ...

    def cancel(self, expected_sent: int) -> bool: ...

    def observed(self) -> np.ndarray | None:
        """The arm's newest checked joint reading, or None without one."""


@dataclasses.dataclass
class Cmd:
    q: np.ndarray
    pose: np.ndarray
    grip: float
    due: float
    leg: int
    s: float
    kind: str  # "start", "leg" or "transition"
    td: float | None = None
    state: tuple[Run, float] | None = (
        None  # the generator's run and time once it was made: where a cancel rewinds to
    )


@dataclasses.dataclass
class Run:
    """A leg, or the move to a state Progress was placed at, as the generator plays it. Times are demo seconds."""

    kind: str  # "leg" or "transition"
    leg: int
    s0: float
    c0: np.ndarray  # the pose it starts from, frozen
    g0: float
    g1: float
    seconds: float  # the arm's motion
    grip_before: float  # the gripper moving with the arm standing, before the motion
    grip_after: float  # ... after it
    m0: float = 0.0
    m1: float = 0.0
    blend: bool = False
    d0: np.ndarray | None = None  # a follow's demo pose where it starts

    @property
    def total(self) -> float:
        return self.grip_before + self.seconds + self.grip_after


@dataclasses.dataclass
class Milestone:
    event: Event
    final: bool = False
    allow_m: float = 0.0
    allow_deg: float = 0.0
    status: str = "ahead"  # ahead, checking, success, failure
    failures: int = 0
    t: float | None = None
    seen: float = -np.inf
    streak: int = 0
    closed: bool = False
    ref_tracks: Tracks | None = None
    data: dict[str, Any] = dataclasses.field(default_factory=dict)
    last: Verdict | None = None

    @property
    def name(self) -> str:
        return f"{self.event.kind} of {self.event.obj}"


def _hermite(u: float, m0: float, m1: float) -> float:
    u2, u3 = u * u, u * u * u
    return (-2 * u3 + 3 * u2) + m0 * (u3 - 2 * u2 + u) + m1 * (u3 - u2)


def _smooth(u: float) -> float:
    return u * u * (3.0 - 2.0 * u)


class Engine:
    """One act. ``objects`` holds the act's objects' poses and noise; ``perturb``, when given, may change a command on
    its way to the motor loop (tests, and fault injection on the rig), never what the engine believes it sent."""

    def __init__(
        self,
        timeline: Timeline,
        arm: Arm,
        scene: SceneLog,
        playback: Playback,
        objects: Objects,
        speed: float = 1.0,
        settings: dict[str, Any] | None = None,
        perturb: Any | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        assert speed > 0.0, "a positive speed"
        self.tl, self.arm, self.scene, self.pb, self.objects = timeline, arm, scene, playback, objects
        self.demo = timeline.demo
        self.speed = speed
        self.settings = dict(settings or {})
        marks = self.settings.get("milestones", {})
        self.milestones = [
            Milestone(
                e,
                final=bool(marks.get(f"{e.kind}:{e.obj}", {}).get("final", False)),
                allow_m=float(marks.get(f"{e.kind}:{e.obj}", {}).get("allow_mm", 0.0)) / 1000.0,
                allow_deg=float(marks.get(f"{e.kind}:{e.obj}", {}).get("allow_deg", 0.0)),
            )
            for e in timeline.events
        ]
        self.wait_view_s = float(
            self.settings.get("wait_view_s", 5.0)
        )  # how long to wait to see a dropped object
        self.perturb = perturb
        self._clock = clock
        self.cmds: list[Cmd] = []
        self.sent = -1
        self.run: Run | None = None
        self.tau = 0.0
        self.progress: tuple[int, float] = (0, 0.0)
        self.hold: dict[
            str, np.ndarray
        ] = {}  # how each held object sits in the gripper, measured at its grasp
        self.fix: dict[str, np.ndarray] = {}  # the gripper-side correction that follows from it
        self.records: list[dict[str, Any]] = []
        self.solve_s = 0.0  # the slowest solve so far
        self.step_gaps: list[float] = []  # the time between the engine's own steps, newest last
        self.waiting: dict[str, Any] | None = None
        self.moved_views: dict[str, int] = {}
        self.ended: float | None = None
        self.end_frames = 0
        self.done = False
        self.result: dict[str, Any] | None = None
        self._last_now: float | None = None
        self._over = False  # the generator has made the act's last command
        self.waiting_arm: float | None = None  # since when the arm trails by more than its catch-up budget
        self.waited_s = 0.0
        self._unreachable = ""
        self.stretched_s = 0.0  # demo seconds the joint bound slowed

    # ── records ──

    def _record(self, what: str, now: float, **info: Any) -> None:
        self.records.append({"t": now, "what": what, "progress": list(self.progress), **info})

    def _finish(self, ok: bool, why: str, now: float) -> None:
        self.done = True
        self.result = {
            "ok": ok,
            "why": why,
            "milestones": [
                {
                    "name": m.name,
                    "status": m.status,
                    "failures": m.failures,
                    "last": None if m.last is None else m.last.why,
                }
                for m in self.milestones
            ],
            "commands": len(self.cmds),
            "stretched_s": self.stretched_s,
            "waited_for_arm_s": self.waited_s,
        }
        self._record("end", now, ok=ok, why=why)

    # ── composition ──

    def _compose(self, leg_n: int, d: np.ndarray) -> np.ndarray:
        leg = self.tl.legs[leg_n]
        out = self.objects.motion(leg.obj) @ d
        if leg.held is not None and leg.held in self.fix:
            out = out @ self.fix[leg.held]
        return out

    def _state_pose(self, leg_n: int, s: float) -> np.ndarray:
        return self._compose(leg_n, self.demo.pose_at(self.tl.time(leg_n, s)))

    def _line_end(self, leg_n: int, c0: np.ndarray) -> np.ndarray:
        leg = self.tl.legs[leg_n]
        if leg.move == "lift":
            end = c0.copy()
            end[2, 3] += leg.up_m
            return end
        end = self._compose(leg_n, self.demo.pose_at(leg.t1))
        if leg.constraint == "orientation":
            end[:3, 3] = c0[:3, 3]
        return end

    def _demo_speed(self, td: float) -> float:
        """The demo's walk speed at demo time ``td``: walk seconds per demo second over the sample there."""
        k = int(np.clip(np.searchsorted(self.demo.t, td, side="right") - 1, 0, len(self.demo.t) - 2))
        dt = float(self.demo.t[k + 1] - self.demo.t[k])
        return 0.0 if dt <= 0 else self.arm.walk_seconds(self.demo.tips[k], self.demo.tips[k + 1]) / dt

    def _slope(self, speed: float, seconds: float, length: float) -> float:
        if length <= 1e-9 or seconds <= 1e-9:
            return 0.0
        return float(np.clip(speed * seconds / length, 0.0, SLOPE_MAX))

    def _arriving_speed(self) -> float:
        """The arm's commanded walk speed now, per demo second: from the last two commands."""
        if len(self.cmds) < 2:
            return 0.0
        a, b = self.cmds[-2], self.cmds[-1]
        return self.arm.walk_seconds(a.pose, b.pose) / (self.speed / self.arm.hz)

    def _leaving_speed(self, leg_n: int) -> float:
        """How fast the arm should be going as leg ``leg_n`` ends, per demo second: the next leg's start."""
        nxt = leg_n + 1
        if nxt >= len(self.tl.legs):
            return 0.0
        leg = self.tl.legs[nxt]
        if leg.move == "follow":
            return self._demo_speed(leg.t0)
        g_from = self.demo.grip_at(leg.t0)
        if leg.grip == "first" and leg.held is None and abs(self.demo.grip_at(leg.t1) - g_from) > 1e-6:
            return 0.0
        span = leg.span * self.tl.straight_share(nxt, self.arm.walk_m_s, self.arm.walk_rad_s)
        if span <= 0:
            return 0.0
        return self.arm.walk_seconds(self.demo.pose_at(leg.t0), self.demo.pose_at(leg.t1)) / span

    def _leg_run(self, leg_n: int, s0: float, c0: np.ndarray, g0: float) -> Run:
        leg = self.tl.legs[leg_n]
        td0 = self.tl.time(leg_n, s0)
        if leg.move == "follow":
            d0 = self.demo.pose_at(td0)
            m, deg = core.pose_residual(c0, self._compose(leg_n, d0))
            close = m <= core.ACT_SOLVE_TOL_M and deg <= core.ACT_SOLVE_TOL_DEG
            return Run(
                "leg",
                leg_n,
                s0,
                c0,
                g0,
                self.demo.grip_at(leg.t1),
                leg.t1 - td0,
                0.0,
                0.0,
                blend=not (leg.same_object and close),
                d0=d0,
            )
        end = self._line_end(leg_n, c0)
        length = self.arm.walk_seconds(c0, end)
        if leg.n == 0 or leg.span <= 0:
            seconds = length  # no key frame before: the walk speed
        else:
            seconds = (
                leg.span * (1.0 - s0) * self.tl.straight_share(leg_n, self.arm.walk_m_s, self.arm.walk_rad_s)
            )
        held = leg.held is not None or leg.move == "lift"
        g1 = g0 if held else self.demo.grip_at(leg.t1)
        grip_before = abs(g1 - g0) / self.arm.grip_units_s if (leg.grip == "first" and not held) else 0.0
        standing = grip_before > 0 or len(self.cmds) < 2
        m0 = 0.0 if standing else self._slope(self._arriving_speed(), seconds, length)
        m1 = self._slope(self._leaving_speed(leg_n), seconds, length)
        return Run("leg", leg_n, s0, c0, g0, g1, seconds, grip_before, 0.0, m0=m0, m1=m1)

    def _transition(self, target: tuple[int, float], c0: np.ndarray, g0: float) -> Run:
        leg_n, s = target
        leg = self.tl.legs[leg_n]
        end = self._state_pose(leg_n, s)
        g1 = (
            g0 if leg.held is not None and leg.held in self.fix else self.demo.grip_at(self.tl.time(leg_n, s))
        )
        grip_s = abs(g1 - g0) / self.arm.grip_units_s
        opening = (g1 - g0) * self.arm.closing < 0
        length = self.arm.walk_seconds(c0, end)
        speed_in = self._demo_speed(self.tl.time(leg_n, s)) if leg.move == "follow" else 0.0
        return Run(
            "transition",
            leg_n,
            s,
            c0,
            g0,
            g1,
            length,
            grip_s if opening else 0.0,
            0.0 if opening else grip_s,
            m0=0.0,
            m1=self._slope(speed_in, length, length),
        )

    def _sample(self, run: Run, tau: float) -> tuple[np.ndarray, float, float, float | None]:
        """The pose, gripper, leg fraction and demo time at ``tau`` demo seconds into ``run``."""
        leg = self.tl.legs[run.leg]
        tau = min(max(tau, 0.0), run.total)
        motion_u = 0.0 if run.seconds <= 0 else min(1.0, max(0.0, (tau - run.grip_before) / run.seconds))
        if tau >= run.grip_before + run.seconds:
            motion_u = 1.0
        grip_u = 1.0
        if run.grip_before > 0:
            grip_u = min(1.0, tau / run.grip_before)
        elif run.grip_after > 0:
            grip_u = max(0.0, (tau - run.seconds) / run.grip_after)
        if run.kind == "transition":
            end = self._state_pose(run.leg, run.s0)
            pose = core.interp_rigid(run.c0, end, _hermite(motion_u, run.m0, run.m1))
            return pose, run.g0 + (run.g1 - run.g0) * grip_u, run.s0, self.tl.time(run.leg, run.s0)
        if run.kind == "retrace":
            s = run.s0 * (1.0 - motion_u)
            td = self.tl.time(run.leg, s)
            pose = run.d0 @ self.demo.pose_at(td)
            if leg.held is not None and leg.held in self.fix:
                pose = pose @ self.fix[leg.held]
            return pose, run.g0, s, td
        s = run.s0 + motion_u * (1.0 - run.s0)
        td = self.tl.time(run.leg, s)
        if leg.move == "follow":
            d = self.demo.pose_at(td)
            end = self._compose(run.leg, d)
            if run.blend:
                start = run.c0 @ np.linalg.inv(run.d0) @ d
                end = core.interp_rigid(start, end, _smooth(motion_u))
            if leg.constraint == "orientation":
                end[:3, 3] = run.c0[:3, 3]
            grip = self.demo.grip_at(td)
            t_star = self.tl.turning_near(td, 0.5 * self.speed / self.arm.hz)
            if t_star is not None:
                grip = self.demo.grip_at(t_star)
            return end, grip, s, td
        sigma = _hermite(motion_u, run.m0, run.m1)
        pose = core.interp_rigid(run.c0, self._line_end(run.leg, run.c0), sigma)
        if leg.grip == "along" and run.grip_before == 0:
            grip = run.g0 + (run.g1 - run.g0) * motion_u
        else:
            grip = run.g0 + (run.g1 - run.g0) * grip_u
        return pose, grip, s, td

    def _next_run(self, run: Run) -> Run | None:
        pose, grip, _, _ = self._sample(run, run.total)
        if run.kind == "transition":
            return self._leg_run(run.leg, run.s0, pose, grip)
        if run.kind == "retrace":
            return self._transition((run.leg, 0.0), pose, grip)
        if run.leg + 1 >= len(self.tl.legs):
            return None
        return self._leg_run(run.leg + 1, 0.0, pose, grip)

    def _advance(self, run: Run, tau: float) -> tuple[Run | None, float]:
        while tau > run.total + 1e-12:
            nxt = self._next_run(run)
            if nxt is None:
                return None, 0.0
            tau -= run.total
            run = nxt
        return run, tau

    # ── the command stage ──

    def _lead(self) -> int:
        """How many commands to keep queued: enough to cover the slowest solve and the time until the engine's next
        step (the 95th percentile of the gaps between its recent steps), plus one."""
        gap = float(np.percentile(self.step_gaps, 95)) if self.step_gaps else 1.0 / self.arm.hz
        return max(2, int(math.ceil((self.solve_s + gap) * self.arm.hz)) + 1)

    def _next(self, prev: Cmd) -> Cmd | None:
        """The next tick's command after ``prev``, or None when the act's motion is over. Advances the timeline by one
        tick times the speed, or less where a joint would move faster than its bound."""
        dtau = self.speed / self.arm.hz
        for _ in range(6):
            run, tau = self._advance(self.run, self.tau + dtau)
            if run is None:
                self._over = True
                return None
            pose, grip, s, td = self._sample(run, tau)
            hint = np.zeros_like(prev.q)
            if (
                run.kind == "leg"
                and self.tl.legs[run.leg].move == "follow"
                and prev.td is not None
                and td is not None
            ):
                hint = self.demo.q_at(td) - self.demo.q_at(prev.td)
                hint[self.arm.gripper] = 0.0
            t0 = self._clock()
            q, err_m, err_deg = self.arm.solve(prev.q, pose, grip, hint)
            self.solve_s = max(self.solve_s, self._clock() - t0)
            if self.arm.joint_step(prev.q, q) <= self.arm.step_deg + 1e-9:
                if dtau < self.speed / self.arm.hz:
                    self.stretched_s += self.speed / self.arm.hz - dtau
                self.run, self.tau = run, tau
                leg = self.tl.legs[run.leg]
                state = (run, tau)
                if (
                    err_m > core.ACT_REACH_TOL_M or err_deg > core.ACT_REACH_TOL_DEG
                ) and leg.constraint != "best_effort":
                    self._unreachable = (
                        f"{leg.label}: the arm reaches only within {err_m * 1000:.1f} mm, {err_deg:.1f} deg"
                    )
                if self.arm.workspace is not None:
                    lo, hi = (np.asarray(b, dtype=float) for b in self.arm.workspace)
                    if np.any(pose[:3, 3] < lo) or np.any(pose[:3, 3] > hi):
                        self._unreachable = f"{leg.label}: outside the workspace"
                return Cmd(q, pose, grip, 0.0, run.leg, s, run.kind, td, state)
            dtau /= 2.0
        self._unreachable = (
            f"{self.tl.legs[self.run.leg].label}: the arm would have to jump to another configuration"
        )
        return None

    def _fill(self, now: float) -> None:
        self._unreachable = ""
        new: list[Cmd] = []
        while len(self.cmds) + len(new) - 1 - self.sent < self._lead():
            c = self._next(new[-1] if new else self.cmds[-1])
            if c is None or self._unreachable:
                break
            new.append(c)
        if new:
            due = self.cmds[-1].due
            clock = self.pb.clock()
            for c in new:
                due = max(due + 1.0 / self.arm.hz, clock)
                c.due = due
            samples = [self._outgoing(c, now) for c in new]
            if not self.pb.append(len(self.cmds) - 1, samples, [c.due for c in new]):
                self._finish(False, "the motor loop's queue changed under the act", now)
                return
            self.cmds.extend(new)
        if self._unreachable:
            self._finish(False, self._unreachable, now)

    def _outgoing(self, c: Cmd, now: float) -> np.ndarray:
        q = c.q.copy()
        if self.perturb is not None:
            q = self.perturb.command(q, c, self, now)
        return q

    # ── what the motor loop sent ──

    def _on_sent(self, now: float) -> None:
        sent = self.pb.sent()
        for k in range(self.sent + 1, sent + 1):
            c = self.cmds[k]
            if c.kind == "leg":
                self.progress = (c.leg, c.s)
                for m in self.milestones:
                    if m.status == "ahead" and (m.event.leg, m.event.s) <= self.progress:
                        self._begin(m, now)
        self.sent = max(self.sent, sent)

    def _begin(self, m: Milestone, now: float) -> None:
        e = m.event
        tip = self.scene.tip_at(now)
        if tip is None:
            tip = self.cmds[self.sent].pose if self.sent >= 0 else self.cmds[0].pose
        m.status, m.t, m.seen, m.streak, m.closed, m.ref_tracks = "checking", now, now, 0, False, None
        if e.kind == "grasp":
            motion = self.objects.motion(e.obj)
            self.hold[e.obj] = np.linalg.inv(tip) @ motion
            self.fix[e.obj] = np.linalg.inv(self.demo.tips[e.i]) @ np.linalg.inv(motion) @ tip
            self.objects.touched(e.obj, now)
            m.data = {"tip": tip, "motion": motion}
        else:
            for g in self.milestones:
                if g.event.kind == "grasp" and g.event.obj == e.obj and g.event.i == e.grasp_i:
                    g.closed = True
            hold = self.hold.get(e.obj)
            ref = self.objects.motion(e.ref)
            put = tip @ hold if hold is not None else None
            m.data = {"tip": tip, "put": put, "ref_motion": ref}
            if put is not None:
                m.data["rel_put"] = np.linalg.inv(ref) @ put
                rel_demo = self.demo.tips[e.i] @ np.linalg.inv(self.demo.tips[e.grasp_i])
                # Both poses come from the arm, put through the reach tolerance; an object's own views add theirs.
                noise = (core.ACT_REACH_TOL_M, core.ACT_REACH_TOL_DEG)
                ref_noise = self.objects.view_noise(e.ref) if e.ref != BASE else (0.0, 0.0)
                noise = None if ref_noise is None else (noise[0] + ref_noise[0], noise[1] + ref_noise[1])
                if noise is not None:
                    m.data["placement"] = pred.related(
                        put,
                        ref,
                        rel_demo,
                        self.objects.centres.get(e.obj, np.zeros(3)),
                        noise[0],
                        noise[1],
                        m.allow_m,
                        m.allow_deg,
                    )
            self.fix.pop(e.obj, None)
        self._record("milestone begins", now, milestone=m.name)

    # ── checks ──

    def _frame_times(self, m: Milestone) -> list[float]:
        times = {f.t for f in self.scene.tracks(m.event.obj, since=m.seen + 1e-9)}
        times |= {d.t for d in self.scene.depths(since=m.seen + 1e-9)}
        return sorted(times)

    def _project_shift(self, centre: np.ndarray, motion: np.ndarray) -> np.ndarray:
        uv, _ = self.scene.camera.project(np.stack([centre, (motion @ np.append(centre, 1.0))[:3]]))
        return uv[1] - uv[0]

    def _evidence(self, m: Milestone, t: float) -> Verdict:
        e, obj = m.event, m.event.obj
        tip_t = self.scene.tip_at(t)
        if tip_t is None:
            return Verdict(Answer.UNKNOWN, "no arm reading at that frame")
        tracks = {f.t: f for f in self.scene.tracks(obj, since=m.t)}
        if m.ref_tracks is None and tracks:
            m.ref_tracks = tracks[min(tracks)]
        tip_ref = None if m.ref_tracks is None else self.scene.tip_at(m.ref_tracks.t)
        c_demo = self.objects.centres.get(obj, np.zeros(3))
        surface = self.objects.surfaces.get(obj)
        cam = self.scene.camera
        out: list[Verdict] = []
        view_noise = self.objects.view_noise(obj)
        depth_noise = self.objects.depth_noise(obj, cam) if surface is not None else None
        depth = next((d for d in self.scene.depths(since=t - 1e-9) if d.t == t), None)
        if e.kind == "grasp":
            where = m.data["motion"]  # where it lay
            c = (where @ np.append(c_demo, 1.0))[:3]
            if tip_ref is not None and t in tracks and cam is not None:
                carry = tip_t @ np.linalg.inv(tip_ref)
                c_ref = (tip_ref @ np.linalg.inv(m.data["tip"]) @ np.append(c, 1.0))[:3]  # had it come along
                noise_px = self.objects.track_noise(obj)
                if noise_px is not None:
                    out.append(
                        pred.moves_with(
                            m.ref_tracks, tracks[t], self._project_shift(c_ref, carry), np.zeros(2), noise_px
                        )
                    )
            carry_event = tip_t @ np.linalg.inv(m.data["tip"])
            apart = float(np.linalg.norm((carry_event @ np.append(c, 1.0))[:3] - c))
            if (
                depth is not None
                and depth_noise is not None
                and surface is not None
                and view_noise is not None
            ):
                extent = float(np.max(np.linalg.norm(surface - c_demo, axis=1)))
                if apart > 2.0 * extent + view_noise[0]:
                    still = pred.sits_at(surface, where, depth, cam, depth_noise, lateral_m=view_noise[0])
                    if still.yes:
                        out.append(Verdict(Answer.NO, f"still where it lay: {still.why}", still.evidence))
            views = [v for v in self.scene.views(obj, since=m.t + 1e-9) if v.t <= t]
            if views and view_noise is not None:
                seen = (views[-1].motion @ np.append(c_demo, 1.0))[:3]
                out.append(
                    pred.decide(
                        seen * 1000,
                        (carry_event @ np.append(c, 1.0))[:3] * 1000,
                        c * 1000,
                        view_noise[0] * 1000,
                        "mm",
                    )
                )
            return pred.combine(out)
        put = m.data.get("put")
        if put is None:
            return Verdict(Answer.UNKNOWN, "how it sat in the gripper was never measured")
        ref_now = self.objects.motion(e.ref)
        stays = ref_now @ m.data["rel_put"]
        c_put = (put @ np.append(c_demo, 1.0))[:3]
        c_stays = (stays @ np.append(c_demo, 1.0))[:3]
        away = tip_t @ np.linalg.inv(m.data["tip"])
        c_away = (away @ np.append(c_put, 1.0))[:3]
        if tip_ref is not None and t in tracks and cam is not None:
            noise_px = self.objects.track_noise(obj)
            gone = tip_t @ np.linalg.inv(tip_ref)
            ref_put = self.scene.tip_at(m.ref_tracks.t) if m.ref_tracks is not None else None
            if noise_px is not None and ref_put is not None:
                c_ref = (np.linalg.inv(m.data["tip"]) @ np.append(c_put, 1.0))[:3]
                c_ref = (ref_put @ np.append(c_ref, 1.0))[
                    :3
                ]  # where the gripper had it at the reference frame
                shift_new = self._project_shift(c_put, stays @ np.linalg.inv(put))
                shift_old = self._project_shift(c_ref, gone)
                out.append(pred.moves_with(m.ref_tracks, tracks[t], shift_new, shift_old, noise_px))
        if "placement" in m.data and m.data["placement"].no:
            out.append(
                Verdict(
                    Answer.NO, f"put down off target: {m.data['placement'].why}", m.data["placement"].evidence
                )
            )
        if depth is not None and depth_noise is not None and surface is not None and view_noise is not None:
            extent = float(np.max(np.linalg.norm(surface - c_demo, axis=1)))
            if float(np.linalg.norm(c_away - c_stays)) > 2.0 * extent + view_noise[0]:
                out.append(
                    pred.sits_at(surface, stays, depth, cam, depth_noise, lateral_m=view_noise[0] + m.allow_m)
                )
        views = [v for v in self.scene.views(obj, since=m.t + 1e-9) if v.t <= t]
        if views and view_noise is not None:
            seen = (views[-1].motion @ np.append(c_demo, 1.0))[:3]
            out.append(pred.decide(seen * 1000, c_stays * 1000, c_away * 1000, view_noise[0] * 1000, "mm"))
        return pred.combine(out)

    def _check(self, now: float) -> None:
        for m in self.milestones:
            if m.status not in ("checking", "success") or m.closed:
                continue
            for t in self._frame_times(m):
                v = self._evidence(m, t)
                m.seen, m.last = t, v
                self._record("check", now, milestone=m.name, frame=t, answer=v.answer.value, why=v.why)
                if v.no:
                    m.streak += 1
                    if m.streak >= FAILURE_FRAMES:
                        self._failed(m, v, now)
                        return
                    continue
                m.streak = 0
                if v.yes and m.status == "checking":
                    m.status = "success"

    # ── going back, or ahead ──

    def _cancel(self) -> None:
        for _ in range(5):
            sent = self.pb.sent()
            if self.pb.cancel(sent):
                del self.cmds[sent + 1 :]
                self.sent = sent
                if self.cmds[sent].state is not None:
                    self.run, self.tau = self.cmds[sent].state
                self._over = False
                return
        raise RuntimeError("could not cancel the motor loop's queue")

    def _failed(self, m: Milestone, v: Verdict, now: float) -> None:
        m.status, m.failures = "failure", m.failures + 1
        self._record("failure", now, milestone=m.name, why=v.why, failures=m.failures)
        self._on_sent(now)
        self._cancel()
        if m.failures >= TRIES:
            self._finish(False, f"the {m.name} failed {m.failures} times: {v.why}", now)
            return
        e = m.event
        hi_i = e.i if e.kind == "grasp" else e.grasp_i
        self.waiting = {"obj": e.obj, "since": now, "hi": float(self.demo.t[hi_i]), "cause": m.name}

    def _lo_time(self, holding: str | None) -> float:
        lo = self.tl.legs[0].t0
        for m in self.milestones:
            t_e = float(self.demo.t[m.event.i])
            if m.final and m.status == "success" and (m.event.leg, m.event.s) <= self.progress:
                lo = max(lo, t_e)
            if (
                holding is not None
                and m.event.kind == "grasp"
                and m.event.obj == holding
                and m.status != "ahead"
            ):
                lo = max(lo, t_e)
        return lo

    def _nearest(
        self, lo_t: float, hi_t: float, here: np.ndarray, t_here: float
    ) -> tuple[tuple[int, float], float]:
        best, best_cost = None, np.inf
        for i in np.flatnonzero((self.demo.t >= lo_t) & (self.demo.t < hi_t)):
            leg_n, s = self.tl.position(int(i))
            if self.tl.legs[leg_n].held is not None and self.tl.legs[leg_n].held not in self.fix:
                continue  # a state with an object in the gripper, which it does not hold
            cost = self.arm.walk_seconds(here, self._state_pose(leg_n, s)) + abs(
                float(self.demo.t[i]) - t_here
            )
            if cost < best_cost:
                best, best_cost = (leg_n, s), cost
        return best, best_cost

    def _place(self, target: tuple[int, float], now: float, cause: str) -> None:
        last = self.cmds[self.sent]
        for m in self.milestones:
            if m.status != "ahead" and (m.event.leg, m.event.s) >= target:
                m.status, m.streak, m.closed, m.data, m.ref_tracks = "ahead", 0, False, {}, None
                if m.event.kind == "grasp":
                    self.fix.pop(m.event.obj, None)
                    self.hold.pop(m.event.obj, None)
        self.run, self.tau = self._transition(target, last.pose, last.grip), 0.0
        self._over, self.ended = False, None
        self._record("progress moved", now, to=list(target), cause=cause)

    def _resume(self, now: float) -> None:
        w = self.waiting
        if self.objects.latest_t[w["obj"]] <= w["since"]:
            if now - w["since"] > self.wait_view_s:
                self._finish(False, f"{w['obj']} was not seen again within {self.wait_view_s:g} s", now)
            return
        self.objects.snap(w["obj"])
        self.objects.resting(w["obj"], self.objects.latest_t[w["obj"]])
        last = self.cmds[self.sent]
        t_here = last.td if last.td is not None else self.tl.time(*self.progress)
        target, _ = self._nearest(self._lo_time(None), w["hi"], last.pose, t_here)
        self.waiting = None
        if target is None:
            self._finish(False, f"no point of the demo before the {w['cause']} to go back to", now)
            return
        self._place(target, now, f"the {w['cause']} failed")

    def _watch_moves(self, now: float) -> None:
        """An object the current leg heads for moved by more than its noise, on two views in a row (R4). During a
        follow — contact, or nearly — the arm backs out along the demo's path relative to where the object was, to the
        follow's start, and goes in again relative to where it is now. During a line the line re-aims from where the
        arm is and keeps its speed, so a target moved nearer is reached sooner."""
        run = self.run
        if run is None or run.kind != "leg":
            return
        leg = self.tl.legs[run.leg]
        obj = leg.obj
        if obj == BASE or obj not in self.objects.used or leg.move == "lift":
            return
        off, threshold = self.objects.off(obj), self.objects.move_threshold(obj)
        t_view = self.objects.latest_t[obj]
        if off <= threshold:
            self.moved_views[obj] = 0
            return
        if t_view <= self.moved_views.get(obj + ":t", -np.inf):
            return
        self.moved_views[obj + ":t"] = t_view
        self.moved_views[obj] = self.moved_views.get(obj, 0) + 1
        if self.moved_views[obj] < FAILURE_FRAMES:
            return
        self.moved_views[obj] = 0
        cause = f"{obj} moved {off * 1000:.0f} mm"
        if leg.move == "follow":
            if any(m.status != "ahead" and m.event.leg == run.leg for m in self.milestones):
                return  # the gripper has acted on it in this follow: the composition follows it
            before = self.objects.used[obj].copy()
            self.objects.snap(obj)
            self._on_sent(now)
            self._cancel()
            last = self.cmds[self.sent]
            s_here = last.s if last.leg == run.leg and last.kind == "leg" else run.s0
            seconds = s_here * leg.span
            self.run, self.tau = (
                Run(
                    "retrace", run.leg, s_here, last.pose, last.grip, last.grip, seconds, 0.0, 0.0, d0=before
                ),
                0.0,
            )
            self._over, self.ended = False, None
            self._record(
                "progress moved", now, to=[run.leg, 0.0], cause=cause, how="backs out along the demo's path"
            )
            return
        here = self.cmds[-1]
        old_end = self._line_end(run.leg, here.pose)
        remaining = max(0.0, run.total - self.tau)
        self.objects.snap(obj)
        new_end = self._line_end(run.leg, here.pose)
        old_len, new_len = (
            self.arm.walk_seconds(here.pose, old_end),
            self.arm.walk_seconds(here.pose, new_end),
        )
        nxt = self._leg_run(run.leg, here.s, here.pose, here.grip)
        if old_len > 1e-9:
            nxt.seconds = remaining * new_len / old_len
        self.run, self.tau = nxt, 0.0
        self._record("line re-aimed", now, cause=cause, seconds=nxt.seconds)

    def _trailing(self, now: float) -> bool:
        """Whether to wait for the arm: a joint trails the last command sent by more than the motor loop's catch-up
        budget. The queue is cancelled so the goal stands still, which lets the motor loop's trim free a sticking
        joint, instead of running on into the loop's own freeze (act_execution.md E21); the clock resumes from the last
        command sent once every joint is back within the budget."""
        observed = getattr(self.pb, "observed", None)
        if self.arm.lag_deg is None or observed is None or self.sent < 0:
            return False
        q = observed()
        if q is None:
            return False
        lag = np.abs(np.asarray(q, dtype=float) - self.cmds[self.sent].q)
        lag[self.arm.gripper] = 0.0
        worst = float(lag.max())
        if worst > self.arm.lag_deg:
            if self.waiting_arm is None:
                self.waiting_arm = now
                self._cancel()
                self._record("waiting for the arm", now, joint=int(np.argmax(lag)), lag_deg=worst)
            return True
        if self.waiting_arm is not None:
            self.waited_s += now - self.waiting_arm
            self._record("the arm caught up", now, waited_s=now - self.waiting_arm)
            self.waiting_arm = None
        return False

    # ── the step ──

    def start(self, q_now: np.ndarray, now: float) -> None:
        """Begin from the arm's present joints ``q_now``, which the motor loop holds as command 0."""
        q = np.asarray(q_now, dtype=float).copy()
        pose = self.arm.fk(q)
        self.run, self.tau = self._leg_run(0, 0.0, pose, float(q[self.arm.gripper])), 0.0
        self.cmds = [Cmd(q, pose, float(q[self.arm.gripper]), 0.0, 0, 0.0, "start", state=(self.run, 0.0))]
        self._record("start", now, speed=self.speed)

    def step(self, now: float) -> None:
        if self.done:
            return
        dt = 0.0 if self._last_now is None else max(0.0, now - self._last_now)
        self._last_now = now
        if dt > 0:
            self.step_gaps = [*self.step_gaps[-99:], dt]
        hold = set()
        if self.run is not None and self.run.kind == "leg" and self.tl.legs[self.run.leg].move == "follow":
            hold = {self.tl.legs[self.run.leg].obj}
        self.objects.update(dt, hold)
        self._on_sent(now)
        self._check(now)
        if self.done:
            return
        if self.waiting is not None:
            self._resume(now)
            if self.done or self.waiting is not None:
                return
        self._watch_moves(now)
        if self._trailing(now):
            return
        if not self._over:
            self._fill(now)
            if self.done:
                return
        if self._over and self.ended is None and self.sent >= len(self.cmds) - 1:
            self.ended = now
            self._record("motion ends", now)
        if self.ended is not None:
            self._end(now)

    def _end(self, now: float) -> None:
        """The motion is over: once every open check has answered, or two camera frames have come without an answer,
        the act ends."""
        open_ms = [m for m in self.milestones if m.status == "checking" and not m.closed]
        frames = {d.t for d in self.scene.depths(since=self.ended)}
        if open_ms and len(frames) < FAILURE_FRAMES:
            return
        failed = [m for m in self.milestones if m.status == "failure"]
        if failed:
            self._finish(False, f"the {failed[0].name} failed", now)
            return
        unsure = [m.name for m in open_ms]
        self._finish(
            True, "done" + (f"; the camera could not tell: {', '.join(unsure)}" if unsure else ""), now
        )
