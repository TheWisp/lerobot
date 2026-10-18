# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The act engine: the control loop's world, task and motion steps (act_execution.md, "The loops, and who writes
what").

``Engine.step(now)`` runs once per control tick, in order: the world step (every object takes its new measurements),
the task step (Progress from what the motor step has sent; each open Milestone answers each new measurement of its
object; going back or ahead when one fails or the target moves) and the motion step (top the command queue up, solved
a few ticks ahead so a slow IK solve never holds up the motor step). It never waits; the arm stands still only when
the queue runs out on purpose (the act ended, a final Milestone, an object to be measured again).
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
from .predicates import Answer, Verdict
from .scene import Measurement, SceneLog
from .timeline import BASE, START, Event, Timeline
from .world import Steady, World

FAILURE_FRAMES = 2  # a choice: the fewest measurements in a row that ignore a single bad one
TRIES = 2  # act_execution.md Q1's leaning: a Milestone may fail once and be tried again
SLOPE_MAX = 3.0  # a cubic from 0 to 1 stays monotonic with end slopes up to 3


@dataclasses.dataclass
class Arm:
    """What the engine needs of an arm (act_execution.md O10). ``kin`` maps joints in degrees, the gripper's column
    included: ``forward_kinematics(q)`` and ``inverse_kinematics(seed, pose)``. ``step_deg`` is the most an arm joint
    may move in one tick, a bound against a solve that jumps rather than a speed limit (the motor loop's own per-tick
    clamp, MAX_STEP_DEG on the jog); ``walk_m_s`` and ``walk_rad_s`` the walk speed the operator set."""

    kin: Any
    gripper: int
    closing: float = 1.0
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    step_deg: float = 12.0
    workspace: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    hz: float = 30.0
    walk_m_s: float = 0.04
    walk_rad_s: float = math.radians(30.0)
    grip_units_s: float = 80.0

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


def _toward(a: np.ndarray, b: np.ndarray, centre: np.ndarray, max_m: float, max_rad: float) -> np.ndarray:
    """Pose ``a`` moved toward ``b`` by at most ``max_m`` at ``centre`` and ``max_rad`` of turn."""
    c = np.append(centre, 1.0)
    dist = float(np.linalg.norm((b @ c)[:3] - (a @ c)[:3]))
    ang = math.radians(core.pose_residual(a, b)[1])
    f = 1.0
    if dist > max_m:
        f = min(f, max_m / dist)
    if ang > max_rad:
        f = min(f, max_rad / ang)
    return b.copy() if f >= 1.0 else core.interp_rigid(a, b, f)


class Engine:
    """One act: the control loop's world, task and motion steps (act_execution.md, "The loops, and who writes what").
    ``world`` holds the act's objects' poses and noise; ``perturb``, when given, may change a command on its way to
    the motor loop (tests, and fault injection on the rig), never what the engine believes it sent."""

    def __init__(
        self,
        timeline: Timeline,
        arm: Arm,
        scene: SceneLog,
        playback: Playback,
        world: World,
        speed: float = 1.0,
        settings: dict[str, Any] | None = None,
        perturb: Any | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        assert speed > 0.0, "a positive speed"
        self.tl, self.arm, self.scene, self.pb, self.world = timeline, arm, scene, playback, world
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
        self.holds: dict[
            str, Steady
        ] = {}  # how each held object sits in the gripper, relative to the fingertip
        self.grasp_i: dict[str, int] = {}  # each held object's grasp, as a demo sample
        self.records: list[dict[str, Any]] = []
        self.solve_s = 0.0  # the slowest solve so far
        self.step_gaps: list[float] = []  # the time between the engine's own steps, newest last
        self.waiting: dict[str, Any] | None = None
        self.watched: dict[
            str, Any
        ] = {}  # the target the commands were made with: its pose, time, far streak
        self.ended: float | None = None
        self.end_frames = 0
        self.done = False
        self.result: dict[str, Any] | None = None
        self._last_now: float | None = None
        self._over = False  # the generator has made the act's last command
        self._unreachable = ""
        self.stretched_s = 0.0  # demo seconds the joint bound slowed
        self.anchors: dict[int, np.ndarray] = {}  # legs held in START, from where each began
        # Each object's pose as the commands use it: the world's, followed at no more than the walk speed during a leg,
        # so an object that moves becomes motion of the arm, not a jump between two commands; a new path (going back,
        # backing out) starts from the world's poses as they are, being a move from where the arm is (E26).
        self.aims: dict[str, np.ndarray] = {}
        self.held_aims: dict[str, np.ndarray] = {}  # the same for how each held object sits in the gripper
        self.let_go: dict[
            str, np.ndarray
        ] = {}  # the gripper-side change each released object had when let go
        self.moves_recorded: dict[str, float] = {}

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
        }
        self._record("end", now, ok=ok, why=why)

    # ── composition ──

    def _anchor(self, leg_n: int) -> np.ndarray:
        """Where a leg held in START puts the demo's pose: the leg's start times the demo's pose there, inverted. Set
        when the leg begins; before that, predicted from where the leg before it ends."""
        if leg_n in self.anchors:
            return self.anchors[leg_n]
        before = self._state_pose(leg_n - 1, 1.0) if leg_n > 0 else self.cmds[0].pose
        return before @ np.linalg.inv(self.demo.pose_at(self.tl.legs[leg_n].t0))

    def _compose(self, leg_n: int, d: np.ndarray, motion: np.ndarray | None = None) -> np.ndarray:
        """The demo's pose ``d`` carried by the leg's object, at its estimate or at ``motion``."""
        leg = self.tl.legs[leg_n]
        if leg.obj == START:  # the demo's gripper motion from where the leg began: any hold rides along
            return self._anchor(leg_n) @ d
        out = (self.aims.get(leg.obj, self.world.motion(leg.obj)) if motion is None else motion) @ d
        return out @ self._held_fix(leg.held)

    def _held_fix(self, held: str | None) -> np.ndarray:
        """The gripper-side change for the object a leg began holding: while it is held, from how it sits; once let
        go in the leg, as it was then, so the gripper goes on from where it let go."""
        if held is None:
            return np.eye(4)
        return self._fix(held) if held in self.holds else self.let_go.get(held, np.eye(4))

    def _fix(self, obj: str) -> np.ndarray:
        """The gripper-side change that puts the held ``obj`` where the demo had it: the demo's fingertip at the grasp
        against how the object sits in the gripper, as the commands follow it."""
        sits = self.held_aims.get(obj, self.holds[obj].estimate)
        return np.linalg.inv(self.demo.tips[self.grasp_i[obj]]) @ np.linalg.inv(sits)

    def _state_pose(self, leg_n: int, s: float) -> np.ndarray:
        return self._compose(leg_n, self.demo.pose_at(self.tl.time(leg_n, s)))

    def _line_end(self, leg_n: int, c0: np.ndarray, motion: np.ndarray | None = None) -> np.ndarray:
        leg = self.tl.legs[leg_n]
        if leg.move == "lift":
            end = c0.copy()
            end[2, 3] += leg.up_m
            return end
        end = self._compose(leg_n, self.demo.pose_at(leg.t1), motion)
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
        if leg.obj == START:
            self.anchors[leg_n] = c0 @ np.linalg.inv(self.demo.pose_at(td0))
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
            g0
            if leg.held is not None and leg.held in self.holds
            else self.demo.grip_at(self.tl.time(leg_n, s))
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
            pose = run.d0 @ self.demo.pose_at(td) @ self._held_fix(leg.held)
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

    def _next_run(self, run: Run, q: np.ndarray) -> Run | None:
        """The run after ``run``, starting where it ends; ``q`` is the arm's last commanded joints."""
        pose, grip, _, _ = self._sample(run, run.total)
        if run.kind == "leg" and self.tl.legs[run.leg].constraint == "best_effort":
            # Its end may be out of reach: the next run starts where the joints get there (act_execution.md E23).
            pose = self.arm.fk(self.arm.solve(q, pose, grip, np.zeros_like(q))[0])
        if run.kind == "transition":
            return self._leg_run(run.leg, run.s0, pose, grip)
        if run.kind == "retrace":
            return self._transition((run.leg, 0.0), pose, grip)
        if run.leg + 1 >= len(self.tl.legs):
            return None
        return self._leg_run(run.leg + 1, 0.0, pose, grip)

    def _advance(self, run: Run, tau: float, q: np.ndarray) -> tuple[Run | None, float]:
        while tau > run.total + 1e-12:
            nxt = self._next_run(run, q)
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
            run, tau = self._advance(self.run, self.tau + dtau, prev.q)
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
                if err_m > core.ACT_REACH_TOL_M or err_deg > core.ACT_REACH_TOL_DEG:
                    if leg.constraint != "best_effort":
                        self._unreachable = f"{leg.label}: the arm reaches only within {err_m * 1000:.1f} mm, {err_deg:.1f} deg"
                    # A best-effort command is where its joints get: the leg after it starts there, not at a pose
                    # no joint reached (a lift stopped short by a joint's range, act_execution.md E23).
                    pose = self.arm.fk(q)
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
        info: dict[str, Any] = {"tip_from": "reading" if tip is not None else "command"}
        if tip is None:
            tip = self.cmds[self.sent].pose if self.sent >= 0 else self.cmds[0].pose
        m.status, m.t, m.seen, m.streak, m.closed = "checking", now, now, 0, False
        if e.kind == "grasp":
            motion = self.world.motion(e.obj)
            # How it sits: its pose and the fingertip now, until its measurements replace it (_sits).
            self.held_aims.pop(e.obj, None)
            self.let_go.pop(e.obj, None)
            self.grasp_i[e.obj] = e.i
            self.holds[e.obj] = Steady(
                self._as_held(e.obj, tip, motion), self.world.centres.get(e.obj, np.zeros(3))
            )
            fix_m, fix_deg = core.pose_residual(np.eye(4), self._fix(e.obj))
            info.update(fix_mm=round(fix_m * 1000.0, 2), fix_deg=round(fix_deg, 2))
            self.world.touched(e.obj, now)
            m.data = {"tip": tip, "motion": motion}
        else:
            for g in self.milestones:
                if g.event.kind == "grasp" and g.event.obj == e.obj and g.event.i == e.grasp_i:
                    g.closed = True
            if e.obj in self.holds:
                self.let_go[e.obj] = self._fix(e.obj)
            hold = self.holds.pop(e.obj, None)
            ref = self.world.motion(e.ref)
            put = tip @ hold.estimate if hold is not None else None
            m.data = {"tip": tip, "put": put}
            if put is not None:
                m.data["rel_put"] = np.linalg.inv(ref) @ put
        self._record("milestone begins", now, milestone=m.name, **info)

    # ── checks ──

    def _evidence(self, m: Milestone, x: Measurement) -> Verdict:
        """A Milestone's answer from one measurement of its object: the measured pose against where each outcome puts
        it (act_execution.md, "What a check answers"). A grasp's success also takes how the object sits from it."""
        e, obj = m.event, m.event.obj
        noise = self.world.noise(obj)
        if noise is None:
            return Verdict(Answer.UNKNOWN, f"{obj}'s noise is not measured yet")
        tip = self.scene.tip_at(x.t)
        if tip is None:
            return Verdict(Answer.UNKNOWN, "no joint reading at that frame")
        seen = self.world.centre(obj, x.motion)
        if e.kind == "grasp":
            hold = self.holds.get(obj)
            if hold is None:
                return Verdict(Answer.UNKNOWN, "not held")
            held, lay = self.world.centre(obj, tip @ hold.estimate), self.world.centre(obj, m.data["motion"])
            return pred.decide(seen * 1000, held * 1000, lay * 1000, noise[0] * 1000, "mm")
        put = m.data.get("put")
        if put is None:
            return Verdict(Answer.UNKNOWN, "it was never held")
        stays = self.world.centre(obj, self.world.motion(e.ref) @ m.data["rel_put"])
        away = self.world.centre(obj, tip @ np.linalg.inv(m.data["tip"]) @ put)
        return pred.decide(seen * 1000, stays * 1000, away * 1000, noise[0] * 1000, "mm")

    def _check(self, now: float) -> None:
        """The task step's checks: every open Milestone answers each new measurement of its object."""
        for m in self.milestones:
            if m.status not in ("checking", "success") or m.closed:
                continue
            for x in self.scene.measurements(m.event.obj, since=m.seen + 1e-9):
                v = self._evidence(m, x)
                m.seen, m.last = x.t, v
                if m.event.kind == "grasp" and not v.no:
                    self._sits(m.event.obj, x, now)
                self._record("check", now, milestone=m.name, frame=x.t, answer=v.answer.value, why=v.why)
                if v.no:
                    m.streak += 1
                    if m.streak >= FAILURE_FRAMES:
                        self._failed(m, v, now)
                        return
                    continue
                m.streak = 0
                if v.yes and m.status == "checking":
                    m.status = "success"

    def _sits(self, obj: str, x: Measurement, now: float) -> None:
        """How the held ``obj`` sits in the gripper takes a measurement, after the check answered from how it sat
        before: every one but those showing it out of the gripper, which say nothing about how it sits there."""
        hold, tip = self.holds.get(obj), self.scene.tip_at(x.t)
        if hold is None or tip is None:
            return
        before = hold.estimate
        if hold.add(x.t, self._as_held(obj, tip, x.motion), self.world.move_threshold(obj)):
            apart = np.linalg.norm(self.world.centre(obj, hold.estimate) - self.world.centre(obj, before))
            self._record(
                "moved in the gripper",
                now,
                obj=obj,
                frame=x.t,
                mm=round(float(apart) * 1000, 1),
                deg=round(core.pose_residual(before, hold.estimate)[1], 1),
            )

    def _as_held(self, obj: str, tip: np.ndarray, motion: np.ndarray) -> np.ndarray:
        """How ``obj`` at ``motion`` sits in the gripper at ``tip``: of the ways its symmetry makes look the same, the
        one nearest how the demo held it."""
        return self.world.equivalent(
            obj, np.linalg.inv(tip) @ motion, np.linalg.inv(self.demo.tips[self.grasp_i[obj]])
        )

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
        self.holds.pop(e.obj, None)  # it is not where the gripper would have it
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
            if self.tl.legs[leg_n].held is not None and self.tl.legs[leg_n].held not in self.holds:
                continue  # a state with an object in the gripper, which it does not hold
            cost = self.arm.walk_seconds(here, self._state_pose(leg_n, s)) + abs(
                float(self.demo.t[i]) - t_here
            )
            if cost < best_cost:
                best, best_cost = (leg_n, s), cost
        return best, best_cost

    def _place(self, target: tuple[int, float], now: float, cause: str) -> None:
        last = self.cmds[self.sent]
        self.aims.clear()  # a new path: to where the objects are now
        self.held_aims.clear()
        self.let_go.clear()
        for m in self.milestones:
            if m.status != "ahead" and (m.event.leg, m.event.s) >= target:
                m.status, m.streak, m.closed, m.data = "ahead", 0, False, {}
                if m.event.kind == "grasp":
                    self.holds.pop(m.event.obj, None)
        self.run, self.tau = self._transition(target, last.pose, last.grip), 0.0
        self._over, self.ended = False, None
        self._record("progress moved", now, to=list(target), cause=cause)

    def _resume(self, now: float) -> None:
        w = self.waiting
        if self.world.t[w["obj"]] <= w["since"]:
            if now - w["since"] > self.wait_view_s:
                self._finish(False, f"{w['obj']} was not seen again within {self.wait_view_s:g} s", now)
            return
        self.world.resting(w["obj"], self.world.t[w["obj"]])
        last = self.cmds[self.sent]
        t_here = last.td if last.td is not None else self.tl.time(*self.progress)
        target, _ = self._nearest(self._lo_time(None), w["hi"], last.pose, t_here)
        self.waiting = None
        if target is None:
            self._finish(False, f"no point of the demo before the {w['cause']} to go back to", now)
            return
        self._place(target, now, f"the {w['cause']} failed")

    def _watch_moves(self, now: float) -> None:
        """The object the current leg heads for moved: the world's pose of it is further than twice its noise from
        the pose the run's commands were first made with (R4). During a follow — contact, or nearly — the arm backs out along
        the demo's path relative to where the object was, to the follow's start, and goes in again relative to where
        it is now. During a line the line re-aims from where the arm is and keeps its speed, so a target moved nearer
        is reached sooner."""
        run = self.run
        if run is None or run.kind != "leg":
            return
        leg = self.tl.legs[run.leg]
        obj = leg.obj
        if obj in (BASE, START) or obj in self.holds or leg.move == "lift" or obj not in self.world.pose:
            return
        w = self.watched.get(obj)
        if w is None or w["run"] is not run:  # a new run: its commands are made with the pose now
            self.watched[obj] = {"run": run, "pose": self.world.motion(obj).copy()}
            return
        here = self.world.motion(obj)
        off = float(np.linalg.norm(self.world.centre(obj, here) - self.world.centre(obj, w["pose"])))
        if off <= self.world.move_threshold(obj):
            return
        before = w["pose"]
        w["pose"] = here.copy()
        cause = f"{obj} moved {off * 1000:.0f} mm"
        if leg.move == "follow":
            if any(m.status != "ahead" and m.event.leg == run.leg for m in self.milestones):
                return  # the gripper has acted on it in this follow: the composition follows it
            self._on_sent(now)
            self._cancel()
            self.aims.pop(obj, None)  # backing out is a new path: in again to where it is now
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
        old_end = self._line_end(run.leg, here.pose, before)
        remaining = max(0.0, run.total - self.tau)
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
        self.world.update()
        for name, (t_move, before) in self.world.moves.items():  # the act's record: every move the world saw
            if t_move > self.moves_recorded.get(name, -np.inf):
                self.moves_recorded[name] = t_move
                _, deg = core.pose_residual(before, self.world.motion(name))
                apart = np.linalg.norm(
                    self.world.centre(name, before) - self.world.centre(name, self.world.motion(name))
                )
                self._record(
                    "object moved",
                    now,
                    obj=name,
                    frame=t_move,
                    mm=round(float(apart) * 1000, 1),
                    deg=round(deg, 1),
                )
        self._on_sent(now)
        self._check(now)
        if self.done:
            return
        self._follow(dt)
        if self.waiting is not None:
            self._resume(now)
            if self.done or self.waiting is not None:
                return
        self._watch_moves(now)
        if not self._over:
            self._fill(now)
            if self.done:
                return
        if self._over and self.ended is None and self.sent >= len(self.cmds) - 1:
            self.ended = now
            self._record("motion ends", now)
        if self.ended is not None:
            self._end(now)

    def _follow(self, dt: float) -> None:
        """The motion step's inputs follow their estimates at no more than the walk speed: each object's pose and how
        each held one sits in the gripper, so a changed estimate becomes motion of the arm, not a jump (E26)."""
        step_m, step_rad = self.arm.walk_m_s * dt, self.arm.walk_rad_s * dt
        for name, pose in self.world.pose.items():
            self.aims[name] = _toward(
                self.aims.get(name, pose), pose, self.world.centres[name], step_m, step_rad
            )
        self.held_aims = {
            name: _toward(
                self.held_aims.get(name, hold.estimate),
                hold.estimate,
                self.world.centres.get(name, np.zeros(3)),
                step_m,
                step_rad,
            )
            for name, hold in self.holds.items()
        }

    def _end(self, now: float) -> None:
        """The motion is over: once every open check has answered, or two camera frames have come without an answer,
        the act ends."""
        open_ms = [m for m in self.milestones if m.status == "checking" and not m.closed]
        frames = self.scene.frames(since=self.ended)
        if open_ms and len(frames) < FAILURE_FRAMES:
            return
        failed = [m for m in self.milestones if m.status == "failure"]
        if failed:
            self._finish(False, f"the {failed[0].name} failed", now)
            return
        unsure = [m.name for m in open_ms]
        if unsure:  # the motion is done, the outcome is not shown: never reported as a success (E24)
            self._finish(False, f"not confirmed: the camera could not tell the {', '.join(unsure)}", now)
            return
        self._finish(True, "done", now)
