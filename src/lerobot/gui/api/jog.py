# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cartesian jog: drag the gripper of ONE SO-107 arm from the URDF tile.

The server owns the arm (bus only, no cameras) and runs a streaming thread that
walks a pose reference toward the operator's target at a bounded speed, solves
the production Cartesian IK each tick, and sends the joints through the
follower — so the servo gains and the gravity feed-forward the profile carries
are exactly what any other run of that profile gets. Every tick also reads the
encoders, so the tile can draw the commanded and the observed pose together and
scale the difference: that gap IS the sag (or its absence) at every pose the
operator drags to.

Passive GET ``/state`` returns cached state; the bus is touched only by the
streaming thread and the connect/disconnect executor.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ._guarded_path import GuardedPath

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jog", tags=["jog"])
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="jog")

HZ = 30.0
# A tick that takes this long holds every stream up: the commands it would have sent in between are never sent, and
# the next one jumps (an act of 2026-10-08 had 33 ticks over 0.2 s, up to 0.67 s; the eight before it 0 to 5). Each is
# kept, with where its time went, for the state to show.
SLOW_TICK_S = 0.1
SLOW_TICKS_KEPT = 50
TICKS_TIMED = 300  # the state's tick statistics cover the last this many ticks
MAX_LINEAR_M_S = 0.04  # default walk speed; the operator moves it with a slider
MAX_ANGULAR_RAD_S = math.radians(30.0)
LINEAR_M_S_RANGE = (0.005, 0.30)
ANGULAR_RAD_S_RANGE = (math.radians(5.0), math.radians(180.0))
# A drag may ask for at most this much rotation away from where the arm is
# now: one bounded move at a time, never a wrist flip the IK has to invent.
MAX_ROT_DELTA_RAD = math.radians(60.0)
ROT_DELTA_RAD_RANGE = (math.radians(10.0), math.radians(150.0))
DIVERGE_DEG = 25.0  # a joint this far behind its command is stalled or blocked: freeze
# A playback waits for a joint this far behind the sample sent (:func:`_advance_playback`). On 2026-10-11 06:50 the
# lift after the release ran its schedule while the loaded shoulder fell behind: 3.2-3.9 deg behind put the fingertip
# 7-9 mm off the planned path, 8.4 deg (its first 0.3 s, from rest) 26 mm, 20 mm down into the stack; through the
# rest of that act the joints stayed within 4.2 deg and the fingertip within 7 mm.
PLAYBACK_LAG_DEG = 3.0
# Successful SO-107 lifts on 2026-10-12 needed 4.4-4.6 deg to break static friction after the 3 deg pause.
# Pausing forever at 3 deg cannot build that effort. Allow this measured breakaway range, bounded at 5 deg,
# then hold/freeze instead of the former unlimited timeout continuation (23 deg before an overload).
# This is a transport following budget, not placement accuracy: opening still requires the tighter tolerance.
PLAYBACK_STOP_LAG_DEG = 5.0
PLAYBACK_HOLD_MAX_S = 0.5  # recoverable startup lag took 0.3 s; each catch-up wait is bounded
# The temperature a servo reports is shown and recorded, not acted on. The servos put a wrong value in it for one read
# now and then, in an otherwise whole reply: at rest the three at the wrist end, +3 to +15 C on 0.2-0.7 % of reads;
# moving under load any of them (2026-10-11: shoulder_lift 33 C, then 88 C and 110 C, a read each). LeRobot's own
# sync_read of the register shows the same; the servos' own overheat protection (70 C, unloading torque) tripped on
# none of them. A servo that does overheat says so in its replies' status, and _faults stops the arm on it.
# Static friction and gear play stop a joint short of its goal on the side it came from, by a pose-dependent
# amount the gravity feed-forward cannot model: at the act's poses (P 32, feed-forward on) the fingertip
# settled 0.7 to 10 mm off a still target, above it after coming down and below it after going up. Once a
# goal holds still, the encoders' error is fed back into the command until each joint sits on its goal.
TRIM_GAIN = 0.1  # share of the remaining joint error added to the trim per tick
TRIM_DEAD_DEG = 0.3  # within this a joint is on its goal and its trim holds
TRIM_MAX_FLIPS = 3  # a joint that has crossed its goal this often stops correcting until the goal moves
# A joint held short by friction works at moderate load; one straining against gravity or contact works near
# its limit, and pushing it further trips the servo's own overload protection (it did, lifting the extended
# arm). Measured with the correction off at the act's poses: stuck joints at 5-38 % of full drive, against
# the servos' overload trip at 80 % (each servo's Overload_Torque, read at connect). The correction grows only
# below TRIM_GROW_LOAD of a joint's trip level, holds up to TRIM_RELEASE_LOAD, and lets go above it.
TRIM_GROW_LOAD = 0.75
TRIM_RELEASE_LOAD = 0.875
TRIM_RELEASE_RATE = 0.2  # share of a straining joint's trim dropped per tick
OVERLOAD_PCT_DEFAULT = 80  # the trip level of a servo that would not say its own
TRIM_MAX_DEG = 3.0  # a joint still off by more than this is blocked, not sticking: the trim stops growing
TRIM_REST_TICKS = 9  # the goal must hold still this long first
TRIM_MOVE_DEG = 0.02  # a goal that changes by more than this in a tick is moving
# The gripper letting go changes what the arm presses on, so it starts the correction over as a moving goal does. A
# trim grown while a held object rests on something throws the arm down the moment the fingers open: in an act of
# 2026-10-08 the wrist's trim reached its 3 deg cap with the gamepad resting on the cube, the arm fell 7 mm as the
# gripper opened, and the jaws slid down around the gamepad and lifted it off the cube. (Closing is left as it was:
# nothing has measured a trim there.)
TRIM_GRIP_OPEN = 0.5  # a gripper goal that opens by more than this in a tick is letting go
# The overload flag in a Feetech reply's status byte (the SDK's ERRBIT_OVERLOAD).
OVERLOAD_ERRBIT = 32
RAMP_DEG_S = 30.0  # joint-space moves (ready, park): slow enough to watch, well under the per-tick clamp
GRIP_UNITS_S = 80.0  # gripper opening walk, in its 0..100 units per second
# The working pose the Cartesian walk starts from: forward of the fold, inside the URDF limits,
# the same seed the benches use. Motor degrees.
READY_DEG = {  # hardcode-ok: SO-107 working pose
    "shoulder_pan": 0.0,
    "shoulder_lift": -45.0,
    "elbow_flex": 74.0,
    "forearm_roll": 2.0,
    "wrist_flex": -41.0,
    "wrist_roll": -12.0,
}


class ConnectBody(BaseModel):
    profile: str
    arm: str = "left"
    p_coefficient: int | None = None
    i_coefficient: int | None = None
    d_coefficient: int | None = None
    gravity_ff_alpha: float | None = None


class TargetBody(BaseModel):
    position: list[float]  # URDF world frame, metres, the virtual gripper tip
    quaternion: list[float]  # x, y, z, w


class GripperBody(BaseModel):
    pos: float  # 0..100, the follower's gripper units


class LimitsBody(BaseModel):
    linear_mm_s: float | None = None
    angular_deg_s: float | None = None
    rotation_cap_deg: float | None = None
    settle: bool | None = None  # the settle correction on a still goal; off to measure the arm without it


@dataclass(frozen=True)
class _Settle:
    """The settle correction's state: per-joint trims, how long the goal has held still, the side of its goal
    each joint was last corrected from, and how often it has crossed its goal since."""

    trim: dict[str, float] = field(default_factory=dict)
    rest: int = 0
    side: dict[str, float] = field(default_factory=dict)
    flips: dict[str, int] = field(default_factory=dict)


@dataclass
class _Jog:
    robot: Any = None
    arm: str = "left"
    kin: Any = None
    ctrl: Any = None
    alignment: dict = field(default_factory=dict)
    thread: threading.Thread | None = None
    stop: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    target: np.ndarray | None = None  # 4x4, URDF world, tip frame
    ref: np.ndarray | None = None  # the walked reference the IK sees
    ref0: np.ndarray | None = None  # controller's latched reference pose
    q_cmd: dict[str, float] = field(default_factory=dict)
    q_obs: dict[str, float] = field(default_factory=dict)
    halted: bool = False
    reason: str = ""
    temps: dict[str, int] = field(default_factory=dict)
    good: _ArmState | None = None  # the loop's last checked read, which the next is held to
    bad_reads: int = 0  # bad reads in a row
    ticks: int = 0
    gains: dict[str, float] = field(default_factory=dict)
    max_linear_m_s: float = MAX_LINEAR_M_S
    max_angular_rad_s: float = MAX_ANGULAR_RAD_S
    max_rot_delta_rad: float = MAX_ROT_DELTA_RAD
    holding: bool = False  # the IK refused the last tick's step (unreachable or an implausible jump)
    robot_id: str = ""
    profile: str = ""
    grip_target: float | None = None  # 0..100; None until the operator asks for a change
    tip_offset: np.ndarray | None = None  # anchor->tip in use (measured when a calibration exists)
    tip_calibrated: bool = False
    joint_zero_deg: dict[str, float] = field(default_factory=dict)
    workspace_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Demo mode: a leader arm's joints pass straight through to the follower each tick, and the
    # follower's own positions are recorded for the demo's keyframes.
    leader: Any = None
    leader_id: str = ""
    mode: str = (
        "cartesian"  # "cartesian" (the IK walk) | "leader" | "joints" (an act streams joint targets) | "path"
    )
    q_target: dict[str, float] | None = None  # joints mode: the configuration the loop streams each tick
    settle: _Settle = field(default_factory=lambda: _Settle())  # the correction of a goal that holds still
    # Off unless asked for: holding the arm still at (-60, -200, 60) mm, it tripled the shoulder's load (11.2 % with P 32
    # alone, 39.2 % with it, 48.8 % with the feed-forward as well, against 10.4 % with the feed-forward alone) and left
    # the fingertip farther off (4.6 mm against 1.4): the trim grows against gear friction that holds the joint, and the
    # servo keeps pushing. Resets overheated the shoulder past 50 C in minutes (2026-10-08).
    settle_on: bool = False
    load: dict[str, int] = field(default_factory=dict)  # Present_Load per motor: signed, 0.1 % of full drive
    protection: dict[str, dict[str, int]] = field(
        default_factory=dict
    )  # each motor's own limits, read at connect
    goal_prev: dict[str, float] | None = None  # last tick's goal, to tell a still goal from a moving one
    grip_goal_prev: float | None = None  # last tick's gripper goal, to tell a gripper letting go
    record: list[dict[str, Any]] | None = None  # samples while recording
    record_t0: float = 0.0
    last_record: list[dict[str, Any]] = field(default_factory=list)
    # A playback (:func:`play_joints`): joint targets the loop sends in order, one per tick, each no earlier than its
    # time from the playback's start; a slow tick delays the samples after it instead of losing them.
    playback: list[dict[str, float]] | None = None
    playback_due: list[float] = field(default_factory=list)  # each sample's time from the start, s
    playback_t0: float = 0.0  # the start, monotonic clock
    playback_sent: int = -1  # the last sample sent
    playback_sent_at: list[float] = field(default_factory=list)  # when each sample was sent, wall clock
    playback_tick: float = 0.0  # the monotonic time of the playback's last tick
    playback_held: float | None = None  # since when the playback waits for a joint behind its target
    playback_hold_spent: bool = False  # the soft wait expired; reset once all arm joints catch up
    playback_blocked: float | None = None  # since exceeding the hard bound, or delaying a release
    guarded_path: GuardedPath | None = None
    path_busy: bool = False  # reservation while a path is validated or handed back
    last_read_at: float = 0.0  # monotonic: the last complete, checked motor read
    tick_s: deque = field(default_factory=lambda: deque(maxlen=TICKS_TIMED))  # the last ticks' durations
    slow_ticks: deque = field(
        default_factory=lambda: deque(maxlen=SLOW_TICKS_KEPT)
    )  # ticks over SLOW_TICK_S: when, how long, the mode, and the milliseconds each phase took

    @property
    def connected(self) -> bool:
        return self.robot is not None


_jog = _Jog()


def _quat_from_matrix(r: np.ndarray) -> list[float]:
    from scipy.spatial.transform import Rotation

    return [float(v) for v in Rotation.from_matrix(r).as_quat()]


def _matrix_from_quat(q: list[float]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(q).as_matrix()


def _urdf_rad(alignment: dict, q_motor: dict[str, float]) -> dict[str, float]:
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES, URDF_JOINT_NAMES

    out = {}
    for m, j in zip(MOTOR_NAMES, URDF_JOINT_NAMES, strict=True):
        a = alignment[m]
        out[j] = math.radians(a.sign * q_motor[m] + a.offset_deg)
    return out


def _step_pose(
    ref: np.ndarray,
    target: np.ndarray,
    max_linear_m_s: float = MAX_LINEAR_M_S,
    max_angular_rad_s: float = MAX_ANGULAR_RAD_S,
) -> np.ndarray:
    """Walk ``ref`` toward ``target`` by at most one tick of bounded linear/angular speed."""
    from scipy.spatial.transform import Rotation

    out = ref.copy()
    dp = target[:3, 3] - ref[:3, 3]
    n = float(np.linalg.norm(dp))
    max_p = max_linear_m_s / HZ
    out[:3, 3] = ref[:3, 3] + (dp if n <= max_p else dp * (max_p / n))
    r_rel = Rotation.from_matrix(target[:3, :3] @ ref[:3, :3].T)
    ang = float(np.linalg.norm(r_rel.as_rotvec()))
    max_a = max_angular_rad_s / HZ
    if ang > 1e-9:
        frac = 1.0 if ang <= max_a else max_a / ang
        out[:3, :3] = Rotation.from_rotvec(r_rel.as_rotvec() * frac).as_matrix() @ ref[:3, :3]
    return out


def _cap_rotation(target: np.ndarray, r_obs: np.ndarray, cap_rad: float) -> tuple[np.ndarray, bool]:
    """Clamp ``target``'s orientation to within ``cap_rad`` of ``r_obs``; the position is untouched.

    Post: the returned pose is ``target`` when it was within the cap, else the same
    axis of rotation from ``r_obs`` stopped at the cap, and the flag says which.
    """
    from scipy.spatial.transform import Rotation

    rv = Rotation.from_matrix(target[:3, :3] @ r_obs.T).as_rotvec()
    ang = float(np.linalg.norm(rv))
    if ang <= cap_rad:
        return target, False
    out = target.copy()
    out[:3, :3] = Rotation.from_rotvec(rv * (cap_rad / ang)).as_matrix() @ r_obs
    return out, True


def _trim_step(
    settle: _Settle,
    goal: dict[str, float],
    goal_prev: dict[str, float] | None,
    q_obs: dict[str, float],
    load: dict[str, float] | None = None,
) -> _Settle:
    """One tick of the settle correction.

    A moving goal clears it, since the friction flips with the direction of travel. After
    the goal has held still for ``TRIM_REST_TICKS``, each joint off by more than
    ``TRIM_DEAD_DEG`` gains ``TRIM_GAIN`` of its error per tick, within ``TRIM_MAX_DEG``,
    from whichever side it is off: a joint that creeps off later is corrected again. A
    sticking joint that breaks free lands past its goal, and correcting it back can repeat
    for as long as the goal holds (measured unbounded: the elbow hunted over two degrees, 73
    trim changes in 7 s); so after ``TRIM_MAX_FLIPS`` crossings a joint keeps its trim.
    ``load`` is each joint's load as a share of its own overload trip level: a joint
    above ``TRIM_RELEASE_LOAD`` is straining, not sticking, and its trim decays; one above
    ``TRIM_GROW_LOAD`` keeps its trim but gets no more.
    """
    if goal_prev is None or any(abs(goal[m] - goal_prev.get(m, goal[m] + 1.0)) > TRIM_MOVE_DEG for m in goal):
        return _Settle(trim=dict.fromkeys(goal, 0.0))
    rest = settle.rest + 1
    if rest < TRIM_REST_TICKS or not q_obs:
        return _Settle(settle.trim, rest, settle.side, settle.flips)
    trim, side, flips = dict(settle.trim), dict(settle.side), dict(settle.flips)
    load = load or {}
    for m, g in goal.items():
        share = abs(load.get(m, 0.0))
        if share >= TRIM_RELEASE_LOAD:
            trim[m] = trim.get(m, 0.0) * (1.0 - TRIM_RELEASE_RATE)
            continue
        if share >= TRIM_GROW_LOAD:
            continue
        e = g - q_obs[m]
        if abs(e) <= TRIM_DEAD_DEG:
            continue
        s = math.copysign(1.0, e)
        if m in side and s != side[m]:
            flips[m] = flips.get(m, 0) + 1
        side[m] = s
        if flips.get(m, 0) > TRIM_MAX_FLIPS:
            continue
        trim[m] = float(np.clip(trim.get(m, 0.0) + TRIM_GAIN * e, -TRIM_MAX_DEG, TRIM_MAX_DEG))
    return _Settle(trim, rest, side, flips)


def _trimmed(j: _Jog, action: dict[str, float]) -> dict[str, float]:
    """The goal with this tick's settle correction on every joint but the gripper, whose goal squeezes on purpose."""
    goal = {
        m: float(v) for m, v in ((k.removesuffix(".pos"), v) for k, v in action.items()) if m != "gripper"
    }
    grip = next((float(v) for k, v in action.items() if k.removesuffix(".pos") == "gripper"), None)
    if not j.settle_on:
        j.settle, j.goal_prev = _Settle(), goal
        return dict(action)
    if grip is not None and j.grip_goal_prev is not None and grip < j.grip_goal_prev - TRIM_GRIP_OPEN:
        j.goal_prev = None  # the gripper lets go: the correction starts over (TRIM_GRIP_OPEN)
    j.grip_goal_prev = grip
    share = {
        m: v / (10.0 * j.protection.get(m, {}).get("Overload_Torque", OVERLOAD_PCT_DEFAULT))
        for m, v in j.load.items()
    }
    j.settle = _trim_step(j.settle, goal, j.goal_prev, j.q_obs, share)
    j.goal_prev = goal
    return {**action, **{f"{m}.pos": g + j.settle.trim.get(m, 0.0) for m, g in goal.items()}}


def _advance_playback(j: _Jog, now: float) -> None:
    """Send the playback's next sample once its time has come, unless a joint is further than PLAYBACK_LAG_DEG behind
    the sample already sent: then the playback's clock waits for it, so the arm keeps to the planned path and arrives
    later. The soft wait expires so normal static friction can be overcome, but error above the measured transport
    budget, or an opening gripper while the arm lags, must catch up within the wait or freeze. A genuinely blocked
    joint cannot let the rest of the plan run ahead indefinitely. Pre: j.lock is held, joints mode, playback under way.
    """
    assert j.playback is not None
    k = j.playback_sent + 1
    behind = max(
        (abs(j.q_target[m] - j.q_obs[m]) for m in j.q_target if m != "gripper" and m in j.q_obs), default=0.0
    )
    pending = k < len(j.playback) and j.playback_sent >= 0
    opening = pending and j.playback[k].get("gripper", 0.0) < j.q_target.get("gripper", 0.0) - 1e-6
    blocked = pending and (behind > PLAYBACK_STOP_LAG_DEG or (opening and behind > PLAYBACK_LAG_DEG))
    j.playback_blocked = (j.playback_blocked if j.playback_blocked is not None else now) if blocked else None
    if blocked and now - j.playback_blocked >= PLAYBACK_HOLD_MAX_S:
        j.halted = True
        context = "before gripper opening" if opening else "beyond transport tolerance"
        j.reason = f"playback joint still {behind:.1f} deg behind {context} after {PLAYBACK_HOLD_MAX_S:.1f} s — frozen"
        return
    if behind <= PLAYBACK_LAG_DEG:
        j.playback_hold_spent = False
    waiting = pending and behind > PLAYBACK_LAG_DEG and (blocked or not j.playback_hold_spent)
    if (
        waiting
        and not blocked
        and j.playback_held is not None
        and now - j.playback_held >= PLAYBACK_HOLD_MAX_S
    ):
        j.playback_hold_spent, waiting = True, False
    if waiting:
        j.playback_held = j.playback_held if j.playback_held is not None else now
        j.playback_t0 += now - j.playback_tick  # the samples still to come wait as long
    else:
        j.playback_held = None
        if k < len(j.playback) and now - j.playback_t0 >= j.playback_due[k]:
            j.q_target = j.playback[k]
            j.playback_sent = k
            j.playback_sent_at.append(time.time())
    j.playback_tick = now


def _bad_read(j: _Jog, e: _BadReadError) -> None:
    """A tick whose read cannot be the arm sends nothing, so the motors hold their last goal; the replies are logged,
    and BAD_READS_FREEZE in a row freeze the arm."""
    with j.lock:
        j.bad_reads += 1
        n = j.bad_reads
        if n >= BAD_READS_FREEZE and not j.halted:
            j.halted, j.reason = True, f"bad joint reads, {n} in a row ({e}) — frozen"
    logger.warning("jog: bad read %d in a row, no command this tick: %s; replies %s", n, e, e.raw)


def _loop(j: _Jog) -> None:
    from scipy.spatial.transform import Rotation

    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    robot = j.robot
    grip = float(j.q_cmd["gripper"])
    period = 1.0 / HZ
    while not j.stop.is_set():
        t0 = time.perf_counter()
        try:
            with j.lock:
                if j.mode == "joints" and j.playback is not None and not j.halted:
                    _advance_playback(j, time.monotonic())
                target, ref, halted = j.target, j.ref, j.halted
                v_lin, v_ang = j.max_linear_m_s, j.max_angular_rad_s
                grip_target = j.grip_target
                ctrl = j.ctrl
                mode, leader, q_target = j.mode, j.leader, j.q_target
            t_locked = time.perf_counter()
            try:
                state = _read_state(robot.bus)
                _check_state(robot.bus, state, j.good)
            except _BadReadError as e:  # no command on what cannot be the arm; a second in a row freezes it
                _bad_read(j, e)
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
                continue
            j.good, j.bad_reads = state, 0
            _faults(robot.bus, state)
            q_obs = state.q
            t_read = time.perf_counter()
            if grip_target is not None:
                step = GRIP_UNITS_S / HZ
                grip += float(np.clip(grip_target - grip, -step, step))
            # A stop may arrive while the bus read is in progress. Every mode must honor it; otherwise a frozen
            # joint stream keeps sending (and trimming) its last goal, and a leader can keep moving the follower.
            with j.lock:
                halted = j.halted
            if halted:
                q_cmd, holding = None, False
            elif mode == "path":
                with j.lock:
                    path = j.guarded_path
                    if path is None:
                        raise RuntimeError("guarded path has no owner")
                    pose = path.kin.forward_kinematics(np.array([q_obs[m] for m in MOTOR_NAMES]))
                    command = path.tick(time.monotonic(), q_obs, pose, state.load, state.temps)
                    if command is None:
                        j.halted, j.reason = True, path.reason
                        q_cmd, holding = None, False
                    else:
                        # The plan checked these joints. Do not add the Cartesian settle trim to them.
                        robot.send_action(_clamped({f"{m}.pos": command[m] for m in MOTOR_NAMES}, q_obs))
                        q_cmd, holding, grip = command, False, command["gripper"]
                        j.q_target = command
            elif mode == "leader" and leader is not None:
                # The human drives: the leader's joints go straight to the follower, gripper included.
                act = leader.get_action()
                q_lead = {m: float(act[f"{m}.pos"]) for m in MOTOR_NAMES}
                robot.send_action(_clamped({f"{m}.pos": q_lead[m] for m in MOTOR_NAMES}, q_obs))
                q_cmd, holding, grip = q_lead, False, q_lead["gripper"]
                j.settle, j.goal_prev = _Settle(), None  # the human closes the loop
            elif mode == "joints" and q_target is not None:
                # An act replays the demo's own joints, corrected: they go straight to the follower, gripper included.
                robot.send_action(
                    _clamped(_trimmed(j, {f"{m}.pos": q_target[m] for m in MOTOR_NAMES}), q_obs)
                )
                q_cmd, holding, grip = dict(q_target), False, q_target["gripper"]
            elif not halted and target is not None and ref is not None:
                ref_prev = ref
                ref = _step_pose(ref, target, v_lin, v_ang)
                d = ref[:3, 3] - j.ref0[:3, 3]
                w = Rotation.from_matrix(ref[:3, :3] @ j.ref0[:3, :3].T).as_rotvec()
                out = ctrl(
                    {
                        "enabled": 1.0,
                        "target_x": float(d[0]),
                        "target_y": float(d[1]),
                        "target_z": float(d[2]),
                        "target_wx": float(w[0]),
                        "target_wy": float(w[1]),
                        "target_wz": float(w[2]),
                        "gripper_pos": grip,
                    }
                )
                robot.send_action(_clamped(_trimmed(j, out), q_obs))
                q_cmd = {m: float(out[f"{m}.pos"]) for m in MOTOR_NAMES}  # the goal itself, untrimmed
                # A held tick (no IK solution, or an implausible joint jump) must
                # not let the reference run ahead of the arm; it waits here and
                # tries the same step again next tick.
                holding = bool(ctrl.is_holding)
                if holding:
                    ref = ref_prev
            else:
                q_cmd, holding = None, False
            t_sent = time.perf_counter()
            loads, temps = state.load, state.temps
            with j.lock:
                t_end = time.perf_counter()
                j.tick_s.append(t_end - t0)
                if t_end - t0 > SLOW_TICK_S:
                    marks = (t0, t_locked, t_read, t_sent, t_end)
                    phases = ("lock", "read_state", "command", "lock_again")
                    j.slow_ticks.append(
                        {
                            "at": time.time(),
                            "ms": round((t_end - t0) * 1000.0, 1),
                            "mode": mode,
                            "phases_ms": {
                                p: round((b - a) * 1000.0, 1)
                                for p, a, b in zip(phases, marks[:-1], marks[1:], strict=True)
                            },
                        }
                    )
                if q_cmd is not None:
                    j.q_cmd, j.ref = q_cmd, ref
                j.holding = holding
                j.q_obs, j.temps, j.load = q_obs, temps, loads
                j.last_read_at = time.monotonic()
                j.ticks += 1
                if j.record is not None:
                    j.record.append(
                        {
                            "t": time.time() - j.record_t0,
                            "obs": dict(q_obs),
                            "cmd": dict(j.q_cmd),
                            "load": dict(loads),
                            "temps": dict(temps),
                        }
                    )
                if not j.halted:
                    # A human on the leader outruns the follower on purpose; only the walk is held to its command.
                    lag = (
                        0.0
                        if j.mode == "leader"
                        else max(abs(j.q_obs[m] - j.q_cmd[m]) for m in MOTOR_NAMES if m != "gripper")
                    )
                    if lag > DIVERGE_DEG:
                        j.halted, j.reason = True, f"joint {lag:.0f} deg behind its command — frozen"
        except Exception as e:  # the arm holds its last goal; the operator sees why
            logger.exception("jog loop stopped")
            with j.lock:
                j.halted, j.reason = True, f"loop error: {e}"
            return
        time.sleep(max(0.0, period - (time.perf_counter() - t0)))


def _connect(body: ConnectBody) -> dict:
    from lerobot.robots.so107_description.cartesian_ik import (
        SO107_WORKSPACE_MAX,
        SO107_WORKSPACE_MIN,
        CartesianIKController,
        make_so107_arm_kinematics,
    )
    from lerobot.robots.so107_description.joint_alignment import (
        LEFT_ARM_ALIGNMENT,
        MOTOR_NAMES,
        RIGHT_ARM_ALIGNMENT,
    )
    from lerobot.robots.so_follower import SO107Follower, SO107FollowerConfig

    from .robot import ROBOT_PROFILES_DIR

    path = ROBOT_PROFILES_DIR / f"{body.profile}.json"
    if not path.exists():
        raise HTTPException(404, f"no robot profile named {body.profile!r}")
    profile = json.loads(path.read_text())
    fields = profile.get("fields", {})
    kind = profile.get("type")
    if kind == "bi_so107_follower":
        port = str(fields[f"{body.arm}_arm_port"])
        motor_id = f"{fields.get('id', body.profile)}_{body.arm}"
    elif kind == "so107_follower":
        port = str(fields["port"])
        motor_id = str(fields.get("id", body.profile))
    else:
        raise HTTPException(422, f"profile {body.profile!r} is {kind!r}; jog needs an SO-107 profile")

    def pick(name, default):
        v = getattr(body, name)
        if v is not None:
            return v
        return fields.get(name, default)

    cfg = SO107FollowerConfig(
        id=motor_id,
        port=port,
        use_degrees=True,
        disable_torque_on_disconnect=False,
        max_relative_target=None,  # the jog clamps every goal itself, against its own checked read (_clamped)
        cameras={},
        p_coefficient=int(pick("p_coefficient", 16)),
        i_coefficient=int(pick("i_coefficient", 0)),
        d_coefficient=int(pick("d_coefficient", 32)),
        gravity_ff_alpha=float(pick("gravity_ff_alpha", 0.0)),
        gravity_ff_arm=body.arm,
    )
    alignment = LEFT_ARM_ALIGNMENT if body.arm == "left" else RIGHT_ARM_ALIGNMENT
    # A measured fingertip replaces the URDF's guess at the tip whenever one is saved.
    from lerobot.gui.config_paths import gui_config_dir
    from lerobot.robots.so107_description.joint_alignment import TIP_OFFSET

    from ._calib_core import (
        calibration_path,
        corrected_alignment,
        joint_zero_from_calibration,
        load_calibration,
        tip_offset_from_calibration,
    )

    calibration = load_calibration(calibration_path(gui_config_dir(), motor_id))
    tip_offset = tip_offset_from_calibration(calibration)
    joint_zero = joint_zero_from_calibration(calibration)
    # Zero corrections from the touch calibration fold into the motor->URDF alignment.
    alignment = corrected_alignment(alignment, joint_zero) if joint_zero else alignment
    kin = make_so107_arm_kinematics(alignment, tip_offset=tip_offset)
    # The workspace floor guards the table. Once the reference point is the real
    # fingertip, the floor is the surface the fingertip was calibrated on, not
    # the default that let the old hinge point hover above it.
    workspace_min = SO107_WORKSPACE_MIN
    if tip_offset is not None:
        surface_z = float(calibration["tool_point"]["point_m"][2])
        workspace_min = (
            SO107_WORKSPACE_MIN[0],
            SO107_WORKSPACE_MIN[1],
            min(SO107_WORKSPACE_MIN[2], surface_z - 0.005),
        )
    robot = SO107Follower(cfg)
    # A motor left latched by an overload fails the connect handshake; clear it first,
    # on the raw port, without touching the motors that are holding the arm.
    robot.bus._connect(handshake=False)
    try:
        _clear_latches(robot.bus)
    finally:
        robot.bus.port_handler.closePort()
    try:
        robot.connect(calibrate=False)
    except Exception:
        # A connect that fails partway leaves the port open in this process, which then holds the arm's port with no
        # arm connected (2026-10-08: a garbled reply to Torque_Enable on motor 4).
        with contextlib.suppress(Exception):
            robot.bus.port_handler.closePort()
        raise
    try:
        if not robot.is_calibrated:
            raise RuntimeError(f"arm {motor_id!r} reports uncalibrated")
        _limit_to_servos(kin, alignment, *_servo_ranges_of(robot.bus))  # no solve asks past a servo's stop
        protection = _read_protection(robot.bus)
        q0 = np.array([_read_checked(robot.bus, None).q[m] for m in MOTOR_NAMES], dtype=float)
        urdf_deg = np.array(
            [alignment[m].sign * q0[i] + alignment[m].offset_deg for i, m in enumerate(MOTOR_NAMES)]
        )
        # The kinematics wrapper clamps to the URDF limits; from outside them (the fold)
        # a joint-space ramp brings the arm to the working pose first.
        lo, hi = np.degrees(kin._inner._inner.q_lo), np.degrees(kin._inner._inner.q_hi)
        if np.any(urdf_deg[:6] < lo[:6] + 2) or np.any(urdf_deg[:6] > hi[:6] - 2):
            logger.info("jog: start pose outside the URDF limits; ramping to the ready pose first")
            _ramp_joints(robot, dict(READY_DEG))
            q0 = np.array([_read_checked(robot.bus, None).q[m] for m in MOTOR_NAMES], dtype=float)
        t0 = kin.forward_kinematics(q0)
        ctrl = CartesianIKController(
            kinematics=kin,
            motor_names=list(MOTOR_NAMES),
            q_init=q0,
            workspace_min=workspace_min,
            workspace_max=SO107_WORKSPACE_MAX,
            label=body.arm,
        )
    except Exception:
        robot.disconnect()
        raise
    global _jog
    j = _Jog(
        robot=robot,
        arm=body.arm,
        kin=kin,
        ctrl=ctrl,
        alignment=alignment,
        max_linear_m_s=_jog.max_linear_m_s,
        max_angular_rad_s=_jog.max_angular_rad_s,
        max_rot_delta_rad=_jog.max_rot_delta_rad,
        settle_on=_jog.settle_on,
        protection=protection,
        robot_id=motor_id,
        profile=body.profile,
        tip_offset=(TIP_OFFSET if tip_offset is None else tip_offset).copy(),
        tip_calibrated=tip_offset is not None,
        joint_zero_deg=dict(joint_zero),
        workspace_min=tuple(workspace_min),
    )
    j.q_cmd = {m: float(q0[i]) for i, m in enumerate(MOTOR_NAMES)}
    j.q_obs = dict(j.q_cmd)
    j.ref0, j.ref, j.target = t0.copy(), t0.copy(), t0.copy()
    j.gains = {
        "p": cfg.p_coefficient,
        "i": cfg.i_coefficient,
        "d": cfg.d_coefficient,
        "gravity_ff_alpha": cfg.gravity_ff_alpha,
    }
    j.thread = threading.Thread(target=_loop, args=(j,), daemon=True, name="jog-stream")
    _jog = j
    j.thread.start()
    return _state_locked(j)


STATE_FIELDS = (
    "Present_Position",
    "Present_Velocity",
    "Present_Load",
    "Present_Voltage",
    "Present_Temperature",
)
MAX_STEP_DEG = 12.0  # a goal at most this far from the joint's checked reading: the feed-forward lead and lag fit under it
RANGE_MARGIN_TICKS = 100  # a position this far past its joint's calibrated range is a misread, not the joint
MAX_JOINT_DEG_S = (
    600.0  # faster than a servo turns: a reading that moved further than this allows is a misread
)
JUMP_SLACK_DEG = 5.0
JUMP_DT_CAP_S = 0.05  # a reading after a longer gap is still held to the move this long allows
BAD_READS_FREEZE = 2  # bad reads in a row that freeze the arm


class _BadReadError(RuntimeError):
    """A read of the arm that cannot be the arm: a missing reply, one of another request's length, or positions or
    temperatures the motors cannot have. ``raw`` is each motor's reply as it came."""

    def __init__(self, why: str, raw: dict[str, list[int]]):
        super().__init__(why)
        self.raw = raw


@dataclass
class _ArmState:
    """One checked read of every motor (:func:`_read_state`)."""

    q: dict[str, float]  # normalized as the follower reads them: degrees, the gripper 0..100
    pos_raw: dict[str, int]  # Present_Position as the motor reported it
    load: dict[str, int]  # Present_Load: signed, 0.1 % of full drive
    temps: dict[str, int]
    errors: dict[str, int]  # each reply's status byte
    raw: dict[str, list[int]]
    at: float  # monotonic, when it was read


def _read_state(bus: Any) -> _ArmState:
    """Every motor's position, speed, load, voltage and temperature in one request, each reply taken whole and checked
    by the bus (:meth:`FeetechMotorsBus.sync_read_block`, which clears the port of late replies after a bad one).

    On 2026-10-10 the positions the follower read for its own clamp were the motors' load replies: a load read that
    timed out left them on the port, the next position read took them for its own (both two bytes, and the Feetech SDK
    slices a reply to the length asked without checking it), and the clamp sent the arm towards them. Pre: the bus is
    connected. Raises _BadReadError."""
    from lerobot.motors.feetech import BusReplyError

    try:
        block = bus.sync_read_block(STATE_FIELDS, normalize=False)
    except BusReplyError as e:
        raise _BadReadError(str(e), e.raw) from e
    ids = {name: bus.motors[name].id for name in block.raw}
    pos = block.values["Present_Position"]
    q = bus._normalize({ids[name]: v for name, v in pos.items()})
    names = {id_: name for name, id_ in ids.items()}
    return _ArmState(
        q={names[i]: float(v) for i, v in q.items()},
        pos_raw={name: int(v) for name, v in pos.items()},
        load={name: int(v) for name, v in block.values["Present_Load"].items()},
        temps={name: int(v) for name, v in block.values["Present_Temperature"].items()},
        errors=dict(block.errors),
        raw=block.raw,
        at=time.monotonic(),
    )


def _check_state(bus: Any, state: _ArmState, prev: _ArmState | None) -> None:
    """Positions the arm can have: each within its joint's calibrated range (a joint calibrated over the servo's whole
    turn has none to keep), and no further from the last good read than a servo turns in the time since. A reply's
    temperature judges nothing: the servos report wrong ones in otherwise whole replies (:data:`DIVERGE_DEG`'s note).
    ``prev`` is the last good read of the same run, or None. Raises _BadReadError."""
    for name, p in state.pos_raw.items():
        cal = bus.calibration[name]
        whole_turn = cal.range_max - cal.range_min >= 4095 - RANGE_MARGIN_TICKS
        if (
            not whole_turn
            and not cal.range_min - RANGE_MARGIN_TICKS <= p <= cal.range_max + RANGE_MARGIN_TICKS
        ):
            raise _BadReadError(
                f"{name} read at {p}, outside its calibrated {cal.range_min}..{cal.range_max}", state.raw
            )
    if prev is None:
        return
    allowed = JUMP_SLACK_DEG + MAX_JOINT_DEG_S * min(state.at - prev.at, JUMP_DT_CAP_S)
    for name, v in state.q.items():
        if abs(v - prev.q[name]) > allowed:
            raise _BadReadError(
                f"{name} read {v:.1f}, {abs(v - prev.q[name]):.0f} from the last read {state.at - prev.at:.2f} s before",
                state.raw,
            )


def _read_checked(bus: Any, prev: _ArmState | None, tries: int = BAD_READS_FREEZE) -> _ArmState:
    """A checked read (:func:`_read_state`, :func:`_check_state`), asked again after a bad one, up to ``tries``. Raises
    RuntimeError when every try was bad."""
    for k in range(tries):
        try:
            state = _read_state(bus)
            _check_state(bus, state, prev)
            return state
        except _BadReadError as e:
            logger.warning("jog: bad read %d of %d: %s; replies %s", k + 1, tries, e, e.raw)
            last = e
    raise RuntimeError(f"bad joint reads, {tries} in a row: {last}")


def _faults(bus: Any, state: _ArmState) -> None:
    """A motor reporting a fault stops the arm. A gripper squeezing an object flags an overload in every reply and keeps
    holding it at the follower's reduced protective torque, so that flag alone on the gripper is not a fault; on any
    other motor it still is. Raises RuntimeError."""
    for name, error in state.errors.items():
        if error and not (name == "gripper" and error == OVERLOAD_ERRBIT):
            raise RuntimeError(f"{name}: {bus.packet_handler.getRxPacketError(error)}")


def _clamped(action: dict[str, float], q: dict[str, float]) -> dict[str, float]:
    """Each joint's goal in ``action`` at most MAX_STEP_DEG from its checked reading ``q``: the follower's own clamp,
    which read the joints again unchecked, is off."""
    out = {}
    for key, v in action.items():
        m = key.removesuffix(".pos")
        out[key] = q[m] + float(np.clip(v - q[m], -MAX_STEP_DEG, MAX_STEP_DEG)) if m in q else v
    return out


PROTECTION_FIELDS = (
    "Torque_Limit",
    "Max_Torque_Limit",
    "Overload_Torque",
    "Protection_Time",
    "Protective_Torque",
    "Protection_Current",
)


def _read_protection(bus: Any) -> dict[str, dict[str, int]]:
    """Each motor's own torque and overload settings, as the servo holds them; a field it will not give is left out."""
    out: dict[str, dict[str, int]] = {}
    for name in bus.motors:
        fields: dict[str, int] = {}
        for f in PROTECTION_FIELDS:
            with contextlib.suppress(Exception):
                fields[f] = int(bus.read(f, name, normalize=False))
        out[name] = fields
    return out


def _clear_latches(bus: Any) -> list[str]:
    """Write Torque_Enable=0 straight to every motor that does not answer PING, and pin its goal where it is.

    Pre: the bus port is open. Post: returns the motors that answered after the
    clear; raises when one still stays silent. Motors that answered the first
    PING are not touched, so the rest of the arm keeps holding.
    """
    from lerobot.motors.feetech.feetech import TorqueMode
    from lerobot.motors.motors_bus import get_address

    cleared, unreachable = [], []
    for name, motor in bus.motors.items():
        if bus.ping(motor.id, num_retry=2) is not None:
            continue
        addr, length = get_address(bus.model_ctrl_table, motor.model, "Torque_Enable")
        bus._write(addr, length, motor.id, TorqueMode.DISABLED.value, num_retry=3, raise_on_error=False)
        if bus.ping(motor.id, num_retry=3) is None:
            unreachable.append(name)
            continue
        # The servo still holds the goal that overloaded it; make the goal its present position
        # before anything re-enables torque, or it lunges straight back into the obstacle.
        p_addr, p_len = get_address(bus.model_ctrl_table, motor.model, "Present_Position")
        g_addr, g_len = get_address(bus.model_ctrl_table, motor.model, "Goal_Position")
        present, comm, _err = bus._read(p_addr, p_len, motor.id, num_retry=3, raise_on_error=False)
        if comm == 0:
            bus._write(g_addr, g_len, motor.id, int(present), num_retry=3, raise_on_error=False)
        cleared.append(name)
    if unreachable:
        raise RuntimeError(
            f"still no answer from {unreachable} after clearing the latch; power-cycle the arm"
        )
    return cleared


def _recover(j: _Jog) -> dict:
    """Clear an overload latch on the jog's own bus and restart the walk from where the arm is.

    Only motors that fail PING are touched, so the rest of the arm keeps
    holding. Post: the loop is running again with a fresh reference at the
    observed pose; the operator's target starts there too.
    """
    _path_owned(j)  # recheck in the worker: a path may reserve the arm while this job is queued
    _stop_loop(j)
    with j.lock:
        grip = j.q_cmd.get("gripper")
    bus = j.robot.bus
    cleared = _clear_latches(bus)
    if cleared:
        bus.enable_torque(motors=cleared)
    # A responsive gripper may have stopped short against a held object. Re-anchoring its command at that observed
    # opening relaxes the grasp for the first resumed ticks, even if grip_target later closes it again. Keep the
    # last issued command, not a pending target. A gripper whose own latch was cleared retains recovery's existing
    # re-anchor behavior: do not immediately restore the command that faulted it.
    _restart_from_present(j, grip=None if "gripper" in cleared else grip)
    return {"cleared": cleared, "temps": dict(j.temps)}


def _stop_loop(j: _Jog) -> None:
    j.stop.set()
    if j.thread is not None and j.thread.is_alive():
        j.thread.join(timeout=2.0)


def _restart_from_present(j: _Jog, grip: float | None = None, require_unhalted: bool = False) -> None:
    """Re-anchor the Cartesian walk at the arm's present pose and start the loop. Pre: the loop is stopped.

    ``grip``, when given, is the gripper command to keep: holding an object, the jaws
    stop short of the command, and taking the observed opening as the new command
    would relax the grasp.
    """
    from lerobot.robots.so107_description.cartesian_ik import SO107_WORKSPACE_MAX, CartesianIKController
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    state = _read_checked(j.robot.bus, None)
    if require_unhalted:
        _faults(j.robot.bus, state)
    q_now = {m: state.q[m] for m in MOTOR_NAMES}
    q0 = np.array([q_now[m] for m in MOTOR_NAMES])
    t0 = j.kin.forward_kinematics(q0)
    ctrl = CartesianIKController(
        kinematics=j.kin,
        motor_names=list(MOTOR_NAMES),
        q_init=q0,
        workspace_min=j.workspace_min,
        workspace_max=SO107_WORKSPACE_MAX,
        label=j.arm,
    )
    q_cmd = dict(q_now)
    if grip is not None:
        q_cmd["gripper"] = float(grip)
    with j.lock:
        if require_unhalted and j.halted:
            raise RuntimeError(f"arm froze during handoff: {j.reason}")
        j.ctrl = ctrl
        j.q_cmd, j.q_obs = q_cmd, dict(q_now)
        if grip is not None:
            j.grip_target = float(grip)
        j.ref0, j.ref, j.target = t0.copy(), t0.copy(), t0.copy()
        j.halted, j.reason, j.holding = False, "", False
        j.good, j.bad_reads = state, 0
        j.stop = threading.Event()
        j.thread = threading.Thread(target=_loop, args=(j,), daemon=True, name="jog-stream")
    j.thread.start()


def _ramp_joints(robot: Any, target: dict[str, float], deg_s: float = RAMP_DEG_S, hz: float = 50.0) -> None:
    """Interpolate every listed joint from its present position to ``target`` at a bounded rate.

    Joint space, so it works from the fold where the IK cannot. Each step is read and checked first and its goal held to
    the reading (:func:`_clamped`); bad reads in a row stop the ramp with the arm on its last goal. Pre: the robot is
    connected and no loop is streaming to it. Post: the last goal sent is ``target``; the gripper is left where it is
    unless listed. Raises RuntimeError on bad reads.
    """
    state = _read_checked(robot.bus, None)
    start = {m: state.q[m] for m in target}
    span = max(abs(target[m] - start[m]) for m in target)
    steps = max(1, int(round(span / deg_s * hz)))
    period = 1.0 / hz
    for i in range(1, steps + 1):
        t0 = time.perf_counter()
        a = i / steps
        state = _read_checked(robot.bus, state)
        robot.send_action(
            _clamped({f"{m}.pos": start[m] * (1.0 - a) + target[m] * a for m in target}, state.q)
        )
        time.sleep(max(0.0, period - (time.perf_counter() - t0)))


def _ready(j: _Jog) -> dict:
    """Bring the arm to the working pose in joint space, then re-anchor the walk there."""
    _path_owned(j)  # recheck in the worker: a path may reserve the arm while this job is queued
    _stop_loop(j)
    _ramp_joints(j.robot, dict(READY_DEG))
    _restart_from_present(j)
    return {"pose": dict(READY_DEG)}


def _park(j: _Jog, rest: dict[str, float]) -> dict:
    """Fold the arm to its profile's rest pose, release torque there, and disconnect the jog."""
    _path_owned(j)  # recheck in the worker: a path may reserve the arm while this job is queued
    _stop_loop(j)
    _drop_leader(j)
    _ramp_joints(j.robot, rest)
    j.robot.bus.disable_torque()
    j.robot.disconnect()
    return {"pose": rest}


def _drop_leader(j: _Jog) -> None:
    """Release the leader arm, if one is attached. Pre: the loop is stopped."""
    leader = j.leader
    with j.lock:
        j.leader, j.leader_id, j.mode = None, "", "cartesian"
    if leader is not None:
        with contextlib.suppress(Exception):
            leader.disconnect()


def _leader_start(j: _Jog, profile: str, arm: str) -> dict:
    """Hand the follower to a leader arm: meet the leader's pose at the ramp rate, then pass its joints through.

    Pre: the jog is connected and walking. Post: the loop runs in leader mode; the
    Cartesian target is parked until :func:`_leader_stop` re-anchors it.
    """
    _path_owned(j)  # recheck in the worker: a path may reserve the arm while this job is queued
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES
    from lerobot.teleoperators.so_leader import SO107Leader, SO107LeaderConfig

    from .robot import TELEOP_PROFILES_DIR

    path = TELEOP_PROFILES_DIR / f"{profile}.json"
    if not path.exists():
        raise HTTPException(404, f"no teleop profile named {profile!r}")
    spec = json.loads(path.read_text())
    fields, kind = spec.get("fields", {}), spec.get("type", "")
    if kind.startswith("bi_so107_leader"):
        port, leader_id = str(fields[f"{arm}_arm_port"]), f"{fields.get('id', profile)}_{arm}"
    elif kind.startswith("so107_leader"):
        port, leader_id = str(fields["port"]), str(fields.get("id", profile))
    else:
        raise HTTPException(422, f"profile {profile!r} is {kind!r}; the demo needs an SO-107 leader")
    leader = SO107Leader(SO107LeaderConfig(id=leader_id, port=port, use_degrees=True))
    leader.connect(calibrate=False)
    try:
        if not leader.is_calibrated:
            raise RuntimeError(f"leader {leader_id!r} reports uncalibrated")
        act = leader.get_action()
        q_lead = {m: float(act[f"{m}.pos"]) for m in MOTOR_NAMES}
    except Exception:
        leader.disconnect()
        raise
    _stop_loop(j)
    _ramp_joints(j.robot, q_lead)  # meet the leader where it is; never snap to it
    with j.lock:
        j.leader, j.leader_id, j.mode = leader, leader_id, "leader"
        j.q_cmd = dict(q_lead)
        j.halted, j.reason, j.holding = False, "", False
        j.stop = threading.Event()
        j.thread = threading.Thread(target=_loop, args=(j,), daemon=True, name="jog-stream")
    j.thread.start()
    return {"leader": leader_id, "mode": "leader"}


def _leader_stop(j: _Jog) -> dict:
    """Take the follower back: release the leader and re-anchor the Cartesian walk where the arm is."""
    _path_owned(j)  # recheck in the worker: a path may reserve the arm while this job is queued
    _stop_loop(j)
    _drop_leader(j)
    _restart_from_present(j)
    return {"mode": "cartesian"}


def _joints_start(j: _Jog, q_first: dict[str, float]) -> dict:
    """Hand the follower to joint targets: meet the first one at the ramp rate, then stream what :func:`set_target_joints` sets.

    Pre: the jog is connected and walking, not under a leader. Post: the loop runs
    in joints mode; the Cartesian target is parked until :func:`_joints_stop`
    re-anchors it where the arm ends up.
    """
    _path_owned(j)  # recheck in the worker: a path may reserve the arm while this job is queued
    _stop_loop(j)
    _ramp_joints(j.robot, dict(q_first))  # meet the first configuration; never snap to it
    with j.lock:
        j.mode, j.q_target, j.playback = "joints", dict(q_first), None
        j.q_cmd = dict(q_first)
        j.halted, j.reason, j.holding = False, "", False
        j.stop = threading.Event()
        j.thread = threading.Thread(target=_loop, args=(j,), daemon=True, name="jog-stream")
    j.thread.start()
    return {"mode": "joints"}


def _joints_stop(j: _Jog, *, from_guarded_path: bool = False) -> dict:
    """Hand back joint control, preserving the grasp and refusing to clear any freeze.

    Only the checked guarded-path finish may bypass its own ownership guard.
    Every handoff requires an unhalted arm; recovery is a separate operation.
    """
    if not from_guarded_path:
        _path_owned(j)
    _stop_loop(j)
    with j.lock:
        last = j.q_target
        j.mode, j.q_target, j.playback = "cartesian", None, None
    _restart_from_present(j, grip=None if last is None else last.get("gripper"), require_unhalted=True)
    return {"mode": "cartesian"}


async def joints_start(q_first: dict[str, float]) -> dict:
    """Ramp to ``q_first`` in joint space, then stream joint targets. Pre: connected and not under a leader."""
    j = _jog
    if not j.connected:
        raise RuntimeError("no arm connected")
    if j.path_busy or j.mode == "path":
        raise RuntimeError("a guarded path owns the arm")
    if j.mode == "leader":
        raise RuntimeError("the leader is driving the arm")
    return await asyncio.get_event_loop().run_in_executor(_EXECUTOR, _joints_start, j, q_first)


async def joints_stop() -> dict:
    """Back to the Cartesian walk, anchored where the arm is. Pre: connected."""
    j = _jog
    if not j.connected:
        raise RuntimeError("no arm connected")
    if j.path_busy or j.mode == "path":
        raise RuntimeError("a guarded path owns the arm")
    return await asyncio.get_event_loop().run_in_executor(_EXECUTOR, _joints_stop, j)


def set_target_joints(q: dict[str, float]) -> None:
    """The configuration the joints-mode loop streams from its next tick. Pre: joints mode, not frozen."""
    j = _jog
    with j.lock:
        if not j.connected:
            raise RuntimeError("no arm connected")
        if j.halted:
            raise RuntimeError(f"jog is frozen: {j.reason}")
        if j.mode != "joints":
            raise RuntimeError("the arm is not taking joint targets")
        j.q_target = dict(q)


def play_joints(samples: list[dict[str, float]], due_s: list[float]) -> None:
    """Hand the loop a stream of joint targets to play back on its own clock: in order, one per tick, each no earlier
    than its time from now. A stream set from another thread sample by sample loses the samples set while a tick is
    slow, and the arm takes only the newest: in an act of 2026-10-08 the fingers' opening and the lift away reached the
    arm in one command, and the gamepad was pulled over. Pre: joints mode (:func:`joints_start`), not frozen, one
    non-decreasing time per sample."""
    assert len(samples) == len(due_s), "one time per sample"
    j = _jog
    with j.lock:
        if not j.connected:
            raise RuntimeError("no arm connected")
        if j.halted:
            raise RuntimeError(f"jog is frozen: {j.reason}")
        if j.mode != "joints":
            raise RuntimeError("the arm is not taking joint targets")
        j.playback = [dict(s) for s in samples]
        j.playback_due = [float(t) for t in due_s]
        j.playback_t0 = j.playback_tick = time.monotonic()
        j.playback_sent, j.playback_sent_at = -1, []
        j.playback_held, j.playback_blocked, j.playback_hold_spent = None, None, False


def playback_state() -> tuple[int, list[float]]:
    """How far the playback has gone: the last sample sent (-1 before the first) and when each was sent."""
    j = _jog
    with j.lock:
        return j.playback_sent, list(j.playback_sent_at)


def playback_stop() -> None:
    """End the playback where it is: the arm holds the last sample sent."""
    j = _jog
    with j.lock:
        j.playback = None


def walk_limits() -> tuple[float, float]:
    """The walk's speed limits the operator set: ``(metres per second, radians per second)``."""
    j = _jog
    with j.lock:
        return float(j.max_linear_m_s), float(j.max_angular_rad_s)


def set_walk_limits(linear_m_s: float, angular_rad_s: float) -> None:
    """Set the walk's speed caps, clamped to the ranges the operator's sliders allow."""
    j = _jog
    with j.lock:
        j.max_linear_m_s = float(np.clip(linear_m_s, *LINEAR_M_S_RANGE))
        j.max_angular_rad_s = float(np.clip(angular_rad_s, *ANGULAR_RAD_S_RANGE))


def workspace_box() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The fingertip box the walk clips to: ``(min, max)`` in metres, base frame; the floor is the calibrated table."""
    from lerobot.robots.so107_description.cartesian_ik import SO107_WORKSPACE_MAX

    j = _jog
    with j.lock:
        return tuple(float(v) for v in j.workspace_min), tuple(float(v) for v in SO107_WORKSPACE_MAX)  # type: ignore[return-value]


def kinematics() -> Any | None:
    """The connected arm's motor-space kinematics (``forward_kinematics``/``inverse_kinematics``), or None."""
    j = _jog
    with j.lock:
        return j.kin if j.connected else None


def servo_ranges() -> tuple[np.ndarray, np.ndarray] | None:
    """Each joint's range in the motor degrees the arm is driven in, ``(lo, hi)`` in MOTOR_NAMES order, or None when no
    arm is connected. A joint's degrees run from the middle of its servo's calibrated span, so it reaches half the span
    either way and the servo stops it there whatever it is asked (the model's own limits are wider). The gripper, in
    its own units, is NaN: no limit here."""

    j = _jog
    with j.lock:
        robot = j.robot if j.connected else None
    if robot is None:
        return None
    return _servo_ranges_of(robot.bus)


def _servo_ranges_of(bus: Any) -> tuple[np.ndarray, np.ndarray]:
    """:func:`servo_ranges` of a bus with its calibration loaded."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    lo, hi = np.full(len(MOTOR_NAMES), np.nan), np.full(len(MOTOR_NAMES), np.nan)
    for i, m in enumerate(MOTOR_NAMES):
        cal = bus.calibration.get(m)
        if m == "gripper" or cal is None:
            continue
        half = (
            (cal.range_max - cal.range_min)
            / 2.0
            * 360.0
            / (bus.model_resolution_table[bus.motors[m].model] - 1)
        )
        lo[i], hi[i] = -half, half
    return lo, hi


def _limit_to_servos(kin: Any, alignment: dict[str, Any], lo: np.ndarray, hi: np.ndarray) -> None:
    """Narrow the model's joint limits, which bound every IK solve made with ``kin`` (the walk's, the plans', the
    reach preview's), to the servos' calibrated ranges ``lo``/``hi`` (motor degrees, MOTOR_NAMES order, NaN for no
    limit) where those are tighter. The model's limits are 5-21 deg wider than the servos': a solve past a servo's
    range asked the arm for what the servo stops at anyway, and the servo pushed against its stop (the wrist roll's
    overloads; a walk asked for -101 deg against 93.3 on 2026-10-08). A joint's degrees map to the model's by the
    alignment, ``urdf = sign * motor + offset``."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    inner = kin._inner._inner
    model = inner.robot.model
    lower, upper = model.lowerPositionLimit.copy(), model.upperPositionLimit.copy()
    for i, m in enumerate(MOTOR_NAMES):
        if i >= len(inner.q_lo) or not (np.isfinite(lo[i]) and np.isfinite(hi[i])):
            continue
        a, b = sorted(
            (
                alignment[m].sign * lo[i] + alignment[m].offset_deg,
                alignment[m].sign * hi[i] + alignment[m].offset_deg,
            )
        )
        inner.q_lo[i] = max(inner.q_lo[i], np.radians(a))
        inner.q_hi[i] = min(inner.q_hi[i], np.radians(b))
        lower[i], upper[i] = inner.q_lo[i], inner.q_hi[i]
    model.lowerPositionLimit, model.upperPositionLimit = lower, upper


def _disconnect(j: _Jog) -> None:
    j.stop.set()
    if j.thread is not None:
        j.thread.join(timeout=2.0)
    _drop_leader(j)
    if j.robot is not None:
        j.robot.disconnect()


def current_tip_and_anchor() -> tuple[np.ndarray, np.ndarray, dict[str, float]] | None:
    """FK of the arm as the encoders read it now: ``(tip 4x4, anchor 4x4, q_obs)``, or None when no arm is up.

    The anchor is the URDF link the tip offset hangs from; the touch calibrations
    need it because the tip is exactly what they are measuring.
    """
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    j = _jog
    with j.lock:
        if not j.connected or not j.q_obs:
            return None
        q_obs, tip_offset = dict(j.q_obs), j.tip_offset
    assert tip_offset is not None, "a connected jog carries its tip offset"
    t_tip = j.kin.forward_kinematics(np.array([q_obs[m] for m in MOTOR_NAMES]))
    return t_tip, t_tip @ np.linalg.inv(tip_offset), q_obs


def current_robot_id() -> str | None:
    j = _jog
    with j.lock:
        return j.robot_id if j.connected else None


def set_target_pose(pose: np.ndarray) -> None:
    """Point the walk at a base-frame tip pose (4x4), as a gizmo drag would. Pre: an arm is connected and not frozen."""
    j = _jog
    with j.lock:
        if not j.connected:
            raise RuntimeError("no arm connected")
        if j.path_busy or j.mode == "path":
            raise RuntimeError("a guarded path owns the arm")
        if j.halted:
            raise RuntimeError(f"jog is frozen: {j.reason}")
        j.target = np.asarray(pose, dtype=float).copy()


def current_arm() -> str | None:
    j = _jog
    with j.lock:
        return j.arm if j.connected else None


def current_calibration_state() -> dict[str, Any]:
    j = _jog
    with j.lock:
        return {
            "tip_calibrated": bool(j.connected and j.tip_calibrated),
            "joint_zero_deg": dict(j.joint_zero_deg),
        }


def current_gripper() -> float | None:
    j = _jog
    with j.lock:
        return float(j.q_obs["gripper"]) if j.connected and j.q_obs else None


def set_gripper(pos: float) -> None:
    """Ask for a gripper opening (0..100); the loop walks there. Pre: an arm is connected and not frozen."""
    j = _jog
    with j.lock:
        if not j.connected:
            raise RuntimeError("no arm connected")
        if j.path_busy or j.mode == "path":
            raise RuntimeError("a guarded path owns the arm")
        if j.halted:
            raise RuntimeError(f"jog is frozen: {j.reason}")
        j.grip_target = float(np.clip(pos, 0.0, 100.0))


def current_status() -> dict[str, Any]:
    """What a sequence needs to know between ticks: holding, frozen, the gripper's reading, the mode, the joints
    commanded and read."""
    j = _jog
    with j.lock:
        if not j.connected:
            return {"connected": False}
        return {
            "connected": True,
            "holding": j.holding,
            "halted": j.halted,
            "reason": j.reason,
            "gripper_obs": float(j.q_obs["gripper"]) if j.q_obs else None,
            "mode": j.mode,
            "q_cmd": dict(j.q_cmd),
            "q_obs": dict(j.q_obs),
        }


def take_record() -> list[dict[str, Any]]:
    """The last recorded demo: samples of ``{"t", "obs", "cmd"}`` at the loop rate, oldest first."""
    j = _jog
    with j.lock:
        return list(j.last_record)


def fk_tip(q_motor: dict[str, float]) -> np.ndarray:
    """The fingertip pose (base frame, 4x4) for motor positions, with the connected arm's model. Pre: connected."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    j = _jog
    if not j.connected:
        raise RuntimeError("no arm connected")
    return j.kin.forward_kinematics(np.array([q_motor[m] for m in MOTOR_NAMES], dtype=float))


def current_tip_calibrated() -> bool:
    j = _jog
    with j.lock:
        return bool(j.connected and j.tip_calibrated)


def _state_locked(j: _Jog) -> dict:
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    with j.lock:
        if not j.connected:
            return {"connected": False}
        q_cmd, q_obs = dict(j.q_cmd), dict(j.q_obs)
        halted, reason, temps, ref, tgt = j.halted, j.reason, dict(j.temps), j.ref, j.target
        holding = j.holding
        tick_s, slow = np.array(j.tick_s), list(j.slow_ticks)
    ticks_ms = (
        {
            "n": int(len(tick_s)),
            "p50": float(np.percentile(tick_s, 50) * 1000.0),
            "p95": float(np.percentile(tick_s, 95) * 1000.0),
            "max": float(tick_s.max() * 1000.0),
            "slow": int((tick_s > SLOW_TICK_S).sum()),
        }
        if len(tick_s)
        else None
    )
    t_cmd = j.kin.forward_kinematics(np.array([q_cmd[m] for m in MOTOR_NAMES]))
    t_obs = j.kin.forward_kinematics(np.array([q_obs[m] for m in MOTOR_NAMES]))
    err_mm = float(np.linalg.norm(t_cmd[:3, 3] - t_obs[:3, 3]) * 1000.0)
    from scipy.spatial.transform import Rotation

    err_deg = float(
        np.degrees(np.linalg.norm(Rotation.from_matrix(t_cmd[:3, :3] @ t_obs[:3, :3].T).as_rotvec()))
    )
    ff = getattr(j.robot, "_gravity_ff", None)
    offsets = ff.offset_deg(q_cmd) if ff is not None else {}
    return {
        "connected": True,
        "arm": j.arm,
        "robot_id": j.robot_id,
        "tip_offset_mm": (j.tip_offset[:3, 3] * 1000.0).tolist() if j.tip_offset is not None else None,
        "tip_calibrated": j.tip_calibrated,
        "joint_zero_deg": dict(j.joint_zero_deg),
        "workspace_min_mm": [v * 1000.0 for v in j.workspace_min],
        "halted": halted,
        "reason": reason,
        "holding": holding,
        "gains": j.gains,
        "temps": temps,
        "q_cmd": q_cmd,
        "q_obs": q_obs,
        "urdf_cmd": _urdf_rad(j.alignment, q_cmd),
        "urdf_obs": _urdf_rad(j.alignment, q_obs),
        "tip_cmd": {"position": t_cmd[:3, 3].tolist(), "quaternion": _quat_from_matrix(t_cmd[:3, :3])},
        "tip_obs": {"position": t_obs[:3, 3].tolist(), "quaternion": _quat_from_matrix(t_obs[:3, :3])},
        "tip_ref": {"position": ref[:3, 3].tolist(), "quaternion": _quat_from_matrix(ref[:3, :3])},
        "tip_target": {"position": tgt[:3, 3].tolist(), "quaternion": _quat_from_matrix(tgt[:3, :3])},
        "err_mm": err_mm,
        "err_deg": err_deg,
        "ff_offsets_deg": offsets,
        "trim_deg": dict(j.settle.trim),
        "settle_on": j.settle_on,
        "load": dict(j.load),
        "protection": j.protection,
        "gripper": {"cmd": q_cmd["gripper"], "obs": q_obs["gripper"], "target": j.grip_target},
        "mode": j.mode,
        "leader": j.leader_id or None,
        "recording": j.record is not None,
        "record_n": len(j.record) if j.record is not None else len(j.last_record),
        "ticks_ms": ticks_ms,
        "slow_ticks": slow[-10:],
        "limits": {
            "linear_mm_s": j.max_linear_m_s * 1000.0,
            "angular_deg_s": math.degrees(j.max_angular_rad_s),
            "rotation_cap_deg": math.degrees(j.max_rot_delta_rad),
        },
    }


@router.get("/meta")
async def meta() -> dict:
    """Tile identity, in the URDF tile's meta vocabulary. ``available`` only while an arm is connected."""
    from lerobot.gui.urdf_viz import resolve_robot

    j = _jog
    if not j.connected:
        return {"available": False}
    spec = resolve_robot(j.robot.observation_features.keys())
    if spec is None:
        return {"available": False}
    return {
        "available": True,
        "name": f"{spec.name} jog ({j.arm})",
        "urdf": f"/urdf-assets/{spec.urdf_url_path}",
        "urdf_right": None,
        "base_offsets": None,
        "bimanual": False,
        "sources": ["state", "command"],
        "ee_link": spec.ee_link,
    }


@router.get("/state")
async def state() -> dict:
    return _state_locked(_jog)


def _register_snapshot(j: _Jog) -> dict:
    """Read actual servo goals/limits while the sole bus owner is frozen.

    Pause its telemetry loop so a diagnostic response cannot be mistaken for
    the regular state packet. Resume reading afterward, without clearing the
    freeze, changing gains, sending a goal, or re-anchoring the controller.
    """
    groups = (
        (
            "Min_Position_Limit",
            "Max_Position_Limit",
            "Operating_Mode",
            "P_Coefficient",
            "I_Coefficient",
            "D_Coefficient",
        ),
        ("Torque_Enable", "Torque_Limit", "Goal_Position", *STATE_FIELDS),
    )
    try:
        _stop_loop(j)
        if j.thread is not None and j.thread.is_alive():
            raise RuntimeError("telemetry loop did not stop; diagnostic refused")
        with j.lock:
            if _jog is not j or not j.connected or not j.halted:
                raise RuntimeError("arm changed or resumed before diagnostic")
        bus = j.robot.bus
        # Reading the entire EEPROM-to-telemetry span failed repeatedly on the
        # live bus. Keep each reply bounded; goal and observed state still come
        # from the SAME runtime packet. Settings do not change while frozen.
        registers, errors = {}, {}
        for fields in groups:
            block = bus.sync_read_block(fields, normalize=False)
            registers.update(block.values)
            for name, error in block.errors.items():
                errors[name] = errors.get(name, 0) | error

        def normalized(field):
            by_id = {bus.motors[name].id: value for name, value in registers[field].items()}
            values = bus._normalize(by_id)
            return {name: float(values[motor.id]) for name, motor in bus.motors.items()}

        return {
            "registers": registers,
            "errors": errors,
            "q_goal": normalized("Goal_Position"),
            "q_observed": normalized("Present_Position"),
        }
    finally:
        with j.lock:
            if j.connected and (j.thread is None or not j.thread.is_alive()):
                j.stop = threading.Event()
                j.thread = threading.Thread(target=_loop, args=(j,), daemon=True, name="jog-diagnostic")
                j.thread.start()
            j.path_busy = False


@router.post("/registers")
async def register_snapshot() -> dict:
    j = _jog
    with j.lock:
        _path_owned(j)
        if not j.connected or not j.halted:
            raise HTTPException(409, "freeze the connected arm before reading diagnostic registers")
        j.path_busy = True
    try:
        return await asyncio.get_event_loop().run_in_executor(_EXECUTOR, _register_snapshot, j)
    except Exception as e:
        raise HTTPException(500, f"register diagnostic failed: {e}") from e


class PathBody(BaseModel):
    samples: list[list[float]]
    tip_poses: list[list[list[float]]]
    grip_contact: bool = False
    previous_id: str | None = None
    stream_samples: bool = False


class PathToken(BaseModel):
    id: str


def _path_owned(j: _Jog) -> None:
    if j.path_busy or j.mode == "path":
        raise HTTPException(409, "a guarded path owns the arm; finish or stop it first")


def _path_snapshot(j: _Jog) -> dict:
    with j.lock:
        path = j.guarded_path
        return {
            "connected": j.connected,
            "halted": j.halted,
            "reason": j.reason,
            "path": None if path is None else path.info(),
            "q_obs": dict(j.q_obs),
            "q_cmd": dict(j.q_cmd),
            "load": dict(j.load),
            "temps": dict(j.temps),
            "read_age_s": None if not j.last_read_at else time.monotonic() - j.last_read_at,
        }


def _guarded_ranges(j: _Jog) -> tuple[np.ndarray, np.ndarray]:
    """Intersection with model limits: FK clamps outside them, so servo ranges alone cannot validate a path."""
    lo, hi = _servo_ranges_of(j.robot.bus)
    inner = j.kin._inner._inner
    bounds = np.array(
        [
            j.kin._urdf_to_motor(np.rad2deg(inner.q_lo)),
            j.kin._urdf_to_motor(np.rad2deg(inner.q_hi)),
        ]
    )
    return np.maximum(lo, bounds.min(axis=0)), np.minimum(hi, bounds.max(axis=0))


def _start_guarded_path(j: _Jog, body: PathBody) -> dict:
    import copy
    import uuid

    from lerobot.robots.so107_description.cartesian_ik import SO107_WORKSPACE_MAX
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    stopped = False
    try:
        # FK mutates Pinocchio's data: validation must not share the live controller's copy.
        kin = copy.deepcopy(j.kin)
        path = GuardedPath.prepare(
            str(uuid.uuid4()),
            tuple(MOTOR_NAMES),
            body.samples,
            body.tip_poses,
            kin,
            _guarded_ranges(j),
            (j.workspace_min, SO107_WORKSPACE_MAX),
            MOTOR_NAMES.index("gripper"),
            body.grip_contact,
            time.monotonic(),
            stream_samples=body.stream_samples,
        )
        with j.lock:
            if _jog is not j or not j.connected or j.halted:
                raise RuntimeError("arm changed or froze during path validation")
            path.check_start(j.q_obs, j.q_cmd, kin)
        _stop_loop(j)
        stopped = True
        if j.thread is not None and j.thread.is_alive():
            raise RuntimeError("hardware loop did not stop; path refused")
        # The same GUI owner takes a fresh read. No ramp to the first sample, and no second serial connection.
        current = _read_checked(j.robot.bus, j.good)
        _faults(j.robot.bus, current)
        with j.lock:
            if j.halted or _jog is not j:
                raise RuntimeError("arm stopped during path handoff")
            path.check_start(current.q, j.q_cmd, kin)
            path.started = path.heartbeat_at = path.sent_at = time.monotonic()
            j.guarded_path, j.mode = path, "path"
            j.q_obs, j.temps, j.load, j.good = current.q, current.temps, current.load, current
            j.last_read_at = time.monotonic()
            j.q_target = dict(zip(MOTOR_NAMES, map(float, path.q[0]), strict=True))
            j.q_cmd = dict(j.q_target)
            j.playback = None
            j.stop = threading.Event()
            j.thread = threading.Thread(target=_loop, args=(j,), daemon=True, name="jog-path")
        j.thread.start()
        return _path_snapshot(j)
    except Exception as e:
        if stopped and (j.thread is None or not j.thread.is_alive()):
            # A bad read or a changed start cannot resume the previous moving target as an error side effect.
            with j.lock:
                j.halted, j.reason = True, f"path handoff refused: {e}"
                j.stop = threading.Event()
                j.thread = threading.Thread(target=_loop, args=(j,), daemon=True, name="jog-path-refused")
            j.thread.start()
        raise
    finally:
        with j.lock:
            j.path_busy = False


@router.post("/path/start")
async def path_start(body: PathBody) -> dict:
    """Execute one externally collision-checked segment with observed-arrival and telemetry guards.

    This endpoint does not certify collision clearance. The client must retain a current scene, check the arm and
    payload, and heartbeat while monitoring observed geometry. Other motion writers are excluded until finish/stop.
    stream_samples opts into continuous arm transit between collision samples, with observed arrival still required
    at the segment endpoint. It retains speed/following-error bounds and does not change gripper contact semantics.
    """
    j = _jog
    with j.lock:
        if not j.connected or j.halted or j.path_busy:
            raise HTTPException(409, "arm disconnected, frozen, or reserved")
        if j.mode == "path":
            if (
                j.guarded_path is None
                or j.guarded_path.status != "done"
                or body.previous_id != j.guarded_path.identity
            ):
                raise HTTPException(409, "previous path must finish before the next segment")
        elif j.mode != "cartesian":
            raise HTTPException(409, "another controller owns the arm")
        j.path_busy = True
    try:
        return await asyncio.get_event_loop().run_in_executor(_EXECUTOR, _start_guarded_path, j, body)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(409, str(e)) from e


@router.get("/path/state")
async def path_state() -> dict:
    return _path_snapshot(_jog)


@router.post("/path/heartbeat")
async def path_heartbeat(body: PathToken) -> dict:
    j = _jog
    with j.lock:
        path = j.guarded_path
        if not j.connected or j.halted or j.mode != "path" or path is None or path.identity != body.id:
            raise HTTPException(409, "path is not running for this client")
        path.heartbeat_at = time.monotonic()
    return _path_snapshot(j)


@router.post("/path/stop")
async def path_stop(body: PathToken) -> dict:
    j = _jog
    with j.lock:
        path = j.guarded_path
        if path is None or path.identity != body.id:
            raise HTTPException(409, "path belongs to another client")
        if path.status != "failed":
            path.fail("stopped by path client")
        j.halted, j.reason = True, path.reason
        j.mode, j.q_target, j.playback = "cartesian", None, None
    return _path_snapshot(j)


@router.post("/path/finish")
async def path_finish(body: PathToken) -> dict:
    j = _jog
    with j.lock:
        path = j.guarded_path
        if (
            not j.connected
            or j.halted
            or j.path_busy
            or j.mode != "path"
            or path is None
            or path.identity != body.id
            or path.status != "done"
        ):
            raise HTTPException(409, "only an arrived path can hand control back")
        j.path_busy = True
    try:
        return await asyncio.get_event_loop().run_in_executor(_EXECUTOR, _finish_guarded_path, j)
    finally:
        with j.lock:
            j.path_busy = False


def _finish_guarded_path(j: _Jog) -> dict:
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    _stop_loop(j)
    if j.thread is not None and j.thread.is_alive():
        raise HTTPException(409, "hardware loop did not stop")
    try:
        current = _read_checked(j.robot.bus, j.good)
        _faults(j.robot.bus, current)
        with j.lock:
            path = j.guarded_path
            if j.halted or path is None:
                raise RuntimeError("arm froze before path handoff")
            pose = path.kin.forward_kinematics(np.array([current.q[n] for n in MOTOR_NAMES]))
            path.tick(time.monotonic(), current.q, pose, current.load, current.temps)
            if path.status != "done":
                raise RuntimeError(path.reason or "arm left the path's final pose")
        return _joints_stop(j, from_guarded_path=True)
    except Exception as e:
        with j.lock:
            j.halted, j.reason = True, f"path finish refused: {e}"
            if j.guarded_path is not None:
                j.guarded_path.fail(j.reason)
            j.stop = threading.Event()
            j.thread = threading.Thread(target=_loop, args=(j,), daemon=True, name="jog-path-refused")
        j.thread.start()
        raise HTTPException(409, j.reason) from e


@router.post("/connect")
async def connect(body: ConnectBody) -> dict:
    if body.arm not in ("left", "right"):
        raise HTTPException(422, "arm must be 'left' or 'right'")
    if _jog.connected:
        raise HTTPException(409, "an arm is already connected to the jog — disconnect it first")
    try:
        return await asyncio.get_event_loop().run_in_executor(_EXECUTOR, _connect, body)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"jog connect failed: {e}") from e


@router.post("/target")
async def target(body: TargetBody) -> dict:
    j = _jog
    _path_owned(j)
    if not j.connected:
        raise HTTPException(409, "no arm connected")
    if len(body.position) != 3 or len(body.quaternion) != 4:
        raise HTTPException(422, "position needs 3 values, quaternion 4")
    pose = np.eye(4)
    pose[:3, :3] = _matrix_from_quat(body.quaternion)
    pose[:3, 3] = body.position
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    with j.lock:
        if j.halted:
            raise HTTPException(409, f"jog is frozen: {j.reason} — disconnect and reconnect")
        q_obs, cap = dict(j.q_obs), j.max_rot_delta_rad
    # The cap is measured from where the arm is, not from where the gizmo was.
    t_obs = j.kin.forward_kinematics(np.array([q_obs[m] for m in MOTOR_NAMES]))
    pose, clamped = _cap_rotation(pose, t_obs[:3, :3], cap)
    with j.lock:
        j.target = pose
    return {
        "status": "ok",
        "clamped": clamped,
        "target": {"position": pose[:3, 3].tolist(), "quaternion": _quat_from_matrix(pose[:3, :3])},
    }


@router.post("/gripper")
async def gripper(body: GripperBody) -> dict:
    """Ask for a gripper opening in the follower's 0..100 units; the loop walks there at a bounded rate."""
    j = _jog
    _path_owned(j)
    if not j.connected:
        raise HTTPException(409, "no arm connected")
    if not 0.0 <= body.pos <= 100.0:
        raise HTTPException(422, "gripper position is 0..100")
    with j.lock:
        if j.halted:
            raise HTTPException(409, f"jog is frozen: {j.reason}")
        j.grip_target = float(body.pos)
    return {"pos": float(body.pos)}


@router.post("/limits")
async def limits(body: LimitsBody) -> dict:
    """Set the reference walk's speed caps; takes effect on the next tick, connected or not."""
    j = _jog
    _path_owned(j)
    with j.lock:
        if body.linear_mm_s is not None:
            v = body.linear_mm_s / 1000.0
            if not LINEAR_M_S_RANGE[0] <= v <= LINEAR_M_S_RANGE[1]:
                raise HTTPException(422, f"linear speed must be within {LINEAR_M_S_RANGE} m/s")
            j.max_linear_m_s = v
        if body.angular_deg_s is not None:
            w = math.radians(body.angular_deg_s)
            if not ANGULAR_RAD_S_RANGE[0] <= w <= ANGULAR_RAD_S_RANGE[1]:
                raise HTTPException(422, "angular speed must be within 5..180 deg/s")
            j.max_angular_rad_s = w
        if body.rotation_cap_deg is not None:
            c = math.radians(body.rotation_cap_deg)
            if not ROT_DELTA_RAD_RANGE[0] <= c <= ROT_DELTA_RAD_RANGE[1]:
                raise HTTPException(422, "rotation cap must be within 10..150 deg")
            j.max_rot_delta_rad = c
        if body.settle is not None:
            j.settle_on = bool(body.settle)
        return {
            "linear_mm_s": j.max_linear_m_s * 1000.0,
            "angular_deg_s": math.degrees(j.max_angular_rad_s),
            "rotation_cap_deg": math.degrees(j.max_rot_delta_rad),
        }


@router.post("/ready")
async def ready() -> dict:
    """Joint-space move to the working pose; the gizmo re-anchors there."""
    j = _jog
    _path_owned(j)
    if not j.connected:
        raise HTTPException(409, "no arm connected")
    try:
        return await asyncio.get_event_loop().run_in_executor(_EXECUTOR, _ready, j)
    except Exception as e:
        raise HTTPException(500, f"ready failed: {e}") from e


@router.post("/park")
async def park() -> dict:
    """Fold the arm to its profile's rest pose, release torque, disconnect. Connect again to resume."""
    global _jog
    j = _jog
    _path_owned(j)
    if not j.connected:
        raise HTTPException(409, "no arm connected")
    rest = _rest_pose_for(j)
    if rest is None:
        raise HTTPException(
            409, "the profile has no recorded rest position for this arm (Robot tab -> record rest)"
        )
    _jog = _Jog(
        max_linear_m_s=j.max_linear_m_s,
        max_angular_rad_s=j.max_angular_rad_s,
        max_rot_delta_rad=j.max_rot_delta_rad,
    )
    try:
        return await asyncio.get_event_loop().run_in_executor(_EXECUTOR, _park, j, rest)
    except Exception as e:
        raise HTTPException(500, f"park failed: {e}") from e


def _rest_pose_for(j: _Jog) -> dict[str, float] | None:
    """This arm's joints from the profile's recorded rest position, keyed by motor name."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    from .robot import ROBOT_PROFILES_DIR

    path = ROBOT_PROFILES_DIR / f"{j.profile}.json"
    if not path.exists():
        return None
    rest = json.loads(path.read_text()).get("rest_position") or {}
    out = {}
    for m in MOTOR_NAMES:
        for key in (f"{j.arm}_{m}.pos", f"{m}.pos"):
            if key in rest:
                out[m] = float(rest[key])
                break
    return out if len(out) == len(MOTOR_NAMES) else None


@router.post("/recover")
async def recover() -> dict:
    """Clear a motor overload latch and resume from the arm's present pose. The gizmo re-anchors there."""
    j = _jog
    _path_owned(j)
    if not j.connected:
        raise HTTPException(409, "no arm connected")
    try:
        return await asyncio.get_event_loop().run_in_executor(_EXECUTOR, _recover, j)
    except Exception as e:
        raise HTTPException(500, f"recover failed: {e}") from e


class LeaderBody(BaseModel):
    profile: str  # a saved SO-107 leader profile; the page offers the saved ones
    arm: str  # "left" or "right"


@router.post("/leader/start")
async def leader_start(body: LeaderBody) -> dict:
    """Demo mode: the named leader arm drives the follower until ``/leader/stop``."""
    j = _jog
    _path_owned(j)
    if not j.connected:
        raise HTTPException(409, "no arm connected")
    if j.mode == "leader":
        raise HTTPException(409, "the leader is already driving")
    if j.mode == "joints":
        raise HTTPException(409, "an act is replaying; stop it first")
    if body.arm not in ("left", "right"):
        raise HTTPException(422, "arm must be 'left' or 'right'")
    try:
        return await asyncio.get_event_loop().run_in_executor(
            _EXECUTOR, _leader_start, j, body.profile, body.arm
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"leader connect failed: {e}") from e


@router.post("/leader/stop")
async def leader_stop() -> dict:
    j = _jog
    if not j.connected:
        raise HTTPException(409, "no arm connected")
    if j.mode != "leader":
        raise HTTPException(409, "the leader is not driving")
    return await asyncio.get_event_loop().run_in_executor(_EXECUTOR, _leader_stop, j)


def start_record() -> float:
    """Begin recording the follower's joints at the loop rate, in any mode. Pre: connected. Post: the wall-clock start."""
    j = _jog
    if not j.connected:
        raise RuntimeError("no arm connected")
    with j.lock:
        j.record, j.record_t0 = [], time.time()
        return j.record_t0


def stop_record() -> list[dict[str, Any]]:
    """End the recording and return its samples, oldest first; they are also kept as the last record."""
    j = _jog
    with j.lock:
        rec = j.record
        j.record = None
        if rec is None:
            raise RuntimeError("not recording")
        j.last_record = rec
        return list(rec)


def record_count() -> int:
    j = _jog
    with j.lock:
        return len(j.record) if j.record is not None else 0


@router.post("/record/start")
async def record_start() -> dict:
    """Record the follower's joints at the loop rate (any mode) until ``/record/stop``."""
    try:
        start_record()
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {"status": "recording"}


@router.post("/record/stop")
async def record_stop() -> dict:
    if not _jog.connected:
        raise HTTPException(409, "no arm connected")
    try:
        rec = stop_record()
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {"n": len(rec), "seconds": rec[-1]["t"] if rec else 0.0}


@router.get("/record")
async def record() -> dict:
    return {"samples": take_record()}


@router.post("/stop")
async def stop() -> dict:
    """Freeze at the current command (torque stays on). Reconnect to resume."""
    j = _jog
    with j.lock:
        if j.connected:
            j.halted, j.reason = True, "stopped by the operator"
    return {"status": "stopped"}


@router.post("/disconnect")
async def disconnect() -> dict:
    global _jog
    j = _jog
    if not j.connected:
        return {"status": "ok"}
    _jog = _Jog(
        max_linear_m_s=j.max_linear_m_s,
        max_angular_rad_s=j.max_angular_rad_s,
        max_rot_delta_rad=j.max_rot_delta_rad,
    )
    await asyncio.get_event_loop().run_in_executor(_EXECUTOR, _disconnect, j)
    return {"status": "ok"}
