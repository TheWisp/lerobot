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
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jog", tags=["jog"])
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="jog")

HZ = 30.0
MAX_LINEAR_M_S = 0.04  # default walk speed; the operator moves it with a slider
MAX_ANGULAR_RAD_S = math.radians(30.0)
LINEAR_M_S_RANGE = (0.005, 0.30)
ANGULAR_RAD_S_RANGE = (math.radians(5.0), math.radians(180.0))
# A drag may ask for at most this much rotation away from where the arm is
# now: one bounded move at a time, never a wrist flip the IK has to invent.
MAX_ROT_DELTA_RAD = math.radians(60.0)
ROT_DELTA_RAD_RANGE = (math.radians(10.0), math.radians(150.0))
DIVERGE_DEG = 25.0  # a joint this far behind its command is stalled or blocked: freeze
MAX_TEMP_C = 60
TEMP_EVERY_TICKS = 60
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
    mode: str = "cartesian"  # "cartesian" (the IK walk) | "leader"
    record: list[dict[str, Any]] | None = None  # samples while recording
    record_t0: float = 0.0
    last_record: list[dict[str, Any]] = field(default_factory=list)

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
                target, ref, halted = j.target, j.ref, j.halted
                v_lin, v_ang = j.max_linear_m_s, j.max_angular_rad_s
                grip_target = j.grip_target
                ctrl = j.ctrl
                mode, leader = j.mode, j.leader
            if grip_target is not None:
                step = GRIP_UNITS_S / HZ
                grip += float(np.clip(grip_target - grip, -step, step))
            if mode == "leader" and leader is not None:
                # The human drives: the leader's joints go straight to the follower, gripper included.
                act = leader.get_action()
                q_lead = {m: float(act[f"{m}.pos"]) for m in MOTOR_NAMES}
                robot.send_action({f"{m}.pos": q_lead[m] for m in MOTOR_NAMES})
                q_cmd, holding, grip = q_lead, False, q_lead["gripper"]
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
                robot.send_action(out)
                q_cmd = {m: float(out[f"{m}.pos"]) for m in MOTOR_NAMES}
                # A held tick (no IK solution, or an implausible joint jump) must
                # not let the reference run ahead of the arm; it waits here and
                # tries the same step again next tick.
                holding = bool(ctrl.is_holding)
                if holding:
                    ref = ref_prev
            else:
                q_cmd, holding = None, False
            obs = robot.get_observation()
            q_obs = {m: float(obs[f"{m}.pos"]) for m in MOTOR_NAMES}
            temps = j.temps
            if j.ticks % TEMP_EVERY_TICKS == 0:
                temps = {
                    m: int(robot.bus.read("Present_Temperature", m, normalize=False)) for m in MOTOR_NAMES
                }
            with j.lock:
                if q_cmd is not None:
                    j.q_cmd, j.ref = q_cmd, ref
                j.holding = holding
                j.q_obs, j.temps = q_obs, temps
                j.ticks += 1
                if j.record is not None:
                    j.record.append(
                        {"t": time.time() - j.record_t0, "obs": dict(q_obs), "cmd": dict(j.q_cmd)}
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
                    elif temps and max(temps.values()) > MAX_TEMP_C:
                        j.halted, j.reason = True, f"motor at {max(temps.values())} C — frozen"
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
        max_relative_target=12.0,  # the feed-forward lead plus tracking lag must fit under this
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
    robot.connect(calibrate=False)
    try:
        if not robot.is_calibrated:
            raise RuntimeError(f"arm {motor_id!r} reports uncalibrated")
        obs = robot.get_observation()
        q0 = np.array([obs[f"{m}.pos"] for m in MOTOR_NAMES], dtype=float)
        urdf_deg = np.array(
            [alignment[m].sign * q0[i] + alignment[m].offset_deg for i, m in enumerate(MOTOR_NAMES)]
        )
        # The kinematics wrapper clamps to the URDF limits; from outside them (the fold)
        # a joint-space ramp brings the arm to the working pose first.
        lo, hi = np.degrees(kin._inner._inner.q_lo), np.degrees(kin._inner._inner.q_hi)
        if np.any(urdf_deg[:6] < lo[:6] + 2) or np.any(urdf_deg[:6] > hi[:6] - 2):
            logger.info("jog: start pose outside the URDF limits; ramping to the ready pose first")
            _ramp_joints(robot, dict(READY_DEG))
            obs = robot.get_observation()
            q0 = np.array([obs[f"{m}.pos"] for m in MOTOR_NAMES], dtype=float)
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
    _stop_loop(j)
    bus = j.robot.bus
    cleared = _clear_latches(bus)
    if cleared:
        bus.enable_torque(motors=cleared)
    _restart_from_present(j)
    return {"cleared": cleared, "temps": dict(j.temps)}


def _stop_loop(j: _Jog) -> None:
    j.stop.set()
    if j.thread is not None and j.thread.is_alive():
        j.thread.join(timeout=2.0)


def _restart_from_present(j: _Jog) -> None:
    """Re-anchor the Cartesian walk at the arm's present pose and start the loop. Pre: the loop is stopped."""
    from lerobot.robots.so107_description.cartesian_ik import SO107_WORKSPACE_MAX, CartesianIKController
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    obs = j.robot.get_observation()
    q_now = {m: float(obs[f"{m}.pos"]) for m in MOTOR_NAMES}
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
    with j.lock:
        j.ctrl = ctrl
        j.q_cmd, j.q_obs = dict(q_now), dict(q_now)
        j.ref0, j.ref, j.target = t0.copy(), t0.copy(), t0.copy()
        j.halted, j.reason, j.holding = False, "", False
        j.stop = threading.Event()
        j.thread = threading.Thread(target=_loop, args=(j,), daemon=True, name="jog-stream")
    j.thread.start()


def _ramp_joints(robot: Any, target: dict[str, float], deg_s: float = RAMP_DEG_S, hz: float = 50.0) -> None:
    """Interpolate every listed joint from its present position to ``target`` at a bounded rate.

    Joint space, so it works from the fold where the IK cannot. Pre: the robot
    is connected and no loop is streaming to it. Post: the last goal sent is
    ``target``; the gripper is left where it is unless listed.
    """
    obs = robot.get_observation()
    start = {m: float(obs[f"{m}.pos"]) for m in target}
    span = max(abs(target[m] - start[m]) for m in target)
    steps = max(1, int(round(span / deg_s * hz)))
    period = 1.0 / hz
    for i in range(1, steps + 1):
        t0 = time.perf_counter()
        a = i / steps
        robot.send_action({f"{m}.pos": start[m] * (1.0 - a) + target[m] * a for m in target})
        time.sleep(max(0.0, period - (time.perf_counter() - t0)))


def _ready(j: _Jog) -> dict:
    """Bring the arm to the working pose in joint space, then re-anchor the walk there."""
    _stop_loop(j)
    _ramp_joints(j.robot, dict(READY_DEG))
    _restart_from_present(j)
    return {"pose": dict(READY_DEG)}


def _park(j: _Jog, rest: dict[str, float]) -> dict:
    """Fold the arm to its profile's rest pose, release torque there, and disconnect the jog."""
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
    _stop_loop(j)
    _drop_leader(j)
    _restart_from_present(j)
    return {"mode": "cartesian"}


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
        if j.halted:
            raise RuntimeError(f"jog is frozen: {j.reason}")
        j.grip_target = float(np.clip(pos, 0.0, 100.0))


def current_status() -> dict[str, Any]:
    """What a sequence needs to know between ticks: holding, frozen, the gripper's reading, the mode."""
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
        "gripper": {"cmd": q_cmd["gripper"], "obs": q_obs["gripper"], "target": j.grip_target},
        "mode": j.mode,
        "leader": j.leader_id or None,
        "recording": j.record is not None,
        "record_n": len(j.record) if j.record is not None else len(j.last_record),
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
        return {
            "linear_mm_s": j.max_linear_m_s * 1000.0,
            "angular_deg_s": math.degrees(j.max_angular_rad_s),
            "rotation_cap_deg": math.degrees(j.max_rot_delta_rad),
        }


@router.post("/ready")
async def ready() -> dict:
    """Joint-space move to the working pose; the gizmo re-anchors there."""
    j = _jog
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
    if not j.connected:
        raise HTTPException(409, "no arm connected")
    try:
        return await asyncio.get_event_loop().run_in_executor(_EXECUTOR, _recover, j)
    except Exception as e:
        raise HTTPException(500, f"recover failed: {e}") from e


class LeaderBody(BaseModel):
    profile: str = "blue"
    arm: str = "left"


@router.post("/leader/start")
async def leader_start(body: LeaderBody) -> dict:
    """Demo mode: the named leader arm drives the follower until ``/leader/stop``."""
    j = _jog
    if not j.connected:
        raise HTTPException(409, "no arm connected")
    if j.mode == "leader":
        raise HTTPException(409, "the leader is already driving")
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


@router.post("/record/start")
async def record_start() -> dict:
    """Record the follower's joints at the loop rate (any mode) until ``/record/stop``."""
    j = _jog
    if not j.connected:
        raise HTTPException(409, "no arm connected")
    with j.lock:
        j.record, j.record_t0 = [], time.time()
    return {"status": "recording"}


@router.post("/record/stop")
async def record_stop() -> dict:
    j = _jog
    if not j.connected:
        raise HTTPException(409, "no arm connected")
    with j.lock:
        rec = j.record
        j.record = None
        if rec is None:
            raise HTTPException(409, "not recording")
        j.last_record = rec
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
