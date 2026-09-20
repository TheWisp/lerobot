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

    robot, ctrl = j.robot, j.ctrl
    grip = float(j.q_cmd["gripper"])
    period = 1.0 / HZ
    while not j.stop.is_set():
        t0 = time.perf_counter()
        try:
            with j.lock:
                target, ref, halted = j.target, j.ref, j.halted
                v_lin, v_ang = j.max_linear_m_s, j.max_angular_rad_s
            if not halted and target is not None and ref is not None:
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
                if not j.halted:
                    lag = max(abs(j.q_obs[m] - j.q_cmd[m]) for m in MOTOR_NAMES if m != "gripper")
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
    kin = make_so107_arm_kinematics(alignment)
    robot = SO107Follower(cfg)
    robot.connect(calibrate=False)
    try:
        if not robot.is_calibrated:
            raise RuntimeError(f"arm {motor_id!r} reports uncalibrated")
        obs = robot.get_observation()
        q0 = np.array([obs[f"{m}.pos"] for m in MOTOR_NAMES], dtype=float)
        urdf_deg = np.array(
            [alignment[m].sign * q0[i] + alignment[m].offset_deg for i, m in enumerate(MOTOR_NAMES)]
        )
        # The kinematics wrapper clamps to the URDF limits; refuse to start from outside them.
        lo, hi = np.degrees(kin._inner._inner.q_lo), np.degrees(kin._inner._inner.q_hi)
        if np.any(urdf_deg[:6] < lo[:6] + 2) or np.any(urdf_deg[:6] > hi[:6] - 2):
            raise RuntimeError(
                f"start pose is outside the URDF joint limits (urdf deg {np.round(urdf_deg[:6], 1)}); "
                "move the arm forward of its fold first"
            )
        t0 = kin.forward_kinematics(q0)
        ctrl = CartesianIKController(
            kinematics=kin,
            motor_names=list(MOTOR_NAMES),
            q_init=q0,
            workspace_min=SO107_WORKSPACE_MIN,
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


def _disconnect(j: _Jog) -> None:
    j.stop.set()
    if j.thread is not None:
        j.thread.join(timeout=2.0)
    if j.robot is not None:
        j.robot.disconnect()


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
