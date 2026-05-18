"""
Quest3 WebXR → IK → simulated arm in rerun. No physical robot.

Clutch-mode position teleop:
    - Squeeze right trigger to ENGAGE: snapshot controller pose + current EE.
    - While engaged: target_EE = snapshot_EE + (controller_now - controller_at_snapshot).
    - Release trigger: freeze arm at current pose.
    - Target rotation is locked to current EE rotation (we're not teleop'ing
      orientation in this first pass; position only).

Naive scaling: 1cm controller motion = 1cm robot motion. Axes:
    robot_x = -quest_z   (Quest forward = robot back)
    robot_y = -quest_x   (Quest right   = robot left)
    robot_z =  quest_y   (Quest up      = robot up)

Run:
    .venv/bin/python -m lerobot.robots.so107_description.experimental.teleop_quest.sim_receiver
On Quest, open https://<PC-IP>:8443/, Connect → Enter VR.
The rerun viewer opens on the PC.
"""

from __future__ import annotations

import asyncio
import json
import ssl
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pinocchio as pin
import rerun as rr
from aiohttp import WSMsgType, web
from scipy.spatial.transform import Rotation as Rot

from ... import get_urdf_path
from ...kinematics import (
    MOTOR_NAMES,
    RIGHT_ARM_MAP,
    motor_pos_to_urdf_q,
)
from ..learned_ik.kinematics_nn import So107NNKinematics
from .latency_receiver import CERT, KEY, PORT, ensure_cert, get_lan_ip

HTML = Path(__file__).parent / "webxr_teleop.html"

# Sim starts the right arm at a "home" pose — values from a typical
# mid-trajectory frame in the training data.
SIM_HOME = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -20.0,
    "elbow_flex": 50.0,
    "forearm_roll": 0.0,
    "wrist_flex": 40.0,
    "wrist_roll": 0.0,
    "gripper": 15.0,
}


# ============================================================================
# Quest <-> Robot frame mapping. Assumes user stands BEHIND the robot, facing
# the same direction the arm naturally reaches. This is the simplest mental
# model: "controller left = robot's left in its own POV". Same orientation you
# get if you watch the workspace through the robot's top-down camera.
#
# SO-107 URDF axis convention (verified empirically via FK at zero joints —
# EE sits at (0.018, -0.276, +0.288), shoulder_pan=+30 swings arm to URDF -X):
#   robot forward (arm reach direction) = URDF -Y
#   robot left                          = URDF +X
#   robot up                            = URDF +Z
#
# Quest local frame: +x=user right, +y=up, +z=toward user (-z = user forward).
# Mapping derives directly: send user_forward to robot_forward, etc.
# ============================================================================
ROBOT_FORWARD_IN_URDF = np.array([0.0, -1.0, 0.0])
ROBOT_UP_IN_URDF = np.array([0.0, 0.0, 1.0])
ROBOT_LEFT_IN_URDF = np.cross(ROBOT_UP_IN_URDF, ROBOT_FORWARD_IN_URDF)  # = (+1, 0, 0)

# Columns of M map (quest_x, quest_y, quest_z) unit vectors to URDF frame.
QUEST_TO_ROBOT_M = np.column_stack(
    [
        -ROBOT_LEFT_IN_URDF,  # quest_x = user_right  -> robot_right (= -robot_left)
        +ROBOT_UP_IN_URDF,  # quest_y = user_up     -> robot_up
        -ROBOT_FORWARD_IN_URDF,  # quest_z = user_back   -> robot_back (= -robot_forward)
    ]
)


def quest_delta_to_robot(delta_quest: np.ndarray) -> np.ndarray:
    """Quest stage-frame delta xyz → robot base-frame delta xyz."""
    return QUEST_TO_ROBOT_M @ delta_quest


def quest_rot_to_robot(quest_quat_xyzw: list[float]) -> Rot:
    """Quest controller quaternion → robot-frame rotation (scipy Rotation)."""
    r_quest = Rot.from_quat(quest_quat_xyzw).as_matrix()
    return Rot.from_matrix(QUEST_TO_ROBOT_M @ r_quest @ QUEST_TO_ROBOT_M.T)


# ============================================================================
# Quest 3 right controller button mapping (confirmed via debug log).
# ============================================================================
CLUTCH_BUTTON_INDEX = 1  # analog grip/squeeze (side button) -> clutch
GRIPPER_BUTTON_INDEX = 0  # analog index trigger -> gripper open/close
CLUTCH_THRESHOLD = 0.5  # squeeze value above this engages tracking

# Trigger value -> gripper position (linear). 0 (released) = open, 1 (squeezed) = closed.
GRIPPER_OPEN_VALUE = 60.0
GRIPPER_CLOSED_VALUE = 5.0

# ============================================================================
# Safety caps. Applied in layers; each is intentional. Logged at startup.
# ============================================================================
# Workspace box (m, robot base frame). Pink's target position is clamped to
# this. Limits derived from training-data EE extents.
WORKSPACE_MIN = np.array([-0.20, -0.35, +0.03])
WORKSPACE_MAX = np.array([+0.25, +0.05, +0.36])

# Pink convergence threshold. If pink's final ik_err exceeds this in a single
# solve, treat as truly stuck. 20-50mm residual is NORMAL during fast hand
# motion (motor lag); only catch genuinely-stuck cases here.
IK_REACHABLE_THRESHOLD_MM = 100.0

# Software per-tick joint motion cap (deg). Wraps pink's output so any one
# tick can't command a wild joint jump. The bus's max_relative_target
# (--max-relative-target, default 30°) is the redundant hardware-side limit.
SOFTWARE_JOINT_CAP_DEG = 25.0

# Watchdogs
FRAME_STALL_THRESHOLD_S = 0.25  # warn if WebXR pose stream pauses
STATIONARY_POS_EPSILON_M = 0.0005  # controller-stale: < this much motion = stationary
STATIONARY_FRAME_THRESHOLD = 60  # ... for this many frames -> warn (≈0.67s @ 90Hz)
PING_INTERVAL = 0.1  # PC -> Quest ping cadence for RTT measurement
# Physical motor limits in degrees (from so107_follower/white_right.json calibration).
# These are degrees-from-midpoint after the bus's DEGREES-mode conversion.
JOINT_LIMITS = {
    "shoulder_pan": (-93.0, 93.0),
    "shoulder_lift": (-105.0, 105.0),
    "elbow_flex": (-97.0, 97.0),
    "forearm_roll": (-96.0, 96.0),
    "wrist_flex": (-98.0, 98.0),
    "wrist_roll": (-180.0, 180.0),  # this joint can do full rotation
    "gripper": (0.0, 100.0),
}


class Sim:
    """Sim arm state + clutch-mode teleop. Trigger held = follow controller delta."""

    def __init__(
        self,
        ik_backend: str = "nn_dls",
        execute: bool = False,
        port: str = "/dev/ttyACM2",
        robot_id: str = "white_right",
        max_relative_target: float = 5.0,
    ) -> None:
        self.motors: dict[str, float] = dict(SIM_HOME)
        nn_model_path = Path(tempfile.gettempdir()) / "so107_ik_model_action.pt"
        if ik_backend == "nn_dls":
            self.kin_nn = So107NNKinematics(model_path=nn_model_path, refine_with_dls=True)
        elif ik_backend == "pink":
            from ..learned_ik.kinematics_pink import So107PinkKinematics

            self.kin_nn = So107PinkKinematics()
        elif ik_backend == "pink_nn":
            from ..learned_ik.kinematics_pink import So107PinkKinematics

            self.kin_nn = So107PinkKinematics(nn_posture_model=nn_model_path)
        else:
            raise ValueError(f"unknown ik_backend: {ik_backend}")
        print(f"  IK backend: {ik_backend}")

        self.robot = None
        if execute:
            from lerobot.robots.so_follower import SO107Follower, SO107FollowerConfig

            print(
                f"  *** EXECUTE MODE: connecting to robot on {port} (id={robot_id}, mrt={max_relative_target}°)"
            )
            config = SO107FollowerConfig(
                port=port,
                id=robot_id,
                use_degrees=True,
                cameras={},
                max_relative_target=max_relative_target,
            )
            self.robot = SO107Follower(config)
            self.robot.connect(calibrate=False)
            # Sync sim state to whatever pose the robot is currently in,
            # so the first engage-snapshot reflects physical reality.
            obs = self.robot.bus.sync_read("Present_Position")
            for nm in MOTOR_NAMES:
                self.motors[nm] = float(obs[nm])
            print(
                "  connected. starting motors: "
                + " ".join(f"{nm.split('_')[0][:3]}={self.motors[nm]:+5.1f}" for nm in MOTOR_NAMES)
            )
        urdf_path = str(get_urdf_path())
        mesh_dir = str(Path(urdf_path).parent)
        self.pin_model = pin.buildModelFromUrdf(urdf_path)
        self.pin_data = self.pin_model.createData()
        # Visual geometries (URDF <visual> meshes) for rendering.
        self.gmodel = pin.buildGeomFromUrdf(self.pin_model, urdf_path, pin.GeometryType.VISUAL, [mesh_dir])
        self.gdata = self.gmodel.createData()
        self.ee_frame_id = self.pin_model.getFrameId("L7_1")
        # Collision geometries: same meshes, separate model, used to log self-collisions.
        self.cmodel = pin.buildGeomFromUrdf(self.pin_model, urdf_path, pin.GeometryType.COLLISION, [mesh_dir])
        self.cmodel.addAllCollisionPairs()
        self.cdata = self.cmodel.createData()
        # Identify baseline collision pairs that already touch at the home pose (adjacent links).
        # We exclude these from per-tick collision warnings.
        zero_q = motor_pos_to_urdf_q(SIM_HOME, RIGHT_ARM_MAP)
        pin.computeCollisions(self.pin_model, self.pin_data, self.cmodel, self.cdata, zero_q, False)
        self.baseline_collisions: set[int] = {
            k for k in range(len(self.cmodel.collisionPairs)) if self.cdata.collisionResults[k].isCollision()
        }
        print(
            f"  collision check: {len(self.cmodel.collisionPairs)} pairs, "
            f"{len(self.baseline_collisions)} adjacent-pair baseline collisions (ignored)"
        )

        # Log each visual mesh once (static); per frame we update Transform3D on its entity.
        # URDF visual scale (mm -> m) is stored in geom.meshScale and applied via the per-frame Transform3D.
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self.mesh_scales: list[np.ndarray] = []
        for geom in self.gmodel.geometryObjects:
            path = f"world/arm/{geom.name}"
            rr.log(path, rr.Asset3D(path=geom.meshPath), static=True)
            self.mesh_scales.append(np.asarray(geom.meshScale, dtype=np.float32))

        # Clutch state
        self.engaged = False
        self.quest_pos_at_engage: np.ndarray | None = None
        self.ee_pos_at_engage: np.ndarray | None = None
        self.quest_rot_at_engage: Rot | None = None
        self.ee_rot_at_engage: Rot | None = None
        # Stale-controller detection state
        self.last_quest_pos: np.ndarray | None = None
        self.stationary_frames: int = 0
        self.stale_warned: bool = False
        # Reset button rising-edge tracking
        self.last_buttons: list[float] | None = None
        self._render()

    def reset_to_home(self, frame_no: int) -> None:
        """Snap motors back to SIM_HOME. On real robot the bus's max_relative_target
        smooths the motion automatically across many ticks."""
        print(f"  [f={frame_no}] >>> RESET to home pose")
        self.motors = dict(SIM_HOME)
        self.engaged = False
        self.quest_pos_at_engage = None
        self.ee_pos_at_engage = None
        self.quest_rot_at_engage = None
        self.ee_rot_at_engage = None
        if self.robot is not None:
            self.robot.send_action({f"{nm}.pos": SIM_HOME[nm] for nm in MOTOR_NAMES})
        self._render()

    def _current_ee_pos_and_rot(self) -> tuple[np.ndarray, np.ndarray]:
        T_ee = self.kin_nn.fk.fk_from_motors(self.motors)
        return T_ee[:3, 3].copy(), T_ee[:3, :3].copy()

    def _render(self, target_xyz: np.ndarray | None = None) -> None:
        q = motor_pos_to_urdf_q(self.motors, RIGHT_ARM_MAP)
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        pin.updateGeometryPlacements(self.pin_model, self.pin_data, self.gmodel, self.gdata)
        # Per-frame: world transform per mesh, with URDF-specified scale applied.
        for geom_id, geom in enumerate(self.gmodel.geometryObjects):
            T = self.gdata.oMg[geom_id]
            rr.log(
                f"world/arm/{geom.name}",
                rr.Transform3D(
                    translation=T.translation,
                    mat3x3=T.rotation,
                    scale=tuple(self.mesh_scales[geom_id].tolist()),
                ),
            )
        # EE marker (small sphere): green when engaged, gray when not.
        ee_pos = self.pin_data.oMf[self.ee_frame_id].translation
        ee_col = [0, 255, 0] if self.engaged else [128, 128, 128]
        rr.log("world/ee", rr.Points3D([ee_pos], radii=0.015, colors=[ee_col]))
        if target_xyz is not None:
            rr.log("world/target", rr.Points3D([target_xyz], radii=0.02, colors=[100, 150, 255]))

    def handle_controller(
        self,
        quest_pos: np.ndarray,
        quest_quat: list[float],
        clutch: float,
        gripper_trigger: float,
        all_buttons: list[float],
        frame_no: int,
    ) -> None:
        """Called once per WebXR frame for the right controller.

        clutch: side-button squeeze value [0..1]. Above CLUTCH_THRESHOLD -> tracking on.
        gripper_trigger: index-trigger value [0..1]. Continuously mapped to gripper position.
        all_buttons: full button array, scanned for rising-edge reset-button detection.
        """
        was_engaged = self.engaged
        self.engaged = clutch > CLUTCH_THRESHOLD

        # If hardware: refresh self.motors from physical state. This keeps the IK
        # seed honest (matches what the motors are actually doing, including any
        # tracking lag) and makes the engage snapshot reflect physical reality.
        if self.robot is not None:
            obs = self.robot.bus.sync_read("Present_Position")
            for nm in MOTOR_NAMES:
                self.motors[nm] = float(obs[nm])

        # Gripper is driven by trigger value at all times (not gated by clutch — you can
        # open/close the gripper without commanding EE motion).
        self.motors["gripper"] = GRIPPER_OPEN_VALUE + gripper_trigger * (
            GRIPPER_CLOSED_VALUE - GRIPPER_OPEN_VALUE
        )

        # Reset detection: while DISENGAGED, a non-clutch / non-trigger button transitioning
        # clearly from off (<0.1) to on (>0.8) fires a reset. The strict thresholds filter
        # out analog touch/proximity sensors (e.g., Quest 3 buttons[7]/[8]) that float near 0.5
        # just from holding the controller.
        if not self.engaged and self.last_buttons is not None and len(all_buttons) == len(self.last_buttons):
            for i, (prev, cur) in enumerate(zip(self.last_buttons, all_buttons, strict=False)):
                if i in (CLUTCH_BUTTON_INDEX, GRIPPER_BUTTON_INDEX):
                    continue
                if prev < 0.1 and cur > 0.8:
                    print(f"  [f={frame_no}] reset triggered by button[{i}] ({prev:.2f}->{cur:.2f})")
                    self.reset_to_home(frame_no)
                    self.last_buttons = list(all_buttons)
                    return
        self.last_buttons = list(all_buttons)

        if self.engaged and not was_engaged:
            cur_ee_pos, cur_ee_rot = self._current_ee_pos_and_rot()
            self.quest_pos_at_engage = quest_pos.copy()
            self.ee_pos_at_engage = cur_ee_pos
            self.quest_rot_at_engage = quest_rot_to_robot(quest_quat)
            self.ee_rot_at_engage = Rot.from_matrix(cur_ee_rot)
            self.last_quest_pos = quest_pos.copy()
            self.stationary_frames = 0
            self.stale_warned = False
            print(
                f"  >>> ENGAGE  quest=({quest_pos[0]:+.3f},{quest_pos[1]:+.3f},{quest_pos[2]:+.3f})  "
                f"ee=({cur_ee_pos[0]:+.3f},{cur_ee_pos[1]:+.3f},{cur_ee_pos[2]:+.3f})"
            )
            self._render()
            return

        if was_engaged and not self.engaged:
            print("  <<< RELEASE")
            self.last_quest_pos = None
            self.stationary_frames = 0
            self.stale_warned = False

        if not self.engaged:
            self._render()
            return

        # Stale controller pose detection (engaged only).
        if self.last_quest_pos is not None:
            move = float(np.linalg.norm(quest_pos - self.last_quest_pos))
            if move < STATIONARY_POS_EPSILON_M:
                self.stationary_frames += 1
            else:
                if self.stale_warned:
                    print(f"  [f={frame_no}] controller pose tracking recovered")
                self.stationary_frames = 0
                self.stale_warned = False
            if self.stationary_frames == STATIONARY_FRAME_THRESHOLD and not self.stale_warned:
                print(
                    f"  [f={frame_no}] WARNING controller pose stationary for "
                    f"{STATIONARY_FRAME_THRESHOLD} frames (~{STATIONARY_FRAME_THRESHOLD / 90:.1f}s) — "
                    f"likely tracking dropout (controller out of cameras' FOV?)"
                )
                self.stale_warned = True
        self.last_quest_pos = quest_pos.copy()

        assert (
            self.quest_pos_at_engage is not None
            and self.ee_pos_at_engage is not None
            and self.quest_rot_at_engage is not None
            and self.ee_rot_at_engage is not None
        )
        # --- POSITION ---
        delta_quest = quest_pos - self.quest_pos_at_engage
        delta_robot = quest_delta_to_robot(delta_quest)
        target_xyz_raw = self.ee_pos_at_engage + delta_robot
        target_xyz = np.clip(target_xyz_raw, WORKSPACE_MIN, WORKSPACE_MAX)

        # Send IK the FULL target (after workspace clamp). Do NOT pre-clamp the
        # target to a small step from current_ee — that defeats the motor's
        # ability to fight gravity. Instead the joint-level software cap (15°/tick)
        # and the bus's max_relative_target (30°/tick) ramp the motor commands.
        cur_ee, _ = self._current_ee_pos_and_rot()
        clamped_target = target_xyz

        # --- ROTATION ---
        # Controller rotation in robot frame, then delta since engage, then composed with engaged EE rotation.
        quest_rot_now = quest_rot_to_robot(quest_quat)
        delta_rot = quest_rot_now * self.quest_rot_at_engage.inv()
        target_rot_full = delta_rot * self.ee_rot_at_engage
        # Send IK the FULL rotation target. Same reasoning as the position case: pre-clamping
        # to "current_rot + 5°/tick" was causing pink to output near-current joints, which
        # when commanded to the motor barely moved it. The joint-level software cap and
        # bus mrt are the rate-limiters.

        target_T = np.eye(4)
        target_T[:3, :3] = target_rot_full.as_matrix()
        target_T[:3, 3] = clamped_target

        motors_before = dict(self.motors)
        new_motors, err_mm = self.kin_nn.ik_to_motors(self.motors, target_T)

        # If IK couldn't reach the target, don't update (avoids garbage joint drift).
        if err_mm > IK_REACHABLE_THRESHOLD_MM:
            if frame_no % 30 == 0:
                print(f"  [f={frame_no}] OUT-OF-REACH  ik_err={err_mm:.2f}mm — freezing arm")
            self._render(target_xyz=target_xyz_raw)
            return

        # Update motors EXCEPT gripper (gripper isn't EE-tracked; driven by trigger).
        # Software per-tick cap (SOFTWARE_JOINT_CAP_DEG) ramps commanded motor motion. The
        # bus's max_relative_target (--max-relative-target, default 30°) is the redundant
        # hardware layer.
        for n in MOTOR_NAMES:
            if n == "gripper":
                continue
            delta = new_motors[n] - self.motors[n]
            delta = max(-SOFTWARE_JOINT_CAP_DEG, min(SOFTWARE_JOINT_CAP_DEG, delta))
            new_val = self.motors[n] + delta
            lo, hi = JOINT_LIMITS[n]
            self.motors[n] = max(lo, min(hi, new_val))

        # Send to physical robot (after software clamping + joint limits, before collision logging).
        # The bus's max_relative_target provides an additional hardware-side rate limit.
        if self.robot is not None:
            self.robot.send_action({f"{nm}.pos": self.motors[nm] for nm in MOTOR_NAMES})

        # Check + LOG self-collisions (not freezing — see Ke's principle: training-data bias should mostly prevent this).
        q_new = motor_pos_to_urdf_q(self.motors, RIGHT_ARM_MAP)
        pin.computeCollisions(self.pin_model, self.pin_data, self.cmodel, self.cdata, q_new, False)
        new_collisions = []
        for k in range(len(self.cmodel.collisionPairs)):
            if k in self.baseline_collisions:
                continue
            cr = self.cdata.collisionResults[k]
            if cr.isCollision():
                pair = self.cmodel.collisionPairs[k]
                first = self.cmodel.geometryObjects[pair.first].name
                second = self.cmodel.geometryObjects[pair.second].name
                # Penetration depth (negative distance = overlapping)
                depth_mm = -cr.distance_lower_bound * 1000 if cr.distance_lower_bound < 0 else 0.0
                new_collisions.append((first, second, depth_mm))
        if new_collisions:
            details = "  ".join(f"{a}↔{b}({d:.1f}mm)" for a, b, d in new_collisions)
            print(f"  [f={frame_no}] COLLISION  {details}")

        if frame_no % 30 == 0:
            max_joint_change = max(
                abs(self.motors[n] - motors_before[n]) for n in MOTOR_NAMES if n != "gripper"
            )
            clamp_hit = "[CLAMP-WS]" if not np.allclose(target_xyz, target_xyz_raw) else ""
            # Report which joints are sitting at their limits.
            at_limit = []
            for n in MOTOR_NAMES:
                if n == "gripper":
                    continue
                lo, hi = JOINT_LIMITS[n]
                if abs(self.motors[n] - lo) < 0.5:
                    at_limit.append(f"{n}=LO")
                elif abs(self.motors[n] - hi) < 0.5:
                    at_limit.append(f"{n}=HI")
            limit_str = f"  AT-LIMIT:{','.join(at_limit)}" if at_limit else ""
            print(
                f"  [f={frame_no}] engaged {clamp_hit} quest_d=({delta_quest[0]:+.3f},{delta_quest[1]:+.3f},{delta_quest[2]:+.3f}) "
                f" target=({clamped_target[0]:+.3f},{clamped_target[1]:+.3f},{clamped_target[2]:+.3f})  "
                f"cur_ee=({cur_ee[0]:+.3f},{cur_ee[1]:+.3f},{cur_ee[2]:+.3f})  "
                f"ik_err={err_mm:5.2f}mm  max_joint_change={max_joint_change:.2f}°{limit_str}"
            )

        self._render(target_xyz=target_xyz)


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=64 * 1024)
    await ws.prepare(request)
    print(f"client connected from {request.remote}")
    sim: Sim = request.app["sim"]
    last_print = time.perf_counter()
    frame_count = 0
    last_frame_t = time.perf_counter()
    rtt_samples: list[float] = []
    pending_pings: dict[int, float] = {}
    stall_warned = False

    async def ping_loop() -> None:
        seq = 0
        while not ws.closed:
            t = time.perf_counter() * 1000
            pending_pings[seq] = t
            try:
                await ws.send_json({"type": "ping", "seq": seq, "t_pc_send": t})
            except ConnectionError:
                return
            seq += 1
            await asyncio.sleep(PING_INTERVAL)

    async def stall_watchdog() -> None:
        nonlocal stall_warned
        while not ws.closed:
            await asyncio.sleep(0.1)
            gap = time.perf_counter() - last_frame_t
            if gap > FRAME_STALL_THRESHOLD_S and not stall_warned:
                print(
                    f"  WARNING no frame for {gap * 1000:.0f}ms — XR session paused "
                    f"(headset removed? proximity sensor triggered?)"
                )
                stall_warned = True

    ping_task = asyncio.create_task(ping_loop())
    watchdog_task = asyncio.create_task(stall_watchdog())

    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)
            mtype = data.get("type")
            if mtype == "pong":
                t_send = pending_pings.pop(data["seq"], None)
                if t_send is not None:
                    rtt = time.perf_counter() * 1000 - t_send
                    rtt_samples.append(rtt)
                    if len(rtt_samples) > 200:
                        rtt_samples = rtt_samples[-200:]
                continue
            if mtype != "frame":
                continue

            t_now = time.perf_counter()
            if stall_warned:
                print(f"  frame stream resumed after {(t_now - last_frame_t) * 1000:.0f}ms gap")
                stall_warned = False
            last_frame_t = t_now

            poses = data.get("poses", [])
            right = next((p for p in poses if p["hand"] == "right"), None)
            if right is not None:
                rr.set_time("frame", sequence=frame_count)
                quest_pos = np.array(right["pos"])
                quest_quat = right.get("rot", [0.0, 0.0, 0.0, 1.0])
                buttons = right.get("buttons", [])
                clutch = buttons[CLUTCH_BUTTON_INDEX] if len(buttons) > CLUTCH_BUTTON_INDEX else 0.0
                grip = buttons[GRIPPER_BUTTON_INDEX] if len(buttons) > GRIPPER_BUTTON_INDEX else 0.0
                sim.handle_controller(quest_pos, quest_quat, clutch, grip, buttons, frame_count)
                frame_count += 1
            if t_now - last_print > 1.0:
                engaged = "ENGAGED" if sim.engaged else "free   "
                motors_str = " ".join(f"{k.split('_')[0][:3]}={v:+5.1f}" for k, v in sim.motors.items())
                rtt_str = ""
                if rtt_samples:
                    s = sorted(rtt_samples)
                    rtt_str = (
                        f"  RTT mean={sum(s) / len(s):4.1f} p95={s[min(int(len(s) * 0.95), len(s) - 1)]:4.1f}"
                        f" max={s[-1]:4.1f}ms"
                    )
                print(f"f={frame_count:5d}  {engaged}  {motors_str}{rtt_str}")
                last_print = t_now
    finally:
        ping_task.cancel()
        watchdog_task.cancel()
        print("client disconnected")
    return ws


async def index_handler(request: web.Request) -> web.Response:
    return web.Response(text=HTML.read_text(), content_type="text/html")


class TeeStdout:
    """Mirror stdout writes to a file so the user can grab the log easily."""

    def __init__(self, file_path: Path) -> None:
        self.file = file_path.open("w", buffering=1)
        self.stdout = sys.stdout

    def write(self, s: str) -> int:
        self.stdout.write(s)
        self.file.write(s)
        return len(s)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ik",
        choices=["nn_dls", "pink", "pink_nn"],
        default="nn_dls",
        help="IK backend. nn_dls = learned NN + DLS refine (default); pink = QP-based 6-DOF; "
        "pink_nn = pink with NN-predicted joints as posture target (best of both).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Drive the physical robot via SO107Follower. WITHOUT this flag, sim_receiver "
        "is sim-only (renders in rerun, no hardware).",
    )
    parser.add_argument("--port", default="/dev/ttyACM2", help="serial port for the robot")
    parser.add_argument("--id", default="white_right", help="robot id (selects calibration file)")
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=30.0,
        help="hardware-side per-tick joint motion cap (deg). The bus clamps each "
        "commanded joint to be within this many degrees of the current motor "
        "position. Lower = safer but the motor will struggle to fight gravity "
        "(small command-gap = small PID torque). The validated value from "
        "trajectory_replay is 30°. The software cap inside sim_receiver "
        "(SOFTWARE_JOINT_CAP_DEG) is 25°/tick.",
    )
    args = parser.parse_args()

    log_path = Path(tempfile.gettempdir()) / f"quest_sim_{datetime.now():%H%M%S}.log"
    sys.stdout = TeeStdout(log_path)
    print(f"  log: {log_path}")
    print()
    print("  === Configured safety caps ===")
    print(f"  ik backend                = {args.ik}")
    print(f"  workspace box (m)         = x{tuple(WORKSPACE_MIN)} -> x{tuple(WORKSPACE_MAX)}")
    print(
        f"  ik reachable threshold    = {IK_REACHABLE_THRESHOLD_MM:.0f} mm (freeze if pink can't reach this)"
    )
    print(f"  software joint cap        = {SOFTWARE_JOINT_CAP_DEG:.0f} °/tick  (per-joint motion ramp)")
    print(f"  hardware mrt (bus)        = {args.max_relative_target:.0f} °/tick  (--max-relative-target)")
    print(
        "  joint limits (deg)        = "
        + ", ".join(f"{n}=[{lo:+.0f},{hi:+.0f}]" for n, (lo, hi) in JOINT_LIMITS.items())
    )
    print(f"  clutch threshold          = {CLUTCH_THRESHOLD:.2f}  (right grip squeeze)")
    print(
        f"  gripper open/closed       = {GRIPPER_OPEN_VALUE:.0f} / {GRIPPER_CLOSED_VALUE:.0f}  (linear from trigger)"
    )
    print()

    if args.execute:
        print()
        print("  *** PHYSICAL ROBOT MODE ***")
        print(f"  port={args.port}  id={args.id}  mrt={args.max_relative_target}°/tick")
        print("  The arm will follow your controller when you squeeze the side grip.")
        print("  Make sure the arm has clearance to move from its current pose.")
        input("  Press Enter to confirm and connect (Ctrl-C to abort)...")

    ensure_cert()
    rr.init("so107_quest_sim", spawn=True)
    sim = Sim(
        ik_backend=args.ik,
        execute=args.execute,
        port=args.port,
        robot_id=args.id,
        max_relative_target=args.max_relative_target,
    )

    app = web.Application()
    app["sim"] = sim
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", ws_handler)

    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(certfile=str(CERT), keyfile=str(KEY))

    print()
    print("  rerun viewer should open in a new window.")
    print(f"  On Quest3 browser, open:  https://{get_lan_ip()}:{PORT}/")
    print("  Connect → Enter VR. Wave your RIGHT controller — sim arm should follow.")
    print()

    try:
        web.run_app(app, host="0.0.0.0", port=PORT, ssl_context=ssl_ctx, print=None)  # nosec B104 (LAN-only dev tool, intentional)
    finally:
        if sim.robot is not None:
            try:
                sim.robot.disconnect()
                print("  robot disconnected cleanly.")
            except Exception as ex:
                print(f"  robot disconnect failed: {ex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
