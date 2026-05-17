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
    .venv/bin/python -m lerobot.robots.so107_description.teleop_quest.sim_receiver
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

from .. import get_urdf_path
from ..kinematics import (
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


# Axis-mapping matrix M such that  v_robot = M @ v_quest.
# Quest local stage frame:  +x = right (user), +y = up, +z = toward user (out of headset).
# Robot base frame:         +x = forward (away from user), +y = left (robot's own), +z = up.
# So:  robot_x = -quest_z,  robot_y = -quest_x,  robot_z =  quest_y
QUEST_TO_ROBOT_M = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)


def quest_delta_to_robot(delta_quest: np.ndarray) -> np.ndarray:
    """Quest stage-frame delta xyz → robot base-frame delta xyz."""
    return QUEST_TO_ROBOT_M @ delta_quest


def quest_rot_to_robot(quest_quat_xyzw: list[float]) -> Rot:
    """Quest controller quaternion → robot-frame rotation (scipy Rotation)."""
    r_quest = Rot.from_quat(quest_quat_xyzw).as_matrix()
    return Rot.from_matrix(QUEST_TO_ROBOT_M @ r_quest @ QUEST_TO_ROBOT_M.T)


def clamp_rotation_step(target_rot: Rot, current_rot: Rot, max_angle_rad: float) -> Rot:
    """Cap rotation step from current_rot to target_rot at max_angle_rad."""
    delta = target_rot * current_rot.inv()
    angle = delta.magnitude()
    if angle <= max_angle_rad:
        return target_rot
    scale = max_angle_rad / angle
    delta_step = Rot.from_rotvec(delta.as_rotvec() * scale)
    return delta_step * current_rot


# Right-hand controller analog trigger is at buttons[0] on Quest 3 (confirmed via debug log).
TRIGGER_BUTTON_INDEX = 0
TRIGGER_THRESHOLD = 0.5
# Max distance per IK tick (m) to clamp wild targets to reachable region.
MAX_TARGET_STEP_M = 0.02
# If IK can't reach within this many mm AND the gap isn't closing, treat as out-of-workspace.
IK_REACHABLE_THRESHOLD_MM = 10.0
# Max rotation step per IK tick (radians, ~5°). Keeps the rotation delta the model sees in-distribution.
MAX_ROT_STEP_RAD = np.radians(5.0)
# Watchdog: warn if no frame received for this long (wall-clock seconds).
FRAME_STALL_THRESHOLD_S = 0.25
# Controller stale: warn if controller position moves less than this (m) for this many engaged frames.
STATIONARY_POS_EPSILON_M = 0.0005
STATIONARY_FRAME_THRESHOLD = 60  # ~0.67s at 90Hz
# Latency probe ping interval (seconds).
PING_INTERVAL = 0.1
# Reachable workspace box (from training data extraction stats). Targets clamped to this.
WORKSPACE_MIN = np.array([-0.20, -0.35, +0.03])
WORKSPACE_MAX = np.array([+0.25, +0.05, +0.36])
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

    def __init__(self) -> None:
        self.motors: dict[str, float] = dict(SIM_HOME)
        self.kin_nn = So107NNKinematics(
            model_path=Path(tempfile.gettempdir()) / "so107_ik_model_action.pt",
            refine_with_dls=True,
        )
        urdf_path = str(get_urdf_path())
        mesh_dir = str(Path(urdf_path).parent)
        self.pin_model = pin.buildModelFromUrdf(urdf_path)
        self.pin_data = self.pin_model.createData()
        # Visual geometries (URDF <visual> meshes) for rendering.
        self.gmodel = pin.buildGeomFromUrdf(self.pin_model, urdf_path, pin.GeometryType.VISUAL, [mesh_dir])
        self.gdata = self.gmodel.createData()
        self.ee_frame_id = self.pin_model.getFrameId("L7_1")

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
        self, quest_pos: np.ndarray, quest_quat: list[float], trigger: float, frame_no: int
    ) -> None:
        """Called once per WebXR frame for the right controller."""
        was_engaged = self.engaged
        self.engaged = trigger > TRIGGER_THRESHOLD

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

        cur_ee, cur_rot = self._current_ee_pos_and_rot()
        step = target_xyz - cur_ee
        step_mag = float(np.linalg.norm(step))
        if step_mag > MAX_TARGET_STEP_M:
            step = step * (MAX_TARGET_STEP_M / step_mag)
        clamped_target = cur_ee + step

        # --- ROTATION ---
        # Controller rotation in robot frame, then delta since engage, then composed with engaged EE rotation.
        quest_rot_now = quest_rot_to_robot(quest_quat)
        delta_rot = quest_rot_now * self.quest_rot_at_engage.inv()
        target_rot_full = delta_rot * self.ee_rot_at_engage
        # Cap the rotation step per tick so the model sees in-distribution rotation deltas.
        cur_rot_R = Rot.from_matrix(cur_rot)
        target_rot_stepped = clamp_rotation_step(target_rot_full, cur_rot_R, MAX_ROT_STEP_RAD)

        target_T = np.eye(4)
        target_T[:3, :3] = target_rot_stepped.as_matrix()
        target_T[:3, 3] = clamped_target

        motors_before = dict(self.motors)
        new_motors, err_mm = self.kin_nn.ik_to_motors(self.motors, target_T)

        # If IK couldn't reach the target, don't update (avoids garbage joint drift).
        if err_mm > IK_REACHABLE_THRESHOLD_MM:
            if frame_no % 30 == 0:
                print(f"  [f={frame_no}] OUT-OF-REACH  ik_err={err_mm:.2f}mm — freezing arm")
            self._render(target_xyz=target_xyz_raw)
            return

        # Update motors EXCEPT gripper (gripper isn't EE-tracked; would drift on bias).
        for n in MOTOR_NAMES:
            if n == "gripper":
                continue
            delta = new_motors[n] - self.motors[n]
            delta = max(-3.0, min(3.0, delta))
            new_val = self.motors[n] + delta
            lo, hi = JOINT_LIMITS[n]
            self.motors[n] = max(lo, min(hi, new_val))

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
                trigger = buttons[TRIGGER_BUTTON_INDEX] if len(buttons) > TRIGGER_BUTTON_INDEX else 0.0
                sim.handle_controller(quest_pos, quest_quat, trigger, frame_count)
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
    log_path = Path(tempfile.gettempdir()) / f"quest_sim_{datetime.now():%H%M%S}.log"
    sys.stdout = TeeStdout(log_path)
    print(f"  log: {log_path}")
    ensure_cert()
    rr.init("so107_quest_sim", spawn=True)
    sim = Sim()

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

    web.run_app(app, host="0.0.0.0", port=PORT, ssl_context=ssl_ctx, print=None)  # nosec B104 (LAN-only dev tool, intentional)
    return 0


if __name__ == "__main__":
    sys.exit(main())
