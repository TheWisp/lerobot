# ruff: noqa: N802, N803, N806
# Rotation and transform names follow the rig API and matrix notation.
"""The SO-107 arm and the overhead camera, as a Python API (`import rig`) and a command line (`./rig`).

Everything is in the ARM FRAME: origin at the arm's base, millimetres, z up. The tray surface is near TRAY_Z (-13.9 mm,
from the arm's touch calibration) but not level: tray_z(x, y) gives it locally.

Seeing (the overhead RealSense; each call reads the camera fresh only in look()):
  rgb, depth = rig.look()       colour HxWx3 uint8 and depth HxW metres (848x480); also saves look.jpg (50 px grid),
                                look_raw.jpg, look_depth.npy. The other seeing calls default to the last look.
  rig.objects(depth, rgb)       every object standing on the tray in the workspace -> list of dicts (below) + "mask"
  rig.obj(u, v)                 the object under pixel (u, v) -> dict (below) + "mask"
  rig.measure(mask, depth, rgb) measure the object under a mask -> {"centre": [x, y], "height" above the tray,
                                "top_z", "axis_deg" (long axis, 0..180 deg from +x toward +y), "length", "width"
                                (outline of the upper half; depth blur adds a few mm), "pixel_centre", "pixels",
                                "hsv" (median colour, OpenCV HSV)}
  rig.where(u, v) / rig.pixel(x, y, z)   pixel -> arm-frame surface point / arm-frame point -> pixel
  rig.tray_z(x, y)              the tray surface height under (x, y)
  rig.target()                  the scene to restore (the demonstration's first frame): {"rgb", "depth", "masks":
                                {"gamepad": HxW bool, "cube": HxW bool}}; measure() works on it the same way

Planning (offline, never moves the arm):
  rig.grasp_yaw(axis_deg)       the yaw that lines the demonstration's gamepad grasp up with a long axis at axis_deg
                                (in -90..90; the other choice, +/-180, grasps the opposite long edge)
  rig.reachable(x, y, z, yaw, seed=None) -> {"ok", "err_mm", "err_deg", "motor_deg", "why", "seed_next"}: the arm's own
                                IK, every servo's calibrated range and wrist_roll's physical stop (it overloads past
                                about +97 motor deg: keep below +90); seeded from the arm's present joints when it is
                                connected; chain plans by passing the previous call's seed_next as seed.

Moving (the arm is the LEFT SO-107; one call at a time; each blocks until done):
  rig.connect()                 power the arm (it holds where it is); limits speed to 40 mm/s and 30 deg/s
  rig.move(x, y, z, yaw=None)   fingertip to (x, y, z) mm, at most 100 mm from where it is, with the gripper turned as in
                                the demonstration's grasp rotated yaw deg about the vertical (None: keep the present
                                yaw); turns over 50 deg go in steps. Returns {"ok", "position", "yaw", "err_mm"}; ok is
                                False when it did not get within 4 mm / 4 deg in 20 s per step, unless it stopped with
                                its joints commanded onto the target and each within 6 deg of its command: then
                                "held_short" is True when the joints stop near their commands but the fingertip has
                                not reached the target (it can still be centimetres away: check the returned error).
  rig.travel(x, y, z, yaw=None) move() along a straight line in steps of at most 95 mm (avoids nothing by itself)
  rig.grip(p) -> {"obs", "holding"}  gripper to p (about 60 open .. 98 closed); holding = it stopped over 0.7 short of p
  rig.state()                   fingertip position, yaw, gripper, freeze reason, motor loads (signed 0.1% drive),
                                temperatures, commanded/observed joints and feed-forward offsets
  rig.recover()                 clear a freeze (a motor's overload latch); the arm holds where it is and moves again
  rig.park()                    fold the arm to rest and switch it off: always last, whatever happened; holding
                                something, it first lowers it onto the tray (clearing a freeze if needed), then opens
Every move is clamped to at least TRAY_Z + 2 mm high and to the workspace box; a frozen arm raises (park it).

Command line: ./rig look | where U V | pixel X Y Z | obj U V | objects | state | connect | move X Y Z [YAW] |
              grip P | park
"""

import contextlib
import io
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

import numpy as np

S = "http://127.0.0.1:9100"
CAMERA_SERIAL = "152222071508"  # the overhead RealSense D435
HERE = pathlib.Path(__file__).resolve().parent
TRAY_Z = -13.9
STEP_MAX = 100.0
TURN_STEP = 50.0
WORK_MIN, WORK_MAX = (-200.0, -350.0), (200.0, 0.0)
ABOVE_TOP_MM = (
    5.0  # measure(): a point this far above an object's top belongs to a taller neighbour, not to it
)
HELD_SHORT_DEG = (
    6.0  # move(): stopped with every joint within this of its command, the arm is as close as it gets
)
STILL_S, STILL_MM = 0.6, 0.3  # stopped: the fingertip stayed within STILL_MM of one point for STILL_S
FACTS = json.loads((HERE / "rig_facts.json").read_text())
DEMO_GRASP_R = np.array(FACTS["demo_grasp_R"])
DEMO_AXIS_DEG = (
    157.3  # measure() of the target gamepad: its long axis when the demonstration grasped it at yaw 0
)
T_BC = np.array(FACTS["t_bc"])
K = FACTS["intr"]


def _call(path, body=None, timeout=60):
    req = urllib.request.Request(
        S + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]


def _rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def _quat_to_R(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _R_to_quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x, y, z = (
        np.copysign(x, R[2, 1] - R[1, 2]),
        np.copysign(y, R[0, 2] - R[2, 0]),
        np.copysign(z, R[1, 0] - R[0, 1]),
    )
    return [float(x), float(y), float(z), float(w)]


def _yaw_of(R):
    rel = R @ DEMO_GRASP_R.T
    return float(np.degrees(np.arctan2(rel[1, 0], rel[0, 0])))


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


# --- seeing ------------------------------------------------------------------------------------------------------


def look():
    import cv2

    code, state = _call("/api/showservo/state")
    if code != 200:
        raise RuntimeError(f"camera state unavailable: {state}")
    if not (state.get("session") or {}).get("live"):
        code, out = _call(
            "/api/showservo/session/start", {"serial": CAMERA_SERIAL, "name": "approach_20261001"}, timeout=60
        )
        if code != 200:
            raise RuntimeError(f"camera start failed: {out}")
        time.sleep(2.0)
    with (
        urllib.request.urlopen(S + "/api/pregrasp/frame.npz", timeout=30) as response,
        np.load(io.BytesIO(response.read()), allow_pickle=False) as frame,
    ):
        rgb = frame["rgb"]
        depth = frame["depth"]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    np.save(HERE / "look_depth.npy", depth)
    grid = bgr.copy()
    for u in range(0, grid.shape[1], 50):
        cv2.line(grid, (u, 0), (u, grid.shape[0]), (80, 80, 80), 1)
        cv2.putText(grid, str(u), (u + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    for v in range(0, grid.shape[0], 50):
        cv2.line(grid, (0, v), (grid.shape[1], v), (80, 80, 80), 1)
        cv2.putText(grid, str(v), (2, v + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.imwrite(str(HERE / "look.jpg"), grid)
    cv2.imwrite(str(HERE / "look_raw.jpg"), bgr)
    return rgb, depth


def _depth(depth=None):
    return np.load(HERE / "look_depth.npy") if depth is None else depth


def _arm_points(depth):
    """Arm-frame xyz (mm) of every pixel, HxWx3; NaN where there is no depth."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    z = depth.astype(np.float64)
    cam = np.stack([(u - K["cx"]) * z / K["fx"], (v - K["cy"]) * z / K["fy"], z], -1)
    arm = cam @ T_BC[:3, :3].T + T_BC[:3, 3]
    arm[z <= 0.05] = np.nan
    return arm * 1000.0


def where(u, v, depth=None):
    pts = _arm_points(_depth(depth))
    patch = pts[max(0, v - 2) : v + 3, max(0, u - 2) : u + 3].reshape(-1, 3)
    patch = patch[~np.isnan(patch).any(1)]
    if not len(patch):
        raise ValueError(f"no depth at ({u}, {v})")
    return np.median(patch, axis=0)


def pixel(x, y, z):
    c = np.linalg.inv(T_BC) @ np.r_[np.array([x, y, z]) / 1000.0, 1.0]
    return int(round(c[0] * K["fx"] / c[2] + K["cx"])), int(round(c[1] * K["fy"] / c[2] + K["cy"]))


def _tray_height(pts):
    """Height (mm) of every pixel above the tray's plane, fitted to the workspace by RANSAC (the tray is not level)."""
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    inbox = (x > WORK_MIN[0]) & (x < WORK_MAX[0]) & (y > WORK_MIN[1]) & (y < WORK_MAX[1]) & ~np.isnan(z)
    P = pts[inbox]
    rng, best = np.random.default_rng(0), (0, None, None)
    for _ in range(300):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-6:
            continue
        n = n / np.linalg.norm(n) * np.sign(n[2] or 1.0)
        k = int((np.abs((P - a) @ n) < 2.0).sum())
        if k > best[0]:
            best = (k, n, a)
    _, n, a = best
    return (pts - a) @ n


def tray_z(x, y, depth=None):
    """The tray surface's height (mm) under (x, y) mm, from the plane fitted to a depth image (default: the last look);
    the tray is not level, so this differs from TRAY_Z by a few mm across the workspace."""
    pts = _arm_points(_depth(depth))
    h = _tray_height(pts)
    ok = ~np.isnan(h) & (np.abs(h) < 1.5)
    P = pts[ok]
    A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    a, b, c = np.linalg.lstsq(A, P[:, 2], rcond=None)[0]
    return float(a * x + b * y + c)


def _components(depth):
    """Label every object standing over 4 mm above the tray; objects touching with a depth step stay apart."""
    import cv2

    pts = _arm_points(depth)
    h = _tray_height(pts)
    hz = np.nan_to_num(h, nan=0.0)
    step = np.hypot(cv2.Sobel(hz, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(hz, cv2.CV_64F, 0, 1, ksize=3)) / 8.0
    body = ((hz > 4.0) & (step < 3.0)).astype(np.uint8)
    _, lab = cv2.connectedComponents(cv2.morphologyEx(body, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)))
    return lab, pts, h


def measure(mask, depth=None, rgb=None):
    """Measure the object covering `mask` (HxW bool) in a depth image (default: the last look).
    -> {"centre": [x, y] mm, "height": mm above the tray, "top_z": mm, "axis_deg": long axis (deg, 0..180, from +x
        toward +y), "length", "width": mm (outline of its upper half, a few mm wide of the truth from depth blur),
        "pixel_centre": [u, v], "pixels": n, "hsv": mean [H 0-180, S, V] when rgb is given}"""
    import cv2

    depth = _depth(depth)
    pts = _arm_points(depth)
    h = _tray_height(pts)
    m = np.asarray(mask, bool) & ~np.isnan(h)
    p, hh = pts[m], h[m]
    top = float(np.percentile(hh, 90))
    # The edges' mixed depth pixels sit low and widen the outline; points well above the object's own top are a taller
    # neighbour its mask touched (the gamepad fallen against the cube measured 102 mm long with the cube's edge in it).
    upper = p[(hh > top / 2) & (hh < top + ABOVE_TOP_MM)]
    (cx, cy), (w, length), ang = cv2.minAreaRect((upper[:, :2] * 10).astype(np.float32))
    axis = (ang if w >= length else ang + 90.0) % 180.0
    ys, xs = np.nonzero(m)
    out = {
        "centre": [round(cx / 10.0, 1), round(cy / 10.0, 1)],
        "height": round(top, 1),
        "top_z": round(float(np.percentile(p[:, 2], 90)), 1),
        "axis_deg": round(axis, 1),
        "length": round(max(w, length) / 10.0, 1),
        "width": round(min(w, length) / 10.0, 1),
        "pixel_centre": [int(xs.mean()), int(ys.mean())],
        "pixels": int(m.sum()),
    }
    if rgb is not None:
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)[m]
        out["hsv"] = [round(float(v), 1) for v in np.median(hsv, axis=0)]
    return out


def objects(depth=None, rgb=None, min_pixels=300):
    """Every object standing on the tray inside the workspace, measured -> list of measure() dicts plus "mask"."""
    depth = _depth(depth)
    lab, pts, _ = _components(depth)
    found = []
    for label in range(1, lab.max() + 1):
        m = lab == label
        if m.sum() < min_pixels:
            continue
        c = np.nanmedian(pts[m], axis=0)
        if not (WORK_MIN[0] < c[0] < WORK_MAX[0] and WORK_MIN[1] < c[1] < WORK_MAX[1]):
            continue
        found.append({**measure(m, depth, rgb), "mask": m})
    return found


def obj(u, v, depth=None, rgb=None):
    """The object under pixel (u, v) (or the nearest within 15 px) -> measure() dict plus "mask"."""
    depth = _depth(depth)
    lab, _, _ = _components(depth)
    label = lab[v, u]
    if label == 0:
        near = np.argwhere(lab > 0)
        if not len(near):
            raise ValueError("nothing stands above the tray")
        i = np.argmin(np.hypot(near[:, 0] - v, near[:, 1] - u))
        if np.hypot(near[i, 0] - v, near[i, 1] - u) > 15:
            raise ValueError(f"nothing stands above the tray within 15 px of ({u}, {v})")
        label = lab[near[i, 0], near[i, 1]]
    m = lab == label
    return {**measure(m, depth, rgb), "mask": m}


def target():
    """The scene to restore, from the demonstration's first frame -> {"rgb", "depth", "masks": {name: HxW bool}}."""
    import cv2

    T = HERE / "target"
    masks = {n: cv2.imread(str(T / f"{n}_mask.png"), cv2.IMREAD_GRAYSCALE) > 127 for n in ("gamepad", "cube")}
    return {
        "rgb": cv2.cvtColor(cv2.imread(str(T / "rgb.jpg")), cv2.COLOR_BGR2RGB),
        "depth": np.load(T / "depth.npz")["depth"],
        "masks": masks,
    }


def grasp_yaw(axis_deg):
    """The yaw that lines the demonstration's grasp up with a long axis at axis_deg (the smaller turn of the two)."""
    return round(((axis_deg - DEMO_AXIS_DEG + 90.0) % 180.0) - 90.0, 1)


# --- planning ------------------------------------------------------------------------------------------------------

SSW_SRC = str(HERE.parents[1] / "src")
READY_MOTOR = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -45.0,
    "elbow_flex": 74.0,
    "forearm_roll": 2.0,
    "wrist_flex": -41.0,
    "wrist_roll": -12.0,
    "gripper": 60.0,
}
# Each servo's calibrated range (motor degrees): the real limit. The URDF's are wider in places, and a target past the
# servo's range is clamped silently by the servo (an act once asked wrist_flex for -101 against +/-93.3).
MOTOR_CAL = str(
    pathlib.Path(
        "~/.cache/huggingface/lerobot/calibration/robots/so107_follower/white_left.json"
    ).expanduser()
)
# wrist_roll has a physical stop far inside its calibrated +/-180: it overloaded twice turning past about +97 motor deg
# (2026-10-07 stalled at 96.9 while asked for 108.9; 2026-10-08 overloaded on a move asking for 108.7). Its negative
# side is not measured.
WRIST_ROLL_MAX_MOTOR = 90.0
_KIN = {}


def _kin():
    if not _KIN:
        if SSW_SRC not in sys.path:
            sys.path.insert(0, SSW_SRC)
        from lerobot.gui.api._calib_core import (
            calibration_path,
            corrected_alignment,
            joint_zero_from_calibration,
            load_calibration,
            tip_offset_from_calibration,
        )
        from lerobot.gui.config_paths import gui_config_dir
        from lerobot.robots.so107_description.cartesian_ik import make_so107_arm_kinematics
        from lerobot.robots.so107_description.joint_alignment import LEFT_ARM_ALIGNMENT, MOTOR_NAMES

        cal = load_calibration(calibration_path(gui_config_dir(), "white_left"))
        jz = joint_zero_from_calibration(cal)
        al = corrected_alignment(LEFT_ARM_ALIGNMENT, jz) if jz else LEFT_ARM_ALIGNMENT
        kin = make_so107_arm_kinematics(al, tip_offset=tip_offset_from_calibration(cal))
        inner = kin._inner._inner
        _KIN.update(
            kin=kin, names=list(MOTOR_NAMES), al=al, lo=np.degrees(inner.q_lo), hi=np.degrees(inner.q_hi)
        )
    return _KIN


def reachable(x, y, z, yaw, seed=None):
    """Can the arm put the fingertip at (x, y, z) mm with grasp yaw `yaw`? Solves the arm's own IK offline, from the
    arm's present joints when it is connected (or `seed`, a dict of motor degrees), without moving anything.
    -> {"ok", "err_mm", "err_deg", "motor_deg": {joint: deg}, "why": [...], "seed_next": motor degrees to chain plans}"""
    k = _kin()
    names = k["names"]
    if seed is None:
        code, st = _call("/api/jog/state")
        seed = st["q_obs"] if code == 200 and st.get("connected") else READY_MOTOR
    q = np.array([float(seed.get(m, READY_MOTOR.get(m, 0.0))) for m in names])
    T = np.eye(4)
    T[:3, :3] = _rz(yaw) @ DEMO_GRASP_R
    T[:3, 3] = np.array([x, y, z], float) / 1000.0
    for _ in range(40):
        q_new = k["kin"].inverse_kinematics(q, T)
        q_new[names.index("gripper")] = q[names.index("gripper")]
        if np.max(np.abs(q_new - q)) < 1e-3:
            q = q_new
            break
        q = q_new
    F = k["kin"].forward_kinematics(q)
    err_mm = float(np.linalg.norm(F[:3, 3] - T[:3, 3]) * 1000.0)
    c = np.clip((np.trace(F[:3, :3].T @ T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    err_deg = float(np.degrees(np.arccos(c)))
    why = []
    if err_mm > 3.0 or err_deg > 3.0:
        why.append(f"IK ends {err_mm:.1f} mm / {err_deg:.1f} deg from the target")
    cal = json.loads(pathlib.Path(MOTOR_CAL).read_text())
    for i, m in enumerate(names):
        if m == "gripper":
            continue
        half = (cal[m]["range_max"] - cal[m]["range_min"]) / 2 * 360.0 / 4095.0
        if abs(q[i]) > half - 2.0:
            why.append(f"{m} at {q[i]:.1f} motor deg is at or past its servo range +/-{half:.1f}")
        if m == "wrist_roll" and q[i] > WRIST_ROLL_MAX_MOTOR:
            why.append(
                f"wrist_roll at {q[i]:.1f} motor deg is past {WRIST_ROLL_MAX_MOTOR:.0f}: it stops near 97 and overloads"
            )
    return {
        "ok": not why,
        "err_mm": round(err_mm, 1),
        "err_deg": round(err_deg, 1),
        "motor_deg": {m: round(float(q[i]), 1) for i, m in enumerate(names) if m != "gripper"},
        "why": why,
        "seed_next": {m: float(q[i]) for i, m in enumerate(names)},
    }


# --- moving ------------------------------------------------------------------------------------------------------


def state():
    code, st = _call("/api/jog/state")
    if code != 200 or not st.get("connected"):
        return {"connected": False}
    tip, cmd, g = st["tip_obs"], st["tip_cmd"], st.get("gripper") or {}
    q_cmd, q_obs = st.get("q_cmd") or {}, st.get("q_obs") or {}
    return {
        "connected": True,
        "position": [round(v * 1000, 1) for v in tip["position"]],
        "yaw": round(_yaw_of(_quat_to_R(tip["quaternion"])), 1),
        "position_cmd": [round(v * 1000, 1) for v in cmd["position"]],
        "yaw_cmd": round(_yaw_of(_quat_to_R(cmd["quaternion"])), 1),
        "lag_deg": round(
            max((abs(q_cmd[m] - q_obs[m]) for m in q_cmd if m != "gripper" and m in q_obs), default=0.0), 1
        ),
        "load": dict(st.get("load") or {}),
        "temps": dict(st.get("temps") or {}),
        "q_cmd": dict(q_cmd),
        "q_obs": dict(q_obs),
        "ff_offsets_deg": dict(st.get("ff_offsets_deg") or {}),
        "trim_deg": dict(st.get("trim_deg") or {}),
        "gripper_cmd": round(g.get("cmd") or 0, 1),
        "gripper_obs": round(g.get("obs") or 0, 1),
        "holding": (g.get("cmd") or 0) - (g.get("obs") or 0) > 0.7,
        "halted": st.get("halted"),
        "reason": st.get("reason"),
        "max_motor_temp_c": max((st.get("temps") or {"x": 0}).values()),
    }


def connect():
    code, out = _call(
        "/api/jog/connect",
        {"profile": "white", "arm": "left"},
        timeout=120,
    )
    if code != 200:
        raise RuntimeError(f"connect failed: {out}")
    _call("/api/jog/limits", {"linear_mm_s": 40, "angular_deg_s": 30})


def _move_once(target, yaw):
    R = _rz(yaw) @ DEMO_GRASP_R
    code, out = _call("/api/jog/target", {"position": (target / 1000).tolist(), "quaternion": _R_to_quat(R)})
    if code != 200:
        raise RuntimeError(f"move refused: {out}")
    t0, s, err, still_at, still_since = time.time(), None, None, None, None
    while time.time() - t0 < 20:
        time.sleep(0.2)
        s = state()
        if not s.get("connected") or s.get("halted"):
            raise RuntimeError(f"arm frozen during the move: {s.get('reason')}; park it")
        err = float(np.linalg.norm(np.array(s["position"]) - target))
        if err < 4.0 and abs(_wrap(s["yaw"] - yaw)) < 4.0:
            return {"ok": True, "position": s["position"], "yaw": s["yaw"], "err_mm": round(err, 1)}
        # Held short by the servos: the joints commanded onto the target, the arm stopped within HELD_SHORT_DEG of
        # them. Waiting does not close it: a descent on 2026-10-08 stood 6.6 mm off for 60 s, commanded onto its
        # target, shoulder_lift 0.9 and elbow_flex 1.1 deg short.
        now, here = time.time(), np.array(s["position"])
        if still_at is None or np.abs(here - still_at).max() >= STILL_MM:
            still_at, still_since = here, now
        stopped = now - still_since >= STILL_S
        commanded = (
            float(np.linalg.norm(np.array(s["position_cmd"]) - target)) < 4.0
            and abs(_wrap(s["yaw_cmd"] - yaw)) < 4.0
        )
        if stopped and commanded and s["lag_deg"] <= HELD_SHORT_DEG:
            return {
                "ok": False,
                "held_short": True,
                "position": s["position"],
                "yaw": s["yaw"],
                "err_mm": round(err, 1),
                "lag_deg": s["lag_deg"],
            }
    return {"ok": False, "position": s["position"], "yaw": s["yaw"], "err_mm": round(err, 1)}


def move(x, y, z, yaw=None):
    s = state()
    if not s.get("connected"):
        raise RuntimeError("arm not connected")
    if s.get("halted"):
        raise RuntimeError(f"arm frozen: {s.get('reason')}; park it")
    target = np.array([float(x), float(y), max(float(z), TRAY_Z + 2.0)])
    target[0] = min(max(target[0], WORK_MIN[0]), WORK_MAX[0])
    target[1] = min(max(target[1], WORK_MIN[1]), WORK_MAX[1])
    now = np.array(s["position"])
    if np.linalg.norm(target - now) > STEP_MAX:
        raise ValueError(
            f"{np.linalg.norm(target - now):.0f} mm from the fingertip at {now.tolist()}; move at most {STEP_MAX:.0f} mm at a time"
        )
    yaw = s["yaw"] if yaw is None else float(yaw)
    turn = _wrap(yaw - s["yaw"])
    steps = max(1, int(np.ceil(abs(turn) / TURN_STEP)))
    result = None
    for k in range(1, steps + 1):  # big turns in steps: the jog caps how far one target turns
        result = _move_once(now + (target - now) * k / steps, s["yaw"] + turn * k / steps)
        if not result["ok"]:
            break
    return result


def travel(x, y, z, yaw=None):
    """move() in steps of at most 100 mm along the straight line (lift first yourself: it does not avoid objects)."""
    s = state()
    if not s.get("connected"):
        raise RuntimeError("arm not connected")
    now, target = np.array(s["position"]), np.array([x, y, z], float)
    n = max(1, int(np.ceil(np.linalg.norm(target - now) / (STEP_MAX - 5.0))))
    result = None
    for k in range(1, n + 1):
        p = now + (target - now) * k / n
        result = move(*p, yaw)
        if not result["ok"]:
            break
    return result


def grip(p):
    code, out = _call("/api/jog/gripper", {"pos": float(p)})
    if code != 200:
        raise RuntimeError(f"gripper refused: {out}")
    time.sleep(1.5)
    s = state()
    return {"obs": s.get("gripper_obs"), "holding": float(p) - (s.get("gripper_obs") or float(p)) > 0.7}


def recover():
    """Clear a frozen arm (a motor's overload latch) and hold where it is; moves work again afterwards."""
    code, out = _call("/api/jog/recover", {}, timeout=60)
    if code != 200:
        raise RuntimeError(f"recover failed: {out}")


def park():
    """Fold the arm to rest and switch it off. Holding something, it first sets it down on the tray below the
    fingertip (clearing a freeze if it has to), so nothing drops from the air; then it opens and lifts clear."""
    s = state()
    if not s.get("connected"):
        return
    if s.get("connected") and s.get("holding"):
        try:
            if s.get("halted"):
                recover()
            x, y, _z = state()["position"]
            _move_once(np.array([x, y, max(TRAY_Z + 2.0, tray_z(x, y) + 2.0)]), state()["yaw"])
        except (RuntimeError, ValueError, OSError):
            pass  # still open and park below: better than leaving the arm powered and holding
    _call("/api/jog/gripper", {"pos": 60.0})
    time.sleep(1.0)
    s = state()
    if s.get("connected") and not s.get("halted"):
        x, y, z = s["position"]
        with contextlib.suppress(RuntimeError):
            _move_once(np.array([x, y, z + 40.0]), s["yaw"])
    code, out = _call("/api/jog/park", {}, timeout=120)
    if code != 200:
        raise RuntimeError(f"park failed: {out}")


# --- command line ------------------------------------------------------------------------------------------------


def main(argv):
    cmd, a = (argv[0] if argv else "help"), argv[1:]
    if cmd == "look":
        look()
        return print(
            "saved look.jpg (grid every 50 px) and look_raw.jpg, 848x480; view it with your image viewer"
        )
    if cmd == "where":
        return print(
            f"pixel ({a[0]}, {a[1]}) is at arm-frame {np.round(where(int(a[0]), int(a[1])), 1).tolist()} mm"
        )
    if cmd == "pixel":
        return print(f"arm-frame {a} mm appears at pixel {pixel(*map(float, a))}")
    if cmd == "obj":
        o = obj(int(a[0]), int(a[1]))
        o.pop("mask")
        return print(json.dumps(o))
    if cmd == "objects":
        return print("\n".join(json.dumps({k: v for k, v in o.items() if k != "mask"}) for o in objects()))
    if cmd == "state":
        return print(json.dumps(state()))
    if cmd == "connect":
        connect()
        return print("connected")
    if cmd == "move":
        return print(json.dumps(move(*map(float, a))))
    if cmd == "grip":
        return print(json.dumps(grip(float(a[0]))))
    if cmd == "park":
        park()
        return print("parked")
    print(__doc__)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}")
