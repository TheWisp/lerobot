#!/usr/bin/env python

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

"""Touch calibrations for one SO-107 arm: the fingertip, then the camera.

Both are driven from the jog: the operator puts the fingertip on a point and
presses "touch", and the server records the arm's FK at that instant. The
tool-point calibration touches ONE point from several orientations; the
camera calibration touches the corners of ArUco markers the camera located
before the gripper covered them. Results are saved per arm under the GUI's
config directory and the jog picks the fingertip up on its next connect.

The arm is read from the jog's cached encoders (no bus traffic here); the
camera is the show-and-servo session's RealSense, read on its own executor.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from . import _calib_core as core

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/calib", tags=["calib"])
# The joint-zero refinement evaluates FK a few thousand times (seconds); it
# must not run on the loop or on the camera's or the arm's worker.
_REFINE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="calib-refine")

MARKER_DICTIONARIES = ("DICT_4X4_50", "DICT_5X5_50", "DICT_6X6_50", "DICT_APRILTAG_36h11")


@dataclass
class _Calib:
    lock: threading.Lock = field(default_factory=threading.Lock)
    tool_touches: list[dict[str, Any]] = field(default_factory=list)
    tool_result: dict[str, Any] | None = None
    markers: dict[str, Any] | None = None  # last detection, corners lifted to 3D
    markers_jpeg: bytes | None = None
    camera_touches: list[dict[str, Any]] = field(default_factory=list)
    camera_result: dict[str, Any] | None = None
    refine_result: dict[str, Any] | None = None
    restored: bool = False


_calib = _Calib()


def _session_path():
    from lerobot.gui.config_paths import gui_config_dir

    return core.calibration_path(gui_config_dir(), "session")


def _persist_locked(c: _Calib) -> None:
    """Touches survive a server restart; the detection frame does not (re-detect)."""
    import json

    from . import jog

    path = _session_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "robot_id": jog.current_robot_id(),
                "tool_touches": c.tool_touches,
                "camera_touches": c.camera_touches,
            }
        )
    )


def _restore_once() -> None:
    """Load the persisted touches on first use (not at import: the config dir is resolved per process)."""
    import json

    c = _calib
    with c.lock:
        if c.restored:
            return
        c.restored = True
        path = _session_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            c.tool_touches = list(data.get("tool_touches", []))
            c.camera_touches = list(data.get("camera_touches", []))
        except Exception:  # a damaged session file is not worth refusing to start over
            logger.exception("could not restore the calibration session")


class MarkersQuery(BaseModel):
    dictionary: str = "DICT_4X4_50"
    side_mm: float | None = None  # printed marker side; enables the perspective-n-point lift


class CameraTouchBody(BaseModel):
    marker_id: int
    corner: int = 0  # ArUco order: 0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left


class CameraSolveBody(BaseModel):
    source: str = "depth"  # "depth" | "pnp"


class GotoBody(BaseModel):
    marker_id: int
    corner: int = 0
    hover_mm: float = 10.0


def _arm_or_409():
    from . import jog

    cur = jog.current_tip_and_anchor()
    if cur is None:
        raise HTTPException(409, "connect an arm in the Jog panel first")
    return cur


def _robot_id_or_409() -> str:
    from . import jog

    rid = jog.current_robot_id()
    if rid is None:
        raise HTTPException(409, "connect an arm in the Jog panel first")
    return rid


def _saved(robot_id: str) -> dict[str, Any]:
    from lerobot.gui.config_paths import gui_config_dir

    return core.load_calibration(core.calibration_path(gui_config_dir(), robot_id))


def _write(robot_id: str, data: dict[str, Any]) -> str:
    from lerobot.gui.config_paths import gui_config_dir

    path = core.calibration_path(gui_config_dir(), robot_id)
    core.save_calibration(path, data)
    return str(path)


def _rotation_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    from scipy.spatial.transform import Rotation

    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(r_a @ r_b.T).as_rotvec())))


def _auto_solve_tool(touches: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The tool fit for the touches so far, or ``{"error": ...}`` once there are enough to try."""
    if len(touches) < core.MIN_TOOL_TOUCHES:
        return None
    try:
        return core.solve_tool_point([np.asarray(t["anchor"]) for t in touches])
    except ValueError as e:
        return {"error": str(e)}


def _auto_solve_camera(touches: list[dict[str, Any]], source: str) -> dict[str, Any] | None:
    key = "cam_depth_m" if source == "depth" else "cam_pnp_m"
    usable = [t for t in touches if t[key] is not None]
    if len(usable) < core.MIN_CAMERA_TOUCHES:
        return None
    try:
        fit = core.rigid_fit(np.array([t[key] for t in usable]), np.array([t["base_m"] for t in usable]))
    except ValueError as e:
        return {"error": str(e)}
    return {**fit, "source": source, "touch_ids": [f"{t['marker_id']}.{t['corner']}" for t in usable]}


@router.get("/state")
async def state() -> dict:
    from lerobot.gui.config_paths import gui_config_dir

    from . import jog, showservo

    _restore_once()
    c = _calib
    rid = jog.current_robot_id()
    saved = _saved(rid) if rid else {}
    path = str(core.calibration_path(gui_config_dir(), rid)) if rid else None
    live = None
    cur = jog.current_tip_and_anchor()
    if cur is not None:
        t_tip, t_anchor, _q = cur
        with c.lock:
            prev = [np.asarray(t["anchor"])[:3, :3] for t in c.tool_touches]
        live = {
            "tip_mm": (t_tip[:3, 3] * 1000.0).tolist(),
            "rotation_from_touches_deg": [_rotation_deg(t_anchor[:3, :3], r) for r in prev],
            "tip_calibrated": jog.current_tip_calibrated(),
        }
    with c.lock:
        tool_result = c.tool_result or _auto_solve_tool(c.tool_touches)
        camera_auto = {
            "depth": _auto_solve_camera(c.camera_touches, "depth"),
            "pnp": _auto_solve_camera(c.camera_touches, "pnp"),
        }
        markers = None
        if c.markers is not None:
            markers = {
                "at": c.markers["at"],
                "dictionary": c.markers["dictionary"],
                "side_mm": c.markers["side_mm"],
                "markers": [
                    {
                        "id": m["id"],
                        "corners_px": m["corners_px"],
                        "depth_ok": m["corners_depth_m"] is not None,
                        "pnp_ok": m["corners_pnp_m"] is not None,
                        "depth_diag": m["depth_diag"],
                    }
                    for m in c.markers["markers"]
                ],
            }
        return {
            "arm_connected": rid is not None,
            "camera_live": showservo.live_camera() is not None,
            "robot_id": rid,
            "live": live,
            "saved": {
                "path": path,
                "tool_point": saved.get("tool_point"),
                "camera": saved.get("camera"),
                "joint_zero_deg": saved.get("joint_zero_deg"),
                "saved_at": saved.get("saved_at"),
            },
            "tool": {"touches": list(c.tool_touches), "result": tool_result},
            "markers": markers,
            "camera": {"touches": list(c.camera_touches), "result": c.camera_result, "auto": camera_auto},
            "refine": c.refine_result,
        }


# ── tool point ──────────────────────────────────────────────────────────────


@router.post("/tool/touch")
async def tool_touch() -> dict:
    _restore_once()
    t_tip, t_anchor, q_obs = _arm_or_409()
    c = _calib
    with c.lock:
        c.tool_touches.append(
            {
                "anchor": t_anchor.tolist(),
                "tip_m": t_tip[:3, 3].tolist(),
                "q_obs": q_obs,
                "at": time.strftime("%H:%M:%S"),
            }
        )
        c.tool_result = None
        n = len(c.tool_touches)
        _persist_locked(c)
    return {"n": n}


@router.post("/tool/undo")
async def tool_undo() -> dict:
    c = _calib
    with c.lock:
        if c.tool_touches:
            c.tool_touches.pop()
        c.tool_result = None
        _persist_locked(c)
        return {"n": len(c.tool_touches)}


@router.post("/tool/clear")
async def tool_clear() -> dict:
    c = _calib
    with c.lock:
        c.tool_touches.clear()
        c.tool_result = None
        _persist_locked(c)
    return {"n": 0}


@router.post("/tool/solve")
async def tool_solve() -> dict:
    c = _calib
    with c.lock:
        poses = [np.asarray(t["anchor"]) for t in c.tool_touches]
    try:
        result = core.solve_tool_point(poses)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    with c.lock:
        c.tool_result = result
    return result


@router.post("/tool/save")
async def tool_save() -> dict:
    """Write the solved fingertip for this arm. The jog applies it on its next connect."""
    rid = _robot_id_or_409()
    c = _calib
    with c.lock:
        touches = list(c.tool_touches)
        result = c.tool_result or _auto_solve_tool(touches)
    if result is None or "error" in result:
        raise HTTPException(409, (result or {}).get("error", "not enough touches to solve the tool point"))
    data = _saved(rid)
    data["tool_point"] = {**result, "touches": touches}
    path = _write(rid, data)
    return {"path": path, "offset_mm": (np.asarray(result["offset_m"]) * 1000.0).tolist()}


# ── markers ──────────────────────────────────────────────────────────────────


def _detect(camera: Any, dictionary: str, side_mm: float | None) -> tuple[dict[str, Any], bytes]:
    import cv2

    rgb, depth_mm = camera.read_color_and_aligned_depth()
    intr = camera.color_intrinsics()
    depth_m = depth_mm.astype(np.float32) / 1000.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    found = core.detect_markers(gray, dictionary)
    markers = []
    for m in found:
        px = np.asarray(m["corners_px"])
        try:
            cd, diag = core.corners_from_depth_plane(px, depth_m, intr)
            corners_depth, depth_diag = cd.tolist(), diag
        except ValueError as e:
            corners_depth, depth_diag = None, {"error": str(e)}
        corners_pnp = None
        if side_mm:
            try:
                corners_pnp = core.corners_from_pnp(px, side_mm / 1000.0, intr).tolist()
            except ValueError:
                corners_pnp = None
        markers.append(
            {
                "id": m["id"],
                "corners_px": m["corners_px"],
                "corners_depth_m": corners_depth,
                "depth_diag": depth_diag,
                "corners_pnp_m": corners_pnp,
            }
        )
    det = {
        "at": time.strftime("%H:%M:%S"),
        "dictionary": dictionary,
        "side_mm": side_mm,
        "intrinsics": intr,
        "markers": markers,
        "rgb": rgb,
    }
    return det, _render(det, set(), None)


def _render(det: dict[str, Any], touched: set[tuple[int, int]], target: int | None) -> bytes:
    """The detection frame with every marker outlined: touched ones green, the target red, the rest yellow."""
    import cv2

    bgr = cv2.cvtColor(det["rgb"], cv2.COLOR_RGB2BGR)
    for m in det["markers"]:
        quad = np.asarray(m["corners_px"], dtype=np.int32)
        done = (m["id"], 0) in touched
        colour = (60, 200, 60) if done else ((0, 0, 255) if m["id"] == target else (0, 220, 255))
        cv2.polylines(bgr, [quad], True, colour, 3 if m["id"] == target else 2)
        tl = tuple(quad[0])
        cv2.circle(bgr, tl, 9 if m["id"] == target else 6, colour, 2)  # corner 0, the one to touch
        label = f"{m['id']}" + (" done" if done else (" <- touch" if m["id"] == target else ""))
        cv2.putText(bgr, label, (tl[0] + 10, tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    _ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg.tobytes()


@router.post("/markers")
async def markers(body: MarkersQuery) -> dict:
    """Detect markers on a fresh frame and lift their corners to camera 3D. Do this before touching."""
    from . import showservo

    if body.dictionary not in MARKER_DICTIONARIES:
        raise HTTPException(422, f"dictionary must be one of {MARKER_DICTIONARIES}")
    camera = showservo.live_camera()
    if camera is None:
        raise HTTPException(409, "start a live camera session in the Servo tab first")
    try:
        det, jpeg = await asyncio.get_event_loop().run_in_executor(
            showservo._EXECUTOR, _detect, camera, body.dictionary, body.side_mm
        )
    except Exception as e:
        raise HTTPException(500, f"marker detection failed: {e}") from e
    c = _calib
    with c.lock:
        c.markers, c.markers_jpeg = det, jpeg
    return {"n": len(det["markers"]), "ids": [m["id"] for m in det["markers"]], "at": det["at"]}


@router.get("/markers/sheet.pdf")
async def marker_sheet(dictionary: str = "DICT_4X4_50", side_mm: float = 40.0, count: int = 8) -> Response:
    """Printable marker sheet at a known physical size (print at 100 %; a scale bar checks it)."""
    if dictionary not in MARKER_DICTIONARIES:
        raise HTTPException(422, f"dictionary must be one of {MARKER_DICTIONARIES}")
    if not 15 <= side_mm <= 80 or not 1 <= count <= 12:
        raise HTTPException(422, "side_mm is 15..80 and count 1..12")
    try:
        pdf = core.marker_sheet_pdf(dictionary, side_mm, count)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="aruco_{dictionary}_{side_mm:g}mm.pdf"'},
    )


@router.get("/markers.jpg")
async def markers_jpeg(target: int | None = None) -> Response:
    """The last detection, redrawn with touched markers and the marker to touch next."""
    c = _calib
    with c.lock:
        det = c.markers
        touched = {(t["marker_id"], t["corner"]) for t in c.camera_touches}
    if det is None:
        raise HTTPException(404, "no detection yet")
    return Response(content=_render(det, touched, target), media_type="image/jpeg")


# ── camera to base ───────────────────────────────────────────────────────────


@router.post("/camera/touch")
async def camera_touch(body: CameraTouchBody) -> dict:
    """Record the fingertip on a marker corner: base side from FK now, camera side from the last detection."""
    if body.corner not in (0, 1, 2, 3):
        raise HTTPException(422, "corner is 0..3")
    _restore_once()
    t_tip, _anchor, q_obs = _arm_or_409()
    c = _calib
    with c.lock:
        det = c.markers
        if det is None:
            raise HTTPException(409, "detect markers first, before the gripper covers them")
        found = [m for m in det["markers"] if m["id"] == body.marker_id]
        if not found:
            raise HTTPException(404, f"marker {body.marker_id} is not in the last detection")
        m = found[0]
        if any(t["marker_id"] == body.marker_id and t["corner"] == body.corner for t in c.camera_touches):
            raise HTTPException(
                409, f"marker {body.marker_id} corner {body.corner} is already touched; undo it first"
            )
        touch = {
            "marker_id": body.marker_id,
            "corner": body.corner,
            "pixel": m["corners_px"][body.corner],
            "cam_depth_m": None if m["corners_depth_m"] is None else m["corners_depth_m"][body.corner],
            "cam_pnp_m": None if m["corners_pnp_m"] is None else m["corners_pnp_m"][body.corner],
            "base_m": t_tip[:3, 3].tolist(),
            "q_obs": q_obs,
            "at": time.strftime("%H:%M:%S"),
        }
        c.camera_touches.append(touch)
        c.camera_result = None
        n = len(c.camera_touches)
        _persist_locked(c)
    return {"n": n, "touch": touch}


@router.post("/camera/undo")
async def camera_undo() -> dict:
    c = _calib
    with c.lock:
        if c.camera_touches:
            c.camera_touches.pop()
        c.camera_result = None
        _persist_locked(c)
        return {"n": len(c.camera_touches)}


@router.post("/camera/clear")
async def camera_clear() -> dict:
    c = _calib
    with c.lock:
        c.camera_touches.clear()
        c.camera_result = None
        _persist_locked(c)
    return {"n": 0}


@router.post("/camera/solve")
async def camera_solve(body: CameraSolveBody) -> dict:
    if body.source not in ("depth", "pnp"):
        raise HTTPException(422, "source is 'depth' or 'pnp'")
    key = "cam_depth_m" if body.source == "depth" else "cam_pnp_m"
    c = _calib
    with c.lock:
        touches = [t for t in c.camera_touches if t[key] is not None]
        skipped = len(c.camera_touches) - len(touches)
    if len(touches) < core.MIN_CAMERA_TOUCHES:
        raise HTTPException(
            422, f"{len(touches)} touches have a {body.source} position; need {core.MIN_CAMERA_TOUCHES}"
        )
    src = np.array([t[key] for t in touches])
    dst = np.array([t["base_m"] for t in touches])
    try:
        fit = core.rigid_fit(src, dst)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    result = {
        **fit,
        "source": body.source,
        "skipped": skipped,
        "touch_ids": [f"{t['marker_id']}.{t['corner']}" for t in touches],
    }
    with c.lock:
        c.camera_result = result
    return result


@router.post("/camera/save")
async def camera_save(body: CameraSolveBody) -> dict:
    """Write the camera-to-base transform for this arm, tagged with the detection's intrinsics."""
    rid = _robot_id_or_409()
    c = _calib
    with c.lock:
        det = c.markers
        touches = list(c.camera_touches)
        result = _auto_solve_camera(touches, body.source)
    if result is None or "error" in result:
        raise HTTPException(
            409, (result or {}).get("error", f"not enough touches with a {body.source} position")
        )
    data = _saved(rid)
    data["camera"] = {
        "T_base_cam": result["transform"],
        "rms_m": result["rms_m"],
        "max_m": result["max_m"],
        "scale": result["scale"],
        "source": result["source"],
        "n": result["n"],
        "intrinsics": det["intrinsics"] if det else None,
        "touches": touches,
    }
    path = _write(rid, data)
    return {"path": path, "rms_mm": result["rms_m"] * 1000.0}


# ── refinement: joint zeros, fingertip and camera pose from every touch at once ─


def _refine(arm: str, tool_touches: list, camera_touches: list, source: str) -> dict[str, Any]:
    from lerobot.robots.so107_description.cartesian_ik import make_so107_arm_kinematics
    from lerobot.robots.so107_description.joint_alignment import (
        LEFT_ARM_ALIGNMENT,
        MOTOR_NAMES,
        RIGHT_ARM_ALIGNMENT,
        TIP_OFFSET,
    )

    # Corrections are absolute, so the FK here is the uncorrected model with the
    # correction added on the motor side: urdf = sign*(q+dq) + offset.
    base = LEFT_ARM_ALIGNMENT if arm == "left" else RIGHT_ARM_ALIGNMENT
    kin = make_so107_arm_kinematics(base)
    inv_tip = np.linalg.inv(TIP_OFFSET)
    idx = {m: i for i, m in enumerate(MOTOR_NAMES)}

    def fk_anchor(q, dq):
        qq = np.array(q, dtype=float)
        for m, v in dq.items():
            qq[idx[m]] += v
        return kin.forward_kinematics(qq) @ inv_tip

    key = "cam_depth_m" if source == "depth" else "cam_pnp_m"
    return core.refine_kinematics(fk_anchor, MOTOR_NAMES, tool_touches, camera_touches, camera_key=key)


@router.post("/refine")
async def refine(body: CameraSolveBody) -> dict:
    """Fit joint-zero corrections, the fingertip and the camera pose to all touches; nothing is saved yet."""
    from . import jog

    if body.source not in ("depth", "pnp"):
        raise HTTPException(422, "source is 'depth' or 'pnp'")
    _restore_once()
    arm = jog.current_arm()
    if arm is None:
        raise HTTPException(409, "connect an arm in the Jog panel first")
    c = _calib
    with c.lock:
        tool, cam = list(c.tool_touches), list(c.camera_touches)
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            _REFINE_EXECUTOR, _refine, arm, tool, cam, body.source
        )
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"refine failed: {e}") from e
    with c.lock:
        c.refine_result = result
    return result


@router.post("/refine/save")
async def refine_save() -> dict:
    """Write the refined zeros, fingertip and camera pose together. The jog applies them on its next connect."""
    rid = _robot_id_or_409()
    c = _calib
    with c.lock:
        result, det = c.refine_result, c.markers
        tool, cam = list(c.tool_touches), list(c.camera_touches)
    if result is None:
        raise HTTPException(409, "run the refinement first")
    data = _saved(rid)
    data["joint_zero_deg"] = result["joint_zero_deg"]
    data["tool_point"] = {
        "offset_m": result["offset_m"],
        "point_m": result["point_m"],
        "residuals_m": result["tool_residuals_m"],
        "rms_m": result["tool_rms_m"],
        "max_m": max(result["tool_residuals_m"]),
        "n": len(result["tool_residuals_m"]),
        "touches": tool,
        "refined": True,
    }
    data["camera"] = {
        "T_base_cam": result["transform"],
        "rms_m": result["camera_rms_m"],
        "max_m": max(result["camera_residuals_m"]),
        "scale": None,
        "source": "depth" if result["source"] == "cam_depth_m" else "pnp",
        "n": len(result["camera_residuals_m"]),
        "intrinsics": det["intrinsics"] if det else None,
        "touches": cam,
        "refined": True,
    }
    path = _write(rid, data)
    return {
        "path": path,
        "camera_rms_mm": result["camera_rms_m"] * 1000.0,
        "tool_rms_mm": result["tool_rms_m"] * 1000.0,
    }


@router.post("/goto")
async def goto(body: GotoBody) -> dict:
    """Walk the fingertip to a detected marker corner, from the camera's coordinates through the saved transform.

    The end-to-end check: camera -> base -> IK -> arm. Pre: the jog is connected
    with the saved calibration loaded (reconnect after saving), and the corner
    is in the last detection. The tip keeps its current orientation; only the
    position moves, to ``hover_mm`` above the corner.
    """
    from . import jog

    rid = _robot_id_or_409()
    saved = _saved(rid)
    cam = saved.get("camera")
    if not cam:
        raise HTTPException(409, "no saved camera transform for this arm")
    st = jog.current_calibration_state()
    if saved.get("joint_zero_deg", {}) != st["joint_zero_deg"] or not st["tip_calibrated"]:
        raise HTTPException(409, "the jog is not running the saved calibration — disconnect and reconnect it")
    c = _calib
    with c.lock:
        det = c.markers
    if det is None:
        raise HTTPException(409, "detect markers first")
    found = [m for m in det["markers"] if m["id"] == body.marker_id]
    if not found or found[0]["corners_depth_m"] is None:
        raise HTTPException(404, f"marker {body.marker_id} has no depth position in the last detection")
    corner_cam = np.asarray(found[0]["corners_depth_m"][body.corner])
    t_bc = np.asarray(cam["T_base_cam"])
    corner_base = t_bc[:3, :3] @ corner_cam + t_bc[:3, 3]
    cur = jog.current_tip_and_anchor()
    if cur is None:
        raise HTTPException(409, "no arm connected")
    pose = cur[0].copy()
    pose[:3, 3] = corner_base + np.array([0.0, 0.0, body.hover_mm / 1000.0])
    try:
        jog.set_target_pose(pose)
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {
        "target_base_mm": (pose[:3, 3] * 1000.0).tolist(),
        "corner_base_mm": (corner_base * 1000.0).tolist(),
    }
