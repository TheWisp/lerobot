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


_calib = _Calib()


class MarkersQuery(BaseModel):
    dictionary: str = "DICT_4X4_50"
    side_mm: float | None = None  # printed marker side; enables the perspective-n-point lift


class CameraTouchBody(BaseModel):
    marker_id: int
    corner: int = 0  # ArUco order: 0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left


class CameraSolveBody(BaseModel):
    source: str = "depth"  # "depth" | "pnp"


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


@router.get("/state")
async def state() -> dict:
    from lerobot.gui.config_paths import gui_config_dir

    from . import jog

    c = _calib
    rid = jog.current_robot_id()
    saved = _saved(rid) if rid else {}
    path = str(core.calibration_path(gui_config_dir(), rid)) if rid else None
    with c.lock:
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
            "robot_id": rid,
            "saved": {
                "path": path,
                "tool_point": saved.get("tool_point"),
                "camera": saved.get("camera"),
                "saved_at": saved.get("saved_at"),
            },
            "tool": {"touches": list(c.tool_touches), "result": c.tool_result},
            "markers": markers,
            "camera": {"touches": list(c.camera_touches), "result": c.camera_result},
        }


# ── tool point ──────────────────────────────────────────────────────────────


@router.post("/tool/touch")
async def tool_touch() -> dict:
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
    return {"n": n}


@router.post("/tool/undo")
async def tool_undo() -> dict:
    c = _calib
    with c.lock:
        if c.tool_touches:
            c.tool_touches.pop()
        c.tool_result = None
        return {"n": len(c.tool_touches)}


@router.post("/tool/clear")
async def tool_clear() -> dict:
    c = _calib
    with c.lock:
        c.tool_touches.clear()
        c.tool_result = None
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
        result = c.tool_result
        touches = list(c.tool_touches)
    if result is None:
        raise HTTPException(409, "solve the tool point first")
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
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for m in markers:
        quad = np.asarray(m["corners_px"], dtype=np.int32)
        cv2.polylines(bgr, [quad], True, (0, 220, 255), 2)
        tl = tuple(quad[0])
        cv2.circle(bgr, tl, 6, (0, 0, 255), 2)  # corner 0, the one to touch
        cv2.putText(
            bgr, str(m["id"]), (tl[0] + 8, tl[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2
        )
        if m["corners_depth_m"] is not None:
            z = m["corners_depth_m"][0][2]
            cv2.putText(
                bgr, f"{z:.3f} m", (tl[0] + 8, tl[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
    _ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    det = {
        "at": time.strftime("%H:%M:%S"),
        "dictionary": dictionary,
        "side_mm": side_mm,
        "intrinsics": intr,
        "markers": markers,
    }
    return det, jpeg.tobytes()


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


@router.get("/markers.jpg")
async def markers_jpeg() -> Response:
    c = _calib
    with c.lock:
        data = c.markers_jpeg
    if data is None:
        raise HTTPException(404, "no detection yet")
    return Response(content=data, media_type="image/jpeg")


# ── camera to base ───────────────────────────────────────────────────────────


@router.post("/camera/touch")
async def camera_touch(body: CameraTouchBody) -> dict:
    """Record the fingertip on a marker corner: base side from FK now, camera side from the last detection."""
    if body.corner not in (0, 1, 2, 3):
        raise HTTPException(422, "corner is 0..3")
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
    return {"n": n, "touch": touch}


@router.post("/camera/undo")
async def camera_undo() -> dict:
    c = _calib
    with c.lock:
        if c.camera_touches:
            c.camera_touches.pop()
        c.camera_result = None
        return {"n": len(c.camera_touches)}


@router.post("/camera/clear")
async def camera_clear() -> dict:
    c = _calib
    with c.lock:
        c.camera_touches.clear()
        c.camera_result = None
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
async def camera_save() -> dict:
    """Write the camera-to-base transform for this arm, tagged with the detection's intrinsics."""
    rid = _robot_id_or_409()
    c = _calib
    with c.lock:
        result, det = c.camera_result, c.markers
        touches = list(c.camera_touches)
    if result is None:
        raise HTTPException(409, "solve the camera transform first")
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
