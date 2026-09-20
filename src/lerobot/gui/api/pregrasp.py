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

"""Teach one pre-grasp pose on an object, move the object, go there again.

The smallest show-and-servo loop that needs no gripper tracking: the taught
pose is the fingertip's FK, the object's motion comes from the top camera, and
the camera-to-base calibration joins the two. Frames come from the
show-and-servo session's RealSense on its executor; the arm is the jog's.
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

from . import _pregrasp_core as core

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/pregrasp", tags=["pregrasp"])


@dataclass
class _Teach:
    at: str
    box: tuple[int, int, int, int]
    rgb: np.ndarray
    depth_m: np.ndarray
    intr: dict[str, float]
    keypoints: dict[str, Any]
    tip_pose: np.ndarray | None = None  # base frame, 4x4


@dataclass
class _Test:
    at: str
    rgb: np.ndarray
    result: dict[str, Any]
    transported: np.ndarray | None = None  # base frame, 4x4


@dataclass
class _State:
    lock: threading.Lock = field(default_factory=threading.Lock)
    teach: _Teach | None = None
    test: _Test | None = None


_state = _State()
_HEAVY = ("delta_cam", "teach_uv", "live_uv", "live_mask")


class TeachBody(BaseModel):
    box: list[int]  # x0, y0, x1, y1 in frame pixels


class GoBody(BaseModel):
    hover_mm: float = 20.0


def _grab(camera: Any) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    rgb, depth_mm = camera.read_color_and_aligned_depth()
    return rgb, depth_mm.astype(np.float32) / 1000.0, camera.color_intrinsics()


async def _frame():
    from . import showservo

    camera = showservo.live_camera()
    if camera is None:
        raise HTTPException(409, "start a live camera session in the Servo tab first")
    try:
        return await asyncio.get_event_loop().run_in_executor(showservo._EXECUTOR, _grab, camera)
    except Exception as e:
        raise HTTPException(500, f"camera read failed: {e}") from e


def _t_base_cam() -> np.ndarray:
    from lerobot.gui.config_paths import gui_config_dir

    from . import _calib_core, jog

    rid = jog.current_robot_id()
    if rid is None:
        raise HTTPException(409, "connect an arm in the Jog panel first")
    saved = _calib_core.load_calibration(_calib_core.calibration_path(gui_config_dir(), rid))
    cam = saved.get("camera")
    if not cam:
        raise HTTPException(409, "no camera-to-base calibration for this arm; run the touch calibration")
    return np.asarray(cam["T_base_cam"], dtype=float)


def _jpeg(bgr: np.ndarray) -> bytes:
    import cv2

    _ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def _project(t_base_cam: np.ndarray, intr: dict[str, float], p_base: np.ndarray) -> tuple[int, int] | None:
    p_cam = np.linalg.inv(t_base_cam) @ np.append(p_base, 1.0)
    if p_cam[2] <= 1e-6:
        return None
    return int(round(intr["fx"] * p_cam[0] / p_cam[2] + intr["cx"])), int(
        round(intr["fy"] * p_cam[1] / p_cam[2] + intr["cy"])
    )


@router.get("/state")
async def state() -> dict:
    from . import jog, showservo

    s = _state
    with s.lock:
        teach, test = s.teach, s.test
    out: dict[str, Any] = {
        "camera_live": showservo.live_camera() is not None,
        "arm_connected": jog.current_robot_id() is not None,
        "teach": None,
        "test": None,
    }
    if teach is not None:
        out["teach"] = {
            "at": teach.at,
            "box": list(teach.box),
            **_teach_info(teach.keypoints),
            "tip_mm": None if teach.tip_pose is None else (teach.tip_pose[:3, 3] * 1000.0).tolist(),
        }
    if test is not None:
        r = test.result
        info = {k: v for k, v in r.items() if k not in _HEAVY}
        if r.get("ok"):
            info["motion"] = core.motion_summary(r["delta_cam"])
        if test.transported is not None:
            info["transported_tip_mm"] = (test.transported[:3, 3] * 1000.0).tolist()
        out["test"] = {"at": test.at, **info}
    return out


@router.get("/frame.jpg")
async def frame_jpeg() -> Response:
    """A fresh colour frame to draw the object box on."""
    import cv2

    rgb, _depth, _intr = await _frame()
    return Response(content=_jpeg(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)), media_type="image/jpeg")


@router.post("/teach/capture")
async def teach_capture(body: TeachBody) -> dict:
    """Keypoints of the object inside the box, lifted by depth. Do this before jogging in over the object."""
    if len(body.box) != 4:
        raise HTTPException(422, "box is x0, y0, x1, y1")
    x0, y0, x1, y1 = body.box
    box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    rgb, depth_m, intr = await _frame()
    # Texture first; a plain object falls back to its shape above the table.
    kp, texture_reason = None, ""
    try:
        kp = core.keypoints_in_box(rgb, depth_m, intr, box)
        if int(kp["valid"].sum()) < core.MIN_TEXTURE_POINTS:
            texture_reason = f"only {int(kp['valid'].sum())} textured points"
            kp = None
    except ValueError as e:
        texture_reason = str(e)
    if kp is None:
        try:
            kp = core.shape_teach(depth_m, intr, box)
        except ValueError as e:
            raise HTTPException(422, f"{texture_reason}; and {e}") from e
    else:
        kp["mode"] = "texture"
        # Keep the shape model too: weak texture that re-matches badly falls back to it at Find.
        try:
            kp["shape"] = core.shape_teach(depth_m, intr, box)
        except ValueError:
            kp["shape"] = None
    with _state.lock:
        _state.teach = _Teach(
            at=time.strftime("%H:%M:%S"), box=box, rgb=rgb, depth_m=depth_m, intr=intr, keypoints=kp
        )
        _state.test = None
    return _teach_info(kp)


def _teach_info(kp: dict[str, Any]) -> dict[str, Any]:
    if kp["mode"] == "texture":
        return {
            "mode": "texture",
            "n_keypoints": int(len(kp["uv"])),
            "n_with_depth": int(kp["valid"].sum()),
            "shape_fallback": kp.get("shape") is not None,
        }
    return {
        "mode": "shape",
        "n_points": int(kp["n_points"]),
        "height_mm": float(kp["height_m"] * 1000.0),
        "symmetric": bool(kp["elongation"] < 1.3),
    }


@router.post("/teach/mark")
async def teach_mark() -> dict:
    """Record the fingertip's present pose as the pre-grasp for the taught object."""
    from . import jog

    cur = jog.current_tip_and_anchor()
    if cur is None:
        raise HTTPException(409, "connect an arm in the Jog panel first")
    with _state.lock:
        if _state.teach is None:
            raise HTTPException(409, "capture the object first")
        _state.teach.tip_pose = cur[0].copy()
        tip = _state.teach.tip_pose[:3, 3]
    return {"tip_mm": (tip * 1000.0).tolist()}


@router.get("/teach.jpg")
async def teach_jpeg() -> Response:
    import cv2

    with _state.lock:
        teach = _state.teach
    if teach is None:
        raise HTTPException(404, "nothing taught")
    bgr = cv2.cvtColor(teach.rgb, cv2.COLOR_RGB2BGR)
    x0, y0, x1, y1 = teach.box
    cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 220, 255), 2)
    if teach.keypoints["mode"] == "texture":
        for (u, v), ok in zip(teach.keypoints["uv"], teach.keypoints["valid"], strict=True):
            cv2.circle(bgr, (int(u), int(v)), 3, (60, 200, 60) if ok else (0, 0, 255), 1)
    else:
        _outline(bgr, teach.keypoints["mask"], (60, 200, 60))
    if teach.tip_pose is not None:
        try:
            px = _project(_t_base_cam(), teach.intr, teach.tip_pose[:3, 3])
        except HTTPException:
            px = None
        if px is not None:
            cv2.drawMarker(bgr, px, (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(
                bgr, "pre-grasp", (px[0] + 10, px[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
    return Response(content=_jpeg(bgr), media_type="image/jpeg")


@router.post("/test/capture")
async def test_capture() -> dict:
    """Find the taught object in a fresh frame and transport the pre-grasp by its motion."""
    with _state.lock:
        teach = _state.teach
    if teach is None:
        raise HTTPException(409, "teach first")
    if teach.tip_pose is None:
        raise HTTPException(409, "mark the pre-grasp first")
    t_bc = _t_base_cam()
    rgb, depth_m, intr = await _frame()
    if teach.keypoints["mode"] == "texture":
        result = core.register(teach.keypoints, rgb, depth_m, intr)
        if not result.get("ok") and teach.keypoints.get("shape") is not None:
            texture_reason = result.get("reason", "texture failed")
            result = core.shape_register(teach.keypoints["shape"], depth_m, intr)
            result["fallback_from"] = f"texture ({texture_reason})"
    else:
        result = core.shape_register(teach.keypoints, depth_m, intr)
    transported = core.transport_pose(t_bc, result["delta_cam"], teach.tip_pose) if result.get("ok") else None
    with _state.lock:
        _state.test = _Test(at=time.strftime("%H:%M:%S"), rgb=rgb, result=result, transported=transported)
    info = {k: v for k, v in result.items() if k not in _HEAVY}
    if result.get("ok"):
        info["motion"] = core.motion_summary(result["delta_cam"])
        info["transported_tip_mm"] = (transported[:3, 3] * 1000.0).tolist()
    return info


def _outline(bgr: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int]) -> None:
    import cv2

    contours, _h = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgr, contours, -1, colour, 2)


@router.get("/test.jpg")
async def test_jpeg() -> Response:
    import cv2

    with _state.lock:
        teach, test = _state.teach, _state.test
    if test is None or teach is None:
        raise HTTPException(404, "no test capture")
    bgr = cv2.cvtColor(test.rgb, cv2.COLOR_RGB2BGR)
    r = test.result
    if r.get("ok"):
        if "live_uv" in r:
            for u, v in r["live_uv"]:
                cv2.circle(bgr, (int(u), int(v)), 3, (60, 200, 60), 1)
        if "live_mask" in r:
            _outline(bgr, r["live_mask"], (60, 200, 60))
        px = _project(_t_base_cam(), teach.intr, test.transported[:3, 3])
        if px is not None:
            cv2.drawMarker(bgr, px, (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(
                bgr, "go here", (px[0] + 10, px[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
    else:
        cv2.putText(bgr, r.get("reason", "no match"), (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return Response(content=_jpeg(bgr), media_type="image/jpeg")


@router.post("/go")
async def go(body: GoBody) -> dict:
    """Walk the fingertip to the transported pre-grasp, ``hover_mm`` above it (0 to go exactly there)."""
    from . import jog

    with _state.lock:
        test = _state.test
    if test is None or test.transported is None:
        raise HTTPException(409, "no transported pose; capture the test frame first")
    pose = test.transported.copy()
    pose[2, 3] += body.hover_mm / 1000.0
    try:
        jog.set_target_pose(pose)
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {"target_mm": (pose[:3, 3] * 1000.0).tolist()}
