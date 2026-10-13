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

"""Frames and calibration from the GUI-owned camera; shared camera lifecycle."""

from __future__ import annotations

import asyncio
import os
import re
import threading
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from lerobot.showservo import motion as core

from . import demos, groups, recording, runtime

router = APIRouter()


def _grab(camera: Any) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """The camera's newest frame not yet returned here, taken from no other reader, when the camera can give one
    (``read_newer_color_and_depth``); otherwise the next frame its ``read_color_and_aligned_depth`` gives."""
    newer = getattr(camera, "read_newer_color_and_depth", None)
    if newer is None:
        rgb, depth_mm = camera.read_color_and_aligned_depth()
    else:
        with runtime._grab_lock:
            last = (
                runtime._last_grab[1]
                if runtime._last_grab is not None and runtime._last_grab[0] is camera
                else None
            )
        rgb, depth_mm, at = newer(last)
        with runtime._grab_lock:
            if (
                runtime._last_grab is None
                or runtime._last_grab[0] is not camera
                or at > runtime._last_grab[1]
            ):
                runtime._last_grab = (camera, at)
    return rgb, depth_mm.astype(np.float32) / 1000.0, camera.color_intrinsics()


async def _frame():
    from .. import showservo

    camera = showservo.live_camera()
    if camera is None:
        raise HTTPException(409, "start a live camera session in the Servo tab first")
    try:
        return await asyncio.get_event_loop().run_in_executor(runtime._GRAB_EXECUTOR, _grab, camera)
    except Exception as e:
        raise HTTPException(500, f"camera read failed: {e}") from e


def _table_normal_cam(t_base_cam: np.ndarray) -> np.ndarray:
    """The table's up direction (base +z) seen from the camera."""
    return np.asarray(t_base_cam, dtype=float)[:3, :3].T @ np.array([0.0, 0.0, 1.0])


def _saved_camera() -> dict[str, Any]:
    """The connected arm's saved camera calibration: transform, touches, marker pixels."""
    from lerobot.gui.config_paths import gui_config_dir

    from .. import _calib_core, jog

    rid = jog.current_robot_id()
    if rid is None:
        raise HTTPException(409, "connect an arm in the Jog panel first")
    saved = _calib_core.load_calibration(_calib_core.calibration_path(gui_config_dir(), rid))
    cam = saved.get("camera")
    if not cam:
        raise HTTPException(409, "no camera-to-base calibration for this arm; run the touch calibration")
    return cam


def _t_base_cam() -> np.ndarray:
    return np.asarray(_saved_camera()["T_base_cam"], dtype=float)


def _camera_check(rgb: np.ndarray) -> dict[str, Any] | None:
    """Have the calibration markers moved in the image since the camera was calibrated?

    The calibration holds only while the camera and the tray stay put, and a
    Find through a stale one is silently wrong. None when there is no
    calibration or it recorded no marker pixels.
    """
    import cv2

    from .. import _calib_core

    try:
        ref = _calib_core.marker_reference(_saved_camera())
    except HTTPException:
        return None
    if not ref:
        return None
    return _calib_core.marker_drift(ref, _calib_core.detect_markers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)))


async def _camera_check_async(rgb: np.ndarray) -> dict[str, Any] | None:
    from .. import showservo

    return await asyncio.get_event_loop().run_in_executor(showservo._EXECUTOR, _camera_check, rgb)


def _arm_motion(t_base_cam: np.ndarray, delta_cam: np.ndarray) -> dict[str, float]:
    """What the object's motion does to the gripper, in the base frame: a turn about vertical and a lean."""
    t_bc = np.asarray(t_base_cam, dtype=float)
    delta_base = t_bc @ np.asarray(delta_cam, dtype=float) @ np.linalg.inv(t_bc)
    tl = core.turn_and_lean(delta_base, np.array([0.0, 0.0, 1.0]))
    return {"arm_turn_deg": tl["turn_deg"], "arm_lean_deg": tl["lean_deg"]}


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


def _page_version() -> str | None:
    """The version of the Approach tab's script the GUI page loads, as index.html names it (``showservo.js?v=N``);
    read again only when the file changes. A tab open since before a change compares itself against this."""
    try:
        mtime = runtime._PAGE.stat().st_mtime
    except OSError:
        return None
    if mtime != runtime._page_seen[0]:
        m = re.search(r"showservo\.js\?v=(\d+)", runtime._PAGE.read_text())
        runtime._page_seen = (mtime, m.group(1) if m else None)
    return runtime._page_seen[1]


@router.get("/frame.jpg")
async def frame_jpeg() -> Response:
    """A fresh colour frame: the camera view while the camera is live and nothing is tracked, and what an object box
    is drawn on. The page polls it, so the frame is read and encoded on the camera's executor in one hop: an encode
    on the event loop would stall the act's ticks for as long as it takes."""
    import cv2

    from .. import showservo

    camera = showservo.live_camera()
    if camera is None:
        raise HTTPException(409, "start a live camera session in the Servo tab first")

    def grab_jpeg() -> bytes:
        rgb, _depth, _intr = _grab(camera)
        return _jpeg(groups._paint_groups(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))

    try:
        data = await asyncio.get_event_loop().run_in_executor(runtime._GRAB_EXECUTOR, grab_jpeg)
    except Exception as e:
        raise HTTPException(500, f"camera read failed: {e}") from e
    return Response(content=data, media_type="image/jpeg")


@router.post("/camera/share/start")
async def camera_share_start() -> dict:
    """Put the camera's frames into shared memory for a reader in another process; the ring's name, size and
    intrinsics. Already sharing: the same ring."""
    from lerobot.showservo.frame_ring import FrameRing

    from .. import showservo

    camera = showservo.live_camera()
    if camera is None:
        raise HTTPException(409, "start the camera first")
    with runtime._state.lock:
        share = runtime._state.camera_share
        if share is not None and share.thread is not None and share.thread.is_alive():
            ring = share.ring
            return {"name": ring.name, "height": ring.h, "width": ring.w, "frames": share.n}
    intr = await asyncio.get_event_loop().run_in_executor(showservo._EXECUTOR, camera.color_intrinsics)
    k = [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]]
    name = f"lerobot_frames_{os.getpid()}_{time.monotonic_ns()}"
    ring = FrameRing(name, int(intr["height"]), int(intr["width"]), slots=4, k=k, create=True)
    share = runtime._FrameShare(ring=ring)
    share.thread = threading.Thread(
        target=recording._share_frames, args=(camera, share), name="pregrasp-share", daemon=True
    )
    with runtime._state.lock:
        old, runtime._state.camera_share = runtime._state.camera_share, share
    if old is not None:
        await _stop_share(old)
    share.thread.start()
    return {"name": ring.name, "height": ring.h, "width": ring.w, "frames": 0}


async def _stop_share(share: runtime._FrameShare) -> None:
    share.stop.set()
    if share.thread is not None:
        await asyncio.get_event_loop().run_in_executor(runtime._RENDER_EXECUTOR, share.thread.join, 5.0)
    share.ring.close()


@router.post("/camera/share/stop")
async def camera_share_stop() -> dict:
    with runtime._state.lock:
        share, runtime._state.camera_share = runtime._state.camera_share, None
    if share is None:
        raise HTTPException(409, "the camera's frames are not being shared")
    await _stop_share(share)
    return {"frames": share.n, "error": share.error}


@router.post("/camera/record/start")
async def camera_record_start() -> dict:
    """Record the camera's colour and depth on their own, as a demo does: a run to replay the tracker over."""
    from .. import showservo

    camera = showservo.live_camera()
    if camera is None:
        raise HTTPException(409, "start the camera first")
    with runtime._state.lock:
        if runtime._state.camera_recording is not None:
            raise HTTPException(409, "the camera is already being recorded")
        if runtime._state.stream is not None:
            raise HTTPException(409, "a demo is recording the camera")  # two readers would split its frames
        rec = runtime._StreamRecorder(
            out=demos._demos_root() / ".recordings" / f"camera_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        runtime._state.camera_recording = rec
    rec.thread = threading.Thread(
        target=recording._record_stream, args=(camera, rec), name="pregrasp-camera", daemon=True
    )
    rec.thread.start()
    return {"status": "recording", "out": str(rec.out)}


@router.post("/camera/record/stop")
async def camera_record_stop() -> dict:
    with runtime._state.lock:
        rec, runtime._state.camera_recording = runtime._state.camera_recording, None
    if rec is None:
        raise HTTPException(409, "the camera is not being recorded")
    rec.stop.set()
    if rec.thread is not None:
        await asyncio.get_event_loop().run_in_executor(runtime._RENDER_EXECUTOR, rec.thread.join, 5.0)
    return {"out": str(rec.out), "frames": rec.n, "error": rec.error}
