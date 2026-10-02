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
import concurrent.futures
import contextlib
import io
import json
import logging
import pathlib
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from . import _pregrasp_core as core

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/pregrasp", tags=["pregrasp"])
_REPO = pathlib.Path(__file__).resolve().parents[4]
_WORKER = _REPO / "benchmarks" / "pregrasp_worker.py"


@dataclass
class _Teach:
    at: str
    box: tuple[int, int, int, int]
    rgb: np.ndarray
    depth_m: np.ndarray
    intr: dict[str, float]
    keypoints: dict[str, Any]
    tip_pose: np.ndarray | None = None  # the pre-grasp: base frame, 4x4
    gripper: float | None = None  # opening at the pre-grasp, the follower's 0..100 units
    grasp_pose: np.ndarray | None = None  # the grasp keyframe, base frame, 4x4
    grasp_gripper: float | None = None  # the closed opening on the object
    demo: dict[str, Any] | None = None  # summary of the recorded demo the keyframes came from


@dataclass
class _Test:
    at: str
    rgb: np.ndarray
    result: dict[str, Any]
    transported: np.ndarray | None = None  # base frame, 4x4


@dataclass
class _Job:
    """One frame handed to the worker, and what came back."""

    id: str
    kind: str  # "teach" | "find"
    concept: str
    rgb: np.ndarray
    depth_m: np.ndarray
    intr: dict[str, float]
    created: float
    taken: bool = False
    result: dict[str, Any] | None = None
    camera_check: dict[str, Any] | None = None  # marker drift on this frame; None without a calibration
    algo: str | None = None  # a track job's algorithm
    compress: bool = True  # the frame's NPZ: compressed for one-off jobs, raw for the tracking stream


@dataclass
class _Worker:
    """The SAM3 + DINO process: spawned here, fed by long-polled jobs, results posted back."""

    proc: subprocess.Popen | None = None
    log: list[str] = field(default_factory=list)
    jobs: dict[str, _Job] = field(default_factory=dict)
    pending: list[str] = field(default_factory=list)  # job ids not yet taken, in order
    wake: asyncio.Event | None = None  # created on the loop, set when a job is queued
    started_at: float = 0.0

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


@dataclass
class _Track:
    """The live tracker: one frame at a time goes to the worker, the newest answer is the object's pose."""

    on: bool = False
    algo: str = "dino"
    follow: bool = False  # the jog's target follows the transported pre-grasp
    hover_mm: float = 20.0
    job: str | None = None  # the track job in flight
    last: dict[str, Any] = field(default_factory=dict)  # the newest result's readout, no arrays
    overlay: bytes | None = None
    fps: float = 0.0
    t_prev: float = 0.0
    done: asyncio.Event | None = None  # set when the in-flight job's result has been applied
    task: asyncio.Task | None = None


@dataclass
class _Run:
    """One grasp attempt: hover, pre-grasp, grasp, close, lift, each step waiting for the arm."""

    on: bool = False
    step: str = ""
    ok: bool | None = None
    reason: str = ""
    stop_requested: bool = False
    grip_at_close: float | None = None
    task: asyncio.Task | None = None


@dataclass
class _State:
    lock: threading.Lock = field(default_factory=threading.Lock)
    teach: _Teach | None = None
    test: _Test | None = None
    worker: _Worker = field(default_factory=_Worker)
    teach_job: str | None = None  # a features teach awaiting its result
    find_job: str | None = None
    flat: bool = True  # objects stay on the table: snap a fitted turn to the table normal
    track: _Track = field(default_factory=_Track)
    run: _Run = field(default_factory=_Run)


_state = _State()
_RENDER_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pregrasp-render")
TRACK_JOB_TIMEOUT_S = 5.0
_HEAVY = ("delta_cam", "teach_uv", "live_uv", "live_mask")
JOB_TIMEOUT_S = 120.0  # the worker's first job loads the models; longer than that and no answer is coming


def _expire_jobs_locked(s: _State) -> None:
    """A pending teach or find the worker has not answered in time stops pending, with a log line saying so.

    Seen once: a worker idle for minutes never received a queued job, and the
    tab said "teaching…" until the page was reloaded. Pre: ``s.lock`` held.
    """
    now = time.time()
    for attr in ("teach_job", "find_job"):
        jid = getattr(s, attr)
        if jid is None:
            continue
        job = s.worker.jobs.get(jid)
        if job is not None and (job.result is not None or now - job.created <= JOB_TIMEOUT_S):
            continue
        setattr(s, attr, None)
        what = (
            "a job"
            if job is None
            else f"{job.kind} {jid} ({'taken' if job.taken else 'never taken'} by the worker)"
        )
        s.worker.log.append(
            f"{what} timed out after {JOB_TIMEOUT_S:.0f} s; try again, restart the worker if it repeats"
        )


class TeachBody(BaseModel):
    box: list[int] = []  # x0, y0, x1, y1 in frame pixels (box mode)
    mode: str = "box"  # "box" | "features" (SAM3 by concept + DINO, in the worker)
    concept: str = ""


class GoBody(BaseModel):
    hover_mm: float = 20.0


class OptionsBody(BaseModel):
    flat: bool = True


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


def _table_normal_cam(t_base_cam: np.ndarray) -> np.ndarray:
    """The table's up direction (base +z) seen from the camera."""
    return np.asarray(t_base_cam, dtype=float)[:3, :3].T @ np.array([0.0, 0.0, 1.0])


def _saved_camera() -> dict[str, Any]:
    """The connected arm's saved camera calibration: transform, touches, marker pixels."""
    from lerobot.gui.config_paths import gui_config_dir

    from . import _calib_core, jog

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

    from . import _calib_core

    try:
        ref = _calib_core.marker_reference(_saved_camera())
    except HTTPException:
        return None
    if not ref:
        return None
    return _calib_core.marker_drift(ref, _calib_core.detect_markers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)))


async def _camera_check_async(rgb: np.ndarray) -> dict[str, Any] | None:
    from . import showservo

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


@router.get("/state")
async def state() -> dict:
    from . import jog, showservo

    s = _state
    with s.lock:
        teach, test = s.teach, s.test
    w = s.worker
    with s.lock:
        _expire_jobs_locked(s)
        teach_pending = s.teach_job is not None
        find_pending = s.find_job is not None
        log_tail = w.log[-12:]
    out: dict[str, Any] = {
        "camera_live": showservo.live_camera() is not None,
        "arm_connected": jog.current_robot_id() is not None,
        "worker": {
            "running": w.running,
            "ready": any("worker ready" in line for line in w.log),
            "log": "\n".join(log_tail),
        },
        "teach_pending": teach_pending,
        "find_pending": find_pending,
        "flat": s.flat,
        "track": {
            "on": s.track.on,
            "algo": s.track.algo,
            "follow": s.track.follow,
            "hover_mm": s.track.hover_mm,
            "fps": s.track.fps,
            "last": dict(s.track.last),
        },
        "run": {
            "on": s.run.on,
            "step": s.run.step,
            "ok": s.run.ok,
            "reason": s.run.reason,
            "grip_at_close": s.run.grip_at_close,
        },
        "teach": None,
        "test": None,
    }
    if teach is not None:
        out["teach"] = {
            "at": teach.at,
            "box": list(teach.box),
            **_teach_info(teach.keypoints),
            "tip_mm": None if teach.tip_pose is None else (teach.tip_pose[:3, 3] * 1000.0).tolist(),
            "gripper": teach.gripper,
            "grasp_mm": None if teach.grasp_pose is None else (teach.grasp_pose[:3, 3] * 1000.0).tolist(),
            "grasp_gripper": teach.grasp_gripper,
            "demo": teach.demo,
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
    """Teach the object: by concept through the worker (SAM3 + DINO), or by a drawn box. Before jogging in."""
    if body.mode == "features":
        if not body.concept.strip():
            raise HTTPException(422, "a concept is required, e.g. 'yellow block'")
        if not _state.worker.running:
            raise HTTPException(409, "start the worker first")
        rgb, depth_m, intr = await _frame()
        job = _queue_job("teach", body.concept.strip(), rgb, depth_m, intr)
        with _state.lock:
            _state.teach_job = job.id
            _state.test = None
        return {"pending": True, "job": job.id, "mode": "features"}
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
            kp = core.shape_teach(depth_m, intr, box, rgb)
        except ValueError as e:
            raise HTTPException(422, f"{texture_reason}; and {e}") from e
    else:
        kp["mode"] = "texture"
        # Keep the shape model too: weak texture that re-matches badly falls back to it at Find.
        try:
            kp["shape"] = core.shape_teach(depth_m, intr, box, rgb)
        except ValueError:
            kp["shape"] = None
    with _state.lock:
        _state.teach = _Teach(
            at=time.strftime("%H:%M:%S"), box=box, rgb=rgb, depth_m=depth_m, intr=intr, keypoints=kp
        )
        _state.test = None
    return _teach_info(kp)


def _teach_info(kp: dict[str, Any]) -> dict[str, Any]:
    if kp["mode"] == "features":
        return {
            "mode": "features",
            "concept": kp["concept"],
            "n_points": int(kp["n_points"]),
            "radius_mm": float(kp["radius_mm"]),
            "shape_class": kp["shape_class"],
            "yaw_observable": bool(kp["yaw_observable"]),
            "face_planarity": None if not kp.get("face") else kp["face"]["planarity"],
            "face_usable": core.face_usable(kp.get("face")),
        }
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
        "colour_cue": kp.get("colour") is not None,
    }


@router.post("/teach/mark")
async def teach_mark(body: MarkBody | None = None) -> dict:
    """Record the fingertip's present pose and opening as a keyframe: the pre-grasp, or the grasp."""
    from . import jog

    which = (body.which if body else "pregrasp").lower()
    if which not in ("pregrasp", "grasp"):
        raise HTTPException(422, "which must be 'pregrasp' or 'grasp'")
    cur = jog.current_tip_and_anchor()
    if cur is None:
        raise HTTPException(409, "connect an arm in the Jog panel first")
    grip = jog.current_gripper()
    with _state.lock:
        if _state.teach is None:
            raise HTTPException(409, "capture the object first")
        if which == "grasp":
            _state.teach.grasp_pose, _state.teach.grasp_gripper = cur[0].copy(), grip
        else:
            _state.teach.tip_pose, _state.teach.gripper = cur[0].copy(), grip
        tip = cur[0][:3, 3]
    return {"which": which, "tip_mm": (tip * 1000.0).tolist(), "gripper": grip}


class MarkBody(BaseModel):
    which: str = "pregrasp"


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
    elif teach.keypoints["mode"] == "features":
        _outline(bgr, teach.keypoints["mask"], (255, 0, 255))
        for u, v in teach.keypoints["uv"][::3]:
            cv2.circle(bgr, (int(u), int(v)), 2, (0, 220, 255), -1)
    else:
        _outline(bgr, teach.keypoints["mask"], (60, 200, 60))
    if teach.tip_pose is not None:
        with contextlib.suppress(HTTPException):  # no arm connected: the image still shows the object
            _draw_tool(bgr, _t_base_cam(), teach.intr, teach.tip_pose, "pre-grasp")
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
    check = await _camera_check_async(rgb)
    if teach.keypoints["mode"] == "features":
        if not _state.worker.running:
            raise HTTPException(409, "start the worker first")
        job = _queue_job("find", teach.keypoints["concept"], rgb, depth_m, intr)
        job.camera_check = check
        with _state.lock:
            _state.find_job = job.id
        return {"pending": True, "job": job.id, "mode": "features", "camera_check": check}
    if teach.keypoints["mode"] == "texture":
        result = core.register(teach.keypoints, rgb, depth_m, intr)
        if result.get("ok") and _state.flat:
            snap = core.snap_to_table_yaw(
                result["delta_cam"],
                _table_normal_cam(t_bc),
                np.asarray(teach.keypoints["xyz"])[teach.keypoints["valid"]].mean(axis=0),
            )
            result["delta_cam"] = snap["delta"]
            result.update(
                {"yaw_deg": snap["yaw_deg"], "tilt_discarded_deg": snap["tilt_deg"], "snapped": True}
            )
        if not result.get("ok") and teach.keypoints.get("shape") is not None:
            texture_reason = result.get("reason", "texture failed")
            result = core.shape_register(teach.keypoints["shape"], depth_m, intr, rgb)
            result["fallback_from"] = f"texture ({texture_reason})"
    else:
        result = core.shape_register(teach.keypoints, depth_m, intr, rgb)
    result["camera_check"] = check
    transported = None
    if result.get("ok"):
        transported = core.transport_pose(t_bc, result["delta_cam"], teach.tip_pose)
        result.update(_arm_motion(t_bc, result["delta_cam"]))
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


JAW_HALF_M = 0.025
APPROACH_M = 0.030


def _draw_tool(
    bgr: np.ndarray, t_base_cam: np.ndarray, intr: dict[str, float], pose: np.ndarray, label: str
) -> None:
    """The fingertip pose on the image: a cross at the tip, the jaw line through it, the approach as an arrow.

    The tip frame is the wrist link's: the jaws close along its x axis and the
    fingers point along -y, so the white line is the jaw direction and the
    arrow points the way the fingers do. Seen from above, a vertical approach
    collapses the arrow to the cross; a leaning one shows it.
    """
    import cv2

    pose = np.asarray(pose, dtype=float)
    p, x, y = pose[:3, 3], pose[:3, 0], pose[:3, 1]
    tip = _project(t_base_cam, intr, p)
    if tip is None:
        return
    a, b = _project(t_base_cam, intr, p + JAW_HALF_M * x), _project(t_base_cam, intr, p - JAW_HALF_M * x)
    back = _project(t_base_cam, intr, p + APPROACH_M * y)
    if a is not None and b is not None:
        cv2.line(bgr, a, b, (255, 255, 255), 3)
    if back is not None:
        cv2.arrowedLine(bgr, back, tip, (0, 220, 255), 2, tipLength=0.3)
    cv2.drawMarker(bgr, tip, (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
    cv2.putText(bgr, label, (tip[0] + 10, tip[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


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
            _outline(bgr, r["live_mask"], (60, 200, 60) if r.get("mode") != "features" else (255, 0, 255))
        if test.transported is not None:
            _draw_tool(bgr, _t_base_cam(), teach.intr, test.transported, "go here")
    else:
        cv2.putText(bgr, r.get("reason", "no match"), (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return Response(content=_jpeg(bgr), media_type="image/jpeg")


@router.post("/go")
async def go(body: GoBody) -> dict:
    """Walk the fingertip to the transported pre-grasp, ``hover_mm`` above it (0 to go exactly there)."""
    from . import jog

    with _state.lock:
        test, teach = _state.test, _state.teach
    if test is None or test.transported is None:
        raise HTTPException(409, "no transported pose; capture the test frame first")
    check = test.result.get("camera_check") or {}
    if check.get("moved"):
        raise HTTPException(
            409,
            f"the camera or the tray moved since the calibration (markers shifted {check['max_px']:.1f} px); "
            "redo the camera calibration before going anywhere",
        )
    pose = test.transported.copy()
    pose[2, 3] += body.hover_mm / 1000.0
    try:
        jog.set_target_pose(pose)
        # The gripper opening is part of the taught pose.
        if teach is not None and teach.gripper is not None:
            jog.set_gripper(teach.gripper)
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {"target_mm": (pose[:3, 3] * 1000.0).tolist(), "gripper": None if teach is None else teach.gripper}


# ── the worker: SAM3 designation + DINO features, in its own process ────────


def _queue_job(
    kind: str,
    concept: str,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intr: dict[str, float],
    algo: str | None = None,
    compress: bool = True,
) -> _Job:
    job = _Job(
        id=uuid.uuid4().hex[:8],
        kind=kind,
        concept=concept,
        rgb=rgb,
        depth_m=depth_m,
        intr=intr,
        created=time.time(),
        algo=algo,
        compress=compress,
    )
    w = _state.worker
    with _state.lock:
        w.jobs[job.id] = job
        w.pending.append(job.id)
        # Forget results nobody will read; keep the last few for the overlays.
        for old in [i for i, j in w.jobs.items() if j.result is not None][:-4]:
            del w.jobs[old]
        wake = w.wake
    if wake is not None:
        wake.set()
    return job


class WorkerStartBody(BaseModel):
    device: str = "cuda"
    dino_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m"


@router.post("/worker/start")
async def worker_start(body: WorkerStartBody, request: Request) -> dict:
    """Spawn the SAM3 + DINO worker against this server. Models load on the first job."""
    w = _state.worker
    with _state.lock:
        if w.running:
            raise HTTPException(409, "the worker is already running")
        cmd = [
            sys.executable,
            str(_WORKER),
            "--server",
            str(request.base_url),
            "--device",
            body.device,
            "--dino-model",
            body.dino_model,
        ]
        import os

        env = {**os.environ, "PYTHONPATH": str(_REPO / "src")}
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=str(_REPO)
        )
        w.proc, w.log, w.started_at = proc, [], time.time()

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            with _state.lock:
                w.log.append(line.rstrip("\n"))
                del w.log[:-200]

    threading.Thread(target=pump, daemon=True, name="pregrasp-worker").start()
    return {"status": "started"}


@router.post("/worker/stop")
async def worker_stop() -> dict:
    w = _state.worker
    with _state.lock:
        proc = w.proc
        w.proc = None
        w.pending.clear()
        wake = w.wake
    if wake is not None:
        wake.set()  # a waiting poll returns and sees no worker
    if proc is not None and proc.poll() is None:
        proc.terminate()
    return {"status": "stopped"}


@router.get("/worker/job")
async def worker_job(wait: float = 20.0) -> Response:
    """The worker's long-poll: the next job, 204 when none within ``wait`` seconds, 410 when stopped."""
    w = _state.worker
    if w.wake is None:
        w.wake = asyncio.Event()
    deadline = time.monotonic() + min(max(wait, 0.0), 60.0)
    while True:
        with _state.lock:
            if not w.running:
                return Response(status_code=410)
            if w.pending:
                job_id = w.pending.pop(0)
                job = w.jobs[job_id]
                job.taken = True
                return Response(
                    content=json.dumps(
                        {"id": job.id, "kind": job.kind, "concept": job.concept, "algo": job.algo}
                    ),
                    media_type="application/json",
                )
            w.wake.clear()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return Response(status_code=204)
        try:
            await asyncio.wait_for(w.wake.wait(), timeout=remaining)
        except TimeoutError:
            return Response(status_code=204)


@router.get("/worker/frame.npz")
async def worker_frame(id: str) -> Response:
    with _state.lock:
        job = _state.worker.jobs.get(id)
    if job is None:
        raise HTTPException(404, "no such job")
    buf = io.BytesIO()
    save = np.savez_compressed if job.compress else np.savez
    save(buf, rgb=job.rgb, depth=job.depth_m, intr=np.array(json.dumps(job.intr)))
    return Response(content=buf.getvalue(), media_type="application/octet-stream")


@router.post("/worker/result")
async def worker_result(id: str, request: Request) -> dict:
    """The worker's answer for a job: NPZ with ``meta`` JSON plus mask, points and, for a find, the delta."""
    body = await request.body()
    data = np.load(io.BytesIO(body), allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    with _state.lock:
        job = _state.worker.jobs.get(id)
    if job is None:
        raise HTTPException(404, "no such job")
    result: dict[str, Any] = dict(meta)
    for key in ("mask", "uv", "xyz", "live_uv", "delta"):
        if key in data.files:
            result[key] = np.asarray(data[key])
    job.result = result
    if job.kind == "teach":
        _apply_teach_result(job)
    elif job.kind == "track":
        await _apply_track_result(job)
    else:
        _apply_find_result(job)
    return {"status": "ok"}


def _apply_teach_result(job: _Job) -> None:
    r = job.result or {}
    with _state.lock:
        if _state.teach_job != job.id:
            return
        _state.teach_job = None
        if not r.get("ok"):
            _state.teach = None
            _state.worker.log.append(f"teach failed: {r.get('reason')}")
            return
        kp = {
            "mode": "features",
            "concept": job.concept,
            "mask": r["mask"].astype(bool),
            "uv": r["uv"],
            "xyz": r["xyz"],
            "n_points": int(r["n_points"]),
            "radius_mm": float(r["radius_mm"]),
            "shape_class": r["shape_class"],
            "yaw_observable": bool(r["yaw_observable"]),
            "face": r.get("face"),
        }
        _state.teach = _Teach(
            at=time.strftime("%H:%M:%S"),
            box=(0, 0, 0, 0),
            rgb=job.rgb,
            depth_m=job.depth_m,
            intr=job.intr,
            keypoints=kp,
        )
        _state.test = None


def _apply_find_result(job: _Job) -> None:
    r = job.result or {}
    with _state.lock:
        if _state.find_job != job.id:
            return
        _state.find_job = None
        teach = _state.teach
    if teach is None or teach.tip_pose is None:
        return
    result: dict[str, Any] = {
        k: v for k, v in r.items() if k not in ("mask", "uv", "xyz", "delta", "face_teach", "face_find")
    }
    result["mode"] = "features"
    result["camera_check"] = job.camera_check
    if r.get("mask") is not None:
        result["live_mask"] = r["mask"].astype(bool)
    transported = None
    if r.get("ok"):
        trusted, why = core.find_trusted(int(r.get("n_inliers", 0)), int(teach.keypoints["n_points"]))
        if not trusted:
            result["ok"], result["reason"] = False, why
    if result.get("ok"):
        try:
            t_bc = _t_base_cam()
        except HTTPException as e:
            result["ok"], result["reason"] = False, e.detail
        else:
            with _state.lock:
                flat = _state.flat
            transported = _compose_motion(result, r, teach, flat, t_bc)
    with _state.lock:
        _state.test = _Test(at=time.strftime("%H:%M:%S"), rgb=job.rgb, result=result, transported=transported)


def _compose_motion(
    result: dict[str, Any], r: dict[str, Any], teach: _Teach, flat: bool, t_bc: np.ndarray | None
) -> np.ndarray | None:
    """Turn the worker's raw fit into the motion the arm uses, in ``result``, and return the transported pre-grasp.

    The axis policy. Under the resting prior the object keeps its bottom on
    the surface it rests on, whose normal is fitted to the depth around the
    object in both frames and carries the axis (the calibration's vertical
    only when no surface could be fitted); the object's face is reported
    only, since a measured face tilt on a resting object is the face's own
    noise (2026-10-01: up to 18 degrees on a rounded object). Without the
    prior the object's face carries the axis when both faces are usable,
    tilt included, else the raw fit does. Without the camera calibration there is no
    transport; without a marked pre-grasp there is nothing to transport.
    Post: ``result['delta_cam']`` is the motion used, with ``axis_source``
    saying which policy chose it.
    """
    result["delta_cam"] = np.asarray(r["delta"], dtype=float)
    centroid = np.asarray(teach.keypoints["xyz"]).mean(axis=0)
    ft, ff = r.get("face_teach"), r.get("face_find")
    faces = core.face_usable(ft) and core.face_usable(ff)
    if faces:
        a, b = np.asarray(ft["normal"], dtype=float), np.asarray(ff["normal"], dtype=float)
        result["face_tilt_deg"] = float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))
        result["face_planarity"] = min(ft["planarity"], ff["planarity"])
    ta, tb = r.get("table_teach"), r.get("table_find")
    if r.get("algo") == "depth":
        # The depth path's motion is a turn about the table normal by construction.
        result.update({"axis_source": "table", "yaw_deg": r.get("yaw_deg"), "symmetric": r.get("symmetric")})
    elif flat and ta is not None and tb is not None:
        # The object keeps its bottom on the surface it rests on: the surface's normal, measured in
        # both frames, carries the axis; a surface that tilted between them (a ramp, a block) tilts
        # the object with it. On a level table this is a pure turn.
        comp = core.compose_with_face(result["delta_cam"], ta, tb, centroid)
        result["delta_cam"] = comp["delta"]
        result.update(
            {
                "axis_source": "surface",
                "yaw_deg": comp["yaw_deg"],
                "surface_tilt_deg": comp["face_tilt_deg"],
                "fit_axis_tilt_deg": comp["fit_axis_tilt_deg"],
                "face_tilt_applied": False,
            }
        )
    elif flat and t_bc is not None:
        # No surface could be fitted around the object: the calibration's vertical stands in.
        snap = core.snap_to_table_yaw(result["delta_cam"], _table_normal_cam(t_bc), centroid)
        result["delta_cam"] = snap["delta"]
        result.update(
            {
                "axis_source": "table",
                "yaw_deg": snap["yaw_deg"],
                "fit_axis_tilt_deg": snap["tilt_deg"],
                "face_tilt_applied": False,
            }
        )
    elif faces:
        # The axis from the face the camera sees (hundreds of points), the turn from the features.
        comp = core.compose_with_face(result["delta_cam"], ft["normal"], ff["normal"], centroid)
        result["delta_cam"] = comp["delta"]
        result.update(
            {
                "axis_source": "face",
                "yaw_deg": comp["yaw_deg"],
                "face_tilt_applied": True,
                "fit_axis_tilt_deg": comp["fit_axis_tilt_deg"],
            }
        )
    else:
        result["axis_source"] = "fit"
    if t_bc is None:
        return None
    result.update(_arm_motion(t_bc, result["delta_cam"]))
    if teach.tip_pose is None:
        return None
    return core.transport_pose(t_bc, result["delta_cam"], teach.tip_pose)


@router.post("/options")
async def options(body: OptionsBody) -> dict:
    """Run-time options: ``flat`` keeps every fitted turn about the table normal (objects do not tilt)."""
    with _state.lock:
        _state.flat = bool(body.flat)
        return {"flat": _state.flat}


# ── live tracking: the worker follows the card frame after frame; the arm may follow the pose ──


class TrackBody(BaseModel):
    algo: str = "dino"
    follow: bool = False
    hover_mm: float = 20.0


def _project_cam(intr: dict[str, float], pts: np.ndarray) -> np.ndarray:
    """Camera-frame points to pixels, (N, 2); points behind the camera are dropped."""
    p = np.asarray(pts, dtype=float).reshape(-1, 3)
    p = p[p[:, 2] > 1e-6]
    return np.stack(
        [intr["fx"] * p[:, 0] / p[:, 2] + intr["cx"], intr["fy"] * p[:, 1] / p[:, 2] + intr["cy"]], axis=1
    )


def _render_live(rgb, r, result, transported, teach, status) -> bytes:
    """The tracking view: the mask edge, the points that agree, the taught cloud carried by the
    motion (where the object is believed to be), the transported pre-grasp, and a status strip."""
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if r.get("mask") is not None:
        _outline(bgr, np.asarray(r["mask"]).astype(bool), (255, 0, 255))
    live = r.get("live_uv")
    if live is not None:
        for u, v in np.asarray(live)[::2]:
            cv2.circle(bgr, (int(u), int(v)), 2, (60, 200, 60), -1)
    if result is not None and result.get("ok"):
        d = result["delta_cam"]
        moved = np.asarray(teach.keypoints["xyz"])[::3] @ d[:3, :3].T + d[:3, 3]
        h, w = bgr.shape[:2]
        for u, v in _project_cam(teach.intr, moved):
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(bgr, (int(u), int(v)), 1, (0, 220, 255), -1)
        if transported is not None:
            with contextlib.suppress(HTTPException):
                _draw_tool(bgr, _t_base_cam(), teach.intr, transported, "pre-grasp")
    state = status.get("state") or ""
    strip = (
        f"[{status.get('algo')}] {state} · {status.get('fps') or 0:.0f} fps · {status.get('ms') or 0:.0f} ms"
    )
    if status.get("n_inliers") is not None:
        strip += f" | {status['n_inliers']} of {status.get('n_matches')} agree"
    if status.get("centre_shift_mm") is not None:
        strip += f" | moved {status['centre_shift_mm']:.0f} mm"
    if status.get("arm_turn_deg") is not None:
        strip += f" | gripper turns {status['arm_turn_deg']:.0f} deg, leans {status['arm_lean_deg']:.0f} deg"
    elif status.get("yaw_deg") is not None:
        strip += f" | turned {status['yaw_deg']:.0f} deg"
    if status.get("reason"):
        strip += f" | {status['reason']}"
    colour = {"tracking": (60, 230, 60), "occluded": (0, 200, 255)}.get(state, (0, 0, 255))
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(bgr, strip, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
    return _jpeg(bgr)


async def _apply_track_result(job: _Job) -> None:
    """One tracked frame: a certified, trusted fit becomes the live pose (and the jog's target when
    following); an occluded or lost frame leaves the last pose in place and only changes the status."""
    from . import jog

    r = job.result or {}
    tr = _state.track
    with _state.lock:
        stale = tr.job != job.id
        teach, flat = _state.teach, _state.flat
    if stale or teach is None:
        return
    status: dict[str, Any] = {
        k: r.get(k)
        for k in ("ok", "state", "algo", "ms", "n_matches", "n_inliers", "rms_m", "scale", "reason")
    }
    result: dict[str, Any] | None = None
    transported = None
    if r.get("ok"):
        trusted, why = core.find_trusted(int(r.get("n_inliers", 0)), int(teach.keypoints["n_points"]))
        if trusted:
            result = {
                k: v
                for k, v in r.items()
                if k not in ("mask", "uv", "xyz", "delta", "face_teach", "face_find")
            }
            result["mode"] = "features"
            if r.get("mask") is not None:
                result["live_mask"] = np.asarray(r["mask"]).astype(bool)
            try:
                t_bc = _t_base_cam()
            except HTTPException:
                t_bc = None
            transported = _compose_motion(result, r, teach, flat, t_bc)
            status.update(
                {
                    k: result.get(k)
                    for k in (
                        "axis_source",
                        "yaw_deg",
                        "face_tilt_deg",
                        "face_tilt_applied",
                        "arm_turn_deg",
                        "arm_lean_deg",
                    )
                }
            )
            status["motion"] = core.motion_summary(result["delta_cam"])
            d = result["delta_cam"]
            c = np.asarray(teach.keypoints["xyz"]).mean(axis=0)
            status["centre_shift_mm"] = float(np.linalg.norm(d[:3, :3] @ c + d[:3, 3] - c) * 1000.0)
            if transported is not None:
                status["transported_tip_mm"] = (transported[:3, 3] * 1000.0).tolist()
                if tr.follow:
                    pose = transported.copy()
                    pose[2, 3] += tr.hover_mm / 1000.0
                    try:
                        jog.set_target_pose(pose)
                    except RuntimeError as e:
                        status["follow_error"] = str(e)
            with _state.lock:
                _state.test = _Test(
                    at=time.strftime("%H:%M:%S"), rgb=job.rgb, result=result, transported=transported
                )
        else:
            status.update(ok=False, state="untrusted", reason=why)
    now = time.perf_counter()
    if tr.t_prev:
        tr.fps = 0.8 * tr.fps + 0.2 / max(now - tr.t_prev, 1e-3)
    tr.t_prev = now
    status["fps"] = tr.fps
    overlay = await asyncio.get_event_loop().run_in_executor(
        _RENDER_EXECUTOR, _render_live, job.rgb, r, result, transported, teach, status
    )
    with _state.lock:
        tr.last, tr.overlay, tr.job = status, overlay, None
    if tr.done is not None:
        tr.done.set()


async def _track_pump() -> None:
    """Feed the worker one frame at a time while tracking is on: the newest frame, never a backlog."""
    tr = _state.track
    try:
        while True:
            with _state.lock:
                on, teach, running = tr.on, _state.teach, _state.worker.running
            if not on:
                return
            if teach is None or teach.keypoints.get("mode") != "features":
                tr.last = {"state": "stopped", "reason": "teach by concept first"}
                return
            if not running:
                tr.last = {"state": "stopped", "reason": "the worker is not running"}
                return
            try:
                rgb, depth_m, intr = await _frame()
            except HTTPException as e:
                tr.last = {"state": "stopped", "reason": e.detail}
                return
            assert tr.done is not None
            tr.done.clear()
            job = _queue_job(
                "track", teach.keypoints["concept"], rgb, depth_m, intr, algo=tr.algo, compress=False
            )
            with _state.lock:
                tr.job = job.id
            try:
                await asyncio.wait_for(tr.done.wait(), timeout=TRACK_JOB_TIMEOUT_S)
            except TimeoutError:
                with _state.lock:
                    tr.job = None
                tr.last = {"state": "waiting", "reason": "the worker did not answer"}
    finally:
        with _state.lock:
            tr.on, tr.job, tr.task = False, None, None


@router.post("/track/start")
async def track_start(body: TrackBody) -> dict:
    """Start following the taught object live with ``algo``; ``follow`` makes the jog walk to the hover above it."""
    from . import showservo

    if body.algo not in core.TRACK_ALGOS:
        raise HTTPException(422, f"algo must be one of {core.TRACK_ALGOS}")
    with _state.lock:
        teach = _state.teach
        running = _state.worker.running
    if teach is None or teach.keypoints.get("mode") != "features":
        raise HTTPException(409, "teach by concept first")
    if not running:
        raise HTTPException(409, "start the worker first")
    if showservo.live_camera() is None:
        raise HTTPException(409, "start a live camera session first")
    tr = _state.track
    with _state.lock:
        tr.algo, tr.follow, tr.hover_mm = body.algo, body.follow, body.hover_mm
        already = tr.on
        if not already:
            tr.on, tr.last, tr.fps, tr.t_prev, tr.overlay = True, {"state": "starting"}, 0.0, 0.0, None
            tr.done = asyncio.Event()
    if not already:
        tr.task = asyncio.create_task(_track_pump())
    return {"status": "tracking", "algo": tr.algo, "follow": tr.follow}


@router.post("/track/stop")
async def track_stop() -> dict:
    tr = _state.track
    with _state.lock:
        tr.on = False
        done = tr.done
    if done is not None:
        done.set()
    return {"status": "stopped"}


@router.post("/track/options")
async def track_options(body: TrackBody) -> dict:
    """Switch the algorithm, the following and the hover while tracking runs; the next frame uses them."""
    if body.algo not in core.TRACK_ALGOS:
        raise HTTPException(422, f"algo must be one of {core.TRACK_ALGOS}")
    tr = _state.track
    with _state.lock:
        tr.algo, tr.follow, tr.hover_mm = body.algo, body.follow, body.hover_mm
    return {"algo": tr.algo, "follow": tr.follow, "hover_mm": tr.hover_mm}


@router.get("/track/live.jpg")
async def track_live() -> Response:
    with _state.lock:
        overlay = _state.track.overlay
    if overlay is None:
        raise HTTPException(404, "no tracking frame yet")
    return Response(content=overlay, media_type="image/jpeg")


# ── the demo and the grasp: keyframes from a leader-arm recording, then hover, pre-grasp, grasp, close, lift ─


class FromDemoBody(BaseModel):
    approach_mm: float = core.DEMO_APPROACH_M * 1000.0


@router.post("/teach/from_demo")
async def teach_from_demo(body: FromDemoBody) -> dict:
    """Both keyframes from the last recorded demo: the grasp where the gripper began to close, the
    pre-grasp where the final approach to it began. The taught object is assumed still during the demo."""
    from . import jog

    samples = jog.take_record()
    if not samples:
        raise HTTPException(409, "record a demo first (Jog panel: Leader drives, Record demo)")
    with _state.lock:
        if _state.teach is None:
            raise HTTPException(409, "capture the object first")

    def extract() -> dict[str, Any]:
        tips = np.stack([jog.fk_tip(s["obs"]) for s in samples])
        times = np.array([s["t"] for s in samples], dtype=float)
        grips = np.array([s["obs"]["gripper"] for s in samples], dtype=float)
        return core.demo_keyframes(times, tips, grips, body.approach_mm / 1000.0)

    try:
        kf = await asyncio.get_event_loop().run_in_executor(_RENDER_EXECUTOR, extract)
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    summary = {
        "n": kf["n"],
        "seconds": kf["seconds"],
        "approach_mm": kf["approach_m"] * 1000.0,
        "lift_mm": kf["lift_m"] * 1000.0,
        "pregrasp_index": kf["pregrasp"]["index"],
        "grasp_index": kf["grasp"]["index"],
    }
    with _state.lock:
        teach = _state.teach
        if teach is None:
            raise HTTPException(409, "capture the object first")
        teach.tip_pose, teach.gripper = kf["pregrasp"]["pose"].copy(), kf["pregrasp"]["gripper"]
        teach.grasp_pose, teach.grasp_gripper = kf["grasp"]["pose"].copy(), kf["grasp"]["gripper"]
        teach.demo = summary
    return {
        **summary,
        "pregrasp_mm": (kf["pregrasp"]["pose"][:3, 3] * 1000.0).tolist(),
        "grasp_mm": (kf["grasp"]["pose"][:3, 3] * 1000.0).tolist(),
        "gripper_open": kf["pregrasp"]["gripper"],
        "gripper_closed": kf["grasp"]["gripper"],
    }


RUN_ARRIVE_M = 0.004  # within this of the target counts as arrived: the servo's stiction band
RUN_STEP_TIMEOUT_S = 20.0
RUN_GRIP_STILL_UNITS = 0.5  # the gripper has stopped when consecutive readings differ by less than this
RUN_GRIP_SETTLE_S = 0.5  # for this long
RUN_TICK_S = 0.1


class RunBody(BaseModel):
    hover_mm: float = 20.0
    lift_mm: float = 50.0
    squeeze: float = 5.0  # gripper units past the taught closed opening, toward closed


async def _run_task(hover_mm: float, lift_mm: float, squeeze: float) -> None:
    from . import jog

    run = _state.run

    def fail(reason: str) -> None:
        run.ok, run.reason, run.step = False, reason, "aborted"

    try:
        with _state.lock:
            teach, test = _state.teach, _state.test
        if teach is None or teach.tip_pose is None or teach.grasp_pose is None:
            fail("mark the pre-grasp and the grasp first (or take them from a demo)")
            return
        if test is None or not test.result.get("ok"):
            fail("find or track the object first")
            return
        try:
            t_bc = _t_base_cam()
        except HTTPException as e:
            fail(e.detail)
            return

        def transported(pose_teach: np.ndarray) -> np.ndarray:
            # Always the newest motion, so a tracked object is followed to the end.
            with _state.lock:
                latest = _state.test
            return core.transport_pose(t_bc, latest.result["delta_cam"], pose_teach)

        def hover_pose() -> np.ndarray:
            p = transported(teach.tip_pose)
            p[2, 3] += hover_mm / 1000.0
            return p

        def pre_pose() -> np.ndarray:
            return transported(teach.tip_pose)

        def grasp_pose() -> np.ndarray:
            return transported(teach.grasp_pose)

        def lift_pose() -> np.ndarray:
            p = transported(teach.grasp_pose)
            p[2, 3] += lift_mm / 1000.0
            return p

        async def go_to(name: str, pose_fn) -> bool:
            run.step = name
            t0 = time.monotonic()
            while True:
                pose = pose_fn()
                try:
                    jog.set_target_pose(pose)
                except RuntimeError as e:
                    fail(str(e))
                    return False
                await asyncio.sleep(RUN_TICK_S)
                st = jog.current_status()
                cur = jog.current_tip_and_anchor()
                if not st.get("connected") or cur is None:
                    fail("the arm went away")
                    return False
                if st["halted"]:
                    fail(f"arm frozen: {st['reason']}")
                    return False
                if run.stop_requested:
                    fail("stopped")
                    return False
                if float(np.linalg.norm(cur[0][:3, 3] - pose[:3, 3])) < RUN_ARRIVE_M and not st["holding"]:
                    return True
                if time.monotonic() - t0 > RUN_STEP_TIMEOUT_S:
                    fail(f"{name}: not there after {RUN_STEP_TIMEOUT_S:.0f} s")
                    return False

        if teach.gripper is not None:
            jog.set_gripper(teach.gripper)  # open as taught before the approach
        for name, fn in (("hover", hover_pose), ("pre-grasp", pre_pose), ("grasp", grasp_pose)):
            if not await go_to(name, fn):
                return
        run.step = "close"
        if teach.grasp_gripper is None:
            fail("the grasp keyframe has no gripper opening")
            return
        toward_closed = (
            np.sign(teach.grasp_gripper - (teach.gripper if teach.gripper is not None else 0.0)) or 1.0
        )
        jog.set_gripper(float(np.clip(teach.grasp_gripper + toward_closed * squeeze, 0.0, 100.0)))
        t0 = time.monotonic()
        last, still_since = None, None
        while True:
            await asyncio.sleep(RUN_TICK_S)
            if run.stop_requested:
                fail("stopped")
                return
            g = jog.current_status().get("gripper_obs")
            if g is None:
                fail("the arm went away")
                return
            if last is not None and abs(g - last) < RUN_GRIP_STILL_UNITS:
                still_since = still_since or time.monotonic()
                if time.monotonic() - still_since >= RUN_GRIP_SETTLE_S:
                    break
            else:
                still_since = None
            last = g
            if time.monotonic() - t0 > RUN_STEP_TIMEOUT_S:
                break
        run.grip_at_close = g
        if not await go_to("lift", lift_pose):
            return
        run.step, run.ok = "done", True
    except Exception as e:  # the arm holds its last target; the operator sees why
        logger.exception("run failed")
        fail(f"run error: {e}")
    finally:
        with contextlib.suppress(Exception):
            _record_trial(hover_mm, lift_mm, squeeze)
        run.on = False


@router.post("/run")
async def run_grasp(body: RunBody) -> dict:
    """Hover, pre-grasp, grasp, close on the object, lift: the whole sequence on the transported keyframes."""
    from . import jog

    with _state.lock:
        teach, test, run = _state.teach, _state.test, _state.run
        if run.on:
            raise HTTPException(409, "a run is in progress")
        if teach is None or teach.tip_pose is None or teach.grasp_pose is None:
            raise HTTPException(409, "mark the pre-grasp and the grasp first (or take them from a demo)")
        if test is None or not test.result.get("ok"):
            raise HTTPException(409, "find or track the object first")
        if (test.result.get("camera_check") or {}).get("moved"):
            raise HTTPException(409, "the camera or the tray moved since the calibration; recalibrate first")
        _state.track.follow = False  # the run owns the target now
        run.on, run.ok, run.reason, run.step, run.stop_requested, run.grip_at_close = (
            True,
            None,
            "",
            "starting",
            False,
            None,
        )
    if jog.current_robot_id() is None:
        with _state.lock:
            run.on = False
        raise HTTPException(409, "connect an arm in the Jog panel first")
    run.task = asyncio.create_task(_run_task(body.hover_mm, body.lift_mm, body.squeeze))
    return {"status": "running"}


@router.post("/run/stop")
async def run_stop() -> dict:
    """Stop the sequence and hold the arm where it is."""
    from . import jog

    with _state.lock:
        _state.run.stop_requested = True
    cur = jog.current_tip_and_anchor()
    if cur is not None:
        with contextlib.suppress(RuntimeError):
            jog.set_target_pose(cur[0])
    return {"status": "stopping"}


# ── trials: one row per run, with the operator's verdict; the milestone's evidence ─────────────

TRIALS_PATH = _REPO / "captures" / "trials.jsonl"
TRIAL_VERDICTS = ("lifted", "missed", "collided", "other")
_trials: list[dict[str, Any]] | None = None


def _load_trials() -> list[dict[str, Any]]:
    global _trials
    if _trials is None:
        rows: list[dict[str, Any]] = []
        if TRIALS_PATH.exists():
            for line in TRIALS_PATH.read_text().splitlines():
                with contextlib.suppress(ValueError):
                    rows.append(json.loads(line))
        _trials = rows
    return _trials


def _save_trials(rows: list[dict[str, Any]]) -> None:
    TRIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRIALS_PATH.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _record_trial(hover_mm: float, lift_mm: float, squeeze: float) -> dict[str, Any]:
    """Append the run that just ended: what was found, what the arm was told, how it ended."""
    with _state.lock:
        teach, test, run, track = _state.teach, _state.test, _state.run, _state.track
    r = test.result if test is not None else {}
    row: dict[str, Any] = {
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "object": (teach.keypoints.get("concept") if teach else None) or "box",
        "source": f"track:{track.algo}" if track.on else "find",
        "n_inliers": r.get("n_inliers"),
        "n_matches": r.get("n_matches"),
        "rms_mm": None if r.get("rms_m") is None else r["rms_m"] * 1000.0,
        "axis_source": r.get("axis_source"),
        "yaw_deg": r.get("yaw_deg"),
        "surface_tilt_deg": r.get("surface_tilt_deg"),
        "face_tilt_deg": r.get("face_tilt_deg"),
        "arm_turn_deg": r.get("arm_turn_deg"),
        "arm_lean_deg": r.get("arm_lean_deg"),
        "hover_mm": hover_mm,
        "lift_mm": lift_mm,
        "squeeze": squeeze,
        "result": "lifted" if run.ok else run.step,
        "reason": run.reason,
        "grip_at_close": run.grip_at_close,
        "grip_taught": teach.grasp_gripper if teach else None,
        "verdict": None,
    }
    if teach is not None and r.get("ok") and r.get("delta_cam") is not None:
        d = np.asarray(r["delta_cam"])
        c = np.asarray(teach.keypoints["xyz"]).mean(axis=0)
        row["centre_shift_mm"] = float(np.linalg.norm(d[:3, :3] @ c + d[:3, 3] - c) * 1000.0)
    rows = _load_trials()
    rows.append(row)
    _save_trials(rows)
    return row


class VerdictBody(BaseModel):
    index: int
    verdict: str


@router.get("/trials")
async def trials() -> dict:
    return {"rows": _load_trials()}


@router.post("/trials/verdict")
async def trial_verdict(body: VerdictBody) -> dict:
    """The operator's word on a run: what the camera cannot see once the gripper covers the object."""
    if body.verdict not in TRIAL_VERDICTS:
        raise HTTPException(422, f"verdict must be one of {TRIAL_VERDICTS}")
    rows = _load_trials()
    if not 0 <= body.index < len(rows):
        raise HTTPException(404, "no such trial")
    rows[body.index]["verdict"] = body.verdict
    _save_trials(rows)
    return {"index": body.index, "verdict": body.verdict}
