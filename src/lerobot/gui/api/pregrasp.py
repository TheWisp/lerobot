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
    tip_pose: np.ndarray | None = (
        None  # the demo's first fingertip pose, base frame, 4x4 (set when a demo is loaded)
    )
    gripper: float | None = None  # its gripper opening, the follower's 0..100 units


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
    history: list = field(default_factory=list)  # (wall time, certified, delta_cam) per frame, bounded


@dataclass
class _Demo:
    """A recorded demonstration: the fingertip's path in the base frame, the gripper, and the object's pose while it ran."""

    name: str
    concept: str
    fps: float
    t: np.ndarray  # (N,) seconds from the start
    tips: np.ndarray  # (N, 4, 4) fingertip poses, base frame, from the follower's observed joints
    grippers: np.ndarray  # (N,) openings, the follower's 0..100 units
    q_obs: np.ndarray  # (N, J) observed joints
    q_cmd: np.ndarray  # (N, J) commanded joints
    deltas: np.ndarray  # (N, 4, 4) the object's motion since teach, camera frame; identity where unseen
    seen: np.ndarray  # (N,) whether the tracker had the object at that sample
    delta0: np.ndarray  # (4, 4) the object's motion since teach when the demo began
    t0: float = 0.0  # wall-clock start of the recording, to pair samples with camera frames
    camera: str = "camera"  # the camera the frames came from, as the dataset names its image feature
    frames: list | None = None  # the top camera while recording, (t, small rgb), when tracking ran
    root: str | None = None  # the dataset folder once saved


@dataclass
class _Act:
    """One replay of the demo on the object where it is now."""

    on: bool = False
    step: str = ""
    ok: bool | None = None
    reason: str = ""
    stop_requested: bool = False
    progress: float = 0.0
    speed: float = 1.0
    task: asyncio.Task | None = None


@dataclass
class _State:
    lock: threading.Lock = field(default_factory=threading.Lock)
    teach: _Teach | None = None
    test: _Test | None = None
    worker: _Worker = field(default_factory=_Worker)
    teach_job: str | None = None  # a features teach awaiting its result
    find_job: str | None = None
    flat: bool = (
        False  # opt-in resting prior: the fit's motion as a turn about the surface the object rests on
    )
    track: _Track = field(default_factory=_Track)
    act: _Act = field(default_factory=_Act)
    demo: _Demo | None = None  # the demo recorded or loaded last
    recording: dict[str, Any] | None = None  # while a demo is being recorded: its start and the frames so far


_state = _State()
_RENDER_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pregrasp-render")
TRACK_JOB_TIMEOUT_S = 5.0
TRACK_HISTORY_MAX = 3000  # about a hundred seconds at the camera's rate: longer than any demo
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
    flat: bool = False


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
        "act": {
            "on": s.act.on,
            "step": s.act.step,
            "ok": s.act.ok,
            "reason": s.act.reason,
            "progress": s.act.progress,
            "speed": s.act.speed,
        },
        "demo": None if s.demo is None else _demo_info(s.demo),
        "recording": None
        if s.recording is None
        else {
            "since": s.recording["t0"],
            "frames": len(s.recording["frames"]),
            "samples": jog.record_count(),
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
        if _state.demo is not None and len(_state.demo.tips):
            # The demo's first pose is the pose that gets transported and drawn; a demo may be applied
            # to a newly taught object on purpose.
            _state.teach.tip_pose = _state.demo.tips[0].copy()
            _state.teach.gripper = float(_state.demo.grippers[0])
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

    The pose is the tracker's own rigid fit. Everything this function used to
    compose on top of it — the resting prior, the axis from the object's face,
    the footprint and long-axis turn rules — was replayed against ground truth
    on the nine YCBInEOAT videos (2026-10-03) and scored below or equal to the
    raw fit on every one: mean ADD-S AUC 44.1 to 67.8 against 80.5, and on the
    videos' at-rest openings the prior turned two still objects by 13 and 115
    degrees. The resting prior survives only as an explicit opt-in (``flat``):
    the fit's motion re-expressed as a turn about the surface the object rests
    on, measured in both frames, with the fit's own turn. Nothing has shown it
    to help; it is kept for an operator who knows the object stays on the
    table and wants to see the difference. Post: ``result['delta_cam']`` is the
    motion used, with ``axis_source`` saying which: "fit", "surface" (opted
    in), or "table" (the depth-only algorithm, whose motion is a turn about the
    table normal by construction).
    """
    result["delta_cam"] = np.asarray(r["delta"], dtype=float)
    centroid = np.asarray(teach.keypoints["xyz"]).mean(axis=0)
    ft, ff = r.get("face_teach"), r.get("face_find")
    if core.face_usable(ft) and core.face_usable(ff):  # reported, never applied
        a, b = np.asarray(ft["normal"], dtype=float), np.asarray(ff["normal"], dtype=float)
        result["face_tilt_deg"] = float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))
        result["face_planarity"] = min(ft["planarity"], ff["planarity"])
    ta, tb = r.get("table_teach"), r.get("table_find")
    if r.get("algo") == "depth":
        result.update({"axis_source": "table", "yaw_deg": r.get("yaw_deg"), "symmetric": r.get("symmetric")})
    elif flat and ta is not None and tb is not None:
        comp = core.compose_with_face(result["delta_cam"], ta, tb, centroid)
        result["delta_cam"] = comp["delta"]
        result.update(
            {
                "axis_source": "surface",
                "yaw_deg": comp["yaw_deg"],
                "surface_tilt_deg": comp["face_tilt_deg"],
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
    """Run-time options: ``flat`` opts into the resting prior (see :func:`_compose_motion`); off by default."""
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


FRAME_AXIS_M = 0.03  # the drawn object frame's axis length


def _draw_frame(bgr: np.ndarray, intr: dict[str, float], delta: np.ndarray, centre_teach: np.ndarray) -> None:
    """The object's frame carried by the motion: origin at the taught centre, axes as taught (the
    camera's at teach time), x red, y green, z blue. A steady frame is a steady pose."""
    import cv2

    r, t = delta[:3, :3], delta[:3, 3]
    origin = r @ centre_teach + t
    pts = np.vstack([origin, origin + FRAME_AXIS_M * r.T])  # the three carried axes, one per row of r.T
    uv = _project_cam(intr, pts)
    if len(uv) < 4:
        return
    o = (int(uv[0, 0]), int(uv[0, 1]))
    for k, colour in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        cv2.line(bgr, o, (int(uv[k + 1, 0]), int(uv[k + 1, 1])), colour, 2, cv2.LINE_AA)


def _render_live(rgb, r, result, transported, teach, status) -> bytes:
    """The tracking view: the mask edge, the points that agree, the taught cloud carried by the
    motion (where the object is believed to be), the transported pre-grasp, and a status strip."""
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if r.get("mask") is not None:
        _outline(bgr, np.asarray(r["mask"]).astype(bool), (255, 0, 255))
    live = r.get("live_uv")
    if live is not None:
        # White on black reads on any object; a green dot vanished on a green one.
        for u, v in np.asarray(live)[::2]:
            cv2.circle(bgr, (int(u), int(v)), 3, (0, 0, 0), -1)
            cv2.circle(bgr, (int(u), int(v)), 2, (255, 255, 255), -1)
    if result is not None and result.get("ok"):
        d = result["delta_cam"]
        # The object as known so far (the teach view plus what the tracker has adopted since), carried
        # by the motion: where the tracker believes the whole object is, seen sides and hidden ones.
        cloud = r.get("model_xyz")
        cloud = np.asarray(teach.keypoints["xyz"] if cloud is None else cloud, dtype=float)
        moved = cloud[::3] @ d[:3, :3].T + d[:3, 3]
        h, w = bgr.shape[:2]
        for u, v in _project_cam(teach.intr, moved):
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(bgr, (int(u), int(v)), 1, (0, 220, 255), -1)
        _draw_frame(bgr, teach.intr, d, np.asarray(teach.keypoints["xyz"], dtype=float).mean(axis=0))
        with contextlib.suppress(HTTPException):
            t_bc = _t_base_cam()
            if transported is not None:
                _draw_tool(bgr, t_bc, teach.intr, transported, "start")
            with _state.lock:
                demo = _state.demo
            if demo is not None:
                _draw_path(bgr, t_bc, teach.intr, _act_tips(demo, result["delta_cam"], t_bc))
    state = status.get("state") or ""
    strip = (
        f"[{status.get('algo')}] {state} | {status.get('fps') or 0:.0f} fps | {status.get('ms') or 0:.0f} ms"
    )
    if status.get("n_inliers") is not None:
        strip += f" | {status['n_inliers']} of {status.get('n_matches')} agree"
    if status.get("centre_shift_mm") is not None:
        strip += f" | moved {status['centre_shift_mm']:.0f} mm"
    if status.get("arm_turn_deg") is not None:
        strip += f" | gripper turns {status['arm_turn_deg']:.0f} deg, leans {status['arm_lean_deg']:.0f} deg"
    elif status.get("yaw_deg") is not None:
        strip += f" | turned {status['yaw_deg']:.0f} deg"
    elif (status.get("motion") or {}).get("rotation_deg") is not None:
        strip += f" | rotated {status['motion']['rotation_deg']:.0f} deg"
    if status.get("card_points"):
        strip += f" | card {status['card_points']}"
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
        for k in (
            "ok",
            "state",
            "algo",
            "ms",
            "n_matches",
            "n_inliers",
            "rms_m",
            "scale",
            "reason",
            "card_points",
            "card_grew",
        )
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
        tr.history.append((time.time(), result is not None, None if result is None else result["delta_cam"]))
        del tr.history[:-TRACK_HISTORY_MAX]
        rec = _state.recording
        if rec is not None:
            rec["frames"].append(
                (time.time(), job.rgb[::2, ::2].copy())
            )  # half size: a demo's video is a record, not evidence
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


def _record_trial() -> dict[str, Any]:
    """Append the act that just ended: what was found, which demo was replayed, how it ended."""
    with _state.lock:
        teach, test, act, track, demo = _state.teach, _state.test, _state.act, _state.track, _state.demo
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
        "demo": None if demo is None else demo.name,
        "speed": act.speed,
        "result": "done" if act.ok else act.step,
        "reason": act.reason,
        "progress": act.progress,
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


# ── the demo: a recorded path saved as a LeRobot dataset; the act: that path on the object where it is now ─

DEMOS_NAMESPACE = "demos"
DEMO_FILE = "showservo_demo.npz"
HISTORY_MATCH_S = 0.2  # a tracker frame this close in time to a joint sample is that sample's object pose
ACT_ARRIVE_M = 0.004  # within this of a target counts as arrived: the servo's stiction band
ACT_STEP_TIMEOUT_S = 20.0
ACT_TICK_S = 0.05


def _demos_root() -> pathlib.Path:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return HF_LEROBOT_HOME / DEMOS_NAMESPACE


def _demo_info(demo: _Demo) -> dict[str, Any]:
    return {
        "name": demo.name,
        "concept": demo.concept,
        "n": int(len(demo.t)),
        "seconds": float(demo.t[-1] - demo.t[0]) if len(demo.t) > 1 else 0.0,
        "fps": demo.fps,
        "seen_fraction": float(demo.seen.mean()) if len(demo.seen) else 0.0,
        "frames": 0 if not demo.frames else len(demo.frames),
        "root": demo.root,
        "repo_id": f"{DEMOS_NAMESPACE}/{demo.name}",
    }


def _demo_from_samples(name: str, concept: str, samples: list, history: list, fk, t0: float) -> _Demo:
    """A demo from the jog's samples and the tracker's history.

    Each sample's object pose is the tracker's nearest certified frame within
    :data:`HISTORY_MATCH_S` of it. The demo's start pose is the first seen
    one, or the identity when the object was never seen, which takes it to sit
    where it was taught. ``fk`` maps observed joints to the fingertip pose.
    """
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    t = np.array([s["t"] for s in samples], dtype=float)
    q_obs = np.array([[s["obs"][m] for m in MOTOR_NAMES] for s in samples], dtype=float)
    q_cmd = np.array([[s["cmd"].get(m, s["obs"][m]) for m in MOTOR_NAMES] for s in samples], dtype=float)
    tips = np.stack([fk(s["obs"]) for s in samples])
    grips = q_obs[:, MOTOR_NAMES.index("gripper")]
    seen_hist = [(w, np.asarray(d, dtype=float)) for (w, ok, d) in history if ok and d is not None]
    deltas = np.tile(np.eye(4), (len(samples), 1, 1))
    seen = np.zeros(len(samples), dtype=bool)
    if seen_hist:
        hw = np.array([w for w, _ in seen_hist])
        for i, ti in enumerate(t):
            k = int(np.argmin(np.abs(hw - (t0 + ti))))
            if abs(hw[k] - (t0 + ti)) <= HISTORY_MATCH_S:
                deltas[i], seen[i] = seen_hist[k][1], True
    first = np.flatnonzero(seen)
    delta0 = deltas[first[0]].copy() if len(first) else np.eye(4)
    fps = float((len(t) - 1) / (t[-1] - t[0])) if len(t) > 1 and t[-1] > t[0] else 30.0
    return _Demo(
        name=name,
        concept=concept,
        fps=fps,
        t=t,
        tips=tips,
        grippers=grips,
        q_obs=q_obs,
        q_cmd=q_cmd,
        deltas=deltas,
        seen=seen,
        delta0=delta0,
        t0=t0,
    )


class DemoNameBody(BaseModel):
    name: str | None = None


@router.post("/demo/record/start")
async def demo_record_start() -> dict:
    """Record the arm (any mode: the leader or the gizmo) and, while tracking runs, the object and the camera."""
    from . import jog

    with _state.lock:
        teach = _state.teach
    if teach is None or teach.keypoints.get("mode") != "features":
        raise HTTPException(409, "teach the object first")
    try:
        t0 = jog.start_record()
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    with _state.lock:
        _state.recording = {"t0": t0, "frames": []}
        tracking = _state.track.on
    return {"status": "recording", "tracking": tracking}


@router.post("/demo/record/stop")
async def demo_record_stop(body: DemoNameBody) -> dict:
    """End the recording and keep it as the current demo (not yet saved)."""
    from . import jog

    with _state.lock:
        rec, teach, history = _state.recording, _state.teach, list(_state.track.history)
        _state.recording = None
    if rec is None:
        raise HTTPException(409, "not recording")
    try:
        samples = jog.stop_record()
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    if len(samples) < 2 or teach is None:
        raise HTTPException(409, "the recording is empty")
    name = _safe_name(body.name) or time.strftime("demo_%Y%m%d_%H%M%S")
    concept = teach.keypoints["concept"]

    def build() -> _Demo:
        return _demo_from_samples(name, concept, samples, history, jog.fk_tip, rec["t0"])

    demo = await asyncio.get_event_loop().run_in_executor(_RENDER_EXECUTOR, build)
    demo.frames = rec["frames"] or None
    demo.camera = _camera_label()
    with _state.lock:
        _state.demo = demo
        teach.tip_pose, teach.gripper = demo.tips[0].copy(), float(demo.grippers[0])
    return _demo_info(demo)


def _camera_label() -> str:
    """The live camera's name, as the dataset's image feature is named after it."""
    from . import showservo

    cam = showservo.live_camera()
    raw = str(getattr(getattr(cam, "config", None), "serial_number_or_name", "") or "camera")
    return _safe_name(raw) or "camera"


def _safe_name(name: str | None) -> str | None:
    if not name:
        return None
    cleaned = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in name.strip())
    return cleaned or None


def _write_demo(demo: _Demo, teach: _Teach) -> pathlib.Path:
    """The demo as a LeRobot dataset the Data tab can play, plus a sidecar with what the act needs."""
    import shutil

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES
    from lerobot.utils.constants import OBS_IMAGES

    root = _demos_root() / demo.name
    if root.exists():
        # safe-destruct: our own demos folder; saving under a taken name replaces that demo
        shutil.rmtree(root)
    nj = len(MOTOR_NAMES)
    matrix_names = [f"m{i}" for i in range(16)]
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": (nj,), "names": list(MOTOR_NAMES)},
        "action": {"dtype": "float32", "shape": (nj,), "names": list(MOTOR_NAMES)},
        "tip.pose": {"dtype": "float32", "shape": (16,), "names": matrix_names},
        "object.delta": {"dtype": "float32", "shape": (16,), "names": matrix_names},
        "object.seen": {"dtype": "float32", "shape": (1,), "names": ["seen"]},
    }
    frames = demo.frames or []
    image_key = f"{OBS_IMAGES}.{demo.camera}"
    if frames:
        h, w = frames[0][1].shape[:2]
        features[image_key] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
        frame_times = np.array([f[0] for f in frames])
    ds = LeRobotDataset.create(
        f"{DEMOS_NAMESPACE}/{demo.name}",
        fps=max(1, int(round(demo.fps))),
        features=features,
        root=root,
        use_videos=True,
    )
    for i in range(len(demo.t)):
        frame: dict[str, Any] = {
            "observation.state": demo.q_obs[i].astype(np.float32),
            "action": demo.q_cmd[i].astype(np.float32),
            "tip.pose": demo.tips[i].reshape(16).astype(np.float32),
            "object.delta": demo.deltas[i].reshape(16).astype(np.float32),
            "object.seen": np.array([float(demo.seen[i])], dtype=np.float32),
            "task": demo.concept,
        }
        if frames:
            k = int(np.argmin(np.abs(frame_times - (demo.t0 + demo.t[i]))))
            frame[image_key] = np.ascontiguousarray(frames[k][1], dtype=np.uint8)
        ds.add_frame(frame)
    ds.save_episode()
    ds.finalize()
    np.savez_compressed(
        root / DEMO_FILE,
        name=demo.name,
        concept=demo.concept,
        fps=demo.fps,
        t=demo.t,
        tips=demo.tips,
        grippers=demo.grippers,
        q_obs=demo.q_obs,
        q_cmd=demo.q_cmd,
        deltas=demo.deltas,
        seen=demo.seen,
        delta0=demo.delta0,
        t0=demo.t0,
        camera=demo.camera,
        teach_rgb=teach.rgb,
        teach_depth=teach.depth_m,
        teach_mask=np.asarray(teach.keypoints.get("mask", np.zeros(teach.depth_m.shape, dtype=bool))),
        intr=json.dumps(teach.intr),
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    return root


@router.post("/demo/save")
async def demo_save(body: DemoNameBody) -> dict:
    """Write the current demo as a dataset under the demos namespace; a name given here renames it."""
    with _state.lock:
        demo, teach = _state.demo, _state.teach
    if demo is None or teach is None:
        raise HTTPException(409, "record a demo first")
    name = _safe_name(body.name)
    if name:
        demo.name = name
    root = await asyncio.get_event_loop().run_in_executor(_RENDER_EXECUTOR, _write_demo, demo, teach)
    demo.root = str(root)
    return _demo_info(demo)


@router.get("/demos")
async def demos() -> dict:
    out = []
    for f in sorted(_demos_root().glob(f"*/{DEMO_FILE}")):
        with contextlib.suppress(Exception):
            z = np.load(f, allow_pickle=False)
            t = z["t"]
            out.append(
                {
                    "name": str(z["name"]),
                    "concept": str(z["concept"]),
                    "n": int(len(t)),
                    "seconds": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
                    "created": str(z["created"]),
                    "root": str(f.parent),
                    "repo_id": f"{DEMOS_NAMESPACE}/{z['name']}",
                }
            )
    return {"demos": out}


class DemoLoadBody(BaseModel):
    name: str


@router.post("/demo/load")
async def demo_load(body: DemoLoadBody) -> dict:
    """A saved demo becomes the current one; its object is re-taught to the worker from the saved frame."""
    f = _demos_root() / body.name / DEMO_FILE
    if not f.exists():
        raise HTTPException(404, f"no demo named {body.name!r}")
    z = np.load(f, allow_pickle=False)
    demo = _Demo(
        name=str(z["name"]),
        concept=str(z["concept"]),
        fps=float(z["fps"]),
        t=np.asarray(z["t"], dtype=float),
        tips=np.asarray(z["tips"], dtype=float),
        grippers=np.asarray(z["grippers"], dtype=float),
        q_obs=np.asarray(z["q_obs"], dtype=float),
        q_cmd=np.asarray(z["q_cmd"], dtype=float),
        deltas=np.asarray(z["deltas"], dtype=float),
        seen=np.asarray(z["seen"]).astype(bool),
        delta0=np.asarray(z["delta0"], dtype=float),
        t0=float(z["t0"]),
        camera=str(z["camera"]) if "camera" in z.files else "camera",
        root=str(f.parent),
    )
    with _state.lock:
        running = _state.worker.running
    if not running:
        raise HTTPException(409, "start the worker first")
    job = _queue_job(
        "teach",
        demo.concept,
        np.asarray(z["teach_rgb"]),
        np.asarray(z["teach_depth"]),
        json.loads(str(z["intr"])),
    )
    with _state.lock:
        _state.teach_job = job.id
        _state.demo = demo
        _state.test = None
    return {**_demo_info(demo), "teach_pending": True}


def _act_tips(demo: _Demo, delta_cam: np.ndarray, t_bc: np.ndarray) -> np.ndarray:
    """The demo's path on the object where it is now: carried by the object's motion since the demo began."""
    rel = np.asarray(delta_cam, dtype=float) @ np.linalg.inv(demo.delta0)
    return core.transport_trajectory(t_bc, rel, demo.tips)


def _draw_path(bgr: np.ndarray, t_bc: np.ndarray, intr: dict[str, float], tips: np.ndarray) -> None:
    """The fingertip path on the image, start as a dot, end as a square."""
    import cv2

    stride = max(1, len(tips) // 200)
    pts = [p for p in (_project(t_bc, intr, tip[:3, 3]) for tip in tips[::stride]) if p is not None]
    for a, b in zip(pts, pts[1:], strict=False):
        cv2.line(bgr, a, b, (255, 200, 0), 2)
    if pts:
        cv2.circle(bgr, pts[0], 5, (255, 200, 0), -1)
        x, y = pts[-1]
        cv2.rectangle(bgr, (x - 4, y - 4), (x + 4, y + 4), (255, 200, 0), -1)


class ActBody(BaseModel):
    speed: float = 1.0  # time scale of the replay; the jog's own speed limits still cap the walk


async def _act_task(speed: float) -> None:
    from . import jog

    act = _state.act

    def fail(reason: str) -> None:
        act.ok, act.reason, act.step = False, reason, "aborted"

    try:
        with _state.lock:
            demo, test = _state.demo, _state.test
        if demo is None:
            fail("record or load a demo first")
            return
        if test is None or not test.result.get("ok"):
            fail("find the object first")
            return
        try:
            t_bc = _t_base_cam()
        except HTTPException as e:
            fail(e.detail)
            return
        tips = _act_tips(demo, test.result["delta_cam"], t_bc)

        async def settle(pose: np.ndarray, name: str) -> bool:
            t0 = time.monotonic()
            while True:
                try:
                    jog.set_target_pose(pose)
                except RuntimeError as e:
                    fail(str(e))
                    return False
                await asyncio.sleep(ACT_TICK_S)
                st = jog.current_status()
                cur = jog.current_tip_and_anchor()
                if not st.get("connected") or cur is None:
                    fail("the arm went away")
                    return False
                if st["halted"]:
                    fail(f"arm frozen: {st['reason']}")
                    return False
                if act.stop_requested:
                    fail("stopped")
                    return False
                if float(np.linalg.norm(cur[0][:3, 3] - pose[:3, 3])) < ACT_ARRIVE_M and not st["holding"]:
                    return True
                if time.monotonic() - t0 > ACT_STEP_TIMEOUT_S:
                    fail(f"{name}: not there after {ACT_STEP_TIMEOUT_S:.0f} s")
                    return False

        act.step = "to the start"
        jog.set_gripper(float(demo.grippers[0]))
        if not await settle(tips[0], "the start"):
            return
        act.step = "replaying"
        n = len(tips)
        t_start = time.monotonic()
        for i in range(n):
            due = t_start + float(demo.t[i] - demo.t[0]) / max(speed, 1e-3)
            while time.monotonic() < due:
                await asyncio.sleep(min(ACT_TICK_S, max(0.0, due - time.monotonic())))
            try:
                jog.set_target_pose(tips[i])
                jog.set_gripper(float(demo.grippers[i]))
            except RuntimeError as e:
                fail(str(e))
                return
            st = jog.current_status()
            if act.stop_requested:
                fail("stopped")
                return
            if st.get("halted"):
                fail(f"arm frozen: {st.get('reason')}")
                return
            act.progress = (i + 1) / n
        act.step = "settling"
        if not await settle(tips[-1], "the end"):
            return
        act.step, act.ok = "done", True
    except Exception as e:  # the arm holds its last target; the operator sees why
        logger.exception("act failed")
        fail(f"act error: {e}")
    finally:
        with contextlib.suppress(Exception):
            _record_trial()
        act.on = False


@router.post("/act")
async def act_start(body: ActBody) -> dict:
    """Replay the demo on the object where it is now: the whole recorded path, carried by the object's motion."""
    from . import jog

    with _state.lock:
        demo, test, act = _state.demo, _state.test, _state.act
        if act.on:
            raise HTTPException(409, "an act is in progress")
        if demo is None:
            raise HTTPException(409, "record or load a demo first")
        if test is None or not test.result.get("ok"):
            raise HTTPException(409, "find the object first")
        if (test.result.get("camera_check") or {}).get("moved"):
            raise HTTPException(409, "the camera or the tray moved since the calibration; recalibrate first")
        _state.track.follow = False  # the act owns the target now
        act.on, act.ok, act.reason, act.step, act.stop_requested, act.progress, act.speed = (
            True,
            None,
            "",
            "starting",
            False,
            0.0,
            body.speed,
        )
    if jog.current_robot_id() is None:
        with _state.lock:
            act.on = False
        raise HTTPException(409, "connect the arm first")
    act.task = asyncio.create_task(_act_task(body.speed))
    return {"status": "acting", "n": int(len(demo.t))}


@router.post("/act/stop")
async def act_stop() -> dict:
    """Stop the replay and hold the arm where it is."""
    from . import jog

    with _state.lock:
        _state.act.stop_requested = True
    cur = jog.current_tip_and_anchor()
    if cur is not None:
        with contextlib.suppress(RuntimeError):
            jog.set_target_pose(cur[0])
    return {"status": "stopping"}
