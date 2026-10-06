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
from collections.abc import Callable
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
    click: list[int] | None = None  # a teach by a clicked pixel instead of by name
    extra: dict[str, Any] = field(
        default_factory=dict
    )  # kind-specific fields handed to the worker as they are
    arrays: dict[str, np.ndarray] = field(default_factory=dict)  # arrays sent with the job's frame
    progress: float = 0.0  # what the worker reported of a long job, 0..1


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
    algo: str = "p2p"  # the one tracker; the DINO modes stay as a comparison
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
    intr: dict[str, float] | None = None  # the camera intrinsics the frames were taken with
    recording: str | None = (
        None  # the camera stream while recording: rgb/%06d.jpg, depth/%06d.png (mm), cam_K.txt, times.txt
    )
    objects: dict[str, dict[str, Any]] = field(default_factory=dict)  # designated on the stream, by name
    keypoints: list[dict[str, Any]] = field(default_factory=list)  # the operator's marks: t, name, anchor
    video: list[bytes] | None = None  # the saved video decoded once for the editor, one JPEG per sample
    taught: bool = False  # recorded with an object taught first; unnamed marks follow that object


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
    plan: dict[str, Any] | None = (
        None  # what the last act judged before moving: per mark reach, the path's worst
    )


@dataclass
class _StreamRecorder:
    """The camera's colour and depth written to disk while a demo is recorded, for designating objects afterwards."""

    out: pathlib.Path
    stop: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    n: int = 0
    error: str = ""


def _record_stream(camera: Any, rec: _StreamRecorder) -> None:
    """Write every new camera frame until stopped: rgb/%06d.jpg, depth/%06d.png (uint16 mm), cam_K.txt, times.txt.

    Reads go through the camera executor like every other reader. A read waits for a
    new frame, so no frame is written twice; when the live tracker also reads, the two
    share the camera's frames.
    """
    import cv2

    from . import showservo

    out = rec.out
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(exist_ok=True)
    intr = camera.color_intrinsics()
    np.savetxt(
        out / "cam_K.txt", [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]]
    )
    times: list[float] = []
    try:
        while not rec.stop.is_set():
            try:
                rgb, depth_mm = showservo._EXECUTOR.submit(camera.read_color_and_aligned_depth).result()
            except TimeoutError:
                continue  # the other reader took that frame
            times.append(time.time())
            cv2.imwrite(
                str(out / "rgb" / f"{rec.n:06d}.jpg"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
            cv2.imwrite(
                str(out / "depth" / f"{rec.n:06d}.png"),
                depth_mm.astype(np.uint16),
                [int(cv2.IMWRITE_PNG_COMPRESSION), 1],
            )
            rec.n += 1
    except Exception as e:  # the stop endpoint reports it; the arm's own recording goes on
        logger.exception("camera stream recording failed")
        rec.error = str(e)
    finally:
        np.savetxt(out / "times.txt", times, fmt="%.6f")


def _stream_times(recording: str) -> np.ndarray:
    f = pathlib.Path(recording) / "times.txt"
    return np.atleast_1d(np.loadtxt(f)) if f.exists() and f.stat().st_size else np.zeros(0)


def _stream_frame(recording: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Frame ``k`` of a recorded stream: (RGB uint8, depth in metres)."""
    import cv2

    root = pathlib.Path(recording)
    bgr = cv2.imread(str(root / "rgb" / f"{k:06d}.jpg"), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(root / "depth" / f"{k:06d}.png"), cv2.IMREAD_UNCHANGED)
    assert bgr is not None and depth is not None, f"frame {k} missing from {root}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth.astype(np.float32) / 1000.0


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
    stream: _StreamRecorder | None = None  # the camera stream being written while a demo is recorded
    run: _Run | None = None  # the act being recorded: its tracker frames, its targets, the arm
    camera_recording: _StreamRecorder | None = (
        None  # the camera recorded on its own, to replay the tracker over
    )


_state = _State()
_RENDER_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pregrasp-render")
_ACT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pregrasp-act")
# An act's recording goes to disk on its own thread, in order; the tracker never waits for it.
_RUN_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pregrasp-run")
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
    mode: str = "box"  # "box" | "features" (SAM3 by concept or by a clicked pixel, in the worker)
    concept: str = ""
    click: list[int] = []  # x, y in frame pixels: SAM3 segments what is under it instead of a name
    ref_object: str = (
        ""  # a demo's designated object: the live view is registered against the demo's view of it
    )


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
            "plan": s.act.plan,
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
        click = [int(v) for v in body.click] if body.click else None
        if click is not None and len(click) != 2:
            raise HTTPException(422, "click is x, y")
        concept = body.concept.strip() or ("clicked object" if click else "")
        if not concept:
            raise HTTPException(422, "a concept is required, e.g. 'yellow block', or click the object")
        job_id = await _start_teach(click, concept, body.ref_object)
        return {"pending": True, "job": job_id, "mode": "features"}
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


async def _start_teach(click: list[int] | None, concept: str, ref_object: str) -> str:
    """Queue a features teach of the object at ``click`` on the live frame; with ``ref_object``, also its find
    against the demo's view of that object. Post: the job's id, the teach pending and the live pose cleared.
    Raises HTTPException when the worker is not running or ``ref_object`` is not tracked in the demo."""
    if not _state.worker.running:
        raise HTTPException(409, "start the worker first")
    rgb, depth_m, intr = await _frame()
    with _state.lock:
        mode = _state.track.algo  # the Point2Pose mode the teach anchors, when one is selected
    ref = None
    if ref_object:
        with _state.lock:
            demo = _state.demo
        ref = None if demo is None else demo.objects.get(ref_object)
        if ref is None or ref.get("status") != "done" or demo.recording is None:
            raise HTTPException(409, f"{ref_object!r} is not a tracked object of the current demo")
        concept = ref_object
    job = _queue_job("teach", concept, rgb, depth_m, intr, algo=mode, click=click)
    if ref is not None:
        job.extra = {
            "ref_recording": demo.recording,
            "ref_frame": int(ref["frame"]),
            "ref_object": ref_object,
        }
        job.arrays = {"ref_mask": np.asarray(ref["mask"], dtype=bool)}
    with _state.lock:
        _state.teach_job = job.id
        _state.test = None
    return job.id


def _deepest_pixel(mask: np.ndarray | None, shape: tuple[int, ...]) -> list[int] | None:
    """``[x, y]`` in an image of ``shape`` at the point deepest inside ``mask``, where a click lands on the
    object rather than on its edge; None for an empty mask."""
    import cv2

    if mask is None or not np.any(mask):
        return None
    m = np.asarray(mask, dtype=np.uint8)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return [int(x * shape[1] / m.shape[1]), int(y * shape[0] / m.shape[0])]


async def _find_afresh(obj: str, stopped: Callable[[], bool]) -> str:
    """Find ``obj`` again where it was last seen, as a click there would, and wait for the restarted tracker to
    certify a view. A track kept since an earlier find goes on adding points and never drops one, and its pose
    drifts with them; a find starts it over from the demo's view of the object.

    Pre: ``stopped()`` says whether the act was stopped. Post: "" when found and tracked since, the teach and
    the live pose fresh; otherwise why not, with nothing moved.
    """
    with _state.lock:
        test, teach = _state.test, _state.teach
    if teach is None:
        return f"click {obj} in the camera view to find it"
    seen = test.result.get("live_mask") if test is not None and test.result.get("ok") else None
    click = _deepest_pixel(seen if seen is not None else teach.keypoints.get("mask"), teach.rgb.shape)
    if click is None:
        return f"click {obj} in the camera view to find it"
    try:
        job_id = await _start_teach(click, obj, obj)
    except HTTPException as e:
        return str(e.detail)
    t0 = time.monotonic()
    while True:
        with _state.lock:
            pending, teach = _state.teach_job == job_id, _state.teach
        if not pending:
            break
        if stopped():
            return "stopped"
        if time.monotonic() - t0 > ACT_STEP_TIMEOUT_S:
            return f"finding {obj} took over {ACT_STEP_TIMEOUT_S:.0f} s"
        await asyncio.sleep(ACT_TICK_S)
    ref = (teach.keypoints.get("ref") if teach is not None else None) or {}
    if not ref.get("ok"):
        return ref.get("reason") or f"{obj} is not where it was last seen: click it in the camera view"
    found, t0 = time.time(), time.monotonic()
    while not _certified_since(found):
        if stopped():
            return "stopped"
        if time.monotonic() - t0 > ACT_STEP_TIMEOUT_S:
            return f"the tracker has not seen {obj} since finding it"
        await asyncio.sleep(ACT_TICK_S)
    return ""


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
            "ref": None
            if not kp.get("ref")
            else {k: kp["ref"][k] for k in ("object", "ok", "inliers", "turn_deg", "reason")},
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
    click: list[int] | None = None,
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
        click=click,
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
                        {
                            "id": job.id,
                            "kind": job.kind,
                            "concept": job.concept,
                            "algo": job.algo,
                            "click": job.click,
                            **job.extra,
                        }
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
    save(buf, rgb=job.rgb, depth=job.depth_m, intr=np.array(json.dumps(job.intr)), **job.arrays)
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
    for key in (
        "mask",
        "uv",
        "xyz",
        "live_uv",
        "delta",
        "deltas",
        "seen",
        "masks",
        "ref_delta",
        "fit_uv",
        "fit_inlier",
    ):
        if key in data.files:
            result[key] = np.asarray(data[key])
    job.result = result
    if job.kind == "teach":
        _apply_teach_result(job)
    elif job.kind == "track":
        await _apply_track_result(job)
    elif job.kind == "stream_object":
        await _apply_object_result(job)
    else:
        _apply_find_result(job)
    return {"status": "ok"}


@router.post("/worker/progress")
async def worker_progress(id: str, done: int, total: int) -> dict:
    """A long job's progress, as the worker reports it."""
    with _state.lock:
        job = _state.worker.jobs.get(id)
        if job is None:
            raise HTTPException(404, "no such job")
        job.progress = float(np.clip(done / max(total, 1), 0.0, 1.0))
    return {"progress": job.progress}


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
        if job.extra.get("ref_object"):
            kp["ref"] = {
                "object": job.extra["ref_object"],
                "ok": bool(r.get("ref_ok")),
                "delta": None if r.get("ref_delta") is None else np.asarray(r["ref_delta"], dtype=float),
                "inliers": r.get("ref_inliers"),
                "turn_deg": r.get("ref_turn_deg"),
                "reason": r.get("ref_reason", ""),
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
        running = _state.worker.running
    # A taught object is tracked from that moment: the guided flow has no separate "start tracking".
    from . import showservo

    if running and showservo.live_camera() is not None:
        _begin_track()


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
    algo: str = "p2p"  # the one tracker; the DINO modes stay as a comparison
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
                path = _act_preview(demo, result["delta_cam"], t_bc)
                if path is not None:
                    _draw_path(bgr, t_bc, teach.intr, path)
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
                # The fitted points are the recording's, not the live readout's: arrays the state cannot serve.
                if k not in ("mask", "uv", "xyz", "delta", "face_teach", "face_find", "fit_uv", "fit_inlier")
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
        if rec is not None and _state.stream is None:
            rec["frames"].append(
                (time.time(), job.rgb[::2, ::2].copy())
            )  # half size: a demo's video is a record, not evidence
        run, step = _state.run, _state.act.step
    if run is not None:
        run.frame(job, r, status, step, None if result is None else result["delta_cam"], transported)
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
    _begin_track()
    return {"status": "tracking", "algo": tr.algo, "follow": tr.follow}


def _begin_track() -> bool:
    """Start the live track with the current settings if it is not running. Post: True when it is
    running after the call. Called on the event loop (an endpoint, or the worker-result handler)."""
    tr = _state.track
    with _state.lock:
        already = tr.on
        if not already:
            tr.on, tr.last, tr.fps, tr.t_prev, tr.overlay = True, {"state": "starting"}, 0.0, 0.0, None
            tr.done = asyncio.Event()
    if not already:
        tr.task = asyncio.create_task(_track_pump())
    return True


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


def _record_trial(run_dir: str | None = None) -> dict[str, Any]:
    """Append the act that just ended: what was found, which demo was replayed, how it ended, its recording."""
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
        "run": run_dir,
    }
    if teach is not None and r.get("ok") and r.get("delta_cam") is not None:
        d = np.asarray(r["delta_cam"])
        c = np.asarray(teach.keypoints["xyz"]).mean(axis=0)
        row["centre_shift_mm"] = float(np.linalg.norm(d[:3, :3] @ c + d[:3, 3] - c) * 1000.0)
    rows = _load_trials()
    rows.append(row)
    _save_trials(rows)
    return row


ACTS_DIR = "acts"  # beside a saved demo: one folder per act, all it saw and did
RUN_FRAME_FIELDS = (
    "ok",
    "state",
    "reason",
    "ms",
    "n_matches",
    "lost_streak",
    "n_tracks",
    "n_model_points",
    "fit_points",
    "fit_inliers",
    "jump_guard_rejected",
)


def _jsonable(o: Any) -> Any:
    return o.tolist() if hasattr(o, "tolist") else str(o)


@dataclass
class _Run:
    """One act's recording: every tracker frame from the arm's first move (image, depth, mask, the pose and
    the points it was fitted on), every target the act gave the arm, and the arm's joints at the loop rate."""

    root: pathlib.Path
    meta: dict[str, Any]
    frames: list[dict[str, Any]] = field(default_factory=list)
    targets: list[dict[str, Any]] = field(default_factory=list)
    arm_recording: bool = False

    def frame(
        self,
        job: _Job,
        r: dict[str, Any],
        status: dict[str, Any],
        step: str,
        delta_used: np.ndarray | None,
        transported: np.ndarray | None,
    ) -> None:
        """One tracker result, on the loop; its files are written on the run's own thread."""
        i = len(self.frames)
        self.frames.append(
            {
                "i": i,
                "t_frame": job.created,
                "t_result": time.time(),
                "step": step,
                "used": bool(status.get("ok")),  # the act follows only a frame the server trusted
                **{k: r.get(k) for k in RUN_FRAME_FIELDS},
            }
        )
        arrays = {
            k: np.asarray(r[k]) for k in ("delta", "fit_uv", "fit_inlier", "live_uv") if r.get(k) is not None
        }
        if delta_used is not None:
            arrays["delta_used"] = np.asarray(delta_used)
        if transported is not None:
            arrays["transported"] = np.asarray(transported)
        _RUN_EXECUTOR.submit(_write_run_frame, self.root, i, job.rgb, job.depth_m, r.get("mask"), arrays)

    def target(self, step: str, pose: np.ndarray | None = None, joints: np.ndarray | None = None) -> None:
        entry: dict[str, Any] = {"t": time.time(), "step": step}
        if pose is not None:
            entry["pose"] = np.asarray(pose, dtype=float).tolist()
        if joints is not None:
            entry["joints"] = np.asarray(joints, dtype=float).tolist()
        self.targets.append(entry)


def _write_run_frame(
    root: pathlib.Path,
    i: int,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray | None,
    arrays: dict[str, np.ndarray],
) -> None:
    import cv2

    d = root / "frames"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{i:06d}.jpg"), cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
    depth_mm = np.round(np.nan_to_num(np.asarray(depth_m, dtype=np.float64)) * 1000.0)
    cv2.imwrite(str(d / f"{i:06d}_depth.png"), np.clip(depth_mm, 0, 65535).astype(np.uint16))
    if mask is not None:
        cv2.imwrite(str(d / f"{i:06d}_mask.png"), np.asarray(mask, dtype=np.uint8) * 255)
    np.savez_compressed(d / f"{i:06d}.npz", **arrays)


def _begin_run(demo: _Demo, speed: float, delta: np.ndarray, t_bc: np.ndarray, plan: dict[str, Any]) -> _Run:
    """Start recording the act, beside its demo when it is saved. Post: ``_state.run`` is the recording."""
    from . import jog

    with _state.lock:
        teach, track = _state.teach, _state.track
        ref = dict((teach.keypoints.get("ref") or {}) if teach is not None else {})
        algo, fps, last = track.algo, track.fps, dict(track.last or {})
    base = pathlib.Path(demo.root) / ACTS_DIR if demo.root else _demos_root() / ".acts"
    meta: dict[str, Any] = {
        "demo": demo.name,
        "demo_root": demo.root,
        "keypoints": list(demo.keypoints),
        "speed": speed,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "t_started": time.time(),
        "t_bc": np.asarray(t_bc, dtype=float).tolist(),
        "intr": None if teach is None else dict(teach.intr),
        "delta_at_start": np.asarray(delta, dtype=float).tolist(),
        "find": ref,
        "tracker_at_start": {"algo": algo, "fps": fps, **{k: last.get(k) for k in RUN_FRAME_FIELDS}},
        "plan": {k: plan.get(k) for k in ("ok", "reason", "marks", "summary")},
    }
    run = _Run(root=base / time.strftime("%Y%m%d_%H%M%S"), meta=meta)
    try:
        meta["arm_t0"] = jog.start_record()
        run.arm_recording = True
    except RuntimeError as e:
        meta["arm_error"] = str(e)
    with _state.lock:
        _state.run = run
    return run


def _finish_run(run: _Run, samples: list[dict[str, Any]], fk: Any, result: dict[str, Any]) -> str:
    """Write the act's summary and the arm's joints, after every queued frame (the run's thread is in order)."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    run.root.mkdir(parents=True, exist_ok=True)
    if samples:
        arm: dict[str, np.ndarray] = {
            "t": np.array([s["t"] for s in samples], dtype=float) + float(run.meta.get("arm_t0", 0.0)),
            "q_obs": np.array([[s["obs"][m] for m in MOTOR_NAMES] for s in samples], dtype=float),
            "q_cmd": np.array(
                [[s["cmd"].get(m, s["obs"][m]) for m in MOTOR_NAMES] for s in samples], dtype=float
            ),
            "motors": np.array(MOTOR_NAMES),
        }
        with contextlib.suppress(Exception):  # the arm may have gone away; the joints are still the record
            arm["tip_obs"] = np.stack([fk(s["obs"]) for s in samples])
        np.savez_compressed(run.root / "arm.npz", **arm)
    summary = {
        **run.meta,
        "ended": time.strftime("%Y-%m-%d %H:%M:%S"),
        "result": result,
        "frames": run.frames,
        "targets": run.targets,
    }
    (run.root / "act.json").write_text(json.dumps(summary, indent=1, default=_jsonable))
    return str(run.root)


class VerdictBody(BaseModel):
    index: int
    verdict: str | None = None  # null clears a verdict given by mistake


@router.get("/trials")
async def trials() -> dict:
    return {"rows": _load_trials()}


@router.post("/trials/verdict")
async def trial_verdict(body: VerdictBody) -> dict:
    """The operator's word on a run: what the camera cannot see once the gripper covers the object."""
    if body.verdict is not None and body.verdict not in TRIAL_VERDICTS:
        raise HTTPException(422, f"verdict must be one of {TRIAL_VERDICTS}, or null to clear it")
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
ACT_SETTLE_DEG = 2.0  # arm joints this close to the last target have arrived: the servo's own band
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
        "stream_frames": int(len(_stream_times(demo.recording))) if demo.recording else 0,
        "objects": [name for name, o in demo.objects.items() if o.get("status") == "done"],
        "root": demo.root,
        "repo_id": f"{DEMOS_NAMESPACE}/{demo.name}",
        "keypoints": list(demo.keypoints),
        "has_frames": _demo_has_frames(demo),
        "taught": demo.taught,
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
    """Record the arm (any mode: the leader or the gizmo) and the camera's colour and depth.

    Nothing has to be taught first: the objects that matter are designated on the
    recording afterwards. While tracking runs, the tracked object's motion is
    recorded too.
    """
    from . import jog, showservo

    with _state.lock:
        if _state.camera_recording is not None:
            raise HTTPException(409, "the camera is being recorded on its own; stop that first")
    try:
        t0 = jog.start_record()
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    camera = showservo.live_camera()
    stream = None
    if camera is not None:
        stream = _StreamRecorder(out=_demos_root() / ".recordings" / time.strftime("%Y%m%d_%H%M%S"))
        stream.thread = threading.Thread(
            target=_record_stream, args=(camera, stream), name="pregrasp-demo-stream", daemon=True
        )
        stream.thread.start()
    with _state.lock:
        _state.recording = {"t0": t0, "frames": []}
        _state.stream = stream
        tracking = _state.track.on
    return {"status": "recording", "tracking": tracking, "camera": stream is not None}


@router.post("/camera/record/start")
async def camera_record_start() -> dict:
    """Record the camera's colour and depth on their own, as a demo does: a run to replay the tracker over."""
    from . import showservo

    camera = showservo.live_camera()
    if camera is None:
        raise HTTPException(409, "start the camera first")
    with _state.lock:
        if _state.camera_recording is not None:
            raise HTTPException(409, "the camera is already being recorded")
        if _state.stream is not None:
            raise HTTPException(409, "a demo is recording the camera")  # two readers would split its frames
        rec = _StreamRecorder(out=_demos_root() / ".recordings" / f"camera_{time.strftime('%Y%m%d_%H%M%S')}")
        _state.camera_recording = rec
    rec.thread = threading.Thread(
        target=_record_stream, args=(camera, rec), name="pregrasp-camera", daemon=True
    )
    rec.thread.start()
    return {"status": "recording", "out": str(rec.out)}


@router.post("/camera/record/stop")
async def camera_record_stop() -> dict:
    with _state.lock:
        rec, _state.camera_recording = _state.camera_recording, None
    if rec is None:
        raise HTTPException(409, "the camera is not being recorded")
    rec.stop.set()
    if rec.thread is not None:
        await asyncio.get_event_loop().run_in_executor(_RENDER_EXECUTOR, rec.thread.join, 5.0)
    return {"out": str(rec.out), "frames": rec.n, "error": rec.error}


@router.post("/demo/record/stop")
async def demo_record_stop(body: DemoNameBody) -> dict:
    """End the recording and keep it as the current demo (not yet saved)."""
    from . import jog, showservo

    with _state.lock:
        rec, teach, history = _state.recording, _state.teach, list(_state.track.history)
        stream, previous = _state.stream, _state.demo
        _state.recording, _state.stream = None, None
    if rec is None:
        raise HTTPException(409, "not recording")
    if stream is not None and stream.thread is not None:
        stream.stop.set()
        await asyncio.get_event_loop().run_in_executor(_RENDER_EXECUTOR, stream.thread.join, 5.0)
    try:
        samples = jog.stop_record()
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    if len(samples) < 2:
        raise HTTPException(409, "the recording is empty")
    name = _safe_name(body.name) or time.strftime("demo_%Y%m%d_%H%M%S")
    concept = teach.keypoints["concept"] if teach is not None else "demo"
    _discard_unsaved_stream(previous)

    def build() -> _Demo:
        return _demo_from_samples(name, concept, samples, history, jog.fk_tip, rec["t0"])

    demo = await asyncio.get_event_loop().run_in_executor(_RENDER_EXECUTOR, build)
    demo.frames = rec["frames"] or None
    demo.taught = teach is not None
    demo.camera = _camera_label()
    camera = showservo.live_camera()
    if teach is not None:
        demo.intr = dict(teach.intr)
    elif camera is not None:
        demo.intr = dict(camera.color_intrinsics())
    if stream is not None and stream.n:
        demo.recording = str(stream.out)
    with _state.lock:
        _state.demo = demo
        if teach is not None:
            teach.tip_pose, teach.gripper = demo.tips[0].copy(), float(demo.grippers[0])
    info = _demo_info(demo)
    if stream is not None and stream.error:
        info["stream_error"] = stream.error
    return info


def _discard_unsaved_stream(demo: _Demo | None) -> None:
    """A recorded demo that was never saved is replaced by the next one; its working stream goes with it."""
    import shutil

    if demo is None or demo.root is not None or demo.recording is None:
        return
    work = pathlib.Path(demo.recording)
    if work.parent == _demos_root() / ".recordings" and work.exists():
        shutil.rmtree(work)  # safe-destruct: our own working copy of a demo the operator never saved


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


def _write_demo(demo: _Demo, teach: _Teach | None) -> pathlib.Path:
    """The demo as a LeRobot dataset the Data tab can play, plus a sidecar with what the act needs.

    The camera stream, when one was recorded, moves into the demo as ``recording/``;
    the dataset's video is made from it. ``teach`` is the object taught before the
    demo, if any; a demo recorded without one designates its objects afterwards.
    """
    import shutil

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES
    from lerobot.utils.constants import OBS_IMAGES

    root = _demos_root() / demo.name
    stream = pathlib.Path(demo.recording) if demo.recording else None
    if stream is not None and root in stream.parents:
        # A re-save replaces the folder below; the stream steps aside first.
        parked = _demos_root() / ".recordings" / f"{demo.name}_resave_{time.strftime('%H%M%S')}"
        parked.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stream), str(parked))  # safe-destruct: our own stream, moved aside, not deleted
        stream = parked
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
    # The video's frames are read one at a time: a long stream does not fit in memory at once.
    frame_times, video_frame = _video_source(demo, stream)
    image_key = f"{OBS_IMAGES}.{demo.camera}"
    frames = frame_times is not None and len(frame_times) > 0
    if frames:
        h, w = video_frame(0).shape[:2]
        features[image_key] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
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
            frame[image_key] = np.ascontiguousarray(video_frame(k), dtype=np.uint8)
        ds.add_frame(frame)
    ds.save_episode()
    ds.finalize()
    taught: dict[str, Any] = {}
    if teach is not None and demo.taught:
        taught = {
            "teach_rgb": teach.rgb,
            "teach_depth": teach.depth_m,
            "teach_mask": np.asarray(teach.keypoints.get("mask", np.zeros(teach.depth_m.shape, dtype=bool))),
        }
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
        intr=json.dumps(teach.intr if teach is not None else (demo.intr or {})),
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        taught=demo.taught,
        **taught,
    )
    _write_keypoints(root, demo.keypoints)
    _write_objects(root, demo.objects)
    if stream is not None and stream.exists():
        # safe-destruct: our own stream, moved into its demo
        shutil.move(str(stream), str(root / DEMO_RECORDING))
        demo.recording = str(root / DEMO_RECORDING)
    return root


DEMO_RECORDING = "recording"


def _video_source(demo: _Demo, stream: pathlib.Path | None) -> tuple[np.ndarray | None, Any]:
    """The dataset video's frame times and a reader for frame ``k`` at half size: the stream, else the small frames.

    The reader keeps the last frame, since consecutive arm samples usually fall on the same camera frame.
    """
    import functools

    if stream is not None and (stream / "times.txt").exists():

        @functools.lru_cache(maxsize=1)
        def from_stream(k: int) -> np.ndarray:
            return _stream_frame(str(stream), k)[0][::2, ::2].copy()

        return _stream_times(str(stream)), from_stream
    if demo.frames:
        return np.array([f[0] for f in demo.frames]), lambda k: demo.frames[k][1]
    return None, None


KEYPOINTS_FILE = "keypoints.json"


def _write_keypoints(root: pathlib.Path, keypoints: list[dict[str, Any]]) -> None:
    """The operator's marks as a sidecar the act reads back; nothing is written when there are none."""
    f = pathlib.Path(root) / KEYPOINTS_FILE
    if keypoints:
        f.write_text(json.dumps({"keypoints": keypoints}, indent=1))
    elif f.exists():
        f.unlink()  # safe-destruct: the marks sidecar we wrote ourselves; the operator cleared the marks


def _read_keypoints(root: pathlib.Path) -> list[dict[str, Any]]:
    f = pathlib.Path(root) / KEYPOINTS_FILE
    if not f.exists():
        return []
    try:
        return list(json.loads(f.read_text()).get("keypoints", []))
    except (OSError, ValueError):
        return []


@router.post("/demo/save")
async def demo_save(body: DemoNameBody) -> dict:
    """Write the current demo as a dataset under the demos namespace; a name given here renames it."""
    with _state.lock:
        demo, teach = _state.demo, _state.teach
    if demo is None:
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
        intr=json.loads(str(z["intr"])),
        keypoints=_read_keypoints(f.parent),
        recording=str(f.parent / DEMO_RECORDING)
        if (f.parent / DEMO_RECORDING / "times.txt").exists()
        else None,
        objects=_read_objects(f.parent),
        taught=bool(z["taught"]) if "taught" in z.files else "teach_rgb" in z.files,
    )
    if core.keypoints_problem(demo.keypoints, float(demo.t[0]), float(demo.t[-1])):
        demo.keypoints = []  # marks in a shape this version does not read
    if "teach_rgb" not in z.files:  # recorded without a taught object: nothing to re-teach
        with _state.lock:
            _discard_unsaved_stream(_state.demo)
            _state.demo = demo
            _state.test = None
        return {**_demo_info(demo), "teach_pending": False}
    with _state.lock:
        running = _state.worker.running
    if not running:
        raise HTTPException(409, "start the worker first")
    mask = np.asarray(z["teach_mask"]).astype(bool) if "teach_mask" in z.files else None
    click = None
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        click = [int(np.median(xs)), int(np.median(ys))]
    with _state.lock:
        algo = _state.track.algo
    job = _queue_job(
        "teach",
        demo.concept,
        np.asarray(z["teach_rgb"]),
        np.asarray(z["teach_depth"]),
        json.loads(str(z["intr"])),
        algo=algo,
        click=click,
    )
    with _state.lock:
        _discard_unsaved_stream(_state.demo)
        _state.teach_job = job.id
        _state.demo = demo
        _state.test = None
    return {**_demo_info(demo), "teach_pending": True}


def _demo_video_file(root: str) -> pathlib.Path | None:
    files = sorted(pathlib.Path(root).glob("videos/**/*.mp4"))
    return files[0] if files else None


def _decode_demo_video(path: pathlib.Path) -> list[bytes]:
    """Every frame of the saved demo video as JPEG, decoded once; the dataset's AV1 needs PyAV's dav1d."""
    import av
    import cv2

    out: list[bytes] = []
    with av.open(str(path)) as container:
        for frame in container.decode(container.streams.video[0]):
            rgb = frame.to_ndarray(format="rgb24")
            ok, buf = cv2.imencode(
                ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            )
            if ok:
                out.append(buf.tobytes())
    return out


def _demo_has_frames(demo: _Demo) -> bool:
    return (
        demo.recording is not None
        or bool(demo.frames)
        or bool(demo.video)
        or (demo.root is not None and _demo_video_file(demo.root) is not None)
    )


def _demo_frame_rgb(demo: _Demo, i: int) -> np.ndarray | None:
    """The camera frame nearest sample ``i``: the recorded stream at full size, else the small frames, else the video."""
    import cv2

    if demo.recording is not None:
        times = _stream_times(demo.recording)
        if len(times):
            return _stream_frame(demo.recording, int(np.argmin(np.abs(times - (demo.t0 + demo.t[i])))))[0]
    if demo.frames:
        times = np.array([f[0] for f in demo.frames])
        k = int(np.argmin(np.abs(times - (demo.t0 + demo.t[i]))))
        return np.asarray(demo.frames[k][1])
    if demo.video is None and demo.root is not None:
        f = _demo_video_file(demo.root)
        demo.video = _decode_demo_video(f) if f is not None else []
    if demo.video:
        bgr = cv2.imdecode(np.frombuffer(demo.video[min(i, len(demo.video) - 1)], np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return None


@router.get("/demo/curve")
async def demo_curve() -> dict:
    """What the editor draws under its slider: the demo's time axis, gripper, visibility, marks and fingertip path."""
    with _state.lock:
        demo = _state.demo
    if demo is None:
        raise HTTPException(404, "no demo")
    return {
        "name": demo.name,
        "n": int(len(demo.t)),
        "t": demo.t.tolist(),
        "gripper": demo.grippers.tolist(),
        "seen": _demo_seen(demo).astype(int).tolist(),
        "keypoints": list(demo.keypoints),
        "has_frames": _demo_has_frames(demo),
        "recording": demo.recording is not None,
        "taught": demo.taught,
        "image_size": None if demo.intr is None else [demo.intr["width"], demo.intr["height"]],
        "uv": _demo_path_uv(demo),
        "pose_t": _pose_frame_t(demo),
    }


def _pose_frame_t(demo: _Demo) -> float | None:
    """When, in the demo's time, the act reads the object's pose: the editor marks it on the timeline."""
    f = _pose_frame(demo)
    return None if f is None else float(_stream_times(demo.recording)[f] - demo.t0)


def _demo_seen(demo: _Demo) -> np.ndarray:
    """Per sample, whether the object the act follows was seen: the designated object's own track on the recording
    when the marks name one (the track whose outline the editor draws), else the live tracker's frames during the
    recording, which come a few times a second and leave samples between them unseen."""
    obj = _marks_object(demo)
    o = demo.objects.get(obj) if obj else None
    times = _stream_times(demo.recording) if demo.recording is not None else np.zeros(0)
    if o is None or o.get("status") != "done" or not len(times):
        return demo.seen
    frames = np.abs(times[None, :] - (demo.t0 + demo.t)[:, None]).argmin(axis=1)
    return np.asarray(o["seen"], dtype=bool)[frames]


def _demo_path_uv(demo: _Demo) -> list[list[int] | None] | None:
    """The fingertip path in the camera image (full-resolution pixels), or None without a camera calibration."""
    try:
        t_bc = _t_base_cam()
    except HTTPException:
        return None
    if demo.intr is None:
        return None
    return [
        None if (uv := _project(t_bc, demo.intr, tip[:3, 3])) is None else [uv[0], uv[1]] for tip in demo.tips
    ]


@router.get("/demo/frame.jpg")
async def demo_frame(i: int = 0) -> Response:
    """Frame ``i`` of the current demo with the designated objects' outlines; the page draws path and marks."""
    with _state.lock:
        demo = _state.demo
    if demo is None:
        raise HTTPException(404, "no demo")
    if not _demo_has_frames(demo):
        raise HTTPException(404, "this demo has no camera frames")
    i = int(np.clip(i, 0, len(demo.t) - 1))

    def render() -> bytes:
        import cv2

        rgb = _demo_frame_rgb(demo, i)
        assert rgb is not None, "has_frames said so"
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        if demo.recording is not None and demo.objects:
            times = _stream_times(demo.recording)
            _draw_objects(bgr, demo, int(np.argmin(np.abs(times - (demo.t0 + demo.t[i])))))
        return _jpeg(bgr)

    jpg = await asyncio.get_event_loop().run_in_executor(_RENDER_EXECUTOR, render)
    return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


OBJECTS_FILE = "objects.npz"
OBJECT_COLOURS = (
    (255, 170, 0),
    (80, 200, 255),
    (120, 230, 120),
    (240, 110, 200),
    (255, 230, 90),
)  # BGR, by order


class ObjectBody(BaseModel):
    i: int  # the demo sample the editor shows
    x: float  # the clicked pixel, in the stream's full-size frame
    y: float
    name: str


@router.post("/demo/objects")
async def demo_object_add(body: ObjectBody) -> dict:
    """Designate an object by clicking it on a frame of the recording; it is then tracked through the whole stream."""
    with _state.lock:
        demo, running = _state.demo, _state.worker.running
    if demo is None:
        raise HTTPException(409, "record or load a demo first")
    if demo.recording is None:
        raise HTTPException(409, "this demo has no camera stream to designate on")
    if not running:
        raise HTTPException(409, "start the worker first")
    name = _safe_name(body.name)
    if not name:
        raise HTTPException(422, "name the object")
    times = _stream_times(demo.recording)
    i = int(np.clip(body.i, 0, len(demo.t) - 1))
    k = int(np.argmin(np.abs(times - (demo.t0 + demo.t[i]))))
    rgb, depth = await asyncio.get_event_loop().run_in_executor(
        _RENDER_EXECUTOR, _stream_frame, demo.recording, k
    )
    if not (0 <= body.x < rgb.shape[1] and 0 <= body.y < rgb.shape[0]):
        raise HTTPException(422, "the click is outside the frame")
    job = _queue_job(
        "stream_object",
        name,
        rgb,
        depth,
        dict(demo.intr or {}),
        click=[int(round(body.x)), int(round(body.y))],
    )
    job.extra = {"recording": demo.recording, "frame": k, "name": name}
    with _state.lock:
        demo.objects[name] = {"frame": k, "click": job.click, "status": "tracking", "job": job.id}
    return _objects_info(demo)


class ObjectNameBody(BaseModel):
    name: str


@router.post("/demo/objects/remove")
async def demo_object_remove(body: ObjectNameBody) -> dict:
    with _state.lock:
        demo = _state.demo
        if demo is None or body.name not in demo.objects:
            raise HTTPException(404, "no such object")
        del demo.objects[body.name]
    if demo.root is not None:
        await asyncio.get_event_loop().run_in_executor(
            _RENDER_EXECUTOR, _write_objects, pathlib.Path(demo.root), demo.objects
        )
    return _objects_info(demo)


@router.get("/demo/objects")
async def demo_objects() -> dict:
    with _state.lock:
        demo = _state.demo
    if demo is None:
        raise HTTPException(404, "no demo")
    return _objects_info(demo)


def _objects_info(demo: _Demo) -> dict[str, Any]:
    """The designated objects for the editor: name, where it was clicked, how far tracking got and how much it saw."""
    times = _stream_times(demo.recording) if demo.recording else np.zeros(0)
    with _state.lock:
        jobs = dict(_state.worker.jobs)
    out = []
    for n, (name, o) in enumerate(demo.objects.items()):
        job = jobs.get(o.get("job", ""))
        b, g, r = OBJECT_COLOURS[n % len(OBJECT_COLOURS)]
        out.append(
            {
                "name": name,
                "colour": f"#{r:02x}{g:02x}{b:02x}",
                "frame": int(o["frame"]),
                "t": float(times[o["frame"]] - demo.t0) if len(times) > o["frame"] else None,
                "status": o["status"],
                "progress": 1.0 if o["status"] != "tracking" else (job.progress if job else 0.0),
                "seen_fraction": float(np.mean(o["seen"])) if o.get("seen") is not None else None,
                "reason": o.get("reason", ""),
            }
        )
    return {"objects": out}


async def _apply_object_result(job: _Job) -> None:
    """Keep what the worker tracked for an object on the demo's stream, and write it beside a saved demo.

    Post on success: the object's entry has ``deltas`` (K, 4, 4), its motion from the
    clicked frame in camera coordinates; ``seen`` (K,); ``masks`` (K, h/4, w/4); ``mask``,
    full size on the clicked frame; one row per stream frame.
    """
    r = job.result or {}
    with _state.lock:
        demo = _state.demo
        entry = None
        if demo is not None:
            entry = next((o for o in demo.objects.values() if o.get("job") == job.id), None)
        if entry is None:
            return  # the object was removed, or another demo is current
        if r.get("ok"):
            entry.update(
                status="done",
                deltas=np.asarray(r["deltas"], dtype=float),
                seen=np.asarray(r["seen"]).astype(bool),
                masks=np.asarray(r["masks"]).astype(bool),
                mask=np.asarray(r["mask"]).astype(bool),
            )
        else:
            entry.update(status="failed", reason=str(r.get("reason", "the worker gave no reason")))
    if demo is not None and demo.root is not None:
        await asyncio.get_event_loop().run_in_executor(
            _RENDER_EXECUTOR, _write_objects, pathlib.Path(demo.root), demo.objects
        )


def _write_objects(root: pathlib.Path, objects: dict[str, dict[str, Any]]) -> None:
    """The tracked objects beside a saved demo; objects still tracking, or failed, are not kept."""
    done = {n: o for n, o in objects.items() if o.get("status") == "done"}
    f = pathlib.Path(root) / OBJECTS_FILE
    if not done:
        if f.exists():
            f.unlink()  # safe-destruct: our own sidecar; the operator removed the last object
        return
    arrays: dict[str, Any] = {
        "names": json.dumps([[n, int(o["frame"]), list(o["click"])] for n, o in done.items()])
    }
    for n, o in enumerate(done.values()):
        for key in ("deltas", "seen", "masks", "mask"):
            arrays[f"o{n}_{key}"] = o[key]
    np.savez_compressed(f, **arrays)


def _read_objects(root: pathlib.Path) -> dict[str, dict[str, Any]]:
    f = pathlib.Path(root) / OBJECTS_FILE
    if not f.exists():
        return {}
    z = np.load(f, allow_pickle=False)
    out: dict[str, dict[str, Any]] = {}
    for n, (name, frame, click) in enumerate(json.loads(str(z["names"]))):
        out[name] = {
            "frame": int(frame),
            "click": list(click),
            "status": "done",
            **{key: np.asarray(z[f"o{n}_{key}"]) for key in ("deltas", "seen", "masks", "mask")},
        }
    return out


def _draw_objects(bgr: np.ndarray, demo: _Demo, k: int) -> None:
    """Each tracked object's pose, outline and name on stream frame ``k``, in its colour."""
    import cv2

    try:
        t_bc = _t_base_cam()
    except HTTPException:
        t_bc = None  # without the arm's camera calibration there is no vertical to draw the pose against
    for n, (name, o) in enumerate(demo.objects.items()):
        if o.get("status") != "done":
            continue
        if t_bc is not None and demo.intr is not None:
            _draw_pose(bgr, demo, o, k, t_bc)
        if not o["seen"][k]:
            continue
        m = o["masks"][k].astype(np.uint8)
        if not m.any():
            continue
        m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colour = OBJECT_COLOURS[n % len(OBJECT_COLOURS)]
        cv2.drawContours(bgr, contours, -1, colour, 2, cv2.LINE_AA)
        x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cv2.putText(bgr, name, (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


POSE_AXIS_M = 0.04  # the drawn x and y axes' length
POSE_UP_M = 0.08  # up and true vertical, longer: seen from above, a tilt barely moves a short one


def _draw_pose(bgr: np.ndarray, demo: _Demo, o: dict[str, Any], k: int, t_bc: np.ndarray) -> None:
    """The object's tracked pose on stream frame ``k``: its axes at its tracked centre, built from the arm base's at
    the frame it was clicked on (x red, y green, up blue), with true vertical as a thin white line beside up. A
    hidden object's held pose is drawn dimmed."""
    import cv2

    if "centre0" not in o:  # the object's centre where it was clicked, from that frame's depth
        depth = _stream_frame(demo.recording, int(o["frame"]))[1]
        mask = cv2.resize(
            np.asarray(o["mask"], dtype=np.uint8),
            (depth.shape[1], depth.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        ys, xs = np.nonzero(mask & (depth > 0.05))
        if not len(xs):
            o["centre0"] = None
        else:
            i, z = demo.intr, depth[ys, xs]
            o["centre0"] = np.array(
                [((xs - i["cx"]) * z / i["fx"]).mean(), ((ys - i["cy"]) * z / i["fy"]).mean(), z.mean()]
            )
    c0 = o["centre0"]
    if c0 is None:
        return
    d = np.asarray(o["deltas"][k], dtype=float)
    c = d[:3, :3] @ c0 + d[:3, 3]
    axes_cam = t_bc[:3, :3].T  # the base's x, y, z as columns, in camera coordinates
    i = demo.intr

    def px(p: np.ndarray) -> tuple[int, int] | None:
        return (
            None
            if p[2] <= 1e-6
            else (int(round(i["fx"] * p[0] / p[2] + i["cx"])), int(round(i["fy"] * p[1] / p[2] + i["cy"])))
        )

    origin = px(c)
    if origin is None:
        return
    seen = bool(o["seen"][k])
    up = px(c + POSE_UP_M * axes_cam[:, 2])
    if up is not None:
        cv2.line(bgr, origin, up, (255, 255, 255), 1, cv2.LINE_AA)
    for axis, colour in zip(range(3), ((0, 0, 255), (0, 200, 0), (255, 80, 0)), strict=True):
        tip = px(c + (POSE_UP_M if axis == 2 else POSE_AXIS_M) * (d[:3, :3] @ axes_cam[:, axis]))
        if tip is not None:
            cv2.arrowedLine(
                bgr,
                origin,
                tip,
                colour if seen else tuple(v // 2 for v in colour),
                2,
                cv2.LINE_AA,
                tipLength=0.2,
            )


class KeypointsBody(BaseModel):
    keypoints: list[dict[str, Any]]


@router.post("/demo/keypoints")
async def demo_keypoints(body: KeypointsBody) -> dict:
    """Replace the demo's marks: pre-grasp points and the grasp's end, each a time in the demo; kept beside a saved demo."""
    with _state.lock:
        demo = _state.demo
    if demo is None:
        raise HTTPException(409, "record or load a demo first")
    try:
        kps = sorted(
            (
                {
                    "t": float(k["t"]),
                    "kind": str(k["kind"]),
                    **({"object": str(k["object"])} if k.get("object") else {}),
                }
                for k in body.keypoints
            ),
            key=lambda k: k["t"],
        )
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(422, f"a mark needs a time and a kind: {e}") from e
    problem = core.keypoints_problem(kps, float(demo.t[0]), float(demo.t[-1]))
    if problem:
        raise HTTPException(422, problem)
    named = {k.get("object", "") for k in kps}
    if len(named) > 1:
        raise HTTPException(422, "the pre-grasp and the grasp are for one object")
    obj = next(iter(named)) if named else ""
    if obj and demo.objects.get(obj, {}).get("status") != "done":
        raise HTTPException(422, f"{obj!r} is not a tracked object of this demo")
    if kps and not obj and not demo.taught:
        raise HTTPException(
            422, "nothing was taught before this demo: the marks follow an object clicked on its recording"
        )
    with _state.lock:
        demo.keypoints = kps
    if demo.root is not None:
        await asyncio.get_event_loop().run_in_executor(
            _RENDER_EXECUTOR, _write_keypoints, pathlib.Path(demo.root), kps
        )
    return _demo_info(demo)


@router.get("/demo/reach")
async def demo_reach() -> dict:
    """Can the arm do the marked pre-grasp and grasp on the object where it is now? The act's own judgement, without moving."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    from . import jog

    with _state.lock:
        demo, test = _state.demo, _state.test
    if demo is None:
        raise HTTPException(409, "record or load a demo first")
    if not _has_pregrasp(demo):
        raise HTTPException(409, "no pre-grasp marked yet")
    if test is None or not test.result.get("ok"):
        raise HTTPException(409, "the object is not found")
    kin, cur = jog.kinematics(), jog.current_tip_and_anchor()
    if kin is None or cur is None:
        raise HTTPException(409, "connect the arm first")
    with _state.lock:
        teach = _state.teach
    problem = _reference_motion(demo, teach)[1]
    if problem:
        raise HTTPException(409, problem)
    t_bc = _t_base_cam()
    q_now = np.array([float(cur[2][m]) for m in MOTOR_NAMES])
    plan = await asyncio.get_event_loop().run_in_executor(
        _ACT_EXECUTOR,
        _plan_act,
        demo,
        test.result["delta_cam"],
        t_bc,
        kin,
        q_now,
        jog.walk_limits(),
        jog.workspace_box(),
        1.0,
    )
    return {k: plan[k] for k in ("ok", "reason", "marks", "summary")}


def _has_pregrasp(demo: _Demo) -> bool:
    return any(k.get("kind") == "pregrasp" for k in demo.keypoints)


def _marks_object(demo: _Demo) -> str | None:
    """The designated object the demo's marks are for, or None for the object taught before the demo."""
    names = {k.get("object") or "" for k in demo.keypoints} - {""}
    return next(iter(names)) if names else None


def _reference_motion(demo: _Demo, teach: _Teach | None) -> tuple[np.ndarray | None, str]:
    """``(R, problem)``: the live track's motion times ``R`` is the object's motion from the demo to now.

    For the object taught before the demo, ``R`` undoes where the object was when the
    demo began. For a designated object it is the live view's registration against the
    demo's view of it, times the inverse of where the demo's track had the object when
    the first pre-grasp was shown, the last frame it was seen at or before then. ``R``
    is None, with the reason, when that object has not been found live yet.
    """
    obj = _marks_object(demo)
    if obj is None:
        if not demo.taught:
            return None, "the marks name no object: open Edit demo and save them for the object clicked there"
        return np.linalg.inv(demo.delta0), ""
    o = demo.objects.get(obj)
    if o is None or o.get("status") != "done" or demo.recording is None:
        return None, f"{obj!r} is not tracked in this demo"
    ref = (teach.keypoints.get("ref") if teach is not None else None) or {}
    if ref.get("object") != obj:
        return None, f"click {obj} in the camera view to find it"
    if not ref.get("ok") or ref.get("delta") is None:
        return None, ref.get("reason") or f"{obj} was not found in the live view"
    f = _pose_frame(demo)
    assert f is not None, "a done object with marks on it has a pose frame"
    return np.asarray(ref["delta"], dtype=float) @ np.linalg.inv(np.asarray(o["deltas"][f], dtype=float)), ""


def _pose_frame(demo: _Demo) -> int | None:
    """The stream frame the act reads the designated object's demo pose from: the last frame the object was seen at
    or before the first pre-grasp. None when the marks name no tracked object."""
    obj = _marks_object(demo)
    o = demo.objects.get(obj) if obj else None
    times = _stream_times(demo.recording) if demo.recording is not None else np.zeros(0)
    pre = [float(k["t"]) for k in demo.keypoints if k["kind"] == "pregrasp"]
    if o is None or o.get("status") != "done" or not len(times) or not pre:
        return None
    f0 = int(np.argmin(np.abs(times - (demo.t0 + min(pre)))))
    seen = np.flatnonzero(np.asarray(o["seen"])[: f0 + 1])
    return int(seen[-1]) if len(seen) else int(o["frame"])


def _delta_base(demo: _Demo, delta_cam: np.ndarray, t_bc: np.ndarray) -> np.ndarray:
    """The object's motion from the demo to now, in the base frame: the tracker's fit as it is.

    Pre: :func:`_reference_motion` has no problem for the current teach.
    """
    with _state.lock:
        teach = _state.teach
    ref, problem = _reference_motion(demo, teach)
    assert ref is not None, problem
    rel = np.asarray(delta_cam, dtype=float) @ ref
    return t_bc @ rel @ np.linalg.inv(t_bc)


def _certified_since(since: float | None) -> list[tuple[float, np.ndarray]]:
    """The tracker's certified motions, oldest first, processed after ``since`` (wall clock), or all of them."""
    with _state.lock:
        hist = list(_state.track.history)
    return [
        (w, np.asarray(d, dtype=float))
        for (w, ok, d) in hist
        if ok and d is not None and (since is None or w > since)
    ]


def _grasp_shift(demo: _Demo, a: np.ndarray, b: np.ndarray, t_bc: np.ndarray) -> tuple[float, float]:
    """How far two tracked motions of the object apart move the marked grasp: ``(metres, degrees)``.

    The largest distance between the grasp's fingertip positions carried by one and
    by the other, and the angle between their turns. Pre: the demo has a pre-grasp
    and a grasp end marked.
    """
    t0 = max(float(k["t"]) for k in demo.keypoints if k["kind"] == "pregrasp")
    t1 = next(float(k["t"]) for k in demo.keypoints if k["kind"] == "grasp_end")
    i0, i1 = int(np.argmin(np.abs(demo.t - t0))), int(np.argmin(np.abs(demo.t - t1)))
    da, db = _delta_base(demo, a, t_bc), _delta_base(demo, b, t_bc)
    pts = demo.tips[i0 : i1 + 1, :3, 3]
    pa = pts @ da[:3, :3].T + da[:3, 3]
    pb = pts @ db[:3, :3].T + db[:3, 3]
    return float(np.max(np.linalg.norm(pa - pb, axis=1))), core.pose_residual(da, db)[1]


def _still_decision(
    fresh: list[tuple[float, np.ndarray]],
    latest: np.ndarray | None,
    visible: bool,
    shift: Any,
) -> tuple[str, Any]:
    """At the last pre-grasp: go with a pose, or wait, from what the tracker sees.

    While the tracker sees the object, it holds still when two consecutive views
    since the arm arrived (``fresh``) move the grasp by less than the act's own reach
    tolerance, a difference the act could not carry out anyway; go with the newer.
    When the tracker no longer sees it, the gripper covering it, go with the latest
    view it had: the views just before a loss are the noisiest, so they are not
    asked to agree. ``shift(a, b)`` is :func:`_grasp_shift`. Post: ``("go", motion)``
    or ``("wait", None)``.
    """
    if len(fresh) >= 2:
        m, deg = shift(fresh[-2][1], fresh[-1][1])
        if m <= core.ACT_REACH_TOL_M and deg <= core.ACT_REACH_TOL_DEG:
            return "go", fresh[-1][1]
    if not visible and latest is not None:
        return "go", latest
    return "wait", None


def _act_preview(demo: _Demo, delta_cam: np.ndarray, t_bc: np.ndarray) -> np.ndarray | None:
    """The act's fingertip path from the first pre-grasp on, for the camera view; None until a pre-grasp is marked."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    from . import jog

    if not _has_pregrasp(demo):
        return None
    with _state.lock:
        teach = _state.teach
    if _reference_motion(demo, teach)[0] is None:
        return None
    gi = MOTOR_NAMES.index("gripper")
    delta_base = _delta_base(demo, delta_cam, t_bc)
    first = min(float(k["t"]) for k in demo.keypoints if k["kind"] == "pregrasp")
    i = int(np.argmin(np.abs(demo.t - first)))
    plan = core.plan_pregrasp_grasp(
        demo.keypoints,
        demo.t,
        demo.tips,
        demo.q_cmd[:, gi],
        demo.q_obs,
        delta_base,
        delta_base @ demo.tips[i],
        float(demo.q_cmd[i, gi]),
        jog.MAX_LINEAR_M_S,
        jog.MAX_ANGULAR_RAD_S,
        jog.GRIP_UNITS_S,
        1.0,
        jog.HZ,
    )
    return plan["poses"]


def _plan_act(
    demo: _Demo,
    delta_cam: np.ndarray,
    t_bc: np.ndarray,
    kin: Any,
    q_now: np.ndarray,
    limits: tuple[float, float],
    box: tuple[tuple[float, float, float], tuple[float, float, float]],
    speed: float,
    skip: int = 0,
) -> dict[str, Any]:
    """What the act will do on the object where it is now, judged before the arm moves.

    Pre: the demo has a pre-grasp mark. From the arm's present joints ``q_now``:
    straight lines through the pre-grasp points, then, when a grasp end is marked,
    the grasp exactly as recorded, all carried by the object's motion; every sample
    solved by IK from the one before. Refuses, naming the reason, when a pre-grasp or
    a grasp sample is out of reach, when a sample leaves the workspace or goes lower
    than the table floor (or than the demo itself went at that sample), or when the
    arm would jump between two samples.
    Post: ``ok``, ``reason``, ``times`` (N,), ``q`` (N, J), ``stage`` (N,), ``marks``
    (label, t, residual_mm, ok) and ``summary``, JSON-safe apart from the arrays.
    """
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    from . import jog

    gi = MOTOR_NAMES.index("gripper")
    q_now = np.asarray(q_now, dtype=float)
    delta_base = _delta_base(demo, delta_cam, t_bc)
    plan = core.plan_pregrasp_grasp(
        demo.keypoints,
        demo.t,
        demo.tips,
        demo.q_cmd[:, gi],
        demo.q_obs,
        delta_base,
        kin.forward_kinematics(q_now),
        float(q_now[gi]),
        limits[0],
        limits[1],
        jog.GRIP_UNITS_S,
        speed,
        jog.HZ,
        skip,
    )
    sol = core.solve_plan_joints(kin, plan["poses"], plan["grips"], plan["hints"], q_now, gi)
    fine = (sol["residual_m"] <= core.ACT_REACH_TOL_M) & (sol["residual_deg"] <= core.ACT_REACH_TOL_DEG)
    pos = plan["poses"][:, :3, 3]
    lo, hi = np.asarray(box[0], dtype=float), np.asarray(box[1], dtype=float)
    floor = np.minimum(
        np.minimum(lo[2], plan["floor_ref"]), pos[0, 2]
    )  # never lower than the arm already stands
    low = pos[:, 2] < floor - core.ACT_REACH_TOL_M
    outside = np.any(pos[:, :2] < lo[:2] - core.ACT_REACH_TOL_M, axis=1) | np.any(
        pos > hi + core.ACT_REACH_TOL_M, axis=1
    )
    marks = []
    pre = sorted(float(k["t"]) for k in demo.keypoints if k["kind"] == "pregrasp")[skip:]
    for n, (tk, idx) in enumerate(zip(pre, plan["arrive"], strict=True), start=skip + 1):
        marks.append(
            {
                "label": f"pre-grasp {n}",
                "t": tk,
                "residual_mm": float(sol["residual_m"][idx] * 1000.0),
                "ok": bool(fine[idx]),
            }
        )
    if plan["grasp"] is not None:
        a, b = plan["grasp"]
        marks.append(
            {
                "label": "grasp",
                "t": next(float(k["t"]) for k in demo.keypoints if k["kind"] == "grasp_end"),
                "residual_mm": float(sol["residual_m"][a : b + 1].max() * 1000.0),
                "ok": bool(fine[a : b + 1].all()),
            }
        )
    reason = ""
    bad = [m for m in marks if not m["ok"]]
    far = np.flatnonzero(~fine)
    jumps = np.flatnonzero(sol["step_deg"] > core.ACT_MAX_JOINT_STEP_DEG)
    if bad:
        reason = (
            f"{bad[0]['label']} is out of reach as the object lies now ({bad[0]['residual_mm']:.0f} mm short)"
        )
    elif len(far):
        n = int(far[0])
        reason = f"the straight line to {plan['stage'][n]} leaves the arm's reach ({sol['residual_m'][n] * 1000.0:.0f} mm short)"
    elif np.any(low):
        n = int(np.argmax(np.where(low, floor - pos[:, 2], -np.inf)))
        reason = f"{plan['stage'][n]} would go {(floor[n] - pos[n, 2]) * 1000.0:.0f} mm below the table"
    elif np.any(outside):
        n = int(np.flatnonzero(outside)[0])
        reason = f"{plan['stage'][n]} leaves the arm's workspace"
    elif len(jumps):
        n = int(jumps[0])
        reason = f"the arm would jump {sol['step_deg'][n]:.0f} deg during {plan['stage'][n]}"
    return {
        "ok": not reason,
        "reason": reason,
        "times": plan["times"],
        "q": sol["q"],
        "stage": plan["stage"],
        "marks": marks,
        "summary": {
            "samples": len(plan["times"]),
            "seconds": float(plan["times"][-1]),
            "worst_residual_mm": float(sol["residual_m"].max() * 1000.0),
            "worst_step_deg": float(sol["step_deg"].max()),
        },
    }


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
    speed: float = (
        1.0  # scales the straight lines (from the jog's walk speed) and the grasp (from the demo's clock)
    )


async def _act_task(speed: float) -> None:
    """The act: follow the object to each pre-grasp, wait for it to hold still, then replay the grasp 1:1.

    An object designated in the demo is found afresh first, where it was last seen, so every act starts
    from the demo's view of it rather than from a track kept since an earlier find.
    The whole act is planned and judged from the arm's present joints before
    anything moves. The approach runs on the jog's walk, re-aimed at every new
    tracker view, so the straight lines bend toward an object that is moved. At the
    last pre-grasp the arm waits until the object holds still, or until the gripper
    covers it, and the grasp is planned from there and streamed as joint targets.
    Without tracking the act runs from the one view it started with.
    """
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    from . import jog

    act = _state.act

    def fail(reason: str) -> None:
        act.ok, act.reason, act.step = False, reason, "aborted"

    def q_dict(q: np.ndarray) -> dict[str, float]:
        return {m: float(q[k]) for k, m in enumerate(MOTOR_NAMES)}

    def interrupted() -> str:
        st = jog.current_status()
        if act.stop_requested:
            return "stopped"
        if not st.get("connected"):
            return "the arm went away"
        if st.get("halted"):
            return f"arm frozen: {st.get('reason')}"
        return ""

    moved = streaming = False
    limits_before: tuple[float, float] | None = None
    run: _Run | None = None
    run_dir: str | None = None
    try:
        with _state.lock:
            demo, test, teach, tracking = _state.demo, _state.test, _state.teach, _state.track.on
        if demo is None:
            fail("record or load a demo first")
            return
        if not _has_pregrasp(demo):
            fail("mark a pre-grasp first")
            return
        obj = _marks_object(demo)
        if obj is not None:
            act.step = f"finding {obj}"
            why = await _find_afresh(obj, lambda: act.stop_requested)
            if why:
                fail(why)
                return
            with _state.lock:
                test, teach, tracking = _state.test, _state.teach, _state.track.on
        if test is None or not test.result.get("ok"):
            fail("find the object first")
            return
        if teach is None:
            fail("teach the object first")
            return
        try:
            t_bc = _t_base_cam()
        except HTTPException as e:
            fail(e.detail)
            return
        kin, cur = jog.kinematics(), jog.current_tip_and_anchor()
        if kin is None or cur is None:
            fail("connect the arm first")
            return
        if tracking:
            with _state.lock:
                state = (_state.track.last or {}).get("state")
            if state != "tracking":
                fail(
                    f"the tracker does not see the object ({state or 'no frame yet'}): "
                    "load the demo again with the object where it was taught"
                )
                return
        problem = _reference_motion(demo, teach)[1]
        if problem:
            fail(problem)
            return
        gi = MOTOR_NAMES.index("gripper")
        delta = np.asarray(test.result["delta_cam"], dtype=float)
        seen_at = max((w for w, _ in _certified_since(None)), default=0.0)
        limits_before = jog.walk_limits()
        act.step = "planning"
        plan = await asyncio.get_event_loop().run_in_executor(
            _ACT_EXECUTOR,
            _plan_act,
            demo,
            delta,
            t_bc,
            kin,
            np.array([float(cur[2][m]) for m in MOTOR_NAMES]),
            limits_before,
            jog.workspace_box(),
            speed,
        )
        act.plan = {k: plan[k] for k in ("ok", "reason", "marks", "summary")}
        if not plan["ok"]:
            fail(plan["reason"])
            return
        pre = sorted(float(k["t"]) for k in demo.keypoints if k["kind"] == "pregrasp")
        idx = [int(np.argmin(np.abs(demo.t - tk))) for tk in pre]
        jog.set_walk_limits(limits_before[0] * speed, limits_before[1] * speed)
        moved = True
        run = _begin_run(demo, speed, delta, t_bc, plan)

        def follow() -> np.ndarray:
            """The newest certified view of the object, or the last one used."""
            nonlocal seen_at, delta
            if tracking:
                newer = _certified_since(seen_at)
                if newer:
                    seen_at, delta = newer[-1]
            return delta

        for n, i in enumerate(idx, start=1):
            g = float(demo.q_cmd[i, gi])
            g_now = jog.current_gripper()
            jog.set_gripper(g)
            act.step = f"pre-grasp {n}: setting the gripper"
            await asyncio.sleep(abs(g - (g if g_now is None else g_now)) / jog.GRIP_UNITS_S + ACT_TICK_S)
            act.step = f"pre-grasp {n}"
            t0 = time.monotonic()
            while True:
                why = interrupted()
                if why:
                    fail(why)
                    return
                target = _delta_base(demo, follow(), t_bc) @ demo.tips[i]
                jog.set_target_pose(target)
                run.target(act.step, pose=target)
                cur = jog.current_tip_and_anchor()
                if cur is not None and not jog.current_status().get("holding"):
                    e_m, e_deg = core.pose_residual(cur[0], target)
                    if e_m <= ACT_ARRIVE_M and e_deg <= core.ACT_REACH_TOL_DEG:
                        break
                if time.monotonic() - t0 > ACT_STEP_TIMEOUT_S:
                    fail(f"pre-grasp {n}: not there after {ACT_STEP_TIMEOUT_S:.0f} s")
                    return
                await asyncio.sleep(ACT_TICK_S)
        if not any(k["kind"] == "grasp_end" for k in demo.keypoints):
            act.step, act.ok = "done", True
            return

        if tracking:
            act.step = "waiting for the object to hold still"
            arrived, t0 = time.time(), time.monotonic()
            while True:
                why = interrupted()
                if why:
                    fail(why)
                    return
                with _state.lock:
                    visible = (_state.track.last or {}).get("state") == "tracking"
                decision, value = _still_decision(
                    _certified_since(arrived), follow(), visible, lambda a, b: _grasp_shift(demo, a, b, t_bc)
                )
                if decision == "go":
                    delta = value
                    break
                hold = _delta_base(demo, follow(), t_bc) @ demo.tips[idx[-1]]
                jog.set_target_pose(hold)  # stay with the object
                run.target(act.step, pose=hold)
                if time.monotonic() - t0 > ACT_STEP_TIMEOUT_S:
                    fail(f"the object did not hold still for {ACT_STEP_TIMEOUT_S:.0f} s")
                    return
                await asyncio.sleep(ACT_TICK_S)

        act.step = "planning the grasp"
        cur = jog.current_tip_and_anchor()
        if cur is None:
            fail("the arm went away")
            return
        grasp = await asyncio.get_event_loop().run_in_executor(
            _ACT_EXECUTOR,
            _plan_act,
            demo,
            delta,
            t_bc,
            kin,
            np.array([float(cur[2][m]) for m in MOTOR_NAMES]),
            limits_before,
            jog.workspace_box(),
            speed,
            len(pre) - 1,
        )
        act.plan = {k: grasp[k] for k in ("ok", "reason", "marks", "summary")}
        if not grasp["ok"]:
            fail(grasp["reason"])
            return
        q, times, stage = grasp["q"], grasp["times"], grasp["stage"]
        try:
            await jog.joints_start(q_dict(q[0]))
        except RuntimeError as e:
            fail(str(e))
            return
        streaming = True
        n = len(q)
        t_start = time.monotonic()
        for i in range(n):
            due = t_start + float(times[i])
            while time.monotonic() < due:
                await asyncio.sleep(min(ACT_TICK_S, max(0.0, due - time.monotonic())))
            why = interrupted()
            if why:
                fail(why)
                return
            try:
                jog.set_target_joints(q_dict(q[i]))
            except RuntimeError as e:
                fail(str(e))
                return
            act.step = stage[i]
            run.target(act.step, joints=q[i])
            act.progress = (i + 1) / n
        act.step = "settling"
        t0 = time.monotonic()
        while True:
            await asyncio.sleep(ACT_TICK_S)
            cur = jog.current_tip_and_anchor()
            why = interrupted()
            if why or cur is None:
                fail(why or "the arm went away")
                return
            lag = max(
                abs(float(cur[2][m]) - float(q[-1][k])) for k, m in enumerate(MOTOR_NAMES) if m != "gripper"
            )
            if lag <= ACT_SETTLE_DEG:
                break
            if time.monotonic() - t0 > ACT_STEP_TIMEOUT_S:
                fail(f"the end: not there after {ACT_STEP_TIMEOUT_S:.0f} s ({lag:.0f} deg off)")
                return
        act.step, act.ok = "done", True
    except Exception as e:  # the arm holds its last target; the operator sees why
        logger.exception("act failed")
        fail(f"act error: {e}")
    finally:
        if limits_before is not None:
            with contextlib.suppress(Exception):
                jog.set_walk_limits(*limits_before)
        if streaming:
            # Back to the Cartesian walk where the arm stopped, still commanding the grasp's closing.
            with contextlib.suppress(Exception):
                await jog.joints_stop()
        if run is not None:
            with _state.lock:
                _state.run = None
            samples: list[dict[str, Any]] = []
            if run.arm_recording:
                with contextlib.suppress(Exception):
                    samples = jog.stop_record()
            result = {"ok": act.ok, "step": act.step, "reason": act.reason, "progress": act.progress}
            try:
                run_dir = await asyncio.get_event_loop().run_in_executor(
                    _RUN_EXECUTOR, _finish_run, run, samples, jog.fk_tip, result
                )
            except Exception:  # the act's outcome stands without its recording
                logger.exception("writing the act's recording failed")
        if moved:
            with contextlib.suppress(Exception):
                _record_trial(run_dir)
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
        if not _has_pregrasp(demo):
            raise HTTPException(409, "mark a pre-grasp first (Edit demo)")
        if test is None or not test.result.get("ok"):
            raise HTTPException(409, "find the object first")
        if (test.result.get("camera_check") or {}).get("moved"):
            raise HTTPException(409, "the camera or the tray moved since the calibration; recalibrate first")
        _state.track.follow = False  # the act owns the target now
        act.plan = None
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
