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
import functools
import io
import json
import logging
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

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
class _GroupsView:
    """The point groups' live view (benchmarks/group_live.py --live --record), started and finished from the
    Approach tab's Groups panel. It holds the camera's recording and TAPIR on the GPU while it runs, so it is
    not left running between sessions."""

    proc: subprocess.Popen | None = None
    log: list[str] = field(default_factory=list)
    last: dict[str, Any] = field(
        default_factory=dict
    )  # what the view said of its recording as it was finished

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
class _TargetTrack:
    """The live track of the object a place goes onto, from its last find: each trusted frame moves that find, so
    whatever reads the find (the live view's outline, the act's aim) follows the object. It is one more object of the
    picked object's Point2Pose session: its share comes back with every tracked frame of the picked one."""

    obj: str | None = None
    anchor: np.ndarray | None = None  # the find's motion from the demo's view to the frame the track began on
    n_points: int = 0  # the tracks it began with, for the trust test
    last: dict[str, Any] = field(default_factory=dict)  # the newest frame's readout


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
    landing: str = (
        "exact"  # the turns the place may land at on its object (core.LANDINGS); kept with the marks
    )
    video: list[bytes] | None = None  # the saved video decoded once for the editor, one JPEG per sample
    taught: bool = False  # recorded with an object taught first; unnamed marks follow that object
    holds: dict[str, Any] = field(
        default_factory=dict
    )  # how the demo held the picked object, by the marks it was measured for
    view_points: dict[str, Any] = field(
        default_factory=dict
    )  # each object's surface on its clicked frame, read once


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
    place: dict[str, Any] | None = (
        None  # what the last act measured for its place: the grasp check, the holds
    )
    inject: dict[str, Any] | None = None  # an error the operator asked this act to inject, for testing
    find_error: np.ndarray | None = (
        None  # while it runs: the injected error in the object's found pose, base frame
    )
    correct_hold: bool = True  # False replays the place uncorrected for the hold, as a baseline


@dataclass
class _StreamRecorder:
    """The camera's colour and depth written to disk while a demo is recorded, for designating objects afterwards."""

    out: pathlib.Path
    stop: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    n: int = 0
    error: str = ""


@dataclass
class _FrameShare:
    """The camera's frames copied into a shared-memory ring as they arrive (lerobot.showservo.frame_ring), so a
    reader in another process takes the newest one with no disk, encoder or wait for a file to be complete."""

    ring: Any
    stop: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    n: int = 0
    error: str | None = None


def _share_frames(camera: Any, share: _FrameShare) -> None:
    """Copy every new camera frame into the ring until stopped. Reads go through the camera executor like every
    other reader; the time stamped on a frame is when its read returned."""
    from . import showservo

    try:
        while not share.stop.is_set():
            try:
                rgb, depth_mm = showservo._EXECUTOR.submit(camera.read_color_and_aligned_depth).result()
            except TimeoutError:
                continue  # another reader took that frame
            share.ring.write(rgb, depth_mm.astype(np.uint16), time.time())
            share.n += 1
    except Exception as e:  # the stop endpoint reports it
        logger.exception("camera frame sharing failed")
        share.error = str(e)


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


# The place object's track is followed only while at least this share of its tracked points is seen: under the
# gripper its points drift onto the arm while the frame is still trusted. Replayed on five recorded acts, 0.97 kept
# the place within 1.2-3.9 mm of where the cube lay (one act 9 mm for a frame, then 3) against 4-163 mm without it.
# A run-time option (the page's slider) while borrowed points are not there yet.
TRUST_SHARE_DEFAULT = 0.97


@dataclass
class _State:
    lock: threading.Lock = field(default_factory=threading.Lock)
    teach: _Teach | None = None
    test: _Test | None = None
    worker: _Worker = field(default_factory=_Worker)
    groups: _GroupsView = field(default_factory=_GroupsView)
    teach_job: str | None = None  # a features teach awaiting its result
    find_job: str | None = None
    flat: bool = (
        False  # opt-in resting prior: the fit's motion as a turn about the surface the object rests on
    )
    trust_share: float = TRUST_SHARE_DEFAULT  # the place object's track is followed only with this share seen
    track: _Track = field(default_factory=_Track)
    act: _Act = field(default_factory=_Act)
    demo: _Demo | None = None  # the demo recorded or loaded last
    recording: dict[str, Any] | None = None  # while a demo is being recorded: its start and the frames so far
    stream: _StreamRecorder | None = None  # the camera stream being written while a demo is recorded
    run: _Run | None = None  # the act being recorded: its tracker frames, its targets, the arm
    camera_recording: _StreamRecorder | None = (
        None  # the camera recorded on its own, to replay the tracker over
    )
    camera_share: _FrameShare | None = (
        None  # the camera's frames in shared memory, for the point groups' view
    )
    located: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )  # objects found against the demo's view of them, by name: the one a place goes onto, moved by its track
    target: _TargetTrack = field(default_factory=_TargetTrack)
    reach_landing: dict[str, Any] = field(
        default_factory=dict
    )  # the reach preview's landing turn and what it was searched for: one search per find of the target


_state = _State()
_RENDER_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pregrasp-render")
_ACT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pregrasp-act")
_SEEN_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pregrasp-seen")
# An act's recording goes to disk on its own thread, in order; the tracker never waits for it.
_RUN_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pregrasp-run")
TRACK_JOB_TIMEOUT_S = 5.0
TRACK_FRAME_RETRY_S = 0.5  # how soon the live track asks the camera again after a frame it could not read
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
    flat: bool | None = None
    trust_share: float | None = Field(None, ge=0.0, le=1.0)


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


_PAGE = pathlib.Path(__file__).resolve().parents[1] / "static" / "index.html"
_page_seen: tuple[float, str | None] = (-1.0, None)


def _page_version() -> str | None:
    """The version of the Approach tab's script the GUI page loads, as index.html names it (``showservo.js?v=N``);
    read again only when the file changes. A tab open since before a change compares itself against this."""
    global _page_seen
    try:
        mtime = _PAGE.stat().st_mtime
    except OSError:
        return None
    if mtime != _page_seen[0]:
        m = re.search(r"showservo\.js\?v=(\d+)", _PAGE.read_text())
        _page_seen = (mtime, m.group(1) if m else None)
    return _page_seen[1]


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
        "page_version": _page_version(),
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
        "trust_share": s.trust_share,
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
            "place": s.act.place,
        },
        "located": {}
        if s.demo is None
        else {
            name: _located_info(f) for name in list(s.located) if (f := _located(s.demo, name)) is not None
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
    """A fresh colour frame: the camera view while the camera is live and nothing is tracked, and what an object box
    is drawn on. The page polls it, so the frame is read and encoded on the camera's executor in one hop: an encode
    on the event loop would stall the act's ticks for as long as it takes."""
    import cv2

    from . import showservo

    camera = showservo.live_camera()
    if camera is None:
        raise HTTPException(409, "start a live camera session in the Servo tab first")

    def grab_jpeg() -> bytes:
        rgb, _depth, _intr = _grab(camera)
        return _jpeg(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    try:
        data = await asyncio.get_event_loop().run_in_executor(showservo._EXECUTOR, grab_jpeg)
    except Exception as e:
        raise HTTPException(500, f"camera read failed: {e}") from e
    return Response(content=data, media_type="image/jpeg")


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


async def _start_teach(
    click: list[int] | None, concept: str, ref_object: str, more: dict[str, np.ndarray] | None = None
) -> str:
    """Queue a features teach of the object at ``click`` on the live frame; with ``ref_object``, also its find
    against the demo's view of that object. The live Point2Pose session starts over with it, keeping the demo's other
    objects it follows, and with ``more`` (name -> mask, found just before on the live view) joining from the start:
    one restart for both. Post: the job's id, the teach pending and the live pose cleared. Raises HTTPException when
    the worker is not running or ``ref_object`` is not tracked in the demo."""
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
            "ref_symmetry": int(ref.get("symmetry") or 1),
            "scene": _session_names(demo, but=ref_object),
            "more": list(more or ()),
        }
        job.arrays = {
            "ref_mask": np.asarray(ref["mask"], dtype=bool),
            **{f"more_mask_{i}": np.asarray(m, dtype=bool) for i, m in enumerate((more or {}).values())},
        }
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


async def _find_afresh(
    obj: str, stopped: Callable[[], bool], more: dict[str, np.ndarray] | None = None, need_track: bool = True
) -> str:
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
        job_id = await _start_teach(click, obj, obj, more)
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
    if core.find_strength(ref.get("inliers"), ref.get("card_points"))[0] is False:
        return (
            f"a weak find: {obj} matched {ref['inliers']} of the demo view's {ref['card_points']} points; "
            "turn it closer to how it lay in the demo, then press Act"
        )
    if not need_track:  # the find's own view is the object's pose until the restarted track sees it
        return ""
    found, t0 = time.time(), time.monotonic()
    while not _certified_since(found):
        if stopped():
            return "stopped"
        if time.monotonic() - t0 > ACT_STEP_TIMEOUT_S:
            return f"the tracker has not seen {obj} since finding it"
        await asyncio.sleep(ACT_TICK_S)
    return ""


async def _locate(
    obj: str,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intr: dict[str, float],
    click: list[int],
    stopped: Callable[[], bool],
    track: bool = False,
) -> dict[str, Any]:
    """Find the demo's object ``obj`` under ``click`` on this frame against the demo's view of it, leaving the live
    track alone: the object a place goes onto, and the held object in the gripper.

    With ``track``, the object joins the live Point2Pose session from this frame (:func:`_store_located` follows it).
    Post: ``object``, ``ok``, ``delta`` (the motion from the demo's view, camera frame) or None, ``inliers``,
    ``card_points``, ``turn_deg``, ``reason``, ``mask`` (what SAM3 cut out at the click, or None), ``view`` (the demo
    and the frame of the view it was matched against), ``at``, ``answered`` (the worker answered) and, with
    ``track``, ``n_points`` and ``tracking``; ``ok`` False with the reason when the object is not tracked in the demo,
    the worker is off or slow, or ``stopped()``.
    """
    with _state.lock:
        demo = _state.demo
    o = None if demo is None else demo.objects.get(obj)
    out: dict[str, Any] = {"object": obj, "ok": False, "delta": None, "mask": None, "at": time.time()}
    if o is None or o.get("status") != "done" or demo.recording is None:
        return {**out, "reason": f"{obj!r} is not a tracked object of the current demo"}
    out["view"] = [demo.name, int(o["frame"])]  # the demo view its motion is from
    if not _state.worker.running:
        return {**out, "reason": "start the worker first"}
    job = _queue_job("locate", obj, rgb, depth_m, intr, click=[int(click[0]), int(click[1])])
    job.extra = {
        "ref_recording": demo.recording,
        "ref_frame": int(o["frame"]),
        "ref_object": obj,
        "ref_symmetry": int(o.get("symmetry") or 1),
    }
    if track:  # into the live session, beside the objects it already follows
        job.extra.update(track=True, scene=_session_names(demo, but=obj))
    job.arrays = {"ref_mask": np.asarray(o["mask"], dtype=bool)}
    t0 = time.monotonic()
    while job.result is None:
        if stopped():
            return {**out, "reason": "stopped"}
        if time.monotonic() - t0 > ACT_STEP_TIMEOUT_S:
            return {**out, "reason": f"finding {obj} took over {ACT_STEP_TIMEOUT_S:.0f} s"}
        await asyncio.sleep(ACT_TICK_S)
    r = job.result
    found = bool(r.get("ok") and r.get("ref_ok") and r.get("ref_delta") is not None)
    return {
        **out,
        "ok": found,
        "delta": np.asarray(r["ref_delta"], dtype=float) if found else None,
        "inliers": r.get("ref_inliers"),
        "card_points": r.get("ref_card_points"),
        "turn_deg": r.get("ref_turn_deg"),
        "reason": "" if found else (r.get("ref_reason") or r.get("reason") or f"{obj} was not found"),
        "mask": None if r.get("mask") is None else np.asarray(r["mask"]).astype(bool),
        "answered": True,  # the worker's own answer, found or not: a measurement
        "n_points": r.get("n_points"),
        "tracking": bool(r.get("tracking")),
    }


LAST_SEEN_FILE = (
    "last_seen.json"  # beside a saved demo: where each of its objects was last seen live, as a click
)
LAST_SEEN_EVERY_S = 2.0  # a tracked object's last-seen point is written at most this often
REFIND_WAIT_S = (
    120.0  # after a load, how long the finds at the last-seen points wait for the worker and the camera
)
_last_seen_written: dict[str, float] = {}
_refinds: set[asyncio.Task] = set()  # the finds a load started, kept until they end


def _write_seen(path: pathlib.Path, obj: str, click: list[int]) -> None:
    """Merge ``obj``'s last-seen click into ``path``; a write that fails leaves the old one."""
    try:
        seen = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        seen = {}
    seen[obj] = {"click": click, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with contextlib.suppress(OSError):
        path.write_text(json.dumps(seen))


def _read_seen(path: pathlib.Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        return {}


def _remember_seen(obj: str, mask: np.ndarray | None, shape: tuple[int, ...]) -> None:
    """Keep where ``obj`` of the current saved demo was last seen live, ``mask`` on a frame of ``shape``, as the click a
    find there would use, so a later load finds it there without one. At most every :data:`LAST_SEEN_EVERY_S` per
    object, written off the event loop."""
    with _state.lock:
        demo = _state.demo
    if demo is None or demo.root is None or mask is None:
        return
    now = time.monotonic()
    if now - _last_seen_written.get(obj, -1e9) < LAST_SEEN_EVERY_S:
        return
    click = _deepest_pixel(mask, shape)
    if click is None:
        return
    _last_seen_written[obj] = now
    _SEEN_EXECUTOR.submit(_write_seen, pathlib.Path(demo.root) / LAST_SEEN_FILE, obj, click)


def _start_refind(demo: _Demo) -> None:
    """Run :func:`_refind_last_seen` for a demo just loaded, kept until it ends."""
    task = asyncio.create_task(_refind_last_seen(demo))
    _refinds.add(task)
    task.add_done_callback(_refinds.discard)


async def _refind_last_seen(demo: _Demo) -> None:
    """After ``demo`` is loaded: once the worker and the camera are up, find its objects again where they were last
    seen, as the operator's clicks there would: the one a place goes onto first, then the picked one by a teach whose
    Point2Pose session starts with both, one start for the two. One not found there waits for a click, as before."""
    from . import showservo

    if demo.root is None:
        return
    seen = await asyncio.get_running_loop().run_in_executor(
        _SEEN_EXECUTOR, _read_seen, pathlib.Path(demo.root) / LAST_SEEN_FILE
    )
    picked, onto = _marks_object(demo), _place_object(demo)
    if not ((picked in seen) or (onto in seen)):
        return
    t0 = time.monotonic()
    while True:  # the worker and the camera are often started after the load, and the load's own teach first
        with _state.lock:
            current, running, busy = _state.demo is demo, _state.worker.running, _state.teach_job is not None
            ready = any("worker ready" in line for line in _state.worker.log)
        if not current or time.monotonic() - t0 > REFIND_WAIT_S:
            return
        if running and ready and not busy and showservo.live_camera() is not None:
            break
        await asyncio.sleep(0.5)

    def gone() -> bool:
        return _state.demo is not demo

    with _state.lock:
        teach, teach_job = _state.teach, _state.teach_job
    found_picked = teach is not None and (teach.keypoints.get("ref") or {}).get("object") == picked
    teach_picked = picked in seen and not found_picked and teach_job is None
    found = None
    if onto in seen and _located(demo, onto) is None:  # first, so the picked object's session starts with it
        rgb, depth_m, intr = await _frame()
        found = await _locate(
            onto, rgb, depth_m, intr, list(seen[onto]["click"]), gone, track=not teach_picked
        )
        if gone():
            return
        _store_located(onto, found)
    if teach_picked:
        more = {onto: found["mask"]} if found and found["ok"] and found.get("mask") is not None else None
        try:
            job_id = await _start_teach(list(seen[picked]["click"]), picked, picked, more)
        except HTTPException:
            return
        t0 = time.monotonic()
        while _state.teach_job == job_id and not gone() and time.monotonic() - t0 < ACT_STEP_TIMEOUT_S:
            await asyncio.sleep(ACT_TICK_S)
        with _state.lock:
            teach = _state.teach
        if more and teach is not None and (teach.keypoints.get("ref") or {}).get("ok") and not gone():
            _store_located(onto, {**found, "tracking": True})


def _session_names(demo: _Demo, but: str | None = None) -> list[str]:
    """The demo's objects the live session keeps following, but ``but``: the picked one and the one a place goes onto."""
    return [
        n for n in dict.fromkeys((_marks_object(demo), _place_object(demo))) if n is not None and n != but
    ]


def _store_located(obj: str, found: dict[str, Any]) -> None:
    """Keep ``found`` as the last find of ``obj``; when it joined the live session, follow it from now on
    (:func:`_apply_others`), and stop following an earlier find. Called on the event loop."""
    if found.get("ok") and found.get("mask") is not None:
        _remember_seen(obj, found["mask"], found["mask"].shape)
    with _state.lock:
        _state.located[obj] = found
        _state.target = (
            _TargetTrack(obj=obj, anchor=np.asarray(found["delta"], dtype=float))
            if found.get("ok") and found.get("tracking")
            else _TargetTrack()
        )


def _apply_others(r: dict[str, Any], shape: tuple[int, ...]) -> None:
    """The other objects of a tracked frame's session (``others``, ``other_delta_i``, ``other_mask_i``): the place
    object's trusted share moves its last find (its motion since its find, times the find's motion from the demo's
    view); anything else leaves the find where it was. Trusted when Point2Pose has it, enough of the tracks it began
    with are seen (:func:`core.find_trusted`) and at least ``trust_share`` of its tracks now are: covered in part, its
    points drift onto what covers it while the frame still looks trusted, so it stays where it was last seen."""
    with _state.lock:
        target = _state.target
        found = _state.located.get(target.obj) if target.obj is not None else None
    if target.obj is None or found is None or target.anchor is None:
        return
    i = next((k for k, o in enumerate(r.get("others") or ()) if o.get("name") == target.obj), None)
    if i is None:
        target.last = {"state": "not in the session"}
        return
    share = r["others"][i]
    target.n_points = target.n_points or int(share.get("n_tracks") or 0)
    trusted, why = (
        (False, "lost")
        if share.get("lost")
        else core.find_trusted(int(share.get("n_visible") or 0), target.n_points)
    )
    seen = int(share.get("n_visible") or 0) / max(1, int(share.get("n_tracks") or 0))
    with _state.lock:
        run, need = _state.run, _state.trust_share
    if trusted and seen < need:
        trusted, why = False, f"only {seen:.0%} of its points are seen; its track is followed from {need:.0%}"
    if (
        run is not None and f"other_delta_{i}" in r
    ):  # the act's record: what the place object's track said, each frame
        d = np.asarray(r[f"other_delta_{i}"], dtype=float) @ target.anchor
        run.meta.setdefault("target_track", []).append(
            {
                "t": time.time(),
                "lost": bool(share.get("lost")),
                "n_visible": share.get("n_visible"),
                "n_tracks": share.get("n_tracks"),
                "trusted": bool(trusted),
                "seen": round(seen, 3),
                "trust_share": need,
                "delta": d.round(
                    5
                ).tolist(),  # the whole motion: its translation alone swings with a small turn
            }
        )
    if not (share.get("ok") and trusted and f"other_delta_{i}" in r):
        target.last = {"state": "untrusted" if share.get("ok") else "lost", "reason": why}
        return
    mask = None if r.get(f"other_mask_{i}") is None else np.asarray(r[f"other_mask_{i}"]).astype(bool)
    with _state.lock:
        found["delta"] = np.asarray(r[f"other_delta_{i}"], dtype=float) @ target.anchor
        found["tracked_at"] = time.time()
        if mask is not None:  # where the act's own find clicks it next
            found["mask"] = mask
    target.last = {"state": "tracking", "n_visible": share.get("n_visible")}
    if mask is not None:
        _remember_seen(target.obj, mask, shape)


def _located(demo: _Demo, obj: str) -> dict[str, Any] | None:
    """The last locate of ``obj`` against this demo's view of it, or None. One made against another demo, or against
    an earlier designation of the object, gives a motion from a different view."""
    o = demo.objects.get(obj)
    with _state.lock:
        found = _state.located.get(obj)
    if found is None or o is None or found.get("view") != [demo.name, int(o["frame"])]:
        return None
    return found


def _located_info(found: dict[str, Any]) -> dict[str, Any]:
    """A locate as the page shows it: like a find, without its arrays; ``track`` is its track's newest state."""
    with _state.lock:
        target = _state.target
    return {
        **{k: found.get(k) for k in ("object", "ok", "inliers", "card_points", "turn_deg", "reason", "at")},
        "track": target.last.get("state") if target.obj == found.get("object") else None,
        **dict(
            zip(
                ("strong", "share"),
                core.find_strength(found.get("inliers"), found.get("card_points")),
                strict=True,
            )
        ),
    }


def _weak(found: dict[str, Any]) -> str:
    """Why a find of the demo's view is too weak to act on, or "" when it is not weak."""
    if core.find_strength(found.get("inliers"), found.get("card_points"))[0] is not False:
        return ""
    return (
        f"a weak find: {found['object']} matched {found['inliers']} of the demo view's {found['card_points']} points; "
        "turn it closer to how it lay in the demo"
    )


class LocateBody(BaseModel):
    click: list[int]
    object: str


@router.post("/locate")
async def locate(body: LocateBody) -> dict:
    """Find a designated object of the demo where the operator clicked it, against the demo's view of it, without
    starting a track: the object a place goes onto. Each act finds it again where this left it."""
    if len(body.click) != 2:
        raise HTTPException(422, "click is x, y")
    with _state.lock:
        demo = _state.demo
    if demo is None or demo.objects.get(body.object, {}).get("status") != "done":
        raise HTTPException(409, f"{body.object!r} is not a tracked object of the current demo")
    if not _state.worker.running:
        raise HTTPException(409, "start the worker first")
    rgb, depth_m, intr = await _frame()
    found = await _locate(body.object, rgb, depth_m, intr, body.click, lambda: False, track=True)
    _store_located(body.object, found)
    return _located_info(found)


async def _locate_afresh(obj: str, stopped: Callable[[], bool], track: bool = True) -> str:
    """Find the place's object again where it was last found (where its track has it), as a click there would; with
    ``track``, it joins the live session from there. Post: "" with ``_state.located[obj]`` fresh and strong;
    otherwise why not."""
    with _state.lock:
        last = _state.located.get(obj)
    mask = None if last is None else last.get("mask")
    click = None if mask is None else _deepest_pixel(mask, mask.shape)
    if click is None:
        return f"click {obj} in the camera view to find it"
    rgb, depth_m, intr = await _frame()
    found = await _locate(obj, rgb, depth_m, intr, click, stopped, track=track)
    _store_located(obj, found)
    if not found["ok"]:
        return found["reason"] or f"{obj} is not where it was last seen: click it in the camera view"
    return _weak(found) + (", then press Act" if _weak(found) else "")


def _target_motion(demo: _Demo, t_bc: np.ndarray) -> tuple[np.ndarray | None, str]:
    """``(motion, problem)``: how the object a place goes onto moved from the demo to now, base frame. Its last locate
    against the demo's view of it, times the inverse of where the demo's track had it at the first pre-place, the
    last frame it was seen at or before then. Pre: a place is marked."""
    obj = _place_object(demo)
    assert obj is not None, "a place is marked"
    o = demo.objects.get(obj)
    if o is None or o.get("status") != "done" or demo.recording is None:
        return None, f"{obj!r} is not tracked in this demo"
    found = _located(demo, obj)
    if found is None:
        return None, f"click {obj} in the camera view to find it"
    if not found["ok"]:
        return None, found["reason"]
    if _weak(found):
        return None, _weak(found)
    f = _pose_frame(demo, obj, "preplace")
    assert f is not None, "a done object with a pre-place on it has a pose frame"
    cam = np.asarray(found["delta"], dtype=float) @ np.linalg.inv(np.asarray(o["deltas"][f], dtype=float))
    return t_bc @ cam @ np.linalg.inv(t_bc), ""


def _object_points(demo: _Demo, obj: str, t_bc: np.ndarray) -> np.ndarray:
    """The designated object's surface on the frame it was clicked on, base frame (N, 3): its depth under its mask
    there, every fourth pixel. The finds' motions are from that view. Pre: the object is done on a stream demo."""
    o = demo.objects[obj]
    _rgb, depth = _stream_frame(demo.recording, int(o["frame"]))
    ys, xs = np.nonzero(np.asarray(o["mask"], dtype=bool) & (depth > 0))
    ys, xs = ys[::4], xs[::4]
    k = np.loadtxt(pathlib.Path(demo.recording) / "cam_K.txt")
    z = depth[ys, xs]
    cam = np.stack([(xs - k[0, 2]) * z / k[0, 0], (ys - k[1, 2]) * z / k[1, 1], z], axis=1)
    return cam @ t_bc[:3, :3].T + t_bc[:3, 3]


def _tip_speed(demo: _Demo) -> np.ndarray:
    """The demo's fingertip speed at every sample, m/s, from its neighbours."""
    pos, n = demo.tips[:, :3, 3], len(demo.t)
    before, after = np.maximum(np.arange(n) - 1, 0), np.minimum(np.arange(n) + 1, n - 1)
    return np.linalg.norm(pos[after] - pos[before], axis=1) / np.maximum(demo.t[after] - demo.t[before], 1e-6)


def _firm_grip(demo: _Demo) -> int | None:
    """The demo sample its grip on the picked object became firm (:func:`core.firm_grip`), within the grasp's replay;
    None without a grasp end or when the gripper never stopped short of its command there."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    pre = sorted(float(k["t"]) for k in demo.keypoints if k["kind"] == "pregrasp")
    end = next((float(k["t"]) for k in demo.keypoints if k["kind"] == "grasp_end"), None)
    if not pre or end is None:
        return None
    gi = MOTOR_NAMES.index("gripper")
    i0, i1 = int(np.argmin(np.abs(demo.t - pre[-1]))), int(np.argmin(np.abs(demo.t - end)))
    closing = float(np.sign(demo.q_cmd[i1, gi] - demo.q_cmd[i0, gi])) or 1.0
    return core.firm_grip(demo.t, demo.q_cmd[:, gi], demo.q_obs[:, gi], i0, i1, closing)


def _hold_window(demo: _Demo, window: str) -> tuple[float, float] | None:
    """The demo's time span a hold is measured over, or None. "grip": from the firm grip until the arm moves again,
    the object gripped and still where it lay. "carry": from the grasp end to the last pre-place, or the end of the
    pause it is marked in, after any shift the lift caused. Pre: a place is marked."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    speed, n = _tip_speed(demo), len(demo.t)
    end = next(float(k["t"]) for k in demo.keypoints if k["kind"] == "grasp_end")
    i_end = int(np.argmin(np.abs(demo.t - end)))
    if window == "grip":
        i = _firm_grip(demo)
        if i is None:
            return None
        j = i
        while j + 1 <= i_end and speed[j + 1] < core.HOLD_STILL_M_S:
            j += 1
        return float(demo.t[i]), float(demo.t[j])
    last = max(float(k["t"]) for k in demo.keypoints if k["kind"] == "preplace")
    # A pause marked at its start goes on after the last pre-place: still the same hold until the arm moves again
    # or the gripper starts to open.
    grip = demo.q_cmd[:, MOTOR_NAMES.index("gripper")]
    j = int(np.argmin(np.abs(demo.t - last)))
    while (
        j + 1 < n
        and speed[j + 1] < core.HOLD_STILL_M_S
        and abs(grip[j + 1] - grip[i_end]) <= core.GRASP_HELD_SHORT
    ):
        j += 1
    return end, max(last, float(demo.t[j]))


def _still_held_frames(
    demo: _Demo, obj: str, window: str = "carry", need_seen: bool = True
) -> list[tuple[int, int]]:
    """Where the demo held the picked object still within a hold ``window`` (:func:`_hold_window`): ``(stream frame,
    demo sample)`` pairs with the fingertip slower than :data:`core.HOLD_STILL_M_S` and the object in its own track;
    nearest the window's end first, at least three frames apart, a few more than a hold needs. Pre: a place is
    marked on a stream demo."""
    span = _hold_window(demo, window)
    if span is None:
        return []
    start, last = span
    o = demo.objects[obj]
    times = _stream_times(demo.recording) - demo.t0
    speed = _tip_speed(demo)
    picked: list[tuple[int, int]] = []
    edge = 1e-3  # a frame on the window's edge stays in it, whatever the stream clock's rounding
    inside = np.flatnonzero((times >= start - edge) & (times <= last + edge))
    for f in sorted(inside, key=lambda f: abs(times[f] - last)):
        i = int(np.argmin(np.abs(demo.t - times[f])))
        if speed[i] >= core.HOLD_STILL_M_S or (need_seen and (not o["seen"][f] or not np.any(o["masks"][f]))):
            continue
        if all(abs(int(f) - g) >= 3 for g, _ in picked):
            picked.append((int(f), i))
        if len(picked) >= core.HOLD_VIEWS + 2:
            break
    return picked


async def _demo_hold(
    demo: _Demo, obj: str, t_bc: np.ndarray, stopped: Callable[[], bool], window: str = "carry"
) -> tuple[dict | None, str]:
    """``(hold, problem)``: how the demo held the picked object in a hold ``window`` (:func:`_hold_window`), as
    :func:`core.average_hold` reports it. A find of the object's view in the demo on each frame where the demo held
    it still there, seen from the gripper. A measured hold is kept on the demo for the marks, window and calibration
    it was measured with, and so is a failure the worker measured; one it did not answer is asked again next time.
    Pre: a place is marked on a stream demo."""
    span = _hold_window(demo, window)
    o = demo.objects[obj]
    key = json.dumps([obj, int(o["frame"]), window, span, np.round(t_bc, 5).tolist()])
    if key in demo.holds:
        kept = demo.holds[key]
        return (None, kept["problem"]) if "problem" in kept else (kept, "")
    frames = _still_held_frames(demo, obj, window)
    if not frames:
        where = (
            "after the gripper closed on it, before the lift"
            if window == "grip"
            else "between the grasp end and the last pre-place"
        )
        if _still_held_frames(demo, obj, window, need_seen=False):
            return None, f"{obj} is hidden in the gripper {where}: its own track does not see it there"
        return None, f"the demo never holds {obj} still in the gripper {where}"
    intr = await asyncio.get_event_loop().run_in_executor(_RENDER_EXECUTOR, _recording_intr, demo)
    holds, why, answered = [], "", True
    for f, i in frames:
        rgb, depth = await asyncio.get_event_loop().run_in_executor(
            _RENDER_EXECUTOR, _stream_frame, demo.recording, f
        )
        found = await _locate(obj, rgb, depth, intr, _deepest_pixel(o["masks"][f], rgb.shape), stopped)
        if stopped():
            return None, "stopped"
        answered = answered and bool(found.get("answered"))
        if found["ok"] and not _weak(found):
            holds.append(np.linalg.inv(demo.tips[i]) @ t_bc @ found["delta"] @ np.linalg.inv(t_bc))
        else:
            why = found["reason"] or _weak(found)
    points = await asyncio.get_event_loop().run_in_executor(_RENDER_EXECUTOR, _object_points, demo, obj, t_bc)
    avg = core.average_hold(holds, points.mean(axis=0))
    if avg is None:
        problem = (
            f"{obj} is not visible enough in the gripper in the demo: {len(holds)} of {len(frames)} still views "
            f"matched the demo's view of it ({why or 'their poses disagree'})"
        )
        if answered:
            demo.holds[key] = {"problem": problem}
        return None, problem
    avg["window"] = window
    demo.holds[key] = avg
    return avg, ""


def _recording_intr(demo: _Demo) -> dict[str, float]:
    """The intrinsics of the demo's recorded stream: as the demo kept them, else from the recording's own file."""
    if demo.intr and "fx" in demo.intr:
        return dict(demo.intr)
    k = np.loadtxt(pathlib.Path(demo.recording) / "cam_K.txt")
    rgb, _depth = _stream_frame(demo.recording, 0)
    return {
        "fx": k[0, 0],
        "fy": k[1, 1],
        "cx": k[0, 2],
        "cy": k[1, 2],
        "width": rgb.shape[1],
        "height": rgb.shape[0],
    }


def _held_click(
    points: np.ndarray, motion: np.ndarray, t_bc: np.ndarray, intr: dict[str, float], shape: tuple[int, ...]
) -> list[int] | None:
    """Where to click the held object: deep inside its surface carried by ``motion`` (base frame) and drawn into the
    image, or None when none of it lands in the frame."""
    moved = points @ motion[:3, :3].T + motion[:3, 3]
    cam = (moved - t_bc[:3, 3]) @ t_bc[:3, :3]
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for u, v in _project_cam(intr, cam):
        if 0 <= u < shape[1] and 0 <= v < shape[0]:
            mask[int(v), int(u)] = 1
    import cv2

    return _deepest_pixel(cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0, shape)


async def _held_view() -> (
    tuple[np.ndarray, np.ndarray, dict[str, float], np.ndarray, np.ndarray | None] | None
):
    """A view of the held object as it is now: ``(rgb, depth, intrinsics, fingertip, live mask)``, the fingertip read
    just before the frame and the live track's mask while it tracks, or None without the arm. Pre: the arm stands
    still, or the fingertip is not where the frame shows it."""
    from . import jog

    cur = jog.current_tip_and_anchor()
    if cur is None:
        return None
    rgb, depth_m, intr = await _frame()
    with _state.lock:  # a finger over the predicted middle: the live track's view is the next place to click
        test, state = _state.test, (_state.track.last or {}).get("state")
    mask = test.result.get("live_mask") if state == "tracking" and test is not None else None
    return rgb, depth_m, intr, cur[0], mask


async def _live_hold(
    demo: _Demo,
    obj: str,
    demo_hold: np.ndarray,
    t_bc: np.ndarray,
    stopped: Callable[[], bool],
    views: list[tuple[np.ndarray, np.ndarray, dict[str, float], np.ndarray, np.ndarray | None]] | None = None,
) -> tuple[dict | None, str]:
    """``(hold, problem)``: how the picked object sits in the gripper, as :func:`core.average_hold` reports it.
    Finds of its view in the demo on fresh frames (:func:`_held_view`), each seen from the gripper where FK had it
    then, with the click where the demo's hold puts the object; after a view that fails, where the live track had
    it, and back. ``views`` taken earlier are read instead of fresh ones: the grip's, read while the act goes on.
    Pre: without ``views``, the arm stands still with the object gripped."""
    points = await asyncio.get_event_loop().run_in_executor(_RENDER_EXECUTOR, _object_points, demo, obj, t_bc)
    pending = None if views is None else list(views)
    holds, tries, choice = [], 0, 0
    while len(holds) < core.HOLD_VIEWS and tries < core.HOLD_VIEWS + 2:
        view = (pending.pop(0) if pending else None) if pending is not None else await _held_view()
        if view is None:
            if pending is not None:
                break
            return None, "the arm went away"
        tries += 1
        rgb, depth_m, intr, tip, mask = view
        clicks = [_held_click(points, tip @ demo_hold, t_bc, intr, rgb.shape)]
        if mask is not None:
            clicks.append(_deepest_pixel(mask, rgb.shape))
        clicks = [c for c in clicks if c is not None]
        if not clicks:
            return None, f"{obj} would be outside the camera's view where the gripper holds it"
        found = await _locate(obj, rgb, depth_m, intr, clicks[choice % len(clicks)], stopped)
        if stopped():
            return None, "stopped"
        if found["ok"] and not _weak(found):
            holds.append(np.linalg.inv(tip) @ t_bc @ found["delta"] @ np.linalg.inv(t_bc))
        else:
            choice += 1  # the other place to click, while there is one
    avg = core.average_hold(holds, points.mean(axis=0))
    if avg is None:
        return None, (
            f"{obj} is not visible enough in the gripper to place it: {len(holds)} of {tries} views matched the "
            "demo's view of it"
        )
    return avg, ""


def _find_info(ref: dict[str, Any]) -> dict[str, Any]:
    """A find as the page shows it: what it matched, of how many of the demo view's points, and whether that is strong."""
    strong, share = core.find_strength(ref.get("inliers"), ref.get("card_points"))
    return {
        **{k: ref.get(k) for k in ("object", "ok", "inliers", "turn_deg", "reason", "card_points")},
        "strong": strong,
        "share": share,
    }


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
            "ref": None if not kp.get("ref") else _find_info(kp["ref"]),
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
        *(k for k in data.files if k.startswith("other_")),
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
    elif job.kind == "locate":
        pass  # :func:`_locate` is waiting on the job itself
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
                "card_points": r.get("ref_card_points"),
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
    if job.extra.get("ref_object") and r.get("ref_ok"):
        _remember_seen(job.extra["ref_object"], np.asarray(r["mask"]).astype(bool), job.rgb.shape)
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
    """Run-time options, each changed only when given: ``flat`` opts into the resting prior (see
    :func:`_compose_motion`), off by default; ``trust_share`` is the share of the place object's tracked points that
    must be seen for its track to be followed (see :func:`_apply_others`)."""
    with _state.lock:
        if body.flat is not None:
            _state.flat = bool(body.flat)
        if body.trust_share is not None:
            _state.trust_share = float(body.trust_share)
        return {"flat": _state.flat, "trust_share": _state.trust_share}


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


FOUND_COLOUR = (
    0,
    140,
    255,
)  # an object as a find placed it: orange, apart from the tracked object's magenta and yellow


def _view_points(demo: _Demo, obj: str) -> np.ndarray:
    """The designated object's surface on the frame it was clicked on, camera frame (N, 3): the points a find's motion
    carries. Read once per designation and kept on the demo. Pre: the object is done on a stream demo."""
    o = demo.objects[obj]
    key = (int(o["frame"]), int(np.count_nonzero(o["mask"])))
    kept = demo.view_points.get(obj)
    if kept is None or kept[0] != key:
        kept = demo.view_points[obj] = (key, _object_points(demo, obj, np.eye(4)))
    return kept[1]


def _found_mask(
    intr: dict[str, float], points: np.ndarray, delta: np.ndarray, shape: tuple[int, ...]
) -> np.ndarray | None:
    """Where an object's surface from the demo's view lands in an image of ``shape`` when carried by a find's motion
    (camera frame): its projected points, their gaps closed. None when fewer than three land in the image."""
    import cv2

    d = np.asarray(delta, dtype=float)
    uv = _project_cam(intr, np.asarray(points, dtype=float) @ d[:3, :3].T + d[:3, 3])
    h, w = shape[:2]
    uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)].astype(int)
    if len(uv) < 3:
        return None
    mask = np.zeros((h, w), np.uint8)
    for u, v in uv:
        cv2.circle(mask, (int(u), int(v)), 3, 1, -1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(
        bool
    )  # the points' gaps only


def _draw_found(
    bgr: np.ndarray, intr: dict[str, float], points: np.ndarray, delta: np.ndarray, label: str
) -> None:
    """An object where a find placed it: the outline of its surface from the demo's view (:func:`_found_mask`), its
    frame at the surface's centre, and its name. Not tracked, it stays where the find put it until the next find."""
    import cv2

    mask = _found_mask(intr, points, delta, bgr.shape)
    if mask is None:
        return
    _outline(bgr, mask, FOUND_COLOUR)
    _draw_frame(bgr, intr, np.asarray(delta, dtype=float), np.asarray(points, dtype=float).mean(axis=0))
    ys, xs = np.nonzero(mask)
    top = int(np.argmin(ys))
    cv2.putText(
        bgr,
        label,
        (int(xs[top]) - 20, max(int(ys[top]) - 8, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        FOUND_COLOUR,
        2,
    )


def _render_live(rgb, r, result, transported, teach, status) -> bytes:
    """The tracking view: the mask edge, the points that agree, the taught cloud carried by the
    motion (where the object is believed to be), the transported pre-grasp, the object a place goes
    onto where its last find put it, and a status strip."""
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
    with _state.lock:
        demo = _state.demo
    onto = None if demo is None else _place_object(demo)
    located = None if onto is None else _located(demo, onto)
    if located is not None and located.get("ok") and located.get("delta") is not None and teach is not None:
        with contextlib.suppress(
            OSError, KeyError, ValueError
        ):  # a demo recording that went away: no outline
            _draw_found(bgr, teach.intr, _view_points(demo, onto), located["delta"], onto)
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
    badges = [_find_badge((teach.keypoints.get("ref") if teach is not None else None) or {})]
    if located is not None:
        with _state.lock:
            tracked = _state.target.last.get("state") if _state.target.obj == onto else None
        badges.append(_find_badge(located, f"{onto}, placed onto, {tracked or 'not tracked'}"))
    y = 32
    for badge in badges:  # each on its own line: the strip above already runs off the frame's edge
        if badge is None:
            continue
        text, badge_colour = badge
        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(bgr, (0, y), (w + 16, y + 26), (0, 0, 0), -1)
        cv2.putText(bgr, text, (8, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, badge_colour, 2, cv2.LINE_AA)
        y += 28
    return _jpeg(bgr)


def _find_badge(ref: dict[str, Any], label: str = "find") -> tuple[str, tuple[int, int, int]] | None:
    """The live view's line about a find of the demo's view: strong in green, weak in orange with what to do; None when
    there is no such find or its strength is unknown. ``label`` names it: the find the track starts from, or the
    object a place goes onto."""
    strong, _share = core.find_strength(ref.get("inliers"), ref.get("card_points"))
    if not ref.get("ok") or strong is None:
        return None
    counts = f"{ref['inliers']} of {ref['card_points']} points"
    if strong:
        return f"{label}: strong, {counts}", (60, 230, 60)
    return f"{label}: weak, {counts}; turn it closer to how it lay in the demo", (0, 165, 255)


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
                # The fitted points are the recording's, not the live readout's: arrays the state cannot serve. The
                # session's other objects are applied on their own (:func:`_apply_others`).
                if k
                not in (
                    "mask",
                    "uv",
                    "xyz",
                    "delta",
                    "face_teach",
                    "face_find",
                    "fit_uv",
                    "fit_inlier",
                    "others",
                )
                and not k.startswith("other_")
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
            name = (teach.keypoints.get("ref") or {}).get("object")
            if name and result.get("live_mask") is not None:
                _remember_seen(name, result["live_mask"], job.rgb.shape)
        else:
            status.update(ok=False, state="untrusted", reason=why)
    _apply_others(r, job.rgb.shape)  # the same step's other objects: the one a place goes onto
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
                # A frame that cannot be read now is waited out: a track that stopped here stayed stopped once the
                # camera was back, while the live view went on showing its last frame as tracking.
                tr.last = {"state": "waiting", "reason": e.detail}
                await asyncio.sleep(TRACK_FRAME_RETRY_S)
                continue
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
        "place": act.place,
        "inject": act.inject,
        "correct_hold": act.correct_hold,
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

    def target(
        self,
        step: str,
        pose: np.ndarray | None = None,
        joints: np.ndarray | None = None,
        at: float | None = None,
    ) -> None:
        """A target the act gave the arm, at ``at`` (wall clock) when it went out, or now."""
        entry: dict[str, Any] = {"t": time.time() if at is None else at, "step": step}
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
# A joint held short under load stops there and waits do not close it: right after the grasp's lift, with the gamepad
# in the gripper, the worst joint of five pick-and-place acts sat 1.2 to 3.4 deg off its last target, the 3.4 holding
# for 20 s (2026-10-07). Stopped (no joint moving ACT_STILL_DEG over ACT_STILL_S) within this, the arm has arrived.
ACT_STALL_DEG = 6.0
ACT_STILL_DEG = 0.3
ACT_STILL_S = 0.5
ACT_TICK_S = 0.05
LANDING_EVERY_S = (
    0.25  # a landing turn is judged on the place's samples this far apart, as well as its pre-places
)
LANDING_TOP_M = 0.004  # the top of the object placed onto: its surface within this of its highest points
GRIP_SETTLE_S = 1.0  # the grasp check waits at most this long for the gripper to stop closing
HOLD_VIEW_GAP_S = (
    0.1  # views of the held object at the grip are this far apart, so each is a new camera frame
)


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
        "place_object": _place_object(demo),
        "landing": demo.landing,
        "hold": _demo_hold_info(demo),
    }


def _demo_hold_info(demo: _Demo) -> dict[str, Any] | None:
    """The demo's measured hold of the picked object for its current marks, as the page shows it; None until an
    act has measured it."""
    obj = _marks_object(demo)
    o = demo.objects.get(obj) if obj else None
    if o is None or _place_object(demo) is None or not demo.holds:
        return None
    out = {}
    for window in ("grip", "carry"):
        span = json.loads(json.dumps(_hold_window(demo, window)))  # as the key holds it
        for key, avg in demo.holds.items():
            if "problem" in avg:  # a failure kept so it is not asked again: nothing measured to show
                continue
            if json.loads(key)[:4] == [obj, int(o["frame"]), window, span]:
                out[window] = {k: avg[k] for k in ("n", "views", "spread_mm", "spread_deg")}
    return out or None


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


GROUPS_VIEW = (
    "127.0.0.1",
    9141,
)  # the point groups' live view's own server (benchmarks/group_live.py --live)
_GROUPS_SCRIPT = _REPO / "benchmarks" / "group_live.py"
# The view runs in Point2Pose's environment, where TAPIR and its checkpoint live, as the worker's bridge does.
P2P_PYTHON = os.environ.get(
    "LEROBOT_P2P_PYTHON", str(pathlib.Path.home() / ".cache/point2pose/venv/bin/python")
)
P2P_REPO = os.environ.get("LEROBOT_P2P_REPO", str(pathlib.Path.home() / ".cache/point2pose/point-to-pose"))
_GROUPS_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pregrasp-groups")
GROUPS_PAGE = """<html><head><title>point groups</title></head><body style='margin:0;background:#111'>
<img src='/api/pregrasp/groups/stream' style='width:100%'
 onerror="setTimeout(()=>{this.src='/api/pregrasp/groups/stream?'+Date.now()},1000)"></body></html>"""


@router.get("/groups/view")
async def groups_view() -> Response:
    """The point groups' live view alone on a page, full width; the Approach tab's Groups panel shows the same
    stream with Start and Finish. A relay until the groups run in the worker and draw on the tab's own camera
    view."""
    return Response(content=GROUPS_PAGE, media_type="text/html")


async def _relay_stream(addr: tuple[str, int]):
    """The view's MJPEG stream with the view's own response headers stripped, over asyncio streams: nothing here
    blocks the loop. While the view is not there (not started yet, restarting) the relay waits and connects again,
    so the page's stream need not be reloaded."""
    while True:
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(*addr), timeout=2.0)
        except (TimeoutError, OSError):
            await asyncio.sleep(1.0)
            continue
        try:
            writer.write(b"GET /stream HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
            await writer.drain()
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b""):
                    break
            while True:
                chunk = await reader.read(1 << 16)
                if not chunk:
                    break
                yield chunk
        finally:
            writer.close()
        await asyncio.sleep(0.5)


@router.get("/groups/stream")
async def groups_stream() -> StreamingResponse:
    return StreamingResponse(
        _relay_stream(GROUPS_VIEW), media_type="multipart/x-mixed-replace; boundary=frame"
    )


async def _groups_recording() -> dict[str, Any] | None:
    """What the view says of its recording ({"recording", "frames", "last"}), or None while it cannot be reached:
    still loading the tracker, or gone."""
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(*GROUPS_VIEW), timeout=2.0)
    except (TimeoutError, OSError):
        return None
    try:
        writer.write(b"GET /record/status HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
        await writer.drain()
        raw = await asyncio.wait_for(reader.read(), timeout=5.0)
    except (TimeoutError, OSError):
        return None
    finally:
        writer.close()
    try:
        return json.loads(raw.partition(b"\r\n\r\n")[2])
    except ValueError:
        return None


@router.get("/groups/status")
async def groups_status() -> dict:
    """The view's process and its recording: {"running", "ready", "recording", "frames", "last", "log"}. ``ready``
    is false while the view is up but not yet answering (the tracker loading); ``last`` is where the finished
    recording went; ``log`` the view's last lines, which say why it stopped when it stopped by itself."""
    g = _state.groups
    rec = await _groups_recording()
    with _state.lock:
        out: dict[str, Any] = {"running": g.running or rec is not None, "ready": rec is not None, **g.last}
        out["log"] = g.log[-5:]
    out.update(rec or {})
    out.setdefault("recording", None)
    out.setdefault("frames", 0)
    out.setdefault("last", None)
    return out


@router.post("/groups/start")
async def groups_start(request: Request) -> dict:
    """Start the view: the camera's frames, TAPIR on the GPU, the groups drawn as a stream, and from the first
    frame a recording (colour, depth, the groups' state) that Finish closes."""
    from . import showservo

    if showservo.live_camera() is None:
        raise HTTPException(409, "start the camera first")
    g = _state.groups
    with _state.lock:
        if g.running:
            raise HTTPException(409, "the groups view is already running")
        cmd = [
            P2P_PYTHON,
            str(_GROUPS_SCRIPT),
            "--live",
            "--record",
            "--server",
            str(request.base_url).rstrip("/"),
            "--port",
            str(GROUPS_VIEW[1]),
            "--pips",
            "2",
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=P2P_REPO
            )
        except OSError as e:
            raise HTTPException(409, f"the view cannot start: {e}") from e
        g.proc, g.log, g.last = proc, [], {}

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            with _state.lock:
                g.log.append(line.rstrip("\n"))
                del g.log[:-200]

    threading.Thread(target=pump, daemon=True, name="pregrasp-groups").start()
    return {"status": "started"}


def _end(proc: subprocess.Popen) -> None:
    """A SIGTERM lets the view close its recording and the camera's; a view that does not go is killed."""
    proc.terminate()
    try:
        proc.wait(10.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(5.0)


@router.post("/groups/stop")
async def groups_stop() -> dict:
    """Finish: the view's recording is closed and the view stopped, the camera's recording and the GPU freed with
    it. Answers where the recording went and how many frames it holds."""
    g = _state.groups
    with _state.lock:
        proc = g.proc if g.running else None
    if proc is None:
        if await _groups_recording() is not None:
            raise HTTPException(409, "the groups view was not started here; stop it where it was started")
        raise HTTPException(409, "the groups view is not running")
    rec = await _groups_recording() or {}
    await asyncio.get_event_loop().run_in_executor(_GROUPS_EXECUTOR, _end, proc)
    with _state.lock:  # a view that was killed rather than stopped leaves the camera's frames shared
        share, _state.camera_share = _state.camera_share, None
    if share is not None:
        await _stop_share(share)
    last = {
        "recording": None,
        "frames": rec.get("frames", 0),
        "last": rec.get("recording") or rec.get("last"),
    }
    with _state.lock:
        g.last = last
    return last


@router.post("/camera/share/start")
async def camera_share_start() -> dict:
    """Put the camera's frames into shared memory for a reader in another process; the ring's name, size and
    intrinsics. Already sharing: the same ring."""
    from lerobot.showservo.frame_ring import FrameRing

    from . import showservo

    camera = showservo.live_camera()
    if camera is None:
        raise HTTPException(409, "start the camera first")
    with _state.lock:
        share = _state.camera_share
        if share is not None and share.thread is not None and share.thread.is_alive():
            ring = share.ring
            return {"name": ring.name, "height": ring.h, "width": ring.w, "frames": share.n}
    intr = await asyncio.get_event_loop().run_in_executor(showservo._EXECUTOR, camera.color_intrinsics)
    k = [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]]
    name = f"lerobot_frames_{os.getpid()}_{time.monotonic_ns()}"
    ring = FrameRing(name, int(intr["height"]), int(intr["width"]), slots=4, k=k, create=True)
    share = _FrameShare(ring=ring)
    share.thread = threading.Thread(
        target=_share_frames, args=(camera, share), name="pregrasp-share", daemon=True
    )
    with _state.lock:
        old, _state.camera_share = _state.camera_share, share
    if old is not None:
        await _stop_share(old)
    share.thread.start()
    return {"name": ring.name, "height": ring.h, "width": ring.w, "frames": 0}


async def _stop_share(share: _FrameShare) -> None:
    share.stop.set()
    if share.thread is not None:
        await asyncio.get_event_loop().run_in_executor(_RENDER_EXECUTOR, share.thread.join, 5.0)
    share.ring.close()


@router.post("/camera/share/stop")
async def camera_share_stop() -> dict:
    with _state.lock:
        share, _state.camera_share = _state.camera_share, None
    if share is None:
        raise HTTPException(409, "the camera's frames are not being shared")
    await _stop_share(share)
    return {"frames": share.n, "error": share.error}


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
    _write_keypoints(root, demo.keypoints, demo.landing)
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


def _write_keypoints(root: pathlib.Path, keypoints: list[dict[str, Any]], landing: str = "exact") -> None:
    """The operator's marks and the place's landing rule as a sidecar the act reads back; nothing is written when
    there are no marks and the landing is the default."""
    f = pathlib.Path(root) / KEYPOINTS_FILE
    if keypoints or landing != "exact":
        f.write_text(
            json.dumps(
                {"keypoints": keypoints, **({"landing": landing} if landing != "exact" else {})}, indent=1
            )
        )
    elif f.exists():
        f.unlink()  # safe-destruct: the marks sidecar we wrote ourselves; the operator cleared the marks


def _read_landing(root: pathlib.Path) -> str:
    """The place's landing rule saved beside a demo: "exact" when none was saved or it is not one this version reads."""
    f = pathlib.Path(root) / KEYPOINTS_FILE
    try:
        landing = json.loads(f.read_text()).get("landing", "exact") if f.exists() else "exact"
    except (OSError, ValueError, AttributeError):
        return "exact"
    return landing if landing in core.LANDINGS else "exact"


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
        landing=_read_landing(f.parent),
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
        _start_refind(demo)
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
    _start_refind(demo)  # after that teach: the objects designated on the demo, where they were last seen
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
        "landing": demo.landing,
        "has_frames": _demo_has_frames(demo),
        "recording": demo.recording is not None,
        "taught": demo.taught,
        "image_size": None if demo.intr is None else [demo.intr["width"], demo.intr["height"]],
        "uv": _demo_path_uv(demo),
        "pose_t": _pose_frame_t(demo),
        "place_pose_t": _pose_frame_t(demo, "preplace"),
        "grip_t": None if (g := _firm_grip(demo)) is None else float(demo.t[g]),
    }


def _pose_frame_t(demo: _Demo, kind: str = "pregrasp") -> float | None:
    """When, in the demo's time, the act reads an object's pose: the picked one's, or with ``kind`` "preplace" the
    one a place goes onto. The editor marks it on the timeline."""
    obj = _marks_object(demo) if kind == "pregrasp" else _place_object(demo)
    f = None if obj is None else _pose_frame(demo, obj, kind)
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


class ObjectSymmetryBody(BaseModel):
    name: str
    order: int = Field(1, ge=1, le=12)  # turns of 360/order deg about its resting axis look and act the same


@router.post("/demo/objects/symmetry")
async def demo_object_symmetry(body: ObjectSymmetryBody) -> dict:
    """Declare an object's rotational symmetry about the axis it rests on: 1 for none, 2 for a shape that reads the
    same turned end to end, 4 for a plain cube. Its finds then report, of the motions it cannot be told apart by,
    the one that turns it least."""
    with _state.lock:
        demo = _state.demo
        if demo is None or body.name not in demo.objects:
            raise HTTPException(404, "no such object")
        demo.objects[body.name]["symmetry"] = int(body.order)
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
                "symmetry": int(o.get("symmetry") or 1),
                **(_object_pose_info(demo, name) if o["status"] == "done" else {}),
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
        "names": json.dumps(
            [[n, int(o["frame"]), list(o["click"]), int(o.get("symmetry") or 1)] for n, o in done.items()]
        )
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
    for n, (name, frame, click, *rest) in enumerate(json.loads(str(z["names"]))):
        out[name] = {
            "frame": int(frame),
            "click": list(click),
            "symmetry": int(rest[0]) if rest else 1,  # demos saved before it was declared have none
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
    for named in sorted({k.get("object", "") for k in kps} - {""}):
        if demo.objects.get(named, {}).get("status") != "done":
            raise HTTPException(422, f"{named!r} is not a tracked object of this demo")
    times = _stream_times(demo.recording) if demo.recording is not None else np.zeros(0)
    for k in kps:
        if k["kind"] == "pose" and len(times):
            f = int(np.argmin(np.abs(times - (demo.t0 + k["t"]))))
            if not demo.objects[k["object"]]["seen"][f]:
                raise HTTPException(
                    422,
                    f"{k['object']} is hidden at {k['t']:.2f} s: read its pose on a frame where it is in view",
                )
    obj = next((k.get("object", "") for k in kps if k["kind"] in core.GRASP_KINDS), "")
    if kps and not obj and not demo.taught:
        raise HTTPException(
            422, "nothing was taught before this demo: the marks follow an object clicked on its recording"
        )
    with _state.lock:
        demo.keypoints = kps
    if demo.root is not None:
        await asyncio.get_event_loop().run_in_executor(
            _RENDER_EXECUTOR, _write_keypoints, pathlib.Path(demo.root), kps, demo.landing
        )
    return _demo_info(demo)


class LandingBody(BaseModel):
    landing: str


@router.post("/demo/landing")
async def demo_landing(body: LandingBody) -> dict:
    """How the place may land on its object: "exact" as shown, "symmetry" turned by any of that object's symmetric
    turns, "turn" turned any amount about its middle. The act takes, of those, a landing the servos reach whose joints
    stay nearest the demo's. Kept beside a saved demo."""
    if body.landing not in core.LANDINGS:
        raise HTTPException(422, f"a landing is one of {', '.join(core.LANDINGS)}")
    with _state.lock:
        demo = _state.demo
        if demo is None:
            raise HTTPException(409, "record or load a demo first")
        demo.landing = body.landing
        kps = list(demo.keypoints)
    if demo.root is not None:
        await asyncio.get_event_loop().run_in_executor(
            _RENDER_EXECUTOR, _write_keypoints, pathlib.Path(demo.root), kps, body.landing
        )
    return _demo_info(demo)


@router.get("/demo/reach")
async def demo_reach() -> dict:
    """Can the arm do the marked pre-grasp and grasp on the object where it is now? The act's own judgement, without moving.

    Not while an act runs, and the landing's turns are searched once per find of the target, that turn reused after.
    The search is a second of CPU-bound solving in this process: polled every 3 s by the editor, it held the arm's
    loop up for a second at a time (2026-10-08: 0.9-1.1 s stalls every 3 s while the arm stood still; in an act the
    release reached the arm together with the lift and the gamepad was pulled over).
    """
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    from . import jog

    with _state.lock:
        demo, test, acting = _state.demo, _state.test, _state.act.on
    if acting:
        raise HTTPException(409, "an act is running")
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
    target_base, place_problem = (None, "") if _place_object(demo) is None else _target_motion(demo, t_bc)
    landing, aim, key, kept = demo.landing, target_base, None, None
    if target_base is not None and landing != "exact":
        found = _located(demo, _place_object(demo)) or {}
        key = [demo.name, json.dumps(demo.keypoints, sort_keys=True), landing, found.get("at")]
        with _state.lock:
            kept = dict(_state.reach_landing) if _state.reach_landing.get("key") == key else None
        if kept is not None:  # searched for this find already: plan that turn alone
            landing = "exact"
            if kept["turn_deg"] is not None:
                aim = core.landed(target_base, kept["centre"], kept["turn_deg"])
    plan = await asyncio.get_event_loop().run_in_executor(
        _ACT_EXECUTOR,
        functools.partial(_plan_act, landing=landing, ranges=jog.servo_ranges()),
        demo,
        test.result["delta_cam"],
        t_bc,
        kin,
        q_now,
        jog.walk_limits(),
        jog.workspace_box(),
        1.0,
        0,
        aim,
    )
    if kept is not None:
        plan = {**plan, "landing": kept["landing"]}
        if kept["turn_deg"] is None:
            plan = {**plan, "ok": False, "reason": kept["reason"]}
    elif key is not None and "landing" in plan:
        with _state.lock:
            _state.reach_landing = {
                "key": key,
                "turn_deg": plan["landing"]["turn_deg"],
                "centre": plan.get("landing_centre"),
                "landing": plan["landing"],
                "reason": plan["reason"],
            }
    return {k: plan[k] for k in ("ok", "reason", "marks", "summary", "landing") if k in plan} | {
        "place_problem": place_problem
    }


def _has_pregrasp(demo: _Demo) -> bool:
    return any(k.get("kind") == "pregrasp" for k in demo.keypoints)


def _marks_object(demo: _Demo) -> str | None:
    """The designated object the pre-grasps and the grasp follow, the one picked and then held; None for the object
    taught before the demo."""
    names = {k.get("object") or "" for k in demo.keypoints if k.get("kind") in core.GRASP_KINDS} - {""}
    return next(iter(names)) if names else None


def _place_object(demo: _Demo) -> str | None:
    """The designated object the pre-places and the place follow, the one the picked object goes onto; None when
    no place is marked."""
    names = {k.get("object") or "" for k in demo.keypoints if k.get("kind") in core.PLACE_KINDS} - {""}
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


def _pose_frame(demo: _Demo, obj: str | None = None, kind: str = "pregrasp") -> int | None:
    """The stream frame the act reads a designated object's demo pose from, as :func:`_pose_choice` picks it; by
    default for the object picked, whose stage marks are pre-grasps."""
    obj = _marks_object(demo) if obj is None else obj
    choice = None if obj is None else _pose_choice(demo, obj, kind)
    return None if choice is None else choice[0]


def _pose_choice(demo: _Demo, obj: str, kind: str) -> tuple[int, str] | None:
    """``(stream frame, where it comes from)``: the frame the act reads ``obj``'s demo pose from, for a stage whose
    marks are of ``kind`` ("pregrasp" for the object picked, "preplace" for the one placed onto).

    The frame the operator set for it with a pose mark ("set"). Else the frame it was clicked on ("clicked"): its
    view there is the one every find matches against, so its pose there is exact, and the operator clicked it where
    it could be seen; taken when it comes no later than the stage's replay begins. Else the last frame it was seen at
    or before the stage's first mark. None when ``obj`` is not tracked in the demo or the stage has no mark yet.
    """
    o = demo.objects.get(obj)
    times = _stream_times(demo.recording) if demo.recording is not None else np.zeros(0)
    pre = [float(k["t"]) for k in demo.keypoints if k["kind"] == kind]
    if o is None or o.get("status") != "done" or not len(times) or not pre:
        return None
    chosen = next(
        (float(k["t"]) for k in demo.keypoints if k["kind"] == "pose" and k.get("object") == obj), None
    )
    if chosen is not None:
        return int(np.argmin(np.abs(times - (demo.t0 + chosen)))), "set"
    click = int(o["frame"])
    if times[click] - demo.t0 <= max(pre):
        return click, "clicked"
    f0 = int(np.argmin(np.abs(times - (demo.t0 + min(pre)))))
    seen = np.flatnonzero(np.asarray(o["seen"])[: f0 + 1])
    stage = "pre-grasp" if kind == "pregrasp" else "pre-place"
    return (int(seen[-1]) if len(seen) else click), f"last seen by the first {stage}"


def _object_pose_info(demo: _Demo, name: str) -> dict[str, Any]:
    """Where the editor says ``name``'s demo pose is read: ``pose_t`` (seconds into the demo) and ``pose_from``. An
    object no mark follows yet reports the frame set for it, else the frame it was clicked on."""
    kind = "pregrasp" if name == _marks_object(demo) else "preplace" if name == _place_object(demo) else None
    choice = None if kind is None else _pose_choice(demo, name, kind)
    times = _stream_times(demo.recording) if demo.recording is not None else np.zeros(0)
    if choice is None:
        o = demo.objects[name]
        chosen = next(
            (float(k["t"]) for k in demo.keypoints if k["kind"] == "pose" and k.get("object") == name), None
        )
        choice = (
            (int(o["frame"]), "clicked")
            if chosen is None
            else (int(np.argmin(np.abs(times - (demo.t0 + chosen)))), "set")
        )
    frame, where = choice
    return {"pose_t": float(times[frame] - demo.t0) if len(times) > frame else None, "pose_from": where}


def _delta_base(demo: _Demo, delta_cam: np.ndarray, t_bc: np.ndarray) -> np.ndarray:
    """The object's motion from the demo to now, in the base frame: the tracker's fit as it is.

    Pre: :func:`_reference_motion` has no problem for the current teach.
    """
    with _state.lock:
        teach = _state.teach
    ref, problem = _reference_motion(demo, teach)
    assert ref is not None, problem
    rel = np.asarray(delta_cam, dtype=float) @ ref
    with _state.lock:
        wrong = _state.act.find_error  # an injected error: the act believes the object is here instead
    base = t_bc @ rel @ np.linalg.inv(t_bc)
    return base if wrong is None else wrong @ base


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
    """The act's fingertip path from the first pre-grasp on, for the camera view, with the carry and the place once
    the object it goes onto is found (as if held as in the demo); None until a pre-grasp is marked."""
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
    target = None if _place_object(demo) is None else _target_motion(demo, t_bc)[0]
    if target is not None:
        then = core.plan_place(
            demo.keypoints,
            demo.t,
            demo.tips,
            demo.q_cmd[:, gi],
            demo.q_obs,
            target,
            np.eye(4),
            plan["poses"][-1],
            float(plan["grips"][-1]),
            jog.MAX_LINEAR_M_S,
            jog.MAX_ANGULAR_RAD_S,
            1.0,
            jog.HZ,
        )
        plan = core.join_plans(plan, then)
    return plan["poses"]


def _plan_act(
    demo: _Demo,
    delta_cam: np.ndarray | None,
    t_bc: np.ndarray,
    kin: Any,
    q_now: np.ndarray,
    limits: tuple[float, float],
    box: tuple[tuple[float, float, float], tuple[float, float, float]],
    speed: float,
    skip: int = 0,
    target_base: np.ndarray | None = None,
    hold_fix: np.ndarray | None = None,
    part: str = "all",
    aim: np.ndarray | None = None,
    landing: str = "exact",
    ranges: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """:func:`_plan_act_once`, the place landed at the turn its rule ``landing`` allows that the arm makes best.

    With "exact", or nothing to place, the demo's own landing. Otherwise the turns the rule allows about the vertical
    through the middle of the object placed onto (:func:`core.landing_turns`) are ranked on the place's judged samples,
    reachable within the servos' ``ranges`` with joints nearest the demo's (:func:`core.rank_landings`), and the
    cheapest that plans in full is taken. Post: as :func:`_plan_act_once`, plus ``landing`` (the rule, how many turns
    the arm reaches, the turn taken and its cost; ``turn_deg`` None when none plans) when a rule other than "exact"
    was searched, and ``landing_centre``, the middle the turn is about, when a turn was taken.
    """
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    onto = _place_object(demo)
    once = functools.partial(
        _plan_act_once,
        demo,
        delta_cam,
        t_bc,
        kin,
        q_now,
        limits,
        box,
        speed,
        skip,
        part=part,
        aim=aim,
        ranges=ranges,
    )
    if landing == "exact" or onto is None or target_base is None or part not in ("all", "place"):
        return once(target_base, hold_fix)
    centre = _landing_centre(demo, onto, t_bc)
    fix = np.eye(4) if hold_fix is None else np.asarray(hold_fix, dtype=float)
    idx = _landing_samples(demo)
    lo, hi = ranges if ranges is not None else (None, None)
    ranked = core.rank_landings(
        kin,
        core.landing_turns(landing, int(demo.objects[onto].get("symmetry") or 1)),
        lambda deg: np.stack([core.landed(target_base, centre, deg) @ demo.tips[i] @ fix for i in idx]),
        demo.q_obs[idx],
        lo,
        hi,
        MOTOR_NAMES.index("gripper"),
    )
    info: dict[str, Any] = {"rule": landing, "reachable": len(ranked), "turn_deg": None}
    tried = []
    for cost, deg in ranked[: core.LANDING_TRIES]:
        plan = once(core.landed(target_base, centre, deg), hold_fix)
        if plan["ok"]:
            return {**plan, "landing": {**info, "turn_deg": deg, "cost": cost}, "landing_centre": centre}
        tried.append((deg, plan))
    if tried:
        deg, plan = tried[0]
        why = f"of the {len(ranked)} landings on {onto} within reach, none plans: turned {deg:.0f} deg, {plan['reason']}"
    else:
        plan = once(target_base, hold_fix)
        why = f"no landing on {onto} is within reach of the arm" + (
            f": as shown, {plan['reason']}" if plan["reason"] else ""
        )
    return {**plan, "ok": False, "reason": why, "landing": info}


def _landing_centre(demo: _Demo, obj: str, t_bc: np.ndarray) -> np.ndarray:
    """The middle of the top of ``obj`` where the demo saw it on its place pose frame, base frame: the turns a landing
    may take are about the vertical through it. The top is its surface within LANDING_TOP_M of its highest points (by
    the 95th percentile, clear of depth noise). Pre: ``obj`` is a done object with a pre-place on it."""
    o = demo.objects[obj]
    f = _pose_frame(demo, obj, "preplace")
    assert f is not None, "a done object with a pre-place on it has a pose frame"
    move = t_bc @ np.asarray(o["deltas"][f], dtype=float) @ np.linalg.inv(t_bc)
    pts = _object_points(demo, obj, t_bc) @ move[:3, :3].T + move[:3, 3]
    return pts[pts[:, 2] >= np.percentile(pts[:, 2], 95) - LANDING_TOP_M].mean(axis=0)


def _landing_samples(demo: _Demo) -> list[int]:
    """The demo samples a landing turn is judged on: each pre-place, then the place every LANDING_EVERY_S to its end.
    Pre: at least one pre-place is marked."""
    pre = sorted(float(k["t"]) for k in demo.keypoints if k["kind"] == "preplace")
    idx = [int(np.argmin(np.abs(demo.t - tk))) for tk in pre]
    end = next((float(k["t"]) for k in demo.keypoints if k["kind"] == "place_end"), None)
    if end is not None:
        i1 = int(np.argmin(np.abs(demo.t - end)))
        step = max(1, int(round(LANDING_EVERY_S * demo.fps)))
        idx += [*range(idx[-1] + step, i1, step), i1]
    return idx


def _plan_act_once(
    demo: _Demo,
    delta_cam: np.ndarray | None,
    t_bc: np.ndarray,
    kin: Any,
    q_now: np.ndarray,
    limits: tuple[float, float],
    box: tuple[tuple[float, float, float], tuple[float, float, float]],
    speed: float,
    skip: int = 0,
    target_base: np.ndarray | None = None,
    hold_fix: np.ndarray | None = None,
    part: str = "all",
    aim: np.ndarray | None = None,
    ranges: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """What the act will do on the objects where they are now, judged before the arm moves.

    Pre: the demo has a pre-grasp mark. From the arm's present joints ``q_now``:
    straight lines through the pre-grasp points, then, when a grasp end is marked,
    the grasp exactly as recorded, all carried by the picked object's motion. When a
    place is marked and ``target_base``, the motion of the object it goes onto, is
    given, then the lines through the pre-places and the place, carried by that motion
    and corrected by ``hold_fix`` (:func:`core.plan_place`; the identity until the hold
    is measured). ``part`` "grasp" plans the approach and grasp only, "place" the place
    only, from the arm's present joints, ``skip`` counting the stage's marks already
    reached. ``aim``, an injected error, carries the approach and grasp off, base frame. Every sample is solved by IK from the one before. Refuses, naming the
    reason, when a mark or a replayed sample is out of reach, when it needs a joint past
    its servo's range (``ranges``: each joint's (lo, hi) in motor degrees, the gripper
    NaN), when a sample leaves the workspace or goes lower than the table floor (or than
    the demo itself went at that sample), or when the arm would jump between two samples.
    Post: ``ok``, ``reason``, ``times`` (N,), ``q`` (N, J), ``stage`` (N,), ``marks``
    (label, t, residual_mm, ok) and ``summary``, JSON-safe apart from the arrays.
    """
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    from . import jog

    gi = MOTOR_NAMES.index("gripper")
    q_now = np.asarray(q_now, dtype=float)
    plan = None
    if part in ("all", "grasp"):
        plan = core.plan_pregrasp_grasp(
            demo.keypoints,
            demo.t,
            demo.tips,
            demo.q_cmd[:, gi],
            demo.q_obs,
            _delta_base(demo, delta_cam, t_bc) if aim is None else aim @ _delta_base(demo, delta_cam, t_bc),
            kin.forward_kinematics(q_now),
            float(q_now[gi]),
            limits[0],
            limits[1],
            jog.GRIP_UNITS_S,
            speed,
            jog.HZ,
            skip,
        )
    placing = part in ("all", "place") and target_base is not None and _place_object(demo) is not None
    if placing:
        then = core.plan_place(
            demo.keypoints,
            demo.t,
            demo.tips,
            demo.q_cmd[:, gi],
            demo.q_obs,
            target_base,
            np.eye(4) if hold_fix is None else hold_fix,
            kin.forward_kinematics(q_now) if plan is None else plan["poses"][-1],
            float(q_now[gi]) if plan is None else float(plan["grips"][-1]),
            limits[0],
            limits[1],
            speed,
            jog.HZ,
            skip if part == "place" else 0,
        )
        plan = then if plan is None else core.join_plans(plan, then)
    assert plan is not None, "a grasp or a place to plan"
    sol = core.solve_plan_joints(
        kin,
        plan["poses"],
        plan["grips"],
        plan["hints"],
        q_now,
        gi,
        *(ranges if ranges is not None else (None, None)),
    )
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
    pre = (
        sorted(float(k["t"]) for k in demo.keypoints if k["kind"] == "pregrasp")[skip:]
        if part != "place"
        else []
    )
    pre_place = (
        sorted(float(k["t"]) for k in demo.keypoints if k["kind"] == "preplace")[
            skip if part == "place" else 0 :
        ]
        if placing
        else []
    )
    first_place = 1 + (skip if part == "place" else 0)
    labels = [f"pre-grasp {n}" for n in range(skip + 1, skip + 1 + len(pre))] + [
        f"pre-place {n}" for n in range(first_place, first_place + len(pre_place))
    ]
    picked, onto = _marks_object(demo), _place_object(demo)
    for n, (label, tk, idx) in enumerate(zip(labels, pre + pre_place, plan["arrive"], strict=True)):
        # How far the object's move carried this mark from where the demo's arm was, and how much of that is away from
        # the arm's base: what puts a mark the demo reached out of reach now.
        was = demo.tips[int(np.argmin(np.abs(demo.t - tk)))][:3, 3]
        now = np.asarray(plan["poses"][idx], dtype=float)[:3, 3]
        marks.append(
            {
                "label": label,
                "t": tk,
                "residual_mm": float(sol["residual_m"][idx] * 1000.0),
                "ok": bool(fine[idx]),
                "object": picked if n < len(pre) else onto,
                "moved_mm": float(np.linalg.norm(now - was) * 1000.0),
                "out_mm": float((np.linalg.norm(now[:2]) - np.linalg.norm(was[:2])) * 1000.0),
            }
        )
    for span, label, kind in (
        (plan.get("grasp"), "grasp", "grasp_end"),
        (plan.get("place"), "place", "place_end"),
    ):
        if span is None:
            continue
        a, b = span
        marks.append(
            {
                "label": label,
                "t": next(float(k["t"]) for k in demo.keypoints if k["kind"] == kind),
                "residual_mm": float(sol["residual_m"][a : b + 1].max() * 1000.0),
                "ok": bool(fine[a : b + 1].all()),
            }
        )
    reason = ""
    bad = [m for m in marks if not m["ok"]]
    far = np.flatnonzero(~fine)
    held = np.flatnonzero(~fine & (sol["held"] >= 0))
    jumps = np.flatnonzero(sol["step_deg"] > core.ACT_MAX_JOINT_STEP_DEG)
    # Out of reach for a joint held at its servo's range: say which, and how far the stage would need it.
    if len(held) and ranges is not None:
        n = max(  # the farthest that stage needs the joint, not just where it first leaves the range
            (int(k) for k in held if plan["stage"][k] == plan["stage"][held[0]]),
            key=lambda k: max(
                sol["residual_m"][k] / core.ACT_REACH_TOL_M, sol["residual_deg"][k] / core.ACT_REACH_TOL_DEG
            ),
        )
        j = int(sol["held"][n])
        lo_j, hi_j = float(ranges[0][j]), float(ranges[1][j])
        free = core.solve_plan_joints(
            kin,
            plan["poses"][n : n + 1],
            plan["grips"][n : n + 1],
            np.zeros((1, len(q_now))),
            sol["q"][n],
            gi,
        )
        reason = (
            f"{plan['stage'][n]} needs {MOTOR_NAMES[j]} at {free['q'][0, j]:.0f} deg, past its servo's range of "
            f"{lo_j:.0f} to {hi_j:.0f} deg"
        )
    elif bad:
        m = bad[0]
        reason = f"{m['label']} is out of reach as the object lies now ({m['residual_mm']:.0f} mm short)"
        if m.get("object") and m.get("moved_mm") is not None:
            reason += (
                f": {m['object']} lies where it carries {m['label']} {m['moved_mm']:.0f} mm from where the demo's arm was"
                f" there, {m['out_mm']:.0f} mm farther out from the arm's base"
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


def _grasp_pose_fix(demo_tip: np.ndarray, object_motion: np.ndarray, live_tip: np.ndarray) -> np.ndarray:
    """The hold's change when the held object cannot be seen: ``demo_tip^-1 . object_motion^-1 . live_tip``.

    ``demo_tip`` is the demo's fingertip at its firm grip, ``object_motion`` the object's motion since the demo that
    the grasp was planned with (base frame), ``live_tip`` the fingertip where the arm actually stands at the grip.
    The grasp was aimed at ``object_motion . demo_tip``, so the result is the identity when the arm landed there,
    and otherwise how far it missed, on the gripper's side. Whatever the closing fingers or the lift did to the
    object is not in it.
    """
    return np.linalg.inv(demo_tip) @ np.linalg.inv(object_motion) @ live_tip


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


INJECT_MAX_MM = 30.0  # an injected error stays a test of the grasp, not a way to drive the arm elsewhere
INJECT_MAX_DEG = 20.0


class InjectBody(BaseModel):
    """An error to inject into one act, base frame: moved by dx, dy, dz and turned by rx, ry, rz about the base's
    axes. ``at`` "aim": the grasp is aimed off, turned about the grasp point, as an arm that misses its plan; the
    act knows where it aimed. "find": the object's found pose is wrong, turned about its centre; the act believes
    it, as a find that erred."""

    at: Literal["aim", "find"] = "aim"
    dx_mm: float = Field(0.0, ge=-INJECT_MAX_MM, le=INJECT_MAX_MM)
    dy_mm: float = Field(0.0, ge=-INJECT_MAX_MM, le=INJECT_MAX_MM)
    dz_mm: float = Field(0.0, ge=-INJECT_MAX_MM, le=INJECT_MAX_MM)
    rx_deg: float = Field(0.0, ge=-INJECT_MAX_DEG, le=INJECT_MAX_DEG)
    ry_deg: float = Field(0.0, ge=-INJECT_MAX_DEG, le=INJECT_MAX_DEG)
    rz_deg: float = Field(0.0, ge=-INJECT_MAX_DEG, le=INJECT_MAX_DEG)


class ActBody(BaseModel):
    speed: float = (
        1.0  # scales the straight lines (from the jog's walk speed) and the grasp (from the demo's clock)
    )
    inject: InjectBody | None = None
    correct_hold: bool = True  # False: a place replays the demo against the target, uncorrected for the hold


def _grasp_index(demo: _Demo) -> int:
    """The demo sample where its fingertip grips the object: the firm grip, else the grasp's end, else the last
    pre-grasp. Pre: a pre-grasp is marked."""
    i = _firm_grip(demo)
    if i is not None:
        return i
    marks = [float(k["t"]) for k in demo.keypoints if k["kind"] in ("pregrasp", "grasp_end")]
    return int(np.argmin(np.abs(demo.t - max(marks))))


def _injects(inject: InjectBody) -> bool:
    """Whether ``inject`` moves or turns anything."""
    return any(getattr(inject, k) for k in ("dx_mm", "dy_mm", "dz_mm", "rx_deg", "ry_deg", "rz_deg"))


def _inject_transform(inject: dict[str, Any], pivot: np.ndarray) -> np.ndarray:
    """The injected error as a base-frame motion: turned by ``rx, ry, rz`` (degrees, about the base's x, y, z) about
    ``pivot``, then moved by ``dx, dy, dz`` (mm)."""
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_euler(
        "xyz", [float(inject.get(k, 0.0)) for k in ("rx_deg", "ry_deg", "rz_deg")], degrees=True
    ).as_matrix()
    p = np.asarray(pivot, dtype=float)[:3]
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = (
        p - rot @ p + np.array([float(inject.get(k, 0.0)) for k in ("dx_mm", "dy_mm", "dz_mm")]) / 1000.0
    )
    return out


async def _act_task(speed: float) -> None:
    """The act: follow the object to each pre-grasp, wait for it to hold still, then replay the grasp 1:1; with a
    place marked, then carry it to the object it goes onto and place it.

    What the act finds runs beside the arm, not before it. A designated object's
    track is started over from a fresh find where it was last seen (a track kept since
    an earlier find drifts), a place's object is found again, and how the demo held
    the picked object is measured from the demo's still frames, all while the arm
    walks toward the first pre-grasp on the track it has; each is awaited at the last
    pre-grasp, before anything is grasped. Only an object with no live track is found
    before the arm moves.

    The whole act is planned and judged from the arm's present joints before
    anything moves. The approach runs on the jog's walk, re-aimed at every new
    tracker view, so the straight lines bend toward an object that is moved. At the
    last pre-grasp the arm waits until the object holds still, or until the gripper
    covers it, and the grasp is planned from there and streamed as joint targets.
    Without tracking the act runs from the one view it started with.

    The grasp streams through without a pause. With a place, what the grip shows is read
    on the fly: when the gripper's reading stops, the gripper must be short of its
    command, as an object between the fingers stops it (a miss halts the stream there);
    the fingertip then is the grasp pose; and while the arm stands still, as the demo's
    did, views of the object in the gripper are taken and read while the act goes on.
    After the lift the gripper is checked again. The arm walks to each pre-place,
    carried by the place object's motion; at the last it stands still while the held
    object is found in the gripper a few times, goes to the pre-place corrected for how
    the object sits there now, and the place is streamed like the grasp, release
    included.
    """
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    from . import jog

    act = _state.act

    def fail(reason: str) -> None:
        act.ok, act.reason, act.step = False, reason, "aborted"

    def q_dict(q: np.ndarray) -> dict[str, float]:
        return {m: float(q[k]) for k, m in enumerate(MOTOR_NAMES)}

    def stopped() -> bool:
        return act.stop_requested

    halt = ""  # why the act stops, found on the fly by something running beside the stream

    def interrupted() -> str:
        st = jog.current_status()
        if act.stop_requested:
            return "stopped"
        if halt:
            return halt
        if not st.get("connected"):
            return "the arm went away"
        if st.get("halted"):
            return f"arm frozen: {st.get('reason')}"
        return ""

    moved = streaming = False
    sent = -1  # the last sample of the plan being streamed that went to the arm
    limits_before: tuple[float, float] | None = None
    run: _Run | None = None
    run_dir: str | None = None
    beside: list[asyncio.Task] = []  # what runs beside the arm; cancelled however the act ends
    try:
        with _state.lock:
            demo, test, teach, tracking = _state.demo, _state.test, _state.teach, _state.track.on
        if demo is None:
            fail("record or load a demo first")
            return
        if not _has_pregrasp(demo):
            fail("mark a pre-grasp first")
            return
        place_obj, obj = _place_object(demo), _marks_object(demo)
        with _state.lock:
            act.find_error = None
            live = bool(
                tracking
                and (_state.track.last or {}).get("state") == "tracking"
                and test is not None
                and test.result.get("ok")
            )
        live = (
            live and not _reference_motion(demo, teach)[1]
        )  # a track to start from, against the demo's view
        # With the picked object's find to follow, the place's object joins its session there, not on its own: one
        # restart of the session for both.
        relocate = (
            asyncio.create_task(_locate_afresh(place_obj, stopped, track=not (obj is not None and live)))
            if place_obj is not None
            else None
        )
        if relocate is not None:
            beside.append(relocate)
        if obj is not None and not live:  # nothing to start from: the find comes first
            act.step = f"finding {obj}"
            why = await _find_afresh(obj, stopped)
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
        ranges = jog.servo_ranges()
        landing_turn: tuple[np.ndarray, float] | None = (
            None  # the middle and the turn the place lands at, once planned
        )
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
        target_base = demo_grip = demo_carry = None
        grip_problem = carry_problem = ""
        holds: asyncio.Task | None = None
        if place_obj is not None:
            assert obj is not None and relocate is not None, "a place names the object picked"
            target_base, problem = _target_motion(demo, t_bc)
            if problem:  # the last find is no use: this one is needed before anything moves
                act.step = f"finding {place_obj}"
                why = await relocate
                if why:
                    fail(why)
                    return
                target_base, problem = _target_motion(demo, t_bc)
                if problem:
                    fail(problem)
                    return

            async def demo_holds() -> tuple[tuple[dict | None, str], tuple[dict | None, str]]:
                return (
                    await _demo_hold(demo, obj, t_bc, stopped, "grip"),
                    await _demo_hold(demo, obj, t_bc, stopped, "carry"),
                )

            holds = asyncio.create_task(demo_holds())
            beside.append(holds)
        gi = MOTOR_NAMES.index("gripper")
        delta = np.asarray(test.result["delta_cam"], dtype=float)
        inject = act.inject
        if inject and inject.get("at") == "find":
            # The act believes the object turned about its centre and moved: everything it does follows that belief.
            truth = _delta_base(demo, delta, t_bc)
            centre = (
                _object_points(demo, obj, t_bc).mean(axis=0)
                if obj is not None and obj in demo.objects
                else demo.tips[_grasp_index(demo)][:3, 3]
            )
            with _state.lock:
                act.find_error = _inject_transform(inject, truth[:3, :3] @ centre + truth[:3, 3])
        grasp_at = _grasp_index(demo)

        def aim_error(d: np.ndarray) -> np.ndarray | None:
            """The injected miss of the approach and grasp, turned about where the fingertip grips the object as it
            lies now; None without one."""
            if not inject or inject.get("at") != "aim":
                return None
            return _inject_transform(inject, (_delta_base(demo, d, t_bc) @ demo.tips[grasp_at])[:3, 3])

        seen_at = max((w for w, _ in _certified_since(None)), default=0.0)
        limits_before = jog.walk_limits()
        act.step = "planning"
        plan = await asyncio.get_event_loop().run_in_executor(
            _ACT_EXECUTOR,
            functools.partial(_plan_act, aim=aim_error(delta), landing=demo.landing, ranges=ranges),
            demo,
            delta,
            t_bc,
            kin,
            np.array([float(cur[2][m]) for m in MOTOR_NAMES]),
            limits_before,
            jog.workspace_box(),
            speed,
            0,
            target_base,
        )
        act.plan = {k: plan[k] for k in ("ok", "reason", "marks", "summary", "landing") if k in plan}
        if not plan["ok"]:
            fail(plan["reason"])
            return
        if (plan.get("landing") or {}).get("turn_deg") is not None:
            # The place lands at this turn from here on: the carry, the hold's check and the place all aim by it.
            landing_turn = (plan["landing_centre"], float(plan["landing"]["turn_deg"]))
        pre = sorted(float(k["t"]) for k in demo.keypoints if k["kind"] == "pregrasp")
        idx = [int(np.argmin(np.abs(demo.t - tk))) for tk in pre]
        jog.set_walk_limits(limits_before[0] * speed, limits_before[1] * speed)
        moved = True
        run = _begin_run(demo, speed, delta, t_bc, plan)
        run.meta["inject"], run.meta["correct_hold"] = inject, act.correct_hold
        # The walk starts on the track the act began with; its find is started over beside the arm, and until the
        # fresh track certifies a view the old one's last motion stands (its views are against the old teach).
        motion0 = _delta_base(demo, delta, t_bc)

        async def refind_with_place() -> str:
            """The picked object's session started over from a fresh find, the place's object (found just before, where
            its track last had it) joining it from the start."""
            if relocate is not None:
                why = await relocate
                if why:
                    return why
            nonlocal delta, seen_at
            found = _located(demo, place_obj) if place_obj is not None else None
            more = (
                {place_obj: found["mask"]}
                if found and found["ok"] and found.get("mask") is not None
                else None
            )
            # The arm is on its way by now and may cover the object from the camera, so the restarted track is not
            # waited for: the find's own view is where the object is (no motion since it, against the new teach)
            # until the track sees it again.
            why = await _find_afresh(obj, stopped, more, need_track=False)
            if not why:  # views from before the new teach was applied are against the old one
                delta, seen_at = np.eye(4), time.time()
                if more:
                    _store_located(place_obj, {**found, "tracking": True})
            return why

        refind = asyncio.create_task(refind_with_place()) if obj is not None and live else None
        if refind is not None:
            beside.append(refind)

        def follow() -> np.ndarray:
            """The newest certified view of the object, or the last one used."""
            nonlocal seen_at, delta
            if tracking:
                newer = _certified_since(seen_at)
                if newer:
                    seen_at, delta = newer[-1]
            return delta

        def approach() -> np.ndarray:
            """The object's motion for the walk to the pre-grasps, base frame: the track the act began with until its
            fresh find is in, then the fresh one's newest view."""
            if refind is not None and not refind.done():
                return motion0
            return _delta_base(demo, follow(), t_bc)

        async def walk_to(label: str, aim: Callable[[], np.ndarray]) -> str:
            """Walk the arm to ``aim()``, asked again every tick, until it arrives: "" or why not. Arrived is within
            ACT_ARRIVE_M of it, or, as for a stream's end, with the commanded joints on it and the arm stopped with
            every joint within ACT_STALL_DEG of its command: held short by the servos, which waiting does not close (an
            act of 2026-10-08 stood 4.9 mm off its pre-place for 20 s, each joint within 0.8 deg of its command). A
            solve that stops short of the target leaves the command off it, and the walk times out as before."""
            act.step = label
            t0 = time.monotonic()
            kin = jog.kinematics()
            arm = [m for m in MOTOR_NAMES if m != "gripper"]
            seen: list[tuple[float, np.ndarray]] = []
            while True:
                why = interrupted()
                if why:
                    return why
                target = aim()
                jog.set_target_pose(target)
                run.target(act.step, pose=target)
                cur, st = jog.current_tip_and_anchor(), jog.current_status()
                if cur is not None and not st.get("holding"):
                    e_m, e_deg = core.pose_residual(cur[0], target)
                    if e_m <= ACT_ARRIVE_M and e_deg <= core.ACT_REACH_TOL_DEG:
                        return ""
                    if kin is not None and st.get("q_cmd") and st.get("q_obs"):
                        now = time.monotonic()
                        q_obs = np.array([st["q_obs"][m] for m in arm])
                        seen = [(w, v) for w, v in seen if now - w <= ACT_STILL_S] + [(now, q_obs)]
                        stopped = (
                            now - seen[0][0] >= 0.8 * ACT_STILL_S
                            and np.ptp([v for _w, v in seen], axis=0).max() < ACT_STILL_DEG
                        )
                        lag = max(abs(st["q_cmd"][m] - st["q_obs"][m]) for m in arm)
                        c_m, c_deg = core.pose_residual(
                            kin.forward_kinematics(np.array([st["q_cmd"][m] for m in MOTOR_NAMES])), target
                        )
                        commanded = c_m <= ACT_ARRIVE_M and c_deg <= core.ACT_REACH_TOL_DEG
                        if stopped and lag <= ACT_STALL_DEG and commanded:
                            shorts = {
                                **(act.place or {}).get("walk_short_mm", {}),
                                label: round(e_m * 1000.0, 1),
                            }
                            act.place = {**(act.place or {}), "walk_short_mm": shorts}
                            return ""
                if time.monotonic() - t0 > ACT_STEP_TIMEOUT_S:
                    return f"{label}: not there after {ACT_STEP_TIMEOUT_S:.0f} s"
                await asyncio.sleep(ACT_TICK_S)

        async def stream(planned: dict[str, Any]) -> str:
            """Stream a plan's joints, then wait for the arm to settle on the last: "" or why not. The jog's loop plays
            them back on its own clock (:func:`jog.play_joints`): every sample in order, none before its time, a slow
            tick delaying the rest instead of losing samples; ``sent`` follows the samples that went out."""
            nonlocal streaming, sent
            q, times, stage = planned["q"], planned["times"], planned["stage"]
            sent = -1
            try:
                await jog.joints_start(q_dict(q[0]))
            except RuntimeError as e:
                return str(e)
            streaming = True
            n = len(q)
            try:
                jog.play_joints([q_dict(x) for x in q], [float(t) for t in times])
            except RuntimeError as e:
                return str(e)
            while sent < n - 1:
                await asyncio.sleep(ACT_TICK_S)
                why = interrupted()
                if why:
                    jog.playback_stop()
                    return why
                done, at = jog.playback_state()
                for i in range(sent + 1, done + 1):
                    run.target(stage[i], joints=q[i], at=at[i])
                if done > sent:
                    sent = done
                    act.step = stage[done]
                    act.progress = (done + 1) / n
            act.step = "settling"
            t0 = time.monotonic()
            arm = [k for k, m in enumerate(MOTOR_NAMES) if m != "gripper"]
            seen: list[tuple[float, np.ndarray]] = []
            while True:
                await asyncio.sleep(ACT_TICK_S)
                cur = jog.current_tip_and_anchor()
                why = interrupted()
                if why or cur is None:
                    return why or "the arm went away"
                q_now = np.array([float(cur[2][MOTOR_NAMES[k]]) for k in arm])
                lag = float(np.abs(q_now - np.asarray(q[-1], dtype=float)[arm]).max())
                if lag <= ACT_SETTLE_DEG:
                    return ""
                now = time.monotonic()
                seen = [(w, v) for w, v in seen if now - w <= ACT_STILL_S] + [(now, q_now)]
                stopped = (
                    now - seen[0][0] >= 0.8 * ACT_STILL_S
                    and np.ptp([v for _w, v in seen], axis=0).max() < ACT_STILL_DEG
                )
                if stopped and lag <= ACT_STALL_DEG:  # held short under load: waiting will not close it
                    act.place = {**(act.place or {}), "settled_short_deg": lag}
                    return ""
                if stopped:
                    return f"the end: the arm stopped {lag:.0f} deg short of it"
                if now - t0 > ACT_STEP_TIMEOUT_S:
                    return f"the end: not there after {ACT_STEP_TIMEOUT_S:.0f} s ({lag:.0f} deg off)"

        async def gripper_still() -> float | None:
            """The gripper's reading once it has stopped moving, as a closing on the object or on nothing ends."""
            g_obs, t0 = jog.current_gripper(), time.monotonic()
            while g_obs is not None and time.monotonic() - t0 < GRIP_SETTLE_S:
                await asyncio.sleep(0.1)
                g_prev, g_obs = g_obs, jog.current_gripper()
                if g_obs is not None and abs(g_obs - g_prev) < core.GRIP_STILL_UNITS:
                    break
            return g_obs

        def grasp_missed(cmd: float, g_obs: float, i_demo: int, closing: float) -> str:
            """Why the grasp missed, or "": closing on nothing reaches the command, an object stops it short."""
            held, short = core.grasp_held(
                cmd, g_obs, float(demo.q_cmd[i_demo, gi]), float(demo.q_obs[i_demo, gi]), closing
            )
            act.place = {**(act.place or {}), "grasp_held": held, "grasp_short": short}
            if held is not False:
                return ""
            return (
                f"the grasp missed: the gripper closed to {g_obs:.1f}, {short:.1f} short of its command; "
                f"an object between the fingers stops it more than {core.GRASP_HELD_SHORT:.1f} short"
            )

        async def still() -> str:
            """Wait until the fingertip has moved less than :data:`core.HOLD_STILL_M_S` allows over half a second: ""
            or why not. Over a single tick, one encoder count of jitter would read as motion."""
            window_s, seen, t0 = 0.5, [], time.monotonic()
            while True:
                why = interrupted()
                if why:
                    return why
                cur = jog.current_tip_and_anchor()
                if cur is None:
                    return "the arm went away"
                now = time.monotonic()
                seen = [(w, p) for w, p in seen if now - w <= window_s] + [(now, cur[0][:3, 3].copy())]
                span = now - seen[0][0]
                if (
                    span >= 0.8 * window_s
                    and np.linalg.norm(seen[-1][1] - seen[0][1]) / span < core.HOLD_STILL_M_S
                ):
                    return ""
                if now - t0 > ACT_STEP_TIMEOUT_S:
                    return f"the arm did not hold still for {ACT_STEP_TIMEOUT_S:.0f} s"
                await asyncio.sleep(ACT_TICK_S)

        for n, i in enumerate(idx, start=1):
            g = float(demo.q_cmd[i, gi])
            g_now = jog.current_gripper()
            jog.set_gripper(g)
            act.step = f"pre-grasp {n}: setting the gripper"
            await asyncio.sleep(abs(g - (g if g_now is None else g_now)) / jog.GRIP_UNITS_S + ACT_TICK_S)
            why = await walk_to(f"pre-grasp {n}", lambda i=i: approach() @ demo.tips[i])
            if why:
                fail(why)
                return
        if not any(k["kind"] == "grasp_end" for k in demo.keypoints):
            act.step, act.ok = "done", True
            return
        # What ran beside the arm is needed from here: the fresh track for the grasp; the place's object and the
        # demo's holds for the place. Each is waited for only if it is not in yet.
        if refind is not None:
            act.step = f"finding {obj}"
            why = await refind
            if why:
                fail(why)
                return
        if relocate is not None:
            act.step = f"finding {place_obj}"
            why = await relocate
            if why:
                fail(why)
                return
            target_base, problem = _target_motion(demo, t_bc)
            if problem:
                fail(problem)
                return
        if place_obj is not None:  # what the place is aimed by: the target's find, and the motion it gives
            found = _located(demo, place_obj) or {}
            run.meta["target"] = {
                **_located_info(found),
                "delta": found.get("delta"),
                "motion_base": target_base,
                "pose_frame": _pose_frame(demo, place_obj, "preplace"),
                "landing": plan.get("landing"),
                "landing_centre": None if landing_turn is None else landing_turn[0],
            }
        if holds is not None:
            act.step = f"measuring how the demo holds {obj}"
            (demo_grip, grip_problem), (demo_carry, carry_problem) = await holds
            if stopped():
                fail("stopped")
                return
            if demo_grip is None and demo_carry is None and _firm_grip(demo) is None:
                fail(
                    f"how the demo holds {obj} cannot be known: no firm grip in the demo; at its grip, "
                    f"{grip_problem}; while carried, {carry_problem}"
                )
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
            functools.partial(_plan_act, aim=aim_error(delta), ranges=ranges),
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
        # With a place, what the grip shows is read on the fly while the grasp streams on, never in a pause: the
        # stream's sample where the demo's grip became firm is where the watcher starts.
        i_end = int(
            np.argmin(
                np.abs(demo.t - next(float(k["t"]) for k in demo.keypoints if k["kind"] == "grasp_end"))
            )
        )
        closing = float(np.sign(demo.q_cmd[i_end, gi] - demo.q_cmd[idx[-1], gi])) or 1.0
        grip_i = _firm_grip(demo) if place_obj is not None else None
        split = None
        if grip_i is not None and "grasp" in grasp["stage"]:
            split = list(grasp["stage"]).index("grasp") + (grip_i - idx[-1] - 1)
            split = split if 0 < split < len(grasp["q"]) - 1 else None
        grip: dict[str, Any] = {}

        async def watch_grip(split: int, grip_i: int) -> None:
            """Beside the stream, from its sample where the demo's grip became firm: once the gripper's reading stops,
            the grasp check (a miss halts the stream) and the fingertip then, the grasp pose; with the demo's grip
            showing the object, views of it in the gripper while the arm stands still, as the demo's did, read while
            the act goes on. Post: ``grip`` holds ``fix`` and ``live`` as far as they were measured."""
            nonlocal halt
            while sent < split:
                if interrupted():
                    return
                await asyncio.sleep(ACT_TICK_S)
            g = g_prev = jog.current_gripper()
            t_prev = t0 = time.monotonic()
            while g is not None and time.monotonic() - t0 < GRIP_SETTLE_S:
                await asyncio.sleep(ACT_TICK_S)
                g = jog.current_gripper()
                if g is not None and time.monotonic() - t_prev >= core.GRIP_STILL_S:
                    if abs(g - g_prev) < core.GRIP_STILL_UNITS:
                        break
                    g_prev, t_prev = g, time.monotonic()
            cur = jog.current_tip_and_anchor()
            if g is None or cur is None or interrupted():
                return
            why = grasp_missed(float(grasp["q"][split][gi]), g, grip_i, closing)
            if why:
                halt = why
                return
            # Without seeing the object: the grasp was aimed by its estimated pose, so the hold is the demo's, changed
            # by however far the arm landed from that aim; what the closing fingers did is not seen.
            grip["fix"] = _grasp_pose_fix(demo.tips[grip_i], _delta_base(demo, delta, t_bc), cur[0])
            if demo_grip is None:
                return
            views: list[Any] = []
            last, t_last = cur[0][:3, 3].copy(), time.monotonic()
            while len(views) < core.HOLD_VIEWS + 2:
                await asyncio.sleep(HOLD_VIEW_GAP_S)
                view = await _held_view()
                if view is None or interrupted():
                    break
                now = time.monotonic()
                if np.linalg.norm(view[3][:3, 3] - last) / max(now - t_last, 1e-6) > core.HOLD_STILL_M_S:
                    break  # the lift has begun
                views.append(view)
                last, t_last = view[3][:3, 3].copy(), now
            if len(views) < core.HOLD_MIN_VIEWS:
                grip["live"] = (None, f"not measured: the arm stood still at the grip for {len(views)} views")
                return
            grip["live"] = await _live_hold(demo, obj, demo_grip["hold"], t_bc, stopped, views=views)

        grip_watch = None
        if split is not None:
            grip_watch = asyncio.create_task(watch_grip(split, grip_i))
            beside.append(grip_watch)
        why = await stream(grasp)
        if why:
            fail(why)
            return
        if place_obj is None:
            act.step, act.ok = "done", True
            return

        # Still held after the lift: a drop closes the fingers onto nothing.
        assert target_base is not None, "found before anything moved"
        target_at_grasp = target_base

        def target_now() -> np.ndarray:
            """The place object's motion as its track has it now, base frame, the last one when it has no newer, turned
            by the landing the plan took (:func:`core.landed`): what the carry and the place are aimed by."""
            nonlocal target_base
            motion, _problem = _target_motion(demo, t_bc)
            if motion is not None:
                target_base = motion
            return target_base if landing_turn is None else core.landed(target_base, *landing_turn)

        act.step = "checking the grasp"
        g_obs = await gripper_still()
        if g_obs is None:
            fail("the arm went away")
            return
        why = grasp_missed(float(grasp["q"][-1][gi]), g_obs, i_end, closing)
        if why:
            fail(why)
            return
        await jog.joints_stop()  # the grasp's closing kept
        streaming = False
        pidx = [
            int(np.argmin(np.abs(demo.t - tk)))
            for tk in sorted(float(k["t"]) for k in demo.keypoints if k["kind"] == "preplace")
        ]
        # The carry streams joints planned from where the arm stands, as the grasp and the place do. A walk solves its
        # target tick by tick from the arm's own joints and can end in another arm configuration than the plan checked:
        # an act of 2026-10-08 walked to its first pre-place asking wrist_flex for -101 deg, past the servo's 93, where
        # the plan of the same line had -90.
        act.step = "planning the carry"
        cur = jog.current_tip_and_anchor()
        if cur is None:
            fail("the arm went away")
            return
        carry = await asyncio.get_event_loop().run_in_executor(
            _ACT_EXECUTOR,
            functools.partial(_plan_act, ranges=ranges),
            demo,
            None,
            t_bc,
            kin,
            np.array([float(cur[2][m]) for m in MOTOR_NAMES]),
            limits_before,
            jog.workspace_box(),
            speed,
            0,
            target_now(),
            None,
            "place",
        )
        act.plan = {k: carry[k] for k in ("ok", "reason", "marks", "summary")}
        if not carry["ok"]:
            fail(carry["reason"])
            return
        last = max(k for k, s in enumerate(carry["stage"]) if s.startswith("pre-place"))
        q_carry = np.array(carry["q"][: last + 1], dtype=float)
        q_carry[:, gi] = float(
            grasp["q"][-1][gi]
        )  # the grasp's closing: on the object the fingers stand short of it
        why = await stream(
            {"q": q_carry, "times": carry["times"][: last + 1], "stage": carry["stage"][: last + 1]}
        )
        if why:
            fail(why)
            return
        await jog.joints_stop()  # the walk again for the hold's correction, the closing kept
        streaming = False
        # The pre-place re-check: the hold as it is now, after the lift and the carry, when both the demo's carry and
        # the live views show enough of the object; otherwise the hold measured at the grip.
        live_place, live_place_problem = None, "not measured: the demo's carry shows too little of it"
        if demo_carry is not None:
            act.step = f"checking how {obj} sits in the gripper"
            why = await still()
            if why:
                fail(f"at pre-place {len(pidx)}: {why}")
                return
            live_place, live_place_problem = await _live_hold(
                demo, obj, demo_carry["hold"], t_bc, lambda: act.stop_requested
            )
            if act.stop_requested:
                fail("stopped")
                return
        if grip_watch is not None:
            await grip_watch
        fix_grasp = grip.get("fix")
        live_grip, live_grip_problem = grip.get(
            "live", (None, "not measured: the demo's grip shows too little of it")
        )
        fix_grip = (
            demo_grip["hold"] @ np.linalg.inv(live_grip["hold"])
            if demo_grip is not None and live_grip is not None
            else None
        )
        fix_place = (
            demo_carry["hold"] @ np.linalg.inv(live_place["hold"])
            if demo_carry is not None and live_place is not None
            else None
        )
        if fix_place is None and fix_grip is None and fix_grasp is None and act.correct_hold:
            fail(
                f"how {obj} sits in the gripper cannot be known: at the grip, "
                f"{grip_problem or live_grip_problem or 'no firm grip in the demo'}; at the pre-place, "
                f"{carry_problem or live_place_problem}"
            )
            return
        fix, used = (
            next(
                (f, name)
                for f, name in ((fix_place, "pre-place"), (fix_grip, "grip"), (fix_grasp, "grasp pose"))
                if f is not None
            )
            if act.correct_hold
            else (np.eye(4), "off")
        )
        aimed = target_now() @ demo.tips[pidx[-1]]
        shift_m, shift_deg = core.pose_residual(aimed @ fix, aimed)

        def summary(hold: dict | None, problem: str) -> dict[str, Any] | str:
            return (
                problem if hold is None else {k: hold[k] for k in ("n", "views", "spread_mm", "spread_deg")}
            )

        act.place = {
            **(act.place or {}),
            "hold_used": used,
            "demo_grip": summary(demo_grip, grip_problem),
            "live_grip": summary(live_grip, live_grip_problem),
            "demo_carry": summary(demo_carry, carry_problem),
            "live_place": summary(live_place, live_place_problem),
            "shift_mm": shift_m * 1000.0,
            "shift_deg": shift_deg,
            "fix": fix.tolist(),
        }
        if (
            fix_grasp is not None
        ):  # what the camera saw against the grasp pose: what the closing and the lift did
            act.place["grasp_pose_mm"], act.place["grasp_pose_deg"] = (
                v * k
                for v, k in zip(core.pose_residual(aimed @ fix_grasp, aimed), (1000.0, 1.0), strict=True)
            )
            if used != "grasp pose":
                seen_m, seen_deg = core.pose_residual(aimed @ fix, aimed @ fix_grasp)
                act.place.update(seen_vs_grasp_pose_mm=seen_m * 1000.0, seen_vs_grasp_pose_deg=seen_deg)
        if fix_grip is not None and fix_place is not None:  # how far the lift and the carry moved the hold
            moved_m, moved_deg = core.pose_residual(aimed @ fix_grip, aimed @ fix_place)
            act.place.update(grip_vs_place_mm=moved_m * 1000.0, grip_vs_place_deg=moved_deg)
        why = await walk_to(
            f"pre-place {len(pidx)}, corrected for the hold", lambda: target_now() @ demo.tips[pidx[-1]] @ fix
        )
        if why:
            fail(why)
            return
        if not any(k["kind"] == "place_end" for k in demo.keypoints):
            act.step, act.ok = "done", True
            return
        act.step = "planning the place"
        cur = jog.current_tip_and_anchor()
        if cur is None:
            fail("the arm went away")
            return
        placing = await asyncio.get_event_loop().run_in_executor(
            _ACT_EXECUTOR,
            functools.partial(
                _plan_act, ranges=ranges
            ),  # the landing as taken: target_now() is turned already
            demo,
            None,
            t_bc,
            kin,
            np.array([float(cur[2][m]) for m in MOTOR_NAMES]),
            limits_before,
            jog.workspace_box(),
            speed,
            len(pidx) - 1,
            target_now(),
            fix,
            "place",
        )
        end_at = int(
            np.argmin(
                np.abs(demo.t - next(float(k["t"]) for k in demo.keypoints if k["kind"] == "place_end"))
            )
        )
        act.place["target_moved_mm"] = (
            1000.0
            * float(  # how far the place object's track moved the place since the grasp
                np.linalg.norm(
                    (target_base @ demo.tips[end_at])[:3, 3] - (target_at_grasp @ demo.tips[end_at])[:3, 3]
                )
            )
        )
        act.plan = {k: placing[k] for k in ("ok", "reason", "marks", "summary")}
        if not placing["ok"]:
            fail(placing["reason"])
            return
        why = await stream(placing)
        if why:
            fail(why)
            return
        act.step, act.ok = "done", True
    except Exception as e:  # the arm holds its last target; the operator sees why
        logger.exception("act failed")
        fail(f"act error: {e}")
    finally:
        for task in beside:
            task.cancel()
        with _state.lock:
            act.find_error = None
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
        act.plan = act.place = None
        act.inject = body.inject.model_dump() if body.inject is not None and _injects(body.inject) else None
        act.correct_hold = body.correct_hold
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
