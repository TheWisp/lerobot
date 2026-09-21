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
    tip_pose: np.ndarray | None = None  # base frame, 4x4
    gripper: float | None = None  # opening at Mark, the follower's 0..100 units


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
class _State:
    lock: threading.Lock = field(default_factory=threading.Lock)
    teach: _Teach | None = None
    test: _Test | None = None
    worker: _Worker = field(default_factory=_Worker)
    teach_job: str | None = None  # a features teach awaiting its result
    find_job: str | None = None


_state = _State()
_HEAVY = ("delta_cam", "teach_uv", "live_uv", "live_mask")


class TeachBody(BaseModel):
    box: list[int] = []  # x0, y0, x1, y1 in frame pixels (box mode)
    mode: str = "box"  # "box" | "features" (SAM3 by concept + DINO, in the worker)
    concept: str = ""


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
    w = s.worker
    with s.lock:
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
async def teach_mark() -> dict:
    """Record the fingertip's present pose as the pre-grasp for the taught object."""
    from . import jog

    cur = jog.current_tip_and_anchor()
    if cur is None:
        raise HTTPException(409, "connect an arm in the Jog panel first")
    grip = jog.current_gripper()
    with _state.lock:
        if _state.teach is None:
            raise HTTPException(409, "capture the object first")
        _state.teach.tip_pose = cur[0].copy()
        _state.teach.gripper = grip
        tip = _state.teach.tip_pose[:3, 3]
    return {"tip_mm": (tip * 1000.0).tolist(), "gripper": grip}


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
    if teach.keypoints["mode"] == "features":
        if not _state.worker.running:
            raise HTTPException(409, "start the worker first")
        job = _queue_job("find", teach.keypoints["concept"], rgb, depth_m, intr)
        with _state.lock:
            _state.find_job = job.id
        return {"pending": True, "job": job.id, "mode": "features"}
    if teach.keypoints["mode"] == "texture":
        result = core.register(teach.keypoints, rgb, depth_m, intr)
        if not result.get("ok") and teach.keypoints.get("shape") is not None:
            texture_reason = result.get("reason", "texture failed")
            result = core.shape_register(teach.keypoints["shape"], depth_m, intr, rgb)
            result["fallback_from"] = f"texture ({texture_reason})"
    else:
        result = core.shape_register(teach.keypoints, depth_m, intr, rgb)
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
            _outline(bgr, r["live_mask"], (60, 200, 60) if r.get("mode") != "features" else (255, 0, 255))
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
        test, teach = _state.test, _state.teach
    if test is None or test.transported is None:
        raise HTTPException(409, "no transported pose; capture the test frame first")
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


def _queue_job(kind: str, concept: str, rgb: np.ndarray, depth_m: np.ndarray, intr: dict[str, float]) -> _Job:
    job = _Job(
        id=uuid.uuid4().hex[:8],
        kind=kind,
        concept=concept,
        rgb=rgb,
        depth_m=depth_m,
        intr=intr,
        created=time.time(),
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
                    content=json.dumps({"id": job.id, "kind": job.kind, "concept": job.concept}),
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
    np.savez_compressed(buf, rgb=job.rgb, depth=job.depth_m, intr=np.array(json.dumps(job.intr)))
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
    result: dict[str, Any] = {k: v for k, v in r.items() if k not in ("mask", "uv", "xyz", "delta")}
    result["mode"] = "features"
    if r.get("mask") is not None:
        result["live_mask"] = r["mask"].astype(bool)
    transported = None
    if r.get("ok"):
        result["delta_cam"] = np.asarray(r["delta"])
        try:
            transported = core.transport_pose(_t_base_cam(), result["delta_cam"], teach.tip_pose)
        except HTTPException as e:
            result["ok"] = False
            result["reason"] = e.detail
    with _state.lock:
        _state.test = _Test(at=time.strftime("%H:%M:%S"), rgb=job.rgb, result=result, transported=transported)
