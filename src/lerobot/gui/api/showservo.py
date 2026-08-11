# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Show-and-Servo capture sessions: the GUI face of the real-rig first-contact bench.

The split keeps the server light. Camera work (connect, preview, capture-to-disk) runs
IN PROCESS — it is pyrealsense2 only. The analysis (SAM3 designation + DINOv3 binding)
runs `benchmarks/showservo_real.py` AS A SUBPROCESS over the captured directory, so the
models never load into the server and the GUI exercises exactly the code path the
command line does — same script, same numbers.

Sessions write the same on-disk format as `benchmarks/showservo_capture.py`
(rgb.png + depth.npy float32 metres + preview.jpg per scene, intrinsics.json once);
either tool can continue a directory the other started. One session at a time.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import pathlib
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Dedicated single worker: camera reads are exclusive-device operations that can
# stall for a frame interval — they must never queue behind (or starve) the GUI's
# shared pool. One worker also serialises capture vs preview by construction.
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="showservo")

router = APIRouter(prefix="/api/showservo", tags=["showservo"])

# Repo root (…/src/lerobot/gui/api -> repo). Captures live under <repo>/captures.
_REPO = pathlib.Path(__file__).resolve().parents[4]
_CAPTURES = _REPO / "captures"
_BENCH = _REPO / "benchmarks" / "showservo_real.py"


@dataclass
class _Session:
    out: pathlib.Path
    camera: Any | None = None  # RealSenseCamera when live; None when opened from disk
    scenes: list[dict] = field(default_factory=list)


@dataclass
class _Bind:
    proc: subprocess.Popen | None = None
    log: list[str] = field(default_factory=list)
    done: bool = False
    ok: bool = False


_session: _Session | None = None
_bind: _Bind = _Bind()
_lock = threading.Lock()


def _scene_info(path: pathlib.Path) -> dict:
    """Post: one thumbnail-strip entry; depth stats read lazily from the npy."""
    info: dict[str, Any] = {"name": path.name}
    depth_file = path / "depth.npy"
    if depth_file.exists():
        depth = np.load(depth_file, mmap_mode="r")
        valid = np.asarray(depth) > 0
        info["depth_valid"] = float(valid.mean())
        info["median_m"] = float(np.median(np.asarray(depth)[valid])) if valid.any() else 0.0
    info["has_overlay"] = (path / "overlay.jpg").exists()
    # A directory made by hand (or by the sim synthesizer) may lack preview.jpg;
    # the frontend then falls back to the raw rgb.png rather than a broken image.
    info["has_preview"] = (path / "preview.jpg").exists()
    return info


@router.get("/cameras")
async def list_cameras() -> list[dict]:
    """RealSense devices currently attached, for the session dropdown."""
    from lerobot.cameras.realsense.camera_realsense import RealSenseCamera

    def scan() -> list[dict]:
        return [
            {"serial": str(c.get("serial_number", c.get("id", "?"))), "name": str(c.get("name", ""))}
            for c in RealSenseCamera.find_cameras()
        ]

    return await asyncio.get_event_loop().run_in_executor(_EXECUTOR, scan)


@router.get("/sessions")
async def list_sessions() -> list[dict]:
    """Existing capture directories, newest first, for the reopen dropdown."""
    if not _CAPTURES.exists():
        return []
    out = []
    for p in sorted(_CAPTURES.iterdir(), key=lambda q: q.stat().st_mtime, reverse=True):
        if p.is_dir() and (p / "intrinsics.json").exists():
            out.append({"name": p.name, "scenes": len(list(p.glob("scene_*")))})
    return out


class StartBody(BaseModel):
    serial: str
    name: str = ""
    width: int = 848
    height: int = 480
    fps: int = 30


@router.post("/session/start")
async def start_session(body: StartBody) -> dict:
    """Connect the camera and open a capture directory."""
    from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    global _session
    with _lock:
        if _session is not None and _session.camera is not None:
            raise HTTPException(409, "a camera session is already live — stop it first")
        name = body.name.strip() or time.strftime("session_%Y%m%d_%H%M%S")
        out = _CAPTURES / name

        # Decimation would halve depth resolution AFTER alignment (breaking the
        # pixel-for-pixel match with color); hole filling invents depth exactly at
        # object boundaries. Both stay off — same choices as the CLI capture tool.
        config = RealSenseCameraConfig(
            serial_number_or_name=body.serial,
            fps=body.fps,
            width=body.width,
            height=body.height,
            use_depth=True,
            enable_decimation=False,
            enable_hole_filling=False,
        )
        camera = RealSenseCamera(config)

    def connect() -> dict:
        camera.connect()
        out.mkdir(parents=True, exist_ok=True)
        intr = camera.color_intrinsics()
        (out / "intrinsics.json").write_text(json.dumps(intr, indent=2))
        return intr

    try:
        intr = await asyncio.get_event_loop().run_in_executor(_EXECUTOR, connect)
    except Exception as e:
        raise HTTPException(500, f"camera connect failed: {e}") from e
    with _lock:
        _session = _Session(out=out, camera=camera)
        _session.scenes = [_scene_info(p) for p in sorted(out.glob("scene_*"))]
    return {"name": out.name, "intrinsics": intr, "scenes": _session.scenes, "live": True}


class OpenBody(BaseModel):
    name: str


@router.post("/session/open")
async def open_session(body: OpenBody) -> dict:
    """Open an existing capture directory without a camera (re-analysis)."""
    global _session
    out = _CAPTURES / body.name
    if not (out / "intrinsics.json").exists():
        raise HTTPException(404, f"no capture session named {body.name!r}")
    with _lock:
        if _session is not None and _session.camera is not None:
            raise HTTPException(409, "a camera session is live — stop it first")
        _session = _Session(out=out)
        _session.scenes = [_scene_info(p) for p in sorted(out.glob("scene_*"))]
        return {"name": out.name, "scenes": _session.scenes, "live": False}


@router.post("/session/stop")
async def stop_session() -> dict:
    global _session
    with _lock:
        session, _session = _session, None
    if session is not None and session.camera is not None:
        await asyncio.get_event_loop().run_in_executor(_EXECUTOR, session.camera.disconnect)
    return {"status": "ok"}


@router.get("/state")
async def state() -> dict:
    with _lock:
        if _session is None:
            return {"session": None}
        return {
            "session": {
                "name": _session.out.name,
                "live": _session.camera is not None,
                "scenes": _session.scenes,
            },
            "bind": {"running": _bind.proc is not None and not _bind.done, "done": _bind.done},
        }


@router.get("/preview.jpg")
async def preview() -> Response:
    """One RGB|depth side-by-side JPEG; the frontend polls this while live."""
    import cv2

    with _lock:
        session = _session
    if session is None or session.camera is None:
        raise HTTPException(409, "no live camera session")

    def grab() -> bytes:
        rgb, depth_mm = session.camera.read_color_and_aligned_depth()
        depth_m = depth_mm.astype(np.float32) / 1000.0
        colored = cv2.applyColorMap(
            cv2.convertScaleAbs(np.clip(depth_m, 0, 1.5), alpha=170), cv2.COLORMAP_TURBO
        )
        colored[depth_m <= 0] = 0
        side = np.concatenate([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), colored], axis=1)
        _ok, jpeg = cv2.imencode(".jpg", side, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpeg.tobytes()

    try:
        data = await asyncio.get_event_loop().run_in_executor(_EXECUTOR, grab)
    except Exception as e:
        raise HTTPException(500, f"preview failed: {e}") from e
    return Response(content=data, media_type="image/jpeg")


@router.post("/capture")
async def capture() -> dict:
    """Save one scene — median of 5 depth frames, same format as the CLI tool."""
    import cv2

    with _lock:
        session = _session
    if session is None or session.camera is None:
        raise HTTPException(409, "no live camera session")

    def snap() -> dict:
        rgb, first = session.camera.read_color_and_aligned_depth()
        depths = [first]
        for _ in range(4):
            _rgb, d = session.camera.read_color_and_aligned_depth()
            depths.append(d)
        depth_m = (np.median(np.stack(depths), axis=0) / 1000.0).astype(np.float32)
        assert depth_m.shape == rgb.shape[:2], "depth/color resolution mismatch"

        index = len(list(session.out.glob("scene_*")))
        scene = session.out / f"scene_{index:03d}"
        scene.mkdir(exist_ok=True)
        cv2.imwrite(str(scene / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(scene / "depth.npy", depth_m)
        valid = depth_m > 0
        colored = cv2.applyColorMap(
            cv2.convertScaleAbs(np.clip(depth_m, 0, 1.5), alpha=170), cv2.COLORMAP_TURBO
        )
        colored[~valid] = 0
        preview_img = np.concatenate([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), colored], axis=1)
        cv2.imwrite(str(scene / "preview.jpg"), preview_img)
        return _scene_info(scene)

    info = await asyncio.get_event_loop().run_in_executor(_EXECUTOR, snap)
    with _lock:
        if _session is session:
            session.scenes.append(info)
    return info


@router.get("/scene/{name}/{kind}")
async def scene_file(name: str, kind: str) -> FileResponse:
    """Serve a scene's preview/overlay/rgb image."""
    if kind not in ("preview.jpg", "overlay.jpg", "rgb.png"):
        raise HTTPException(404, "unknown file kind")
    with _lock:
        session = _session
    if session is None:
        raise HTTPException(409, "no session")
    path = (session.out / name / kind).resolve()
    if not path.is_relative_to(session.out.resolve()) or not path.exists():
        raise HTTPException(404, f"{name}/{kind} not found")
    return FileResponse(path)


class BindBody(BaseModel):
    concept: str = ""
    teach: list[int] = [0]
    mask: str = "sam3"  # or "files"


@router.post("/bind")
async def bind(body: BindBody) -> dict:
    """Run the first-contact bench over the session directory, as a subprocess."""
    global _bind
    with _lock:
        session = _session
        if session is None:
            raise HTTPException(409, "no session")
        if _bind.proc is not None and not _bind.done:
            raise HTTPException(409, "a bind run is already in progress")
        if body.mask == "sam3" and not body.concept.strip():
            raise HTTPException(422, "concept is required with SAM3 designation")

        cmd = [
            sys.executable,
            str(_BENCH),
            "--captures",
            str(session.out),
            "--mask",
            body.mask,
            "--teach",
            *[str(i) for i in body.teach],
        ]
        if body.concept.strip():
            cmd += ["--concept", body.concept.strip()]
        env = {"PYTHONPATH": str(_REPO / "src")}
        import os

        env = {**os.environ, **env}
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=str(_REPO)
        )
        _bind = _Bind(proc=proc)

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            with _lock:
                _bind.log.append(line.rstrip("\n"))
        code = proc.wait()
        with _lock:
            _bind.done = True
            _bind.ok = code == 0
            if session is _session:
                session.scenes = [_scene_info(p) for p in sorted(session.out.glob("scene_*"))]

    threading.Thread(target=pump, daemon=True, name="showservo-bind").start()
    return {"status": "started"}


@router.get("/bind/log")
async def bind_log() -> dict:
    with _lock:
        return {
            "running": _bind.proc is not None and not _bind.done,
            "done": _bind.done,
            "ok": _bind.ok,
            "log": "\n".join(_bind.log),
        }
