"""Live perception debugging using the shared Run subprocess owner."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

router = APIRouter(prefix="/api/run/flash/tracking", tags=["flash-tracking"])
_io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flash-debug-state")
_directory: Path | None = None


class TrackingStart(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    serial: str = ""
    width: int = Field(default=1280, gt=0)
    height: int = Field(default=720, gt=0)
    fps: int = Field(default=30, gt=0)
    preview_fps: int = Field(default=10, gt=0)
    vision_pipeline: VisionPipelineConfig = Field(default_factory=VisionPipelineConfig)

    @model_validator(mode="before")
    @classmethod
    def migrate_filter_settings(cls, values):
        # Accept the original debugger payload, but publish only one canonical
        # configuration so snapshots can be used for preparation and live runs.
        values = dict(values)
        legacy = {
            key: values.pop(key)
            for key in ("observation", "shape", "motion", "sam_resolution")
            if key in values
        }
        if legacy:
            pipeline = dict(values.get("vision_pipeline") or {})
            if set(legacy) & set(pipeline):
                raise ValueError("provide each tracking setting only once")
            values["vision_pipeline"] = {**pipeline, **legacy}
        if isinstance(values.get("vision_pipeline"), dict):
            try:
                values["vision_pipeline"] = VisionPipelineConfig.from_dict(values["vision_pipeline"])
            except TypeError as exc:
                raise ValueError(str(exc)) from exc
        return values


class TrackingControl(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    action: Literal["freeze", "clear", "select", "record"]
    id: str
    prompt: str = ""
    point: tuple[float, float] | None = None
    frame_id: int | None = None
    duration_s: float | None = Field(default=None, gt=0, le=120)


def _python() -> str:
    return os.environ.get("LEROBOT_FLASH_TRACKING_PYTHON", sys.executable)


@router.get("/defaults")
async def defaults() -> dict:
    return TrackingStart().model_dump()


@router.post("/cameras")
async def cameras() -> dict:
    proc = await asyncio.create_subprocess_exec(
        _python(),
        "-c",
        "import json; from lerobot.cameras.realsense import RealSenseCamera; "
        "print(json.dumps(RealSenseCamera.find_cameras()))",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=20)
    except TimeoutError as exc:
        proc.kill()
        await proc.communicate()
        raise HTTPException(504, "Camera discovery timed out") from exc
    if proc.returncode:
        raise HTTPException(400, stderr.decode(errors="replace")[-2000:])
    return {"cameras": json.loads(stdout)}


@router.post("/start")
async def start(req: TrackingStart) -> dict:
    from . import run

    if req.vision_pipeline.backend != "surface":
        raise HTTPException(
            400, "Camera-only inspection uses a local surface reference; appearance needs a prepared demo"
        )
    async with run._launch_lock:
        run._ensure_no_active_process()
        global _directory

        def prepare():
            if _directory is not None:
                # safe-destruct: our own mkdtemp snapshot directory; the previous worker has stopped.
                shutil.rmtree(_directory, ignore_errors=True)
            directory = Path(tempfile.mkdtemp(prefix="lerobot-surface-debug-"))
            (directory / "config.json").write_text(json.dumps(req.model_dump()))
            return directory

        _directory = await asyncio.get_running_loop().run_in_executor(_io, prepare)
        # Close previews on this dedicated pool before the generic launcher
        # performs its now-empty preview cleanup.
        from .robot import release_preview_cameras

        await asyncio.get_running_loop().run_in_executor(_io, release_preview_cameras)
        await run._launch_subprocess(
            [_python(), "-u", "-m", "lerobot.policies.flash.tracking_debug_worker", str(_directory)],
            command="flash_tracking",
            config=req.model_dump(),
        )
        return {"status": "started"}


@router.post("/control")
async def control(req: TrackingControl) -> dict:
    from . import run

    proc = run._active_process
    if run._active_command != "flash_tracking" or proc is None or proc.returncode is not None:
        raise HTTPException(409, "Open the tracking camera first")
    if req.action == "select" and req.point is None and not req.prompt.strip():
        raise HTTPException(400, "Enter an object name or select a point")
    if req.action == "record" and req.duration_s is None:
        raise HTTPException(400, "Set a diagnostic recording duration")
    if "." in req.prompt:
        raise HTTPException(400, "Select one object; the name must not contain a period")
    try:
        proc.stdin.write((req.model_dump_json() + "\n").encode())
        await proc.stdin.drain()
    except (BrokenPipeError, ConnectionResetError) as exc:
        raise HTTPException(409, "The tracking process has stopped") from exc
    return {"id": req.id}


@router.get("/state")
async def state() -> dict:
    from . import run

    directory = _directory
    running = run._active_command == "flash_tracking" and run.is_run_active()

    def read():
        result = {"running": running, "server_time_s": time.perf_counter()}
        if directory:
            result["session_id"] = directory.name
            for key, name in (
                ("preview", "preview.json"),
                ("estimate", "state.json"),
                ("config", "config.json"),
            ):
                with contextlib.suppress(FileNotFoundError):
                    result[key] = json.loads((directory / name).read_text())
        return result

    return await asyncio.get_running_loop().run_in_executor(_io, read)


@router.post("/stop")
async def stop() -> dict:
    from . import run

    if run._active_command != "flash_tracking" or not run.is_run_active():
        return {"status": "stopped"}
    return await run.stop_process()
