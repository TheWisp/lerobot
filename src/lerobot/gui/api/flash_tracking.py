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
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

router = APIRouter(prefix="/api/run/flash/tracking", tags=["flash-tracking"])
_io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flash-debug-state")
_directory: Path | None = None


class TrackingStart(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    engine: Literal["experiment"] = "experiment"
    pipeline_settings: dict | None = None
    method: Literal["conditioned_fixed", "conditioned_growing", "group_fixed", "group_growing"] = (
        "group_growing"
    )
    serial: str = ""
    width: int = Field(default=1280, gt=0)
    height: int = Field(default=720, gt=0)
    fps: int = Field(default=30, gt=0)
    preview_fps: int = Field(default=10, gt=0)

    @model_validator(mode="before")
    @classmethod
    def shared_pipeline_settings(cls, values):
        from lerobot.policies.flash.surface_session_config import defaults, validate

        values = dict(values)
        settings = validate(values.get("pipeline_settings") or defaults())
        if VisionPipelineConfig.from_dict(settings["config"]).backend != "surface":
            raise ValueError("Camera inspection requires the shared surface pipeline")
        if settings["surface_experiment"] is None:
            raise ValueError("Live surface inspection requires surface settings")
        if (
            values.get("method", "group_growing").startswith("group_")
            and settings["group_experiment"] is None
        ):
            raise ValueError("Selected method requires group settings")
        values["pipeline_settings"] = settings
        return values


class TrackingControl(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    action: Literal["freeze", "clear", "select"]
    id: str
    prompt: str = ""
    point: tuple[float, float] | None = None
    frame_id: int | None = None


def _python() -> str:
    return os.environ.get("LEROBOT_FLASH_TRACKING_PYTHON", sys.executable)


@router.get("/defaults")
async def defaults() -> dict:
    return TrackingStart().model_dump()


@router.get("/shared-defaults")
async def shared_defaults() -> dict:
    from lerobot.policies.flash.surface_registration import MATCHERS
    from lerobot.policies.flash.surface_session_config import defaults

    return {"settings": defaults(), "matchers": MATCHERS}


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
            [_python(), "-u", "-m", "lerobot.policies.flash.live_surface_worker", str(_directory)],
            command="flash_surface_live",
            config=req.model_dump(),
        )
        return {"status": "started"}


@router.post("/control")
async def control(req: TrackingControl) -> dict:
    from . import run

    proc = run._active_process
    if run._active_command != "flash_surface_live" or proc is None or proc.returncode is not None:
        raise HTTPException(409, "Open the tracking camera first")
    if req.action == "select" and req.point is None and not req.prompt.strip():
        raise HTTPException(400, "Enter an object name or select a point")
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
    running = run._active_command == "flash_surface_live" and run.is_run_active()

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

    if run._active_command != "flash_surface_live" or not run.is_run_active():
        return {"status": "stopped"}
    return await run.stop_process()


@router.get("/scene")
async def scene():
    directory = _directory

    def read():
        if directory is None or not (directory / "scene.bin").is_file():
            raise HTTPException(404, "No processed live frame yet")
        return (directory / "scene.bin").read_bytes()

    packet = await asyncio.get_running_loop().run_in_executor(_io, read)
    return Response(packet, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})
