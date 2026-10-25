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
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from lerobot.policies.flash.surface_correspondence import SurfaceMotionConfig
from lerobot.policies.flash.surface_grounding import ShapeConfig
from lerobot.policies.flash.surface_observation import SurfaceObservationConfig

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
    sam_resolution: int = Field(default=504, ge=224, multiple_of=14)
    observation: SurfaceObservationConfig = Field(default_factory=SurfaceObservationConfig)
    shape: ShapeConfig = Field(default_factory=ShapeConfig)
    motion: SurfaceMotionConfig = Field(default_factory=SurfaceMotionConfig)


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

    # Negative filter sizes and non-finite solver settings must fail before a
    # camera or model is opened. Zero mask erosion is a valid experiment.
    for key, value in asdict(req.observation).items():
        if value is not None and (value < 0 or (value == 0 and key != "mask_margin_px")):
            raise HTTPException(400, f"Invalid observation setting: {key}")
    if any(
        value < 0 or (value == 0 and key != "mask_boundary_tolerance_px")
        for key, value in asdict(req.shape).items()
    ):
        raise HTTPException(400, "Shape settings must be positive")
    motion = req.motion
    if (
        any(value <= 0 for key, value in asdict(motion).items() if key != "covisibility_min")
        or not 0 <= motion.covisibility_min <= 1
        or not 0 < motion.ransac_confidence < 1
        or motion.min_ransac_pairs < 4
        or motion.crop_min_px > motion.crop_max_px
        or 2 * motion.grid_step_px >= motion.input_size_px
    ):
        raise HTTPException(400, "Invalid RGB motion settings")
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
