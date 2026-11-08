"""Live perception debugging using the shared Run subprocess owner."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
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

from lerobot.policies.flash.live_surface_io import write_snapshot
from lerobot.policies.flash.tracking_history import (
    export_episode,
    history_root,
    read_index,
    read_packet,
    save_window,
)
from lerobot.policies.flash.vision_pipeline import VisionPipelineConfig

router = APIRouter(prefix="/api/run/flash/tracking", tags=["flash-tracking"])
_io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flash-debug-state")
_export = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flash-episode-export")
_directory: Path | None = None


class TrackingStart(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    engine: Literal["experiment"] = "experiment"
    pipeline_settings: dict | None = None
    serial: str = ""
    width: int = Field(default=1280, gt=0)
    height: int = Field(default=720, gt=0)
    fps: int = Field(default=30, gt=0)
    preview_fps: int = Field(default=10, gt=0)
    history_seconds: int = Field(default=60, ge=1, le=600)
    # Model outputs are opt-in and never replace the recorded sensor arrays.
    depth_source: Literal[
        "sensor",
        "lingbot",
        "foundation_stereo",
        "lite_any_stereo_v2",
        "lite_any_stereo_v2_s",
        "lite_any_stereo_v2_m",
        "lite_any_stereo_v2_l",
    ] = "sensor"
    # Applied by the camera before FLASH sees the depth map. Off by default so
    # the live depth view exposes sensor holes and measured noise.
    depth_processing: dict[Literal["spatial", "temporal", "hole_filling"], bool] = Field(
        default_factory=lambda: dict.fromkeys(("spatial", "temporal", "hole_filling"), False)
    )

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
        if settings["group_experiment"] is None:
            raise ValueError("Live tracking requires group settings")
        values["pipeline_settings"] = settings
        return values


class TrackingControl(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    action: Literal["freeze", "clear", "select", "clear_history", "depth_filters"]
    id: str
    prompt: str = ""
    point: tuple[float, float] | None = None
    frame_id: int | None = None
    depth_processing: dict[Literal["spatial", "temporal", "hole_filling"], bool] | None = None

    @model_validator(mode="after")
    def complete_filter_change(self):
        if self.action == "depth_filters" and (
            self.depth_processing is None
            or set(self.depth_processing) != {"spatial", "temporal", "hole_filling"}
        ):
            raise ValueError("Supply all three depth-filter switches")
        return self


class RecordingRename(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recording: str
    name: str = Field(min_length=1, max_length=80)


class EpisodeExport(BaseModel):
    recording: str


def _python() -> str:
    return os.environ.get("LEROBOT_FLASH_TRACKING_PYTHON", sys.executable)


async def _check_registration_backend(backend: str) -> None:
    """Fail before camera launch if the selected worker lacks its optional solver."""
    module = {"small_gicp": "small_gicp", "open3d_plane": "open3d", "open3d_color": "open3d"}.get(backend)
    if module is None:
        return
    try:
        proc = await asyncio.create_subprocess_exec(
            _python(),
            "-c",
            "import importlib.util, sys; sys.exit(importlib.util.find_spec(sys.argv[1]) is None)",
            module,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        except TimeoutError as exc:
            proc.kill()
            await proc.communicate()
            raise HTTPException(504, f"Could not check {backend} in the tracking environment") from exc
    except OSError as exc:
        raise HTTPException(400, f"Cannot start the tracking Python environment: {exc}") from exc
    if proc.returncode:
        raise HTTPException(
            400,
            f"{backend} is unavailable in the tracking Python environment. Install {module} there or choose Surface fit.",
        )


@router.get("/defaults")
async def defaults() -> dict:
    return TrackingStart().model_dump()


@router.get("/shared-defaults")
async def shared_defaults() -> dict:
    from lerobot.gui.flash_pipeline_fields import pipeline_fields
    from lerobot.policies.flash.surface_registration import MATCHERS
    from lerobot.policies.flash.surface_session_config import defaults

    return {"settings": defaults(), "matchers": MATCHERS, "fields": pipeline_fields()}


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

    await _check_registration_backend(req.pipeline_settings["config"]["shape"]["registration"]["backend"])
    async with run._launch_lock:
        run._ensure_no_active_process()
        global _directory

        def prepare():
            if _directory is not None:
                # safe-destruct: our own mkdtemp snapshot directory; the previous worker has stopped.
                shutil.rmtree(_directory, ignore_errors=True)
            directory = Path(tempfile.mkdtemp(prefix="lerobot-surface-debug-"))
            recent = history_root() / "recent"
            # The prior worker is stopped. Explicitly saved windows have their
            # own hard links and survive replacement of the rolling buffer.
            if recent.exists():
                # safe-destruct: replace only this application's stopped rolling buffer.
                shutil.rmtree(recent)
            recent.mkdir(parents=True)
            (recent / "config.json").write_text(req.model_dump_json())
            (directory / "config.json").write_text(
                json.dumps({**req.model_dump(), "history_directory": str(recent)})
            )
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


@router.get("/preview-depth")
async def preview_depth():
    directory = _directory

    def read():
        if directory is None or not (directory / "preview_depth.bin").is_file():
            raise HTTPException(404, "No live depth preview yet")
        return (directory / "preview_depth.bin").read_bytes()

    packet = await asyncio.get_running_loop().run_in_executor(_io, read)
    return Response(packet, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})


def _history_directory(identity):
    if identity != "recent" and not re.fullmatch(r"saved-[0-9a-f]{32}", identity):
        raise HTTPException(404, "Unknown recording")
    directory = history_root() / identity
    if not directory.is_dir():
        raise HTTPException(404, "Recording is unavailable")
    return directory


@router.get("/history")
async def history(recording: str = "recent"):
    def read():
        root = history_root()
        recordings = []
        for directory in root.glob("saved-*"):
            index = read_index(directory)
            names = sorted(
                {
                    obj.get("name")
                    for row in index["frames"]
                    for obj in row.get("objects", {}).values()
                    if obj.get("name")
                }
            )
            recordings.append(
                {
                    "id": directory.name,
                    "name": index.get("name"),
                    "episode_root": str(directory / "episode")
                    if (directory / "episode" / "meta" / "info.json").is_file()
                    else None,
                    "saved_at": index.get("saved_at"),
                    "frames": len(index["frames"]),
                    "objects": names,
                }
            )
        recordings.sort(key=lambda row: row["saved_at"] or 0, reverse=True)
        recent = read_index(root / "recent")
        index = recent if recording == "recent" else read_index(_history_directory(recording))
        camera_file = root / recording / "episode" / "flash" / "v1" / "camera.json"
        streams_file = root / recording / "camera_streams.json"
        return {
            **index,
            "recording": recording,
            "recordings": recordings,
            "recent_frames": len(recent["frames"]),
            "episode_camera": json.loads(camera_file.read_text()) if camera_file.is_file() else None,
            "camera_streams": json.loads(streams_file.read_text()) if streams_file.is_file() else None,
            "overlay_indices": [
                i
                for i, row in enumerate(index["frames"])
                if (root / recording / f"{row['frame']:010d}.overlay.json").is_file()
            ],
        }

    return await asyncio.get_running_loop().run_in_executor(_io, read)


@router.get("/history/scene")
async def history_scene(frame: int, recording: str = "recent"):
    def read():
        try:
            return read_packet(_history_directory(recording), frame)
        except FileNotFoundError as exc:
            raise HTTPException(404, "Frame expired from recent history; choose a retained frame") from exc

    packet = await asyncio.get_running_loop().run_in_executor(_io, read)
    return Response(packet, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})


@router.post("/history/save")
async def save_history():
    def save():
        try:
            return {"id": save_window(_history_directory("recent"), history_root())}
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    return await asyncio.get_running_loop().run_in_executor(_io, save)


@router.post("/history/episode")
async def create_history_episode(request: EpisodeExport):
    if request.recording == "recent":
        raise HTTPException(400, "Save recent history before exporting an episode")
    directory = _history_directory(request.recording)
    root = await asyncio.get_running_loop().run_in_executor(_export, export_episode, directory)
    return {"recording": request.recording, "episode_root": str(root)}


@router.get("/history/overlay")
async def history_overlay(recording: str, index: int):
    directory = _history_directory(recording)

    def read():
        rows = read_index(directory)["frames"]
        if index < 0 or index >= len(rows):
            raise HTTPException(404, "Frame is outside this recording")
        for i in range(index, -1, -1):
            path = directory / "episode" / "flash" / "v1" / "overlays" / f"{i:06d}.json"
            if path.is_file():
                return {"source_index": i, **json.loads(path.read_text())}
        return {"source_index": None, "objects": {}, "scene_group_tracks": []}

    return await asyncio.get_running_loop().run_in_executor(_io, read)


@router.patch("/history/name")
async def rename_history(request: RecordingRename):
    name = request.name.strip()
    if not name:
        raise HTTPException(422, "Recording name cannot be blank")
    if request.recording == "recent":
        raise HTTPException(400, "Save recent history before naming it")

    def rename():
        directory = _history_directory(request.recording)
        index = read_index(directory)
        index["name"] = name
        write_snapshot(directory / "index.json", index)
        return {"id": request.recording, "name": name}

    return await asyncio.get_running_loop().run_in_executor(_io, rename)


@router.delete("/history")
async def delete_history(recording: str):
    from . import run

    if recording == "recent" and run._active_command == "flash_surface_live" and run.is_run_active():
        await control(TrackingControl(action="clear_history", id=f"history-{time.time_ns()}"))
        return {"status": "clearing"}

    def remove():
        directory = _history_directory(recording)
        # Only the stopped rolling window or a validated saved-* recording is removable.
        # safe-destruct: user requested deletion of this validated recording through the GUI.
        shutil.rmtree(directory)
        if recording == "recent":
            directory.mkdir(parents=True)
        return {"status": "deleted"}

    return await asyncio.get_running_loop().run_in_executor(_io, remove)
