"""FLASH editor metadata; recording and playback stay with the shared Run/Data APIs."""

import asyncio
import copy
import gzip
import json
import struct
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from lerobot.gui.config_paths import gui_config_dir
from lerobot.policies.flash.demo_document import (
    DemoDocument,
    atomic_json,
    document_path,
    load_document,
    save_document,
    tracking_selection,
)

from . import run

router = APIRouter(prefix="/api/run/flash/editor", tags=["flash-editor"])
_io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flash-editor")
_recording: dict | None = None
_record_process = None
_tracking_job = None
_preview_process = None
_record_tasks = set()
EDITOR_DIR = gui_config_dir() / "flash" / "editor"


class EditorSettings(BaseModel):
    """One local setup; robot and teleop definitions remain in their own profiles."""

    model_config = ConfigDict(extra="forbid", validate_default=True)
    robot: str = ""
    teleop: str = ""
    camera: str = ""
    fps: int = Field(default=30, ge=1, le=120)
    pipeline_settings: dict = Field(default_factory=dict)
    depth_source: str = "sensor"

    @field_validator("robot", "teleop")
    @classmethod
    def profile_name(cls, value):
        if value and (value in {".", ".."} or "/" in value or "\\" in value):
            raise ValueError("Invalid profile name")
        return value

    @field_validator("pipeline_settings")
    @classmethod
    def pipeline(cls, value):
        from lerobot.policies.flash.surface_session_config import defaults, validate

        return validate(value or defaults())

    @field_validator("depth_source")
    @classmethod
    def depth(cls, value):
        from .flash_tracking import TrackingStart

        return TrackingStart(depth_source=value).depth_source


def _read_settings():
    from lerobot.policies.flash.depth_enhancement import preferred_depth_source
    from lerobot.policies.flash.surface_session_config import defaults

    path = EDITOR_DIR / "settings.json"
    if path.exists():
        return EditorSettings.model_validate_json(path.read_text())
    return EditorSettings(pipeline_settings=defaults(), depth_source=preferred_depth_source())


@router.get("/settings")
async def settings():
    return await asyncio.get_running_loop().run_in_executor(_io, _read_settings)


@router.put("/settings")
async def save_settings(request: EditorSettings):
    def write():
        # Resolve names now; never silently switch hardware after a profile was deleted.
        _profiles(request, require_teleop=False)
        atomic_json(EDITOR_DIR / "settings.json", request.model_dump())
        return request

    return await asyncio.get_running_loop().run_in_executor(_io, write)


def _profiles(settings, *, require_teleop=True):
    from .robot import ROBOT_PROFILES_DIR, TELEOP_PROFILES_DIR, _read_profile

    if not settings.robot or (require_teleop and not settings.teleop):
        raise HTTPException(400, "Choose a robot and teleoperator in Settings")
    return (
        _read_profile(ROBOT_PROFILES_DIR, settings.robot),
        _read_profile(TELEOP_PROFILES_DIR, settings.teleop) if settings.teleop else None,
    )


def _read_library():
    path = EDITOR_DIR / "demos.json"
    library = json.loads(path.read_text()) if path.exists() else {"recent": None, "demos": []}
    # Migrate the old staging entry once. Naming was never required to retain
    # the underlying dataset, and it is no longer required to list the demo.
    recent = library.get("recent")
    if recent and not any(d["id"] == recent["id"] for d in library["demos"]):
        library["demos"].insert(0, {**recent, "name": _demo_name(recent["created_at"])})
        atomic_json(path, library)
    return library


def _demo_name(created_at):
    return "Demo · " + datetime.fromtimestamp(created_at).strftime("%d %b, %H:%M")


def _remember_recording(recording=None, process=None):
    library = _read_library()
    recording = recording or _recording
    process = process or _record_process
    if recording and process and process.returncode == 0:
        identity = recording["id"]
        if library.get("completed") != identity:
            library["recent"] = {
                "id": identity,
                "root": recording["root"],
                "episode": 0,
                "name": _demo_name(recording["created_at"]),
                "created_at": recording["created_at"],
            }
            if not any(d["id"] == identity for d in library["demos"]):
                library["demos"].insert(0, dict(library["recent"]))
            library["completed"] = identity
            atomic_json(EDITOR_DIR / "demos.json", library)
    return library


async def _record_finished(recording, process):
    await process.wait()
    await asyncio.get_running_loop().run_in_executor(_io, _remember_recording, recording, process)


@router.get("/library")
async def library():
    return await asyncio.get_running_loop().run_in_executor(_io, _remember_recording)


class KeepDemo(BaseModel):
    id: str
    name: str = Field(min_length=1, max_length=80)


@router.post("/library/save")
async def keep_demo(request: KeepDemo):
    def write():
        library = _remember_recording()
        entry = next((d for d in library["demos"] if d["id"] == request.id), None)
        if entry is None:
            recent = library["recent"]
            if not recent or recent["id"] != request.id:
                raise HTTPException(404, "Demo no longer available")
            entry = dict(recent)
            library["demos"].insert(0, entry)
        entry["name"] = request.name.strip() or "Untitled demo"
        atomic_json(EDITOR_DIR / "demos.json", library)
        return library

    return await asyncio.get_running_loop().run_in_executor(_io, write)


@router.post("/record-current")
async def record_current():
    def prepare():
        settings = _read_settings()
        robot, teleop = _profiles(settings)
        return settings, robot, teleop

    settings, robot, teleop = await asyncio.get_running_loop().run_in_executor(_io, prepare)
    await stop_preview()
    identity = uuid.uuid4().hex
    return await record(
        run.RecordRequest(
            robot=robot,
            teleop=teleop,
            repo_id=f"local/flash-demo-{identity}",
            root=str(EDITOR_DIR / "recordings" / identity),
            single_task="FLASH demonstration",
            fps=settings.fps,
            episode_time_s=600,
            reset_time_s=0,
            num_episodes=1,
            play_sounds=False,
        )
    )


@router.post("/preview")
async def preview():
    """Open the existing FLASH camera worker; no robot connection or duplicate camera."""
    from .flash_tracking import TrackingStart, start

    global _preview_process
    if (
        _preview_process is not None
        and run._active_process is _preview_process
        and _preview_process.returncode is None
    ):
        return {"status": "started"}

    def prepare():
        settings = _read_settings()
        robot, _ = _profiles(settings, require_teleop=False)
        cameras = _calibrated_cameras(robot)
        if settings.camera not in cameras:
            raise ValueError("Choose a calibrated camera in Settings")
        camera = robot["cameras"][settings.camera]
        if camera["type"] != "intelrealsense":
            raise ValueError("FLASH live depth preview currently requires a RealSense camera")
        return TrackingStart(
            serial=str(camera["serial_number_or_name"]),
            width=camera["width"],
            height=camera["height"],
            fps=camera.get("fps", settings.fps),
            pipeline_settings=settings.pipeline_settings,
            depth_source=settings.depth_source,
        )

    try:
        request = await asyncio.get_running_loop().run_in_executor(_io, prepare)
    except (KeyError, ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    result = await start(request)
    _preview_process = run._active_process
    return result


@router.post("/preview/stop")
async def stop_preview():
    if (
        _preview_process is not None
        and run._active_process is _preview_process
        and _preview_process.returncode is None
    ):
        return await run.stop_process()
    return {"status": "stopped"}


def _calibrated_cameras(profile: dict) -> list[str]:
    from lerobot.robots.geometry import RobotGeometry

    fields = profile.get("fields", {})
    geometry = RobotGeometry.load(fields.get("geometry_path"), robot_id=fields.get("id"))
    if geometry is None:
        raise ValueError("This robot needs camera-to-base calibration. Set geometry_path in Robot Setup.")
    cameras = []
    for name, config in profile.get("cameras", {}).items():
        if name in geometry.to_dict()["cameras"]:
            geometry.camera_pose(name, config)
            cameras.append(name)
    if not cameras:
        raise ValueError("Add a camera matching this robot's geometry in Robot Setup.")
    return cameras


@router.post("/robot-geometry")
async def robot_geometry(profile: dict):
    try:
        cameras = await asyncio.get_running_loop().run_in_executor(_io, _calibrated_cameras, profile)
        return {"ready": True, "cameras": cameras}
    except (ValueError, OSError) as exc:
        return {"ready": False, "reason": str(exc)}


@router.post("/record")
async def record(request: run.RecordRequest):
    """Use the ordinary recorder and its hardware ownership, factories and processors."""
    from lerobot.utils.constants import HF_LEROBOT_HOME

    global _recording, _record_process
    if request.teleop is None or request.policy_path is not None:
        raise HTTPException(400, "Select a teleoperator to demonstrate the task")
    # A demo retains model inputs even when preview uses sensor depth. Keep the
    # user's general Robot profile unchanged; this launch records raw RGB-D + IR.
    robot = copy.deepcopy(request.robot)
    for camera in robot.get("cameras", {}).values():
        if camera.get("type") == "intelrealsense":
            camera.update(
                use_rgb=True,
                use_depth=True,
                enable_infrared=True,
                enable_decimation=False,
                enable_spatial=False,
                enable_temporal=False,
                enable_hole_filling=False,
            )
            camera.pop("record_depth", None)  # This is a robot field, not a camera setting.
    if "record_depth" in robot.get("fields", {}):
        robot["fields"]["record_depth"] = True
    request = request.model_copy(update={"robot": robot})
    if not 1 <= request.fps <= 120:
        raise HTTPException(400, "Recording FPS must be between 1 and 120")
    if len(request.repo_id.split("/")) != 2 or any(
        part in {"", ".", ".."} for part in request.repo_id.split("/")
    ):
        raise HTTPException(400, "Use a dataset name such as local/my-demo")
    root = Path(request.root).expanduser() if request.root else HF_LEROBOT_HOME / request.repo_id
    if await asyncio.get_running_loop().run_in_executor(_io, root.exists):
        raise HTTPException(409, "This dataset already exists. Choose a new dataset name for this demo.")
    try:
        await asyncio.get_running_loop().run_in_executor(_io, _calibrated_cameras, request.robot)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    camera_rates = [
        camera["fps"] for camera in request.robot.get("cameras", {}).values() if camera.get("fps")
    ]
    request = request.model_copy(
        update={
            "root": str(root),
            "num_episodes": 1,
            "reset_time_s": 0,
            "fps": min(request.fps, *camera_rates) if camera_rates else request.fps,
        }
    )
    launched = await run.start_record(request)
    _record_process = run._active_process
    _recording = {
        "id": uuid.uuid4().hex,
        "created_at": time.time(),
        "pid": launched["pid"],
        "root": str(root),
        "repo_id": request.repo_id,
        "fps": request.fps,
    }
    task = asyncio.create_task(_record_finished(dict(_recording), _record_process))
    _record_tasks.add(task)
    task.add_done_callback(_record_tasks.discard)
    return {**launched, **_recording}


@router.get("/recording")
async def recording():
    from ._run_core import get_run_status

    status = get_run_status()
    # The last editor recording is discoverable after a page reload. A different
    # Run process must never acquire the editor's Stop button or completion path.
    own = None
    if _recording and _record_process:
        own = {
            **_recording,
            "running": _record_process.returncode is None,
            "returncode": _record_process.returncode,
        }
    return {
        "run": status,
        "recording": own,
        "preview_running": _preview_process is not None
        and run._active_process is _preview_process
        and _preview_process.returncode is None,
    }


class StopRecording(BaseModel):
    pid: int


@router.post("/stop")
async def stop(request: StopRecording):
    if not _recording or request.pid != _recording["pid"] or run._active_process is not _record_process:
        raise HTTPException(409, "This recording is no longer the active run")
    return await run.stop_process()


class SaveDemo(BaseModel):
    dataset_id: str
    episode: int = Field(ge=0)
    document: DemoDocument


class TrackDemo(BaseModel):
    dataset_id: str
    episode: int = Field(ge=0)
    revision: int = Field(ge=1)


@router.post("/track")
async def track(request: TrackDemo):
    global _tracking_job
    dataset = _dataset(request.dataset_id)
    async with run._launch_lock:
        run._ensure_no_active_process()

        def prepare():
            from lerobot.policies.flash.demo_tracking import DemoFrames

            document, _ = load_document(dataset, request.episode)
            if document.revision != request.revision:
                raise ValueError("Save the current edit before tracking")
            if not document.objects:
                raise ValueError("Select at least one object")
            frames = DemoFrames(dataset, request.episode, document)
            setup = _read_settings()
            frames.require_depth_source(setup.depth_source)
            folder = document_path(dataset, request.episode).with_suffix("")
            destination = folder / uuid.uuid4().hex
            atomic_json(folder / "latest.json", {"directory": destination.name})
            atomic_json(destination / "status.json", {"state": "loading", "done": 0})
            atomic_json(destination / "settings.json", setup.pipeline_settings)
            return destination, setup.depth_source

        try:
            destination, depth_source = await asyncio.get_running_loop().run_in_executor(_io, prepare)
        except (KeyError, ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc
        args = [
            sys.executable,
            "-m",
            "lerobot.policies.flash.demo_tracking",
            "--root",
            str(dataset.root),
            "--episode",
            str(request.episode),
            "--revision",
            str(request.revision),
            "--output",
            str(destination),
            "--settings",
            str(destination / "settings.json"),
            "--depth-source",
            depth_source,
        ]
        await run._launch_subprocess(args, command="flash_editor", config=request.model_dump())
        _tracking_job = (destination, run._active_process)
        return {"pid": run._active_process.pid}


@router.post("/cancel-tracking")
async def cancel_tracking(request: StopRecording):
    if (
        not _tracking_job
        or request.pid != _tracking_job[1].pid
        or run._active_process is not _tracking_job[1]
    ):
        raise HTTPException(409, "This tracking job is no longer the active run")
    return await run.stop_process()


@router.get("/tracking")
async def tracking(dataset_id: str, episode: int = 0, frame: int | None = None):
    dataset = _dataset(dataset_id)

    def read():
        document, _ = load_document(dataset, episode)
        folder = document_path(dataset, episode).with_suffix("")
        latest = folder / "latest.json"
        if not latest.exists():
            return {"state": "none"}
        destination = folder / json.loads(latest.read_text())["directory"]
        if destination.parent != folder:
            raise ValueError("Invalid tracking output directory")
        status = json.loads((destination / "status.json").read_text())
        status["result_id"] = destination.name
        if _tracking_job and _tracking_job[0] == destination:
            proc = _tracking_job[1]
            status["pid"] = proc.pid
            if proc.returncode is not None and status["state"] in {"loading", "running"}:
                status.update(state="failed", reason="Tracking worker stopped. See Run output.")
        inputs = destination / "input.json"
        if inputs.exists():
            previous = json.loads(inputs.read_text())["document"]
            status["stale"] = tracking_selection(DemoDocument.model_validate(previous)) != tracking_selection(
                document
            )
        if frame is not None and status["state"] == "complete" and not status.get("stale"):
            packet = destination / "frames" / f"{frame:06d}.json"
            if packet.exists():
                status["frame_result"] = json.loads(packet.read_text())
        return status

    try:
        return await asyncio.get_running_loop().run_in_executor(_io, read)
    except (OSError, KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/display")
async def display_frames(
    dataset_id: str,
    episode: int = 0,
    start: int = Query(ge=0),
    count: int = Query(ge=1, le=240),
    scene: bool = False,
):
    """Prefetch display evidence alongside the same bounded video chunk."""
    dataset = _dataset(dataset_id)

    def read():
        document, source = load_document(dataset, episode)
        if start + count > source["length"]:
            raise ValueError("Display interval is outside the episode")
        folder = document_path(dataset, episode).with_suffix("")
        latest = folder / "latest.json"
        if not latest.exists():
            return Response(json.dumps([None] * count), media_type="application/json")
        destination = folder / json.loads(latest.read_text())["directory"]
        if destination.parent != folder:
            raise ValueError("Invalid tracking output directory")
        status = json.loads((destination / "status.json").read_text())
        if status["state"] != "complete":
            return Response(json.dumps([None] * count), media_type="application/json")
        previous = DemoDocument.model_validate(
            json.loads((destination / "input.json").read_text())["document"]
        )
        if tracking_selection(previous) != tracking_selection(document):
            return Response(json.dumps([None] * count), media_type="application/json")
        if scene and not status.get("display_schema"):
            raise ValueError("Track this demo again to save processed depth and surfaces")
        rows, packets = [], []
        for frame in range(start, start + count):
            path = destination / "frames" / f"{frame:06d}.json"
            result = json.loads(path.read_text()) if path.exists() else None
            if scene:
                packet = (
                    (destination / "scenes" / f"{result['observation_frame']:06d}.bin.gz").read_bytes()
                    if result
                    else b""
                )
                packets.extend([struct.pack("<I", len(packet)), packet])
            else:
                if result:
                    result = {key: value for key, value in result.items() if key != "objects"} | {
                        "objects": {
                            role: {
                                key: value
                                for key, value in obj.items()
                                if key not in {"diagnostics", "group_tracks"}
                            }
                            for role, obj in result["objects"].items()
                        }
                    }
                rows.append(result)
        if scene:
            return Response(b"".join(packets), media_type="application/octet-stream")
        return Response(
            gzip.compress(json.dumps(rows, separators=(",", ":")).encode(), compresslevel=1),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"},
        )

    try:
        return await asyncio.get_running_loop().run_in_executor(_io, read)
    except (OSError, KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


def _dataset(identity):
    from . import datasets

    dataset = datasets._app_state.datasets.get(identity)
    if dataset is None:
        raise HTTPException(404, "Open the demo dataset first")
    return dataset


@router.get("/document")
async def document(dataset_id: str, episode: int = 0):
    dataset = _dataset(dataset_id)

    def read():
        try:
            value, source = load_document(dataset, episode)
            return {"document": value.model_dump(mode="json"), "source": source}
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    return await asyncio.get_running_loop().run_in_executor(_io, read)


@router.put("/document")
async def save(request: SaveDemo):
    dataset = _dataset(request.dataset_id)

    def write():
        try:
            return save_document(dataset, request.episode, request.document).model_dump(mode="json")
        except (ValueError, OSError) as exc:
            raise HTTPException(409, str(exc)) from exc

    return await asyncio.get_running_loop().run_in_executor(_io, write)


@router.get("/trajectory")
async def demo_trajectory(dataset_id: str, camera: str, episode: int = 0):
    dataset = _dataset(dataset_id)

    def read():
        from lerobot.policies.flash.demo_trajectory import read_demo_trajectory

        try:
            return read_demo_trajectory(dataset, episode, camera)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise HTTPException(422, "This recording has no calibrated robot actions") from exc

    return await asyncio.get_running_loop().run_in_executor(_io, read)


class PrepareDemo(BaseModel):
    id: str


class ApplyDemo(BaseModel):
    plan: str = Field(pattern=r"^[0-9a-f]{32}$")
    simulate: bool = True


_execution_process = None
_execution_directory = None


@router.post("/run-demo")
async def run_demo(request: PrepareDemo):
    """One user action; preparation and validation still precede robot launch."""
    prepared = await prepare_demo(request)
    return await apply_demo(ApplyDemo(plan=prepared["plan"], simulate=False))


def _rollout_directory(identity):
    import re

    if not re.fullmatch(r"rollout-[0-9a-f]{32}", identity):
        raise HTTPException(404, "Unknown rollout")
    directory = EDITOR_DIR / "rollouts" / identity
    if not (directory / "run.json").is_file():
        raise HTTPException(404, "Rollout is unavailable")
    return directory


@router.get("/rollouts")
async def rollouts():
    def read():
        rows = []
        for path in (EDITOR_DIR / "rollouts").glob("rollout-*/run.json"):
            value = json.loads(path.read_text())
            if value["state"] == "running" and not (
                _execution_directory == path.parent
                and _execution_process is not None
                and _execution_process.returncode is None
            ):
                value["state"] = "interrupted"
            rows.append(value)
        return sorted(rows, key=lambda row: row["created_at"], reverse=True)

    return await asyncio.get_running_loop().run_in_executor(_io, read)


@router.get("/rollouts/{identity}/events")
async def rollout_events(
    identity: str,
    cursor: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    kind: str = "",
):
    def read():
        directory = _rollout_directory(identity)
        path = directory / "events.jsonl"
        if not path.exists():
            return {"events": [], "cursor": cursor, "more": False}
        rows = []
        with path.open("rb") as stream:
            stream.seek(cursor)
            while len(rows) < limit:
                before = stream.tell()
                line = stream.readline()
                if not line.endswith(b"\n"):
                    stream.seek(before)
                    break
                event = json.loads(line)
                if not kind or event["kind"].startswith(kind):
                    rows.append(event)
            next_cursor = stream.tell()
            more = bool(stream.read(1))
        return {"events": rows, "cursor": next_cursor, "more": more}

    return await asyncio.get_running_loop().run_in_executor(_io, read)


@router.get("/rollouts/{identity}/trajectory")
async def rollout_trajectory(identity: str):
    def read():
        from lerobot.policies.flash.rollout_trajectory import read_trajectory

        try:
            return read_trajectory(_rollout_directory(identity))
        except (ValueError, KeyError, FileNotFoundError) as exc:
            raise HTTPException(422, "This recording does not have a usable commanded tool path") from exc

    return await asyncio.get_running_loop().run_in_executor(_io, read)


@router.get("/rollouts/{identity}/log")
async def rollout_log(identity: str):
    def read():
        directory = _rollout_directory(identity)
        output = directory / "output.log"
        if output.exists():
            return {"text": output.read_text(errors="replace")}
        # Older runs have the full event journal but no captured console output.
        # Show their recorded lifecycle messages, without inventing stdout.
        lines = ["Console output was not saved for this rollout. Recorded execution messages:"]
        origin = None
        path = directory / "events.jsonl"
        if path.exists():
            with path.open() as stream:
                for line in stream:
                    if not line.endswith("\n"):
                        break
                    event = json.loads(line)
                    origin = event["time_s"] if origin is None else origin
                    if event["kind"] in {
                        "run.started",
                        "execution.state",
                        "run.finished",
                        "robot.command_failed",
                        "robot.read_failed",
                    }:
                        detail = event.get("state", event.get("mode", ""))
                        reason = event.get("reason")
                        lines.append(
                            f"+{event['time_s'] - origin:.2f}s {event['kind']}: {detail}"
                            + (f"\n{reason}" if reason else "")
                        )
        return {"text": "\n".join(lines)}

    return await asyncio.get_running_loop().run_in_executor(_io, read)


@router.post("/prepare")
async def prepare_demo(request: PrepareDemo):
    """Pin an edited episode, calibrated robot and shared tracker inputs."""

    def prepare():
        import shutil

        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.flash.demo_execution import compile_demo

        from .robot import _make_robot_from_profile

        setup = _read_settings()
        profile, _ = _profiles(setup, require_teleop=False)
        if setup.camera not in _calibrated_cameras(profile):
            raise ValueError("Choose a calibrated camera in Settings")
        entries = _read_library()["demos"]
        entry = next((item for item in entries if item["id"] == request.id), None)
        if entry is None:
            raise ValueError("Choose a saved demo")
        dataset = LeRobotDataset(Path(entry["root"]).name, root=entry["root"])
        document, _ = load_document(dataset, entry["episode"])
        folder = document_path(dataset, entry["episode"]).with_suffix("")
        if not (folder / "latest.json").exists():
            raise ValueError("Track objects in this demo before applying it")
        latest = json.loads((folder / "latest.json").read_text())["directory"]
        tracking_folder = folder / latest
        if tracking_folder.parent != folder:
            raise ValueError("Invalid tracking output directory")
        robot = _make_robot_from_profile(profile)  # No connect, no camera construction.
        plan = compile_demo(dataset, entry["episode"], document, tracking_folder, robot, setup.camera)
        if (
            setup.depth_source != plan["depth_source"]
            or setup.model_dump(mode="json")["pipeline_settings"] != plan["settings"]
        ):
            raise ValueError(
                "Setup changed since tracking. Track the demo with this setup before applying it."
            )
        identity = uuid.uuid4().hex
        destination = EDITOR_DIR / "plans" / identity
        atomic_json(destination / "plan.json", plan)
        atomic_json(
            destination / "source.json", {"entry": entry, "profile": profile, "setup": setup.model_dump()}
        )
        shutil.copyfile(tracking_folder / "reference.npz", destination / "reference.npz")
        return {
            "plan": identity,
            "name": document.name,
            "keyframes": [
                {
                    "frame": k["frame"],
                    "reference": k["reference"],
                    "reference_frame": k["reference_frame"],
                    "gripper": k["gripper"],
                }
                for k in plan["keyframes"]
            ],
            "robot": setup.robot,
            "fps": dataset.fps,
            "objects": {obj.id: obj.name for obj in document.objects},
        }

    try:
        return await asyncio.get_running_loop().run_in_executor(_io, prepare)
    except (ValueError, KeyError, OSError, NotImplementedError, ImportError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/apply")
async def apply_demo(request: ApplyDemo):
    """Test and robot runs use the same controller, tracker and camera owner."""
    from . import flash_tracking

    global _preview_process, _execution_process, _execution_directory

    def prepare():
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.flash.demo_execution import PLAN_VERSION

        folder = EDITOR_DIR / "plans" / request.plan
        plan = json.loads((folder / "plan.json").read_text())
        if plan.get("version") != PLAN_VERSION:
            raise ValueError("Execution rules changed; prepare the demo again")
        source = json.loads((folder / "source.json").read_text())
        setup = _read_settings()
        profile, _ = _profiles(setup, require_teleop=False)
        if setup.model_dump(mode="json") != source["setup"] or profile != source["profile"]:
            raise ValueError("Setup changed; prepare the demo again")
        if not request.simulate and not profile.get("rest_position"):
            raise ValueError("Save a rest position in the robot profile before running a demo")
        entry = source["entry"]
        dataset = LeRobotDataset(Path(entry["root"]).name, root=entry["root"])
        document, _ = load_document(dataset, entry["episode"])
        if document.model_dump(mode="json") != plan["document"]:
            raise ValueError("Demo changed; prepare it again")
        if setup.camera not in _calibrated_cameras(profile):
            raise ValueError("Selected camera is not calibrated")
        camera = profile["cameras"][setup.camera]
        if camera["type"] != "intelrealsense":
            raise ValueError("The shared live worker currently requires RealSense")
        return (
            flash_tracking.TrackingStart(
                serial=str(camera["serial_number_or_name"]),
                width=camera["width"],
                height=camera["height"],
                fps=camera.get("fps", setup.fps),
                pipeline_settings=plan["settings"],
                depth_source=plan["depth_source"],
            ),
            {
                "demo_execution": {
                    "plan_directory": str(folder),
                    "robot": profile,
                    "simulate": request.simulate,
                },
                "initial_selection": {"id": "editor-demo", "action": "select", "prompt": "demo objects"},
            },
            source,
        )

    try:
        camera, extra, source = await asyncio.get_running_loop().run_in_executor(_io, prepare)
    except (ValueError, KeyError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    # Stop only the editor's own camera preview. An unrelated Run is rejected
    # by start_worker instead of being stolen by this page.
    await stop_preview()
    identity = "rollout-" + uuid.uuid4().hex
    destination = EDITOR_DIR / "rollouts" / identity

    def snapshot():
        import shutil

        folder = Path(extra["demo_execution"]["plan_directory"])
        atomic_json(
            destination / "run.json",
            {
                "schema_version": 1,
                "id": identity,
                "name": source["entry"]["name"],
                "demo_id": source["entry"]["id"],
                "created_at": time.time(),
                "state": "running",
                "outcome": "unknown",
                "mode": "simulation" if request.simulate else "robot",
            },
        )
        for name in ("plan.json", "source.json", "reference.npz"):
            shutil.copyfile(folder / name, destination / name)

    await asyncio.get_running_loop().run_in_executor(_io, snapshot)
    extra["demo_execution"]["plan_directory"] = str(destination)
    try:
        await flash_tracking.start_worker(
            camera,
            module="lerobot.policies.flash.demo_live_worker",
            extra=extra,
            output_directory=destination,
        )
    except BaseException as exc:
        from lerobot.policies.flash.rollout_recording import finish_run

        await asyncio.get_running_loop().run_in_executor(
            _io, partial(finish_run, destination, state="failed", reason=str(exc))
        )
        raise
    _preview_process = _execution_process = run._active_process
    _execution_directory = flash_tracking._directory
    return {
        "pid": _execution_process.pid,
        "mode": "simulation" if request.simulate else "robot",
        "rollout": identity,
    }


@router.get("/execution")
async def execution():
    def read():
        if _execution_process is None:
            return {"running": False}
        path = _execution_directory / "execution.json"
        value = json.loads(path.read_text()) if path.exists() else {"state": "starting"}
        return {
            **value,
            "running": _execution_process.returncode is None,
            "pid": _execution_process.pid,
            "returncode": _execution_process.returncode,
        }

    return await asyncio.get_running_loop().run_in_executor(_io, read)
