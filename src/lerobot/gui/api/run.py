"""Run tab API: launch teleoperate/record/replay subprocesses and stream output."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/run", tags=["run"])

_app_state: "AppState" = None  # type: ignore


def set_app_state(state: "AppState") -> None:
    global _app_state
    _app_state = state


# ============================================================================
# Profile → CLI args conversion
# ============================================================================


def _get_known_fields(profile_type: str, prefix: str) -> set[str] | None:
    """Return the set of valid field names for a config type, or None if unknown.

    Imports all robot/teleoperator modules to ensure config subclasses are registered.
    """
    import dataclasses
    import importlib
    import pkgutil

    import lerobot.robots
    import lerobot.teleoperators
    from lerobot.robots.config import RobotConfig
    from lerobot.teleoperators.config import TeleoperatorConfig

    # Trigger registration of all config subclasses
    pkg = lerobot.robots if prefix == "robot" else lerobot.teleoperators
    for _importer, modname, _ispkg in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        try:
            importlib.import_module(modname)
        except Exception:
            pass

    base_cls = RobotConfig if prefix == "robot" else TeleoperatorConfig
    choices = base_cls.get_known_choices()
    config_cls = choices.get(profile_type)
    if config_cls is None:
        return None
    return {f.name for f in dataclasses.fields(config_cls)}


def _profile_to_cli_args(profile_data: dict, prefix: str, *, include_cameras: bool = True) -> list[str]:
    """Convert a profile data dict to draccus CLI arguments.

    Args:
        profile_data: {type, fields, cameras} from the frontend.
        prefix: "robot" or "teleop".
        include_cameras: Whether to include camera args (False for replay,
            which doesn't import camera config modules).

    Returns:
        List of CLI arg strings like ["--robot.type=bi_so107_follower", ...].
    """
    args = [f"--{prefix}.type={profile_data['type']}"]
    known_fields = _get_known_fields(profile_data["type"], prefix)

    for key, value in profile_data.get("fields", {}).items():
        if value is None:
            continue
        if known_fields is not None and key not in known_fields:
            logger.warning(
                f"Ignoring unknown {prefix} field '{key}' (not in {profile_data['type']} config). "
                f"It may belong to a different branch or have been removed."
            )
            continue
        if isinstance(value, bool):
            args.append(f"--{prefix}.{key}={str(value).lower()}")
        else:
            args.append(f"--{prefix}.{key}={value}")

    if include_cameras:
        cameras = profile_data.get("cameras", {})
        if cameras:
            args.append(f"--{prefix}.cameras={json.dumps(cameras)}")

    return args


# ============================================================================
# Request models
# ============================================================================


class TeleoperateRequest(BaseModel):
    robot: dict[str, Any]
    teleop: dict[str, Any]
    fps: int = 60


class RecordRequest(BaseModel):
    robot: dict[str, Any]
    teleop: dict[str, Any] | None = None
    repo_id: str
    root: str | None = None
    policy_path: str | None = None
    single_task: str
    fps: int = 30
    episode_time_s: float = 60
    reset_time_s: float = 60
    num_episodes: int = 50
    video: bool = True
    vcodec: str = "libsvtav1"
    play_sounds: bool = True
    resume: bool = False


class ReplayRequest(BaseModel):
    robot: dict[str, Any]
    repo_id: str
    root: str | None = None
    episode: int
    fps: int = 30


class HVLARunRequest(BaseModel):
    robot: dict[str, Any]
    s1_checkpoint: str
    s2_checkpoint: str
    task: str
    fps: int = 30
    s1_type: str = "flow"
    decode_subtask: bool = False
    record_dataset: str | None = None
    num_episodes: int = 1
    episode_time_s: float = 60
    reset_time_s: float = 20


# ============================================================================
# Subprocess state
# ============================================================================

_active_process: asyncio.subprocess.Process | None = None
_active_command: str | None = None
_active_config: dict | None = None
_output_lines: list[str] = []
_output_event: asyncio.Event = asyncio.Event()
_OUTPUT_MAX_LINES = 2000


def _append_output(line: str) -> None:
    """Append a line to the output buffer and notify SSE waiters."""
    global _output_lines
    _output_lines.append(line)
    if len(_output_lines) > _OUTPUT_MAX_LINES:
        _output_lines = _output_lines[-_OUTPUT_MAX_LINES:]
    _output_event.set()


def _ensure_no_active_process() -> None:
    """Raise 409 if a process is already running."""
    if _active_process is not None and _active_process.returncode is None:
        raise HTTPException(
            409, f"A '{_active_command}' process is already running (PID {_active_process.pid})"
        )


async def _read_stream(stream: asyncio.StreamReader, prefix: str = "") -> None:
    """Read lines from a subprocess stream and append to output buffer."""
    while True:
        line = await stream.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip("\n\r")
        _append_output(f"{prefix}{text}")


_stream_tasks: list[asyncio.Task] = []


async def _wait_for_exit() -> None:
    """Wait for the subprocess to finish, log the exit, and clean up."""
    global _active_process, _active_command, _active_config
    if _active_process is None:
        return
    proc = _active_process  # capture reference before clearing
    await proc.wait()
    # Wait for stream readers to finish draining pipes before clearing state,
    # otherwise the SSE sends "done" before error output is captured.
    if _stream_tasks:
        await asyncio.gather(*_stream_tasks, return_exceptions=True)
    rc = proc.returncode
    _append_output(f"\n--- Process exited with code {rc} ---")
    # Only clear if this is still the same process (not replaced by a new launch)
    if _active_process is proc:
        _active_process = None
        _active_command = None
        _active_config = None
        _close_obs_reader()
        logger.info(f"Process exited (rc={rc}), state cleared")
    _output_event.set()




async def _launch_subprocess(args: list[str], command: str, config: dict) -> None:
    """Launch a subprocess and start reading its output."""
    global _active_process, _active_command, _active_config, _output_lines, _stream_tasks

    _output_lines = []
    _active_command = command
    _active_config = config

    env = {**__import__("os").environ, "LEROBOT_OBS_STREAM": "1"}
    cmd_str = " ".join(args)
    logger.info(f"Launching: {cmd_str}")
    _append_output(f"--- Starting {command} ---")
    _append_output(f"$ {cmd_str}\n")

    _active_process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    _stream_tasks = [
        asyncio.create_task(_read_stream(_active_process.stdout)),
        asyncio.create_task(_read_stream(_active_process.stderr, prefix="[stderr] ")),
    ]
    asyncio.create_task(_wait_for_exit())


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/teleoperate")
async def start_teleoperate(req: TeleoperateRequest) -> dict:
    _ensure_no_active_process()

    args = ["lerobot-teleoperate"]
    args.extend(_profile_to_cli_args(req.robot, "robot"))
    args.extend(_profile_to_cli_args(req.teleop, "teleop"))
    args.append(f"--fps={req.fps}")

    await _launch_subprocess(args, command="teleoperate", config=req.model_dump())
    return {"status": "started", "command": "teleoperate", "pid": _active_process.pid}


@router.post("/record")
async def start_record(req: RecordRequest) -> dict:
    _ensure_no_active_process()

    if req.teleop is None and req.policy_path is None:
        raise HTTPException(400, "Either teleop or policy_path must be provided")

    args = ["lerobot-record"]
    args.extend(_profile_to_cli_args(req.robot, "robot"))
    if req.teleop is not None:
        args.extend(_profile_to_cli_args(req.teleop, "teleop"))
    if req.policy_path:
        args.append(f"--policy.path={req.policy_path}")
    args.append(f"--dataset.repo_id={req.repo_id}")
    if req.root:
        args.append(f"--dataset.root={req.root}")
    args.append(f"--dataset.single_task={req.single_task}")
    args.append(f"--dataset.fps={req.fps}")
    args.append(f"--dataset.episode_time_s={req.episode_time_s}")
    args.append(f"--dataset.reset_time_s={req.reset_time_s}")
    args.append(f"--dataset.num_episodes={req.num_episodes}")
    args.append(f"--dataset.video={'true' if req.video else 'false'}")
    args.append("--dataset.push_to_hub=false")
    args.append(f"--dataset.vcodec={req.vcodec}")
    args.append(f"--play_sounds={'true' if req.play_sounds else 'false'}")
    if req.resume:
        args.append("--resume=true")

    await _launch_subprocess(args, command="record", config=req.model_dump())
    return {"status": "started", "command": "record", "pid": _active_process.pid}


@router.post("/replay")
async def start_replay(req: ReplayRequest) -> dict:
    _ensure_no_active_process()

    args = ["lerobot-replay"]
    args.extend(_profile_to_cli_args(req.robot, "robot"))
    args.append(f"--dataset.repo_id={req.repo_id}")
    if req.root:
        args.append(f"--dataset.root={req.root}")
    args.append(f"--dataset.episode={req.episode}")
    args.append(f"--dataset.fps={req.fps}")

    await _launch_subprocess(args, command="replay", config=req.model_dump())
    return {"status": "started", "command": "replay", "pid": _active_process.pid}


@router.post("/hvla")
async def start_hvla(req: HVLARunRequest) -> dict:
    """Launch HVLA dual-system inference (S1 + S2)."""
    import tempfile

    _ensure_no_active_process()

    # Write robot profile to temp file (HVLA launch reads robot config from file)
    robot_config = dict(req.robot)
    if "fields" in robot_config:
        flat = {"type": robot_config["type"]}
        flat.update(robot_config["fields"])
        if "cameras" in robot_config:
            flat["cameras"] = robot_config["cameras"]
        robot_config = flat

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, prefix="hvla_robot_")
    tmp.write(json.dumps(robot_config, indent=2))
    tmp.close()

    args = [
        "python", "-m", "lerobot.policies.hvla.launch",
        f"--s1-checkpoint={req.s1_checkpoint}",
        f"--s2-checkpoint={Path(req.s2_checkpoint).expanduser()}",
        f"--task={req.task}",
        f"--robot-config={tmp.name}",
        f"--fps={req.fps}",
        f"--s1-type={req.s1_type}",
    ]
    if req.decode_subtask:
        args.append("--decode-subtask")
    if req.record_dataset:
        args.append(f"--record-dataset={req.record_dataset}")
    args.append(f"--num-episodes={req.num_episodes}")
    args.append(f"--episode-time-s={req.episode_time_s}")
    args.append(f"--reset-time-s={req.reset_time_s}")

    await _launch_subprocess(args, command="hvla", config=req.model_dump())
    return {"status": "started", "command": "hvla", "pid": _active_process.pid}


@router.post("/stop")
async def stop_process() -> dict:
    global _active_process, _active_command, _active_config

    if _active_process is None:
        raise HTTPException(409, "No active process to stop")

    pid = _active_process.pid
    try:
        _active_process.send_signal(signal.SIGINT)
        try:
            await asyncio.wait_for(_active_process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            _active_process.kill()
            await _active_process.wait()
    except ProcessLookupError:
        pass

    _append_output(f"\n--- Process stopped (PID {pid}) ---\n")
    _active_process = None
    _active_command = None
    _active_config = None
    _close_obs_reader()
    return {"status": "stopped", "pid": pid}


@router.get("/status")
async def get_status() -> dict:
    if _active_process is None:
        return {"running": False, "command": None}

    returncode = _active_process.returncode
    if returncode is not None:
        return {
            "running": False,
            "command": _active_command,
            "returncode": returncode,
        }
    return {
        "running": True,
        "command": _active_command,
        "pid": _active_process.pid,
    }


@router.get("/output")
async def stream_output() -> StreamingResponse:
    """Server-Sent Events stream of subprocess output."""

    async def event_generator():
        sent_lines = 0
        proc_at_start = _active_process

        logger.info(f"SSE: connected, buffered={len(_output_lines)} lines, process={'running' if proc_at_start and proc_at_start.returncode is None else 'none/done'}")

        # Send existing buffered lines first
        while sent_lines < len(_output_lines):
            line = _output_lines[sent_lines]
            yield f"data: {json.dumps({'line': line})}\n\n"
            sent_lines += 1

        # Then wait for new lines
        while True:
            _output_event.clear()

            while sent_lines < len(_output_lines):
                line = _output_lines[sent_lines]
                yield f"data: {json.dumps({'line': line})}\n\n"
                sent_lines += 1

            # Check if process is done
            if _active_process is None or _active_process.returncode is not None:
                while sent_lines < len(_output_lines):
                    line = _output_lines[sent_lines]
                    yield f"data: {json.dumps({'line': line})}\n\n"
                    sent_lines += 1
                logger.info(f"SSE: done (process={'exited rc='+str(_active_process.returncode) if _active_process else 'None'}, sent={sent_lines} lines)")
                yield f"data: {json.dumps({'done': True})}\n\n"
                return

            try:
                await asyncio.wait_for(_output_event.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================================================
# Observation stream endpoints (shared-memory camera/state viewer)
# ============================================================================

_obs_reader = None  # ObservationStreamReader | None
_jpeg_cache: dict[str, tuple[int, bytes]] = {}  # cam_key → (seq, jpeg_bytes)


def _get_obs_reader():
    """Lazily attach to the robot's observation stream."""
    global _obs_reader
    if _obs_reader is not None:
        return _obs_reader
    try:
        from lerobot.robots.obs_stream import ObservationStreamReader

        _obs_reader = ObservationStreamReader()
        logger.info(
            "ObservationStreamReader attached: %d scalars, %d cameras",
            len(_obs_reader.obs_scalar_keys),
            len(_obs_reader.image_keys),
        )
        return _obs_reader
    except Exception:
        return None


def _close_obs_reader():
    global _obs_reader
    if _obs_reader is not None:
        _obs_reader.close()
        _obs_reader = None
    _jpeg_cache.clear()


@router.get("/obs-stream/meta")
async def obs_stream_meta() -> dict:
    """Return observation stream layout (feature names, image dims)."""
    reader = _get_obs_reader()
    if reader is None:
        return {"available": False}
    return {
        "available": True,
        "obs_scalar_keys": reader.obs_scalar_keys,
        "action_keys": reader.action_keys,
        "image_keys": reader.image_keys,
    }


@router.get("/obs-stream/state")
async def obs_stream_state() -> dict:
    """Return latest scalar observations and actions."""
    reader = _get_obs_reader()
    if reader is None:
        raise HTTPException(503, "Observation stream not available")
    obs_result = reader.read_obs()
    act_result = reader.read_action()
    return {
        "obs": obs_result[0] if obs_result else None,
        "obs_ts": obs_result[1] if obs_result else None,
        "action": act_result[0] if act_result else None,
        "action_ts": act_result[1] if act_result else None,
    }


@router.get("/obs-stream/image/{cam_key}")
async def obs_stream_image(cam_key: str) -> Response:
    """Return latest camera frame as JPEG (cached until new frame arrives)."""
    reader = _get_obs_reader()
    if reader is None:
        raise HTTPException(503, "Observation stream not available")

    # Skip re-encoding if the frame hasn't changed
    seq = reader.image_seq(cam_key)
    cached = _jpeg_cache.get(cam_key)
    if cached is not None and cached[0] == seq:
        return Response(
            content=cached[1],
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    result = reader.read_image(cam_key)
    if result is None:
        raise HTTPException(404, f"No image for camera '{cam_key}'")
    img, _ts = result

    import cv2

    _, jpeg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    jpeg_bytes = jpeg.tobytes()
    _jpeg_cache[cam_key] = (seq, jpeg_bytes)
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )
