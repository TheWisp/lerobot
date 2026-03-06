"""Run tab API: launch teleoperate/record/replay subprocesses and stream output."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import subprocess
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
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
# Rerun server
# ============================================================================

_rerun_started = False
RERUN_GRPC_PORT = 9876
RERUN_WEB_PORT = 9090


def _is_port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def init_rerun_server() -> None:
    """Serve the Rerun web viewer (static files only).

    The subprocess itself hosts a gRPC server via LEROBOT_RERUN_SERVE_PORT,
    and the web viewer connects to it via the ?url= query parameter in the
    iframe src.
    """
    global _rerun_started
    if _rerun_started:
        return
    try:
        import rerun as rr

        # Kill anything on the web viewer port so we own it
        if _is_port_in_use(RERUN_WEB_PORT):
            logger.info(f"Killing stale process on web viewer port {RERUN_WEB_PORT}")
            try:
                subprocess.run(["fuser", "-k", f"{RERUN_WEB_PORT}/tcp"], capture_output=True, timeout=5)
            except Exception as e:
                logger.warning(f"fuser -k {RERUN_WEB_PORT}/tcp failed: {e}")
            import time
            for i in range(10):
                if not _is_port_in_use(RERUN_WEB_PORT):
                    break
                time.sleep(0.2)

        rr.serve_web_viewer(web_port=RERUN_WEB_PORT, open_browser=False)
        logger.info(f"Rerun web viewer serving on port {RERUN_WEB_PORT}")
        _rerun_started = True
    except Exception as e:
        logger.warning(f"Failed to start Rerun web viewer: {e}")


@router.get("/rerun-ports")
async def get_rerun_ports() -> dict:
    """Return Rerun server ports for the frontend iframe."""
    web_ok = _is_port_in_use(RERUN_WEB_PORT)
    logger.info(f"Rerun health: started={_rerun_started}, web={web_ok}")
    return {
        "available": _rerun_started and web_ok,
        "web_port": RERUN_WEB_PORT if _rerun_started else None,
        "grpc_port": RERUN_GRPC_PORT if _rerun_started else None,
    }


# ============================================================================
# Profile → CLI args conversion
# ============================================================================


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

    for key, value in profile_data.get("fields", {}).items():
        if value is None:
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
        logger.info(f"Process exited (rc={rc}), state cleared")
    _output_event.set()




async def _launch_subprocess(args: list[str], command: str, config: dict) -> None:
    """Launch a subprocess and start reading its output."""
    global _active_process, _active_command, _active_config, _output_lines, _stream_tasks

    # Kill any leftover gRPC server on the port from a previous run
    if _is_port_in_use(RERUN_GRPC_PORT):
        logger.info(f"Killing stale process on gRPC port {RERUN_GRPC_PORT}")
        try:
            subprocess.run(["fuser", "-k", f"{RERUN_GRPC_PORT}/tcp"], capture_output=True, timeout=5)
        except Exception as e:
            logger.warning(f"fuser -k {RERUN_GRPC_PORT}/tcp failed: {e}")

    _output_lines = []
    _active_command = command
    _active_config = config

    rerun_extra = _rerun_env()
    env = {**__import__("os").environ, **rerun_extra}
    cmd_str = " ".join(args)
    logger.info(f"Launching: {cmd_str}")
    if rerun_extra:
        logger.info(f"Rerun env vars: {rerun_extra}")
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


def _display_args() -> list[str]:
    """CLI args to enable Rerun display in the subprocess."""
    if not _rerun_started:
        logger.info("Rerun not started — skipping display args")
        return []
    args = ["--display_data=true", "--display_compressed_images=true"]
    logger.info(f"Rerun display args: {args}")
    return args


def _rerun_env() -> dict[str, str]:
    """Extra env vars for the subprocess to self-host a gRPC server."""
    if not _rerun_started:
        return {}
    return {"LEROBOT_RERUN_SERVE_PORT": str(RERUN_GRPC_PORT)}


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
    args.extend(_display_args())

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
    args.extend(_display_args())

    await _launch_subprocess(args, command="record", config=req.model_dump())
    return {"status": "started", "command": "record", "pid": _active_process.pid}


@router.post("/replay")
async def start_replay(req: ReplayRequest) -> dict:
    _ensure_no_active_process()

    args = ["lerobot-replay"]
    args.extend(_profile_to_cli_args(req.robot, "robot", include_cameras=False))
    args.append(f"--dataset.repo_id={req.repo_id}")
    if req.root:
        args.append(f"--dataset.root={req.root}")
    args.append(f"--dataset.episode={req.episode}")
    args.append(f"--dataset.fps={req.fps}")

    await _launch_subprocess(args, command="replay", config=req.model_dump())
    return {"status": "started", "command": "replay", "pid": _active_process.pid}


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
