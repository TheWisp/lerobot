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
_rerun_process: subprocess.Popen | None = None
RERUN_GRPC_PORT = 9876
RERUN_WEB_PORT = 9090


def _is_port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def init_rerun_server() -> None:
    """Start a standalone Rerun server process (gRPC + web viewer)."""
    global _rerun_started, _rerun_process
    if _rerun_started:
        return
    try:
        import shutil
        import time

        rerun_bin = shutil.which("rerun")
        if not rerun_bin:
            logger.warning("'rerun' CLI not found in PATH — Rerun viewer disabled")
            return

        # If ports are already in use (stale process), treat as already running
        grpc_used = _is_port_in_use(RERUN_GRPC_PORT)
        web_used = _is_port_in_use(RERUN_WEB_PORT)
        logger.info(f"Rerun port check: gRPC={RERUN_GRPC_PORT} {'IN USE' if grpc_used else 'free'}, web={RERUN_WEB_PORT} {'IN USE' if web_used else 'free'}")
        if grpc_used and web_used:
            logger.info("Rerun ports already in use — reusing existing server")
            _rerun_started = True
            return

        cmd = [
            rerun_bin,
            "--serve-web",
            "--port", str(RERUN_GRPC_PORT),
            "--web-viewer-port", str(RERUN_WEB_PORT),
            "--server-memory-limit", "25%",
        ]
        logger.info(f"Starting Rerun server: {' '.join(cmd)}")
        _rerun_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # Give it a moment to start (or fail)
        time.sleep(1)
        if _rerun_process.poll() is not None:
            stderr = _rerun_process.stderr.read().decode() if _rerun_process.stderr else ""
            logger.warning(f"Rerun server exited immediately (rc={_rerun_process.returncode}): {stderr.strip()}")
            _rerun_process = None
            return

        # Verify ports are actually open
        grpc_ok = _is_port_in_use(RERUN_GRPC_PORT)
        web_ok = _is_port_in_use(RERUN_WEB_PORT)
        logger.info(f"Rerun server PID {_rerun_process.pid}: gRPC={RERUN_GRPC_PORT} {'OK' if grpc_ok else 'FAILED'}, web={RERUN_WEB_PORT} {'OK' if web_ok else 'FAILED'}")

        if not grpc_ok or not web_ok:
            logger.warning("Rerun server started but ports are not listening — viewer may not work")

        _rerun_started = True
    except Exception as e:
        logger.warning(f"Failed to start Rerun server: {e}")


@router.get("/rerun-ports")
async def get_rerun_ports() -> dict:
    """Return Rerun server ports for the frontend iframe."""
    # Live-check if the rerun process is still alive
    process_alive = _rerun_process is not None and _rerun_process.poll() is None
    grpc_ok = _is_port_in_use(RERUN_GRPC_PORT)
    web_ok = _is_port_in_use(RERUN_WEB_PORT)
    logger.info(f"Rerun health: started={_rerun_started}, process_alive={process_alive}, gRPC={grpc_ok}, web={web_ok}")
    return {
        "available": _rerun_started and (grpc_ok or web_ok),
        "web_port": RERUN_WEB_PORT if _rerun_started else None,
        "grpc_port": RERUN_GRPC_PORT if _rerun_started else None,
    }


# ============================================================================
# Profile → CLI args conversion
# ============================================================================


def _profile_to_cli_args(profile_data: dict, prefix: str) -> list[str]:
    """Convert a profile data dict to draccus CLI arguments.

    Args:
        profile_data: {type, fields, cameras} from the frontend.
        prefix: "robot" or "teleop".

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
    teleop: dict[str, Any]
    repo_id: str
    single_task: str
    fps: int = 30
    episode_time_s: float = 60
    reset_time_s: float = 60
    num_episodes: int = 50
    video: bool = True
    push_to_hub: bool = False
    vcodec: str = "libsvtav1"
    play_sounds: bool = True
    resume: bool = False


class ReplayRequest(BaseModel):
    robot: dict[str, Any]
    repo_id: str
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


async def _wait_for_exit() -> None:
    """Wait for the subprocess to finish, log the exit, and clean up."""
    global _active_process, _active_command, _active_config
    if _active_process is None:
        return
    proc = _active_process  # capture reference before clearing
    await proc.wait()
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
    global _active_process, _active_command, _active_config, _output_lines

    _output_lines = []
    _active_command = command
    _active_config = config

    cmd_str = " ".join(args)
    logger.info(f"Launching: {cmd_str}")
    _append_output(f"--- Starting {command} ---")
    _append_output(f"$ {cmd_str}\n")

    _active_process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    asyncio.create_task(_read_stream(_active_process.stdout))
    asyncio.create_task(_read_stream(_active_process.stderr, prefix="[stderr] "))
    asyncio.create_task(_wait_for_exit())


def _display_args() -> list[str]:
    """CLI args to connect subprocess to our Rerun server."""
    if not _rerun_started:
        logger.info("Rerun not started — skipping display args")
        return []
    args = [
        "--display_data=true",
        f"--display_ip=127.0.0.1",
        f"--display_port={RERUN_GRPC_PORT}",
    ]
    logger.info(f"Rerun display args: {args}")
    return args


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

    args = ["lerobot-record"]
    args.extend(_profile_to_cli_args(req.robot, "robot"))
    args.extend(_profile_to_cli_args(req.teleop, "teleop"))
    args.append(f"--dataset.repo_id={req.repo_id}")
    args.append(f"--dataset.single_task={req.single_task}")
    args.append(f"--dataset.fps={req.fps}")
    args.append(f"--dataset.episode_time_s={req.episode_time_s}")
    args.append(f"--dataset.reset_time_s={req.reset_time_s}")
    args.append(f"--dataset.num_episodes={req.num_episodes}")
    args.append(f"--dataset.video={'true' if req.video else 'false'}")
    args.append(f"--dataset.push_to_hub={'true' if req.push_to_hub else 'false'}")
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
    args.extend(_profile_to_cli_args(req.robot, "robot"))
    args.append(f"--dataset.repo_id={req.repo_id}")
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
