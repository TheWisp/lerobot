"""Run tab API: launch teleoperate/record/replay subprocesses and stream output."""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import json
import logging
import os
import re
import secrets
import signal
import sys
import tempfile
import time
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/run", tags=["run"])

# Tuple of timeout-error classes to catch from `asyncio.wait_for(...)`.
#
# Why this exists: pyupgrade and ruff UP041 ("timeout-error-alias") rewrite
# any literal `asyncio.TimeoutError` token in the source to `TimeoutError`
# under py311+ targets, and pyproject.toml declares
# `requires-python = ">=3.12"`. But the project's actual conda env runs
# Python 3.10 (per CLAUDE.md / project memory), where
# `asyncio.TimeoutError` is still a *distinct* class from builtin
# `TimeoutError` — the alias only landed in 3.11. So a naive
# `except asyncio.TimeoutError:` (or even
# `except (TimeoutError, asyncio.TimeoutError):`) gets rewritten to
# `except TimeoutError:`, which then doesn't catch on 3.10 and the SSE
# keepalive crashes with an unhandled traceback every 2s.
#
# Resolution: the except clauses reference _TIMEOUT_EXCS by name so no
# rewriteable literal `asyncio.TimeoutError` token appears in an except
# context. The constant assignment itself isn't a UP041 target.
_TIMEOUT_EXCS: tuple[type[BaseException], ...] = (TimeoutError, asyncio.TimeoutError)

# Where the teleop subprocess writes its latency_snapshot.json. The frontend
# reads this via /api/run/latency-metrics (cross-process). Mirrors the RLT
# pattern (outputs/rlt_online/metrics.json).
LATENCY_OUTPUT_DIR_TELEOP = "outputs/teleop"
LATENCY_OUTPUT_DIR_RECORD = "outputs/record"
# Back-compat alias: existing call sites pass this for teleop. Don't rename
# without updating every caller.
LATENCY_OUTPUT_DIR = LATENCY_OUTPUT_DIR_TELEOP

# Known latency snapshot sources. Each subprocess (teleop / record / HVLA)
# writes ``latency_snapshot.json`` into its OWN directory so the dashboard
# doesn't render the same data twice when two source keys would otherwise
# share a file. /api/run/latency-metrics?source=<key> reads from the
# matching directory. Adding a new loop kind means appending to this map
# and (if needed) plumbing the output_dir through that script's CLI args.
# Keys must match the ``loop_kind`` field the writer publishes.
# HVLA writes one snapshot per thread under outputs/hvla_runs/<track>/, so
# the dashboard can render the control thread (track=main) and the
# inference thread (track=inference) as stacked tracks of the same
# process. Single-track loops (teleop, record) write directly into their
# loop directory.
LATENCY_SOURCES: dict[str, str] = {
    "teleop": LATENCY_OUTPUT_DIR_TELEOP,
    "record": LATENCY_OUTPUT_DIR_RECORD,
    "hvla_main": "outputs/hvla_runs/main",
    "hvla_infer": "outputs/hvla_runs/inference",
}

_app_state: AppState = None  # type: ignore


def set_app_state(state: AppState) -> None:
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
    for _importer, modname, _ispkg in pkgutil.walk_packages(
        pkg.__path__, prefix=pkg.__name__ + ".", onerror=lambda _name: None
    ):
        with contextlib.suppress(Exception):
            importlib.import_module(modname)

    base_cls = RobotConfig if prefix == "robot" else TeleoperatorConfig
    choices = base_cls.get_known_choices()
    config_cls = choices.get(profile_type)
    if config_cls is None:
        return None

    def leaf_paths(cls: type, path_prefix: str = "") -> set[str]:
        try:
            import typing

            type_hints = typing.get_type_hints(cls)
        except Exception:
            type_hints = {}

        result = set()
        for field in dataclasses.fields(cls):
            path = f"{path_prefix}.{field.name}" if path_prefix else field.name
            resolved_type = type_hints.get(field.name, field.type)
            if dataclasses.is_dataclass(resolved_type):
                result.update(leaf_paths(resolved_type, path))
            else:
                result.add(path)
        return result

    return leaf_paths(config_cls)


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

    def iter_values(values: dict, path_prefix: str = ""):
        for key, value in values.items():
            path = f"{path_prefix}.{key}" if path_prefix else key
            # Only recurse when the config schema says this mapping is a
            # nested dataclass. Dict-valued leaf fields (joint limits, motor
            # config, etc.) must remain one JSON CLI value.
            is_nested_config = known_fields is not None and any(
                field.startswith(f"{path}.") for field in known_fields
            )
            if isinstance(value, dict) and is_nested_config:
                yield from iter_values(value, path)
            else:
                yield path, value

    for key, value in iter_values(profile_data.get("fields", {})):
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
        elif isinstance(value, (dict, list)):
            args.append(f"--{prefix}.{key}={json.dumps(value)}")
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


class DebugModelConfig(BaseModel):
    checkpoint: str
    policy_type: str
    task: str = ""
    decode_subtask: bool = True


class TeleoperateRequest(BaseModel):
    robot: dict[str, Any]
    teleop: dict[str, Any]
    fps: int = 60
    debug_model: DebugModelConfig | None = None


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
    debug_model: DebugModelConfig | None = None
    intervention_repo_id: str | None = None


class ReplayRequest(BaseModel):
    robot: dict[str, Any]
    repo_id: str
    root: str | None = None
    episode: int
    # No fps: replay must always pace at the dataset's recording fps. Upstream
    # `lerobot-replay` declares `cfg.dataset.fps` but its loop paces by
    # `dataset.fps` directly, so the field is dead config.


class FlashInspectRequest(BaseModel):
    skill_path: str


class FlashDatasetInspectRequest(BaseModel):
    dataset_root: str
    episode: int


class FlashRunRequest(FlashDatasetInspectRequest):
    bundle_path: str
    config_path: str


class FlashConfigInspectRequest(BaseModel):
    config_path: str


class FlashDemoPrepareRequest(FlashDatasetInspectRequest):
    bundle_path: str | None = None
    annotation: dict[str, Any]
    keypoints_document: dict[str, Any]
    output_name: str | None = None


class FlashTrackerImportRequest(FlashDatasetInspectRequest):
    bundle_path: str
    tracker: str
    role: str
    source_path: str


class FlashTrackerProcessRequest(FlashDatasetInspectRequest):
    bundle_path: str
    tracker: str
    role: str | None = None
    max_points_per_object: int | None = None
    new_points_per_view: int | None = None
    surface_tolerance_m: float | None = None


class FlashTrackerApplyRequest(FlashDatasetInspectRequest):
    bundle_path: str
    candidate_id: str
    output_name: str | None = None


class HVLARunRequest(BaseModel):
    robot: dict[str, Any]
    s1_checkpoint: str
    s2_checkpoint: str | None = None
    task: str
    fps: int = 30
    s1_type: str = "flow"
    decode_subtask: bool = False
    s1_query_interval: int | None = None
    denoise_steps: int | None = None
    record_dataset: str | None = None
    num_episodes: int = 1
    episode_time_s: float = 60
    reset_time_s: float = 20
    teleop: dict[str, Any] | None = None
    intervention_dataset: str | None = None
    # RLT (RL Token)
    rlt_mode: bool = False
    rlt_token_checkpoint: str | None = None  # Phase 1: RL token encoder
    rlt_checkpoint: str | None = None  # Phase 2: existing actor checkpoint dir
    rlt_deploy: bool = False  # True = inference only, no training
    rlt_chunk_length: int = 10
    rlt_output_dir: str = "outputs/rlt_online"
    rlt_start_engaged: bool = True
    rlt_shared_noise_per_chunk: bool = True  # default after v2_widened A/B
    # Action send shape on the policy path. "chunk" (default) routes the
    # remaining chunk frames as an ActionChunk → predictive robots use
    # the chunk-exact-lookup path (zero estimation residual). "dict"
    # sends only the single frame at idx → predictive robots fall back
    # to the velocity-LSQ extrapolation path. Non-predictive robots are
    # unaffected (they use frames[0] either way). Primarily for A/B
    # testing the chunk-vs-dict latency improvement; default matches
    # the recommended best-perf path.
    send_action_shape: str = "chunk"


# ============================================================================
# Subprocess state
# ============================================================================

_active_process: asyncio.subprocess.Process | None = None
# The event loop `_active_process` was created on. Only meaningful while that
# process is not None. A subprocess can only be awaited from its own loop, so a
# server running on a different one must not try to stop it — see
# `_terminate_active_process` in gui/server.py, and issue #128 for what happened
# when it did.
_active_loop: asyncio.AbstractEventLoop | None = None
_active_command: str | None = None
_flash_live_path: Path | None = None
_flash_telemetry_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gui-flash-telemetry")
_flash_config_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gui-flash-config")
_flash_candidate_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gui-flash-candidate")
_flash_tracker_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gui-flash-tracker")
_active_config: dict | None = None
_debug_process: asyncio.subprocess.Process | None = None  # optional model debug alongside teleop
_debug_output_path: Path | None = None  # log file for debug model output
_debug_output_lines: list[str] = []
_debug_output_event: asyncio.Event = asyncio.Event()
_debug_read_task: asyncio.Task | None = None
_output_lines: list[str] = []
_output_event: asyncio.Event = asyncio.Event()
_OUTPUT_MAX_LINES = 2000


_overlay_state: dict | None = None  # {"text": "...", "color": "..."}

# Current record-phase shown next to the Run tab's flow-control buttons.
# TODO(run-state): this is parsed from subprocess stdout text, which is
# brittle — any rewording of the log_say messages in lerobot_record breaks
# it silently. Replace with a structured run_state.json published by the
# subprocess (the RLT metrics.json pattern: writer in lerobot_record,
# reader here, polled by the frontend).
_active_phase: str | None = None

# Ordered (pattern, phase-label) rules; first match wins. The {episode}
# placeholder is filled from the capture group when present.
_RUN_PHASE_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"Recording episode (\d+)"), "recording episode {episode}"),
    (re.compile(r"Re-record episode"), "re-recording"),
    (re.compile(r"Reset the environment"), "resetting"),
    (re.compile(r"Auto-reset: moving to trajectory start"), "resetting"),
    (re.compile(r"Stopping data recording"), "stopping"),
]


def _track_run_phase(line: str) -> None:
    """Update _active_phase when a subprocess stdout line marks a phase transition."""
    global _active_phase
    for pattern, label in _RUN_PHASE_RULES:
        m = pattern.search(line)
        if m:
            _active_phase = label.format(episode=m.group(1) if m.groups() else "")
            return


def _append_output(line: str) -> None:
    """Append a line to the output buffer and notify SSE waiters.

    Lines matching ##OVERLAY:text:color## are intercepted as overlay
    updates and not appended to the terminal. Any subprocess can set
    the camera-feed overlay by printing this format to stdout.
    """
    global _output_lines, _overlay_state
    _track_run_phase(line)
    if line.startswith("##OVERLAY:"):
        parts = line.strip().strip("#").split(":")
        # OVERLAY:text or OVERLAY:text:color
        text = parts[1] if len(parts) > 1 else ""
        color = parts[2] if len(parts) > 2 else "#ffffff"
        _overlay_state = {"text": text, "color": color}
        _output_event.set()  # wake SSE to send overlay update
        return
    _output_lines.append(line)
    if len(_output_lines) > _OUTPUT_MAX_LINES:
        _output_lines = _output_lines[-_OUTPUT_MAX_LINES:]
    _output_event.set()


def _refuse_if_deps_missing(robot: dict, teleop: dict | None = None) -> None:
    """Stop before launching when the chosen hardware cannot be driven here.

    The guards inside the subprocess would catch this anyway, several seconds
    later, after cameras and serial ports have been opened and with the
    sentence reaching a log. This is the same question asked of the same
    declarations, at the moment the answer is still useful.
    """
    from lerobot.gui.api.robot import _ensure_configs_loaded
    from lerobot.utils.hardware_extras import extras_for_robot, install_command, missing_extras

    _ensure_configs_loaded()
    missing = missing_extras(
        extras_for_robot(
            robot.get("type"),
            cameras=robot.get("cameras") or {},
            teleop_type=(teleop or {}).get("type"),
        )
    )
    if missing:
        raise HTTPException(
            400,
            f"This robot needs {', '.join(sorted(missing))}, which is not installed here. "
            f"Install it with: {install_command(missing)}",
        )


def _ensure_no_active_process() -> None:
    """Raise 409 if a process is already running; otherwise sweep leftover obs-stream segments.

    Reaching past the check means no run subprocess is alive, so any ``lerobot_obs_*`` segments in
    /dev/shm are orphans from a crashed or hard-stopped run. Sweep them now (and log the count even
    when the cause is unclear — the overlay's surface-loudly policy) so the next run's readers — the
    overlay worker, the run-view camera viewer — can't latch a dead one and freeze (the stale-stream
    bug). The new run creates a fresh stream immediately after.
    """
    if _active_process is not None and _active_process.returncode is None:
        raise HTTPException(
            409, f"A '{_active_command}' process is already running (PID {_active_process.pid})"
        )
    from lerobot.robots.obs_stream import cleanup_stale_streams

    # respect_liveness: this fires mid-session (no GUI-tracked run), where a LIVE external writer (a
    # teleop/feeder started outside the GUI) may own the stream — sweeping it would freeze that reader.
    # Only orphan (no recent write) streams are removed; an active writer makes this a no-op.
    swept = cleanup_stale_streams(respect_liveness=True)
    if swept:
        logger.warning("stale-stream guard: swept %d leftover obs-stream segment(s) before launch", swept)


def is_run_active() -> bool:
    """True while a run subprocess (teleop / record / replay / policy) is alive. It owns the
    obs stream, so the data-tab overlay publisher must yield — see gui/api/overlays.py."""
    return _active_process is not None and _active_process.returncode is None


async def _read_stream(stream: asyncio.StreamReader, prefix: str = "") -> None:
    """Read lines from a subprocess stream and append to output buffer."""
    while True:
        line = await stream.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip("\n\r")
        _append_output(f"{prefix}{text}")


_stream_tasks: list[asyncio.Task] = []

#: Why the last run stopped, when it stopped badly: ``{command, returncode,
#: reason}``. Cleared when the next one is launched. The subprocess's own
#: message is the only thing that knows why — a robot's dependencies are
#: guarded where they are used, inside that process — and clearing the run's
#: state on exit used to take it with it.
_last_failure: dict | None = None

#: The last line of a traceback, which is where Python puts the sentence a
#: person can act on. Anchored so a line merely mentioning an error in
#: passing does not match.
_EXCEPTION_LINE = re.compile(r"^(?:\w+\.)*\w*(?:Error|Exception|Interrupt)\b\s*:\s*(.+)$")


def _reason_from(lines: list[str]) -> str | None:
    """The sentence to show for a run that exited badly.

    Read from the end: the last exception line is the one that stopped it,
    and the frames above it are noise to everyone but whoever fixes the code.
    """
    for line in reversed(lines):
        match = _EXCEPTION_LINE.match(line.strip())
        if match:
            return f"{line.strip().split(':', 1)[0]}: {match.group(1).strip()}"
    return None


def _note_exit(command: str | None, returncode: int | None, lines: list[str]) -> None:
    """Record why a run stopped, if it stopped badly."""
    global _last_failure
    if not returncode:
        return
    _last_failure = {
        "command": command,
        "returncode": returncode,
        "reason": _reason_from(lines) or f"the {command or 'run'} process exited with code {returncode}",
    }


def _clear_last_failure() -> None:
    global _last_failure
    _last_failure = None


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
    _note_exit(_active_command, rc, list(_output_lines))
    _append_output(f"\n--- Process exited with code {rc} ---")
    # Only clear if this is still the same process (not replaced by a new launch)
    if _active_process is proc:
        _active_process = None
        _active_command = None
        _active_config = None
        _close_obs_reader()
        # NOTE: debug model is NOT stopped here — it stays warm for reuse
        logger.info(f"Process exited (rc={rc}), state cleared")
    _output_event.set()


# ── Parent-death detector ────────────────────────────────────────────────
#
# Why: a SIGKILL on the GUI server (e.g. VS Code closing the terminal,
# OOM kill, segfault) leaves any in-flight teleop/record subprocess
# running indefinitely — observed in the wild as a `lerobot-teleoperate`
# subprocess alive almost 2 hours after the GUI exited, holding ttyACM
# ports and cameras and blocking the next launch.
#
# Phase 1 fix: use Linux's PR_SET_PDEATHSIG so the kernel sends SIGTERM
# to the child the moment its parent dies, independent of FastAPI's
# lifespan hooks. Set via preexec_fn (runs in the forked child before
# exec, async-signal-safe — only a single libc.prctl call). Linux-only;
# no-op elsewhere.
#
# Phase 2 (still TODO in gui/TODO.md): file-based heartbeat for the
# corner case where SIGTERM races a stuck child.

# PR_SET_PDEATHSIG value from <sys/prctl.h>. Kept as a module constant so
# the preexec function doesn't reach into ctypes at signal-handler time.
_PR_SET_PDEATHSIG = 1


def _set_pdeathsig_preexec() -> None:
    """preexec_fn for subprocess: ask the kernel to SIGTERM us if our parent dies.

    Runs in the forked child between fork and exec. Stays minimal: a
    single libc.prctl call. Failure is silent (we can't log from
    preexec_fn — the parent would never see it). Worst case we just
    don't get the auto-cleanup behaviour, which is the current state
    anyway.
    """
    if sys.platform != "linux":
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        # prctl(PR_SET_PDEATHSIG, SIGTERM, 0, 0, 0). The signal is
        # cleared on exec of a setuid binary; not a concern here.
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except (OSError, AttributeError):
        pass


async def _launch_subprocess(
    args: list[str], command: str, config: dict, extra_env: dict[str, str] | None = None
) -> None:
    """Launch a subprocess and start reading its output."""
    # TODO(gui-hardware): Manage SocketCAN interface lifecycle here before
    # launching hardware subprocesses. Kernel CAN interfaces lose their
    # config (bitrate / FD mode / up state) on reboot or link-down, and
    # nothing on a stock system re-arms them — today the operator must run
    # `ip link set canX type can bitrate ... fd on up` manually after every
    # reboot (see docs/source/damiao.mdx), and a forgotten bring-up surfaces
    # only as a late "Network is down" ConnectionError deep inside robot
    # connect. The GUI should instead: (1) parse the required CAN channels
    # from the robot profile (e.g. left/right_arm_config.port for
    # bi_openarm_follower), (2) check each interface's state via
    # `ip -details -j link show canX`, (3) bring it up with the right
    # bitrate/FD settings when down or misconfigured, and (4) report the
    # action in the run terminal. Privilege design is the open question:
    # `ip link` needs CAP_NET_ADMIN — options are a narrowly-scoped sudoers
    # entry for the exact commands, pkexec, or a tiny privileged helper
    # service (systemd/D-Bus); running the whole GUI as root is NOT
    # acceptable. Until this lands, the interim fix is persistent host
    # config via systemd-networkd (.netdev with BitRate/DataBitRate/FDMode).
    global _active_process, _active_command, _active_config, _output_lines, _stream_tasks, _active_phase
    global _active_loop

    _output_lines = []
    _active_command = command
    _active_config = config
    _active_phase = None

    # Release camera previews before the subprocess opens the same devices.
    # The Robot tab holds a V4L2 / librealsense handle per previewed camera for
    # as long as the GUI runs — nothing closed them except server shutdown — so a
    # run launched with previews open competed with the GUI for its own cameras.
    # The complementary guard (refusing to *open* a preview while a run is
    # active) already exists on /api/robot/detect-cameras; this is the other
    # half. Imported here rather than at module scope because robot.py imports
    # is_run_active from this module.
    from lerobot.gui.api.robot import release_preview_cameras

    released = release_preview_cameras()
    if released:
        logger.info(f"Released {released} camera preview(s) before launching {command}")

    # Close stale obs reader — the new process will create fresh shared memory
    # segments; any existing reader is mapped to old (possibly unlinked) segments.
    _close_obs_reader()
    # The last run's reason belongs to the last run.
    _clear_last_failure()

    # A data-tab overlay may own the obs stream; a run must be its sole writer, so tear the data
    # publisher down before the subprocess's robot-connect recreates the segment (else
    # _Block(create=True) unlinks + recreates it under the publisher — see overlays.py).
    try:
        from lerobot.gui.api.overlays import stop_data_publisher

        stop_data_publisher()
    except Exception:
        pass

    env = {**__import__("os").environ, "LEROBOT_OBS_STREAM": "1"}
    # The GUI owns flow control for every subprocess it launches (via POST
    # /api/run/control -> subprocess stdin), so suppress the subprocesses' local
    # keyboard listeners: single source of truth, and avoids the X11 footgun of
    # a global pynput listener firing on keypresses meant for other windows.
    env["LEROBOT_KEYBOARD_LISTENER"] = "0"
    if extra_env:
        env.update(extra_env)
    cmd_str = " ".join(args)
    logger.info(f"Launching: {cmd_str}")
    _append_output(f"--- Starting {command} ---")
    _append_output(f"$ {cmd_str}\n")

    # Claim ownership before the process exists, not after. A reader on another
    # thread's loop that catches the in-between state must never see the new
    # process paired with a previous launch's loop — that pairing is what lets
    # a foreign server decide the process is its own. This order makes the
    # in-between state "previous process, this loop", which the check declines.
    _active_loop = asyncio.get_running_loop()
    _active_process = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        preexec_fn=_set_pdeathsig_preexec,
    )

    _stream_tasks = [
        asyncio.create_task(_read_stream(_active_process.stdout)),
        asyncio.create_task(_read_stream(_active_process.stderr, prefix="[stderr] ")),
    ]
    asyncio.create_task(_wait_for_exit())


async def _launch_debug_s2(config: DebugModelConfig) -> None:
    """Launch HVLA S2 standalone as a debug model process alongside teleop."""
    global _debug_process, _debug_output_path, _debug_output_lines, _debug_read_task
    import tempfile

    await _stop_debug_process()

    args = [
        sys.executable,
        "-u",
        "-m",
        "lerobot.policies.hvla.s2_standalone",
        f"--checkpoint={Path(config.checkpoint).expanduser() / 'model.safetensors'}"
        if not config.checkpoint.endswith(".safetensors")
        else f"--checkpoint={Path(config.checkpoint).expanduser()}",
        f"--task={config.task}",
    ]
    if config.decode_subtask:
        args.append("--decode-subtask")

    # Write output to a dedicated log file (not mixed with main process output).
    # mkstemp -> path-only: the fd is closed immediately, then the subprocess
    # opens the path with O_TRUNC. Avoids mktemp's TOCTOU race.
    _fd, _log_path = tempfile.mkstemp(prefix="lerobot_debug_model_", suffix=".log")
    os.close(_fd)
    _debug_output_path = Path(_log_path)
    _debug_output_lines = []

    env = {**__import__("os").environ}
    logger.info(f"Launching debug S2: {' '.join(args)}")
    logger.info(f"Debug model output: {_debug_output_path}")

    debug_log_fd = os.open(str(_debug_output_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    _debug_process = await asyncio.create_subprocess_exec(
        *args,
        stdout=debug_log_fd,
        stderr=debug_log_fd,
        env=env,
        preexec_fn=_set_pdeathsig_preexec,
    )
    os.close(debug_log_fd)  # subprocess inherited the fd, we can close our copy

    # Tail the log file in background
    _debug_read_task = asyncio.create_task(_tail_debug_log())


async def _tail_debug_log() -> None:
    """Tail the debug model log file, appending lines to _debug_output_lines."""
    global _debug_output_lines
    if _debug_output_path is None:
        return
    # Wait for file to exist
    for _ in range(50):
        if _debug_output_path.exists():
            break
        await asyncio.sleep(0.1)
    if not _debug_output_path.exists():
        return
    with open(_debug_output_path) as f:
        while True:
            line = f.readline()
            if line:
                line = line.rstrip("\n\r")
                if line:
                    _debug_output_lines.append(line)
                    if len(_debug_output_lines) > 1000:
                        _debug_output_lines = _debug_output_lines[-1000:]
                    _debug_output_event.set()
            else:
                # No new data — check if process exited
                if _debug_process is None or _debug_process.returncode is not None:
                    break
                await asyncio.sleep(0.1)


async def _stop_debug_process() -> None:
    """Stop the debug model process if running."""
    global _debug_process, _debug_read_task, _debug_output_path, _s2_subtask_cache
    _s2_subtask_cache = None
    if _debug_process is not None and _debug_process.returncode is None:
        _debug_process.terminate()
        try:
            await asyncio.wait_for(_debug_process.wait(), timeout=5.0)
        except _TIMEOUT_EXCS:
            _debug_process.kill()
            await _debug_process.wait()
    _debug_process = None
    if _debug_read_task is not None:
        _debug_read_task.cancel()
        _debug_read_task = None
    # Clean up log file
    if _debug_output_path is not None:
        with contextlib.suppress(Exception):
            # safe-destruct: our debug log temp file
            _debug_output_path.unlink(missing_ok=True)
        _debug_output_path = None


# ============================================================================
# Debug model management
# ============================================================================

_debug_lock = asyncio.Lock()  # prevent concurrent load/unload

# Serializes the check-then-launch sequence in start_teleoperate / start_record /
# start_replay / start_hvla. Without it, two requests arriving close together
# can both see `_active_process is None` (the synchronous check passes),
# both call `await asyncio.create_subprocess_exec(...)`, and the second
# overwrites `_active_process` before the first launch finishes — orphaning
# the first subprocess holding cameras and serial ports. Acquired around the
# whole `_ensure_no_active_process() + _launch_subprocess(...)` block so the
# check and the assign-to-`_active_process` happen atomically.
_launch_lock = asyncio.Lock()


async def _release_preview_cameras() -> None:
    """Release Robot-tab preview devices before a hardware subprocess starts."""
    from lerobot.gui.api.robot import _close_preview_cameras

    await asyncio.get_running_loop().run_in_executor(None, _close_preview_cameras)


def _is_debug_loaded() -> bool:
    return _debug_process is not None and _debug_process.returncode is None


@router.post("/debug/load")
async def load_debug_model(config: DebugModelConfig) -> dict:
    """Load a debug model process (keeps it warm for reuse across teleop sessions)."""
    async with _debug_lock:
        if _is_debug_loaded():
            raise HTTPException(409, "Debug model already loaded — unload first")

        if config.policy_type == "hvla_s2_vlm":
            await _launch_debug_s2(config)
        else:
            raise HTTPException(400, f"Unsupported debug model type: {config.policy_type}")

        return {"status": "loaded", "policy_type": config.policy_type, "pid": _debug_process.pid}


@router.post("/debug/unload")
async def unload_debug_model() -> dict:
    """Unload the debug model process (frees GPU memory)."""
    async with _debug_lock:
        if not _is_debug_loaded():
            return {"status": "not_loaded"}
        await _stop_debug_process()
        return {"status": "unloaded"}


@router.get("/debug/status")
async def debug_model_status() -> dict:
    """Check if a debug model is loaded."""
    return {"loaded": _is_debug_loaded(), "pid": _debug_process.pid if _is_debug_loaded() else None}


@router.get("/debug/output")
async def debug_output_sse():
    """SSE stream of debug model output lines."""
    sent = 0

    async def event_generator():
        nonlocal sent
        while True:
            _debug_output_event.clear()
            # Send any new lines
            while sent < len(_debug_output_lines):
                line = _debug_output_lines[sent]
                sent += 1
                yield f"data: {line}\n\n"
            # Wait for new data or timeout (keepalive)
            try:
                await asyncio.wait_for(_debug_output_event.wait(), timeout=5.0)
            except _TIMEOUT_EXCS:
                yield ": keepalive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# TODO: generalize — currently hardcoded for HVLA S2 subtask.
# When multiple debug model types exist, this should read from a generic
# model output schema rather than importing HVLA-specific IPC.
_s2_subtask_cache = None  # SharedLatentCache | None (read-only attach)


@router.get("/debug/subtask")
async def debug_subtask() -> dict:
    """Read the latest S2 subtask text directly from shared memory.

    Returns {"subtask": str, "age_ms": float} or {"subtask": ""} if unavailable.
    """
    global _s2_subtask_cache
    if not _is_debug_loaded():
        _s2_subtask_cache = None
        return {"subtask": ""}
    # Lazily attach to S2's shared memory
    if _s2_subtask_cache is None:
        try:
            from lerobot.policies.hvla.ipc import SharedLatentCache

            _s2_subtask_cache = SharedLatentCache(create=False)
        except FileNotFoundError:
            return {"subtask": ""}
    try:
        text, ts, confidence = _s2_subtask_cache.read_subtask()
        age_ms = (time.time() - ts) * 1000 if ts > 0 else 0.0
        return {"subtask": text, "age_ms": age_ms, "confidence": confidence}
    except Exception:
        _s2_subtask_cache = None
        return {"subtask": ""}


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/teleoperate")
async def start_teleoperate(req: TeleoperateRequest) -> dict:
    async with _launch_lock:
        _ensure_no_active_process()
        _refuse_if_deps_missing(req.robot, req.teleop)
        await _release_preview_cameras()

        # Use the interpreter running the GUI instead of relying on console
        # scripts being discoverable on PATH. This also preserves the GUI's
        # active virtual environment and editable source tree.
        args = [sys.executable, "-m", "lerobot.scripts.lerobot_teleoperate"]
        args.extend(_profile_to_cli_args(req.robot, "robot"))
        args.extend(_profile_to_cli_args(req.teleop, "teleop"))
        args.append(f"--fps={req.fps}")
        # Always-on latency monitoring for GUI sessions; the GUI polls
        # outputs/teleop/latency_snapshot.json for the live overlay.
        args.append("--latency_monitor=true")
        args.append(f"--latency_output_dir={LATENCY_OUTPUT_DIR}")

        # Ensure debug model is loaded if selected (lazy load, stays warm after teleop).
        # _debug_lock is nested under _launch_lock: this lock-ordering is safe
        # because _debug_lock holders (load_debug_model / unload_debug_model)
        # never reach for _launch_lock — no cycle is possible.
        extra_env = None
        if req.debug_model and req.debug_model.policy_type == "hvla_s2_vlm":
            async with _debug_lock:
                if _debug_process is None or _debug_process.returncode is not None:
                    await _launch_debug_s2(req.debug_model)
            extra_env = {"LEROBOT_S2_IMAGE_BUFFER": "1"}

        await _launch_subprocess(args, command="teleoperate", config=req.model_dump(), extra_env=extra_env)
        return {"status": "started", "command": "teleoperate", "pid": _active_process.pid}


@router.post("/record")
async def start_record(req: RecordRequest) -> dict:
    async with _launch_lock:
        _ensure_no_active_process()
        _refuse_if_deps_missing(req.robot, req.teleop)
        await _release_preview_cameras()

        if req.teleop is None and req.policy_path is None:
            raise HTTPException(400, "Either teleop or policy_path must be provided")

        args = [sys.executable, "-m", "lerobot.scripts.lerobot_record"]
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
        # Upstream moved the codec under the nested rgb_encoder config; the flat
        # --dataset.vcodec this used to emit is no longer a recognised argument.
        args.append(f"--dataset.rgb_encoder.vcodec={req.vcodec}")
        args.append(f"--play_sounds={'true' if req.play_sounds else 'false'}")
        if req.resume:
            args.append("--resume=true")
        if req.intervention_repo_id:
            args.append(f"--intervention_repo_id={req.intervention_repo_id}")
        # Always-on latency monitoring for GUI sessions. Record writes to its
        # own directory so the dashboard doesn't double-render when both
        # teleop and record source keys exist in LATENCY_SOURCES.
        # _ensure_no_active_process guarantees only one writer at a time.
        args.append("--latency_monitor=true")
        args.append(f"--latency_output_dir={LATENCY_OUTPUT_DIR_RECORD}")

        extra_env = None
        if req.debug_model and req.debug_model.policy_type == "hvla_s2_vlm":
            async with _debug_lock:
                if _debug_process is None or _debug_process.returncode is not None:
                    await _launch_debug_s2(req.debug_model)
            extra_env = {"LEROBOT_S2_IMAGE_BUFFER": "1"}

        await _launch_subprocess(args, command="record", config=req.model_dump(), extra_env=extra_env)
        return {"status": "started", "command": "record", "pid": _active_process.pid}


@router.post("/flash/inspect")
async def inspect_flash_skill(req: FlashInspectRequest) -> dict:
    """Inspect a local FLASH sidecar without loading it on the GUI event loop."""
    return await _inspect_flash_subprocess(req.skill_path)


@router.post("/flash/dataset-inspect")
async def inspect_flash_dataset(req: FlashDatasetInspectRequest) -> dict:
    """Discover FLASH data beside an ordinary local LeRobot episode."""
    return await _inspect_flash_subprocess("--dataset-root", req.dataset_root, "--episode", str(req.episode))


@router.post("/flash/config-inspect")
async def inspect_flash_config(req: FlashConfigInspectRequest) -> dict:
    """Show the exact trial parameters without connecting to a robot."""
    from lerobot.policies.flash.causal_service import CausalTrackingConfig
    from lerobot.policies.flash.runtime import AttachmentRuntimeLimits

    def read() -> dict:
        path = Path(req.config_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError("Select an existing local robot/camera trial config")
        settings = json.loads(path.read_text(encoding="utf-8"))
        AttachmentRuntimeLimits.from_config(settings["attachment"])
        if settings.get("causal_tracking") is not None:
            CausalTrackingConfig.from_config(settings["causal_tracking"])
        return {"path": str(path), "settings": settings}

    try:
        return await asyncio.get_running_loop().run_in_executor(_flash_config_pool, read)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise HTTPException(400, f"Invalid FLASH trial config: {exc}") from exc


@router.post("/flash/run")
async def start_flash(req: FlashRunRequest) -> dict:
    """Launch a prepared spatial skill through the shared Run process owner."""
    async with _launch_lock:
        _ensure_no_active_process()
        inspection = await _inspect_flash_subprocess(
            "--dataset-root", req.dataset_root, "--episode", str(req.episode)
        )
        bundle = Path(req.bundle_path).expanduser().resolve()
        if not any(
            item["kind"] == "prepared"
            and Path(item["path"]).resolve() == bundle
            and item["inspection"]["runtime_bindings_complete"]
            for item in inspection["skills"]
        ):
            raise HTTPException(400, "Select a complete prepared FLASH skill from this dataset episode")
        config = Path(req.config_path).expanduser().resolve()
        if not config.is_file():
            raise HTTPException(400, "Select an existing local robot/camera trial config")
        await _release_preview_cameras()
        global _flash_live_path
        _flash_live_path = Path(tempfile.gettempdir()) / f"lerobot-flash-{secrets.token_hex(16)}.json"
        args = [
            os.environ.get("LEROBOT_FLASH_TRACKING_PYTHON", sys.executable),
            "-u",
            "-m",
            "lerobot.policies.flash.live_trial",
            str(bundle),
            str(config),
            "--auto-output",
            "--source-dataset-root",
            str(Path(req.dataset_root).expanduser().resolve()),
            "--telemetry-path",
            str(_flash_live_path),
        ]
        await _launch_subprocess(args, command="flash", config=req.model_dump())
        return {"status": "started", "command": "flash", "pid": _active_process.pid}


@router.get("/flash/telemetry/live")
async def flash_live_telemetry() -> dict:
    """Return the current trial's latest tracker layers without touching its loops."""
    path = _flash_live_path if _active_command == "flash" and is_run_active() else None
    if path is None:
        return {"available": False}

    def read() -> dict:
        try:
            return {"available": True, "snapshot": json.loads(path.read_text(encoding="utf-8"))}
        except FileNotFoundError:
            return {"available": False}

    return await asyncio.get_running_loop().run_in_executor(_flash_telemetry_pool, read)


@router.get("/flash/telemetry/frame/{frame_id}/{kind}")
async def flash_live_frame(frame_id: int, kind: str) -> Response:
    """Read the display copy of an exact live RGB-D frame, outside the GUI loop."""
    path = _flash_live_path if _active_command == "flash" and is_run_active() else None
    if path is None:
        raise HTTPException(404, "No FLASH trial is running")
    from lerobot.policies.flash.live_telemetry import live_frame_path

    try:
        frame_path = live_frame_path(path, frame_id, kind)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    def read() -> bytes:
        return frame_path.read_bytes()

    try:
        data = await asyncio.get_running_loop().run_in_executor(_flash_telemetry_pool, read)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Live display frame has expired") from exc
    return Response(data, media_type="image/jpeg" if kind == "rgb" else "application/octet-stream")


def _flash_tracker_replay(root: Path, episode: int) -> dict:
    info_path = root.expanduser().resolve() / "meta" / "info.json"
    camera_features = None
    if info_path.is_file():
        features = json.loads(info_path.read_text(encoding="utf-8")).get("features", {})
        for depth_key, feature in features.items():
            if not feature.get("info", {}).get("is_depth_map") or not depth_key.endswith("_depth"):
                continue
            rgb_key = depth_key.removesuffix("_depth")
            if rgb_key in features:
                camera_features = {"rgb": rgb_key, "depth": depth_key}
                break
    extension = root.expanduser().resolve() / "flash" / "v1"
    trace_path = extension / f"episode_{episode:06d}.jsonl"
    if not trace_path.is_file():
        raise HTTPException(404, "No FLASH trace is stored for this episode")
    ticks = [json.loads(line) for line in trace_path.open(encoding="utf-8") if line.strip()]
    if not ticks:
        raise HTTPException(404, "FLASH trace has no frames")
    camera_ticks = [tick for tick in ticks if tick.get("camera_source_time_s") is not None]
    times = [tick["camera_source_time_s"] for tick in camera_ticks]
    ticks_by_camera_id: dict[int, dict] = {}
    for tick in camera_ticks:
        if tick.get("camera_frame_id") is not None:
            ticks_by_camera_id.setdefault(tick["camera_frame_id"], tick)
    layers_path = extension / f"tracker_episode_{episode:06d}.jsonl"
    snapshots = (
        [json.loads(line) for line in layers_path.open(encoding="utf-8") if line.strip()]
        if layers_path.is_file()
        else []
    )
    if snapshots:
        frames = []
        for snapshot in snapshots:
            camera = snapshot.get("camera")
            source_s = camera.get("source_time_s") if camera else None
            if camera is None and ticks_by_camera_id:
                continue
            if camera and ticks_by_camera_id:
                tick = ticks_by_camera_id.get(camera.get("frame_id"))
                if tick is None:
                    # The camera ran before or between recorded motor ticks.
                    # No episode image exists for this exact tracker snapshot.
                    continue
                frame_index = tick["frame_index"]
                alignment = "camera_frame_id"
            elif source_s is not None and camera_ticks:
                pos = bisect_left(times, source_s)
                pos = min(pos, len(camera_ticks) - 1)
                if pos and abs(times[pos - 1] - source_s) < abs(times[pos] - source_s):
                    pos -= 1
                frame_index = camera_ticks[pos]["frame_index"]
                alignment = "nearest_camera_time"
            else:
                frame_index = 0
                alignment = "none"
            frames.append({"frame_index": frame_index, "alignment": alignment, "snapshot": snapshot})
        if not frames:
            raise HTTPException(404, "No tracker snapshot has a recorded camera frame")
        return {
            "available": True,
            "kind": "rollout",
            "layer_history": True,
            "camera_features": camera_features,
            "frames": frames,
        }
    # Older trials recorded the accepted planner pose at motor ticks, but not
    # every rejected/direct/group layer. Show only what those traces establish.
    return {
        "available": True,
        "kind": "rollout",
        "layer_history": False,
        "camera_features": camera_features,
        "frames": [
            {
                "frame_index": tick["frame_index"],
                "snapshot": {
                    "recorded_at_s": tick.get("observed_at_s") or tick.get("scheduled_s"),
                    "camera": None,
                    "centers_object_m": {},
                    "proposals": {},
                    "layers": {},
                    "accepted": tick.get("poses", {}),
                },
            }
            for tick in ticks
        ],
    }


@router.get("/flash/telemetry/replay")
async def flash_tracker_replay(dataset_root: str, episode: int = 0) -> dict:
    if episode < 0:
        raise HTTPException(400, "Episode must be nonnegative")
    return await asyncio.get_running_loop().run_in_executor(
        _flash_telemetry_pool, _flash_tracker_replay, Path(dataset_root), episode
    )


@router.get("/flash/demo/replay")
async def flash_demo_replay(
    dataset_root: str, bundle_path: str, episode: int = 0, candidate_id: str | None = None
) -> dict:
    """Play the original camera frames with the exact prepared spatial samples."""
    from lerobot.policies.flash.demo_replay import load_legacy_demo_replay
    from lerobot.policies.flash.demo_tracker_candidates import load_candidate_set

    if episode < 0:
        raise HTTPException(400, "Episode must be nonnegative")
    try:

        def read() -> dict:
            result = load_legacy_demo_replay(Path(dataset_root), episode, Path(bundle_path))
            if candidate_id is not None:
                manifest, poses_by_role = load_candidate_set(
                    Path(dataset_root), Path(bundle_path), candidate_id
                )
                for frame in result["frames"]:
                    for role, poses in poses_by_role.items():
                        pose = poses.get(frame["frame_index"])
                        if pose is not None:
                            snapshot = frame["snapshot"]
                            snapshot.setdefault("layers", {}).setdefault(role, {})["candidate_tracker"] = {
                                "pose": pose.tolist(),
                                "evidence": "candidate",
                                "estimator": manifest["tracker"],
                                "source_time_s": snapshot["recorded_at_s"],
                            }
                result["tracker_candidate"] = {"candidate_id": candidate_id, **manifest}
            return result

        return await asyncio.get_running_loop().run_in_executor(_flash_candidate_pool, read)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"Cannot replay FLASH demo: {exc}") from exc


@router.get("/flash/demo/track-points")
async def flash_demo_track_points(dataset_root: str, bundle_path: str, candidate_id: str) -> dict:
    """Read the candidate's raw model tracks for replay; no fitting runs here."""
    from lerobot.policies.flash.demo_tracker_candidates import load_candidate_point_tracks

    try:
        return await asyncio.get_running_loop().run_in_executor(
            _flash_candidate_pool,
            load_candidate_point_tracks,
            Path(dataset_root),
            Path(bundle_path),
            candidate_id,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"Cannot read FLASH point tracks: {exc}") from exc


@router.get("/flash/demo/trackers")
async def flash_demo_trackers(dataset_root: str, bundle_path: str) -> dict:
    """List supported trace identities and versioned proposals for this demo."""
    from lerobot.policies.flash.demo_tracker_candidates import list_candidates, runner_availability

    try:
        trackers, candidates = await asyncio.get_running_loop().run_in_executor(
            _flash_candidate_pool,
            lambda: (runner_availability(), list_candidates(Path(dataset_root), Path(bundle_path))),
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"Cannot list demo trackers: {exc}") from exc
    return {
        "trackers": trackers,
        "candidates": candidates,
    }


@router.post("/flash/demo/trackers/process")
async def flash_demo_tracker_process(req: FlashTrackerProcessRequest) -> dict:
    """Run offline inference without occupying the GUI event loop or robot owner."""
    from lerobot.policies.flash.demo_tracker_candidates import process_demo

    candidate_id = f"{req.tracker}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
    try:
        return await asyncio.get_running_loop().run_in_executor(
            _flash_tracker_pool,
            lambda: process_demo(
                Path(req.dataset_root),
                Path(req.bundle_path),
                tracker=req.tracker,
                role=req.role,
                candidate_id=candidate_id,
                expected_episode=req.episode,
                max_points_per_object=req.max_points_per_object,
                new_points_per_view=req.new_points_per_view,
                surface_tolerance_m=req.surface_tolerance_m,
            ),
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"Cannot process FLASH demo: {exc}") from exc


@router.post("/flash/demo/trackers/import")
async def flash_demo_tracker_import(req: FlashTrackerImportRequest) -> dict:
    """Validate and align a saved tracker output as a review-only proposal."""
    from lerobot.policies.flash.demo_tracker_candidates import import_relative_trace

    try:
        return await asyncio.get_running_loop().run_in_executor(
            _flash_candidate_pool,
            lambda: import_relative_trace(
                Path(req.dataset_root),
                Path(req.bundle_path),
                tracker=req.tracker,
                role=req.role,
                source_path=Path(req.source_path),
                candidate_id=f"{req.tracker}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}",
                expected_episode=req.episode,
            ),
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"Cannot import tracker output: {exc}") from exc


@router.post("/flash/demo/trackers/apply")
async def flash_demo_tracker_apply(req: FlashTrackerApplyRequest) -> dict:
    """Create a new active prepared skill from one reviewed pose proposal."""
    from lerobot.policies.flash.demo_tracker_candidates import apply_candidate

    try:
        name = req.output_name or f"flash-tracker-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
        bundle, report = await asyncio.get_running_loop().run_in_executor(
            _flash_candidate_pool,
            lambda: apply_candidate(
                Path(req.dataset_root),
                Path(req.bundle_path),
                req.candidate_id,
                output_name=name,
                expected_episode=req.episode,
            ),
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"Cannot apply tracker output: {exc}") from exc
    return {"bundle_path": str(bundle), "report": report}


@router.get("/flash/demo/raw-replay")
async def flash_raw_demo_replay(dataset_root: str, episode: int = 0) -> dict:
    """Review a raw RGB-D demo before any object roles or skill are prepared."""
    from lerobot.policies.flash.demo_replay import load_raw_legacy_demo_replay

    try:
        return await asyncio.get_running_loop().run_in_executor(
            _flash_telemetry_pool, load_raw_legacy_demo_replay, Path(dataset_root), episode
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"Cannot replay raw FLASH demo: {exc}") from exc


@router.get("/flash/demo/frame")
def flash_demo_frame(dataset_root: str, frame_index: int) -> FileResponse:
    """Serve the same numbered JPEG frame the preparation adapter consumed."""
    root = Path(dataset_root).expanduser().resolve()
    if frame_index < 0 or not (root / "meta" / "info.json").is_file():
        raise HTTPException(400, "Select a valid local demo dataset and frame")
    path = root / "recording" / "rgb" / f"{frame_index:06d}.jpg"
    if not path.is_file():
        raise HTTPException(404, "Demo camera frame is missing")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/flash/demo/depth-frame")
async def flash_demo_depth_frame(dataset_root: str, frame_index: int) -> Response:
    """Aligned metric depth for the 3D inspector; read outside the GUI loop."""
    if frame_index < 0:
        raise HTTPException(400, "Frame index must be nonnegative")

    def read() -> bytes:
        import cv2
        import numpy as np

        root = Path(dataset_root).expanduser().resolve()
        if not (root / "meta" / "info.json").is_file():
            raise FileNotFoundError("Demo dataset is missing")
        depth = cv2.imread(str(root / "recording" / "depth" / f"{frame_index:06d}.png"), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.ndim != 2 or depth.dtype != np.uint16:
            raise FileNotFoundError("Aligned 16-bit demo depth frame is missing")
        return depth.astype("<u2", copy=False).tobytes()

    try:
        data = await asyncio.get_running_loop().run_in_executor(_flash_telemetry_pool, read)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return Response(data, media_type="application/octet-stream")


@router.get("/flash/demo/reference-surfaces")
async def flash_demo_reference_surfaces(dataset_root: str, bundle_path: str) -> dict:
    """Build only measured reference patches; no inferred back surface."""
    from lerobot.policies.flash.demo_scene import reference_surfaces

    try:

        def read() -> dict:
            root, bundle = _flash_demo_paths(dataset_root, bundle_path)
            return reference_surfaces(root, bundle)

        return await asyncio.get_running_loop().run_in_executor(_flash_config_pool, read)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"Cannot build demo surfaces: {exc}") from exc


@router.get("/flash/demo/scene-evidence/status")
async def flash_demo_scene_evidence_status(dataset_root: str, bundle_path: str) -> dict:
    from lerobot.policies.flash.demo_scene import scene_evidence_status

    try:
        _, bundle = _flash_demo_paths(dataset_root, bundle_path)
        return await asyncio.get_running_loop().run_in_executor(
            _flash_config_pool, scene_evidence_status, bundle
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"Cannot read demo scene evidence: {exc}") from exc


@router.get("/flash/demo/scene-evidence/frame")
async def flash_demo_scene_evidence_frame(dataset_root: str, bundle_path: str, frame_index: int) -> dict:
    from lerobot.policies.flash.demo_scene import scene_evidence_frame

    try:
        _, bundle = _flash_demo_paths(dataset_root, bundle_path)
        return await asyncio.get_running_loop().run_in_executor(
            _flash_config_pool, scene_evidence_frame, bundle, frame_index
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"Cannot read demo scene evidence: {exc}") from exc


def _flash_demo_paths(dataset_root: str, bundle_path: str) -> tuple[Path, Path]:
    root = Path(dataset_root).expanduser().resolve()
    bundle = Path(bundle_path).expanduser().resolve()
    prepared_root = root / "flash" / "v1" / "prepared"
    try:
        relative = bundle.relative_to(prepared_root)
    except ValueError as exc:
        raise ValueError("selected skill must be stored with this demo dataset") from exc
    if (
        relative.parts != (bundle.parent.name, "bundle.json")
        or not bundle.is_file()
        or not (root / "meta" / "info.json").is_file()
    ):
        raise ValueError("select a prepared FLASH skill beside a local demo")
    return root, bundle


def _flash_demo_preparation(dataset_root: str, bundle_path: str, episode: int) -> dict:
    root, bundle = _flash_demo_paths(dataset_root, bundle_path)
    manifest = json.loads(bundle.read_text(encoding="utf-8"))
    skill = json.loads((bundle.parent / manifest["skill"]).read_text(encoding="utf-8"))
    if skill["source_episode"] != episode:
        raise ValueError("prepared skill belongs to a different demo episode")
    annotation_path = bundle.parent / "annotation.json"
    if not annotation_path.is_file():
        annotation_path = root / "flash" / "v1" / "import-annotations.json"
    keypoints_path = bundle.parent / "keypoints.json"
    if not keypoints_path.is_file():
        keypoints_path = root / "keypoints.json"
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    keypoints = json.loads(keypoints_path.read_text(encoding="utf-8"))
    if annotation["episode"] != episode or not isinstance(keypoints["keypoints"], list):
        raise ValueError("preparation inputs do not match the selected demo episode")
    path_migrated = False
    arm_path = root / "showservo_demo.npz"
    if skill.get("intervals") and arm_path.is_file():
        import numpy as np

        from lerobot.policies.flash.sealed_demo_import import explicit_paths_from_prepared_skill

        with np.load(arm_path, allow_pickle=False) as arm:
            keypoints, path_migrated = explicit_paths_from_prepared_skill(
                keypoints, skill, np.asarray(arm["t"], dtype=float)
            )
    report_path = bundle.parent / "preparation-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
    return {
        "annotation": annotation,
        "keypoints_document": keypoints,
        "path_selection_migrated": path_migrated,
        "preparation_report": report,
        "annotation_source": str(annotation_path),
        "keypoints_source": str(keypoints_path),
        "reference_masks": {
            reference["role"]: {
                "frame_index": reference["snippets"][0]["frame_index"],
                "available": (root / reference["snippets"][0]["mask_artifact"]).is_file(),
            }
            for reference in manifest["references"]
            if reference.get("snippets")
        },
    }


@router.get("/flash/demo/preparation")
async def flash_demo_preparation(dataset_root: str, bundle_path: str, episode: int = 0) -> dict:
    """Read the selected demo's editable recipe without loading GPU models."""
    try:
        return await asyncio.get_running_loop().run_in_executor(
            _flash_config_pool, _flash_demo_preparation, dataset_root, bundle_path, episode
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"Cannot load FLASH demo preparation: {exc}") from exc


@router.get("/flash/demo/reference-mask")
def flash_demo_reference_mask(dataset_root: str, bundle_path: str, role: str) -> FileResponse:
    """Serve a prepared reference mask from its dataset, not an arbitrary path."""
    try:
        root, bundle = _flash_demo_paths(dataset_root, bundle_path)
        manifest = json.loads(bundle.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(400, f"Cannot read prepared reference mask: {exc}") from exc
    reference = next((item for item in manifest["references"] if item["role"] == role), None)
    if reference is None or not reference["snippets"]:
        raise HTTPException(404, "Selected role has no prepared reference mask")
    mask = (root / reference["snippets"][0]["mask_artifact"]).resolve()
    try:
        mask.relative_to((root / "flash" / "v1" / "masks").resolve())
    except ValueError as exc:
        raise HTTPException(400, "Reference mask is outside this dataset") from exc
    if not mask.is_file():
        raise HTTPException(404, "Prepared reference mask is missing")
    return FileResponse(mask, media_type="image/png")


@router.post("/flash/demo/prepare")
async def prepare_flash_demo(req: FlashDemoPrepareRequest) -> dict:
    """Version the edited inputs and launch offline preparation asynchronously."""
    async with _launch_lock:
        _ensure_no_active_process()

        def save_inputs() -> tuple[Path, Path, Path, str]:
            if req.bundle_path is not None:
                root, _ = _flash_demo_paths(req.dataset_root, req.bundle_path)
                current = _flash_demo_preparation(req.dataset_root, req.bundle_path, req.episode)
                if set(req.annotation.get("prompts_by_role", {})) != set(
                    current["annotation"]["prompts_by_role"]
                ):
                    raise ValueError("this importer cannot change the prepared object roles")
            else:
                root = Path(req.dataset_root).expanduser().resolve()
                if (
                    req.episode != 0
                    or not (root / "showservo_demo.npz").is_file()
                    or not (root / "meta" / "info.json").is_file()
                ):
                    raise ValueError("select episode 0 of a local raw RGB-D demo")
                roles = req.annotation.get("prompts_by_role", {})
                held, target = req.annotation.get("held_role"), req.annotation.get("target_role")
                if (
                    not isinstance(roles, dict)
                    or set(roles) != {held, target}
                    or held == target
                    or any(
                        not isinstance(role, str)
                        or re.fullmatch(r"[A-Za-z][A-Za-z0-9 _-]{0,39}", role) is None
                        or not isinstance(prompt, str)
                        or not prompt.strip()
                        for role, prompt in roles.items()
                    )
                ):
                    raise ValueError("name two distinct objects and give each a segmentation prompt")
            if req.annotation.get("episode") != req.episode:
                raise ValueError("annotation episode differs from the selected demo")
            keypoints = req.keypoints_document.get("keypoints")
            if not isinstance(keypoints, list) or not keypoints:
                raise ValueError("the edited demo needs at least one keypoint")
            name = (
                req.output_name or f"flash-prepared-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
            )
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name) is None:
                raise ValueError(
                    "prepared version name must use letters, digits, dots, dashes or underscores"
                )
            if (root / "flash" / "v1" / "prepared" / name).exists():
                raise ValueError("that prepared version already exists")
            input_dir = root / "flash" / "v1" / "preparation-inputs" / name
            input_dir.mkdir(parents=True, exist_ok=False)
            annotation_path = input_dir / "annotation.json"
            keypoints_path = input_dir / "keypoints.json"
            annotation_path.write_text(json.dumps(req.annotation, indent=2) + "\n", encoding="utf-8")
            keypoints_path.write_text(json.dumps(req.keypoints_document, indent=2) + "\n", encoding="utf-8")
            return root, annotation_path, keypoints_path, name

        try:
            root, annotation_path, keypoints_path, name = await asyncio.get_running_loop().run_in_executor(
                _flash_config_pool, save_inputs
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, f"Cannot prepare FLASH demo: {exc}") from exc
        importer = Path(__file__).resolve().parents[4] / "benchmarks" / "flash_prepare_legacy.py"
        args = [
            os.environ.get("LEROBOT_FLASH_TRACKING_PYTHON", sys.executable),
            "-u",
            str(importer),
            str(root),
            str(annotation_path),
            "--keypoints",
            str(keypoints_path),
            "--output-name",
            name,
        ]
        await _launch_subprocess(args, command="flash_prepare", config=req.model_dump())
        return {
            "status": "started",
            "command": "flash_prepare",
            "pid": _active_process.pid,
            "bundle_path": str(root / "flash" / "v1" / "prepared" / name / "bundle.json"),
            "output_name": name,
        }


def _flash_recordings(source_paths: tuple[Path, ...]) -> list[dict]:
    """Find demo episodes and rollouts under the same bounded source folders as Data."""
    extensions: set[Path] = set()
    for source in source_paths:
        root = source.expanduser()
        for depth in range(4):
            pattern = "/".join(["*"] * depth + ["flash", "v1"])
            extensions.update(path for path in root.glob(pattern) if path.is_dir())
            raw_pattern = "/".join(["*"] * depth + ["meta", "info.json"])
            for info_path in root.glob(raw_pattern):
                dataset_root = info_path.parent.parent
                if (dataset_root / "showservo_demo.npz").is_file() and (
                    dataset_root / "recording" / "times.txt"
                ).is_file():
                    extensions.add(dataset_root / "flash" / "v1")
    found = []
    for extension in extensions:
        dataset_root = extension.parent.parent
        info_path = dataset_root / "meta" / "info.json"
        if not info_path.is_file():
            continue
        try:
            episode_count = json.loads(info_path.read_text(encoding="utf-8"))["total_episodes"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not isinstance(episode_count, int) or episode_count < 0:
            continue
        demos: dict[int, dict] = {}
        raw_arm = dataset_root / "showservo_demo.npz"
        if episode_count > 0 and raw_arm.is_file() and (dataset_root / "recording" / "times.txt").is_file():
            demos[0] = {
                "kind": "demo",
                "dataset_root": str(dataset_root.resolve()),
                "episode": 0,
                "modified_at_s": raw_arm.stat().st_mtime,
                "skills": [],
                "prepared_versions": [],
                "bundle_path": None,
                "prepared_skill": None,
            }
        skill_paths = list((extension / "prepared").glob("*/bundle.json"))
        skill_paths += list((extension / "skills").glob("*.json"))
        skill_paths.sort(key=lambda path: path.stat().st_mtime)
        for path in skill_paths:
            try:
                if path.stat().st_size > 16 * 1024 * 1024:
                    continue
                metadata = json.loads(path.read_text(encoding="utf-8"))
                if path.name == "bundle.json":
                    path_to_skill = path.parent / metadata["skill"]
                    if path_to_skill.stat().st_size > 16 * 1024 * 1024:
                        continue
                    metadata = json.loads(path_to_skill.read_text(encoding="utf-8"))
                episode = metadata["source_episode"]
                if not isinstance(episode, int) or not 0 <= episode < episode_count:
                    continue
                modified_at_s = path.stat().st_mtime
            except (OSError, ValueError, KeyError, TypeError):
                continue
            demo = demos.setdefault(
                episode,
                {
                    "kind": "demo",
                    "dataset_root": str(dataset_root.resolve()),
                    "episode": episode,
                    "modified_at_s": 0,
                    "skills": [],
                    "prepared_versions": [],
                    "bundle_path": None,
                    "prepared_skill": None,
                },
            )
            demo["modified_at_s"] = max(demo["modified_at_s"], modified_at_s)
            demo["skills"].append(path.parent.name if path.name == "bundle.json" else path.stem)
            if path.name == "bundle.json":
                report = path.parent / "preparation-report.json"
                try:
                    candidate = (
                        json.loads(report.read_text(encoding="utf-8")).get("review_status") == "candidate"
                    )
                except (OSError, ValueError, TypeError):
                    candidate = False
                demo["prepared_versions"].append(
                    {"name": path.parent.name, "bundle_path": str(path.resolve()), "candidate": candidate}
                )
                if not candidate:
                    demo["bundle_path"] = str(path.resolve())
                    demo["prepared_skill"] = path.parent.name
        for demo in demos.values():
            if demo["bundle_path"] is None and demo["prepared_versions"]:
                chosen = demo["prepared_versions"][-1]
                demo["bundle_path"] = chosen["bundle_path"]
                demo["prepared_skill"] = chosen["name"]
        found.extend(demos.values())
        manifest_path = extension / "manifest.json"
        try:
            episodes = json.loads(manifest_path.read_text(encoding="utf-8")).get("episodes", {})
        except (OSError, ValueError):
            episodes = {}
        for trace in extension.glob("episode_*.jsonl"):
            try:
                episode = int(trace.stem.removeprefix("episode_"))
            except ValueError:
                continue
            detail = episodes.get(str(episode), {})
            found.append(
                {
                    "kind": "rollout",
                    "dataset_root": str(dataset_root.resolve()),
                    "episode": episode,
                    "modified_at_s": trace.stat().st_mtime,
                    "frames": detail.get("frames"),
                    "outcome": detail.get("outcome", "recording"),
                    "reason": detail.get("reason", ""),
                }
            )
    return sorted(found, key=lambda item: item["modified_at_s"], reverse=True)


@router.get("/flash/recordings")
async def flash_recordings() -> list[dict]:
    from lerobot.gui.api.datasets import _read_sources

    sources = tuple(Path(item["path"]) for item in _read_sources())
    return await asyncio.get_running_loop().run_in_executor(_flash_telemetry_pool, _flash_recordings, sources)


async def _inspect_flash_subprocess(*args: str) -> dict:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "lerobot.policies.flash.inspect",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(process.communicate(), timeout=20)
    except _TIMEOUT_EXCS as exc:
        process.kill()
        await process.wait()
        raise HTTPException(408, "FLASH skill inspection timed out") from exc
    if process.returncode:
        message = err.decode(errors="replace").strip().splitlines()
        raise HTTPException(400, message[-1] if message else "Invalid FLASH skill")
    return json.loads(out)


@router.post("/replay")
async def start_replay(req: ReplayRequest) -> dict:
    async with _launch_lock:
        _ensure_no_active_process()
        _refuse_if_deps_missing(req.robot)
        await _release_preview_cameras()

        # Note: --dataset.fps is intentionally omitted — `lerobot-replay` declares
        # it as config but ignores it (the loop paces by `dataset.fps` directly),
        # so passing it from the GUI was dead wiring.
        args = [sys.executable, "-m", "lerobot.scripts.lerobot_replay"]
        args.extend(_profile_to_cli_args(req.robot, "robot"))
        args.append(f"--dataset.repo_id={req.repo_id}")
        if req.root:
            args.append(f"--dataset.root={req.root}")
        args.append(f"--dataset.episode={req.episode}")

        await _launch_subprocess(args, command="replay", config=req.model_dump())
        return {"status": "started", "command": "replay", "pid": _active_process.pid}


@router.post("/hvla")
async def start_hvla(req: HVLARunRequest) -> dict:
    """Launch HVLA dual-system inference (S1 + S2)."""
    import tempfile

    async with _launch_lock:
        _ensure_no_active_process()
        await _release_preview_cameras()

        # Write robot profile to temp file (HVLA launch reads robot config from file)
        robot_config = dict(req.robot)
        if "fields" in robot_config:
            flat = {"type": robot_config["type"]}
            flat.update(robot_config["fields"])
            if "cameras" in robot_config:
                flat["cameras"] = robot_config["cameras"]
            robot_config = flat

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, prefix="hvla_robot_") as tmp:
            tmp.write(json.dumps(robot_config, indent=2))
            tmp_name = tmp.name

        args = [
            sys.executable,
            "-m",
            "lerobot.policies.hvla.launch",
            f"--s1-checkpoint={req.s1_checkpoint}",
            f"--task={req.task}",
            f"--robot-config={tmp_name}",
            f"--fps={req.fps}",
            f"--s1-type={req.s1_type}",
        ]
        if req.s1_query_interval is not None:
            args.append(f"--s1-query-interval={req.s1_query_interval}")
        if req.denoise_steps is not None:
            args.append(f"--denoise-steps={req.denoise_steps}")
        if req.s2_checkpoint:
            args.append(f"--s2-checkpoint={Path(req.s2_checkpoint).expanduser()}")
        else:
            args.append("--zero-s2")
        if req.decode_subtask:
            args.append("--decode-subtask")
        if req.record_dataset:
            args.append(f"--record-dataset={req.record_dataset}")
            args.append(f"--num-episodes={req.num_episodes}")
            args.append(f"--episode-time-s={req.episode_time_s}")
            args.append(f"--reset-time-s={req.reset_time_s}")

        # Teleop for intervention / inverse follow
        if req.teleop:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="hvla_teleop_"
            ) as teleop_tmp:
                teleop_tmp.write(json.dumps(req.teleop, indent=2))
                teleop_tmp_name = teleop_tmp.name
            args.append(f"--teleop-config={teleop_tmp_name}")
        if req.intervention_dataset:
            args.append(f"--intervention-dataset={req.intervention_dataset}")

        # RLT
        if req.rlt_mode:
            # Refuse to launch without an explicit RL Token Encoder. Every other
            # path tried — silent warning + actor.pt load → state_dict size
            # mismatch crash deep inside model construction. Fail fast at the
            # request boundary instead.
            if not req.rlt_token_checkpoint or not str(req.rlt_token_checkpoint).strip():
                raise HTTPException(
                    400,
                    "rlt_token_checkpoint is required when rlt_mode is True. "
                    "Set the 'RL Token Encoder' field to a trained Phase-1 "
                    "encoder dir (e.g. outputs/rlt_token_v4_4layer_d2048/"
                    "checkpoint-10000). The actor's input dim depends on it.",
                )
            args.append("--rlt-mode")
            args.append(f"--rl-token-checkpoint={Path(req.rlt_token_checkpoint).expanduser()}")
            if req.rlt_checkpoint:
                args.append(f"--rlt-checkpoint={Path(req.rlt_checkpoint).expanduser()}")
            if req.rlt_deploy:
                args.append("--rlt-deploy")
            args.append(f"--rl-chunk-length={req.rlt_chunk_length}")
            args.append(f"--rlt-output-dir={req.rlt_output_dir}")
            if not req.rlt_start_engaged:
                args.append("--rlt-start-disengaged")
            if req.rlt_shared_noise_per_chunk:
                args.append("--rlt-shared-noise-per-chunk")
            # RLT needs multi-episode mode
            if not req.record_dataset:
                args.append(f"--num-episodes={req.num_episodes}")
                args.append(f"--episode-time-s={req.episode_time_s}")
                args.append(f"--reset-time-s={req.reset_time_s}")

        # Always enable latency monitoring when launched from the GUI so the
        # dashboard's hvla tracks have data without requiring an extra
        # checkbox. s1_process appends /main and /inference to this parent
        # path; the resulting subdirs match LATENCY_SOURCES.
        args.append("--latency-monitor")
        args.append("--latency-output-dir=outputs/hvla_runs")

        # Action send shape (default "chunk"). Only emit the flag when the
        # value differs from the default to keep the CLI tidy in the GUI's
        # subprocess log.
        if req.send_action_shape != "chunk":
            args.append(f"--send-action-shape={req.send_action_shape}")

        await _launch_subprocess(args, command="hvla", config=req.model_dump())
        return {"status": "started", "command": "hvla", "pid": _active_process.pid}


@router.get("/rlt-metrics")
async def get_rlt_metrics() -> dict:
    """Return current RLT training metrics from file (cross-process)."""
    from lerobot.gui.api._run_core import get_rlt_metrics as _impl

    return _impl()


@router.get("/latency-metrics")
async def get_latency_metrics(source: str = "teleop") -> dict:
    """Return the latest latency snapshot for the requested source."""
    from lerobot.gui.api._run_core import get_latency_metrics as _impl

    return _impl(source)


@router.get("/latency-sources")
async def list_latency_sources() -> dict:
    """List which latency snapshot sources are available and recent.

    A source is considered "fresh" when its snapshot file was written in the
    last 5 seconds (the writer publishes at ~1 Hz; 5s tolerates jitter and
    pause/resume). Stale snapshots are reported with ``fresh=False`` so the
    dashboard can grey them out instead of dropping them entirely (useful
    for post-mortem viewing of the last run).

    Returns: ``{"sources": [{"key": str, "loop_kind": str, "fresh": bool,
    "age_s": float | None}, ...]}``.
    """
    out: list[dict] = []
    now = time.time()
    for key, dir_path in LATENCY_SOURCES.items():
        snap_path = Path(dir_path) / "latency_snapshot.json"
        if not snap_path.exists():
            out.append({"key": key, "loop_kind": None, "fresh": False, "age_s": None})
            continue
        try:
            mtime = snap_path.stat().st_mtime
            age_s = max(0.0, now - mtime)
            with open(snap_path, encoding="utf-8") as f:
                snap = json.load(f)
            out.append(
                {
                    "key": key,
                    "loop_kind": snap.get("loop_kind"),
                    "fresh": age_s <= 5.0,
                    "age_s": age_s,
                }
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("latency-sources read failed (key=%s): %s", key, e)
            out.append({"key": key, "loop_kind": None, "fresh": False, "age_s": None})
    return {"sources": out}


@router.get("/rlt-config")
async def get_rlt_config() -> dict:
    """Return current RLT override values so the GUI sliders can show the
    actually-applied settings instead of their hardcoded HTML defaults.

    Falls back to ``RLTConfig`` defaults for any key not present in the file,
    so the GUI always has a value to display. Returns empty dict when no RLT
    session is active.
    """
    import json as _json

    from lerobot.policies.hvla.rlt.config import RLTConfig

    defaults = RLTConfig()
    # ``active`` lets the GUI distinguish "no session — defaults" from
    # "session running — these are the live values". Without it the GUI
    # silently lets users toggle the diag button before launch / after
    # stop and the click goes nowhere (POST returns 409, frontend used
    # to swallow it). See ``set_rlt_config``.
    active = bool(_active_config and _active_config.get("rlt_output_dir"))
    result = {
        "active": active,
        "beta": defaults.beta,
        "exploration_sigma": defaults.exploration_sigma,
        "target_sigma": defaults.target_sigma,
        "dump_chunks": False,
    }
    if not active:
        return result
    override_path = Path(_active_config["rlt_output_dir"]) / "rlt_overrides.json"
    if not override_path.exists():
        return result
    try:
        with open(override_path) as f:
            overrides = _json.load(f)
        # Back-compat: legacy "actor_sigma" is a synonym for exploration_sigma.
        if "actor_sigma" in overrides and "exploration_sigma" not in overrides:
            overrides["exploration_sigma"] = overrides["actor_sigma"]
        for key in ("beta", "exploration_sigma", "target_sigma"):
            if key in overrides:
                result[key] = float(overrides[key])
        if "dump_chunks" in overrides:
            result["dump_chunks"] = bool(overrides["dump_chunks"])
    except Exception as e:
        logger.warning("RLT config read failed: %s", e)
    return result


@router.post("/rlt-config")
async def set_rlt_config(body: dict) -> dict:
    """Write RLT config overrides for the training subprocess to pick up."""
    import json as _json

    if not _active_config or not _active_config.get("rlt_output_dir"):
        raise HTTPException(409, "No active RLT session")
    # Validate: only known keys, numeric values in range. Legacy "actor_sigma"
    # is accepted as a synonym for exploration_sigma so older GUI builds don't
    # break mid-session.
    if "actor_sigma" in body and "exploration_sigma" not in body:
        body["exploration_sigma"] = body["actor_sigma"]
    allowed = {
        "beta": (0.0, 10.0),
        "exploration_sigma": (0.0, 1.0),
        "target_sigma": (0.0, 1.0),
    }
    filtered = {}
    for key, (lo, hi) in allowed.items():
        if key in body:
            val = float(body[key])
            filtered[key] = max(lo, min(hi, val))
    # Boolean flags (no range)
    if "dump_chunks" in body:
        filtered["dump_chunks"] = bool(body["dump_chunks"])
    if not filtered:
        raise HTTPException(
            400,
            "No valid keys. Allowed: beta, exploration_sigma, target_sigma, dump_chunks",
        )
    override_path = Path(_active_config["rlt_output_dir"]) / "rlt_overrides.json"
    # Ensure the output dir exists. The subprocess creates it inside ``run_s1``
    # but only after model imports finish — there's a multi-second window
    # right after launch where _active_config is set but the dir doesn't
    # exist yet, and a Diagnostic-toggle click in that window used to fail
    # with FileNotFoundError → 500 → red toast. Creating it here closes
    # that window without needing to rendezvous with the subprocess.
    override_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Merge with existing file so partial updates (e.g. only dump_chunks)
        # don't wipe other keys (beta, actor_sigma).
        existing = {}
        if override_path.exists():
            with contextlib.suppress(Exception), open(override_path) as f:
                existing = _json.load(f)
        merged = {**existing, **filtered}
        tmp = str(override_path) + ".tmp"
        with open(tmp, "w") as f:
            _json.dump(merged, f)
        os.replace(tmp, str(override_path))
        logger.info("RLT config override written: %s", filtered)
        return {"status": "ok", **filtered}
    except Exception as e:
        logger.warning("RLT config write failed: %s", e)
        raise HTTPException(500, str(e)) from e


class ControlRequest(BaseModel):
    """Flow-control command forwarded to the active subprocess's stdin.

    The command vocabulary matches the stdin control protocol consumed by
    ``lerobot.utils.keyboard_input.StdinControlListener`` — the same events the
    record loop's right/left/esc keyboard controls set.
    """

    cmd: str


_CONTROL_COMMANDS = {"exit_early", "rerecord_episode", "stop_recording"}


@router.post("/control")
async def send_control(req: ControlRequest) -> dict:
    """Write one JSON control line to the active subprocess's stdin."""
    if req.cmd not in _CONTROL_COMMANDS:
        raise HTTPException(
            400, f"Unknown control command {req.cmd!r}; expected one of {sorted(_CONTROL_COMMANDS)}"
        )
    # Bind the process once. `drain()` yields, and a concurrent /stop clears
    # _active_process — re-reading the global after the await would raise
    # AttributeError on None and surface as a 500 instead of the 409 this is.
    proc = _active_process
    if proc is None or proc.returncode is not None:
        raise HTTPException(409, "No active process to control")
    if proc.stdin is None:
        raise HTTPException(409, "Active process has no control channel")

    line = json.dumps({"v": 1, "cmd": req.cmd}) + "\n"
    try:
        proc.stdin.write(line.encode())
        await proc.stdin.drain()
    except (BrokenPipeError, ConnectionResetError) as e:
        raise HTTPException(409, f"Control channel broken: {e}") from e
    logger.info(f"Control channel: {req.cmd} forwarded to PID {proc.pid}")
    return {"status": "sent", "cmd": req.cmd, "pid": proc.pid}


# How long Stop waits for a record run to save and exit on its own before
# escalating to SIGINT. Save runs AFTER the reset phase, so at the moment the
# operator clicks Stop a finished episode is usually still only in memory;
# killing before save_episode() completes silently discards it. The window
# must cover a buffered video encode of one episode.
STOP_GRACE_S = 30.0

# True while a graceful stop is waiting on the subprocess. A second Stop click
# during the wait escalates to SIGINT immediately instead of queueing another
# grace period — the operator pressing Stop twice means "kill it".
_graceful_stop_inflight = False


@router.post("/stop")
async def stop_process() -> dict:
    global _active_process, _active_command, _active_config, _active_phase, _graceful_stop_inflight

    # Bind once: every await below yields, and the globals can be cleared by a
    # concurrent caller in the meantime (same race send_control guards against).
    proc = _active_process
    if proc is None:
        raise HTTPException(409, "No active process to stop")
    pid = proc.pid

    # Record runs get a graceful stop first. The subprocess's stop_recording
    # command skips the reset phase, SAVES the buffered episode and exits
    # cleanly — whereas SIGINT during the reset phase discards an episode the
    # operator already finished recording. Observed in the field: record one
    # episode, press Stop during reset, dataset ends up empty.
    graceful = False
    if (
        _active_command == "record"
        and proc.stdin is not None
        and proc.returncode is None
        and not _graceful_stop_inflight
    ):
        _graceful_stop_inflight = True
        try:
            proc.stdin.write((json.dumps({"v": 1, "cmd": "stop_recording"}) + "\n").encode())
            await proc.stdin.drain()
            _append_output(f"\n--- Stop requested: finishing save, then exiting (PID {pid}) ---\n")
            await asyncio.wait_for(proc.wait(), timeout=STOP_GRACE_S)
            graceful = True
        except (*_TIMEOUT_EXCS, BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.warning(f"Graceful stop failed for PID {pid} ({e}); escalating to SIGINT")
        finally:
            _graceful_stop_inflight = False

    if _active_command == "flash" and proc.returncode is None and not _graceful_stop_inflight:
        _graceful_stop_inflight = True
        try:
            proc.send_signal(signal.SIGINT)
            _append_output(f"\n--- Stop requested: saving FLASH trial (PID {pid}) ---\n")
            await asyncio.wait_for(proc.wait(), timeout=STOP_GRACE_S)
            graceful = True
        except (*_TIMEOUT_EXCS, ProcessLookupError) as e:
            logger.warning(f"Graceful FLASH stop failed for PID {pid} ({e}); escalating")
        finally:
            _graceful_stop_inflight = False

    if proc.returncode is None:
        try:
            proc.send_signal(signal.SIGINT)
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except _TIMEOUT_EXCS:
                proc.kill()
                await proc.wait()
        except ProcessLookupError:
            pass

    _append_output(f"\n--- Process stopped (PID {pid}) ---\n")
    _active_process = None
    _active_command = None
    _active_config = None
    _active_phase = None
    _close_obs_reader()
    # NOTE: debug model is NOT stopped here — it stays warm for reuse
    return {"status": "stopped", "pid": pid, "graceful": graceful}


@router.get("/status")
async def get_status() -> dict:
    from lerobot.gui.api._run_core import get_run_status

    return get_run_status()


@router.get("/output")
async def stream_output() -> StreamingResponse:
    """Server-Sent Events stream of subprocess output."""

    async def event_generator():
        sent_lines = 0
        sent_overlay = None  # track last overlay sent to avoid duplicates
        proc_at_start = _active_process

        logger.info(
            f"SSE: connected, buffered={len(_output_lines)} lines, process={'running' if proc_at_start and proc_at_start.returncode is None else 'none/done'}"
        )

        # Send existing buffered lines first
        while sent_lines < len(_output_lines):
            line = _output_lines[sent_lines]
            yield f"data: {json.dumps({'line': line})}\n\n"
            sent_lines += 1

        # Then wait for new lines
        while True:
            _output_event.clear()

            # Send overlay update if changed
            if _overlay_state is not None and _overlay_state != sent_overlay:
                sent_overlay = _overlay_state.copy()
                yield f"data: {json.dumps({'overlay': sent_overlay})}\n\n"

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
                logger.info(
                    f"SSE: done (process={'exited rc=' + str(_active_process.returncode) if _active_process else 'None'}, sent={sent_lines} lines)"
                )
                yield f"data: {json.dumps({'done': True})}\n\n"
                return

            try:
                await asyncio.wait_for(_output_event.wait(), timeout=2.0)
            except _TIMEOUT_EXCS:
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
_obs_reader_meta_ino: int | None = None  # inode of /dev/shm/lerobot_obs_meta at attach time
_jpeg_cache: dict[str, tuple[int, bytes]] = {}  # cam_key → (seq, jpeg_bytes)


def _get_obs_reader():
    """Lazily attach to the robot's observation stream.

    Detects stream recreation (e.g. new teleop session after a crash) by
    the tap's own identity — if it changed, the old reader is stale (mapped
    to unlinked segments) and we re-attach.
    """
    from lerobot.robots.obs_stream import stream_identity

    global _obs_reader, _obs_reader_meta_ino

    if _obs_reader is not None:
        # Cheap staleness check: has the meta segment been recreated?
        current_ino = stream_identity()
        if current_ino is None:
            _close_obs_reader()
            return None
        if current_ino != _obs_reader_meta_ino:
            logger.info("ObservationStream recreated (inode changed), re-attaching reader")
            _close_obs_reader()
            # Fall through to create new reader
        else:
            return _obs_reader

    try:
        from lerobot.robots.obs_stream import ObservationStreamReader

        _obs_reader = ObservationStreamReader()
        _obs_reader_meta_ino = stream_identity()
        logger.info(
            "ObservationStreamReader attached: %d scalars, %d cameras",
            len(_obs_reader.obs_scalar_keys),
            len(_obs_reader.image_keys),
        )
        return _obs_reader
    except Exception:
        return None


def _close_obs_reader():
    global _obs_reader, _obs_reader_meta_ino
    if _obs_reader is not None:
        _obs_reader.close()
        _obs_reader = None
    _obs_reader_meta_ino = None
    _jpeg_cache.clear()


@router.get("/obs-stream/meta")
async def obs_stream_meta() -> dict:
    """Return observation stream layout (feature names, image dims)."""
    reader = _get_obs_reader()
    if reader is None:
        logger.debug("obs-stream/meta: reader not available")
        return {"available": False}
    logger.info("obs-stream/meta: available, cameras=%s", list(reader.image_keys.keys()))
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


@router.get("/urdf-viz/meta")
async def urdf_viz_meta() -> dict:
    """One-shot identity + advertised sources for the in-browser URDF viewer.

    The frontend fetches this once at iframe init to learn (1) which robot is
    on the run (so it can load the URDF and arrange arms in the scene) and
    (2) which pose sources are currently available. ``sources`` is a list of
    user-facing names — today ``"state"`` (always) and ``"action"`` (when the
    run is writing the commanded action to the obs stream). Future sources
    (e.g. ``"prediction"`` from a debug model) appear here without changing
    the rest of the contract.

    ``{"available": false}`` when no run is streaming or the robot has no
    vendored URDF.
    """
    from lerobot.gui.urdf_viz import resolve_robot

    reader = _get_obs_reader()
    if reader is None:
        return {"available": False}
    obs_result = reader.read_obs()
    if not obs_result:
        return {"available": False}
    spec = resolve_robot(obs_result[0].keys())
    if spec is None:
        return {"available": False}
    sources = ["state"]
    if reader.read_action():
        sources.append("action")
    return {
        "available": True,
        "name": spec.name,
        "urdf": f"/urdf-assets/{spec.urdf_url_path}",
        # Mirrored-arm robots (e.g. OpenArm) ship a separate right-arm URDF;
        # None means both arms load ``urdf``.
        "urdf_right": f"/urdf-assets/{spec.urdf_url_path_right}" if spec.urdf_url_path_right else None,
        # Per-arm base offsets (URDF world frame) from the description;
        # None lets the frontend use its default side-by-side spacing.
        "base_offsets": spec.base_offsets,
        "bimanual": len(spec.arms) == 2,
        "sources": sources,
        # ee_link is None for descriptions that didn't declare one; the
        # frontend skips polyline rendering in that case.
        "ee_link": spec.ee_link,
    }


@router.get("/urdf-viz")
async def urdf_viz_source(source: str = "state") -> dict:
    """Per-arm URDF joint angles for one named source.

    ``source`` mirrors the user-facing names advertised by
    :func:`urdf_viz_meta` (``"state"`` / ``"action"``); the backend decides
    how to fulfil each one. The response is the same shape regardless of
    source — a list of arms, each with one or more ``frames``. A teleop's
    action is a single frame; a future chunk-output policy would return many
    with an ``fps`` for playback timing. The renderer applies frames[0] for
    a pose and steps through for a chunk; it does not need to know which
    backend produced the data.

    ``{"available": false}`` when no run is streaming, the robot has no
    vendored URDF, or the requested source has no current data.
    """
    from lerobot.gui.urdf_viz import compute_joint_angles, resolve_robot

    reader = _get_obs_reader()
    if reader is None:
        return {"available": False}
    obs_result = reader.read_obs()
    if not obs_result:
        return {"available": False}
    obs = obs_result[0]
    spec = resolve_robot(obs.keys())
    if spec is None:
        return {"available": False}

    if source == "state":
        sample = obs
    elif source == "action":
        act_result = reader.read_action()
        if not act_result:
            return {"available": False}
        sample = act_result[0]
    else:
        raise HTTPException(status_code=400, detail=f"unknown source: {source!r}")

    angles = compute_joint_angles(spec, sample)
    arms_payload = [
        {"prefix": a.obs_prefix, "frames": [{"joints": angles.get(a.obs_prefix, {})}]} for a in spec.arms
    ]
    return {"available": True, "arms": arms_payload}


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
        # A large frame may be overwritten during all bounded read attempts.
        # Keep displaying the last coherent JPEG instead of encoding bytes
        # that failed the shared-memory sequence check.
        if cached is not None:
            return Response(
                content=cached[1],
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store", "X-Lerobot-Frame-Stale": "1"},
            )
        raise HTTPException(503, f"No coherent image available yet for camera '{cam_key}'")
    img, _ts = result

    import cv2

    def _encode() -> bytes:
        # cv2.cvtColor + imencode is 5–10 ms for a 720p frame. Push it off
        # the event loop so concurrent obs-stream image requests for other
        # cameras + the obs_stream_state poll don't queue up behind it.
        # cv2 releases the GIL during the C-side encode, so the default
        # thread pool scales naturally.
        _, jpeg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpeg.tobytes()

    jpeg_bytes = await asyncio.get_event_loop().run_in_executor(None, _encode)
    _jpeg_cache[cam_key] = (seq, jpeg_bytes)
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )
