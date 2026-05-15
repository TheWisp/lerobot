# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FastAPI server for the LeRobot Dataset GUI."""

from __future__ import annotations

import argparse
import contextlib
import logging
import logging.config
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from lerobot.gui.api import bug_reports, datasets, edits, models, playback, robot, run
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LeRobot Dataset GUI",
    description="Web-based visualization and editing tool for LeRobot datasets",
    version="0.1.0",
)

# Global app state (initialized on startup)
_app_state: AppState | None = None


def parse_cache_size(size_str: str) -> int:
    """Parse cache size string like '500MB' or '1GB' to bytes."""
    size_str = size_str.strip().upper()
    # Check longer suffixes first to avoid "MB" matching "B"
    multipliers = [
        ("GB", 1024 * 1024 * 1024),
        ("MB", 1024 * 1024),
        ("KB", 1024),
        ("B", 1),
    ]
    for suffix, mult in multipliers:
        if size_str.endswith(suffix):
            return int(float(size_str[: -len(suffix)]) * mult)
    return int(size_str)


@app.on_event("startup")
async def startup_event():
    """Initialize app state on startup."""
    global _app_state
    # Default cache size, can be overridden via CLI
    cache_size = getattr(app.state, "cache_size", 1_000_000_000)
    _app_state = AppState(frame_cache=FrameCache(max_bytes=cache_size))
    datasets.set_app_state(_app_state)
    playback.set_app_state(_app_state)
    edits.set_app_state(_app_state)
    robot.set_app_state(_app_state)
    run.set_app_state(_app_state)
    models.set_app_state(_app_state)
    bug_reports.set_app_state(_app_state)
    logger.info(f"Initialized frame cache with {cache_size / 1_000_000:.0f} MB budget")
    # Sweep stale obs-stream shared-memory segments left by a previously-
    # crashed teleop/record subprocess. Without this, the GUI's reader
    # auto-attaches to the leftover segments and serves frozen data,
    # making it look like teleop is running when it isn't.
    from lerobot.robots.obs_stream import cleanup_stale_streams

    n = cleanup_stale_streams()
    if n:
        logger.info("Swept %d stale obs-stream shm segment(s) from a previous run", n)


async def _terminate_active_process(*, sigint_grace_s: float = 5.0) -> bool:
    """Kill the active teleop/record subprocess, if any.

    Returns True if a subprocess was found and terminated (or was already
    dead), False if there was no subprocess to begin with. Bounded: this
    function will always return within roughly ``sigint_grace_s`` seconds
    even if the subprocess ignores SIGINT.

    Extracted from the shutdown hook so it can be unit-tested with a
    mocked subprocess. A reference to ``_active_process`` is fetched
    lazily here (not via top-level import) so we always see the current
    value of the module global rather than a stale snapshot.
    """
    import asyncio
    import signal

    from lerobot.gui.api import run as run_module

    proc = run_module._active_process
    if proc is None or proc.returncode is not None:
        return False

    # SIGINT first so the subprocess gets a chance to run its
    # disconnect() cleanup (which includes ObservationStream.cleanup()
    # that unlinks the shm). Fall back to SIGKILL after the grace
    # period. We intentionally do NOT await proc.wait() after kill():
    # the original code didn't either, and uvicorn's shutdown hook is
    # sensitive to long awaits (one stuck wait can wedge the whole
    # event loop and the user's Ctrl+C is silently ignored). The OS
    # reaps zombies when the parent (us) exits, and the defensive
    # cleanup_stale_streams() below catches any leaked shm anyway.
    proc.send_signal(signal.SIGINT)
    try:
        await asyncio.wait_for(proc.wait(), timeout=sigint_grace_s)
    except Exception:
        with contextlib.suppress(Exception):
            proc.kill()
    return True


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up subprocesses + their shared memory on server shutdown.

    Each step is best-effort: a failure in one step MUST NOT block the
    others, or a single misbehaving cleanup callback would silently leak
    every later resource. Failures are logged with exc_info so the cause
    isn't lost. Tracked partially in gui/TODO.md's "Foolproof shutdown-
    cleanup registry" entry — this is the safety-net version pending
    the registry pattern.
    """
    import asyncio

    from lerobot.gui.api.datasets import shutdown_decode_executor, shutdown_prefetch_executor
    from lerobot.gui.api.robot import cleanup_in_process_resources
    from lerobot.gui.api.run import _stop_debug_process
    from lerobot.robots.obs_stream import cleanup_stale_streams

    # Stop debug model process
    try:
        await _stop_debug_process()
    except Exception:
        logger.exception("shutdown: _stop_debug_process failed")
    # Stop active teleop/record subprocess (no-op if none).
    try:
        await _terminate_active_process()
    except Exception:
        logger.exception("shutdown: _terminate_active_process failed")
    # Defensive sweep: even if the subprocess ran its own cleanup, also
    # unlink any segments it may have leaked (SIGKILL path, abort, etc.).
    # Safe at this point because the subprocess is no longer running.
    try:
        n = cleanup_stale_streams()
        if n:
            logger.info("Swept %d stale obs-stream shm segment(s) on shutdown", n)
    except Exception:
        logger.exception("shutdown: cleanup_stale_streams failed")
    # Release in-process hardware: preview cameras the Robot tab opened,
    # plus any rest-position / safe-trajectory recording robot the user
    # started but never finished. Run in the default executor — the
    # underlying disconnect()s do blocking serial / V4L2 I/O.
    try:
        await asyncio.get_event_loop().run_in_executor(None, cleanup_in_process_resources)
    except Exception:
        logger.exception("shutdown: cleanup_in_process_resources failed")
    # Cancel any in-flight prefetch work + release the executor's worker
    # thread so uvicorn's shutdown doesn't race against a long-running
    # video-decode pass.
    try:
        shutdown_prefetch_executor()
    except Exception:
        logger.exception("shutdown: shutdown_prefetch_executor failed")
    try:
        shutdown_decode_executor()
    except Exception:
        logger.exception("shutdown: shutdown_decode_executor failed")
    # Send mDNS goodbye packet so clients don't keep a stale
    # lerobot.local entry cached after we stop.
    mdns_handle = getattr(app.state, "mdns_handle", None)
    if mdns_handle is not None:
        with contextlib.suppress(Exception):
            mdns_handle.unregister()


# Include API routers
app.include_router(datasets.router)
app.include_router(playback.router)
app.include_router(edits.router)
app.include_router(robot.router)
app.include_router(run.router)
app.include_router(models.router)
app.include_router(bug_reports.router)

# Serve static files (CSS, JS)
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/")
async def root():
    """Serve the HTML viewer."""
    return FileResponse(_static_dir / "index.html", media_type="text/html")


def run_server(host: str = "127.0.0.1", port: int = 8000, cache_size: int = 1_000_000_000):
    """Run the GUI server."""
    import uvicorn

    from lerobot.gui.mdns import advertise, detect_lan_ip

    app.state.cache_size = cache_size

    # Advertise on mDNS so phones / laptops on the same LAN can hit us
    # at http://lerobot.local:<port>/ without knowing the IP. The
    # advertise call returns None if we're loopback-only or zeroconf
    # isn't installed; in either case we still print the always-available
    # raw URL below.
    app.state.mdns_handle = advertise(host=host, port=port)

    # Startup banner: spell out every URL the user can reach us at, in
    # order of "nicest to type". Pre-uvicorn so it's not buried under
    # the uvicorn boot logs.
    lines = ["LeRobot host service ready:"]
    if app.state.mdns_handle is not None:
        lines.append(f"  mDNS (no IP needed): http://{app.state.mdns_handle.hostname}:{port}/")
    lan_ip = detect_lan_ip()
    if lan_ip is not None and host not in ("127.0.0.1", "::1", "localhost"):
        lines.append(f"  LAN:                 http://{lan_ip}:{port}/")
    # The "0.0.0.0" literal is for display-name comparison only (we
    # echo "localhost" instead in the banner); not a bind() argument.
    all_ifaces = "0.0.0.0"  # nosec B104 - label comparison, not a network bind
    local_label = "localhost" if host in (all_ifaces, "127.0.0.1") else host
    lines.append(f"  Local:               http://{local_label}:{port}/")
    for line in lines:
        logger.info(line)

    # log_config=None: keep the uvicorn handlers we attached in setup_logging.
    # Without this, uvicorn calls dictConfig at startup and replaces them.
    uvicorn.run(app, host=host, port=port, access_log=False, log_config=None)


def setup_logging(log_dir: Path | None = None) -> Path:
    """Configure persistent file logging for the GUI server.

    Only configures logging for lerobot.gui.* loggers, not the root logger.
    Called once at server startup.

    Args:
        log_dir: Directory for log files. Defaults to ~/.cache/huggingface/lerobot/gui/logs/

    Returns:
        Path to the log directory.
    """
    if log_dir is None:
        log_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "gui" / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log file with date in name
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"server_{date_str}.log"

    # Configure lerobot.gui plus uvicorn/uvicorn.error in one shared dictConfig so
    # both share a single rotating file handler (avoids RotatingFileHandler races
    # when two handlers point at the same path) and so uvicorn/starlette ASGI
    # tracebacks reach the file log — not just the terminal that started the
    # server. We pair this with `log_config=None` in run_server to stop uvicorn
    # from clobbering these loggers via its own dictConfig at startup.
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"default": {"format": log_format}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "level": "INFO",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": str(log_file),
                    "maxBytes": 10 * 1024 * 1024,
                    "backupCount": 10,
                    "encoding": "utf-8",
                    "formatter": "default",
                    "level": "INFO",
                },
            },
            "loggers": {
                "lerobot.gui": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
                "uvicorn": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
                "uvicorn.error": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    logging.getLogger("lerobot.gui").info(f"Logging to {log_file}")

    return log_dir


def main():
    """CLI entry point."""
    # Capture native-crash stack traces (SIGSEGV / SIGABRT in C extensions
    # like torchcodec) to stderr so they show up in the server log instead
    # of vanishing as a silent "core dumped".
    import faulthandler

    faulthandler.enable()

    parser = argparse.ArgumentParser(description="LeRobot Dataset GUI Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument(
        "--cache-size",
        default="1GB",
        help="Frame cache size (default: 1GB). Examples: 500MB, 1GB, 2GB",
    )

    args = parser.parse_args()

    # Setup persistent logging before starting server
    setup_logging()

    cache_bytes = parse_cache_size(args.cache_size)
    run_server(host=args.host, port=args.port, cache_size=cache_bytes)
