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
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from lerobot.gui.api import datasets, edits, playback
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
    cache_size = getattr(app.state, "cache_size", 500_000_000)
    _app_state = AppState(frame_cache=FrameCache(max_bytes=cache_size))
    datasets.set_app_state(_app_state)
    playback.set_app_state(_app_state)
    edits.set_app_state(_app_state)
    logger.info(f"Initialized frame cache with {cache_size / 1_000_000:.0f} MB budget")


# Include API routers
app.include_router(datasets.router)
app.include_router(playback.router)
app.include_router(edits.router)

# Serve static files (CSS, JS)
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/")
async def root():
    """Serve the HTML viewer."""
    return FileResponse(_static_dir / "index.html", media_type="text/html")


def run_server(host: str = "127.0.0.1", port: int = 8000, cache_size: int = 500_000_000):
    """Run the GUI server."""
    import uvicorn

    app.state.cache_size = cache_size
    logger.info(f"Starting LeRobot Dataset GUI at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, access_log=False)


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

    # Configure only the lerobot.gui logger (not root)
    gui_logger = logging.getLogger("lerobot.gui")
    gui_logger.setLevel(logging.INFO)
    gui_logger.propagate = False  # Don't propagate to root logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)

    # File handler (rotating, max 10MB per file, keep 10 files)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_format)

    gui_logger.addHandler(console_handler)
    gui_logger.addHandler(file_handler)

    gui_logger.info(f"Logging to {log_file}")

    return log_dir


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="LeRobot Dataset GUI Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument(
        "--cache-size",
        default="500MB",
        help="Frame cache size (default: 500MB). Examples: 500MB, 1GB, 2GB",
    )

    args = parser.parse_args()

    # Setup persistent logging before starting server
    setup_logging()

    cache_bytes = parse_cache_size(args.cache_size)
    run_server(host=args.host, port=args.port, cache_size=cache_bytes)
