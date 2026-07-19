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

"""Backend-agnostic visualization dispatch.

Selects a visualization backend at runtime via a display-mode string (e.g. a ``--display_mode`` CLI
flag) so callers never branch on the backend. The concrete implementations live in
:mod:`lerobot.utils.rerun_visualization` and :mod:`lerobot.utils.foxglove_visualization`; importing
this module does not import ``rerun`` or ``foxglove`` (each backend imports its SDK lazily behind a
``require_package`` guard).
"""

import os

from lerobot.types import RobotAction, RobotObservation

from .foxglove_visualization import init_foxglove, log_foxglove_data, shutdown_foxglove
from .import_utils import require_package
from .rerun_visualization import log_rerun_data, shutdown_rerun
from .rerun_visualization import init_rerun as _init_rerun

# Visualization backends selectable at runtime via a display-mode string (e.g. a --display_mode flag).
VISUALIZATION_MODES = ("rerun", "foxglove")


def init_rerun(session_name: str = "lerobot_control_loop", ip: str | None = None, port: int | None = None) -> None:
    """Initializes the Rerun SDK, honoring the fork's ``LEROBOT_RERUN_SERVE_PORT`` override.

    Delegates to :func:`lerobot.utils.rerun_visualization.init_rerun` normally.

    fork-only: when ``LEROBOT_RERUN_SERVE_PORT`` is set, host a gRPC server on
    this port instead of connecting to an external one (or spawning a local
    viewer). Used by the GUI so the web viewer can pull data directly from this
    process (connect_grpc to an external ``rerun --serve-web`` drops images in
    the web viewer — Rerun 0.26 bug).
    """
    serve_port = os.environ.get("LEROBOT_RERUN_SERVE_PORT")
    if not serve_port:
        _init_rerun(session_name=session_name, ip=ip, port=port)
        return

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    log_rerun_data.blueprint = None  # Reset blueprint cache for new session

    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    rr.serve_grpc(grpc_port=int(serve_port), server_memory_limit=memory_limit)


def init_visualization(
    display_mode: str,
    *,
    session_name: str = "lerobot_control_loop",
    ip: str | None = None,
    port: int | None = None,
) -> None:
    """Initializes the visualization backend selected by ``display_mode``.

    For ``"rerun"``, ``ip``/``port`` point at an optional remote Rerun server. For ``"foxglove"``,
    ``ip`` is the interface to bind the WebSocket server to (``127.0.0.1`` for local only, ``0.0.0.0``
    for all interfaces) and ``port`` is its port.
    """

    if display_mode == "rerun":
        init_rerun(session_name=session_name, ip=ip, port=port)
    elif display_mode == "foxglove":
        init_foxglove(host=ip or "127.0.0.1", port=port)
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")


def log_visualization_data(
    display_mode: str,
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """Logs observation/action data to the backend selected by ``display_mode``."""

    if display_mode == "rerun":
        log_rerun_data(observation=observation, action=action, compress_images=compress_images)
    elif display_mode == "foxglove":
        log_foxglove_data(observation=observation, action=action, compress_images=compress_images)
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")


def shutdown_visualization(display_mode: str) -> None:
    """Shuts down the backend selected by ``display_mode``."""

    if display_mode == "rerun":
        shutdown_rerun()
    elif display_mode == "foxglove":
        shutdown_foxglove()
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")
