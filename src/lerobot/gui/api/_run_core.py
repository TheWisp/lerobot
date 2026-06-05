# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Shared read-only helpers for the run-tracking surface.

Unlike ``_edits_core.py`` (which centers on ``AppState.pending_edits``)
and ``_hub_core.py`` (which centers on ``AppState.hub_jobs``), the
run-state lives in module globals on ``lerobot.gui.api.run``
(``_active_process``, ``_active_command``, ``_active_config``,
``_output_lines``). The subprocess lifecycle is tightly coupled to the
asyncio event loop, which makes storing the state on an immutable
dataclass awkward; the helpers below read those globals through their
public proxies on ``run`` instead.

Read-only by design: no helper here mutates state, sends signals, or
spawns subprocesses. Stopping / starting runs is a separate operate-tier
surface that lands behind explicit operator sign-off.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_EMPTY_LATENCY_SNAPSHOT: dict[str, Any] = {
    "n_records": 0,
    "dropped_records": 0,
    "overrun_ratio": 0.0,
    "stages": {},
    "series": {},
}

_EMPTY_RLT_METRICS: dict[str, Any] = {
    "episode": 0,
    "step_count": 0,
    "buffer_size": 0,
    "total_updates": 0,
    "mode": "IDLE",
    "success_rate": 0,
    "total_successes": 0,
    "total_episodes": 0,
    "series": {},
}


def get_run_status() -> dict[str, Any]:
    """Current state of the GUI-managed subprocess.

    Three-way response — the AI branches on ``running``:

    - ``{"running": False, "command": None}`` — no subprocess at all.
    - ``{"running": False, "command": "<kind>", "returncode": int}`` —
      a subprocess existed but has since exited; ``returncode`` is
      its exit status (0 = clean, non-zero = error or signal).
    - ``{"running": True, "command": "<kind>", "pid": int}`` —
      subprocess is currently active.

    ``command`` is one of the launcher kinds (``'teleoperate'``,
    ``'record'``, ``'replay'``, ``'hvla'``) — the same value the GUI's
    "Run" tab shows.
    """
    from lerobot.gui.api import run as run_mod

    proc = run_mod._active_process
    if proc is None:
        return {"running": False, "command": None}
    returncode = proc.returncode
    if returncode is not None:
        return {
            "running": False,
            "command": run_mod._active_command,
            "returncode": returncode,
        }
    return {
        "running": True,
        "command": run_mod._active_command,
        "pid": proc.pid,
    }


def get_run_output_tail(last_n: int = 200) -> dict[str, Any]:
    """Snapshot of the tail of the active subprocess's captured output.

    The FastAPI ``/api/run/output`` route streams via Server-Sent Events
    for the GUI's live log panel; this helper exists for callers that
    want a discrete snapshot (the MCP tool, ad-hoc scripts) without
    holding a streaming connection.

    Args:
        last_n: Max number of lines to include. The buffer is capped at
            2000 lines on the GUI side; asking for more silently caps
            at the buffer size.

    Returns ``{"lines": [...], "total_buffered": N, "truncated": bool}``.
    ``truncated=True`` when the buffer held more lines than were returned.
    """
    from lerobot.gui.api import run as run_mod

    if last_n < 0:
        raise ValueError(f"last_n must be >= 0; got {last_n}")
    buf = list(run_mod._output_lines)  # snapshot in case of concurrent append
    total = len(buf)
    if last_n == 0:
        return {"lines": [], "total_buffered": total, "truncated": total > 0}
    take = min(last_n, total)
    return {
        "lines": buf[-take:],
        "total_buffered": total,
        "truncated": total > take,
    }


def get_latency_metrics(source: str = "teleop") -> dict[str, Any]:
    """Latest latency snapshot for the requested loop source.

    Each subprocess (teleop / record / HVLA inference) atomically
    replaces ``<source_dir>/latency_snapshot.json`` once per second; this
    function reads the matching file. Returns the empty-snapshot stub
    when no snapshot exists yet (fresh session, source not running,
    unknown source) so callers can branch on ``n_records`` without
    error-handling.

    Args:
        source: One of the keys in ``LATENCY_SOURCES`` on the GUI side
            — currently ``'teleop'``, ``'record'``, or ``'hvla'``.
            Unknown sources return the empty stub.

    Returns ``{"n_records": N, "dropped_records": N, "overrun_ratio":
    float, "stages": {...}, "series": {...}}``. ``stages`` are per-stage
    latency stats (mean/p95/max); ``series`` is the windowed time-series
    the dashboard plots.
    """
    from lerobot.gui.api.run import LATENCY_SOURCES

    source_dir = LATENCY_SOURCES.get(source)
    if source_dir is None:
        return dict(_EMPTY_LATENCY_SNAPSHOT)
    snapshot_path = Path(source_dir) / "latency_snapshot.json"
    try:
        if snapshot_path.exists():
            with open(snapshot_path, encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:  # noqa: BLE001
        # Atomic-replace race: a concurrent rename can leave the path
        # momentarily missing or partially-readable. Harmless — the next
        # poll succeeds.
        logger.debug("latency snapshot read failed (source=%s): %s", source, e)
    return dict(_EMPTY_LATENCY_SNAPSHOT)


_RLT_RANGES = {
    # Numeric override keys + their valid range. Updates outside the range
    # are clamped, not rejected — same behavior the FastAPI handler has
    # had since it shipped; the response surfaces the clamping so the
    # caller knows what actually landed.
    "beta": (0.0, 10.0),
    "exploration_sigma": (0.0, 1.0),
    "target_sigma": (0.0, 1.0),
}


class NoActiveRunError(RuntimeError):
    """No active RLT training run — overrides have nowhere to land."""


class EditValidationError(ValueError):
    """Update call carried no valid override keys."""


def update_rlt_overrides(
    *,
    beta: float | None = None,
    exploration_sigma: float | None = None,
    target_sigma: float | None = None,
    dump_chunks: bool | None = None,
) -> dict[str, Any]:
    """Write RLT config overrides for the active training subprocess.

    Numeric values are clamped to their valid range (``beta`` to
    ``[0, 10]``, both sigma fields to ``[0, 1]``); the response carries
    a ``clamped`` dict whenever a value was adjusted so the caller can
    tell the difference between "I set what I asked for" and "the
    server pinned my request to the boundary." Partial updates merge
    with any existing override file — keys you don't pass are left
    alone.

    Raises:
        NoActiveRunError: ``_active_config`` is unset or has no
            ``rlt_output_dir`` — there's no live RLT session to update.
        EditValidationError: every argument was ``None`` (nothing to do).
    """
    import json as _json
    import os

    from lerobot.gui.api import run as run_mod

    if not run_mod._active_config or not run_mod._active_config.get("rlt_output_dir"):
        raise NoActiveRunError("No active RLT session — start an HVLA run first, then update overrides.")

    requested: dict[str, Any] = {}
    if beta is not None:
        requested["beta"] = float(beta)
    if exploration_sigma is not None:
        requested["exploration_sigma"] = float(exploration_sigma)
    if target_sigma is not None:
        requested["target_sigma"] = float(target_sigma)
    if dump_chunks is not None:
        requested["dump_chunks"] = bool(dump_chunks)

    if not requested:
        raise EditValidationError(
            "No override fields provided. Pass at least one of: "
            "beta, exploration_sigma, target_sigma, dump_chunks."
        )

    filtered: dict[str, Any] = {}
    clamped: dict[str, dict[str, Any]] = {}
    for key, (lo, hi) in _RLT_RANGES.items():
        if key not in requested:
            continue
        raw = requested[key]
        applied = max(lo, min(hi, raw))
        filtered[key] = applied
        if applied != raw:
            clamped[key] = {"requested": raw, "applied": applied, "range": [lo, hi]}
    if "dump_chunks" in requested:
        filtered["dump_chunks"] = requested["dump_chunks"]

    override_path = Path(run_mod._active_config["rlt_output_dir"]) / "rlt_overrides.json"
    override_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing file so partial updates don't wipe other keys.
    previous: dict[str, Any] = {}
    if override_path.exists():
        try:
            with open(override_path) as f:
                previous = _json.load(f)
        except Exception:  # noqa: BLE001 — corrupt file becomes "empty"
            previous = None  # type: ignore[assignment]
            previous = {}
    merged = {**previous, **filtered}
    tmp = str(override_path) + ".tmp"
    with open(tmp, "w") as f:
        _json.dump(merged, f)
    os.replace(tmp, str(override_path))
    logger.info("RLT config override written via _run_core: %s", filtered)

    result: dict[str, Any] = {
        "status": "ok",
        "applied": filtered,
        "previous_values": {k: previous.get(k) for k in filtered},
        "override_path": str(override_path),
    }
    if clamped:
        result["clamped"] = clamped
    return result


def get_rlt_metrics() -> dict[str, Any]:
    """RLT training metrics for the currently-active RLT run.

    Reads the ``metrics.json`` that belongs to the active RLT session
    (resolved from ``_active_config["rlt_output_dir"]``). Returns the
    empty-metrics stub when no RLT run is active, so callers can branch
    on ``mode == "IDLE"`` instead of error-handling.

    Returns ``{"episode", "step_count", "buffer_size", "total_updates",
    "mode", "success_rate", "total_successes", "total_episodes",
    "series"}`` — ``mode == "IDLE"`` means no live RLT session.
    """
    from lerobot.gui.api import run as run_mod

    try:
        from lerobot.policies.hvla.rlt.metrics import load_metrics_from_file

        path = None
        if run_mod._active_config and run_mod._active_config.get("rlt_output_dir"):
            path = str(Path(run_mod._active_config["rlt_output_dir"]) / "metrics.json")
        data = load_metrics_from_file(path)
        if data:
            return data
    except Exception as e:  # noqa: BLE001
        logger.warning("RLT metrics read failed: %s", e)
    return dict(_EMPTY_RLT_METRICS)
