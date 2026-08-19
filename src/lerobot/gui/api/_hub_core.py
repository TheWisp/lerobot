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
"""Shared business logic for Hugging Face Hub read-only operations.

Mirrors the pattern of ``_edits_core.py``: sync pure-Python helpers that
both the FastAPI ``/api/hub/*`` routes and the new MCP ``hub_*`` tools
call. No HTTP self-call, no cross-surface duplication.

Only one typed exception is needed:

- ``HubJobNotFoundError`` → FastAPI 404 / MCP error (caller passed an
  unknown ``job_id``).

Auth and repo-info probes deliberately catch all exceptions and return
a transparent ``{"logged_in": False, ...}`` / ``{"exists": False, ...}``
shape so the agent can branch on the result without parsing error text.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

# Hub reachability checks (`whoami`) run here rather than on the default
# executor, which is contended with frame decode and camera work. The call is a
# sync network round-trip that can hang for minutes when the Hub is
# unreachable — the case this check exists to catch — so it must not occupy a
# thread anything else is waiting for.
hub_auth_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gui-hub-auth")


# ── Typed exceptions ──────────────────────────────────────────────────────


class HubJobNotFoundError(KeyError):
    """Hub job id not present in ``AppState.hub_jobs``."""


# ── Public helpers — called by both FastAPI handlers and MCP tools ────────


def get_auth_status() -> dict[str, Any]:
    """Probe HF Hub auth via ``whoami()``.

    Returns ``{"logged_in": bool, "username": str | None}``. Cheap
    single GET; not cached because auth state changes rarely and
    callers want freshness. Any exception path collapses to
    ``logged_in=False`` — the agent should treat absence of a valid
    token, expired tokens, and network failures uniformly (re-run
    ``huggingface-cli login`` or set ``HF_TOKEN``).
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi().whoami()
        return {
            "logged_in": True,
            "username": info.get("name", info.get("fullname", "unknown")),
        }
    except Exception:  # noqa: BLE001 — probe collapses all failures
        return {"logged_in": False, "username": None}


def get_repo_info(repo_id: str, repo_type: str = "dataset") -> dict[str, Any]:
    """Look up a repo on the Hub.

    ``repo_type`` selects the namespace: models live at the Hub root, datasets
    under ``/datasets``, and the two are separate ID spaces — a model lookup
    against the dataset API reports "not found" for a repo that exists.

    Returns ``{"exists": bool, ...}``. When ``exists=False`` (repo
    missing, private with no access, network down) only ``repo_id``
    is filled — fields like ``total_episodes`` are omitted rather than
    nulled so an agent can branch unambiguously on ``"exists"``.

    Best-effort enriches with remote ``meta/info.json`` (episode and
    frame counts, fps) when available; that fetch is wrapped in its
    own try/except so a missing or unreadable info.json doesn't fail
    the whole call.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = (
            api.model_info(repo_id, files_metadata=True)
            if repo_type == "model"
            else api.dataset_info(repo_id, files_metadata=True)
        )
    except Exception as e:  # noqa: BLE001 — repo missing / network / auth
        return {"exists": False, "repo_id": repo_id, "error": f"{type(e).__name__}: {e}"}

    siblings = info.siblings or []
    total_size = sum(s.size for s in siblings if s.size)
    remote_episodes = None
    remote_frames = None
    remote_fps = None
    try:
        if repo_type != "dataset":
            raise RuntimeError("episode counts are a dataset notion")
        import json as _json
        from pathlib import Path

        from huggingface_hub import hf_hub_download

        info_path = hf_hub_download(repo_id, "meta/info.json", repo_type="dataset")
        remote_info = _json.loads(Path(info_path).read_text())
        remote_episodes = remote_info.get("total_episodes")
        remote_frames = remote_info.get("total_frames")
        remote_fps = remote_info.get("fps")
    except Exception:  # noqa: BLE001 — best-effort enrichment
        pass

    return {
        "exists": True,
        "repo_id": info.id,
        "private": info.private,
        "last_modified": str(info.last_modified) if info.last_modified else None,
        "downloads": info.downloads,
        "files": len(siblings),
        "total_size_mb": round(total_size / 1e6, 1),
        "sha": info.sha[:12] if info.sha else None,
        "total_episodes": remote_episodes,
        "total_frames": remote_frames,
        "fps": remote_fps,
    }


def list_hub_jobs(app_state: AppState) -> dict[str, Any]:
    """All Hub transfers known to the server, newest-first.

    Reads ``app_state.hub_jobs``, refreshes pending/running jobs
    from each worker's progress JSON, and opportunistically GCs
    terminal jobs older than 30 minutes. Same code path as the
    existing ``GET /api/hub/jobs`` route the GUI's Transfers tray
    polls.

    Returns ``{"jobs": [...], "total": N, "active": N_active}`` —
    the ``total`` / ``active`` summary is the outcome-transparent
    bit so the agent doesn't have to count the array.
    """
    # Lazy import to avoid a circular gui.api.datasets → _hub_core cycle.
    from lerobot.gui.api.datasets import (
        _escalate_cancel_if_overdue,
        _fail_if_heartbeat_dead,
        _refresh_progress_from_file,
    )
    from lerobot.gui.hub_jobs import ACTIVE_STATUSES, reap_if_dead

    app_state.gc_finished_hub_jobs()
    for j in app_state.hub_jobs.values():
        if j.status in ACTIVE_STATUSES:
            _refresh_progress_from_file(j)
            # A worker that swallowed SIGTERM is killed here rather than on
            # a second user click — polling is what makes cancel eventually
            # terminate on its own.
            if not _escalate_cancel_if_overdue(j):
                _fail_if_heartbeat_dead(j)
        elif j.pid is not None and reap_if_dead(j.pid):
            # Terminal job: reap the child if it hasn't been already. The
            # spawn path drops its Popen, so without this every completed
            # transfer leaves a zombie for the life of the server session.
            # Clearing the pid stops us re-waiting on it every poll.
            j.pid = None

    jobs = sorted(
        (j.to_dict() for j in app_state.hub_jobs.values()),
        key=lambda d: d["started_at"],
        reverse=True,
    )
    active = sum(1 for d in jobs if d["status"] in ACTIVE_STATUSES)
    return {"jobs": jobs, "total": len(jobs), "active": active}


def list_hub_history(*, limit: int = 20) -> dict[str, Any]:
    """Terminal outcomes of past transfers, newest first.

    Answers the question the live job list cannot: *did my upload land?*
    ``list_hub_jobs`` drops a job 30 minutes after it finishes and loses
    everything on a server restart, so a long upload can complete and leave
    no trace. This reads the durable record instead.

    Returns ``{"transfers": [...], "total": N}``.
    """
    from lerobot.gui.hub_history import read_recent

    transfers = read_recent(limit=limit)
    return {"transfers": transfers, "total": len(transfers)}


def get_job_progress(app_state: AppState, job_id: str) -> dict[str, Any]:
    """Snapshot of one Hub job's state + latest progress merge.

    Raises ``HubJobNotFoundError`` if ``job_id`` is unknown. For
    active jobs, refreshes from the worker's progress file before
    returning so the snapshot is current at call time.
    """
    from lerobot.gui.api.datasets import (
        _escalate_cancel_if_overdue,
        _fail_if_heartbeat_dead,
        _refresh_progress_from_file,
    )
    from lerobot.gui.hub_jobs import ACTIVE_STATUSES

    job = app_state.hub_jobs.get(job_id)
    if job is None:
        raise HubJobNotFoundError(f"Hub job not found: {job_id}")
    if job.status in ACTIVE_STATUSES:
        _refresh_progress_from_file(job)
        if not _escalate_cancel_if_overdue(job):
            _fail_if_heartbeat_dead(job)
    return job.to_dict()
