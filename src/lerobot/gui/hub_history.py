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
"""Durable record of how each Hub transfer ended.

The in-memory job registry answers "what is happening now". It cannot answer
"did my upload land", because it is dropped 30 minutes after a job finishes
and erased outright when the server restarts. A real 8.4 GB upload completed,
was collected, and left the user with no way to tell success from failure.

This module is the answer to that question: one JSON line appended per
terminal outcome, in a file the server does not own.

Two properties do the work:

* **Append-only, one line per outcome.** A single ``write()`` of a short line
  to a file opened ``O_APPEND`` is atomic on POSIX, so concurrent writers —
  and there are several, in different processes — interleave whole lines
  rather than corrupting each other. This is deliberately *not* the
  read-modify-write shape that raced in ``atomic_write_json``.

* **Last line per ``job_id`` wins on read.** Both the worker and the server
  can record the same job: the worker at its terminal write, the server when
  it force-terminates a worker that cannot record its own ending. Rather than
  coordinating who writes, both do, and the reader keeps the last. The server
  writes later than the worker in every case where both write, and the server
  is the authority in exactly those cases.

See :doc:`gui/docs/hub_transfers.md`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Not under ~/.cache: this outlives the per-job IPC files by design, and a
# cache directory is a place things are allowed to disappear from.
HISTORY_PATH = Path.home() / ".config" / "lerobot" / "gui" / "hub_transfers.jsonl"

# Overrides HISTORY_PATH. The worker is a separate process, so a test cannot
# monkeypatch its module — the override has to survive the process boundary,
# and an env var is the only channel the spawn path already carries.
HISTORY_PATH_ENV = "LEROBOT_HUB_HISTORY_PATH"


def history_path() -> Path:
    """The file outcomes are appended to. Env var wins; read at call time."""
    override = os.environ.get(HISTORY_PATH_ENV)
    return Path(override) if override else HISTORY_PATH


# Read cap. The file is a few hundred bytes per transfer; this is about
# bounding work per poll, not about the file getting large.
DEFAULT_READ_LIMIT = 50

# `prune()` keeps the newest this many entries. Called at server startup, not
# on append — see prune(). At ~400 bytes per record that is under a megabyte,
# and a heavy user reaches it in years.
MAX_LINES = 2000


def _record_from_job(job: Any, *, now: float | None = None) -> dict[str, Any]:
    """Build the outcome record for a terminal ``HubJobState``.

    Pre: ``job.status`` is terminal. Post: the returned dict is JSON-safe and
    contains no field the reader has to guess at.
    """
    finished = job.finished_at or (time.time() if now is None else now)
    started = job.started_at or finished
    return {
        "ts": finished,
        "job_id": job.job_id,
        "direction": job.direction,
        "repo_id": job.repo_id,
        "repo_type": getattr(job, "repo_type", "dataset"),
        "dataset_id": job.dataset_id,
        "status": job.status,
        "error": job.error,
        "error_class": job.error_class,
        "bytes_total": job.bytes_total,
        "bytes_done_estimate": job.bytes_done_estimate,
        "files_total": job.files_total,
        "duration_s": max(0.0, finished - started),
        "pr_num": job.pr_num,
        "pr_url": job.pr_url,
        "disable_xet": bool(getattr(job, "disable_xet", False)),
    }


def append_outcome(record: dict[str, Any], *, path: Path | None = None) -> bool:
    """Append one terminal-outcome record. Returns True if it was written.

    Never raises: a transfer must not fail because its bookkeeping did. A
    failure here is logged and the caller continues — the worst case is a
    missing history entry, which is what we had before this existed.

    Pre: ``record`` is JSON-serialisable and contains ``job_id``.
    Post: on True, exactly one line was appended.
    """
    target = history_path() if path is None else path
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str) + "\n"
        # One write of one short line to an O_APPEND handle. Concurrent
        # writers in other processes cannot interleave within it.
        with open(target, "a", encoding="utf-8") as f:
            f.write(line)
        return True
    except Exception:  # noqa: BLE001 — bookkeeping must not break a transfer
        logger.warning("could not record transfer outcome", exc_info=True)
        return False


def prune(path: Path | None = None, *, max_lines: int | None = None) -> int:
    """Keep the newest ``max_lines`` entries. Returns the number dropped.

    Read-modify-write, and therefore **not** safe to call while workers may
    be appending: a line written between the read and the ``os.replace``
    is lost. That is the shape this module exists to avoid, so it is kept
    off the append path entirely and called once at server startup, where
    nothing of ours is mid-transfer. Bounding growth is not urgent enough
    to justify racing the record it is bounding.

    ``MAX_LINES`` is read at call time rather than bound as a default
    argument: a default is evaluated once at import, so the cap could never
    be changed afterwards — not by a caller, and not by a test trying to
    prove the pruning works at all.
    """
    target = history_path() if path is None else path
    limit = MAX_LINES if max_lines is None else max_lines
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        if len(lines) <= limit:
            return 0
        tmp = target.with_name(f"{target.name}.{os.getpid()}.prune")
        tmp.write_text("".join(lines[-limit:]), encoding="utf-8")
        os.replace(tmp, target)
        return len(lines) - limit
    except FileNotFoundError:
        return 0
    except Exception:  # noqa: BLE001
        logger.warning("could not prune transfer history", exc_info=True)
        return 0


def read_recent(*, limit: int = DEFAULT_READ_LIMIT, path: Path | None = None) -> list[dict[str, Any]]:
    """Most recent terminal outcomes, newest first, one entry per job.

    Malformed lines are skipped rather than failing the read: this file is
    appended to by more than one process, and a torn line from a crash must
    not cost the user the rest of their history.

    Post: at most ``limit`` entries; no two share a ``job_id``; sorted by
    ``ts`` descending.
    """
    target = history_path() if path is None else path
    try:
        raw = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return []
    except OSError:
        logger.warning("could not read transfer history", exc_info=True)
        return []

    by_job: dict[str, dict[str, Any]] = {}
    for line in raw:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        job_id = rec.get("job_id")
        if not isinstance(job_id, str):
            continue
        # `ts` is sorted on below, and a line with a string timestamp would
        # raise TypeError there — outside any try, so a single malformed
        # entry would 500 the history endpoint permanently. That is exactly
        # the whole-history loss this function promises cannot happen, so
        # the type is checked here rather than trusted.
        if not isinstance(rec.get("ts"), (int, float)) or isinstance(rec.get("ts"), bool):
            rec["ts"] = 0.0
        # Later line wins: the server records after the worker in every case
        # where both do, and in those cases the server is the authority.
        by_job[job_id] = rec

    out = sorted(by_job.values(), key=lambda r: r.get("ts") or 0.0, reverse=True)
    return out[:limit]
