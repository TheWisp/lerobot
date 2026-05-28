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
"""Subprocess worker that owns one Hub transfer end-to-end.

Spawned by the GUI server, never directly by the user. Reads its config
from the ``LEROBOT_HUB_WORKER_CONFIG`` env var (a JSON blob) at startup
and writes progress + lifecycle state to per-job files the server polls.

The worker is the only place that calls ``huggingface_hub``'s helpers
(`upload_large_folder`, `snapshot_download`, etc.). The server speaks
only to the worker via file IPC and POSIX signals — never to HF directly
for transfer-bound operations.

For the full design see :doc:`gui/docs/hub_transfers.md`.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import re
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from lerobot.gui.hub_jobs import (
    PROGRESS_WRITE_INTERVAL_S,
    JobConfig,
    JobPaths,
    atomic_write_json,
    classify_error,
    pid_file_payload,
)

logger = logging.getLogger(__name__)


# ── Worker state ────────────────────────────────────────────────────────────


class _WorkerState:
    """In-memory state of one running worker. Owns the progress file write.

    A single instance per worker process. Threadsafe writes through
    ``_lock`` since the milestone-extraction thread also mutates fields.
    """

    def __init__(self, config: JobConfig, paths: JobPaths) -> None:
        self.config = config
        self.paths = paths
        self.started_at = time.time()
        self.finished_at: float | None = None
        # Mutable progress fields. Read-and-write atomic on CPython for the
        # primitive types we use here; ``_lock`` only protects the multi-field
        # write to disk.
        self.status: str = "pending"  # pending | running | complete | failed | cancelled
        self.stage: str = "starting"
        self.milestone: str = f"Starting {config.direction}"
        self.milestone_at: float = self.started_at
        self.files_total: int = 0
        self.files_done_estimate: int = 0
        self.bytes_total: int = 0
        self.bytes_done_estimate: int = 0
        self.current_file: str | None = None
        self.error: str | None = None
        self.error_class: str | None = None
        # Upload-only.
        self.pr_num: int | None = None
        self.pr_url: str | None = None
        # Cancellation flag set by the SIGTERM handler. The HF calls don't
        # poll this; we check it between pipeline stages where possible.
        self.cancel_requested: bool = False
        self._lock = threading.Lock()
        self._stop_writer = threading.Event()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "job_id": self.config.job_id,
                "dataset_id": self.config.dataset_id,
                "direction": self.config.direction,
                "repo_id": self.config.repo_id,
                "status": self.status,
                "stage": self.stage,
                "milestone": self.milestone,
                "milestone_at": self.milestone_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "files_total": self.files_total,
                "files_done_estimate": self.files_done_estimate,
                "bytes_total": self.bytes_total,
                "bytes_done_estimate": self.bytes_done_estimate,
                "current_file": self.current_file,
                "error": self.error,
                "error_class": self.error_class,
                "pr_num": self.pr_num,
                "pr_url": self.pr_url,
            }

    def set_milestone(self, milestone: str, *, stage: str | None = None) -> None:
        with self._lock:
            self.milestone = milestone
            self.milestone_at = time.time()
            if stage is not None:
                self.stage = stage
        # Flush immediately on a milestone change so the GUI tray sees state
        # transitions without waiting for the next poll-interval tick.
        self.write_progress()

    def write_progress(self) -> None:
        """Atomically write the current snapshot to the progress JSON file."""
        atomic_write_json(self.paths.progress, self.snapshot())

    def start_writer_thread(self) -> threading.Thread:
        """Background thread that flushes the snapshot at the configured rate.

        Milestone transitions also flush directly (see set_milestone), so
        the thread is the "no news is good news" heartbeat — it keeps the
        server's poll seeing fresh-enough timestamps even when nothing
        changed.
        """

        def _run() -> None:
            while not self._stop_writer.wait(PROGRESS_WRITE_INTERVAL_S):
                self.write_progress()

        t = threading.Thread(target=_run, name="hub-worker-progress", daemon=True)
        t.start()
        return t

    def stop_writer_thread(self) -> None:
        self._stop_writer.set()


# ── Milestone extraction from HF stderr ────────────────────────────────────
#
# HF's helpers write progress to stderr via tqdm. We capture stderr,
# extract structured milestones from it, and persist the rest verbatim to
# the per-job log for debugging.
#
# The patterns below are intentionally lossy — if HF's format shifts in a
# future version, our milestone string falls back to "running" and the
# rest of the system keeps working. The log file always has the raw
# stderr regardless of what we successfully matched.


_PATTERNS_UPLOAD = [
    # upload_large_folder progress reports
    (re.compile(r"Hashing file ([\d.]+\s*\w+)/([\d.]+\s*\w+)"), "Hashing files {0} / {1}"),
    (re.compile(r"Pre-uploading file ([\d.]+\s*\w+)/([\d.]+\s*\w+)"), "Pre-uploading {0} / {1}"),
    (re.compile(r"Processing Files\s*\((\d+)\s*/\s*(\d+)\)"), "Processing files {0} / {1}"),
    (
        re.compile(r"New Data Upload\s*:\s*\|.*\|\s*([\d.]+\s*\w+)\s*/\s*([\d.]+\s*\w+)"),
        "Uploading {0} / {1}",
    ),
    # HF's upload helpers also emit "Fetching N files" during the pre-upload
    # remote-state probe (before any byte is sent). Calling that "Downloading"
    # while the user is uploading is confusing — relabel as a check pass.
    (re.compile(r"Fetching\s+(\d+)\s+files:.*?(\d+)/\1"), "Checking remote files {1} / {0}"),
    (re.compile(r"Committing files\s*\((\d+)\s*/\s*(\d+)\)"), "Committing {0} / {1}"),
]

_PATTERNS_DOWNLOAD = [
    # snapshot_download progress
    (re.compile(r"Fetching\s+(\d+)\s+files:.*?(\d+)/\1"), "Downloading {1} / {0} files"),
    (re.compile(r"(\d+)%\|"), "Downloading {0}%"),
]


def extract_milestone(line: str, direction: str) -> str | None:
    """Match a tqdm/HF-stderr line against known patterns; return milestone or None.

    Best-effort parsing. Lines that don't match any pattern are kept in the
    log file but produce no milestone update.
    """
    patterns = _PATTERNS_UPLOAD if direction == "upload" else _PATTERNS_DOWNLOAD
    for pat, template in patterns:
        m = pat.search(line)
        if m:
            return template.format(*m.groups())
    return None


_SEPARATORS = b"\r\n"


def stream_stderr_to_log_and_state(
    pipe: io.BufferedIOBase,
    log_path: Path,
    state: _WorkerState,
) -> None:
    """Read the HF helpers' stderr stream, splitting on ``\\r`` and ``\\n``.

    HF / tqdm use ``\\r`` carriage returns to overwrite progress lines on
    the same terminal line. A line-buffered reader misses every progress
    tick, so we read in chunks and split on both separators ourselves.

    Each parsed "line" is:
      (1) appended to the per-job log file verbatim (post-pended ``\\n``),
      (2) optionally matched against milestone patterns to update state.

    Exits when ``pipe`` returns EOF.
    """
    buf = bytearray()
    with open(log_path, "ab") as log_f:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                break
            buf.extend(chunk)
            # Drain whole records out of the buffer; anything past the last
            # separator stays for the next chunk.
            start = 0
            for i, byte in enumerate(buf):
                if byte in _SEPARATORS:
                    if i > start:
                        line = bytes(buf[start:i]).decode("utf-8", errors="replace")
                        log_f.write(line.encode("utf-8", errors="replace") + b"\n")
                        milestone = extract_milestone(line, state.config.direction)
                        if milestone is not None:
                            state.set_milestone(milestone)
                    start = i + 1
            if start:
                log_f.flush()
                del buf[:start]


# ── Signal handling ─────────────────────────────────────────────────────────


def _install_signal_handlers(state: _WorkerState) -> None:
    def _on_sigterm(signum, frame):  # noqa: ARG001
        # Set the flag and let the main thread observe it at the next
        # pipeline-stage boundary. Don't try to abort the HF call mid-flight;
        # the server's SIGKILL escalation will handle that case if needed.
        #
        # Crucially: do NOT acquire ``state._lock`` here. Python delivers
        # signals synchronously on the main thread between bytecodes; if
        # the main thread is already inside ``set_milestone`` (or any
        # other ``with state._lock`` block) when SIGTERM arrives, this
        # handler would re-acquire a non-reentrant lock held by the same
        # thread → self-deadlock, wedging the worker until SIGKILL.
        #
        # The individual field assignments below are atomic in CPython
        # (refcount manipulation + pointer write), so a snapshot taken
        # mid-handler may see "cancelling" milestone with the previous
        # timestamp (or vice versa). That transient inconsistency is
        # strictly preferable to a frozen worker, and the next writer-
        # thread tick (within PROGRESS_WRITE_INTERVAL_S) will write a
        # coherent snapshot anyway.
        state.cancel_requested = True
        state.milestone = "cancelling"
        state.milestone_at = time.time()

    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)

    # On Linux, opt-in to parent-death signal so we don't outlive the GUI
    # server. macOS doesn't have this; that's part of the broader orphan-
    # subprocess work tracked separately.
    if sys.platform == "linux":
        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            PR_SET_PDEATHSIG = 1  # noqa: N806 — POSIX constant
            libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
        except Exception:  # noqa: BLE001 — best effort
            pass


# ── PID file lifecycle ──────────────────────────────────────────────────────


def _write_pid_file(paths: JobPaths) -> None:
    """Record our identity so the server can later verify we're alive.

    Pre: ``paths.pid``'s parent directory exists.
    Post: ``paths.pid`` contains a JSON payload with ``pid``, ``start_time``,
    ``started_at`` — the server uses ``(pid, start_time)`` to disambiguate
    against a recycled PID.
    """
    atomic_write_json(paths.pid, pid_file_payload(os.getpid()))


def _cleanup_pid_file(paths: JobPaths) -> None:
    with contextlib.suppress(Exception):
        # safe-destruct: our own PID file at terminal-state exit
        paths.pid.unlink(missing_ok=True)


# ── Direction-specific pipelines ────────────────────────────────────────────


def _do_upload(state: _WorkerState) -> None:
    """Upload pipeline: create_pr → upload_large_folder → squash → merge.

    Each step is its own ``stage`` value the server can render. Errors at
    any step raise; the outer ``main`` catches and classifies.

    Squash-failure fallback: if ``super_squash_history`` times out or
    otherwise fails, we proceed straight to merge. The fast-forward merge
    of an unsquashed PR branch is still atomic on ``main``; only the
    commit-history hygiene degrades.
    """
    from huggingface_hub import HfApi, upload_large_folder

    cfg = state.config
    api = HfApi()

    # ── Stage 1: ensure repo exists, create or reuse a draft PR ────────
    state.set_milestone("Preparing PR", stage="preparing")
    api.create_repo(
        repo_id=cfg.repo_id,
        repo_type=cfg.repo_type,
        exist_ok=True,
        private=cfg.private,
    )

    pr_num: int
    if cfg.reuse_pr_num is not None:
        # Resume path: the server told us to use an existing draft PR.
        state.set_milestone(f"Resuming PR #{cfg.reuse_pr_num}", stage="preparing")
        pr_num = cfg.reuse_pr_num
    else:
        pr = api.create_pull_request(
            repo_id=cfg.repo_id,
            repo_type=cfg.repo_type,
            title=cfg.commit_message or f"Upload from LeRobot GUI ({cfg.dataset_id})",
            description="Pending upload via LeRobot GUI Hub transfers.",
        )
        pr_num = pr.num
    state.pr_num = pr_num
    state.pr_url = f"https://huggingface.co/datasets/{cfg.repo_id}/discussions/{pr_num}"
    state.write_progress()

    if state.cancel_requested:
        raise InterruptedError("cancel requested before upload")

    # ── Stage 2: upload to the PR branch ───────────────────────────────
    state.set_milestone("Uploading files", stage="uploading")
    revision = f"refs/pr/{pr_num}"
    # `upload_large_folder` writes report text to stdout every
    # `print_report_every` seconds. We don't need that — we parse the
    # stderr/tqdm stream — but disabling reduces double-noise in the log.
    upload_large_folder(
        repo_id=cfg.repo_id,
        repo_type=cfg.repo_type,
        folder_path=str(cfg.local_path),
        revision=revision,
        allow_patterns=cfg.allow_patterns,
        ignore_patterns=list(cfg.ignore_patterns) if cfg.ignore_patterns else None,
        print_report=False,
    )

    if state.cancel_requested:
        raise InterruptedError("cancel requested after upload")

    # ── Stage 3: squash is currently disabled ─────────────────────────
    # super_squash_history rewrites the PR branch in a way that doesn't
    # always preserve fast-forward-ability to main — observed on
    # second-upload-to-same-repo where HF reports "merge conflicts" after
    # squash. The design's squash-failure-fallback path covers exactly this
    # case (merge unsquashed); we're just always taking that path until the
    # HF interaction is understood. Multi-commit main history is acceptable
    # per the design.
    #
    # When the right API usage is figured out, the squash call goes back
    # here gated by a JobConfig flag.
    squash_ok = False

    # ── Stage 4: move PR out of draft and merge ───────────────────────
    state.set_milestone("Merging PR", stage="merging")
    api.change_discussion_status(
        repo_id=cfg.repo_id,
        repo_type=cfg.repo_type,
        discussion_num=pr_num,
        new_status="open",
    )
    api.merge_pull_request(
        repo_id=cfg.repo_id,
        repo_type=cfg.repo_type,
        discussion_num=pr_num,
    )

    state.set_milestone(
        "Upload complete" + ("" if squash_ok else " (merged unsquashed)"),
        stage="done",
    )


def _do_download(state: _WorkerState) -> None:
    """Download pipeline: snapshot_download into dataset.root, no temp+swap."""
    from huggingface_hub import snapshot_download

    cfg = state.config
    state.set_milestone("Downloading files", stage="downloading")
    snapshot_download(
        repo_id=cfg.repo_id,
        repo_type=cfg.repo_type,
        local_dir=str(cfg.local_path),
        allow_patterns=cfg.allow_patterns,
        ignore_patterns=list(cfg.ignore_patterns) if cfg.ignore_patterns else None,
        max_workers=8,
    )
    state.set_milestone("Download complete", stage="done")


# ── Main entry point ────────────────────────────────────────────────────────


def _load_config() -> tuple[JobConfig, JobPaths]:
    raw = os.environ.get("LEROBOT_HUB_WORKER_CONFIG")
    if raw is None:
        print("hub_worker: missing LEROBOT_HUB_WORKER_CONFIG env var", file=sys.stderr)
        sys.exit(2)
    cfg = JobConfig.from_json(raw)
    paths = JobPaths.for_job(cfg.job_id, cfg.jobs_dir)
    paths.jobs_dir.mkdir(parents=True, exist_ok=True)
    return cfg, paths


def main() -> int:
    cfg, paths = _load_config()
    state = _WorkerState(cfg, paths)
    _install_signal_handlers(state)
    _write_pid_file(paths)

    state.status = "running"
    state.write_progress()
    writer = state.start_writer_thread()

    # Redirect HF's stderr through our reader-thread so we can extract
    # milestones AND keep the verbatim text in the per-job log. We replace
    # the worker's own stderr fd to point at a writable pipe; a background
    # thread drains the read end.
    r_fd, w_fd = os.pipe()
    original_stderr_fd = os.dup(2)
    os.dup2(w_fd, 2)
    os.close(w_fd)
    sys.stderr = os.fdopen(2, "w", buffering=1)  # line-buffered text wrapper

    reader_thread = threading.Thread(
        target=stream_stderr_to_log_and_state,
        args=(os.fdopen(r_fd, "rb", buffering=0), paths.log, state),
        daemon=True,
        name="hub-worker-stderr-reader",
    )
    reader_thread.start()

    rc = 0
    try:
        if cfg.direction == "upload":
            _do_upload(state)
        elif cfg.direction == "download":
            _do_download(state)
        else:  # pragma: no cover — guarded by JobConfig.__post_init__
            raise ValueError(f"unknown direction: {cfg.direction!r}")
        state.status = "complete"
    except InterruptedError:
        # Raised on our cancel path between pipeline stages. Local
        # `.cache/.huggingface/` + the draft PR remain intact for resume.
        state.status = "cancelled"
        state.error = "Cancelled by user"
        state.error_class = "cancelled"
    except KeyboardInterrupt:
        # SIGINT delivered by tests or interactive run.
        state.status = "cancelled"
        state.error = "Cancelled (SIGINT)"
        state.error_class = "cancelled"
    except Exception as e:  # noqa: BLE001 — terminal-error catch is intentional
        state.status = "failed"
        state.error = f"{type(e).__name__}: {e}"
        state.error_class = classify_error(e)
        # Append full traceback to the log for post-mortem.
        try:
            paths.log.parent.mkdir(parents=True, exist_ok=True)
            with open(paths.log, "a") as f:
                f.write("\n--- terminal exception ---\n")
                traceback.print_exc(file=f)
        except Exception:  # noqa: BLE001
            pass
        rc = 1
    finally:
        state.finished_at = time.time()
        # Restore stderr first so any late writes (from the writer thread's
        # final flush, for instance) don't deadlock against a half-closed pipe.
        try:
            os.dup2(original_stderr_fd, 2)
            os.close(original_stderr_fd)
        except Exception:  # noqa: BLE001
            pass
        # Stop + join the writer thread BEFORE the final write_progress so
        # two threads don't race on the same .tmp path under atomic_write_json
        # (writer-thread tick + main-thread final write would otherwise
        # interleave, potentially leaving the progress file with garbage
        # content or stale-snapshot content).
        state.stop_writer_thread()
        writer.join(timeout=1.0)
        # Final state write so the server sees terminal status.
        state.write_progress()
        # Reader is daemon; it exits when stderr pipe closes (above).
        _cleanup_pid_file(paths)

    return rc


if __name__ == "__main__":
    sys.exit(main())
