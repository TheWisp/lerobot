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
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.gui.hub_jobs import (
    CANCEL_GRACE_S,
    DEFAULT_UPLOAD_IGNORES,
    PROGRESS_WRITE_INTERVAL_S,
    JobConfig,
    JobPaths,
    atomic_write_json,
    classify_error,
    enumerate_upload_files,
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
        self.transfer_rate_bps: float = 0.0
        self.last_progress_at: float = 0.0
        self.error: str | None = None
        self.error_class: str | None = None
        # Upload-only.
        self.pr_num: int | None = None
        self.pr_url: str | None = None
        # Cancellation flag set by the SIGTERM handler. The HF calls don't
        # poll this; we check it between pipeline stages where possible,
        # and the cancel watchdog force-exits if they never return.
        self.cancel_requested: bool = False
        # Once set, milestone updates from the output-reader thread are
        # ignored. Without it, the next tqdm tick (milliseconds away)
        # overwrites "Cancelling…" with a routine progress string and the
        # user sees the cancel silently un-happen.
        self.milestone_locked: bool = False
        # Set by the SIGTERM handler; awaited by the cancel watchdog.
        self.cancel_event = threading.Event()
        # Set by main()'s finally block so the watchdog can tell a clean
        # unwind from a wedged one and skip the force-exit.
        self.exit_event = threading.Event()
        self._lock = threading.Lock()
        self._stop_writer = threading.Event()
        # Populated by start_writer_thread; the cancel watchdog joins it
        # before writing terminal state so the two don't race.
        self.writer_thread: threading.Thread | None = None
        # Set by the output reader once it starts aggregating. Held here so
        # the periodic snapshot can re-derive the rate on every tick rather
        # than only when bytes move — otherwise a stalled transfer keeps
        # reporting the last healthy throughput indefinitely.
        self.progress: TransferProgress | None = None

    def snapshot(self) -> dict[str, Any]:
        # Re-derive outside the lock: the aggregator has its own state and
        # taking our lock here would serialise against the output reader.
        if self.progress is not None:
            self.transfer_rate_bps = self.progress.rate_bps()
        with self._lock:
            return {
                "job_id": self.config.job_id,
                "dataset_id": self.config.dataset_id,
                "direction": self.config.direction,
                "repo_id": self.config.repo_id,
                # Carried into the durable record, not just held in memory: the
                # tray's history reads this file after the in-memory job is
                # dropped, and without it a past model transfer renders under
                # the dataset namespace and links to a URL that does not exist.
                "repo_type": self.config.repo_type,
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
                "transfer_rate_bps": self.transfer_rate_bps,
                "last_progress_at": self.last_progress_at,
                "error": self.error,
                "error_class": self.error_class,
                "pr_num": self.pr_num,
                "pr_url": self.pr_url,
            }

    def set_milestone(self, milestone: str, *, stage: str | None = None) -> None:
        if self.milestone_locked:
            # Cancelling: the pinned milestone outranks any routine update.
            return
        with self._lock:
            # Re-check under the lock. The flag is set from the SIGTERM
            # handler, which can land between the check above and this
            # assignment — and a routine string that wins that race is
            # never corrected, because every later update is suppressed by
            # the flag. The tray would then read "Processing files 0 / 1"
            # for the whole cancellation: the exact symptom the pin exists
            # to prevent, through a narrower window.
            if self.milestone_locked:
                return
            unchanged = self.milestone == milestone and (stage is None or self.stage == stage)
            if unchanged:
                # HF redraws its bars several times a second, and the
                # milestone string derived from them is usually identical
                # across a whole file. Re-writing the progress file for
                # each of those is pure contention against the writer
                # thread on the same path, for no new information.
                return
            self.milestone = milestone
            self.milestone_at = time.time()
            if stage is not None:
                self.stage = stage
        # Flush immediately on a milestone change so the GUI tray sees state
        # transitions without waiting for the next poll-interval tick.
        self.write_progress()

    def record_bytes(
        self,
        *,
        bytes_done: int,
        files_done: int,
        rate_bps: float,
        at: float,
        current_file: str | None,
    ) -> None:
        """Publish a byte-progress observation from the output reader.

        Pre: ``bytes_done`` is non-decreasing across calls (guaranteed by
        :class:`TransferProgress`). Post: the next ``snapshot()`` reflects
        these counters. Deliberately does not flush to disk — the writer
        thread picks it up within ``PROGRESS_WRITE_INTERVAL_S``, and byte
        observations arrive far too often to write on each one.
        """
        with self._lock:
            self.bytes_done_estimate = bytes_done
            self.files_done_estimate = files_done
            self.transfer_rate_bps = rate_bps
            self.last_progress_at = at
            self.current_file = current_file

    def mark_complete(self) -> None:
        """Enter the ``complete`` state, and make the counters say so.

        ``bytes_done_estimate`` / ``files_done_estimate`` are documented *lower
        bounds*, recovered from the progress bars huggingface_hub prints. Files
        the server already has never produce a bar at all, so on a successful
        transfer the estimate stops well short of the total: a real 8.4 GB
        upload finished reading ``3.99 GB / 8.41 GB, 29/84 files`` — a bar
        frozen near 47% on a transfer that fully succeeded, which reads as a
        partial upload and is exactly the ambiguity the durable outcome record
        exists to remove.

        Success is the one moment the true values are known without estimating:
        everything the worker set out to transfer is on the remote. The
        lower-bound semantics stay for in-flight display, where a number that
        undercounts is honest and one that guesses is not.
        """
        with self._lock:
            self.status = "complete"
            self.bytes_done_estimate = self.bytes_total
            self.files_done_estimate = self.files_total

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
                try:
                    self.write_progress()
                except Exception:  # noqa: BLE001 — heartbeat must outlive one bad write
                    # This thread is the server's only liveness signal for a
                    # transfer. Letting an exception end it means the progress
                    # file silently freezes and a healthy transfer reads as
                    # hung — the exact production failure that motivated the
                    # unique-temp-name fix in atomic_write_json. A transient
                    # I/O error must cost one tick, not the whole heartbeat.
                    logger.warning("progress write failed; continuing", exc_info=True)

        t = threading.Thread(target=_run, name="hub-worker-progress", daemon=True)
        t.start()
        self.writer_thread = t
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


# ── Byte-level progress extraction ─────────────────────────────────────────
#
# The milestone strings above are file *counts*, and for a Xet-backed
# upload the count moves once per completed file — which for a dataset of
# a few large mp4s means the tray shows a constant "Processing files 0 / 1"
# for the entire multi-GB transfer. Users read that (correctly) as "hung".
#
# The bytes are right there on the same lines. `XetProgressReporter`
# renders every bar with an explicit
#   bar_format="{l_bar}{bar}| {n_fmt:>5}B / {total_fmt:>5}B{postfix:>12}"
# so each rendered line carries "<done>B / <total>B" and, for the two
# summary bars, a ", <rate>B/s" postfix. We parse those and aggregate.
#
# Two structural facts about that output drive the design here:
#
#  1. `upload_large_folder` runs several HF worker threads, each with its
#     OWN XetProgressReporter, all writing to the same captured stream.
#     So "Processing Files (…)" lines from unrelated reporters interleave
#     with unrelated totals. Trusting the most recent one makes the number
#     jump around. Per-file bars are keyed by filename, so aggregating
#     those is stable across reporters.
#  2. tqdm reuses a per-file bar slot for a new file once the previous one
#     completes. Keying on the rendered description (not the bar slot)
#     means each file keeps its own last-known counter.
#
# Everything here is a documented *lower bound* on bytes processed, never
# an overestimate: a file we've never seen a bar for contributes 0.


# tqdm's format_sizeof prefixes, in ascending order. unit_divisor is 1000
# for HF's reporter, so these are decimal (kB = 1000 B), matching what the
# text on screen means.
_SIZE_MULTIPLIER = {
    "": 1,
    "k": 1_000,
    "K": 1_000,  # tqdm's default format_sizeof emits "K", the Xet one "k"
    "M": 1_000_000,
    "G": 1_000_000_000,
    "T": 1_000_000_000_000,
    "P": 1_000_000_000_000_000,
}

# Xet's reporter sets an explicit bar_format, so its bars render as
# "<desc>: <pct>%|<bar>| <done>B / <total>B[, <rate>B/s]".
_BAR_LINE = re.compile(
    r"^(?P<desc>.*?):\s*\d{1,3}%\|[^|]*\|"
    r"\s*(?P<done>[\d.]+)\s*(?P<done_unit>[kMGTP]?)B"
    r"\s*/\s*(?P<total>[\d.]+)\s*(?P<total_unit>[kMGTP]?)B"
    r"(?:\s*,\s*(?P<rate>[\d.]+)\s*(?P<rate_unit>[kMGTP]?)B/s)?"
)

# The classic (non-Xet) LFS upload path uses tqdm's *default* bar_format
# instead, which prints no "B" after the numbers:
#   "model.bin:  45%|████      | 1.05G/2.34G [00:12<00:15, 84.2MB/s]"
#
# This path matters more than it looks: HF_HUB_DISABLE_XET=1 is the
# supported way to avoid the Xet CAS endpoints, and on a link where those
# endpoints stall it is the difference between an upload completing and not.
# Without this pattern the progress readout is blank for exactly the users
# who had to reach for that workaround.
#
# A unit suffix on the total is required. tqdm renders plain counters the
# same way ("Fetching 84 files: 100%|...| 84/84"), and matching those would
# feed file counts into a byte aggregator. Byte bars below 1 kB would also
# be skipped by that rule, which is harmless — they contribute nothing.
_BAR_LINE_DEFAULT_FMT = re.compile(
    r"^(?P<desc>.*?):\s*\d{1,3}%\|[^|]*\|"
    r"\s*(?P<done>[\d.]+)(?P<done_unit>[kKMGTP]?)"
    r"\s*/\s*(?P<total>[\d.]+)(?P<total_unit>[kKMGTP])"
    r"(?:\s*\[.*?,\s*(?P<rate>[\d.]+)(?P<rate_unit>[kKMGTP]?)B/s)?"
)

# Descriptions belonging to the two whole-transfer summary bars rather than
# to an individual file.
_SUMMARY_DESCS = ("Processing Files", "New Data Upload")


def _parse_size(number: str, unit: str) -> int:
    """Convert a tqdm-rendered size such as ``("1.05", "M")`` to bytes.

    Pre: ``number`` parses as a float and ``unit`` is a key of
    ``_SIZE_MULTIPLIER`` (both guaranteed by ``_BAR_LINE``'s character
    classes). Post: return value is >= 0.
    """
    value = int(float(number) * _SIZE_MULTIPLIER[unit])
    assert value >= 0, f"negative size from {number!r}{unit!r}"
    return value


@dataclass(frozen=True)
class BarSample:
    """One parsed tqdm bar line."""

    desc: str
    done: int
    total: int
    rate_bps: float | None
    is_summary: bool


def parse_bar_line(line: str) -> BarSample | None:
    """Parse one tqdm progress line into a :class:`BarSample`, else None.

    Handles both the Xet reporter's explicit bar_format and tqdm's default
    one (the classic LFS upload path). Lines that aren't tqdm bars — log
    text, tracebacks, HF's own prints, plain file counters — return None
    and are simply not counted.
    """
    cleaned = line.strip("\x00").replace("\x1b[A", "").rstrip()
    m = _BAR_LINE.match(cleaned) or _BAR_LINE_DEFAULT_FMT.match(cleaned)
    if m is None:
        return None
    desc = m.group("desc").strip()
    # Truncated per-file descriptions are rendered with a leading "...".
    stripped = desc.lstrip(".")
    rate = m.group("rate")
    return BarSample(
        desc=desc,
        done=_parse_size(m.group("done"), m.group("done_unit")),
        total=_parse_size(m.group("total"), m.group("total_unit")),
        rate_bps=(_parse_size(rate, m.group("rate_unit")) if rate else None),
        is_summary=any(stripped.startswith(d) for d in _SUMMARY_DESCS),
    )


class TransferProgress:
    """Aggregates tqdm bar lines into monotonic byte counters + a rate.

    One instance per worker, fed every parsed output line. All counters it
    exposes are lower bounds — they never overshoot the real transfer, so a
    progress bar driven by them cannot run backwards or exceed 100%.

    The rate is computed here rather than taken from HF's postfix because
    that postfix is per-reporter: with several reporters running, the last
    one printed is an arbitrary fraction of the real aggregate throughput.

    Pre: ``window_s`` > 0. Post: ``bytes_done`` is non-decreasing across
    calls for the lifetime of the instance.
    """

    def __init__(self, *, window_s: float = 15.0, clock: Callable[[], float] = time.time) -> None:
        assert window_s > 0, "window_s must be positive"
        self._window_s = window_s
        self._clock = clock
        # desc → (done, total). Counters derived from it are maintained
        # incrementally rather than recomputed: this runs on every line of
        # HF's output, and a dataset can have thousands of files, so an
        # O(files) sweep per line makes the reader thread quadratic. A
        # reader that falls behind stops draining the capture pipe, which
        # back-pressures HF's own writes — the progress readout would
        # become the stall it exists to detect.
        self._per_file: dict[str, tuple[int, int]] = {}
        self._per_file_sum = 0
        self._files_complete = 0
        self._summary_peak = 0
        self._bytes_done = 0
        self._samples: deque[tuple[float, int]] = deque()
        self.last_progress_at: float = 0.0
        self.current_file: str | None = None

    @property
    def bytes_done(self) -> int:
        return self._bytes_done

    @property
    def files_done(self) -> int:
        """Files whose bar reached its total. Undercounts deduped files.

        A file that Xet dedups away entirely never gets a bar, so this can
        sit below the true count. It is a supplementary readout only — the
        progress bar itself is byte-driven precisely so this undercount
        can't make a finished transfer look stuck at 90%.
        """
        return self._files_complete

    def rate_bps(self) -> float:
        """Bytes/s over the trailing window, or 0.0 with too few samples.

        The window ends at *now*, not at the last observation, so the rate
        decays toward zero while a transfer is stalled instead of holding
        the last healthy figure. This matters: uploads through a stalling
        link freeze for tens of seconds at a time, and a readout still
        claiming "540 kB/s" through a 55-second freeze is the same kind of
        confidently-wrong number this whole readout exists to replace.
        """
        if len(self._samples) < 2:
            return 0.0
        (t0, b0), (t1, b1) = self._samples[0], self._samples[-1]
        span = max(t1, self._clock()) - t0
        if span <= 0:
            return 0.0
        return max(0.0, (b1 - b0) / span)

    def feed(self, line: str) -> bool:
        """Consume one output line. Returns True if byte progress advanced.

        Post: on a True return, ``last_progress_at`` has been set to now
        and ``bytes_done`` is strictly greater than before the call.
        """
        sample = parse_bar_line(line)
        if sample is None:
            return False

        if sample.is_summary:
            # A lower bound from whichever reporter printed most recently;
            # kept as a floor so we still show movement in the (unlikely)
            # case that per-file bars are absent.
            self._summary_peak = max(self._summary_peak, sample.done)
        else:
            previous = self._per_file.get(sample.desc)
            was_complete = previous is not None and previous[1] > 0 and previous[0] >= previous[1]
            is_complete = sample.total > 0 and sample.done >= sample.total
            self._per_file_sum += sample.done - (previous[0] if previous else 0)
            self._files_complete += int(is_complete) - int(was_complete)
            self._per_file[sample.desc] = (sample.done, sample.total)
            self.current_file = sample.desc

        aggregate = max(self._per_file_sum, self._summary_peak)
        if aggregate <= self._bytes_done:
            return False

        self._bytes_done = aggregate
        now = self._clock()
        self.last_progress_at = now
        self._samples.append((now, aggregate))
        while len(self._samples) > 2 and now - self._samples[0][0] > self._window_s:
            self._samples.popleft()
        return True


_SEPARATORS = b"\r\n"


def stream_stderr_to_log_and_state(
    pipe: io.BufferedIOBase,
    log_path: Path,
    state: _WorkerState,
) -> None:
    """Read the HF helpers' merged stdout+stderr stream, splitting on ``\\r``
    and ``\\n``.

    The caller dup2s both fd 1 and fd 2 to the same pipe before spawning
    this thread, so the byte stream interleaves stdout and stderr in
    write order. We don't distinguish them at parse time — for our
    purposes the per-job log is the authoritative "what did HF actually
    say" record.

    HF / tqdm use ``\\r`` carriage returns to overwrite progress lines on
    the same terminal line. A line-buffered reader misses every progress
    tick, so we read in chunks and split on both separators ourselves.

    Each parsed "line" is:
      (1) appended to the per-job log file verbatim (post-pended ``\\n``),
      (2) fed to the byte-progress aggregator,
      (3) optionally matched against milestone patterns to update state.

    Exits when ``pipe`` returns EOF.
    """
    buf = bytearray()
    progress = TransferProgress()
    # Publish it so the periodic snapshot can decay the rate during stalls.
    state.progress = progress
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
                        if progress.feed(line):
                            state.record_bytes(
                                bytes_done=progress.bytes_done,
                                files_done=progress.files_done,
                                rate_bps=progress.rate_bps(),
                                at=progress.last_progress_at,
                                current_file=progress.current_file,
                            )
                        milestone = extract_milestone(line, state.config.direction)
                        if milestone is not None:
                            state.set_milestone(milestone)
                    start = i + 1
            if start:
                log_f.flush()
                del buf[:start]


# ── Fail-fast on unrecoverable HF responses ────────────────────────────────
#
# `huggingface_hub.upload_large_folder` catches HTTP errors at the worker
# level and adaptively shrinks its commit batch on any failure. The
# behavior is intentional for 504 Gateway Timeouts (HF's commit endpoint
# times out validating large batches; shrinking + retrying is the right
# move). But the same blanket `except Exception` swallows 429/401/403/404,
# producing a multi-hour, thousand-attempts-per-minute storm that ignores
# HF's own Retry-After header.
#
# This is a known upstream issue (huggingface/huggingface_hub#3325). The
# sanctioned consumer-side fix is to hook the httpx client HF shares via
# `get_session()` and react to the unrecoverable status codes.
#
# We originally tried raising a BaseException subclass from the hook —
# Python's `except Exception` would skip it and the exception would
# propagate to our outer try in main(). What we missed: HF uses raw
# `threading.Thread`, not `concurrent.futures.ThreadPoolExecutor`. When
# a BaseException raises in a plain Thread, the thread dies with a
# traceback to stderr and the MAIN thread never sees it. The process
# stays wedged.
#
# Real fix: have the hook synchronously write terminal failure state to
# the progress JSON and call `os._exit(1)`. Works from any thread, no
# propagation needed, no library cooperation needed. The `_FatalHFError`
# class is retained for use by tests (which can't tolerate os._exit on
# their process) and as a structured exception type for the abort path.
#
# We deliberately do NOT intercept 5xx codes — that's the case the
# adaptive shrink-and-retry was designed for.


class _FatalHFError(BaseException):
    """Structured carrier for a fatal HF response.

    Used by the test harness (which installs a raising hook variant) and
    as the type the production hook passes to the abort helper. Inherits
    from BaseException for symmetry with the design constraint that
    HF's `except Exception` not catch it — relevant only in the raising
    path, since the production path calls os._exit directly.
    """

    def __init__(self, status: int, error_class: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.error_class = error_class
        self.message = message


# Status codes we treat as terminal. 5xx is intentionally absent — the
# library's shrink-and-retry exists for 504 commit timeouts under load
# and intercepting those would defeat the design.
#
# 4xx codes are all "the request will fail again on retry without
# something changing" — and the library doesn't change the request, it
# only shrinks the batch. Smaller-of-the-same-bad-request still fails.
# We've observed 400 storms on LFS-pointer-without-blob state (corrupt
# upload cache after a prior abort) keeping the worker busy long enough
# to also trip 429 — fail-fasting on 400 lets the user see the actual
# cause (HF's body text) and act on it (clear local cache, fix the bad
# files) instead of watching a retry loop chew through their commit cap.
_FATAL_STATUS_TO_CLASS: dict[int, str] = {
    400: "bad_request",  # malformed request — observed on LFS-pointer corruption
    401: "auth",
    403: "auth",  # storage-quota exhausted also surfaces as 403, same retry storm
    404: "auth",  # repo missing / renamed mid-flight; no retry helps
    422: "bad_request",  # unprocessable entity — validation failure equivalent of 400
    429: "rate_limit",
}


# S3 returns 400 for conditions that are purely transient, and LFS uploads
# go straight to S3. Observed in the wild on a link that stalls mid-transfer:
#
#   HTTP 400 from hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com/...UploadPart
#   RequestTimeout: Your socket connection to the server was not read from or
#   written to within the timeout period. Idle connections will be closed.
#
# Nothing about that request is malformed — the connection went quiet and S3
# hung up. `huggingface_hub` retries the part on its own, so fail-fasting
# converts a recoverable hiccup into a dead upload. The 400 entry in the
# fatal table exists for HF's *own* 400s (LFS-pointer corruption), which do
# repeat forever; these are a different animal that happens to share a code.
_TRANSIENT_STORAGE_ERROR_CODES = (
    "RequestTimeout",  # idle connection closed mid-part
    "RequestTimeTooSkewed",  # clock drift; resolves on retry with a fresh signature
    "SlowDown",  # S3 asking for backoff
    "InternalError",
    "ServiceUnavailable",
)


def _is_transient_storage_error(body: str) -> bool:
    """True if an error body names a retryable object-storage condition.

    Matches S3's XML ``<Code>…</Code>`` element rather than a substring of
    the whole body, so prose that merely mentions a timeout doesn't
    accidentally excuse a genuinely fatal response.
    """
    match = re.search(r"<Code>\s*([A-Za-z]+)\s*</Code>", body)
    if match is None:
        return False
    return match.group(1) in _TRANSIENT_STORAGE_ERROR_CODES


def _classify_response(response: Any) -> _FatalHFError | None:
    """Pure function: build the exception we'd raise, or None.

    Separated so the test harness can call it on a synthetic response
    without going through the install path. Reads the body when a fatal
    code matches — HF's actionable error text lives there.
    """
    status = response.status_code
    error_class = _FATAL_STATUS_TO_CLASS.get(status)
    if error_class is None:
        return None
    try:
        body = response.read().decode("utf-8", errors="replace").strip()
    except Exception:  # noqa: BLE001 — best-effort message enrichment
        body = "(body unavailable)"
    if _is_transient_storage_error(body):
        # The storage backend, not HF, and retryable. Killing the transfer
        # here would turn a hiccup the library already knows how to retry
        # into a failed upload.
        return None
    if len(body) > 800:
        body = body[:800] + "…"
    retry_after = response.headers.get("Retry-After")
    url = str(response.url)
    parts = [f"HTTP {status} from {url}"]
    if retry_after:
        parts.append(f"Retry-After: {retry_after}s")
    if body:
        parts.append(body)
    return _FatalHFError(status, error_class, " — ".join(parts))


def _record_outcome(state: _WorkerState) -> None:
    """Append this transfer's ending to the durable history. Never raises.

    Called from every path that reaches a terminal state, including the two
    that ``os._exit`` — those skip ``main``'s ``finally`` entirely, and they
    are the endings a user is most likely to come back asking about.

    The progress JSON is not a substitute: the server deletes it, and it is
    keyed to a job the registry forgets 30 minutes after it finishes.
    """
    from lerobot.gui.hub_history import _record_from_job, append_outcome

    snap = state.snapshot()

    class _JobView:
        """Adapts the worker's snapshot to the shape _record_from_job reads."""

        def __init__(self, d: dict[str, Any], cfg: JobConfig) -> None:
            self.__dict__.update(d)
            self.repo_type = cfg.repo_type
            self.disable_xet = os.environ.get("HF_HUB_DISABLE_XET") == "1"

    with contextlib.suppress(Exception):
        append_outcome(_record_from_job(_JobView(snap, state.config)))


def _abort_to_terminal_state(state: _WorkerState, exc: _FatalHFError) -> None:
    """Persist the failure to disk and kill the worker process.

    Called from inside the httpx hook, which runs on whichever thread
    HF dispatched the failing request from — typically one of HF's
    `_worker_job` threads. We can't rely on the main thread observing
    anything, so we update the on-disk progress JSON synchronously
    (the GUI server reads only this file for state), drop the PID file
    so the server's PID-liveness sweep doesn't overwrite our specific
    error class with a generic "Worker exited without finalizing", and
    os._exit(1).

    os._exit (not sys.exit) because the main thread is wedged in HF's
    library and any cooperative-shutdown path would deadlock against
    its threading state.
    """
    with state._lock:
        state.status = "failed"
        state.error_class = exc.error_class
        state.error = exc.message
        state.finished_at = time.time()
        state.milestone = f"Failed ({exc.error_class})"
    with contextlib.suppress(Exception):  # last-ditch terminal write
        state.write_progress()
    with contextlib.suppress(Exception):
        # safe-destruct: our own pid file at terminal-failed exit
        state.paths.pid.unlink(missing_ok=True)
    _record_outcome(state)
    os._exit(1)


def _install_fatal_http_hook(
    on_fatal: Callable[[_FatalHFError], None] | None = None,
) -> None:
    """Attach an httpx response hook to HF's shared client.

    ``on_fatal`` is invoked with the constructed ``_FatalHFError`` on
    every fatal-status response. Two production-relevant choices:
      - In the worker subprocess: a closure that calls
        ``_abort_to_terminal_state(state, exc)``.
      - In tests and other non-process-killing contexts:
        ``operator.iadd``-style ``raise exc`` (the default when
        ``on_fatal`` is None) so the caller can pytest-assert the type.

    Idempotent per-callback by function identity. Best-effort: if
    huggingface_hub's internal API changes and we can't reach the
    session, the worker logs a warning and runs without fail-fast.
    """
    if on_fatal is None:

        def on_fatal(exc: _FatalHFError) -> None:  # noqa: E306
            raise exc

    try:
        from huggingface_hub.utils._http import get_session
    except ImportError:
        logger.warning(
            "huggingface_hub.utils._http.get_session not importable; "
            "fail-fast on rate-limit/auth disabled for this session"
        )
        return

    try:
        client = get_session()
    except Exception:  # noqa: BLE001 — guard against any internal HF changes
        logger.warning(
            "Couldn't obtain HF shared http client; fail-fast disabled",
            exc_info=True,
        )
        return

    hooks = getattr(client, "event_hooks", None)
    if hooks is None or "response" not in hooks:
        logger.warning(
            "HF http client has no response event_hooks; fail-fast disabled (client type=%s)",
            type(client).__name__,
        )
        return

    def _hook(response: Any) -> None:
        exc = _classify_response(response)
        if exc is not None:
            on_fatal(exc)

    # Idempotency: don't stack duplicate hooks if install is called
    # multiple times in the same process (re-imports in tests, etc.).
    # We tag hooks we install so dedup is robust to closure identity.
    for existing in hooks["response"]:
        if getattr(existing, "_lerobot_fatal_hook", False):
            return
    _hook._lerobot_fatal_hook = True  # type: ignore[attr-defined]
    hooks["response"].append(_hook)


# ── Signal handling ─────────────────────────────────────────────────────────


def _force_cancel_exit(state: _WorkerState) -> None:
    """Write terminal ``cancelled`` state and kill the process.

    The escape hatch for the case the old cancel path had no answer for:
    ``upload_large_folder`` blocks the main thread for the whole transfer
    and offers no interruption point, so the between-stages
    ``cancel_requested`` check is never reached and a cancelled job runs to
    completion anyway. Setting a flag is not a cancel if nothing can
    observe it.

    Same mechanism as :func:`_abort_to_terminal_state`: persist the outcome
    the server reads, drop our PID file so the server's liveness sweep
    doesn't overwrite it with a generic "exited without finalizing", then
    ``os._exit``. Cooperative shutdown is not available — the main thread
    is inside HF's thread pool and joining it is exactly what we can't do.

    Pre: called from the cancel watchdog thread, after the grace period.
    Post: does not return.
    """
    # Stop and join the periodic writer first: it and this thread would
    # otherwise race on the same .tmp path inside atomic_write_json, and a
    # writer tick that snapshotted before our mutation could land last and
    # leave "running" as the final on-disk word.
    state.stop_writer_thread()
    if state.writer_thread is not None:
        state.writer_thread.join(timeout=PROGRESS_WRITE_INTERVAL_S * 4)
    with state._lock:
        state.status = "cancelled"
        state.error = "Cancelled by user"
        state.error_class = "cancelled"
        state.finished_at = time.time()
        state.milestone = "Cancelled"
        state.stage = "cancelled"
    with contextlib.suppress(Exception):  # last-ditch terminal write
        state.write_progress()
    with contextlib.suppress(Exception):
        # safe-destruct: our own pid file at terminal-cancelled exit
        state.paths.pid.unlink(missing_ok=True)
    _record_outcome(state)
    os._exit(130)  # 128 + SIGINT, the conventional "terminated by user" code


def _cancel_grace_s() -> float:
    """Grace period before a cancel is forced. Env var is a test hook.

    ``LEROBOT_HUB_CANCEL_GRACE_S`` lets the test suite exercise the
    force-exit path in under a second instead of waiting out the real
    deadline. Malformed values fall back to the constant rather than
    crashing a transfer over a typo'd env var.
    """
    raw = os.environ.get("LEROBOT_HUB_CANCEL_GRACE_S")
    if raw is None:
        return CANCEL_GRACE_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning("Ignoring malformed LEROBOT_HUB_CANCEL_GRACE_S=%r", raw)
        return CANCEL_GRACE_S


def _start_cancel_watchdog(state: _WorkerState, *, grace_s: float | None = None) -> threading.Thread:
    """Force-terminate the worker if a cancel request goes unhonoured.

    Started unconditionally at worker startup and parked on
    ``state.cancel_event`` so that the SIGTERM handler only has to call
    ``Event.set()`` — starting a thread *from* the handler would risk
    deadlocking against ``threading``'s own module-level lock if the main
    thread happened to be inside ``Thread.start()`` at signal time.

    After a cancel request it waits ``grace_s`` for the main thread to
    unwind on its own (the clean path, when the cancel lands between
    pipeline stages) and force-exits only if that hasn't happened.
    """

    grace = _cancel_grace_s() if grace_s is None else grace_s

    def _run() -> None:
        state.cancel_event.wait()
        if state.exit_event.wait(grace):
            return  # main thread finalised on its own; nothing to force.
        _force_cancel_exit(state)

    t = threading.Thread(target=_run, name="hub-worker-cancel-watchdog", daemon=True)
    t.start()
    return t


def _install_signal_handlers(state: _WorkerState) -> None:
    def _on_sigterm(signum, frame):  # noqa: ARG001
        # Set the flag so the main thread can exit cleanly at the next
        # pipeline-stage boundary, and arm the watchdog for the case where
        # it never reaches one (the common case: we're inside
        # upload_large_folder, which returns only when the transfer ends).
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
        # mid-handler may see the new milestone with the previous
        # timestamp (or vice versa). That transient inconsistency is
        # strictly preferable to a frozen worker, and the next writer-
        # thread tick (within PROGRESS_WRITE_INTERVAL_S) will write a
        # coherent snapshot anyway.
        #
        # ``milestone_locked`` is set last and pins the milestone: the
        # output-reader thread is emitting a tqdm tick every few
        # milliseconds and would otherwise overwrite "Cancelling…" before
        # the server's next poll ever saw it.
        state.cancel_requested = True
        state.milestone = "Cancelling…"
        state.milestone_at = time.time()
        state.milestone_locked = True
        # Safe from a signal handler: this event's lock is only ever held
        # briefly by the watchdog thread, never by the main thread, so
        # there is no re-entrancy hazard of the kind described above.
        state.cancel_event.set()

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

    # Establish the denominator up front. Without it the tray has no total
    # to render a percentage against, and every upload shows "0 / 0 files,
    # 0%" for its whole life however much data is moving. A stat() sweep of
    # the folder is the one total we can know exactly and that never
    # changes mid-transfer, unlike HF's per-batch bar totals.
    upload_files = enumerate_upload_files(
        Path(cfg.local_path),
        ignore_patterns=cfg.ignore_patterns or DEFAULT_UPLOAD_IGNORES,
    )
    with state._lock:
        state.files_total = len(upload_files)
        state.bytes_total = sum(p.stat().st_size for p in upload_files)
    state.write_progress()

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
    # Models live at the Hub root, datasets under /datasets. Hardcoding the
    # dataset prefix gave model uploads a PR link to a URL that does not exist.
    _ns = "datasets/" if cfg.repo_type == "dataset" else ""
    state.pr_url = f"https://huggingface.co/{_ns}{cfg.repo_id}/discussions/{pr_num}"
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

    state.set_milestone("Upload complete", stage="done")


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
    # Parked on state.cancel_event until a cancel arrives; see
    # _start_cancel_watchdog for why it can't be started from the handler.
    _start_cancel_watchdog(state)
    _write_pid_file(paths)

    state.status = "running"
    state.write_progress()
    writer = state.start_writer_thread()

    # Redirect HF's stderr AND stdout through our reader-thread so we can
    # extract milestones AND keep the verbatim text in the per-job log.
    # We replace the worker's own fd 1 and fd 2 to point at the same
    # writable pipe; a background thread drains the read end.
    #
    # Both streams unified into one pipe (rather than two pipes + two
    # reader threads) keeps the ordering between stdout and stderr writes
    # faithful to the kernel's, and avoids interleaved log files where
    # cause-and-effect text from the same library is split across two
    # capture channels.
    r_fd, w_fd = os.pipe()
    original_stderr_fd = os.dup(2)
    original_stdout_fd = os.dup(1)
    os.dup2(w_fd, 2)
    os.dup2(w_fd, 1)
    os.close(w_fd)
    sys.stderr = os.fdopen(2, "w", buffering=1)  # line-buffered text wrapper
    sys.stdout = os.fdopen(1, "w", buffering=1)

    reader_thread = threading.Thread(
        target=stream_stderr_to_log_and_state,
        args=(os.fdopen(r_fd, "rb", buffering=0), paths.log, state),
        daemon=True,
        name="hub-worker-output-reader",
    )
    reader_thread.start()

    # Install the fail-fast httpx hook BEFORE the pipeline runs so the
    # first 429/401/403/404 from any HF call (create_repo, create_pr,
    # the commit step inside upload_large_folder, etc.) terminates the
    # process immediately. The hook runs on whichever HF thread issued
    # the request — usually one of upload_large_folder's _worker_job
    # threads — and calls os._exit after writing terminal state to the
    # progress JSON. We cannot use exception-propagation because plain
    # threading.Thread doesn't surface exceptions to the main thread.
    _install_fatal_http_hook(lambda exc: _abort_to_terminal_state(state, exc))

    rc = 0
    try:
        if cfg.direction == "upload":
            _do_upload(state)
        elif cfg.direction == "download":
            _do_download(state)
        else:  # pragma: no cover — guarded by JobConfig.__post_init__
            raise ValueError(f"unknown direction: {cfg.direction!r}")
        state.mark_complete()
    except InterruptedError:
        # Raised on our cancel path between pipeline stages. Local
        # `.cache/.huggingface/` + the draft PR remain intact for resume.
        state.status = "cancelled"
        state.error = "Cancelled by user"
        state.error_class = "cancelled"
        # Assigned directly: milestone_locked (set by the SIGTERM handler
        # to stop tqdm ticks clobbering "Cancelling…") also blocks
        # set_milestone, and this is the transition it must not block.
        state.milestone = "Cancelled"
        state.stage = "cancelled"
    except KeyboardInterrupt:
        # SIGINT delivered by tests or interactive run.
        state.status = "cancelled"
        state.error = "Cancelled (SIGINT)"
        state.error_class = "cancelled"
        state.milestone = "Cancelled"
        state.stage = "cancelled"
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
        # Restore stderr + stdout first so any late writes (from the
        # writer thread's final flush, for instance) don't deadlock
        # against a half-closed pipe.
        try:
            os.dup2(original_stderr_fd, 2)
            os.close(original_stderr_fd)
            os.dup2(original_stdout_fd, 1)
            os.close(original_stdout_fd)
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
        _record_outcome(state)
        # Tell the cancel watchdog we finalised under our own power, so it
        # stands down instead of force-exiting over a completed job.
        state.exit_event.set()

    return rc


if __name__ == "__main__":
    sys.exit(main())
