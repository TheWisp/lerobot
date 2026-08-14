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
"""Tests for the durable Hub-transfer history.

This exists because a real 8.4 GB upload completed, was garbage-collected 30
minutes later, and left the user with no way to tell success from failure. The
properties below are the ones that make the record trustworthy after the fact:
it survives concurrent writers in separate processes, it survives a torn line,
and the authoritative writer wins.
"""

from __future__ import annotations

import json
import multiprocessing
import threading
import time
from pathlib import Path

import pytest

from lerobot.gui import hub_history


def _rec(job_id: str, **kw):
    base = {"job_id": job_id, "ts": kw.pop("ts", time.time()), "status": "complete"}
    base.update(kw)
    return base


class TestAppendAndRead:
    def test_round_trip(self, tmp_path):
        p = tmp_path / "h.jsonl"
        assert hub_history.append_outcome(_rec("a", ts=1.0), path=p) is True
        out = hub_history.read_recent(path=p)
        assert [r["job_id"] for r in out] == ["a"]

    def test_missing_file_reads_empty(self, tmp_path):
        assert hub_history.read_recent(path=tmp_path / "nope.jsonl") == []

    def test_newest_first(self, tmp_path):
        p = tmp_path / "h.jsonl"
        for i, jid in enumerate("abc"):
            hub_history.append_outcome(_rec(jid, ts=float(i)), path=p)
        assert [r["job_id"] for r in hub_history.read_recent(path=p)] == ["c", "b", "a"]

    def test_limit_applies_after_dedup(self, tmp_path):
        p = tmp_path / "h.jsonl"
        for i in range(10):
            hub_history.append_outcome(_rec(f"j{i}", ts=float(i)), path=p)
        assert len(hub_history.read_recent(limit=3, path=p)) == 3


class TestLastWriterWins:
    """Both the worker and the server record; the server writes last.

    A SIGKILLed worker cannot record its own ending, so the server records
    those. Rather than coordinating who writes, both do — which only works if
    the reader keeps one entry per job and keeps the *later* one.
    """

    def test_later_line_supersedes_earlier_for_same_job(self, tmp_path):
        p = tmp_path / "h.jsonl"
        hub_history.append_outcome(_rec("j", ts=1.0, status="running"), path=p)
        hub_history.append_outcome(_rec("j", ts=2.0, status="cancelled"), path=p)
        out = hub_history.read_recent(path=p)
        assert len(out) == 1, "one entry per job, not one per write"
        assert out[0]["status"] == "cancelled"

    def test_duplicate_does_not_inflate_the_count(self, tmp_path):
        p = tmp_path / "h.jsonl"
        for _ in range(5):
            hub_history.append_outcome(_rec("j"), path=p)
        assert len(hub_history.read_recent(path=p)) == 1


class TestCorruptionTolerance:
    """A torn line must not cost the user the rest of their history."""

    @pytest.mark.parametrize(
        "junk",
        ['{"job_id": "x", "ts"', "not json at all", "", "   ", "[1,2,3]", "null"],
    )
    def test_bad_lines_are_skipped(self, tmp_path, junk):
        p = tmp_path / "h.jsonl"
        hub_history.append_outcome(_rec("good", ts=1.0), path=p)
        with open(p, "a") as f:
            f.write(junk + "\n")
        hub_history.append_outcome(_rec("also-good", ts=2.0), path=p)
        assert [r["job_id"] for r in hub_history.read_recent(path=p)] == ["also-good", "good"]

    def test_entry_without_job_id_is_skipped(self, tmp_path):
        p = tmp_path / "h.jsonl"
        with open(p, "a") as f:
            f.write(json.dumps({"ts": 1.0, "status": "complete"}) + "\n")
        assert hub_history.read_recent(path=p) == []


def _append_many(path_str: str, prefix: str, n: int) -> None:
    """Separate-process writer — the real shape, since worker and server differ."""
    p = Path(path_str)
    for i in range(n):
        hub_history.append_outcome({"job_id": f"{prefix}{i}", "ts": float(i), "status": "complete"}, path=p)


class TestConcurrentWriters:
    """Append-only is the point: the writers are in different processes.

    The progress file's read-modify-write shape raced and killed a thread
    (see TestAtomicWriteJsonConcurrency). This file is appended to instead,
    one short line at a time, so concurrent writers interleave whole lines.
    """

    def test_threads_do_not_lose_or_corrupt_entries(self, tmp_path):
        p = tmp_path / "h.jsonl"
        n_threads, per = 8, 40
        errs: list[BaseException] = []

        def run(w: int) -> None:
            try:
                _append_many(str(p), f"t{w}_", per)
            except BaseException as e:  # noqa: BLE001
                errs.append(e)

        ts = [threading.Thread(target=run, args=(w,)) for w in range(n_threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=60)

        assert not errs, f"concurrent writers raised: {errs!r}"
        assert len(hub_history.read_recent(limit=10_000, path=p)) == n_threads * per

    def test_separate_processes_do_not_corrupt_lines(self, tmp_path):
        """The real configuration: the worker is not in the server's process."""
        p = tmp_path / "h.jsonl"
        ctx = multiprocessing.get_context("spawn")
        procs = [ctx.Process(target=_append_many, args=(str(p), f"p{i}_", 30)) for i in range(4)]
        for pr in procs:
            pr.start()
        for pr in procs:
            pr.join(timeout=120)
        assert all(pr.exitcode == 0 for pr in procs), [pr.exitcode for pr in procs]

        # Every line must be parseable — a torn write would show up here.
        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        assert len(lines) == 120
        for ln in lines:
            json.loads(ln)


class TestPruning:
    def test_file_does_not_grow_without_bound(self, tmp_path, monkeypatch):
        """Bounded, not exact.

        Pruning is gated on a cheap size check so that appending does not read
        the whole file every time, which means the line count can overshoot the
        cap slightly before a rewrite happens. The guarantee is that it stays
        bounded — 200 appends against a cap of 50 must not leave 200 lines.
        """
        p = tmp_path / "h.jsonl"
        monkeypatch.setattr(hub_history, "MAX_LINES", 50)
        for i in range(200):
            hub_history.append_outcome(_rec(f"j{i}", ts=float(i)), path=p)
        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        assert len(lines) <= 2 * 50, f"unbounded growth: {len(lines)} lines"
        # Pruning keeps the newest, which is what a user would look for.
        assert hub_history.read_recent(limit=1, path=p)[0]["job_id"] == "j199"

    def test_pruning_leaves_no_temp_files(self, tmp_path, monkeypatch):
        p = tmp_path / "h.jsonl"
        monkeypatch.setattr(hub_history, "MAX_LINES", 20)
        for i in range(100):
            hub_history.append_outcome(_rec(f"j{i}", ts=float(i)), path=p)
        assert sorted(x.name for x in tmp_path.iterdir()) == ["h.jsonl"]


class TestNeverBreaksATransfer:
    """Bookkeeping failing must not fail the transfer it is describing."""

    def test_unwritable_path_returns_false_rather_than_raising(self, tmp_path):
        # A directory where the file should be: open() will fail.
        blocked = tmp_path / "h.jsonl"
        blocked.mkdir()
        assert hub_history.append_outcome(_rec("j"), path=blocked) is False

    def test_unserialisable_record_is_rejected(self, tmp_path):
        class Weird:
            pass

        # default=str makes most things serialisable; keys must still be str.
        assert hub_history.append_outcome({Weird(): 1, "job_id": "j"}, path=tmp_path / "h.jsonl") is False


class TestRecordFromJob:
    def test_carries_what_the_user_would_ask_about(self):
        from lerobot.gui import hub_jobs

        j = hub_jobs.make_job(dataset_id="ds", direction="upload", repo_id="u/r")
        j.status = "complete"
        j.started_at = 100.0
        j.finished_at = 400.0
        j.bytes_total = 8_414_284_276
        j.bytes_done_estimate = 3_988_804_000
        j.files_total = 84
        j.pr_num = 2

        rec = hub_history._record_from_job(j)
        assert rec["status"] == "complete"
        assert rec["repo_id"] == "u/r"
        assert rec["duration_s"] == pytest.approx(300.0)
        assert rec["bytes_total"] == 8_414_284_276
        assert rec["pr_num"] == 2
        json.dumps(rec)  # must be serialisable as-is

    def test_duration_is_never_negative(self):
        from lerobot.gui import hub_jobs

        j = hub_jobs.make_job(dataset_id="ds", direction="upload", repo_id="u/r")
        j.status = "failed"
        j.started_at = 500.0
        j.finished_at = 100.0  # clock moved backwards
        assert hub_history._record_from_job(j)["duration_s"] == 0.0
