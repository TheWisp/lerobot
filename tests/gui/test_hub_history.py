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
    """Trimming is explicit and off the append path.

    It is a read-modify-write, so calling it while a worker may be appending
    would drop the very record it is trimming — the shape this module exists
    to avoid. It runs once at server startup instead.
    """

    def test_prune_keeps_the_newest(self, tmp_path, monkeypatch):
        p = tmp_path / "h.jsonl"
        monkeypatch.setattr(hub_history, "MAX_LINES", 50)
        for i in range(200):
            hub_history.append_outcome(_rec(f"j{i}", ts=float(i)), path=p)
        assert len([x for x in p.read_text().splitlines() if x.strip()]) == 200, (
            "append must not trim; that is what makes it race-free"
        )

        dropped = hub_history.prune(p)
        assert dropped == 150
        lines = [x for x in p.read_text().splitlines() if x.strip()]
        assert len(lines) == 50
        assert hub_history.read_recent(limit=1, path=p)[0]["job_id"] == "j199"

    def test_prune_is_a_noop_below_the_cap(self, tmp_path, monkeypatch):
        p = tmp_path / "h.jsonl"
        monkeypatch.setattr(hub_history, "MAX_LINES", 50)
        for i in range(10):
            hub_history.append_outcome(_rec(f"j{i}", ts=float(i)), path=p)
        assert hub_history.prune(p) == 0

    def test_prune_on_a_missing_file_is_a_noop(self, tmp_path):
        assert hub_history.prune(tmp_path / "never-written.jsonl") == 0

    def test_pruning_leaves_no_temp_files(self, tmp_path, monkeypatch):
        p = tmp_path / "h.jsonl"
        monkeypatch.setattr(hub_history, "MAX_LINES", 20)
        for i in range(100):
            hub_history.append_outcome(_rec(f"j{i}", ts=float(i)), path=p)
        hub_history.prune(p)
        assert sorted(x.name for x in tmp_path.iterdir()) == ["h.jsonl"]


class TestNeverBreaksATransfer:
    """Bookkeeping failing must not fail the transfer it is describing."""

    def test_unwritable_path_returns_false_rather_than_raising(self, tmp_path):
        # A directory where the file should be: open() will fail.
        blocked = tmp_path / "h.jsonl"
        blocked.mkdir()
        assert hub_history.append_outcome(_rec("j"), path=blocked) is False

    def test_unserialisable_record_returns_false(self, tmp_path):
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


class TestMalformedTimestampCannotBreakTheHistory:
    """A single bad line must not take the whole record with it.

    `ts` is sorted on, so a string timestamp raised TypeError outside any
    try — permanently 500-ing the history endpoint. That is exactly the
    whole-history loss this module promises cannot happen from one torn line.
    """

    @pytest.mark.parametrize("bad_ts", ['"not-a-number"', "null", "{}", "[]", "true"])
    def test_bad_timestamp_does_not_raise(self, tmp_path, bad_ts):
        p = tmp_path / "h.jsonl"
        hub_history.append_outcome(_rec("good", ts=5.0), path=p)
        with open(p, "a") as f:
            f.write(f'{{"job_id": "bad", "ts": {bad_ts}, "status": "complete"}}\n')
        out = hub_history.read_recent(path=p)
        assert [r["job_id"] for r in out][0] == "good", "the good entry must still sort first"
        assert len(out) == 2

    def test_a_bad_timestamp_sorts_last_rather_than_winning(self, tmp_path):
        p = tmp_path / "h.jsonl"
        with open(p, "a") as f:
            f.write('{"job_id": "bad", "ts": "9999", "status": "complete"}\n')
        hub_history.append_outcome(_rec("good", ts=1.0), path=p)
        assert hub_history.read_recent(path=p)[0]["job_id"] == "good"


class TestAppendNeverRewritesTheFile:
    """Appending must stay append-only; that is the race-free property.

    Trimming inside `append_outcome` made it a read-modify-write, so a line
    written by another process between the read and the replace was lost —
    reintroducing the failure mode that killed a worker thread through the
    progress file.
    """

    def test_append_only_ever_grows_the_file(self, tmp_path, monkeypatch):
        p = tmp_path / "h.jsonl"
        monkeypatch.setattr(hub_history, "MAX_LINES", 5)
        seen = []
        for i in range(40):
            hub_history.append_outcome(_rec(f"j{i}", ts=float(i)), path=p)
            seen.append(len([x for x in p.read_text().splitlines() if x.strip()]))
        assert seen == sorted(seen), "line count must be monotonic — no rewrite on append"
        assert seen[-1] == 40


class TestReadsWhatOlderAndNewerVersionsWrote:
    """The file outlives the version that wrote it, so the reader must not
    assume the writer was this one.

    Nothing here needs a migration: an install upgrading into this feature has
    no history file at all, and one downgrading leaves records a later version
    can still read. What that costs is a reader that treats every field except
    ``job_id`` as optional — which is only true if something checks.
    """

    def test_a_record_with_only_a_job_id_survives(self, tmp_path):
        """The floor. Every other field is optional by construction."""
        p = tmp_path / "h.jsonl"
        p.write_text('{"job_id": "minimal"}\n')
        out = hub_history.read_recent(path=p)
        assert [r["job_id"] for r in out] == ["minimal"]
        assert out[0]["ts"] == 0.0, "a missing timestamp must sort, not raise"

    def test_unknown_fields_from_a_newer_version_are_preserved(self, tmp_path):
        """Dropping them would silently downgrade the file on the next prune."""
        p = tmp_path / "h.jsonl"
        p.write_text('{"job_id": "j", "ts": 1.0, "field_from_the_future": {"a": [1, 2]}}\n')
        assert hub_history.read_recent(path=p)[0]["field_from_the_future"] == {"a": [1, 2]}

    def test_a_file_that_is_not_json_at_all_reads_empty(self, tmp_path):
        """Whatever else is at that path, the endpoint must still answer."""
        p = tmp_path / "h.jsonl"
        p.write_bytes(b"\x00\xff\xfe not json\n" * 3)
        assert hub_history.read_recent(path=p) == []


class TestPruneCannotBreakServerStartup:
    """`prune()` runs in the startup event, unguarded by its caller.

    It is bookkeeping on a path where a raise costs the user the whole GUI, so
    the containment has to be inside prune itself.
    """

    def test_unreadable_file_returns_zero(self, tmp_path):
        p = tmp_path / "h.jsonl"
        p.write_text('{"job_id": "a", "ts": 1}\n' * 10)
        p.chmod(0o000)
        try:
            assert hub_history.prune(p, max_lines=1) == 0
        finally:
            p.chmod(0o644)

    def test_a_directory_where_the_file_should_be_returns_zero(self, tmp_path):
        p = tmp_path / "h.jsonl"
        p.mkdir()
        assert hub_history.prune(p, max_lines=1) == 0
        assert hub_history.append_outcome({"job_id": "x", "ts": 1.0}, path=p) is False
