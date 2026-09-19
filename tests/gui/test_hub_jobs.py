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
"""Unit tests for the Hub-transfer helper layer.

What this file covers:
  * JobConfig JSON round-trip
  * JobPaths derives consistent locations from a job_id
  * HubJobState shape + merge_progress invariants (won't un-terminalize)
  * PID-file identity: pid_file_payload, is_worker_alive against live + dead processes
  * Error classification across the HF exception hierarchy
  * Milestone extraction against recorded HF stderr samples
  * Upload-side file enumeration (respects ignore patterns, sorted)
  * check_upload_completeness logic (fresh repo, missing locally, incomplete)

What this file does NOT cover (separate test files):
  * Worker subprocess end-to-end — see test_hub_worker.py
  * Endpoint flow with FastAPI TestClient — see test_hub_endpoints.py
  * Real HF interaction — see test_hub_live.py (gated by ``@pytest.mark.hub_live``)
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time

import pytest

from lerobot.gui import hub_jobs

# ── JobConfig ───────────────────────────────────────────────────────────────


class TestJobConfig:
    def _good(self, **overrides):
        defaults = {
            "job_id": "abc123",
            "dataset_id": "user/foo",
            "direction": "upload",
            "repo_id": "user/foo",
            "repo_type": "dataset",
            "local_path": "/tmp/foo",
            "jobs_dir": "/tmp/jobs",
        }
        defaults.update(overrides)
        return hub_jobs.JobConfig(**defaults)

    def test_json_roundtrip_preserves_all_fields(self):
        cfg = self._good(
            private=False,
            commit_message="hello",
            ignore_patterns=(".cache/", ".DS_Store"),
            reuse_pr_num=42,
        )
        round_tripped = hub_jobs.JobConfig.from_json(cfg.to_json())
        assert round_tripped == cfg

    def test_rejects_bad_direction(self):
        with pytest.raises(ValueError, match="bad direction"):
            self._good(direction="sideways")

    def test_rejects_bad_repo_type(self):
        with pytest.raises(ValueError, match="bad repo_type"):
            self._good(repo_type="bucket")

    def test_ignore_patterns_normalized_to_tuple(self):
        cfg = self._good(ignore_patterns=(".cache/",))
        # Round-trip through JSON to confirm the list-roundtrip turns it back to tuple.
        rt = hub_jobs.JobConfig.from_json(cfg.to_json())
        assert isinstance(rt.ignore_patterns, tuple)
        assert rt.ignore_patterns == (".cache/",)

    def test_ignore_patterns_none_stays_none(self):
        cfg = self._good(ignore_patterns=None)
        rt = hub_jobs.JobConfig.from_json(cfg.to_json())
        assert rt.ignore_patterns is None


# ── JobPaths ────────────────────────────────────────────────────────────────


class TestJobPaths:
    def test_paths_share_directory_and_job_id_prefix(self, tmp_path):
        paths = hub_jobs.JobPaths.for_job("abc123", tmp_path)
        assert paths.progress == tmp_path / "abc123.json"
        assert paths.log == tmp_path / "abc123.log"
        assert paths.pid == tmp_path / "abc123.pid"

    def test_paths_for_different_jobs_dont_collide(self, tmp_path):
        p1 = hub_jobs.JobPaths.for_job("aaa", tmp_path)
        p2 = hub_jobs.JobPaths.for_job("bbb", tmp_path)
        assert p1.progress != p2.progress
        assert p1.log != p2.log
        assert p1.pid != p2.pid


# ── HubJobState ─────────────────────────────────────────────────────────────


class TestHubJobState:
    def test_make_job_starts_pending(self):
        j = hub_jobs.make_job(dataset_id="ds", direction="upload", repo_id="u/r")
        assert j.status == "pending"
        assert j.started_at > 0
        assert j.finished_at is None
        assert j.pr_num is None
        assert j.error is None

    def test_initial_milestone_is_direction_aware(self):
        """Pre-worker-spawn milestone must say which direction the job is.

        A bare "starting" was historically rendered in the tray before the
        worker process attached, leading users to misread an upload as a
        download (see the Hub-transfers UX bug). The default now spells the
        direction out so the tray cannot ambiguate.
        """
        up = hub_jobs.make_job(dataset_id="ds", direction="upload", repo_id="u/r")
        down = hub_jobs.make_job(dataset_id="ds", direction="download", repo_id="u/r")
        assert up.milestone == "Starting upload"
        assert down.milestone == "Starting download"

    def test_unique_job_ids(self):
        ids = {
            hub_jobs.make_job(dataset_id="ds", direction="upload", repo_id="u/r").job_id for _ in range(20)
        }
        assert len(ids) == 20

    def test_merge_progress_updates_live_state(self):
        j = hub_jobs.make_job(dataset_id="ds", direction="upload", repo_id="u/r")
        j.status = "running"
        j.merge_progress(
            {
                "status": "running",
                "milestone": "Uploading files",
                "milestone_at": 12345.0,
                "files_total": 10,
                "files_done_estimate": 4,
                "pr_num": 7,
            }
        )
        assert j.milestone == "Uploading files"
        assert j.files_total == 10
        assert j.files_done_estimate == 4
        assert j.pr_num == 7

    def test_merge_progress_cannot_un_terminalize(self):
        """Once status is terminal, no merge from the worker drags it back.

        Protects against a confused worker writing a stale snapshot after
        we've already marked the job failed/cancelled server-side.
        """
        j = hub_jobs.make_job(dataset_id="ds", direction="upload", repo_id="u/r")
        j.status = "failed"
        j.error = "auth"
        j.merge_progress({"status": "running", "milestone": "Uploading"})
        assert j.status == "failed"
        # Other fields also don't move (the snapshot is ignored entirely).
        # The default milestone is direction-aware so a user looking at the
        # tray during the brief pre-spawn window sees "Starting upload" rather
        # than a context-free "starting" that historically read as ambiguous
        # (could be download — see the original Hub-transfers UX bug report).
        assert j.milestone == "Starting upload"

    def test_to_dict_omits_nothing_serialisable(self):
        """Sanity: every field in to_dict() is JSON-serialisable."""
        j = hub_jobs.make_job(dataset_id="ds", direction="upload", repo_id="u/r")
        json.dumps(j.to_dict())  # raises if any value is non-JSON


# ── PID-file identity ───────────────────────────────────────────────────────


class TestPidFileIdentity:
    def test_payload_contains_pid_and_started_at(self):
        payload = hub_jobs.pid_file_payload(os.getpid())
        assert payload["pid"] == os.getpid()
        assert "start_time" in payload
        assert "started_at" in payload
        assert payload["started_at"] > 0

    def test_alive_check_for_current_process_is_true(self):
        payload = hub_jobs.pid_file_payload(os.getpid())
        assert hub_jobs.is_worker_alive(payload) is True

    def test_alive_check_for_dead_pid_is_false(self):
        # Use PID 1 (init) but tamper the start_time so the identity check
        # fails. init's process exists but its start_time differs from any
        # we'd record, simulating PID reuse.
        payload = {"pid": 1, "start_time": -1.0, "started_at": 0.0}
        # On Linux, init's start_time is well-defined and != -1.0; the
        # check should reject. On non-Linux, start_time is None so we
        # degrade to "process exists" — which also returns True for init.
        # We accept either outcome on non-Linux; on Linux we expect False.
        result = hub_jobs.is_worker_alive(payload)
        if hub_jobs._process_start_time(1) is not None:
            assert result is False, "Linux should reject mismatched start_time"
        # Otherwise the result is platform-dependent; just verify no crash.

    def test_alive_check_for_truly_dead_pid_is_false(self):
        # Spawn a short-lived child, capture its pid, wait for it to die.
        import subprocess

        proc = subprocess.Popen(["true"])
        proc.wait()
        payload = {"pid": proc.pid, "start_time": None, "started_at": time.time()}
        # The pid is now dead. is_worker_alive should return False.
        assert hub_jobs.is_worker_alive(payload) is False

    def test_alive_check_handles_missing_pid_key(self):
        assert hub_jobs.is_worker_alive({}) is False
        assert hub_jobs.is_worker_alive({"pid": "not an int"}) is False

    def test_read_pid_file_returns_none_for_missing_path(self, tmp_path):
        assert hub_jobs.read_pid_file(tmp_path / "nope.pid") is None

    def test_read_pid_file_returns_none_for_malformed_json(self, tmp_path):
        path = tmp_path / "bad.pid"
        path.write_text("not json {")
        assert hub_jobs.read_pid_file(path) is None

    def test_read_pid_file_roundtrips(self, tmp_path):
        path = tmp_path / "ok.pid"
        payload = hub_jobs.pid_file_payload(os.getpid())
        path.write_text(json.dumps(payload))
        assert hub_jobs.read_pid_file(path) == payload


# ── Error classification ────────────────────────────────────────────────────


class TestErrorClassification:
    def _hf_http_error(self, status_code: int, message: str):
        """Construct an HfHubHTTPError with the given status, version-agnostic.

        Newer huggingface_hub (1.x) requires a real httpx.Response. We build
        a minimal one — only the status_code attribute matters for our
        classification.
        """
        import httpx
        from huggingface_hub.errors import HfHubHTTPError

        response = httpx.Response(status_code, request=httpx.Request("GET", "http://x"))
        return HfHubHTTPError(message, response=response)

    def test_classifies_auth_via_status_code(self):
        assert hub_jobs.classify_error(self._hf_http_error(401, "unauthorized")) == "auth"
        assert hub_jobs.classify_error(self._hf_http_error(403, "forbidden")) == "auth"

    def test_classifies_rate_limit(self):
        assert hub_jobs.classify_error(self._hf_http_error(429, "too many requests")) == "rate_limit"

    def test_classifies_5xx_as_network(self):
        assert hub_jobs.classify_error(self._hf_http_error(503, "service unavailable")) == "network"
        assert hub_jobs.classify_error(self._hf_http_error(502, "bad gateway")) == "network"

    def test_classifies_repository_not_found_as_auth(self):
        """Private-repo-no-access surfaces as RepositoryNotFoundError; treat as auth.

        Construct one with the required response kwarg (newer HF) or fall
        back to setting response post-init (older HF).
        """
        import httpx
        from huggingface_hub.errors import RepositoryNotFoundError

        response = httpx.Response(404, request=httpx.Request("GET", "http://x"))
        try:
            rnf = RepositoryNotFoundError("private_inaccessible", response=response)
        except TypeError:
            rnf = RepositoryNotFoundError("private_inaccessible")  # type: ignore[call-arg]
            rnf.response = response
        assert hub_jobs.classify_error(rnf) == "auth"

    def test_classifies_connection_error_as_network(self):
        err = ConnectionError("name resolution failed")
        assert hub_jobs.classify_error(err) == "network"

    def test_classifies_timeout_error_as_network(self):
        err = TimeoutError("read timed out")
        assert hub_jobs.classify_error(err) == "network"

    def test_classifies_text_substring_fallback_auth(self):
        # Plain Exception with auth-related message → "auth"
        assert hub_jobs.classify_error(Exception("401 Unauthorized")) == "auth"
        assert hub_jobs.classify_error(Exception("403 Forbidden")) == "auth"

    def test_classifies_unknown_as_other(self):
        assert hub_jobs.classify_error(Exception("something else")) == "other"
        assert hub_jobs.classify_error(ValueError("bad input")) == "other"


# ── Milestone extraction ────────────────────────────────────────────────────
#
# Pinned samples of HF/tqdm stderr. If HF's format shifts in a future
# version, update the samples + fixtures here; the parser falls back to
# unmatched and the rest of the system stays functional.


class TestMilestoneExtraction:
    """The parser is best-effort — these tests pin the patterns we recognise.

    The graceful-degradation contract is more important than format
    coverage: if the parser returns None on a line, the system uses the
    fallback ``"running"`` milestone. So a regression here is a UX
    degradation, not a correctness bug.
    """

    def test_upload_processing_files(self):
        from lerobot.gui.hub_worker import extract_milestone

        result = extract_milestone("Processing Files (3 / 47)", "upload")
        assert result == "Processing files 3 / 47"

    def test_upload_committing(self):
        from lerobot.gui.hub_worker import extract_milestone

        assert extract_milestone("Committing files (5 / 12)", "upload") == "Committing 5 / 12"

    def test_download_fetching(self):
        from lerobot.gui.hub_worker import extract_milestone

        result = extract_milestone("Fetching 8 files: 100%|########| 5/8 [00:02<00:00]", "download")
        # Pattern matches "Fetching {N} files" and "{k}/{N}"
        assert result == "Downloading 5 / 8 files"

    def test_download_bare_percentage(self):
        from lerobot.gui.hub_worker import extract_milestone

        # Generic tqdm percentage matches the fallback pattern.
        assert extract_milestone("42%|####     |  4/10", "download") == "Downloading 42%"

    def test_unmatched_returns_none(self):
        from lerobot.gui.hub_worker import extract_milestone

        assert extract_milestone("Some unrelated log line", "upload") is None
        assert extract_milestone("", "upload") is None
        assert extract_milestone("Some unrelated log line", "download") is None


# ── enumerate_upload_files ──────────────────────────────────────────────────


class TestEnumerateUploadFiles:
    def test_returns_every_regular_file(self, tmp_path):
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "info.json").write_text("{}")
        (tmp_path / "data.parquet").write_bytes(b"x")
        files = hub_jobs.enumerate_upload_files(tmp_path)
        rels = sorted(p.relative_to(tmp_path).as_posix() for p in files)
        assert rels == ["data.parquet", "meta/info.json"]

    def test_skips_default_ignores(self, tmp_path):
        (tmp_path / "data.parquet").write_bytes(b"x")
        (tmp_path / ".lerobot_gui_edits.json").write_text("{}")
        (tmp_path / ".cache").mkdir()
        (tmp_path / ".cache" / "stuff").write_bytes(b"y")
        (tmp_path / ".DS_Store").write_bytes(b"z")
        files = hub_jobs.enumerate_upload_files(tmp_path)
        rels = sorted(p.relative_to(tmp_path).as_posix() for p in files)
        assert rels == ["data.parquet"]

    def test_returns_sorted_order(self, tmp_path):
        for name in ["z.bin", "a.bin", "m.bin"]:
            (tmp_path / name).write_bytes(b"")
        files = hub_jobs.enumerate_upload_files(tmp_path)
        rels = [p.relative_to(tmp_path).as_posix() for p in files]
        assert rels == ["a.bin", "m.bin", "z.bin"]

    def test_asserts_on_missing_root(self, tmp_path):
        with pytest.raises(AssertionError):
            hub_jobs.enumerate_upload_files(tmp_path / "nope")


# ── check_upload_completeness ──────────────────────────────────────────────
#
# Defends against the download-fail-then-upload corruption scenario.


class _FakeApi:
    """Minimal HfApi mock for the completeness check."""

    def __init__(self, siblings=None, raise_not_found=False):
        self._siblings = siblings or []
        self._raise = raise_not_found

    def repo_info(self, repo_id, repo_type="dataset", files_metadata=False):
        if self._raise:
            import httpx
            from huggingface_hub.errors import RepositoryNotFoundError

            response = httpx.Response(404, request=httpx.Request("GET", "http://x"))
            try:
                err: Exception = RepositoryNotFoundError(f"{repo_id} not found", response=response)
            except TypeError:
                err = RepositoryNotFoundError(f"{repo_id} not found")  # type: ignore[call-arg]
            raise err

        class _Sib:
            def __init__(self, rfilename):
                self.rfilename = rfilename

        class _Info:
            siblings = [_Sib(s) for s in self._siblings]

        return _Info()


class TestCheckUploadCompleteness:
    def test_fresh_repo_returns_empty(self, tmp_path):
        api = _FakeApi(raise_not_found=True)
        out = hub_jobs.check_upload_completeness(tmp_path, "user/new_repo", api=api)
        assert out == {"missing_locally": [], "incomplete_locally": []}

    def test_all_files_present_returns_empty(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"x")
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "info.json").write_text("{}")
        api = _FakeApi(siblings=["a.bin", "meta/info.json"])
        out = hub_jobs.check_upload_completeness(tmp_path, "user/repo", api=api)
        assert out["missing_locally"] == []
        assert out["incomplete_locally"] == []

    def test_detects_missing_locally(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"x")
        # remote has a.bin AND b.bin; b.bin is missing locally
        api = _FakeApi(siblings=["a.bin", "b.bin"])
        out = hub_jobs.check_upload_completeness(tmp_path, "user/repo", api=api)
        assert out["missing_locally"] == ["b.bin"]
        assert out["incomplete_locally"] == []

    def test_detects_incomplete_marker(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"x")
        # Simulate a half-finished download: HF leaves <name>.incomplete.
        (tmp_path / "a.bin.incomplete").write_bytes(b"partial")
        api = _FakeApi(siblings=["a.bin"])
        out = hub_jobs.check_upload_completeness(tmp_path, "user/repo", api=api)
        assert out["incomplete_locally"] == ["a.bin"]


class TestCancellingStatusIsServerOwned:
    """A worker snapshot must not un-cancel a job the user cancelled.

    The worker keeps writing ``running`` until it finishes unwinding. If
    that snapshot wins the merge, the tray flips back to a normal running
    card one poll after the click — the "cancel did nothing" symptom.
    """

    def _cancelling_job(self):
        job = hub_jobs.make_job(dataset_id="u/ds", direction="upload", repo_id="u/ds")
        job.status = "cancelling"
        return job

    def test_running_snapshot_does_not_revert_cancelling(self):
        job = self._cancelling_job()
        job.merge_progress({"status": "running", "milestone": "Uploading files"})
        assert job.status == "cancelling"

    def test_progress_numbers_still_merge_while_cancelling(self):
        """Pinning the status must not freeze the rest of the readout."""
        job = self._cancelling_job()
        job.merge_progress({"status": "running", "bytes_done_estimate": 4096, "transfer_rate_bps": 12.5})
        assert job.status == "cancelling"
        assert job.bytes_done_estimate == 4096
        assert job.transfer_rate_bps == 12.5

    def test_stale_milestone_does_not_flicker_off_cancelling(self):
        """The worker's last pre-SIGTERM snapshot must not retake the label."""
        job = self._cancelling_job()
        job.milestone = "Cancelling…"
        job.merge_progress({"status": "running", "milestone": "Processing files 0 / 1"})
        assert job.milestone == "Cancelling…"

    @pytest.mark.parametrize("terminal", ["cancelled", "failed", "complete"])
    def test_terminal_snapshot_is_accepted(self, terminal):
        job = self._cancelling_job()
        job.merge_progress({"status": terminal})
        assert job.status == terminal

    def test_cancelling_counts_as_active_not_terminal(self):
        assert "cancelling" in hub_jobs.ACTIVE_STATUSES
        assert "cancelling" not in hub_jobs.TERMINAL_STATUSES


class TestStallClock:
    """``stalled_for_s`` is what lets the tray say "stuck" instead of implying it."""

    def _running_job(self, *, last_progress_at):
        job = hub_jobs.make_job(dataset_id="u/ds", direction="upload", repo_id="u/ds")
        job.status = "running"
        job.last_progress_at = last_progress_at
        return job

    def test_reports_seconds_since_the_last_movement(self):
        now = time.time()
        job = self._running_job(last_progress_at=now - 120.0)
        assert job.stalled_for_s(now=now) == pytest.approx(120.0, abs=0.1)

    def test_job_that_never_reported_progress_is_not_stalled(self):
        """A job still starting up hasn't moved yet; that isn't a stall."""
        job = self._running_job(last_progress_at=0.0)
        assert job.stalled_for_s() == 0.0

    def test_terminal_job_is_never_stalled(self):
        job = self._running_job(last_progress_at=time.time() - 10_000)
        job.status = "complete"
        assert job.stalled_for_s() == 0.0

    def test_serialized_snapshot_carries_the_stall_and_rate(self):
        now = time.time()
        job = self._running_job(last_progress_at=now - 200.0)
        job.transfer_rate_bps = 1024.0
        d = job.to_dict()
        assert d["stalled_for_s"] > 90.0
        assert d["transfer_rate_bps"] == 1024.0


class TestAtomicWriteJsonConcurrency:
    """Concurrent writers of one path must not destroy each other's temp file.

    Regression for the failure that froze a live transfer's GUI. With a
    shared ``<path>.tmp``, two threads both write the temp, the first
    ``os.replace`` renames it away, and the second raises FileNotFoundError:

        File ".../hub_jobs.py", in atomic_write_json
          os.replace(tmp, path)
        FileNotFoundError: '.../<job>.json.tmp' -> '.../<job>.json'

    That exception killed the worker's progress-writer thread, after which
    the progress file was never updated again and the GUI showed an
    18-minute-stale snapshot of a transfer that was still running.
    """

    def test_concurrent_writers_never_raise(self, tmp_path):
        target = tmp_path / "progress.json"
        errors: list[BaseException] = []
        start = threading.Barrier(8)

        def writer(worker_id: int) -> None:
            start.wait()
            try:
                for i in range(150):
                    hub_jobs.atomic_write_json(target, {"worker": worker_id, "i": i})
            except BaseException as e:  # noqa: BLE001 — the assertion is "nothing escaped"
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(w,)) for w in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert not errors, f"concurrent writers raised: {errors!r}"

    def test_file_is_always_complete_json_never_truncated(self, tmp_path):
        """A reader polling mid-write sees coherent contents, never a partial."""
        target = tmp_path / "progress.json"
        hub_jobs.atomic_write_json(target, {"seed": True})
        stop = threading.Event()
        bad: list[str] = []

        def writer() -> None:
            i = 0
            while not stop.is_set():
                hub_jobs.atomic_write_json(target, {"i": i, "pad": "x" * 2000})
                i += 1

        def reader() -> None:
            while not stop.is_set():
                try:
                    json.loads(target.read_text())
                except json.JSONDecodeError as e:
                    bad.append(str(e))
                except FileNotFoundError:
                    bad.append("progress file vanished")

        threads = [threading.Thread(target=writer) for _ in range(4)]
        threads += [threading.Thread(target=reader) for _ in range(2)]
        for t in threads:
            t.start()
        time.sleep(1.0)
        stop.set()
        for t in threads:
            t.join(timeout=30)

        assert not bad, f"reader saw incoherent state: {bad[:3]}"

    def test_no_temp_files_are_left_behind(self, tmp_path):
        target = tmp_path / "progress.json"
        for i in range(20):
            hub_jobs.atomic_write_json(target, {"i": i})
        assert sorted(p.name for p in tmp_path.iterdir()) == ["progress.json"]

    def test_temp_file_is_removed_when_the_write_fails(self, tmp_path):
        """A failed write must not litter the jobs dir with orphan temps."""

        class _Unserializable:
            pass

        target = tmp_path / "progress.json"
        with pytest.raises(TypeError):
            hub_jobs.atomic_write_json(target, {"bad": _Unserializable()})
        assert list(tmp_path.iterdir()) == []


class TestZombieWorkerDetection:
    """An exited-but-unreaped worker must read as dead, not alive.

    The spawn path drops its ``Popen`` object, so nothing ever waits on
    these children and a finished worker stays in the PID table as a
    zombie. ``os.kill(pid, 0)`` succeeds against a zombie and
    ``/proc/<pid>/stat`` still reports its original start_time, so the
    liveness probe reported crashed workers as healthy indefinitely — the
    "worker exited without finalizing" path could never fire.
    """

    def _zombie(self):
        """A real zombie: a child that exited and is deliberately not waited."""
        import subprocess

        proc = subprocess.Popen(["true"])  # noqa: S603,S607 — fixed argv
        deadline = time.time() + 5
        while time.time() < deadline:
            if hub_jobs._process_is_zombie(proc.pid):
                return proc
            time.sleep(0.02)
        proc.wait()
        pytest.skip("could not produce a zombie on this platform")

    def test_zombie_is_reported_dead(self):
        proc = self._zombie()
        try:
            payload = {"pid": proc.pid, "start_time": None, "started_at": time.time()}
            assert hub_jobs.is_worker_alive(payload) is False
        finally:
            with contextlib.suppress(ChildProcessError):
                proc.wait(timeout=5)

    def test_liveness_probe_reaps_the_zombie(self):
        """Detection also clears the PID-table entry, so they don't accumulate."""
        proc = self._zombie()
        hub_jobs.is_worker_alive({"pid": proc.pid, "start_time": None, "started_at": time.time()})
        assert not hub_jobs._process_is_zombie(proc.pid), "zombie survived the liveness probe"

    def test_reap_if_dead_is_safe_on_a_foreign_pid(self):
        """Never our child: must be a quiet no-op, not an exception."""
        assert hub_jobs.reap_if_dead(os.getpid()) is False

    def test_live_process_is_still_reported_alive(self):
        """The zombie check must not make healthy workers look dead."""
        payload = hub_jobs.pid_file_payload(os.getpid())
        assert hub_jobs.is_worker_alive(payload) is True


class TestCompletenessIgnoresHubManagedFiles:
    """`.gitattributes` must not be reported as missing locally.

    The Hub creates it when the repo is created, so it is always on the
    remote and never in a local dataset directory. Counting it made the
    guardrail fire on every upload to an existing repo, forever, with a
    warning the user could do nothing about — which trains them to click
    past the same dialog that carries the real warnings.
    """

    class _FakeApi:
        def __init__(self, names):
            self._names = names

        def repo_info(self, repo_id, repo_type="dataset", files_metadata=False):
            sibs = [type("S", (), {"rfilename": n})() for n in self._names]
            return type("I", (), {"siblings": sibs})()

    def test_gitattributes_alone_is_not_a_missing_file(self, tmp_path):
        out = hub_jobs.check_upload_completeness(tmp_path, "u/r", api=self._FakeApi([".gitattributes"]))
        assert out["missing_locally"] == []

    def test_real_missing_files_are_still_reported(self, tmp_path):
        out = hub_jobs.check_upload_completeness(
            tmp_path, "u/r", api=self._FakeApi([".gitattributes", "data/chunk-000/file.parquet"])
        )
        assert out["missing_locally"] == ["data/chunk-000/file.parquet"]

    def test_present_local_file_is_not_reported(self, tmp_path):
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "info.json").write_text("{}")
        out = hub_jobs.check_upload_completeness(
            tmp_path, "u/r", api=self._FakeApi([".gitattributes", "meta/info.json"])
        )
        assert out["missing_locally"] == []
