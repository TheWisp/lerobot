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
"""End-to-end tests for the /hub/* HTTP endpoints.

What this file covers:
  * A stalled Hub auth probe does not block unrelated GUI requests
  * POST /hub/upload + /hub/download return job_id without blocking
  * GET /hub/jobs returns sorted list with merged worker progress
  * GET /hub/progress/{id} same shape for one job
  * POST /hub/progress/{id}/cancel signals the worker (verified via state)
  * POST /hub/progress/{id}/dismiss removes the entry
  * Concurrent upload returns 409 with existing job_id
  * Completeness-check 409 path (incomplete_local_state) with confirm_force override
  * Server-startup PID sweep marks orphan workers as failed

We mock subprocess.Popen so tests don't actually fork; the tests verify
the endpoint plumbing + state machine, not the subprocess lifecycle (that
lives in test_hub_worker.py).
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import threading
import time
import types
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from lerobot.gui import hub_jobs
from lerobot.gui.api import datasets as datasets_module
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def app_with_state(tmp_path, monkeypatch):
    """Fresh FastAPI app + AppState + an isolated JOBS_DIR per test."""
    jobs_dir = tmp_path / "hub_jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr(hub_jobs, "JOBS_DIR", jobs_dir)

    app = FastAPI()
    app.include_router(datasets_module.router)
    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    datasets_module.set_app_state(state)

    # Always-OK auth for these tests; specific tests can override per-call.
    import huggingface_hub

    class _FakeApi:
        def whoami(self):
            return {"name": "test"}

        def repo_info(self, repo_id, repo_type="dataset", files_metadata=False):
            class _Info:
                siblings = []

            return _Info()

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    yield app, state, monkeypatch, jobs_dir


def _make_open_dataset(state: AppState, dataset_id: str, root) -> None:
    state.datasets[dataset_id] = types.SimpleNamespace(
        root=str(root),
        repo_id=dataset_id,
        meta=types.SimpleNamespace(total_episodes=0, total_frames=0),
    )


class _FakePopen:
    """Captures subprocess args; pretends to be a long-running worker."""

    instances: list[_FakePopen] = []

    def __init__(self, args, **kwargs):
        self.args = args
        self.env = kwargs.get("env", {})
        self.pid = 9000 + len(_FakePopen.instances)
        self._terminated = False
        _FakePopen.instances.append(self)

    def terminate(self):
        self._terminated = True

    def wait(self, timeout=None):
        return 0


@pytest.fixture(autouse=True)
def reset_fake_popen():
    _FakePopen.instances.clear()
    yield
    _FakePopen.instances.clear()


# ── GET /hub/auth-status ────────────────────────────────────────────────────


class TestHubAuthEndpoint:
    def test_stalled_auth_probe_does_not_block_other_requests(self, app_with_state):
        """A stuck synchronous ``whoami`` must not freeze the ASGI event loop.

        The watchdog makes this test terminate against the old inline-call
        implementation: there, the datasets request cannot run until the
        watchdog releases ``whoami``, and the ordering assertion fails.
        """
        app, _, monkeypatch, _ = app_with_state
        auth_started = threading.Event()
        auth_release = threading.Event()
        watchdog_released = threading.Event()

        import huggingface_hub

        class _BlockingApi:
            def whoami(self):
                auth_started.set()
                auth_release.wait()
                return {"name": "test"}

        monkeypatch.setattr(huggingface_hub, "HfApi", _BlockingApi)

        def release_from_watchdog() -> None:
            watchdog_released.set()
            auth_release.set()

        watchdog = threading.Timer(2.0, release_from_watchdog)
        watchdog.daemon = True
        watchdog.start()

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                auth_request = asyncio.create_task(client.get("/api/datasets/hub/auth-status"))
                datasets_request = asyncio.create_task(client.get("/api/datasets"))
                try:
                    while not auth_started.is_set() and not auth_request.done():
                        await asyncio.sleep(0)
                    assert auth_started.is_set(), "the auth request exited before calling whoami()"
                    response = await datasets_request
                    assert response.status_code == 200
                    assert response.json() == []
                    assert not watchdog_released.is_set(), (
                        "the unrelated datasets request only completed after the "
                        "watchdog released the stalled Hub auth call"
                    )
                finally:
                    auth_release.set()
                    auth_response = await auth_request
                    assert auth_response.status_code == 200
                    assert auth_response.json() == {"logged_in": True, "username": "test"}

        try:
            asyncio.run(run())
        finally:
            auth_release.set()
            watchdog.cancel()


# ── POST /hub/upload, /hub/download ─────────────────────────────────────────


class TestUploadEndpoint:
    def test_returns_job_id_without_blocking(self, app_with_state, tmp_path):
        app, state, monkeypatch, jobs_dir = app_with_state
        ds_root = tmp_path / "ds"
        ds_root.mkdir()
        (ds_root / "data.bin").write_bytes(b"x")
        _make_open_dataset(state, "user/ds", ds_root)

        with patch("subprocess.Popen", _FakePopen):

            async def run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    t0 = time.monotonic()
                    resp = await client.post(
                        "/api/datasets/user%2Fds/hub/upload",
                        json={"repo_id": "user/ds"},
                    )
                    elapsed = time.monotonic() - t0
                    assert elapsed < 1.0
                    assert resp.status_code == 200, resp.text
                    body = resp.json()
                    assert "job_id" in body
                    assert body["status"] == "started"
                    # Job is registered server-side immediately.
                    assert body["job_id"] in state.hub_jobs
                    # Worker subprocess was spawned with the right args.
                    assert len(_FakePopen.instances) == 1
                    proc = _FakePopen.instances[0]
                    assert "lerobot.gui.hub_worker" in " ".join(proc.args)
                    assert "LEROBOT_HUB_WORKER_CONFIG" in proc.env

            asyncio.run(run())

    def test_second_upload_returns_409_with_existing_job_id(self, app_with_state, tmp_path):
        app, state, monkeypatch, jobs_dir = app_with_state
        ds_root = tmp_path / "ds"
        ds_root.mkdir()
        (ds_root / "data.bin").write_bytes(b"x")
        _make_open_dataset(state, "user/ds", ds_root)

        with patch("subprocess.Popen", _FakePopen):

            async def run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    first = await client.post(
                        "/api/datasets/user%2Fds/hub/upload", json={"repo_id": "user/ds"}
                    )
                    assert first.status_code == 200
                    first_job = first.json()["job_id"]
                    state.hub_jobs[first_job].status = "running"  # simulate worker liftoff

                    second = await client.post(
                        "/api/datasets/user%2Fds/hub/upload", json={"repo_id": "user/ds"}
                    )
                    assert second.status_code == 409, second.text
                    assert second.json()["detail"]["job_id"] == first_job

            asyncio.run(run())

    def test_concurrent_uploads_funnel_to_single_spawn(self, app_with_state, tmp_path):
        """N truly-concurrent POSTs on the same dataset → 1 spawn, N-1 conflicts.

        Defends against the "user double-clicks Upload" / "two tabs racing"
        case. The test fires ``n`` POSTs in parallel via asyncio.gather +
        httpx ASGI transport and asserts exactly one Popen actually ran,
        the others all got 409 referencing the winner's job_id.

        What protects the invariant: ``async with _hub_spawn_lock_for(ds)``
        wraps both the active-job check AND the new-job registration.
        Two coroutines cannot both observe ``active_hub_job_for() is None``
        and proceed to spawn.

        Caveat — what this test does NOT distinguish on its own: the
        production critical section happens to be entirely synchronous
        today (no awaits between lock-acquire and lock-release), so
        Python's cooperative scheduling alone funnels concurrent POSTs
        even with the lock removed. The lock is defense-in-depth against
        anyone adding an ``await`` (e.g. an async hub call) inside the
        critical section in the future — if that happens, the post-
        condition this test asserts must continue to hold, and only the
        lock guarantees that.
        """
        app, state, monkeypatch, jobs_dir = app_with_state
        ds_root = tmp_path / "ds"
        ds_root.mkdir()
        (ds_root / "data.bin").write_bytes(b"x")
        _make_open_dataset(state, "user/ds", ds_root)

        n = 5

        with patch("subprocess.Popen", _FakePopen):
            datasets_module._hub_spawn_locks.clear()

            async def post_one(client):
                return await client.post("/api/datasets/user%2Fds/hub/upload", json={"repo_id": "user/ds"})

            async def run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    posts = [asyncio.create_task(post_one(client)) for _ in range(n)]
                    return await asyncio.gather(*posts)

            results = asyncio.run(run())

        statuses = sorted(r.status_code for r in results)
        # Exactly one winner, N-1 conflicts.
        assert statuses == [200] + [409] * (n - 1), [r.text for r in results]

        # The losers all reference the same winning job_id — proves the
        # 409s came from "an active job exists," not from some unrelated
        # error path.
        winner = next(r for r in results if r.status_code == 200).json()["job_id"]
        losers = [r for r in results if r.status_code == 409]
        for loser in losers:
            assert loser.json()["detail"]["job_id"] == winner, loser.text

        # Decisive invariant: exactly one worker subprocess was spawned and
        # one job registered — user clicking Upload twice cannot result in
        # two HF transfers fighting each other.
        assert len(_FakePopen.instances) == 1, (
            f"{len(_FakePopen.instances)} workers spawned for the same dataset"
        )
        assert len(state.hub_jobs) == 1


class TestUploadCompletenessGuardrail:
    """Defends against download-fail-then-upload corruption."""

    def test_missing_locally_returns_409_with_code(self, app_with_state, tmp_path, monkeypatch):
        """When local is missing files present on remote, refuse the upload."""
        app, state, monkeypatch_, jobs_dir = app_with_state
        ds_root = tmp_path / "ds"
        ds_root.mkdir()
        (ds_root / "have.bin").write_bytes(b"x")
        _make_open_dataset(state, "user/ds", ds_root)

        # Override the fake API to advertise an extra remote file.
        import huggingface_hub

        class _FakeApiWithExtra:
            def whoami(self):
                return {"name": "test"}

            def repo_info(self, repo_id, repo_type="dataset", files_metadata=False):
                class _Sib:
                    def __init__(self, name):
                        self.rfilename = name

                class _Info:
                    siblings = [_Sib("have.bin"), _Sib("missing.bin")]

                return _Info()

        monkeypatch_.setattr(huggingface_hub, "HfApi", _FakeApiWithExtra)

        with patch("subprocess.Popen", _FakePopen):

            async def run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.post(
                        "/api/datasets/user%2Fds/hub/upload", json={"repo_id": "user/ds"}
                    )
                    assert resp.status_code == 409
                    detail = resp.json()["detail"]
                    assert detail["code"] == "incomplete_local_state"
                    assert "missing.bin" in detail["missing_locally"]
                    # No worker should have been spawned.
                    assert _FakePopen.instances == []

            asyncio.run(run())

    def test_confirm_force_bypasses_guardrail(self, app_with_state, tmp_path, monkeypatch):
        app, state, monkeypatch_, jobs_dir = app_with_state
        ds_root = tmp_path / "ds"
        ds_root.mkdir()
        (ds_root / "have.bin").write_bytes(b"x")
        _make_open_dataset(state, "user/ds", ds_root)

        import huggingface_hub

        class _FakeApiWithExtra:
            def whoami(self):
                return {"name": "test"}

            def repo_info(self, repo_id, repo_type="dataset", files_metadata=False):
                class _Sib:
                    def __init__(self, name):
                        self.rfilename = name

                class _Info:
                    siblings = [_Sib("have.bin"), _Sib("missing.bin")]

                return _Info()

        monkeypatch_.setattr(huggingface_hub, "HfApi", _FakeApiWithExtra)

        with patch("subprocess.Popen", _FakePopen):

            async def run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.post(
                        "/api/datasets/user%2Fds/hub/upload",
                        json={"repo_id": "user/ds", "confirm_force": True},
                    )
                    assert resp.status_code == 200
                    # Worker spawned despite the guardrail trigger.
                    assert len(_FakePopen.instances) == 1

            asyncio.run(run())


class TestJobsList:
    def test_lists_all_jobs_sorted_newest_first(self, app_with_state):
        app, state, _, _ = app_with_state
        for ds, t in [("a", 100.0), ("c", 300.0), ("b", 200.0)]:
            j = hub_jobs.make_job(dataset_id=ds, direction="upload", repo_id=ds)
            j.started_at = t
            state.hub_jobs[j.job_id] = j

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/datasets/hub/jobs")
                jobs = resp.json()["jobs"]
                assert [j["dataset_id"] for j in jobs] == ["c", "b", "a"]

        asyncio.run(run())

    def test_merges_worker_progress_into_in_memory_state(self, app_with_state, tmp_path):
        app, state, _, jobs_dir = app_with_state
        j = hub_jobs.make_job(dataset_id="u/ds", direction="upload", repo_id="u/ds")
        j.status = "running"
        state.hub_jobs[j.job_id] = j

        # Worker has written progress JSON with files_done_estimate=5.
        paths = hub_jobs.JobPaths.for_job(j.job_id, jobs_dir)
        paths.progress.write_text(
            json.dumps(
                {
                    "status": "running",
                    "milestone": "Uploading files",
                    "milestone_at": 12345.0,
                    "files_total": 10,
                    "files_done_estimate": 5,
                    "pr_num": 42,
                }
            )
        )

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/datasets/hub/jobs")
                jobs = resp.json()["jobs"]
                assert len(jobs) == 1
                assert jobs[0]["files_done_estimate"] == 5
                assert jobs[0]["milestone"] == "Uploading files"
                assert jobs[0]["pr_num"] == 42

        asyncio.run(run())


class TestCancel:
    def test_cancel_signals_worker_via_identity_check(self, app_with_state, tmp_path):
        """Cancel sends SIGTERM to the worker pid recorded in the pid file."""
        app, state, monkeypatch, jobs_dir = app_with_state
        j = hub_jobs.make_job(dataset_id="u/ds", direction="upload", repo_id="u/ds")
        j.status = "running"
        state.hub_jobs[j.job_id] = j

        # Write a PID file that points at the current test process so the
        # identity check succeeds.
        paths = hub_jobs.JobPaths.for_job(j.job_id, jobs_dir)
        paths.pid.write_text(json.dumps(hub_jobs.pid_file_payload(os.getpid())))

        # Intercept os.kill so we don't actually signal ourselves.
        signals_sent: list[tuple[int, int]] = []
        monkeypatch.setattr(os, "kill", lambda pid, sig: signals_sent.append((pid, sig)))

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(f"/api/datasets/hub/progress/{j.job_id}/cancel")
                assert resp.status_code == 200
                assert resp.json()["status"] == "cancel_requested"
                # signal 0 happens first (alive check), then SIGTERM.
                import signal as sigmod

                assert (os.getpid(), sigmod.SIGTERM) in signals_sent

        asyncio.run(run())

    def test_cancel_with_dead_worker_finalises_as_cancelled(self, app_with_state, tmp_path):
        """A PID file naming a dead process is proof the worker is gone.

        This used to finalise as `failed`. On the cancel path that reads
        wrong: the user asked to stop it, and it is stopped. `failed` also
        invites a Retry for something that did not fail. The PID file's
        presence is what makes this knowable — without one the worker may
        merely be starting, which TestCancelBeforeWorkerIsIdentifiable
        covers."""
        app, state, _, jobs_dir = app_with_state
        j = hub_jobs.make_job(dataset_id="u/ds", direction="upload", repo_id="u/ds")
        j.status = "running"
        state.hub_jobs[j.job_id] = j

        import subprocess

        proc = subprocess.Popen(["true"])
        proc.wait()
        paths = hub_jobs.JobPaths.for_job(j.job_id, jobs_dir)
        paths.pid.write_text(json.dumps({"pid": proc.pid, "start_time": None, "started_at": time.time()}))

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(f"/api/datasets/hub/progress/{j.job_id}/cancel")
                assert resp.status_code == 200
                assert state.hub_jobs[j.job_id].status == "cancelled"
                assert state.hub_jobs[j.job_id].error_class == "cancelled"

        asyncio.run(run())


class TestDismiss:
    def test_dismiss_removes_terminal_job_and_files(self, app_with_state, tmp_path):
        app, state, _, jobs_dir = app_with_state
        j = hub_jobs.make_job(dataset_id="ds", direction="upload", repo_id="user/ds")
        j.status = "complete"
        j.finished_at = time.time()
        state.hub_jobs[j.job_id] = j
        paths = hub_jobs.JobPaths.for_job(j.job_id, jobs_dir)
        paths.progress.write_text("{}")
        paths.log.write_text("logs")

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(f"/api/datasets/hub/progress/{j.job_id}/dismiss")
                assert resp.status_code == 200
                assert j.job_id not in state.hub_jobs
                assert not paths.progress.exists()
                assert not paths.log.exists()

        asyncio.run(run())

    def test_dismiss_refuses_active_job(self, app_with_state):
        app, state, _, _ = app_with_state
        j = hub_jobs.make_job(dataset_id="ds", direction="upload", repo_id="user/ds")
        j.status = "running"
        state.hub_jobs[j.job_id] = j

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(f"/api/datasets/hub/progress/{j.job_id}/dismiss")
                assert resp.status_code == 409
                assert j.job_id in state.hub_jobs

        asyncio.run(run())


class TestRetryPRTransfer:
    """Regression for the retry/dismiss interaction bug.

    Before the fix: Retry on a failed upload spawned a new worker with
    reuse_pr_num=N, but the frontend's follow-up Dismiss then called
    change_discussion_status(closed) on that same PR — destroying the
    very PR the new worker was resuming into.

    After the fix: _find_existing_pr_for_retry transfers PR ownership
    by clearing pr_num on every source-entry pointing at the resumed
    PR, so the subsequent dismiss skips the close branch.
    """

    def test_pr_ownership_transferred_off_source_on_retry(self, app_with_state, tmp_path, monkeypatch):
        app, state, monkeypatch_, jobs_dir = app_with_state
        ds_root = tmp_path / "ds"
        ds_root.mkdir()
        (ds_root / "data.bin").write_bytes(b"x")
        _make_open_dataset(state, "user/ds", ds_root)

        # Source entry: a failed upload that left a draft PR behind.
        source = hub_jobs.make_job(dataset_id="user/ds", direction="upload", repo_id="user/ds")
        source.status = "failed"
        source.finished_at = time.time()
        source.pr_num = 42
        state.hub_jobs[source.job_id] = source

        # Patch HfApi so _find_existing_pr_for_retry sees a draft PR.
        import huggingface_hub

        class _FakeApi:
            def whoami(self):
                return {"name": "test"}

            def repo_info(self, *a, **k):
                class _Info:
                    siblings = []

                return _Info()

            def get_discussion_details(self, *a, **k):
                class _Details:
                    status = "draft"

                return _Details()

        monkeypatch_.setattr(huggingface_hub, "HfApi", _FakeApi)

        with patch("subprocess.Popen", _FakePopen):
            datasets_module._hub_spawn_locks.clear()

            async def run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.post(
                        "/api/datasets/user%2Fds/hub/upload", json={"repo_id": "user/ds"}
                    )
                    assert resp.status_code == 200, resp.text
                    new_job_id = resp.json()["job_id"]

                    new_job = state.hub_jobs[new_job_id]
                    # New job inherited pr_num=42.
                    assert new_job.pr_num == 42
                    # Source's pr_num was transferred away — this is the
                    # invariant that makes the follow-up dismiss safe.
                    assert source.pr_num is None, (
                        "pr_num was not transferred off the source; "
                        "subsequent dismiss would close the resumed PR"
                    )

            asyncio.run(run())

    def test_dismiss_after_retry_does_not_close_resumed_pr(self, app_with_state, tmp_path, monkeypatch):
        """End-to-end: source dismiss after retry must not call change_discussion_status."""
        app, state, monkeypatch_, jobs_dir = app_with_state
        ds_root = tmp_path / "ds"
        ds_root.mkdir()
        (ds_root / "data.bin").write_bytes(b"x")
        _make_open_dataset(state, "user/ds", ds_root)

        source = hub_jobs.make_job(dataset_id="user/ds", direction="upload", repo_id="user/ds")
        source.status = "failed"
        source.finished_at = time.time()
        source.pr_num = 42
        state.hub_jobs[source.job_id] = source

        change_status_calls: list[dict] = []

        import huggingface_hub

        class _FakeApi:
            def whoami(self):
                return {"name": "test"}

            def repo_info(self, *a, **k):
                class _Info:
                    siblings = []

                return _Info()

            def get_discussion_details(self, *a, **k):
                class _Details:
                    status = "draft"

                return _Details()

            def change_discussion_status(self, **kwargs):
                change_status_calls.append(kwargs)

        monkeypatch_.setattr(huggingface_hub, "HfApi", _FakeApi)

        with patch("subprocess.Popen", _FakePopen):
            datasets_module._hub_spawn_locks.clear()

            async def run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    # Retry → spawns new worker with reuse_pr_num=42.
                    resp = await client.post(
                        "/api/datasets/user%2Fds/hub/upload", json={"repo_id": "user/ds"}
                    )
                    assert resp.status_code == 200

                    # Mirror the frontend: dismiss the source job after retry.
                    dismiss = await client.post(f"/api/datasets/hub/progress/{source.job_id}/dismiss")
                    assert dismiss.status_code == 200

                    # The critical assertion: dismiss must NOT have closed
                    # the resumed PR.
                    assert change_status_calls == [], (
                        f"dismiss closed the resumed PR — calls: {change_status_calls}"
                    )

            asyncio.run(run())


class TestStartupSweep:
    """Server-startup PID sweep reaps orphan workers from a previous run."""

    def test_sweeps_dead_worker_pid_files(self, app_with_state, tmp_path, monkeypatch):
        app, state, _, jobs_dir = app_with_state
        j = hub_jobs.make_job(dataset_id="ds", direction="upload", repo_id="user/ds")
        j.status = "running"  # pretending the prior server thought it was alive
        state.hub_jobs[j.job_id] = j

        # PID file points at a process that's already dead.
        import subprocess

        proc = subprocess.Popen(["true"])
        proc.wait()
        paths = hub_jobs.JobPaths.for_job(j.job_id, jobs_dir)
        paths.pid.write_text(json.dumps({"pid": proc.pid, "start_time": None, "started_at": time.time()}))

        # Run the sweep (what server.startup_event calls).
        reaped = datasets_module._sweep_orphan_pid_files()
        assert reaped >= 1
        assert state.hub_jobs[j.job_id].status == "failed"
        assert "Worker exited without finalizing" in state.hub_jobs[j.job_id].error
        assert not paths.pid.exists()


class TestCancelEscalation:
    """Cancel must reach a terminal state without further user action.

    Previously cancel sent one SIGTERM and left the job in ``running``. A
    worker blocked inside ``upload_large_folder`` ignores that signal for
    as long as the transfer takes, so the tray kept rendering a normal
    running card and the upload continued to completion.
    """

    def _running_job_with_live_pid(self, state, jobs_dir):
        j = hub_jobs.make_job(dataset_id="u/ds", direction="upload", repo_id="u/ds")
        j.status = "running"
        state.hub_jobs[j.job_id] = j
        paths = hub_jobs.JobPaths.for_job(j.job_id, jobs_dir)
        paths.pid.write_text(json.dumps(hub_jobs.pid_file_payload(os.getpid())))
        return j

    def test_cancel_moves_job_to_cancelling_immediately(self, app_with_state):
        """The status change is what makes the click visible on the next poll."""
        app, state, monkeypatch, jobs_dir = app_with_state
        j = self._running_job_with_live_pid(state, jobs_dir)
        monkeypatch.setattr(os, "kill", lambda pid, sig: None)

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(f"/api/datasets/hub/progress/{j.job_id}/cancel")
                assert resp.status_code == 200
                assert state.hub_jobs[j.job_id].status == "cancelling"
                assert state.hub_jobs[j.job_id].cancel_requested_at is not None

        asyncio.run(run())

    def test_stale_running_snapshot_does_not_undo_the_cancel(self, app_with_state):
        """A poll after cancel must not re-render the job as running."""
        app, state, monkeypatch, jobs_dir = app_with_state
        j = self._running_job_with_live_pid(state, jobs_dir)
        monkeypatch.setattr(os, "kill", lambda pid, sig: None)
        # The worker is still writing "running" while it unwinds.
        paths = hub_jobs.JobPaths.for_job(j.job_id, jobs_dir)
        paths.progress.write_text(json.dumps({"status": "running", "milestone": "Processing files 0 / 1"}))

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                await client.post(f"/api/datasets/hub/progress/{j.job_id}/cancel")
                resp = await client.get("/api/datasets/hub/jobs")
                job = next(x for x in resp.json()["jobs"] if x["job_id"] == j.job_id)
                assert job["status"] == "cancelling"

        asyncio.run(run())

    def test_polling_escalates_to_sigkill_after_the_grace_period(self, app_with_state):
        """A wedged worker is killed by the poll loop, with no second click."""
        app, state, monkeypatch, jobs_dir = app_with_state
        j = self._running_job_with_live_pid(state, jobs_dir)
        signals: list[int] = []
        monkeypatch.setattr(os, "kill", lambda pid, sig: signals.append(sig))

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                await client.post(f"/api/datasets/hub/progress/{j.job_id}/cancel")
                assert signal.SIGKILL not in signals

                # Pretend the grace period has elapsed with the worker
                # still alive — i.e. it swallowed the SIGTERM.
                state.hub_jobs[j.job_id].cancel_requested_at = time.time() - hub_jobs.CANCEL_GRACE_S - 1
                resp = await client.get("/api/datasets/hub/jobs")

                assert signal.SIGKILL in signals
                job = next(x for x in resp.json()["jobs"] if x["job_id"] == j.job_id)
                assert job["status"] == "cancelled"
                assert job["error_class"] == "cancelled"
                assert job["finished_at"] is not None

        asyncio.run(run())

    def test_second_cancel_click_force_kills_without_waiting(self, app_with_state):
        """Clicking again is the user saying the polite path isn't working."""
        app, state, monkeypatch, jobs_dir = app_with_state
        j = self._running_job_with_live_pid(state, jobs_dir)
        signals: list[int] = []
        monkeypatch.setattr(os, "kill", lambda pid, sig: signals.append(sig))

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                url = f"/api/datasets/hub/progress/{j.job_id}/cancel"
                await client.post(url)
                assert signal.SIGKILL not in signals
                resp = await client.post(url)
                assert signal.SIGKILL in signals
                assert resp.json()["job_status"] == "cancelled"

        asyncio.run(run())

    def test_cancel_is_idempotent_once_terminal(self, app_with_state):
        app, state, monkeypatch, jobs_dir = app_with_state
        j = self._running_job_with_live_pid(state, jobs_dir)
        j.status = "cancelled"
        monkeypatch.setattr(os, "kill", lambda pid, sig: pytest.fail("signalled a dead job"))

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(f"/api/datasets/hub/progress/{j.job_id}/cancel")
                assert resp.status_code == 200
                assert state.hub_jobs[j.job_id].status == "cancelled"

        asyncio.run(run())

    def test_dismiss_refuses_a_cancelling_job(self, app_with_state):
        """Its IPC files are still owned by a live worker."""
        app, state, _, jobs_dir = app_with_state
        j = hub_jobs.make_job(dataset_id="u/ds", direction="upload", repo_id="u/ds")
        j.status = "cancelling"
        state.hub_jobs[j.job_id] = j

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(f"/api/datasets/hub/progress/{j.job_id}/dismiss")
                assert resp.status_code == 409

        asyncio.run(run())

    def test_cancelling_job_still_blocks_a_second_upload(self, app_with_state, tmp_path):
        """Two workers must never share one dataset's upload cache + draft PR."""
        app, state, _, jobs_dir = app_with_state
        j = hub_jobs.make_job(dataset_id="u/ds", direction="upload", repo_id="u/ds")
        j.status = "cancelling"
        state.hub_jobs[j.job_id] = j
        assert state.active_hub_job_for("u/ds") is j


class TestHeartbeatFault:
    """A worker that goes silent must become a visible failure, not stay 'running'.

    Regression for the production incident: the worker's progress-writer
    thread died on an unhandled exception while the transfer kept running.
    Process liveness — the server's only health signal at the time — stayed
    true, so the tray rendered a healthy running job backed by an
    18-minute-stale progress file for as long as the user cared to watch.
    """

    def _running_job(self, state, jobs_dir, *, age_s: float):
        j = hub_jobs.make_job(dataset_id="u/ds", direction="upload", repo_id="u/ds")
        j.status = "running"
        j.started_at = time.time() - age_s - 1
        state.hub_jobs[j.job_id] = j
        paths = hub_jobs.JobPaths.for_job(j.job_id, jobs_dir)
        paths.pid.write_text(json.dumps(hub_jobs.pid_file_payload(os.getpid())))
        paths.progress.write_text(json.dumps({"status": "running"}))
        # Backdate the progress file to simulate a dead heartbeat.
        stale = time.time() - age_s
        os.utime(paths.progress, (stale, stale))
        return j

    def test_silent_worker_is_failed_and_killed(self, app_with_state):
        app, state, monkeypatch, jobs_dir = app_with_state
        j = self._running_job(state, jobs_dir, age_s=hub_jobs.HEARTBEAT_FAULT_S + 5)
        signals: list[int] = []
        monkeypatch.setattr(os, "kill", lambda pid, sig: signals.append(sig))

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/datasets/hub/jobs")
                job = next(x for x in resp.json()["jobs"] if x["job_id"] == j.job_id)
                assert job["status"] == "failed"
                assert job["error_class"] == "unresponsive"
                assert job["finished_at"] is not None
                # Left running it would keep uploading invisibly, and a
                # Retry would race a second worker onto the same PR.
                assert signal.SIGKILL in signals

        asyncio.run(run())

    def test_fresh_heartbeat_is_left_alone(self, app_with_state):
        """A healthy job must never be faulted by this check."""
        app, state, monkeypatch, jobs_dir = app_with_state
        j = self._running_job(state, jobs_dir, age_s=0)
        monkeypatch.setattr(os, "kill", lambda pid, sig: pytest.fail("killed a healthy worker"))

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/datasets/hub/jobs")
                job = next(x for x in resp.json()["jobs"] if x["job_id"] == j.job_id)
                assert job["status"] == "running"

        asyncio.run(run())

    def test_slow_transfer_is_stalled_not_faulted(self, app_with_state):
        """Stalled bytes and a dead heartbeat are different faults.

        A worker reporting "no bytes moved for 10 minutes" is working
        correctly — it must be flagged as stalled, never killed.
        """
        app, state, monkeypatch, jobs_dir = app_with_state
        j = self._running_job(state, jobs_dir, age_s=0)
        j.last_progress_at = time.time() - 600
        monkeypatch.setattr(os, "kill", lambda pid, sig: pytest.fail("killed a stalled-but-live worker"))

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/datasets/hub/jobs")
                job = next(x for x in resp.json()["jobs"] if x["job_id"] == j.job_id)
                assert job["status"] == "running"
                assert job["stalled_for_s"] > hub_jobs.STALL_THRESHOLD_S

        asyncio.run(run())

    def test_pending_job_is_never_faulted_for_missing_heartbeat(self, app_with_state):
        """A job whose worker hasn't spawned yet has no heartbeat to miss."""
        app, state, _, jobs_dir = app_with_state
        j = hub_jobs.make_job(dataset_id="u/ds", direction="upload", repo_id="u/ds")
        j.started_at = time.time() - 3600
        state.hub_jobs[j.job_id] = j
        assert j.status == "pending"

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/datasets/hub/jobs")
                job = next(x for x in resp.json()["jobs"] if x["job_id"] == j.job_id)
                assert job["status"] == "pending"

        asyncio.run(run())


class TestTempFileHousekeeping:
    """Unique temp names must not turn into an accumulating leak.

    A shared ``<path>.tmp`` was self-limiting — the next writer overwrote
    it. Per-(pid, thread) names are required so concurrent writers don't
    destroy each other's staging file, but they mean a hard kill between
    the write and the rename (SIGKILL, or the worker's own os._exit paths)
    leaves an orphan that nothing reclaims.
    """

    def test_startup_sweep_removes_stale_temps(self, app_with_state):
        _, _, _, jobs_dir = app_with_state
        stale = jobs_dir / "abc123.json.999.888.tmp"
        stale.write_text("{}")
        old = time.time() - 3600
        os.utime(stale, (old, old))

        assert datasets_module._sweep_orphan_temp_files() == 1
        assert not stale.exists()

    def test_sweep_leaves_in_flight_temps_alone(self, app_with_state):
        """A write happening right now must not have its staging file pulled."""
        _, _, _, jobs_dir = app_with_state
        fresh = jobs_dir / "abc123.json.999.888.tmp"
        fresh.write_text("{}")

        assert datasets_module._sweep_orphan_temp_files() == 0
        assert fresh.exists()

    def test_sweep_never_touches_real_job_files(self, app_with_state):
        _, _, _, jobs_dir = app_with_state
        keep = [jobs_dir / "abc123.json", jobs_dir / "abc123.log", jobs_dir / "abc123.pid"]
        for p in keep:
            p.write_text("{}")
            os.utime(p, (time.time() - 3600, time.time() - 3600))

        datasets_module._sweep_orphan_temp_files()
        assert all(p.exists() for p in keep)

    def test_dismiss_removes_stray_temps_for_that_job(self, app_with_state):
        app, state, _, jobs_dir = app_with_state
        j = hub_jobs.make_job(dataset_id="u/ds", direction="upload", repo_id="u/ds")
        j.status = "cancelled"
        state.hub_jobs[j.job_id] = j
        paths = hub_jobs.JobPaths.for_job(j.job_id, jobs_dir)
        paths.progress.write_text("{}")
        stray = jobs_dir / f"{j.job_id}.json.111.222.tmp"
        stray.write_text("{}")
        other = jobs_dir / "otherjob.json.111.222.tmp"
        other.write_text("{}")

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(f"/api/datasets/hub/progress/{j.job_id}/dismiss")
                assert resp.status_code == 200

        asyncio.run(run())
        assert not stray.exists()
        assert other.exists(), "dismiss must not touch another job's files"


class TestProgressSnapshotCompatibility:
    """A snapshot written by an older worker must still merge.

    Workers are separate processes with their own copy of the code. A
    worker spawned before an upgrade keeps running against the new server,
    so the server has to tolerate a progress file that predates the new
    fields.
    """

    def test_old_format_snapshot_without_new_fields_merges(self):
        j = hub_jobs.make_job(dataset_id="ds", direction="upload", repo_id="u/r")
        j.status = "running"
        j.merge_progress(
            {
                "status": "running",
                "stage": "uploading",
                "milestone": "Processing files 0 / 1",
                "files_total": 3,
                "files_done_estimate": 1,
                "bytes_total": 100,
                "bytes_done_estimate": 40,
            }
        )
        assert j.milestone == "Processing files 0 / 1"
        assert j.bytes_done_estimate == 40
        # Fields the old worker never wrote keep their defaults rather
        # than blowing up or poisoning the readout.
        assert j.transfer_rate_bps == 0.0
        assert j.last_progress_at == 0.0
        # And a job that has never reported progress is not "stalled".
        assert j.to_dict()["stalled_for_s"] == 0.0

    def test_old_format_snapshot_serialises_for_the_tray(self):
        j = hub_jobs.make_job(dataset_id="ds", direction="upload", repo_id="u/r")
        j.status = "running"
        j.merge_progress({"status": "running", "files_total": 2})
        json.dumps(j.to_dict())  # raises if any new field is non-serialisable


class TestDisableXetOption:
    """Per-transfer choice of upload path, not a process-wide env var.

    Whether Xet works is a property of the network path, not of the
    install: on a link where the Xet CAS endpoints stall, a 200 MB upload
    that never completed via Xet finished in 405 s over classic LFS. A
    global flag would also silently apply to every future transfer.

    The flag has to arrive as an environment variable at spawn time —
    huggingface_hub reads it into a module constant at import, so setting
    it inside the worker after any part of the library is imported does
    nothing.
    """

    def _upload(self, app, body):
        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.post("/api/datasets/user%2Fds/hub/upload", json=body)

        return asyncio.run(run())

    def _prepare(self, state, tmp_path):
        ds_root = tmp_path / "ds"
        ds_root.mkdir()
        (ds_root / "data.bin").write_bytes(b"x")
        _make_open_dataset(state, "user/ds", ds_root)

    def test_flag_reaches_the_worker_env(self, app_with_state, tmp_path):
        app, state, _, _ = app_with_state
        self._prepare(state, tmp_path)
        with patch("subprocess.Popen", _FakePopen):
            resp = self._upload(app, {"repo_id": "user/ds", "disable_xet": True})
        assert resp.status_code == 200, resp.text
        assert _FakePopen.instances[-1].env.get("HF_HUB_DISABLE_XET") == "1"

    def test_default_leaves_xet_enabled(self, app_with_state, tmp_path):
        """Absent the option, we must not touch HF's default behaviour."""
        app, state, _, _ = app_with_state
        self._prepare(state, tmp_path)
        with patch("subprocess.Popen", _FakePopen):
            resp = self._upload(app, {"repo_id": "user/ds"})
        assert resp.status_code == 200
        assert "HF_HUB_DISABLE_XET" not in _FakePopen.instances[-1].env

    def test_choice_is_recorded_on_the_job_for_retry(self, app_with_state, tmp_path):
        """The tray reads it back to re-post a retry down the same path."""
        app, state, _, _ = app_with_state
        self._prepare(state, tmp_path)
        with patch("subprocess.Popen", _FakePopen):
            resp = self._upload(app, {"repo_id": "user/ds", "disable_xet": True})
        job = state.hub_jobs[resp.json()["job_id"]]
        assert job.disable_xet is True
        assert job.to_dict()["disable_xet"] is True

    def test_download_ignores_the_flag(self, app_with_state, tmp_path):
        """Not offered for downloads; an unknown field must not 500."""
        app, state, _, _ = app_with_state
        self._prepare(state, tmp_path)

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.post(
                    "/api/datasets/user%2Fds/hub/download",
                    json={"repo_id": "user/ds", "disable_xet": True},
                )

        with patch("subprocess.Popen", _FakePopen):
            resp = asyncio.run(run())
        assert resp.status_code == 200
        assert "HF_HUB_DISABLE_XET" not in _FakePopen.instances[-1].env

    def test_ambient_env_does_not_override_a_xet_selection(self, app_with_state, tmp_path, monkeypatch):
        """Picking Xet must mean Xet, even under a global opt-out.

        The worker inherits the server's environment. A GUI started with
        HF_HUB_DISABLE_XET=1 already exported — the obvious workaround for a
        stalling link, and the one this selector replaces — would otherwise
        pin every transfer to LFS while the modal reported Xet.
        """
        app, state, _, _ = app_with_state
        self._prepare(state, tmp_path)
        monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")
        with patch("subprocess.Popen", _FakePopen):
            resp = self._upload(app, {"repo_id": "user/ds"})
        assert resp.status_code == 200
        assert "HF_HUB_DISABLE_XET" not in _FakePopen.instances[-1].env

    def test_ambient_env_still_allows_an_lfs_selection(self, app_with_state, tmp_path, monkeypatch):
        app, state, _, _ = app_with_state
        self._prepare(state, tmp_path)
        monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")
        with patch("subprocess.Popen", _FakePopen):
            resp = self._upload(app, {"repo_id": "user/ds", "disable_xet": True})
        assert resp.status_code == 200
        assert _FakePopen.instances[-1].env.get("HF_HUB_DISABLE_XET") == "1"


class TestCancelBeforeWorkerIsIdentifiable:
    """Cancel must not mistake a starting worker for a dead one.

    The worker writes its PID file at startup, so a Cancel clicked in the
    first moments of a transfer arrives before there is anything to signal.
    Treating that as "worker exited" ends the job terminal while the worker
    is alive and still uploading — and being terminal stops the poll loop
    escalating and stops it blocking a second transfer on the same dataset,
    so two workers could then run against one upload cache and draft PR.
    """

    def _job_without_pid_file(self, state):
        j = hub_jobs.make_job(dataset_id="u/ds", direction="upload", repo_id="u/ds")
        j.status = "running"  # spawned; worker still starting, no PID file yet
        state.hub_jobs[j.job_id] = j
        return j

    def test_cancel_stays_cancelling_when_no_pid_file_yet(self, app_with_state):
        app, state, _, jobs_dir = app_with_state
        j = self._job_without_pid_file(state)
        assert not hub_jobs.JobPaths.for_job(j.job_id, jobs_dir).pid.exists()

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(f"/api/datasets/hub/progress/{j.job_id}/cancel")
                assert resp.status_code == 200
                assert state.hub_jobs[j.job_id].status == "cancelling", (
                    "a starting worker must not be reported as failed"
                )

        asyncio.run(run())

    def test_that_job_still_blocks_a_second_transfer(self, app_with_state):
        app, state, _, _ = app_with_state
        j = self._job_without_pid_file(state)

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                await client.post(f"/api/datasets/hub/progress/{j.job_id}/cancel")

        asyncio.run(run())
        assert state.active_hub_job_for("u/ds") is j

    def test_escalation_finishes_it_even_if_the_worker_never_appears(self, app_with_state):
        """A worker that died before writing its PID file still terminates."""
        app, state, _, _ = app_with_state
        j = self._job_without_pid_file(state)

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                await client.post(f"/api/datasets/hub/progress/{j.job_id}/cancel")
                state.hub_jobs[j.job_id].cancel_requested_at = time.time() - hub_jobs.CANCEL_GRACE_S - 1
                resp = await client.get("/api/datasets/hub/jobs")
                job = next(x for x in resp.json()["jobs"] if x["job_id"] == j.job_id)
                assert job["status"] == "cancelled"

        asyncio.run(run())

    def test_a_genuinely_dead_worker_is_still_reported_failed(self, app_with_state):
        """The non-cancel paths must keep their fail-fast behaviour."""
        app, state, _, jobs_dir = app_with_state
        j = self._job_without_pid_file(state)
        import signal as sigmod

        sent = datasets_module._send_signal_with_identity_check(j, sigmod.SIGTERM)
        assert sent is False
        assert j.status == "failed"

        asyncio.run(asyncio.sleep(0))


class TestHistoryEndpoint:
    """`/hub/history` answers what `/hub/jobs` structurally cannot.

    The live list drops a job 30 minutes after it finishes and is erased by a
    server restart. The history is a file, so it survives both.
    """

    def _write_history(self, records):
        """Seed the history the endpoint will read.

        Deliberately `history_path()` rather than a path of this test's own:
        that is the one the suite-wide isolation fixture put in place, and the
        endpoint resolves the same way, so the two cannot drift apart.
        """
        from lerobot.gui import hub_history

        p = hub_history.history_path()
        for r in records:
            hub_history.append_outcome(r, path=p)
        return p

    def _get(self, app, url):
        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.get(url)

        return asyncio.run(run())

    def test_returns_past_outcomes_newest_first(self, app_with_state, tmp_path):
        app, _, monkeypatch, _ = app_with_state
        self._write_history(
            [
                {"job_id": "a", "ts": 1.0, "status": "complete", "repo_id": "u/one"},
                {"job_id": "b", "ts": 2.0, "status": "failed", "repo_id": "u/two"},
            ],
        )
        resp = self._get(app, "/api/datasets/hub/history")
        assert resp.status_code == 200
        body = resp.json()
        assert [t["job_id"] for t in body["transfers"]] == ["b", "a"]
        assert body["total"] == 2

    def test_survives_an_empty_registry(self, app_with_state, tmp_path):
        """The point of the feature: the job is long gone from memory."""
        app, state, monkeypatch, _ = app_with_state
        self._write_history([{"job_id": "gone", "ts": 1.0, "status": "complete"}])
        assert state.hub_jobs == {}
        resp = self._get(app, "/api/datasets/hub/history")
        assert [t["job_id"] for t in resp.json()["transfers"]] == ["gone"]

    def test_missing_history_file_is_not_an_error(self, app_with_state):
        app, _, _, _ = app_with_state  # nothing has written the isolated history yet
        resp = self._get(app, "/api/datasets/hub/history")
        assert resp.status_code == 200
        assert resp.json() == {"transfers": [], "total": 0}

    def test_limit_is_clamped(self, app_with_state, tmp_path):
        """An unbounded limit would read and serialise the whole file."""
        app, _, monkeypatch, _ = app_with_state
        self._write_history(
            [{"job_id": f"j{i}", "ts": float(i), "status": "complete"} for i in range(30)],
        )
        assert len(self._get(app, "/api/datasets/hub/history?limit=5").json()["transfers"]) == 5
        assert len(self._get(app, "/api/datasets/hub/history?limit=99999").json()["transfers"]) == 30
        # limit=0 would render an empty list forever; clamped to at least 1.
        assert len(self._get(app, "/api/datasets/hub/history?limit=0").json()["transfers"]) == 1


class TestServerRecordsEndingsTheWorkerCannot:
    """A SIGKILLed worker writes nothing — the server must record for it."""

    def _history(self):
        from lerobot.gui import hub_history

        return hub_history.history_path()

    def _read(self, p):
        if not p.exists():
            return []
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

    def test_forced_cancel_is_recorded(self, app_with_state, tmp_path):
        app, state, monkeypatch, jobs_dir = app_with_state
        hist = self._history()
        j = hub_jobs.make_job(dataset_id="u/ds", direction="upload", repo_id="u/ds")
        j.status = "running"
        state.hub_jobs[j.job_id] = j
        paths = hub_jobs.JobPaths.for_job(j.job_id, jobs_dir)
        paths.pid.write_text(json.dumps(hub_jobs.pid_file_payload(os.getpid())))
        monkeypatch.setattr(os, "kill", lambda pid, sig: None)

        datasets_module._request_cancel(j)
        j.cancel_requested_at = time.time() - hub_jobs.CANCEL_GRACE_S - 1
        assert datasets_module._escalate_cancel_if_overdue(j) is True

        recs = self._read(hist)
        assert [r["status"] for r in recs] == ["cancelled"]
        assert recs[0]["job_id"] == j.job_id

    def test_heartbeat_fault_is_recorded(self, app_with_state, tmp_path):
        app, state, monkeypatch, jobs_dir = app_with_state
        hist = self._history()
        j = hub_jobs.make_job(dataset_id="u/ds", direction="upload", repo_id="u/ds")
        j.status = "running"
        j.started_at = time.time() - hub_jobs.HEARTBEAT_FAULT_S - 10
        state.hub_jobs[j.job_id] = j
        paths = hub_jobs.JobPaths.for_job(j.job_id, jobs_dir)
        paths.pid.write_text(json.dumps(hub_jobs.pid_file_payload(os.getpid())))
        paths.progress.write_text(json.dumps({"status": "running"}))
        stale = time.time() - hub_jobs.HEARTBEAT_FAULT_S - 5
        os.utime(paths.progress, (stale, stale))
        monkeypatch.setattr(os, "kill", lambda pid, sig: None)

        assert datasets_module._fail_if_heartbeat_dead(j) is True
        recs = self._read(hist)
        assert [r["status"] for r in recs] == ["failed"]
        assert recs[0]["error_class"] == "unresponsive"


class TestClearingAListIsNotDestroyingAnArtifact:
    """Clearing a card must never close the PR it could resume from.

    The rule browser download managers follow: removing an entry from the
    downloads panel never deletes the file, and deleting the file is its own
    explicit action. Ours conflated the two — a failed card offered only
    Retry and Discard, so tidying the tray cost the user the draft PR.
    """

    def _failed_job_with_pr(self, state, jobs_dir):
        j = hub_jobs.make_job(dataset_id="u/ds", direction="upload", repo_id="u/ds")
        j.status = "failed"
        j.pr_num = 7
        j.finished_at = time.time()
        state.hub_jobs[j.job_id] = j
        hub_jobs.JobPaths.for_job(j.job_id, jobs_dir).progress.write_text("{}")
        return j

    def _dismiss(self, app, job_id, query=""):
        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.post(f"/api/datasets/hub/progress/{job_id}/dismiss{query}")

        return asyncio.run(run())

    def test_close_pr_false_leaves_the_pr_open(self, app_with_state, tmp_path, monkeypatch):
        app, state, _, jobs_dir = app_with_state
        j = self._failed_job_with_pr(state, jobs_dir)
        import huggingface_hub

        class _Api:
            def get_discussion_details(self, **kw):
                pytest.fail("clearing a card must not touch the PR")

            def change_discussion_status(self, **kw):
                pytest.fail("clearing a card must not close the PR")

        monkeypatch.setattr(huggingface_hub, "HfApi", _Api)

        resp = self._dismiss(app, j.job_id, "?close_pr=false")
        assert resp.status_code == 200
        assert j.job_id not in state.hub_jobs, "the card should still be cleared"

    def test_default_still_closes_the_pr(self, app_with_state, tmp_path, monkeypatch):
        """Discard keeps its teeth; only the new opt-out is non-destructive."""
        app, state, _, jobs_dir = app_with_state
        j = self._failed_job_with_pr(state, jobs_dir)
        closed: list[int] = []
        import huggingface_hub

        class _Api:
            def get_discussion_details(self, **kw):
                return types.SimpleNamespace(status="draft")

            def change_discussion_status(self, **kw):
                closed.append(kw.get("discussion_num"))

        monkeypatch.setattr(huggingface_hub, "HfApi", _Api)

        resp = self._dismiss(app, j.job_id)
        assert resp.status_code == 200
        assert closed == [7], "Discard must still close the draft PR"

    def test_clearing_a_complete_job_is_unaffected(self, app_with_state, tmp_path, monkeypatch):
        app, state, _, jobs_dir = app_with_state
        j = hub_jobs.make_job(dataset_id="u/ds", direction="upload", repo_id="u/ds")
        j.status = "complete"
        j.finished_at = time.time()
        state.hub_jobs[j.job_id] = j
        resp = self._dismiss(app, j.job_id, "?close_pr=false")
        assert resp.status_code == 200
        assert j.job_id not in state.hub_jobs


class TestClearingDoesNotOrphanTheDraftPR:
    """The ✕ promises the draft PR is kept, so uploading again resumes.

    Keeping the PR open on HF is only half of that. PR reuse read the
    in-memory registry, and dismiss deletes the entry — so after a clear the
    PR existed but nothing could find it: no card to Retry or Discard from,
    and the next upload opened a fresh PR and re-sent everything. The
    durable record carries pr_num, so it can answer instead.
    """

    def _cleared_upload(self, state, *, status="failed"):
        from lerobot.gui import hub_history

        hist = hub_history.history_path()
        j = hub_jobs.make_job(dataset_id="/local/ds", direction="upload", repo_id="u/ds")
        j.status = status
        j.pr_num = 7
        j.finished_at = time.time()
        hub_history.append_outcome(hub_history._record_from_job(j), path=hist)
        # The registry entry is gone, exactly as ✕ leaves it.
        state.hub_jobs.clear()
        return j

    def test_retry_recovers_the_pr_from_history(self, app_with_state, tmp_path, monkeypatch):
        app, state, _, _ = app_with_state
        self._cleared_upload(state)
        import huggingface_hub

        class _Api:
            def get_discussion_details(self, **kw):
                return types.SimpleNamespace(status="draft")

        monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
        found = datasets_module._find_existing_pr_for_retry("/local/ds", "u/ds")
        assert found == 7, "a cleared card must not orphan its draft PR"

    def test_a_merged_pr_from_history_is_not_reused(self, app_with_state, tmp_path, monkeypatch):
        """The record can be stale; HF is the authority on the PR's state."""
        app, state, _, _ = app_with_state
        self._cleared_upload(state)
        import huggingface_hub

        class _Api:
            def get_discussion_details(self, **kw):
                return types.SimpleNamespace(status="merged")

        monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
        assert datasets_module._find_existing_pr_for_retry("/local/ds", "u/ds") is None

    def test_a_different_dataset_does_not_reuse_the_pr(self, app_with_state, tmp_path, monkeypatch):
        app, state, _, _ = app_with_state
        self._cleared_upload(state)
        assert datasets_module._find_existing_pr_for_retry("/other/ds", "u/ds") is None

    def test_a_completed_transfer_is_not_a_resume_candidate(self, app_with_state, tmp_path, monkeypatch):
        """Its PR was merged; resuming into it would be wrong."""
        app, state, _, _ = app_with_state
        self._cleared_upload(state, status="complete")
        assert datasets_module._find_existing_pr_for_retry("/local/ds", "u/ds") is None

    def test_unreachable_hf_does_not_break_the_upload(self, app_with_state, tmp_path, monkeypatch):
        app, state, _, _ = app_with_state
        self._cleared_upload(state)
        import huggingface_hub

        class _Api:
            def get_discussion_details(self, **kw):
                raise RuntimeError("hub unreachable")

        monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
        assert datasets_module._find_existing_pr_for_retry("/local/ds", "u/ds") is None


def _write_minimal_dataset(root, repo_id: str):
    """A real on-disk LeRobotDataset, so the endpoint opens what it would in production."""
    import numpy as np

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    }
    ds = LeRobotDataset.create(repo_id=repo_id, fps=10, features=features, root=str(root))
    for _ in range(2):
        ds.add_frame(
            {
                "observation.state": np.zeros(2, np.float32),
                "action": np.zeros(2, np.float32),
                "task": "t",
            }
        )
    ds.save_episode()
    ds.finalize()


class TestTransfersDoNotRequireAnOpenDataset:
    """A Hub transfer needs a directory and a repo id, not a loaded dataset.

    The routes only ever read ``root`` and ``repo_id`` off the object, and the
    worker subprocess does the transfer from the directory. Requiring the object
    tied both routes to the open-dataset registry — process-local state a GUI
    restart clears while the page still shows the dataset open — so uploading
    something plainly visible in the tree returned 404, and later a 500 when a
    fallback was added that had never worked.

    These drive the real routes against a real dataset directory the server has
    never opened, and assert the registry stays empty: the transfer must not
    need it, and must not quietly populate it either.
    """

    @staticmethod
    def _dataset_dir(tmp_path):
        root = tmp_path / "cache" / "owner" / "name"
        root.parent.mkdir(parents=True)
        _write_minimal_dataset(root, "owner/name")
        return root

    def _post(self, app, root, action, **body):
        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.post(
                    f"/api/datasets/{quote(str(root), safe='')}/hub/{action}",
                    json={"repo_id": "owner/name", **body},
                )

        return asyncio.run(run())

    @pytest.mark.parametrize("action", ["upload", "download"])
    def test_a_dataset_never_opened_can_still_transfer(self, app_with_state, tmp_path, action):
        app, state, monkeypatch, jobs_dir = app_with_state
        root = self._dataset_dir(tmp_path)
        assert not state.datasets, "precondition: the server holds nothing"

        with patch("subprocess.Popen", _FakePopen):
            resp = self._post(app, root, action)

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "started"
        assert body["job_id"] in state.hub_jobs, "the worker was actually dispatched"
        assert not state.datasets, "a transfer must not need to load the dataset"

    @pytest.mark.parametrize("action", ["upload", "download"])
    def test_the_worker_is_pointed_at_the_directory(self, app_with_state, tmp_path, action):
        """Deriving the target from the path must still send the right folder."""
        app, state, monkeypatch, jobs_dir = app_with_state
        root = self._dataset_dir(tmp_path)

        with patch("subprocess.Popen", _FakePopen):
            resp = self._post(app, root, action)

        job = state.hub_jobs[resp.json()["job_id"]]
        assert job.repo_id == "owner/name", "repo id derived from <owner>/<name>"
        # The directory reaches the worker through its config, not the job state.
        config = json.loads(_FakePopen.instances[-1].env["LEROBOT_HUB_WORKER_CONFIG"])
        assert Path(config["local_path"]) == root, config

    @pytest.mark.parametrize("action", ["upload", "download"])
    def test_open_under_a_repo_id_but_clicked_from_the_tree(self, app_with_state, tmp_path, action):
        """The identifier the click carries need not be the one it was opened under.

        `DatasetInfo` reports both: `id` is the registry key, `root` is the
        directory. They are the same string for a dataset opened by path and
        differ for one opened by repo id — and the Hub menu is the only caller
        that passes `root`, while episode browsing passes `id`. So a dataset
        could be fully browsable and still fail to upload, which is precisely
        what a user hits and cannot explain.

        Resolving from either identifier is what makes that impossible.
        """
        app, state, monkeypatch, jobs_dir = app_with_state
        root = self._dataset_dir(tmp_path)
        _make_open_dataset(state, "user/opened-name", root)  # keyed by repo id
        assert str(root) not in state.datasets, "the path is not the key here"

        with patch("subprocess.Popen", _FakePopen):
            resp = self._post(app, root, action)  # clicked from the tree: sends the path

        assert resp.status_code == 200, resp.text
        config = json.loads(_FakePopen.instances[-1].env["LEROBOT_HUB_WORKER_CONFIG"])
        assert Path(config["local_path"]) == root

    @pytest.mark.parametrize("action", ["upload", "download"])
    def test_a_path_that_is_not_a_dataset_still_404s(self, app_with_state, tmp_path, action):
        """Dropping the requirement must not turn 'no such dataset' into a transfer."""
        app, state, monkeypatch, jobs_dir = app_with_state
        missing = tmp_path / "not-a-dataset"
        missing.mkdir()

        assert self._post(app, missing, action).status_code == 404

    @pytest.mark.parametrize("action", ["upload", "download"])
    def test_an_opened_dataset_keeps_its_own_repo_id(self, app_with_state, tmp_path, action):
        """The registry still wins when it has an entry: a dataset opened under a
        repo id that does not match its location must not be renamed by its path."""
        app, state, monkeypatch, jobs_dir = app_with_state
        root = self._dataset_dir(tmp_path)
        _make_open_dataset(state, "user/opened-name", root)

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.post(
                    f"/api/datasets/{quote('user/opened-name', safe='')}/hub/{action}", json={}
                )

        with patch("subprocess.Popen", _FakePopen):
            resp = asyncio.run(run())

        assert resp.status_code == 200, resp.text
        assert state.hub_jobs[resp.json()["job_id"]].repo_id == "user/opened-name"


class TestResolveHubTarget:
    """The resolver itself, at its edges.

    The endpoint tests drive it through FastAPI; these pin the function, because
    its two jobs — find the directory, name the repo — fail in different ways and
    only one of them is visible from a route that also takes a repo id.
    """

    @staticmethod
    def _dataset_dir(root):
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text("{}")
        return root

    def test_the_registry_wins_and_keeps_its_own_repo_id(self, app_with_state, tmp_path):
        """An opened dataset's repo id need not match where it sits on disk."""
        app, state, monkeypatch, jobs_dir = app_with_state
        root = self._dataset_dir(tmp_path / "cache" / "owner" / "name")
        _make_open_dataset(state, "someone/else", root)

        assert datasets_module._resolve_hub_target("someone/else") == (root, "someone/else")

    def test_a_path_names_the_repo_from_its_last_two_components(self, app_with_state, tmp_path):
        app, state, monkeypatch, jobs_dir = app_with_state
        root = self._dataset_dir(tmp_path / "cache" / "owner" / "name")

        assert datasets_module._resolve_hub_target(str(root)) == (root, "owner/name")

    def test_a_trailing_slash_is_the_same_target(self, app_with_state, tmp_path):
        app, state, monkeypatch, jobs_dir = app_with_state
        root = self._dataset_dir(tmp_path / "cache" / "owner" / "name")

        assert datasets_module._resolve_hub_target(str(root) + "/") == (root, "owner/name")

    def test_a_path_too_shallow_to_name_a_repo_yields_none(self, app_with_state, tmp_path, monkeypatch):
        """`f"{parent.name}/{name}"` fabricates "/name" for a top-level directory.

        An invalid repo id sent to the worker fails somewhere far from here, so
        the resolver declines to invent one and the route asks to be told.
        """
        app, state, mp, jobs_dir = app_with_state
        self._dataset_dir(tmp_path / "toplevel")
        monkeypatch.chdir(tmp_path)  # so "toplevel" is a relative, one-component path

        resolved_root, repo_id = datasets_module._resolve_hub_target("toplevel")

        assert resolved_root == Path("toplevel")
        assert repo_id is None, "no owner component means no derivable repo id"

    def test_a_directory_without_metadata_is_not_a_dataset(self, app_with_state, tmp_path):
        app, state, monkeypatch, jobs_dir = app_with_state
        (tmp_path / "empty").mkdir()

        with pytest.raises(HTTPException) as exc:
            datasets_module._resolve_hub_target(str(tmp_path / "empty"))
        assert exc.value.status_code == 404

    def test_a_path_that_does_not_exist_is_not_a_dataset(self, app_with_state, tmp_path):
        app, state, monkeypatch, jobs_dir = app_with_state

        with pytest.raises(HTTPException) as exc:
            datasets_module._resolve_hub_target(str(tmp_path / "nope"))
        assert exc.value.status_code == 404

    def test_it_reads_nothing_and_opens_nothing(self, app_with_state, tmp_path, monkeypatch):
        """`.get()` on a plain dict cannot load a dataset or reach the Hub.

        Pinned because the previous implementation *did* construct a
        `LeRobotDataset` here, which reads metadata and resolves against the Hub
        when it cannot satisfy itself locally — on the event loop.
        """
        app, state, mp, jobs_dir = app_with_state
        root = self._dataset_dir(tmp_path / "cache" / "owner" / "name")

        def _explode(*a, **k):
            raise AssertionError("the resolver must not construct a dataset")

        monkeypatch.setattr("lerobot.datasets.lerobot_dataset.LeRobotDataset", _explode)

        assert datasets_module._resolve_hub_target(str(root)) == (root, "owner/name")
        assert not state.datasets, "and it must not register anything"
