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
"""Tests for the Hub-transfer worker subprocess.

We spawn the worker as a real subprocess (not just import + call ``main``)
because the worker:
  - installs SIGTERM/SIGINT handlers that we test against
  - redirects stderr at the file-descriptor level
  - writes its own PID file

All of those need a real process to exercise correctly. We mock the
``huggingface_hub`` calls inside the worker so the tests don't hit the
network, but the IPC + signal handling + PID-file lifecycle are real.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from lerobot.gui import hub_jobs, hub_worker

# ── Test helpers ────────────────────────────────────────────────────────────


_MOCK_HF_MODULE = """
\"\"\"Inserted ahead of huggingface_hub by sys.path manipulation in the test.

Provides controlled mock implementations whose behaviour is steered by env
vars the test sets. The real huggingface_hub is shadowed for the worker's
lifetime; tests that need real HF should not use this fixture.
\"\"\"

import json
import os
import time
import sys


def _mock_config():
    return json.loads(os.environ.get('LEROBOT_HUB_TEST_MOCK_CONFIG', '{}'))


class _PRDetails:
    def __init__(self, num):
        self.num = num
        self.status = 'draft'


class HfApi:
    _create_repo_calls = []
    _pr_num = 1

    def __init__(self, *a, **kw): pass

    def create_repo(self, **kwargs):
        type(self)._create_repo_calls.append(kwargs)

    def create_pull_request(self, **kwargs):
        if _mock_config().get('fail_create_pr'):
            raise RuntimeError('mock: create_pull_request failed')
        type(self)._pr_num += 1
        return _PRDetails(type(self)._pr_num)

    def super_squash_history(self, **kwargs):
        if _mock_config().get('fail_squash'):
            raise TimeoutError('mock: squash timed out (simulated)')

    def change_discussion_status(self, **kwargs):
        if _mock_config().get('fail_change_status'):
            raise RuntimeError('mock: change_discussion_status failed')

    def merge_pull_request(self, **kwargs):
        if _mock_config().get('fail_merge'):
            raise RuntimeError('mock: merge failed')

    def get_discussion_details(self, **kwargs):
        return _PRDetails(kwargs.get('discussion_num', 1))

    def whoami(self):
        return {'name': 'test-user'}


def upload_large_folder(**kwargs):
    if _mock_config().get('upload_forever'):
        # Reproduces the real failure condition: upload_large_folder blocks
        # the main thread for the entire transfer with no interruption
        # point, while emitting Xet-style tqdm bars whose *file* counter
        # never moves and whose *byte* counters do. Verbatim line shapes
        # from an observed multi-GB upload (see TestByteProgress).
        done = 0
        while True:
            done += 1_050_000
            mb = done / 1_000_000
            sys.stderr.write(
                '  ...st/chunk-000/file-001.mp4:   1%|          | '
                f'{mb:.2f}MB /  201MB            \\r'
            )
            sys.stderr.write(
                'Processing Files (0 / 1)      :   1%|          | '
                f'{mb:.2f}MB /  201MB,  749kB/s  \\r'
            )
            sys.stderr.flush()
            time.sleep(0.02)

    sleep_s = float(_mock_config().get('upload_sleep_s', 0.0))
    if sleep_s > 0:
        # Sleep in small increments so a SIGTERM is responsive during the call.
        end = time.time() + sleep_s
        while time.time() < end:
            sys.stderr.write(f'Processing Files (1 / 1)\\r')
            sys.stderr.flush()
            time.sleep(0.05)
    sys.stderr.write('Upload done\\n')
    if _mock_config().get('fail_upload'):
        raise RuntimeError('mock: upload failed')


def snapshot_download(**kwargs):
    sleep_s = float(_mock_config().get('download_sleep_s', 0.0))
    if sleep_s > 0:
        end = time.time() + sleep_s
        while time.time() < end:
            sys.stderr.write(f'Fetching 1 files: 1/1\\r')
            sys.stderr.flush()
            time.sleep(0.05)


from huggingface_hub import errors  # re-export for import compat
"""


@pytest.fixture
def mock_hf_install(tmp_path, monkeypatch):
    """Install a mock ``huggingface_hub`` module at the front of sys.path.

    The fixture writes a fake huggingface_hub package into ``tmp_path``
    that overrides the real one for any subprocess that inherits the
    test's ``PYTHONPATH``. Cleanup is automatic via tmp_path teardown.
    """
    mock_pkg = tmp_path / "mock_hf" / "huggingface_hub"
    mock_pkg.mkdir(parents=True)
    (mock_pkg / "__init__.py").write_text(_MOCK_HF_MODULE)
    # Re-export the errors module from the real HF so our worker can import it.
    (mock_pkg / "errors.py").write_text("from huggingface_hub.errors import *  # noqa: F401, F403\n")
    monkeypatch.setenv("PYTHONPATH", str(mock_pkg.parent) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return mock_pkg.parent


def _build_config(tmp_path: Path, **overrides) -> tuple[hub_jobs.JobConfig, hub_jobs.JobPaths]:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    local_path = tmp_path / "local"
    local_path.mkdir()
    (local_path / "data.bin").write_bytes(b"x")
    defaults = {
        "job_id": "job-test",
        "dataset_id": "ds-1",
        "direction": "upload",
        "repo_id": "user/repo",
        "repo_type": "dataset",
        "local_path": str(local_path),
        "jobs_dir": str(jobs_dir),
        "private": True,
        "commit_message": "Test upload",
    }
    defaults.update(overrides)
    cfg = hub_jobs.JobConfig(**defaults)
    paths = hub_jobs.JobPaths.for_job(cfg.job_id, jobs_dir)
    return cfg, paths


def _spawn_worker(
    cfg: hub_jobs.JobConfig,
    mock_config: dict | None = None,
    *,
    extra_env: dict | None = None,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["LEROBOT_HUB_WORKER_CONFIG"] = cfg.to_json()
    env["LEROBOT_HUB_TEST_MOCK_CONFIG"] = json.dumps(mock_config or {})
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(  # noqa: S603 — args controlled
        [sys.executable, "-m", "lerobot.gui.hub_worker"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _wait_until_status(paths: hub_jobs.JobPaths, *, timeout_s: float = 30.0) -> dict:
    """Poll the progress JSON until status is terminal."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if paths.progress.exists():
            try:
                snap = json.loads(paths.progress.read_text())
                if snap.get("status") in ("complete", "failed", "cancelled"):
                    return snap
            except (OSError, json.JSONDecodeError):
                pass
        time.sleep(0.05)
    raise TimeoutError(f"Worker didn't reach terminal status in {timeout_s}s")


def _wait_for(paths: hub_jobs.JobPaths, predicate, *, timeout_s: float = 15.0) -> dict:
    """Poll the progress JSON until ``predicate(snapshot)`` holds.

    Returns the snapshot that satisfied it. Partial/absent files are
    tolerated — the worker rewrites this file ~2 Hz while we read it.
    """
    deadline = time.time() + timeout_s
    last: dict = {}
    while time.time() < deadline:
        try:
            last = json.loads(paths.progress.read_text())
            if predicate(last):
                return last
        except (OSError, json.JSONDecodeError):
            pass
        time.sleep(0.02)
    raise TimeoutError(f"Predicate unmet in {timeout_s}s; last snapshot: {last}")


def _kill(proc: subprocess.Popen) -> None:
    """Ensure a spawned worker is gone, whatever state the test left it in."""
    if proc.poll() is not None:
        return
    proc.kill()
    proc.wait(timeout=5)


# ── End-to-end worker flows ────────────────────────────────────────────────


class TestWorkerLifecycle:
    """High-level worker behaviour — spawn, run, exit, terminal state visible."""

    def test_successful_upload_writes_complete_status(self, tmp_path, mock_hf_install):
        cfg, paths = _build_config(tmp_path)
        proc = _spawn_worker(cfg, mock_config={"upload_sleep_s": 0.2})
        try:
            snap = _wait_until_status(paths)
            assert snap["status"] == "complete"
            assert snap["error"] is None
        finally:
            proc.wait(timeout=5)

    def test_successful_download_writes_complete_status(self, tmp_path, mock_hf_install):
        cfg, paths = _build_config(tmp_path, direction="download")
        proc = _spawn_worker(cfg, mock_config={"download_sleep_s": 0.1})
        try:
            snap = _wait_until_status(paths)
            assert snap["status"] == "complete"
        finally:
            proc.wait(timeout=5)

    def test_writes_pid_file_with_identity(self, tmp_path, mock_hf_install):
        cfg, paths = _build_config(tmp_path)
        # Long-running upload so the PID file is observable mid-flight.
        proc = _spawn_worker(cfg, mock_config={"upload_sleep_s": 2.0})
        try:
            # Wait for the PID file to appear.
            deadline = time.time() + 5
            while time.time() < deadline and not paths.pid.exists():
                time.sleep(0.05)
            assert paths.pid.exists(), "Worker should write pid file shortly after spawn"
            payload = hub_jobs.read_pid_file(paths.pid)
            assert payload is not None
            assert payload["pid"] == proc.pid
            # Identity check accepts the live worker.
            assert hub_jobs.is_worker_alive(payload)
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_cleans_up_pid_file_after_terminal_exit(self, tmp_path, mock_hf_install):
        cfg, paths = _build_config(tmp_path)
        proc = _spawn_worker(cfg, mock_config={"upload_sleep_s": 0.1})
        proc.wait(timeout=10)
        assert not paths.pid.exists(), "Worker should remove its PID file on exit"

    def test_failure_in_upload_pipeline_classifies_error(self, tmp_path, mock_hf_install):
        cfg, paths = _build_config(tmp_path)
        proc = _spawn_worker(cfg, mock_config={"fail_upload": True})
        try:
            snap = _wait_until_status(paths)
            assert snap["status"] == "failed"
            assert "mock: upload failed" in snap["error"]
            # Generic Exception → "other"
            assert snap["error_class"] == "other"
        finally:
            proc.wait(timeout=5)


class TestSquashFallback:
    """Squash is currently disabled in the worker — see hub_transfers.md.

    The pipeline takes the unsquashed-merge path unconditionally, and must
    still complete when squash would have failed, so re-enabling squash
    (when the HF API issue is resolved) requires an explicit test update.

    This used to assert on the milestone text, which read "Upload complete
    (merged unsquashed)". That coupled an internal branch decision to a
    string the user reads, and the string has since been reworded for
    saying nothing to the user that they can act on. The behaviour is the
    same; only the way it is pinned changed.
    """

    def test_pipeline_completes_without_squashing(self, tmp_path, mock_hf_install):
        cfg, paths = _build_config(tmp_path)
        proc = _spawn_worker(cfg, mock_config={"fail_squash": True})
        try:
            snap = _wait_until_status(paths)
            assert snap["status"] == "complete", f"Expected complete; got {snap}"
            assert snap["stage"] == "done"
            # A squash attempt would have raised from the mock; reaching a
            # clean terminal state is the evidence it was never called.
            assert snap["error"] is None
        finally:
            proc.wait(timeout=5)


class TestCancellation:
    """SIGTERM handling: cancel mid-flight, leave resumable state intact."""

    def test_sigterm_during_upload_yields_cancelled_status(self, tmp_path, mock_hf_install):
        cfg, paths = _build_config(tmp_path)
        # Long-running upload so we can interrupt mid-flight.
        proc = _spawn_worker(cfg, mock_config={"upload_sleep_s": 5.0})
        try:
            # Wait for the upload to actually start.
            deadline = time.time() + 3
            while time.time() < deadline:
                if paths.progress.exists():
                    try:
                        snap = json.loads(paths.progress.read_text())
                        if snap.get("stage") == "uploading":
                            break
                    except (OSError, json.JSONDecodeError):
                        pass
                time.sleep(0.05)

            proc.terminate()  # SIGTERM
            snap = _wait_until_status(paths, timeout_s=15.0)
            assert snap["status"] == "cancelled"
            assert snap["error_class"] == "cancelled"
        finally:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def test_sigkill_escalation_still_leaves_no_zombie(self, tmp_path, mock_hf_install):
        """If the worker ignores SIGTERM, SIGKILL still cleans up the PID file.

        The worker's at-exit hook tries to remove the PID file but SIGKILL
        skips Python's atexit. So we assert the *parent* of the worker
        (the OS) reaps the PID, and that a subsequent is_worker_alive call
        returns False — which is what the server-startup sweep uses to
        reap stale entries.
        """
        cfg, paths = _build_config(tmp_path)
        proc = _spawn_worker(cfg, mock_config={"upload_sleep_s": 10.0})
        try:
            deadline = time.time() + 3
            while time.time() < deadline and not paths.pid.exists():
                time.sleep(0.05)
            payload = hub_jobs.read_pid_file(paths.pid)
            assert payload is not None
            proc.kill()  # SIGKILL — bypasses Python signal handlers
            proc.wait(timeout=5)
            # Process is gone; is_worker_alive recognises that even though
            # the PID file may still exist.
            assert hub_jobs.is_worker_alive(payload) is False
        finally:
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


class TestMissingConfig:
    """Worker should exit cleanly with a useful error when config is missing."""

    def test_no_config_env_var_exits_with_code_2(self, tmp_path):
        env = {k: v for k, v in os.environ.items() if k != "LEROBOT_HUB_WORKER_CONFIG"}
        # NOTE: not using mock_hf_install — we exit before any HF import.
        proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", "lerobot.gui.hub_worker"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _stdout, stderr = proc.communicate(timeout=5)
        assert proc.returncode == 2
        assert b"LEROBOT_HUB_WORKER_CONFIG" in stderr


class TestSignalHandlerNoDeadlock:
    """Regression: SIGTERM handler must not acquire state._lock.

    Python delivers signals synchronously on the main thread between
    bytecodes. If the main thread already holds ``state._lock`` (e.g.,
    is inside ``set_milestone``), a handler that does ``with state._lock``
    would self-deadlock on the non-reentrant lock — wedging the worker
    until SIGKILL.

    We test by directly invoking the handler function with state._lock
    pre-acquired. If the handler still tried to re-acquire, this test
    would block; we wrap in a watchdog thread that fails the test if it
    runs too long.
    """

    def test_handler_completes_while_lock_held(self):
        import threading

        from lerobot.gui import hub_jobs, hub_worker

        cfg = hub_jobs.JobConfig(
            job_id="test-handler",
            dataset_id="user/ds",
            direction="upload",
            repo_id="user/ds",
            repo_type="dataset",
            local_path="/tmp",
            jobs_dir="/tmp",
            private=True,
            commit_message=None,
            allow_patterns=None,
            ignore_patterns=None,
            reuse_pr_num=None,
        )
        paths = hub_jobs.JobPaths.for_job("test-handler", Path("/tmp"))
        state = hub_worker._WorkerState(cfg, paths)

        # Build the same handler function that _install_signal_handlers builds.
        # We don't actually register it with signal.signal (that would mess with
        # the test runner); we just need to verify it doesn't try to acquire
        # state._lock when called while the lock is already held.
        captured: list = []

        def _on_sigterm(signum, frame):
            # Mirror the production handler exactly. If this implementation
            # ever changes to acquire state._lock, this test will deadlock
            # (and fail via the watchdog).
            state.cancel_requested = True
            state.milestone = "cancelling"
            state.milestone_at = time.time()
            captured.append("done")

        # Hold the lock on the "main" (test) thread, then invoke the handler.
        # Signal handlers in production also run on the main thread, so the
        # re-acquire path is what we're regression-testing.
        done = threading.Event()

        def call_handler():
            _on_sigterm(15, None)  # 15 = SIGTERM
            done.set()

        with state._lock:
            t = threading.Thread(target=call_handler)
            t.start()
            # In the buggy version, the handler would block on `with state._lock`
            # while THIS thread holds it — done.wait would time out.
            # (Note: this only catches the bug if the handler tries to acquire
            # synchronously; in real signal delivery the handler runs on the
            # same thread that holds the lock, which is even more deadly.)
            assert done.wait(timeout=1.0), "handler did not complete; suspected re-acquire deadlock"
            t.join(timeout=1.0)

        assert captured == ["done"]
        assert state.cancel_requested is True
        assert state.milestone == "cancelling"

    def test_handler_in_source_does_not_acquire_state_lock(self):
        """Static guard: the production handler must not contain a
        ``with state._lock`` (or any other ``with ... _lock``) acquisition.

        Belt-and-suspenders complement to the runtime test: catches the
        regression at lint time without needing the runtime watchdog to
        fire. We match the executable pattern, not the bare string, so a
        comment explaining the lock-avoidance rule doesn't false-positive.
        """
        import re

        from lerobot.gui import hub_worker

        src = Path(hub_worker.__file__).read_text()
        start = src.find("def _on_sigterm(")
        assert start >= 0, "couldn't locate _on_sigterm"
        body = src[start : start + 2000]
        # Strip comment lines so the rationale comment doesn't match.
        code_only = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))
        assert not re.search(r"with\s+\w+\._lock\s*:", code_only), (
            "SIGTERM handler must not acquire any _lock — Python delivers "
            "signals synchronously on the main thread; re-acquiring a "
            "non-reentrant lock the same thread already holds would deadlock"
        )
        # Also guard against an explicit acquire call.
        assert "_lock.acquire" not in code_only, (
            "SIGTERM handler must not call _lock.acquire() for the same deadlock reason"
        )


class TestWriterShutdownOrdering:
    """Regression: worker main() must stop+join the writer thread BEFORE the
    final write_progress() call.

    Otherwise two threads call atomic_write_json(path, ...) concurrently —
    both writing to the same ``.tmp`` path — and the os.replace can land
    a partially-written tmp, corrupting the terminal progress file the
    server polls.
    """

    def test_main_finally_stops_writer_before_final_write(self):
        """Static guard on the ordering inside main()'s finally block."""
        from lerobot.gui import hub_worker

        src = Path(hub_worker.__file__).read_text()
        # Find main()'s finally block.
        main_start = src.find("def main(")
        assert main_start >= 0, "couldn't locate main()"
        finally_start = src.find("finally:", main_start)
        assert finally_start >= 0, "couldn't locate finally: in main()"
        # The end of main()'s finally is bounded by the next top-level def
        # or 'return rc'.
        end = src.find("return rc", finally_start)
        assert end >= 0
        body = src[finally_start:end]
        stop_pos = body.find("stop_writer_thread()")
        join_pos = body.find("writer.join(")
        final_write_pos = body.find("state.write_progress()")
        assert stop_pos >= 0, "stop_writer_thread() call missing"
        assert join_pos >= 0, "writer.join() call missing"
        assert final_write_pos >= 0, "final state.write_progress() call missing"
        assert stop_pos < final_write_pos, (
            "stop_writer_thread() must be called BEFORE the final "
            "state.write_progress() — otherwise the writer thread can race "
            "the main thread on the same .tmp path under atomic_write_json"
        )
        assert join_pos < final_write_pos, (
            "writer.join() must complete BEFORE the final state.write_progress() — same race-on-tmp concern"
        )


# ── Capture both stdout and stderr in the per-job log ──────────────────────


# Mock that writes to BOTH stdout and stderr so we can verify the worker
# captures both into the same per-job log file. Without dup'ing fd 1 too,
# the stdout writes would silently disappear (server spawns the worker
# with stdout=DEVNULL) — exactly the leak that hid HF's rate-limit
# messages from view in the original bug investigation.
_MOCK_HF_STDOUT_AND_STDERR = """
import json
import os
import sys

def _mock_config():
    return json.loads(os.environ.get('LEROBOT_HUB_TEST_MOCK_CONFIG', '{}'))


class _PRDetails:
    def __init__(self, num):
        self.num = num


class HfApi:
    def create_repo(self, **kwargs): return None
    def create_pull_request(self, **kwargs): return _PRDetails(1)
    def change_discussion_status(self, **kwargs): return None
    def merge_pull_request(self, **kwargs): return None
    def super_squash_history(self, **kwargs): return None
    def whoami(self): return {'name': 'test-user'}


def upload_large_folder(**kwargs):
    # Emit a recognizable marker to each stream. The captured log must
    # contain BOTH or the dup is broken.
    sys.stdout.write('MARKER_STDOUT_from_hf\\n')
    sys.stdout.flush()
    sys.stderr.write('MARKER_STDERR_from_hf\\n')
    sys.stderr.flush()


def snapshot_download(**kwargs):
    sys.stdout.write('MARKER_STDOUT_from_hf\\n')
    sys.stdout.flush()
    sys.stderr.write('MARKER_STDERR_from_hf\\n')
    sys.stderr.flush()


from huggingface_hub import errors  # re-export for import compat
"""


@pytest.fixture
def mock_hf_stdout_and_stderr(tmp_path, monkeypatch):
    """Variant of the mock_hf_install fixture whose mock emits to both
    streams, so we can prove the worker captures both into ``paths.log``.
    """
    mock_pkg = tmp_path / "mock_hf" / "huggingface_hub"
    mock_pkg.mkdir(parents=True)
    (mock_pkg / "__init__.py").write_text(_MOCK_HF_STDOUT_AND_STDERR)
    (mock_pkg / "errors.py").write_text("from huggingface_hub.errors import *  # noqa: F401, F403\n")
    monkeypatch.setenv("PYTHONPATH", str(mock_pkg.parent) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return mock_pkg.parent


class TestCapturesStdoutAndStderr:
    """Both stdout and stderr from the HF library land in the per-job log.

    The original visibility bug had HF's rate-limit error message in
    stderr (where we already captured it), but parts of HF's library
    output go to stdout. The server spawns the worker with stdout=DEVNULL,
    so without an explicit fd-1 dup the worker's own stdout would be
    sent into the void. This test guards against silently regressing
    that dup.
    """

    def test_marker_from_stderr_is_in_log(self, tmp_path, mock_hf_stdout_and_stderr):
        cfg, paths = _build_config(tmp_path)
        proc = _spawn_worker(cfg)
        try:
            snap = _wait_until_status(paths, timeout_s=10)
        finally:
            proc.wait(timeout=5)
        assert snap["status"] == "complete"
        log_text = paths.log.read_text()
        assert "MARKER_STDERR_from_hf" in log_text, f"stderr marker missing from log; got:\n{log_text[-500:]}"

    def test_marker_from_stdout_is_in_log(self, tmp_path, mock_hf_stdout_and_stderr):
        """The regression: without dup2(w_fd, 1), this assertion fails
        because the server's stdout=DEVNULL eats the marker.
        """
        cfg, paths = _build_config(tmp_path)
        proc = _spawn_worker(cfg)
        try:
            snap = _wait_until_status(paths, timeout_s=10)
        finally:
            proc.wait(timeout=5)
        assert snap["status"] == "complete"
        log_text = paths.log.read_text()
        assert "MARKER_STDOUT_from_hf" in log_text, (
            f"stdout marker missing from log — worker is not dup'ing fd 1 "
            f"into the same pipe as fd 2. Got:\n{log_text[-500:]}"
        )


# ── Fail-fast httpx hook on unrecoverable HF responses ─────────────────────


class _Synthetic429Transport:
    """httpx transport stub that responds to every request with 429.

    Used to drive the fail-fast hook without monkey-patching
    huggingface_hub internals beyond the public-but-private transport
    swap. The Retry-After + body mirror what HF actually returns on a
    repo-commit-rate-limit so the assertion on the surfaced message
    text exercises real wording.
    """

    def __init__(self, status: int = 429, retry_after: str = "130") -> None:
        self.status = status
        self.retry_after = retry_after

    def handle_request(self, request):  # signature matches httpx.BaseTransport
        import httpx

        body = (
            b"429 Too Many Requests: you have reached your 'api' rate limit. "
            b"Retry after 130 seconds. "
            b"You have exceeded the rate limit for repository commits (128 per hour)."
        )
        return httpx.Response(
            status_code=self.status,
            headers={"Retry-After": self.retry_after, "Content-Type": "text/plain"},
            content=body,
            request=request,
        )


class TestFatalHttpHookUnit:
    """Direct exercise of the worker's httpx-hook helpers, no subprocess.

    Production uses an abort callback that os._exit's the process, so
    these tests use the default (raising) install variant. The abort
    helper itself is covered by a separate subprocess-level test that
    actually invokes it and asserts the resulting on-disk state.
    """

    def _uninstall_all(self, client):
        """Strip every tagged hook from the shared HF client between tests."""
        client.event_hooks["response"] = [
            h for h in client.event_hooks.get("response", []) if not getattr(h, "_lerobot_fatal_hook", False)
        ]

    def test_classify_response_pure_function(self):
        """The classifier is the unit that decides "is this fatal" and
        builds the message HF's text gets stitched into. Cover it directly
        so the message-construction contract is pinned independently of
        the hook install plumbing.
        """
        import httpx

        from lerobot.gui.hub_worker import _classify_response

        req = httpx.Request("POST", "https://huggingface.co/api/test")

        resp_429 = httpx.Response(429, headers={"Retry-After": "200"}, content=b"go away", request=req)
        exc = _classify_response(resp_429)
        assert exc is not None
        assert exc.status == 429
        assert exc.error_class == "rate_limit"
        assert "Retry-After: 200s" in exc.message
        assert "go away" in exc.message

        # Non-fatal short-circuits before touching the body.
        assert _classify_response(httpx.Response(200, content=b"ok", request=req)) is None

        # 5xx deliberately ignored — HF's adaptive retry was designed for
        # 504 commit timeouts and intercepting would defeat it.
        assert _classify_response(httpx.Response(504, content=b"timeout", request=req)) is None
        assert _classify_response(httpx.Response(500, content=b"boom", request=req)) is None

        # 4xx are all fatal — the library only changes the batch SIZE on
        # retry, not the request CONTENTS, so a malformed-request 400/422
        # will fail identically on every retry until the rate limit fires.
        # Pin the specific case we observed in the wild (LFS pointer
        # corruption surfaces as 400 with that body text).
        resp_400 = httpx.Response(
            400, content=b"LFS pointer pointed to a file that does not exist", request=req
        )
        exc_400 = _classify_response(resp_400)
        assert exc_400 is not None
        assert exc_400.status == 400
        assert exc_400.error_class == "bad_request"
        assert "LFS pointer" in exc_400.message

        resp_422 = httpx.Response(422, content=b"validation failed", request=req)
        exc_422 = _classify_response(resp_422)
        assert exc_422 is not None
        assert exc_422.error_class == "bad_request"

    def test_default_hook_raises_on_429(self):
        """Calling install() with no callback registers a hook that
        raises _FatalHFError — useful for tests that want to assert
        against the exception type without going through process exit.
        """
        import huggingface_hub
        from huggingface_hub.utils._http import get_session

        from lerobot.gui.hub_worker import _FatalHFError, _install_fatal_http_hook

        client = get_session()
        original_transport = client._transport
        client._transport = _Synthetic429Transport()
        _install_fatal_http_hook()
        try:
            with pytest.raises(_FatalHFError) as exc_info:
                huggingface_hub.HfApi().create_repo(
                    repo_id="not-real/repro", repo_type="dataset", exist_ok=True
                )
        finally:
            client._transport = original_transport
            self._uninstall_all(client)

        assert exc_info.value.status == 429
        assert exc_info.value.error_class == "rate_limit"
        msg = exc_info.value.message
        assert "Retry-After: 130" in msg
        assert "128 per hour" in msg, f"verbatim HF text should pass through; got {msg!r}"

    def test_abort_callback_receives_constructed_exception(self):
        """The production install path: pass an on_fatal callback (e.g.
        the closure that calls _abort_to_terminal_state). Verify the
        callback gets called with a fully-populated _FatalHFError on a
        fatal response.
        """
        import huggingface_hub
        from huggingface_hub.utils._http import get_session

        from lerobot.gui.hub_worker import _FatalHFError, _install_fatal_http_hook

        captured: list[_FatalHFError] = []

        def _record(exc: _FatalHFError) -> None:
            captured.append(exc)
            # Raise something Exception-derived so HF's call returns
            # (HF catches Exception in its worker thread; this lets
            # the test continue rather than hang).
            raise RuntimeError("test stub raised after recording")

        client = get_session()
        original_transport = client._transport
        client._transport = _Synthetic429Transport()
        _install_fatal_http_hook(_record)
        try:
            with pytest.raises(RuntimeError):
                huggingface_hub.HfApi().create_repo(
                    repo_id="not-real/repro", repo_type="dataset", exist_ok=True
                )
        finally:
            client._transport = original_transport
            self._uninstall_all(client)

        assert len(captured) == 1
        exc = captured[0]
        assert exc.status == 429
        assert exc.error_class == "rate_limit"
        assert "128 per hour" in exc.message

    def test_install_is_idempotent_across_callback_variants(self):
        """Calling install repeatedly — with or without a callback — must
        not stack multiple hooks on the shared HF client. We tag by
        attribute so dedup is robust to closure identity differences.
        """
        from huggingface_hub.utils._http import get_session

        from lerobot.gui.hub_worker import _install_fatal_http_hook

        client = get_session()
        try:
            _install_fatal_http_hook()
            _install_fatal_http_hook(lambda exc: None)
            _install_fatal_http_hook()
            tagged = [
                h for h in client.event_hooks.get("response", []) if getattr(h, "_lerobot_fatal_hook", False)
            ]
            assert len(tagged) == 1, f"install should be idempotent — got {len(tagged)} tagged hooks"
        finally:
            self._uninstall_all(client)


class TestAbortToTerminalState:
    """``_abort_to_terminal_state`` is what production hooks call when
    they detect a fatal HF response. It must (1) mark the in-memory
    state as failed with the right fields, (2) write the terminal
    snapshot to disk synchronously, (3) remove the pid file so the
    server's PID-liveness sweep doesn't overwrite our error_class
    with a generic one, and (4) os._exit(1) so the wedged main thread
    inside HF's library doesn't keep the process alive. We exercise
    the helper directly with os._exit monkey-patched so it raises
    SystemExit instead of killing the test process.
    """

    def _build_state(self, tmp_path):
        from lerobot.gui.hub_jobs import JobConfig, JobPaths
        from lerobot.gui.hub_worker import _WorkerState

        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        cfg = JobConfig(
            job_id="job-test",
            dataset_id="ds",
            direction="upload",
            repo_id="u/r",
            repo_type="dataset",
            local_path=str(tmp_path),
            jobs_dir=str(jobs_dir),
        )
        paths = JobPaths.for_job(cfg.job_id, jobs_dir)
        # Write a pid file so we can verify the abort removes it.
        paths.pid.write_text('{"pid": 99999, "started_at": 0, "start_time": 0}')
        return _WorkerState(cfg, paths)

    def test_abort_writes_failed_state_and_calls_exit(self, tmp_path, monkeypatch):
        from lerobot.gui import hub_worker
        from lerobot.gui.hub_worker import _abort_to_terminal_state, _FatalHFError

        state = self._build_state(tmp_path)
        exc = _FatalHFError(429, "rate_limit", "HTTP 429 — 128 per hour cap exceeded")

        # Replace os._exit with one that raises SystemExit so we can
        # assert post-state. The production behavior is process death.
        called_with: list[int] = []

        def _fake_exit(code: int) -> None:
            called_with.append(code)
            raise SystemExit(code)

        monkeypatch.setattr(hub_worker.os, "_exit", _fake_exit)

        with pytest.raises(SystemExit) as exit_info:
            _abort_to_terminal_state(state, exc)

        assert exit_info.value.code == 1
        assert called_with == [1]

        # In-memory state mutated to terminal failed.
        assert state.status == "failed"
        assert state.error_class == "rate_limit"
        assert "128 per hour" in state.error
        assert state.finished_at is not None
        assert "Failed" in state.milestone

        # Progress JSON on disk reflects the same terminal state — this is
        # what the GUI server polls, so the user sees the real reason.
        import json as _json

        snap = _json.loads(state.paths.progress.read_text())
        assert snap["status"] == "failed"
        assert snap["error_class"] == "rate_limit"
        assert "128 per hour" in snap["error"]

        # PID file removed so PID-liveness sweep doesn't second-guess us.
        assert not state.paths.pid.exists()


# ── Byte-level progress extraction ─────────────────────────────────────────
#
# Regression cover for a transfer that ran for 37 minutes while the tray
# showed a frozen "Processing files 0 / 1" and a 0% bar. Nothing was stuck:
# the worker never populated any byte counter, and the one string it did
# publish was a file count that legitimately doesn't move until a
# multi-hundred-MB file finishes. The user read that as a hang and
# cancelled a healthy upload.


# Verbatim lines from the observed incident's per-job log. Three HF worker
# threads, each with its own XetProgressReporter, interleaved on one stream —
# note the three different "Processing Files" totals, which is why the
# summary bars can't simply be believed as-is.
_REAL_UPLOAD_EXCERPT = [
    "  ...st/chunk-000/file-008.mp4:   1%|          | 1.05MB /  203MB            ",
    "Processing Files (0 / 1)      :   1%|          | 1.05MB /  203MB,  749kB/s  ",
    "  ...st/chunk-000/file-001.mp4:   1%|          | 1.05MB /  201MB            ",
    "Processing Files (0 / 1)      :   1%|          | 1.05MB /  201MB,  749kB/s  ",
    "  ...op/chunk-000/file-002.mp4:   1%|          | 1.05MB /  183MB            ",
    "Processing Files (0 / 1)      :   0%|          | 1.05MB / 3.09GB,  655kB/s  ",
    "  ...st/chunk-000/file-008.mp4:  11%|█         | 22.6MB /  203MB            ",
    "  ...st/chunk-000/file-001.mp4:   6%|▌         | 11.5MB /  201MB            ",
    "  ...op/chunk-000/file-002.mp4:  28%|██▊       | 50.3MB /  183MB            ",
]


class TestParseBarLine:
    """Parsing one rendered tqdm bar out of HF's captured output."""

    def test_per_file_bar(self):
        s = hub_worker.parse_bar_line(_REAL_UPLOAD_EXCERPT[0])
        assert s is not None
        assert s.is_summary is False
        assert s.desc == "...st/chunk-000/file-008.mp4"
        # HF renders with unit_divisor=1000, so these are decimal sizes.
        assert s.done == 1_050_000
        assert s.total == 203_000_000
        assert s.rate_bps is None

    def test_summary_bar_carries_rate(self):
        s = hub_worker.parse_bar_line(_REAL_UPLOAD_EXCERPT[1])
        assert s is not None
        assert s.is_summary is True
        assert s.rate_bps == 749_000

    def test_new_data_upload_is_a_summary_bar(self):
        s = hub_worker.parse_bar_line(
            "New Data Upload               :   2%|▏         | 4.20MB / 67.0MB, 2.62MB/s  "
        )
        assert s is not None and s.is_summary is True
        assert s.done == 4_200_000 and s.rate_bps == 2_620_000

    def test_gigabyte_suffix(self):
        s = hub_worker.parse_bar_line(_REAL_UPLOAD_EXCERPT[5])
        assert s is not None and s.total == 3_090_000_000

    def test_leading_cursor_up_escape_is_tolerated(self):
        """tqdm emits \\x1b[A between bars; a split record can start with one."""
        s = hub_worker.parse_bar_line("\x1b[A" + _REAL_UPLOAD_EXCERPT[0])
        assert s is not None and s.desc == "...st/chunk-000/file-008.mp4"

    @pytest.mark.parametrize(
        "line",
        [
            "",
            "Upload done",
            "Recovering from metadata files: 100%|##########| 84/84 [00:00<00:00, 9401.51it/s]",
            "--- terminal exception ---",
        ],
    )
    def test_non_bar_lines_are_ignored(self, line):
        assert hub_worker.parse_bar_line(line) is None


class TestTransferProgress:
    """Aggregation across interleaved reporters and reused bar slots."""

    def test_real_excerpt_yields_moving_bytes_while_file_count_is_static(self):
        """The exact condition that looked like a hang.

        Every "Processing Files" line in the excerpt says 0 / 1 — the file
        counter the old tray rendered — yet 84.4 MB of real movement is
        recoverable from the same lines.
        """
        p = hub_worker.TransferProgress()
        for line in _REAL_UPLOAD_EXCERPT:
            p.feed(line)

        # 22.6 + 11.5 + 50.3 MB across the three in-flight files.
        assert p.bytes_done == 84_400_000
        # ...while the milestone string the old code published never moved.
        milestones = {
            hub_worker.extract_milestone(line, "upload")
            for line in _REAL_UPLOAD_EXCERPT
            if hub_worker.extract_milestone(line, "upload")
        }
        assert milestones == {"Processing files 0 / 1"}

    def test_bytes_done_is_monotonic_over_the_full_incident_log(self):
        """Interleaved reporters must never make the bar run backwards."""
        p = hub_worker.TransferProgress()
        previous = 0
        # Replay the excerpt repeatedly with the summary bars out of order,
        # which is how concurrent reporters actually arrive.
        for _ in range(3):
            for line in reversed(_REAL_UPLOAD_EXCERPT):
                p.feed(line)
                assert p.bytes_done >= previous
                previous = p.bytes_done

    def test_feed_reports_whether_progress_advanced(self):
        p = hub_worker.TransferProgress()
        assert p.feed(_REAL_UPLOAD_EXCERPT[0]) is True
        # Same line again: no new bytes, so no advance.
        assert p.feed(_REAL_UPLOAD_EXCERPT[0]) is False
        assert p.feed("not a progress bar") is False

    def test_reused_bar_slot_does_not_double_count(self):
        """tqdm reuses a bar for the next file; keys are descriptions."""
        p = hub_worker.TransferProgress()
        p.feed("  a.mp4:  50%|x| 50.0MB /  100MB            ")
        p.feed("  a.mp4: 100%|x|  100MB /  100MB            ")
        p.feed("  b.mp4:  10%|x| 10.0MB /  100MB            ")
        assert p.bytes_done == 110_000_000
        assert p.files_done == 1
        assert p.current_file == "b.mp4"

    def test_summary_bar_is_a_floor_when_no_per_file_bars_exist(self):
        p = hub_worker.TransferProgress()
        p.feed("Processing Files (0 / 3)      :  10%|x| 40.0MB /  400MB,  749kB/s  ")
        assert p.bytes_done == 40_000_000

    def test_rate_is_measured_over_the_window_not_taken_from_hf(self):
        """HF's postfix is per-reporter; we measure the aggregate ourselves.

        The clock is held still at the last observation so this isolates the
        measurement from the stall-decay behaviour, which
        ``TestRateDecaysDuringStall`` covers separately.
        """
        now = [0.0]
        p = hub_worker.TransferProgress(clock=lambda: now[0])
        now[0] = 0.0
        p.feed("  a.mp4:   1%|x| 10.0MB /  100MB            ")
        now[0] = 1.0
        p.feed("  a.mp4:   2%|x| 20.0MB /  100MB            ")
        now[0] = 2.0
        p.feed("  a.mp4:   3%|x| 30.0MB /  100MB            ")
        # 20 MB gained over the 2 seconds spanned by the observations.
        assert p.rate_bps() == pytest.approx(10_000_000, rel=0.01)

    def test_rate_is_zero_before_two_observations(self):
        p = hub_worker.TransferProgress()
        assert p.rate_bps() == 0.0
        p.feed(_REAL_UPLOAD_EXCERPT[0])
        assert p.rate_bps() == 0.0

    def test_last_progress_at_only_moves_on_real_movement(self):
        """The stall clock must not be reset by repeated identical ticks."""
        ticks = iter([100.0, 200.0, 300.0])
        p = hub_worker.TransferProgress(clock=lambda: next(ticks))
        p.feed("  a.mp4:   1%|x| 10.0MB /  100MB            ")
        assert p.last_progress_at == 100.0
        for _ in range(50):
            p.feed("  a.mp4:   1%|x| 10.0MB /  100MB            ")
        assert p.last_progress_at == 100.0, "redundant tqdm redraws faked liveness"


class TestWorkerPublishesByteProgress:
    """End-to-end: a running worker's progress JSON carries real numbers."""

    def test_progress_json_has_totals_and_moving_bytes(self, tmp_path, mock_hf_install):
        cfg, paths = _build_config(tmp_path)
        # Give the upload root some real bytes to enumerate.
        (Path(cfg.local_path) / "data.bin").write_bytes(b"x" * 4096)
        proc = _spawn_worker(cfg, mock_config={"upload_forever": True})
        try:
            snap = _wait_for(
                paths,
                lambda s: s.get("bytes_done_estimate", 0) > 0,
                timeout_s=15.0,
            )
            # The denominator the tray needs for a percentage — previously
            # hardcoded to 0 for the lifetime of every transfer.
            assert snap["files_total"] == 1
            assert snap["bytes_total"] == 4096
            assert snap["bytes_done_estimate"] > 0
            assert snap["last_progress_at"] > 0

            # And it keeps moving.
            first = snap["bytes_done_estimate"]
            later = _wait_for(
                paths,
                lambda s: s.get("bytes_done_estimate", 0) > first,
                timeout_s=15.0,
            )
            assert later["transfer_rate_bps"] > 0
            # Meanwhile the old readout is still frozen — which is exactly
            # why it can't be the thing the user is asked to interpret.
            assert later["milestone"] == "Processing files 0 / 1"
        finally:
            _kill(proc)


# ── Cancelling a transfer that will not stop on its own ────────────────────


class TestCancelForcesTermination:
    """SIGTERM must end the transfer even inside an uninterruptible call.

    ``upload_large_folder`` returns only when the whole transfer finishes,
    so the pre-existing "set a flag, check it between pipeline stages"
    cancel was unobservable in the case that matters: the observed incident
    had the worker still uploading 37 minutes after the user cancelled.
    """

    GRACE = {"LEROBOT_HUB_CANCEL_GRACE_S": "1.0"}

    def test_sigterm_ends_an_upload_that_never_returns(self, tmp_path, mock_hf_install):
        cfg, paths = _build_config(tmp_path)
        proc = _spawn_worker(cfg, mock_config={"upload_forever": True}, extra_env=self.GRACE)
        try:
            _wait_for(paths, lambda s: s.get("bytes_done_estimate", 0) > 0, timeout_s=15.0)
            proc.terminate()

            snap = _wait_until_status(paths, timeout_s=15.0)
            assert snap["status"] == "cancelled"
            assert snap["error_class"] == "cancelled"
            # The worker force-exits itself rather than waiting for the
            # server's SIGKILL; 130 is the conventional user-terminated code.
            assert proc.wait(timeout=5) == 130
            # PID file dropped so the server's liveness sweep doesn't
            # overwrite our specific outcome with a generic failure.
            assert not paths.pid.exists()
        finally:
            _kill(proc)

    def test_cancelling_milestone_is_not_clobbered_by_tqdm_ticks(self, tmp_path, mock_hf_install):
        """The "showed cancelling then nothing happened" symptom.

        The output reader publishes a milestone every few milliseconds. Once
        a cancel is requested the pinned milestone has to win, or the tray
        reverts to a routine progress string before the next poll.
        """
        cfg, paths = _build_config(tmp_path)
        # Long grace so we can observe the pre-terminal window.
        proc = _spawn_worker(
            cfg,
            mock_config={"upload_forever": True},
            extra_env={"LEROBOT_HUB_CANCEL_GRACE_S": "30.0"},
        )
        try:
            _wait_for(paths, lambda s: s.get("bytes_done_estimate", 0) > 0, timeout_s=15.0)
            proc.terminate()

            snap = _wait_for(paths, lambda s: s.get("milestone") == "Cancelling…", timeout_s=10.0)
            assert snap["status"] == "running", "still unwinding, not yet terminal"

            # Hold across many tqdm ticks (the mock emits every 20ms).
            time.sleep(1.0)
            snap = json.loads(paths.progress.read_text())
            assert snap["milestone"] == "Cancelling…"
        finally:
            _kill(proc)

    def test_clean_cancel_between_stages_does_not_force_exit(self, tmp_path, mock_hf_install):
        """When the main thread *can* unwind, the watchdog stands down.

        Distinguishable by exit code: a cooperative cancel exits 0 through
        main()'s normal return, a forced one exits 130.
        """
        cfg, paths = _build_config(tmp_path)
        proc = _spawn_worker(
            cfg, mock_config={"upload_sleep_s": 3.0}, extra_env={"LEROBOT_HUB_CANCEL_GRACE_S": "30.0"}
        )
        try:
            _wait_for(paths, lambda s: s.get("stage") == "uploading", timeout_s=10.0)
            proc.terminate()
            snap = _wait_until_status(paths, timeout_s=20.0)
            assert snap["status"] == "cancelled"
            assert snap["milestone"] == "Cancelled"
            assert proc.wait(timeout=5) == 0, "should not have needed the watchdog"
        finally:
            _kill(proc)

    def test_completed_transfer_is_never_force_cancelled(self, tmp_path, mock_hf_install):
        """The watchdog must not fire on a job that finished on its own."""
        cfg, paths = _build_config(tmp_path)
        proc = _spawn_worker(cfg, mock_config={}, extra_env=self.GRACE)
        try:
            snap = _wait_until_status(paths, timeout_s=20.0)
            assert snap["status"] == "complete"
            assert proc.wait(timeout=5) == 0
            # Well past the (1s) grace period: still complete, not cancelled.
            time.sleep(1.5)
            assert json.loads(paths.progress.read_text())["status"] == "complete"
        finally:
            _kill(proc)


class TestTransferProgressScaling:
    """Per-line cost must not grow with the number of files in the dataset.

    ``feed`` runs on every line HF writes. A per-line sweep over all known
    files is quadratic in the file count, and a reader thread that falls
    behind stops draining the capture pipe — which back-pressures HF's own
    writes and stalls the transfer. The progress readout would then be
    causing the very hang it exists to reveal.
    """

    @staticmethod
    def _bars(n_files: int, steps: int = 5) -> list[str]:
        return [
            f"  chunk-000/file-{f:05d}.mp4:  {s * 10}%|x| {s * 10}.0MB /  100MB            "
            for s in range(1, steps + 1)
            for f in range(n_files)
        ]

    @staticmethod
    def _us_per_line(lines: list[str]) -> float:
        p = hub_worker.TransferProgress()
        for line in lines:  # warm the regex cache before timing
            p.feed(line)
        p = hub_worker.TransferProgress()
        start = time.perf_counter()
        for line in lines:
            p.feed(line)
        return (time.perf_counter() - start) / len(lines) * 1e6

    def test_cost_per_line_does_not_grow_with_file_count(self):
        small = self._us_per_line(self._bars(100))
        large = self._us_per_line(self._bars(4000))
        # Constant-time would be 1.0x. The pre-fix implementation was ~27x
        # at this ratio; the bound is loose enough to survive a noisy CI box
        # while still catching a return to O(files) per line.
        assert large < small * 5, f"{large:.2f}us/line at 4000 files vs {small:.2f}us at 100"

    def test_counters_are_correct_under_incremental_accounting(self):
        """The incremental sum must agree with a full recomputation."""
        p = hub_worker.TransferProgress()
        p.feed("  a.mp4:  50%|x| 50.0MB /  100MB            ")
        p.feed("  b.mp4:  25%|x| 25.0MB /  100MB            ")
        assert p.bytes_done == 75_000_000 and p.files_done == 0
        # a completes...
        p.feed("  a.mp4: 100%|x|  100MB /  100MB            ")
        assert p.bytes_done == 125_000_000 and p.files_done == 1
        # ...and b follows.
        p.feed("  b.mp4: 100%|x|  100MB /  100MB            ")
        assert p.bytes_done == 200_000_000 and p.files_done == 2

    def test_completed_file_is_not_double_counted_on_repeat_ticks(self):
        p = hub_worker.TransferProgress()
        for _ in range(10):
            p.feed("  a.mp4: 100%|x|  100MB /  100MB            ")
        assert p.files_done == 1
        assert p.bytes_done == 100_000_000


class TestServerWorkerCancelIntegration:
    """Real worker subprocess driven by the real server-side cancel path.

    The unit tests either mock the worker (endpoints) or drive the worker
    without a server (above). This is the seam the reported incident
    actually crossed: a live worker blocked in an uninterruptible upload,
    a server holding a HubJobState, and a user clicking Cancel. It exercises
    the signal, the progress-file merge, and the escalation together.
    """

    def _register(self, state, cfg, proc):
        """Build the server-side job entry the GUI would have created."""
        job = hub_jobs.make_job(dataset_id=cfg.dataset_id, direction="upload", repo_id=cfg.repo_id)
        job.job_id = cfg.job_id
        job.status = "running"
        state.hub_jobs[job.job_id] = job
        return job

    @pytest.fixture
    def server_state(self, tmp_path, monkeypatch):
        from lerobot.gui.api import datasets as datasets_module
        from lerobot.gui.frame_cache import FrameCache
        from lerobot.gui.state import AppState

        state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
        datasets_module.set_app_state(state)
        return state

    def test_cancel_of_an_unstoppable_upload_reaches_cancelled(
        self, tmp_path, mock_hf_install, server_state, monkeypatch
    ):
        from lerobot.gui.api import datasets as datasets_module
        from lerobot.gui.api._hub_core import list_hub_jobs

        cfg, paths = _build_config(tmp_path)
        (Path(cfg.local_path) / "data.bin").write_bytes(b"x" * 8192)
        # The server computes job paths from the module-level JOBS_DIR.
        monkeypatch.setattr(hub_jobs, "JOBS_DIR", paths.jobs_dir)

        proc = _spawn_worker(
            cfg,
            mock_config={"upload_forever": True},
            extra_env={"LEROBOT_HUB_CANCEL_GRACE_S": "1.0"},
        )
        try:
            job = self._register(server_state, cfg, proc)

            # 1. The tray sees real, moving bytes — not a frozen file count.
            _wait_for(paths, lambda s: s.get("bytes_done_estimate", 0) > 0, timeout_s=15.0)
            listing = list_hub_jobs(server_state)["jobs"][0]
            assert listing["status"] == "running"
            assert listing["bytes_total"] == 8192
            assert listing["bytes_done_estimate"] > 0
            assert listing["stalled_for_s"] < 5.0, "healthy transfer must not read as stalled"

            # 2. Cancel is visible immediately, before the worker reacts.
            datasets_module._request_cancel(job)
            assert list_hub_jobs(server_state)["jobs"][0]["status"] == "cancelling"

            # 3. The worker force-exits itself inside the grace period, and
            #    the server converges on the terminal state by polling alone.
            deadline = time.time() + 20
            while time.time() < deadline:
                listing = list_hub_jobs(server_state)["jobs"][0]
                if listing["status"] == "cancelled":
                    break
                time.sleep(0.05)
            assert listing["status"] == "cancelled", listing

            # 4. The transfer is genuinely over — no process still uploading.
            assert proc.wait(timeout=10) is not None
            assert not paths.pid.exists()
        finally:
            _kill(proc)

    def test_wedged_worker_is_killed_by_the_poll_loop(
        self, tmp_path, mock_hf_install, server_state, monkeypatch
    ):
        """Belt-and-braces: if the worker can't self-exit, the server kills it.

        Simulated by giving the worker an unreachably long grace period, so
        only the server's SIGKILL escalation can end the transfer.
        """
        from lerobot.gui.api import datasets as datasets_module
        from lerobot.gui.api._hub_core import list_hub_jobs

        cfg, paths = _build_config(tmp_path)
        monkeypatch.setattr(hub_jobs, "JOBS_DIR", paths.jobs_dir)
        proc = _spawn_worker(
            cfg,
            mock_config={"upload_forever": True},
            extra_env={"LEROBOT_HUB_CANCEL_GRACE_S": "9999"},
        )
        try:
            job = self._register(server_state, cfg, proc)
            _wait_for(paths, lambda s: s.get("bytes_done_estimate", 0) > 0, timeout_s=15.0)

            datasets_module._request_cancel(job)
            # Worker is alive and will never self-terminate.
            assert proc.poll() is None
            # Pretend the grace period elapsed.
            job.cancel_requested_at = time.time() - hub_jobs.CANCEL_GRACE_S - 1

            listing = list_hub_jobs(server_state)["jobs"][0]
            assert listing["status"] == "cancelled"
            assert proc.wait(timeout=10) is not None, "SIGKILL did not end the worker"
        finally:
            _kill(proc)


class TestProgressWriterResilience:
    """The heartbeat thread must outlive a failing write.

    In production a single ``FileNotFoundError`` from a temp-path race ended
    this thread outright. Nothing supervised it, so the progress file simply
    stopped updating for the remaining hour of the transfer. The race itself
    is fixed in ``atomic_write_json``; this is the second line of defence,
    because *any* transient I/O error would have had the same effect.
    """

    def _state(self, tmp_path):
        cfg, paths = _build_config(tmp_path)
        return hub_worker._WorkerState(cfg, paths)

    def test_writer_thread_survives_a_failing_write(self, tmp_path, monkeypatch):
        state = self._state(tmp_path)
        calls = {"n": 0}
        real = hub_worker.atomic_write_json

        def flaky(path, data):
            calls["n"] += 1
            if calls["n"] <= 3:
                raise OSError("simulated transient I/O failure")
            return real(path, data)

        monkeypatch.setattr(hub_worker, "atomic_write_json", flaky)
        monkeypatch.setattr(hub_worker, "PROGRESS_WRITE_INTERVAL_S", 0.01)

        writer = state.start_writer_thread()
        try:
            deadline = time.time() + 10
            while time.time() < deadline and not state.paths.progress.exists():
                time.sleep(0.01)
            assert state.paths.progress.exists(), "heartbeat never recovered from the failures"
            assert writer.is_alive(), "writer thread died on a transient write failure"
        finally:
            state.stop_writer_thread()
            writer.join(timeout=5)

    def test_repeated_identical_milestones_do_not_rewrite_the_file(self, tmp_path):
        """The write storm that widened the race window is gone.

        HF redraws its bars several times a second and the derived milestone
        is usually unchanged; each redundant write was contention against
        the heartbeat thread on the same path for no new information.
        """
        state = self._state(tmp_path)
        state.set_milestone("Uploading files", stage="uploading")
        first = state.paths.progress.stat().st_mtime_ns

        for _ in range(50):
            state.set_milestone("Uploading files", stage="uploading")
        assert state.paths.progress.stat().st_mtime_ns == first, "redundant milestone rewrote the file"

        # A genuine change still flushes immediately.
        state.set_milestone("Merging PR", stage="merging")
        assert json.loads(state.paths.progress.read_text())["milestone"] == "Merging PR"


class TestRateDecaysDuringStall:
    """A stalled transfer must not keep reporting its last healthy rate.

    Observed on a real 200 MB upload through a stalling link: the byte
    counter froze at 32.50 MB for 55 seconds while the readout continued
    to claim 539.7 kB/s. A confidently-wrong throughput figure is the same
    class of defect as the frozen file counter this readout replaced.
    """

    def test_rate_decays_as_wall_clock_advances_without_bytes(self):
        now = [0.0]
        p = hub_worker.TransferProgress(clock=lambda: now[0])
        now[0] = 1.0
        p.feed("  a.mp4:   1%|x| 10.0MB /  100MB            ")
        now[0] = 2.0
        p.feed("  a.mp4:   2%|x| 20.0MB /  100MB            ")
        healthy = p.rate_bps()
        assert healthy == pytest.approx(10_000_000, rel=0.01)

        # Transfer stalls: no new bars arrive, only time passes.
        now[0] = 12.0
        stalled = p.rate_bps()
        assert stalled < healthy / 5, f"rate held at {stalled} through a 10s stall"
        now[0] = 102.0
        assert p.rate_bps() < healthy / 50

    def test_rate_recovers_when_bytes_resume(self):
        now = [0.0]
        p = hub_worker.TransferProgress(clock=lambda: now[0])
        now[0] = 1.0
        p.feed("  a.mp4:   1%|x| 10.0MB /  100MB            ")
        now[0] = 30.0  # long stall
        assert p.rate_bps() == 0.0  # only one sample so far
        now[0] = 31.0
        p.feed("  a.mp4:   3%|x| 30.0MB /  100MB            ")
        assert p.rate_bps() > 0.0

    def test_periodic_snapshot_refreshes_the_rate_without_new_bytes(self, tmp_path):
        """The heartbeat must re-derive it; feed() alone never runs while stalled."""
        cfg, _ = _build_config(tmp_path)
        state = hub_worker._WorkerState(cfg, hub_jobs.JobPaths.for_job(cfg.job_id, cfg.jobs_dir))
        now = [0.0]
        progress = hub_worker.TransferProgress(clock=lambda: now[0])
        state.progress = progress
        now[0] = 1.0
        progress.feed("  a.mp4:   1%|x| 10.0MB /  100MB            ")
        now[0] = 2.0
        progress.feed("  a.mp4:   2%|x| 20.0MB /  100MB            ")
        assert state.snapshot()["transfer_rate_bps"] == pytest.approx(10_000_000, rel=0.01)

        # No further feed() calls — exactly what a stall looks like.
        now[0] = 62.0
        assert state.snapshot()["transfer_rate_bps"] < 400_000


class TestClassicLfsBarFormat:
    """The non-Xet upload path must produce byte progress too.

    ``HF_HUB_DISABLE_XET=1`` is the supported way to avoid the Xet CAS
    endpoints, and it is a verified workaround on links where those
    endpoints stall — a 200 MB upload that never completed via Xet
    finished in 405 s with it set. That path uses tqdm's *default*
    bar_format, which prints no "B" after the numbers, so without a second
    pattern the readout is blank for exactly the users who needed the
    workaround.
    """

    def test_default_format_bar_is_parsed(self):
        s = hub_worker.parse_bar_line("model.bin:  45%|████      | 1.05G/2.34G [00:12<00:15, 84.2MB/s]")
        assert s is not None
        assert s.desc == "model.bin"
        assert s.done == 1_050_000_000
        assert s.total == 2_340_000_000
        assert s.rate_bps == 84_200_000

    def test_default_format_without_a_rate(self):
        s = hub_worker.parse_bar_line("data/chunk-000/file-001.mp4:   6%|▌  | 11.5M/201M [00:20<05:12]")
        assert s is not None
        assert (s.done, s.total) == (11_500_000, 201_000_000)
        assert s.rate_bps is None

    @pytest.mark.parametrize(
        "line",
        [
            # Plain item counters render identically apart from the missing
            # unit suffix. Feeding these to a byte aggregator would report
            # a 3 GB upload as "84 bytes of 84".
            "Fetching 84 files: 100%|██████████| 84/84 [00:00<00:00, 9401.51it/s]",
            "Fetching 84 files:  12%|█▏        | 10/84 [00:03<00:22]",
            "Recovering from metadata files: 100%|███| 84/84 [00:00<00:00, 9401.51it/s]",
        ],
    )
    def test_item_counters_are_not_mistaken_for_bytes(self, line):
        assert hub_worker.parse_bar_line(line) is None

    def test_both_formats_aggregate_together(self):
        """A worker only ever sees one format, but mixing must not corrupt."""
        p = hub_worker.TransferProgress()
        p.feed("a.mp4:  50%|x| 50.0M/100M [00:10<00:10]")
        p.feed("  b.mp4:  25%|x| 25.0MB /  100MB            ")
        assert p.bytes_done == 75_000_000


class TestTransientStorageErrorsAreNotFatal:
    """S3's transient 400s must not kill an upload.

    LFS uploads go straight to S3, which answers 400 for conditions that
    are purely transient. Observed on a stalling link:

        HTTP 400 from hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com/…
        UploadPart — RequestTimeout: Your socket connection to the server
        was not read from or written to within the timeout period.

    huggingface_hub retries the part itself, so fail-fasting on it turns a
    recoverable hiccup into a failed transfer. The 400 entry in the fatal
    table is for HF's own 400s (LFS-pointer corruption), which really do
    repeat forever.
    """

    S3_TIMEOUT_BODY = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<Error><Code>RequestTimeout</Code><Message>Your socket connection to the "
        "server was not read from or written to within the timeout period. Idle "
        "connections will be closed.</Message><RequestId>PEBZBV3CJWG2MDR8</RequestId>"
        "</Error>"
    )

    def _response(self, status: int, body: str):
        class _Resp:
            status_code = status
            headers: dict[str, str] = {}
            url = "https://hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com/repos/x?partNumber=1"

            def read(self):
                return body.encode()

        return _Resp()

    def test_s3_request_timeout_is_not_fatal(self):
        assert hub_worker._classify_response(self._response(400, self.S3_TIMEOUT_BODY)) is None

    @pytest.mark.parametrize("code", ["SlowDown", "InternalError", "ServiceUnavailable"])
    def test_other_transient_storage_codes_are_not_fatal(self, code):
        body = f"<Error><Code>{code}</Code><Message>try again</Message></Error>"
        assert hub_worker._classify_response(self._response(400, body)) is None

    def test_hf_own_bad_request_is_still_fatal(self):
        """The case the fatal entry exists for must keep failing fast."""
        body = '{"error":"Your push was rejected because it contains an LFS pointer"}'
        exc = hub_worker._classify_response(self._response(400, body))
        assert exc is not None and exc.error_class == "bad_request"

    def test_s3_access_denied_is_still_fatal(self):
        """A genuine S3 error that retrying cannot fix stays fatal."""
        body = "<Error><Code>AccessDenied</Code><Message>Access Denied</Message></Error>"
        exc = hub_worker._classify_response(self._response(403, body))
        assert exc is not None and exc.error_class == "auth"

    def test_prose_mentioning_a_timeout_does_not_excuse_a_fatal_response(self):
        """Match the XML Code element, not any mention of the word."""
        body = '{"error":"bad request; a previous RequestTimeout left the repo dirty"}'
        exc = hub_worker._classify_response(self._response(400, body))
        assert exc is not None, "substring matching would have wrongly excused this"

    def test_rate_limit_is_unaffected(self):
        exc = hub_worker._classify_response(self._response(429, "too many requests"))
        assert exc is not None and exc.error_class == "rate_limit"


class TestMilestonePinSurvivesTheSignalRace:
    """The cancel pin must not lose a check-then-act race.

    ``milestone_locked`` is set from the SIGTERM handler, which can land
    between an unlocked check and the assignment that follows it. A routine
    string that wins that race is never corrected — every later update is
    suppressed by the flag — so the tray would read "Processing files 0 / 1"
    for the whole cancellation, which is the symptom the pin exists to
    prevent.
    """

    def _state(self, tmp_path):
        cfg, _ = _build_config(tmp_path)
        return hub_worker._WorkerState(cfg, hub_jobs.JobPaths.for_job(cfg.job_id, cfg.jobs_dir))

    def test_lock_set_after_the_entry_check_still_wins(self, tmp_path):
        state = self._state(tmp_path)
        state.milestone = "Cancelling…"

        real_lock = state._lock

        class _LockThatCancelsOnAcquire:
            """Simulates SIGTERM landing inside set_milestone's window."""

            def __enter__(self):
                real_lock.acquire()
                state.milestone_locked = True  # the handler runs, right here
                return self

            def __exit__(self, *exc):
                real_lock.release()
                return False

        state._lock = _LockThatCancelsOnAcquire()
        state.set_milestone("Processing files 0 / 1")
        assert state.milestone == "Cancelling…", (
            "a routine milestone overwrote the pin and would never be corrected"
        )

    def test_normal_updates_still_apply_when_not_cancelling(self, tmp_path):
        state = self._state(tmp_path)
        state.set_milestone("Uploading files", stage="uploading")
        assert state.milestone == "Uploading files"
        assert state.stage == "uploading"


class TestWorkerRecordsItsOutcome:
    """A transfer's ending must outlive the job registry.

    The registry drops a finished job after 30 minutes and loses everything on
    a server restart, so "did my 8 GB upload land?" was unanswerable hours
    later. The worker appends its ending to a durable file — including on the
    force-exit paths, which skip main()'s finally entirely and are exactly the
    endings a user comes back asking about.
    """

    def _history(self, tmp_path):
        return tmp_path / "history.jsonl"

    def _read(self, path):
        import json as _json

        if not path.exists():
            return []
        return [_json.loads(x) for x in path.read_text().splitlines() if x.strip()]

    def test_completed_transfer_is_recorded(self, tmp_path, mock_hf_install):
        cfg, paths = _build_config(tmp_path)
        hist = self._history(tmp_path)
        proc = _spawn_worker(cfg, mock_config={}, extra_env={"LEROBOT_HUB_HISTORY_PATH": str(hist)})
        try:
            snap = _wait_until_status(paths, timeout_s=20.0)
            assert snap["status"] == "complete"
            proc.wait(timeout=5)
            recs = self._read(hist)
            assert len(recs) == 1, recs
            assert recs[0]["status"] == "complete"
            assert recs[0]["repo_id"] == "user/repo"
            assert recs[0]["job_id"] == cfg.job_id
        finally:
            _kill(proc)

    def test_force_cancelled_transfer_is_still_recorded(self, tmp_path, mock_hf_install):
        """The os._exit path skips main()'s finally — it must record first."""
        cfg, paths = _build_config(tmp_path)
        hist = self._history(tmp_path)
        proc = _spawn_worker(
            cfg,
            mock_config={"upload_forever": True},
            extra_env={"LEROBOT_HUB_CANCEL_GRACE_S": "1.0", "LEROBOT_HUB_HISTORY_PATH": str(hist)},
        )
        try:
            _wait_for(paths, lambda s: s.get("bytes_done_estimate", 0) > 0, timeout_s=15.0)
            proc.terminate()
            snap = _wait_until_status(paths, timeout_s=15.0)
            assert snap["status"] == "cancelled"
            assert proc.wait(timeout=5) == 130, "expected the force-exit path"

            recs = self._read(hist)
            assert len(recs) == 1, recs
            assert recs[0]["status"] == "cancelled"
            assert recs[0]["bytes_done_estimate"] > 0, "should carry what it managed to move"
        finally:
            _kill(proc)

    def test_failed_transfer_records_the_reason(self, tmp_path, mock_hf_install):
        cfg, paths = _build_config(tmp_path)
        hist = self._history(tmp_path)
        proc = _spawn_worker(
            cfg, mock_config={"fail_upload": True}, extra_env={"LEROBOT_HUB_HISTORY_PATH": str(hist)}
        )
        try:
            snap = _wait_until_status(paths, timeout_s=20.0)
            assert snap["status"] == "failed"
            proc.wait(timeout=5)
            recs = self._read(hist)
            assert len(recs) == 1
            assert recs[0]["status"] == "failed"
            assert recs[0]["error"], "a failure with no reason is what we are fixing"
        finally:
            _kill(proc)
