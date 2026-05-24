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

from lerobot.gui import hub_jobs

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

    The pipeline takes the unsquashed-merge path unconditionally, so the
    end-state milestone always says "merged unsquashed". This test pins
    that behaviour so re-enabling squash (when the HF API issue is
    resolved) requires an explicit test update.
    """

    def test_pipeline_always_takes_unsquashed_merge_path(self, tmp_path, mock_hf_install):
        cfg, paths = _build_config(tmp_path)
        proc = _spawn_worker(cfg, mock_config={"fail_squash": True})
        try:
            snap = _wait_until_status(paths)
            assert snap["status"] == "complete", f"Expected complete; got {snap}"
            assert "unsquashed" in snap["milestone"], (
                f"Milestone should signal the unsquashed merge; got {snap['milestone']!r}"
            )
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
