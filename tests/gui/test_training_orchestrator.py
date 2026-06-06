# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""End-to-end orchestrator tests using the real subprocess training runner.

These tests actually invoke `python -m lerobot.gui.training_runner` (the
fake-training stub) and verify the orchestrator drives it through the full
state machine, reads its outputs, and reconciles state.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from lerobot.gui.training_hosts import HostRegistry, TrainingHost
from lerobot.gui.training_orchestrator import (
    HostBusyError,
    Orchestrator,
    StartRequest,
    UnknownHostError,
    UnknownRunError,
)
from lerobot.gui.training_runs import RunRegistry, RunState
from lerobot.gui.training_transport import SubprocessTransport

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def host(tmp_path: Path) -> TrainingHost:
    return TrainingHost(
        id="test-host",
        display_name="Test Host",
        transport=SubprocessTransport(workdir=tmp_path / "workdir"),
    )


@pytest.fixture
def orch(host: TrainingHost, tmp_path: Path) -> Orchestrator:
    hr = HostRegistry(hosts=[host])
    rr = RunRegistry(runs_dir=tmp_path / "runs")
    return Orchestrator(host_registry=hr, run_registry=rr)


def _wait_until_state(orch: Orchestrator, run_id: str, want: RunState, *, timeout: float = 30.0):
    """Poll until the run reaches ``want`` (or any terminal state on failure)."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        snap = orch.poll(run_id)
        last = snap
        if snap.run.state == want or snap.run.state.value in {"completed", "failed", "aborted"}:
            return snap
        time.sleep(0.05)
    raise AssertionError(
        f"timed out waiting for state {want.value}; last={last.run.state.value if last else None}"
    )


# ── Start ──────────────────────────────────────────────────────────────────────


def test_start_returns_running_run(orch: Orchestrator) -> None:
    req = StartRequest(
        host_id="test-host",
        recipe_name="fake",
        dataset_id="fake/ds",
        args={"num_steps": 5, "save_every": 10, "step_seconds": 0.05},
    )
    run = orch.start(req)
    assert run.state == RunState.RUNNING
    assert run.session_id is not None
    assert run.host_id == "test-host"


def test_start_unknown_host_raises(orch: Orchestrator) -> None:
    req = StartRequest(host_id="nope", recipe_name="r", dataset_id="d")
    with pytest.raises(UnknownHostError):
        orch.start(req)


def test_start_idempotency_key_returns_same_run(orch: Orchestrator) -> None:
    req1 = StartRequest(
        host_id="test-host",
        recipe_name="fake",
        dataset_id="fake/ds",
        args={"num_steps": 5, "save_every": 10, "step_seconds": 0.05},
        idempotency_key="abc",
    )
    run1 = orch.start(req1)
    _wait_until_state(orch, run1.run_id, RunState.COMPLETED)
    # Same key → same run id, even though the original has finished
    req2 = StartRequest(
        host_id="test-host",
        recipe_name="other",
        dataset_id="other/ds",
        idempotency_key="abc",
    )
    run2 = orch.start(req2)
    assert run2.run_id == run1.run_id
    # And the original recipe is preserved (we didn't replace)
    assert run2.recipe_name == "fake"


def test_start_refuses_when_host_busy(orch: Orchestrator) -> None:
    req = StartRequest(
        host_id="test-host",
        recipe_name="fake",
        dataset_id="fake/ds",
        args={"num_steps": 100, "save_every": 25, "step_seconds": 0.05},  # ~5s, plenty of time
    )
    run1 = orch.start(req)
    try:
        req2 = StartRequest(
            host_id="test-host",
            recipe_name="other",
            dataset_id="other/ds",
            args={"num_steps": 5, "save_every": 10, "step_seconds": 0.05},
        )
        with pytest.raises(HostBusyError, match="busy"):
            orch.start(req2)
    finally:
        orch.stop(run1.run_id)
        _wait_until_state(orch, run1.run_id, RunState.ABORTED)


# ── End-to-end happy path ──────────────────────────────────────────────────────


def test_end_to_end_natural_completion(orch: Orchestrator) -> None:
    """Real subprocess runner runs to completion, orchestrator sees the
    progress / checkpoints / completion event correctly."""
    req = StartRequest(
        host_id="test-host",
        recipe_name="fake",
        dataset_id="fake/ds",
        args={"num_steps": 10, "save_every": 5, "step_seconds": 0.05},
    )
    run = orch.start(req)
    snap = _wait_until_state(orch, run.run_id, RunState.COMPLETED)
    # State machine landed correctly
    assert snap.run.state == RunState.COMPLETED
    assert snap.run.started_at is not None
    assert snap.run.finished_at is not None
    # Progress was written
    assert snap.progress is not None
    assert snap.progress["step"] == 10
    assert snap.progress["loss"] > 0
    # Checkpoints surfaced via manifest
    assert len(snap.checkpoints) == 2  # 10 steps / save_every=5 = 2
    assert snap.checkpoints[0].step == 5
    assert snap.checkpoints[1].step == 10
    assert all(c.sha256 for c in snap.checkpoints)
    # stderr_tail surfaced
    assert "[runner]" in snap.stderr_tail


# ── Stop ───────────────────────────────────────────────────────────────────────


def test_stop_aborts_running_run(orch: Orchestrator) -> None:
    req = StartRequest(
        host_id="test-host",
        recipe_name="fake",
        dataset_id="fake/ds",
        args={"num_steps": 1000, "save_every": 100, "step_seconds": 0.05},
    )
    run = orch.start(req)
    # Give the worker a moment to actually start its loop
    time.sleep(0.3)
    orch.stop(run.run_id)
    snap = _wait_until_state(orch, run.run_id, RunState.ABORTED)
    assert snap.run.state == RunState.ABORTED


def test_stop_unknown_run_raises(orch: Orchestrator) -> None:
    with pytest.raises(UnknownRunError):
        orch.stop("nope")


def test_stop_idempotent_on_terminal(orch: Orchestrator) -> None:
    req = StartRequest(
        host_id="test-host",
        recipe_name="fake",
        dataset_id="fake/ds",
        args={"num_steps": 3, "save_every": 5, "step_seconds": 0.05},
    )
    run = orch.start(req)
    snap = _wait_until_state(orch, run.run_id, RunState.COMPLETED)
    # Calling stop on a completed run is a no-op
    same = orch.stop(snap.run.run_id)
    assert same.state == RunState.COMPLETED


# ── Poll ───────────────────────────────────────────────────────────────────────


def test_poll_unknown_run_raises(orch: Orchestrator) -> None:
    with pytest.raises(UnknownRunError):
        orch.poll("nope")


def test_poll_returns_progress_during_run(orch: Orchestrator) -> None:
    req = StartRequest(
        host_id="test-host",
        recipe_name="fake",
        dataset_id="fake/ds",
        args={"num_steps": 200, "save_every": 50, "step_seconds": 0.05},
    )
    run = orch.start(req)
    try:
        # Wait for first checkpoint to appear
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            snap = orch.poll(run.run_id)
            if snap.progress is not None and snap.progress.get("step", 0) > 0:
                break
            time.sleep(0.05)
        else:
            pytest.fail("progress did not appear within 30s")
        assert snap.progress["loss"] > 0
        assert snap.run.state == RunState.RUNNING
    finally:
        orch.stop(run.run_id)
        _wait_until_state(orch, run.run_id, RunState.ABORTED)


# ── List ───────────────────────────────────────────────────────────────────────


def test_list_runs_includes_started_run(orch: Orchestrator) -> None:
    req = StartRequest(
        host_id="test-host",
        recipe_name="fake",
        dataset_id="fake/ds",
        args={"num_steps": 3, "save_every": 5, "step_seconds": 0.05},
    )
    run = orch.start(req)
    runs = orch.list_runs()
    assert any(r.run_id == run.run_id for r in runs)
    _wait_until_state(orch, run.run_id, RunState.COMPLETED)
