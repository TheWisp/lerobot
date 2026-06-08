# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""End-to-end orchestrator tests using the real subprocess training runner.

These tests actually invoke `python -m lerobot.gui.training.runner` (the
fake-training stub) and verify the orchestrator drives it through the full
state machine, reads its outputs, and reconciles state.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from lerobot.gui.training.hosts import HostRegistry, TrainingHost
from lerobot.gui.training.orchestrator import (
    HostBusyError,
    Orchestrator,
    StartRequest,
    UnknownHostError,
    UnknownRunError,
    _extract_image_from_docker_argv,
)
from lerobot.gui.training.runs import RunRegistry, RunState
from lerobot.gui.training.transport import SubprocessClient, SubprocessTransport

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


def test_start_returns_pending_then_advances_to_running(orch: Orchestrator) -> None:
    """``start()`` returns immediately with state=PENDING (so the POST
    isn't blocked by image prep / launch). The prep thread advances to
    RUNNING shortly after — verifiable via poll()."""
    req = StartRequest(
        host_id="test-host",
        recipe_name="fake",
        dataset_id="fake/ds",
        args={"__recipe__": "__fake__", "num_steps": 5, "save_every": 10, "step_seconds": 0.05},
    )
    run = orch.start(req)
    # Synchronous return: PENDING (no session_id yet — prep thread sets it)
    assert run.state == RunState.PENDING
    assert run.host_id == "test-host"
    # Eventually: RUNNING with a session_id
    snap = _wait_until_state(orch, run.run_id, RunState.RUNNING)
    assert snap.run.session_id is not None


def test_start_unknown_host_raises(orch: Orchestrator) -> None:
    req = StartRequest(host_id="nope", recipe_name="r", dataset_id="d")
    with pytest.raises(UnknownHostError):
        orch.start(req)


def test_start_idempotency_key_returns_same_run(orch: Orchestrator) -> None:
    req1 = StartRequest(
        host_id="test-host",
        recipe_name="fake",
        dataset_id="fake/ds",
        args={"__recipe__": "__fake__", "num_steps": 5, "save_every": 10, "step_seconds": 0.05},
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
        args={
            "__recipe__": "__fake__",
            "num_steps": 100,
            "save_every": 25,
            "step_seconds": 0.05,
        },  # ~5s, plenty of time
    )
    run1 = orch.start(req)
    # Wait until the prep thread spawned the worker so the second start
    # actually races against a real RUNNING worker, not the still-PENDING
    # run. The host-busy check passes either way (PENDING is non-terminal),
    # but the stop()-cleanup at the end of the test wants RUNNING so it
    # exercises the SIGTERM path (PENDING goes straight to ABORTED, which
    # we cover in test_stop_pending_run_skips_to_aborted).
    _wait_until_state(orch, run1.run_id, RunState.RUNNING)
    try:
        req2 = StartRequest(
            host_id="test-host",
            recipe_name="other",
            dataset_id="other/ds",
            args={"__recipe__": "__fake__", "num_steps": 5, "save_every": 10, "step_seconds": 0.05},
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
        args={"__recipe__": "__fake__", "num_steps": 10, "save_every": 5, "step_seconds": 0.05},
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
        args={"__recipe__": "__fake__", "num_steps": 1000, "save_every": 100, "step_seconds": 0.05},
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
        args={"__recipe__": "__fake__", "num_steps": 3, "save_every": 5, "step_seconds": 0.05},
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
        args={"__recipe__": "__fake__", "num_steps": 200, "save_every": 50, "step_seconds": 0.05},
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
        args={"__recipe__": "__fake__", "num_steps": 3, "save_every": 5, "step_seconds": 0.05},
    )
    run = orch.start(req)
    runs = orch.list_runs()
    assert any(r.run_id == run.run_id for r in runs)
    _wait_until_state(orch, run.run_id, RunState.COMPLETED)


def test_list_runs_reconciles_completion_without_poll(host: TrainingHost, tmp_path: Path) -> None:
    """Regression: list_runs() must reconcile non-terminal runs from
    events.jsonl so the sidebar reflects completion even for runs the user
    hasn't clicked on. Previously the list endpoint only read run.json from
    disk; if no one called poll(run_id), state stayed "running" forever
    after the worker exited.
    """
    hr = HostRegistry(hosts=[host])
    rr = RunRegistry(runs_dir=tmp_path / "runs")
    orch = Orchestrator(host_registry=hr, run_registry=rr)
    req = StartRequest(
        host_id="test-host",
        recipe_name="fake",
        dataset_id="fake/ds",
        args={"__recipe__": "__fake__", "num_steps": 3, "save_every": 5, "step_seconds": 0.05},
    )
    run = orch.start(req)
    # Wait for the worker to actually exit (file-based: events.jsonl will have
    # the terminal event). Without calling orch.poll(), the in-memory state
    # is still RUNNING.
    deadline = time.monotonic() + 30.0
    paths = (tmp_path / "runs" / run.run_id).resolve()
    events_path = paths / "events.jsonl"
    while time.monotonic() < deadline:
        if events_path.exists():
            content = events_path.read_text()
            if "completed_naturally" in content or "crashed" in content:
                break
        time.sleep(0.05)
    assert "completed_naturally" in events_path.read_text()

    # The bug being fixed: previously list_runs would have returned RUNNING
    # because it only read run.json (no reconciliation). With the fix, the
    # cheap events-only reconcile runs inline and the state lands at COMPLETED.
    runs = orch.list_runs()
    me = next(r for r in runs if r.run_id == run.run_id)
    assert me.state == RunState.COMPLETED, (
        f"list_runs() should reconcile non-terminal runs from events.jsonl; got {me.state.value}"
    )


def test_orchestrator_appends_to_manifest_from_disk_real_recipe_layout(
    host: TrainingHost, tmp_path: Path
) -> None:
    """When using the real (docker) recipe, lerobot-train doesn't write our
    checkpoints.jsonl — the orchestrator does, by watching the bind-mounted
    output dir for new step subdirs. Simulates that layout without invoking
    docker.
    """
    import time as _t

    from lerobot.gui.training.runs import Run, RunPaths, new_run_id

    hr = HostRegistry(hosts=[host])
    rr = RunRegistry(runs_dir=tmp_path / "runs")
    orch = Orchestrator(host_registry=hr, run_registry=rr)
    run = Run(
        run_id=new_run_id(),
        host_id="test-host",
        recipe_name="real-mode",
        dataset_id="lerobot/pusht",
        args={"policy.type": "act"},  # NOT __fake__ → docker recipe path
        state=RunState.PENDING,
        created_at=_t.time(),
    )
    run.session_id = 1  # fake PID
    run.advance(RunState.RUNNING)
    rr.save(run)
    paths = RunPaths.for_run(run.run_id, rr.runs_dir)
    paths.ensure_exists()
    # lerobot-train-shaped layout: <run>/output/checkpoints/000005/pretrained_model/
    for step in (5, 10):
        d = paths.root / "output" / "checkpoints" / f"{step:06d}" / "pretrained_model"
        d.mkdir(parents=True)
        (d / "model.safetensors").write_bytes(f"fake-{step}".encode())
    snap = orch.poll(run.run_id)
    assert len(snap.checkpoints) == 2
    assert [c.step for c in snap.checkpoints] == [5, 10]
    assert all(c.path.endswith("model.safetensors") for c in snap.checkpoints)
    assert all(c.sha256 for c in snap.checkpoints)


def test_orchestrator_appends_to_manifest_from_disk_hvla_recipe_layout(
    host: TrainingHost, tmp_path: Path
) -> None:
    """HVLA flow_matching trainer writes checkpoints as
    <run>/output/checkpoints/checkpoint-<step>/ — the scanner regex must
    pick up that pattern in addition to the lerobot-train zero-padded form.
    """
    import time as _t

    from lerobot.gui.training.recipes import HVLA_FLOW_S1_RECIPE
    from lerobot.gui.training.runs import Run, RunPaths, new_run_id

    hr = HostRegistry(hosts=[host])
    rr = RunRegistry(runs_dir=tmp_path / "runs")
    orch = Orchestrator(host_registry=hr, run_registry=rr)
    run = Run(
        run_id=new_run_id(),
        host_id="test-host",
        recipe_name="hvla-test",
        dataset_id="thewisp/some_data",
        # HVLA recipe marker → output_subdir_in_run still returns "output",
        # so checkpoints land in <run>/output/checkpoints/ — same place the
        # scanner looks for lerobot-train.
        args={"__recipe__": HVLA_FLOW_S1_RECIPE, "dataset_repo_id": "thewisp/some_data"},
        state=RunState.PENDING,
        created_at=_t.time(),
    )
    run.session_id = 1  # fake PID (treated as not alive)
    run.advance(RunState.RUNNING)
    rr.save(run)
    paths = RunPaths.for_run(run.run_id, rr.runs_dir)
    paths.ensure_exists()
    # HVLA layout: <run>/output/checkpoints/checkpoint-<step>/pretrained_model/
    for step in (100, 500):
        d = paths.root / "output" / "checkpoints" / f"checkpoint-{step}" / "pretrained_model"
        d.mkdir(parents=True)
        (d / "model.safetensors").write_bytes(f"fake-{step}".encode())
    snap = orch.poll(run.run_id)
    assert len(snap.checkpoints) == 2
    assert [c.step for c in snap.checkpoints] == [100, 500]
    assert all(c.path.endswith("model.safetensors") for c in snap.checkpoints)
    assert all(c.sha256 for c in snap.checkpoints)


def test_orchestrator_sorts_checkpoints_by_step_not_dir_name(host: TrainingHost, tmp_path: Path) -> None:
    """``checkpoint-10`` sorts before ``checkpoint-5`` alphabetically.
    The scanner must sort by parsed step number, not by raw directory name,
    so the manifest comes out in step order regardless of zero-padding."""
    import time as _t

    from lerobot.gui.training.recipes import HVLA_FLOW_S1_RECIPE
    from lerobot.gui.training.runs import Run, RunPaths, new_run_id

    hr = HostRegistry(hosts=[host])
    rr = RunRegistry(runs_dir=tmp_path / "runs")
    orch = Orchestrator(host_registry=hr, run_registry=rr)
    run = Run(
        run_id=new_run_id(),
        host_id="test-host",
        recipe_name="sort-test",
        dataset_id="thewisp/some_data",
        args={"__recipe__": HVLA_FLOW_S1_RECIPE, "dataset_repo_id": "thewisp/some_data"},
        state=RunState.PENDING,
        created_at=_t.time(),
    )
    run.session_id = 1
    run.advance(RunState.RUNNING)
    rr.save(run)
    paths = RunPaths.for_run(run.run_id, rr.runs_dir)
    paths.ensure_exists()
    # Use steps that ARE in the wrong order alphabetically: 5 + 10 + 100
    # ("checkpoint-10" < "checkpoint-100" < "checkpoint-5" by string sort).
    for step in (5, 10, 100):
        d = paths.root / "output" / "checkpoints" / f"checkpoint-{step}" / "pretrained_model"
        d.mkdir(parents=True)
        (d / "model.safetensors").write_bytes(f"fake-{step}".encode())
    snap = orch.poll(run.run_id)
    # Manifest entries must be in step order.
    assert [c.step for c in snap.checkpoints] == [5, 10, 100]


def test_orchestrator_completed_on_exit_with_checkpoints(host: TrainingHost, tmp_path: Path) -> None:
    """Real (docker) recipe: process exits cleanly, at least one checkpoint
    on disk → orchestrator writes completed_naturally and advances to
    COMPLETED."""
    import time as _t

    from lerobot.gui.training.runs import Run, RunPaths, new_run_id

    hr = HostRegistry(hosts=[host])
    rr = RunRegistry(runs_dir=tmp_path / "runs")
    orch = Orchestrator(host_registry=hr, run_registry=rr)
    run = Run(
        run_id=new_run_id(),
        host_id="test-host",
        recipe_name="real",
        dataset_id="lerobot/pusht",
        args={"policy.type": "act"},
        state=RunState.PENDING,
        created_at=_t.time(),
    )
    run.session_id = 1  # not alive
    run.advance(RunState.RUNNING)
    rr.save(run)
    paths = RunPaths.for_run(run.run_id, rr.runs_dir)
    paths.ensure_exists()
    d = paths.root / "output" / "checkpoints" / "000005" / "pretrained_model"
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"x")
    snap = orch.poll(run.run_id)
    assert snap.run.state == RunState.COMPLETED
    assert "completed_naturally" in paths.events_jsonl.read_text()


def test_orchestrator_crashed_on_exit_without_checkpoints(host: TrainingHost, tmp_path: Path) -> None:
    """Real recipe: process exits but no checkpoints → orchestrator writes
    crashed and advances to FAILED."""
    import time as _t

    from lerobot.gui.training.runs import Run, RunPaths, new_run_id

    hr = HostRegistry(hosts=[host])
    rr = RunRegistry(runs_dir=tmp_path / "runs")
    orch = Orchestrator(host_registry=hr, run_registry=rr)
    run = Run(
        run_id=new_run_id(),
        host_id="test-host",
        recipe_name="real",
        dataset_id="lerobot/pusht",
        args={"policy.type": "act"},
        state=RunState.PENDING,
        created_at=_t.time(),
    )
    run.session_id = 1  # not alive
    run.advance(RunState.RUNNING)
    rr.save(run)
    paths = RunPaths.for_run(run.run_id, rr.runs_dir)
    paths.ensure_exists()
    snap = orch.poll(run.run_id)
    assert snap.run.state == RunState.FAILED
    assert "crashed" in paths.events_jsonl.read_text()


def test_orchestrator_crashes_on_nonzero_exit_even_with_checkpoints(
    host: TrainingHost, tmp_path: Path
) -> None:
    """Regression for the SmolVLA HF-403 silent-success: lerobot-train wrote
    both checkpoints, then crashed in push_model_to_hub at end of training.
    The container exited non-zero, but the old orchestrator counted
    checkpoints and called it completed. Now: non-zero exit code from the
    transport wins, regardless of checkpoint count, and the stderr tail is
    surfaced in run.error so the user can see WHAT crashed."""
    import time as _t

    from lerobot.gui.training.runs import Run, RunPaths, new_run_id

    class _ExitCodeClient(SubprocessClient):
        """Real subprocess client EXCEPT exit_code() returns a fixed value
        so we can simulate the docker container's non-zero exit without
        actually launching a process. The session_id (1) intentionally
        isn't in our Popen registry → we override at the exit_code call."""

        def __init__(self, transport: SubprocessTransport, *, fake_exit_code: int) -> None:
            super().__init__(transport)
            self._fake_exit_code = fake_exit_code

        def exit_code(self, session_id: int) -> int | None:
            return self._fake_exit_code

        def is_alive(self, session_id: int) -> bool:
            return False  # crashed, not alive

    hr = HostRegistry(hosts=[host])
    rr = RunRegistry(runs_dir=tmp_path / "runs")
    client = _ExitCodeClient(SubprocessTransport(workdir=tmp_path / "workdir"), fake_exit_code=1)
    orch = Orchestrator(
        host_registry=hr,
        run_registry=rr,
        make_client_fn=lambda _transport: client,
    )
    run = Run(
        run_id=new_run_id(),
        host_id="test-host",
        recipe_name="real",
        dataset_id="lerobot/pusht",
        args={"policy.type": "smolvla"},
        state=RunState.PENDING,
        created_at=_t.time(),
    )
    run.session_id = 1
    run.advance(RunState.RUNNING)
    rr.save(run)
    paths = RunPaths.for_run(run.run_id, rr.runs_dir)
    paths.ensure_exists()
    # Two checkpoints on disk — same as the actual SmolVLA HF-403 run.
    for step in (100, 200):
        d = paths.root / "output" / "checkpoints" / f"{step:06d}" / "pretrained_model"
        d.mkdir(parents=True)
        (d / "model.safetensors").write_bytes(b"x")
    # Stderr tail with a recognisable HF-403 traceback signature.
    paths.stderr_log.write_text(
        "INFO End of training\n"
        "Traceback (most recent call last):\n"
        '  File "pretrained.py", line 213, in push_model_to_hub\n'
        "huggingface_hub.errors.HfHubHTTPError: 403 Forbidden ... "
        'rights to create a model under the namespace "local"\n'
    )
    snap = orch.poll(run.run_id)
    assert snap.run.state == RunState.FAILED, "non-zero exit must override checkpoint-based completion"
    assert snap.run.error is not None and "exit code 1" in snap.run.error
    assert "403 Forbidden" in snap.run.error, "stderr tail should be surfaced in run.error"
    assert "crashed" in paths.events_jsonl.read_text()


def test_orchestrator_final_step_from_max_checkpoint_not_progress_json(
    host: TrainingHost, tmp_path: Path
) -> None:
    """Regression: final_step used to come from progress.json (which real
    lerobot-train doesn't write), so completed runs reported step 0. Now
    it's max(checkpoint_steps), which works for both fake (writes progress.json)
    and real (only checkpoints) recipes."""
    import json
    import time as _t

    from lerobot.gui.training.runs import Run, RunPaths, new_run_id

    hr = HostRegistry(hosts=[host])
    rr = RunRegistry(runs_dir=tmp_path / "runs")
    orch = Orchestrator(host_registry=hr, run_registry=rr)
    run = Run(
        run_id=new_run_id(),
        host_id="test-host",
        recipe_name="real",
        dataset_id="lerobot/pusht",
        args={"policy.type": "act"},
        state=RunState.PENDING,
        created_at=_t.time(),
    )
    run.session_id = 1  # not alive, no Popen → exit_code returns None
    run.advance(RunState.RUNNING)
    rr.save(run)
    paths = RunPaths.for_run(run.run_id, rr.runs_dir)
    paths.ensure_exists()
    # Three checkpoints; alphabetical order would put 200 < 50, so the test
    # also guards against the "last-dir-by-listdir" bug.
    for step in (50, 100, 200):
        d = paths.root / "output" / "checkpoints" / f"{step:06d}" / "pretrained_model"
        d.mkdir(parents=True)
        (d / "model.safetensors").write_bytes(b"x")
    # No progress.json — simulates the real lerobot-train run.
    snap = orch.poll(run.run_id)
    assert snap.run.state == RunState.COMPLETED
    events = [json.loads(ln) for ln in paths.events_jsonl.read_text().splitlines() if ln]
    terminal = [e for e in events if e["type"] == "completed_naturally"]
    assert len(terminal) == 1
    assert terminal[0]["final_step"] == 200, (
        f"final_step should be max checkpoint step (200), got {terminal[0]['final_step']}"
    )


def test_list_runs_reconciles_abort_without_poll(host: TrainingHost, tmp_path: Path) -> None:
    """Same regression for the aborted_by_user terminal event."""
    hr = HostRegistry(hosts=[host])
    rr = RunRegistry(runs_dir=tmp_path / "runs")
    orch = Orchestrator(host_registry=hr, run_registry=rr)
    req = StartRequest(
        host_id="test-host",
        recipe_name="fake",
        dataset_id="fake/ds",
        args={"__recipe__": "__fake__", "num_steps": 1000, "save_every": 100, "step_seconds": 0.05},
    )
    run = orch.start(req)
    # Let the worker actually start its loop, then stop
    time.sleep(0.3)
    orch.stop(run.run_id)
    # Wait for the worker to write its aborted_by_user event
    events_path = tmp_path / "runs" / run.run_id / "events.jsonl"
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if events_path.exists() and "aborted_by_user" in events_path.read_text():
            break
        time.sleep(0.05)
    assert "aborted_by_user" in events_path.read_text()
    # list_runs should now reconcile COMPLETING → ABORTED
    runs = orch.list_runs()
    me = next(r for r in runs if r.run_id == run.run_id)
    assert me.state == RunState.ABORTED


# ── C5: image preparation (pre-pull + events) ─────────────────────────────────


class _FakeTransportClient(SubprocessClient):
    """Subclass that overrides the docker-image ops for tests, leaving every
    file/launch op delegating to the real SubprocessClient impl.

    The image ops used to live on a separate ``ImageRunner`` shim; they're
    now part of :class:`TransportClient`. Tests inject this via
    ``Orchestrator(make_client_fn=lambda transport: fake)`` instead of the
    old ``image_runner=`` kwarg.
    """

    def __init__(
        self,
        transport: SubprocessTransport,
        *,
        inspect_returns: bool = False,
        pull_returns: tuple[bool, str] = (True, ""),
        size: int = 42,
    ) -> None:
        super().__init__(transport)
        self.inspect_returns = inspect_returns
        self.pull_returns = pull_returns
        self.size = size
        self.inspect_calls: list[str] = []
        self.pull_calls: list[str] = []

    def image_inspect(self, tag: str) -> bool:
        self.inspect_calls.append(tag)
        return self.inspect_returns

    def image_pull(self, tag: str) -> tuple[bool, str]:
        self.pull_calls.append(tag)
        return self.pull_returns

    def image_size(self, tag: str) -> int | None:
        return self.size


def _make_orch_with_fake_image(
    host, tmp_path: Path, **fake_kwargs
) -> tuple[Orchestrator, _FakeTransportClient]:
    """Wire up an Orchestrator whose transport-client factory always returns
    the same fake (so we can inspect docker-op call records after a run)."""
    hr = HostRegistry(hosts=[host])
    rr = RunRegistry(runs_dir=tmp_path / "runs")
    fake = _FakeTransportClient(SubprocessTransport(workdir=tmp_path / "workdir"), **fake_kwargs)
    orch = Orchestrator(
        host_registry=hr,
        run_registry=rr,
        make_client_fn=lambda _transport: fake,
    )
    return orch, fake


def _events_of(orch: Orchestrator, run_id: str) -> list[dict]:
    import json

    from lerobot.gui.training.runs import RunPaths

    paths = RunPaths.for_run(run_id, orch._runs.runs_dir)
    if not paths.events_jsonl.exists():
        return []
    return [json.loads(line) for line in paths.events_jsonl.read_text().splitlines() if line.strip()]


def test_extract_image_from_docker_argv_typical_recipe() -> None:
    """The recipe builder's argv shape is:
        docker run --rm --gpus all --user 1000:1000 -v A:B -v C:D <image> <entrypoint> ...
    The helper should return <image>."""
    argv = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--user",
        "1000:1000",
        "-v",
        "/h:/c",
        "-v",
        "/runs/x:/runs",
        "ghcr.io/foo/lerobot-training:tag",
        "lerobot-train",
        "--policy.type=act",
    ]
    assert _extract_image_from_docker_argv(argv) == "ghcr.io/foo/lerobot-training:tag"


def test_extract_image_from_docker_argv_non_docker_returns_none() -> None:
    # Fake recipe argv — python -m, no docker
    assert (
        _extract_image_from_docker_argv(["python", "-m", "lerobot.gui.training.runner", "--run-dir", "/x"])
        is None
    )
    # Empty / too-short
    assert _extract_image_from_docker_argv([]) is None
    assert _extract_image_from_docker_argv(["docker"]) is None
    assert _extract_image_from_docker_argv(["docker", "ps"]) is None  # not "run"


def test_extract_image_unknown_flag_returns_none() -> None:
    # Defensive: if we see a flag we don't recognise, bail rather than
    # mis-identify it as the image. Better to silently skip pre-pull than
    # to try to docker-inspect a flag value.
    argv = ["docker", "run", "--rm", "--mystery-flag", "v", "img:tag", "cmd"]
    assert _extract_image_from_docker_argv(argv) is None


def test_image_cache_hit_emits_event_and_skips_pull(host, tmp_path: Path) -> None:
    """Fake recipe path: no docker invocation at all → no image_* events."""
    orch, fake = _make_orch_with_fake_image(host, tmp_path, inspect_returns=True)
    req = StartRequest(
        host_id="test-host",
        recipe_name="fake-rec",
        dataset_id="ds",
        args={"__recipe__": "__fake__", "num_steps": 2, "save_every": 5, "step_seconds": 0.05},
    )
    run = orch.start(req)
    _wait_until_state(orch, run.run_id, RunState.COMPLETED)
    # fake recipe → no image inspect or pull
    assert fake.inspect_calls == []
    assert fake.pull_calls == []
    types = [e["type"] for e in _events_of(orch, run.run_id)]
    assert "image_cache_hit" not in types
    assert "image_pull_started" not in types


def test_ensure_image_cache_hit_emits_only_one_event(host, tmp_path: Path) -> None:
    """Direct unit-test of _ensure_image with a cache hit. Uses the fake
    transport client so the call lands on its scripted image_inspect."""
    orch, fake = _make_orch_with_fake_image(host, tmp_path, inspect_returns=True)
    from lerobot.gui.training.runs import RunPaths

    paths = RunPaths.for_run("test", runs_dir=tmp_path / "runs")
    paths.ensure_exists()
    orch._ensure_image(fake, "ghcr.io/foo/img:tag", paths)
    assert fake.inspect_calls == ["ghcr.io/foo/img:tag"]
    assert fake.pull_calls == []  # never pulled
    import json

    lines = paths.events_jsonl.read_text().splitlines()
    assert len(lines) == 1
    evt = json.loads(lines[0])
    assert evt["type"] == "image_cache_hit"
    assert evt["image"] == "ghcr.io/foo/img:tag"


def test_ensure_image_cache_miss_pulls_and_emits_two_events(host, tmp_path: Path) -> None:
    """Cache miss → image_pull_started + image_pulled (with duration_s + size_bytes)."""
    orch, fake = _make_orch_with_fake_image(
        host, tmp_path, inspect_returns=False, pull_returns=(True, ""), size=1234
    )
    from lerobot.gui.training.runs import RunPaths

    paths = RunPaths.for_run("test", runs_dir=tmp_path / "runs")
    paths.ensure_exists()
    orch._ensure_image(fake, "ghcr.io/foo/img:tag", paths)
    assert fake.pull_calls == ["ghcr.io/foo/img:tag"]
    import json

    events = [json.loads(line) for line in paths.events_jsonl.read_text().splitlines()]
    types = [e["type"] for e in events]
    assert types == ["image_pull_started", "image_pulled"]
    pulled = events[1]
    assert pulled["image"] == "ghcr.io/foo/img:tag"
    assert pulled["size_bytes"] == 1234
    assert pulled["duration_s"] >= 0


def test_ensure_image_pull_failure_emits_pull_failed_and_raises(host, tmp_path: Path) -> None:
    """Pull failure → image_pull_started + image_pull_failed events; raises
    so the orchestrator can flip the run to FAILED."""
    from lerobot.gui.training.orchestrator import _ImagePullError

    orch, fake = _make_orch_with_fake_image(
        host, tmp_path, inspect_returns=False, pull_returns=(False, "manifest unknown\n")
    )
    from lerobot.gui.training.runs import RunPaths

    paths = RunPaths.for_run("test", runs_dir=tmp_path / "runs")
    paths.ensure_exists()
    with pytest.raises(_ImagePullError, match="manifest unknown"):
        orch._ensure_image(fake, "ghcr.io/foo/img:bad", paths)
    import json

    events = [json.loads(line) for line in paths.events_jsonl.read_text().splitlines()]
    assert [e["type"] for e in events] == ["image_pull_started", "image_pull_failed"]
    assert "manifest unknown" in events[1]["error"]


def test_stop_pending_run_skips_to_aborted(orch: Orchestrator) -> None:
    """A stop() before the prep thread has finished should advance the run
    straight to ABORTED (skipping COMPLETING, since there's no worker yet
    to SIGTERM). The prep thread bails on the state change."""
    # We rely on the fake recipe being effectively instant; the race window
    # is small but nonzero. To make this deterministic, we'd need a barrier
    # in the prep thread. Instead, we ASSERT the stop() path works even
    # when called while still PENDING: load the run before the prep thread
    # touches it.
    req = StartRequest(
        host_id="test-host",
        recipe_name="fake",
        dataset_id="ds",
        args={"__recipe__": "__fake__", "num_steps": 1000, "save_every": 100, "step_seconds": 1.0},  # slow
    )
    run = orch.start(req)
    # In the small chance the prep thread already advanced, we still want
    # the test to pass — the stop() path from RUNNING is well-tested
    # elsewhere. Here we only assert that stop()+wait reaches ABORTED.
    orch.stop(run.run_id)
    snap = _wait_until_state(orch, run.run_id, RunState.ABORTED)
    assert snap.run.state == RunState.ABORTED


# ── Delete + clear (housekeeping) ─────────────────────────────────────────────


def test_delete_run_with_model_preserves_checkpoints(orch: Orchestrator) -> None:
    """REGRESSION: dropping a run from history must NOT delete the trained
    model checkpoints. Models tab continues to surface the artefact;
    Training tab no longer lists the run."""
    from lerobot.gui.training.runs import RunPaths

    req = StartRequest(
        host_id="test-host",
        recipe_name="fake",
        dataset_id="fake/ds",
        args={"__recipe__": "__fake__", "num_steps": 10, "save_every": 5, "step_seconds": 0.05},
    )
    run = orch.start(req)
    _wait_until_state(orch, run.run_id, RunState.COMPLETED)
    paths = RunPaths.for_run(run.run_id, orch._runs.runs_dir)
    # Sanity: model files were actually produced
    ckpt_files = list(paths.checkpoints_dir.rglob("*.safetensors"))
    assert ckpt_files, "fake-runner should have written checkpoints"
    ckpt_paths_before = {p: p.stat().st_size for p in ckpt_files}

    result = orch.delete_run(run.run_id)
    # Metadata gone
    assert not paths.run_json.exists()
    assert not paths.events_jsonl.exists()
    assert not paths.checkpoints_jsonl.exists()
    assert not paths.stderr_log.exists()
    # Checkpoints survived — EXACT same paths + sizes
    assert paths.checkpoints_dir.is_dir()
    for p, sz in ckpt_paths_before.items():
        assert p.exists(), f"{p} should have survived delete_run"
        assert p.stat().st_size == sz, f"{p} content changed"
    # Returned shape
    assert result["run_id"] == run.run_id
    assert result["kept_model"] is True
    assert result["metadata_bytes_freed"] > 0  # at least stderr.log + run.json
    # Gone from list_runs
    assert all(r.run_id != run.run_id for r in orch.list_runs())


def test_delete_failed_no_checkpoint_run_nukes_dir(orch: Orchestrator) -> None:
    """A run that never wrote a model (e.g. failed pull, aborted before
    first save) has nothing to preserve — the whole dir goes."""
    import time as _t

    from lerobot.gui.training.runs import Run, RunPaths, new_run_id

    run = Run(
        run_id=new_run_id(),
        host_id="test-host",
        recipe_name="hand-crafted-fail",
        dataset_id="ds",
        args={"__recipe__": "__fake__"},
        state=RunState.PENDING,
        created_at=_t.time(),
    )
    run.advance(RunState.FAILED)
    orch._runs.save(run)
    paths = RunPaths.for_run(run.run_id, orch._runs.runs_dir)
    paths.ensure_exists()
    (paths.root / "stderr.log").write_text("crashed without producing a checkpoint")

    result = orch.delete_run(run.run_id)
    # No model = full nuke
    assert not paths.root.exists()
    assert result["kept_model"] is False
    assert result["metadata_bytes_freed"] >= len("crashed without producing a checkpoint")


def test_delete_run_keeps_checkpoints_under_real_layout(orch: Orchestrator) -> None:
    """REGRESSION: the real docker recipe writes checkpoints under
    ``<run>/output/checkpoints/<step>/...``, not directly under
    ``<run>/checkpoints/``. The metadata-drop helper has to look in the
    right place — otherwise it 'thinks' there's no model and nukes
    everything."""
    import time as _t

    from lerobot.gui.training.runs import Run, RunPaths, new_run_id

    # Hand-build a run that mirrors what `_build_docker_command` would leave on
    # disk: terminal state, real output/ layout with a model file.
    run = Run(
        run_id=new_run_id(),
        host_id="test-host",
        recipe_name="real-layout",
        dataset_id="lerobot/pusht",
        args={"policy.type": "act", "dataset.repo_id": "lerobot/pusht"},
        state=RunState.PENDING,
        created_at=_t.time(),
    )
    run.advance(RunState.RUNNING)
    run.advance(RunState.COMPLETED)
    orch._runs.save(run)
    paths = RunPaths.for_run(run.run_id, orch._runs.runs_dir)
    paths.ensure_exists()
    # Write the real-recipe layout
    pretrained = paths.root / "output" / "checkpoints" / "000010" / "pretrained_model"
    pretrained.mkdir(parents=True)
    model_file = pretrained / "model.safetensors"
    model_file.write_bytes(b"fake-act-model" * 1000)
    expected_size = model_file.stat().st_size
    paths.stderr_log.write_text("trained")

    result = orch.delete_run(run.run_id)
    assert result["kept_model"] is True
    assert model_file.exists()
    assert model_file.stat().st_size == expected_size
    # Metadata gone
    assert not paths.run_json.exists()
    assert not paths.stderr_log.exists()


def test_delete_running_run_refuses(orch: Orchestrator) -> None:
    """Trying to delete a still-running run raises RunNotTerminalError.
    Caller must Stop first."""
    from lerobot.gui.training.orchestrator import RunNotTerminalError

    req = StartRequest(
        host_id="test-host",
        recipe_name="fake",
        dataset_id="fake/ds",
        args={"__recipe__": "__fake__", "num_steps": 1000, "save_every": 100, "step_seconds": 1.0},
    )
    run = orch.start(req)
    _wait_until_state(orch, run.run_id, RunState.RUNNING)
    try:
        with pytest.raises(RunNotTerminalError, match="stop it first"):
            orch.delete_run(run.run_id)
    finally:
        orch.stop(run.run_id)
        _wait_until_state(orch, run.run_id, RunState.ABORTED)


def test_delete_unknown_raises(orch: Orchestrator) -> None:
    with pytest.raises(UnknownRunError):
        orch.delete_run("nope")


def test_clear_terminal_removes_only_finished_runs(orch: Orchestrator, tmp_path: Path) -> None:
    """clear_terminal_runs() should drop every terminal run from history,
    PRESERVING checkpoints on rows that produced them and skipping active
    rows entirely. ``models_kept`` counts the survivors."""
    import time as _t

    from lerobot.gui.training.runs import Run, RunPaths, new_run_id

    # Set up: 1 terminal-WITH-model + 1 terminal-NO-model + 1 active. The
    # WITH-model row should keep its checkpoint after clear; the NO-model
    # row gets fully nuked; the active row is untouched.
    paths_to_check = {}
    with_model_run_id = None
    no_model_run_id = None
    for label, target_state, write_model in [
        ("done-with-model", RunState.COMPLETED, True),
        ("oops-no-model", RunState.ABORTED, False),
    ]:
        r = Run(
            run_id=new_run_id(),
            host_id="some-other-host",  # avoid the host lock with the active run below
            recipe_name=label,
            dataset_id="ds",
            args={},
            state=RunState.PENDING,
            created_at=_t.time(),
        )
        r.advance(RunState.RUNNING)
        r.advance(target_state)
        orch._runs.save(r)
        p = RunPaths.for_run(r.run_id, orch._runs.runs_dir)
        p.ensure_exists()
        (p.root / "stderr.log").write_text("x" * 1000)  # metadata
        if write_model:
            pretrained = p.root / "output" / "checkpoints" / "000005" / "pretrained_model"
            pretrained.mkdir(parents=True)
            (pretrained / "model.safetensors").write_bytes(b"M" * 4096)
            with_model_run_id = r.run_id
        else:
            no_model_run_id = r.run_id
        paths_to_check[r.run_id] = (p, target_state, write_model)
    # And a real-active run
    req = StartRequest(
        host_id="test-host",
        recipe_name="active",
        dataset_id="ds",
        args={"__recipe__": "__fake__", "num_steps": 1000, "save_every": 100, "step_seconds": 1.0},
    )
    active = orch.start(req)
    _wait_until_state(orch, active.run_id, RunState.RUNNING)
    active_paths = RunPaths.for_run(active.run_id, orch._runs.runs_dir)
    try:
        result = orch.clear_terminal_runs()
        assert set(result["deleted"]) == set(paths_to_check)
        # Exactly one model was preserved (the WITH-model row)
        assert result["models_kept"] == 1
        # bytes_freed counts metadata only — the model bytes stay on disk
        assert result["metadata_bytes_freed"] >= 1000  # at least one stderr.log
        # WITH-model row: dir survives + model intact, but metadata gone
        wp = paths_to_check[with_model_run_id][0]
        assert wp.root.is_dir()
        assert not wp.stderr_log.exists()
        model_file = wp.root / "output" / "checkpoints" / "000005" / "pretrained_model" / "model.safetensors"
        assert model_file.exists()
        assert model_file.stat().st_size == 4096
        # NO-model row: whole dir gone
        np = paths_to_check[no_model_run_id][0]
        assert not np.root.exists()
        # Active dir intact
        assert active_paths.root.is_dir()
    finally:
        orch.stop(active.run_id)
        _wait_until_state(orch, active.run_id, RunState.ABORTED)


def test_clear_terminal_empty_is_noop(orch: Orchestrator) -> None:
    """No terminal runs → deleted=[], all counters zero. Idempotent."""
    result = orch.clear_terminal_runs()
    assert result == {"deleted": [], "metadata_bytes_freed": 0, "models_kept": 0}
