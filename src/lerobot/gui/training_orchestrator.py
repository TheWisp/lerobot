# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Run orchestrator — starts, polls, stops training runs.

The orchestrator is the bridge between the API layer (which receives Start /
Stop / List from the frontend) and the lower-level pieces:

- :class:`HostRegistry` — picks which transport to use for a given host id
- :class:`TransportClient` — launches / reads from the training process
- :class:`RunRegistry` — persists run metadata + state machine

It does NOT own background polling. The API layer (FastAPI) is expected to
call :meth:`poll` periodically, which reads the structured files the worker
writes and updates the Run's state. This keeps the orchestrator stateless
beyond the registry, so resume across GUI server restarts is automatic.

DESIGN.md § Concurrency:
- Idempotency keys deflect double-Start clicks
- Per-host single-active-run lock prevents two trainings on the same host
- State machine prevents stale duplicate Start from re-running a finished run
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.gui.training_hosts import HostRegistry, TrainingHost
from lerobot.gui.training_runs import (
    TERMINAL_STATES,
    Run,
    RunPaths,
    RunRegistry,
    RunState,
    append_event,
    new_run_id,
)
from lerobot.gui.training_transport import (
    TransportClient,
    make_client,
)

# ── Requests / responses ──────────────────────────────────────────────────────


@dataclass(frozen=True, kw_only=True)
class StartRequest:
    """What the API layer hands to :meth:`Orchestrator.start`."""

    host_id: str
    recipe_name: str
    dataset_id: str
    args: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None


@dataclass(frozen=True)
class CheckpointEntry:
    """One line of ``checkpoints.jsonl`` — a completed checkpoint."""

    step: int
    path: str  # relative to the run dir
    sha256: str
    ts: float


@dataclass(frozen=True)
class RunSnapshot:
    """Polling result — current state of a run plus what's been observed.

    All fields readable by the API to render the UI without round-tripping
    to the orchestrator again.
    """

    run: Run
    progress: dict[str, Any] | None  # contents of progress.json, or None if not written yet
    checkpoints: list[CheckpointEntry]  # all manifest entries
    stderr_tail: str  # last N bytes of stderr.log (configurable on poll)


# ── Orchestrator ──────────────────────────────────────────────────────────────


# Number of bytes of stderr.log we surface back to the caller. Cheap to read;
# the full log is on disk for "open in files" later.
DEFAULT_STDERR_TAIL_BYTES = 16 * 1024


class HostBusyError(RuntimeError):
    """A start request targeted a host that already has an active run."""


class UnknownHostError(KeyError):
    """The requested host id isn't in the registry."""


class UnknownRunError(KeyError):
    """The requested run id doesn't exist."""


class Orchestrator:
    """Owns start / poll / stop for training runs.

    Stateless beyond the two registries it composes: any state survives
    GUI server restart by reading from disk on the next call. No background
    threads — the API layer drives polling.
    """

    def __init__(
        self,
        host_registry: HostRegistry,
        run_registry: RunRegistry,
        *,
        runner_module: str = "lerobot.gui.training_runner",
    ) -> None:
        self._hosts = host_registry
        self._runs = run_registry
        # Module name (not file path) so we invoke via `python -m`. Tests can
        # substitute a stub runner module without monkey-patching subprocess.
        self._runner_module = runner_module

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, req: StartRequest) -> Run:
        """Create + launch a new run.

        Deflects double-clicks via idempotency_key. Refuses if the target
        host already has an active run.
        """
        # 1. Idempotency: same key → return same run
        if req.idempotency_key:
            existing = self._runs.find_by_idempotency_key(req.idempotency_key)
            if existing is not None:
                return existing

        # 2. Host exists?
        host = self._hosts.get(req.host_id)
        if host is None:
            raise UnknownHostError(f"unknown host id: {req.host_id!r}")

        # 3. Per-host single-active-run lock
        busy = self._runs.active_run_on_host(req.host_id)
        if busy is not None:
            raise HostBusyError(
                f"host {req.host_id!r} is busy with run {busy.run_id!r} (state={busy.state.value})"
            )

        # 4. Create run, persist as PENDING
        run = Run(
            run_id=new_run_id(),
            host_id=req.host_id,
            recipe_name=req.recipe_name,
            dataset_id=req.dataset_id,
            args=dict(req.args),
            state=RunState.PENDING,
            created_at=time.time(),
            idempotency_key=req.idempotency_key,
        )
        self._runs.save(run)
        paths = RunPaths.for_run(run.run_id, self._runs.runs_dir)
        paths.ensure_exists()

        # 5. Launch the worker via the host's transport
        try:
            session_id = self._launch_worker(host, run, paths)
        except Exception as exc:
            run.error = f"launch failed: {exc!r}"
            run.advance(RunState.FAILED)
            self._runs.save(run)
            append_event(paths.events_jsonl, "crashed", error=str(exc), final_step=0)
            raise

        # 6. Mark RUNNING
        run.session_id = session_id
        run.advance(RunState.RUNNING)
        self._runs.save(run)
        append_event(paths.events_jsonl, "started", session_id=session_id, host_id=host.id)
        return run

    def poll(
        self,
        run_id: str,
        *,
        stderr_tail_bytes: int = DEFAULT_STDERR_TAIL_BYTES,
    ) -> RunSnapshot:
        """Read the worker's state files and reconcile with the state machine.

        Detects natural completion / abort / crash by reading the final
        ``events.jsonl`` entry (worker writes it before exit) cross-checked
        against the transport's ``is_alive``.
        """
        run = self._runs.load(run_id)
        if run is None:
            raise UnknownRunError(f"unknown run id: {run_id!r}")

        paths = RunPaths.for_run(run.run_id, self._runs.runs_dir)
        host = self._hosts.get(run.host_id)
        client = make_client(host.transport) if host is not None else None

        # Reconcile state with the worker, if it's still in a live state.
        if run.state in (RunState.RUNNING, RunState.COMPLETING) and client is not None:
            self._reconcile_state(run, paths, client)

        progress = self._read_progress(paths.progress_json)
        checkpoints = self._read_manifest(paths.checkpoints_jsonl)
        stderr_tail = self._read_stderr_tail(paths.stderr_log, stderr_tail_bytes)

        return RunSnapshot(run=run, progress=progress, checkpoints=checkpoints, stderr_tail=stderr_tail)

    def stop(self, run_id: str) -> Run:
        """User-initiated stop — SIGTERM the worker, mark COMPLETING.

        The worker writes its final ``aborted_by_user`` event then exits;
        next ``poll()`` reconciles to ABORTED. Idempotent on terminal runs.
        """
        run = self._runs.load(run_id)
        if run is None:
            raise UnknownRunError(f"unknown run id: {run_id!r}")
        if run.state in TERMINAL_STATES:
            return run  # idempotent — already stopped
        if run.state == RunState.COMPLETING:
            return run  # idempotent — already stopping
        host = self._hosts.get(run.host_id)
        if host is None:
            # Host went away (deleted profile) — best-effort mark aborted.
            run.advance(RunState.ABORTED)
            self._runs.save(run)
            return run
        client = make_client(host.transport)
        if run.session_id is not None:
            client.stop(run.session_id, force=False)
        run.advance(RunState.COMPLETING)
        self._runs.save(run)
        append_event(
            RunPaths.for_run(run.run_id, self._runs.runs_dir).events_jsonl,
            "stop_requested",
        )
        return run

    def list_runs(self) -> list[Run]:
        """List all runs. Cheaply reconciles each non-terminal run from its
        ``events.jsonl`` so the list view shows up-to-date state even for
        runs the user hasn't clicked on (no transport calls — just a file
        read per non-terminal run).

        Full reconciliation including the process-liveness probe still lives
        in :meth:`poll` for the selected run.
        """
        runs = self._runs.list_all()
        for run in runs:
            if run.state in TERMINAL_STATES:
                continue
            paths = RunPaths.for_run(run.run_id, self._runs.runs_dir)
            if self._reconcile_from_events_only(run, paths):
                self._runs.save(run)
        return runs

    # ── Internals ─────────────────────────────────────────────────────────────

    def _launch_worker(self, host: TrainingHost, run: Run, paths: RunPaths) -> int:
        """Build the worker command + invoke via the host's transport."""
        client = make_client(host.transport)
        command = self._build_command(run, paths)
        env = self._build_env(run, paths)
        # For subprocess transport, workdir is the run dir (worker writes here).
        # For SSH (future), the workdir param becomes the remote per-run dir
        # (e.g. /workspace/runs/<run_id>); SshClient will translate. For now,
        # paths.root is the right thing to pass in either case.
        return client.launch(command=command, env=env, workdir=paths.root, log_path=paths.stderr_log)

    def _build_command(self, run: Run, paths: RunPaths) -> list[str]:
        """Compose the runner CLI from a Run's args. Keeps the runner's CLI
        contract — args dict forwarded as ``--key value`` (kebab-case)."""
        cmd = [sys.executable, "-m", self._runner_module, "--run-dir", str(paths.root)]
        for k, v in run.args.items():
            flag = "--" + k.replace("_", "-")
            cmd.extend([flag, str(v)])
        return cmd

    def _build_env(self, run: Run, paths: RunPaths) -> dict[str, str]:
        """Env vars passed to the worker. Minimal in v1; HF_TOKEN goes here
        once we wire HF auth (deferred to v2 per DESIGN.md)."""
        return {
            "LEROBOT_RUN_ID": run.run_id,
            "LEROBOT_RUN_DIR": str(paths.root),
        }

    def _reconcile_from_events_only(self, run: Run, paths: RunPaths) -> bool:
        """Cheap reconciliation path: only reads ``events.jsonl``.

        Used by :meth:`list_runs` to keep the sidebar fresh without a
        per-run transport probe. Returns True iff the state actually changed
        (so the caller can save).

        Caveat: cannot detect crashes (worker died without writing a
        terminal event) — that's covered by the full :meth:`_reconcile_state`
        path on the selected run's poll.
        """
        before = run.state
        terminal_event = self._read_terminal_event(paths.events_jsonl)
        if terminal_event == "completed_naturally" and run.state != RunState.COMPLETED:
            run.advance(RunState.COMPLETED)
        elif terminal_event == "aborted_by_user" and run.state != RunState.ABORTED:
            run.advance(RunState.ABORTED)
        elif terminal_event == "crashed" and run.state != RunState.FAILED:
            run.advance(RunState.FAILED)
        return run.state != before

    def _reconcile_state(self, run: Run, paths: RunPaths, client: TransportClient) -> None:
        """Update ``run.state`` based on (a) what the worker wrote to
        events.jsonl and (b) whether the process is still alive.

        DESIGN.md § Health "Completion signal" — we read the worker's final
        events.jsonl line as canonical; falling back to "process gone" only
        as a crash detection.
        """
        terminal_event = self._read_terminal_event(paths.events_jsonl)
        alive = client.is_alive(run.session_id) if run.session_id is not None else False

        if terminal_event == "completed_naturally":
            if run.state != RunState.COMPLETED:
                run.advance(RunState.COMPLETED)
                self._runs.save(run)
        elif terminal_event == "aborted_by_user":
            if run.state != RunState.ABORTED:
                run.advance(RunState.ABORTED)
                self._runs.save(run)
        elif terminal_event == "crashed":
            if run.state != RunState.FAILED:
                run.advance(RunState.FAILED)
                self._runs.save(run)
        elif not alive:
            # Process gone, no terminal event — treat as crash.
            run.error = "process exited without writing a completion event"
            run.advance(RunState.FAILED)
            self._runs.save(run)
            append_event(
                paths.events_jsonl,
                "crashed",
                error=run.error,
                final_step=self._read_progress(paths.progress_json).get("step", 0)
                if self._read_progress(paths.progress_json)
                else 0,
            )

    @staticmethod
    def _read_progress(progress_path: Path) -> dict[str, Any] | None:
        if not progress_path.exists():
            return None
        try:
            return json.loads(progress_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _read_manifest(manifest_path: Path) -> list[CheckpointEntry]:
        if not manifest_path.exists():
            return []
        entries: list[CheckpointEntry] = []
        for line in manifest_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                entries.append(
                    CheckpointEntry(
                        step=int(d["step"]),
                        path=str(d["path"]),
                        sha256=str(d["sha256"]),
                        ts=float(d["ts"]),
                    )
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        return entries

    @staticmethod
    def _read_stderr_tail(stderr_path: Path, n_bytes: int) -> str:
        if not stderr_path.exists() or n_bytes <= 0:
            return ""
        try:
            size = stderr_path.stat().st_size
            offset = max(0, size - n_bytes)
            with stderr_path.open("rb") as f:
                f.seek(offset)
                return f.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    @staticmethod
    def _read_terminal_event(events_path: Path) -> str | None:
        """Scan events.jsonl for a terminal event type. Returns the type name
        or None. Terminal events: completed_naturally / aborted_by_user / crashed.
        """
        if not events_path.exists():
            return None
        terminal = {"completed_naturally", "aborted_by_user", "crashed"}
        for line in events_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if evt.get("type") in terminal:
                return evt["type"]
        return None
