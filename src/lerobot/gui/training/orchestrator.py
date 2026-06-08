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

import contextlib
import json
import logging
import re
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.gui.training.hosts import HostRegistry, TrainingHost
from lerobot.gui.training.recipes import (
    build_lerobot_train_command,
    output_subdir_in_run,
)
from lerobot.gui.training.runs import (
    TERMINAL_STATES,
    Run,
    RunPaths,
    RunRegistry,
    RunState,
    new_run_id,
)
from lerobot.gui.training.transport import (
    SubprocessClient,
    SubprocessTransport,
    TransportClient,
    make_client,
)

logger = logging.getLogger(__name__)

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
    events: list[dict[str, Any]]  # all events.jsonl entries (oldest first)


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


class RunNotTerminalError(RuntimeError):
    """A delete request hit a run that's still pending / running / completing.

    Refuses to delete an active run because:
      - the prep thread or worker subprocess is writing to files we'd
        rmtree out from under them
      - the per-host single-active-run lock would silently un-lock
      - the user's most likely intent was "stop and forget", not "kill
        and delete"; surfacing this as an error makes the two-step
        explicit (stop, then delete).
    """


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
        runner_module: str = "lerobot.gui.training.runner",
        make_client_fn: Callable[[Any], TransportClient] | None = None,
    ) -> None:
        self._hosts = host_registry
        self._runs = run_registry
        # Module name (not file path) so we invoke via `python -m`. Tests can
        # substitute a stub runner module without monkey-patching subprocess.
        self._runner_module = runner_module
        # All host-state ops (file reads, dir listings, image docker calls,
        # checkpoint sha256s) go through the resolved TransportClient. Tests
        # inject a fake-transport factory; production uses :func:`make_client`
        # which picks SubprocessClient or SshClient based on the host's
        # transport type. See [DESIGN.md § HostProvider] + the
        # ``TransportClient`` Protocol in ``training/transport.py``.
        self._make_client_fn = make_client_fn or make_client
        # Background prep threads (image pull + worker spawn). Keyed by run_id;
        # daemon=True so they don't block process shutdown. We keep refs so
        # tests can join them; in production they're fire-and-forget.
        self._prep_threads: dict[str, threading.Thread] = {}

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

        # 5. Prepare image + launch worker in a background thread.
        #
        # The HTTP request returns immediately with state=PENDING. The
        # background thread emits ``image_cache_hit`` / ``image_pull_started`` /
        # ``image_pulled`` / ``image_pull_failed`` events into events.jsonl
        # as it works, and the frontend's existing run-detail poller renders
        # them as visible status. On success it spawns the worker and
        # advances to RUNNING; on failure it advances to FAILED.
        #
        # Why a thread rather than a synchronous pull: first-time image
        # pulls measured ~13 min on this workstation's connection. Blocking
        # the POST that long times out the browser's fetch — and the user
        # sees nothing in the meantime since the run row only appears
        # after the response.
        t = threading.Thread(
            target=self._prepare_and_launch,
            args=(host, run.run_id, paths),
            daemon=True,
            name=f"prepare-{run.run_id[:8]}",
        )
        self._prep_threads[run.run_id] = t
        t.start()
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
        client = self._client_for_host(host, paths)

        # Reconcile state with the worker, if it's still in a live state.
        # We only do the liveness probe when the host is known; otherwise
        # the run is treated as "we can read what we have, but we can't
        # check on it." Same semantic as before the refactor.
        if run.state in (RunState.RUNNING, RunState.COMPLETING) and host is not None:
            self._reconcile_state(run, paths, client)

        progress = self._read_progress(client, paths.progress_json)
        checkpoints = self._read_manifest(client, paths.checkpoints_jsonl)
        stderr_tail = self._read_stderr_tail(client, paths.stderr_log, stderr_tail_bytes)
        events = self._read_events(client, paths.events_jsonl)

        return RunSnapshot(
            run=run,
            progress=progress,
            checkpoints=checkpoints,
            stderr_tail=stderr_tail,
            events=events,
        )

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
        paths = RunPaths.for_run(run.run_id, self._runs.runs_dir)
        host = self._hosts.get(run.host_id)
        client = self._client_for_host(host, paths)
        if run.state == RunState.PENDING:
            # Prep thread is still running (image pull or pre-launch). No
            # worker to SIGTERM. Skip COMPLETING straight to ABORTED — the
            # prep thread will reload run state before launching and bail
            # if it sees a terminal state. (If it raced past that check
            # already, the spawned worker is --rm so it cleans up on exit.)
            run.advance(RunState.ABORTED)
            self._runs.save(run)
            self._emit_event(client, paths.events_jsonl, "aborted_by_user", final_step=0)
            return run
        if host is None:
            # Host went away (deleted profile) — best-effort mark aborted.
            run.advance(RunState.ABORTED)
            self._runs.save(run)
            return run
        if run.session_id is not None:
            client.stop(run.session_id, force=False)
        run.advance(RunState.COMPLETING)
        self._runs.save(run)
        self._emit_event(client, paths.events_jsonl, "stop_requested")
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

    def delete_run(self, run_id: str) -> dict[str, Any]:
        """Drop a run from the training history.

        Removes the run's metadata files (run.json, events.jsonl,
        checkpoints.jsonl, stderr.log, progress.json) so the row stops
        showing up in the Training list. **Preserves
        ``output/checkpoints/`` intact** — dropping a run from history is
        NOT throwing away the model the run produced. The trained
        checkpoint continues to surface in the Models tab; disk cleanup
        of the model itself is a separate Models-tab action.

        If the run produced no checkpoints (failed pull, aborted before
        first save), the whole dir is removed — nothing to preserve.

        Returns ``{"run_id": str, "metadata_bytes_freed": int,
        "kept_model": bool}``.

        Refuses with :class:`RunNotTerminalError` if the run is still in a
        pending / running / completing state — stop it first. Refuses
        with :class:`UnknownRunError` if the run id is unknown.
        """
        run = self._runs.load(run_id)
        if run is None:
            raise UnknownRunError(f"unknown run id: {run_id!r}")
        if run.state not in TERMINAL_STATES:
            raise RunNotTerminalError(
                f"run {run_id!r} is in state {run.state.value!r}; stop it first, then delete"
            )
        paths = RunPaths.for_run(run.run_id, self._runs.runs_dir)
        bytes_freed, kept_model = _drop_run_metadata(paths)
        return {
            "run_id": run_id,
            "metadata_bytes_freed": bytes_freed,
            "kept_model": kept_model,
        }

    def clear_terminal_runs(self) -> dict[str, Any]:
        """Bulk-drop every terminal-state run from training history.

        Per-row semantics match :meth:`delete_run`: keeps checkpoints,
        removes only metadata. Returns ``{"deleted": [run_ids],
        "metadata_bytes_freed": int, "models_kept": int}`` —
        ``models_kept`` counts the rows whose ``output/checkpoints/``
        survived (and thus stay visible in the Models tab). Idempotent.
        """
        deleted: list[str] = []
        bytes_freed = 0
        models_kept = 0
        for run in self._runs.list_all():
            if run.state not in TERMINAL_STATES:
                continue
            paths = RunPaths.for_run(run.run_id, self._runs.runs_dir)
            b, kept = _drop_run_metadata(paths)
            bytes_freed += b
            if kept:
                models_kept += 1
            deleted.append(run.run_id)
        return {
            "deleted": deleted,
            "metadata_bytes_freed": bytes_freed,
            "models_kept": models_kept,
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _client_for_host(self, host: TrainingHost | None, paths: RunPaths) -> TransportClient:
        """Resolve a :class:`TransportClient` for ops on a given host.

        Falls back to a local :class:`SubprocessClient` if the host is
        unknown (e.g. deleted from the registry). That fallback preserves
        the pre-refactor behavior — for workstation runs the run dir is
        on the GUI server's filesystem anyway, so local reads still
        surface the last known state. For SSH-host runs with a deleted
        profile, we lose remote access but the run.json + locally-synced
        artefacts remain readable.
        """
        if host is not None:
            return self._make_client_fn(host.transport)
        return SubprocessClient(SubprocessTransport(workdir=paths.root))

    @staticmethod
    def _emit_event(client: TransportClient, events_path: Path, type_: str, **fields: Any) -> None:
        """Append a JSON event line to the host's events.jsonl via the
        transport. Used by the orchestrator for its own event emits
        (``started`` / ``image_*`` / ``aborted_by_user`` / ``crashed`` /
        ``completed_naturally``). The worker still uses the direct-file
        :func:`runs.append_event` because it runs inside the container
        with local filesystem access.
        """
        line = json.dumps({"type": type_, "ts": time.time(), **fields})
        client.append_text(events_path, line + "\n")

    # ── Image prep + launch (background thread entry point) ───────────────────

    def _prepare_and_launch(self, host: TrainingHost, run_id: str, paths: RunPaths) -> None:
        """Pre-pull the image if needed, then launch the worker.

        Runs in a daemon thread spawned from :meth:`start`. On success,
        advances run to RUNNING and emits ``started`` event. On failure,
        advances to FAILED and emits ``image_pull_failed`` or ``crashed``
        as appropriate. All errors are caught here — never bubble up;
        the run state IS the error channel.
        """
        # Re-load — start() saved PENDING; we own the lifecycle now.
        run = self._runs.load(run_id)
        if run is None:
            logger.error("prepare-and-launch: run %s vanished", run_id)
            return
        client = self._client_for_host(host, paths)
        try:
            cmd = self._build_command(run, paths)
            image = _extract_image_from_docker_argv(cmd)
            if image is not None:
                self._ensure_image(client, image, paths)
        except _ImagePullError as exc:
            # Already emitted image_pull_failed; flip state to FAILED.
            run.error = f"image pull failed: {exc}"
            run.advance(RunState.FAILED)
            self._runs.save(run)
            return
        except Exception as exc:
            logger.exception("prepare-and-launch: unexpected error before launch")
            run.error = f"prepare failed: {exc!r}"
            run.advance(RunState.FAILED)
            self._runs.save(run)
            self._emit_event(client, paths.events_jsonl, "crashed", error=str(exc), final_step=0)
            return
        # Race check: did the user stop us between image-prep and launch?
        # (stop() on PENDING transitions directly to ABORTED.)
        run = self._runs.load(run_id)
        if run is None or run.state != RunState.PENDING:
            logger.info(
                "prepare-and-launch: run %s no longer PENDING (state=%s) — skipping launch",
                run_id,
                run.state.value if run else "?",
            )
            return
        try:
            session_id = self._launch_worker(host, run, paths)
        except Exception as exc:
            logger.exception("prepare-and-launch: worker launch failed")
            run.error = f"launch failed: {exc!r}"
            run.advance(RunState.FAILED)
            self._runs.save(run)
            self._emit_event(client, paths.events_jsonl, "crashed", error=str(exc), final_step=0)
            return
        # Final race check: stop() can land between launch and advance.
        # If so, kill what we just spawned. --rm on docker run cleans up the
        # container even if we miss this; the kill is best-effort UX.
        run_after = self._runs.load(run_id)
        if run_after is None or run_after.state != RunState.PENDING:
            try:
                client.stop(session_id, force=True)
            except Exception:
                logger.exception("prepare-and-launch: post-launch kill failed (best-effort)")
            return
        run_after.session_id = session_id
        run_after.advance(RunState.RUNNING)
        self._runs.save(run_after)
        self._emit_event(client, paths.events_jsonl, "started", session_id=session_id, host_id=host.id)

    def _ensure_image(self, client: TransportClient, image: str, paths: RunPaths) -> None:
        """Make sure ``image`` is present in the host's docker cache.

        Uses the transport client's image ops — ``image_inspect`` to check,
        ``image_pull`` to fetch, ``image_size`` for the post-pull size.
        For SubprocessClient those shell out to local ``docker``; for
        SshClient they run ``docker`` over SSH on the remote host. The
        orchestrator's view is identical either way.

        Emits one of:
          - ``image_cache_hit`` — image already local; no pull.
          - ``image_pull_started`` + ``image_pulled`` — pull succeeded; latter
            carries ``duration_s`` and (when available) ``size_bytes``.
          - ``image_pull_started`` + ``image_pull_failed`` — pull failed;
            raises :class:`_ImagePullError` so the caller can flip the
            run state to FAILED.

        Always emits AT LEAST ONE event so the frontend can render a
        deterministic "what's happening" status. Pre: ``paths.root`` exists.
        """
        if client.image_inspect(image):
            self._emit_event(client, paths.events_jsonl, "image_cache_hit", image=image)
            return
        self._emit_event(client, paths.events_jsonl, "image_pull_started", image=image)
        t0 = time.time()
        ok, err = client.image_pull(image)
        duration_s = time.time() - t0
        if not ok:
            self._emit_event(
                client,
                paths.events_jsonl,
                "image_pull_failed",
                image=image,
                duration_s=round(duration_s, 3),
                error=err[:500],
            )
            raise _ImagePullError(err[:200])
        # Best-effort size after pull (the docker manifest gives the
        # compressed size; the inspect gives the on-disk uncompressed size
        # — the latter is what most people mean by "image size").
        size_bytes = client.image_size(image)
        self._emit_event(
            client,
            paths.events_jsonl,
            "image_pulled",
            image=image,
            duration_s=round(duration_s, 3),
            size_bytes=size_bytes,
        )

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
        """Compose the worker command via the recipe builder.

        Returns the full argv: for real training, that's the
        ``docker run … training-image lerobot-train …`` argv; for the fake
        recipe (``__recipe__=__fake__``), the legacy
        ``python -m lerobot.gui.training.runner …`` argv.
        """
        cmd, _ = build_lerobot_train_command(run, paths)
        return cmd

    def _build_env(self, run: Run, paths: RunPaths) -> dict[str, str]:
        """Env vars passed to the worker subprocess.

        Recipe-builder env (e.g., HF_HUB_OFFLINE for future scratch-staging)
        plus a few orchestrator-side breadcrumbs the existing fake runner
        consults.
        """
        _, recipe_env = build_lerobot_train_command(run, paths)
        return {
            "LEROBOT_RUN_ID": run.run_id,
            "LEROBOT_RUN_DIR": str(paths.root),
            **recipe_env,
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
        host = self._hosts.get(run.host_id)
        client = self._client_for_host(host, paths)
        terminal_event = self._read_terminal_event(client, paths.events_jsonl)
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

        DESIGN.md § Health "Completion signal" — for the fake-training
        recipe, the worker writes the terminal event itself. For the real
        (docker) recipe, the orchestrator writes the terminal event when it
        detects the process has exited (since lerobot-train doesn't write
        our events.jsonl format).

        For both recipes, the orchestrator also incrementally appends to
        ``checkpoints.jsonl`` whenever it sees a new checkpoint directory
        on disk (via the bind-mounted output dir for docker mode, or in
        the worker's local dir for the fake mode).
        """
        # New checkpoints discovered on disk → appended to manifest. Cheap
        # filesystem scan, idempotent on re-poll.
        self._sync_checkpoints_manifest(client, run, paths)

        terminal_event = self._read_terminal_event(client, paths.events_jsonl)
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
            # Process gone, no terminal event. For the real-training (docker)
            # recipe this is the EXPECTED path — lerobot-train doesn't write
            # our events.jsonl. For the fake recipe, this is a crash.
            self._write_terminal_event_from_exit(client, run, paths)

    def _write_terminal_event_from_exit(self, client: TransportClient, run: Run, paths: RunPaths) -> None:
        """Process exited without writing a terminal event. Decide whether
        to call it ``completed_naturally`` or ``crashed`` based on artifacts.

        Heuristic: if at least one checkpoint dir was written, treat as
        completed; otherwise crashed. This is the right call for the docker
        recipe (lerobot-train only saves checkpoints on successful step
        boundaries; if it crashed early, no checkpoints exist).

        Aborted runs (state==COMPLETING after a Stop) → ``aborted_by_user``.
        """
        if run.state == RunState.COMPLETING:
            self._emit_event(client, paths.events_jsonl, "aborted_by_user")
            run.advance(RunState.ABORTED)
            self._runs.save(run)
            return

        ckpt_count = sum(1 for _ in self._iter_checkpoint_dirs(client, run, paths))
        progress = self._read_progress(client, paths.progress_json) or {}
        final_step = progress.get("step", 0) if isinstance(progress, dict) else 0

        if ckpt_count > 0:
            self._emit_event(client, paths.events_jsonl, "completed_naturally", final_step=final_step)
            run.advance(RunState.COMPLETED)
        else:
            run.error = "process exited without writing a checkpoint"
            self._emit_event(client, paths.events_jsonl, "crashed", error=run.error, final_step=final_step)
            run.advance(RunState.FAILED)
        self._runs.save(run)

    def _sync_checkpoints_manifest(self, client: TransportClient, run: Run, paths: RunPaths) -> None:
        """Append newly-discovered checkpoint dirs to ``checkpoints.jsonl``.

        Idempotent — skips dirs already in the manifest. Called on every
        poll, so the manifest catches up incrementally during a long run.
        Routes every file op via the transport so the same code works for
        local (subprocess) and remote (ssh) hosts.
        """
        already_seen_steps = {e.step for e in self._read_manifest(client, paths.checkpoints_jsonl)}
        for ckpt_dir, step in self._iter_checkpoint_dirs(client, run, paths):
            if step in already_seen_steps:
                continue
            # Locate the model file in the dir. Real lerobot-train layout
            # is ``<ckpt>/pretrained_model/model.safetensors``; fake-runner
            # writes ``<ckpt>/model.safetensors`` directly. Try both via
            # transport-routed listing.
            model_file: Path | None = None
            for child in client.list_dir(ckpt_dir):
                if child.name == "pretrained_model":
                    # nested layout — model.safetensors lives one level deeper
                    for grandchild in client.list_dir(child):
                        if grandchild.name == "model.safetensors":
                            model_file = grandchild
                            break
                    if model_file:
                        break
                elif child.name == "model.safetensors":
                    model_file = child
                    break
            if model_file is None:
                continue
            digest = client.sha256_of(model_file)
            if digest is None:
                continue
            rel_path = str(model_file.relative_to(paths.root))
            line = json.dumps({"step": step, "path": rel_path, "sha256": digest, "ts": time.time()})
            client.append_text(paths.checkpoints_jsonl, line + "\n")
            already_seen_steps.add(step)

    @staticmethod
    def _iter_checkpoint_dirs(client: TransportClient, run: Run, paths: RunPaths):
        """Yield ``(checkpoint_dir, step_number)`` pairs found on disk.

        Looks in the right place for the recipe:

        - lerobot-train: ``paths.root / "output" / "checkpoints" / <NNNNNN>/``
        - HVLA flow_s1:  ``paths.root / "output" / "checkpoints" / checkpoint-<N>/``
        - Fake:          ``paths.root / "checkpoints" / <NNNNNN>/``
        """
        subdir = output_subdir_in_run(run)
        ckpts_base = paths.root / subdir / "checkpoints" if subdir else paths.root / "checkpoints"
        # Parse first, then sort by step number — sorting directories by
        # name puts ``checkpoint-10`` before ``checkpoint-5`` (alphabetical)
        # which would write the manifest out of step order.
        pairs: list[tuple[Path, int]] = []
        for child in client.list_dir(ckpts_base):
            # Parse step number from the dir name. Two layouts supported:
            #   "000005"        → lerobot-train (zero-padded)
            #   "checkpoint-5"  → HVLA flow_matching trainer
            m = re.fullmatch(r"(?:checkpoint-)?0*(\d+)", child.name)
            if not m:
                continue
            pairs.append((child, int(m.group(1))))
        pairs.sort(key=lambda p: p[1])
        yield from pairs

    @staticmethod
    def _read_progress(client: TransportClient, progress_path: Path) -> dict[str, Any] | None:
        text = client.read_text(progress_path)
        if text is None:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _read_manifest(client: TransportClient, manifest_path: Path) -> list[CheckpointEntry]:
        text = client.read_text(manifest_path)
        if text is None:
            return []
        entries: list[CheckpointEntry] = []
        for line in text.splitlines():
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
    def _read_stderr_tail(client: TransportClient, stderr_path: Path, n_bytes: int) -> str:
        if n_bytes <= 0:
            return ""
        return client.read_tail(stderr_path, n_bytes).decode("utf-8", errors="replace")

    @staticmethod
    def _read_events(client: TransportClient, events_path: Path) -> list[dict[str, Any]]:
        """Read events.jsonl entries in order. Routes via the transport so
        SSH hosts work without bind-mounting events.jsonl back to the GUI
        server."""
        text = client.read_text(events_path)
        if text is None:
            return []
        out: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    @staticmethod
    def _read_terminal_event(client: TransportClient, events_path: Path) -> str | None:
        """Scan events.jsonl for a terminal event type. Returns the type name
        or None. Terminal events: completed_naturally / aborted_by_user / crashed.
        """
        text = client.read_text(events_path)
        if text is None:
            return None
        terminal = {"completed_naturally", "aborted_by_user", "crashed"}
        for line in text.splitlines():
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


# ── Module helpers ────────────────────────────────────────────────────────────


def _drop_run_metadata(paths: RunPaths) -> tuple[int, bool]:
    """Remove a run's metadata files (run.json, events.jsonl,
    checkpoints.jsonl, stderr.log, progress.json) but PRESERVE the
    ``output/checkpoints/`` subtree if it exists. That subtree is the
    trained model artefact; the Models tab continues to surface it after
    the run is dropped from history.

    If no model was ever written (no ``output/`` or it's empty of
    safetensors), the run dir is wholly removed — there's nothing to
    preserve.

    Returns ``(bytes_freed_metadata, kept_model)``. ``bytes_freed_metadata``
    counts only what we actually deleted (tens to hundreds of KB for the
    metadata case; full dir size for the no-model case). ``kept_model`` is
    True iff the model artefacts remain on disk.

    Idempotent: a second call on a run that's already been dropped (no
    metadata files present, only checkpoints) returns ``(0, True)``.
    """
    if not paths.root.exists():
        return 0, False
    metadata_files = [
        paths.run_json,
        paths.events_jsonl,
        paths.checkpoints_jsonl,
        paths.stderr_log,
        paths.progress_json,
    ]
    # Has-model probe — look for ANY .safetensors anywhere under
    # output/checkpoints/. If none, this is a no-model run (failed early
    # or aborted before the first save) and the whole dir is dead weight.
    ckpt_dir = (
        paths.checkpoints_dir
        if not (paths.root / "output").exists()
        else paths.root / "output" / "checkpoints"
    )
    has_model = ckpt_dir.is_dir() and any(ckpt_dir.rglob("*.safetensors"))

    if not has_model:
        # Nothing worth preserving. Tally + nuke. Note: callers gate this
        # behind a TERMINAL state check, so no worker is writing.
        total = 0
        for p in paths.root.rglob("*"):
            try:
                if p.is_file() and not p.is_symlink():
                    total += p.stat().st_size
            except OSError:
                continue
        # safe-destruct: orchestrator-owned <runs_dir>/<run_id>/ in terminal state with no model artefact
        shutil.rmtree(paths.root, ignore_errors=False)
        return total, False

    # Has a model → drop ONLY the metadata files. Checkpoints stay.
    freed = 0
    for f in metadata_files:
        if f.exists() and f.is_file():
            with contextlib.suppress(OSError):
                freed += f.stat().st_size
            with contextlib.suppress(OSError):
                # safe-destruct: orchestrator-owned metadata file in terminal-state run dir
                f.unlink()
    return freed, True


# ── Image preparation (pre-pull + cache check) ────────────────────────────────


class _ImagePullError(RuntimeError):
    """Raised internally when ``docker pull`` exits non-zero. Caught by
    :meth:`Orchestrator._prepare_and_launch` to flip the run to FAILED."""


def _extract_image_from_docker_argv(cmd: list[str]) -> str | None:
    """Pick the image tag out of a ``docker run ...`` argv.

    Returns ``None`` if ``cmd`` isn't a docker invocation (e.g., the fake
    recipe's ``python -m ...``). Recognising the docker recipe shape is
    cheap; full POSIX-style flag parsing would over-fit a structure we
    control.

    Convention from :func:`training_recipes.build_lerobot_train_command`:
    the image is the first positional after the last ``-v`` mount pair, and
    is followed by the entrypoint (``lerobot-train`` or
    ``python -u -m ...``). We just look for the first token after a
    sequence of recognised docker-flag pairs.
    """
    if not cmd or cmd[0] != "docker" or len(cmd) < 3 or cmd[1] != "run":
        return None
    i = 2
    # Skip flag pairs and standalone flags until we hit the image.
    # Recognised: --rm, --gpus all, --user UID:GID, -v X:Y, --network host, etc.
    while i < len(cmd):
        tok = cmd[i]
        if tok == "--rm":
            i += 1
            continue
        if tok in {
            "--gpus",
            "--user",
            "--network",
            "-v",
            "--volume",
            "-e",
            "--env",
            "--name",
            "--ipc",
            "--shm-size",
        }:
            i += 2
            continue
        if tok.startswith("-"):
            # Unknown flag — bail rather than guess pair vs. standalone.
            return None
        # First non-flag positional: this is the image.
        return tok
    return None
