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

import hashlib
import json
import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.gui.training_hosts import HostRegistry, TrainingHost
from lerobot.gui.training_recipes import (
    build_lerobot_train_command,
    output_subdir_in_run,
)
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
        image_runner: ImageRunner | None = None,
    ) -> None:
        self._hosts = host_registry
        self._runs = run_registry
        # Module name (not file path) so we invoke via `python -m`. Tests can
        # substitute a stub runner module without monkey-patching subprocess.
        self._runner_module = runner_module
        # Swappable shim for the two docker calls (inspect / pull). Tests
        # inject a fake; the default talks to the local docker daemon.
        self._image = image_runner or _DefaultImageRunner()
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
        # background thread emits ``image_cache_hit`` / ``pulling_image`` /
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
        client = make_client(host.transport) if host is not None else None

        # Reconcile state with the worker, if it's still in a live state.
        if run.state in (RunState.RUNNING, RunState.COMPLETING) and client is not None:
            self._reconcile_state(run, paths, client)

        progress = self._read_progress(paths.progress_json)
        checkpoints = self._read_manifest(paths.checkpoints_jsonl)
        stderr_tail = self._read_stderr_tail(paths.stderr_log, stderr_tail_bytes)
        events = _read_events(paths.events_jsonl)

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
        if run.state == RunState.PENDING:
            # Prep thread is still running (image pull or pre-launch). No
            # worker to SIGTERM. Skip COMPLETING straight to ABORTED — the
            # prep thread will reload run state before launching and bail
            # if it sees a terminal state. (If it raced past that check
            # already, the spawned worker is --rm so it cleans up on exit.)
            run.advance(RunState.ABORTED)
            self._runs.save(run)
            append_event(paths.events_jsonl, "aborted_by_user", final_step=0)
            return run
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
        append_event(paths.events_jsonl, "stop_requested")
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
        try:
            cmd = self._build_command(run, paths)
            image = _extract_image_from_docker_argv(cmd)
            if image is not None:
                self._ensure_image(image, paths)
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
            append_event(paths.events_jsonl, "crashed", error=str(exc), final_step=0)
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
            append_event(paths.events_jsonl, "crashed", error=str(exc), final_step=0)
            return
        # Final race check: stop() can land between launch and advance.
        # If so, kill what we just spawned. --rm on docker run cleans up the
        # container even if we miss this; the kill is best-effort UX.
        run_after = self._runs.load(run_id)
        if run_after is None or run_after.state != RunState.PENDING:
            try:
                make_client(host.transport).stop(session_id, force=True)
            except Exception:
                logger.exception("prepare-and-launch: post-launch kill failed (best-effort)")
            return
        run_after.session_id = session_id
        run_after.advance(RunState.RUNNING)
        self._runs.save(run_after)
        append_event(paths.events_jsonl, "started", session_id=session_id, host_id=host.id)

    def _ensure_image(self, image: str, paths: RunPaths) -> None:
        """Make sure ``image`` is present in the local docker cache.

        Emits one of:
          - ``image_cache_hit`` — image already local; no pull.
          - ``pulling_image`` + ``image_pulled`` — pull succeeded; latter
            carries ``duration_s`` and (when available) ``size_bytes``.
          - ``pulling_image`` + ``image_pull_failed`` — pull failed;
            raises :class:`_ImagePullError` so the caller can flip the
            run state to FAILED.

        Always emits AT LEAST ONE event so the frontend can render a
        deterministic "what's happening" status. Pre: ``paths.root`` exists.
        """
        if self._image.inspect(image):
            append_event(paths.events_jsonl, "image_cache_hit", image=image)
            return
        append_event(paths.events_jsonl, "pulling_image", image=image)
        t0 = time.time()
        ok, err = self._image.pull(image)
        duration_s = time.time() - t0
        if not ok:
            append_event(
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
        size_bytes = self._image.image_size(image)
        append_event(
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
        ``python -m lerobot.gui.training_runner …`` argv.
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
        self._sync_checkpoints_manifest(run, paths)

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
            # Process gone, no terminal event. For the real-training (docker)
            # recipe this is the EXPECTED path — lerobot-train doesn't write
            # our events.jsonl. For the fake recipe, this is a crash.
            self._write_terminal_event_from_exit(run, paths)

    def _write_terminal_event_from_exit(self, run: Run, paths: RunPaths) -> None:
        """Process exited without writing a terminal event. Decide whether
        to call it ``completed_naturally`` or ``crashed`` based on artifacts.

        Heuristic: if at least one checkpoint dir was written, treat as
        completed; otherwise crashed. This is the right call for the docker
        recipe (lerobot-train only saves checkpoints on successful step
        boundaries; if it crashed early, no checkpoints exist).

        Aborted runs (state==COMPLETING after a Stop) → ``aborted_by_user``.
        """
        if run.state == RunState.COMPLETING:
            append_event(paths.events_jsonl, "aborted_by_user")
            run.advance(RunState.ABORTED)
            self._runs.save(run)
            return

        ckpt_count = sum(1 for _ in self._iter_checkpoint_dirs(run, paths))
        progress = self._read_progress(paths.progress_json) or {}
        final_step = progress.get("step", 0) if isinstance(progress, dict) else 0

        if ckpt_count > 0:
            append_event(paths.events_jsonl, "completed_naturally", final_step=final_step)
            run.advance(RunState.COMPLETED)
        else:
            run.error = "process exited without writing a checkpoint"
            append_event(paths.events_jsonl, "crashed", error=run.error, final_step=final_step)
            run.advance(RunState.FAILED)
        self._runs.save(run)

    def _sync_checkpoints_manifest(self, run: Run, paths: RunPaths) -> None:
        """Append newly-discovered checkpoint dirs to ``checkpoints.jsonl``.

        Idempotent — skips dirs already in the manifest. Called on every
        poll, so the manifest catches up incrementally during a long run.
        """
        already_seen_steps = {e.step for e in self._read_manifest(paths.checkpoints_jsonl)}
        for ckpt_dir, step in self._iter_checkpoint_dirs(run, paths):
            if step in already_seen_steps:
                continue
            model_file = ckpt_dir / "pretrained_model" / "model.safetensors"
            if not model_file.exists():
                # Real lerobot-train layout; fake-runner uses checkpoints/<step>/model.safetensors
                # at the top of the step dir. Fall back to direct model file.
                candidates = list(ckpt_dir.glob("**/model.safetensors"))
                if not candidates:
                    continue
                model_file = candidates[0]
            try:
                digest = _sha256_of(model_file)
            except OSError:
                continue
            rel_path = str(model_file.relative_to(paths.root))
            line = json.dumps({"step": step, "path": rel_path, "sha256": digest, "ts": time.time()})
            paths.checkpoints_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with paths.checkpoints_jsonl.open("a") as f:
                f.write(line + "\n")
            already_seen_steps.add(step)

    @staticmethod
    def _iter_checkpoint_dirs(run: Run, paths: RunPaths):
        """Yield ``(checkpoint_dir, step_number)`` pairs found on disk.

        Looks in the right place for the recipe:

        - lerobot-train: ``paths.root / "output" / "checkpoints" / <NNNNNN>/``
        - HVLA flow_s1:  ``paths.root / "output" / "checkpoints" / checkpoint-<N>/``
        - Fake:          ``paths.root / "checkpoints" / <NNNNNN>/``
        """
        subdir = output_subdir_in_run(run)
        ckpts_base = paths.root / subdir / "checkpoints" if subdir else paths.root / "checkpoints"
        if not ckpts_base.is_dir():
            return
        for child in sorted(ckpts_base.iterdir()):
            if not child.is_dir():
                continue
            # Parse step number from the dir name. Two layouts supported:
            #   "000005"        → lerobot-train (zero-padded)
            #   "checkpoint-5"  → HVLA flow_matching trainer
            m = re.fullmatch(r"(?:checkpoint-)?0*(\d+)", child.name)
            if not m:
                continue
            yield child, int(m.group(1))

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


# ── Module helpers ────────────────────────────────────────────────────────────


def _sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming sha256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _read_events(events_path: Path) -> list[dict[str, Any]]:
    """Read all events from events.jsonl. Returns [] if the file is missing
    or every line malformed; otherwise returns valid entries in file order.

    Cheap (file is small — events.jsonl never exceeds a few KB per run).
    """
    if not events_path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in events_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


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


class ImageRunner:
    """Shim for the two docker calls C5 needs (``inspect`` and ``pull``).

    The default talks to the local docker daemon via subprocess; tests
    inject a fake to assert on which calls happen without touching docker.
    Kept narrow on purpose — the orchestrator should never grow a generic
    "run docker" surface, and this is the only API it needs.
    """

    def inspect(self, image: str) -> bool:  # pragma: no cover - abstract
        """Return True iff ``image`` exists locally."""
        raise NotImplementedError

    def pull(self, image: str) -> tuple[bool, str]:  # pragma: no cover
        """Pull ``image``. Returns ``(ok, stderr_tail)``."""
        raise NotImplementedError

    def image_size(self, image: str) -> int | None:  # pragma: no cover
        """On-disk size of the image in bytes, or None if unknown."""
        raise NotImplementedError


class _DefaultImageRunner(ImageRunner):
    """Production impl: shells out to ``docker`` on PATH.

    All three calls are bounded — ``inspect`` is ~50ms cache-hit, much
    faster on misses (immediate non-zero exit); ``image_size`` is similarly
    fast; ``pull`` is what we're trying to expose timing for, so it gets
    no extra timeout beyond what the user's host configures.
    """

    def inspect(self, image: str) -> bool:
        try:
            r = subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True,
                timeout=30,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        return r.returncode == 0

    def pull(self, image: str) -> tuple[bool, str]:
        try:
            r = subprocess.run(
                ["docker", "pull", image],
                capture_output=True,
                text=True,
                # No timeout — pulls genuinely take 10+ min on slow links.
            )
        except FileNotFoundError as exc:
            return False, f"docker binary not found: {exc}"
        if r.returncode != 0:
            # Combined tail — stderr is where pull failures land; include
            # stdout in case of a "Status: ..." line that helps diagnose.
            return False, (r.stderr or r.stdout or "")[-1000:]
        return True, ""

    def image_size(self, image: str) -> int | None:
        try:
            r = subprocess.run(
                ["docker", "image", "inspect", "-f", "{{.Size}}", image],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
        if r.returncode != 0:
            return None
        try:
            return int(r.stdout.strip())
        except ValueError:
            return None
