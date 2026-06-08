# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Training API: orchestrator-backed run CRUD + hosts listing.

Endpoints, all under ``/api/training``:

- ``GET /hosts``                — list training hosts (workstation auto-detected)
- ``GET /runs``                 — list all runs (newest first)
- ``POST /runs``                — start a new run
- ``GET /runs/{run_id}``        — snapshot for one run (state, progress, checkpoints, log tail)
- ``POST /runs/{run_id}/stop``  — user-initiated stop

The router is constructed lazily — the orchestrator + host registry are
injected via a module-level state object so server.py can wire them at startup
with the right runs_dir. See :func:`init_state` and :func:`get_state`.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lerobot.gui.training.hosts import HostRegistry
from lerobot.gui.training.orchestrator import (
    HostBusyError,
    Orchestrator,
    RunNotTerminalError,
    RunSnapshot,
    StartRequest,
    UnknownHostError,
    UnknownRunError,
)
from lerobot.gui.training.runs import RUNS_DIR, RunRegistry

router = APIRouter(prefix="/api/training", tags=["training"])


# ── Lazy state ────────────────────────────────────────────────────────────────


# Held in module scope so server.py can call init_state() once at startup
# and the route handlers can read it. Keeps this module importable for tests
# without a full app context.
_state: dict[str, Any] = {"orch": None, "host_registry": None}


def init_state(orch: Orchestrator, host_registry: HostRegistry) -> None:
    """Wire the orchestrator + host registry the routes will use.

    Called once at GUI server startup. Tests can call it with a custom
    orchestrator + registry without spinning up the full app.
    """
    _state["orch"] = orch
    _state["host_registry"] = host_registry


def get_state() -> tuple[Orchestrator, HostRegistry]:
    """Return the wired-up orchestrator + host registry.

    Raises if :func:`init_state` was never called.
    """
    orch = _state["orch"]
    host_registry = _state["host_registry"]
    if orch is None or host_registry is None:
        raise RuntimeError("Training API state not initialized; call init_state() at server startup.")
    return orch, host_registry


def reset_state_for_testing() -> None:
    """Test helper: reset the module-level state between fixtures."""
    _state["orch"] = None
    _state["host_registry"] = None


def make_default_orchestrator(runs_dir: Path | None = None) -> Orchestrator:
    """Build a stock Orchestrator + auto-detected HostRegistry pair.

    Convenience for server startup. Pass ``runs_dir`` to override the default
    (``~/.cache/lerobot/runs``) — tests use a tmp dir.
    """
    runs_dir = runs_dir or RUNS_DIR
    host_registry = HostRegistry.auto(workdir=runs_dir)
    run_registry = RunRegistry(runs_dir=runs_dir)
    return Orchestrator(host_registry=host_registry, run_registry=run_registry)


# ── Schemas ───────────────────────────────────────────────────────────────────


class HostInfo(BaseModel):
    id: str
    display_name: str
    transport_kind: str  # "subprocess" or "ssh"
    capabilities: dict[str, Any]


class StartRunBody(BaseModel):
    host_id: str
    recipe_name: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None


class RunDTO(BaseModel):
    run_id: str
    host_id: str
    recipe_name: str
    dataset_id: str
    args: dict[str, Any]
    state: str
    created_at: float
    started_at: float | None
    finished_at: float | None
    session_id: int | None
    error: str | None


class CheckpointDTO(BaseModel):
    step: int
    path: str
    sha256: str
    ts: float


class RunSnapshotDTO(BaseModel):
    run: RunDTO
    progress: dict[str, Any] | None
    checkpoints: list[CheckpointDTO]
    stderr_tail: str
    # Raw events.jsonl entries (oldest first). Frontend filters for
    # image-prep events to render the "Pulling image…" status banner.
    events: list[dict[str, Any]]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _run_to_dto(run) -> RunDTO:
    d = asdict(run)
    d["state"] = run.state.value
    # Drop fields that aren't part of the API surface
    d.pop("idempotency_key", None)
    return RunDTO(**d)


def _snapshot_to_dto(snap: RunSnapshot) -> RunSnapshotDTO:
    return RunSnapshotDTO(
        run=_run_to_dto(snap.run),
        progress=snap.progress,
        checkpoints=[
            CheckpointDTO(step=c.step, path=c.path, sha256=c.sha256, ts=c.ts) for c in snap.checkpoints
        ],
        stderr_tail=snap.stderr_tail,
        events=list(snap.events),
    )


def _transport_kind(transport: Any) -> str:
    cls = transport.__class__.__name__
    if cls == "SubprocessTransport":
        return "subprocess"
    if cls == "SshTransport":
        return "ssh"
    return cls


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/hosts", response_model=list[HostInfo])
def list_hosts() -> list[HostInfo]:
    """List all training hosts. v1: workstation host if a GPU is auto-detected."""
    _, hosts = get_state()
    return [
        HostInfo(
            id=h.id,
            display_name=h.display_name,
            transport_kind=_transport_kind(h.transport),
            capabilities=h.capabilities,
        )
        for h in hosts.list_hosts()
    ]


@router.get("/runs", response_model=list[RunDTO])
def list_runs() -> list[RunDTO]:
    """List all runs, newest first."""
    orch, _ = get_state()
    return [_run_to_dto(r) for r in orch.list_runs()]


@router.post("/runs", response_model=RunDTO, status_code=201)
def start_run(body: StartRunBody) -> RunDTO:
    """Start a training run.

    Returns 409 if the target host already has an active run.
    Returns 404 if the host id isn't registered.
    """
    orch, _ = get_state()
    try:
        run = orch.start(
            StartRequest(
                host_id=body.host_id,
                recipe_name=body.recipe_name,
                dataset_id=body.dataset_id,
                args=body.args,
                idempotency_key=body.idempotency_key,
            )
        )
    except UnknownHostError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HostBusyError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return _run_to_dto(run)


@router.get("/runs/{run_id}", response_model=RunSnapshotDTO)
def get_run(run_id: str) -> RunSnapshotDTO:
    """Snapshot one run: state, progress.json, checkpoints manifest, stderr tail."""
    orch, _ = get_state()
    try:
        snap = orch.poll(run_id)
    except UnknownRunError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return _snapshot_to_dto(snap)


@router.post("/runs/{run_id}/stop", response_model=RunDTO)
def stop_run(run_id: str) -> RunDTO:
    """User-initiated stop. Idempotent on already-terminal runs."""
    orch, _ = get_state()
    try:
        run = orch.stop(run_id)
    except UnknownRunError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return _run_to_dto(run)


class DeleteRunResponse(BaseModel):
    run_id: str
    # Metadata-only delete; ``kept_model`` says whether the checkpoint
    # directory survived (= Models tab still has the artefact).
    metadata_bytes_freed: int
    kept_model: bool


class ClearTerminalResponse(BaseModel):
    deleted: list[str]
    metadata_bytes_freed: int
    models_kept: int


@router.delete("/runs/{run_id}", response_model=DeleteRunResponse)
def delete_run(run_id: str) -> DeleteRunResponse:
    """Drop a terminal-state run from training history.

    Metadata-only delete: removes the run row from the Training list but
    keeps ``output/checkpoints/`` so the trained model continues to show
    up in the Models tab. If the run produced no model (failed pull,
    aborted before first save), the whole dir is removed.

    Returns 409 if the run is still active (caller must Stop first).
    Returns 404 if the run id is unknown.
    """
    orch, _ = get_state()
    try:
        result = orch.delete_run(run_id)
    except UnknownRunError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RunNotTerminalError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return DeleteRunResponse(**result)


@router.post("/runs/clear", response_model=ClearTerminalResponse)
def clear_terminal_runs() -> ClearTerminalResponse:
    """Drop every terminal-state run from training history in one shot.

    Metadata-only per row (see :func:`delete_run`). Active runs are
    skipped. Idempotent: a second call after the first returns
    ``deleted=[]``.
    """
    orch, _ = get_state()
    result = orch.clear_terminal_runs()
    return ClearTerminalResponse(**result)
