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

import contextlib
import dataclasses
import importlib
import pkgutil
import typing
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
from lerobot.gui.training.recipes import HVLA_FLOW_S1_FIELD_TO_FLAG, HVLA_FLOW_S1_RECIPE
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


# ── Policy catalog (GET /api/training/policies) ───────────────────────────────
#
# Dynamically discovers every PreTrainedConfig subclass via
# ``PreTrainedConfig.get_known_choices()`` and introspects its dataclass
# fields. Mirrors the same pattern used by api/robot.py's `_introspect_fields`
# + `/api/robot/schemas` — so what works for robot/teleop configs also
# works for policy configs without rebuilding from scratch.
#
# Manually-registered (non-draccus) recipes — currently just HVLA — are
# spliced in alongside the auto-discovered draccus policies so the
# frontend sees one uniform catalog.


_policies_loaded = False


def _ensure_policy_configs_loaded() -> None:
    """Import every submodule under ``lerobot.policies`` so each
    ``@PreTrainedConfig.register_subclass(...)`` decorator runs.

    Mirrors :func:`api.robot._ensure_configs_loaded` — auto-discovery via
    ``pkgutil.walk_packages`` so a new policy package picked up at runtime
    is exposed in the catalog without a code edit. Each import is wrapped
    in ``contextlib.suppress(Exception)`` so policies whose optional deps
    aren't installed (e.g. ``smolvla`` without ``transformers``) silently
    drop out instead of breaking discovery for the rest.
    """
    global _policies_loaded
    if _policies_loaded:
        return
    import lerobot.policies  # noqa: PLC0415

    for _importer, modname, _ispkg in pkgutil.walk_packages(
        lerobot.policies.__path__, prefix=lerobot.policies.__name__ + "."
    ):
        with contextlib.suppress(Exception):
            importlib.import_module(modname)
    _policies_loaded = True


# Field type classification for the frontend form renderer. Anything not
# in here is dropped from the schema — the form can't render a `list[str]`
# or a nested dataclass usefully, and falling through to a free-text input
# silently misleads the user. If you need a complex-typed field, hit
# /api/training/runs directly with the args dict.
_FORM_KIND_FROM_PY = {
    int: "int",
    float: "float",
    bool: "bool",
    str: "string",
}


def _classify_field(annotation: Any) -> tuple[str | None, list[str] | None]:
    """Map a type annotation to ``(form_kind, choices)`` for the frontend.

    Returns ``(None, None)`` if the field type isn't renderable in a form
    (lists, dicts, nested dataclasses, callables, ...). Handles
    ``Optional[X]`` by unwrapping. Handles ``Literal["a", "b"]`` by
    returning ``("select", ["a", "b"])``.
    """
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    # Optional[X] → Union[X, None]: unwrap and recurse on X.
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _classify_field(non_none[0])
        return None, None

    if origin is typing.Literal:
        return "select", [str(v) for v in args]

    if annotation in _FORM_KIND_FROM_PY:
        return _FORM_KIND_FROM_PY[annotation], None

    return None, None


def _introspect_policy_fields(cls: type) -> list[dict]:
    """Extract a frontend-renderable schema from a dataclass config class.

    Skips fields whose type isn't a scalar (int/float/bool/str) or a
    ``Literal[...]``. Skips fields without a default (we can't pre-fill
    the form and don't want to require them blindly). Sorts: required-ish
    first, then alphabetical name.
    """
    try:
        type_hints = typing.get_type_hints(cls)
    except Exception:
        type_hints = {}

    out: list[dict] = []
    for f in dataclasses.fields(cls):
        resolved = type_hints.get(f.name, f.type)
        form_kind, choices = _classify_field(resolved)
        if form_kind is None:
            continue  # not renderable in a form

        # Resolve default: literal default beats factory beats None.
        default: Any = None
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            try:
                default = f.default_factory()  # type: ignore[misc]
            except Exception:
                default = None

        entry: dict[str, Any] = {
            "name": f.name,
            "label": _humanize_field_name(f.name),
            "type": form_kind,
            "default": default,
        }
        if choices is not None:
            entry["choices"] = choices
        description = f.metadata.get("description")
        if description:
            entry["description"] = description
        out.append(entry)

    out.sort(key=lambda e: (e["default"] is None and e["type"] != "bool", e["name"]))
    return out


def _humanize_field_name(name: str) -> str:
    """Render a snake_case field name as a Title Case label."""
    return name.replace("_", " ").strip().capitalize()


def _humanize_policy_name(type_name: str) -> str:
    """Fallback label for a registered policy with no curated display name."""
    return type_name.replace("_", " ").upper()


# Curated display labels for the policies the GUI exposes by default. Any
# auto-discovered policy without an entry here gets the humanized fallback
# from :func:`_humanize_policy_name`. The point is to keep the *catalog*
# fully dynamic while letting the label remain hand-tuned where it
# matters (so "act" → "ACT (Action Chunking Transformer)" instead of
# "ACT").
_POLICY_LABELS = {
    "act": "ACT (Action Chunking Transformer)",
    "diffusion": "Diffusion Policy",
    "smolvla": "SmolVLA",
    "sac": "SAC (Soft Actor-Critic)",
    "tdmpc": "TD-MPC",
    "vqbet": "VQ-BeT",
    "pi0": "Pi-0",
    "pi05": "Pi-0.5",
    "pi0_fast": "Pi-0 FAST",
    "multi_task_dit": "Multi-task DiT",
    "groot": "GR00T",
    "wall_x": "Wall-X",
    "xvla": "X-VLA",
    "act_vlm": "ACT-VLM",
    "sarm": "SARM",
    "reward_classifier": "Reward classifier",
}


# Manually-registered recipes — anything that doesn't go through
# ``lerobot-train``'s draccus CLI. HVLA's flow_matching trainer has its
# own argparse; the args-key convention is snake_case (no ``policy.``
# prefix). We surface a hand-curated field list here because HVLA's
# argparse isn't a dataclass — there's nothing to introspect dynamically.
# This is the one place hand-curation survives, and only because the
# underlying trainer hasn't yet been migrated to a dataclass-based config.
_NON_DRACCUS_RECIPES: list[dict[str, Any]] = [
    {
        "type_name": "hvla_flow_s1",
        "label": "HVLA Flow Matching S1 (no S2)",
        "recipe": HVLA_FLOW_S1_RECIPE,
        "arg_key_prefix": "",  # HVLA's keys are bare snake_case
        "fields": [
            {"name": "chunk_size", "label": "Chunk size", "type": "int", "default": 50},
            {"name": "num_inference_steps", "label": "Inference steps", "type": "int", "default": 15},
            {"name": "hidden_dim", "label": "Hidden dim", "type": "int", "default": 768},
            {"name": "num_decoder_layers", "label": "Decoder layers", "type": "int", "default": 6},
            {"name": "num_workers", "label": "Data workers", "type": "int", "default": 4},
        ],
        # Make explicit which form keys map to the trainer's CLI; the
        # frontend doesn't need to know but it's useful in tests + docs.
        "_known_flag_map": list(HVLA_FLOW_S1_FIELD_TO_FLAG),
    },
]


@router.get("/policies")
def list_policies() -> list[dict]:
    """Catalog of policies the GUI can launch.

    Auto-discovers everything registered with ``PreTrainedConfig`` (via
    ``@PreTrainedConfig.register_subclass(name)``) — adding a new policy
    upstream that follows the registration pattern surfaces in the GUI
    with no code edit here. Manually-registered non-draccus recipes
    (HVLA) are spliced in alongside.

    Each entry: ``{type_name, label, recipe, arg_key_prefix, fields}``
    where ``recipe`` is null for plain ``lerobot-train`` and a marker
    string (e.g. ``"hvla_flow_s1"``) for routed recipes, and
    ``arg_key_prefix`` tells the frontend whether to prepend
    ``"policy."`` to each field name when building the args dict.
    """
    _ensure_policy_configs_loaded()
    from lerobot.configs.policies import PreTrainedConfig  # noqa: PLC0415

    schemas: list[dict] = []
    for type_name, cls in sorted(PreTrainedConfig.get_known_choices().items()):
        fields = _introspect_policy_fields(cls)
        if not fields:
            # Policy registered but no renderable fields — skip rather
            # than emit a useless entry. Picked up automatically if the
            # config grows scalar fields later.
            continue
        schemas.append(
            {
                "type_name": type_name,
                "label": _POLICY_LABELS.get(type_name) or _humanize_policy_name(type_name),
                "recipe": None,  # default: lerobot-train via draccus
                "arg_key_prefix": "policy.",
                "fields": fields,
            }
        )

    # Splice in non-draccus recipes (HVLA, future custom trainers).
    for entry in _NON_DRACCUS_RECIPES:
        # Don't leak the introspection helper into the API; strip
        # internal-only keys before serialising.
        external = {k: v for k, v in entry.items() if not k.startswith("_")}
        schemas.append(external)
    return schemas
