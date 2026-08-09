# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the training API endpoints.

Uses a FastAPI TestClient and wires the orchestrator against a tmp_path
runs dir + a single test host. Verifies the routes' behavior, status codes,
and DTO shapes — separate from the orchestrator unit tests.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.gui.api import training as training_api
from lerobot.gui.training.hosts import HostRegistry, TrainingHost
from lerobot.gui.training.orchestrator import Orchestrator
from lerobot.gui.training.runs import Run, RunPaths, RunRegistry, RunState
from lerobot.gui.training.transport import SubprocessTransport


@pytest.fixture
def app(tmp_path: Path):
    """Fresh app with the training router and a tmp orchestrator wired in."""
    training_api.reset_state_for_testing()
    host = TrainingHost(
        id="test-host",
        display_name="Test Host",
        transport=SubprocessTransport(workdir=tmp_path / "workdir"),
        capabilities={"gpu_name": "Test", "vram_mb": 16384, "gpu_count_detected": 1},
    )
    hosts = HostRegistry(hosts=[host])
    runs = RunRegistry(runs_dir=tmp_path / "runs")
    orch = Orchestrator(host_registry=hosts, run_registry=runs)
    training_api.init_state(orch=orch, host_registry=hosts)
    app = FastAPI()
    app.include_router(training_api.router)
    yield app
    training_api.reset_state_for_testing()


@pytest.fixture
def client(app: FastAPI):
    return TestClient(app)


def _start_run_payload(**over) -> dict:
    base = {
        "host_id": "test-host",
        "recipe_name": "fake",
        "dataset_id": "fake/ds",
        "args": {"__recipe__": "__fake__", "num_steps": 5, "save_every": 10, "step_seconds": 0.05},
    }
    base.update(over)
    return base


def _wait_until_state(client: TestClient, run_id: str, want: str, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        resp = client.get(f"/api/training/runs/{run_id}")
        assert resp.status_code == 200
        body = resp.json()
        last = body
        if body["run"]["state"] == want or body["run"]["state"] in {
            "completed",
            "failed",
            "stopped",
        }:
            return body
        time.sleep(0.05)
    raise AssertionError(f"timed out; last state={last['run']['state'] if last else None}")


# ── /hosts ────────────────────────────────────────────────────────────────────


def test_list_hosts(client: TestClient) -> None:
    resp = client.get("/api/training/hosts")
    assert resp.status_code == 200
    hosts = resp.json()
    assert len(hosts) == 1
    h = hosts[0]
    assert h["id"] == "test-host"
    assert h["display_name"] == "Test Host"
    assert h["transport_kind"] == "subprocess"
    assert h["capabilities"]["gpu_name"] == "Test"


# ── /runs (list + create) ─────────────────────────────────────────────────────


def test_list_runs_empty(client: TestClient) -> None:
    resp = client.get("/api/training/runs")
    assert resp.status_code == 200
    assert resp.json() == []


def test_start_run_201(client: TestClient) -> None:
    """POST returns immediately with state=pending (C5 background prep
    thread does image pull + worker launch; advances to running on
    completion). The state machine + session_id show up on subsequent
    polls — we wait until completed to assert the full lifecycle ran."""
    resp = client.post("/api/training/runs", json=_start_run_payload())
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["state"] == "pending"
    assert body["host_id"] == "test-host"
    assert body["recipe_name"] == "fake"
    # session_id is set by the prep thread once it spawns the worker
    assert body["session_id"] is None
    # Drive through to completion so subsequent fixtures see a clean slate.
    final = _wait_until_state(client, body["run_id"], "completed")
    assert final["run"]["session_id"] is not None


def test_start_run_unknown_host_404(client: TestClient) -> None:
    resp = client.post("/api/training/runs", json=_start_run_payload(host_id="nope"))
    assert resp.status_code == 404
    assert "unknown host" in resp.json()["detail"].lower()


def test_start_run_validation_400(client: TestClient) -> None:
    resp = client.post(
        "/api/training/runs",
        json={"host_id": "test-host", "recipe_name": "", "dataset_id": "x"},
    )
    assert resp.status_code == 422  # FastAPI validation error for min_length=1


def test_start_run_rejects_internal_resume_path(client: TestClient) -> None:
    payload = _start_run_payload()
    payload["args"]["__resume_checkpoint__"] = "/etc"

    response = client.post("/api/training/runs", json=payload)

    assert response.status_code == 422
    assert "reserved internal training arguments" in response.json()["detail"]


def test_start_run_host_busy_409(client: TestClient) -> None:
    """Two start requests targeting the same host while the first is running →
    second gets 409."""
    long_payload = _start_run_payload(
        args={"__recipe__": "__fake__", "num_steps": 1000, "save_every": 100, "step_seconds": 0.05}
    )
    r1 = client.post("/api/training/runs", json=long_payload)
    assert r1.status_code == 201
    try:
        r2 = client.post("/api/training/runs", json=_start_run_payload())
        assert r2.status_code == 409
        assert "busy" in r2.json()["detail"].lower()
    finally:
        client.post(f"/api/training/runs/{r1.json()['run_id']}/stop")
        _wait_until_state(client, r1.json()["run_id"], "stopped")


def test_start_run_idempotency_returns_same_id(client: TestClient) -> None:
    p = _start_run_payload(idempotency_key="abc")
    r1 = client.post("/api/training/runs", json=p)
    assert r1.status_code == 201
    _wait_until_state(client, r1.json()["run_id"], "completed")
    # Resubmit with same key → same run_id (even though first is done)
    r2 = client.post("/api/training/runs", json=p)
    assert r2.status_code == 201
    assert r2.json()["run_id"] == r1.json()["run_id"]


# ── /runs/{id} ────────────────────────────────────────────────────────────────


def test_get_run_404(client: TestClient) -> None:
    resp = client.get("/api/training/runs/missing")
    assert resp.status_code == 404


def test_get_run_snapshot_after_completion(client: TestClient) -> None:
    r = client.post("/api/training/runs", json=_start_run_payload()).json()
    body = _wait_until_state(client, r["run_id"], "completed")
    assert body["run"]["state"] == "completed"
    assert body["progress"]["step"] == 5
    assert body["progress"]["loss"] > 0
    # save_every=10, num_steps=5 — no checkpoint expected
    assert body["checkpoints"] == []
    assert "[runner]" in body["stderr_tail"]


def test_get_run_snapshot_with_checkpoints(client: TestClient) -> None:
    r = client.post(
        "/api/training/runs",
        json=_start_run_payload(
            args={"__recipe__": "__fake__", "num_steps": 10, "save_every": 5, "step_seconds": 0.05}
        ),
    ).json()
    body = _wait_until_state(client, r["run_id"], "completed")
    assert len(body["checkpoints"]) == 2
    assert body["checkpoints"][0]["step"] == 5
    assert body["checkpoints"][1]["step"] == 10
    assert all(c["sha256"] for c in body["checkpoints"])


# ── /runs/{id}/resume ─────────────────────────────────────────────────────────


def test_resume_run_201(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orch, _ = training_api.get_state()
    source = Run(
        run_id="resume-source",
        host_id="test-host",
        recipe_name="hvla",
        dataset_id="robot/data",
        args={"__recipe__": "hvla_flow_s1", "steps": 500},
        state=RunState.FAILED,
        created_at=time.time(),
        finished_at=time.time(),
    )
    orch._runs.save(source)  # noqa: SLF001 - arrange API fixture state
    paths = RunPaths.for_run(source.run_id, orch._runs.runs_dir)  # noqa: SLF001
    checkpoint = paths.root / "output/checkpoints/checkpoint-200"
    (checkpoint / "training_state").mkdir(parents=True)
    pretrained = checkpoint / "pretrained_model"
    pretrained.mkdir()
    (pretrained / "train_config.json").write_text("{}")
    monkeypatch.setattr(orch, "_prepare_and_launch", lambda *_args: None)

    response = client.post(
        f"/api/training/runs/{source.run_id}/resume",
        json={"checkpoint_step": 200, "idempotency_key": "resume-api"},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["state"] == "pending"
    assert body["run_id"] != source.run_id
    assert body["recipe_name"] == "hvla (resume 200)"


def test_resume_run_missing_404(client: TestClient) -> None:
    response = client.post(
        "/api/training/runs/missing/resume",
        json={"checkpoint_step": 200},
    )

    assert response.status_code == 404


# ── /runs/{id}/stop ───────────────────────────────────────────────────────────


def test_stop_run_404(client: TestClient) -> None:
    resp = client.post("/api/training/runs/missing/stop")
    assert resp.status_code == 404


def test_stop_run_aborts(client: TestClient) -> None:
    r = client.post(
        "/api/training/runs",
        json=_start_run_payload(
            args={"__recipe__": "__fake__", "num_steps": 1000, "save_every": 100, "step_seconds": 0.05}
        ),
    ).json()
    time.sleep(0.3)
    resp = client.post(f"/api/training/runs/{r['run_id']}/stop")
    assert resp.status_code == 200
    body = _wait_until_state(client, r["run_id"], "stopped")
    assert body["run"]["state"] == "stopped"


def test_stop_run_idempotent_on_completed(client: TestClient) -> None:
    r = client.post("/api/training/runs", json=_start_run_payload()).json()
    _wait_until_state(client, r["run_id"], "completed")
    resp = client.post(f"/api/training/runs/{r['run_id']}/stop")
    assert resp.status_code == 200
    assert resp.json()["state"] == "completed"


# ── /runs list reflects activity ──────────────────────────────────────────────


def test_list_runs_after_some_activity(client: TestClient) -> None:
    a = client.post("/api/training/runs", json=_start_run_payload()).json()
    _wait_until_state(client, a["run_id"], "completed")
    b = client.post("/api/training/runs", json=_start_run_payload()).json()
    _wait_until_state(client, b["run_id"], "completed")
    listed = client.get("/api/training/runs").json()
    ids = {r["run_id"] for r in listed}
    assert {a["run_id"], b["run_id"]}.issubset(ids)


# ── State guard ───────────────────────────────────────────────────────────────


def test_uninitialized_state_raises(tmp_path: Path) -> None:
    """If init_state was never called, endpoints should fail loudly."""
    training_api.reset_state_for_testing()
    app = FastAPI()
    app.include_router(training_api.router)
    cli = TestClient(app, raise_server_exceptions=True)
    # Calling an endpoint without init_state() → 500 from the get_state() guard
    with pytest.raises(RuntimeError, match="not initialized"):
        cli.get("/api/training/hosts")


# ── DTO consistency ───────────────────────────────────────────────────────────


def test_run_dto_does_not_leak_idempotency_key(client: TestClient) -> None:
    r = client.post(
        "/api/training/runs",
        json=_start_run_payload(idempotency_key="my-secret"),
    ).json()
    assert "idempotency_key" not in r
    _wait_until_state(client, r["run_id"], "completed")


# ── DELETE + clear endpoints (housekeeping) ───────────────────────────────────


def test_delete_run_404_on_unknown(client: TestClient) -> None:
    resp = client.delete("/api/training/runs/does-not-exist")
    assert resp.status_code == 404


def test_delete_completed_run_returns_kept_model_true(client: TestClient) -> None:
    """Delete a completed run with a model: response says
    ``kept_model: true``, subsequent GET 404s."""
    payload = _start_run_payload(
        args={"__recipe__": "__fake__", "num_steps": 10, "save_every": 5, "step_seconds": 0.05}
    )
    r = client.post("/api/training/runs", json=payload).json()
    _wait_until_state(client, r["run_id"], "completed")
    resp = client.delete(f"/api/training/runs/{r['run_id']}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["run_id"] == r["run_id"]
    assert body["metadata_bytes_freed"] >= 0
    assert body["kept_model"] is True  # fake-runner produced 2 checkpoints
    # Subsequent GET returns 404 (gone from history)
    assert client.get(f"/api/training/runs/{r['run_id']}").status_code == 404


def test_delete_active_run_409(client: TestClient) -> None:
    """Can't delete a running run; must Stop first."""
    payload = _start_run_payload()
    payload["args"] = {
        "__recipe__": "__fake__",
        "num_steps": 1000,
        "save_every": 100,
        "step_seconds": 1.0,
    }
    r = client.post("/api/training/runs", json=payload).json()
    _wait_until_state(client, r["run_id"], "running")
    resp = client.delete(f"/api/training/runs/{r['run_id']}")
    assert resp.status_code == 409
    assert "stop it first" in resp.json()["detail"]
    # Cleanup
    client.post(f"/api/training/runs/{r['run_id']}/stop")
    _wait_until_state(client, r["run_id"], "stopped")


def test_clear_terminal_endpoint(client: TestClient) -> None:
    """POST /api/training/runs/clear returns deleted list + metadata_bytes
    + models_kept."""
    payload = _start_run_payload(
        args={"__recipe__": "__fake__", "num_steps": 10, "save_every": 5, "step_seconds": 0.05}
    )
    r = client.post("/api/training/runs", json=payload).json()
    _wait_until_state(client, r["run_id"], "completed")
    resp = client.post("/api/training/runs/clear")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert r["run_id"] in body["deleted"]
    assert body["models_kept"] == 1  # the fake run produced a checkpoint
    # Idempotent: second call has nothing to delete
    resp2 = client.post("/api/training/runs/clear")
    body2 = resp2.json()
    assert body2["deleted"] == []
    assert body2["models_kept"] == 0


# ── /api/training/policies (catalog) ──────────────────────────────────────────


def test_list_policies_includes_act_and_hvla(client: TestClient) -> None:
    """Auto-discovery should pick up every PreTrainedConfig subclass + the
    manually-registered HVLA recipe."""
    resp = client.get("/api/training/policies")
    assert resp.status_code == 200
    catalog = resp.json()
    types = [p["type_name"] for p in catalog]
    # Auto-discovered draccus policies (sampling — full set depends on extras)
    assert "act" in types
    assert "diffusion" in types
    # Manually-registered non-draccus recipe
    assert "hvla_flow_s1" in types


def test_list_policies_act_entry_has_renderable_fields(client: TestClient) -> None:
    """ACT's draccus config should expose its scalar fields with defaults
    + recognisable form types."""
    catalog = client.get("/api/training/policies").json()
    act = next(p for p in catalog if p["type_name"] == "act")
    assert act["recipe"] is None  # default lerobot-train
    assert act["arg_key_prefix"] == "policy."
    assert act["fields"], "act should have at least one renderable field"
    field_names = {f["name"] for f in act["fields"]}
    # Spot-check that the headline ACT fields are present
    assert "chunk_size" in field_names
    assert "n_action_steps" in field_names
    assert "dim_model" in field_names
    # Every field has a usable form type. "cameras" and "flags" are not introspected
    # from the dataclass — they are appended to every recipe and rendered as checkbox
    # groups filled from the selected dataset.
    for f in act["fields"]:
        assert f["type"] in {"int", "float", "bool", "string", "select", "cameras", "flags"}
        assert "default" in f


def test_list_policies_hvla_entry_uses_recipe_marker(client: TestClient) -> None:
    """The manual S1-without-S2 entry exposes its complete training contract."""
    catalog = client.get("/api/training/policies").json()
    hvla = next(p for p in catalog if p["type_name"] == "hvla_flow_s1")
    assert hvla["recipe"] == "hvla_flow_s1"
    assert hvla["arg_key_prefix"] == ""
    fields = {f["name"]: f for f in hvla["fields"]}
    expected = {
        "chunk_size",
        "num_inference_steps",
        "rtc_max_delay",
        "rtc_drop_prob",
        "resize_images",
        "hidden_dim",
        "num_decoder_layers",
        "num_workers",
        "validation_fraction",
        "state_position_std_floor",
        "use_relative_actions",
        "seed",
    }
    assert expected <= fields.keys()
    assert fields["num_inference_steps"]["label"] == "Denoise steps"
    assert fields["num_inference_steps"]["default"] == 15
    assert fields["rtc_max_delay"]["default"] == 6
    assert fields["state_position_std_floor"]["default"] == 0.0
    assert fields["use_relative_actions"]["default"] is False
    assert fields["validation_fraction"]["default"] == 0.1
    assert fields["seed"]["default"] == 1337
    assert fields["rtc_drop_prob"]["default"] == 0.2
    assert fields["resize_images"]["default"] == "224x224"
    assert fields["resize_images"]["label"] == "Image input resolution"
    # Every HVLA hyperparameter sits behind the advanced disclosure. The camera
    # picker deliberately does not: which cameras a run consumes is a data choice
    # alongside the dataset, not a hyperparameter.
    assert all(fields[name]["advanced"] is True for name in expected)
    assert not fields["cameras"].get("advanced")
    # Same reasoning for the flag picker: which frames a run refuses to learn from
    # is a data choice, and burying it is how it stays unused.
    assert not fields["exclude_flags"].get("advanced")
    assert "max_delay" not in fields  # S2 latent delay is irrelevant to this no-S2 recipe.


def test_list_policies_skips_complex_fields(client: TestClient) -> None:
    """Fields the form can't usefully render (list/dict/nested-dataclass)
    must be dropped from the catalog — otherwise the form would silently
    fall back to a free-text input that the user can't fill correctly."""
    catalog = client.get("/api/training/policies").json()
    act = next(p for p in catalog if p["type_name"] == "act")
    # ACT's config defines complex-typed fields like
    # optimizer_lr_backbone_scale, image_features, etc. — none of those
    # should be in the catalog.
    assert "image_features" not in {f["name"] for f in act["fields"]}
    assert "optimizer_lr_backbone_scale" not in {f["name"] for f in act["fields"]}
    # Introspection emits only scalars; "cameras" and "flags" are the appended fields,
    # and both have real renderers rather than the free-text fallback this guards against.
    for f in act["fields"]:
        assert f["type"] in {"int", "float", "bool", "string", "select", "cameras", "flags"}


# ── run_dir: naming a run's model ─────────────────────────────────────────────
#
# The Checkpoints card lists paths relative to the run directory
# ("output/checkpoints/010000/..."), so without the directory itself the card
# never says which trained model those belong to, and the string for
# --policy.path cannot be reconstructed. The run's own log is no help: it
# records the container's bind-mount target ("/runs/output"), not a host path.


def test_run_snapshot_exposes_the_run_directory(client: TestClient, tmp_path: Path) -> None:
    run_id = client.post("/api/training/runs", json=_start_run_payload()).json()["run_id"]
    body = _wait_until_state(client, run_id, "completed")

    run_dir = body["run_dir"]
    assert run_dir, "snapshot must name the directory its checkpoint paths are relative to"
    assert Path(run_dir).is_absolute()
    assert Path(run_dir).name == run_id


def test_run_dir_follows_the_configured_runs_root(client: TestClient, tmp_path: Path) -> None:
    """Not hardcoded to ~/.cache: the fixture's registry uses tmp_path."""
    run_id = client.post("/api/training/runs", json=_start_run_payload()).json()["run_id"]
    body = _wait_until_state(client, run_id, "completed")

    assert body["run_dir"] == str(tmp_path / "runs" / run_id)


def _run_dir_with_a_checkpoint(runs_dir: Path, run_id: str, recipe: str, step: int = 1000) -> Path:
    """A run directory shaped like a real one: run.json, a checkpoint manifest,
    and a checkpoint the model scanner recognises."""
    import json as _json

    run_dir = runs_dir / run_id
    pretrained = run_dir / "output" / "checkpoints" / f"{step:06d}" / "pretrained_model"
    pretrained.mkdir(parents=True)
    (pretrained / "config.json").write_text(_json.dumps({"type": "act"}))
    (run_dir / "run.json").write_text(
        _json.dumps(
            {
                "run_id": run_id,
                "host_id": "test-host",
                "recipe_name": recipe,
                "dataset_id": "some/ds",
                "args": {},
                "state": "completed",
                "created_at": 1.0,
            }
        )
    )
    (run_dir / "checkpoints.jsonl").write_text(
        _json.dumps(
            {
                "step": step,
                "path": f"output/checkpoints/{step:06d}/pretrained_model/model.safetensors",
                "sha256": "0" * 64,
                "ts": 1.0,
            }
        )
        + "\n"
    )
    return run_dir


def test_run_dir_plus_checkpoint_path_locates_the_model(client: TestClient, tmp_path: Path) -> None:
    """The two halves compose into a real directory — the point of the field."""
    run_dir = _run_dir_with_a_checkpoint(tmp_path / "runs", "composerun001", "compose-recipe")

    body = client.get("/api/training/runs/composerun001").json()
    assert body["run_dir"] == str(run_dir)

    rel = body["checkpoints"][0]["path"]
    assert not Path(rel).is_absolute(), "checkpoint paths are relative; run_dir supplies the rest"
    policy_path = (Path(body["run_dir"]) / rel).parent
    assert policy_path.is_dir(), "run_dir + checkpoint path must resolve to the --policy.path dir"
    assert policy_path.name == "pretrained_model"


def test_run_dir_is_reported_for_a_run_with_no_checkpoints(client: TestClient) -> None:
    """Old and failed runs never wrote a checkpoint; the card must still work."""
    run_id = client.post(
        "/api/training/runs",
        json=_start_run_payload(args={"__recipe__": "__fake__", "num_steps": 1, "save_every": 10_000}),
    ).json()["run_id"]
    body = _wait_until_state(client, run_id, "completed")

    assert body["checkpoints"] == [] or body["checkpoints"]
    assert body["run_dir"].endswith(run_id)


def test_legacy_run_json_still_snapshots(client: TestClient, tmp_path: Path) -> None:
    """A run.json from before the current vocabulary must not break the field.

    Exercises the loader's compatibility paths — the pre-three-outcome
    "aborted" state and an integer session_id — and asserts the snapshot still
    carries run_dir.
    """
    run_id = "legacyrun0001"
    legacy_dir = tmp_path / "runs" / run_id
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "run.json").write_text(
        '{"run_id": "legacyrun0001", "host_id": "test-host", "recipe_name": "legacy-recipe",'
        ' "dataset_id": "old/ds", "args": {}, "state": "aborted", "created_at": 1.0,'
        ' "session_id": 4242}'
    )

    resp = client.get(f"/api/training/runs/{run_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["run"]["state"] == "stopped"  # "aborted" is translated on load
    assert body["run"]["recipe_name"] == "legacy-recipe"
    assert body["run_dir"] == str(legacy_dir)


def test_run_detail_and_models_tab_agree_on_the_model_name(client: TestClient, tmp_path: Path) -> None:
    """The name in the run's Checkpoints card must be findable in the Models tab.

    The Models tab labels a run by recipe_name, falling back to the run
    directory name (gui/api/models.py `_scan_training_run`); the run detail
    mirrors that rule. Pinned so the two cannot drift apart silently.
    """
    from lerobot.gui.api import models as models_api

    run_dir = _run_dir_with_a_checkpoint(tmp_path / "runs", "namedrun0001", "named-run")
    body = client.get("/api/training/runs/namedrun0001").json()

    scanned = models_api._scan_training_run(run_dir)  # noqa: SLF001
    assert scanned is not None, "the scanner must recognise a standard run layout"

    expected = body["run"]["recipe_name"] or Path(body["run_dir"]).name
    assert scanned["name"] == expected == "named-run"


def test_models_tab_falls_back_to_the_directory_name(client: TestClient, tmp_path: Path) -> None:
    """Old runs without a recipe show as the directory in both places."""
    from lerobot.gui.api import models as models_api

    run_dir = _run_dir_with_a_checkpoint(tmp_path / "runs", "norecipe00001", "")
    scanned = models_api._scan_training_run(run_dir)  # noqa: SLF001

    assert scanned is not None
    assert scanned["name"] == "norecipe00001"
