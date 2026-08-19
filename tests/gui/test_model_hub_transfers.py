# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Hub transfers for model repos (#87).

No test anywhere constructed a job with ``repo_type="model"`` — the only
coverage was ``JobConfig.__post_init__`` rejecting a third value, which proves
the literal is validated, not that a model transfer works. Everything below the
endpoint already reads ``repo_type``; what was missing was an endpoint that
sets it, and a tray that does not assume ``/datasets/``.

The worker is never spawned here: these assert the job the endpoint builds and
the guards around it, which is where the dataset assumptions were.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lerobot.gui.api import datasets as datasets_module, models as models_module
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState


@pytest.fixture
def client(tmp_path):
    """A model run on disk, with the worker spawn stubbed out."""
    run = tmp_path / "outputs" / "act_pick_place"
    (run / "pretrained_model").mkdir(parents=True)
    (run / "pretrained_model" / "config.json").write_text('{"type": "act"}')

    app = FastAPI()
    app.include_router(models_module.router)
    state = AppState(frame_cache=FrameCache(max_bytes=1000))
    orig_m, orig_d = models_module._app_state, datasets_module._app_state
    models_module.set_app_state(state)
    datasets_module.set_app_state(state)
    spawned: list = []
    try:
        with (
            patch.object(datasets_module, "_verify_hub_auth", lambda: None),
            patch.object(datasets_module, "_spawn_hub_worker", lambda **kw: spawned.append(kw)),
            TestClient(app) as c,
        ):
            yield c, run, state, spawned
    finally:
        models_module._app_state = orig_m
        datasets_module._app_state = orig_d


@pytest.mark.parametrize("direction", ["upload", "download"])
def test_transfer_builds_a_model_typed_job(client, direction):
    """The job must carry repo_type='model' — everything downstream reads it."""
    c, run, state, spawned = client
    r = c.post(f"/api/models/hub/{direction}", json={"path": str(run), "repo_id": "me/act-pick"})
    assert r.status_code == 200, r.text

    job = state.hub_jobs[r.json()["job_id"]]
    assert job.repo_type == "model", "a dataset-typed job would push to the wrong namespace"
    assert job.direction == direction
    assert job.repo_id == "me/act-pick"
    assert spawned and spawned[0]["local_path"] == run


def test_a_second_transfer_for_the_same_run_is_refused(client):
    """Two workers writing one directory is the race the spawn lock exists for."""
    c, run, _state, _spawned = client
    first = c.post("/api/models/hub/upload", json={"path": str(run), "repo_id": "me/act-pick"})
    assert first.status_code == 200
    second = c.post("/api/models/hub/upload", json={"path": str(run), "repo_id": "me/act-pick"})
    assert second.status_code == 409, second.text
    assert second.json()["detail"]["job_id"] == first.json()["job_id"]


def test_a_path_that_is_not_a_run_is_refused(client, tmp_path):
    c, _run, _state, spawned = client
    missing = c.post("/api/models/hub/upload", json={"path": str(tmp_path / "nope"), "repo_id": "me/x"})
    assert missing.status_code == 404, missing.text

    empty = tmp_path / "empty_run"
    empty.mkdir()
    r = c.post("/api/models/hub/upload", json={"path": str(empty), "repo_id": "me/x"})
    assert r.status_code == 400, r.text
    assert not spawned, "nothing may be spawned for a path that is not a run"


def test_model_pr_url_points_at_the_model_namespace():
    """Models live at the Hub root; datasets under /datasets.

    The worker hardcoded the dataset prefix, so a model upload's PR link
    pointed at a URL that does not exist.
    """
    import re

    src = Path("src/lerobot/gui/hub_worker.py").read_text()
    assert 'f"https://huggingface.co/datasets/{cfg.repo_id}/discussions/' not in src
    assert re.search(r'_ns = "datasets/" if cfg\.repo_type == "dataset" else ""', src)


def test_model_sources_honour_the_config_dir_override(tmp_path, monkeypatch):
    """The model tree's config file must be redirectable like every other one.

    It hardcoded `~/.config/lerobot/model_sources.json`, so a test or a script
    that set `LEROBOT_GUI_CONFIG_DIR` still wrote to the developer's real
    config — the one GUI config file the isolation did not cover. Found by the
    user-state guard when an end-to-end script registered a temp source and the
    guard reported a real file had changed.
    """
    import importlib

    monkeypatch.setenv("LEROBOT_GUI_CONFIG_DIR", str(tmp_path / "cfg"))
    from lerobot.gui.api import models as models_mod

    reloaded = importlib.reload(models_mod)
    try:
        assert tmp_path / "cfg" / "model_sources.json" == reloaded.SOURCES_FILE
        assert str(Path.home()) not in str(reloaded.SOURCES_FILE)
    finally:
        monkeypatch.delenv("LEROBOT_GUI_CONFIG_DIR", raising=False)
        importlib.reload(models_mod)


def test_the_worker_records_repo_type_in_its_durable_progress_file():
    """The tray's history reads this file, not the in-memory job.

    A live card gets `repo_type` from `HubJobState`, so the namespace is right
    while the transfer runs. The registry drops a job 30 minutes after it
    finishes and loses everything on restart, after which the history renders
    from the worker's JSON — which omitted `repo_type`, so
    `hubRepoUrl(id, undefined)` fell back to `/datasets/` and every past model
    transfer linked to a URL that does not exist.

    Found by uploading a real model: the persisted record read
    `repo_type: None` while the transfer itself had been correct throughout.
    """
    from lerobot.gui.hub_worker import _WorkerState

    src = Path("src/lerobot/gui/hub_worker.py").read_text()
    assert '"repo_type": self.config.repo_type,' in src, (
        "the progress dict must carry repo_type into the durable record"
    )
    # And it sits in the same dict as repo_id, so the two cannot drift apart.
    snapshot = src[src.index('"repo_id": self.config.repo_id,') :][:400]
    assert '"repo_type"' in snapshot
    assert _WorkerState is not None


# ── Failure and edge cases, by direction ────────────────────────────────────
#
# Upload and download have opposite preconditions, so a single gate cannot
# serve both: upload needs content to send, download needs somewhere to put it.
# The table below is the coverage claim — one row per way each can be asked for
# something it cannot do.


def test_download_into_a_new_folder_is_the_normal_case(client, tmp_path):
    """A fresh directory is where a model is usually fetched to.

    The upload gate rejects an empty directory, correctly — there is nothing to
    send. Applying it to download made the ordinary case impossible.
    """
    c, _run, _state, spawned = client
    target = tmp_path / "outputs" / "fetched_here"
    target.parent.mkdir(parents=True, exist_ok=True)

    r = c.post("/api/models/hub/download", json={"path": str(target), "repo_id": "me/act"})
    assert r.status_code == 200, r.text
    assert target.is_dir(), "the target is created for the worker to write into"
    assert spawned and spawned[0]["local_path"] == target


def test_download_into_an_existing_empty_folder_works(client, tmp_path):
    c, _run, _state, _spawned = client
    target = tmp_path / "already_there"
    target.mkdir()
    assert (
        c.post("/api/models/hub/download", json={"path": str(target), "repo_id": "me/act"}).status_code == 200
    )


def test_download_will_not_conjure_a_whole_tree_from_a_typo(client, tmp_path):
    """Creating a leaf beside directories the user chose is reasonable;
    creating three levels from a mistyped path is how files end up nowhere."""
    c, _run, _state, spawned = client
    typo = tmp_path / "no" / "such" / "place"
    r = c.post("/api/models/hub/download", json={"path": str(typo), "repo_id": "me/act"})
    assert r.status_code == 404, r.text
    assert not typo.exists()
    assert not spawned


def test_download_refuses_a_path_that_is_a_file(client, tmp_path):
    c, _run, _state, _spawned = client
    f = tmp_path / "a_file.txt"
    f.write_text("not a directory")
    r = c.post("/api/models/hub/download", json={"path": str(f), "repo_id": "me/act"})
    assert r.status_code == 400, r.text
    assert f.read_text() == "not a directory", "the file must be untouched"


def test_upload_still_refuses_an_empty_run(client, tmp_path):
    """Splitting the gates must not loosen the upload side."""
    c, _run, _state, spawned = client
    empty = tmp_path / "nothing_here"
    empty.mkdir()
    assert c.post("/api/models/hub/upload", json={"path": str(empty), "repo_id": "me/act"}).status_code == 400
    assert not spawned


def test_a_transfer_is_refused_when_the_hub_rejects_the_credentials(client, monkeypatch):
    """Auth is checked before a worker is spawned, so a bad token costs a
    request rather than a subprocess that fails minutes later."""
    from lerobot.gui.api import datasets as dm

    c, run, _state, spawned = client

    from fastapi import HTTPException

    def no_auth():
        # What the real check raises: a 401 the frontend can surface at once,
        # rather than a job that fails minutes later in the tray.
        raise HTTPException(status_code=401, detail="Not logged in to HuggingFace Hub.")

    monkeypatch.setattr(dm, "_verify_hub_auth", no_auth)
    r = c.post("/api/models/hub/upload", json={"path": str(run), "repo_id": "me/act"})
    assert r.status_code == 401, r.text
    assert "Not logged in" in r.json()["detail"]
    assert not spawned, "no worker may be spawned without credentials"
