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
    pointed at a URL that does not exist. Asserted on the URL the worker
    builds: the earlier version of this test matched the source line that
    builds it, which pinned a local variable's name and would have passed a
    regression that rebuilt the URL correctly somewhere else.
    """
    from lerobot.gui.hub_worker import pr_url_for

    assert pr_url_for("u/thing", "model", 7) == "https://huggingface.co/u/thing/discussions/7"
    assert pr_url_for("u/thing", "dataset", 7) == "https://huggingface.co/datasets/u/thing/discussions/7"


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

    # `importlib.reload` re-executes the module body in the *existing* globals,
    # which every already-registered route resolves through — so it also resets
    # `_app_state` to None for the rest of the session. Put it back.
    saved_state = models_mod._app_state
    reloaded = importlib.reload(models_mod)
    try:
        assert tmp_path / "cfg" / "model_sources.json" == reloaded.SOURCES_FILE
        assert str(Path.home()) not in str(reloaded.SOURCES_FILE)
    finally:
        monkeypatch.delenv("LEROBOT_GUI_CONFIG_DIR", raising=False)
        importlib.reload(models_mod)
        models_mod._app_state = saved_state


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
    from lerobot.gui.hub_jobs import JobPaths, make_job
    from lerobot.gui.hub_worker import _WorkerState

    job = make_job(dataset_id="/runs/x", direction="upload", repo_id="u/thing", repo_type="model")
    snapshot = _WorkerState(job, JobPaths(jobs_dir=Path("/tmp"), job_id=job.job_id)).snapshot()
    assert snapshot["repo_type"] == "model"
    assert snapshot["repo_id"] == "u/thing"


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


def test_a_transfer_is_refused_when_the_hub_rejects_the_credentials(client):
    """Auth is checked before a worker is spawned, so a bad token costs a
    request rather than a subprocess that fails minutes later."""
    from lerobot.gui.api import datasets as dm

    c, run, _state, spawned = client

    from fastapi import HTTPException

    def no_auth():
        # What the real check raises: a 401 the frontend can surface at once,
        # rather than a job that fails minutes later in the tray.
        raise HTTPException(status_code=401, detail="Not logged in to HuggingFace Hub.")

    # `patch.object`, not monkeypatch: the fixture installs its own stub with
    # `patch.object`, and pytest tears monkeypatch down *after* it — so undoing
    # a monkeypatch here reinstalled the always-passes stub for every later test
    # in the session, silently disarming the auth gate they rely on.
    with patch.object(dm, "_verify_hub_auth", no_auth):
        r = c.post("/api/models/hub/upload", json={"path": str(run), "repo_id": "me/act"})
    assert r.status_code == 401, r.text
    assert "Not logged in" in r.json()["detail"]
    assert not spawned, "no worker may be spawned without credentials"


# ── Guardrails the dataset path runs, on the model path ─────────────────────
#
# These were dropped when the model endpoint was written, on the belief that
# they were dataset-shaped. Both helpers already take `repo_type` and read no
# dataset layout, so what they defend against applies unchanged to a checkpoint.


def _remote_with(*names: str):
    """A stub `HfApi` whose repo holds exactly ``names``."""

    class _Sib:
        def __init__(self, name):
            self.rfilename = name
            self.size = 1

    class _Info:
        siblings = [_Sib(n) for n in names]

    class _Api:
        def repo_info(self, repo_id, repo_type=None, files_metadata=False):
            assert repo_type == "model", f"looked the repo up as {repo_type!r}"
            return _Info()

    return _Api()


def test_a_partial_local_copy_is_refused_before_it_can_overwrite_the_remote(client):
    """Download dies halfway, user clicks Upload: the truncated tree must not ship.

    This is the download-fail-then-upload sequence the completeness check exists
    for. Without it `upload_large_folder` commits whatever is on disk over a
    complete remote and merges the PR — a silent, unrecoverable loss.
    """
    c, run, _state, spawned = client
    remote = _remote_with("pretrained_model/config.json", "pretrained_model/model.safetensors")

    with patch("huggingface_hub.HfApi", lambda *a, **k: remote):
        r = c.post("/api/models/hub/upload", json={"path": str(run), "repo_id": "me/act"})

    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "incomplete_local_state"
    assert "pretrained_model/model.safetensors" in detail["missing_locally"]
    assert not spawned, "a worker must not start for a known-partial upload"


def test_the_user_can_override_the_guardrail(client):
    """The dialog re-issues with confirm_force after the user says to go ahead."""
    c, run, _state, spawned = client
    remote = _remote_with("pretrained_model/config.json", "pretrained_model/model.safetensors")

    with patch("huggingface_hub.HfApi", lambda *a, **k: remote):
        r = c.post(
            "/api/models/hub/upload",
            json={"path": str(run), "repo_id": "me/act", "confirm_force": True},
        )

    assert r.status_code == 200, r.text
    assert len(spawned) == 1


def test_a_complete_local_copy_passes_the_guardrail(client):
    """The guardrail must not block the ordinary case."""
    c, run, _state, spawned = client
    remote = _remote_with("pretrained_model/config.json")

    with patch("huggingface_hub.HfApi", lambda *a, **k: remote):
        r = c.post("/api/models/hub/upload", json={"path": str(run), "repo_id": "me/act"})

    assert r.status_code == 200, r.text
    assert len(spawned) == 1


def test_a_download_is_not_subjected_to_the_upload_guardrail(client, tmp_path):
    """Nothing to compare: the local side is what the download is about to write."""
    c, _run, _state, spawned = client
    target = tmp_path / "fetch_here"

    r = c.post("/api/models/hub/download", json={"path": str(target), "repo_id": "me/act"})

    assert r.status_code == 200, r.text
    assert len(spawned) == 1


def test_a_retry_resumes_into_the_draft_pr_the_last_attempt_left(client):
    """Otherwise every retry re-sends the whole checkpoint and orphans a PR.

    A multi-GB upload that dies at 80% leaves a draft PR holding what was sent.
    Opening a second one starts from zero, and nothing in `hub_jobs` points at
    the first any more, so no dismiss can ever close it.
    """
    c, run, state, spawned = client
    from lerobot.gui.hub_jobs import make_job

    failed = make_job(dataset_id=str(run), direction="upload", repo_id="me/act", repo_type="model")
    failed.status = "failed"
    failed.pr_num = 11
    state.hub_jobs[failed.job_id] = failed

    with (
        patch.object(datasets_module, "_find_existing_pr_for_retry", lambda *a, **k: 11),
        patch("huggingface_hub.HfApi", lambda *a, **k: _remote_with("pretrained_model/config.json")),
    ):
        r = c.post("/api/models/hub/upload", json={"path": str(run), "repo_id": "me/act"})

    assert r.status_code == 200, r.text
    assert spawned[0]["reuse_pr_num"] == 11, spawned[0]
    assert state.hub_jobs[r.json()["job_id"]].pr_num == 11


def test_the_retry_lookup_reads_the_model_namespace(client):
    """A dataset-typed lookup would not find the model job, and silently open a
    second PR — the failure this guards against is invisible without the arg."""
    c, run, state, spawned = client
    from lerobot.gui.hub_jobs import make_job

    failed = make_job(dataset_id=str(run), direction="upload", repo_id="me/act", repo_type="model")
    failed.status = "failed"
    failed.pr_num = 11
    state.hub_jobs[failed.job_id] = failed

    seen: list[str] = []

    def _spy(dataset_id, repo_id, repo_type="dataset"):
        seen.append(repo_type)
        return None

    with (
        patch.object(datasets_module, "_find_existing_pr_for_retry", _spy),
        patch("huggingface_hub.HfApi", lambda *a, **k: _remote_with("pretrained_model/config.json")),
    ):
        c.post("/api/models/hub/upload", json={"path": str(run), "repo_id": "me/act"})

    assert seen == ["model"], seen


# ── Freshness ───────────────────────────────────────────────────────────────


def test_run_mtime_ignores_the_hubs_own_download_bookkeeping(client, tmp_path):
    """Otherwise a finished download reports "local is newer" than what it fetched.

    `snapshot_download` writes `.cache/huggingface/download/<name>.metadata`
    beside every file it pulls, stamped now. A walk that counts them makes the
    dialog invite the user to push back what they just pulled — and the ignore
    list both sides of a transfer already share exists to prevent exactly this.
    """
    import os

    c, run, _state, _spawned = client
    real = run / "pretrained_model" / "config.json"
    os.utime(real, (1_700_000_000, 1_700_000_000))
    os.utime(run / "pretrained_model", (1_700_000_000, 1_700_000_000))
    os.utime(run, (1_700_000_000, 1_700_000_000))

    cache = run / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    meta = cache / "config.json.metadata"
    meta.write_text("{}")
    os.utime(meta, (1_900_000_000, 1_900_000_000))

    r = c.get("/api/models/run-mtime", params={"path": str(run)})

    assert r.status_code == 200, r.text
    assert r.json()["mtime"] == 1_700_000_000, "the download cache leaked into the verdict"


def test_run_mtime_still_notices_a_new_checkpoint(client, tmp_path):
    """The ignore list must not swallow the thing being measured."""
    import os

    c, run, _state, _spawned = client
    for p in (run, run / "pretrained_model", run / "pretrained_model" / "config.json"):
        os.utime(p, (1_700_000_000, 1_700_000_000))
    fresh = run / "pretrained_model" / "model.safetensors"
    fresh.write_bytes(b"w")
    os.utime(fresh, (1_800_000_000, 1_800_000_000))

    r = c.get("/api/models/run-mtime", params={"path": str(run)})

    assert r.json()["mtime"] == 1_800_000_000


# ── A rejected request leaves nothing behind ────────────────────────────────


def test_a_download_rejected_for_credentials_creates_no_directory(client, tmp_path):
    """The target used to be created while resolving the path, before any gate.

    An empty directory left by a 401 is not inert: the model scanner lists it as
    a run, so a failed click adds a phantom entry to the tree.
    """
    from fastapi import HTTPException

    c, _run, _state, spawned = client
    target = tmp_path / "never_created"

    def no_auth():
        raise HTTPException(status_code=401, detail="Not logged in to HuggingFace Hub.")

    with patch.object(datasets_module, "_verify_hub_auth", no_auth):
        r = c.post("/api/models/hub/download", json={"path": str(target), "repo_id": "me/act"})

    assert r.status_code == 401, r.text
    assert not target.exists(), "a rejected download must not leave a directory behind"
    assert not spawned


def test_a_download_rejected_as_already_running_creates_no_directory(client, tmp_path):
    c, _run, state, _spawned = client
    from lerobot.gui.hub_jobs import make_job

    # The target does not exist yet, which is the ordinary case for a download —
    # and the case where creating it while resolving the path leaves a directory
    # behind that the refusal then never cleans up.
    target = tmp_path / "busy_target"
    active = make_job(dataset_id=str(target), direction="download", repo_id="me/act", repo_type="model")
    state.hub_jobs[active.job_id] = active

    r = c.post("/api/models/hub/download", json={"path": str(target), "repo_id": "me/act"})

    assert r.status_code == 409, r.text
    assert not target.exists(), "a refused duplicate transfer must not leave a directory behind"


def test_a_download_into_an_existing_directory_does_not_disturb_it(client, tmp_path):
    c, _run, _state, spawned = client
    target = tmp_path / "already_here"
    target.mkdir()
    (target / "keep.txt").write_text("x")

    r = c.post("/api/models/hub/download", json={"path": str(target), "repo_id": "me/act"})

    assert r.status_code == 200, r.text
    assert (target / "keep.txt").read_text() == "x"
    assert spawned[0]["local_path"] == target
