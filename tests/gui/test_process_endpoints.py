# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the /api/process/* endpoints + ProcessJobConfig roundtrip.

The worker subprocess is stubbed (``_spawn_worker`` patched), so these cover the
endpoint plumbing + validation + job registry, not the SAM3 pass itself (that is
``test_dataset_postprocess``).
"""

from __future__ import annotations

import time
import types

import httpx
import pytest
from fastapi import FastAPI

from lerobot.gui import process_jobs
from lerobot.gui.api import process as process_module
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState


def test_config_roundtrip():
    cfg = process_jobs.ProcessJobConfig(
        job_id="j1",
        source_id="/d",
        source_repo_id="o/n",
        source_root="/d",
        out_repo_id="o/n_aug",
        out_root="/o",
        model="sam3_track",
        objects=[
            {"name": "ring", "sign": "+", "treatment": {"key": "tint", "params": {"color": [0, 255, 0]}}}
        ],
        background_treatment={"key": "random", "params": {}},
        apply_mode="per_episode",
        variants=2,
        multi_instance=True,
        cameras=["observation.images.top"],
        episodes=[0, 1],
        preview=False,
        jobs_dir="/j",
    )
    assert process_jobs.ProcessJobConfig.from_json(cfg.to_json()) == cfg


def test_merge_progress_never_un_terminalizes():
    job = process_jobs.make_job(source_id="/d", out_repo_id="o/n_aug", out_root="/o", effect="bg_solid")
    job.status = "complete"
    job.merge_progress({"status": "running", "frames_done": 3})
    assert job.status == "complete"  # terminal wins


def test_job_config_roundtrips_model_and_resolution():
    # The worker re-reads the config from env JSON — model + resolution must survive
    # (preview == commit includes the segmenter and its resolution).
    cfg = process_jobs.ProcessJobConfig(
        job_id="j1",
        source_id="/d",
        source_repo_id="me/demo",
        source_root="/src",
        out_repo_id="me/demo_aug",
        out_root="/out",
        model="sam3_track",
        resolution=672,
        objects=[{"name": "ring", "sign": "+", "treatment": {"key": "none"}}],
        background_treatment={"key": "random", "params": {}},
        apply_mode="per_episode",
        variants=1,
        multi_instance=True,
        cameras=None,
        episodes=None,
        preview=False,
        jobs_dir="/jobs",
    )
    back = process_jobs.ProcessJobConfig.from_json(cfg.to_json())
    assert back.model == "sam3_track" and back.resolution == 672
    # Back-compat: a config written before the knob existed loads with resolution=None.
    import json as _json

    d = _json.loads(cfg.to_json())
    del d["resolution"]
    old = process_jobs.ProcessJobConfig.from_json(_json.dumps(d))
    assert old.resolution is None


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(process_jobs, "JOBS_DIR", tmp_path / "process_jobs")
    monkeypatch.setattr(process_module, "JOBS_DIR", tmp_path / "process_jobs")
    state = AppState(frame_cache=FrameCache(max_bytes=1 << 20))
    state.datasets["/d"] = types.SimpleNamespace(repo_id="me/demo", root=tmp_path / "demo")  # type: ignore
    process_module.set_app_state(state)

    spawned = {}

    def fake_spawn(*, job, req, src, out_repo_id, out_root):
        job.pid = 4242
        spawned["job_id"] = job.job_id

    monkeypatch.setattr(process_module, "_spawn_worker", fake_spawn)
    # Reset the process-wide aux-GPU slot — a stubbed job holds it without a
    # heartbeat (never lapses), so leftover state would bleed across tests.
    from lerobot.gui.gpu_slot import SLOT

    SLOT._holder = None
    # Neutralise the overlay teardown (nothing running in a unit test).
    import lerobot.gui.api.overlays as ov

    monkeypatch.setattr(ov, "stop_data_publisher", lambda: None)

    async def _noop():
        pass

    monkeypatch.setattr(ov, "_stop_live", _noop)

    app = FastAPI()
    app.include_router(process_module.router)
    transport = httpx.ASGITransport(app=app)
    # `set_app_state` is a plain module write, so monkeypatch cannot undo it.
    # Left in place it points the process API at this stub for the rest of the
    # session: a later test that serves the real GUI in-process finds a dataset
    # registry holding only "/d", and its own dataset 404s.
    previous = process_module._app_state
    yield httpx.AsyncClient(transport=transport, base_url="http://t"), state
    process_module.set_app_state(previous)
    # Resetting the slot on the way IN is not enough. These tests end with a
    # stubbed job holding it, and a job's hold has no heartbeat, so it never
    # lapses: the next test to want the GPU is refused 409 by a job that only
    # ever existed in this file.
    SLOT._holder = None


@pytest.mark.asyncio
async def test_effects_listed(client):
    c, _ = client
    async with c:
        r = await c.get("/api/process/treatments")
    assert r.status_code == 200
    body = r.json()
    keys = {t["key"] for t in body["treatments"]}
    assert "tint" in keys and "blur" in keys and "none" in keys


@pytest.mark.asyncio
async def test_start_validation(client):
    c, _ = client
    async with c:
        # unknown dataset
        r = await c.post(
            "/api/process/start",
            json={
                "source_id": "/missing",
                "objects": [{"name": "r"}],
                "background_treatment": {"key": "random"},
            },
        )
        assert r.status_code == 404
        # unknown treatment
        r = await c.post(
            "/api/process/start",
            json={"source_id": "/d", "objects": [{"name": "r"}], "background_treatment": {"key": "nope"}},
        )
        assert r.status_code == 400
        # no named object
        r = await c.post(
            "/api/process/start",
            json={"source_id": "/d", "objects": [{"name": "  "}], "background_treatment": {"key": "random"}},
        )
        assert r.status_code == 400
        # all-none = nothing to do
        r = await c.post(
            "/api/process/start",
            json={"source_id": "/d", "objects": [{"name": "r"}], "background_treatment": {"key": "none"}},
        )
        assert r.status_code == 400
        # static + variants>1 would write byte-identical copies: one draw is shared
        # by the whole run, so the variants differ in nothing but disk usage.
        r = await c.post(
            "/api/process/start",
            json={
                "source_id": "/d",
                "objects": [{"name": "r"}],
                "background_treatment": {"key": "random"},
                "apply_mode": "static",
                "variants": 3,
            },
        )
        assert r.status_code == 400 and "identical copies" in r.json()["detail"]
        # ...and the same request with variants=1 is a legitimate single-look run.
        # an unrecognised cadence must not silently fall through to per_episode
        r = await c.post(
            "/api/process/start",
            json={
                "source_id": "/d",
                "objects": [{"name": "r"}],
                "background_treatment": {"key": "random"},
                "apply_mode": "per_decade",
            },
        )
        assert r.status_code == 400 and "apply_mode" in r.json()["detail"]
        # unknown segmentation model (saliency is an overlay, not a segmenter)
        r = await c.post(
            "/api/process/start",
            json={
                "source_id": "/d",
                "objects": [{"name": "r"}],
                "background_treatment": {"key": "random"},
                "model": "policy_saliency",
            },
        )
        assert r.status_code == 400
        # resolution outside the adapter presets
        r = await c.post(
            "/api/process/start",
            json={
                "source_id": "/d",
                "objects": [{"name": "r"}],
                "background_treatment": {"key": "random"},
                "resolution": 999,
            },
        )
        assert r.status_code == 400


@pytest.mark.asyncio
async def test_start_jobs_dismiss_flow(client):
    c, state = client
    async with c:
        r = await c.post(
            "/api/process/start",
            json={
                "source_id": "/d",
                "objects": [{"name": "ring", "sign": "+", "treatment": {"key": "none"}}],
                "background_treatment": {"key": "random"},
                "variants": 2,
                "out_name": "demo_rand",
            },
        )
        assert r.status_code == 200, r.text
        jid = r.json()["job_id"]
        assert r.json()["out_repo_id"] == "me/demo_rand"

        # duplicate in-flight -> 409
        r = await c.post(
            "/api/process/start",
            json={"source_id": "/d", "objects": [{"name": "r"}], "background_treatment": {"key": "random"}},
        )
        assert r.status_code == 409

        # listed + active
        r = await c.get("/api/process/jobs")
        assert r.json()["total"] == 1 and r.json()["active"] == 1

        # dismiss refused while running
        r = await c.post(f"/api/process/{jid}/dismiss")
        assert r.status_code == 409

        # complete it, then dismiss works
        state.process_jobs[jid].status = "complete"
        state.process_jobs[jid].finished_at = time.time()
        r = await c.post(f"/api/process/{jid}/dismiss")
        assert r.status_code == 200
        assert jid not in state.process_jobs


@pytest.mark.asyncio
async def test_preview_fixed_name_overwrites_and_is_findable(client, monkeypatch, tmp_path):
    c, state = client
    # Preview writes to the normal datasets dir (so it's a detectable Source),
    # under a fixed __preview name we overwrite each run. Patch HF_LEROBOT_HOME to
    # tmp so the test never touches the real home.
    monkeypatch.setattr(process_module, "HF_LEROBOT_HOME", tmp_path / "hf")
    async with c:
        payload = {
            "source_id": "/d",
            "objects": [{"name": "ring"}],
            "background_treatment": {"key": "random"},
            "preview": True,
            "episodes": [0],
        }
        r = await c.post("/api/process/start", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["preview"] is True
        assert body["out_repo_id"] == "me/demo__preview"  # fixed, suffix-named
        assert str(tmp_path / "hf") in body["out_root"]  # in the datasets dir -> findable
        jid = body["job_id"]
        assert state.process_jobs[jid].preview is True

        # A prior __preview on disk is overwritten (not 409'd) on the next preview.
        state.process_jobs[jid].status = "complete"
        state.process_jobs[jid].finished_at = time.time()
        (tmp_path / "hf" / "me/demo__preview").mkdir(parents=True, exist_ok=True)
        r = await c.post("/api/process/start", json=payload)
        assert r.status_code == 200, r.text  # overwrote, no collision error
