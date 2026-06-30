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
        objects=[{"name": "ring", "color": [0, 255, 0], "sign": "+"}],
        effect="bg_random_color",
        effect_params={"sigma": 9},
        apply_mode="per_episode",
        variants=2,
        cameras=["observation.images.top"],
        jobs_dir="/j",
    )
    assert process_jobs.ProcessJobConfig.from_json(cfg.to_json()) == cfg


def test_merge_progress_never_un_terminalizes():
    job = process_jobs.make_job(source_id="/d", out_repo_id="o/n_aug", out_root="/o", effect="bg_solid")
    job.status = "complete"
    job.merge_progress({"status": "running", "frames_done": 3})
    assert job.status == "complete"  # terminal wins


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
    # Neutralise the overlay teardown (nothing running in a unit test).
    import lerobot.gui.api.overlays as ov

    monkeypatch.setattr(ov, "stop_data_publisher", lambda: None)

    async def _noop():
        pass

    monkeypatch.setattr(ov, "_stop_live", _noop)

    app = FastAPI()
    app.include_router(process_module.router)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t"), state


@pytest.mark.asyncio
async def test_effects_listed(client):
    c, _ = client
    async with c:
        r = await c.get("/api/process/effects")
    assert r.status_code == 200
    body = r.json()
    keys = {e["key"] for e in body["effects"]}
    assert "bg_random_color" in keys and "brightness" in keys
    assert {m["key"] for m in body["apply_modes"]} == {"per_episode", "per_frame", "static"}


@pytest.mark.asyncio
async def test_start_validation(client):
    c, _ = client
    async with c:
        # unknown dataset
        r = await c.post(
            "/api/process/start",
            json={"source_id": "/missing", "objects": [{"name": "r"}], "effect": "bg_solid"},
        )
        assert r.status_code == 404
        # unknown effect
        r = await c.post(
            "/api/process/start", json={"source_id": "/d", "objects": [{"name": "r"}], "effect": "nope"}
        )
        assert r.status_code == 400
        # no named object
        r = await c.post(
            "/api/process/start", json={"source_id": "/d", "objects": [{"name": "  "}], "effect": "bg_solid"}
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
                "objects": [{"name": "ring", "color": [0, 255, 0], "sign": "+"}],
                "effect": "bg_random_color",
                "variants": 2,
                "out_name": "demo_rand",
            },
        )
        assert r.status_code == 200, r.text
        jid = r.json()["job_id"]
        assert r.json()["out_repo_id"] == "me/demo_rand"

        # duplicate in-flight -> 409
        r = await c.post(
            "/api/process/start", json={"source_id": "/d", "objects": [{"name": "r"}], "effect": "bg_solid"}
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
