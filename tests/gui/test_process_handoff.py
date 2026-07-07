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
"""A batch process job is an aux-GPU activity over the same slot as the overlay.

Starting a job hands the slot off from THIS tab's own preview overlay (tear it
down, take the slot as a background activity), but refuses (409 overlay_busy) if
another client's overlay or job holds it. The worker spawn is stubbed — this
covers the slot handoff, not SAM.
"""

from __future__ import annotations

import time
import types

import httpx
import pytest
from fastapi import FastAPI

from lerobot.gui.api import overlays as ov, process as pr
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.gpu_slot import SLOT
from lerobot.gui.state import AppState


@pytest.fixture
def client(monkeypatch):
    state = AppState(frame_cache=FrameCache(max_bytes=1 << 20))
    state.datasets["/d"] = types.SimpleNamespace(repo_id="me/demo", root="/d")  # type: ignore
    pr.set_app_state(state)
    ov.set_app_state(state)
    # Stub the overlay teardown the handoff calls, and the worker spawn.
    monkeypatch.setattr(ov, "stop_data_publisher", lambda *a, **k: None, raising=False)

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(ov, "_stop_live", _noop)
    monkeypatch.setattr(pr, "_spawn_worker", lambda **k: None)
    SLOT._holder = None

    app = FastAPI()
    app.include_router(pr.router)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def _start(session, preview=True):
    return {
        "headers": {"X-Overlay-Session": session},
        "json": {
            "source_id": "/d",
            "model": "sam3_track",
            "objects": [{"name": "arm"}],
            "effect": "bg_random_color",
            "preview": preview,
        },
    }


@pytest.mark.asyncio
async def test_start_hands_off_from_own_preview_overlay(client):
    async with client as c:
        now = time.time()
        # This tab's own data overlay holds the slot (live now)...
        assert SLOT.acquire(ov._data_key("A"), "SAM3 overlay", now=now) is True
        r = await c.post("/api/process/start", **_start("A"))
        assert r.status_code == 200
        # ...handed off to the job (background activity, our overlay's claim dropped).
        h = SLOT.holder(now=time.time())
        assert h is not None and h.key.startswith("process:") and h.heartbeat is False


@pytest.mark.asyncio
async def test_start_refused_when_another_client_holds_the_slot(client):
    async with client as c:
        SLOT.acquire(ov._data_key("other"), "SAM3 overlay", now=time.time())  # a different client's overlay
        r = await c.post("/api/process/start", **_start("A"))
        assert r.status_code == 409
        assert r.json()["detail"]["code"] == "overlay_busy"
        assert r.json()["detail"]["holder"] == "SAM3 overlay"
        # The other client's slot is untouched.
        assert SLOT.holder(now=time.time()).key == ov._data_key("other")


@pytest.mark.asyncio
async def test_start_from_free_slot_takes_it_as_background(client):
    async with client as c:
        r = await c.post("/api/process/start", **_start("A"))
        assert r.status_code == 200
        h = SLOT.holder(now=time.time())
        assert h.heartbeat is False  # a job never lapses; it holds until it finishes


@pytest.mark.asyncio
async def test_finished_job_slot_self_heals_for_the_next_start(client):
    # A finished job holds the slot until /jobs settles it; a fresh start must not be
    # blocked by that stale hold (the next preview right after one completes).
    async with client as c:
        r1 = await c.post("/api/process/start", **_start("A"))
        jid = r1.json()["job_id"]
        job = pr._app_state.process_jobs[jid]
        job.status = "complete"  # finished, but no /jobs poll yet → slot still held
        job.finished_at = time.time()
        r2 = await c.post("/api/process/start", **_start("A"))
        assert r2.status_code == 200  # self-healed: the terminal job's slot was freed
        assert SLOT.holder(now=time.time()).key != f"process:{jid}"
