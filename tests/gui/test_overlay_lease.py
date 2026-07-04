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
"""Single-owner lease for the shared data overlay.

One worker + one obs stream serve the whole server, so two browser tabs can't
independently drive the data overlay. The first tab (session token) owns it; a
second is told it's busy; the owner's status poll is a heartbeat, so the lease
frees when the owner goes quiet. The worker spawn / obs stream are stubbed — this
covers the lease logic, not SAM3.
"""

from __future__ import annotations

import types

import httpx
import pytest
from fastapi import FastAPI

from lerobot.gui.api import overlays as ov
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState


@pytest.fixture
def client(monkeypatch):
    state = AppState(frame_cache=FrameCache(max_bytes=1 << 20))
    state.datasets["/d"] = types.SimpleNamespace(repo_id="me/demo", root="/d")  # type: ignore
    ov.set_app_state(state)
    # Stub the heavy bits: no obs stream, no subprocess, no shm control writes.
    monkeypatch.setattr(ov, "_dataset_camera_dims", lambda ds: {"observation.images.cam": (4, 4)})
    monkeypatch.setattr(ov, "start_data_publisher", lambda *a, **k: True)
    monkeypatch.setattr(ov, "_write_data_control", lambda: None)
    monkeypatch.setattr(ov, "_data_publisher_active", lambda: True)

    async def _noop_spawn(*a, **k):
        return None

    monkeypatch.setattr(ov, "_spawn_worker", _noop_spawn)
    # Reset the module-level lease between tests.
    ov._data_owner = None
    ov._data_owner_seen = 0.0
    monkeypatch.setattr(ov, "_data_pub_dataset", None, raising=False)

    app = FastAPI()
    app.include_router(ov.router)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def _cfg(session):
    return {
        "headers": {"X-Overlay-Session": session},
        "json": {"dataset_id": "/d", "model": "sam3_track", "objects": [{"name": "x"}]},
    }


@pytest.mark.asyncio
async def test_second_session_is_refused_busy(client):
    async with client as c:
        r = await c.post("/api/overlays/data/configure", **_cfg("A"))
        assert r.status_code == 200, r.text  # A owns it

        r = await c.post("/api/overlays/data/configure", **_cfg("B"))
        assert r.status_code == 409
        assert r.json()["detail"]["code"] == "overlay_busy"

        # A re-configures freely (still the owner).
        r = await c.post("/api/overlays/data/configure", **_cfg("A"))
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_status_reports_owner_and_busy(client):
    async with client as c:
        await c.post("/api/overlays/data/configure", **_cfg("A"))
        ra = await c.get("/api/overlays/data/status", headers={"X-Overlay-Session": "A"})
        rb = await c.get("/api/overlays/data/status", headers={"X-Overlay-Session": "B"})
        assert ra.json()["owner"] is True and ra.json()["busy"] is False
        assert rb.json()["owner"] is False and rb.json()["busy"] is True


@pytest.mark.asyncio
async def test_cancel_releases_only_for_owner(client):
    async with client as c:
        await c.post("/api/overlays/data/configure", **_cfg("A"))
        # Non-owner cancel is a no-op (can't kill A's overlay).
        r = await c.post("/api/overlays/data/cancel", headers={"X-Overlay-Session": "B"})
        assert r.status_code == 200 and "note" in r.json()
        r = await c.post("/api/overlays/data/configure", **_cfg("B"))
        assert r.status_code == 409  # A still owns it

        # Owner cancel releases; now B can take it.
        await c.post("/api/overlays/data/cancel", headers={"X-Overlay-Session": "A"})
        r = await c.post("/api/overlays/data/configure", **_cfg("B"))
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_expired_lease_lets_another_take_over(client, monkeypatch):
    async with client as c:
        await c.post("/api/overlays/data/configure", **_cfg("A"))
        # Simulate A going silent past the timeout — B should be able to acquire.
        ov._data_owner_seen -= ov.LEASE_TIMEOUT_S + 1
        r = await c.post("/api/overlays/data/configure", **_cfg("B"))
        assert r.status_code == 200
        assert ov._data_owner == "B"
