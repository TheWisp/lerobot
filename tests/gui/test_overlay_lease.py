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
"""Overlay resource mutex — the single shared SAM worker.

One worker + one obs stream serve the whole server, so it's a plain mutex: whoever
holds it holds it, and EVERY other user — another data tab/machine, or the run-tab
overlay — is treated the same (busy, no priority, no preemption). The holder's
status poll is the heartbeat, so the mutex frees when it goes quiet. The worker
spawn / obs stream are stubbed — this covers the mutex, not SAM.
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
    monkeypatch.setattr(ov, "_dataset_camera_dims", lambda ds: {"observation.images.cam": (4, 4)})
    monkeypatch.setattr(ov, "start_data_publisher", lambda *a, **k: True)
    monkeypatch.setattr(ov, "_write_data_control", lambda: None)
    monkeypatch.setattr(ov, "_data_publisher_active", lambda: True)

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(ov, "_spawn_worker", _noop)
    monkeypatch.setattr(ov, "_stop_live", _noop)
    # Reset the module-level mutex between tests.
    ov._overlay_holder = None
    ov._overlay_holder_seen = 0.0

    app = FastAPI()
    app.include_router(ov.router)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def _data(session):
    return {
        "headers": {"X-Overlay-Session": session},
        "json": {"dataset_id": "/d", "model": "sam3_track", "objects": [{"name": "x"}]},
    }


def _live():
    return {"json": {"model": "sam3_track", "objects": [{"name": "x"}]}}


@pytest.mark.asyncio
async def test_second_data_client_is_busy(client):
    async with client as c:
        assert (await c.post("/api/overlays/data/configure", **_data("A"))).status_code == 200
        r = await c.post("/api/overlays/data/configure", **_data("B"))
        assert r.status_code == 409 and r.json()["detail"]["code"] == "overlay_busy"
        # The holder re-configures freely.
        assert (await c.post("/api/overlays/data/configure", **_data("A"))).status_code == 200


@pytest.mark.asyncio
async def test_status_reports_holder_and_busy(client):
    async with client as c:
        await c.post("/api/overlays/data/configure", **_data("A"))
        ra = (await c.get("/api/overlays/data/status", headers={"X-Overlay-Session": "A"})).json()
        rb = (await c.get("/api/overlays/data/status", headers={"X-Overlay-Session": "B"})).json()
        assert ra["owner"] is True and ra["busy"] is False
        assert rb["owner"] is False and rb["busy"] is True and rb["holder"] == "A"


@pytest.mark.asyncio
async def test_run_overlay_and_data_share_one_mutex_symmetrically(client):
    async with client as c:
        # Data holds it -> the run overlay is refused, treated identically.
        await c.post("/api/overlays/data/configure", **_data("A"))
        r = await c.post("/api/overlays/live/start", **_live())
        assert r.status_code == 409 and r.json()["detail"]["holder"] == "A"

        # A releases; now the run overlay can hold it -> a data client is refused with holder "run".
        await c.post("/api/overlays/data/cancel", headers={"X-Overlay-Session": "A"})
        assert (await c.post("/api/overlays/live/start", **_live())).status_code == 200
        r = await c.post("/api/overlays/data/configure", **_data("B"))
        assert r.status_code == 409 and r.json()["detail"]["holder"] == "run"
        # And the data status shows the run as the busy holder.
        rb = (await c.get("/api/overlays/data/status", headers={"X-Overlay-Session": "B"})).json()
        assert rb["busy"] is True and rb["holder"] == "run"


@pytest.mark.asyncio
async def test_cancel_only_by_holder_and_expiry_frees(client):
    async with client as c:
        await c.post("/api/overlays/data/configure", **_data("A"))
        # Another client's cancel is a no-op — can't kill the holder's overlay.
        r = await c.post("/api/overlays/data/cancel", headers={"X-Overlay-Session": "B"})
        assert "note" in r.json()
        assert (await c.post("/api/overlays/data/configure", **_data("B"))).status_code == 409

        # Holder goes silent past the timeout -> the mutex frees, anyone can take over.
        ov._overlay_holder_seen -= ov.LEASE_TIMEOUT_S + 1
        assert (await c.post("/api/overlays/data/configure", **_data("B"))).status_code == 200
        assert ov._overlay_holder == "B"
