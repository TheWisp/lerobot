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

import json
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
    yield httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
    # These tests deliberately end with the slot held by a job. It is process
    # global, so leaving it held makes the NEXT test's fill refuse with 409 --
    # a failure with no connection to whatever that test was checking.
    SLOT._holder = None


def _start(session, preview=True):
    return {
        "headers": {"X-Overlay-Session": session},
        "json": {
            "source_id": "/d",
            "model": "sam3_track",
            "objects": [{"name": "arm"}],
            "background_treatment": {"key": "random"},
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


# ── the same handoff, for the endpoint the UI actually reaches ───────────────
#
# Everything above drives /api/process/start -- the bake-a-copy path that
# nothing in the GUI reaches any more. The dataset-wide fill goes to
# /api/process/episode-masks, which writes out the same slot dance a second
# time. One rule, two implementations, and only one of them was tested: the
# tested one is the dead path.


@pytest.fixture
def masks_client(monkeypatch, tmp_path):
    """The fill endpoint, with the worker spawn and job directory stubbed."""
    import subprocess as _sp

    state = AppState(frame_cache=FrameCache(max_bytes=1 << 20))
    meta = types.SimpleNamespace(
        total_episodes=2,
        camera_keys=["observation.images.top"],
        features={"masks.top": {"mask_labels": ["arm"], "mask_encoding": "coco_rle"}},
    )
    state.datasets["/d"] = types.SimpleNamespace(repo_id="me/demo", root=str(tmp_path), meta=meta)  # type: ignore
    pr.set_app_state(state)
    ov.set_app_state(state)
    monkeypatch.setattr(ov, "stop_data_publisher", lambda *a, **k: None, raising=False)

    async def _noop(*a, **k):
        return None

    # Nothing is stubbed here on purpose. The disarm lives in `_teardown_current`,
    # and stubbing either that or `_stop_live` would hide the behaviour these
    # tests exist to check. With no worker running the real teardown clears its
    # globals and returns, which is exactly what a test wants it to do.
    # Never launch a real worker, and never write into the user's job directory.
    monkeypatch.setattr(pr, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(_sp, "Popen", lambda *a, **k: types.SimpleNamespace(pid=4242))
    monkeypatch.setattr(pr, "_rebind_when_done", _noop)
    SLOT._holder = None

    app = FastAPI()
    app.include_router(pr.router)
    # Both routers: the fill's hand-off reaches into the overlay's teardown, and
    # the disarm it causes is only observable through the overlay's own status.
    app.include_router(ov.router)
    ov._data_apply_on = False
    yield httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
    SLOT._holder = None
    ov._data_apply_on = False


def _fill(session, episodes=(0, 1)):
    return {
        "headers": {"X-Overlay-Session": session},
        "json": {
            "source_id": "/d",
            "episode": episodes[0],
            "episodes": list(episodes),
            "objects": [{"name": "arm", "sign": "+", "treatment": {"key": "none"}}],
            "cameras": ["observation.images.top"],
            "confirm_adopt": True,
            "confirm_overwrite": True,
        },
    }


@pytest.mark.asyncio
async def test_fill_hands_off_from_this_tab_s_own_overlay(masks_client):
    """The operator's likely state: the SAM3 panel is open and previewing, and
    they press Fill gaps. The preview must be torn down and the slot taken, not
    left racing the batch pass for the same GPU."""
    async with masks_client as c:
        assert SLOT.acquire(ov._data_key("A"), "SAM3 overlay", now=time.time()) is True
        r = await c.post("/api/process/episode-masks", **_fill("A"))
        assert r.status_code == 200, r.text
        h = SLOT.holder(now=time.time())
        assert h is not None and h.key.startswith("process:"), f"the fill did not take the slot: {h}"
        assert h.heartbeat is False, "a batch hold must not lapse while the pass runs"


@pytest.mark.asyncio
async def test_fill_is_refused_while_another_client_holds_the_slot(masks_client):
    """Two people, one GPU. The refusal names the holder so the second one knows
    what to wait for."""
    async with masks_client as c:
        SLOT.acquire(ov._data_key("other"), "SAM3 overlay", now=time.time())
        r = await c.post("/api/process/episode-masks", **_fill("A"))
        assert r.status_code == 409
        assert r.json()["detail"]["code"] == "overlay_busy"
        assert SLOT.holder(now=time.time()).key == ov._data_key("other"), "the other client lost its slot"


@pytest.mark.asyncio
async def test_the_slot_is_free_again_once_the_fill_settles(masks_client):
    """Why the overlay could be reopened seconds after a fill "was still running":
    the pass had finished. A terminal job must not keep the GPU."""
    async with masks_client as c:
        r = await c.post("/api/process/episode-masks", **_fill("A"))
        jid = r.json()["job_id"]
        job = pr._app_state.process_jobs[jid]
        job.status = "complete"
        job.finished_at = time.time()
        await c.get("/api/process/jobs")  # the tray poll is what settles it
        assert SLOT.free(now=time.time()), "a finished fill still holds the GPU slot"


# ── a run that fails must give the GPU back ─────────────────────────────────
#
# The worst failure this feature has: a job that dies holding the aux-GPU slot
# locks the operator out of BOTH the SAM3 panel and every further fill, with no
# in-app way to recover -- and it presents exactly like "SAM3 is broken". The
# release happens in `_settle`, which the tray poll drives, so these go through
# GET /api/process/jobs rather than calling it directly.


def _progress(job_id, tmp_path, **fields):
    from lerobot.gui.process_jobs import ProcessJobPaths

    paths = ProcessJobPaths.for_job(job_id, tmp_path / "jobs")
    paths.progress.parent.mkdir(parents=True, exist_ok=True)
    paths.progress.write_text(json.dumps({"job_id": job_id, **fields}))
    return paths


@pytest.mark.asyncio
async def test_a_worker_that_reports_failure_gives_the_slot_back(masks_client, tmp_path):
    async with masks_client as c:
        r = await c.post("/api/process/episode-masks", **_fill("A"))
        jid = r.json()["job_id"]
        assert SLOT.holder(now=time.time()).key == f"process:{jid}"

        _progress(jid, tmp_path, status="failed", stage="error", error="SAM3 failed to load")
        await c.get("/api/process/jobs")

        assert SLOT.free(now=time.time()), (
            f"a failed fill kept the GPU: holder={SLOT.holder(now=time.time())}"
        )


@pytest.mark.asyncio
async def test_a_worker_that_dies_silently_gives_the_slot_back(masks_client, tmp_path, monkeypatch):
    """The harder case: killed outright, so it never writes a terminal status.
    Left running forever, its slot would never be released -- a job's hold has no
    heartbeat precisely so a long pass cannot lapse."""
    async with masks_client as c:
        r = await c.post("/api/process/episode-masks", **_fill("A"))
        jid = r.json()["job_id"]

        paths = _progress(jid, tmp_path, status="running", stage="segmenting")
        paths.pid.write_text(json.dumps({"pid": 999999, "started_at": time.time()}))
        monkeypatch.setattr(pr, "is_worker_alive", lambda payload: False, raising=False)
        from lerobot.gui import hub_jobs

        monkeypatch.setattr(hub_jobs, "is_worker_alive", lambda payload: False)

        await c.get("/api/process/jobs")

        job = pr._app_state.process_jobs[jid]
        assert job.status == "failed", f"a dead worker left the job {job.status}"
        assert SLOT.free(now=time.time()), "a dead worker kept the GPU"


@pytest.mark.asyncio
async def test_the_slot_survives_a_job_whose_pid_file_never_appeared(masks_client, tmp_path):
    """The gap in the liveness check: `read_pid_file` returning None means "no
    evidence", and no evidence is not treated as death, so the job stays
    `running` and keeps the slot indefinitely -- measured: held is True here.

    The exposure is narrow. The worker writes its pid file immediately after
    parsing its config, before any model load, so only a death inside those few
    milliseconds leaves none behind: an import error, an unusable config, an
    interpreter that will not start. Narrow is not never, and the failure has no
    automatic recovery.

    What is asserted is therefore the property that must hold regardless of that
    judgement -- the operator is not permanently locked out, because cancelling
    is terminal and terminal frees the slot. Asserting the hold itself would
    entrench it; a fix that reaps these should not have to edit this test.
    """
    async with masks_client as c:
        r = await c.post("/api/process/episode-masks", **_fill("A"))
        jid = r.json()["job_id"]
        _progress(jid, tmp_path, status="running", stage="segmenting")  # no .pid written

        await c.get("/api/process/jobs")
        stuck = not SLOT.free(now=time.time())

        # Recovery: cancelling is terminal, and terminal frees the slot.
        await c.post(f"/api/process/{jid}/cancel")
        await c.get("/api/process/jobs")
        assert SLOT.free(now=time.time()), (
            "cancelling a job with no pid file did not free the GPU — there is then no "
            f"in-app way back (was stuck before cancel: {stuck})"
        )


@pytest.mark.asyncio
async def test_a_fill_disarms_apply_because_it_takes_the_worker(masks_client):
    """Apply-and-play has no GPU claim of its own: it rides the live overlay's,
    which is an interactive claim indistinguishable from an idle preview. A fill
    therefore takes the GPU from a run mid-write, exactly as if nothing were
    happening — and the run's masks come from the worker the fill tears down.

    So the mode cannot outlive the worker. Disarming happens in `_stop_live`,
    where the worker actually dies, so every path that kills it disarms too
    rather than each caller remembering; the flag is then reported by
    `/data/status` so the panel mirrors it instead of keeping a second copy.
    """
    async with masks_client as c:
        await c.post("/api/overlays/apply/arm", json={"armed": True})
        assert ov._data_apply_on is True, "arming did not take"

        r = await c.post("/api/process/episode-masks", **_fill("A"))
        assert r.status_code == 200, r.text
        assert ov._data_apply_on is False, (
            "a fill took the GPU and tore down the worker while Apply stayed armed; "
            "Play would then publish frames nobody is segmenting"
        )


@pytest.mark.asyncio
async def test_the_armed_flag_is_reported_so_the_client_need_not_guess(masks_client):
    """The panel used to keep its own copy and could not learn it had been
    disarmed. One fact, reported by the side that owns it."""
    async with masks_client as c:
        await c.post("/api/overlays/apply/arm", json={"armed": True})
        before = (await c.get("/api/overlays/data/status")).json()
        assert before.get("apply_armed") is True, before

        await c.post("/api/process/episode-masks", **_fill("A"))
        after = (await c.get("/api/overlays/data/status")).json()
        assert after.get("apply_armed") is False, (
            f"status still reports Apply armed after a fill took the GPU: {after.get('apply_armed')}"
        )


@pytest.mark.asyncio
async def test_only_the_gpu_slot_holder_can_arm_apply(masks_client):
    """Apply is a writing mode over ONE shared worker and ONE shared drain queue.

    Two tabs arming it is not two runs: the drain hands each batch back once, so
    they take half the frames each, and either tab's disarm stopped the other's
    run because the flag was global and carried no session.

    The owner is whoever holds the GPU slot rather than a second notion of
    ownership: a parallel one needs its own lifetime, and a tab that closes
    mid-run would then own Apply for the life of the process. The slot already
    leases with a heartbeat and reclaims from a tab that stopped polling.
    """
    async with masks_client as c:
        SLOT.acquire(ov._data_key("A"), "SAM3 overlay", now=time.time())

        mine = await c.post(
            "/api/overlays/apply/arm", json={"armed": True}, headers={"X-Overlay-Session": "A"}
        )
        assert mine.status_code == 200 and mine.json()["armed"] is True

        theirs = await c.post(
            "/api/overlays/apply/arm", json={"armed": True}, headers={"X-Overlay-Session": "B"}
        )
        assert theirs.status_code == 409, f"a tab without the GPU armed Apply: {theirs.status_code}"
        assert theirs.json()["detail"]["code"] == "overlay_busy"
        assert ov._data_apply_on is True, "the refused tab changed the owner's mode"

        # And the lifetime comes free: once the holder lets go, anyone may arm.
        SLOT.release(ov._data_key("A"))
        now_theirs = await c.post(
            "/api/overlays/apply/arm", json={"armed": True}, headers={"X-Overlay-Session": "B"}
        )
        assert now_theirs.status_code == 200, "the mode stayed owned after the slot was released"
