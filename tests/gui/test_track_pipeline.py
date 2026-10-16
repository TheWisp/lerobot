"""The worker must not wait for surface verification before starting its next frame."""

import asyncio
import io
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from starlette.requests import Request

from lerobot.gui.api.manipulation import runtime, tracking, worker


@pytest.mark.asyncio
async def test_slow_overlay_does_not_delay_pipelined_pose_application(monkeypatch):
    old = (runtime._state.track, runtime._state.teach, runtime._state.run)
    tr = runtime._Track(on=True, epoch=1)
    teach = SimpleNamespace(keypoints={"concept": "object"})
    runtime._state.track, runtime._state.teach, runtime._state.run = tr, teach, None
    drawing, release = threading.Event(), threading.Event()
    monkeypatch.setattr(tracking, "_verify_track_poses", lambda *args: [])

    def draw(*args):
        drawing.set()
        assert release.wait(2)
        return b"image"

    monkeypatch.setattr(tracking.rendering, "_render_live", draw)
    job = runtime._Job(
        id="frame",
        kind="track",
        concept="object",
        rgb=np.zeros((2, 2, 3), np.uint8),
        depth_m=np.ones((2, 2), np.float32),
        intr={},
        created=0.0,
        extra={"pipeline": True, "track_epoch": 1},
        result={"ok": False, "algo": "p2p", "state": "lost"},
    )
    try:
        await asyncio.wait_for(tracking._apply_track_result(job), 1)
        assert job.worker_timing_ms["verify_work"] >= 0
        assert job.worker_timing_ms["verify_thread_cpu"] >= 0
        assert job.worker_timing_ms["verify_resume_wait"] >= 0
        assert tr.last["state"] == "lost"
        assert tr.overlay is None and job.rendered_at is None
        await asyncio.wait_for(asyncio.to_thread(drawing.wait, 1), 2)
        release.set()
        await asyncio.wait_for(tr.overlay_task, 2)
        assert tr.overlay == b"image" and job.rendered_at is not None
    finally:
        release.set()
        if tr.overlay_task is not None:
            await tr.overlay_task
        runtime._state.track, runtime._state.teach, runtime._state.run = old


@pytest.mark.asyncio
async def test_track_replies_return_while_results_apply_in_order(monkeypatch):
    old_track = runtime._state.track
    tr = runtime._Track(on=True, epoch=1, done=asyncio.Event(), slots=asyncio.Semaphore(2))
    runtime._state.track = tr
    first_started, finish_first = asyncio.Event(), asyncio.Event()
    order = []

    async def apply(job):
        order.append(f"start {job.id}")
        if job.id == "first":
            first_started.set()
            await finish_first.wait()
        order.append(f"end {job.id}")

    monkeypatch.setattr(tracking, "_apply_track_result", apply)

    def make_job(name):
        job = runtime._Job(
            id=name,
            kind="track",
            concept="object",
            rgb=np.zeros((2, 2, 3), np.uint8),
            depth_m=np.ones((2, 2), np.float32),
            intr={},
            created=0.0,
            extra={"pipeline": True, "track_epoch": 1},
        )
        runtime._state.worker.jobs[name] = job
        return job

    async def reply(job):
        buf = io.BytesIO()
        np.savez(buf, meta=json.dumps({"ok": True}))

        async def receive():
            return {"type": "http.request", "body": buf.getvalue(), "more_body": False}

        request = Request({"type": "http", "method": "POST", "headers": []}, receive)
        return await worker.worker_result(job.id, request)

    try:
        await tr.slots.acquire()
        tr.job = "first"
        assert (await reply(make_job("first")))["status"] == "ok"
        await asyncio.wait_for(first_started.wait(), 1)
        assert tr.done.is_set() and order == ["start first"]

        tr.done.clear()
        await tr.slots.acquire()
        tr.job = "second"
        assert (await reply(make_job("second")))["status"] == "ok"
        assert tr.done.is_set() and order == ["start first"]
        finish_first.set()
        await asyncio.wait_for(tr.apply_task, 1)
        assert order == ["start first", "end first", "start second", "end second"]
        assert tr.slots._value == 2

        tr.job = "late"
        await tracking.track_stop()
        assert (await reply(make_job("late")))["status"] == "ok"
        assert order == ["start first", "end first", "start second", "end second"]
    finally:
        finish_first.set()
        if tr.apply_task is not None:
            await tr.apply_task
        runtime._state.track = old_track
        runtime._state.worker.jobs.pop("first", None)
        runtime._state.worker.jobs.pop("second", None)
        runtime._state.worker.jobs.pop("late", None)


@pytest.mark.asyncio
async def test_worker_records_time_waiting_for_dispatch():
    old = runtime._state.worker
    runtime._state.worker = runtime._Worker(proc=SimpleNamespace(poll=lambda: None))
    try:
        job = worker._queue_job("track", "object", np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)), {})
        job.created = time.time() - 0.5
        response = await worker.worker_job(wait=0)
        assert json.loads(response.body)["id"] == job.id
        assert job.worker_timing_ms["queue"] >= 500
    finally:
        runtime._state.worker = old


@pytest.mark.asyncio
async def test_recovery_lane_cannot_consume_or_block_stateful_jobs():
    old = runtime._state.worker
    proc = SimpleNamespace(poll=lambda: None)
    runtime._state.worker = runtime._Worker(proc=proc, recovery_proc=proc)
    try:

        def queue(kind, **extra):
            return worker._queue_job(
                kind, "object", np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)), {}, extra=extra
            )

        recovery = queue("locate")
        tracked = queue("track")
        designation = queue("locate", track=True)
        # An older recovery job never holds up the main worker.
        assert json.loads((await worker.worker_job(wait=0)).body)["id"] == tracked.id
        assert json.loads((await worker.worker_job(wait=0, lane="recovery")).body)["id"] == recovery.id
        assert (await worker.worker_job(wait=0, lane="recovery")).status_code == 204
        assert json.loads((await worker.worker_job(wait=0)).body)["id"] == designation.id
        # If the optional recovery process exits, its pending work remains usable.
        late = queue("locate")
        runtime._state.worker.recovery_proc = SimpleNamespace(poll=lambda: 1)
        assert (await worker.worker_job(wait=0, lane="recovery")).status_code == 410
        assert json.loads((await worker.worker_job(wait=0)).body)["id"] == late.id
    finally:
        runtime._state.worker = old


@pytest.mark.asyncio
async def test_read_only_locate_does_not_replace_live_pose(monkeypatch):
    from lerobot.gui.api.manipulation import acquisition

    state = runtime._state
    old = state.demo, state.worker
    state.demo = SimpleNamespace(objects={"object": {"status": "done"}})
    state.worker = runtime._Worker(proc=SimpleNamespace(poll=lambda: None))
    calls = []

    async def frame():
        return np.zeros((2, 2, 3)), np.ones((2, 2)), {}

    async def locate(*args, **kwargs):
        calls.append(kwargs["track"])
        return {"object": "object", "ok": False, "reason": "no match"}

    monkeypatch.setattr(acquisition.camera, "_frame", frame)
    monkeypatch.setattr(acquisition, "_locate", locate)
    monkeypatch.setattr(acquisition, "_store_located", lambda *args: pytest.fail("must not replace pose"))
    try:
        result = await acquisition.locate(acquisition.LocateBody(object="object", click=[0, 0], track=False))
        assert calls == [False]
        assert not result["ok"]
    finally:
        state.demo, state.worker = old
