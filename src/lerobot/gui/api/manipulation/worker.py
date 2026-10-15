#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Worker process and asynchronous acquisition/tracking job transport."""

from __future__ import annotations

import asyncio
import io
import json
import subprocess
import sys
import threading
import time
import uuid
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from . import acquisition, demos, runtime, tracking

router = APIRouter()


# ── the worker: SAM3 designation + DINO features, in its own process ────────


def _queue_job(
    kind: str,
    concept: str,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intr: dict[str, float],
    algo: str | None = None,
    compress: bool = True,
    click: list[int] | None = None,
    camera_stamp_mono: float | None = None,
) -> runtime._Job:
    job = runtime._Job(
        id=uuid.uuid4().hex[:8],
        kind=kind,
        concept=concept,
        rgb=rgb,
        depth_m=depth_m,
        intr=intr,
        created=time.time(),
        camera_stamp_mono=camera_stamp_mono,
        algo=algo,
        compress=compress,
        click=click,
    )
    w = runtime._state.worker
    with runtime._state.lock:
        w.jobs[job.id] = job
        w.pending.append(job.id)
        # Forget results nobody will read; keep the last few for the overlays.
        for old in [i for i, j in w.jobs.items() if j.result is not None][:-4]:
            del w.jobs[old]
        wake = w.wake
    if wake is not None:
        wake.set()
    return job


class WorkerStartBody(BaseModel):
    device: str = "cuda"
    dino_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m"


@router.post("/worker/start")
async def worker_start(body: WorkerStartBody, request: Request) -> dict:
    """Spawn the SAM3 + DINO worker against this server. Models load on the first job."""
    w = runtime._state.worker
    with runtime._state.lock:
        if w.running:
            raise HTTPException(409, "the worker is already running")
        cmd = [
            sys.executable,
            str(runtime._WORKER),
            "--server",
            str(request.base_url),
            "--device",
            body.device,
            "--dino-model",
            body.dino_model,
        ]
        import os

        env = {**os.environ, "PYTHONPATH": str(runtime._REPO / "src")}
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=str(runtime._REPO)
        )
        w.proc, w.log, w.started_at = proc, [], time.time()
        runtime._state.server_url = str(request.base_url).rstrip("/")

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            with runtime._state.lock:
                w.log.append(line.rstrip("\n"))
                del w.log[:-200]

    threading.Thread(target=pump, daemon=True, name="pregrasp-worker").start()
    return {"status": "started"}


@router.post("/worker/stop")
async def worker_stop() -> dict:
    w = runtime._state.worker
    with runtime._state.lock:
        proc = w.proc
        w.proc = None
        w.pending.clear()
        wake = w.wake
    if wake is not None:
        wake.set()  # a waiting poll returns and sees no worker
    if proc is not None and proc.poll() is None:
        proc.terminate()
    return {"status": "stopped"}


@router.get("/worker/job")
async def worker_job(wait: float = 20.0) -> Response:
    """The worker's long-poll: the next job, 204 when none within ``wait`` seconds, 410 when stopped."""
    w = runtime._state.worker
    if w.wake is None:
        w.wake = asyncio.Event()
    deadline = time.monotonic() + min(max(wait, 0.0), 60.0)
    while True:
        with runtime._state.lock:
            if not w.running:
                return Response(status_code=410)
            if w.pending:
                job_id = w.pending.pop(0)
                job = w.jobs[job_id]
                job.taken = True
                return Response(
                    content=json.dumps(
                        {
                            "id": job.id,
                            "kind": job.kind,
                            "concept": job.concept,
                            "algo": job.algo,
                            "click": job.click,
                            **job.extra,
                        }
                    ),
                    media_type="application/json",
                )
            w.wake.clear()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return Response(status_code=204)
        try:
            await asyncio.wait_for(w.wake.wait(), timeout=remaining)
        except TimeoutError:
            return Response(status_code=204)


@router.get("/worker/frame.npz")
async def worker_frame(id: str) -> Response:
    with runtime._state.lock:
        job = runtime._state.worker.jobs.get(id)
    if job is None:
        raise HTTPException(404, "no such job")
    buf = io.BytesIO()
    save = np.savez_compressed if job.compress else np.savez
    save(buf, rgb=job.rgb, depth=job.depth_m, intr=np.array(json.dumps(job.intr)), **job.arrays)
    return Response(content=buf.getvalue(), media_type="application/octet-stream")


@router.post("/worker/result")
async def worker_result(id: str, request: Request) -> dict:
    """The worker's answer for a job: NPZ with ``meta`` JSON plus mask, points and, for a find, the delta."""
    body = await request.body()
    data = np.load(io.BytesIO(body), allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    with runtime._state.lock:
        job = runtime._state.worker.jobs.get(id)
    if job is None:
        raise HTTPException(404, "no such job")
    result: dict[str, Any] = dict(meta)
    for key in (
        "mask",
        "uv",
        "xyz",
        "live_uv",
        "model_xyz",
        "model_valid",
        "delta",
        "deltas",
        "seen",
        "masks",
        "ref_delta",
        "ref_mask",
        "fit_uv",
        "fit_inlier",
        "track_idx",
        "track_uv",
        "track_vis",
        *(k for k in data.files if k.startswith("other_")),
    ):
        if key in data.files:
            result[key] = np.asarray(data[key])
    if job.kind in ("teach", "locate", "find"):
        result = await asyncio.get_running_loop().run_in_executor(
            runtime._RENDER_EXECUTOR, tracking._constrain_find_result, job, result
        )
    job.result = result
    if job.kind == "teach":
        acquisition._apply_teach_result(job)
    elif job.kind == "track":
        await tracking._apply_track_result(job)
    elif job.kind == "stream_object":
        await demos._apply_object_result(job)
    elif job.kind == "locate":
        pass  # :func:`_locate` is waiting on the job itself
    else:
        acquisition._apply_find_result(job)
    return {"status": "ok"}


@router.post("/worker/progress")
async def worker_progress(id: str, done: int, total: int) -> dict:
    """A long job's progress, as the worker reports it."""
    with runtime._state.lock:
        job = runtime._state.worker.jobs.get(id)
        if job is None:
            raise HTTPException(404, "no such job")
        job.progress = float(np.clip(done / max(total, 1), 0.0, 1.0))
    return {"progress": job.progress}
