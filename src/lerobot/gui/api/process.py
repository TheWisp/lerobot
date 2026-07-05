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
"""Dataset post-processing API — segment objects + apply an effect → new dataset.

The "Edit data" feature in the data tab. Reuses the overlay's already-configured
objects (the protected foreground) and runs an offline pass in a subprocess (the
:mod:`lerobot.gui.process_worker`), modelled on the Hub-transfer job tray: the
server registers a :class:`ProcessJobState`, spawns the worker, and the GUI polls
``/api/process/jobs`` for frame-count progress. The produced dataset is local
under ``$HF_LEROBOT_HOME``; the frontend opens it on completion.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from lerobot.datasets.dataset_postprocess import EFFECTS
from lerobot.gui.process_jobs import (
    JOBS_DIR,
    ProcessJobConfig,
    ProcessJobPaths,
    make_job,
)
from lerobot.utils.constants import HF_LEROBOT_HOME

# Previews are single-episode runs written to the normal dataset location (so
# they're detectable in the default Source + open like any dataset) under a
# ``__preview`` suffix that we overwrite each run — ephemeral, but findable.
PREVIEW_SUFFIX = "__preview"

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/process", tags=["process"])

_app_state: AppState = None  # type: ignore  # set by server.py

_VALID_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_EFFECT_KEYS = {e.key for e in EFFECTS}


def set_app_state(state: AppState) -> None:
    global _app_state
    _app_state = state


@router.get("/effects")
async def list_effects() -> dict:
    """The effect menu the frontend renders, grouped (background vs global) so the
    UI can label "foreground protected" effects distinctly from whole-frame ones.

    Randomized effects always sample once per episode (consistent within a
    trajectory); the mode isn't user-exposed."""
    return {
        "effects": [
            {
                "key": e.key,
                "label": e.label,
                "group": e.group,
                "controls": e.controls,
                "randomized": e.randomized,
            }
            for e in EFFECTS
        ]
    }


class StartRequest(BaseModel):
    source_id: str
    objects: list[dict]  # [{name, color:[r,g,b], sign}]
    effect: str
    effect_params: dict | None = None
    apply_mode: str = "per_episode"
    variants: int = 1
    multi_instance: bool = True  # segment all instances of each object (both arms) vs largest
    cameras: list[str] | None = None
    model: str = "sam3_track"
    out_name: str | None = None  # dataset name part; combined with the source owner
    preview: bool = False  # quick single-episode run to an ephemeral dir, auto-opened
    episodes: list[int] | None = None  # subset to process (preview passes [current])


def _refresh(job) -> None:
    """Merge the worker's progress JSON into the in-memory job state."""
    paths = ProcessJobPaths.for_job(job.job_id, JOBS_DIR)
    try:
        snap = json.loads(paths.progress.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return
    job.merge_progress(snap)


@router.post("/start")
async def start(req: StartRequest, x_overlay_session: str | None = Header(default=None)) -> dict:
    """Validate, acquire the aux-GPU slot, and spawn the post-process worker.

    A batch job is an aux-GPU activity: it acquires the same slot as the live overlay
    (see gpu_slot). If your OWN preview overlay holds it, we hand off (tear the overlay
    down, take the slot); if another client's overlay/job holds it, refuse (409
    overlay_busy). The slot is held for the job's whole lifetime and released when it
    reaches a terminal state. Also 409 if a job is already running for this source, or
    400 on a bad effect / missing objects / colliding output path."""
    if _app_state is None or req.source_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {req.source_id}")
    if req.effect not in _EFFECT_KEYS:
        raise HTTPException(status_code=400, detail=f"Unknown effect: {req.effect}")
    named = [o for o in (req.objects or []) if str(o.get("name", "")).strip()]
    if not named:
        raise HTTPException(status_code=400, detail="Name at least one object to keep as foreground")
    if _app_state.active_process_job_for(req.source_id) is not None:
        raise HTTPException(status_code=409, detail="A processing job is already running for this dataset")

    src = _app_state.datasets[req.source_id]
    owner = src.repo_id.split("/")[0] if "/" in src.repo_id else "local"
    src_name = src.repo_id.split("/")[-1]

    if req.preview:
        # Single-episode run in the normal datasets dir (detectable + findable),
        # under a fixed __preview name we overwrite each time. Auto-opened by the
        # frontend on completion.
        out_repo_id = f"{owner}/{src_name}{PREVIEW_SUFFIX}"
        out_root = HF_LEROBOT_HOME / out_repo_id
        if out_root.exists():
            assert out_root.name.endswith(PREVIEW_SUFFIX), f"refusing to rm non-preview {out_root}"
            shutil.rmtree(out_root)  # safe-destruct: our own prior preview (suffix-guarded)
    else:
        name = (req.out_name or f"{src_name}_{req.effect}").strip()
        if not _VALID_NAME.match(name):
            raise HTTPException(status_code=400, detail="Output name may only contain letters, digits, . _ -")
        out_repo_id = f"{owner}/{name}"
        out_root = HF_LEROBOT_HOME / out_repo_id
        if out_root.exists():
            raise HTTPException(status_code=409, detail=f"Output dataset already exists: {out_repo_id}")

    # Acquire the aux-GPU slot. If your OWN preview overlay holds it, hand off (tear it
    # down + take the slot); if another activity holds it, refuse.
    from lerobot.gui.api.overlays import _data_key, _stop_live, stop_data_publisher
    from lerobot.gui.gpu_slot import SLOT

    now = time.time()
    own_overlay = _data_key(x_overlay_session)
    holder = SLOT.holder(now)
    job = make_job(
        source_id=req.source_id,
        out_repo_id=out_repo_id,
        out_root=str(out_root),
        effect=req.effect,
        preview=req.preview,
    )
    proc_key = f"process:{job.job_id}"
    if holder is not None and holder.key not in (proc_key, own_overlay):
        raise HTTPException(status_code=409, detail={"code": "overlay_busy", "holder": holder.label})
    # Free (or held by our own preview overlay) → hand off: drop the overlay's claim, tear
    # its worker down (the batch worker loads its own SAM3), and take the slot for the job.
    SLOT.release(own_overlay)
    stop_data_publisher()
    await _stop_live()
    label = f"processing {out_repo_id.split('/')[-1]} ({'preview' if req.preview else 'full'})"
    SLOT.acquire(proc_key, label, now, heartbeat=False)  # background: held until the job ends

    _app_state.process_jobs[job.job_id] = job
    _spawn_worker(job=job, req=req, src=src, out_repo_id=out_repo_id, out_root=out_root)
    return {
        "job_id": job.job_id,
        "status": "started",
        "out_repo_id": out_repo_id,
        "out_root": str(out_root),
        "preview": req.preview,
    }


def _spawn_worker(*, job, req: StartRequest, src, out_repo_id: str, out_root: Path) -> None:
    """Launch the detached post-process worker subprocess for ``job``."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    paths = ProcessJobPaths.for_job(job.job_id, JOBS_DIR)
    cfg = ProcessJobConfig(
        job_id=job.job_id,
        source_id=req.source_id,
        source_repo_id=src.repo_id,
        source_root=str(src.root),
        out_repo_id=out_repo_id,
        out_root=str(out_root),
        model=req.model,
        objects=req.objects,
        effect=req.effect,
        effect_params=req.effect_params or {},
        apply_mode=req.apply_mode,
        variants=max(1, int(req.variants)),
        multi_instance=req.multi_instance,
        cameras=req.cameras,
        episodes=req.episodes,
        preview=req.preview,
        jobs_dir=str(JOBS_DIR),
    )
    # Stub the progress file so a poll right after spawn reads something.
    from lerobot.gui.hub_jobs import atomic_write_json

    atomic_write_json(paths.progress, {"job_id": job.job_id, "status": "pending", "stage": "starting"})

    env = os.environ.copy()
    env["LEROBOT_PROCESS_WORKER_CONFIG"] = cfg.to_json()
    proc = subprocess.Popen(  # noqa: S603 — args are well-controlled
        [sys.executable, "-m", "lerobot.gui.process_worker"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    job.pid = proc.pid
    logger.info(
        "spawned post-process worker pid=%d job=%s effect=%s -> %s",
        proc.pid,
        job.job_id,
        req.effect,
        out_repo_id,
    )


def _settle(job) -> None:
    """Refresh a running job; if its worker died without finalizing, mark it failed
    (so the aux-GPU slot frees); release the slot once the job is terminal."""
    from lerobot.gui.gpu_slot import SLOT
    from lerobot.gui.hub_jobs import is_worker_alive, read_pid_file

    if job.status in ("pending", "running"):
        _refresh(job)
        if job.status in ("pending", "running"):
            payload = read_pid_file(ProcessJobPaths.for_job(job.job_id, JOBS_DIR).pid)
            if payload is not None and not is_worker_alive(payload):
                job.status = "failed"
                job.error = "Worker exited without finalizing"
                job.finished_at = time.time()
    if job.status in ("complete", "failed", "cancelled"):
        SLOT.release(f"process:{job.job_id}")  # give the aux-GPU slot back


@router.get("/jobs")
async def jobs() -> dict:
    """All post-process jobs, newest-first, refreshed from the workers' progress
    files (the GUI tray polls this). Frees the aux-GPU slot for terminal jobs and
    GCs jobs older than 30 min."""
    _app_state.gc_finished_process_jobs()
    for j in list(_app_state.process_jobs.values()):
        _settle(j)
    out = sorted(
        (j.to_dict() for j in _app_state.process_jobs.values()), key=lambda d: d["started_at"], reverse=True
    )
    active = sum(1 for d in out if d["status"] in ("pending", "running"))
    return {"jobs": out, "total": len(out), "active": active}


@router.post("/{job_id}/cancel")
async def cancel(job_id: str) -> dict:
    """Request a graceful cancel (SIGTERM) of a running job after a (pid,
    start_time) identity check, so a recycled PID is never signalled."""
    job = _app_state.process_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    from lerobot.gui.gpu_slot import SLOT
    from lerobot.gui.hub_jobs import is_worker_alive, read_pid_file

    paths = ProcessJobPaths.for_job(job_id, JOBS_DIR)
    payload = read_pid_file(paths.pid)
    if payload is None or not is_worker_alive(payload):
        if job.status not in ("complete", "failed", "cancelled"):
            job.status = "failed"
            job.error = "Worker exited without finalizing"
            job.finished_at = time.time()
        SLOT.release(f"process:{job_id}")  # give the aux-GPU slot back
        paths.pid.unlink(missing_ok=True)  # safe-destruct: stale PID file we own
        return {"status": "already_gone", "job_id": job_id}
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(payload["pid"], signal.SIGTERM)
    return {"status": "cancel_requested", "job_id": job_id}


@router.post("/{job_id}/dismiss")
async def dismiss(job_id: str) -> dict:
    """Drop a terminal job from the registry and clean up its IPC files."""
    job = _app_state.process_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.status in ("pending", "running"):
        raise HTTPException(status_code=409, detail="Cancel the job before dismissing it")
    from lerobot.gui.gpu_slot import SLOT

    SLOT.release(f"process:{job_id}")  # give the aux-GPU slot back (belt-and-suspenders)
    paths = ProcessJobPaths.for_job(job_id, JOBS_DIR)
    for p in (paths.progress, paths.log, paths.pid):
        p.unlink(missing_ok=True)  # safe-destruct: this job's own IPC files
    del _app_state.process_jobs[job_id]
    return {"status": "dismissed", "job_id": job_id}
