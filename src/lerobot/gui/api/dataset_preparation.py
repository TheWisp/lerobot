# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Backend for the Preprocess tab: HVLA 224x224 dataset preparation jobs.

Deliberately minimal: one job at a time, in-memory state only (lost on
server restart), no cancel/retry/queue. The actual work is
``lerobot.datasets.hvla_preparation.prepare_hvla_dataset`` — this module is
only a thin async wrapper so FastAPI is not blocked by the long conversion.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lerobot.utils.constants import HF_LEROBOT_HOME

router = APIRouter(prefix="/api/dataset-preparation", tags=["dataset-preparation"])


class PrepareHvlaRequest(BaseModel):
    source_repo_id: str = Field(min_length=1)
    source_root: str | None = None
    output_repo_id: str | None = None
    output_root: str | None = None


@dataclass
class PreparationJob:
    job_id: str
    status: Literal["pending", "running", "complete", "failed"]
    done: int
    total: int
    current_file: str
    source_repo_id: str
    output_repo_id: str
    output_root: str
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


_jobs: dict[str, PreparationJob] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1)


def _active_job() -> PreparationJob | None:
    for job in _jobs.values():
        if job.status in ("pending", "running"):
            return job
    return None


def _run_job(job: PreparationJob, source_root: str | None) -> None:
    # Deferred import: heavy (torch/av) and only needed once a job starts.
    from lerobot.datasets.hvla_preparation import prepare_hvla_dataset

    job.status = "running"

    def on_progress(done: int, total: int, current: str) -> None:
        job.done = done
        job.total = total
        job.current_file = current

    try:
        prepare_hvla_dataset(
            source_repo_id=job.source_repo_id,
            source_root=source_root,
            output_repo_id=job.output_repo_id,
            output_root=job.output_root,
            progress=on_progress,
        )
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
    else:
        job.done = job.total
        job.status = "complete"


@router.post("/hvla", status_code=201)
def start_hvla_preparation(body: PrepareHvlaRequest) -> dict:
    output_repo_id = body.output_repo_id or f"{body.source_repo_id}_hvla224"
    output_root = body.output_root or str(HF_LEROBOT_HOME / output_repo_id)

    if Path(output_root).exists():
        raise HTTPException(status_code=409, detail=f"Output already exists: {output_root}")
    with _jobs_lock:
        active = _active_job()
        if active is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Preparation already {active.status}: {active.output_repo_id} (job {active.job_id})",
            )
        job = PreparationJob(
            job_id=uuid.uuid4().hex[:8],
            status="pending",
            done=0,
            total=0,
            current_file="",
            source_repo_id=body.source_repo_id,
            output_repo_id=output_repo_id,
            output_root=output_root,
        )
        _jobs[job.job_id] = job
    _executor.submit(_run_job, job, body.source_root)
    return {"job_id": job.job_id, "status": job.status}


@router.get("/jobs/{job_id}")
def get_preparation_job(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return job.to_dict()
