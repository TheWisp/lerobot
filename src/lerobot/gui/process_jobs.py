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
"""Dataset post-processing jobs — types + IPC boundary.

Mirrors :mod:`lerobot.gui.hub_jobs` for a different unit of work: editing a
dataset's camera frames offline (segment + effect) into a NEW dataset. Like the
Hub transfer, the heavy work runs in a subprocess (SAM3 is GPU-bound and must not
block the FastAPI loop); the server polls a per-job progress JSON file.

* :class:`ProcessJobConfig` — server → worker payload (``LEROBOT_PROCESS_WORKER_CONFIG``).
* :class:`ProcessJobState` — the server's in-memory mirror, polled by the GUI.
* :class:`ProcessJobPaths` — per-job IPC files under ``~/.cache/lerobot/gui/process_jobs``.

The generic process-identity / atomic-write helpers (``is_worker_alive``,
``atomic_write_json`` …) are reused from :mod:`lerobot.gui.hub_jobs` — only the
job *shape* differs here.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# Where per-job IPC files live (one progress/log/pid trio per job).
JOBS_DIR = Path.home() / ".cache" / "lerobot" / "gui" / "process_jobs"

# Worker rewrites its progress file at this interval; the server polls at ~1 Hz.
PROGRESS_WRITE_INTERVAL_S = 0.5

# Hold a terminal job (+ its files) this long so the GUI can show the final
# state and a "open the new dataset" affordance before it is GC'd.
STALE_TERMINAL_RETENTION_S = 1800.0  # 30 min

ProcessStatus = Literal["pending", "running", "complete", "failed", "cancelled"]


@dataclass(frozen=True)
class ProcessJobConfig:
    """Immutable config passed from server to the post-process worker.

    Pre: ``source_root`` is a readable LeRobotDataset; ``effect`` is a known
    effect key; ``objects`` is the overlay object list. The worker parses this
    once at startup from ``LEROBOT_PROCESS_WORKER_CONFIG`` and never re-reads.
    """

    job_id: str
    source_id: str  # the GUI's dataset id for the source (its open key)
    source_repo_id: str  # the source dataset's repo_id (owner/name), for re-opening it
    source_root: str  # resolved absolute path of the source dataset
    out_repo_id: str  # repo_id of the dataset to create
    out_root: str  # resolved absolute path for the new dataset
    model: str  # segmentation model, e.g. "sam3_track"
    objects: list[dict]  # [{name, color:[r,g,b], sign:'+'/'-'}]
    effect: str  # effect key (see dataset_postprocess.EFFECTS)
    effect_params: dict  # effect-specific params (color, sigma, amount, ...)
    apply_mode: str  # per_episode | per_frame | static
    variants: int
    cameras: list[str] | None  # subset to edit; None/[] = all
    episodes: list[int] | None  # subset to process; None = all (used by preview)
    preview: bool  # a quick single-episode run written to an ephemeral dir
    jobs_dir: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "job_id": self.job_id,
                "source_id": self.source_id,
                "source_repo_id": self.source_repo_id,
                "source_root": self.source_root,
                "out_repo_id": self.out_repo_id,
                "out_root": self.out_root,
                "model": self.model,
                "objects": self.objects,
                "effect": self.effect,
                "effect_params": self.effect_params,
                "apply_mode": self.apply_mode,
                "variants": self.variants,
                "cameras": self.cameras,
                "episodes": self.episodes,
                "preview": self.preview,
                "jobs_dir": self.jobs_dir,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> ProcessJobConfig:
        d = json.loads(raw)
        return cls(
            job_id=d["job_id"],
            source_id=d["source_id"],
            source_repo_id=d["source_repo_id"],
            source_root=d["source_root"],
            out_repo_id=d["out_repo_id"],
            out_root=d["out_root"],
            model=d["model"],
            objects=d["objects"],
            effect=d["effect"],
            effect_params=d.get("effect_params", {}),
            apply_mode=d.get("apply_mode", "per_episode"),
            variants=int(d.get("variants", 1)),
            cameras=d.get("cameras"),
            episodes=d.get("episodes"),
            preview=bool(d.get("preview", False)),
            jobs_dir=d["jobs_dir"],
        )


@dataclass(frozen=True)
class ProcessJobPaths:
    """Per-job IPC file locations; both server and worker compute the same set."""

    jobs_dir: Path
    job_id: str

    @property
    def progress(self) -> Path:
        return self.jobs_dir / f"{self.job_id}.json"

    @property
    def log(self) -> Path:
        return self.jobs_dir / f"{self.job_id}.log"

    @property
    def pid(self) -> Path:
        return self.jobs_dir / f"{self.job_id}.pid"

    @classmethod
    def for_job(cls, job_id: str, jobs_dir: str | os.PathLike[str]) -> ProcessJobPaths:
        return cls(jobs_dir=Path(jobs_dir), job_id=job_id)


@dataclass
class ProcessJobState:
    """Server-side mirror of one post-process job. Polled by the GUI tray.

    Progress is frame-counted: ``frames_done / frames_total`` drives the bar.
    The server's ``status`` is authoritative until the worker writes a terminal
    value, after which the worker (which knows the truth) wins — same rule as
    :class:`lerobot.gui.hub_jobs.HubJobState`.
    """

    job_id: str
    source_id: str
    out_repo_id: str
    out_root: str
    effect: str
    status: ProcessStatus
    started_at: float
    preview: bool = False
    finished_at: float | None = None
    stage: str = "starting"
    frames_total: int = 0
    frames_done: int = 0
    episodes_total: int = 0
    episodes_done: int = 0
    current_episode: int | None = None
    error: str | None = None
    # Server-side worker tracking; None until the worker has spawned.
    pid: int | None = None
    process_start_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "source_id": self.source_id,
            "out_repo_id": self.out_repo_id,
            "out_root": self.out_root,
            "effect": self.effect,
            "preview": self.preview,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stage": self.stage,
            "frames_total": self.frames_total,
            "frames_done": self.frames_done,
            "episodes_total": self.episodes_total,
            "episodes_done": self.episodes_done,
            "current_episode": self.current_episode,
            "error": self.error,
        }

    def merge_progress(self, snapshot: dict[str, Any]) -> None:
        """Pull worker-owned fields from a progress JSON snapshot.

        Never lets a snapshot drag a terminal job back to running."""
        if self.status in ("complete", "failed", "cancelled"):
            return
        for key in (
            "status",
            "stage",
            "finished_at",
            "frames_total",
            "frames_done",
            "episodes_total",
            "episodes_done",
            "current_episode",
            "error",
        ):
            if key in snapshot and snapshot[key] is not None:
                setattr(self, key, snapshot[key])


def make_job(
    *,
    source_id: str,
    out_repo_id: str,
    out_root: str,
    effect: str,
    preview: bool = False,
) -> ProcessJobState:
    """Build a fresh server-side ``ProcessJobState`` in ``pending``."""
    return ProcessJobState(
        job_id=uuid.uuid4().hex,
        source_id=source_id,
        out_repo_id=out_repo_id,
        out_root=out_root,
        effect=effect,
        preview=preview,
        status="pending",
        started_at=time.time(),
    )
