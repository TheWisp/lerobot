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
"""Dataset post-processing worker — one job per process.

Loads its :class:`~lerobot.gui.process_jobs.ProcessJobConfig` from
``LEROBOT_PROCESS_WORKER_CONFIG``, runs
:func:`lerobot.datasets.dataset_postprocess.process_dataset`, and writes a
progress JSON file the GUI server polls. Subprocess (not an asyncio task) so the
GPU-bound SAM3 pass never blocks the server loop, and a crash here can't take
the GUI down. SIGTERM/SIGINT request a graceful cancel (the partial dataset is
finalized and the job reports ``cancelled``).

Run as: ``python -m lerobot.gui.process_worker`` with the config env var set.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass

from lerobot.gui.hub_jobs import atomic_write_json, pid_file_payload
from lerobot.gui.process_jobs import (
    PROGRESS_WRITE_INTERVAL_S,
    ProcessJobConfig,
    ProcessJobPaths,
    ProcessStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class _WorkerState:
    """In-process progress, flushed to the job's progress JSON ~2 Hz.

    Individual field writes are atomic under the GIL; the writer thread takes a
    coherent snapshot via :meth:`snapshot`. Mirrors the fields the server's
    :class:`ProcessJobState` reads back."""

    status: ProcessStatus = "running"
    stage: str = "starting"
    frames_total: int = 0
    frames_done: int = 0
    #: Per mask-feature count of frames that actually carry a mask. A camera
    #: at zero means the pass found nothing at all — worth surfacing, since it
    #: is otherwise indistinguishable from success.
    coverage: dict | None = None
    episodes_total: int = 0
    episodes_done: int = 0
    current_episode: int | None = None
    finished_at: float | None = None
    error: str | None = None
    cancel_requested: bool = False

    def snapshot(self) -> dict:
        d = asdict(self)
        d.pop("cancel_requested", None)
        return d


def _run(cfg: ProcessJobConfig, state: _WorkerState) -> None:
    """The actual work: open the source, transform, write the new dataset."""
    from lerobot.datasets.dataset_postprocess import process_dataset, split_stereo_cameras
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    state.stage = "opening dataset"
    src = LeRobotDataset(repo_id=cfg.source_repo_id, root=cfg.source_root)

    def on_progress(p: dict) -> None:
        state.stage = p.get("stage", state.stage)
        state.frames_total = p.get("frames_total", state.frames_total)
        state.frames_done = p.get("frames_done", state.frames_done)
        state.episodes_total = p.get("episodes_total", state.episodes_total)
        state.episodes_done = p.get("episodes_done", state.episodes_done)
        state.current_episode = p.get("current_episode", state.current_episode)

    if getattr(cfg, "kind", "segment") == "episode_masks":
        from lerobot.datasets.dataset_postprocess import generate_episode_masks

        result = generate_episode_masks(
            src,
            episode=int((cfg.episodes or [0])[0]),
            objects=cfg.objects,
            cameras=cfg.cameras,
            model=cfg.model,
            resolution=cfg.resolution,
            multi_instance=cfg.multi_instance,
            background_treatment=cfg.background_treatment,
            adopt=bool(getattr(cfg, "adopt", False)),
            progress=on_progress,
            should_cancel=lambda: state.cancel_requested,
        )
        state.coverage = result.get("coverage")
        state.status = "cancelled" if result.get("cancelled") else "complete"
        state.stage = "cancelled" if result.get("cancelled") else "done"
        return

    if getattr(cfg, "kind", "segment") == "split_stereo":
        result = split_stereo_cameras(
            src,
            out_repo_id=cfg.out_repo_id,
            cameras=cfg.cameras or [],
            episodes=cfg.episodes,
            out_root=cfg.out_root,
            progress=on_progress,
            should_cancel=lambda: state.cancel_requested,
        )
        state.status = "cancelled" if result.cancelled else "complete"
        state.stage = "cancelled" if result.cancelled else "done"
        return

    result = process_dataset(
        src,
        out_repo_id=cfg.out_repo_id,
        objects=cfg.objects,
        background_treatment=cfg.background_treatment,
        apply_mode=cfg.apply_mode,
        variants=cfg.variants,
        multi_instance=cfg.multi_instance,
        cameras=cfg.cameras,
        episodes=cfg.episodes,
        out_root=cfg.out_root,
        model=cfg.model,
        resolution=cfg.resolution,
        progress=on_progress,
        should_cancel=lambda: state.cancel_requested,
    )
    if result.cancelled:
        state.status = "cancelled"
        state.stage = "cancelled"
    else:
        state.status = "complete"
        state.stage = "done"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = os.environ.get("LEROBOT_PROCESS_WORKER_CONFIG")
    if not raw:
        print("ERROR: LEROBOT_PROCESS_WORKER_CONFIG not set", file=sys.stderr)
        return 2
    cfg = ProcessJobConfig.from_json(raw)
    paths = ProcessJobPaths.for_job(cfg.job_id, cfg.jobs_dir)
    paths.jobs_dir.mkdir(parents=True, exist_ok=True)

    state = _WorkerState()
    atomic_write_json(paths.pid, pid_file_payload(os.getpid()))

    # SIGTERM/SIGINT = graceful cancel. The flag is checked between frames in
    # process_dataset; the handler only flips a bool (no locks, reentrant-safe).
    def _on_signal(*_a) -> None:
        state.cancel_requested = True
        state.stage = "cancelling"

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    # Tee logs to the job's log file so the GUI's "open log" can tail loading /
    # detections / errors, mirroring the overlay + hub workers.
    fh = logging.FileHandler(paths.log)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)

    stop_writer = threading.Event()

    def _writer() -> None:
        while not stop_writer.is_set():
            try:
                atomic_write_json(paths.progress, state.snapshot())
            except Exception:  # noqa: BLE001 — heartbeat must outlive one bad write
                # This thread is the server's only liveness signal for the
                # job. Letting an exception end it freezes the progress file
                # while the work continues, and the GUI then shows a stale
                # snapshot indefinitely with no indication anything is wrong
                # — the failure mode that motivated the same guard in
                # hub_worker. A transient I/O error must cost one tick.
                logger.warning("progress write failed; continuing", exc_info=True)
            stop_writer.wait(PROGRESS_WRITE_INTERVAL_S)

    writer = threading.Thread(target=_writer, name="process-progress-writer", daemon=True)
    writer.start()

    try:
        _run(cfg, state)
    except Exception as e:  # noqa: BLE001 — any failure becomes a terminal job state
        logger.exception("post-process job failed")
        state.status = "failed"
        state.stage = "failed"
        state.error = f"{type(e).__name__}: {e}"
        paths.log.open("a").write("\n" + traceback.format_exc())
    finally:
        state.finished_at = time.time()
        stop_writer.set()
        writer.join(timeout=1.0)
        atomic_write_json(paths.progress, state.snapshot())  # terminal write
        # Leave the pid file; the server's identity check + GC clean it up.
    return 0 if state.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
