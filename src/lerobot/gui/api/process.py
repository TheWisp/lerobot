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

import asyncio
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from lerobot.gui.process_jobs import (
    JOBS_DIR,
    ProcessJobConfig,
    ProcessJobPaths,
    make_job,
)
from lerobot.overlays.effects import TREATMENTS
from lerobot.utils.constants import HF_LEROBOT_HOME

# Previews are single-episode runs written to the normal dataset location (so
# they're detectable in the default Source + open like any dataset) under a
# ``__preview`` suffix that we overwrite each run — ephemeral, but findable.
PREVIEW_SUFFIX = "__preview"

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

OBS_IMAGES = "observation.images."

router = APIRouter(prefix="/api/process", tags=["process"])

_app_state: AppState = None  # type: ignore  # set by server.py

_VALID_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_TREATMENT_KEYS = {t.key for t in TREATMENTS} | {"none"}


def set_app_state(state: AppState) -> None:
    global _app_state
    _app_state = state


@router.get("/treatments")
async def list_treatments() -> dict:
    """The per-region treatment palette the frontend renders as each row's segmented
    control (Tint / Random / Blur / None …). Every region — each object and the
    background — chooses one. Randomized treatments sample once per episode."""
    return {
        "treatments": [
            {"key": t.key, "label": t.label, "controls": t.controls, "randomized": t.randomized}
            for t in TREATMENTS
        ]
    }


class StartRequest(BaseModel):
    source_id: str
    objects: list[dict]  # [{name, sign, treatment:{key,params}}]
    background_treatment: dict | None = None  # {key, params}; default = random colour
    apply_mode: str = "per_episode"
    variants: int = 1
    multi_instance: bool = True  # segment all instances of each object (both arms) vs largest
    cameras: list[str] | None = None
    model: str = "sam3_track"
    # SAM inference resolution preset (ConceptMaskAdapter.RESOLUTIONS); None = adapter
    # default. Must match the live preview's — preview == commit includes resolution.
    resolution: int | None = None
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
    named = [o for o in (req.objects or []) if str(o.get("name", "")).strip()]
    if not named:
        raise HTTPException(status_code=400, detail="Name at least one object to segment")
    bg_treatment = req.background_treatment or {"key": "random", "params": {}}
    all_treatments = [bg_treatment, *[o.get("treatment") or {"key": "none"} for o in named]]
    for tr in all_treatments:
        if (tr.get("key") or "none") not in _TREATMENT_KEYS:
            raise HTTPException(status_code=400, detail=f"Unknown treatment: {tr.get('key')}")
    if all((tr.get("key") or "none") == "none" for tr in all_treatments):
        raise HTTPException(
            status_code=400, detail="Set at least one treatment (an object or the background)"
        )
    from lerobot.overlays.adapters import SEGMENTER_KEYS, ConceptMaskAdapter

    if req.model not in SEGMENTER_KEYS:
        raise HTTPException(
            status_code=400, detail=f"Unknown segmentation model: {req.model}; have {list(SEGMENTER_KEYS)}"
        )
    if req.resolution is not None and req.resolution not in ConceptMaskAdapter.RESOLUTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown resolution {req.resolution}; presets: {list(ConceptMaskAdapter.RESOLUTIONS)}",
        )
    if req.apply_mode not in ("per_episode", "per_frame", "static"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown apply_mode: {req.apply_mode!r} (expected per_episode, per_frame or static)",
        )
    if req.apply_mode == "static" and req.variants > 1:
        # "static" draws ONE look for the whole run, so every variant would be a
        # byte-identical copy — N times the GPU cost and N times the disk for no
        # added diversity. Refuse rather than silently writing duplicates.
        raise HTTPException(
            status_code=400,
            detail="apply_mode='static' uses a single draw for the entire run, so "
            f"variants={req.variants} would write identical copies. Use "
            "apply_mode='per_episode' for independently randomized variants, or set variants=1.",
        )
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
        name = (req.out_name or f"{src_name}_aug").strip()
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
    # Self-heal: a job that already finished but hasn't been polled by /jobs still
    # holds the slot (background activities don't heartbeat-expire). Settle terminal
    # jobs first so a just-finished preview doesn't block the next one.
    for j in list(_app_state.process_jobs.values()):
        _settle(j)
    holder = SLOT.holder(now)
    # A human summary of the per-region edits for the job card (no single "effect").
    summary_parts = []
    if (bg_treatment.get("key") or "none") != "none":
        summary_parts.append(f"{bg_treatment['key']} bg")
    summary_parts += [
        f"{(o.get('treatment') or {}).get('key')} {o['name']}"
        for o in named
        if ((o.get("treatment") or {}).get("key") or "none") != "none"
    ]
    job = make_job(
        source_id=req.source_id,
        out_repo_id=out_repo_id,
        out_root=str(out_root),
        effect=", ".join(summary_parts) or "edit",
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
        resolution=req.resolution,
        objects=req.objects,
        background_treatment=req.background_treatment or {"key": "random", "params": {}},
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
        "spawned post-process worker pid=%d job=%s edit=%s -> %s",
        proc.pid,
        job.job_id,
        job.effect,
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


class SplitStereoRequest(BaseModel):
    source_id: str
    cameras: list[str]  # camera keys to split, bare ("top") or fully qualified
    out_name: str | None = None  # dataset name part; combined with the source owner
    episodes: list[int] | None = None  # subset to convert; None = all


@router.get("/stereo-candidates/{source_id:path}")
async def stereo_candidates(source_id: str) -> dict:
    """Cameras in a dataset that could be a side-by-side stereo pair.

    A pair is not detectable from metadata alone, so this reports the even-width
    cameras and leaves the choice to the operator. Width is the only hard
    requirement: an odd-width frame cannot be halved.
    """
    if _app_state is None or source_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {source_id}")
    src = _app_state.datasets[source_id]
    out = []
    for key in src.meta.camera_keys:
        h, w = (int(x) for x in src.meta.features[key]["shape"][:2])
        name = key.removeprefix(OBS_IMAGES)
        out.append(
            {
                "name": name,
                "width": w,
                "height": h,
                "splittable": w % 2 == 0,
                # A frame twice as wide as it is tall is the usual shape of a
                # side-by-side pair, so it is worth pointing at — but only as a hint.
                "likely_stereo": w % 2 == 0 and w >= 2 * h,
                "channels": [f"{name}_l", f"{name}_r"] if w % 2 == 0 else [],
            }
        )
    return {"cameras": out}


@router.post("/split-stereo")
async def split_stereo(req: SplitStereoRequest) -> dict:
    """Convert side-by-side stereo cameras into one channel per eye.

    Writes a NEW dataset; the source is untouched. No GPU slot is taken, because
    the transform is decode/encode only — a live overlay can keep running
    alongside it.
    """
    if _app_state is None or req.source_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {req.source_id}")
    if not req.cameras:
        raise HTTPException(status_code=400, detail="Select at least one camera to split")

    src = _app_state.datasets[req.source_id]
    known = {k.removeprefix(OBS_IMAGES) for k in src.meta.camera_keys}
    unknown = [c for c in req.cameras if c.removeprefix(OBS_IMAGES) not in known]
    if unknown:
        raise HTTPException(
            status_code=400, detail=f"Not cameras of this dataset: {unknown}; have {sorted(known)}"
        )
    for cam in req.cameras:
        key = cam if cam.startswith(OBS_IMAGES) else f"{OBS_IMAGES}{cam}"
        width = int(src.meta.features[key]["shape"][1])
        if width % 2:
            raise HTTPException(
                status_code=400,
                detail=f"{cam} is {width} px wide; a side-by-side pair must have an even width",
            )
    if _app_state.active_process_job_for(req.source_id) is not None:
        raise HTTPException(status_code=409, detail="A processing job is already running for this dataset")

    owner = src.repo_id.split("/")[0] if "/" in src.repo_id else "local"
    src_name = src.repo_id.split("/")[-1]
    name = (req.out_name or f"{src_name}_split").strip()
    if not _VALID_NAME.match(name):
        raise HTTPException(status_code=400, detail="Output name may only contain letters, digits, . _ -")
    out_repo_id = f"{owner}/{name}"
    out_root = HF_LEROBOT_HOME / out_repo_id
    if out_root.exists():
        raise HTTPException(status_code=409, detail=f"Output dataset already exists: {out_repo_id}")

    for j in list(_app_state.process_jobs.values()):
        _settle(j)

    job = make_job(
        source_id=req.source_id,
        out_repo_id=out_repo_id,
        out_root=str(out_root),
        effect=f"split {', '.join(req.cameras)}",
        preview=False,
    )
    _app_state.process_jobs[job.job_id] = job

    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    paths = ProcessJobPaths.for_job(job.job_id, JOBS_DIR)
    cfg = ProcessJobConfig(
        job_id=job.job_id,
        source_id=req.source_id,
        source_repo_id=src.repo_id,
        source_root=str(src.root),
        out_repo_id=out_repo_id,
        out_root=str(out_root),
        model="",
        objects=[],
        background_treatment={"key": "none", "params": {}},
        apply_mode="per_episode",
        variants=1,
        multi_instance=False,
        cameras=req.cameras,
        episodes=req.episodes,
        preview=False,
        jobs_dir=str(JOBS_DIR),
        kind="split_stereo",
    )
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
    logger.info("spawned stereo-split worker pid=%d job=%s -> %s", proc.pid, job.job_id, out_repo_id)
    return {"job_id": job.job_id, "out_repo_id": out_repo_id}


def _ep_label(episodes: list[int]) -> str:
    """Human label for one episode or many, used in the slot and the job effect."""
    return f"ep{episodes[0]}" if len(episodes) == 1 else f"{len(episodes)} episodes"


class EpisodeMasksRequest(BaseModel):
    source_id: str
    episode: int
    #: Several episodes in ONE job. The single-episode `episode` above stays the
    #: interactive case (save what you just tuned); this is how a whole dataset
    #: gets masked without spawning a worker — and reloading SAM3 — per episode.
    #: None means "just `episode`", so existing callers are unaffected.
    episodes: list[int] | None = None
    confirm_adopt: bool = False
    #: Overwriting an episode that already has saved masks needs consent too:
    #: the natural tuning loop keeps the overlay on, and without this gate any
    #: later save would silently replace masks someone had already confirmed.
    confirm_overwrite: bool = False
    #: Update the recipe (treatments/background) without re-segmenting: a
    #: metadata write, instant, trivially reversible — no job, no consent.
    effects_only: bool = False
    # Everything below defaults to the LIVE overlay's current settings — the
    # operator tunes against the preview, then saves what they see.
    objects: list[dict] | None = None
    cameras: list[str] | None = None
    model: str | None = None
    resolution: int | None = None
    multi_instance: bool | None = None
    background_treatment: dict | None = None


@router.post("/episode-masks")
async def start_episode_masks(
    req: EpisodeMasksRequest, x_overlay_session: str | None = Header(default=None)
) -> dict:
    """Save masks for ONE episode as a frame-aligned feature, from live settings.

    The recipe is stored, never pixels: rows carry COCO RLE per camera, the
    feature metadata carries labels + per-label treatments + background — so
    playback and training reproduce the effect and a later treatment change is
    a metadata edit. First save on a dataset is a schema change and returns
    409 ``adopt_masks_feature`` until ``confirm_adopt`` is true. A different
    label vocabulary is **not** refused: the writer normalises it to the
    stored one plus any new names, so a reorder is discarded and a rename
    becomes an append. The ``mask_labels_differ`` branch below is
    unreachable -- see the annotation at it.
    """
    if _app_state is None or req.source_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {req.source_id}")
    src = _app_state.datasets[req.source_id]
    episode_list = [int(e) for e in (req.episodes if req.episodes is not None else [req.episode])]
    if not episode_list:
        raise HTTPException(400, "no episodes requested")
    bad = [e for e in episode_list if e < 0 or e >= src.meta.total_episodes]
    if bad:
        raise HTTPException(status_code=404, detail=f"Episode not found: {bad[:5]}")

    # Live overlay settings are the default recipe (the collaboration link).
    from lerobot.gui.api import overlays as _ovl

    live = getattr(_ovl, "_data_pub_config", None) or {}
    objects = req.objects if req.objects is not None else list(live.get("objects") or [])
    cameras = req.cameras if req.cameras is not None else list(live.get("cameras") or [])
    model = req.model or live.get("model") or "sam3_track"
    resolution = req.resolution if req.resolution is not None else live.get("resolution")
    multi_instance = (
        req.multi_instance if req.multi_instance is not None else bool(live.get("multi_instance", True))
    )
    background = (
        req.background_treatment
        if req.background_treatment is not None
        else (live.get("background_treatment") or {"key": "none"})
    )
    # Deduped like the writer's own list, so the gates below reason about the
    # vocabulary that will actually be stored (see `generate_episode_masks`).
    labels = list(
        dict.fromkeys(str(o.get("name", "")).strip() for o in objects if str(o.get("name", "")).strip())
    )
    if not labels:
        raise HTTPException(400, "no named objects — configure the overlay (or pass objects) first")
    cam_keys = [c for c in src.meta.camera_keys if not cameras or c in set(cameras)]
    if not cam_keys:
        raise HTTPException(400, "no cameras selected")

    from lerobot.datasets.mask_compositing import mask_keys_for

    # ── gates, answered structurally so the client can drive the dialog ──────
    mask_key_of = mask_keys_for(cam_keys)
    missing = [mask_key_of[c] for c in cam_keys if mask_key_of[c] not in src.meta.features]
    if missing and not req.confirm_adopt:
        raise HTTPException(
            409,
            detail={
                "code": "adopt_masks_feature",
                "message": "Saving masks adds a dataset-wide feature (values are per frame). "
                "Confirm to adopt it; afterwards saves only rewrite the episode in view.",
                "features": missing,
                "labels": labels,
            },
        )
    for c in cam_keys:
        key = mask_key_of[c]
        if key in src.meta.features:
            have = list(src.meta.features[key].get("mask_labels", []))
            # Appending is safe — stored ids keep their meaning — so only a
            # change that would move an existing label is refused. This is what
            # lets one episode be re-run with an extra object without
            # regenerating the dataset.
            merged = have + [name for name in labels if name not in have]
            # **NOT IMPLEMENTED.** `merged` is built by appending to `have`, so
            # its first len(have) entries ARE `have` for every possible input:
            # this condition is a tautology and the 409 below is unreachable,
            # including for the reorder and rename it names. The stored ids are
            # still safe -- the writer normalises -- but the client never
            # receives this error, so the intent is dropped silently instead.
            # Pinned in tests/datasets/test_mask_vocabulary.py.
            if merged[: len(have)] != have:
                raise HTTPException(
                    409,
                    detail={
                        "code": "mask_labels_differ",
                        "existing": have,
                        "requested": labels,
                        "message": "That would move a label the stored masks already use, which "
                        "changes what every other episode's rows mean. Adding objects is fine; "
                        "renaming or reordering them is not.",
                    },
                )

    # Effects-only: rewrite the recipe metadata; nothing touches rows or
    # video, so this is instant and needs no consent dialog.
    if req.effects_only:
        # Every mask column, not the selected ones. A treatment is a property
        # of the object rather than the view -- blurring the arm in one camera
        # and tinting it in another describes nothing a model could learn --
        # so editing one with a single camera selected must not split them.
        from lerobot.datasets.mask_store import mask_columns

        existing_keys = sorted(
            set(mask_columns(src).values())
            | {mask_key_of[c] for c in cam_keys if mask_key_of[c] in src.meta.features}
        )
        if not existing_keys:
            raise HTTPException(
                409,
                detail={"code": "adopt_masks_feature", "message": "No masks feature yet — save masks first."},
            )
        from lerobot.datasets.dataset_postprocess import _update_mask_feature_info

        treatments = {
            str(o.get("name", "")).strip(): (o.get("treatment") or {"key": "none"})
            for o in objects
            if str(o.get("name", "")).strip()
        }
        # Only the reproduction options. model/resolution/multi_instance are
        # provenance of the STORED rows, which an effects edit does not touch —
        # rewriting them here stamped in live-panel defaults, wrong whenever
        # the worker was off.
        _update_mask_feature_info(
            Path(src.root),
            {key: {"mask_treatments": treatments, "mask_background": background} for key in existing_keys},
        )
        # Composited playback reads the recipe from disk; refresh in-memory
        # meta in place for the remaining readers (masks read-back, status).
        # Rows and videos are untouched, so nothing else needs reloading.
        from lerobot.datasets.io_utils import load_info

        src.meta.info = load_info(src.meta.root)
        # New fingerprints, per image camera, so the client can cache-bust its
        # composited URLs without a second round trip.
        from lerobot.datasets.mask_compositing import (
            camera_feature_of,
            load_recipe_from_disk,
            recipe_fingerprint,
        )

        fingerprints = {}
        for key in existing_keys:
            cam = camera_feature_of(key, cam_keys)
            spec = load_recipe_from_disk(src.root, cam)
            if spec is not None:
                fingerprints[cam] = recipe_fingerprint(spec)
        return {"updated": "effects", "features": existing_keys, "fingerprints": fingerprints}

    # Overwrite gate: an episode with existing rows is confirmed work. The
    # client auto-confirms for its OWN just-saved episodes (smooth iteration)
    # and asks the user before replacing anything it did not save itself.
    existing = [k for k in (mask_key_of[c] for c in cam_keys) if k in src.meta.features]
    if existing and not req.confirm_overwrite:
        # Every requested episode, or a multi-episode run would report the first
        # one as empty and quietly replace the other 273.
        coverage = dict.fromkeys(existing, 0)
        for ep_i in episode_list:
            start_idx = int(src.meta.episodes["dataset_from_index"][ep_i])
            length = int(src.meta.episodes["length"][ep_i])
            for key in existing:
                col = src.hf_dataset[key][start_idx : start_idx + length]
                for cell in col:
                    v = cell[0] if isinstance(cell, (list, tuple)) else cell
                    if v and str(v) not in ("", "[]"):
                        coverage[key] += 1
        if any(coverage.values()):
            raise HTTPException(
                409,
                detail={
                    "code": "masks_exist",
                    "coverage": coverage,
                    "frames": length,
                    "message": f"Episode {req.episode} already has saved masks. Overwrite them "
                    "with the current settings?",
                },
            )

    # ── slot + job, the same rules as every batch pass ───────────────────────
    from lerobot.gui.api.overlays import _data_key, _stop_live, stop_data_publisher
    from lerobot.gui.gpu_slot import SLOT

    now = time.time()
    own_overlay = _data_key(x_overlay_session)
    for j in list(_app_state.process_jobs.values()):
        _settle(j)
    holder = SLOT.holder(now)
    job = make_job(
        source_id=req.source_id,
        out_repo_id=src.repo_id,  # in place: the "output" IS the source
        out_root=str(src.root),
        effect=f"masks {_ep_label(episode_list)}: {', '.join(labels)}",
        preview=False,
    )
    proc_key = f"process:{job.job_id}"
    if holder is not None and holder.key not in (proc_key, own_overlay):
        raise HTTPException(status_code=409, detail={"code": "overlay_busy", "holder": holder.label})
    SLOT.release(own_overlay)
    stop_data_publisher()
    await _stop_live()
    SLOT.acquire(proc_key, f"saving masks {_ep_label(episode_list)}", now, heartbeat=False)

    _app_state.process_jobs[job.job_id] = job
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    paths = ProcessJobPaths.for_job(job.job_id, JOBS_DIR)
    cfg = ProcessJobConfig(
        job_id=job.job_id,
        source_id=req.source_id,
        source_repo_id=src.repo_id,
        source_root=str(src.root),
        out_repo_id=src.repo_id,
        out_root=str(src.root),
        model=model,
        resolution=resolution,
        objects=objects,
        background_treatment=background,
        apply_mode="per_episode",
        variants=1,
        multi_instance=multi_instance,
        cameras=cam_keys,
        episodes=episode_list,
        preview=False,
        kind="episode_masks",
        adopt=bool(req.confirm_adopt or not missing),
        jobs_dir=str(JOBS_DIR),
    )
    # Same launch tail as _spawn_worker: progress stub so an immediate poll
    # reads something, config through the environment, detached process group.
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
    logger.info("spawned episode-masks worker pid=%d job=%s ep=%d", proc.pid, job.job_id, req.episode)
    asyncio.get_event_loop().create_task(_rebind_when_done(job.job_id, req.source_id))
    return {
        "job_id": job.job_id,
        "status": "started",
        "episode": req.episode,
        "episodes": episode_list,
        "labels": labels,
    }


# Dataset reconstruction after an in-place save can take hundreds of ms on
# big datasets; keep it off the shared default pool (gui-async-hygiene).
_rebind_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gui-rebind")


async def _rebind_when_done(job_id: str, source_id: str) -> None:
    """Reload the dataset into AppState once an in-place job finishes.

    In-place writes change parquet under the server's cached LeRobotDataset,
    whose hf_dataset would keep serving pre-save rows; a stale editor after a
    successful save is indistinguishable from the save having failed.
    """
    while True:
        await asyncio.sleep(2.0)
        job = _app_state.process_jobs.get(job_id) if _app_state else None
        if job is None:
            return
        _settle(job)
        if job.status in ("complete", "failed", "cancelled"):
            break
    if job.status != "complete":
        return
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = _app_state.datasets.get(source_id)
        root = getattr(ds, "root", None) or source_id
        # NOTE: not local_files_only=True — LeRobotDataset has no such kwarg
        # (api/datasets.py's auto-open-for-upload path passes it and would
        # TypeError if ever hit). Passing root= keeps resolution local.
        fresh = await asyncio.get_event_loop().run_in_executor(
            _rebind_executor,
            lambda: LeRobotDataset(source_id, root=root),  # blocking-ok: runs on _rebind_executor
        )
        _app_state.datasets[source_id] = fresh
        # The job rewrote mask ROWS; the recipe fingerprint that keys
        # composited caches only covers the recipe, so those entries are now
        # stale by content. Clear the frame cache and drop this dataset's
        # composited transcodes (plain transcodes stay — videos are untouched).
        from lerobot.gui import api as _gui_api
        from lerobot.gui.cache_invalidation import invalidate_caches

        invalidate_caches(_app_state, source_id)
        # Composited transcodes are served by the playback rewrite, which is not
        # on every branch that has masks. Where it is absent there is nothing to
        # drop; importing its cache directory unconditionally aborted this
        # handler BEFORE invalidate_caches above, so a mask save left the editor
        # serving frames from before the save.
        _cache_dir = getattr(_gui_api.datasets, "_playback_cache_dir", None)
        removed = 0
        if _cache_dir is not None:
            for f in _cache_dir().glob(f"{source_id.replace('/', '_')}__*__m*.mp4"):
                f.unlink(missing_ok=True)  # safe-destruct: derived cache entry; rebuilt on next request
                removed += 1
        if removed:
            logger.info("episode-masks: dropped %d composited transcodes for %s", removed, source_id)
        logger.info("episode-masks: dataset %s rebound after in-place save", source_id)
    except Exception:
        logger.exception("episode-masks: rebind failed for %s — editor may serve stale rows", source_id)


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
