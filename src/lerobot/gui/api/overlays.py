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
"""Overlays API — run one processing step on the current observation.

Both tabs drive the SAME out-of-process worker (``overlays/standalone.py``),
which reads the obs stream (``lerobot_obs_*``) and writes per-camera RGBA overlays
to a SharedOverlayBuffer the GUI serves as PNG. The only difference is who publishes
the obs stream: teleop/policy/record for the run tab; the GUI itself for the data
tab (it publishes each scrubbed frame — see ``start_data_publisher`` /
``publish_data_frame``). One obs-stream writer at a time (run XOR data). The worker's
lifecycle is the shared state machine (``overlay_state.py``). See gui/docs/overlays.md.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from lerobot.gui.gpu_slot import SLOT
from lerobot.overlays.overlay_state import Event, OverlayStateMachine, State

# The live stream decodes dataset frames for as long as it runs; keep that off
# the shared default pool so it cannot starve unrelated offloaded work. One
# worker: the loop is sequential by design (latest-wins frame choice).
_stream_decode_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gui-stream-decode")

if TYPE_CHECKING:
    from lerobot.gui.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/overlays", tags=["overlays"])

_app_state: AppState = None  # type: ignore  # set by server.py


def set_app_state(state: AppState) -> None:
    global _app_state
    _app_state = state


# ---------------------------------------------------------------------------
# Processing-step registry — the productized subset (SAM3 first). Each step
# declares its body (controls), its result kind, and a load-cost hint. The
# frontend prepends a "none" entry (the off switch); it is not listed here.
# ---------------------------------------------------------------------------
_STEPS: list[dict] = [
    {
        "key": "sam3_track",
        "label": "SAM3",
        "result_kind": "spatial",
        "load_cost": "slow",  # gated weights + encoder warmup; a few seconds
        "controls": [
            {
                "type": "objects",
                "key": "prompt",
                "label": "Objects",
                "placeholder": "green ring",
                "hint": "Each object is detected once then locked + tracked in its own color.",
            }
        ],
    },
    {
        "key": "policy_saliency",
        "label": "Attention map",  # colloquial name; technically gradient saliency / attention rollout (see the doc)
        "result_kind": "spatial",
        "load_cost": "fast",  # no model of its own — reads the running policy's saliency via shm
        "controls": [
            {
                "type": "select",
                "key": "method",
                "label": "Method",
                "default": "gradient",
                "options": [
                    {"value": "gradient", "label": "Gradient (causal — what the action uses)"},
                    {"value": "rollout", "label": "Rollout (attention — where it routes from)"},
                ],
            },
            {
                "type": "select",
                "key": "style",
                "label": "Style",
                "default": "blue_yellow",
                "options": [
                    {"value": "blue_yellow", "label": "Blue → Yellow"},
                    {"value": "cividis", "label": "Cividis (uniform)"},
                    {"value": "spotlight", "label": "Spotlight (hotspots only)"},
                    {"value": "heatmap", "label": "Full heatmap"},
                    {"value": "inferno", "label": "Inferno (golden)"},
                ],
            },
            {
                "type": "slider",
                "key": "smooth",
                "label": "Smoothing",
                "min": 0.0,
                "max": 3.0,
                "step": 0.1,
                "default": 1.2,
            },
        ],
    },
]


# SAM inference-resolution presets, single-sourced from the adapter (a LOAD-TIME knob:
# the GUI reads these from /models; endpoints validate against them). Labels carry the
# measured trade-off so the picker is self-explanatory.
_RESOLUTIONS: list[dict] = [
    {"value": 1008, "label": "Full (1008 px)"},
    {"value": 672, "label": "Balanced (672 px) — default"},
    {"value": 504, "label": "Fast (504 px)"},
]


def _validate_resolution(resolution: int | None) -> None:
    """400 on a resolution outside the adapter presets (None = adapter default is fine)."""
    from lerobot.overlays.adapters import ConceptMaskAdapter

    if resolution is not None and resolution not in ConceptMaskAdapter.RESOLUTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown resolution {resolution}; presets: {list(ConceptMaskAdapter.RESOLUTIONS)}",
        )


@router.get("/models")
async def list_models() -> dict:
    """The processing steps the picker offers (besides 'none'), plus which of them are
    text-prompted segmenters (valid for data editing) and the resolution presets."""
    from lerobot.overlays.adapters import SEGMENTER_KEYS

    return {"models": _STEPS, "segmenters": list(SEGMENTER_KEYS), "resolutions": _RESOLUTIONS}


_pmon_last: tuple[float, dict[int, int]] = (0.0, {})  # (monotonic_t, {pid: sm%}) — cache ~1 Hz


def _proc_sm(pid: int | None) -> int:
    """A process's OWN GPU utilization % (SM %) via ``nvidia-smi pmon`` — the
    model's share of the card, NOT whole-card. 0 if the PID is idle ('-') or
    unavailable. One pmon call (~30 ms) lists every PID, cached ~1 Hz, so the
    live (subprocess) and data (this process) statuses share it.

    (torch.cuda.utilization() needs nvidia-ml-py and only gives whole-card; pmon
    gives per-process without a new dependency.)"""
    if pid is None:
        return 0
    global _pmon_last
    now = time.monotonic()
    if now - _pmon_last[0] >= 1.0:
        table: dict[int, int] = {}
        with contextlib.suppress(Exception):
            out = subprocess.run(["nvidia-smi", "pmon", "-c", "1"], capture_output=True, text=True, timeout=3)
            for line in out.stdout.splitlines():
                cols = line.split()  # gpu_idx pid type sm mem ... ; header lines start with '#'
                if len(cols) >= 4 and cols[1].isdigit():
                    table[int(cols[1])] = int(cols[3]) if cols[3].isdigit() else 0
        _pmon_last = (now, table)
    return _pmon_last[1].get(pid, 0)


class ConfigureRequest(BaseModel):
    dataset_id: str
    model: str
    objects: list[dict] | None = None  # [{name, sign:'+'/'-', treatment:{key,params}}]
    cameras: list[str] | None = None  # active subset — worker infers + we publish only these; None = all
    # Per-region background treatment ({key, params}) for the region behind every object.
    # It no longer selects a render mode: the worker composites regions + draws detection
    # chrome for ANY segmenter, on both tabs (it branches on the adapter type, not on this).
    background_treatment: dict | None = None
    # Segment ALL instances of each object (both arms) vs the single largest.
    multi_instance: bool = True
    # SAM inference resolution preset (ConceptMaskAdapter.RESOLUTIONS). Load-time
    # knob: changing it respawns the worker. None = the adapter default.
    resolution: int | None = None
    # False when every region was designated by clicking. With no word to search for, the
    # adapter would otherwise fall back to hunting DEFAULT_PROMPT, and each time it lost that
    # phantom it rebuilt the tracker — which destroyed the clicked objects every few seconds.
    text_detection: bool = True
    # Which SAM3 box API a drag uses: "tracker" (promptable segmenter) or "exemplar" (visual
    # prompt to the detector). Both are exposed because they fail differently in clutter.
    box_method: str = "tracker"
    # Batch the vision encode across cameras (experimental; default on). Runtime
    # toggle — rides the control push, no respawn.


def _frame_rgb(item: dict, cam: str) -> np.ndarray:
    """A dataset item's camera tensor -> contiguous HxWx3 uint8 RGB (what the
    adapter's infer() expects)."""
    import torch

    t = item[cam]
    if t.dim() == 3 and t.shape[0] in (1, 3, 4):  # CHW -> HWC
        t = t.permute(1, 2, 0)
    if t.is_floating_point():
        t = (t * 255).clamp(0, 255).to(torch.uint8)
    elif t.dtype != torch.uint8:
        t = t.to(torch.uint8)
    a = t.cpu().numpy()
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[:, :, :3]
    assert a.ndim == 3 and a.shape[2] == 3, f"expected HxWx3, got {a.shape}"
    return np.ascontiguousarray(a)


def _png(rgba: np.ndarray) -> bytes:
    """Encode an HxWx4 RGBA overlay to PNG (preserves transparency).

    Pre: HxWx4 uint8 RGBA. Post: PNG bytes whose visible result is unchanged —
    only pixels the viewer cannot see are rewritten.

    Fully-transparent pixels keep whatever RGB the compositor left there, which for
    an overlay that is mostly a transparent diff is the WHOLE camera frame: PNG then
    compresses a full photo nobody will ever see. Measured on a live 1280x720 run-tab
    overlay that is 94.2% transparent: 782 KB as-is vs 88 KB with the invisible RGB
    zeroed — 8.8x for pixels alpha discards anyway. Size matters here because these
    are pulled per tile continuously, and an overlay that cannot finish downloading
    before the next pull replaces it never draws at all.
    """
    from PIL import Image

    assert rgba.ndim == 3 and rgba.shape[2] == 4, f"expected HxWx4 RGBA, got {rgba.shape}"
    invisible = rgba[..., 3] == 0
    if invisible.any():
        rgba = rgba.copy()  # never mutate the caller's shm-backed view
        rgba[invisible] = 0
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _dataset_camera_dims(ds) -> dict[str, tuple[int, int]]:
    """{cam: (h, w)} for the dataset's cameras — for the obs-stream publisher. Dims come from a
    single decoded sample (unambiguous about the real resolution)."""
    item = ds[0]
    out: dict[str, tuple[int, int]] = {}
    for cam in ds.meta.camera_keys:
        if cam in item:
            h, w = _frame_rgb(item, cam).shape[:2]
            out[cam] = (int(h), int(w))
    return out


class DataPublishRequest(BaseModel):
    dataset_id: str
    episode: int
    frame: int
    # Re-publish the same frame without resetting tracking. A paused episode publishes
    # nothing, and the worker only reads controls when it processes a frame — so a gesture
    # has to hand it one, or the click is never seen.
    force: bool = False


# ── Overlay = one activity on the shared aux-GPU slot ─────────────────────────
#
# The live overlay is a heavy aux-GPU activity, so it acquires the process-wide
# aux-GPU slot (see lerobot.gui.gpu_slot). The slot is a plain mutex over the one
# GPU resource; batch augment jobs acquire the SAME slot. So a data tab, another
# machine's data tab, the run-tab overlay, and a running process job all contend
# for it identically. The overlay is an INTERACTIVE activity — it heartbeats via
# its ~2 Hz status poll, so a closed tab frees the slot after the timeout.
_RUN_KEY = "overlay:run"  # nosec B105 - the run-tab overlay's activity key (a label, not a secret)


def _data_key(session: str | None) -> str:
    return f"overlay:data:{session or 'default'}"


@router.post("/data/configure")
async def data_configure(req: ConfigureRequest, x_overlay_session: str | None = Header(default=None)) -> dict:
    """Turn on the data overlay: publish the scrubbed frames to the obs stream (the GUI is the
    writer) and spawn the worker, which reads that stream exactly like the run path. Acquires the
    aux-GPU slot as an activity; refuses (409 ``overlay_busy``) if any other activity (another data
    tab, the run overlay, or a batch job) holds it. A re-configure by the same session refreshes it."""
    if _app_state is None or req.dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {req.dataset_id}")
    if req.model not in {s["key"] for s in _STEPS}:
        raise HTTPException(status_code=400, detail=f"Unknown overlay model: {req.model}")
    _validate_resolution(req.resolution)
    key = _data_key(x_overlay_session)
    now = time.time()
    if not SLOT.acquire(key, "SAM3 overlay", now, heartbeat=True):
        raise HTTPException(
            status_code=409, detail={"code": "overlay_busy", "holder": SLOT.holder(now).label}
        )
    ds = _app_state.datasets[req.dataset_id]
    cameras = _dataset_camera_dims(ds)
    if not cameras:
        SLOT.release(key)
        raise HTTPException(status_code=400, detail="Dataset has no camera/image keys")
    config = {
        "objects": req.objects or [],
        "background_treatment": req.background_treatment or {"key": "random", "params": {}},
        "multi_instance": req.multi_instance,
        "text_detection": req.text_detection,
        "box_method": req.box_method,
    }
    # start_data_publisher enforces the obs-stream physical constraint (teleop is the sole writer
    # during a run) — separate from the aux-GPU slot; surface it as the same overlay_busy state.
    if not start_data_publisher(req.dataset_id, cameras, config):
        SLOT.release(key)
        raise HTTPException(status_code=409, detail={"code": "overlay_busy", "holder": "teleop run"})
    m = _machine(req.model)
    async with _live_lock:
        # A dataset switch replaces the obs stream, but the worker only needs a RESPAWN when
        # the stream's SHAPE changes: camera names and dims are baked into the segments the
        # worker mapped and the overlay buffers it created. Keying the teardown on dataset
        # IDENTITY (as this used to) threw away the worker — and its ~6 s SAM3 load — for
        # every switch, including a dataset and its own __preview, whose shapes are equal by
        # construction. Same shape: the worker re-attaches to the replaced segments by inode
        # (the same machinery that survives a teleop restart on the run tab) and the
        # publisher's generation bump reseeds tracking — measured ~12 s of "model reload"
        # becomes a ~2 s reseed. Different shape (or a worker of unknown shape): respawn.
        global _data_worker_dims
        if not _worker_serves(req.model, req.resolution, cameras):
            await _teardown_current()
        await _spawn_worker(
            req.model,
            objects=req.objects,
            background_treatment=req.background_treatment or {"key": "random", "params": {}},
            resolution=req.resolution,
            multi_instance=req.multi_instance,
            text_detection=req.text_detection,
            box_method=req.box_method,
        )
        _data_worker_dims = dict(cameras)
    # Narrow the worker to the panel's selected cameras so disabling one actually cuts its work:
    # publish only those + filter inference to them (None/absent = keep the default = all cameras).
    global _data_pub_cameras
    if req.cameras is not None:
        _data_pub_cameras = req.cameras
    # Always push the control so the worker picks up the effect/objects/background (the
    # config lives in _data_pub_config; the reader attaches lazily, so this is a no-op
    # until the worker's buffer exists, then the frontend's re-sync delivers it).
    _write_data_control()
    return {"ok": True, "state": m.state.value}


@router.get("/data/status")
async def data_status(x_overlay_session: str | None = Header(default=None)) -> dict:
    """Badge/state for the data overlay — the worker's lifecycle machine + fps/util/vram (the
    worker is identical to the run path). ``publishing`` reflects the obs-stream writer. Also the
    slot heartbeat: the holder's poll refreshes its lease; anyone else sees ``busy`` + the holder."""
    key = _data_key(x_overlay_session)
    now = time.time()
    is_owner = SLOT.touch(key, now)  # heartbeat iff this session holds the slot
    busy = SLOT.blocks(key, now)
    _observe()
    target = _live_model
    machine = _machines.get(target) if target else None
    state = machine.state if machine is not None else State.INACTIVE
    running = target is not None and _live_proc is not None and _live_proc.returncode is None
    st = _read_status() if running else {}
    reader = _get_live_reader() if running else None
    # Re-push the config each poll while the worker is up (control writes are a no-op until the
    # worker's shm buffer exists) — but only the OWNER drives it, so a non-owner poll never clobbers.
    if is_owner and reader is not None and _data_publisher_active():
        _write_data_control()
    return {
        "state": state.value,
        "available": state is State.ACTIVE,
        "model": target,
        "cameras": list(reader.cameras) if reader is not None else [],
        "fps": float(st.get("fps", 0.0)),
        "vram": float(st.get("vram", 0.0)),
        # The worker's measured per-camera compute (model + effects), from its ~1 Hz
        # latency block. None until it has actually inferred — consumers must treat
        # absence as "no measurement", not substitute a constant.
        "compute_ms": (lambda lat: float(lat["compute_ms"]) if lat.get("compute_ms") else None)(
            reader.read_latency() if reader is not None else {}
        ),
        "util": _proc_sm(_live_proc.pid) if running else 0,
        "publishing": _data_publisher_active(),
        # The client kept its own copy of this and could not learn that a batch
        # job had disarmed the mode. One fact, reported by the side that owns it.
        "apply_armed": _data_apply_on,
        "owner": is_owner,
        "busy": busy,
        "holder": SLOT.holder(now).label if busy else None,
    }


@router.post("/data/cancel")
async def data_cancel(x_overlay_session: str | None = Header(default=None)) -> dict:
    """Turn the data overlay off — release the slot and stop the obs-stream publisher, but PARK the
    worker instead of killing it. Cancel fires on every switch to a dataset with no overlay config,
    so tearing down here made a dataset bounce cost a full SAM3 reload (~10 s); parked, the worker
    idles in its wait-for-stream loop with the model warm, and the next configure either re-attaches
    it (same shape/model/resolution) or respawns it. The slot IS released — any other activity that
    acquires it (a batch job via its takeover in api/process.py, the run overlay via live_start)
    evicts the parked worker before using the GPU, so the slot's free-VRAM contract still holds.
    /data/free remains the explicit kill. Only the HOLDER cancels the shared publisher; another
    activity's cancel is a no-op so it can't stop the holder's overlay."""
    key = _data_key(x_overlay_session)
    now = time.time()
    if SLOT.blocks(key, now):  # someone else holds the slot — don't touch their overlay
        return {"ok": True, "note": "not the holder; nothing torn down"}
    SLOT.release(key)
    stop_data_publisher()
    parked = _live_proc is not None and _live_proc.returncode is None
    if parked:
        logger.info("data overlay canceled — worker parked warm (model %r loaded)", _live_model)
    return {"ok": True, "parked": parked}


@router.post("/data/free")
async def data_free() -> dict:
    """Free VRAM — same as cancel for the worker (tear it down)."""
    stop_data_publisher()
    await _stop_live()
    return {"ok": True}


@router.get("/data/log")
async def data_log(lines: int = 400) -> dict:
    """The worker's log tail (loading / detections / seeds / errors) for the panel's 'open log'."""
    return await live_log(lines)


@router.post("/data/publish")
async def data_publish(
    req: DataPublishRequest, x_overlay_session: str | None = Header(default=None)
) -> Response:
    """Frontend calls this on every frame change: decode the landed frame (all cameras) and publish
    it to the obs stream so the worker overlays it. The decode runs off the event loop. No-op unless
    a data publisher is active for this dataset AND the caller holds the slot (only the holder drives
    the single frame slot, so a background client can't fight over which frame is segmented)."""
    if SLOT.blocks(_data_key(x_overlay_session), time.time()):
        return Response(status_code=204)  # not the holder — don't publish into the shared slot
    if not _data_publisher_active() or _app_state is None or req.dataset_id not in _app_state.datasets:
        return Response(status_code=204)
    ds = _app_state.datasets[req.dataset_id]

    def _decode_and_publish() -> None:
        from lerobot.gui.api.datasets import _get_episode_start_index

        start = _get_episode_start_index(req.dataset_id, req.episode)
        # ds[i] decodes EVERY camera's video for this frame, not just the ones the
        # overlay wants — the dominant unmeasured term in the scrub-to-overlay path.
        t_dec = time.perf_counter()
        item = ds[start + req.frame]
        dec_ms = (time.perf_counter() - t_dec) * 1000.0
        t_pub = time.perf_counter()
        publish_data_frame(req.dataset_id, req.episode, req.frame, item, force=bool(req.force))
        pub_ms = (time.perf_counter() - t_pub) * 1000.0
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "publish ep=%d frame=%d: decode %.1fms (%d cams) + publish %.1fms",
                req.episode,
                req.frame,
                dec_ms,
                len(ds.meta.camera_keys),
                pub_ms,
            )

    t_all = time.perf_counter()
    await asyncio.get_event_loop().run_in_executor(None, _decode_and_publish)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "publish ep=%d frame=%d: handler total %.1fms (includes executor queueing)",
            req.episode,
            req.frame,
            (time.perf_counter() - t_all) * 1000.0,
        )
    return Response(status_code=204)


@router.get("/data/{dataset_id:path}/frame/{episode_idx}/{frame_idx}")
async def data_overlay_frame(
    dataset_id: str, episode_idx: int, frame_idx: int, camera: str | None = None
) -> Response:
    """The LATEST overlay for a camera (the worker's output). The episode/frame in the path are
    cache-busters; we serve the newest result (it lags playback a little, like the live overlay —
    no frame-matching)."""
    if camera is None or not _data_publisher_active():
        return Response(status_code=204)
    return await _serve_overlay(camera)


# ---------------------------------------------------------------------------
# Data obs-stream publisher — the data tab feeds the overlay worker by publishing
# the scrubbed frame into the SAME obs stream a run uses (lerobot_obs_*). ONE writer
# at a time: it refuses while a run owns the stream (run.is_run_active) and run-start
# tears it down (run._launch_subprocess), so a robot-connect can't clobber it. Created
# on overlay-on, torn down on overlay-off / dataset-change / leaving the data tab.
# ---------------------------------------------------------------------------
_data_pub = None  # ObservationStream | None — the active data writer
_data_pub_dataset: str | None = None
_data_pub_cameras: list[str] = []
_data_pub_config: dict | None = None  # the step's config (objects, ...) — pushed via the control
_data_pub_last_pos: tuple[int, int] | None = None  # (episode, frame) for jump detection
#: Apply is armed by the operator and only acts while frames are played into the
#: overlay loop. Ticking it is not a write, so this only sets a mode.
_data_apply_on: bool = False
#: Last time we said the run is armed with no worker behind it. Throttled: the
#: run drains several times a second, and a line per poll would bury the log.
_data_apply_no_worker_at: float = 0.0
#: (camera, obs_seq) -> (episode, frame), for masks coming back from the worker.
#: The worker reports the seq it CONSUMED; by the time a batch arrives the
#: playhead may have moved, so the position has to be remembered from when the
#: frame was published rather than read from wherever the playhead is now.
#: Bounded: a run at ~10 fps on two cameras fills this at ~20 entries/s, and the
#: worker is at most a flush (~1 s) behind, so a few hundred is generous.
_APPLY_POS_KEEP = 600
_data_apply_pos: OrderedDict[tuple[str, int], tuple[int, int]] = OrderedDict()
#: Last mask-block counter drained, so a poll returns each batch once.
_data_apply_last_seq: int = -1
# The camera shape map ({cam: (h, w)}) the LIVE worker attached to. The worker's obs
# mapping and its overlay buffers are sized by this, so it — not the dataset's identity —
# is what decides whether a dataset switch needs a respawn.
_data_worker_dims: dict[str, tuple[int, int]] | None = None
# The latest click/box op, re-sent on every control write so it is not overwritten before
# the worker reads it. Cleared in _teardown_current — it addresses one worker.
from lerobot.overlays.adapters import _CLICK_OP_KEYS  # noqa: E402  (one definition of "an op")

_last_click_op: dict = {}
# The op sequence is assigned HERE, not by the browser. It was Date.now()-based per client,
# which silently drops gestures whenever more than one client drives the worker: two tabs
# land on the same millisecond, two laptops disagree by their clock skew, and the worker's
# "already applied" gate cannot tell a stale id from another client's. One counter, one
# authority; reset with the worker, whose own counter restarts at 0.
_click_seq = 0
_data_pub_generation = 0  # bumped on a new stream (jump / episode / wrap) -> the worker resets
# How far the playhead may jump forward and still count as the same continuous video. Half a
# second at 30 fps: enough to absorb the frames playback drops when inference is the slower
# side, short enough that a real scrub still reads as a discontinuity.
_CONTINUOUS_SKIP = 15


def _data_publisher_active() -> bool:
    return _data_pub is not None


def _write_data_control() -> None:
    """Push {generation, cameras, config} to the worker's control channel. No-op until the worker
    has created the overlay buffer (the reader attaches lazily)."""
    reader = _get_live_reader()
    if reader is None:
        return
    try:
        reader.write_control(
            {
                "generation": _data_pub_generation,
                "cameras": _data_pub_cameras or None,
                "config": _data_pub_config or {},
                # A mode, not an action: the worker only publishes masks for the
                # frames that are played into it while this is set.
                "apply": bool(_data_apply_on),
                # This runs on every status poll and overwrites the whole block, so without
                # carrying the op a data-tab gesture is erased within ~1 s.
                **_last_click_op,
            }
        )
    except Exception:
        logger.warning("data control write failed — the worker won't see this config change", exc_info=True)


def start_data_publisher(dataset_id: str, cameras: dict[str, tuple[int, int]], config: dict) -> bool:
    """Create the obs-stream writer for the data tab. Returns False (does nothing) if a run already
    owns the stream — one writer at a time. ``cameras`` is {cam: (h, w)}; ``config`` is the step's
    opaque config (objects, ...). Precondition: a data overlay is being turned on."""
    global _data_pub, _data_pub_dataset, _data_pub_cameras, _data_pub_config, _data_pub_last_pos
    from lerobot.gui.api.run import is_run_active

    if is_run_active():
        logger.info("data publisher: a run owns the obs stream — not starting")
        return False
    if _data_pub is not None and _data_pub_dataset == dataset_id:
        _data_pub_config = config  # same dataset already up — just refresh the step config
        return True
    stop_data_publisher()  # replace any prior writer (different dataset)
    from lerobot.robots.obs_stream import ObservationStream

    obs_features = {cam: (h, w, 3) for cam, (h, w) in cameras.items()}
    _data_pub = ObservationStream(obs_features, {})
    _data_pub_dataset = dataset_id
    _data_pub_cameras = list(cameras)
    _data_pub_config = config
    _data_pub_last_pos = None
    logger.info("data publisher: obs stream up for %s (%d cameras)", dataset_id, len(cameras))
    return True


def stop_data_publisher() -> None:
    """Tear down the data obs-stream writer (overlay off / dataset change / leaving the data tab /
    a run starting). Idempotent."""
    global _data_pub, _data_pub_dataset, _data_pub_cameras, _data_pub_config, _data_pub_last_pos
    if _data_pub is not None:
        with contextlib.suppress(Exception):
            _data_pub.cleanup()
        _data_pub = None
        logger.info("data publisher: obs stream torn down")
    _data_pub_dataset = None
    _data_pub_cameras = []
    _data_pub_config = None
    _data_pub_last_pos = None


def publish_data_frame(dataset_id: str, episode: int, frame: int, item: dict, *, force: bool = False) -> None:
    """Publish the decoded frame to the obs stream + bump ``generation`` on a new stream (scrub jump
    / episode change / wrap) so the worker drops its stale tracker and reseeds. A re-published *same*
    frame (a paused playhead, or the status poll re-sending the current frame) is a no-op: republishing
    it would reset the tracker and re-run the detector on a frame already done — the data path's old
    ~3fps. No-op unless the publisher is active for this dataset. The caller passes the already-decoded
    dataset ``item`` so there is no extra decode.

    ``force`` re-publishes even an unchanged frame, WITHOUT the generation bump (so tracking is
    kept, not reset). The worker samples the control block once per frame it processes, so on a
    paused episode it is parked and never sees a click/box op — and a single latched slot means
    the next op overwrites the unread one (measured: a click and a box 218 ms apart, only the box
    ever reached the worker). A gesture therefore hands the worker a frame to sample on."""
    global _data_pub_last_pos, _data_pub_generation
    if _data_pub is None or _data_pub_dataset != dataset_id:
        return
    pos = (int(episode), int(frame))
    if pos == _data_pub_last_pos and not force:
        return  # same frame already published — let the worker keep tracking, don't reset + re-infer
    # Plain playback advances one frame at a time; that +1 step is continuous, so the worker's tracker
    # just propagates. Any other move — a scrub, an episode change, or the wrap to the loop start — is
    # a new stream: bump generation so the worker resets its per-camera tracking and reseeds.
    # Continuity, not adjacency. Requiring exactly last+1 made a single dropped frame a new
    # stream: playback advances on a timer while inference runs slower, so skips are routine,
    # and each one reset the tracker and re-ran the detector for every concept (~30 ms each).
    # A short forward gap within the same episode is the same video; the tracker propagates
    # across it. Backwards, a different episode, or a long jump is still a scrub.
    step = (
        pos[1] - _data_pub_last_pos[1]
        if _data_pub_last_pos is not None and pos[0] == _data_pub_last_pos[0]
        else None
    )
    sequential = step is not None and 1 <= step <= _CONTINUOUS_SKIP
    if not sequential and not force:
        _data_pub_generation += 1
        _write_data_control()
    _data_pub_last_pos = pos
    obs = {cam: _frame_rgb(item, cam) for cam in _data_pub_cameras if cam in item}
    try:
        _data_pub.write_obs(obs)
    except Exception:
        logger.warning("data obs publish failed for frame %s — the overlay won't update", pos, exc_info=True)
        return
    if _data_apply_on:
        # Remember where each camera's freshly written frame sits, so the masks
        # the worker returns for that seq can be filed against it. Outside the
        # try above on purpose: an error in this bookkeeping is not a failed
        # publish, and reporting it as one hid an AttributeError here behind a
        # message about the overlay not updating.
        for cam in obs:
            _data_apply_pos[(cam, int(_data_pub.image_seq(cam)))] = pos
        while len(_data_apply_pos) > _APPLY_POS_KEEP:
            _data_apply_pos.popitem(last=False)


# ---------------------------------------------------------------------------
# Live (run) path — a debug-vision standalone subprocess reads the live
# ObservationStream, runs the adapter, and writes per-camera RGBA overlays to a
# SharedOverlayBuffer; we attach read-only and serve them as PNG. Control
# (objects/prompt) is pushed back through the buffer. Teleop/record must be
# publishing the obs stream for the subprocess to start producing overlays.
# ---------------------------------------------------------------------------
_live_proc: asyncio.subprocess.Process | None = None
_live_reader = None  # SharedOverlayBuffer | None (read-only attach)
_live_png_cache: dict[str, tuple[int, bytes]] = {}
_live_frame_warned: set[str] = set()  # cam keys we've logged "never produced" for (once each, per run)
_live_frame_served: set[str] = set()  # cam keys we've logged a first successful serve for (once each)
_live_model: str | None = None
_live_resolution: int | None = None  # the running worker's SAM resolution (load-time; change = respawn)
_live_log_path: Path | None = None
_live_log_file = None  # parent's handle to the worker log; the next spawn closes it, so fds don't accumulate across respawns
_live_lock = asyncio.Lock()  # serialises start/stop; a queued start waits for a teardown, never dropped
_live_stopping = False  # a commanded teardown is in flight — the status poll defers its event-firing to it
_live_status_reader = None  # OverlayStatus(create=False): the standalone's self-reported phase/fps/vram
_machines: dict[str, OverlayStateMachine] = {}  # PER MODEL — switching A->B keeps A/B states independent


def _get_live_reader():
    """Lazily attach to the subprocess's overlay buffer. It creates the segments
    only after the obs stream exists, so attach fails (None) until then."""
    global _live_reader
    if _live_reader is not None:
        return _live_reader
    try:
        from lerobot.overlays.overlay_ipc import SharedOverlayBuffer

        _live_reader = SharedOverlayBuffer(create=False)
    except FileNotFoundError:
        _live_reader = None
    except Exception:
        logger.exception("live overlay reader attach failed")
        _live_reader = None
    return _live_reader


# --------------------------------------------------------------------------
# Live preview as one H.264 stream (verification slice)
#
# The per-frame publish/pull loop costs two round trips and a ~115 KB PNG per
# frame, which a remote link cannot sustain; measured 0.8 fps against 10 fps on
# localhost. This endpoint moves the loop server-side: settings arrive lazily
# via /data/configure as before, and the composited preview leaves as a single
# fragmented-MP4 stream the browser buffers like any video. Encoder settings
# are the run-tab preview's, which already stream live over the same links.
# --------------------------------------------------------------------------


def _stream_encoder_command(ffmpeg: str, width: int, height: int, fps: int, bitrate_kbps: int) -> list[str]:
    """The run-tab preview's encoder settings, parametrized for the atlas size."""
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-threads",
        "1",
        "-profile:v",
        "baseline",
        "-level:v",
        "3.0",
        "-b:v",
        f"{bitrate_kbps}k",
        "-maxrate",
        f"{bitrate_kbps}k",
        "-bufsize",
        f"{max(128, bitrate_kbps // 5)}k",
        "-g",
        str(fps),
        "-keyint_min",
        str(fps),
        "-bf",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+empty_moov+default_base_moof+omit_tfhd_offset+frag_every_frame",
        "-f",
        "mp4",
        "pipe:1",
    ]


def _frame_rgb_uint8(value) -> np.ndarray:
    """A dataset camera tensor as contiguous HxWx3 uint8 (what the worker saw)."""
    import torch

    t = value
    if t.dim() == 3 and t.shape[0] in (1, 3, 4):
        t = t.permute(1, 2, 0)
    if t.is_floating_point():
        t = (t * 255).clamp(0, 255).to(torch.uint8)
    return np.ascontiguousarray(t.cpu().numpy())


#: Common tile height of the streamed atlas. Every selected camera is scaled to
#: this height and tiled horizontally; one frame carrying all cameras is what
#: makes cross-camera sync structural rather than maintained.
_STREAM_ATLAS_H = 360
#: Encoder timeline rate. Matches the dataset fps so stream time IS episode
#: time: the feeder picks frames by wall clock (latest-wins, like the live
#: worker's own seq gate) and pads the timeline with duplicate frames, which
#: H.264 encodes to almost nothing. Playback is then 1x with the overlay
#: refreshing at whatever rate segmentation sustains — the localhost feel.
_STREAM_FPS = 30


@router.get("/data/stream.mp4")
async def data_overlay_stream(
    request: Request,
    dataset_id: str,
    episode: int = 0,
    from_frame: int = 0,
    cameras: str = "",
    max_frames: int = 0,
    bitrate_kbps: int = 900,
    x_overlay_session: str | None = Header(default=None),
):
    """The live composited preview of one episode as ONE fragmented-MP4 stream.

    Pre: the data overlay is configured and ACTIVE (the caller went through
    /data/configure) and the caller holds the slot. ``cameras`` is a csv of
    camera keys (short names accepted); every one must be produced by the
    worker. Post: streams until the episode ends, ``max_frames`` were sent, or
    the client disconnects. Pacing is the worker's own — the next frame is
    published only after every selected camera's overlay for the previous one
    arrived, so the single frame slot is never contended and the tiles are of
    the same instant by construction.

    The tile layout rides in the ``X-Overlay-Layout`` response header (same-
    origin fetch can read it): ``{atlas:[W,H], fps, from_frame, cameras:{key:
    [x,y,w,h]}}``.
    """
    import shutil as _shutil

    import cv2

    if _app_state is None or dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    if SLOT.blocks(_data_key(x_overlay_session), time.time()):
        raise HTTPException(status_code=409, detail="another activity holds the overlay slot")
    if not _data_publisher_active():
        raise HTTPException(
            status_code=409, detail="data overlay is not configured; POST /data/configure first"
        )
    reader = _get_live_reader()
    if reader is None:
        raise HTTPException(status_code=503, detail="overlay worker not running")
    ds = _app_state.datasets[dataset_id]

    cam_list = [c.strip() for c in cameras.split(",") if c.strip()]
    cam_list = [c if c.startswith("observation.") else f"observation.images.{c}" for c in cam_list]
    if not cam_list:
        cam_list = [next(iter(ds.meta.camera_keys))]
    unknown = [c for c in cam_list if c not in ds.meta.camera_keys]
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown cameras: {unknown}")
    not_produced = [c for c in cam_list if c not in reader.cameras]
    if not_produced:
        raise HTTPException(
            status_code=409,
            detail=f"worker does not produce {not_produced}; its cameras: {list(reader.cameras)}",
        )
    ffmpeg = _shutil.which("ffmpeg")
    if ffmpeg is None:
        raise HTTPException(status_code=503, detail="ffmpeg not found")

    from lerobot.gui.api.datasets import _get_episode_start_index

    start = _get_episode_start_index(dataset_id, episode)
    length = int(ds.meta.episodes["length"][episode])
    first = max(0, min(from_frame, length - 1))
    last = length if max_frames <= 0 else min(length, first + max_frames)

    # Layout from the first frame's true dimensions: each camera scaled to the
    # common height, width rounded to even (yuv420 requires it), tiled in order.
    item0 = ds[start + first]
    tiles: dict[str, tuple[int, int, int, int]] = {}
    x = 0
    for cam in cam_list:
        h, w = _frame_rgb_uint8(item0[cam]).shape[:2]
        sw = max(2, int(round(w * _STREAM_ATLAS_H / h)) & ~1)
        tiles[cam] = (x, 0, sw, _STREAM_ATLAS_H)
        x += sw
    atlas_w = x
    layout = {
        "atlas": [atlas_w, _STREAM_ATLAS_H],
        "fps": _STREAM_FPS,
        "from_frame": first,
        "cameras": {cam: list(r) for cam, r in tiles.items()},
    }
    command = _stream_encoder_command(ffmpeg, atlas_w, _STREAM_ATLAS_H, _STREAM_FPS, bitrate_kbps)

    async def gen():
        t_started = time.monotonic()
        frames = 0
        bytes_out = 0
        atlas = np.empty((_STREAM_ATLAS_H, atlas_w, 3), dtype=np.uint8)
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stop = asyncio.Event()

        async def feed():
            nonlocal frames
            emitted = 0
            t_play = time.monotonic()
            ds_fps = float(ds.meta.fps or _STREAM_FPS)
            prev_f = None
            try:
                while True:
                    if stop.is_set() or proc.returncode is not None:
                        return
                    # Latest-wins frame choice: the episode clock runs at 1x
                    # wall time and segmentation covers whichever frame is
                    # current, skipping what it cannot keep up with — within
                    # the publish continuity tolerance, so the tracker
                    # propagates instead of reseeding.
                    f = first + int((time.monotonic() - t_play) * ds_fps)
                    if f >= last:
                        return
                    if prev_f is not None:
                        f = min(f, prev_f + _CONTINUOUS_SKIP)
                        if f <= prev_f:
                            await asyncio.sleep(0.005)
                            continue
                    prev_f = f
                    t_f = time.perf_counter()
                    item = await asyncio.get_event_loop().run_in_executor(
                        _stream_decode_executor, lambda f=f: ds[start + f]
                    )
                    bases = {cam: _frame_rgb_uint8(item[cam]) for cam in cam_list}
                    seq0 = {cam: reader.overlay_seq(cam) for cam in cam_list}
                    publish_data_frame(dataset_id, episode, f, item, force=(f == first))
                    t_pub = time.perf_counter()
                    pending = set(cam_list)
                    while pending:
                        if stop.is_set() or proc.returncode is not None:
                            return
                        pending = {c for c in pending if reader.overlay_seq(c) == seq0[c]}
                        if not pending:
                            break
                        if time.perf_counter() - t_pub > 5.0:
                            logger.warning("stream: overlays for frame %d never arrived (%s)", f, pending)
                            return
                        await asyncio.sleep(0.003)
                    t_blend = time.perf_counter()
                    for cam in cam_list:
                        out = bases[cam]
                        result = reader.read_overlay(cam)
                        if result is not None:
                            rgba, _ts = result
                            a = rgba[..., 3]
                            sel = a > 0
                            if sel.any():
                                out = out.copy()
                                af = (a[sel].astype(np.float32) / 255.0)[:, None]
                                out[sel] = (
                                    out[sel].astype(np.float32) * (1.0 - af)
                                    + rgba[..., :3][sel].astype(np.float32) * af
                                    + 0.5
                                ).astype(np.uint8)
                        tx, _ty, tw, th = tiles[cam]
                        atlas[:, tx : tx + tw] = cv2.resize(out, (tw, th), interpolation=cv2.INTER_AREA)
                    if proc.stdin is None:
                        return
                    # Pad the encoder timeline up to the wall clock with
                    # duplicates of this frame, so the stream's clock stays 1x
                    # regardless of how fast segmentation runs.
                    target = int((time.monotonic() - t_play) * _STREAM_FPS) + 1
                    n_emit = max(1, min(target - emitted, _STREAM_FPS * 3))
                    payload = atlas.tobytes()
                    for _ in range(n_emit):
                        proc.stdin.write(payload)
                    await proc.stdin.drain()
                    emitted += n_emit
                    frames += 1
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "stream frame %d: decode+seg %.0fms · blend+tile %.0fms · total %.0fms",
                            f,
                            (t_blend - t_f) * 1000.0,
                            (time.perf_counter() - t_blend) * 1000.0,
                            (time.perf_counter() - t_f) * 1000.0,
                        )
            finally:
                if proc.stdin is not None:
                    with contextlib.suppress(Exception):
                        proc.stdin.close()

        feeder = asyncio.create_task(feed())
        try:
            while True:
                chunk = await proc.stdout.read(64 * 1024)
                if not chunk:
                    break
                bytes_out += len(chunk)
                yield chunk
        finally:
            stop.set()
            feeder.cancel()
            with contextlib.suppress(Exception):
                proc.kill()
            elapsed = time.monotonic() - t_started
            logger.info(
                "stream done: %d frames x %d cams in %.1fs (%.2f fps) · %.0f KB (%.0f kbit/s)",
                frames,
                len(cam_list),
                elapsed,
                frames / elapsed if elapsed > 0 else 0.0,
                bytes_out / 1024.0,
                bytes_out * 8 / elapsed / 1000.0 if elapsed > 0 else 0.0,
            )

    return StreamingResponse(
        gen(),
        media_type="video/mp4",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "X-Overlay-Layout": json.dumps(layout, separators=(",", ":")),
        },
    )


@router.get("/data/events")
async def data_overlay_events(request: Request):
    """SSE stream: push ``{cam, seq}`` the instant a camera's overlay advances, so the data
    tab re-pulls that frame immediately instead of waiting on its ~2 Hz status poll (the
    dominant felt lag). Cheap: a fine server-side shm-header poll — no browser cost; the
    browser just gets an event and pulls the one changed frame. One stream per data tab."""

    async def gen():
        last: dict[str, int] = {}
        idle_ticks = 0
        while True:
            if await request.is_disconnected():
                return
            reader = _get_live_reader()
            emitted = False
            if reader is not None:
                for cam in list(reader.cameras):
                    seq = reader.overlay_seq(cam)
                    if seq and seq != last.get(cam):
                        last[cam] = seq
                        emitted = True
                        yield f"data: {json.dumps({'cam': cam, 'seq': seq})}\n\n"
            idle_ticks = 0 if emitted else idle_ticks + 1
            if idle_ticks >= 300:  # ~3 s quiet -> comment keepalive so proxies don't drop the stream
                idle_ticks = 0
                yield ": keepalive\n\n"
            await asyncio.sleep(
                0.01
            )  # ~100 Hz seq poll (cheap shm reads), well under the ~90 ms produce time

    return StreamingResponse(gen(), media_type="text/event-stream")


def _close_live_reader() -> None:
    global _live_reader, _live_png_cache
    if _live_reader is not None:
        with contextlib.suppress(Exception):
            _live_reader.cleanup()
        _live_reader = None
    _live_png_cache = {}
    _live_frame_warned.clear()
    _live_frame_served.clear()


def _machine(model: str) -> OverlayStateMachine:
    """Per-model lifecycle machine — switching models keeps each model's state independent."""
    if model not in _machines:
        _machines[model] = OverlayStateMachine(
            on_transition=lambda prev, ev, nxt, m=model: logger.info(
                "overlay[%s]: %s --%s--> %s", m, prev.value, ev.value, nxt.value
            )
        )
    return _machines[model]


def _read_status() -> dict:
    """The running standalone's self-reported {phase, fps, vram} ({} until it exists)."""
    global _live_status_reader
    if _live_status_reader is None:
        try:
            from lerobot.overlays.overlay_ipc import OverlayStatus

            _live_status_reader = OverlayStatus(create=False)
        except (FileNotFoundError, RuntimeError):
            return {}
        except Exception:
            logger.exception("overlay status reader attach failed")
            return {}
    with contextlib.suppress(Exception):
        return _live_status_reader.read()
    return {}


def _close_status_reader() -> None:
    global _live_status_reader
    if _live_status_reader is not None:
        with contextlib.suppress(Exception):
            _live_status_reader.cleanup()
        _live_status_reader = None


def _observe() -> None:
    """Poll-side event source: map every observation of the standalone (reported phase +
    process liveness) to a state-machine event. Skips while a commanded teardown owns the
    transitions. Exhaustiveness matters here: a silently-dropped observation is how the
    badge ends up lying (frozen "active" on a dead worker, stuck "off" on a live one) —
    if reality contradicts the machine, either fire the mapping event or log the desync."""
    global _live_proc, _live_model, _live_resolution, _data_worker_dims
    if _live_stopping or _live_model is None or _live_proc is None:
        return
    m = _machine(_live_model)
    rc = _live_proc.returncode
    if rc is None:  # alive
        phase = _read_status().get("phase")
        if m.state is State.LOADING and phase == "active":
            m.fire(Event.LOADED)
        elif m.state is State.INACTIVE and phase:
            # Impossible if our invariants hold (segments are swept before every spawn,
            # and a spawn fires START before the worker exists) — a report here means
            # machine and reality disagree. Never drop that silently.
            logger.warning("overlay state desync: worker reports %r while machine is inactive", phase)
        return
    # The process ended. Every exit maps to an event — returning silently here would
    # freeze the badge on a dead process.
    if rc in (0, -15):
        # A clean/SIGTERM exit we did NOT command (teardown sets _live_stopping): the
        # worker ended on its own. The overlay is simply gone — back to inactive.
        logger.warning("overlay worker exited on its own (rc=%s) — resetting to inactive", rc)
        m.fire(Event.STOP)
        m.fire(Event.STOPPED)
    else:
        m.fire(Event.CRASH)  # died abnormally; the machine holds ERROR until start/stop
    _close_live_reader()
    _close_status_reader()
    _live_proc = None
    _data_worker_dims = None  # the process the shape described is gone
    if m.state is State.INACTIVE:  # self-exit path: nothing is running, clear ownership
        _live_model = None
        _live_resolution = None


def _worker_serves(model: str, resolution: int | None, dims: dict | None) -> bool:
    """Can the running worker serve this request as it stands?

    The identity is `(model, resolution, bound stream shape)`. It used to be split: the first
    two were checked inside ``_spawn_worker`` and the shape by each caller, with a different
    ad-hoc rule each — which is why a shape mismatch needed its own eviction path in
    ``live_start``. ``dims`` is the shape the caller needs: a camera->(h, w) map for the data
    tab, and None for the run tab, whose worker reads teleop's stream and so can never reuse
    one bound to a dataset.

    False here means the caller must tear down first. ``_spawn_worker`` tears down anyway
    before spawning, so an early teardown only ever moves that work earlier.
    """
    return (
        _live_proc is not None
        and _live_proc.returncode is None
        and _live_model == model
        and _live_resolution == resolution
        and _data_worker_dims == dims
    )


async def _teardown_current() -> None:
    """Stop the running standalone. Caller MUST hold _live_lock. Fires STOP -> STOPPED (or RESET
    if it had already crashed); no-op when nothing is running. Also forgets the data-stream shape
    the worker was bound to, and the last click/box op — neither record may outlive the process it
    describes. A kept op is worse than useless: the replacement worker starts at click_seq 0, so the
    next control write would replay a click made on the PREVIOUS dataset, at that frame's pixel
    coordinates, seeding a tracker on whatever now happens to lie under them."""
    global _live_proc, _live_model, _live_resolution, _live_stopping, _data_worker_dims
    global _last_click_op, _click_seq, _data_apply_on
    _data_worker_dims = None
    _last_click_op = {}
    _click_seq = 0  # the replacement worker's own counter restarts at 0 too
    # An armed Apply cannot outlive the worker either: its masks are what that
    # worker produces. Here rather than in `_stop_live`, which is only one of the
    # four callers of this -- a model switch and the data-tab cancel tear a
    # worker down too, and each would otherwise leave the mode armed over
    # nothing. Same reason as the lines above: no record of a process may
    # outlive the process it describes.
    _data_apply_on = False
    _data_apply_pos.clear()
    if _live_model is None:
        return
    m = _machine(_live_model)
    if m.state is State.ERROR:  # already dead — just clear the error, nothing to terminate
        m.fire(Event.RESET)
        _close_live_reader()
        _close_status_reader()
        _live_proc = None
        _live_model = None
        _live_resolution = None
        return
    m.fire(Event.STOP)  # active/loading -> stopping
    _live_stopping = True
    try:
        if _live_proc is not None and _live_proc.returncode is None:
            _live_proc.terminate()
            try:
                await asyncio.wait_for(_live_proc.wait(), timeout=5.0)
            except Exception:
                with contextlib.suppress(Exception):
                    _live_proc.kill()
                    await _live_proc.wait()
        _close_live_reader()
        _close_status_reader()
        m.fire(Event.STOPPED)  # stopping -> inactive
    finally:
        _live_proc = None
        _live_model = None
        _live_resolution = None
        _live_stopping = False


async def _stop_live() -> None:
    """Lock-wrapped teardown for external callers (server shutdown, /live/stop).

    Disarms Apply, because an armed run's masks are produced by the worker this
    tears down: the mode cannot outlive it. Done here rather than at each call
    site so every path that kills the worker gets it -- a batch job taking the
    GPU, the operator stopping the overlay, shutdown -- instead of each caller
    having to remember. Leaving it set left a ticked box over a worker that no
    longer existed, and Play then published frames nobody was segmenting.
    """
    async with _live_lock:
        await _teardown_current()


class LiveStartRequest(BaseModel):
    model: str
    objects: list[dict] | None = None  # [{name, sign, treatment:{key,params}}] — same shape as the data tab
    background_treatment: dict | None = None  # {key, params}; None = background kept as-is
    cameras: list[str] | None = None
    style: str | None = None  # policy_saliency render style (see PolicySaliencyAdapter.STYLES)
    smooth: float | None = None  # policy_saliency smoothing sigma (0 = raw 64x64)
    method: str | None = None  # policy_saliency source: "gradient" | "rollout" (read by the policy)
    resolution: int | None = None  # SAM inference resolution preset (load-time; None = adapter default)
    # Segment ALL instances of each object (both arms) vs the single largest — the same
    # control the data tab has. Defaults FALSE here, where the tab is a live debug view and
    # every extra masklet costs frame rate; the data tab defaults True because a treatment
    # that protects only one of two arms silently corrupts the written dataset.
    multi_instance: bool = False
    # False = no text detector; the only concepts are the ones clicked on a tile.
    text_detection: bool = True
    # See ConfigureRequest.box_method — the same knob, the same two SAM3 box APIs.
    box_method: str = "tracker"


class LiveDiagRequest(BaseModel):
    model: str | None = None
    fps: float | None = None
    objects: int = 0
    started: bool = False
    reason: str = ""
    available: list[str] = []
    selected: list[str] | None = None
    drawn: list[str] = []
    blank: list[str] = []


async def _spawn_worker(
    model: str,
    *,
    objects=None,
    cameras=None,
    background_treatment=None,
    style=None,
    smooth=None,
    method=None,
    resolution=None,
    multi_instance=None,
    text_detection=None,
    box_method=None,
) -> None:
    """Spawn (or push control to) the single overlay worker for ``model``. Caller MUST hold
    ``_live_lock``. The worker is identical for live + data — it reads the obs stream; only the
    publisher differs (teleop for run, the GUI data publisher for data). A same-model call just
    pushes control; a different model — or a different ``resolution``, which is baked into the
    model at load — tears the old worker down first."""
    global _live_proc, _live_model, _live_resolution, _live_log_path, _live_log_file
    m = _machine(model)
    if (
        _live_model == model
        and _live_resolution == resolution
        and _live_proc is not None
        and _live_proc.returncode is None
    ):
        reader = _get_live_reader()  # already up — push control, don't restart
        if reader is not None:
            reader.write_control(
                {
                    **_last_click_op,  # same slot, same rule: don't erase an unread op
                    "config": {
                        "objects": objects or [],
                        "background_treatment": background_treatment,
                        "style": style,
                        "smooth": smooth,
                        "method": method,  # read by the POLICY (gradient|rollout), not the worker
                        **({} if multi_instance is None else {"multi_instance": multi_instance}),
                        **({} if text_detection is None else {"text_detection": text_detection}),
                        **({} if box_method is None else {"box_method": box_method}),
                    },
                }
            )
        return
    # Teardown BEFORE firing START. A same-model respawn (a resolution change) shares
    # one state machine: START-first put it in `loading`, then the OLD worker's
    # STOP/STOPPED knocked it back to `inactive`, the new worker's LOADED was invalid
    # from there and dropped — badge permanently "off" while the worker served fine.
    await _teardown_current()  # stop the running worker first (serialised)
    # An uncleanly-killed worker (server death SIGTERMs it mid-CUDA) leaves its shm
    # segments behind; the fixed-name status segment frozen at "active" would make
    # this spawn report loaded instantly. Nothing is running now — sweep them so
    # every segment that exists after this point belongs to the worker we spawn.
    from lerobot.overlays.overlay_ipc import unlink_stale_segments

    n = unlink_stale_segments()
    if n:
        logger.info("swept %d stale overlay shm segment(s) before spawn", n)
    m.fire(Event.START)  # -> loading
    args = [sys.executable, "-u", "-m", "lerobot.overlays.standalone", f"--model={model}"]
    if objects:
        args.append(f"--objects={json.dumps(objects)}")
    # Seed the background treatment at spawn (like objects) — a control-channel push
    # is a no-op until the worker's buffer exists, so the FIRST inference would
    # otherwise miss it.
    if background_treatment is not None:
        args.append(f"--background-treatment={json.dumps(background_treatment)}")
    # Same reason: without seeding, the worker's first inferences use the adapter's own
    # default instead of the panel's instance policy. The data tab happened to converge
    # because its config is re-pushed on every status poll; the run tab has no such
    # re-push, so an unseeded value would simply never arrive.
    if multi_instance is not None:
        args.append(f"--multi-instance={'1' if multi_instance else '0'}")
    if text_detection is not None:
        args.append(f"--text-detection={'1' if text_detection else '0'}")
    if box_method is not None:
        args.append(f"--box-method={box_method}")
    if style:
        args.append(f"--style={style}")
    if method:
        args.append(f"--method={method}")  # worker seeds it into its control block at creation
    if smooth is not None:
        args.append(f"--smooth={smooth}")
    if resolution is not None:
        args.append(f"--resolution={resolution}")
    if cameras:
        args.append("--cameras")
        args.extend(cameras)
    # Per GUI process. A fixed name means a second server — another port, a test instance —
    # truncates the first's worker log the moment it spawns a worker, destroying the evidence
    # of a session still in progress. The log endpoint reads this same variable, so it follows.
    _live_log_path = Path(tempfile.gettempdir()) / f"lerobot_overlays_{os.getpid()}.log"
    if _live_log_file is not None:
        with contextlib.suppress(Exception):
            _live_log_file.close()  # release the prior worker's handle so fds don't accumulate
    _live_log_file = logf = _live_log_path.open("w")
    # Parent-death cleanup: the kernel SIGTERMs the worker if the GUI dies (even on SIGKILL),
    # so it can't orphan and keep hogging the GPU.
    from lerobot.gui.api.run import _set_pdeathsig_preexec

    try:
        _live_proc = await asyncio.create_subprocess_exec(
            *args, stdout=logf, stderr=asyncio.subprocess.STDOUT, preexec_fn=_set_pdeathsig_preexec
        )
    except Exception as e:
        # Exec failure (interpreter gone, fork limits): without an event the machine
        # would sit in `loading` forever with no process behind it.
        m.fire(Event.CRASH)
        raise HTTPException(status_code=500, detail=f"overlay worker failed to spawn: {e}") from e
    _live_model = model
    _live_resolution = resolution


@router.post("/live/start")
async def live_start(req: LiveStartRequest) -> dict:
    """Launch the worker for a model (run tab) — fires START on that model's machine. Serialised
    with stops by _live_lock. Acquires the SAME aux-GPU slot as the data tab + batch jobs, so any
    of them holding it blocks the run overlay and vice versa — refuses (409 overlay_busy)."""
    if req.model not in {s["key"] for s in _STEPS}:
        raise HTTPException(status_code=400, detail=f"Unknown overlay model: {req.model}")
    _validate_resolution(req.resolution)
    now = time.time()
    if not SLOT.acquire(_RUN_KEY, "SAM3 overlay (run)", now, heartbeat=True):
        raise HTTPException(
            status_code=409, detail={"code": "overlay_busy", "holder": SLOT.holder(now).label}
        )
    m = _machine(req.model)
    async with _live_lock:
        # A worker parked by the data tab is bound to a DATASET's stream shape; the teleop
        # stream this overlay will read almost never matches it, and the worker's reattach
        # refuses shape mismatches by design (it would wait forever, badge stuck). Same-model
        # reuse inside _spawn_worker would keep exactly that worker — evict it instead.
        # (Reusing the warm model across the data->run boundary is a recorded follow-up.)
        if not _worker_serves(req.model, req.resolution, None):
            await _teardown_current()
        await _spawn_worker(
            req.model,
            objects=req.objects,
            background_treatment=req.background_treatment,
            cameras=req.cameras,
            style=req.style,
            smooth=req.smooth,
            method=req.method,
            resolution=req.resolution,
            multi_instance=req.multi_instance,
            text_detection=req.text_detection,
            box_method=req.box_method,
        )
    return {"ok": True, "state": m.state.value}


class ApplyArmRequest(BaseModel):
    armed: bool


@router.post("/apply/arm")
async def apply_arm(req: ApplyArmRequest, x_overlay_session: str | None = Header(default=None)) -> dict:
    """Arm or disarm Apply. Ticking the box is NOT a write.

    The design is explicit that ticking stores nothing: a run's masks are the
    ones the preview loop already computes, and they only begin flowing once
    frames are played into it. Arming therefore sets a mode on the worker and
    clears any positions left from a previous run.
    """
    global _data_apply_on
    key = _data_key(x_overlay_session)
    # Apply is a WRITING mode over ONE shared worker and ONE shared drain queue,
    # so it can have only one owner: two tabs arming it split the drained frames
    # between them, and either tab's disarm stopped the other's run.
    #
    # The owner is whoever holds the GPU slot, rather than a second notion of
    # ownership kept here. A parallel one would need its own lifetime, and the
    # first draft of this had none: a tab closed mid-run would have owned Apply
    # for the life of the process. The slot already leases with a heartbeat and
    # reclaims from a tab that stopped polling, which is exactly the lifetime
    # this needs.
    if SLOT.blocks(key, time.time()):
        raise HTTPException(
            status_code=409,
            detail={"code": "overlay_busy", "holder": SLOT.holder(time.time()).label},
        )
    _data_apply_on = bool(req.armed)
    if not _data_apply_on:
        _data_apply_pos.clear()
    _write_data_control()
    logger.info("apply %s (%s)", "armed" if _data_apply_on else "disarmed", key)
    return {"armed": _data_apply_on}


@router.post("/apply/drain")
async def apply_drain() -> dict:
    """Hand back the run's segmented frames, resolved to (episode, frame).

    The worker publishes the masks it computed for the frames it consumed; this
    turns each camera's obs sequence back into the position that frame was
    published at. A sequence with no remembered position is DROPPED rather than
    guessed at -- it means the frame was published before the run was armed, or
    so long ago that the bounded map has rotated past it, and filing masks
    against the wrong frame is worse than not filing them.

    Returns ``{"frames": [{"episode", "frame", "camera", "rle": {name: counts}}]}``.
    The caller decides what to keep: the write rule is applied client-side, where
    the stored coverage already is.
    """
    global _data_apply_last_seq, _data_apply_no_worker_at
    reader = _get_live_reader()
    if reader is None:
        # Armed, playing, and nothing segmenting: the run waits out its whole
        # per-frame deadline on every frame, so the tiles advance about once
        # every eight seconds and the GUI looks frozen. Said here because this
        # is the only side that can see both facts at once, and because the
        # symptom -- "Play does nothing" -- points nowhere near the cause.
        if _data_apply_on and time.monotonic() - _data_apply_no_worker_at > 5.0:
            _data_apply_no_worker_at = time.monotonic()
            logger.warning(
                "apply/drain: the run is armed but no overlay worker is running — "
                "every frame will wait out its deadline. Turn the segmenter on, or untick Apply."
            )
        return {"frames": [], "dropped": 0}
    seq = reader.masks_seq()
    if seq == _data_apply_last_seq:
        return {"frames": [], "dropped": 0}  # nothing new since the last poll
    _data_apply_last_seq = seq
    out: list[dict] = []
    dropped = 0
    for entry in reader.read_masks():
        for cam, payload in (entry or {}).items():
            pos = _data_apply_pos.get((cam, int(payload.get("seq", -1))))
            if pos is None:
                dropped += 1
                continue
            episode, frame = pos
            out.append(
                {
                    "episode": int(episode),
                    "frame": int(frame),
                    "camera": cam,
                    "rle": dict(payload.get("rle") or {}),
                }
            )
    if dropped:
        logger.info("apply drain: %d camera-frames had no remembered position", dropped)
    return {"frames": out, "dropped": dropped}


@router.post("/live/control")
async def live_control(body: dict) -> dict:
    """Push a control update (e.g. {"prompt": "green ring . robot arm"}) to the worker.

    A click/box op is remembered and re-sent on every later write, because the control block
    is one slot that each write overwrites — without this a prompt edit erases a click the
    worker has not read yet. Repeating is safe: the worker applies each op once, by
    ``click_seq``. ``_teardown_current`` forgets it, so it cannot reach the next worker."""
    global _last_click_op, _click_seq
    reader = _get_live_reader()
    if reader is None:
        raise HTTPException(status_code=409, detail="No live overlay producer yet")
    body = {k: v for k, v in body.items() if k != "click_seq"}  # the sequence is ours to assign
    op = {k: v for k, v in body.items() if k in _CLICK_OP_KEYS}
    if op:
        _click_seq += 1
        _last_click_op = {**op, "click_seq": _click_seq}
        # Gestures are the one control whose delivery we cannot infer from anything else:
        # they are events on a latched slot, and a lost one looks exactly like a no-op.
        logger.info("click op %d: %s", _click_seq, op)
    # _last_click_op last: it carries the stamped sequence, which must not lose to anything
    # a client put in the body.
    reader.write_control({**body, **_last_click_op})
    return {"ok": True}


@router.post("/live/stop")
async def live_stop() -> dict:
    SLOT.release(_RUN_KEY)  # release the aux-GPU slot so another activity can take over
    await _stop_live()
    return {"ok": True}


@router.get("/live/status")
async def live_status(model: str | None = None) -> dict:
    """The lifecycle-machine state for `model` (default: the running one), plus live fps/util/
    vram when that model is the running, active one. The state is the machine's — never a string
    assembled here. States: inactive / loading / active / stopping / error."""
    now = time.time()
    SLOT.touch(_RUN_KEY, now)  # heartbeat while the run overlay panel polls
    busy = SLOT.blocks(_RUN_KEY, now)  # another activity holds the aux-GPU slot
    _observe()  # fire LOADED / CRASH for the running standalone from its reported phase + liveness
    target = model or _live_model
    machine = _machines.get(target) if target else None
    state = machine.state if machine is not None else State.INACTIVE
    running = (
        target is not None
        and target == _live_model
        and _live_proc is not None
        and _live_proc.returncode is None
    )
    st = _read_status() if running else {}
    reader = _get_live_reader() if running else None
    resp = {
        "state": state.value,
        "available": state is State.ACTIVE,
        "model": target,
        "cameras": list(reader.cameras) if reader is not None else [],
        "fps": float(st.get("fps", 0.0)),
        "vram": float(st.get("vram", 0.0)),
        # The worker's measured per-camera compute (model + effects), from its ~1 Hz
        # latency block. None until it has actually inferred — consumers must treat
        # absence as "no measurement", not substitute a constant.
        "compute_ms": (lambda lat: float(lat["compute_ms"]) if lat.get("compute_ms") else None)(
            reader.read_latency() if reader is not None else {}
        ),
        "util": _proc_sm(_live_proc.pid) if running else 0,
        "busy": busy,
        "holder": SLOT.holder(now).label if busy else None,
    }
    # A policy-internal overlay's real cost lives in the POLICY process (the worker only
    # colorizes), published as pass_ms through the aux stats block. Attach just that one block —
    # not the whole SharedAuxBuffer (meta + a grid mmap per camera) — this runs on every poll.
    # Absent until the first publish; dropped when stale (policy stopped publishing).
    if running:
        with contextlib.suppress(Exception):  # no aux stats (non-publishing model) — omit
            from lerobot.overlays.aux_ipc import read_stats_pass_ms

            pm = read_stats_pass_ms()
            if pm is not None and time.time() - pm[1] < 30.0:
                resp["sal_ms"] = round(pm[0], 1)
    return resp


@router.get("/live/log")
async def live_log(lines: int = 400) -> dict:
    """The live standalone's log tail (loading / ready / per-frame errors) for the
    panel's 'log' viewer. Empty before the first live run."""
    if _live_log_path is None or not _live_log_path.exists():
        return {"log": ""}
    try:
        text = _live_log_path.read_text(errors="replace")
    except Exception as e:  # noqa: BLE001
        return {"log": f"(could not read {_live_log_path}: {e})"}
    return {"log": "\n".join(text.splitlines()[-lines:])}


@router.post("/live/diag")
async def live_diag(req: LiveDiagRequest) -> dict:
    """The frontend reports its live-overlay state here so a failure is visible in the server
    log, not only the browser console. selected=[] => the panel has no camera chosen (nothing
    will draw — the ordering bug's signature); 'blank' => selected cameras whose overlay <img>
    hasn't rendered."""
    logger.info(
        "live/diag: model=%s started=%s reason=%r objects=%s selected=%s drawn=%s blank=%s fps=%s",
        req.model,
        req.started,
        req.reason,
        req.objects,
        req.selected,
        req.drawn,
        req.blank,
        req.fps,
    )
    return {"ok": True}


async def _serve_overlay(cam_key: str) -> Response:
    """A camera's LATEST overlay RGBA as PNG, from the worker's buffer (PNG-cached by seq). Shared
    by the run + data tabs — always the newest result (it lags playback a little, like the live
    feed; no frame-matching). 404 = a camera the worker never produced; 204 = warming / none yet."""
    reader = _get_live_reader()
    if reader is None:
        return Response(status_code=204)  # worker hasn't created the buffer yet
    if cam_key not in reader.cameras:
        if cam_key not in _live_frame_warned:
            _live_frame_warned.add(cam_key)
            logger.warning(
                "overlay/frame: requested camera %r was never produced — producer cameras=%s "
                "(frontend / stream camera-key mismatch).",
                cam_key,
                list(reader.cameras),
            )
        return Response(status_code=404)
    seq = reader.overlay_seq(cam_key)
    if seq == 0:
        return Response(status_code=204)  # known camera, overlay not written yet
    if cam_key not in _live_frame_served:
        _live_frame_served.add(cam_key)
        logger.info("overlay/frame: first overlay served for %r (seq=%d)", cam_key, seq)
    cached = _live_png_cache.get(cam_key)
    if cached is not None and cached[0] == seq:
        return Response(content=cached[1], media_type="image/png", headers={"Cache-Control": "no-store"})
    result = reader.read_overlay(cam_key)
    if result is None:
        return Response(status_code=204)
    rgba, _ts = result
    t_png = time.perf_counter()
    png = await asyncio.get_event_loop().run_in_executor(None, _png, rgba)
    if logger.isEnabledFor(logging.DEBUG):
        # Size matters as much as time: this payload crosses the operator's link
        # once per frame, and at 237 ms RTT bandwidth becomes the next wall.
        png_ms = (time.perf_counter() - t_png) * 1000.0
        # Encode the candidate on the SAME buffer, so the comparison is against
        # what is actually served rather than a reconstruction. JPEG is not a
        # candidate — it has no alpha, and the client stacks this over the frame.
        alt = ""
        try:
            import io as _io

            from PIL import Image as _Image

            _z = rgba.copy()
            _z[..., :3][rgba[..., 3] == 0] = 0
            _b = _io.BytesIO()
            _t = time.perf_counter()
            _Image.fromarray(_z).save(_b, format="WEBP", quality=80, method=2)
            _ms = (time.perf_counter() - _t) * 1000.0
            _n = len(_b.getvalue())
            alt = f" | webp q80 {_ms:.1f}ms -> {_n / 1024.0:.0f} KB ({len(png) / max(_n, 1):.1f}x)"
        except Exception as exc:  # measurement must never break serving
            alt = f" | webp probe failed: {type(exc).__name__}"
        logger.debug(
            "overlay %s seq=%d: alpha>0 %.1f%% · png encode %.1fms -> %.0f KB (%dx%d)%s",
            cam_key,
            seq,
            float((rgba[..., 3] > 0).mean()) * 100.0,
            png_ms,
            len(png) / 1024.0,
            rgba.shape[1],
            rgba.shape[0],
            alt,
        )
    _live_png_cache[cam_key] = (seq, png)
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.get("/live/frame/{cam_key}")
async def live_frame(cam_key: str) -> Response:
    """Latest RGBA overlay for a camera as PNG (run tab)."""
    return await _serve_overlay(cam_key)
