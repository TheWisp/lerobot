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
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from lerobot.gui.gpu_slot import SLOT
from lerobot.overlays.overlay_state import Event, OverlayStateMachine, State

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
        same_worker_reusable = (
            _live_model == req.model
            and _live_resolution == req.resolution
            and _live_proc is not None
            and _live_proc.returncode is None
        )
        if same_worker_reusable and _data_worker_dims != cameras:
            await _teardown_current()
        await _spawn_worker(
            req.model,
            objects=req.objects,
            background_treatment=req.background_treatment or {"key": "random", "params": {}},
            resolution=req.resolution,
            multi_instance=req.multi_instance,
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
        "owner": is_owner,
        "busy": busy,
        "holder": SLOT.holder(now).label if busy else None,
    }


@router.post("/data/cancel")
async def data_cancel(x_overlay_session: str | None = Header(default=None)) -> dict:
    """Turn the data overlay off — release the slot, stop the obs-stream publisher, tear the worker
    down. Only the HOLDER tears down the shared worker; another activity's cancel is a no-op so it
    can't kill the holder's overlay."""
    key = _data_key(x_overlay_session)
    now = time.time()
    if SLOT.blocks(key, now):  # someone else holds the slot — don't touch their overlay
        return {"ok": True, "note": "not the holder; nothing torn down"}
    SLOT.release(key)
    stop_data_publisher()
    await _stop_live()
    return {"ok": True}


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
        publish_data_frame(req.dataset_id, req.episode, req.frame, ds[start + req.frame])

    await asyncio.get_event_loop().run_in_executor(None, _decode_and_publish)
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
# The camera shape map ({cam: (h, w)}) the LIVE worker attached to. The worker's obs
# mapping and its overlay buffers are sized by this, so it — not the dataset's identity —
# is what decides whether a dataset switch needs a respawn.
_data_worker_dims: dict[str, tuple[int, int]] | None = None
_data_pub_generation = 0  # bumped on a new stream (jump / episode / wrap) -> the worker resets


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


def publish_data_frame(dataset_id: str, episode: int, frame: int, item: dict) -> None:
    """Publish the decoded frame to the obs stream + bump ``generation`` on a new stream (scrub jump
    / episode change / wrap) so the worker drops its stale tracker and reseeds. A re-published *same*
    frame (a paused playhead, or the status poll re-sending the current frame) is a no-op: republishing
    it would reset the tracker and re-run the detector on a frame already done — the data path's old
    ~3fps. No-op unless the publisher is active for this dataset. The caller passes the already-decoded
    dataset ``item`` so there is no extra decode."""
    global _data_pub_last_pos, _data_pub_generation
    if _data_pub is None or _data_pub_dataset != dataset_id:
        return
    pos = (int(episode), int(frame))
    if pos == _data_pub_last_pos:
        return  # same frame already published — let the worker keep tracking, don't reset + re-infer
    # Plain playback advances one frame at a time; that +1 step is continuous, so the worker's tracker
    # just propagates. Any other move — a scrub, an episode change, or the wrap to the loop start — is
    # a new stream: bump generation so the worker resets its per-camera tracking and reseeds.
    sequential = _data_pub_last_pos is not None and pos == (_data_pub_last_pos[0], _data_pub_last_pos[1] + 1)
    if not sequential:
        _data_pub_generation += 1
        _write_data_control()
    _data_pub_last_pos = pos
    obs = {cam: _frame_rgb(item, cam) for cam in _data_pub_cameras if cam in item}
    try:
        _data_pub.write_obs(obs)
    except Exception:
        logger.warning("data obs publish failed for frame %s — the overlay won't update", pos, exc_info=True)


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
    global _live_proc, _live_model, _live_resolution
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
    if m.state is State.INACTIVE:  # self-exit path: nothing is running, clear ownership
        _live_model = None
        _live_resolution = None


async def _teardown_current() -> None:
    """Stop the running standalone. Caller MUST hold _live_lock. Fires STOP -> STOPPED (or RESET
    if it had already crashed); no-op when nothing is running."""
    global _live_proc, _live_model, _live_resolution, _live_stopping
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
    """Lock-wrapped teardown for external callers (server shutdown, /live/stop)."""
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
                    "config": {
                        "objects": objects or [],
                        "background_treatment": background_treatment,
                        "style": style,
                        "smooth": smooth,
                        "method": method,  # read by the POLICY (gradient|rollout), not the worker
                        **({} if multi_instance is None else {"multi_instance": multi_instance}),
                    }
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
    _live_log_path = Path(tempfile.gettempdir()) / "lerobot_overlays.log"
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
        )
    return {"ok": True, "state": m.state.value}


@router.post("/live/control")
async def live_control(body: dict) -> dict:
    """Push a control update (e.g. {"prompt": "green ring . robot arm"}) to the
    running subprocess via the overlay buffer's reverse channel."""
    reader = _get_live_reader()
    if reader is None:
        raise HTTPException(status_code=409, detail="No live overlay producer yet")
    reader.write_control(body)
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
    png = await asyncio.get_event_loop().run_in_executor(None, _png, rgba)
    _live_png_cache[cam_key] = (seq, png)
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.get("/live/frame/{cam_key}")
async def live_frame(cam_key: str) -> Response:
    """Latest RGBA overlay for a camera as PNG (run tab)."""
    return await _serve_overlay(cam_key)
