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
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

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
        "label": "Policy saliency",
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


@router.get("/models")
async def list_models() -> dict:
    """The processing steps the picker offers (besides 'none')."""
    return {"models": _STEPS}


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
    objects: list[dict] | None = None  # [{name, color:[r,g,b], sign:'+'/'-'}]
    background: dict | None = None  # {color: [r,g,b] | null}
    cameras: list[str] | None = None  # active subset — worker infers + we publish only these; None = all


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
    """Encode an HxWx4 RGBA overlay to PNG (preserves transparency)."""
    from PIL import Image

    assert rgba.ndim == 3 and rgba.shape[2] == 4, f"expected HxWx4 RGBA, got {rgba.shape}"
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


@router.post("/data/configure")
async def data_configure(req: ConfigureRequest) -> dict:
    """Turn on the data overlay: publish the scrubbed frames to the obs stream (the GUI is the
    writer) and spawn the worker, which reads that stream exactly like the run path. Refuses (409)
    if a run already owns the stream — one writer at a time. A re-configure (same dataset/model)
    just refreshes the step config."""
    if _app_state is None or req.dataset_id not in _app_state.datasets:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {req.dataset_id}")
    if req.model not in {s["key"] for s in _STEPS}:
        raise HTTPException(status_code=400, detail=f"Unknown overlay model: {req.model}")
    ds = _app_state.datasets[req.dataset_id]
    cameras = _dataset_camera_dims(ds)
    if not cameras:
        raise HTTPException(status_code=400, detail="Dataset has no camera/image keys")
    config = {"objects": req.objects or [], "background": req.background}
    prev_dataset = _data_pub_dataset  # capture before start_data_publisher updates it
    if not start_data_publisher(req.dataset_id, cameras, config):
        raise HTTPException(
            status_code=409,
            detail="A run owns the observation stream; stop the run to use the data overlay.",
        )
    m = _machine(req.model)
    async with _live_lock:
        # A dataset switch recreates the obs stream (cameras/dims may differ), so the worker must
        # RESPAWN to rebuild its overlay buffer — a same-model control push wouldn't pick up the
        # new shape. Same dataset: _spawn_worker just pushes control (no restart).
        if prev_dataset is not None and prev_dataset != req.dataset_id:
            await _teardown_current()
        await _spawn_worker(req.model, objects=req.objects, background=req.background)
    # Narrow the worker to the panel's selected cameras so disabling one actually cuts its work:
    # publish only those + filter inference to them (None/absent = keep the default = all cameras).
    global _data_pub_cameras
    if req.cameras is not None:
        _data_pub_cameras = req.cameras
        _write_data_control()
    return {"ok": True, "state": m.state.value}


@router.get("/data/status")
async def data_status() -> dict:
    """Badge/state for the data overlay — the worker's lifecycle machine + fps/util/vram (the
    worker is identical to the run path). ``publishing`` reflects the obs-stream writer."""
    _observe()
    target = _live_model
    machine = _machines.get(target) if target else None
    state = machine.state if machine is not None else State.INACTIVE
    running = target is not None and _live_proc is not None and _live_proc.returncode is None
    st = _read_status() if running else {}
    reader = _get_live_reader() if running else None
    return {
        "state": state.value,
        "available": state is State.ACTIVE,
        "model": target,
        "cameras": list(reader.cameras) if reader is not None else [],
        "fps": float(st.get("fps", 0.0)),
        "vram": float(st.get("vram", 0.0)),
        "util": _proc_sm(_live_proc.pid) if running else 0,
        "publishing": _data_publisher_active(),
    }


@router.post("/data/cancel")
async def data_cancel() -> dict:
    """Turn the data overlay off — stop the obs-stream publisher (frees it for a run) and tear the
    worker down."""
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
async def data_publish(req: DataPublishRequest) -> Response:
    """Frontend calls this on every frame change: decode the landed frame (all cameras) and publish
    it to the obs stream so the worker overlays it. The decode runs off the event loop. No-op unless
    a data publisher is active for this dataset."""
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
    """Poll-side event source: fire LOADED / CRASH for the running standalone from its reported
    phase + process liveness. Skips while a commanded teardown owns the transitions."""
    global _live_proc
    if _live_stopping or _live_model is None or _live_proc is None:
        return
    m = _machine(_live_model)
    rc = _live_proc.returncode
    if rc is None:  # alive
        if m.state is State.LOADING and _read_status().get("phase") == "active":
            m.fire(Event.LOADED)
    elif rc not in (0, -15):  # died abnormally, NOT via our teardown
        m.fire(Event.CRASH)
        _close_live_reader()
        _close_status_reader()
        _live_proc = None  # reaped; the machine holds ERROR until the next start/stop


async def _teardown_current() -> None:
    """Stop the running standalone. Caller MUST hold _live_lock. Fires STOP -> STOPPED (or RESET
    if it had already crashed); no-op when nothing is running."""
    global _live_proc, _live_model, _live_stopping
    if _live_model is None:
        return
    m = _machine(_live_model)
    if m.state is State.ERROR:  # already dead — just clear the error, nothing to terminate
        m.fire(Event.RESET)
        _close_live_reader()
        _close_status_reader()
        _live_proc = None
        _live_model = None
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
        _live_stopping = False


async def _stop_live() -> None:
    """Lock-wrapped teardown for external callers (server shutdown, /live/stop)."""
    async with _live_lock:
        await _teardown_current()


class LiveStartRequest(BaseModel):
    model: str
    objects: list[dict] | None = None
    background: dict | None = None
    cameras: list[str] | None = None
    style: str | None = None  # policy_saliency render style (see PolicySaliencyAdapter.STYLES)
    smooth: float | None = None  # policy_saliency smoothing sigma (0 = raw 64x64)
    method: str | None = None  # policy_saliency source: "gradient" | "rollout" (read by the policy)


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
    model: str, *, objects=None, background=None, cameras=None, style=None, smooth=None, method=None
) -> None:
    """Spawn (or push control to) the single overlay worker for ``model``. Caller MUST hold
    ``_live_lock``. The worker is identical for live + data — it reads the obs stream; only the
    publisher differs (teleop for run, the GUI data publisher for data). A same-model call just
    pushes control; a different model tears the old worker down first."""
    global _live_proc, _live_model, _live_log_path, _live_log_file
    m = _machine(model)
    if _live_model == model and _live_proc is not None and _live_proc.returncode is None:
        reader = _get_live_reader()  # already up — push control, don't restart
        if reader is not None:
            reader.write_control(
                {
                    "config": {
                        "objects": objects or [],
                        "background": background,
                        "style": style,
                        "smooth": smooth,
                        "method": method,  # read by the POLICY (gradient|rollout), not the worker
                    }
                }
            )
        return
    m.fire(Event.START)  # -> loading; the badge reflects it immediately
    await _teardown_current()  # stop a different running model first (serialised)
    args = [sys.executable, "-u", "-m", "lerobot.overlays.standalone", f"--model={model}"]
    if objects:
        args.append(f"--objects={json.dumps(objects)}")
    if background is not None:
        args.append(f"--background={json.dumps(background)}")
    if style:
        args.append(f"--style={style}")
    if smooth is not None:
        args.append(f"--smooth={smooth}")
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

    _live_proc = await asyncio.create_subprocess_exec(
        *args, stdout=logf, stderr=asyncio.subprocess.STDOUT, preexec_fn=_set_pdeathsig_preexec
    )
    _live_model = model
    if method:
        # The method is read by the POLICY from the control block (the worker doesn't use it). A fresh
        # spawn's buffer isn't up instantly, so push it once it exists — else a method chosen BEFORE
        # start is silently dropped (only a mid-run /live/control would set it).
        for _ in range(20):
            await asyncio.sleep(0.3)
            r = _get_live_reader()
            if r is not None:
                r.write_control({"config": {"method": method}})
                break


@router.post("/live/start")
async def live_start(req: LiveStartRequest) -> dict:
    """Launch the worker for a model (run tab) — fires START on that model's machine. Serialised
    with stops by _live_lock. Teleop/record/policy publishes the obs stream the worker reads."""
    if req.model not in {s["key"] for s in _STEPS}:
        raise HTTPException(status_code=400, detail=f"Unknown overlay model: {req.model}")
    m = _machine(req.model)
    async with _live_lock:
        await _spawn_worker(
            req.model,
            objects=req.objects,
            background=req.background,
            cameras=req.cameras,
            style=req.style,
            smooth=req.smooth,
            method=req.method,
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
    await _stop_live()
    return {"ok": True}


@router.get("/live/status")
async def live_status(model: str | None = None) -> dict:
    """The lifecycle-machine state for `model` (default: the running one), plus live fps/util/
    vram when that model is the running, active one. The state is the machine's — never a string
    assembled here. States: inactive / loading / active / stopping / error."""
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
    return {
        "state": state.value,
        "available": state is State.ACTIVE,
        "model": target,
        "cameras": list(reader.cameras) if reader is not None else [],
        "fps": float(st.get("fps", 0.0)),
        "vram": float(st.get("vram", 0.0)),
        "util": _proc_sm(_live_proc.pid) if running else 0,
    }


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
