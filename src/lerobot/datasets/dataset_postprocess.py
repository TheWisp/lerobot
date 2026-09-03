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
"""Offline visual data editing: segment the task-relevant objects, then transform
the rest of the frame, and write the result as a NEW LeRobotDataset.

This is camera-side domain randomization for imitation-learning data (the
GreenAug / RoboEngine recipe): the user's segmented objects + everything they
mark are the **protected foreground**; an effect rewrites the **background**
(or, for global effects, the whole frame) of every frame. Only the camera
pixels change — actions, states, tasks, and timing are copied verbatim, so the
edited dataset is trained on exactly like the original.

Segmentation reuses the live overlay's SAM3 tracker (``lerobot.overlays``), so
"what counts as foreground" matches what the user already previewed in the data
tab. Randomized effects are sampled ONCE per episode by default (per-frame
flicker destroys the motion cues a policy learns from); see ``ApplyMode``.

No GUI/IPC here — that lives in :mod:`lerobot.gui.process_worker`. This module
is a pure dataset transform, a peer of :mod:`lerobot.datasets.dataset_tools`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.mask_compositing import mask_keys_for
from lerobot.overlays.effects import (
    TREATMENTS,
    TREATMENTS_BY_KEY as _TREATMENTS_BY_KEY,
    build_and_sample_regions,
    composite_regions,
)
from lerobot.utils.constants import DEFAULT_FEATURES, HF_LEROBOT_HOME

logger = logging.getLogger(__name__)

#: Prefix every camera feature key carries.
OBS_STR = "observation.images."

# When to re-sample a randomized effect's parameters. Per-episode is the
# default and the right answer for trajectory data: a fixed look per episode
# preserves temporal/motion cues, whereas per-frame resampling makes the
# background flicker and corrupts the dynamics the policy learns. Additive
# sensor noise is the one effect where per-frame is physically correct.
ApplyMode = Literal["per_episode", "per_frame", "static"]

# Frames a worker may run ahead of the writer, per in-flight episode. Bounds the
# memory cost of parallelism to a few frames rather than a whole episode.
_PARALLEL_QUEUE_FRAMES = 24
# Episode-level parallelism, proven correct (byte-identical output) and ON for long
# jobs. Measured on real footage with real SAM3, 1720 frames, model load discounted
# (negligible on an hours-long dataset): 70.0 s -> 63.0 s, i.e. 1.11x.
#
# It does NOT come from filling an idle GPU: mean utilization is 54% with one worker
# and 55% with two. Whatever leaves the card idle 46% of the time is not something a
# second CUDA stream can fill, so adding workers is not the lever for utilization.
#
# Each worker costs a model load (~6 s) and its own ~3 GB VRAM copy, so a short job
# would lose more than it gains: below this many frames the job stays serial.
_PARALLEL_MIN_FRAMES = 2000


# The treatment registry + per-region composite live in the shared, dependency-free
# lerobot.overlays.effects module, so the live overlay worker renders exactly what
# this batch pass commits (preview == commit).
__all__ = ["TREATMENTS", "composite_regions", "process_dataset", "ProcessResult", "ApplyMode"]


# ── Frame I/O helpers ────────────────────────────────────────────────────────


def _to_rgb_uint8(t) -> np.ndarray:
    """A decoded dataset camera tensor -> contiguous HxWx3 uint8 RGB (what the
    SAM adapter expects and what add_frame stores for image/video features)."""
    import torch

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


# ── Main entry point ─────────────────────────────────────────────────────────


def _copyable_feature_keys(features: dict) -> list[str]:
    """The feature keys ``process_dataset`` copies/creates on the output — everything
    except the special fields. ``task`` is special (stored as ``task_index`` + a
    ``frame["task"]`` string, and stripped by ``validate_frame`` before feature
    validation), so it must be excluded **even when a source materialises it as a
    regular feature** (some merged/raw datasets do). Otherwise the output would list
    ``task`` as a normal feature that ``validate_frame`` then reports as "missing" on
    every frame."""
    special = set(DEFAULT_FEATURES) | {"task"}
    return [k for k in features if k not in special]


@dataclass
class ProcessResult:
    out_root: Path
    out_repo_id: str
    episodes_written: int
    frames_written: int
    cancelled: bool = False


def _job_rng(seed: int, variant: int, ep: int):
    """A generator derived from (seed, variant, episode) rather than drawn from one
    shared stream. Draws then depend only on WHICH episode this is, never on the order
    episodes happen to run in — which is what lets a parallel run produce byte-identical
    output to a serial one."""
    return np.random.default_rng([seed, variant, ep])


def _process_one_episode(
    *,
    src,
    adapter,
    ep: int,
    start: int,
    length: int,
    cam_keys,
    feature_keys,
    edit_cams,
    obj_treatment_by_name,
    background_treatment,
    apply_mode: str,
    static_cache: dict,
    rng,
    emit,
    cancelled_flag,
) -> int:
    """Segment + composite one episode, calling ``emit(frame)`` per frame IN ORDER.

    Pre: ``adapter`` is exclusively this call's — tracking state is per-episode, so a
    shared adapter cannot serve two episodes concurrently. Post: returns the number of
    frames emitted, short of ``length`` only when cancelled.
    """
    # New tracker session per (camera, episode): each episode is an independent video
    # stream, so reseed rather than propagate.
    for cam in cam_keys:
        adapter.set_camera(cam)
        adapter.reset()
    ep_cache: dict[str, dict] = {}  # per-camera region cache for per_episode mode
    emitted = 0
    for f in range(length):
        if cancelled_flag():
            break
        item = src[start + f]
        # One segmentation pass for the timestep across all edited cameras — the same
        # segment_many the live preview runs, so preview == commit.
        rgb_by_cam = {k: _to_rgb_uint8(item[k]) for k in feature_keys if k in edit_cams}
        if hasattr(adapter, "segment_many"):
            masks_by_cam = adapter.segment_many(rgb_by_cam)
        else:  # minimal duck-typed adapters (tests) only implement segment()
            masks_by_cam = {}
            for k, rgb in rgb_by_cam.items():
                adapter.set_camera(k)
                masks_by_cam[k] = adapter.segment(rgb)
        frame: dict[str, Any] = {}
        for k in feature_keys:
            if k in edit_cams:
                rgb = rgb_by_cam[k]
                h, w = rgb.shape[:2]
                masks_by_name = masks_by_cam[k]
                # Reuse randomized draws at the ApplyMode cadence (per (cam, region)).
                if apply_mode == "per_frame":
                    cache: dict = {}
                elif apply_mode == "static":
                    cache = static_cache.setdefault(k, {})
                else:  # per_episode
                    cache = ep_cache.setdefault(k, {})
                regions, sampled = build_and_sample_regions(
                    masks_by_name, obj_treatment_by_name, background_treatment, h, w, rng, cache
                )
                frame[k] = composite_regions(rgb, regions, sampled)
            elif k in src.meta.camera_keys:
                frame[k] = _to_rgb_uint8(item[k])  # a camera the user excluded
            else:
                frame[k] = item[k]
        frame["task"] = item["task"]
        emit(frame)
        emitted += 1
    return emitted


def process_dataset(
    src: LeRobotDataset,
    *,
    out_repo_id: str,
    objects: list[dict],
    background_treatment: dict | None = None,
    apply_mode: ApplyMode = "per_episode",
    variants: int = 1,
    multi_instance: bool = True,
    cameras: list[str] | None = None,
    episodes: list[int] | None = None,
    out_root: str | Path | None = None,
    device: str = "cuda",
    model: str = "sam3_track",
    resolution: int | None = None,
    seed: int = 0,
    parallel_episodes: int = 2,
    adapter: Any = None,
    progress: Callable[[dict], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> ProcessResult:
    """Segment ``objects`` in every frame of ``src`` and write an edited copy.

    Per-region model: each ``object`` carries a ``treatment`` (``{key, params}``)
    applied to its own mask, and ``background_treatment`` applies to everything else.
    A ``none`` treatment leaves that region's real pixels — so the default (objects
    ``none``, background ``random``) is the GreenAug recipe. This is the exact same
    composite the live overlay renders (see :func:`composite_regions`).

    Pre: ``src`` is a readable LeRobotDataset; ``objects`` is the overlay object list
    (``[{name, sign, treatment}]``); every treatment ``key`` is in :data:`TREATMENTS`.
    ``model`` is a segmenter key and ``resolution`` (when set) one of its presets —
    both should match the live preview that tuned this run (preview == commit).
    ``cameras``/``episodes`` default to all. ``variants`` > 1 writes that many
    independently-randomized copies of each source episode.

    Post: a new dataset is written under ``out_root`` (default
    ``$HF_LEROBOT_HOME/out_repo_id``) with identical features and per-frame
    non-camera data; only camera pixels are transformed. Returns a
    :class:`ProcessResult`. If ``should_cancel`` flips True mid-run the partial
    dataset is finalized and ``cancelled=True`` is returned.

    ``progress`` is called with ``{stage, frames_done, frames_total,
    episodes_done, episodes_total, current_episode}`` roughly per frame.
    """
    background_treatment = background_treatment or {"key": "random", "params": {}}
    obj_treatment_by_name = {
        str(o.get("name", "")).strip(): (o.get("treatment") or {"key": "none"})
        for o in objects
        if str(o.get("name", "")).strip()
    }
    for tr in [background_treatment, *obj_treatment_by_name.values()]:
        key = (tr or {}).get("key") or "none"
        if key not in _TREATMENTS_BY_KEY and key != "none":
            raise ValueError(f"unknown treatment {key!r}; have {list(_TREATMENTS_BY_KEY)}")
    cancelled_flag = should_cancel or (lambda: False)

    cam_keys = list(src.meta.camera_keys)
    if cameras:
        cam_keys = [c for c in cam_keys if c in set(cameras)]
    if not cam_keys:
        raise ValueError("no camera keys selected to process")
    edit_cams = set(cam_keys)

    if episodes is None:
        episodes = list(range(src.meta.total_episodes))

    out_root = Path(out_root) if out_root is not None else HF_LEROBOT_HOME / out_repo_id
    feature_keys = _copyable_feature_keys(src.meta.features)
    create_features = {k: src.meta.features[k] for k in feature_keys}

    def _emit(stage: str, fd: int, ft: int, ed: int, et: int, cur: int | None) -> None:
        if progress is not None:
            progress(
                {
                    "stage": stage,
                    "frames_done": fd,
                    "frames_total": ft,
                    "episodes_done": ed,
                    "episodes_total": et,
                    "current_episode": cur,
                }
            )

    ep_lengths = {ep: int(src.meta.episodes["length"][ep]) for ep in episodes}
    frames_total = variants * sum(ep_lengths.values())
    episodes_total = variants * len(episodes)
    _emit("loading model", 0, frames_total, 0, episodes_total, None)

    injected_adapter = adapter is not None
    if adapter is None:
        from lerobot.overlays.adapters import build_adapter

        # resolution matches the live preview's (preview == commit includes resolution).
        adapter = build_adapter(model, device=device, resolution=resolution)
    adapter.set_control({"objects": objects, "multi_instance": multi_instance})

    out = LeRobotDataset.create(
        repo_id=out_repo_id,
        fps=src.meta.fps,
        features=create_features,
        root=out_root,
        robot_type=src.meta.robot_type,
        use_videos=len(src.meta.video_keys) > 0,
    )

    # Randomized-treatment draws, memoized per (camera → per-region) at the ApplyMode
    # cadence: `static_cache` persists across episodes; a fresh `ep_cache` per episode
    # gives the default per-episode coherence; per_frame passes a throwaway cache.
    static_cache: dict[str, dict] = {}
    frames_done = 0
    episodes_done = 0
    cancelled = False

    # Episodes are independent (each reseeds its tracker), so they can run
    # concurrently — the ONE unit of parallelism this pipeline allows, since frames
    # within an episode depend on the previous frame's tracker state. Each worker owns
    # its own adapter (tracking state is not shareable) and still runs batch-1
    # inference, so the masks are bit-identical to a serial run; only the ORDER of
    # execution changes, and per-job RNG makes even the random draws order-independent.
    jobs = [(v, ep) for v in range(variants) for ep in episodes]
    n_workers = max(1, min(int(parallel_episodes), len(jobs)))
    if frames_total < _PARALLEL_MIN_FRAMES:
        n_workers = 1  # too short to repay the extra model load
    if injected_adapter:
        n_workers = 1  # a caller-supplied adapter cannot be cloned per worker
    if apply_mode == "static":
        n_workers = 1  # one draw shared by the whole run: order would decide who draws

    def _run_serial() -> tuple[int, int, bool]:
        fd = ed = 0
        for v, ep in jobs:
            if cancelled_flag():
                return fd, ed, True
            n = _process_one_episode(
                src=src,
                adapter=adapter,
                ep=ep,
                start=int(src.meta.episodes["dataset_from_index"][ep]),
                length=ep_lengths[ep],
                cam_keys=cam_keys,
                feature_keys=feature_keys,
                edit_cams=edit_cams,
                obj_treatment_by_name=obj_treatment_by_name,
                background_treatment=background_treatment,
                apply_mode=apply_mode,
                static_cache=static_cache,
                rng=_job_rng(seed, v, ep),
                emit=out.add_frame,
                cancelled_flag=cancelled_flag,
            )
            fd += n
            if n < ep_lengths[ep]:
                return fd, ed, True
            out.save_episode()
            ed += 1
            _emit("processing", fd, frames_total, ed, episodes_total, ep)
        return fd, ed, False

    def _run_parallel(workers: int) -> tuple[int, int, bool]:
        import queue as _queue
        import threading

        from lerobot.overlays.adapters import build_adapter

        sentinel = object()
        queues = [_queue.Queue(maxsize=_PARALLEL_QUEUE_FRAMES) for _ in jobs]
        cursor, lock = [0], threading.Lock()
        abort, errors = threading.Event(), []

        def take() -> int | None:
            with lock:
                if cursor[0] >= len(jobs):
                    return None
                cursor[0] += 1
                return cursor[0] - 1

        def put(q, item) -> None:
            while not abort.is_set():  # bounded queue: block, but stay abortable
                try:
                    q.put(item, timeout=0.2)
                    return
                except _queue.Full:
                    continue

        def work(ad) -> None:
            try:
                while not abort.is_set():
                    idx = take()
                    if idx is None:
                        return
                    v, ep = jobs[idx]
                    _process_one_episode(
                        src=src,
                        adapter=ad,
                        ep=ep,
                        start=int(src.meta.episodes["dataset_from_index"][ep]),
                        length=ep_lengths[ep],
                        cam_keys=cam_keys,
                        feature_keys=feature_keys,
                        edit_cams=edit_cams,
                        obj_treatment_by_name=obj_treatment_by_name,
                        background_treatment=background_treatment,
                        apply_mode=apply_mode,
                        static_cache=static_cache,
                        rng=_job_rng(seed, v, ep),
                        emit=lambda frame, q=queues[idx]: put(q, frame),
                        cancelled_flag=lambda: abort.is_set() or cancelled_flag(),
                    )
                    put(queues[idx], sentinel)
            except BaseException as exc:  # a dead worker must not hang the writer
                errors.append(exc)
                abort.set()

        adapters = [adapter] + [
            build_adapter(model, device=device, resolution=resolution) for _ in range(workers - 1)
        ]
        for ad in adapters[1:]:
            ad.set_control({"objects": objects, "multi_instance": multi_instance})
        threads = [
            threading.Thread(target=work, args=(ad,), name=f"process-ep-{i}", daemon=True)
            for i, ad in enumerate(adapters)
        ]
        fd = ed = 0
        stopped = False
        for t in threads:
            t.start()
        try:
            # The writer consumes jobs IN ORDER, so the output dataset is byte-identical
            # to a serial run no matter which worker finished first.
            for idx, (_v, ep) in enumerate(jobs):
                got = 0
                while True:
                    try:
                        item = queues[idx].get(timeout=0.2)
                    except _queue.Empty:
                        if abort.is_set():
                            break
                        continue
                    if item is sentinel:
                        break
                    out.add_frame(item)
                    got += 1
                    fd += 1
                    if fd % 10 == 0:
                        _emit("processing", fd, frames_total, ed, episodes_total, ep)
                if got < ep_lengths[ep] or cancelled_flag() or errors:
                    stopped = True
                    break
                out.save_episode()
                ed += 1
                _emit("processing", fd, frames_total, ed, episodes_total, ep)
        finally:
            abort.set()
            for t in threads:
                t.join(timeout=15)
        if errors:
            raise errors[0]
        return fd, ed, stopped

    try:
        if n_workers > 1:
            logger.info("post-process: %d episodes in parallel", n_workers)
            frames_done, episodes_done, cancelled = _run_parallel(n_workers)
        else:
            frames_done, episodes_done, cancelled = _run_serial()
        _emit("finalizing", frames_done, frames_total, episodes_done, episodes_total, None)
    finally:
        if out.has_pending_frames():
            out.clear_episode_buffer()
        out.finalize()

    logger.info(
        "post-process done: %d episodes / %d frames -> %s%s",
        episodes_done,
        frames_done,
        out_root,
        " (cancelled)" if cancelled else "",
    )
    return ProcessResult(
        out_root=Path(out_root),
        out_repo_id=out_repo_id,
        episodes_written=episodes_done,
        frames_written=frames_done,
        cancelled=cancelled,
    )


def split_stereo_cameras(
    src: LeRobotDataset,
    *,
    out_repo_id: str,
    cameras: list[str],
    out_root: str | Path | None = None,
    episodes: list[int] | None = None,
    passthrough: bool = False,
    progress: Callable[[dict], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> ProcessResult:
    """Split side-by-side stereo cameras into one channel per eye.

    Side-by-side stereo devices (the ZED family) have no single-eye capture
    mode, so a dataset recorded through the plain UVC path stores both eyes
    concatenated in one frame. Consumed whole it carries the scene twice and
    halves the horizontal resolution surviving into a policy's square encoder
    input — a ball ~60 px across lands on fewer pixels than one ViT patch.

    Each named camera's key is REPLACED by two keys at half the width, named by
    :func:`~lerobot.cameras.stereo.stereo_channel_keys` — the same convention
    the live camera publishes, so a converted dataset and a rollout agree on
    which half is which. Both eyes are kept: which one a model consumes is a
    training-config choice (``--cameras``), not a decision frozen into data
    that cannot be re-recorded.

    ``passthrough`` carries the cameras that are NOT being split through by
    hardlink instead of re-encoding them: bit-identical, no disk, and half the
    encoding work. It is off by default because it preserves each carried
    camera's original codec, so a source that mixes codecs stays mixed — and one
    codec throughout is usually worth more than the few dB the re-encode costs.
    It is also only valid for a whole-dataset conversion, since the copied video
    metadata indexes the source's episode numbering, so passing ``episodes``
    disables it regardless.

    Pre: ``src`` is readable; every entry of ``cameras`` is one of its camera
    keys, named bare (``top``) or fully (``observation.images.top``); each has
    an even width.

    Post: a new dataset under ``out_root`` (default ``$HF_LEROBOT_HOME/
    out_repo_id``) whose stereo keys are gone and replaced by their eyes at
    ``(height, width // 2, 3)``. Non-stereo cameras and every non-camera field
    are copied verbatim; per-episode stats are recomputed by ``save_episode``,
    so no statistic for a removed key survives. If ``should_cancel`` flips True
    the partial dataset is finalized and ``cancelled=True`` returned.
    """
    from lerobot.cameras.stereo import split_stereo_frame, stereo_channel_keys

    cancelled_flag = should_cancel or (lambda: False)

    wanted = [c if c.startswith(OBS_STR) else f"{OBS_STR}{c}" for c in cameras]
    unknown = [c for c in wanted if c not in src.meta.camera_keys]
    if unknown:
        raise ValueError(f"not cameras of this dataset: {unknown}; have {list(src.meta.camera_keys)}")
    if not wanted:
        raise ValueError("no cameras selected to split")
    stereo = set(wanted)

    # Subset conversions cannot carry source video metadata; see the docstring.
    carried: set[str] = set()
    if passthrough and episodes is None and len(src.meta.video_keys) > 0:
        # video_keys, not camera_keys: hardlinking moves video files, so an
        # image-dtype camera listed here would be dropped from the writer and
        # then not carried either, vanishing from the output.
        carried = {k for k in src.meta.video_keys if k not in stereo}
    elif passthrough and episodes is not None:
        logger.info("passthrough disabled: converting a subset of episodes")

    feature_keys = _copyable_feature_keys(src.meta.features)
    create_features: dict[str, Any] = {}
    for k in feature_keys:
        feat = src.meta.features[k]
        if k in carried:
            continue  # hardlinked in after the writer finishes
        if k not in stereo:
            create_features[k] = feat
            continue
        h, w, c = (int(x) for x in feat["shape"])
        if w % 2:
            raise ValueError(f"{k} has odd width {w}; it cannot be a side-by-side pair")
        for eye_key in stereo_channel_keys(k):
            create_features[eye_key] = {**feat, "shape": (h, w // 2, c)}
            # Encoder metadata describes the source file, which is about to be
            # re-encoded at a different width; save_episode rewrites it.
            create_features[eye_key].pop("info", None)

    if episodes is None:
        episodes = list(range(src.meta.total_episodes))
    out_root = Path(out_root) if out_root is not None else HF_LEROBOT_HOME / out_repo_id
    ep_lengths = {ep: int(src.meta.episodes["length"][ep]) for ep in episodes}
    frames_total = sum(ep_lengths.values())

    logger.info(
        "Splitting %s -> %s in %d episodes (%d frames)",
        sorted(stereo),
        sorted(k for k in create_features if k not in src.meta.features),
        len(episodes),
        frames_total,
    )

    out = LeRobotDataset.create(
        repo_id=out_repo_id,
        fps=src.meta.fps,
        features=create_features,
        root=out_root,
        robot_type=src.meta.robot_type,
        use_videos=len(src.meta.video_keys) > 0,
    )

    frames_done = 0
    episodes_done = 0
    cancelled = False
    for ep in episodes:
        if cancelled_flag():
            cancelled = True
            break
        start = int(src.meta.episodes["dataset_from_index"][ep])
        for f in range(ep_lengths[ep]):
            if cancelled_flag():
                cancelled = True
                break
            item = src[start + f]
            frame: dict[str, Any] = {}
            for k in feature_keys:
                if k in stereo:
                    left, right = split_stereo_frame(_to_rgb_uint8(item[k]))
                    left_key, right_key = stereo_channel_keys(k)
                    frame[left_key], frame[right_key] = left, right
                elif k in carried:
                    continue  # not written by the writer; hardlinked afterwards
                elif k in src.meta.camera_keys:
                    frame[k] = _to_rgb_uint8(item[k])
                else:
                    # A 1-element feature decodes to a 0-d tensor, which
                    # validate_frame rejects against its declared (1,) shape.
                    # The quality flags are stored exactly this way.
                    value = item[k]
                    want = tuple(src.meta.features[k].get("shape", ()))
                    if want == (1,) and getattr(value, "ndim", None) == 0:
                        value = value.reshape(1)
                    frame[k] = value
            frame["task"] = item["task"]
            out.add_frame(frame)
            frames_done += 1
            if progress is not None and frames_done % 50 == 0:
                progress(
                    {
                        "stage": "splitting",
                        "frames_done": frames_done,
                        "frames_total": frames_total,
                        "episodes_done": episodes_done,
                        "episodes_total": len(episodes),
                        "current_episode": ep,
                    }
                )
        if cancelled:
            break
        out.save_episode()
        episodes_done += 1

    # Episode metadata is buffered; without this the output has no meta/episodes
    # and cannot be reopened. A cancelled run finalizes what it completed.
    #
    # Cancel breaks before save_episode(), so the abandoned episode's frames are
    # still buffered. Clearing them keeps its temp image directories out of the
    # finished dataset; the sibling process_dataset guards the same way.
    if out.has_pending_frames():
        out.clear_episode_buffer()
    out.finalize()

    if carried and not cancelled:
        _carry_videos_through(src, out_root, sorted(carried))
        logger.info("carried %d camera(s) through by hardlink: %s", len(carried), sorted(carried))

    return ProcessResult(
        out_root=out_root,
        out_repo_id=out_repo_id,
        episodes_written=episodes_done,
        frames_written=frames_done,
        cancelled=cancelled,
    )


def _carry_videos_through(src: LeRobotDataset, out_root: Path, keys: list[str]) -> None:
    """Hardlink ``keys``' video files into ``out_root`` and copy their metadata.

    Pre: the output holds every episode of ``src``, in the same order and with
    the same lengths — otherwise the copied chunk/file indices and timestamps,
    which refer to the source's numbering, would address the wrong frames. The
    caller enforces this by only carrying cameras on a whole-dataset conversion.

    Post: ``out_root`` declares each key with the source's feature entry, its
    video files are hardlinks of the source's (same bytes, no extra disk), and
    each episode's four video-locator columns are the source's values.
    """
    import json
    import os

    import pyarrow as pa
    import pyarrow.parquet as pq

    info_path = out_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())

    for key in keys:
        src_dir = src.root / "videos" / key
        dst_dir = out_root / "videos" / key
        for f in sorted(src_dir.rglob("*.mp4")):
            target = dst_dir / f.relative_to(src_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(f, target)  # same bytes, no copy
        info["features"][key] = src.meta.features[key]
    info_path.write_text(json.dumps(info, indent=4))

    # Merge the source's per-episode video locators into the output's episode
    # metadata, matched on episode_index rather than on row order.
    src_cols = [
        f"videos/{k}/{c}"
        for k in keys
        for c in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
    ]
    src_tbl = (
        pq.ParquetDataset([str(p) for p in sorted((src.root / "meta" / "episodes").rglob("*.parquet"))])
        .read(columns=["episode_index", *src_cols])
        .to_pydict()
    )
    by_ep = {int(e): i for i, e in enumerate(src_tbl["episode_index"])}

    for out_file in sorted((out_root / "meta" / "episodes").rglob("*.parquet")):
        tbl = pq.read_table(out_file)
        eps = [int(e) for e in tbl.column("episode_index").to_pylist()]
        missing = [e for e in eps if e not in by_ep]
        assert not missing, f"output episodes absent from source: {missing[:5]}"
        for col in src_cols:
            values = [src_tbl[col][by_ep[e]] for e in eps]
            tbl = tbl.append_column(col, pa.array(values))
        pq.write_table(tbl, out_file)


def _gpu_frame_sources(src: LeRobotDataset, cam_keys: list[str], device: str) -> dict | None:
    """One NVDEC source per camera, or None if the GPU path cannot serve this
    dataset here. Never raises: an unavailable GPU decode is a fallback, not a
    failure of the mask job."""
    # Every exit says which path this pass took. Two of these used to return
    # None in silence, so "did the fill use the GPU?" was answerable only by
    # inference from an unrelated calibration line -- and not at all for the
    # model, which is the 90% of the time.
    if not str(device).startswith("cuda"):
        logger.info("mask pass decode: CPU reader (device is %s, not CUDA)", device)
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            logger.info("mask pass decode: CPU reader (torch reports no CUDA device)")
            return None
        from lerobot.datasets.gpu_data_pipeline import GpuFrameSource

        sources = {cam: GpuFrameSource(src, cam, device=device) for cam in cam_keys}
        logger.info("mask pass decode: GPU NVDEC on %s for %d camera(s)", device, len(cam_keys))
        return sources
    except Exception as e:  # noqa: BLE001 - any failure means "use the CPU read"
        logger.info(
            "GPU decode unavailable for the mask pass (%s: %s); using the CPU read", type(e).__name__, e
        )
        return None


class _MaskFramePrefetch:
    """Decode the episode in chunks, ahead of the tracker.

    Batched NVDEC decode is ~2x the per-frame rate (measured 3.0 against 5.6
    ms/frame), and the tracker is strictly sequential, so decoding a chunk
    while it works through the previous one costs nothing and hides the decode
    almost entirely. Chunks are small because the frames are held as full-size
    device tensors.
    """

    CHUNK = 32

    def __init__(self, sources: dict, start: int, length: int, cam_keys: list[str]):
        self._sources = sources
        self._start = start
        self._length = length
        self._cams = cam_keys
        self._base = -1
        self._chunk: dict = {}

    def frame(self, f: int) -> dict:
        import numpy as _np

        if not (self._base <= f < self._base + self.CHUNK) or not self._chunk:
            self._base = f
            n = min(self.CHUNK, self._length - f)
            idx = _np.arange(self._start + f, self._start + f + n, dtype=_np.int64)
            self._chunk = {cam: self._sources[cam].fetch(idx) for cam in self._cams}
        k = f - self._base
        # DeviceFrame keeps these on the GPU for the tracker and copies to host
        # only if the detector needs them.
        from lerobot.overlays.adapters import DeviceFrame

        return {cam: DeviceFrame(tensor=self._chunk[cam][k]) for cam in self._cams}

    def close(self) -> None:
        self._chunk = {}


def _fill_gaps(stored_value, found: dict, labels: list[str], shape) -> str:
    """One row: everything already stored, plus what this pass found in the gaps.

    Pre: ``stored_value`` is that frame's current cell (possibly ``""``);
    ``found`` maps label -> boolean mask for what the segmenter detected.
    Post: a row carrying every stored label unchanged -- pixels and enabled
    flag both -- plus each found label that the frame did not already carry.

    The enabled flags have to survive this. A disabled mask that came back
    enabled would silently rejoin training, which is the one thing muting
    exists to prevent.
    """
    from lerobot.datasets.mask_codec import decode_frame, encode_frame, frame_states

    stored = frame_states(stored_value, labels) if stored_value else {}
    merged = decode_frame(stored_value, labels, shape, include_disabled=True) if stored_value else {}
    for name, mask in found.items():
        if name not in stored:
            merged[name] = mask
    muted = [n for n, on in stored.items() if not on]
    return encode_frame(merged, labels, disabled=muted)


def generate_episode_masks(
    src: LeRobotDataset,
    *,
    episode: int,
    objects: list[dict],
    cameras: list[str] | None = None,
    model: str = "sam3_track",
    resolution: int | None = None,
    multi_instance: bool = True,
    background_treatment: dict | None = None,
    adopt: bool = False,
    device: str = "cuda",
    adapter: Any = None,
    progress: Callable[[dict], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """Segment ONE episode and store the masks as a frame-aligned feature, in place.

    The inverse trade of :func:`process_dataset`: nothing is baked. Masks land in
    ``masks.<camera>`` as COCO RLE (see ``lerobot.datasets.mask_codec``)
    and the EFFECT OPTIONS — per-label treatments, background treatment, model,
    resolution — are recorded in the feature's metadata. Playback and training
    reproduce the composite from (masks, options); changing a treatment later is
    a metadata edit, not a segmentation re-run, and the source video is never
    rewritten.

    Adoption: the first save on a dataset is a SCHEMA change (a column for every
    selected camera, empty everywhere). It only happens with ``adopt=True`` —
    the caller is expected to have asked the user. Once adopted, saves rewrite
    only this episode's rows via the same global-index edit path every other
    frame edit uses, so trims and deletions keep masks aligned structurally.

    Vocabulary: rows store label IDS against the feature's ``mask_labels``.
    Regenerating one episode with a DIFFERENT label set would silently corrupt
    every other episode's rows, so a mismatch raises instead; changing the
    vocabulary means regenerating the dataset's masks, deliberately.

    Pre: ``src`` is readable and writable; ``episode`` exists; ``objects`` is the
    overlay object list (named entries carry the vocabulary and treatments).
    Post: every frame of the episode has a row for every selected camera (empty
    string = segmented, nothing found); returns per-camera coverage counts.
    """
    from lerobot.datasets.dataset_tools import add_features_inplace
    from lerobot.datasets.feature_value_edits import set_feature_values
    from lerobot.datasets.mask_codec import EMPTY, feature_spec

    # Deduped, first occurrence winning. The vocabulary is POSITIONAL: two ids
    # sharing a name means a stored row can decode to either, the timeline draws
    # two identical lanes, and every by-name lookup silently picks one. Nothing
    # stops an operator naming two object rows the same thing, so the writer has
    # to be the one that refuses to store it twice.
    # Clock the WHOLE call, not just the frame loop: reading the episode's stored
    # rows, building the prefetcher and resetting the tracker all sit outside it,
    # and measured against the job log they are ~2.2 s per episode -- about 18%
    # of a 274-episode run. A summary whose stages only cover the loop reports
    # percentages of the wrong total and hides that entirely.
    t_started = time.perf_counter()
    labels, requested_labels, treatments = _requested_vocabulary(objects)
    cam_keys, mask_key_of = _resolve_cameras(src, cameras)

    cancelled_flag = should_cancel or (lambda: False)
    start = int(src.meta.episodes["dataset_from_index"][episode])
    length = int(src.meta.episodes["length"][episode])

    # ── schema: adopt or validate ────────────────────────────────────────────
    missing = [cam for cam in cam_keys if mask_key_of[cam] not in src.meta.features]
    if missing and not adopt:
        raise ValueError(
            f"masks feature not adopted for {[mask_key_of[c] for c in missing]}; "
            "adoption is a dataset-wide schema change and needs explicit consent (adopt=True)"
        )
    # One vocabulary for the pass, appended to rather than replaced: a stored
    # id keeps its meaning, so an episode can be re-run with an extra object
    # without touching the episodes that never had it.
    #
    # Read from EVERY mask column, not the selected ones. The vocabulary is a
    # dataset-level fact -- the same object seen from three cameras is one
    # label -- and a pass over one camera that consulted only that camera would
    # append a name the others never learn, which is how they drift apart.
    from lerobot.datasets.mask_store import mask_columns

    all_mask_keys = sorted(set(mask_columns(src).values()) | {mask_key_of[c] for c in cam_keys})
    for key in all_mask_keys:
        stored = list(src.meta.features.get(key, {}).get("mask_labels", []))
        if stored and labels[: len(stored)] != stored:
            labels = stored + [name for name in labels if name not in stored]
    # A STORED treatment always wins over one this pass carries.
    #
    # Treatments are edited in one place -- the Inspector's dataset tier -- and
    # a segmentation pass has no opinion about them. But the caller still sends
    # a treatment per object, defaulting to "none", so preferring the caller
    # would silently reset every label it named back to none on the next save:
    # segment again to add one object, lose the effects on all the others.
    #
    # A label the dataset has never seen keeps whatever the caller supplied,
    # which is how a new object arrives untreated.
    for key in all_mask_keys:
        for name, effect in (src.meta.features.get(key, {}).get("mask_treatments") or {}).items():
            treatments[name] = effect

    # And so does a stored BACKGROUND, for exactly the same reason. The panel
    # that calls this has no background control -- it moved to the Inspector
    # with the per-label treatments -- so the value arriving here is a default,
    # not an intent, and letting it win reset a stored `blur` to `none` on the
    # next save of any episode. A dataset that has never stored one keeps what
    # the caller supplied, which is how the first save sets it.
    for key in all_mask_keys:
        stored_bg = (src.meta.features.get(key) or {}).get("mask_background")
        if stored_bg:
            background_treatment = stored_bg
            break

    # prompt -> stored label, for translating what the adapter echoes back.
    # Objects carrying no prompt map their own name, so a dataset written before
    # prompts existed behaves exactly as it did.
    from lerobot.overlays.adapters import prompt_of

    name_of_prompt = {
        prompt_of(o): str(o.get("name", "")).strip() for o in objects if str(o.get("name", "")).strip()
    }
    prompts_by_label = {
        str(o.get("name", "")).strip(): prompt_of(o)
        for o in objects
        if str(o.get("name", "")).strip() and prompt_of(o) != str(o.get("name", "")).strip()
    }

    for cam in cam_keys:
        key = mask_key_of[cam]
        if key in src.meta.features:
            have = list(src.meta.features[key].get("mask_labels", []))
            # **NOT REACHABLE for a single camera.** The loop above already
            # rewrote `labels` to `stored + new`, so this prefix check holds by
            # construction and the refusal below never fires: a reorder is
            # discarded and a rename becomes an append, both without a word to
            # the caller. It can only fire when two cameras carry DIFFERENT
            # stored vocabularies, which the comment above says cannot happen.
            # The invariant it guards -- a stored id never moves -- does hold,
            # via the normalisation rather than via this raise. Pinned in
            # tests/datasets/test_mask_vocabulary.py.
            if labels[: len(have)] != have:
                raise ValueError(
                    f"{key} already carries vocabulary {have}; regenerating one episode with "
                    f"{labels} would move a stored label and corrupt other episodes' rows. "
                    "Adding labels is fine; reordering or removing is not."
                )
    if missing:
        item0 = src[start]
        new_features = {}
        for cam in missing:
            h, w = _to_rgb_uint8(item0[cam]).shape[:2]
            spec = feature_spec(labels, (h, w))
            spec["mask_treatments"] = treatments
            spec["mask_background"] = background_treatment or {"key": "none"}
            if prompts_by_label:
                # Only the labels whose prompt differs from their name: the
                # common case stores nothing, so the spec does not grow for
                # datasets that never sharpen a prompt.
                spec["mask_prompts"] = dict(prompts_by_label)
            spec["mask_model"] = model
            spec["mask_resolution"] = resolution
            spec["mask_multi_instance"] = bool(multi_instance)
            new_features[mask_key_of[cam]] = (EMPTY, spec)
        add_features_inplace(src, new_features, recompute_stats=False)

    # ── segment the episode, the batch pass's own way ────────────────────────
    adapter_was_injected = adapter is not None
    if adapter is None:
        from lerobot.overlays.adapters import build_adapter

        adapter = build_adapter(model, device=device, resolution=resolution)
    adapter.set_control({"objects": objects, "multi_instance": multi_instance})
    # The model is ~90% of this pass, and nothing said where it ran. `injected`
    # matters too: a multi-episode job builds ONE adapter and reuses it, so a
    # per-episode model load would be a real regression and is invisible without
    # this.
    logger.info(
        "mask pass model: %s on %s (%s adapter)",
        model,
        device,
        "reused" if adapter_was_injected else "built here",
    )
    for cam in cam_keys:
        adapter.set_camera(cam)
        adapter.reset()  # one tracker session per (camera, episode)

    rows: dict[str, list[str]] = {cam: [] for cam in cam_keys}
    # What is ALREADY stored for these frames, read once. The write rule below
    # fills only where a label is absent, so every row has to be merged against
    # the one it replaces -- and the enabled flags carried across, or a muted
    # mask would come back enabled and silently rejoin training.
    stored_rows: dict[str, list[str]] = {}
    mask_shape = (0, 0)
    for cam in cam_keys:
        key = mask_key_of[cam]
        ft = src.meta.features.get(key) or {}
        if ft.get("mask_encoding") == "coco_rle":
            mask_shape = tuple(ft.get("mask_size") or (0, 0))
            col = src.hf_dataset[key][start : start + length]
            stored_rows[cam] = [str(c[0] if isinstance(c, (list, tuple)) and c else (c or "")) for c in col]
        else:
            stored_rows[cam] = [""] * length
    # Decode on the GPU when we can. Measured on this pass: CPU decode is 20.4
    # ms/frame against 3.0 batched on NVDEC, out of a ~79 ms frame, and the
    # frames land as device tensors that the adapter preprocesses without a
    # round trip to host memory. Falls back to the dataset's own read whenever
    # the GPU path is unavailable, which keeps this working on machines without
    # one -- the reason two paths exist at all.
    # Nothing to look for? Then do not look. The write rule fills only where a
    # label is ABSENT, and it is applied AFTER the model has run -- so a pass over
    # ground that is already covered used to segment every frame and discard every
    # result, costing exactly as much as the first pass. Measured on a 294-frame
    # two-camera episode: 18.7 s either way.
    #
    # The check is per EPISODE, not per frame, and deliberately so: `sam3_track`
    # is a tracker, and skipping frames inside an episode would break the tracking
    # state that makes the rest of the episode work. Skipping the episode entirely
    # has no such problem -- there is no session to keep.
    gpu_sources = _gpu_frame_sources(src, cam_keys, device)
    prefetch = _MaskFramePrefetch(gpu_sources, start, length, cam_keys) if gpu_sources else None
    # ── incremental flush ────────────────────────────────────────────────────
    # Rows reach disk as the pass goes, not in one write at the end. Two
    # reasons, and the second is a defect this replaces:
    #
    # A whole episode buffered in memory is minutes of segmentation to lose to
    # a crash -- ~3 min for 1,777 frames at ~10 fps on two cameras. About a
    # second's worth caps that at a second.
    #
    # And cancelling used to return here having written NOTHING, so every frame
    # already computed was discarded. That loss is invisible: the tracks show
    # those frames as absent, exactly as if they had never been segmented.
    flush_frames = 50
    flushed = 0
    declared = False
    # Per-stage timing. The pass had none, so a slow fill could not be
    # attributed: from the log alone, 2.5 s between two shard writes reads as
    # write cost, and the write measures 0.28 s on a 47,803-row dataset -- the
    # rest was the next camera's segmentation. Wall-clock deltas between events
    # invite exactly that mistake, so the stages are timed rather than inferred.
    t_stage = {"decode": 0.0, "segment": 0.0, "encode": 0.0, "write": 0.0}

    def _write_mask_feature_info() -> None:
        """Declare the vocabulary and the recipe, once, on the first flush.

        Split by scope. The vocabulary, treatments and background describe the
        DATASET, so they go to every mask column -- writing them only to the
        cameras this pass touched is what let one camera learn a label the
        others never did, and a treatment reach some views and not others. The
        model, resolution and multi-instance flag are provenance of the rows
        this pass WROTE, so they stay on the cameras it wrote.
        """
        dataset_wide = {
            "mask_labels": labels,
            "mask_treatments": treatments,
            "mask_background": background_treatment or {"key": "none"},
            # Recorded on every save, not only on adopt: sharpening a prompt is
            # a re-run of an existing column, which is the whole point of
            # separating the two. Written only when some prompt differs from its
            # label, so a dataset that never sharpens one stores nothing.
            **({"mask_prompts": dict(prompts_by_label)} if prompts_by_label else {}),
        }
        per_camera = {
            "mask_model": model,
            "mask_resolution": resolution,
            "mask_multi_instance": bool(multi_instance),
        }
        written = {mask_key_of[cam] for cam in cam_keys}
        _update_mask_feature_info(
            Path(src.root),
            {key: {**dataset_wide, **(per_camera if key in written else {})} for key in all_mask_keys},
        )

    def _flush(upto: int) -> None:
        """Write rows ``[flushed, upto)`` and, on the first call, the vocabulary."""
        nonlocal flushed, declared
        if upto <= flushed:
            return
        batch = [
            {
                "feature": mask_key_of[cam],
                "from_index": start + f,
                "to_index": start + f + 1,
                "value": rows[cam][f],
            }
            for cam in cam_keys
            for f in range(flushed, upto)
        ]
        if batch:
            _t = time.perf_counter()
            set_feature_values(src, batch, in_place=True)
            t_stage["write"] += time.perf_counter() - _t
        if not declared:
            # A prompt becomes a LABEL on the first flush, not when the run
            # starts. The vocabulary is positional and can never shrink, so a
            # label declared up front would outlive a run cancelled a second
            # later, with no way to take it back.
            _write_mask_feature_info()
            declared = True
        flushed = upto

    if _episode_is_covered(stored_rows, labels, requested_labels, cam_keys, length):
        # Skip the MODEL, not the metadata. A pass carries recipe and provenance
        # with it -- a sharpened prompt, a changed background -- and those are a
        # dataset-scope write that has nothing to do with whether any row moved.
        # Returning before this dropped them silently, which two existing tests
        # caught: the point of the optimisation is that the expensive part does
        # not run, not that the pass stops meaning anything.
        _write_mask_feature_info()
        if prefetch is not None:
            prefetch.close()
        logger.info(
            "episode %d skipped: every requested label already present on all %d frames x %d cam",
            episode,
            length,
            len(cam_keys),
        )
        coverage = {
            mask_key_of[cam]: sum(1 for v in stored_rows[cam] if v not in ("", "[]")) for cam in cam_keys
        }
        return {
            "cancelled": False,
            "frames_done": length,
            "frames_total": length,
            "episode": episode,
            "coverage": coverage,
            "labels": labels,
            "skipped": True,
        }

    for f in range(length):
        if cancelled_flag():
            # The boundary: what was computed before the stop is kept.
            _flush(f)
            if prefetch is not None:
                prefetch.close()
            _log_stage_timing(t_stage, t_started, f, cam_keys, episode, cancelled=True)
            return {"cancelled": True, "frames_done": f, "frames_total": length}
        _t = time.perf_counter()
        if prefetch is not None:
            rgb_by_cam = prefetch.frame(f)
        else:
            item = src[start + f]
            rgb_by_cam = {cam: _to_rgb_uint8(item[cam]) for cam in cam_keys}
        t_stage["decode"] += time.perf_counter() - _t
        _t = time.perf_counter()
        if hasattr(adapter, "segment_many"):
            masks_by_cam = adapter.segment_many(rgb_by_cam)
        else:  # duck-typed test adapters
            masks_by_cam = {}
            for cam, rgb in rgb_by_cam.items():
                adapter.set_camera(cam)
                masks_by_cam[cam] = adapter.segment(rgb)
        t_stage["segment"] += time.perf_counter() - _t
        for cam in cam_keys:
            by_label = {}
            for key, m in (masks_by_cam.get(cam) or {}).items():
                # The adapter echoes the PROMPT it was given; the dataset stores
                # the label. They are the same string until someone sharpens a
                # prompt, and the translation has to happen here because this is
                # the only place that knows the vocabulary.
                name = name_of_prompt.get(key, key)
                if name not in treatments:
                    continue  # e.g. clicked objects outside the vocabulary
                a = np.asarray(m, dtype=np.float32)
                if a.ndim == 3:
                    a = a[..., 0]
                by_label[name] = a > 0.5
            # THE WRITE RULE. Masks are expensive to produce and cheap to
            # delete, so a pass fills gaps and leaves what is already there
            # alone: a (frame, label) is written only where that label is
            # ABSENT. Detected masks keep what they had -- even if this run
            # would have found something different -- and disabled ones stay
            # disabled and stay out of training.
            #
            # Without this a re-run silently replaced hours of segmentation,
            # and muting a bad detection was pointless because the next pass
            # put it straight back. To replace something deliberately: delete
            # it over that range, then run again.
            _t = time.perf_counter()
            rows[cam].append(_fill_gaps(stored_rows[cam][f], by_label, labels, mask_shape))
            t_stage["encode"] += time.perf_counter() - _t
        if (f + 1) % flush_frames == 0:
            _flush(f + 1)
        if progress and (f % 10 == 0 or f == length - 1):
            progress(
                {
                    "stage": "segmenting",
                    "frames_done": f + 1,
                    "frames_total": length,
                    "episodes_total": 1,
                    "episodes_done": 0,
                    "current_episode": episode,
                }
            )

    # ── store: this episode's rows only, by global index ─────────────────────
    if progress:
        progress({"stage": "writing masks", "frames_done": length, "frames_total": length})
    _flush(length)

    _log_stage_timing(t_stage, t_started, length, cam_keys, episode)
    coverage = {mask_key_of[cam]: sum(1 for v in rows[cam] if v not in ("", "[]")) for cam in cam_keys}
    return {
        "cancelled": False,
        "frames_done": length,
        "frames_total": length,
        "episode": episode,
        "coverage": coverage,
        "labels": labels,
    }


def _requested_vocabulary(objects: list[dict]) -> tuple[list[str], list[str], dict]:
    """The names this run is looking for, deduped, with their treatments.

    Pre: ``objects`` is the run's object list; blank names are ignored.
    Post: ``(labels, requested, treatments)``. ``requested`` is a copy taken
    before the caller merges ``labels`` with the stored vocabulary: the skip
    check asks whether THESE are covered, not whether every label the dataset
    has ever seen is, which would almost never be true.

    Repeats collapse, first occurrence winning -- a positional vocabulary cannot
    hold a name twice, and someone who typed one twice has done nothing harmful.
    """
    named = [str(o.get("name", "")).strip() for o in objects]
    labels = list(dict.fromkeys(n for n in named if n))
    if not labels:
        raise ValueError("no named objects — the vocabulary would be empty")
    treatments = {
        n: (o.get("treatment") or {"key": "none"}) for o, n in zip(objects, named, strict=True) if n
    }
    return labels, list(labels), treatments


def _resolve_cameras(src: LeRobotDataset, cameras: list[str] | None) -> tuple[list[str], dict[str, str]]:
    """The cameras this run writes, and each one's mask column.

    Pre: ``cameras`` is a subset of the dataset's camera keys, or None for all.
    Post: a non-empty camera list and its ``camera -> masks.<name>`` mapping.
    """
    cam_keys = list(src.meta.camera_keys)
    if cameras:
        cam_keys = [c for c in cam_keys if c in set(cameras)]
    if not cam_keys:
        raise ValueError("no camera keys selected")
    mask_key_of = mask_keys_for(cam_keys)
    for c in cam_keys:
        # Writers must refuse cameras with no derivable mask column — adopting
        # would try to replace the camera column itself with mask rows.
        assert mask_key_of[c] != c, f"camera key {c!r} has no '.images.' segment to derive a mask column"
    return cam_keys, mask_key_of


def _episode_is_covered(
    stored_rows: dict[str, list[str]],
    labels: list[str],
    requested: list[str],
    cam_keys: list[str],
    length: int,
) -> bool:
    """True iff every requested label is already present on every frame, everywhere.

    ``labels`` is the stored vocabulary, needed to decode a row's ids into names;
    ``requested`` is what this run was asked to find, which is what decides
    whether there is any work.

    "Present" means detected OR disabled: the write rule leaves both alone, so
    neither is a gap. A single absent (frame, label) makes the pass worth running,
    because that is the one thing it would write.
    """
    from lerobot.datasets.mask_codec import frame_states

    if not requested or not length:
        return False
    for cam in cam_keys:
        rows = stored_rows.get(cam) or []
        if len(rows) < length:
            return False
        for value in rows[:length]:
            if not value:
                return False
            present = frame_states(value, labels)
            if any(name not in present for name in requested):
                return False
    return True


def _log_stage_timing(
    t_stage: dict[str, float],
    started: float,
    frames: int,
    cam_keys: list[str],
    episode: int,
    *,
    cancelled: bool = False,
) -> None:
    """One line per episode saying where the time went.

    Per frame-camera rather than per frame, because the cameras are what scales:
    a four-camera dataset does four times the work per frame, and a rate quoted
    per frame hides that. `other` is the remainder -- model load, seeding,
    recovery, metadata -- and is worth watching: when it dominates, the cost is
    not in the loop this instruments.
    """
    total = time.perf_counter() - started
    fc = max(1, frames * max(1, len(cam_keys)))
    known = sum(t_stage.values())
    logger.info(
        "episode %d %s: %.1fs for %d frames x %d cam = %.1f ms/frame-camera "
        "(decode %.0f%% segment %.0f%% encode %.0f%% write %.0f%% other %.0f%%)",
        episode,
        "cancelled" if cancelled else "done",
        total,
        frames,
        len(cam_keys),
        1000.0 * total / fc,
        100.0 * t_stage["decode"] / total if total else 0,
        100.0 * t_stage["segment"] / total if total else 0,
        100.0 * t_stage["encode"] / total if total else 0,
        100.0 * t_stage["write"] / total if total else 0,
        100.0 * max(0.0, total - known) / total if total else 0,
    )


def _update_mask_feature_info(root: Path, updates: dict[str, dict]) -> None:
    """Merge effect options into mask features' info.json entries, atomically.

    The options are metadata about how to REPRODUCE the composite, not about the
    stored rows, so changing them must not touch parquet. Written via tmp+replace
    like every other info.json writer here.
    """
    import json as _json
    import os as _os

    info_path = root / "meta" / "info.json"
    with info_path.open("r") as fh:
        info = _json.load(fh)
    for key, fields in updates.items():
        if key in info.get("features", {}):
            info["features"][key].update(fields)
    tmp = info_path.with_suffix(".json.tmp")
    with tmp.open("w") as fh:
        _json.dump(info, fh, indent=4)
    _os.replace(tmp, info_path)
