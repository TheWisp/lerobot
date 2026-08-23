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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
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
