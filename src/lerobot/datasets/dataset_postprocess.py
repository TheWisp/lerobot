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

# When to re-sample a randomized effect's parameters. Per-episode is the
# default and the right answer for trajectory data: a fixed look per episode
# preserves temporal/motion cues, whereas per-frame resampling makes the
# background flicker and corrupts the dynamics the policy learns. Additive
# sensor noise is the one effect where per-frame is physically correct.
ApplyMode = Literal["per_episode", "per_frame", "static"]


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

    rng = np.random.default_rng(seed)
    # Randomized-treatment draws, memoized per (camera → per-region) at the ApplyMode
    # cadence: `static_cache` persists across episodes; a fresh `ep_cache` per episode
    # gives the default per-episode coherence; per_frame passes a throwaway cache.
    static_cache: dict[str, dict] = {}
    frames_done = 0
    episodes_done = 0
    cancelled = False

    try:
        for _variant in range(variants):
            if cancelled:
                break
            for ep in episodes:
                if cancelled_flag():
                    cancelled = True
                    break
                start = int(src.meta.episodes["dataset_from_index"][ep])
                length = ep_lengths[ep]
                # New tracker session per (camera, episode): each episode is an
                # independent video stream, so reseed rather than propagate.
                for cam in cam_keys:
                    adapter.set_camera(cam)
                    adapter.reset()
                ep_cache: dict[str, dict] = {}  # per-camera region cache for per_episode mode

                for f in range(length):
                    if cancelled_flag():
                        cancelled = True
                        break
                    item = src[start + f]
                    # One segmentation pass for the timestep across all edited cameras —
                    # the same segment_many the live preview runs, so the batching flag
                    # (and its numeric trajectory) is shared: preview == commit.
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
                            # A camera the user excluded — copy through untouched.
                            frame[k] = _to_rgb_uint8(item[k])
                        else:
                            frame[k] = item[k]
                    frame["task"] = item["task"]
                    out.add_frame(frame)
                    frames_done += 1
                    if frames_done % 10 == 0 or f == length - 1:
                        _emit("processing", frames_done, frames_total, episodes_done, episodes_total, ep)
                out.save_episode()
                episodes_done += 1
                _emit("processing", frames_done, frames_total, episodes_done, episodes_total, ep)
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
