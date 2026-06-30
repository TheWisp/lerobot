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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import DEFAULT_FEATURES, HF_LEROBOT_HOME

logger = logging.getLogger(__name__)

# When to re-sample a randomized effect's parameters. Per-episode is the
# default and the right answer for trajectory data: a fixed look per episode
# preserves temporal/motion cues, whereas per-frame resampling makes the
# background flicker and corrupts the dynamics the policy learns. Additive
# sensor noise is the one effect where per-frame is physically correct.
ApplyMode = Literal["per_episode", "per_frame", "static"]


# ── Effects ──────────────────────────────────────────────────────────────────
#
# Each effect rewrites an HxWx3 uint8 RGB frame given a foreground alpha
# (float [0,1], 1.0 = keep the original pixel). "background" effects composite a
# replacement behind the foreground; "global" effects ignore the mask and touch
# the whole frame. Randomness is drawn in ``sample()`` at the cadence ApplyMode
# dictates, so the same draw is reused across a whole episode when per-episode.


@dataclass(frozen=True)
class EffectSpec:
    """A productized effect: its identity + what the GUI should render for it."""

    key: str
    label: str
    group: Literal["background", "global"]
    # UI control declarations the frontend renders (color swatch / slider).
    controls: list[dict] = field(default_factory=list)
    randomized: bool = False  # does sample() draw anything? (drives the Apply-mode control)


EFFECTS: list[EffectSpec] = [
    EffectSpec(
        key="bg_random_color",
        label="Randomize background (color)",
        group="background",
        randomized=True,
    ),
    EffectSpec(
        key="bg_random_noise",
        label="Randomize background (texture)",
        group="background",
        randomized=True,
    ),
    EffectSpec(
        key="bg_solid",
        label="Solid background color",
        group="background",
        controls=[{"type": "color", "key": "color", "label": "Color", "default": [0, 200, 0]}],
    ),
    EffectSpec(
        key="bg_blur",
        label="Blur background",
        group="background",
        controls=[{"type": "range", "key": "sigma", "label": "Blur", "min": 3, "max": 41, "default": 15}],
    ),
    EffectSpec(
        key="brightness",
        label="Jitter brightness / contrast",
        group="global",
        randomized=True,
        controls=[
            {"type": "range", "key": "amount", "label": "Amount %", "min": 5, "max": 60, "default": 25}
        ],
    ),
]

_EFFECTS_BY_KEY = {e.key: e for e in EFFECTS}


def _solid(h: int, w: int, color) -> np.ndarray:
    out = np.empty((h, w, 3), dtype=np.uint8)
    out[:] = np.asarray(color, dtype=np.uint8)
    return out


def _noise_texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """A blobby low-frequency colour texture (random patches upsampled), not
    per-pixel static — closer to the random-texture backgrounds GreenAug found
    most effective, and far less jarring than white noise."""
    import cv2

    blocks = int(rng.integers(6, 16))
    small = rng.integers(0, 256, size=(blocks, blocks, 3), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def sample_effect(effect_key: str, params: dict, h: int, w: int, rng: np.random.Generator) -> dict:
    """Draw the per-application random values for one effect (or ``{}`` if it is
    deterministic). Called once per episode / per frame / per dataset depending
    on ApplyMode; the returned dict is passed to :func:`apply_effect`."""
    if effect_key == "bg_random_color":
        return {"color": [int(c) for c in rng.integers(0, 256, size=3)]}
    if effect_key == "bg_random_noise":
        return {"bg": _noise_texture(h, w, rng)}
    if effect_key == "brightness":
        amt = float(params.get("amount", 25)) / 100.0
        return {"bright": float(rng.uniform(-amt, amt)), "contrast": float(rng.uniform(1 - amt, 1 + amt))}
    return {}


def apply_effect(
    rgb: np.ndarray, alpha: np.ndarray, effect_key: str, params: dict, sampled: dict
) -> np.ndarray:
    """Rewrite one frame. ``rgb`` is HxWx3 uint8; ``alpha`` is HxW float in
    [0,1] (1.0 = protected foreground). Returns a new HxWx3 uint8 frame."""
    import cv2

    h, w = rgb.shape[:2]
    if effect_key == "brightness":  # global — ignores the mask
        b, c = sampled.get("bright", 0.0), sampled.get("contrast", 1.0)
        out = rgb.astype(np.float32) * c + b * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)

    if effect_key == "bg_solid":
        bg = _solid(h, w, params.get("color", [0, 200, 0]))
    elif effect_key == "bg_random_color":
        bg = _solid(h, w, sampled.get("color", [0, 0, 0]))
    elif effect_key == "bg_random_noise":
        bg = sampled.get("bg")
        if bg is None or bg.shape[:2] != (h, w):
            bg = _solid(h, w, [0, 0, 0])
    elif effect_key == "bg_blur":
        k = int(params.get("sigma", 15)) | 1  # ksize must be odd
        bg = cv2.GaussianBlur(rgb, (k, k), 0)
    else:
        raise ValueError(f"unknown effect {effect_key!r}")

    a = alpha[:, :, None]
    out = rgb.astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def _feathered_alpha(masks: list[np.ndarray], h: int, w: int, feather: int = 5) -> np.ndarray:
    """Union the per-object boolean masks into a soft foreground alpha. A small
    dilation + blur softens the hard SAM edge so the composited seam isn't a
    crisp cut-out (GreenAug shows imperfect masks are fine; this just avoids the
    worst artefacts). Returns HxW float in [0,1]; all-zero (no detection) means
    the whole frame is treated as background."""
    import cv2

    union = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        if m is not None and m.shape == (h, w):
            union |= m.astype(np.uint8)
    if not union.any():
        return np.zeros((h, w), dtype=np.float32)
    if feather > 0:
        ksz = feather * 2 + 1
        union = cv2.dilate(union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)))
        soft = cv2.GaussianBlur(union.astype(np.float32) * 255.0, (ksz, ksz), 0) / 255.0
        return np.clip(soft, 0.0, 1.0)
    return union.astype(np.float32)


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
    effect: str,
    effect_params: dict | None = None,
    apply_mode: ApplyMode = "per_episode",
    variants: int = 1,
    cameras: list[str] | None = None,
    episodes: list[int] | None = None,
    out_root: str | Path | None = None,
    device: str = "cuda",
    model: str = "sam3_track",
    seed: int = 0,
    adapter: Any = None,
    progress: Callable[[dict], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> ProcessResult:
    """Segment ``objects`` in every frame of ``src`` and write an edited copy.

    Pre: ``src`` is a readable LeRobotDataset; ``effect`` is a key in
    :data:`EFFECTS`; ``objects`` is the overlay object list
    (``[{name, color, sign}]``). ``cameras``/``episodes`` default to all.
    ``variants`` > 1 writes that many independently-randomized copies of each
    source episode (the GreenAug "N augmented copies" knob).

    Post: a new dataset is written under ``out_root`` (default
    ``$HF_LEROBOT_HOME/out_repo_id``) with identical features and per-frame
    non-camera data; only camera pixels are transformed. Returns a
    :class:`ProcessResult`. If ``should_cancel`` flips True mid-run the partial
    dataset is finalized and ``cancelled=True`` is returned.

    ``progress`` is called with ``{stage, frames_done, frames_total,
    episodes_done, episodes_total, current_episode}`` roughly per frame.
    """
    if effect not in _EFFECTS_BY_KEY:
        raise ValueError(f"unknown effect {effect!r}; have {list(_EFFECTS_BY_KEY)}")
    effect_params = effect_params or {}
    spec = _EFFECTS_BY_KEY[effect]
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
    feature_keys = [k for k in src.meta.features if k not in DEFAULT_FEATURES]
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

        adapter = build_adapter(model, device=device)
    adapter.set_control({"objects": objects})

    out = LeRobotDataset.create(
        repo_id=out_repo_id,
        fps=src.meta.fps,
        features=create_features,
        root=out_root,
        robot_type=src.meta.robot_type,
        use_videos=len(src.meta.video_keys) > 0,
    )

    rng = np.random.default_rng(seed)
    static_sample: dict[str, dict] = {}  # per-camera, for ApplyMode == "static"
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
                ep_sample: dict[str, dict] = {}  # per-camera draw for per_episode mode

                for f in range(length):
                    if cancelled_flag():
                        cancelled = True
                        break
                    item = src[start + f]
                    frame: dict[str, Any] = {}
                    for k in feature_keys:
                        if k in edit_cams:
                            rgb = _to_rgb_uint8(item[k])
                            h, w = rgb.shape[:2]
                            adapter.set_camera(k)
                            masks = list(adapter.segment(rgb).values())
                            alpha = _feathered_alpha(masks, h, w)
                            # Draw the effect's randomness at the cadence ApplyMode wants.
                            if spec.randomized:
                                if apply_mode == "per_frame":
                                    s = sample_effect(effect, effect_params, h, w, rng)
                                elif apply_mode == "static":
                                    s = static_sample.setdefault(
                                        k, sample_effect(effect, effect_params, h, w, rng)
                                    )
                                else:  # per_episode
                                    s = ep_sample.setdefault(
                                        k, sample_effect(effect, effect_params, h, w, rng)
                                    )
                            else:
                                s = {}
                            frame[k] = apply_effect(rgb, alpha, effect, effect_params, s)
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
