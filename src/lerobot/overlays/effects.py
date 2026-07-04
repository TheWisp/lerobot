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
"""Visual effects for data editing — the pixel transforms applied to a frame
given a protected-foreground mask.

Pure numpy/cv2, no dataset or model dependencies, so BOTH consumers share one
source of truth:
  * the live overlay worker (:mod:`lerobot.overlays.standalone`) renders the
    selected effect on the scrubbed frame — the WYSIWYG preview;
  * the offline batch pass (:mod:`lerobot.datasets.dataset_postprocess`) applies
    the same effect to every frame when committing to a new dataset.

An effect rewrites an HxWx3 uint8 RGB frame given a foreground alpha (float
[0,1], 1.0 = keep the original pixel). "background" effects composite a
replacement behind the foreground; "global" effects ignore the mask and touch
the whole frame. Randomness is drawn in :func:`sample_effect` at the cadence the
caller chooses (once per episode for trajectory coherence), so the same draw is
reused across a whole episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class EffectSpec:
    """A productized effect: its identity + what the GUI should render for it."""

    key: str
    label: str
    group: Literal["background", "global"]
    # UI control declarations the frontend renders (color swatch / slider).
    controls: list[dict] = field(default_factory=list)
    randomized: bool = False  # does sample_effect() draw anything? (per-episode vs static)


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
        controls=[
            {"type": "range", "key": "strength", "label": "Blur strength", "min": 2, "max": 40, "default": 12}
        ],
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

EFFECTS_BY_KEY = {e.key: e for e in EFFECTS}


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
    on the caller; the returned dict is passed to :func:`apply_effect`."""
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
        # Treat the control value as the ACTUAL gaussian sigma (blur strength), not
        # the kernel size — cv2 derives a tiny sigma from ksize otherwise, so the
        # blur was near-invisible on flat backgrounds. ksize wide enough for sigma.
        sigma = max(1.0, float(params.get("strength", 12)))
        k = int(sigma * 4) | 1
        bg = cv2.GaussianBlur(rgb, (k, k), sigma)
    else:
        raise ValueError(f"unknown effect {effect_key!r}")

    a = alpha[:, :, None]
    out = rgb.astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def feathered_alpha(masks: list[np.ndarray], h: int, w: int, feather: int = 5) -> np.ndarray:
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
