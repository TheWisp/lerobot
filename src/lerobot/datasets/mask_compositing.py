# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Reproduce the saved effect from stored masks — the recipe's consumer.

The masks feature stores per-frame COCO RLE rows and the feature metadata
stores the effect options (per-label treatments, background, model). This
module turns the two back into pixels, and it is the ONE implementation both
playback and training use: the same `effects.py` composite the live preview and
the batch bake run, fed from storage instead of a segmenter. Measured cost is
5-15 ms/frame against 80-110 ms for segmentation, which is what makes "apply
effects freely" true.

Determinism: the `random` background draws a per-episode texture. Reproduction
seeds the generator from (episode, recipe fingerprint), so every call — a still
today, a video tomorrow, a training epoch next week — composites the identical
texture without any stored pixels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets.mask_codec import decode_frame


def recipe_fingerprint(spec: dict) -> str:
    """Stable 8-hex id of everything that changes the composited pixels.

    Cache keys carry it, so a treatments edit invalidates composited caches by
    construction — no cache-bust bookkeeping anywhere.
    """
    payload = {
        "labels": spec.get("mask_labels", []),
        "treatments": spec.get("mask_treatments", {}),
        "background": spec.get("mask_background", {"key": "none"}),
        "size": spec.get("mask_size"),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


def episode_rng(episode: int, fingerprint: str) -> np.random.Generator:
    """The per-episode generator for randomized treatments, reproducibly.

    Same (episode, recipe) -> same first draw -> the identical background
    texture on every reproduction, per-episode coherent exactly like the bake.
    """
    seed = int.from_bytes(hashlib.sha1(f"{episode}:{fingerprint}".encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


def composite_from_store(
    rgb: np.ndarray,
    row_value: str,
    spec: dict,
    *,
    episode: int,
    cache: dict | None = None,
) -> np.ndarray:
    """One stored row + the recipe -> the composited frame.

    Pre: ``rgb`` is HxWx3 uint8 at the SEGMENTED resolution (``mask_size`` —
    masks are stored at source scale, deliberately; composite first, downscale
    after). ``row_value`` is the feature cell ("" / "[]" = segmented, nothing
    found: the whole frame is background). Post: a new HxWx3 uint8 frame; the
    input is not modified.

    ``cache`` carries randomized draws across the frames of one episode — pass
    one dict per (episode, recipe) for per-episode coherence; omitting it still
    yields identical pixels via the seeded generator, just re-drawn per call.
    """
    from lerobot.overlays.effects import build_and_sample_regions, composite_regions

    labels = spec.get("mask_labels", [])
    h, w = (int(x) for x in spec.get("mask_size", rgb.shape[:2]))
    if rgb.shape[:2] != (h, w):
        raise ValueError(
            f"frame is {rgb.shape[:2]} but masks were segmented at {(h, w)}; "
            "composite at source resolution, then scale"
        )
    masks = decode_frame(row_value or "[]", labels, (h, w))
    treatments = spec.get("mask_treatments", {}) or {}
    background = spec.get("mask_background") or {"key": "none"}
    rng = episode_rng(episode, recipe_fingerprint(spec))
    regions, sampled = build_and_sample_regions(
        masks,
        {name: (treatments.get(name) or {"key": "none"}) for name in labels},
        background,
        h,
        w,
        rng,
        cache if cache is not None else {},
    )
    return composite_regions(rgb, regions, sampled)


def load_recipe_from_disk(root, camera_key: str) -> dict | None:
    """The camera's mask spec as info.json holds it right now, or None.

    The effects editor writes recipes straight to info.json (a metadata edit,
    no job), so consumers must read disk rather than a dataset object's
    in-memory meta — an in-memory copy is only as fresh as its last reload,
    and a stale recipe silently composites yesterday's effects. The read is
    ~0.1 ms against a 5-15 ms composite.
    """
    info_path = Path(root) / "meta" / "info.json"
    try:
        with info_path.open() as fh:
            features = json.load(fh).get("features", {})
    except (FileNotFoundError, ValueError):
        return None
    ft = features.get(mask_feature_of(camera_key))
    return ft if ft is not None and ft.get("mask_encoding") == "coco_rle" else None


def mask_feature_of(camera_key: str) -> str:
    """The mask column that describes ``camera_key``."""
    return camera_key.replace(".images.", ".masks.")


def composited_available(dataset: Any, camera_key: str) -> dict | None:
    """The recipe spec if this camera has an adopted mask feature, else None."""
    key = mask_feature_of(camera_key)
    ft = dataset.meta.features.get(key)
    return ft if ft is not None and ft.get("mask_encoding") == "coco_rle" else None
