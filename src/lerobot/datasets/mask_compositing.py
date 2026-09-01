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
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from lerobot.datasets.mask_codec import decode_frame

logger = logging.getLogger(__name__)

#: How often each DataLoader worker reports its running composite count. The
#: first report comes early so a short run still produces one — the count is
#: what an operator checks a finished run against — and the rest are rare
#: enough to stay invisible in a long one.
_REPORT_FIRST = 100
_REPORT_EVERY = 2000


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
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode(), usedforsecurity=False).hexdigest()[:8]


def episode_rng(episode: int, fingerprint: str) -> np.random.Generator:
    """The per-episode generator for randomized treatments, reproducibly.

    Same (episode, recipe) -> same first draw -> the identical background
    texture on every reproduction, per-episode coherent exactly like the bake.
    """
    seed = int.from_bytes(
        hashlib.sha1(f"{episode}:{fingerprint}".encode(), usedforsecurity=False).digest()[:8], "big"
    )
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

    Pre: ``rgb`` is HxWx3 uint8. It may be at the segmented resolution
    (``mask_size``) or at any scale of it — the masks are resized to the frame
    with nearest-neighbour, so a display-sized frame composites at display
    cost. ``row_value`` is the feature cell ("" / "[]" = segmented, nothing
    found: the whole frame is background). Post: a new HxWx3 uint8 frame; the
    input is not modified.

    Training composites at source scale, where the pixels are the ones the
    recipe describes; the scaled path exists for playback, which is going to
    be downscaled for the screen either way.

    ``cache`` carries randomized draws across the frames of one episode — pass
    one dict per (episode, recipe) for per-episode coherence; omitting it still
    yields identical pixels via the seeded generator, just re-drawn per call.
    """
    from lerobot.overlays.effects import build_and_sample_regions, composite_regions

    labels = spec.get("mask_labels", [])
    mh, mw = (int(x) for x in spec.get("mask_size", rgb.shape[:2]))
    h, w = rgb.shape[:2]
    if (h, w) != (mh, mw) and abs((w / h) - (mw / mh)) > 0.02:
        # A different SHAPE is a different picture — the wrong camera, or a
        # frame these masks were never computed on. Only a rescale of the same
        # picture is accepted, which is what the display path produces.
        raise ValueError(
            f"frame is {(h, w)} but masks were segmented at {(mh, mw)}; "
            "these are not the same picture at a different scale"
        )
    masks = decode_frame(row_value or "[]", labels, (mh, mw))
    if (h, w) != (mh, mw):
        # Resize the decoded masks rather than decoding straight at this size:
        # sampling the run structure per target pixel was measured SLOWER than
        # letting numpy fill the full mask and cv2 shrink it (11.2 vs 7.4
        # ms/frame on a 720p source at 640x360), because both of those are C
        # loops and the sampling is a searchsorted over every output pixel.
        # Nearest-neighbour is the only honest filter for a label image.
        import cv2

        masks = {
            name: cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            for name, m in masks.items()
        }

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
    key = mask_feature_of(camera_key)
    if key == camera_key:
        return None  # no derivable mask column for this naming; nothing saved
    info_path = Path(root) / "meta" / "info.json"
    try:
        with info_path.open() as fh:
            features = json.load(fh).get("features", {})
    except (FileNotFoundError, ValueError):
        return None
    ft = features.get(key)
    return ft if ft is not None and ft.get("mask_encoding") == "coco_rle" else None


#: Namespace the mask columns live under. Deliberately NOT inside
#: ``observation.``: ``dataset_to_policy_features`` classifies every non-image
#: ``observation.*`` feature as policy STATE, so a mask column there is
#: declared a policy input, handed to the normalizer, and then dropped by the
#: reader before collation -- a model that trains without an input its own
#: checkpoint says it has. Anything outside ``observation.`` and ``action``
#: hits that function's ``else: continue``, which is the same reason the flags
#: column is ``quality.human_flags``.
MASK_NAMESPACE = "masks"

#: What the columns were called before the namespace moved out of
#: ``observation.``. Datasets written then are still on disk, and the reader
#: cannot see their masks: it looks under MASK_NAMESPACE.
LEGACY_MASK_NAMESPACE = "observation.masks"

#: The camera-key convention masks are derived from. A camera outside it (e.g.
#: pusht's ``observation.image``) has no derivable mask column.
_CAMERA_INFIX = ".images."


def mask_feature_of(camera_key: str) -> str:
    """The mask column that describes ``camera_key``.

    Pre: ``camera_key`` is a dataset feature key.
    Post: ``masks.<suffix>`` for a camera following the ``*.images.*``
    convention; ``camera_key`` unchanged otherwise, which readers treat as "no
    masks" and writers must refuse (adopting would replace the camera column
    itself).

    This is the only place the mask column's name is constructed. Eleven call
    sites once spelled the swap inline, which is what made the namespace
    expensive to correct.
    """
    if _CAMERA_INFIX not in camera_key:
        return camera_key
    return f"{MASK_NAMESPACE}.{camera_key.split(_CAMERA_INFIX, 1)[1]}"


def mask_keys_for(camera_keys: Iterable[str]) -> dict[str, str]:
    """``{camera_key: mask_key}`` for a whole dataset, refusing collisions.

    Pre: ``camera_keys`` are the cameras masks are being written for.
    Post: one entry per camera, every value distinct.

    The mask key is derived from the part of the camera key AFTER
    ``.images.``, so two cameras whose names differ only before it --
    ``observation.images.top`` and ``sensors.images.top`` -- would share a
    column, and the second pass would overwrite the first's rows with masks of
    a different scene. The previous namespace could not collide because it kept
    the whole prefix, so this refuses rather than lets the narrowing corrupt a
    dataset. Nothing in the standard convention hits it: every camera lives
    under ``observation.images.``, where the suffix is already unique.
    """
    out: dict[str, str] = {}
    seen: dict[str, str] = {}
    for cam in camera_keys:
        key = mask_feature_of(cam)
        if key in seen:
            raise ValueError(
                f"cameras {seen[key]!r} and {cam!r} both map to the mask column {key!r}; "
                "masks for one would overwrite the other. Rename one camera so the part "
                f"after '{_CAMERA_INFIX}' is unique."
            )
        seen[key] = cam
        out[cam] = key
    return out


def camera_feature_of(mask_key: str, camera_keys: Iterable[str] = ()) -> str:
    """The camera ``mask_key`` describes -- the inverse of `mask_feature_of`.

    Pre: ``mask_key`` is a mask column key. ``camera_keys``, when given, is the
    dataset's camera keys and is used to resolve the prefix exactly.
    Post: the camera key, or ``mask_key`` unchanged if it names no mask column.

    Without ``camera_keys`` the standard ``observation.images.`` prefix is
    assumed. Passing them is preferred: a dataset whose cameras sit under some
    other prefix would otherwise get a key that does not exist.
    """
    prefix = f"{MASK_NAMESPACE}."
    if not mask_key.startswith(prefix):
        return mask_key
    suffix = mask_key[len(prefix) :]
    for cam in camera_keys:
        if cam.endswith(_CAMERA_INFIX + suffix):
            return cam
    return f"observation{_CAMERA_INFIX}{suffix}"


def legacy_mask_columns(features: Iterable[str]) -> list[str]:
    """Feature keys carrying masks under the pre-rename namespace."""
    return sorted(k for k in features if k.startswith(f"{LEGACY_MASK_NAMESPACE}."))
