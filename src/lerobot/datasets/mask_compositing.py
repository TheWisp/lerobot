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

import collections
import hashlib
import json
import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from lerobot.datasets.mask_codec import decode_frame, frame_states

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


class SavedMaskCompositor:
    """Reproduce the saved effect on decoded camera frames at dataset load.

    Pre: ``root`` is a dataset root whose info.json may declare mask features;
    cameras without one pass through untouched. The recipes are snapshotted
    from disk ONCE here — a training run must read a stable recipe even if
    effects are edited in the GUI mid-run.

    Post: :meth:`apply` returns the item with each mask-bearing camera's frame
    replaced by its composite, same dtype and layout as decoded (uint8 or
    float CHW). Randomized treatments draw from the (episode, fingerprint)
    seeded generator, so training sees bit-identical pixels to playback.
    """

    def __init__(self, root, camera_keys) -> None:
        self.specs: dict[str, dict] = {}
        self.fingerprints: dict[str, str] = {}
        for cam in camera_keys:
            spec = load_recipe_from_disk(root, cam)
            if spec is not None:
                self.specs[cam] = spec
                self.fingerprints[cam] = recipe_fingerprint(spec)
        # Counters are per process: the composite runs in DataLoader workers,
        # so each keeps its own and reports separately. Summing across workers
        # is the reader's job and the worker id is in the line.
        self._composited = 0
        self._empty = 0
        self._per_label: dict[str, int] = {}
        self._total_ms = 0.0
        # A ring of recent samples, for a tail figure without an unbounded list.
        self._recent_ms: collections.deque = collections.deque(maxlen=512)
        self._announced = False
        # Say what will be applied, once, where the dataset is built. A run
        # that found no recipe says so too — "nothing applied" has to be
        # visible, since it is indistinguishable from success in the loss.
        if self.specs:
            for cam, spec in self.specs.items():
                treatments = {
                    label: (spec.get("mask_treatments", {}).get(label) or {}).get("key", "none")
                    for label in spec.get("mask_labels", [])
                }
                logger.info(
                    "saved masks: %s -> recipe %s, labels %s, background %s, segmented at %s",
                    cam,
                    self.fingerprints[cam],
                    treatments,
                    (spec.get("mask_background") or {}).get("key", "none"),
                    spec.get("mask_size"),
                )
        else:
            logger.info(
                "saved masks: none of %d cameras carries a mask recipe under %s — "
                "frames are served exactly as recorded",
                len(list(camera_keys)),
                root,
            )
        # Per-(episode, camera) randomized-draw caches, bounded: a background
        # texture is ~3 MB per camera and datasets can hold many episodes.
        self._caches: dict[tuple[int, str], dict] = {}
        self._cache_order: list[tuple[int, str]] = []

    def __bool__(self) -> bool:
        return bool(self.specs)

    def _cache_for(self, episode: int, cam: str) -> dict:
        key = (episode, cam)
        if key not in self._caches:
            self._caches[key] = {}
            self._cache_order.append(key)
            if len(self._cache_order) > 8:
                self._caches.pop(self._cache_order.pop(0), None)
        return self._caches[key]

    def apply(self, item: dict, episode_index: int) -> dict:
        """Composite every mask-bearing camera frame in ``item`` in place.

        Pre: decoded camera values are CHW (uint8 or float in [0, 1]) at the
        resolution the masks were segmented at; stacked windows (4-D, from
        delta_timestamps on a camera key) are not supported and raise.
        """
        import torch

        if self.specs and not self._announced:
            self._announced = True
            info = torch.utils.data.get_worker_info()
            logger.info(
                "saved masks: compositing live in %s for %s",
                f"dataloader worker {info.id}" if info is not None else "the main process",
                sorted(c.split(".")[-1] for c in self.specs),
            )

        for cam, spec in self.specs.items():
            if cam not in item:
                continue
            frames = item[cam]
            if frames.dim() == 4:
                raise NotImplementedError(
                    f"saved-mask compositing with stacked camera frames ({cam} has a "
                    "delta_timestamps window) is not implemented; disable "
                    "dataset.apply_saved_masks or drop the camera from delta_timestamps"
                )
            assert frames.dim() == 3, f"{cam}: expected CHW, got {tuple(frames.shape)}"
            row = item.get(mask_feature_of(cam))
            if isinstance(row, (list, tuple)):
                row = row[0] if row else ""
            row = "" if row is None else str(row)

            was_float = frames.is_floating_point()
            rgb = frames
            if rgb.shape[0] in (1, 3, 4):
                rgb = rgb.permute(1, 2, 0)
            if was_float:
                rgb = (rgb * 255).round().clamp(0, 255).to(torch.uint8)
            _t0 = time.perf_counter()
            composited = composite_from_store(
                np.ascontiguousarray(rgb.cpu().numpy()),
                row,
                spec,
                episode=episode_index,
                cache=self._cache_for(episode_index, cam),
            )
            _ms = (time.perf_counter() - _t0) * 1000.0
            self._total_ms += _ms
            self._recent_ms.append(_ms)
            out = torch.from_numpy(composited).permute(2, 0, 1).contiguous()
            item[cam] = out.to(frames.dtype) / 255.0 if was_float else out
            self._composited += 1
            # Per LABEL, not just per frame. The aggregate below answers "is
            # anything being composited"; it cannot answer "is `tray` being
            # composited", and a label that never applies -- a bad id mapping, a
            # name the writer stored differently -- looks exactly like an object
            # that happened not to be in frame. One counter per label makes the
            # difference visible without another log line.
            for name in frame_states(row, spec.get("mask_labels") or []) if row else ():
                self._per_label[name] = self._per_label.get(name, 0) + 1
            if not row or row == "[]":
                # Segmented and found nothing: the whole frame becomes
                # background. Legitimate when the object is out of view, and a
                # silent disaster when it means the pass failed, so it is
                # counted rather than assumed.
                self._empty += 1

        if self._composited == _REPORT_FIRST or (self._composited and self._composited % _REPORT_EVERY == 0):
            # Timing, aggregated. Compositing is the dominant cost of reading a
            # masked dataset — it was ~90% of a training step's data time when
            # this was written — and a run that reports only counts cannot say
            # so: the breakdown had to be reconstructed offline from step times.
            # A mean hides the tail that actually stalls a dataloader, so the
            # slowest of the recent samples is reported too, from a small ring
            # buffer rather than a growing list.
            recent = sorted(self._recent_ms)
            # Every declared label, including the ones at zero: a label missing
            # from the line reads as an oversight, a label showing 0 reads as the
            # fact it is.
            declared = sorted({n for sp in self.specs.values() for n in (sp.get("mask_labels") or [])})
            per_label = ", ".join(f"{n} {self._per_label.get(n, 0)}" for n in declared) or "none declared"
            logger.info("saved masks: applied per label — %s", per_label)
            logger.info(
                "saved masks: %d camera-frames composited in pid %d (%.1f%% had no mask, "
                "rendered as all-background) | %.2f ms/frame mean, %.2f ms p95 over the "
                "last %d, %.1f s spent compositing in total",
                self._composited,
                os.getpid(),
                100.0 * self._empty / self._composited,
                self._total_ms / self._composited,
                recent[int(len(recent) * 0.95)] if recent else 0.0,
                len(recent),
                self._total_ms / 1000.0,
            )
        return item


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


def refuse_legacy_mask_columns(features: Iterable[str]) -> None:
    """Raise if a dataset's masks are under the pre-rename namespace.

    Pre: ``features`` are the dataset's feature keys.
    Post: returns only if no legacy mask column is present.

    Called where a caller has asked for saved masks. Reading such a dataset
    would composite nothing -- raw pixels, a healthy-looking run, and no signal
    anywhere -- so it fails instead, naming the migration.

    Lives here rather than in the format layer because this is where its only
    caller is: a guard shipped ahead of the thing that invokes it reads as
    protection that does not exist.
    """
    legacy = legacy_mask_columns(features)
    if not legacy:
        return
    raise ValueError(
        f"This dataset's masks are under the old {LEGACY_MASK_NAMESPACE}.* namespace "
        f"({', '.join(legacy)}), which the reader no longer applies: loading it with "
        "apply_saved_masks would train on raw pixels. Convert it with "
        "`lerobot.datasets.mask_migrate.migrate_root(root)`, or pass "
        "apply_saved_masks=False (--ignore-saved-masks) to read the raw frames on purpose."
    )
