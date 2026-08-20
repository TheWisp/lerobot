# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Lossless run-length coding for per-frame segmentation masks.

Masks are sparse and blobby, so a bitmap wastes almost all of its bytes: a
224x224 binary mask is 6 KB packed, while the same region as run lengths is a
few hundred. Storing them as video was the other candidate and is rejected here
— a lossy codec perturbs pixel values, and an object-index image cannot survive
that, while lossless video encoders give up the compression that made video
attractive in the first place.

The encoding is COCO's, byte-for-byte, rather than something bespoke:

  * Runs are counted in COLUMN-major (Fortran) order, starting with a run of
    zeros, which is what ``pycocotools`` does and what every consumer of a COCO
    ``segmentation`` field expects.
  * Each run length is delta-coded against the run two positions back and
    written LEB128-style in 5-bit groups, biased into printable ASCII. That is
    why the result is a ``str`` and not ``bytes``: COCO chose it so masks embed
    in JSON, and it is what lets these live in a dataset ``string`` feature
    without a new dtype or a base64 wrapper.

Interoperability is the point of matching it: a mask written here can be read by
``pycocotools.mask.decode`` unchanged, and any COCO tooling can consume the
dataset. Correctness is pinned by a round-trip test over real SAM3 masks, not
just synthetic ones — see ``tests/datasets/test_mask_codec.py``.

Encoding is exact. ``decode(encode(m)) == m`` for every binary mask, so nothing
here is an approximation of the segmenter's output.

Why this is hand-rolled rather than ``pycocotools.mask``: there is no blocker,
only a trade. pycocotools is a C extension built at install time, and this is
~40 lines against a format that has not changed in a decade, so the dependency
buys little. It does buy speed — the C encoder is far faster than this pure
Python one — which does not matter while encoding happens once per frame in an
offline pass beside a 50 ms segmentation, and would matter if RLE ever lands in
a hot path. The wire format is byte-identical either way, so the swap is
mechanical: replace ``encode_mask``/``decode_mask`` with
``pycocotools.mask.encode/``decode`` and delete the two helpers. Revisit that
if profiling ever shows this module in a live loop.
"""

from __future__ import annotations

import json

import numpy as np

#: Value stored in a frame's mask feature when nothing was detected. An empty
#: list rather than an empty string, so a reader never has to distinguish
#: "no instances" from "field not written".
EMPTY = "[]"


def _counts_to_string(counts: list[int]) -> str:
    """COCO's ``rleToString``: delta-code, then 5 bits per character."""
    out: list[str] = []
    for i, count in enumerate(counts):
        x = int(count)
        if i > 2:
            x -= int(counts[i - 2])
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            # Bit 0x10 is the sign bit of the 5-bit group: when it is set the
            # value continues only while x is not -1 (arithmetic shift of a
            # negative), otherwise while x is not 0.
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def _string_to_counts(s: str) -> list[int]:
    """COCO's ``rleFrString``: inverse of :func:`_counts_to_string`."""
    counts: list[int] = []
    p, n = 0, len(s)
    while p < n:
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)  # sign-extend
        if len(counts) > 2:
            x += counts[len(counts) - 2]
        counts.append(x)
    return counts


def encode_mask(mask: np.ndarray) -> str:
    """One boolean HxW mask -> COCO RLE ``counts`` string.

    Pre: ``mask`` is 2-D and castable to bool. Post: the returned string decodes
    back to exactly this mask via :func:`decode_mask` given the same shape.
    """
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {mask.shape}")
    flat = np.asfortranarray(mask.astype(bool)).ravel(order="F")
    n = flat.size
    # Run boundaries, then lengths. Taking boundaries from value CHANGES (rather
    # than diffing a padded array) is what keeps a uniform mask correct: an
    # all-zero frame has no changes and must still emit one run covering every
    # pixel, since the counts have to sum to h*w for the frame to decode.
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], changes, [n]))).tolist()
    # The format defines the first run as zeros; a mask that starts set needs a
    # leading empty run so the alternation lines up.
    if n and flat[0]:
        counts = [0, *counts]
    return _counts_to_string(counts)


def decode_mask(counts: str, shape: tuple[int, int]) -> np.ndarray:
    """COCO RLE ``counts`` string -> boolean HxW mask. Inverse of :func:`encode_mask`."""
    h, w = shape
    flat = np.zeros(h * w, dtype=bool)
    pos, value = 0, False
    for run in _string_to_counts(counts):
        if value and run:
            flat[pos : pos + run] = True
        pos += run
        value = not value
    if pos != h * w:
        raise ValueError(f"RLE covers {pos} pixels, but {shape} needs {h * w}")
    return flat.reshape((h, w), order="F")


def encode_frame(masks_by_label: dict[str, np.ndarray], labels: list[str]) -> str:
    """All instances in one frame -> the string stored in the feature column.

    Pre: every key of ``masks_by_label`` is in ``labels``; ``labels`` is the
    vocabulary recorded in the feature metadata, so the stored rows carry small
    integer ids rather than repeating the label text on every frame.

    Post: returns JSON ``[[label_id, counts], ...]``, ordered by label id so two
    frames with the same content encode identically and diff cleanly.
    """
    index = {name: i for i, name in enumerate(labels)}
    rows = []
    for name, mask in masks_by_label.items():
        if name not in index:
            raise KeyError(f"label {name!r} is not in the declared vocabulary {labels}")
        if mask is None or not mask.any():
            continue  # an absent object is absent, not an empty run
        rows.append([index[name], encode_mask(mask)])
    rows.sort(key=lambda r: r[0])
    return json.dumps(rows, separators=(",", ":"))


def decode_frame(value: str, labels: list[str], shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Inverse of :func:`encode_frame`: the stored string -> ``{label: mask}``.

    Labels absent from the frame are absent from the result rather than present
    and empty, so a caller can tell "not detected" from "detected nothing".
    """
    out: dict[str, np.ndarray] = {}
    for label_id, counts in json.loads(value or EMPTY):
        if not 0 <= label_id < len(labels):
            raise ValueError(f"label id {label_id} outside the declared vocabulary of {len(labels)}")
        out[labels[label_id]] = decode_mask(counts, shape)
    return out


def feature_spec(labels: list[str], shape: tuple[int, int]) -> dict:
    """The dataset feature declaration for one camera's masks.

    Mirrors how ``quality.human_flags`` carries its ``flags`` vocabulary: the
    per-frame rows stay small and the meaning lives once, in the metadata. The
    mask resolution is recorded because it is the SEGMENTED resolution, which is
    deliberately not the stored video's — masks are computed at source scale,
    where SAM3 can still see a 50 px object, and may be composited against a
    downscaled frame later.
    """
    return {
        "dtype": "string",
        "shape": [1],
        "names": None,
        "mask_encoding": "coco_rle",
        "mask_labels": list(labels),
        "mask_size": [int(shape[0]), int(shape[1])],
    }
