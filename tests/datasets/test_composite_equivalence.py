# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The composited pixels are a contract, so speeding it up must not move them.

`composite_regions` is the single source of truth for committed pixels: the
live preview, saved playback and training all render through it, and the whole
saved-mask design rests on "what you previewed is what trains". That makes
every optimisation of it a refactor under an exactness obligation rather than
a free win.

So the outputs are pinned by hash, over a matrix that exercises the paths that
differ: an all-`none` recipe (the common one — keep the objects, treat the
background), per-object treatments, overlapping masks (where the
smallest-claims-the-pixel arbitration decides), a randomized background (which
must stay reproducible from its seed), and an empty row. The hashes were
recorded from the implementation as it stood before the optimisation work; a
change that moves any of them is either a bug or a deliberate change of the
rendering contract, and both should be argued for rather than discovered.
"""

import hashlib

import numpy as np
import pytest

from lerobot.datasets.mask_compositing import composite_from_store, episode_rng, recipe_fingerprint
from lerobot.overlays.effects import build_and_sample_regions, composite_regions

H, W = 96, 128


def _frame() -> np.ndarray:
    """A deterministic frame with structure in it (a flat one hides blur bugs)."""
    y, x = np.mgrid[0:H, 0:W]
    rgb = np.stack([(x * 2) % 256, (y * 3) % 256, ((x + y) * 5) % 256], axis=-1)
    return np.ascontiguousarray(rgb.astype(np.uint8))


def _masks(overlapping: bool) -> dict[str, np.ndarray]:
    a = np.zeros((H, W), bool)
    a[10:60, 12:70] = True  # the larger object
    b = np.zeros((H, W), bool)
    if overlapping:
        b[40:80, 50:100] = True  # overlaps `a`, and is smaller
    else:
        b[70:90, 90:120] = True
    return {"tray": a, "ball": b}


def _digest(img: np.ndarray) -> str:
    assert img.dtype == np.uint8
    return hashlib.sha256(img.tobytes()).hexdigest()[:16]


TINT = {"key": "tint", "params": {"color": [0, 200, 255]}}
BLUR = {"key": "blur", "params": {}}
NONE = {"key": "none", "params": {}}
RANDOM = {"key": "random", "params": {}}

CASES = {
    # name: (object treatments, background treatment, overlapping masks)
    "objects_none_bg_blur": ({"tray": NONE, "ball": NONE}, BLUR, False),
    "objects_none_bg_random": ({"tray": NONE, "ball": NONE}, RANDOM, False),
    "tint_and_blur": ({"tray": TINT, "ball": BLUR}, NONE, False),
    "overlap_smallest_wins": ({"tray": TINT, "ball": BLUR}, BLUR, True),
    "all_none": ({"tray": NONE, "ball": NONE}, NONE, False),
}

#: Re-recorded when the composite moved from a float32 round-trip to cv2's uint8
#: blend and the feather to a uint8 blur (12.76 -> 4.61 ms per 720p composite).
#: The change was accepted deliberately, not discovered: every case above was
#: rendered by both implementations and compared pixel by pixel first — the
#: maximum difference anywhere is ONE level out of 255, including the
#: overlapping multi-region case where successive blends could have
#: accumulated. A future change that moves these hashes owes the same check.
GOLDEN = {
    "all_none": "e8f484d47529ca53",
    "objects_none_bg_blur": "13a73373a3ee8727",
    "objects_none_bg_random": "b36b1decdf450bdf",
    "overlap_smallest_wins": "99a35f78633bc111",
    "tint_and_blur": "03eac6e35eedd95f",
}


def _render(case: str) -> np.ndarray:
    treatments, background, overlapping = CASES[case]
    rgb = _frame()
    rng = episode_rng(0, "goldenfp")
    regions, sampled = build_and_sample_regions(_masks(overlapping), treatments, background, H, W, rng, {})
    return composite_regions(rgb, regions, sampled)


@pytest.mark.parametrize("case", sorted(CASES))
def test_composited_pixels_are_unchanged(case: str):
    assert _digest(_render(case)) == GOLDEN[case], (
        f"{case}: the composite moved. Preview, saved playback and training all render "
        "through this function, so a change here changes what trains."
    )


def test_all_none_is_exactly_the_original_frame():
    """The degenerate case worth stating: treat nothing, change nothing."""
    assert np.array_equal(_render("all_none"), _frame())


def test_randomized_background_is_reproducible_from_the_seed():
    """A random background must be a function of (episode, recipe), or the same
    dataset composites differently on every read."""
    first, second = _render("objects_none_bg_random"), _render("objects_none_bg_random")
    assert np.array_equal(first, second)


def test_scaled_composite_matches_scaling_the_masks_by_hand():
    """Playback composites at display size. That path must agree with doing the
    resize explicitly — it is the same picture, not a different rendering."""
    import cv2

    from lerobot.datasets.mask_codec import encode_frame

    labels = ["tray", "ball"]
    masks = _masks(False)
    row = encode_frame(masks, labels)
    spec = {
        "mask_encoding": "coco_rle",
        "mask_labels": labels,
        "mask_size": [H, W],
        "mask_treatments": {"tray": NONE, "ball": TINT},
        "mask_background": BLUR,
    }
    small = cv2.resize(_frame(), (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    got = composite_from_store(small, row, spec, episode=0)

    rng = episode_rng(0, recipe_fingerprint(spec))
    scaled_masks = {
        n: cv2.resize(m.astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST).astype(bool)
        for n, m in masks.items()
    }
    regions, sampled = build_and_sample_regions(
        scaled_masks, spec["mask_treatments"], spec["mask_background"], H // 2, W // 2, rng, {}
    )
    assert np.array_equal(got, composite_regions(small, regions, sampled))
