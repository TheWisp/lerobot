# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Contracts for reproducing saved effects from stored masks.

The recipe fingerprint is a cache key: everything that changes composited
pixels must change it, per camera, and nothing else may. The composite must be
bit-reproducible from storage alone — a still today, a training epoch next
week — including randomized backgrounds, which draw from a generator seeded by
(episode, fingerprint) rather than from stored pixels. And the recipe is read
from DISK at each use: an effects edit is a metadata write, and any consumer
trusting an in-memory copy composites yesterday's effects.
"""

import json

import numpy as np
import pytest

from lerobot.datasets.mask_codec import encode_frame
from lerobot.datasets.mask_compositing import (
    composite_from_store,
    episode_rng,
    load_recipe_from_disk,
    recipe_fingerprint,
)


def _spec(**over):
    spec = {
        "mask_encoding": "coco_rle",
        "mask_labels": ["tray", "ball"],
        "mask_size": [12, 10],
        "mask_treatments": {
            "tray": {"key": "none"},
            "ball": {"key": "tint", "params": {"color": [255, 0, 0]}},
        },
        "mask_background": {"key": "none"},
        "mask_model": "sam3_track",
    }
    spec.update(over)
    return spec


def _frame_and_row():
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 255, (12, 10, 3), dtype=np.uint8)
    tray = np.zeros((12, 10), bool)
    tray[2:8, 1:9] = True
    ball = np.zeros((12, 10), bool)
    ball[9:11, 2:5] = True
    return rgb, encode_frame({"tray": tray, "ball": ball}, ["tray", "ball"])


def test_fingerprint_tracks_every_field_that_changes_pixels():
    base = recipe_fingerprint(_spec())
    assert recipe_fingerprint(_spec(mask_labels=["tray", "cup"])) != base
    assert recipe_fingerprint(_spec(mask_size=[24, 20])) != base
    assert recipe_fingerprint(_spec(mask_background={"key": "blur", "params": {}})) != base
    # Treatment PARAMS count: a tint's color changes the pixels.
    other_tint = {"tray": {"key": "none"}, "ball": {"key": "tint", "params": {"color": [0, 255, 0]}}}
    assert recipe_fingerprint(_spec(mask_treatments=other_tint)) != base


def test_fingerprint_ignores_fields_that_do_not_change_pixels():
    """The model that produced the masks is provenance, not recipe: switching
    it must NOT orphan caches whose pixels would be identical."""
    assert recipe_fingerprint(_spec(mask_model="other")) == recipe_fingerprint(_spec())


def test_cameras_with_different_recipes_get_different_fingerprints():
    """Regression: the frame cache once keyed EVERY camera under the first
    camera's fingerprint, so edits to another camera's recipe stopped
    invalidating anything. Distinct recipes must yield distinct keys."""
    a = _spec()
    b = _spec(
        mask_treatments={"tray": {"key": "tint", "params": {"color": [0, 0, 255]}}, "ball": {"key": "none"}}
    )
    assert recipe_fingerprint(a) != recipe_fingerprint(b)


def test_episode_rng_is_reproducible_and_separates_episodes():
    fp = recipe_fingerprint(_spec())
    assert episode_rng(3, fp).random() == episode_rng(3, fp).random()
    assert episode_rng(3, fp).random() != episode_rng(4, fp).random()
    assert episode_rng(3, fp).random() != episode_rng(3, "0" * 8).random()


def test_composite_is_reproducible_from_storage_alone():
    """Two independent calls — no shared cache, randomized background — must
    agree bit-for-bit, or playback and training would see different data."""
    rgb, row = _frame_and_row()
    spec = _spec(mask_background={"key": "random", "params": {}})
    one = composite_from_store(rgb, row, spec, episode=2)
    two = composite_from_store(rgb, row, spec, episode=2)
    assert np.array_equal(one, two)
    assert one.shape == rgb.shape and one.dtype == np.uint8


def test_composite_changes_with_the_recipe():
    rgb, row = _frame_and_row()
    tinted = composite_from_store(rgb, row, _spec(), episode=0)
    plain = composite_from_store(
        rgb, row, _spec(mask_treatments={"tray": {"key": "none"}, "ball": {"key": "none"}}), episode=0
    )
    assert not np.array_equal(tinted, plain)
    assert np.array_equal(plain, rgb)  # all-none recipe is the identity


def test_composite_does_not_mutate_the_input():
    rgb, row = _frame_and_row()
    before = rgb.copy()
    composite_from_store(rgb, row, _spec(), episode=0)
    assert np.array_equal(rgb, before)


def test_composite_scales_masks_to_a_resized_frame():
    """Playback composites at display size — a quarter of the pixels for the
    same picture — so a rescaled frame is composited, not refused. Masks are
    label images, so they are resized nearest-neighbour: every pixel keeps a
    membership some pixel actually had."""
    import cv2

    rgb, row = _frame_and_row()
    small = cv2.resize(rgb, (5, 6), interpolation=cv2.INTER_AREA)
    out = composite_from_store(small, row, _spec(), episode=0)
    assert out.shape == small.shape and out.dtype == np.uint8


def test_composite_rejects_a_frame_of_another_shape():
    """A different aspect ratio is a different picture — the wrong camera, or
    a frame these masks were never computed on — and misaligning every region
    silently is exactly the failure this guard exists for."""
    rgb, row = _frame_and_row()
    with pytest.raises(ValueError, match="not the same picture"):
        composite_from_store(rgb[:6, :], row, _spec(), episode=0)


def test_empty_row_means_everything_is_background():
    rgb, _ = _frame_and_row()
    spec = _spec(mask_background={"key": "solid", "params": {"color": [0, 0, 0]}})
    out = composite_from_store(rgb, "", spec, episode=0)
    assert (out == 0).all()


def test_load_recipe_from_disk_sees_the_current_write(tmp_path):
    """The staleness contract: an effects edit rewrites info.json, and the
    very next read must return the new recipe — no caching layer allowed."""
    meta = tmp_path / "meta"
    meta.mkdir()
    features = {"masks.top": _spec()}
    (meta / "info.json").write_text(json.dumps({"features": features}))
    first = load_recipe_from_disk(tmp_path, "observation.images.top")
    assert first["mask_background"] == {"key": "none"}

    features["masks.top"] = _spec(mask_background={"key": "blur", "params": {}})
    (meta / "info.json").write_text(json.dumps({"features": features}))
    second = load_recipe_from_disk(tmp_path, "observation.images.top")
    assert second["mask_background"] == {"key": "blur", "params": {}}
    assert recipe_fingerprint(first) != recipe_fingerprint(second)


def test_load_recipe_from_disk_returns_none_when_absent(tmp_path):
    assert load_recipe_from_disk(tmp_path, "observation.images.top") is None
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"features": {"observation.images.top": {"dtype": "video"}}}))
    assert load_recipe_from_disk(tmp_path, "observation.images.top") is None


# ── a disabled mask must not reach training ─────────────────────────────────
# This is the reason the codec's decode excludes disabled by default rather
# than taking a flag here: the compositor is the training-side read, and the
# safe direction is the one you get by writing nothing.


def test_a_disabled_mask_is_not_composited():
    """The operator muted this detection; training must see the raw pixels
    there, exactly as if the label had never been found."""
    rgb, _ = _frame_and_row()
    tray = np.zeros((12, 10), bool)
    tray[2:8, 1:9] = True
    labels = ["tray", "ball"]
    spec = _spec(mask_treatments={"tray": {"key": "tint", "params": {"color": [0, 255, 0]}}})

    on = composite_from_store(rgb, encode_frame({"tray": tray}, labels), spec, episode=0)
    off = composite_from_store(rgb, encode_frame({"tray": tray}, labels, disabled=["tray"]), spec, episode=0)
    assert not np.array_equal(on, rgb), "the enabled mask must change pixels, or this proves nothing"
    assert np.array_equal(off, rgb), "a disabled mask reached the composite"


def test_disabling_one_label_leaves_the_others_composited():
    rgb, _ = _frame_and_row()
    labels = ["tray", "ball"]
    tray = np.zeros((12, 10), bool)
    tray[2:8, 1:9] = True
    ball = np.zeros((12, 10), bool)
    ball[9:11, 2:5] = True
    spec = _spec(
        mask_treatments={
            "tray": {"key": "tint", "params": {"color": [0, 255, 0]}},
            "ball": {"key": "tint", "params": {"color": [255, 0, 0]}},
        }
    )
    both = composite_from_store(rgb, encode_frame({"tray": tray, "ball": ball}, labels), spec, episode=0)
    ball_only = composite_from_store(
        rgb, encode_frame({"tray": tray, "ball": ball}, labels, disabled=["tray"]), spec, episode=0
    )
    just_ball = composite_from_store(rgb, encode_frame({"ball": ball}, labels), spec, episode=0)
    assert not np.array_equal(both, ball_only)
    assert np.array_equal(ball_only, just_ball), "disabling must equal never having detected it"


def test_a_disabled_mask_does_not_become_background():
    """The failure that would look almost right: excluding the mask from its
    treatment but still counting it as foreground would blank it instead."""
    rgb, _ = _frame_and_row()
    tray = np.zeros((12, 10), bool)
    tray[2:8, 1:9] = True
    spec = _spec(
        mask_treatments={"tray": {"key": "none"}},
        mask_background={"key": "solid", "params": {"color": [0, 0, 0]}},
    )
    out = composite_from_store(
        rgb, encode_frame({"tray": tray}, ["tray", "ball"], disabled=["tray"]), spec, episode=0
    )
    assert (out == 0).all(), "a disabled label must not hold back the background"


def test_timeline_summarizes_a_mask_row_as_a_presence_bitset():
    """The timeline asks which objects were found, not what the RLE said.

    The row is `[[label_id, rle], ...]`, so presence needs the ids alone — no
    decoding, and one integer per frame instead of a ~2 KB string. Sent as a
    bitset because several labels hold on the same frame, exactly like the
    flags columns the lane renderer was built for.
    """
    from lerobot.gui.api.datasets import _mask_presence_bits

    rgb, row = _frame_and_row()  # labels: tray (bit 0), ball (bit 1)
    assert _mask_presence_bits(row) == 0b11
    assert _mask_presence_bits([row]) == 0b11, "a list-wrapped cell reads the same"

    # "segmented, found nothing" and "never written" both answer 0: the row
    # has no object to draw either way.
    for empty in ("", "[]", None):
        assert _mask_presence_bits(empty) == 0

    # One label only.
    single = encode_frame({"ball": np.ones((12, 10), bool)}, ["tray", "ball"])
    assert _mask_presence_bits(single) == 0b10

    # A corrupt cell must not take the timeline down with it.
    assert _mask_presence_bits("{not json") == 0


def test_the_timeline_does_not_mark_a_disabled_label_as_present():
    """Every surface answers "what does training see here" — the track, the
    tile and the composite — so a muted mask must not draw a bar the tile then
    declines to paint."""
    from lerobot.gui.api.datasets import _mask_presence_bits

    labels = ["tray", "ball"]
    both = {n: np.ones((12, 10), bool) for n in labels}
    assert _mask_presence_bits(encode_frame(both, labels)) == 0b11
    assert _mask_presence_bits(encode_frame(both, labels, disabled=["tray"])) == 0b10
    assert _mask_presence_bits(encode_frame(both, labels, disabled=labels)) == 0

    # The form the encoder never writes, and the one it wrote before the flag.
    assert _mask_presence_bits('[[0,"abc",1]]') == 0b01
    assert _mask_presence_bits('[[0,"abc"]]') == 0b01, "a pre-flag row must stay present"
