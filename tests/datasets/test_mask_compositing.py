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
