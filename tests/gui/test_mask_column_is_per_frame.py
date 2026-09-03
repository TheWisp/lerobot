# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""A mask column is per-frame however uniform its values are.

`_detect_per_episode_features` decides a column is per-episode when its value
never changes within an episode. Mask columns used to be excluded from that
scan only incidentally: they were named `observation.masks.*`, and the scan
skips everything under `observation.`. Moving them to their own namespace
removed that accident.

A segmenter that finds the same region in every frame -- a fixed tray, a
static background, anything bolted down -- writes an identical RLE string
throughout. The scan reads that as uniform-per-episode, and the feature panel
renders such columns in the episode inspector rather than as a timeline track:
the mask lanes disappear, and they disappear for exactly the datasets whose
masks are most stable.

The exclusion is explicit now, and keyed on `mask_encoding` rather than on the
name, because the name is a namespace that has already moved once.
"""

import types

import pandas as pd
import pytest

from lerobot.gui.api.datasets import _detect_per_episode_features

MASK_KEY = "masks.top"
UNIFORM = "quality.grade"


@pytest.fixture
def dataset_with_a_constant_mask(tmp_path):
    """A dataset whose mask cell is byte-identical in every frame.

    Alongside it, a genuinely per-episode string column, so a fix that simply
    stopped detecting anything could not pass.
    """
    shard = tmp_path / "data" / "chunk-000"
    shard.mkdir(parents=True)
    frames = 6
    pd.DataFrame(
        {
            "episode_index": [0] * frames,
            MASK_KEY: ['[[0,"PPk0"]]'] * frames,  # identical every frame
            UNIFORM: ["good"] * frames,  # genuinely per-episode
        }
    ).to_parquet(shard / "file-000.parquet")

    features = {
        "action": {"dtype": "float32", "shape": [6], "names": None},
        MASK_KEY: {
            "dtype": "string",
            "shape": [1],
            "names": None,
            "mask_encoding": "coco_rle",
            "mask_labels": ["tray"],
            "mask_size": [8, 12],
        },
        UNIFORM: {"dtype": "string", "shape": [1], "names": None},
    }
    return types.SimpleNamespace(root=tmp_path, meta=types.SimpleNamespace(features=features))


def test_a_constant_mask_column_is_not_per_episode(dataset_with_a_constant_mask):
    detected = _detect_per_episode_features("ds-under-test", dataset_with_a_constant_mask)
    assert MASK_KEY not in detected, (
        f"{MASK_KEY} was detected as per-episode; the feature panel renders those in the "
        "episode inspector, so the mask lanes vanish for any dataset whose masks do not move"
    )


def test_the_detector_still_finds_a_real_per_episode_column(dataset_with_a_constant_mask):
    """Complement: without this, excluding everything would pass the test above."""
    detected = _detect_per_episode_features("ds-under-test-2", dataset_with_a_constant_mask)
    assert UNIFORM in detected, "per-episode detection stopped working altogether"
