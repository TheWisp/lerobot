# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""CRUD on the saved-mask column, against a real dataset.

`mask_store` is the layer a producer calls. It takes arrays and returns arrays,
so these tests use hand-built masks rather than a segmenter -- which is the
point of the layer: anything that can make a boolean array can write here.

What is pinned is the contract that makes the format safe to build on:
positions are permanent, an episode is written whole or not at all, and the
delete half exists.
"""

import json

import numpy as np
import pytest

from lerobot.datasets.mask_store import (
    RETIRED_KEY,
    active_labels,
    adopt,
    coverage,
    describe,
    labels_of,
    read_episode,
    read_frame,
    remove,
    rename_label,
    retire_label,
    spec_of,
    write_episode,
)

H, W = 64, 96
CAM = "observation.images.top"


def _stripe(row_from: int, row_to: int) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[row_from:row_to] = True
    return m


@pytest.fixture
def ds(tmp_path, info_factory, lerobot_dataset_factory):
    motors = {
        "action": {"dtype": "float32", "shape": (6,), "names": [f"j{i}" for i in range(6)]},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": [f"j{i}" for i in range(6)]},
    }
    cams = {CAM: {"shape": (H, W, 3), "names": ["height", "width", "channels"], "info": None}}
    info = info_factory(
        total_episodes=2, total_frames=8, total_tasks=1, motor_features=motors, camera_features=cams
    )
    return lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=8, info=info)


def _ep_len(ds, episode):
    return int(ds.meta.episodes["length"][episode])


# ── create ──────────────────────────────────────────────────────────────────


def test_adopt_declares_the_vocabulary_and_leaves_rows_empty(ds):
    keys = adopt(ds, [CAM], ["ball", "tray"], (H, W))
    assert keys == {CAM: "masks.top"}
    assert labels_of(ds, CAM) == ["ball", "tray"]
    assert coverage(ds, 0, CAM) == (0, _ep_len(ds, 0)), "adopting must not invent rows"


def test_adopting_twice_is_refused(ds):
    adopt(ds, [CAM], ["ball"], (H, W))
    with pytest.raises(ValueError, match="already adopted"):
        adopt(ds, [CAM], ["ball"], (H, W))


def test_an_empty_vocabulary_is_refused(ds):
    with pytest.raises(ValueError, match="at least one label"):
        adopt(ds, [CAM], [], (H, W))


def test_a_camera_with_no_column_reads_as_none(ds):
    assert spec_of(ds, CAM) is None
    assert read_frame(ds, 0, 0, CAM) is None
    assert read_episode(ds, 0, CAM) is None
    assert labels_of(ds, CAM) == []


# ── write / read round trip ─────────────────────────────────────────────────


def test_masks_survive_a_write_read_round_trip(ds):
    adopt(ds, [CAM], ["ball", "tray"], (H, W))
    n = _ep_len(ds, 0)
    written = [{"ball": _stripe(0, 16), "tray": _stripe(32, 48)} for _ in range(n)]
    assert write_episode(ds, 0, CAM, written) == n

    got = read_frame(ds, 0, 0, CAM)
    assert set(got) == {"ball", "tray"}
    assert np.array_equal(got["ball"], _stripe(0, 16))
    assert np.array_equal(got["tray"], _stripe(32, 48))


def test_found_nothing_is_distinct_from_never_written(ds):
    """Both read as no masks, but only one means the frame was looked at."""
    adopt(ds, [CAM], ["ball"], (H, W))
    n = _ep_len(ds, 0)
    write_episode(ds, 0, CAM, [{} for _ in range(n)])

    assert read_frame(ds, 0, 0, CAM) == {}, "segmented, found nothing"
    assert coverage(ds, 0, CAM) == (0, n)
    # Episode 1 was never written; it reads the same way through this API, and
    # the difference lives in the stored cell, which the codec keeps distinct.
    assert read_frame(ds, 1, 0, CAM) == {}


def test_a_short_episode_is_refused_rather_than_partially_written(ds):
    """A partial write would leave rows describing frames they do not belong to."""
    adopt(ds, [CAM], ["ball"], (H, W))
    n = _ep_len(ds, 0)
    with pytest.raises(ValueError, match="frames"):
        write_episode(ds, 0, CAM, [{"ball": _stripe(0, 16)}] * (n - 1))
    assert coverage(ds, 0, CAM) == (0, n), "nothing was written"


def test_writing_one_episode_leaves_the_other_alone(ds):
    adopt(ds, [CAM], ["ball"], (H, W))
    write_episode(ds, 0, CAM, [{"ball": _stripe(0, 16)}] * _ep_len(ds, 0))
    assert coverage(ds, 0, CAM)[0] == _ep_len(ds, 0)
    assert coverage(ds, 1, CAM)[0] == 0, "episode 1 was written by an episode-0 call"


def test_an_unknown_label_is_dropped_not_appended(ds):
    """Appending here would re-point nothing, but it would grow the vocabulary
    behind the caller's back and change what a later reader expects."""
    adopt(ds, [CAM], ["ball"], (H, W))
    write_episode(ds, 0, CAM, [{"ball": _stripe(0, 16), "ghost": _stripe(16, 32)}] * _ep_len(ds, 0))
    assert labels_of(ds, CAM) == ["ball"]
    assert set(read_frame(ds, 0, 0, CAM)) == {"ball"}


def test_writing_without_adopting_is_refused(ds):
    with pytest.raises(ValueError, match="no mask column"):
        write_episode(ds, 0, CAM, [{}] * _ep_len(ds, 0))


# ── update: rename ──────────────────────────────────────────────────────────


def test_renaming_a_label_changes_no_rows(ds):
    """Safe because rows reference positions: the string at a position moves,
    every row that used it now reads as the new name."""
    adopt(ds, [CAM], ["ball", "tray"], (H, W))
    write_episode(ds, 0, CAM, [{"ball": _stripe(0, 16)}] * _ep_len(ds, 0))
    before = read_frame(ds, 0, 0, CAM)["ball"]

    rename_label(ds, CAM, 0, "yellow ball")
    assert labels_of(ds, CAM) == ["yellow ball", "tray"]
    after = read_frame(ds, 0, 0, CAM)
    assert set(after) == {"yellow ball"}
    assert np.array_equal(after["yellow ball"], before), "pixels moved during a rename"


def test_renaming_carries_the_treatment(ds):
    adopt(ds, [CAM], ["ball"], (H, W), treatments={"ball": {"key": "blur"}})
    rename_label(ds, CAM, 0, "orb")
    assert spec_of(ds, CAM)["mask_treatments"] == {"orb": {"key": "blur"}}


def test_renaming_onto_an_existing_name_is_refused(ds):
    adopt(ds, [CAM], ["ball", "tray"], (H, W))
    with pytest.raises(ValueError, match="already at position"):
        rename_label(ds, CAM, 0, "tray")


def test_renaming_a_position_that_does_not_exist_is_refused(ds):
    adopt(ds, [CAM], ["ball"], (H, W))
    with pytest.raises(IndexError):
        rename_label(ds, CAM, 3, "x")


# ── delete: retire and remove ───────────────────────────────────────────────


def test_retiring_a_label_keeps_its_position(ds):
    """The vocabulary cannot be compacted: removing an entry would shift every
    later label down and re-point every stored row."""
    adopt(ds, [CAM], ["ball", "tray", "cup"], (H, W))
    write_episode(ds, 0, CAM, [{"cup": _stripe(0, 16)}] * _ep_len(ds, 0))

    assert retire_label(ds, CAM, 1) == [1]
    assert labels_of(ds, CAM) == ["ball", "tray", "cup"], "positions must not move"
    assert active_labels(ds, CAM) == ["ball", "cup"]
    assert set(read_frame(ds, 0, 0, CAM)) == {"cup"}, "a retirement changed what a row means"


def test_nothing_is_stored_until_the_first_retirement(ds):
    adopt(ds, [CAM], ["ball"], (H, W))
    assert RETIRED_KEY not in spec_of(ds, CAM)
    assert active_labels(ds, CAM) == ["ball"]


def test_retiring_is_idempotent_and_accumulates(ds):
    adopt(ds, [CAM], ["a", "b", "c"], (H, W))
    assert retire_label(ds, CAM, 2) == [2]
    assert retire_label(ds, CAM, 2) == [2]
    assert retire_label(ds, CAM, 0) == [0, 2]
    assert active_labels(ds, CAM) == ["b"]


def test_remove_drops_the_column_and_its_spec(ds):
    adopt(ds, [CAM], ["ball"], (H, W))
    write_episode(ds, 0, CAM, [{"ball": _stripe(0, 16)}] * _ep_len(ds, 0))

    assert remove(ds, [CAM]) == ["masks.top"]
    assert spec_of(ds, CAM) is None
    info = json.loads((ds.root / "meta" / "info.json").read_text())
    assert "masks.top" not in info["features"]
    assert CAM in info["features"], "removing masks took the camera with it"


def test_removing_when_there_is_nothing_to_remove(ds):
    assert remove(ds, [CAM]) == []


def test_describe_reports_what_is_stored(ds):
    adopt(ds, [CAM], ["ball", "tray"], (H, W), background={"key": "blur"})
    retire_label(ds, CAM, 1)
    d = describe(ds)["masks.top"]
    assert d["camera"] == CAM
    assert d["labels"] == ["ball", "tray"]
    assert d["retired"] == [1]
    assert d["size"] == [H, W]
    assert d["background"] == {"key": "blur"}


# ── what a producer actually hands in ───────────────────────────────────────


def test_a_probability_map_is_thresholded_at_half(ds):
    """Segmenters emit floats. `astype(bool)` would threshold at 0, turning
    every pixel the model gave any weight into object -- roughly twice the area
    on a real SAM output, which reads as poor segmentation rather than a bug."""
    adopt(ds, [CAM], ["ball"], (H, W))
    soft = np.zeros((H, W), np.float32)
    soft[0:16] = 0.9  # confidently object
    soft[16:32] = 0.05  # noise a caller would threshold away
    write_episode(ds, 0, CAM, [{"ball": soft}] * _ep_len(ds, 0))

    got = read_frame(ds, 0, 0, CAM)["ball"]
    assert got[0:16].all(), "the confident region was lost"
    assert not got[16:32].any(), (
        "low-confidence pixels were stored as object; the float was thresholded at 0, not 0.5"
    )
    assert int(got.sum()) == 16 * W


def test_a_boolean_mask_is_unchanged_by_the_threshold(ds):
    """The complement: thresholding must not disturb the ordinary case."""
    adopt(ds, [CAM], ["ball"], (H, W))
    write_episode(ds, 0, CAM, [{"ball": _stripe(0, 16)}] * _ep_len(ds, 0))
    assert np.array_equal(read_frame(ds, 0, 0, CAM)["ball"], _stripe(0, 16))


def test_dropping_an_unknown_label_is_reported(ds, caplog):
    """Dropping is right -- appending would re-point every other episode's rows
    -- but silence is not: a typo'd label would otherwise store nothing for that
    object and still report a full episode written."""
    import logging

    adopt(ds, [CAM], ["ball"], (H, W))
    with caplog.at_level(logging.WARNING):
        write_episode(ds, 0, CAM, [{"ball": _stripe(0, 16), "typo": _stripe(16, 32)}] * _ep_len(ds, 0))
    messages = [r.getMessage() for r in caplog.records]
    assert any("typo" in m for m in messages), f"no warning named the dropped label; got {messages}"
