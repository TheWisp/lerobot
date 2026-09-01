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
import random

import numpy as np
import pytest

from lerobot.datasets.mask_compositing import mask_feature_of
from lerobot.datasets.mask_store import (
    RETIRED_KEY,
    active_labels,
    adopt,
    append_labels,
    coverage,
    delete_label_range,
    describe,
    labels_of,
    mask_columns,
    read_episode,
    read_frame,
    remove,
    rename_label,
    retire_label,
    set_label_enabled,
    spec_of,
    states,
    unify_vocabulary,
    vocabulary_of,
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
    # `episodes_factory` splits the frames with an unseeded multinomial, so
    # episode lengths -- and whether episode 0 is long enough to hold a range
    # -- differ per run. Seeding makes the shape a fact these tests can assert
    # on rather than a die roll; without it a range test fails about one run in
    # five, and blames whatever it was pointed at.
    random.seed(0)
    np.random.seed(0)
    motors = {
        "action": {"dtype": "float32", "shape": (6,), "names": [f"j{i}" for i in range(6)]},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": [f"j{i}" for i in range(6)]},
    }
    cams = {CAM: {"shape": (H, W, 3), "names": ["height", "width", "channels"], "info": None}}
    info = info_factory(
        total_episodes=2, total_frames=8, total_tasks=1, motor_features=motors, camera_features=cams
    )
    ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=8, info=info)
    assert int(ds.meta.episodes["length"][0]) >= 4, "the range tests below index into episode 0"
    return ds


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

    rename_label(ds, 0, "yellow ball")
    assert labels_of(ds, CAM) == ["yellow ball", "tray"]
    after = read_frame(ds, 0, 0, CAM)
    assert set(after) == {"yellow ball"}
    assert np.array_equal(after["yellow ball"], before), "pixels moved during a rename"


def test_renaming_carries_the_treatment(ds):
    adopt(ds, [CAM], ["ball"], (H, W), treatments={"ball": {"key": "blur"}})
    rename_label(ds, 0, "orb")
    assert spec_of(ds, CAM)["mask_treatments"] == {"orb": {"key": "blur"}}


def test_renaming_onto_an_existing_name_is_refused(ds):
    adopt(ds, [CAM], ["ball", "tray"], (H, W))
    with pytest.raises(ValueError, match="already at position"):
        rename_label(ds, 0, "tray")


def test_renaming_a_position_that_does_not_exist_is_refused(ds):
    adopt(ds, [CAM], ["ball"], (H, W))
    with pytest.raises(IndexError):
        rename_label(ds, 3, "x")


# ── delete: retire and remove ───────────────────────────────────────────────


def test_retiring_a_label_keeps_its_position(ds):
    """The vocabulary cannot be compacted: removing an entry would shift every
    later label down and re-point every stored row."""
    adopt(ds, [CAM], ["ball", "tray", "cup"], (H, W))
    write_episode(ds, 0, CAM, [{"cup": _stripe(0, 16)}] * _ep_len(ds, 0))

    assert retire_label(ds, 1) == [1]
    assert labels_of(ds, CAM) == ["ball", "tray", "cup"], "positions must not move"
    assert active_labels(ds, CAM) == ["ball", "cup"]
    assert set(read_frame(ds, 0, 0, CAM)) == {"cup"}, "a retirement changed what a row means"


def test_nothing_is_stored_until_the_first_retirement(ds):
    adopt(ds, [CAM], ["ball"], (H, W))
    assert RETIRED_KEY not in spec_of(ds, CAM)
    assert active_labels(ds, CAM) == ["ball"]


def test_retiring_is_idempotent_and_accumulates(ds):
    adopt(ds, [CAM], ["a", "b", "c"], (H, W))
    assert retire_label(ds, 2) == [2]
    assert retire_label(ds, 2) == [2]
    assert retire_label(ds, 0) == [0, 2]
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
    retire_label(ds, 1)
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


# ── disable and delete over a range ──────────────────────────────────────────
# The pair a repair loop needs. Deleting returns a frame to absent so a later
# gap-filling write refills it; disabling keeps the pixels, stops them reaching
# training, and protects them from that same write. Without both, "this
# detection is wrong here" and "nothing was found here" are one state, and
# every pass puts the wrong detection back.


def _filled(ds, labels=("ball", "tray")):
    """Both labels on every frame of episode 0."""
    adopt(ds, [CAM], list(labels), (H, W))
    n = _ep_len(ds, 0)
    write_episode(ds, 0, CAM, [{"ball": _stripe(0, 16), "tray": _stripe(32, 48)} for _ in range(n)])
    return n


def test_disabling_mutes_a_label_over_a_range_only(ds):
    n = _filled(ds)
    assert set_label_enabled(ds, 0, CAM, "ball", False, frames=range(1, 3)) == 2
    got = states(ds, 0, CAM)
    assert [s["ball"] for s in got] == [True, False, False] + [True] * (n - 3)
    assert all(s["tray"] for s in got), "tray was not the target"


def test_a_disabled_mask_is_not_returned_by_a_normal_read(ds):
    """What the compositor sees: read_frame is the training-side read."""
    _filled(ds)
    set_label_enabled(ds, 0, CAM, "ball", False, frames=range(0, 1))
    assert set(read_frame(ds, 0, 0, CAM)) == {"tray"}


def test_disabling_keeps_the_pixels(ds):
    """Muting is reversible; that is the whole difference from deleting."""
    _filled(ds)
    set_label_enabled(ds, 0, CAM, "ball", False, frames=range(0, 2))
    set_label_enabled(ds, 0, CAM, "ball", True, frames=range(0, 2))
    assert np.array_equal(read_frame(ds, 0, 0, CAM)["ball"], _stripe(0, 16))


def test_deleting_returns_a_frame_to_absent(ds):
    """The complement of the test above: after a delete there is nothing to
    unmute, and a later write is free to fill."""
    n = _filled(ds)
    assert delete_label_range(ds, 0, CAM, "ball", frames=range(0, 2)) == 2
    got = states(ds, 0, CAM)
    assert "ball" not in got[0] and "ball" not in got[1]
    assert got[2]["ball"] is True
    assert set(read_frame(ds, 0, 0, CAM)) == {"tray"}
    assert coverage(ds, 0, CAM) == (n, n), "the frames still hold tray"


def test_deleting_every_label_leaves_found_nothing_not_never_written(ds):
    n = _filled(ds)
    delete_label_range(ds, 0, CAM, "ball")
    delete_label_range(ds, 0, CAM, "tray")
    assert coverage(ds, 0, CAM) == (0, n)
    assert read_frame(ds, 0, 0, CAM) == {}


def test_disabling_a_label_that_is_absent_on_a_frame_changes_nothing(ds):
    """There is no mask to mute. Storing a flag anyway would invent a mask."""
    _filled(ds)
    delete_label_range(ds, 0, CAM, "ball", frames=range(0, 2))
    assert set_label_enabled(ds, 0, CAM, "ball", False, frames=range(0, 2)) == 0
    assert "ball" not in states(ds, 0, CAM)[0]


def test_toggling_to_the_state_it_already_has_writes_nothing(ds):
    _filled(ds)
    assert set_label_enabled(ds, 0, CAM, "ball", True) == 0


def test_disabling_one_label_leaves_the_others_enabled(ds):
    """Labels are independent: what happens to ball says nothing about tray."""
    _filled(ds)
    set_label_enabled(ds, 0, CAM, "ball", False)
    st = states(ds, 0, CAM)[0]
    assert st == {"ball": False, "tray": True}


def test_disabling_two_labels_accumulates(ds):
    """The second write must carry the first's flag, not drop it."""
    _filled(ds)
    set_label_enabled(ds, 0, CAM, "ball", False)
    set_label_enabled(ds, 0, CAM, "tray", False)
    assert states(ds, 0, CAM)[0] == {"ball": False, "tray": False}


def test_deleting_a_label_carries_the_others_flags(ds):
    """A delete re-encodes the frame; a disabled neighbour must stay disabled."""
    adopt(ds, [CAM], ["ball", "tray", "arm"], (H, W))
    n = _ep_len(ds, 0)
    write_episode(
        ds,
        0,
        CAM,
        [{"ball": _stripe(0, 8), "tray": _stripe(16, 24), "arm": _stripe(32, 40)} for _ in range(n)],
    )
    set_label_enabled(ds, 0, CAM, "tray", False)
    delete_label_range(ds, 0, CAM, "ball")
    assert states(ds, 0, CAM)[0] == {"tray": False, "arm": True}


def test_a_range_outside_the_episode_is_refused(ds):
    """Episode-relative indices; an off-by-one here would edit the neighbour."""
    n = _filled(ds)
    with pytest.raises(IndexError, match="outside episode"):
        set_label_enabled(ds, 0, CAM, "ball", False, frames=range(0, n + 1))


def test_a_range_operation_does_not_reach_the_next_episode(ds):
    n = _filled(ds)
    write_episode(ds, 1, CAM, [{"ball": _stripe(0, 16)}] * _ep_len(ds, 1))
    set_label_enabled(ds, 0, CAM, "ball", False, frames=range(0, n))
    assert all(s["ball"] for s in states(ds, 1, CAM)), "episode 1 was muted by an episode-0 call"


def test_a_label_outside_the_vocabulary_is_refused(ds):
    _filled(ds)
    for fn in (
        lambda: set_label_enabled(ds, 0, CAM, "ghost", False),
        lambda: delete_label_range(ds, 0, CAM, "ghost"),
    ):
        with pytest.raises(ValueError, match="not in the vocabulary"):
            fn()


def test_range_operations_need_a_column(ds):
    with pytest.raises(ValueError, match="no mask column"):
        set_label_enabled(ds, 0, CAM, "ball", False)


def test_states_reports_nothing_for_a_camera_with_no_column(ds):
    assert states(ds, 0, CAM) == []


def test_write_episode_can_write_a_label_already_disabled(ds):
    """`write_episode` replaces the episode, so it takes the flags with it --
    otherwise a producer would have to re-disable after every pass."""
    adopt(ds, [CAM], ["ball", "tray"], (H, W))
    n = _ep_len(ds, 0)
    write_episode(
        ds,
        0,
        CAM,
        [{"ball": _stripe(0, 16), "tray": _stripe(32, 48)} for _ in range(n)],
        disabled_per_frame=[["ball"]] * n,
    )
    assert states(ds, 0, CAM)[0] == {"ball": False, "tray": True}
    assert set(read_frame(ds, 0, 0, CAM)) == {"tray"}


def test_an_unknown_disabled_name_is_reported_like_an_unknown_label(ds, caplog):
    """Filtering it silently would leave the mask enabled and reaching
    training, with nothing said -- the failure the label warning exists for."""
    adopt(ds, [CAM], ["ball"], (H, W))
    n = _ep_len(ds, 0)
    with caplog.at_level("WARNING"):
        write_episode(ds, 0, CAM, [{"ball": _stripe(0, 16)}] * n, disabled_per_frame=[["bal"]] * n)
    assert "bal" in caplog.text
    assert states(ds, 0, CAM)[0] == {"ball": True}, "the mask stays enabled"


# ── one vocabulary across every camera ──────────────────────────────────────
# A label is one physical object, so the same name must mean the same thing in
# every camera. `info.json` cannot hold that at the top level -- `DatasetInfo`
# is a closed dataclass that drops unknown keys on read and never writes them
# back -- so the list is mirrored into each column and the invariant enforced
# in code. These pin the enforcement, since nothing structural provides it.

CAM2 = "observation.images.wrist"


@pytest.fixture
def ds2(tmp_path, info_factory, lerobot_dataset_factory):
    """Two cameras, which is where a per-column vocabulary can go wrong."""
    random.seed(0)
    np.random.seed(0)
    motors = {
        "action": {"dtype": "float32", "shape": (6,), "names": [f"j{i}" for i in range(6)]},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": [f"j{i}" for i in range(6)]},
    }
    cams = {
        c: {"shape": (H, W, 3), "names": ["height", "width", "channels"], "info": None} for c in (CAM, CAM2)
    }
    info = info_factory(
        total_episodes=2, total_frames=8, total_tasks=1, motor_features=motors, camera_features=cams
    )
    ds = lerobot_dataset_factory(root=tmp_path / "ds2", total_episodes=2, total_frames=8, info=info)
    assert int(ds.meta.episodes["length"][0]) >= 4
    return ds


def test_adopt_gives_every_camera_the_same_vocabulary(ds2):
    adopt(ds2, [CAM, CAM2], ["ball", "tray"], (H, W))
    assert vocabulary_of(ds2) == ["ball", "tray"]
    assert labels_of(ds2, CAM) == labels_of(ds2, CAM2)


def test_mask_columns_finds_only_mask_columns(ds2):
    """The namespace prefix alone is not the test -- `action` and the image
    features must not be mistaken for columns with a vocabulary."""
    adopt(ds2, [CAM, CAM2], ["ball"], (H, W))
    assert set(mask_columns(ds2)) == {CAM, CAM2}


def test_appending_reaches_every_camera(ds2):
    """The defect this exists for: appending only to the camera a detection
    came from is what lets the vocabularies drift apart."""
    adopt(ds2, [CAM, CAM2], ["ball", "tray"], (H, W))
    assert append_labels(ds2, ["banana"]) == ["ball", "tray", "banana"]
    assert labels_of(ds2, CAM) == ["ball", "tray", "banana"]
    assert labels_of(ds2, CAM2) == ["ball", "tray", "banana"], "the other camera never learned it"


def test_appending_gives_a_label_the_same_id_everywhere(ds2):
    """Rows reference positions, so equal lists mean a label id is meaningful
    dataset-wide -- which is the property that makes cross-camera work safe."""
    adopt(ds2, [CAM, CAM2], ["ball"], (H, W))
    append_labels(ds2, ["tray", "banana"])
    for cam in (CAM, CAM2):
        assert labels_of(ds2, cam).index("banana") == 2


def test_appending_declares_a_label_without_writing_rows(ds2):
    """A label is declared here and detected elsewhere; the camera that saw
    nothing must not gain coverage."""
    adopt(ds2, [CAM, CAM2], ["ball"], (H, W))
    write_episode(ds2, 0, CAM, [{"ball": _stripe(0, 16)}] * _ep_len(ds2, 0))
    append_labels(ds2, ["banana"])
    assert coverage(ds2, 0, CAM2)[0] == 0
    assert read_frame(ds2, 0, 0, CAM2) == {}


def test_appending_a_name_that_exists_is_a_no_op(ds2):
    adopt(ds2, [CAM, CAM2], ["ball", "tray"], (H, W))
    assert append_labels(ds2, ["ball", "  ", ""]) == ["ball", "tray"]


def test_renaming_reaches_every_camera(ds2):
    adopt(ds2, [CAM, CAM2], ["apple", "orange"], (H, W))
    rename_label(ds2, 0, "fruit")
    assert labels_of(ds2, CAM) == ["fruit", "orange"]
    assert labels_of(ds2, CAM2) == ["fruit", "orange"], "one object, two names"


def test_renaming_carries_each_camera_s_treatment(ds2):
    adopt(ds2, [CAM, CAM2], ["apple"], (H, W), treatments={"apple": {"key": "blur"}})
    rename_label(ds2, 0, "fruit")
    for cam in (CAM, CAM2):
        assert spec_of(ds2, cam)["mask_treatments"] == {"fruit": {"key": "blur"}}


def test_retiring_reaches_every_camera(ds2):
    adopt(ds2, [CAM, CAM2], ["ball", "tray"], (H, W))
    assert retire_label(ds2, 1) == [1]
    for cam in (CAM, CAM2):
        assert spec_of(ds2, cam)[RETIRED_KEY] == [1]
        assert active_labels(ds2, cam) == ["ball"]


def test_a_rename_moves_no_rows_in_either_camera(ds2):
    adopt(ds2, [CAM, CAM2], ["ball", "tray"], (H, W))
    for cam in (CAM, CAM2):
        write_episode(ds2, 0, cam, [{"ball": _stripe(0, 16)}] * _ep_len(ds2, 0))
    rename_label(ds2, 0, "orb")
    for cam in (CAM, CAM2):
        assert np.array_equal(read_frame(ds2, 0, 0, cam)["orb"], _stripe(0, 16))


def test_a_diverged_vocabulary_is_refused_not_silently_picked(ds2):
    """Choosing one column's list would keep a treatment pointed at a name only
    some cameras use, and the dataset would read as consistent."""
    adopt(ds2, [CAM, CAM2], ["ball"], (H, W))
    ds2.meta.features[mask_feature_of(CAM2)]["mask_labels"] = ["ball", "rogue"]
    with pytest.raises(ValueError, match="different vocabularies"):
        vocabulary_of(ds2)


@pytest.mark.parametrize("op", ["append", "rename", "retire"])
def test_every_vocabulary_operation_refuses_a_diverged_dataset(ds2, op):
    adopt(ds2, [CAM, CAM2], ["ball"], (H, W))
    ds2.meta.features[mask_feature_of(CAM2)]["mask_labels"] = ["ball", "rogue"]
    calls = {
        "append": lambda: append_labels(ds2, ["x"]),
        "rename": lambda: rename_label(ds2, 0, "x"),
        "retire": lambda: retire_label(ds2, 0),
    }
    with pytest.raises(ValueError, match="different vocabularies"):
        calls[op]()


def test_unify_appends_the_union_and_leaves_positions_alone(ds2):
    """For datasets written before appends were dataset-wide. Positions cannot
    move -- rows already reference them -- so the union only appends."""
    adopt(ds2, [CAM, CAM2], ["ball"], (H, W))
    ds2.meta.features[mask_feature_of(CAM2)]["mask_labels"] = ["ball", "banana"]
    assert unify_vocabulary(ds2) == ["ball", "banana"]
    assert labels_of(ds2, CAM) == labels_of(ds2, CAM2) == ["ball", "banana"]
    assert vocabulary_of(ds2) == ["ball", "banana"], "and the check now passes"


def test_unify_refuses_when_it_would_move_a_label(ds2):
    """Reordered vocabularies cannot be unioned: any fix re-points rows."""
    adopt(ds2, [CAM, CAM2], ["ball", "tray"], (H, W))
    ds2.meta.features[mask_feature_of(CAM2)]["mask_labels"] = ["tray", "ball"]
    with pytest.raises(ValueError, match="would move a label"):
        unify_vocabulary(ds2)


def test_unify_is_a_no_op_when_they_already_agree(ds2):
    adopt(ds2, [CAM, CAM2], ["ball", "tray"], (H, W))
    assert unify_vocabulary(ds2) == ["ball", "tray"]
    assert labels_of(ds2, CAM) == ["ball", "tray"]


def test_one_camera_is_trivially_consistent(ds):
    """The single-camera case must not need unifying to work."""
    adopt(ds, [CAM], ["ball"], (H, W))
    assert vocabulary_of(ds) == ["ball"]
    assert append_labels(ds, ["tray"]) == ["ball", "tray"]


def test_a_dataset_with_no_mask_column_has_no_vocabulary(ds):
    assert vocabulary_of(ds) == []
    assert mask_columns(ds) == {}
    with pytest.raises(ValueError, match="no camera has a mask column"):
        append_labels(ds, ["ball"])


# ── one id per name, at the store's write paths ─────────────────────────────


def test_adopting_a_repeated_label_is_refused(ds):
    """`adopt` inherits the vocabulary guard rather than restating it.

    Asserted here and not only at `feature_spec`, because `adopt` is what a
    caller with a user in front of it actually reaches for; a refusal that
    existed one layer down and was bypassed here would leave the column written.
    """
    with pytest.raises(ValueError, match="more than once"):
        adopt(ds, [CAM], ["ball", "tray", "ball"], (H, W))
    # And the refusal happened before anything was written, so the dataset is
    # still adoptable -- a half-adopted column would be the worse outcome.
    assert spec_of(ds, CAM) is None
    adopt(ds, [CAM], ["ball", "tray"], (H, W))
    assert labels_of(ds, CAM) == ["ball", "tray"]


def test_appending_the_same_new_name_twice_declares_it_once(ds):
    """`append_labels` filtered only against the STORED list, so a name new to
    the dataset and given twice in one call was appended twice -- writing a
    duplicate into a vocabulary whose ids rows already reference, with no error.

    Appending is "ensure these exist", so a repeat inside the call means what a
    name already present means: nothing more to do.
    """
    adopt(ds, [CAM], ["ball"], (H, W))
    assert append_labels(ds, ["tray", "tray"]) == ["ball", "tray"]
    assert labels_of(ds, CAM) == ["ball", "tray"], "the repeat reached the column"

    # Mixed: one already stored, one new and repeated, in any order.
    assert append_labels(ds, ["ball", "cube", "cube", "ball"]) == ["ball", "tray", "cube"]
    assert labels_of(ds, CAM) == ["ball", "tray", "cube"]


def test_appending_still_appends_what_is_actually_new(ds):
    """The complement: deduping must not turn the call into a no-op.

    "no duplicates appear" is satisfied by an implementation that appends
    nothing at all, which is why this sits beside the test above.
    """
    adopt(ds, [CAM], ["ball"], (H, W))
    assert append_labels(ds, ["tray", "cube"]) == ["ball", "tray", "cube"]
    assert labels_of(ds, CAM) == ["ball", "tray", "cube"]
    # Order given is the order stored: it is the id assignment.
    assert labels_of(ds, CAM).index("tray") < labels_of(ds, CAM).index("cube")


def test_every_stored_vocabulary_holds_each_name_once(ds):
    """The invariant itself, over the store's write paths together.

    Stated as a property rather than against a literal, so a path added later
    that writes `mask_labels` is covered by the same assertion.
    """
    adopt(ds, [CAM], ["ball", "tray"], (H, W))
    append_labels(ds, ["cube", "cube", "ball", "ring"])
    unify_vocabulary(ds)
    for key in set(mask_columns(ds).values()):
        stored = list(ds.meta.features[key]["mask_labels"])
        assert len(stored) == len(set(stored)), f"{key} declares a name twice: {stored}"
