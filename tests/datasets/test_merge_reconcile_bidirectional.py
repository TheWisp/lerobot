# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Reconciliation when BOTH sides are missing something, across feature kinds.

The straightforward case -- one side has an extra column -- says little about
the one that actually turns up: two datasets recorded months apart, each
carrying columns the other never had, of several different kinds. Flags are
integer vocabularies, masks are RLE strings whose meaning lives in the feature
metadata, and the rest are ordinary scalars. They fill differently, and a
neutral value that is wrong for one of them is wrong silently.
"""

from __future__ import annotations

import pytest
import torch

from lerobot.datasets.dataset_tools import add_features_inplace, merge_into
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.mask_codec import EMPTY, decode_frame, feature_spec
from lerobot.datasets.mask_store import NEVER_WRITTEN
from tests.fixtures.constants import DUMMY_REPO_ID

# On A only
A_FLAGS = "quality.human_flags"
A_MASKS = "masks.top"
A_SCORE = "custom.score"
# On B only
B_FLAGS = "quality.auto_flags"
B_MASKS = "masks.wrist"
B_NOTE = "custom.note"

ONLY_A = {
    A_FLAGS: (5, {"dtype": "int64", "shape": (1,), "names": None}),
    A_MASKS: (EMPTY, feature_spec(["ball", "holder"], (240, 320))),
    A_SCORE: (1.5, {"dtype": "float32", "shape": (1,), "names": None}),
}
ONLY_B = {
    B_FLAGS: (3, {"dtype": "int64", "shape": (1,), "names": None}),
    B_MASKS: (EMPTY, feature_spec(["gripper"], (240, 320))),
    B_NOTE: ("recorded-in-july", {"dtype": "string", "shape": (1,), "names": None}),
}


@pytest.fixture
def pair(tmp_path, lerobot_dataset_factory):
    a = lerobot_dataset_factory(
        root=tmp_path / "a", repo_id=f"{DUMMY_REPO_ID}_a", total_episodes=3, total_frames=30
    )
    b = lerobot_dataset_factory(
        root=tmp_path / "b", repo_id=f"{DUMMY_REPO_ID}_b", total_episodes=2, total_frames=20
    )
    add_features_inplace(a, ONLY_A)
    add_features_inplace(b, ONLY_B)
    return (
        LeRobotDataset(a.repo_id, root=a.root),
        LeRobotDataset(b.repo_id, root=b.root),
    )


def test_without_reconciliation_a_two_sided_difference_is_refused(pair):
    a, b = pair
    with pytest.raises(ValueError):
        merge_into(a, b)


def test_both_sides_gain_the_other_s_columns(pair):
    a, b = pair
    a_frames, b_frames = a.num_frames, b.num_frames

    merge_into(a, b, reconcile_features=True)

    merged = LeRobotDataset(a.repo_id, root=a.root)
    for key in (*ONLY_A, *ONLY_B):
        assert key in merged.meta.features, f"{key} missing after a two-sided reconcile"

    assert merged.num_frames == a_frames + b_frames

    def col(name):
        return [merged[i][name] for i in range(merged.num_frames)]

    # A's own columns: its frames keep their recorded values, B's read neutral.
    flags_a = [int(torch.as_tensor(v).flatten()[0]) for v in col(A_FLAGS)]
    assert flags_a[:a_frames] == [5] * a_frames
    assert flags_a[a_frames:] == [0] * b_frames

    score = [float(torch.as_tensor(v).flatten()[0]) for v in col(A_SCORE)]
    assert score[:a_frames] == pytest.approx([1.5] * a_frames)
    assert score[a_frames:] == pytest.approx([0.0] * b_frames)

    # B's own columns, the other direction: A's frames are the filled ones.
    flags_b = [int(torch.as_tensor(v).flatten()[0]) for v in col(B_FLAGS)]
    assert flags_b[:a_frames] == [0] * a_frames
    assert flags_b[a_frames:] == [3] * b_frames

    notes = [v if isinstance(v, str) else v[0] for v in col(B_NOTE)]
    assert notes[a_frames:] == ["recorded-in-july"] * b_frames
    assert set(notes[:a_frames]) == {""}, "a string column should fill as empty, not as text"


def test_the_mask_columns_stay_decodable_on_both_sides(pair):
    """A mask column is an RLE string whose vocabulary lives in the feature
    metadata. A fill that the codec cannot read would surface far away -- as a
    training run compositing nothing, or a viewer showing no overlay -- so the
    filled rows are decoded here rather than merely counted."""
    a, b = pair
    a_frames = a.num_frames

    merge_into(a, b, reconcile_features=True)
    merged = LeRobotDataset(a.repo_id, root=a.root)

    # The reader strips masks.* once it has composited with them, so the rows
    # are read the way the mask code itself does: with decoding off, which is
    # what LeRobotDataset.delivers_mask_rows names.
    merged.set_video_decoding(False)
    assert merged.delivers_mask_rows

    for key, labels in ((A_MASKS, ["ball", "holder"]), (B_MASKS, ["gripper"])):
        spec = merged.meta.features[key]
        assert spec["mask_labels"] == labels, f"{key} lost its vocabulary: {spec.get('mask_labels')}"
        for i in (0, a_frames, merged.num_frames - 1):
            value = merged[i][key]
            value = value if isinstance(value, str) else value[0]
            decoded = decode_frame(value, labels, tuple(spec["mask_size"]))
            assert isinstance(decoded, dict), f"{key} row {i} did not decode"
            assert decoded == {}, (
                f"{key} row {i} decoded to {sorted(decoded)}; every row here is a fill "
                "or an untouched EMPTY, so none should carry a mask"
            )


def test_the_source_ends_up_with_the_same_feature_set(pair):
    """Reconciliation modifies both datasets. The source keeping a different
    schema would leave it un-mergeable next time for the opposite reason."""
    a, b = pair
    b_root, b_repo = b.root, b.repo_id

    merge_into(a, b, reconcile_features=True)

    merged_a = LeRobotDataset(a.repo_id, root=a.root)
    after_b = LeRobotDataset(b_repo, root=b_root)
    for key in (*ONLY_A, *ONLY_B):
        assert key in after_b.meta.features, f"source is missing {key}"
    assert set(merged_a.meta.features) == set(after_b.meta.features)


def test_a_filled_mask_row_says_never_segmented_not_segmented_and_empty(pair):
    """The distinction the three mask states rest on.

    ``mask_store.NEVER_WRITTEN`` ("") and ``mask_codec.EMPTY`` ("[]") both
    decode to no masks, so nothing downstream crashes either way -- but they do
    not mean the same thing. A side that never carried the column was never
    segmented; writing EMPTY there would claim it was looked at and nothing was
    found, which is exactly the confusion the two sentinels exist to prevent.
    """
    a, b = pair
    a_frames = a.num_frames

    merge_into(a, b, reconcile_features=True)
    merged = LeRobotDataset(a.repo_id, root=a.root)
    merged.set_video_decoding(False)

    def row(i, key):
        v = merged[i][key]
        return v if isinstance(v, str) else v[0]

    # A never had B's mask column: its frames must read as never-segmented.
    assert row(0, B_MASKS) == NEVER_WRITTEN
    # B's own frames keep what B stored, which the fixture left at EMPTY.
    assert row(a_frames, B_MASKS) == EMPTY
    # And the mirror image for A's column.
    assert row(0, A_MASKS) == EMPTY
    assert row(a_frames, A_MASKS) == NEVER_WRITTEN
