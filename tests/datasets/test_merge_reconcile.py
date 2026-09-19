# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""A reconciled merge, end to end, checked on the result rather than the call.

Reconciliation rewrites BOTH datasets before merging, so the thing worth
asserting is not that it ran but what it left behind: the column that existed
on one side only is present on both, the rows that never had it read as "not
recorded", and the rows that did keep their own values.
"""

from __future__ import annotations

import pytest
import torch

from lerobot.datasets.dataset_tools import add_features_inplace, merge_into
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tests.fixtures.constants import DUMMY_REPO_ID

FLAG = "quality.human_flags"
FLAG_SPEC = {"dtype": "int64", "shape": (1,), "names": None}
LABELLED_VALUE = 7


def _pair(tmp_path, factory, episodes=(3, 2)):
    """A labelled dataset and an unlabelled one, differing by exactly one column."""
    labelled = factory(
        root=tmp_path / "labelled",
        repo_id=f"{DUMMY_REPO_ID}_labelled",
        total_episodes=episodes[0],
        total_frames=episodes[0] * 10,
    )
    plain = factory(
        root=tmp_path / "plain",
        repo_id=f"{DUMMY_REPO_ID}_plain",
        total_episodes=episodes[1],
        total_frames=episodes[1] * 10,
    )
    add_features_inplace(labelled, {FLAG: (LABELLED_VALUE, dict(FLAG_SPEC))})
    return LeRobotDataset(labelled.repo_id, root=labelled.root), plain


def test_without_reconciliation_the_merge_still_refuses(tmp_path, lerobot_dataset_factory):
    """The complement. If a merge went through regardless, every assertion
    below would be about a code path nobody reaches."""
    labelled, plain = _pair(tmp_path, lerobot_dataset_factory)
    with pytest.raises(ValueError):
        merge_into(labelled, plain)


def test_the_missing_column_is_filled_and_the_recorded_one_is_kept(tmp_path, lerobot_dataset_factory):
    labelled, plain = _pair(tmp_path, lerobot_dataset_factory)
    labelled_frames, plain_frames = labelled.num_frames, plain.num_frames

    merge_into(labelled, plain, reconcile_features=True)
    merged = LeRobotDataset(labelled.repo_id, root=labelled.root)

    assert FLAG in merged.meta.features, "the reconciled column is missing from the merged dataset"
    assert merged.num_frames == labelled_frames + plain_frames

    values = [int(torch.as_tensor(merged[i][FLAG]).flatten()[0]) for i in range(merged.num_frames)]
    recorded = [v for v in values if v == LABELLED_VALUE]
    filled = [v for v in values if v == 0]

    # Every frame is one or the other -- nothing was corrupted into a third value.
    assert len(recorded) + len(filled) == merged.num_frames, f"unexpected values: {set(values)}"
    # The side that was labelled keeps its labels; the side that never was reads
    # as not-recorded rather than as a value someone might filter on.
    assert len(recorded) == labelled_frames
    assert len(filled) == plain_frames


def test_the_source_gains_the_column_too(tmp_path, lerobot_dataset_factory):
    """Reconciliation is documented as modifying BOTH datasets. If it quietly
    stopped doing so, the source would be left un-mergeable next time and the
    docstring would be describing something that no longer happens."""
    labelled, plain = _pair(tmp_path, lerobot_dataset_factory)
    plain_root, plain_repo = plain.root, plain.repo_id

    merge_into(labelled, plain, reconcile_features=True)

    assert FLAG in LeRobotDataset(plain_repo, root=plain_root).meta.features


def test_a_genuine_conflict_is_still_refused(tmp_path, lerobot_dataset_factory):
    """A column both sides declare but describe differently is not a gap to
    fill -- a neutral value would be a fabrication, so reconciliation must not
    quietly paper over it."""
    labelled, plain = _pair(tmp_path, lerobot_dataset_factory)
    add_features_inplace(plain, {FLAG: (0.5, {"dtype": "float32", "shape": (1,), "names": None})})
    plain = LeRobotDataset(plain.repo_id, root=plain.root)

    with pytest.raises(ValueError, match="different things"):
        merge_into(labelled, plain, reconcile_features=True)


def test_the_merge_says_which_columns_it_filled(tmp_path, lerobot_dataset_factory, caplog):
    """A reconciled merge that succeeds is the case with no other trace: the
    run looks like any other merge while a not-recorded column has appeared.
    A later filter over that column passes every filled frame silently, so
    which ones they are has to be recoverable afterwards."""
    import logging

    labelled, plain = _pair(tmp_path, lerobot_dataset_factory)
    with caplog.at_level(logging.INFO):
        merge_into(labelled, plain, reconcile_features=True)

    reconciled = [r.getMessage() for r in caplog.records if "Reconciled features" in r.getMessage()]
    assert reconciled, "a reconciled merge left no record of what it changed"
    message = reconciled[0]
    assert FLAG in message, f"the filled column is not named: {message}"
    assert plain.repo_id in message, f"the side it was added to is not named: {message}"
    assert "NOT RECORDED" in message


def test_a_plain_merge_says_nothing_about_reconciling(tmp_path, lerobot_dataset_factory, caplog):
    """The complement: a log line emitted unconditionally would carry no
    information about whether anything was reconciled."""
    import logging

    plain_a = lerobot_dataset_factory(
        root=tmp_path / "a", repo_id=f"{DUMMY_REPO_ID}_a", total_episodes=2, total_frames=20
    )
    plain_b = lerobot_dataset_factory(
        root=tmp_path / "b", repo_id=f"{DUMMY_REPO_ID}_b", total_episodes=2, total_frames=20
    )
    with caplog.at_level(logging.INFO):
        merge_into(plain_a, plain_b, reconcile_features=True)

    assert not [r for r in caplog.records if "Reconciled features" in r.getMessage()]


def test_the_gui_path_reconciles_all_the_way_to_disk(tmp_path, lerobot_dataset_factory):
    """The whole chain, unstubbed: the flag the dialog sends, through the GUI's
    canonical merge entry point, down to the bytes on disk.

    The endpoint tests in tests/gui/test_merge_reconcile_endpoint.py stop at
    ``merge_dataset_into`` because they stub the merge. This picks up there and
    runs the real one, so no link in between is taken on trust.
    """
    import asyncio

    from lerobot.gui.api._edits_core import merge_dataset_into

    labelled, plain = _pair(tmp_path, lerobot_dataset_factory)

    class _State:
        """Only what merge_dataset_into reaches for: the dataset registry and
        one real asyncio lock per id, which it takes for the merge's duration."""

        def __init__(self, mapping):
            self.datasets = mapping
            self._locks = {k: asyncio.Lock() for k in mapping}

        def is_locked(self, dataset_id):
            return self._locks[dataset_id].locked()

        def get_lock(self, dataset_id):
            return self._locks[dataset_id]

    state = _State({labelled.repo_id: labelled, plain.repo_id: plain})
    result = asyncio.run(
        merge_dataset_into(state, plain.repo_id, labelled.repo_id, force=False, reconcile_features=True)
    )

    assert result["source_episodes_merged"] == plain.num_episodes

    merged = LeRobotDataset(labelled.repo_id, root=labelled.root)
    values = [int(torch.as_tensor(merged[i][FLAG]).flatten()[0]) for i in range(merged.num_frames)]
    assert sorted(set(values)) == [0, LABELLED_VALUE]
    assert values.count(0) == plain.num_frames


def test_the_gui_path_without_the_flag_still_refuses(tmp_path, lerobot_dataset_factory):
    """The complement for the chain above -- without it, an implementation that
    always reconciled would pass."""
    import asyncio

    from lerobot.gui.api._edits_core import EditValidationError, merge_dataset_into

    labelled, plain = _pair(tmp_path, lerobot_dataset_factory)

    class _State:
        def __init__(self, mapping):
            self.datasets = mapping
            self._locks = {k: asyncio.Lock() for k in mapping}

        def is_locked(self, dataset_id):
            return self._locks[dataset_id].locked()

        def get_lock(self, dataset_id):
            return self._locks[dataset_id]

    state = _State({labelled.repo_id: labelled, plain.repo_id: plain})
    with pytest.raises(EditValidationError):
        asyncio.run(merge_dataset_into(state, plain.repo_id, labelled.repo_id))
