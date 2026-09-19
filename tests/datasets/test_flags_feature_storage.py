#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Storing a bitset column: it must survive every write path, and the object
that wrote it must not disagree with the bytes it just wrote.

``LeRobotDataset`` is a read/write facade, not a view -- it owns a writer and
``dataset_tools`` mutates schema through a live instance. So the risks worth
testing are the ones a read-only object would not have: a vocabulary silently
dropped by a writer, and an in-memory object left describing a schema that no
longer matches the disk.
"""

import json

import pytest
import torch

from lerobot.datasets.dataset_tools import (
    add_features_inplace,
    remove_features_inplace,
    rename_features_inplace,
)
from lerobot.datasets.feature_utils import is_categorical_feature
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FPS = 10
FRAMES = 6
LABELS = ["blurry", "fumble"]
QUALITY_SPEC = {"dtype": "int64", "shape": [1], "names": None, "flags": LABELS}
BASE_FEATURES = {
    "action": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
    "observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
}


def info_json(root):
    return json.loads((root / "meta" / "info.json").read_text())


@pytest.fixture
def plain_root(tmp_path):
    """A finalized two-episode dataset with no bitset column yet."""
    root = tmp_path / "plain"
    dataset = LeRobotDataset.create(
        repo_id="test/store", fps=FPS, root=root, features=BASE_FEATURES, use_videos=False
    )
    for _ in range(2):
        for index in range(FRAMES):
            value = float(index)
            dataset.add_frame(
                {
                    "action": torch.tensor([value, value]),
                    "observation.state": torch.tensor([value, value]),
                    "task": "storing",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    return root


@pytest.fixture
def flagled_root(tmp_path):
    """A finalized dataset that declared the bitset column at creation."""
    root = tmp_path / "flagled"
    features = {**BASE_FEATURES, "quality": QUALITY_SPEC}
    dataset = LeRobotDataset.create(
        repo_id="test/store2", fps=FPS, root=root, features=features, use_videos=False
    )
    for index in range(FRAMES):
        value = float(index)
        dataset.add_frame(
            {
                "action": torch.tensor([value, value]),
                "observation.state": torch.tensor([value, value]),
                "quality": torch.tensor([0b11 if index == 2 else 0]),
                "task": "storing",
            }
        )
    dataset.save_episode()
    dataset.finalize()
    return root


# --- the vocabulary survives every writer -----------------------------------


def test_creating_a_dataset_persists_the_vocabulary(flagled_root):
    """Bit i means flags[i]. A writer that dropped the list would leave every
    stored value undecodable, and the symptom would be "nothing matched"
    rather than anything that looks like data loss."""
    assert info_json(flagled_root)["features"]["quality"]["flags"] == LABELS


def test_adding_the_column_to_an_existing_dataset_persists_the_vocabulary(plain_root):
    """The path the GUI takes: the column is added long after recording."""
    dataset = LeRobotDataset(repo_id="test/store", root=plain_root)
    add_features_inplace(dataset, features={"quality": (0, QUALITY_SPEC)})
    assert info_json(plain_root)["features"]["quality"]["flags"] == LABELS


def test_renaming_the_column_keeps_the_vocabulary(plain_root):
    dataset = LeRobotDataset(repo_id="test/store", root=plain_root)
    add_features_inplace(dataset, features={"quality": (0, QUALITY_SPEC)})
    rename_features_inplace(dataset, renames={"quality": "frame_quality"})
    assert info_json(plain_root)["features"]["frame_quality"]["flags"] == LABELS
    assert "quality" not in info_json(plain_root)["features"]


def test_removing_the_column_leaves_no_trace(plain_root):
    dataset = LeRobotDataset(repo_id="test/store", root=plain_root)
    add_features_inplace(dataset, features={"quality": (0, QUALITY_SPEC)})
    remove_features_inplace(dataset, names="quality")
    assert "quality" not in info_json(plain_root)["features"]


def test_values_survive_a_reopen(flagled_root):
    reopened = LeRobotDataset(repo_id="test/store2", root=flagled_root)
    assert reopened.meta.features["quality"]["flags"] == LABELS
    assert int(reopened[2]["quality"]) == 0b11
    assert int(reopened[0]["quality"]) == 0


def test_a_new_column_starts_with_nothing_set(plain_root):
    """The fill value is not a flag; an unannotated frame must decode to []."""
    dataset = LeRobotDataset(repo_id="test/store", root=plain_root)
    add_features_inplace(dataset, features={"quality": (0, QUALITY_SPEC)})
    reopened = LeRobotDataset(repo_id="test/store", root=plain_root)
    assert all(int(reopened[i]["quality"]) == 0 for i in range(FRAMES))


# --- consistency: the object must not disagree with the disk ----------------


def test_a_schema_mutation_leaves_the_object_self_consistent(plain_root):
    """The regression this guards is silent, not loud.

    ``add_features_inplace`` used to rebind ``dataset.meta`` while leaving the
    reader holding pre-mutation metadata *and* a pre-mutation table. The
    reader's copy is what selects columns on load, so the object reported a
    column that its own reads did not return.
    """
    dataset = LeRobotDataset(repo_id="test/store", root=plain_root)
    dataset[0]  # force the table to load, so there is a stale copy to catch
    add_features_inplace(dataset, features={"quality": (0, QUALITY_SPEC)})

    assert "quality" in dataset.meta.features
    assert "quality" in dataset.reader._meta.features, "the reader kept stale metadata"
    assert "quality" in dataset[0], "metadata reports a column that reads do not return"
    assert dataset.meta is dataset.reader._meta, "two metadata objects will drift apart again"


def test_a_removal_is_visible_through_the_same_object(plain_root):
    dataset = LeRobotDataset(repo_id="test/store", root=plain_root)
    add_features_inplace(dataset, features={"quality": (0, QUALITY_SPEC)})
    dataset[0]
    remove_features_inplace(dataset, names="quality")
    assert "quality" not in dataset.meta.features
    assert "quality" not in dataset[0], "a removed column is still being served"


def test_a_rename_is_visible_through_the_same_object(plain_root):
    dataset = LeRobotDataset(repo_id="test/store", root=plain_root)
    add_features_inplace(dataset, features={"quality": (0, QUALITY_SPEC)})
    dataset[0]
    rename_features_inplace(dataset, renames={"quality": "frame_quality"})
    item = dataset[0]
    assert "frame_quality" in item
    assert "quality" not in item


def test_a_second_mutation_builds_on_the_first(plain_root):
    """Compounding is where a stale schema turns into a failed cast rather than
    a missing column, so the second mutation must see the first."""
    dataset = LeRobotDataset(repo_id="test/store", root=plain_root)
    dataset[0]
    add_features_inplace(dataset, features={"quality": (0, QUALITY_SPEC)})
    add_features_inplace(dataset, features={"note": (0, {"dtype": "int64", "shape": [1], "names": None})})
    item = dataset[0]
    assert {"quality", "note"} <= set(item)
    assert {"quality", "note"} <= set(dataset.meta.features)


# --- reentrance: a write in the middle of a read must refuse ----------------


def test_a_read_mode_dataset_refuses_to_be_written(plain_root):
    """No lock, just a refusal: the object opened for reading has no writer,
    and accepting the frame would drop it silently."""
    dataset = LeRobotDataset(repo_id="test/store", root=plain_root)
    assert dataset.writer is None
    with pytest.raises(RuntimeError, match="read-only dataset"):
        dataset.add_frame({"action": torch.zeros(2), "observation.state": torch.zeros(2), "task": "storing"})


def test_a_read_mode_dataset_refuses_to_save_an_episode(plain_root):
    dataset = LeRobotDataset(repo_id="test/store", root=plain_root)
    with pytest.raises(RuntimeError, match="read-only dataset"):
        dataset.save_episode()


def test_an_undeclared_bit_cannot_be_written(tmp_path):
    """The value validator is reachable from add_frame, not only from unit
    tests: an undeclared bit decodes to no flag and would survive every round
    trip meaning nothing."""
    root = tmp_path / "reject"
    features = {**BASE_FEATURES, "quality": QUALITY_SPEC}
    dataset = LeRobotDataset.create(
        repo_id="test/reject", fps=FPS, root=root, features=features, use_videos=False
    )
    with pytest.raises(ValueError, match="outside the 2 declared flags"):
        dataset.add_frame(
            {
                "action": torch.zeros(2),
                "observation.state": torch.zeros(2),
                "quality": torch.tensor([0b100]),
                "task": "storing",
            }
        )


def test_a_bitset_declares_names_as_null(plain_root):
    """``names`` describes the dimensions of a vector or the classes of a
    categorical, so a bitset has nothing to put there -- but it declares the
    key anyway, as every other feature does.

    Several readers index ``names`` directly rather than with ``.get``, so
    omitting it would break them somewhere far from the feature that left it
    out. The flags live in ``flags``; ``names`` stays null and unused.
    """
    dataset = LeRobotDataset(repo_id="test/store", root=plain_root)
    add_features_inplace(dataset, features={"quality": (0, QUALITY_SPEC)})

    assert QUALITY_SPEC["names"] is None
    assert dataset.meta.names["quality"] is None
    assert not is_categorical_feature(dataset.meta.features["quality"])
    reopened = LeRobotDataset(repo_id="test/store", root=plain_root)
    assert reopened.meta.features["quality"]["flags"] == LABELS


@pytest.mark.parametrize("value", [0b00, 0b01, 0b10, 0b11])
def test_every_bit_pattern_survives_the_writer(tmp_path, value):
    """The unit tests prove the encoding; this proves the column actually
    carries it through parquet and back."""
    root = tmp_path / f"pattern{value}"
    features = {**BASE_FEATURES, "quality": QUALITY_SPEC}
    dataset = LeRobotDataset.create(
        repo_id="test/pattern", fps=FPS, root=root, features=features, use_videos=False
    )
    dataset.add_frame(
        {
            "action": torch.zeros(2),
            "observation.state": torch.zeros(2),
            "quality": torch.tensor([value]),
            "task": "storing",
        }
    )
    dataset.save_episode()
    dataset.finalize()

    reopened = LeRobotDataset(repo_id="test/pattern", root=root)
    assert int(reopened[0]["quality"]) == value
