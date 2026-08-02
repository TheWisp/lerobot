# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Checkpoints trained before the feature contract must have an upgrade path.

The loader rejects a checkpoint without an ordered contract and tells the
operator to migrate. The migrator only ever recognised the *flat* legacy layout
(``model.safetensors`` with no ``pretrained_model/``), which is the opposite of
what every run since then produces — so the advice pointed at a tool that
structurally could not help, and real 50k-step runs were unloadable with no way
forward.

Backfill sources the ordered names from the run's own training dataset — the
same source training read — and refuses whenever it cannot verify them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
from lerobot.policies.hvla.scripts.hvla_migrate_checkpoints import backfill_contract

ACTION_NAMES = [f"joint_{i}.pos" for i in range(4)]
STATE_NAMES = [f"joint_{i}.pos" for i in range(4)]
CAMERAS = ["observation.images.front", "observation.images.wrist"]


def _pre_contract_checkpoint(tmp_path: Path, *, resize: str = "224x224") -> Path:
    """A standard-layout checkpoint carrying dims but no ordered contract."""
    pretrained = tmp_path / "checkpoints" / "checkpoint-10000" / "pretrained_model"
    pretrained.mkdir(parents=True)
    (pretrained / "config.json").write_text(
        json.dumps(
            {
                "type": "hvla_flow_s1",
                "action_dim": len(ACTION_NAMES),
                "state_dim": len(STATE_NAMES),
                "image_features": dict.fromkeys(CAMERAS, 224),
                "chunk_size": 50,
            }
        )
    )
    (pretrained / "train_config.json").write_text(
        json.dumps({"dataset": {"repo_id": "test/ds"}, "resize_images": resize})
    )
    return pretrained


@pytest.fixture
def fake_dataset(monkeypatch):
    """Stand in for the training dataset's metadata."""

    def _install(features: dict):
        class _Meta:
            def __init__(self, repo_id, *args, **kwargs):
                if repo_id != "test/ds":
                    raise FileNotFoundError(repo_id)
                self.features = features

        monkeypatch.setattr("lerobot.datasets.dataset_metadata.LeRobotDatasetMetadata", _Meta, raising=True)

    return _install


def _features(action_names=ACTION_NAMES, state_names=STATE_NAMES, cameras=CAMERAS) -> dict:
    features = {
        "action": {"names": list(action_names)},
        "observation.state": {"names": list(state_names)},
    }
    for cam in cameras:
        features[cam] = {"dtype": "video"}
    return features


def test_backfilled_checkpoint_loads(tmp_path, fake_dataset):
    """The whole point: a rejected checkpoint loads after backfill."""
    fake_dataset(_features())
    pretrained = _pre_contract_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="ambiguous or missing"):
        FlowMatchingS1Config.from_checkpoint_dict(json.loads((pretrained / "config.json").read_text()))

    assert backfill_contract(pretrained) == "backfilled"

    config = FlowMatchingS1Config.from_checkpoint_dict(json.loads((pretrained / "config.json").read_text()))
    assert config.action_feature_names == ACTION_NAMES
    assert config.state_feature_names == STATE_NAMES
    assert config.robot_state_feature is True
    assert config.image_resize_shape == (224, 224)


def test_backfill_preserves_training_order(tmp_path, fake_dataset):
    """Order is the contract. Alphabetical or robot order would be a silent mis-drive."""
    shuffled = ["z.pos", "a.pos", "m.pos", "b.pos"]
    fake_dataset(_features(action_names=shuffled, state_names=shuffled))
    pretrained = _pre_contract_checkpoint(tmp_path)

    assert backfill_contract(pretrained) == "backfilled"
    assert json.loads((pretrained / "config.json").read_text())["action_feature_names"] == shuffled


def test_dry_run_writes_nothing(tmp_path, fake_dataset):
    fake_dataset(_features())
    pretrained = _pre_contract_checkpoint(tmp_path)
    before = (pretrained / "config.json").read_text()

    assert backfill_contract(pretrained, dry_run=True) == "backfilled"
    assert (pretrained / "config.json").read_text() == before


def test_already_contracted_checkpoint_is_untouched(tmp_path, fake_dataset):
    fake_dataset(_features())
    pretrained = _pre_contract_checkpoint(tmp_path)
    backfill_contract(pretrained)
    after_first = (pretrained / "config.json").read_text()

    assert backfill_contract(pretrained) == "complete"
    assert (pretrained / "config.json").read_text() == after_first


def test_stateless_checkpoint_records_empty_state_layout(tmp_path, fake_dataset):
    fake_dataset(_features(state_names=[]))
    pretrained = _pre_contract_checkpoint(tmp_path)
    config = json.loads((pretrained / "config.json").read_text())
    config["state_dim"] = 0
    (pretrained / "config.json").write_text(json.dumps(config))

    assert backfill_contract(pretrained) == "backfilled"
    written = json.loads((pretrained / "config.json").read_text())
    assert written["robot_state_feature"] is False
    assert written["state_feature_names"] == []


class TestRefusesRatherThanGuesses:
    """Every refusal below would otherwise write an unverified order.

    A wrong order is worse than a failed load: the checkpoint loads, the robot
    moves, and the joints are permuted.
    """

    def test_dimension_mismatch_refuses(self, tmp_path, fake_dataset):
        fake_dataset(_features(action_names=["only_one.pos"]))
        pretrained = _pre_contract_checkpoint(tmp_path)

        assert "refusing to guess" in backfill_contract(pretrained)
        assert "action_feature_names" not in json.loads((pretrained / "config.json").read_text())

    def test_state_dimension_mismatch_refuses(self, tmp_path, fake_dataset):
        fake_dataset(_features(state_names=["a.pos", "b.pos"]))
        pretrained = _pre_contract_checkpoint(tmp_path)

        assert "refusing to guess" in backfill_contract(pretrained)

    def test_camera_mismatch_refuses(self, tmp_path, fake_dataset):
        """A dataset missing a camera the model was trained on is the wrong dataset."""
        fake_dataset(_features(cameras=["observation.images.front"]))
        pretrained = _pre_contract_checkpoint(tmp_path)

        assert "absent from dataset" in backfill_contract(pretrained)

    def test_missing_train_config_refuses(self, tmp_path, fake_dataset):
        fake_dataset(_features())
        pretrained = _pre_contract_checkpoint(tmp_path)
        (pretrained / "train_config.json").unlink()

        assert "cannot verify feature order" in backfill_contract(pretrained)

    def test_unavailable_dataset_refuses(self, tmp_path, fake_dataset):
        fake_dataset(_features())
        pretrained = _pre_contract_checkpoint(tmp_path)
        train_config = json.loads((pretrained / "train_config.json").read_text())
        train_config["dataset"]["repo_id"] = "gone/missing"
        (pretrained / "train_config.json").write_text(json.dumps(train_config))

        assert "is unavailable" in backfill_contract(pretrained)

    def test_unparsable_resize_refuses(self, tmp_path, fake_dataset):
        fake_dataset(_features())
        pretrained = _pre_contract_checkpoint(tmp_path, resize="big")

        assert "cannot recover the input resolution" in backfill_contract(pretrained)
