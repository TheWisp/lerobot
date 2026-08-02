# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The checkpoint writer and reader must agree, and only a test can hold them so.

Contract tests covered the reader against hand-authored dicts, and the writer
lived as a closure inside ``train()`` — unreachable from the suite. Dropping a
contract field from the writer therefore left every test green while every new
checkpoint became unloadable: the failure would surface only at the next
inference run, hours of training later.

These pin the two ends against each other, which is the only assertion that
actually protects the round trip.
"""

from __future__ import annotations

import json

import pytest

from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
from lerobot.policies.hvla.s1.flow_matching.train import checkpoint_config_dict

ACTION_NAMES = ["a.pos", "b.pos", "c.pos"]
STATE_NAMES = ["s0.pos", "s1.pos"]


def _config(**overrides) -> FlowMatchingS1Config:
    base = {
        "action_dim": len(ACTION_NAMES),
        "action_feature_names": list(ACTION_NAMES),
        "robot_state_feature": True,
        "state_dim": len(STATE_NAMES),
        "state_feature_names": list(STATE_NAMES),
        "image_features": {"observation.images.front": 224},
        "image_resize_shape": (224, 224),
    }
    base.update(overrides)
    return FlowMatchingS1Config(**base)


def test_what_training_writes_is_what_inference_can_load():
    """The round trip, end to end — the invariant the contract exists for."""
    written = checkpoint_config_dict(_config())

    # Survives serialization: a checkpoint is JSON on disk, not a live dict.
    loaded = FlowMatchingS1Config.from_checkpoint_dict(json.loads(json.dumps(written)))

    assert loaded.action_feature_names == ACTION_NAMES
    assert loaded.state_feature_names == STATE_NAMES
    assert loaded.action_dim == len(ACTION_NAMES)
    assert loaded.state_dim == len(STATE_NAMES)
    assert loaded.robot_state_feature is True
    assert loaded.image_resize_shape == (224, 224)


def test_stateless_run_round_trips_as_stateless():
    written = checkpoint_config_dict(_config(robot_state_feature=False, state_dim=0, state_feature_names=[]))
    loaded = FlowMatchingS1Config.from_checkpoint_dict(json.loads(json.dumps(written)))

    assert loaded.robot_state_feature is False
    assert loaded.state_feature_names == []


def test_writer_stamps_the_current_contract_version():
    """A version drift between writer and reader silently rejects every checkpoint."""
    written = checkpoint_config_dict(_config())
    assert written["feature_contract_version"] == FlowMatchingS1Config.FEATURE_CONTRACT_VERSION


@pytest.mark.parametrize("dropped", ["action_feature_names", "state_feature_names", "image_resize_shape"])
def test_dropping_a_contract_field_breaks_the_load(dropped):
    """The mutation the old closure allowed to ship: this is what must fail loudly."""
    written = checkpoint_config_dict(_config())
    assert dropped in written, f"writer no longer emits {dropped} — the contract regressed"

    del written[dropped]
    with pytest.raises(ValueError, match="ambiguous or missing"):
        FlowMatchingS1Config.from_checkpoint_dict(written)


class TestRobotStateFeatureIsRecoveredNotGuessed:
    """``robot_state_feature`` is the one field the loader may reconstruct.

    First-generation contract checkpoints predate the flag but already carry
    complete ordered state metadata, and a self-consistent ``state_dim`` /
    ``state_feature_names`` pair determines it exactly — no robot identity and
    no runtime order is involved. The writer still emits it; these pin that the
    reconstruction stays exact rather than becoming a default.
    """

    def test_recovered_when_state_metadata_is_self_consistent(self):
        written = checkpoint_config_dict(_config())
        del written["robot_state_feature"]

        assert FlowMatchingS1Config.from_checkpoint_dict(written).robot_state_feature is True

    def test_recovered_as_false_for_an_explicitly_empty_state_layout(self):
        written = checkpoint_config_dict(
            _config(robot_state_feature=False, state_dim=0, state_feature_names=[])
        )
        del written["robot_state_feature"]

        assert FlowMatchingS1Config.from_checkpoint_dict(written).robot_state_feature is False

    def test_refused_when_state_metadata_disagrees_with_itself(self):
        """dim and names in conflict is ambiguous — the loader must not pick one."""
        written = checkpoint_config_dict(_config(state_dim=2, state_feature_names=["s0.pos", "s1.pos"]))
        del written["robot_state_feature"]
        written["state_dim"] = 5

        with pytest.raises(ValueError, match="ambiguous or missing"):
            FlowMatchingS1Config.from_checkpoint_dict(written)

    def test_refused_when_state_metadata_is_absent_entirely(self):
        written = checkpoint_config_dict(_config())
        del written["robot_state_feature"]
        del written["state_feature_names"]

        with pytest.raises(ValueError, match="ambiguous or missing"):
            FlowMatchingS1Config.from_checkpoint_dict(written)


def test_training_order_is_preserved_verbatim():
    """Not sorted, not the robot's order — the order training saw."""
    shuffled = ["z.pos", "a.pos", "m.pos"]
    written = checkpoint_config_dict(_config(action_feature_names=shuffled))

    loaded = FlowMatchingS1Config.from_checkpoint_dict(json.loads(json.dumps(written)))
    assert loaded.action_feature_names == shuffled
