# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""A recorded dataset must describe the columns it actually ships.

Once ``run_s1`` resolves feature order from the checkpoint rather than the
connected robot, the two can legitimately differ — that is the whole point of
the contract. The recording path built its schema from the robot while filling
frames in checkpoint order, so an episode recorded through a re-ordered
checkpoint carried permuted columns under confident-looking names. Feeding that
dataset back into training bakes the permutation into the next checkpoint's
contract: the exact failure the contract exists to prevent, laundered through a
dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.policies.hvla.s1_process import _add_frame_to_dataset, _create_recording_dataset

# The robot's own order. A checkpoint trained elsewhere may name the same
# joints in a different order — supported, and the reason this test exists.
ROBOT_ACTION_ORDER = ["base.turn", "arm.lift", "tool.close"]
CHECKPOINT_ACTION_ORDER = ["tool.close", "base.turn", "arm.lift"]


class _FakeRobot:
    robot_type = "fake"

    @property
    def action_features(self):
        return dict.fromkeys(ROBOT_ACTION_ORDER, float)

    @property
    def observation_features(self):
        features = dict.fromkeys(ROBOT_ACTION_ORDER, float)
        features["front"] = (480, 640, 3)
        return features


@pytest.fixture
def captured_features(monkeypatch):
    """Capture the schema without creating a dataset on disk."""
    captured = {}

    def _fake_create(repo_id, fps, features, robot_type):
        captured["features"] = features
        captured["robot_type"] = robot_type

        class _Meta:
            total_episodes = 0

        class _Dataset:
            meta = _Meta()

        return _Dataset()

    monkeypatch.setattr("lerobot.policies.hvla.s1_process._create_or_resume_dataset", _fake_create)
    return captured


def test_schema_uses_checkpoint_order_not_robot_order(captured_features):
    _create_recording_dataset(
        "test/rec",
        30,
        _FakeRobot(),
        "task",
        action_names=CHECKPOINT_ACTION_ORDER,
        state_names=CHECKPOINT_ACTION_ORDER,
    )

    features = captured_features["features"]
    assert features["action"]["names"] == CHECKPOINT_ACTION_ORDER
    assert features["action"]["shape"] == (3,)
    assert features["observation.state"]["names"] == CHECKPOINT_ACTION_ORDER
    assert "observation.images.front" in features


def test_state_schema_is_independent_of_action_schema(captured_features):
    """A stateless-action / stated-observation checkpoint must not be forced to match."""
    state_only = ["arm.lift", "base.turn"]
    _create_recording_dataset(
        "test/rec", 30, _FakeRobot(), "task", action_names=CHECKPOINT_ACTION_ORDER, state_names=state_only
    )

    features = captured_features["features"]
    assert features["action"]["names"] == CHECKPOINT_ACTION_ORDER
    assert features["observation.state"]["names"] == state_only
    assert features["observation.state"]["shape"] == (2,)


def test_frames_are_written_in_the_declared_state_order():
    """The frame's state vector follows the declared names, not the obs dict's order."""
    written = []

    class _Dataset:
        def add_frame(self, frame):
            written.append(frame)

    # obs arrives in the robot's order; the declared layout is the checkpoint's.
    obs = {"base.turn": 1.0, "arm.lift": 2.0, "tool.close": 3.0}
    _add_frame_to_dataset(_Dataset(), obs, np.zeros(3, dtype=np.float32), CHECKPOINT_ACTION_ORDER, "task")

    assert written[0]["observation.state"].tolist() == [3.0, 1.0, 2.0]


def test_schema_and_frames_agree_end_to_end(captured_features):
    """The invariant that was broken: column i of every frame is feature i of the schema."""
    written = []

    class _Dataset:
        def add_frame(self, frame):
            written.append(frame)

    _create_recording_dataset(
        "test/rec",
        30,
        _FakeRobot(),
        "task",
        action_names=CHECKPOINT_ACTION_ORDER,
        state_names=CHECKPOINT_ACTION_ORDER,
    )
    declared = captured_features["features"]["observation.state"]["names"]

    obs = {"base.turn": 1.0, "arm.lift": 2.0, "tool.close": 3.0}
    _add_frame_to_dataset(_Dataset(), obs, np.zeros(3, dtype=np.float32), declared, "task")

    state = written[0]["observation.state"].tolist()
    assert [obs[name] for name in declared] == state
