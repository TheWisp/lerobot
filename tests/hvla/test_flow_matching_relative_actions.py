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

"""Training/inference contract tests for Flow S1 relative arm actions."""

from __future__ import annotations

import torch

from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
from lerobot.policies.hvla.s1.flow_matching.model import FlowMatchingS1Policy
from lerobot.policies.hvla.s1.flow_matching.train import (
    FlowMatchingDataset,
    split_train_validation_frames_by_episode,
)
from lerobot.policies.hvla.s1.protocol import ACTION_PREFIX_KEY

ACTION_NAMES = ["joint.pos", "gripper.pos"]
STATE_NAMES = ["joint.pos", "joint.vel", "gripper.pos"]


class _HFDataset:
    column_names = ["action", "observation.state", "episode_index"]

    def __init__(self) -> None:
        self._columns = {
            "action": [[10.0, 50.0], [12.0, 55.0], [15.0, 60.0]],
            "observation.state": [[10.0, 0.0, 50.0], [11.0, 0.0, 51.0], [12.0, 0.0, 52.0]],
            "episode_index": [0, 0, 0],
        }

    def __getitem__(self, key: str):
        return self._columns[key]


class _Dataset:
    def __init__(self) -> None:
        self.hf_dataset = _HFDataset()

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "action": torch.tensor(self.hf_dataset["action"][index]),
            "observation.state": torch.tensor(self.hf_dataset["observation.state"][index]),
        }


def test_relative_stats_cover_valid_chunk_tokens_and_keep_gripper_absolute():
    dataset = FlowMatchingDataset(
        _Dataset(),
        s2_latents=None,
        chunk_size=2,
        action_feature_names=ACTION_NAMES,
        state_feature_names=STATE_NAMES,
        use_relative_actions=True,
    )

    # Valid arm residuals are [0, 2], [1, 4], [3]; padded tail tokens
    # are deliberately excluded.  Gripper targets stay absolute.
    torch.testing.assert_close(dataset.action_mean, torch.tensor([2.0, 56.0]))

    sample = dataset[0]
    denormalized = sample["action"] * dataset.action_std + dataset.action_mean
    torch.testing.assert_close(denormalized, torch.tensor([[0.0, 50.0], [2.0, 55.0]]))


def test_relative_checkpoint_reanchors_arm_output_and_rtc_prefix_by_feature_name():
    config = FlowMatchingS1Config(
        action_dim=2,
        action_feature_names=ACTION_NAMES,
        use_relative_actions=True,
        chunk_size=2,
        hidden_dim=8,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
        use_dino_backbone=False,
        image_features={},
        robot_state_feature=True,
        state_dim=3,
        state_feature_names=STATE_NAMES,
    )
    policy = FlowMatchingS1Policy(config)
    policy._action_mean = torch.zeros(2)
    policy._action_std = torch.ones(2)
    policy._state_mean = torch.zeros(3)
    policy._state_std = torch.ones(3)
    captured = {}

    def fake_sample_actions(batch, *, num_steps, action_prefix, prefix_len, context):
        captured["prefix"] = action_prefix.clone()
        captured["state"] = batch["observation.state"].clone()
        assert prefix_len == 1
        return torch.tensor([[[2.0, 40.0], [2.5, 42.0]]])

    policy.model.sample_actions = fake_sample_actions
    output = policy.predict_action_chunk(
        {
            "observation.state": torch.tensor([[10.0, 99.0, 30.0]]),
            ACTION_PREFIX_KEY: torch.tensor([[[13.0, 41.0]]]),
        }
    )

    # Arm residuals are anchored to joint.pos=10, not the interleaved vel=99.
    # Gripper remains absolute; the same conversion is applied to the RTC prefix.
    torch.testing.assert_close(output, torch.tensor([[[12.0, 40.0], [12.5, 42.0]]]))
    torch.testing.assert_close(captured["prefix"], torch.tensor([[[3.0, 41.0]]]))
    torch.testing.assert_close(captured["state"], torch.tensor([[10.0, 99.0, 30.0]]))


def test_validation_split_holds_out_complete_episodes_deterministically():
    episodes = [7, 7, 12, 12, 12, 20, 20, 31]

    first = split_train_validation_frames_by_episode(episodes, validation_fraction=0.25, seed=1337)
    second = split_train_validation_frames_by_episode(episodes, validation_fraction=0.25, seed=1337)

    assert first == second
    train_frames, validation_frames, validation_episode_ids = first
    train_episode_ids = {episodes[index] for index in train_frames}
    assert train_episode_ids.isdisjoint(validation_episode_ids)
    assert {episodes[index] for index in validation_frames} == set(validation_episode_ids)
    assert len(validation_episode_ids) == 1
