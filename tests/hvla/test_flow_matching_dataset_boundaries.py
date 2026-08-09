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

"""Regression tests for HVLA Flow S1 action chunks at episode boundaries."""

from __future__ import annotations

import torch
import pytest

from lerobot.policies.hvla.s1.flow_matching.train import FlowMatchingDataset


class _V3HFDataset:
    column_names = ["action", "observation.state", "episode_index"]

    def __init__(self) -> None:
        self._columns = {
            "action": [[0.0], [1.0], [2.0], [100.0], [101.0], [102.0]],
            "observation.state": [[0.0], [1.0], [2.0], [100.0], [101.0], [102.0]],
            "episode_index": [7, 7, 7, 12, 12, 12],
        }

    def __getitem__(self, key: str):
        return self._columns[key]


class _V3LeRobotDataset:
    """Current LeRobot contract: episode_index column, no episode_data_index."""

    def __init__(self) -> None:
        self.hf_dataset = _V3HFDataset()

    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "action": torch.tensor(self.hf_dataset["action"][index]),
            "observation.state": torch.tensor(self.hf_dataset["observation.state"][index]),
        }


def test_action_chunk_pads_at_v3_episode_boundary_instead_of_reading_next_demo():
    dataset = FlowMatchingDataset(_V3LeRobotDataset(), s2_latents=None, chunk_size=4)

    sample = dataset[1]
    denormalized_actions = sample["action"] * dataset.action_std + dataset.action_mean

    assert denormalized_actions[:, 0].tolist() == [1.0, 2.0, 2.0, 2.0]
    assert sample["action_is_pad"].tolist() == [False, False, True, True]


def test_action_chunk_starts_normally_at_next_v3_episode():
    dataset = FlowMatchingDataset(_V3LeRobotDataset(), s2_latents=None, chunk_size=3)

    sample = dataset[3]
    denormalized_actions = sample["action"] * dataset.action_std + dataset.action_mean

    assert denormalized_actions[:, 0].tolist() == [100.0, 101.0, 102.0]
    assert sample["action_is_pad"].tolist() == [False, False, False]


def test_normalization_statistics_can_be_fit_on_train_episodes_only():
    dataset = FlowMatchingDataset(
        _V3LeRobotDataset(),
        s2_latents=None,
        chunk_size=2,
        statistics_indices=[0, 1, 2],
    )

    assert dataset.action_mean.item() == 1.0
    assert dataset.state_mean.item() == 1.0


class _NoBoundaryHFDataset:
    """Neither contract: no episode_data_index attribute, no episode_index column."""

    column_names = ["action", "observation.state"]

    def __init__(self) -> None:
        self._columns = {
            "action": [[0.0], [1.0], [2.0]],
            "observation.state": [[0.0], [1.0], [2.0]],
        }

    def __getitem__(self, key: str):
        return self._columns[key]


class _NoBoundaryLeRobotDataset:
    def __init__(self) -> None:
        self.hf_dataset = _NoBoundaryHFDataset()

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "action": torch.tensor(self.hf_dataset["action"][index]),
            "observation.state": torch.tensor(self.hf_dataset["observation.state"][index]),
        }


def test_training_refuses_to_start_when_boundaries_cannot_be_derived():
    """The original defect was a silent fallback, so silence is what must not recur.

    Deriving boundaries correctly is only half the fix. If a future dataset
    exposes neither contract, falling through to a single global interval would
    reproduce the bug exactly — with no error, no warning, and a 10% target
    corruption rate visible only as poor policy behaviour. Failing loudly is the
    property under test; without it, nothing in this file would notice.
    """
    with pytest.raises(ValueError, match="episode boundaries for every frame"):
        FlowMatchingDataset(_NoBoundaryLeRobotDataset(), s2_latents=None, chunk_size=2)
