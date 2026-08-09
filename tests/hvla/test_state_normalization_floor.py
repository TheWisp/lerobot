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

"""Regression tests for HVLA Flow S1's position-state normalization.

The failure being protected against is not a generic divide-by-zero. A valid
embodiment can contain a joint that did not move in a particular dataset. A
1e-6 z-score scale then turns normal encoder noise or a small posture mismatch
into an enormous model input. The treatment must therefore floor only named
position observations while preserving both action normalization and the
legacy default.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from lerobot.policies.hvla.s1.flow_matching.train import FlowMatchingDataset, seed_training

STATE_NAMES = ["left_stationary.pos", "right_moving.pos", "left_stationary.vel"]


class _FakeHFDataset:
    column_names = ["action", "observation.state", "episode_index"]

    def __init__(self) -> None:
        self._columns = {
            "action": [
                [7.0, 0.0],
                [7.0, 2.0],
                [7.0, 4.0],
                [7.0, 6.0],
            ],
            "observation.state": [
                [10.0, 0.0, 0.0],
                [10.0, 2.0, 0.0],
                [10.0, 4.0, 0.0],
                [10.0, 6.0, 0.0],
            ],
            "episode_index": [0, 0, 0, 0],
        }

    def __getitem__(self, key: str):
        return self._columns[key]


class _FakeLeRobotDataset:
    def __init__(self) -> None:
        self.hf_dataset = _FakeHFDataset()

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "action": torch.tensor(self.hf_dataset["action"][index]),
            "observation.state": torch.tensor(self.hf_dataset["observation.state"][index]),
        }


def _dataset(*, floor: float = 0.0) -> FlowMatchingDataset:
    return FlowMatchingDataset(
        _FakeLeRobotDataset(),
        s2_latents=None,
        chunk_size=2,
        state_feature_names=STATE_NAMES,
        state_position_std_floor=floor,
    )


def test_position_floor_changes_only_degenerate_position_observations():
    dataset = _dataset(floor=0.5)

    raw_state_std = torch.tensor(_FakeHFDataset()["observation.state"], dtype=torch.float32).std(dim=0)
    raw_action_std = torch.tensor(_FakeHFDataset()["action"], dtype=torch.float32).std(dim=0)

    assert dataset.state_std[0].item() == 0.5
    assert dataset.state_std[1].item() == raw_state_std[1].item()
    assert dataset.state_std[2].item() == pytest.approx(1e-6)
    assert dataset.action_std[0].item() == pytest.approx(1e-6)
    assert dataset.action_std[1].item() == raw_action_std[1].item()


def test_zero_floor_preserves_legacy_normalization():
    dataset = _dataset()

    assert dataset.state_std[0].item() == pytest.approx(1e-6)
    assert dataset.state_std[2].item() == pytest.approx(1e-6)


def test_positive_floor_requires_an_exact_named_state_contract():
    try:
        FlowMatchingDataset(
            _FakeLeRobotDataset(),
            s2_latents=None,
            chunk_size=2,
            state_feature_names=STATE_NAMES[:-1],
            state_position_std_floor=0.5,
        )
    except ValueError as exc:
        assert "state feature name" in str(exc).lower()
    else:
        raise AssertionError("an ambiguous state order must not receive a positional floor")


def test_explicit_training_seed_replays_all_rng_sources():
    first_generator = seed_training(1337)
    first = (
        random.random(),
        np.random.random(),
        torch.rand(1).item(),
        torch.rand(1, generator=first_generator).item(),
    )

    second_generator = seed_training(1337)
    second = (
        random.random(),
        np.random.random(),
        torch.rand(1).item(),
        torch.rand(1, generator=second_generator).item(),
    )

    assert first == second
