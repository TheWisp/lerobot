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
"""HVLA honours excluded flags with the same rule as the generic trainer.

HVLA does not go through ``delta_timestamps``: it wraps a ``LeRobotDataset`` in
its own ``FlowMatchingDataset`` and builds chunks itself, so the reader's flag
boundary never reaches it. That makes "every policy gets this by derivation"
false for exactly the policy this feature started from, and it is why the rule
is restated there -- and why these tests check it independently rather than
trusting the shared one.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

FPS = 10
FRAMES_PER_EPISODE = 12
FLAGGED = 6  # index within its episode
CHUNK = 5
BLURRY, FUMBLE = 0b01, 0b10

FEATURES = {
    "action": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
    "observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
    "quality": {"dtype": "int64", "shape": (1,), "names": None, "flags": ["blurry", "fumble"]},
}


@pytest.fixture
def flagged_dataset(tmp_path):
    """Two episodes; frame 6 of each carries `blurry`. Action at frame f is (f, f)."""
    root = tmp_path / "hvla"
    ds = LeRobotDataset.create(
        repo_id="test/hvla-flags", fps=FPS, root=root, features=FEATURES, use_videos=False
    )
    for _ in range(2):
        for frame in range(FRAMES_PER_EPISODE):
            value = float(frame)
            ds.add_frame(
                {
                    "action": torch.tensor([value, value], dtype=torch.float32),
                    "observation.state": torch.tensor([value, value], dtype=torch.float32),
                    "quality": torch.tensor([BLURRY if frame == FLAGGED else 0], dtype=torch.int64),
                    "task": "flagging",
                }
            )
        ds.save_episode()
    ds.finalize()
    return LeRobotDataset(repo_id="test/hvla-flags", root=root)


def build(lerobot_dataset, flags):
    from lerobot.policies.hvla.s1.flow_matching.train import FlowMatchingDataset

    return FlowMatchingDataset(
        lerobot_dataset,
        s2_latents=None,
        chunk_size=CHUNK,
        max_delay_seconds=0.0,
        fps=FPS,
        image_keys=[],
        exclude_flags=flags,
    )


def test_the_cli_flag_exists_and_reaches_the_dataset():
    """The first version of this shipped a trainer that read `args.exclude_flags`
    without any `add_argument` defining it -- an AttributeError on every HVLA
    run, including ones not using flags at all."""
    import inspect

    from lerobot.policies.hvla.s1.flow_matching import train as hvla_train

    source = inspect.getsource(hvla_train)
    assert '"--exclude-flags"' in source, "the CLI flag is not defined"
    assert "exclude_flags=exclude_flags" in source, "the parsed value never reaches the dataset"
    assert "_log_flag_exclusion" not in source, "references a helper defined in another module"
    assert "exclude_flags" in inspect.signature(hvla_train.FlowMatchingDataset.__init__).parameters


def test_a_chunk_reaching_a_flagged_frame_stops_there(flagged_dataset):
    ds = build(flagged_dataset, ["blurry"])
    sample = ds[FLAGGED - 3]
    assert sample["action_is_pad"].tolist() == [False, False, False, True, True]


def test_the_truncated_positions_repeat_the_last_good_action(flagged_dataset):
    """Asserted as a shape, not as values: this trainer normalises actions, so
    the stored numbers are not the ones written."""
    ds = build(flagged_dataset, ["blurry"])
    actions = ds[FLAGGED - 3]["action"][:, 0].tolist()
    real, padded = actions[:3], actions[3:]
    assert real[0] < real[1] < real[2], f"the supervised part should still advance: {real}"
    assert padded == [real[2], real[2]], f"padding should hold the last good action: {actions}"


def test_truncation_matches_what_the_episode_end_does(flagged_dataset):
    """The two boundaries must be indistinguishable in the sample they produce,
    which is the whole claim of "a flag is a virtual episode end"."""
    ds = build(flagged_dataset, ["blurry"])
    at_flag = ds[FLAGGED - 3]["action"][:, 0].tolist()
    # Episode 0 ends at 12; starting 3 before it truncates the same way.
    at_end = ds[FRAMES_PER_EPISODE - 3]["action"][:, 0].tolist()
    assert at_flag[3:] == [at_flag[2]] * 2
    assert at_end[3:] == [at_end[2]] * 2


def test_without_flags_the_dataset_is_unchanged(flagged_dataset):
    plain, excluded = build(flagged_dataset, None), build(flagged_dataset, ["blurry"])
    assert plain[FLAGGED - 3]["action_is_pad"].tolist() == [False] * CHUNK
    assert excluded[FLAGGED - 3]["action_is_pad"].tolist() != [False] * CHUNK


def test_a_flag_the_frames_do_not_carry_changes_nothing(flagged_dataset):
    ds = build(flagged_dataset, ["fumble"])
    assert ds[FLAGGED - 3]["action_is_pad"].tolist() == [False] * CHUNK


def test_the_pad_mask_never_reopens(flagged_dataset):
    """The no-hole-punching invariant, over every start."""
    ds = build(flagged_dataset, ["blurry"])
    for i in range(len(flagged_dataset)):
        mask = ds[i]["action_is_pad"].tolist()
        assert mask == sorted(mask), f"chunk at {i} resumes after padding: {mask}"


def test_a_chunk_starting_on_a_flagged_frame_reads_inside_its_episode(tmp_path):
    """With ep_end == idx the upper clamp alone gives idx - 1, which at index 0
    is -1 -- silently reading the last row of the whole dataset.

    Built with frame 0 flagged specifically, because that is the only index
    where the wrap is reachable, and compared against the value at frame 0
    rather than a raw range, since actions are normalised.
    """
    root = tmp_path / "first-frame-flagged"
    ds = LeRobotDataset.create(
        repo_id="test/hvla-first", fps=FPS, root=root, features=FEATURES, use_videos=False
    )
    for frame in range(FRAMES_PER_EPISODE):
        value = float(frame)
        ds.add_frame(
            {
                "action": torch.tensor([value, value], dtype=torch.float32),
                "observation.state": torch.tensor([value, value], dtype=torch.float32),
                "quality": torch.tensor([BLURRY if frame == 0 else 0], dtype=torch.int64),
                "task": "flagging",
            }
        )
    ds.save_episode()
    ds.finalize()
    opened = LeRobotDataset(repo_id="test/hvla-first", root=root)

    flagged = build(opened, ["blurry"])
    plain = build(opened, None)
    sample = flagged[0]
    assert sample["action_is_pad"].tolist() == [True] * CHUNK
    first = plain[0]["action"][0, 0].item()
    last = plain[FRAMES_PER_EPISODE - 1]["action"][0, 0].item()
    assert first != last, "fixture must distinguish the two ends"
    for value in sample["action"][:, 0].tolist():
        assert value == pytest.approx(first), (
            f"read {value}, expected frame 0's action {first} -- {last} would mean it wrapped"
        )


def test_the_flag_does_not_reach_across_an_episode_boundary(flagged_dataset):
    """Episode 1's flag must not truncate a chunk in episode 0."""
    ds = build(flagged_dataset, ["blurry"])
    last_start = FRAMES_PER_EPISODE - 1
    assert ds[last_start]["action_is_pad"].tolist() == [False, True, True, True, True]


def test_an_undeclared_flag_is_refused(flagged_dataset):
    with pytest.raises(ValueError, match="Unknown flag"):
        build(flagged_dataset, ["nonexistent"])


def test_flagged_frames_are_located_by_absolute_index(flagged_dataset):
    ds = build(flagged_dataset, ["blurry"])
    assert ds._flagged_indices.tolist() == [FLAGGED, FRAMES_PER_EPISODE + FLAGGED]
    assert np.array_equal(ds._flagged_indices, np.sort(ds._flagged_indices))


def test_resume_advances_the_sampler_epoch():
    """Sharing the trainer's sampler made the order a pure function of
    (seed, epoch), which is what makes it reproducible -- and what makes a
    resumed run replay the identical first epoch if nothing advances it.
    `shuffle=True` never had to, because it was never reproducible.
    """
    import inspect

    from lerobot.policies.hvla.s1.flow_matching import train as hvla_train

    source = inspect.getsource(hvla_train.train)
    assert "sampler.set_epoch(" in source, "a resumed run never advances the sampler's epoch"
    # Advanced from the step it resumed at, not from a constant.
    assert "start_step // len(dataloader)" in source


def test_the_sampler_permutation_repeats_without_set_epoch():
    """The mechanism the test above guards, shown directly: two fresh samplers
    with the same seed produce the same first epoch."""
    from lerobot.datasets.sampler import EpisodeAwareSampler

    a = EpisodeAwareSampler([0], [20], shuffle=True, seed=7)
    b = EpisodeAwareSampler([0], [20], shuffle=True, seed=7)
    assert list(a) == list(b)

    b.set_epoch(3)
    assert list(b) != list(a), "advancing the epoch must change the order"
