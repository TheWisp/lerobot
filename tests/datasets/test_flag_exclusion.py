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
"""A flagged frame ends an action window the way an episode end does.

The alternative -- masking only the flagged positions and supervising the real
actions on both sides -- would teach the policy to jump across data we decided
not to trust, and it emits those unsupervised positions at inference anyway.
So the invariant under test is that the pad mask never reopens: once a window
stops, it stays stopped.
"""

import numpy as np
import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

FPS = 10
CHUNK = 5
FRAMES_PER_EPISODE = 12
FLAGGED_FRAME = 6  # index within its episode
QUALITY = "quality"
LABELS = ["blurry", "fumble"]

FEATURES = {
    "action": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
    "observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
    QUALITY: {"dtype": "int64", "shape": (1,), "names": None, "flags": LABELS},
}


def build_dataset(root, *, flagged_label: str | None, episodes: int = 2, **kwargs):
    """A dataset whose action at frame f is (f, f), with one frame flagged per episode."""
    dataset = LeRobotDataset.create(
        repo_id="test/flags", fps=FPS, root=root, features=FEATURES, use_videos=False
    )
    bit = LABELS.index(flagged_label) if flagged_label else None
    for _ in range(episodes):
        for frame_index in range(FRAMES_PER_EPISODE):
            value = float(frame_index)
            flags = (1 << bit) if (bit is not None and frame_index == FLAGGED_FRAME) else 0
            dataset.add_frame(
                {
                    "action": torch.tensor([value, value], dtype=torch.float32),
                    "observation.state": torch.tensor([value, value], dtype=torch.float32),
                    QUALITY: torch.tensor([flags], dtype=torch.int64),
                    "task": "flagging",
                }
            )
        dataset.save_episode()
    dataset.finalize()

    delta = {"action": [i / FPS for i in range(CHUNK)]}
    return LeRobotDataset(repo_id="test/flags", root=root, delta_timestamps=delta, **kwargs)


@pytest.fixture
def flagged_root(tmp_path):
    return tmp_path / "flagged"


def pad_mask(dataset, frame_index: int) -> list[bool]:
    return dataset[frame_index]["action_is_pad"].tolist()


def first_component(dataset, frame_index: int) -> list[float]:
    return dataset[frame_index]["action"][:, 0].tolist()


def test_a_window_reaching_a_flagged_frame_stops_there(flagged_root):
    dataset = build_dataset(flagged_root, flagged_label="blurry", exclude_flags=["blurry"])
    # Starting 3 before the flag: positions 0..2 are real, the flag is position 3.
    assert pad_mask(dataset, FLAGGED_FRAME - 3) == [False, False, False, True, True]


def test_the_truncated_positions_repeat_the_last_good_action(flagged_root):
    """Same filler the episode end uses -- a real pose, held, not a zero."""
    dataset = build_dataset(flagged_root, flagged_label="blurry", exclude_flags=["blurry"])
    actions = first_component(dataset, FLAGGED_FRAME - 3)
    assert actions == [3.0, 4.0, 5.0, 5.0, 5.0]


def test_the_pad_mask_never_reopens(flagged_root):
    """The no-hole-punching invariant, over every start in the dataset."""
    dataset = build_dataset(flagged_root, flagged_label="blurry", exclude_flags=["blurry"])
    for index in range(len(dataset)):
        mask = pad_mask(dataset, index)
        assert mask == sorted(mask), f"window at {index} resumes after padding: {mask}"


def test_a_window_entirely_before_the_flag_is_untouched(flagged_root):
    dataset = build_dataset(flagged_root, flagged_label="blurry", exclude_flags=["blurry"])
    assert pad_mask(dataset, 0) == [False] * CHUNK
    assert first_component(dataset, 0) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_a_window_starting_after_the_flag_is_untouched(flagged_root):
    """The good data past a flag is still learned -- from later starts."""
    dataset = build_dataset(flagged_root, flagged_label="blurry", exclude_flags=["blurry"])
    assert pad_mask(dataset, FLAGGED_FRAME + 1) == [False] * CHUNK
    assert first_component(dataset, FLAGGED_FRAME + 1) == [7.0, 8.0, 9.0, 10.0, 11.0]


def test_a_window_starting_on_the_flag_supervises_nothing(flagged_root):
    dataset = build_dataset(flagged_root, flagged_label="blurry", exclude_flags=["blurry"])
    assert pad_mask(dataset, FLAGGED_FRAME) == [True] * CHUNK


def test_the_flag_does_not_reach_across_an_episode_boundary(flagged_root):
    """Episode 1's flag must not truncate a window in episode 0."""
    dataset = build_dataset(flagged_root, flagged_label="blurry", exclude_flags=["blurry"])
    last_start = FRAMES_PER_EPISODE - 1  # final frame of episode 0
    mask = pad_mask(dataset, last_start)
    # Padded because the *episode* ends, and its filler is episode 0's last action.
    assert mask == [False, True, True, True, True]
    assert first_component(dataset, last_start) == [11.0] * CHUNK


def test_without_a_selection_the_dataset_reads_exactly_as_before(flagged_root):
    """A dataset that carries labels but is not filtering must be unchanged."""
    flagged = build_dataset(flagged_root, flagged_label="blurry", exclude_flags=["blurry"])
    unfiltered = LeRobotDataset(
        repo_id="test/flags",
        root=flagged_root,
        delta_timestamps={"action": [i / FPS for i in range(CHUNK)]},
    )
    assert pad_mask(unfiltered, FLAGGED_FRAME - 3) == [False] * CHUNK
    assert pad_mask(flagged, FLAGGED_FRAME - 3) != [False] * CHUNK


def test_selecting_a_different_label_selects_different_frames(flagged_root):
    """The exclusion is a property of the run, not of the data: same bytes,
    different masks, no rewrite."""
    dataset = build_dataset(flagged_root, flagged_label="blurry", exclude_flags=["fumble"])
    assert pad_mask(dataset, FLAGGED_FRAME - 3) == [False] * CHUNK


def test_an_unknown_label_is_refused_at_construction(flagged_root):
    with pytest.raises(ValueError, match="Unknown flag"):
        build_dataset(flagged_root, flagged_label="blurry", exclude_flags=["nonexistent"])


def test_a_dataset_with_no_flags_column_refuses_a_selection(tmp_path):
    """Silently matching nothing would train on everything while reporting
    itself filtered."""
    root = tmp_path / "plain"
    plain = {k: v for k, v in FEATURES.items() if k != QUALITY}
    dataset = LeRobotDataset.create(
        repo_id="test/plain", fps=FPS, root=root, features=plain, use_videos=False
    )
    for frame_index in range(FRAMES_PER_EPISODE):
        value = float(frame_index)
        dataset.add_frame(
            {
                "action": torch.tensor([value, value], dtype=torch.float32),
                "observation.state": torch.tensor([value, value], dtype=torch.float32),
                "task": "plain",
            }
        )
    dataset.save_episode()
    dataset.finalize()
    with pytest.raises(ValueError, match="declares none"):
        LeRobotDataset(repo_id="test/plain", root=root, exclude_flags=["blurry"])


def test_flagged_frames_are_located_by_absolute_index(flagged_root):
    """Both episodes carry a flag; the second must truncate at its own."""
    dataset = build_dataset(flagged_root, flagged_label="blurry", exclude_flags=["blurry"])
    second_episode_flag = FRAMES_PER_EPISODE + FLAGGED_FRAME
    assert pad_mask(dataset, second_episode_flag - 2) == [False, False, True, True, True]


def test_the_flag_index_lookup_is_sorted(flagged_root):
    dataset = build_dataset(flagged_root, flagged_label="blurry", exclude_flags=["blurry"])
    flagged = dataset.reader._flagged_indices
    assert flagged is not None
    assert np.array_equal(flagged, np.sort(flagged))
    assert flagged.tolist() == [FLAGGED_FRAME, FRAMES_PER_EPISODE + FLAGGED_FRAME]


# --- the run-config surface -------------------------------------------------


def make_config(**kwargs):
    from lerobot.configs.default import DatasetConfig

    return DatasetConfig(repo_id="test/flags", **kwargs)


def test_the_default_excludes_nothing():
    assert make_config().exclude_flags is None


def test_an_empty_selection_is_refused():
    """None already means "exclude nothing"; a second spelling lets a run report
    itself filtered when it is not."""
    with pytest.raises(ValueError, match="at least one flag"):
        make_config(exclude_flags=[])


def test_a_repeated_label_is_refused():
    with pytest.raises(ValueError, match="duplicates"):
        make_config(exclude_flags=["blurry", "blurry"])


def test_a_selection_survives_the_config():
    assert make_config(exclude_flags=["blurry", "fumble"]).exclude_flags == ["blurry", "fumble"]


# --- the stored contract ----------------------------------------------------


def test_the_vocabulary_survives_into_info_json(flagged_root):
    """The load-bearing persistence fact. Bit i means flags[i], so a writer that
    dropped the list would leave every stored value undecodable -- and the
    failure would look like "no frames matched" rather than like data loss."""
    import json

    build_dataset(flagged_root, flagged_label="blurry")
    info = json.loads((flagged_root / "meta" / "info.json").read_text())
    assert info["features"][QUALITY]["flags"] == LABELS


def test_an_undeclared_bit_cannot_be_written(tmp_path):
    """The validator is reachable from add_frame, not only from unit tests."""
    root = tmp_path / "bad"
    dataset = LeRobotDataset.create(
        repo_id="test/bad", fps=FPS, root=root, features=FEATURES, use_videos=False
    )
    with pytest.raises(ValueError, match="outside the 2 declared flags"):
        dataset.add_frame(
            {
                "action": torch.zeros(2),
                "observation.state": torch.zeros(2),
                QUALITY: torch.tensor([0b100]),  # bit 2, with two labels declared
                "task": "flagging",
            }
        )


def test_observation_history_still_reads_across_a_flag(flagged_root):
    """Pins a KNOWN LIMITATION rather than a desired behaviour.

    Only the forward direction is bounded. No policy consumes an
    observation-side pad mask today, so marking one would change nothing, and
    clamping history at a flag is a separate decision that deserves its own
    measurement. This test exists so that decision is made deliberately: if
    someone bounds the backward direction, this fails and points here.
    """
    dataset = build_dataset(flagged_root, flagged_label="blurry", exclude_flags=["blurry"])
    history = {"observation.state": [-2 / FPS, -1 / FPS, 0.0], "action": [i / FPS for i in range(CHUNK)]}
    with_history = LeRobotDataset(repo_id="test/flags", root=flagged_root, delta_timestamps=history)
    item = with_history[FLAGGED_FRAME + 2]
    assert item["observation.state_is_pad"].tolist() == [False, False, False]
    assert float(item["observation.state"][0, 0]) == float(FLAGGED_FRAME), (
        "the flagged frame is expected to appear in history -- see the docstring"
    )
    assert dataset is not None


# --- several labels and several columns -------------------------------------


TWO_COLUMN_FEATURES = {
    **FEATURES,
    "take": {"dtype": "int64", "shape": (1,), "names": None, "flags": ["fumble", "retake"]},
}


def build_two_column_dataset(root):
    """Frame 4 of episode 0 carries blurry+fumble in `quality`; frame 2 of
    episode 1 carries fumble in `take`. `fumble` is declared in both columns."""
    dataset = LeRobotDataset.create(
        repo_id="test/two", fps=FPS, root=root, features=TWO_COLUMN_FEATURES, use_videos=False
    )
    for episode in range(2):
        for frame_index in range(FRAMES_PER_EPISODE):
            value = float(frame_index)
            dataset.add_frame(
                {
                    "action": torch.tensor([value, value], dtype=torch.float32),
                    "observation.state": torch.tensor([value, value], dtype=torch.float32),
                    QUALITY: torch.tensor([0b11 if (episode == 0 and frame_index == 4) else 0]),
                    "take": torch.tensor([0b01 if (episode == 1 and frame_index == 2) else 0]),
                    "task": "flagging",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    return root


def flagged_for(root, labels):
    dataset = LeRobotDataset(
        repo_id="test/two",
        root=root,
        delta_timestamps={"action": [i / FPS for i in range(CHUNK)]},
        exclude_flags=labels,
    )
    found = dataset.reader._flagged_indices
    return [] if found is None else found.tolist()


def test_a_frame_is_selected_by_any_one_of_its_labels(tmp_path):
    root = build_two_column_dataset(tmp_path / "two")
    assert flagged_for(root, ["blurry"]) == [4]


def test_a_label_declared_in_two_columns_selects_from_both(tmp_path):
    """The same vocabulary spans granularities, and a caller excluding 'fumble'
    should not have to know which column holds it."""
    root = build_two_column_dataset(tmp_path / "two")
    assert flagged_for(root, ["fumble"]) == [4, FRAMES_PER_EPISODE + 2]


def test_several_labels_union_rather_than_intersect(tmp_path):
    root = build_two_column_dataset(tmp_path / "two")
    assert flagged_for(root, ["blurry", "retake"]) == [4]


def test_a_declared_but_unused_label_selects_nothing_without_erroring(tmp_path):
    """Declared-and-absent is a legitimate state; only *undeclared* is an error."""
    root = build_two_column_dataset(tmp_path / "two")
    assert flagged_for(root, ["retake"]) == []


def test_an_episode_subset_still_locates_flags_by_absolute_index(tmp_path):
    """The reader indexes absolutely while hf_dataset is a relative subset;
    mixing the two would truncate at the wrong frame or not at all."""
    root = build_two_column_dataset(tmp_path / "two")
    dataset = LeRobotDataset(
        repo_id="test/two",
        root=root,
        episodes=[1],
        delta_timestamps={"action": [i / FPS for i in range(CHUNK)]},
        exclude_flags=["fumble"],
    )
    assert dataset.reader._flagged_indices.tolist() == [FRAMES_PER_EPISODE + 2]
    # Episode 1's own frame 0 is two frames before its flag.
    assert dataset[0]["action_is_pad"].tolist() == [False, False, True, True, True]
    assert dataset[0]["action"][:, 0].tolist() == [0.0, 1.0, 1.0, 1.0, 1.0]


def test_a_query_on_an_excluded_frame_reads_that_frame_not_the_one_before(flagged_root):
    """The boundary sets ``ep_end == abs_idx`` when the frame itself is excluded,
    and an upper clamp alone then resolves *every* delta to ``abs_idx - 1`` --
    delta 0 included, so this frame's row is paired with the previous frame's
    images.

    Training never reaches it, because the sampler does not draw such a start.
    ``dataset[i]`` does: the eval loop and the viewers index directly.
    """
    dataset = build_dataset(flagged_root, flagged_label=LABELS[0], exclude_flags=[LABELS[0]])
    reader = dataset.reader
    reader.delta_indices = {"observation.state": [0]}

    query, pad = reader._get_query_indices(FLAGGED_FRAME, 0)
    assert query["observation.state"] == [FLAGGED_FRAME], (
        f"delta 0 must read frame {FLAGGED_FRAME}, got {query['observation.state']}"
    )
    # Still padding -- this decides which row is read, not whether the position
    # counts toward the loss.
    assert pad["observation.state_is_pad"].tolist() == [True]


def test_an_unexcluded_frame_clamps_exactly_as_before(flagged_root):
    """The floor must not loosen the ordinary case: a window running past the
    episode end still stops at its last frame."""
    dataset = build_dataset(flagged_root, flagged_label=None)
    reader = dataset.reader
    reader.delta_indices = {"observation.state": [0, 5, 50]}

    last = FRAMES_PER_EPISODE - 1
    query, _ = reader._get_query_indices(last, 0)
    assert query["observation.state"] == [last, last, last]


def test_the_logged_share_is_over_the_frames_this_run_loaded(flagged_root, caplog):
    """`_flagged_indices` is scoped to the loaded subset. Dividing it by the
    whole dataset's length understates the exclusion by the subset ratio -- in
    the one line the run leaves behind as its record of what it trained on.
    """
    import logging as _logging

    from lerobot.datasets.factory import _log_flag_exclusion

    build_dataset(flagged_root, flagged_label=LABELS[0], episodes=2)
    delta = {"action": [i / FPS for i in range(CHUNK)]}
    one_episode = LeRobotDataset(
        repo_id="test/flags",
        root=flagged_root,
        delta_timestamps=delta,
        exclude_flags=[LABELS[0]],
        episodes=[0],
    )
    assert len(one_episode) == FRAMES_PER_EPISODE, "fixture must load one of two episodes"

    with caplog.at_level(_logging.INFO):
        _log_flag_exclusion(one_episode, [LABELS[0]])
    line = next(r.getMessage() for r in caplog.records if "Flags to exclude" in r.getMessage())

    assert f"1 of {FRAMES_PER_EPISODE} frames" in line, line
    # 1/12, not 1/24.
    assert "8.33%" in line, line


def test_streaming_refuses_a_selection_rather_than_ignoring_it(flagged_root):
    """The streaming reader builds its own padding masks and never sees
    `_flagged_indices`, so a selection would be accepted and do nothing -- a run
    reporting itself filtered while training on every frame. Refused loudly at
    construction instead, which is the only place it can still be noticed.
    """
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.factory import make_policy_config

    build_dataset(flagged_root, flagged_label=LABELS[0])
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id="test/flags",
            root=str(flagged_root),
            streaming=True,
            exclude_flags=[LABELS[0]],
        ),
        policy=make_policy_config("act"),
    )
    with pytest.raises(NotImplementedError, match="not supported for streaming"):
        make_dataset(cfg)
