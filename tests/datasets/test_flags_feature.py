"""Bitset ("flags") feature columns: contract, editing, and exclusion.

A flags column stores one integer per frame with bit ``i`` meaning
``flags[i]``, so labels written by different passes coexist on the same frame.
These tests pin the properties that make that true, each of which was a real
defect: an edit that wrote the whole integer erased every label it did not
know about, an Inspector that read a value nobody passed rendered every bit
off regardless of the data, and a derived column offered edits that the next
recompute would discard.
"""

import asyncio

import numpy as np
import pytest
import torch

from lerobot.datasets.dataset_tools import add_features_inplace
from lerobot.datasets.feature_utils import (
    flags_to_labels,
    is_derived_feature,
    is_flags_feature,
    labels_to_flags,
)
from lerobot.datasets.feature_value_edits import set_feature_values
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FLAGS = ["tool_a:x", "tool_b:y", "human:collision"]
N_FRAMES = 12


@pytest.fixture
def flagged_dataset(tmp_path):
    """A two-episode dataset with a bitset column already carrying a varied mix.

    Frames 0-1 hold tool_a, 2-3 hold both tools, 4-5 hold tool_b, the rest are
    clear — so a test that flattens the range to one value is visible.
    """
    root = tmp_path / "flagged"
    ds = LeRobotDataset.create(
        repo_id="test/flagged",
        fps=10,
        root=root,
        features={
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
            "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        },
        use_videos=False,
    )
    for ep in range(2):
        for i in range(N_FRAMES):
            ds.add_frame({
                "observation.state": torch.tensor([float(i), float(ep)]),
                "action": torch.tensor([float(i), float(ep)]),
                "task": "flags",
            })
        ds.save_episode()
    ds.finalize()

    add_features_inplace(ds, {"quality.flags": (0, {
        "dtype": "int64", "shape": (1,), "names": None, "flags": FLAGS, "per_episode": False,
    })})

    ds = LeRobotDataset("test/flagged", root=root)
    set_feature_values(ds, [
        {"feature": "quality.flags", "from_index": 0, "to_index": 2, "value": 0b001},
        {"feature": "quality.flags", "from_index": 2, "to_index": 4, "value": 0b011},
        {"feature": "quality.flags", "from_index": 4, "to_index": 6, "value": 0b010},
    ], in_place=True)
    return LeRobotDataset("test/flagged", root=root)


def _column(dataset) -> np.ndarray:
    return np.asarray(dataset.hf_dataset["quality.flags"], dtype=np.int64).reshape(-1)


def _reopen(dataset) -> LeRobotDataset:
    return LeRobotDataset("test/flagged", root=dataset.root)


@pytest.fixture
def gui_state(flagged_dataset):
    """App state wired the way the server wires it, with the dataset loaded."""
    from lerobot.gui.api import datasets as datasets_api
    from lerobot.gui.api import edits as edits_api
    from lerobot.gui.frame_cache import FrameCache
    from lerobot.gui.state import AppState

    state = AppState(frame_cache=FrameCache(max_bytes=1))
    edits_api.set_app_state(state)
    datasets_api.set_app_state(state)
    state.datasets[str(flagged_dataset.root)] = flagged_dataset
    return state


# ── contract ───────────────────────────────────────────────────────────────


def test_flags_contract_distinguishes_bitset_from_categorical():
    bitset = {"dtype": "int64", "shape": [1], "flags": FLAGS}
    categorical = {"dtype": "int64", "shape": [1], "names": ["a", "b"]}
    assert is_flags_feature(bitset)
    assert not is_flags_feature(categorical)
    # A vector is not a bitset even with a vocabulary: one value per frame.
    assert not is_flags_feature({"dtype": "int64", "shape": [4], "flags": FLAGS})


def test_labels_round_trip_through_the_integer():
    spec = {"dtype": "int64", "shape": [1], "flags": FLAGS}
    value = labels_to_flags(spec, ["tool_a:x", "human:collision"])
    assert value == 0b101
    assert flags_to_labels(spec, value) == ["tool_a:x", "human:collision"]


def test_derived_defaults_to_false_so_existing_datasets_are_unchanged():
    assert not is_derived_feature({"dtype": "int64", "shape": [1], "flags": FLAGS})
    assert is_derived_feature({"dtype": "int64", "shape": [1], "flags": FLAGS, "derived": True})


# ── editing ────────────────────────────────────────────────────────────────


def test_adding_one_label_preserves_the_others_per_frame(flagged_dataset, gui_state):
    """The defect: staging a whole integer flattened every other tool's labels."""
    from lerobot.gui.api import edits as edits_api
    from lerobot.gui.api._edits_core import propose_feature_set

    dataset_id = str(flagged_dataset.root)
    before = _column(flagged_dataset).copy()
    assert len(set(before[:6].tolist())) > 1, "fixture must have a varied range"

    propose_feature_set(gui_state, dataset_id, 0, "quality.flags", 2, 10, None, set_mask=0b100)
    asyncio.run(edits_api._apply_edits_locked(dataset_id))

    expected = before.copy()
    expected[2:10] |= 0b100
    assert _column(_reopen(flagged_dataset)).tolist() == expected.tolist()


def test_removing_a_label_restores_the_previous_state_exactly(flagged_dataset, gui_state):
    from lerobot.gui.api import edits as edits_api
    from lerobot.gui.api._edits_core import propose_feature_set

    dataset_id = str(flagged_dataset.root)
    before = _column(flagged_dataset).copy()

    propose_feature_set(gui_state, dataset_id, 0, "quality.flags", 2, 10, None, set_mask=0b100)
    asyncio.run(edits_api._apply_edits_locked(dataset_id))

    gui_state.pending_edits.clear()
    gui_state.datasets[dataset_id] = _reopen(flagged_dataset)
    propose_feature_set(gui_state, dataset_id, 0, "quality.flags", 2, 10, None, clear_mask=0b100)
    asyncio.run(edits_api._apply_edits_locked(dataset_id))

    assert _column(_reopen(flagged_dataset)).tolist() == before.tolist()


def test_undeclared_mask_bit_is_refused_at_stage_time(flagged_dataset, gui_state):
    """Rejecting on Save would be after a whole labelling session's work."""
    from lerobot.gui.api._edits_core import EditValidationError, propose_feature_set

    with pytest.raises(EditValidationError, match="not declared"):
        propose_feature_set(
            gui_state, str(flagged_dataset.root), 0, "quality.flags", 0, 4, None,
            set_mask=1 << (len(FLAGS) + 2),
        )


def test_whole_value_edits_still_work(flagged_dataset, gui_state):
    """Masks are an addition; replacing the integer outright still means that."""
    from lerobot.gui.api import edits as edits_api
    from lerobot.gui.api._edits_core import propose_feature_set

    dataset_id = str(flagged_dataset.root)
    propose_feature_set(gui_state, dataset_id, 0, "quality.flags", 0, 6, 0b010)
    asyncio.run(edits_api._apply_edits_locked(dataset_id))
    assert _column(_reopen(flagged_dataset))[:6].tolist() == [0b010] * 6


# ── training-side exclusion ────────────────────────────────────────────────


def test_exclusion_resolves_a_label_across_every_flags_column(flagged_dataset):
    """A caller names a label; which column carries it is not their problem."""
    from lerobot.policies.hvla.s1.flow_matching.train import load_excluded_frames

    add_features_inplace(flagged_dataset, {"quality.episode_flags": (0, {
        "dtype": "int64", "shape": (1,), "names": None,
        "flags": ["tool_c:short"], "per_episode": True,
    })})
    ds = _reopen(flagged_dataset)
    set_feature_values(ds, [
        {"feature": "quality.episode_flags", "from_index": N_FRAMES,
         "to_index": 2 * N_FRAMES, "value": 1},
    ], in_place=True)
    ds = _reopen(flagged_dataset)

    # tool_a:x sits in the per-frame column, tool_c:short in the per-episode one.
    assert load_excluded_frames(ds, "tool_a:x") == {0, 1, 2, 3}
    assert load_excluded_frames(ds, "tool_c:short") == set(range(N_FRAMES, 2 * N_FRAMES))
    assert load_excluded_frames(ds, "tool_a:x,tool_c:short") == {0, 1, 2, 3} | set(
        range(N_FRAMES, 2 * N_FRAMES)
    )
    assert load_excluded_frames(ds, None) == set()

    with pytest.raises(ValueError, match="unknown flag"):
        load_excluded_frames(ds, "nobody:declared_this")


# ── schema mutation keeps the loaded table in step ─────────────────────────


def test_added_column_is_readable_without_reopening(flagged_dataset):
    """Not flags-specific: any column added to an open dataset must be readable.

    ``add_features_inplace`` rebinds ``dataset.meta``, but the reader holds its
    own reference and that copy decides which columns the loaded table has. The
    two disagreeing meant a freshly added column was missing from
    ``hf_dataset`` (reads raised "Column 'x' doesn't exist") and the next
    reload failed casting the new parquet to the old schema — a CastError
    naming a data file, which reads as corruption but is not.

    Add-then-edit-then-save is the labelling workflow, which is why this went
    unnoticed: the schema tools are otherwise used on a dataset that is then
    reopened from scratch.
    """
    add_features_inplace(flagged_dataset, {"quality.extra": (0, {
        "dtype": "int64", "shape": (1,), "names": None, "per_episode": False,
    })})

    assert "quality.extra" in flagged_dataset.meta.features
    # The loaded table, not just the metadata, must know about it.
    assert "quality.extra" in flagged_dataset.hf_dataset.column_names
    assert np.asarray(flagged_dataset.hf_dataset["quality.extra"]).reshape(-1).tolist() == (
        [0] * (2 * N_FRAMES)
    )

    # And a reload must not fail casting the new parquet to the old schema.
    from lerobot.gui.dataset_reload import reload_dataset_from_disk

    reload_dataset_from_disk(flagged_dataset)
    assert "quality.extra" in flagged_dataset.hf_dataset.column_names


def test_values_written_to_a_freshly_added_column_survive_apply(flagged_dataset, gui_state):
    """The end of the labelling loop: add a column, label in it, save."""
    from lerobot.gui.api import edits as edits_api
    from lerobot.gui.api._edits_core import propose_feature_set

    add_features_inplace(flagged_dataset, {"quality.human_flags": (0, {
        "dtype": "int64", "shape": (1,), "names": None, "per_episode": False,
        "flags": ["human:bad_frame"],
    })})
    dataset_id = str(flagged_dataset.root)
    gui_state.datasets[dataset_id] = flagged_dataset

    propose_feature_set(gui_state, dataset_id, 0, "quality.human_flags", 2, 6, None, set_mask=0b1)
    result = asyncio.run(edits_api._apply_edits_locked(dataset_id))
    assert result["applied"] == 1, result

    written = np.asarray(
        _reopen(flagged_dataset).hf_dataset["quality.human_flags"], dtype=np.int64
    ).reshape(-1)
    assert written[2:6].tolist() == [1, 1, 1, 1]
    assert written[:2].tolist() == [0, 0] and written[6:].sum() == 0


# ── labels that share frames compose, they do not contest ──────────────────


def test_two_labels_on_the_same_range_both_survive(flagged_dataset, gui_state):
    """The reported bug: ticking a second label unticked the first.

    Last-write-wins dropped any prior edit fully inside the new range, and
    ticking a second label on the same selection is exactly that shape.
    """
    from lerobot.gui.api import edits as edits_api
    from lerobot.gui.api._edits_core import propose_feature_set

    dataset_id = str(flagged_dataset.root)
    propose_feature_set(gui_state, dataset_id, 0, "quality.flags", 6, 10, None, set_mask=0b001)
    propose_feature_set(gui_state, dataset_id, 0, "quality.flags", 6, 10, None, set_mask=0b100)
    assert len(gui_state.pending_edits) == 2, "neither edit may be dropped"

    asyncio.run(edits_api._apply_edits_locked(dataset_id))
    written = _column(_reopen(flagged_dataset))
    assert written[6:10].tolist() == [0b101] * 4


def test_partially_overlapping_label_edits_compose(flagged_dataset, gui_state):
    """Where the ranges overlap, both labels hold; elsewhere, only one does."""
    from lerobot.gui.api import edits as edits_api
    from lerobot.gui.api._edits_core import propose_feature_set

    dataset_id = str(flagged_dataset.root)
    propose_feature_set(gui_state, dataset_id, 0, "quality.flags", 6, 10, None, set_mask=0b001)
    propose_feature_set(gui_state, dataset_id, 0, "quality.flags", 8, 12, None, set_mask=0b100)
    asyncio.run(edits_api._apply_edits_locked(dataset_id))

    written = _column(_reopen(flagged_dataset))
    assert written[6:8].tolist() == [0b001, 0b001], "first label only"
    assert written[8:10].tolist() == [0b101, 0b101], "both, on the shared frames"
    assert written[10:12].tolist() == [0b100, 0b100], "second label only"


def test_a_label_edit_does_not_prompt_about_another_label(flagged_dataset, gui_state):
    """No overlap dialog between mask edits — there is nothing to resolve.

    ``confirm_overlap`` is left False: if the edits were still treated as
    contesting, this would raise EditConflictError instead of staging.
    """
    from lerobot.gui.api._edits_core import propose_feature_set

    dataset_id = str(flagged_dataset.root)
    propose_feature_set(gui_state, dataset_id, 0, "quality.flags", 4, 10, None, set_mask=0b001)
    propose_feature_set(gui_state, dataset_id, 0, "quality.flags", 6, 12, None, set_mask=0b100)
    assert len(gui_state.pending_edits) == 2


def test_toggling_one_label_off_and_on_leaves_one_edit(flagged_dataset, gui_state):
    """Repeated toggling must not pile up edits that cancel out."""
    from lerobot.gui.api import edits as edits_api
    from lerobot.gui.api._edits_core import propose_feature_set

    dataset_id = str(flagged_dataset.root)
    for masks in ({"set_mask": 0b100}, {"clear_mask": 0b100}, {"set_mask": 0b100}):
        propose_feature_set(gui_state, dataset_id, 0, "quality.flags", 6, 10, None, **masks)
    assert len(gui_state.pending_edits) == 1, gui_state.pending_edits

    asyncio.run(edits_api._apply_edits_locked(dataset_id))
    assert (_column(_reopen(flagged_dataset))[6:10] & 0b100).tolist() == [0b100] * 4


def test_a_value_edit_still_contests_the_frames_it_covers(flagged_dataset, gui_state):
    """Replacing the integer is not composition, so it must still prompt."""
    from lerobot.gui.api._edits_core import EditConflictError, propose_feature_set

    dataset_id = str(flagged_dataset.root)
    propose_feature_set(gui_state, dataset_id, 0, "quality.flags", 6, 10, None, set_mask=0b001)
    with pytest.raises(EditConflictError):
        propose_feature_set(gui_state, dataset_id, 0, "quality.flags", 6, 10, 0b010)


def test_clearing_a_label_leaves_the_others_on_the_shared_frames(flagged_dataset, gui_state):
    """Untick must remove one bit, not reset the frames."""
    from lerobot.gui.api import edits as edits_api
    from lerobot.gui.api._edits_core import propose_feature_set

    dataset_id = str(flagged_dataset.root)
    propose_feature_set(gui_state, dataset_id, 0, "quality.flags", 0, 6, None, set_mask=0b100)
    propose_feature_set(gui_state, dataset_id, 0, "quality.flags", 0, 6, None, clear_mask=0b001)
    asyncio.run(edits_api._apply_edits_locked(dataset_id))

    # Fixture: 0-1 = tool_a (001), 2-3 = both (011), 4-5 = tool_b (010).
    written = _column(_reopen(flagged_dataset))
    assert written[:6].tolist() == [0b100, 0b100, 0b110, 0b110, 0b110, 0b110]


# ── a flagged frame disqualifies every chunk containing it ─────────────────


def _starts(flagged, chunk=4, episodes=(0, 0, 0, 0, 0, 1, 1, 1, 1, 1)):
    from lerobot.policies.hvla.s1.flow_matching.train import clean_chunk_starts

    return clean_chunk_starts(list(range(len(episodes))), set(flagged), list(episodes), chunk)


def test_a_flagged_frame_disqualifies_every_chunk_containing_it():
    """Masking the flagged position's loss does not keep it out of training.

    ``x_t`` is built from the whole action chunk and ``denoise_step`` attends
    across all positions with no causal mask, so a flagged action reaches every
    other position's prediction — and those predictions are in the loss.
    """
    # Frame 3 is flagged; with chunk=4 that rules out starts 0,1,2,3.
    assert _starts({3}) == [4, 5, 6, 7, 8, 9]


def test_the_window_stops_at_the_episode_boundary():
    """A flag in the next episode must not disqualify this one's chunks.

    Chunks never read across a demonstration — positions past the end repeat
    the episode's last action — so a flag on frame 5, episode 1's first frame,
    leaves every episode 0 start intact. Within episode 1 it costs only the
    start that is itself frame 5; 6..9 look forward, away from it.
    """
    assert _starts({5}) == [0, 1, 2, 3, 4, 6, 7, 8, 9]


def test_no_flags_keeps_every_start():
    assert _starts(set()) == list(range(10))


def test_padding_is_not_contamination():
    """Starts near an episode end survive: padding repeats a real pose.

    Frame 4 is the last of episode 0, so a chunk starting at 4 is one real
    action plus padding. Nothing there is flagged, so it stays.
    """
    kept = _starts({0})
    assert 4 in kept, kept


def test_every_surviving_window_is_clean_on_a_real_shaped_case():
    """Property check over a longer, clustered case rather than a hand example."""
    import numpy as np

    from lerobot.policies.hvla.s1.flow_matching.train import clean_chunk_starts

    episodes = np.repeat(np.arange(6), 40)
    flagged = set(range(17, 23)) | set(range(100, 104)) | {200}
    chunk = 12
    kept = clean_chunk_starts(list(range(len(episodes))), flagged, episodes, chunk)

    bad = np.zeros(len(episodes), dtype=bool)
    bad[list(flagged)] = True
    for i in kept:
        end = (int(episodes[i]) + 1) * 40
        assert not bad[i:min(i + chunk, end)].any(), f"chunk at {i} carries a flagged frame"
    # And nothing clean was dropped.
    dropped = set(range(len(episodes))) - set(kept)
    for i in dropped:
        end = (int(episodes[i]) + 1) * 40
        assert bad[i:min(i + chunk, end)].any(), f"chunk at {i} was clean but dropped"


# ── the blast radius of a single flagged frame ─────────────────────────────
#
# A chunk starting at i covers [i, i + chunk), so it contains frame b exactly
# when b - chunk + 1 <= i <= b. The disqualified starts therefore all sit AT OR
# BEFORE the flagged frame, never after it — which is the property that makes
# labelling a bad frame cost bounded, predictable data rather than truncating
# the rest of the episode.


def _dropped(flagged_frame, n_frames=200, chunk=50, episodes=None):
    import numpy as np

    from lerobot.policies.hvla.s1.flow_matching.train import clean_chunk_starts

    if episodes is None:
        episodes = np.zeros(n_frames, dtype=np.int64)
    kept = clean_chunk_starts(list(range(n_frames)), {flagged_frame}, episodes, chunk)
    return sorted(set(range(n_frames)) - set(kept))


def test_a_flagged_frame_only_costs_starts_at_or_before_it():
    dropped = _dropped(100)
    assert dropped == list(range(51, 101))
    assert max(dropped) == 100, "nothing after the flagged frame is affected"
    assert len(dropped) == 50, "exactly chunk_size starts, mid-episode"


def test_the_frame_after_a_flagged_one_is_still_trainable():
    """Recovery is learnable: only the approach to the bad frame is lost."""
    assert 101 not in _dropped(100)
    assert 150 not in _dropped(100)


def test_near_an_episode_start_there_is_less_to_disqualify():
    """The radius is bounded by the episode, so it is at most chunk_size."""
    assert _dropped(10) == list(range(0, 11))


def test_near_an_episode_end_the_radius_is_still_the_full_chunk():
    """Clipping shortens windows; it does not change which contain the frame."""
    assert _dropped(195) == list(range(146, 196))


def test_the_radius_cannot_reach_back_into_the_previous_episode():
    import numpy as np

    episodes = np.repeat([0, 1], 100)
    dropped = _dropped(105, episodes=episodes)
    assert dropped == list(range(100, 106))
    assert all(d >= 100 for d in dropped), "episode 0 must be untouched"


def test_the_radius_scales_with_chunk_size():
    """Halving the horizon halves what one bad frame costs."""
    assert len(_dropped(100, chunk=50)) == 50
    assert len(_dropped(100, chunk=25)) == 25
    assert len(_dropped(100, chunk=1)) == 1
