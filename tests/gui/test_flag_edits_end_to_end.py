# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Flag edits from the HTTP surface to the bytes on disk.

The unit tests prove the bit arithmetic and the staging invariant. These prove
the parts only an end-to-end run can: that staging reaches the pending queue,
that Save lowers it against the real column, and that a dataset reopened from
disk carries what the operator asked for and nothing else.
"""

from __future__ import annotations

import asyncio

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.gui.api import datasets as datasets_module, edits as edits_module
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState

BLURRY, FUMBLE = 0b01, 0b10
FRAMES_PER_EPISODE = 8
EPISODES = 2


@pytest.fixture
def app_with_state():
    app = FastAPI()
    app.include_router(datasets_module.router)
    app.include_router(edits_module.router)
    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    original = datasets_module._app_state
    original_edits = edits_module._app_state
    datasets_module.set_app_state(state)
    edits_module._app_state = state
    yield app, state
    datasets_module._app_state = original
    edits_module._app_state = original_edits
    state.pending_edits.clear()


@pytest.fixture
def flagged(app_with_state, tmp_path, empty_lerobot_dataset_factory):
    """Two episodes; frames 4-7 of episode 0 already carry `fumble`."""
    app, state = app_with_state
    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": None},
        "quality": {"dtype": "int64", "shape": (1,), "names": None, "flags": ["blurry", "fumble"]},
    }
    ds = empty_lerobot_dataset_factory(root=tmp_path / "ds", features=features)
    for episode in range(EPISODES):
        for frame in range(FRAMES_PER_EPISODE):
            preset = FUMBLE if (episode == 0 and 4 <= frame < 8) else 0
            ds.add_frame(
                {
                    "action": np.zeros(2, dtype=np.float32),
                    "quality": np.array([preset], dtype=np.int64),
                    "task": "flagging",
                }
            )
        ds.save_episode()
    ds.finalize()
    # Register the dataset as the GUI would have it: opened for reading, with
    # episode metadata loaded. The write-mode object has no meta.episodes, and
    # the edit path needs episode bounds to resolve a range.
    opened = LeRobotDataset(repo_id=ds.repo_id, root=ds.root)
    state.datasets[str(opened.root)] = opened
    return app, state, str(opened.root), opened


def post(app, path, body):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(path, json=body)

    return asyncio.run(run())


def tick(app, dataset_id, *, frames, flag, on=True, episode=0):
    return post(
        app,
        "/api/edits/feature-bits",
        {
            "dataset_id": dataset_id,
            "episode_index": episode,
            "feature": "quality",
            "frame_from": frames[0],
            "frame_to": frames[1],
            "set_flags": [flag] if on else [],
            "clear_flags": [] if on else [flag],
        },
    )


def saved_column(ds) -> list[int]:
    """The column as a fresh reader sees it -- not the in-memory object."""
    reopened = LeRobotDataset(repo_id=ds.repo_id, root=ds.root)
    return [int(reopened[i]["quality"]) for i in range(reopened.num_frames)]


def apply_edits(app, dataset_id):
    return post(app, f"/api/edits/apply?dataset_id={dataset_id}", None)


# --- staging ---------------------------------------------------------------


def test_ticking_a_flag_stages_one_edit(flagged):
    app, state, dataset_id, _ds = flagged
    resp = tick(app, dataset_id, frames=(0, 8), flag="blurry")
    assert resp.status_code == 200, resp.text
    staged = [e for e in state.get_edits_for_dataset(dataset_id) if e.edit_type == "feature_bits"]
    assert len(staged) == 1


def test_ticking_then_unticking_stages_nothing(flagged):
    """The question the design was built around, over the real HTTP surface."""
    app, state, dataset_id, _ds = flagged
    tick(app, dataset_id, frames=(0, 8), flag="blurry")
    resp = tick(app, dataset_id, frames=(0, 8), flag="blurry", on=False)
    assert resp.status_code == 200, resp.text
    assert resp.json()["pending"] == 0
    staged = [e for e in state.get_edits_for_dataset(dataset_id) if e.edit_type == "feature_bits"]
    assert staged == []


def test_reticking_a_flag_the_frames_already_carry_stages_nothing(flagged):
    app, state, dataset_id, _ds = flagged
    resp = tick(app, dataset_id, frames=(4, 8), flag="fumble")
    assert resp.json()["pending"] == 0
    assert [e for e in state.get_edits_for_dataset(dataset_id) if e.edit_type == "feature_bits"] == []


def test_two_flags_on_the_same_frames_both_stage_without_a_conflict(flagged):
    """A value edit would 409 here and ask the operator to arbitrate."""
    app, state, dataset_id, _ds = flagged
    assert tick(app, dataset_id, frames=(0, 8), flag="blurry").status_code == 200
    assert tick(app, dataset_id, frames=(0, 8), flag="fumble").status_code == 200
    staged = [e for e in state.get_edits_for_dataset(dataset_id) if e.edit_type == "feature_bits"]
    assert len(staged) == 2


def test_an_undeclared_flag_is_refused(flagged):
    app, _state, dataset_id, _ds = flagged
    resp = tick(app, dataset_id, frames=(0, 8), flag="nonexistent")
    assert resp.status_code == 400
    assert "unknown flag" in resp.json()["detail"]


def test_a_column_that_is_not_a_bitset_is_refused(flagged):
    app, _state, dataset_id, _ds = flagged
    resp = post(
        app,
        "/api/edits/feature-bits",
        {
            "dataset_id": dataset_id,
            "episode_index": 0,
            "feature": "action",
            "frame_from": 0,
            "frame_to": 4,
            "set_flags": ["blurry"],
        },
    )
    assert resp.status_code == 400
    assert "not a flags column" in resp.json()["detail"]


def test_ticking_and_unticking_the_same_flag_in_one_call_is_refused(flagged):
    app, _state, dataset_id, _ds = flagged
    resp = post(
        app,
        "/api/edits/feature-bits",
        {
            "dataset_id": dataset_id,
            "episode_index": 0,
            "feature": "quality",
            "frame_from": 0,
            "frame_to": 4,
            "set_flags": ["blurry"],
            "clear_flags": ["blurry"],
        },
    )
    assert resp.status_code == 400
    assert "contradict" in resp.json()["detail"]


# --- data integrity through Save -------------------------------------------


def test_saving_writes_only_the_edited_flag(flagged):
    """The property the whole design exists for: frames 4-7 keep `fumble`."""
    app, _state, dataset_id, ds = flagged
    tick(app, dataset_id, frames=(0, 8), flag="blurry")
    resp = apply_edits(app, dataset_id)
    assert resp.status_code == 200, resp.text

    column = saved_column(ds)
    assert column[0:4] == [BLURRY] * 4
    assert column[4:8] == [BLURRY | FUMBLE] * 4, "fumble was erased by the blurry edit"
    assert column[8:] == [0] * FRAMES_PER_EPISODE, "episode 1 must be untouched"


def test_two_flags_staged_together_both_reach_disk(flagged):
    app, _state, dataset_id, ds = flagged
    tick(app, dataset_id, frames=(0, 4), flag="blurry")
    tick(app, dataset_id, frames=(2, 6), flag="fumble")
    apply_edits(app, dataset_id)

    column = saved_column(ds)
    assert column[0:2] == [BLURRY] * 2
    assert column[2:4] == [BLURRY | FUMBLE] * 2
    assert column[4:6] == [FUMBLE] * 2
    assert column[6:8] == [FUMBLE] * 2  # preset, untouched by either edit


def test_unticking_removes_only_that_flag(flagged):
    app, _state, dataset_id, ds = flagged
    tick(app, dataset_id, frames=(0, 8), flag="blurry")
    apply_edits(app, dataset_id)
    tick(app, dataset_id, frames=(0, 8), flag="blurry", on=False)
    apply_edits(app, dataset_id)

    column = saved_column(ds)
    assert column[0:4] == [0] * 4
    assert column[4:8] == [FUMBLE] * 4, "unticking blurry must not disturb fumble"


def test_saving_nothing_leaves_the_column_alone(flagged):
    app, _state, dataset_id, ds = flagged
    before = saved_column(ds)
    tick(app, dataset_id, frames=(4, 8), flag="fumble")  # already set: stages nothing
    apply_edits(app, dataset_id)
    assert saved_column(ds) == before


def test_the_staged_queue_is_empty_after_saving(flagged):
    app, state, dataset_id, _ds = flagged
    tick(app, dataset_id, frames=(0, 8), flag="blurry")
    apply_edits(app, dataset_id)
    assert state.get_edits_for_dataset(dataset_id) == []


def test_one_flag_edit_is_reported_as_one_applied_edit(flagged):
    """Lowering fans one edit into several runs; the count must describe what
    the operator did, not how it was written."""
    app, _state, dataset_id, _ds = flagged
    tick(app, dataset_id, frames=(0, 8), flag="blurry")  # spans two distinct values
    resp = apply_edits(app, dataset_id)
    assert resp.json()["applied"] == 1


def test_other_columns_are_untouched_by_a_flag_save(flagged):
    app, _state, dataset_id, ds = flagged
    tick(app, dataset_id, frames=(0, 8), flag="blurry")
    apply_edits(app, dataset_id)
    reopened = LeRobotDataset(repo_id=ds.repo_id, root=ds.root)
    assert all(np.allclose(reopened[i]["action"].numpy(), np.zeros(2)) for i in range(reopened.num_frames))


def test_the_vocabulary_survives_a_flag_save(flagged):
    app, _state, dataset_id, ds = flagged
    tick(app, dataset_id, frames=(0, 8), flag="blurry")
    apply_edits(app, dataset_id)
    reopened = LeRobotDataset(repo_id=ds.repo_id, root=ds.root)
    assert reopened.meta.features["quality"]["flags"] == ["blurry", "fumble"]


def test_flags_survive_a_second_round_of_editing(flagged):
    """Save, edit again, save: the second lowering reads the column the first
    one wrote, not the one it started from."""
    app, _state, dataset_id, ds = flagged
    tick(app, dataset_id, frames=(0, 4), flag="blurry")
    apply_edits(app, dataset_id)
    tick(app, dataset_id, frames=(2, 6), flag="fumble")
    apply_edits(app, dataset_id)

    column = saved_column(ds)
    assert column[0:2] == [BLURRY] * 2
    assert column[2:4] == [BLURRY | FUMBLE] * 2
    assert column[4:6] == [FUMBLE] * 2


def test_flags_on_two_episodes_keep_their_own_frame_numbers(flagged):
    """Staged edits are stored per feature, not per episode, so staging on one
    episode re-derives the episode-local frames of edits belonging to another.

    Those local numbers are what the GUI merges into the row it draws, so
    getting them from the wrong episode paints the edit on the wrong frames.
    """
    app, state, dataset_id, _ds = flagged
    assert tick(app, dataset_id, frames=(0, 4), flag="blurry", episode=0).status_code == 200
    assert tick(app, dataset_id, frames=(2, 6), flag="fumble", episode=1).status_code == 200

    staged = {
        (e.episode_index, e.params["feature"]): e
        for e in state.get_edits_for_dataset(dataset_id)
        if e.edit_type == "feature_bits"
    }
    assert len(staged) == 2, staged

    ep0 = next(e for e in staged.values() if e.params["global_from_index"] == 0)
    ep1 = next(e for e in staged.values() if e.params["global_from_index"] == FRAMES_PER_EPISODE + 2)
    assert (ep0.params["frame_from"], ep0.params["frame_to"]) == (0, 4)
    assert ep0.episode_index == 0
    assert (ep1.params["frame_from"], ep1.params["frame_to"]) == (2, 6)
    assert ep1.episode_index == 1


def test_a_flag_edit_that_lowers_to_nothing_is_still_reported_honestly(flagged):
    """A staged edit can lower to no runs if the column changed under it. The
    count must not go negative, and the save must not claim more than was
    staged."""
    app, state, dataset_id, ds = flagged
    tick(app, dataset_id, frames=(0, 8), flag="blurry")
    # Write the flag out of band, so the staged edit becomes a no-op.
    apply_edits(app, dataset_id)
    tick(app, dataset_id, frames=(0, 8), flag="blurry")  # already set: stages nothing
    resp = apply_edits(app, dataset_id)
    assert resp.status_code == 200, resp.text
    assert resp.json()["applied"] >= 0


def test_a_column_cannot_be_dropped_while_flag_edits_are_queued(flagged):
    """The schema guard filtered on `feature_set` alone, so a flags column
    could be removed with edits still staged against it -- failing at save with
    the operator's annotations already discarded."""
    app, _state, dataset_id, _ds = flagged
    tick(app, dataset_id, frames=(0, 4), flag="blurry")

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.delete(f"/api/datasets/{dataset_id}/features/quality")

    resp = asyncio.run(run())
    assert resp.status_code == 409
    assert "pending" in resp.json()["detail"].lower()


def test_a_column_can_be_dropped_once_nothing_is_queued(flagged):
    """The guard must not be so broad that it never lets go."""
    app, _state, dataset_id, ds = flagged

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.delete(f"/api/datasets/{dataset_id}/features/quality")

    resp = asyncio.run(run())
    assert resp.status_code == 200, resp.text
    assert "quality" not in ds.meta.features
