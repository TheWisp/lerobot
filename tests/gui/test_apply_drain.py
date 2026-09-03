# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Resolving a run's masks back to the frames they were computed from.

The overlay worker reports the obs-stream sequence it CONSUMED. By the time a
batch reaches the server the playhead has usually moved on — a run is lock-step
at ~10 fps and the worker is up to a flush (~1 s) behind — so the position has
to be remembered from when the frame was published. Reading "where is the
playhead now" would file masks against the wrong frame under exactly the
conditions the feature exists for.
"""

from __future__ import annotations

import pytest

import lerobot.gui.api.overlays as ovl


@pytest.fixture(autouse=True)
def clean_apply_state():
    ovl._data_apply_pos.clear()
    ovl._data_apply_last_seq = -1
    ovl._data_apply_on = False
    ovl._data_pub_last_pos = None
    yield
    ovl._data_apply_pos.clear()
    ovl._data_apply_on = False


class _Reader:
    """Stands in for the worker's side of the shared block."""

    def __init__(self, batches):
        self._batches = list(batches)
        self._seq = 0

    def publish(self):
        self._seq += 1

    def masks_seq(self):
        return self._seq

    def read_masks(self):
        return self._batches


async def _drain(monkeypatch, reader):
    monkeypatch.setattr(ovl, "_get_live_reader", lambda: reader)
    return await ovl.apply_drain()


@pytest.mark.asyncio
async def test_masks_are_filed_against_the_frame_they_came_from(monkeypatch):
    cam = "observation.images.top"
    # The frame was published at episode 3, frame 100, and got seq 7.
    ovl._data_apply_pos[(cam, 7)] = (3, 100)
    reader = _Reader([{cam: {"seq": 7, "rle": {"ring": "PPP"}}}])
    reader.publish()

    out = await _drain(monkeypatch, reader)
    assert out["frames"] == [{"episode": 3, "frame": 100, "camera": cam, "rle": {"ring": "PPP"}}]


@pytest.mark.asyncio
async def test_a_sequence_with_no_remembered_position_is_dropped(monkeypatch):
    """Not guessed at. It means the frame predates the run being armed, or the
    bounded map has rotated past it — and a mask on the wrong frame is worse
    than a mask that never arrived."""
    cam = "observation.images.top"
    # The playhead has moved on, which is the normal state of affairs while a run
    # is in flight: the worker is up to a flush behind. A fallback to "wherever
    # the playhead is" would file these masks at frame 512 instead of dropping
    # them, so the test only means something with a current position set.
    ovl._data_pub_last_pos = (0, 512)
    reader = _Reader([{cam: {"seq": 999, "rle": {"ring": "PPP"}}}])
    reader.publish()

    out = await _drain(monkeypatch, reader)
    assert out["frames"] == [], "masks with no remembered position were filed anyway"
    assert out["dropped"] == 1


@pytest.mark.asyncio
async def test_each_camera_resolves_through_its_own_sequence(monkeypatch):
    """The image blocks' counters diverge whenever a camera is missing from a
    published frame, so one seq per entry would cross-file the cameras."""
    top, wrist = "observation.images.top", "observation.images.wrist"
    ovl._data_apply_pos[(top, 11)] = (0, 40)
    ovl._data_apply_pos[(wrist, 9)] = (0, 40)
    reader = _Reader([{top: {"seq": 11, "rle": {"a": "X"}}, wrist: {"seq": 9, "rle": {"a": "Y"}}}])
    reader.publish()

    out = await _drain(monkeypatch, reader)
    by_cam = {f["camera"]: f for f in out["frames"]}
    assert by_cam[top]["frame"] == 40 and by_cam[wrist]["frame"] == 40
    assert by_cam[top]["rle"] == {"a": "X"} and by_cam[wrist]["rle"] == {"a": "Y"}


@pytest.mark.asyncio
async def test_a_batch_is_handed_back_once(monkeypatch):
    """Polling is repeated; re-staging the same frames would multiply the run's
    edits and, with the write rule, quietly do nothing while looking busy."""
    cam = "observation.images.top"
    ovl._data_apply_pos[(cam, 1)] = (0, 0)
    reader = _Reader([{cam: {"seq": 1, "rle": {"ring": "P"}}}])
    reader.publish()

    assert len((await _drain(monkeypatch, reader))["frames"]) == 1
    assert (await _drain(monkeypatch, reader))["frames"] == [], "the same batch came back twice"

    reader.publish()  # a new batch from the worker
    assert len((await _drain(monkeypatch, reader))["frames"]) == 1


@pytest.mark.asyncio
async def test_no_worker_is_not_an_error(monkeypatch):
    monkeypatch.setattr(ovl, "_get_live_reader", lambda: None)
    assert await ovl.apply_drain() == {"frames": [], "dropped": 0}


@pytest.mark.asyncio
async def test_arming_is_not_a_write_and_disarming_forgets_positions():
    """Ticking the box stores nothing — it sets a mode. Disarming clears the
    remembered positions so a later run cannot resolve a stale sequence."""
    ovl._data_apply_pos[("cam", 1)] = (0, 0)
    out = await ovl.apply_arm(ovl.ApplyArmRequest(armed=True))
    assert out == {"armed": True} and ovl._data_apply_on is True

    out = await ovl.apply_arm(ovl.ApplyArmRequest(armed=False))
    assert out == {"armed": False} and ovl._data_apply_on is False
    assert not ovl._data_apply_pos, "positions survived disarming"
