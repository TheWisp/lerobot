# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The segment-level edits the timeline makes, staged then applied over HTTP.

These cross the process boundary the UI actually uses -- a request body in,
stored rows out -- rather than calling `mask_store` directly, because staging
is where the frame range, the action verb and the label have to survive
translation, and because a staged edit that never lowers correctly at save time
looks exactly like one that was never made.

Nothing is written until Save. That is what makes Discard the undo, and it is
why deleting a segment needs no confirmation dialog.

Three states have to stay distinguishable end to end: the two bitsets the
timeline draws from must never report the same label as both enabled and
disabled, and must report nothing at all where it is absent.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lerobot.datasets.mask_store import adopt, read_frame, states, write_episode

H, W = 32, 48
CAM = "observation.images.top"


def _stripe(a: int, b: int) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[a:b] = True
    return m


@pytest.fixture
def client_with_masks(tmp_path, info_factory, lerobot_dataset_factory, monkeypatch):
    import random

    random.seed(0)
    np.random.seed(0)
    from lerobot.gui.api import datasets as ds_api

    motors = {
        "action": {"dtype": "float32", "shape": (6,), "names": [f"j{i}" for i in range(6)]},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": [f"j{i}" for i in range(6)]},
    }
    cams = {CAM: {"shape": (H, W, 3), "names": ["height", "width", "channels"], "info": None}}
    info = info_factory(
        total_episodes=2, total_frames=12, total_tasks=1, motor_features=motors, camera_features=cams
    )
    ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=12, info=info)
    n = int(ds.meta.episodes["length"][0])
    assert n >= 4, "this test edits sub-ranges of episode 0"
    adopt(ds, [CAM], ["ball", "tray"], (H, W))
    write_episode(ds, 0, CAM, [{"ball": _stripe(0, 8), "tray": _stripe(16, 24)} for _ in range(n)])

    from fastapi import FastAPI

    from lerobot.gui.api import edits as edits_api
    from lerobot.gui.frame_cache import FrameCache
    from lerobot.gui.state import AppState

    app = FastAPI()
    app.include_router(ds_api.router)
    app.include_router(edits_api.router)
    original, original_edits = ds_api._app_state, edits_api._app_state
    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    ds_api.set_app_state(state)
    edits_api.set_app_state(state)
    state.datasets["d"] = ds
    try:
        with TestClient(app) as c:
            yield c, ds, n
    finally:
        ds_api._app_state = original
        edits_api._app_state = original_edits


def _stage(client, body):
    """Stage one segment edit. Nothing is written yet."""
    return client.post(
        "/api/edits/mask-range",
        json={"dataset_id": "d", "episode_index": 0, **body},
    )


def _save(client):
    r = client.post("/api/edits/apply", params={"dataset_id": "d"})
    assert r.status_code == 200, r.text
    assert not r.json().get("errors"), r.json()
    return r


def _post(client, body):
    """Stage and commit, for the tests that assert on stored rows."""
    r = _stage(client, body)
    if r.status_code != 200:
        return r
    _save(client)
    return r


def test_disabling_a_range_mutes_only_those_frames(client_with_masks):
    client, ds, n = client_with_masks
    r = _post(client, {"camera": CAM, "label": "ball", "from_frame": 1, "to_frame": 3, "action": "disable"})
    assert r.status_code == 200, r.text
    assert r.json()["frames"] == 2
    got = states(ds, 0, CAM)
    assert [s["ball"] for s in got] == [True, False, False] + [True] * (n - 3)
    assert all(s["tray"] for s in got), "tray was not the target"


def test_a_disabled_mask_leaves_the_training_read(client_with_masks):
    client, ds, _ = client_with_masks
    _post(client, {"camera": CAM, "label": "ball", "from_frame": 0, "to_frame": 1, "action": "disable"})
    assert set(read_frame(ds, 0, 0, CAM)) == {"tray"}


def test_enabling_restores_the_same_pixels(client_with_masks):
    """Muting is reversible; that is the whole difference from deleting."""
    client, ds, _ = client_with_masks
    _post(client, {"camera": CAM, "label": "ball", "from_frame": 0, "to_frame": 2, "action": "disable"})
    _post(client, {"camera": CAM, "label": "ball", "from_frame": 0, "to_frame": 2, "action": "enable"})
    assert np.array_equal(read_frame(ds, 0, 0, CAM)["ball"], _stripe(0, 8))


def test_deleting_returns_the_range_to_absent(client_with_masks):
    client, ds, _ = client_with_masks
    r = _post(client, {"camera": CAM, "label": "ball", "from_frame": 0, "to_frame": 2, "action": "delete"})
    assert r.json()["frames"] == 2
    got = states(ds, 0, CAM)
    assert "ball" not in got[0] and "ball" not in got[1]
    assert got[2]["ball"] is True


def test_a_range_spanning_a_gap_skips_rather_than_refuses(client_with_masks):
    """A selection that covers frames without the label is normal, not an error."""
    client, ds, n = client_with_masks
    _post(client, {"camera": CAM, "label": "ball", "from_frame": 0, "to_frame": 2, "action": "delete"})
    r = _post(client, {"camera": CAM, "label": "ball", "from_frame": 0, "to_frame": n, "action": "disable"})
    assert r.status_code == 200, r.text
    got = states(ds, 0, CAM)
    assert "ball" not in got[0] and "ball" not in got[1], "deleted frames stayed deleted"
    assert all(s["ball"] is False for s in got[2:]), "the rest were muted"


@pytest.mark.parametrize(
    ("body", "why"),
    [
        ({"from_frame": -1, "to_frame": 2}, "negative start"),
        ({"from_frame": 0, "to_frame": 999}, "past the end"),
        ({"from_frame": 2, "to_frame": 2}, "empty range"),
        ({"from_frame": 3, "to_frame": 1}, "reversed"),
    ],
)
def test_a_range_outside_the_episode_is_refused(client_with_masks, body, why):
    """An off-by-one here would edit the neighbouring episode's frames."""
    client, _, _ = client_with_masks
    r = _post(client, {"camera": CAM, "label": "ball", "action": "disable", **body})
    assert r.status_code == 400, f"{why}: {r.text}"


def test_an_unknown_action_is_refused(client_with_masks):
    client, _, _ = client_with_masks
    r = _post(client, {"camera": CAM, "label": "ball", "from_frame": 0, "to_frame": 1, "action": "clear"})
    assert r.status_code == 400


def test_an_unknown_label_is_refused(client_with_masks):
    client, _, _ = client_with_masks
    r = _post(client, {"camera": CAM, "label": "ghost", "from_frame": 0, "to_frame": 1, "action": "disable"})
    assert r.status_code == 400


def test_the_two_bitsets_never_overlap_and_cover_three_states(client_with_masks):
    """What the timeline draws. A label reports in exactly one bitset, or in
    neither when absent -- otherwise a lane cannot tell muted from missing."""
    from lerobot.datasets.mask_codec import encode_frame
    from lerobot.gui.api.datasets import _mask_disabled_bits, _mask_presence_bits

    labels = ["ball", "tray"]
    masks = {"ball": _stripe(0, 8), "tray": _stripe(16, 24)}
    both = encode_frame(masks, labels)
    muted = encode_frame(masks, labels, disabled=["ball"])
    empty = encode_frame({}, labels)

    assert (_mask_presence_bits(both), _mask_disabled_bits(both)) == (0b11, 0)
    assert (_mask_presence_bits(muted), _mask_disabled_bits(muted)) == (0b10, 0b01)
    assert (_mask_presence_bits(empty), _mask_disabled_bits(empty)) == (0, 0)
    for row in (both, muted, empty):
        assert _mask_presence_bits(row) & _mask_disabled_bits(row) == 0, "a label reported twice"


# ── staging: nothing is written until Save, and the queue stays readable ─────


def _pending(client):
    return client.get("/api/edits", params={"dataset_id": "d"}).json()["edits"]


def test_staging_writes_nothing(client_with_masks):
    """Discard is the undo, which is only true if staging touched no rows."""
    client, ds, _ = client_with_masks
    before = states(ds, 0, CAM)
    r = _stage(client, {"camera": CAM, "label": "ball", "from_frame": 0, "to_frame": 2, "action": "delete"})
    assert r.status_code == 200, r.text
    assert states(ds, 0, CAM) == before, "a staged edit reached the rows"
    assert len(_pending(client)) == 1


def test_discard_undoes_a_staged_delete(client_with_masks):
    client, ds, _ = client_with_masks
    before = states(ds, 0, CAM)
    _stage(client, {"camera": CAM, "label": "ball", "from_frame": 0, "to_frame": 2, "action": "delete"})
    assert client.post("/api/edits/discard", params={"dataset_id": "d"}).status_code == 200
    assert _pending(client) == []
    assert states(ds, 0, CAM) == before


def test_touching_spans_of_one_label_coalesce(client_with_masks):
    """Dragging across adjacent segments must not fill the queue: [0,2) and
    [2,4) are one span to the eye and must be one entry."""
    client, _, _ = client_with_masks
    _stage(client, {"camera": CAM, "label": "ball", "from_frame": 0, "to_frame": 2, "action": "disable"})
    r = _stage(client, {"camera": CAM, "label": "ball", "from_frame": 2, "to_frame": 4, "action": "disable"})
    assert r.json()["merged"] == 1
    pend = _pending(client)
    assert len(pend) == 1
    assert (pend[0]["params"]["from_frame"], pend[0]["params"]["to_frame"]) == (0, 4)


def test_disjoint_spans_stay_separate(client_with_masks):
    """Coalescing must not swallow a gap the operator deliberately left."""
    client, _, n = client_with_masks
    _stage(client, {"camera": CAM, "label": "ball", "from_frame": 0, "to_frame": 1, "action": "disable"})
    _stage(client, {"camera": CAM, "label": "ball", "from_frame": 3, "to_frame": 4, "action": "disable"})
    assert len(_pending(client)) == 2


def test_a_later_edit_supersedes_an_earlier_one_over_the_same_frames(client_with_masks):
    """The queue records the intended END STATE, not the click history. Two
    entries that cancel at save time would read as two changes to review."""
    client, _, _ = client_with_masks
    _stage(client, {"camera": CAM, "label": "ball", "from_frame": 0, "to_frame": 2, "action": "disable"})
    _stage(client, {"camera": CAM, "label": "ball", "from_frame": 0, "to_frame": 2, "action": "delete"})
    pend = _pending(client)
    assert len(pend) == 1
    assert pend[0]["params"]["action"] == "delete", "the last intent is the one that stands"


def test_undoing_an_edit_empties_the_queue(client_with_masks):
    """Disable a segment, re-enable the same one: the dataset is back where it
    started, so there is nothing left to save."""
    client, ds, _ = client_with_masks
    before = states(ds, 0, CAM)
    _stage(client, {"camera": CAM, "label": "ball", "from_frame": 0, "to_frame": 2, "action": "disable"})
    assert len(_pending(client)) == 1
    r = _stage(client, {"camera": CAM, "label": "ball", "from_frame": 0, "to_frame": 2, "action": "enable"})
    assert r.json()["pending"] is False, "re-enabling reported a change that is not one"
    assert _pending(client) == [], "the queue still holds a round trip that nets to nothing"
    assert states(ds, 0, CAM) == before


def test_an_edit_matching_what_is_stored_stages_nothing(client_with_masks):
    """Enabling something already enabled is a change to review that is not a
    change."""
    client, _, _ = client_with_masks
    r = _stage(client, {"camera": CAM, "label": "ball", "from_frame": 0, "to_frame": 2, "action": "enable"})
    assert r.json()["pending"] is False
    assert _pending(client) == []


def test_a_partial_overlap_trims_rather_than_dropping_the_rest(client_with_masks):
    """Superseding must not discard intent the newer edit does not cover."""
    client, _, n = client_with_masks
    _stage(client, {"camera": CAM, "label": "ball", "from_frame": 0, "to_frame": n, "action": "disable"})
    _stage(client, {"camera": CAM, "label": "ball", "from_frame": 0, "to_frame": 2, "action": "delete"})
    pend = sorted(_pending(client), key=lambda e: e["params"]["from_frame"])
    assert len(pend) == 2, pend
    assert (pend[0]["params"]["action"], pend[0]["params"]["from_frame"], pend[0]["params"]["to_frame"]) == (
        "delete",
        0,
        2,
    )
    assert (pend[1]["params"]["action"], pend[1]["params"]["from_frame"]) == ("disable", 2), (
        "the disable lost the frames the delete did not cover"
    )


def test_a_different_label_does_not_merge(client_with_masks):
    client, _, _ = client_with_masks
    _stage(client, {"camera": CAM, "label": "ball", "from_frame": 0, "to_frame": 2, "action": "disable"})
    _stage(client, {"camera": CAM, "label": "tray", "from_frame": 0, "to_frame": 2, "action": "disable"})
    assert len(_pending(client)) == 2


def test_two_labels_edited_over_the_same_frames_compose(client_with_masks):
    """Lowering happens at save time against the rows as they then are, so the
    second edit must not overwrite the first."""
    client, ds, _ = client_with_masks
    _stage(client, {"camera": CAM, "label": "ball", "from_frame": 0, "to_frame": 2, "action": "disable"})
    _stage(client, {"camera": CAM, "label": "tray", "from_frame": 0, "to_frame": 2, "action": "disable"})
    _save(client)
    assert states(ds, 0, CAM)[0] == {"ball": False, "tray": False}


def test_staging_an_unknown_label_is_refused(client_with_masks):
    client, _, _ = client_with_masks
    r = _stage(client, {"camera": CAM, "label": "ghost", "from_frame": 0, "to_frame": 1, "action": "disable"})
    assert r.status_code >= 400
    assert _pending(client) == []


# ── what the whole-dataset dialog needs to know ─────────────────────────────


def test_label_coverage_counts_episodes_and_frames(client_with_masks):
    """A label found in one episode out of many is local to it; running it
    everywhere would spend hours finding nothing. The dialog shows this so the
    choice is obvious rather than a memory test."""
    client, ds, n = client_with_masks
    r = client.get("/api/datasets/d/masks/label-coverage")
    assert r.status_code == 200, r.text
    body = r.json()
    by_name = {x["name"]: x for x in body["labels"]}
    assert set(by_name) == {"ball", "tray"}, by_name
    assert body["total_episodes"] == ds.meta.total_episodes
    # Episode 0 was written; episode 1 never was.
    assert by_name["ball"]["episodes"] == 1
    assert by_name["ball"]["frames"] == n


def test_label_coverage_counts_a_disabled_label_as_seen(client_with_masks):
    """Muting says "this detection is wrong here", not "this object does not
    appear". Excluding it would under-report a label that is in fact all over
    the dataset."""
    client, ds, n = client_with_masks
    _post(client, {"camera": CAM, "label": "ball", "from_frame": 0, "to_frame": n, "action": "disable"})
    by_name = {x["name"]: x for x in client.get("/api/datasets/d/masks/label-coverage").json()["labels"]}
    assert by_name["ball"]["episodes"] == 1, "a muted label vanished from the coverage count"
    assert by_name["ball"]["frames"] == n


def test_label_coverage_drops_a_deleted_label_to_zero(client_with_masks):
    """The complement: deleting really does remove it, so the count is not just
    reporting the vocabulary back."""
    client, ds, n = client_with_masks
    _post(client, {"camera": CAM, "label": "ball", "from_frame": 0, "to_frame": n, "action": "delete"})
    by_name = {x["name"]: x for x in client.get("/api/datasets/d/masks/label-coverage").json()["labels"]}
    assert by_name["ball"]["episodes"] == 0, "a deleted label is still counted as seen"
    assert by_name["tray"]["episodes"] == 1, "the other label was collateral"


def test_label_coverage_counts_an_episode_once_across_cameras(
    tmp_path, info_factory, lerobot_dataset_factory, monkeypatch
):
    """A label detected in two cameras of one episode has been seen in ONE
    episode. Counting per camera reported "seen in 4/2 ep" on a two-episode,
    two-camera dataset -- a count larger than the dataset itself, in the one
    place meant to make the choice obvious.

    Needs TWO cameras carrying the same label, which is why it builds its own
    dataset rather than reusing the single-camera fixture: there, a per-camera
    count and a union are the same number and the defect is invisible.
    """
    import random

    from fastapi import FastAPI

    from lerobot.gui.api import datasets as ds_api
    from lerobot.gui.frame_cache import FrameCache
    from lerobot.gui.state import AppState

    random.seed(0)
    np.random.seed(0)
    cam2 = "observation.images.wrist"
    motors = {
        "action": {"dtype": "float32", "shape": (6,), "names": [f"j{i}" for i in range(6)]},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": [f"j{i}" for i in range(6)]},
    }
    cams = {
        c: {"shape": (H, W, 3), "names": ["height", "width", "channels"], "info": None} for c in (CAM, cam2)
    }
    info = info_factory(
        total_episodes=2, total_frames=12, total_tasks=1, motor_features=motors, camera_features=cams
    )
    ds = lerobot_dataset_factory(root=tmp_path / "two", total_episodes=2, total_frames=12, info=info)
    adopt(ds, [CAM, cam2], ["ball"], (H, W))
    for cam in (CAM, cam2):
        for ep in range(2):
            n = int(ds.meta.episodes["length"][ep])
            write_episode(ds, ep, cam, [{"ball": _stripe(0, 8)}] * n)

    app = FastAPI()
    app.include_router(ds_api.router)
    original = ds_api._app_state
    ds_api.set_app_state(AppState(frame_cache=FrameCache(max_bytes=1_000_000)))
    ds_api._app_state.datasets["two"] = ds
    try:
        with TestClient(app) as c:
            body = c.get("/api/datasets/two/masks/label-coverage").json()
    finally:
        ds_api._app_state = original

    total = body["total_episodes"]
    assert total == 2
    row = next(x for x in body["labels"] if x["name"] == "ball")
    assert row["episodes"] == 2, (
        f"ball is in both episodes of a 2-episode set, reported {row['episodes']} "
        "-- the count is summing cameras rather than taking their union"
    )
