# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""An apply-while-playing run is a pending EDIT, not a write.

Ticking Apply stores nothing; playing produces masks that join one growing
pending edit, and Save commits it while Discard throws the run away whole. This
covers that edit: its coalescing, when it declares a label, and the write rule
it obeys when it finally lands.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lerobot.datasets.mask_codec import encode_mask
from lerobot.datasets.mask_store import adopt, labels_of, read_frame, states, write_episode

H, W = 32, 48
CAM = "observation.images.top"
CAM2 = "observation.images.wrist"


def _stripe(a: int, b: int) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[a:b] = True
    return m


@pytest.fixture
def client_with_masks(tmp_path, info_factory, lerobot_dataset_factory):
    import random

    random.seed(0)
    np.random.seed(0)
    from fastapi import FastAPI

    from lerobot.gui.api import datasets as ds_api, edits as edits_api
    from lerobot.gui.frame_cache import FrameCache
    from lerobot.gui.state import AppState

    motors = {
        "action": {"dtype": "float32", "shape": (6,), "names": [f"j{i}" for i in range(6)]},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": [f"j{i}" for i in range(6)]},
    }
    cams = {
        CAM: {"shape": (H, W, 3), "names": ["height", "width", "channels"], "info": None},
        CAM2: {"shape": (H, W, 3), "names": ["height", "width", "channels"], "info": None},
    }
    info = info_factory(
        total_episodes=2, total_frames=12, total_tasks=1, motor_features=motors, camera_features=cams
    )
    ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=12, info=info)
    n = int(ds.meta.episodes["length"][0])
    assert n >= 4
    adopt(ds, [CAM, CAM2], ["ball"], (H, W))
    # Episode 0 starts EMPTY on the top camera: a run fills gaps, so there has
    # to be a gap for it to fill.
    write_episode(ds, 0, CAM, [{} for _ in range(n)])
    write_episode(ds, 0, CAM2, [{} for _ in range(n)])

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
            yield c, ds, state, n
    finally:
        ds_api._app_state = original
        edits_api._app_state = original_edits


def _flush(client, rows, episode=0):
    return client.post(
        "/api/edits/mask-run",
        json={"dataset_id": "d", "episode_index": episode, "rows": rows},
    )


def _row(frame, labels, cam=CAM):
    return {"camera": cam, "frame": frame, "rle": {n: encode_mask(m) for n, m in labels.items()}}


def _save(client):
    return client.post("/api/edits/apply?dataset_id=d")


# ── one edit, extended ──────────────────────────────────────────────────────


def test_a_run_is_one_pending_edit_however_many_flushes(client_with_masks):
    """A 1,440-frame episode must not become 1,440 queue entries."""
    client, _ds, state, _n = client_with_masks
    for f in range(3):
        assert _flush(client, [_row(f, {"ball": _stripe(0, 8)})]).status_code == 200
    runs = [e for e in state.pending_edits if e.edit_type == "mask_run"]
    assert len(runs) == 1, f"{len(runs)} run edits after three flushes"
    assert len(runs[0].params["rows"]) == 3, "the flushes did not accumulate"


def test_a_frame_resent_in_one_run_keeps_the_first_masks(client_with_masks):
    """The write rule fills a gap once, so a repeat is a repeat, not a
    correction — and a run that re-sent frames would otherwise grow without
    bound on a loop."""
    client, _ds, state, _n = client_with_masks
    _flush(client, [_row(0, {"ball": _stripe(0, 8)})])
    first = dict(next(e for e in state.pending_edits if e.edit_type == "mask_run").params["rows"])
    _flush(client, [_row(0, {"ball": _stripe(20, 28)})])
    after = next(e for e in state.pending_edits if e.edit_type == "mask_run").params["rows"]
    assert after == first, "a re-sent frame overwrote the run's earlier masks"


def test_nothing_is_written_until_save(client_with_masks):
    """Ticking Apply is not a write, and neither is playing: the run stages."""
    client, ds, _state, _n = client_with_masks
    _flush(client, [_row(0, {"ball": _stripe(0, 8)})])
    assert read_frame(ds, 0, 0, CAM) == {}, "the run reached the dataset before Save"


# ── when a label becomes real ───────────────────────────────────────────────


def test_a_new_label_is_declared_on_the_first_flush(client_with_masks):
    """Not on tick — a slot in a positional vocabulary can never be taken back,
    so a label declared when Apply is ticked would outlive a run cancelled a
    second later. Declaring on the first flush is also what makes the new lane
    appear and fill while the operator watches."""
    client, ds, _state, _n = client_with_masks
    assert "cube" not in labels_of(ds, CAM)
    _flush(client, [_row(0, {"cube": _stripe(0, 8)})])
    assert "cube" in labels_of(ds, CAM), "the label was not declared on the first flush"


def test_a_label_is_declared_in_every_camera_column(client_with_masks):
    """A label names an object; the same object in three views is one label. Rows
    still land only where the run looked."""
    client, ds, _state, _n = client_with_masks
    _flush(client, [_row(0, {"cube": _stripe(0, 8)}, cam=CAM)])
    assert "cube" in labels_of(ds, CAM) and "cube" in labels_of(ds, CAM2), (
        "the declaration did not reach every mask column"
    )


def test_a_run_that_never_flushes_declares_nothing(client_with_masks):
    """The complement: the vocabulary must be untouched by arming alone."""
    _client, ds, _state, _n = client_with_masks
    assert labels_of(ds, CAM) == ["ball"], labels_of(ds, CAM)


# ── the write rule, at save ─────────────────────────────────────────────────


def test_save_fills_only_where_the_label_was_absent(client_with_masks):
    client, ds, _state, n = client_with_masks
    write_episode(ds, 0, CAM, [{"ball": _stripe(0, 4)} if f == 1 else {} for f in range(n)])
    _flush(client, [_row(f, {"ball": _stripe(20, 28)}) for f in range(3)])
    assert _save(client).status_code == 200

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds2 = LeRobotDataset(ds.repo_id, root=ds.root)
    assert "ball" in read_frame(ds2, 0, 0, CAM), "an absent frame was not filled"
    kept = read_frame(ds2, 0, 1, CAM)["ball"]
    assert kept[:4].any() and not kept[20:].any(), "a frame that already had the label was overwritten"


def test_save_leaves_a_disabled_mask_disabled(client_with_masks):
    """The case muting exists for: a run must not put back a detection the
    operator rejected."""
    client, ds, _state, n = client_with_masks
    from lerobot.datasets.mask_store import set_label_enabled

    write_episode(ds, 0, CAM, [{"ball": _stripe(0, 4)} for _ in range(n)])
    set_label_enabled(ds, 0, CAM, "ball", False, frames=range(0, 2))
    _flush(client, [_row(f, {"ball": _stripe(20, 28)}) for f in range(2)])
    assert _save(client).status_code == 200

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds2 = LeRobotDataset(ds.repo_id, root=ds.root)
    st = states(ds2, 0, CAM)
    assert st[0]["ball"] is False and st[1]["ball"] is False, "the run re-enabled a muted mask"


def test_discard_throws_the_run_away_whole(client_with_masks):
    client, ds, state, _n = client_with_masks
    _flush(client, [_row(f, {"ball": _stripe(0, 8)}) for f in range(3)])
    assert client.post("/api/edits/discard", json={"dataset_id": "d"}).status_code in (200, 204)
    assert not [e for e in state.pending_edits if e.edit_type == "mask_run"]
    assert read_frame(ds, 0, 0, CAM) == {}


# ── validation ──────────────────────────────────────────────────────────────


def test_a_frame_outside_the_episode_is_refused(client_with_masks):
    client, _ds, _state, n = client_with_masks
    assert _flush(client, [_row(n + 5, {"ball": _stripe(0, 8)})]).status_code >= 400


def test_a_camera_without_masks_is_refused(client_with_masks):
    client, _ds, _state, _n = client_with_masks
    bad = {"camera": "observation.images.nope", "frame": 0, "rle": {"ball": "P"}}
    assert _flush(client, [bad]).status_code >= 400


def test_a_run_adding_a_label_carries_the_row_s_existing_flags(client_with_masks):
    """The row is REWRITTEN when a run fills a gap on it, so every flag already
    on that row has to be carried across.

    `test_save_leaves_a_disabled_mask_disabled` does not reach this: there the
    only label is the disabled one, so the write rule skips the frame entirely
    and the row is never rebuilt. The flags only have to survive when the run
    adds something else to the same frame — a mask muted at frame 0 while
    `cube` is discovered there must still be muted afterwards.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.mask_store import set_label_enabled

    client, ds, _state, n = client_with_masks
    write_episode(ds, 0, CAM, [{"ball": _stripe(0, 4)} for _ in range(n)])
    set_label_enabled(ds, 0, CAM, "ball", False, frames=range(0, 2))
    assert states(ds, 0, CAM)[0]["ball"] is False

    # The run finds a DIFFERENT object on the same frames.
    _flush(client, [_row(f, {"cube": _stripe(20, 28)}) for f in range(2)])
    assert _save(client).status_code == 200

    ds2 = LeRobotDataset(ds.repo_id, root=ds.root)
    st = states(ds2, 0, CAM)
    assert "cube" in st[0], "the run's new label did not land"
    assert st[0]["ball"] is False, "rewriting the row re-enabled a muted mask"
    assert st[1]["ball"] is False


def test_a_run_against_an_unadopted_dataset_is_refused_not_ignored(
    tmp_path, info_factory, lerobot_dataset_factory
):
    """Every dataset has no mask column until one is adopted, so this is the
    FIRST thing apply-while-playing meets on anything new.

    The endpoint refuses with 400. `fetch` resolves for 4xx rather than throwing,
    so the client's try/catch never saw it: the run played on, segmenting every
    frame and discarding every result to the end of the episode, with no error
    anywhere. Measured on a real 274-episode dataset -- 75 frames played, zero
    pending edits, no message.

    Pinned at the endpoint so the refusal, and a message naming what is missing,
    stay a contract rather than an accident of phrasing.
    """
    from fastapi import FastAPI

    from lerobot.gui.api import datasets as ds_api, edits as edits_api
    from lerobot.gui.frame_cache import FrameCache
    from lerobot.gui.state import AppState

    motors = {
        "action": {"dtype": "float32", "shape": (6,), "names": [f"j{i}" for i in range(6)]},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": [f"j{i}" for i in range(6)]},
    }
    cams = {CAM: {"shape": (H, W, 3), "names": ["height", "width", "channels"], "info": None}}
    info = info_factory(
        total_episodes=1, total_frames=6, total_tasks=1, motor_features=motors, camera_features=cams
    )
    # Deliberately NOT adopted: no `adopt(...)` call, which is the state every
    # dataset is in before its first save.
    ds = lerobot_dataset_factory(root=tmp_path / "raw", total_episodes=1, total_frames=6, info=info)

    app = FastAPI()
    app.include_router(ds_api.router)
    app.include_router(edits_api.router)
    original, original_edits = ds_api._app_state, edits_api._app_state
    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    ds_api.set_app_state(state)
    edits_api.set_app_state(state)
    state.datasets["raw"] = ds
    try:
        with TestClient(app) as c:
            resp = c.post(
                "/api/edits/mask-run",
                json={
                    "dataset_id": "raw",
                    "episode_index": 0,
                    "rows": [{"camera": CAM, "frame": 0, "rle": {"ball": "abc"}}],
                },
            )
        assert resp.status_code == 400, f"an unadopted dataset must refuse, got {resp.status_code}"
        detail = str(resp.json().get("detail", ""))
        assert "mask column" in detail, f"the refusal must name what is missing: {detail!r}"
    finally:
        ds_api._app_state = original
        edits_api._app_state = original_edits


def test_saving_a_run_refuses_a_frame_outside_the_episode(client_with_masks):
    """A pending edit outlives the proposal that made it.

    `propose_mask_run` bounds the frame when the run stages it. The save runs
    later, against a dataset that may have changed, and an out-of-range frame
    there is not an error but a wrong WRITE: `column[-1]` reads the episode's
    last row, and `start + frame` addresses the previous episode's frames. So
    the bound is checked again where the write happens.
    """
    from lerobot.gui.api._edits_core import EditValidationError, apply_mask_run

    c, ds, state, n = client_with_masks
    for bad in (-1, n + 5):
        with pytest.raises(EditValidationError, match="outside episode"):
            apply_mask_run(ds, 0, {"rows": {f"{CAM}:{bad}": {"ball": encode_mask(_stripe(0, 4))}}})

    # And the complement: an in-range frame still writes, or the guard above is
    # satisfied by a function that refuses everything.
    changed = apply_mask_run(ds, 0, {"rows": {f"{CAM}:0": {"ball": encode_mask(_stripe(0, 4))}}})
    assert changed == 1, "the in-range write was refused too"


def test_a_fill_between_staging_and_saving_is_not_overwritten(client_with_masks):
    """Pending edits and the dataset-wide fill both write masks, and neither
    invalidates the other. The reconciliation is that a pending edit is lowered
    against the rows as they are at SAVE time, not as they were when staged.

    So a fill that lands between the two -- which is exactly what happens when an
    apply run is staged and the operator then fills the dataset -- keeps its
    masks: the write rule re-runs here and the staged pairs are no longer absent.
    Nothing is silently replaced, and the fill is not undone.
    """
    from lerobot.datasets.mask_store import read_frame, write_episode
    from lerobot.gui.api._edits_core import apply_mask_run

    c, ds, state, n = client_with_masks
    staged = {f"{CAM}:{f}": {"ball": encode_mask(_stripe(0, 4))} for f in range(3)}

    # The fill gets there first, with a DIFFERENT mask for the same label.
    fill_mask = _stripe(20, 30)
    write_episode(ds, 0, CAM, [{"ball": fill_mask} for _ in range(n)])
    before = read_frame(ds, 0, 0, CAM)["ball"].copy()

    changed = apply_mask_run(ds, 0, {"rows": staged})
    assert changed == 0, f"the staged run overwrote {changed} frame(s) the fill had already written"
    after = read_frame(ds, 0, 0, CAM)["ball"]
    assert (after == before).all(), "the fill's pixels were replaced by the staged run's"


def test_a_pending_run_still_fills_what_the_fill_missed(client_with_masks):
    """The complement: the write rule leaves what is there and fills what is not,
    so a staged run is not discarded wholesale either -- without this the test
    above passes for a save that does nothing at all.
    """
    from lerobot.datasets.mask_store import read_frame, write_episode
    from lerobot.gui.api._edits_core import apply_mask_run

    c, ds, state, n = client_with_masks
    # The fill covered only frame 0; the run staged frames 0..2.
    rows = [{"ball": _stripe(20, 30)}] + [{} for _ in range(n - 1)]
    write_episode(ds, 0, CAM, rows)

    staged = {f"{CAM}:{f}": {"ball": encode_mask(_stripe(0, 4))} for f in range(3)}
    changed = apply_mask_run(ds, 0, {"rows": staged})
    assert changed == 2, f"expected the two absent frames to be filled, got {changed}"

    # Re-open to read back: an in-place write changes parquet under the cached
    # `hf_dataset`, which keeps serving pre-save rows. That is the whole reason
    # the server rebinds the dataset when a job finishes.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    fresh = LeRobotDataset(ds.repo_id, root=ds.root)
    assert "ball" in read_frame(fresh, 0, 1, CAM), "the frame the fill missed was not filled"
