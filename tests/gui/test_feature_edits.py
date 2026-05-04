# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the GUI's feature-value editing path:

* ``POST /api/edits/feature-set`` — staging.
* ``DELETE /api/edits/{edit_index}`` — per-chip removal already covers feature_set.
* ``POST /api/edits/apply`` — translating staged feature_set edits into a
  single ``set_feature_values`` call and reloading the dataset.
"""

from __future__ import annotations

import asyncio
import random

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI

from lerobot.gui.api import datasets as datasets_module, edits as edits_module
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState


@pytest.fixture(autouse=True)
def _seed_rng():
    """Seed numpy + random before each test to make ``lerobot_dataset_factory``
    deterministic.

    Why: ``episodes_factory`` distributes ``total_frames`` across episodes via
    ``np.random.multinomial``, so episode lengths drift with the global RNG
    state — which is consumed by other tests in the suite. Tests in this file
    assert specific frame ranges (e.g. ``[0, 9)``) that require a known
    minimum episode length, so the RNG must be seeded to make them
    order-independent.
    """
    np.random.seed(0)
    random.seed(0)
    yield


@pytest.fixture
def app_with_state():
    """Mounts both routers (datasets + edits) on the same FastAPI app and clears module state on teardown."""
    app = FastAPI()
    app.include_router(datasets_module.router)
    app.include_router(edits_module.router)

    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    original_dat_state = datasets_module._app_state
    original_edits_state = edits_module._app_state
    original_indices = datasets_module._episode_start_indices.copy()
    datasets_module.set_app_state(state)
    edits_module.set_app_state(state)

    yield app, state

    datasets_module._app_state = original_dat_state
    edits_module._app_state = original_edits_state
    datasets_module._episode_start_indices.clear()
    datasets_module._episode_start_indices.update(original_indices)


def _post_feature_set(app, body: dict):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post("/api/edits/feature-set", json=body)

    return asyncio.run(run())


def _list_pending(app, dataset_id: str | None = None):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            url = "/api/edits"
            if dataset_id:
                url += f"?dataset_id={dataset_id}"
            return await client.get(url)

    return asyncio.run(run())


def _post_apply(app, dataset_id: str):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(f"/api/edits/apply?dataset_id={dataset_id}")

    return asyncio.run(run())


def _delete_edit(app, idx: int):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.delete(f"/api/edits/{idx}")

    return asyncio.run(run())


# ── /api/edits/feature-set — staging & validation ──────────────────────────


class TestStageFeatureSet:
    def test_stages_a_valid_edit(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        # action is read-only — pick a feature that's NOT in the readonly list.
        editable = [
            n
            for n, ft in ds.meta.features.items()
            if ft["dtype"] not in ("image", "video")
            and n not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
            and n != "action"
            and not n.startswith("observation.")
        ]
        if not editable:
            pytest.skip("factory dataset has no editable features for V1 (only action/observation/defaults)")

        feature = editable[0]
        ft_info = ds.meta.features[feature]
        # Build a value matching the dtype.
        shape = ft_info.get("shape") or [1]
        if ft_info["dtype"] == "string":
            value = "ok"
        elif shape in ([1], []):
            value = 0.5 if ft_info["dtype"].startswith("float") else 1
        else:
            value = [0.5] * shape[0]

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 2,
                "frame_to": 5,
                "value": value,
            },
        )
        assert resp.status_code == 200, resp.text

        # The edit shows up in the pending list.
        pending = _list_pending(app, dataset_id).json()
        assert pending["total"] == 1
        assert pending["edits"][0]["edit_type"] == "feature_set"
        assert pending["edits"][0]["params"]["feature"] == feature
        assert pending["edits"][0]["params"]["frame_from"] == 2
        assert pending["edits"][0]["params"]["frame_to"] == 5

    def test_action_feature_rejected(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "action",
                "frame_from": 0,
                "frame_to": 5,
                "value": [0.0] * ds.meta.features["action"]["shape"][0],
            },
        )
        assert resp.status_code == 400
        assert "read-only" in resp.json()["detail"]

    def test_observation_feature_rejected(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        obs_features = [n for n in ds.meta.features if n.startswith("observation.")]
        if not obs_features:
            pytest.skip("no observation.* features in factory dataset")

        feature = obs_features[0]
        ft = ds.meta.features[feature]
        if ft["dtype"] in ("image", "video"):
            # That path returns a different message; pick a non-image observation if possible.
            non_img = [n for n in obs_features if ds.meta.features[n]["dtype"] not in ("image", "video")]
            if not non_img:
                pytest.skip("only image observation features in factory dataset")
            feature = non_img[0]
            ft = ds.meta.features[feature]

        shape = ft.get("shape") or [1]
        value = 0.0 if shape in ([1], []) else [0.0] * shape[0]
        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 0,
                "frame_to": 5,
                "value": value,
            },
        )
        assert resp.status_code == 400
        assert "read-only" in resp.json()["detail"]

    def test_default_feature_rejected(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "frame_index",
                "frame_from": 0,
                "frame_to": 5,
                "value": 0,
            },
        )
        assert resp.status_code == 400
        assert "auto-managed" in resp.json()["detail"]

    def test_unknown_feature_rejected(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "totally_made_up",
                "frame_from": 0,
                "frame_to": 5,
                "value": 0,
            },
        )
        assert resp.status_code == 400
        assert "Unknown feature" in resp.json()["detail"]

    def test_invalid_range_rejected(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        # Pick something editable and try frame_from >= frame_to.
        editable = [
            n
            for n, ft in ds.meta.features.items()
            if ft["dtype"] not in ("image", "video")
            and n not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
            and n != "action"
            and not n.startswith("observation.")
        ]
        if not editable:
            pytest.skip("no editable features in factory")

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": editable[0],
                "frame_from": 5,
                "frame_to": 5,  # empty range
                "value": 0,
            },
        )
        assert resp.status_code == 400
        assert "Invalid range" in resp.json()["detail"]


# ── per-chip removal works on feature_set edits too ────────────────────────


class TestRemoveFeatureSetEdit:
    def test_delete_removes_one_feature_set(self, app_with_state, tmp_path, lerobot_dataset_factory):
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        editable = [
            n
            for n, ft in ds.meta.features.items()
            if ft["dtype"] not in ("image", "video")
            and n not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
            and n != "action"
            and not n.startswith("observation.")
        ]
        if not editable:
            pytest.skip("no editable features in factory")
        feature = editable[0]
        ft = ds.meta.features[feature]
        shape = ft.get("shape") or [1]
        value = 0.0 if shape in ([1], []) else [0.0] * shape[0]

        # Stage two distinct edits.
        _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 0,
                "frame_to": 3,
                "value": value,
            },
        )
        _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 5,
                "frame_to": 8,
                "value": value,
            },
        )

        before = _list_pending(app, dataset_id).json()
        assert before["total"] == 2

        resp = _delete_edit(app, 0)
        assert resp.status_code == 200

        after = _list_pending(app, dataset_id).json()
        assert after["total"] == 1
        assert after["edits"][0]["params"]["frame_from"] == 5  # the second one survived


# ── apply pipeline writes to disk via set_feature_values ───────────────────


class TestApplyFeatureSet:
    def test_apply_persists_value_edit_to_disk(self, app_with_state, tmp_path, lerobot_dataset_factory):
        """End-to-end: stage an edit, hit /apply, confirm the parquet shard
        on disk now reflects the new value."""
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        # Find a non-action, non-observation, non-default editable scalar / vector feature.
        editable = [
            n
            for n, ft in ds.meta.features.items()
            if ft["dtype"] not in ("image", "video", "string")
            and n not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
            and n != "action"
            and not n.startswith("observation.")
        ]
        if not editable:
            pytest.skip("no editable numeric feature in factory dataset")

        feature = editable[0]
        ft = ds.meta.features[feature]
        shape = ft.get("shape") or [1]
        marker = -42.0 if shape in ([1], []) else [-42.0] * shape[0]

        # Stage an edit on episode 0, frames 1..4 (3 frames).
        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 1,
                "frame_to": 4,
                "value": marker,
            },
        )
        assert resp.status_code == 200, resp.text

        # Apply.
        resp = _post_apply(app, dataset_id)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["applied"] == 1, body

        # Read the data shard(s) directly and verify the edit landed.
        parts = []
        for shard in sorted((ds.root / "data").glob("*/*.parquet")):
            parts.append(pd.read_parquet(shard))
        df = pd.concat(parts, ignore_index=True).sort_values("index").reset_index(drop=True)
        ep0 = df[df["episode_index"] == 0].sort_values("frame_index").reset_index(drop=True)

        # Frames 1..3 (inclusive) should show the marker.
        for i in range(1, 4):
            cell = ep0.iloc[i][feature]
            arr = np.asarray(cell)
            if arr.ndim == 0:
                assert float(arr.item()) == pytest.approx(-42.0)
            else:
                assert np.allclose(arr, -42.0)

        # Frames 0 and 4 must be untouched (compare to baseline = anything not the marker).
        cell0 = np.asarray(ep0.iloc[0][feature])
        cell4 = np.asarray(ep0.iloc[4][feature])
        assert not (np.allclose(cell0, -42.0) if cell0.ndim else float(cell0) == -42.0)
        assert not (np.allclose(cell4, -42.0) if cell4.ndim else float(cell4) == -42.0)

        # And the pending edits list is now empty.
        pending = _list_pending(app, dataset_id).json()
        assert pending["total"] == 0


# ── Overlapping staged edits — 409 + clip-on-confirm ────────────────────────


class TestOverlappingEdits:
    def test_overlap_returns_409_with_structured_detail(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """Two staged edits on the same (feature, episode) with overlapping
        ranges must trigger a 409 unless ``confirm_overlap=True``."""
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        editable = [
            n
            for n, ft in ds.meta.features.items()
            if ft["dtype"] not in ("image", "video", "string")
            and n not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
            and n != "action"
            and not n.startswith("observation.")
        ]
        if not editable:
            pytest.skip("no editable feature in factory dataset")
        feature = editable[0]
        ft = ds.meta.features[feature]
        shape = ft.get("shape") or [1]
        value = 1.5 if shape in ([1], []) else [1.5] * shape[0]

        # First edit lands fine.
        r1 = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 0,
                "frame_to": 5,
                "value": value,
            },
        )
        assert r1.status_code == 200, r1.text

        # Overlapping second edit returns 409.
        r2 = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 3,
                "frame_to": 7,
                "value": value,
            },
        )
        assert r2.status_code == 409
        detail = r2.json()["detail"]
        assert detail["code"] == "overlapping_edit"
        assert detail["feature"] == feature
        assert detail["episode_index"] == 0
        assert detail["new_range"] == [3, 7]
        assert detail["overlapping"][0]["frame_from"] == 0
        assert detail["overlapping"][0]["frame_to"] == 5

    def test_confirm_overlap_clips_prior_edit(self, app_with_state, tmp_path, lerobot_dataset_factory):
        """With ``confirm_overlap=True``, the prior edit is clipped (last-write-wins)."""
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        editable = [
            n
            for n, ft in ds.meta.features.items()
            if ft["dtype"] not in ("image", "video", "string")
            and n not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
            and n != "action"
            and not n.startswith("observation.")
        ]
        if not editable:
            pytest.skip("no editable feature in factory dataset")
        feature = editable[0]
        ft = ds.meta.features[feature]
        shape = ft.get("shape") or [1]
        value = 1.5 if shape in ([1], []) else [1.5] * shape[0]

        _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 0,
                "frame_to": 5,
                "value": value,
            },
        )
        r2 = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 3,
                "frame_to": 7,
                "value": value,
                "confirm_overlap": True,
            },
        )
        assert r2.status_code == 200, r2.text

        pending = _list_pending(app, dataset_id).json()
        # We expect two edits now: the clipped prior [0, 3) and the new [3, 7).
        ranges = sorted(
            (int(e["params"]["frame_from"]), int(e["params"]["frame_to"]))
            for e in pending["edits"]
            if e["edit_type"] == "feature_set"
        )
        assert ranges == [(0, 3), (3, 7)], ranges

    def test_fully_contained_prior_is_removed(self, app_with_state, tmp_path, lerobot_dataset_factory):
        """If the new edit fully contains a prior one, the prior is removed entirely."""
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        editable = [
            n
            for n, ft in ds.meta.features.items()
            if ft["dtype"] not in ("image", "video", "string")
            and n not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
            and n != "action"
            and not n.startswith("observation.")
        ]
        if not editable:
            pytest.skip("no editable feature in factory dataset")
        feature = editable[0]
        ft = ds.meta.features[feature]
        shape = ft.get("shape") or [1]
        value = 1.5 if shape in ([1], []) else [1.5] * shape[0]

        # Prior edit at [3, 5) — fully inside the new [0, 8).
        _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 3,
                "frame_to": 5,
                "value": value,
            },
        )
        r2 = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 0,
                "frame_to": 8,
                "value": value,
                "confirm_overlap": True,
            },
        )
        assert r2.status_code == 200

        pending = _list_pending(app, dataset_id).json()
        ranges = [
            (int(e["params"]["frame_from"]), int(e["params"]["frame_to"]))
            for e in pending["edits"]
            if e["edit_type"] == "feature_set"
        ]
        assert ranges == [(0, 8)], ranges

    def test_prior_split_when_new_strictly_inside(self, app_with_state, tmp_path, lerobot_dataset_factory):
        """If the new edit is strictly inside the prior, the prior splits in two."""
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        editable = [
            n
            for n, ft in ds.meta.features.items()
            if ft["dtype"] not in ("image", "video", "string")
            and n not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
            and n != "action"
            and not n.startswith("observation.")
        ]
        if not editable:
            pytest.skip("no editable feature in factory dataset")
        feature = editable[0]
        ft = ds.meta.features[feature]
        shape = ft.get("shape") or [1]
        value = 1.5 if shape in ([1], []) else [1.5] * shape[0]

        # Prior wide edit, then new narrow inside.
        _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 0,
                "frame_to": 9,
                "value": value,
            },
        )
        r2 = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": feature,
                "frame_from": 3,
                "frame_to": 6,
                "value": value,
                "confirm_overlap": True,
            },
        )
        assert r2.status_code == 200

        pending = _list_pending(app, dataset_id).json()
        ranges = sorted(
            (int(e["params"]["frame_from"]), int(e["params"]["frame_to"]))
            for e in pending["edits"]
            if e["edit_type"] == "feature_set"
        )
        # Expect: left clip [0, 3), the new [3, 6), right clip [6, 9).
        assert ranges == [(0, 3), (3, 6), (6, 9)], ranges


# ── Subtask synthesis at the staging endpoint ──────────────────────────────


class TestSubtaskTranslation:
    """The user-facing ``subtask`` (string) translates to storage ``subtask_index``
    at stage time. Edits land in PendingEdit with the storage feature name so
    the apply pipeline doesn't need a special case. Resolution of new strings
    to fresh int indices happens at Save (covered by
    ``tests/datasets/test_set_feature_values.py::TestSubtaskStringResolution``).
    """

    @staticmethod
    def _ds_with_subtask_lookup(tmp_path, lerobot_dataset_factory):
        """Build a dataset and inject ``subtask_index`` + ``meta/subtasks.parquet``."""
        import numpy as np

        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        ds.meta.features["subtask_index"] = {"dtype": "int64", "shape": [1], "names": None}
        # Add a subtask_index column to each shard, varying so it's not detected as per-episode.
        for shard in (ds.root / "data").glob("*/*.parquet"):
            df = pd.read_parquet(shard)
            df["subtask_index"] = (df["frame_index"].astype(int) % 3).astype(np.int64)
            df.to_parquet(shard, compression="snappy", index=False)
        # Plant a starter lookup table.
        subtasks_df = pd.DataFrame(
            {"subtask_index": [0, 1, 2]},
            index=pd.Index(["approach", "grasp", "release"], name="subtask"),
        )
        (ds.root / "meta").mkdir(exist_ok=True)
        subtasks_df.to_parquet(ds.root / "meta" / "subtasks.parquet")
        # Reload metadata so the dataset picks up the new column + lookup.
        from lerobot.datasets.io_utils import load_subtasks

        ds.meta.subtasks = load_subtasks(ds.root)
        return ds

    def test_stage_subtask_translates_to_subtask_index_storage(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        app, state = app_with_state
        ds = self._ds_with_subtask_lookup(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "subtask",  # ← user-facing display name
                "frame_from": 1,
                "frame_to": 5,
                "value": "grasp",
            },
        )
        assert resp.status_code == 200, resp.text

        pending = _list_pending(app, dataset_id).json()
        assert pending["total"] == 1
        edit = pending["edits"][0]
        # PendingEdit stores the storage name — apply pipeline never sees "subtask".
        assert edit["params"]["feature"] == "subtask_index", edit
        assert edit["params"]["value"] == "grasp"

    def test_stage_subtask_with_brand_new_string_value(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """Strings absent from the lookup table are accepted at stage time —
        resolution to a fresh int index is deferred to Save by design (so
        concurrent new-string Saves on different sessions converge)."""
        app, state = app_with_state
        ds = self._ds_with_subtask_lookup(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "subtask",
                "frame_from": 0,
                "frame_to": 3,
                "value": "inspect knot",  # not in the lookup
            },
        )
        assert resp.status_code == 200, resp.text

    def test_stage_subtask_index_directly_still_works(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """The raw / programmatic API path: callers may use the storage name
        directly with an int value. Should pass through unchanged."""
        app, state = app_with_state
        ds = self._ds_with_subtask_lookup(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "subtask_index",
                "frame_from": 1,
                "frame_to": 4,
                "value": 2,
            },
        )
        assert resp.status_code == 200, resp.text
        pending = _list_pending(app, dataset_id).json()
        assert pending["edits"][0]["params"]["feature"] == "subtask_index"
        assert pending["edits"][0]["params"]["value"] == 2

    def test_stage_subtask_rejected_when_no_lookup_table(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """Without ``meta/subtasks.parquet`` the synthesis doesn't apply, so
        ``feature='subtask'`` falls through to the standard 'unknown feature'
        path — the dataset has no real ``subtask`` feature."""
        app, state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        # No subtask_index, no lookup table.
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "subtask",
                "frame_from": 0,
                "frame_to": 5,
                "value": "anything",
            },
        )
        assert resp.status_code == 400
        assert "Unknown feature" in resp.json()["detail"]

    def test_overlap_detection_normalizes_subtask_to_storage(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """Stage one edit as ``subtask`` and another as ``subtask_index`` on the
        same range — the overlap detector should see them as the same feature
        because both are stored under the storage name."""
        app, state = app_with_state
        ds = self._ds_with_subtask_lookup(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        # First stage uses the display name → stored as subtask_index.
        r1 = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "subtask",
                "frame_from": 0,
                "frame_to": 5,
                "value": "approach",
            },
        )
        assert r1.status_code == 200

        # Second stage uses the storage name on an overlapping range → 409.
        r2 = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "subtask_index",
                "frame_from": 3,
                "frame_to": 7,
                "value": 1,
            },
        )
        assert r2.status_code == 409, r2.text
        assert r2.json()["detail"]["code"] == "overlapping_edit"
        assert r2.json()["detail"]["feature"] == "subtask_index"

    def test_apply_subtask_string_lands_as_int_in_data(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """End-to-end: stage feature='subtask' with a string value, hit /apply,
        confirm the data parquet now has the resolved integer index in those rows."""
        app, state = app_with_state
        ds = self._ds_with_subtask_lookup(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        # Stage edit using the display name with an existing string value.
        resp = _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "subtask",
                "frame_from": 1,
                "frame_to": 4,
                "value": "release",  # already in lookup as index 2
            },
        )
        assert resp.status_code == 200, resp.text

        # Apply.
        apply_resp = _post_apply(app, dataset_id)
        assert apply_resp.status_code == 200, apply_resp.text
        assert apply_resp.json()["applied"] == 1

        # Read back from disk: frames [1, 4) should be the int index for "release".
        parts = [pd.read_parquet(p) for p in sorted((ds.root / "data").glob("*/*.parquet"))]
        merged = pd.concat(parts, ignore_index=True).sort_values("index").reset_index(drop=True)
        ep0 = merged[merged["episode_index"] == 0].sort_values("frame_index").reset_index(drop=True)

        # Look up the expected index from the (possibly updated) lookup table.
        subtasks_df = pd.read_parquet(ds.root / "meta" / "subtasks.parquet")
        release_idx = int(subtasks_df.loc["release", "subtask_index"])
        for fi in (1, 2, 3):
            assert int(ep0.iloc[fi]["subtask_index"]) == release_idx, (
                f"frame {fi} should be index {release_idx} (release), got {ep0.iloc[fi]['subtask_index']!r}"
            )
        # Untouched frames keep their (frame_index % 3) initialization.
        assert int(ep0.iloc[0]["subtask_index"]) == 0  # was: 0 % 3
        assert int(ep0.iloc[4]["subtask_index"]) == 1  # was: 4 % 3

    def test_apply_subtask_with_new_string_appends_to_lookup_table(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """Brand-new string is appended to ``meta/subtasks.parquet`` with a
        fresh index, then written into the data shards."""
        app, state = app_with_state
        ds = self._ds_with_subtask_lookup(tmp_path, lerobot_dataset_factory)
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        before = pd.read_parquet(ds.root / "meta" / "subtasks.parquet")
        assert "inspect knot" not in before.index

        _post_feature_set(
            app,
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "subtask",
                "frame_from": 0,
                "frame_to": 2,
                "value": "inspect knot",
            },
        )
        assert _post_apply(app, dataset_id).status_code == 200

        after = pd.read_parquet(ds.root / "meta" / "subtasks.parquet")
        assert "inspect knot" in after.index, "new subtask string should be appended to lookup"
        # And its index should be > all the previously-existing ones.
        assert int(after.loc["inspect knot", "subtask_index"]) >= len(before)
