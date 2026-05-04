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

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI

from lerobot.gui.api import datasets as datasets_module, edits as edits_module
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState


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
