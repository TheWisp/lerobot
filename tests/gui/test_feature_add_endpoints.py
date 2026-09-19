# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the in-place schema-add endpoints.

Endpoints under test:
  - POST /api/datasets/{id}/features          (generic)
  - POST /api/datasets/{id}/features/defaults (reward + success banner)

Plus the FeatureSchema declared-per-episode behavior (T11) and the
pending-edits guard (T9).
"""

from __future__ import annotations

import asyncio
import json
import pathlib

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

from lerobot.gui.api import datasets as datasets_module
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState, PendingEdit

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def app_with_state():
    """FastAPI app with the datasets + edits routers and a clean module-level state."""
    from lerobot.gui.api import edits as edits_module

    app = FastAPI()
    # Routers already declare their own prefixes (/api/datasets, /api/edits).
    app.include_router(datasets_module.router)
    app.include_router(edits_module.router)

    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    original_state = datasets_module._app_state
    original_edits_state = edits_module._app_state
    original_indices = datasets_module._episode_start_indices.copy()
    datasets_module.set_app_state(state)
    edits_module._app_state = state

    yield app, state

    datasets_module._app_state = original_state
    edits_module._app_state = original_edits_state
    datasets_module._episode_start_indices.clear()
    datasets_module._episode_start_indices.update(original_indices)
    state.pending_edits.clear()


@pytest.fixture
def opened_dataset(app_with_state, tmp_path, empty_lerobot_dataset_factory):
    """Tiny in-memory dataset registered with AppState."""
    _app, state = app_with_state
    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
    }
    ds = empty_lerobot_dataset_factory(
        root=tmp_path / "ds",
        features=features,
    )
    for _ in range(2):
        for _ in range(4):
            ds.add_frame(
                {
                    "action": np.zeros(2, dtype=np.float32),
                    "observation.state": np.zeros(2, dtype=np.float32),
                    "task": "t",
                }
            )
        ds.save_episode()
    ds.finalize()

    dataset_id = str(ds.root)
    state.datasets[dataset_id] = ds
    return dataset_id, ds


# ── Helpers ──────────────────────────────────────────────────────────


def _post_json(app, path: str, body=None):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(path, json=body)

    return asyncio.run(run())


def _get_json(app, path: str):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.get(path)

    return asyncio.run(run())


# ── POST /api/datasets/{id}/features (T7) ────────────────────────────


class TestPostFeatures:
    def test_unknown_dataset_returns_404(self, app_with_state):
        app, _state = app_with_state
        resp = _post_json(
            app,
            "/api/datasets/no-such-ds/features",
            {
                "name": "x",
                "dtype": "float32",
                "shape": [1],
                "per_episode": False,
                "fill_value": 0.0,
            },
        )
        assert resp.status_code == 404

    def test_adds_per_frame_column(self, app_with_state, opened_dataset):
        app, _state = app_with_state
        dataset_id, _ds = opened_dataset
        resp = _post_json(
            app,
            f"/api/datasets/{dataset_id}/features",
            {
                "name": "custom_metric",
                "dtype": "float32",
                "shape": [1],
                "per_episode": False,
                "fill_value": 0.0,
            },
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["added"] == ["custom_metric"]
        # The new info payload reflects the schema add.
        assert "custom_metric" in payload["info"]["features_schema"]

    def test_rejects_default_feature_name(self, app_with_state, opened_dataset):
        app, _state = app_with_state
        dataset_id, _ds = opened_dataset
        resp = _post_json(
            app,
            f"/api/datasets/{dataset_id}/features",
            {
                "name": "timestamp",
                "dtype": "float32",
                "shape": [1],
                "per_episode": False,
                "fill_value": 0.0,
            },
        )
        assert resp.status_code == 400
        assert "DEFAULT_FEATURE" in resp.json()["detail"]

    def test_rejects_existing_name(self, app_with_state, opened_dataset):
        app, _state = app_with_state
        dataset_id, _ds = opened_dataset
        resp = _post_json(
            app,
            f"/api/datasets/{dataset_id}/features",
            {
                "name": "action",
                "dtype": "float32",
                "shape": [1],
                "per_episode": False,
                "fill_value": 0.0,
            },
        )
        assert resp.status_code == 400
        assert "already exists" in resp.json()["detail"]

    def test_rejects_default_feature_via_dialog_path(self, app_with_state, opened_dataset):
        """`reward` and `success` go through the banner endpoint, not the dialog."""
        app, _state = app_with_state
        dataset_id, _ds = opened_dataset
        for name in ("reward", "success"):
            resp = _post_json(
                app,
                f"/api/datasets/{dataset_id}/features",
                {
                    "name": name,
                    "dtype": "float32",
                    "shape": [1],
                    "per_episode": False,
                    "fill_value": 0.0,
                },
            )
            assert resp.status_code == 400, f"{name}: {resp.text}"
            assert "default feature" in resp.json()["detail"].lower()

    def test_adds_per_episode_bool(self, app_with_state, opened_dataset):
        """per_episode flag round-trips and surfaces in features_schema."""
        app, _state = app_with_state
        dataset_id, _ds = opened_dataset
        resp = _post_json(
            app,
            f"/api/datasets/{dataset_id}/features",
            {
                "name": "pe_flag",
                "dtype": "bool",
                "shape": [1],
                "per_episode": True,
                "fill_value": False,
            },
        )
        assert resp.status_code == 200, resp.text
        info = resp.json()["info"]
        assert info["features_schema"]["pe_flag"]["is_per_episode"] is True


# ── POST /api/datasets/{id}/features/defaults (T8) ───────────────────


class TestPostFeaturesDefaults:
    def test_adds_missing_reward_and_success(self, app_with_state, opened_dataset):
        app, _state = app_with_state
        dataset_id, _ds = opened_dataset
        resp = _post_json(app, f"/api/datasets/{dataset_id}/features/defaults", None)
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert sorted(payload["added"]) == ["reward", "success"]

        schema = payload["info"]["features_schema"]
        assert "reward" in schema and "success" in schema
        assert schema["reward"]["dtype"] == "float32"
        assert schema["success"]["dtype"] == "int8"
        assert schema["success"]["is_per_episode"] is True

    def test_idempotent_when_present(self, app_with_state, opened_dataset):
        app, _state = app_with_state
        dataset_id, _ds = opened_dataset
        # First call adds both.
        _post_json(app, f"/api/datasets/{dataset_id}/features/defaults", None)
        # Second call adds nothing.
        resp2 = _post_json(app, f"/api/datasets/{dataset_id}/features/defaults", None)
        assert resp2.status_code == 200
        assert resp2.json()["added"] == []
        assert resp2.json()["renamed"] == []

    def test_renames_existing_next_reward_to_reward(
        self, app_with_state, tmp_path, empty_lerobot_dataset_factory
    ):
        """When the dataset has next.reward and lacks reward, rename instead of add."""
        app, state = app_with_state
        features = {
            "action": {"dtype": "float32", "shape": (2,), "names": None},
            "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
            # Pre-populate with the Gym-convention reward column.
            "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        }
        ds = empty_lerobot_dataset_factory(root=tmp_path / "ds_with_next", features=features)
        for _ in range(2):
            for _ in range(3):
                ds.add_frame(
                    {
                        "action": np.zeros(2, dtype=np.float32),
                        "observation.state": np.zeros(2, dtype=np.float32),
                        "next.reward": np.array([0.5], dtype=np.float32),
                        "task": "t",
                    }
                )
            ds.save_episode()
        ds.finalize()

        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _post_json(app, f"/api/datasets/{dataset_id}/features/defaults", None)
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        # success was added; reward came from a rename.
        assert "success" in payload["added"]
        assert "reward" not in payload["added"]
        assert payload["renamed"] == ["next.reward→reward"]

        # The renamed column carries the original 0.5 fill, not 0.0.
        import pyarrow.parquet as pq

        for f in (ds.root / "data").rglob("*.parquet"):
            t = pq.read_table(f)
            assert "reward" in t.column_names
            assert "next.reward" not in t.column_names
            assert all(v == 0.5 for v in t.column("reward").to_pylist())

    def test_added_reward_does_not_get_inferred_as_per_episode(self, app_with_state, opened_dataset):
        """After adding reward (declared per_episode=false), staging a range
        edit on it does NOT get coerced to whole-episode by inference.

        Bug repro: the constant 0.0 fill made every episode look uniform,
        so _detect_per_episode_features inferred per_episode=True and the
        staging endpoint silently widened the user's range edit to the
        whole episode. The declared per_episode=false hint must win.
        """
        app, _state = app_with_state
        dataset_id, ds = opened_dataset

        # Add reward via the defaults endpoint.
        _post_json(app, f"/api/datasets/{dataset_id}/features/defaults", None)

        # Stage a range edit on reward — should be accepted as a range edit.
        ep_length = int(ds.meta.episodes[0]["length"])
        # Pick a sub-range strictly inside the episode.
        sub_from, sub_to = 1, max(2, ep_length - 1)
        resp = _post_json(
            app,
            "/api/edits/feature-set",
            {
                "dataset_id": dataset_id,
                "episode_index": 0,
                "feature": "reward",
                "frame_from": sub_from,
                "frame_to": sub_to,
                "value": 0.5,
            },
        )
        assert resp.status_code == 200, resp.text

        # The pending edit should preserve the staged sub-range, not be
        # widened to [0, ep_length).
        async def get_edits():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.get("/api/edits")

        pending = asyncio.run(get_edits()).json()["edits"]
        feature_set_edits = [e for e in pending if e["params"].get("feature") == "reward"]
        assert feature_set_edits, "no pending feature_set edit found for reward"
        e = feature_set_edits[-1]["params"]
        assert (e["frame_from"], e["frame_to"]) == (sub_from, sub_to), (
            f"reward edit was coerced from [{sub_from}, {sub_to}) to "
            f"[{e['frame_from']}, {e['frame_to']}) — declared per_episode=false should prevent this"
        )

    def test_skips_rename_when_dtype_incompatible(
        self, app_with_state, tmp_path, empty_lerobot_dataset_factory
    ):
        """If next.reward has the wrong dtype, fall back to adding a new column."""
        app, state = app_with_state
        features = {
            "action": {"dtype": "float32", "shape": (2,), "names": None},
            "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
            # Wrong dtype: int64 instead of float32 — must not be auto-renamed.
            "next.reward": {"dtype": "int64", "shape": (1,), "names": None},
        }
        ds = empty_lerobot_dataset_factory(root=tmp_path / "ds_wrong_dtype", features=features)
        for _ in range(2):
            for _ in range(3):
                ds.add_frame(
                    {
                        "action": np.zeros(2, dtype=np.float32),
                        "observation.state": np.zeros(2, dtype=np.float32),
                        "next.reward": np.array([1], dtype=np.int64),
                        "task": "t",
                    }
                )
            ds.save_episode()
        ds.finalize()
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        resp = _post_json(app, f"/api/datasets/{dataset_id}/features/defaults", None)
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert sorted(payload["added"]) == ["reward", "success"]
        assert payload["renamed"] == []  # incompatible dtype → no rename


# ── Pending-edits guard (T9) ─────────────────────────────────────────


class TestPendingEditGuard:
    def test_state_helper_filters_correctly(self):
        state = AppState(frame_cache=FrameCache(max_bytes=1_000))
        state.add_edit(
            PendingEdit(
                edit_type="feature_set",
                dataset_id="ds1",
                episode_index=0,
                params={"feature": "reward", "frame_from": 0, "frame_to": 5, "value": 1.0},
            )
        )
        state.add_edit(
            PendingEdit(
                edit_type="trim",
                dataset_id="ds1",
                episode_index=0,
                params={},
            )
        )
        state.add_edit(
            PendingEdit(
                edit_type="feature_set",
                dataset_id="other_ds",
                episode_index=0,
                params={"feature": "x", "frame_from": 0, "frame_to": 1, "value": 0},
            )
        )
        assert len(state.pending_feature_set_edits_for_dataset("ds1")) == 1
        assert state.pending_feature_set_edits_for_dataset("ds_missing") == []

    def test_post_features_blocked_by_pending_feature_edits(self, app_with_state, opened_dataset):
        app, state = app_with_state
        dataset_id, _ds = opened_dataset
        state.add_edit(
            PendingEdit(
                edit_type="feature_set",
                dataset_id=dataset_id,
                episode_index=0,
                params={"feature": "action", "frame_from": 0, "frame_to": 1, "value": [0.0, 0.0]},
            )
        )
        try:
            resp = _post_json(
                app,
                f"/api/datasets/{dataset_id}/features",
                {
                    "name": "x",
                    "dtype": "float32",
                    "shape": [1],
                    "per_episode": False,
                    "fill_value": 0.0,
                },
            )
            assert resp.status_code == 409
            assert "pending" in resp.json()["detail"].lower()
        finally:
            state.pending_edits.clear()

    def test_dataset_open_sweeps_orphan_tmp(self, app_with_state, tmp_path, empty_lerobot_dataset_factory):
        """A stale .tmp left from a crashed save is removed when the dataset is opened."""
        app, _state = app_with_state
        features = {
            "action": {"dtype": "float32", "shape": (2,), "names": None},
            "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
        }
        ds = empty_lerobot_dataset_factory(root=tmp_path / "ds", features=features)
        for _ in range(2):
            for _ in range(3):
                ds.add_frame(
                    {
                        "action": np.zeros(2, dtype=np.float32),
                        "observation.state": np.zeros(2, dtype=np.float32),
                        "task": "t",
                    }
                )
            ds.save_episode()
        ds.finalize()

        # Drop a stale .tmp file in the data dir.
        stale = next((ds.root / "data").rglob("*.parquet")).with_suffix(".parquet.tmp")
        stale.write_text("orphan")
        assert stale.exists()

        # Open via the API.
        async def open_call():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.post(
                    "/api/datasets",
                    json={"local_path": str(ds.root), "confirm_hub_sync": True},
                )

        resp = asyncio.run(open_call())
        assert resp.status_code == 200, resp.text
        assert not stale.exists(), "stale .tmp not cleaned on open"

    def test_delete_feature_drops_column(self, app_with_state, opened_dataset):
        """DELETE /features/{name} drops the column."""

        async def del_call(client, path):
            return await client.delete(path)

        app, _state = app_with_state
        dataset_id, _ds = opened_dataset

        # Add a column we can then drop.
        _post_json(
            app,
            f"/api/datasets/{dataset_id}/features",
            {
                "name": "scratch",
                "dtype": "float32",
                "shape": [1],
                "per_episode": False,
                "fill_value": 0.0,
            },
        )

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.delete(f"/api/datasets/{dataset_id}/features/scratch")

        resp = asyncio.run(run())
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["removed"] == ["scratch"]
        assert "scratch" not in payload["info"]["features_schema"]

    def test_delete_feature_rejects_default(self, app_with_state, opened_dataset):
        app, _state = app_with_state
        dataset_id, _ds = opened_dataset

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.delete(f"/api/datasets/{dataset_id}/features/timestamp")

        resp = asyncio.run(run())
        assert resp.status_code == 400
        assert "DEFAULT_FEATURE" in resp.json()["detail"]

    def test_delete_feature_unknown_returns_404(self, app_with_state, opened_dataset):
        app, _state = app_with_state
        dataset_id, _ds = opened_dataset

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.delete(f"/api/datasets/{dataset_id}/features/nonexistent")

        resp = asyncio.run(run())
        assert resp.status_code == 404

    def test_post_defaults_blocked_by_pending_feature_edits(self, app_with_state, opened_dataset):
        app, state = app_with_state
        dataset_id, _ds = opened_dataset
        state.add_edit(
            PendingEdit(
                edit_type="feature_set",
                dataset_id=dataset_id,
                episode_index=0,
                params={"feature": "action", "frame_from": 0, "frame_to": 1, "value": [0.0, 0.0]},
            )
        )
        try:
            resp = _post_json(app, f"/api/datasets/{dataset_id}/features/defaults", None)
            assert resp.status_code == 409
        finally:
            state.pending_edits.clear()


# ── Open-response shape contract ─────────────────────────────────────
#
# Pins what the GET /api/datasets and POST /api/datasets responses must
# include. The frontend's banner / Inspector / row rendering all depend
# on `features_schema` being populated for each open dataset; if it ever
# starts coming back empty, the banner falsely claims reward/success
# are missing for datasets that already have them (user-reported regression).


class TestOpenResponseSchemaContract:
    def _build_dataset_with(self, root, factory, *, extra_features: dict, frames_per_ep: int = 4):
        """Construct a dataset with action / observation.state plus arbitrary
        extra_features. Returns the LeRobotDataset (write-mode finalized)."""
        features = {
            "action": {"dtype": "float32", "shape": (2,), "names": None},
            "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
            **extra_features,
        }
        ds = factory(root=root, features=features)
        for _ in range(2):
            for _ in range(frames_per_ep):
                frame = {
                    "action": np.zeros(2, dtype=np.float32),
                    "observation.state": np.zeros(2, dtype=np.float32),
                    "task": "t",
                }
                # Fill any extra features with sensible defaults so add_frame
                # accepts them (it requires every declared feature to be present).
                for fname, spec in extra_features.items():
                    dtype = spec["dtype"]
                    shape = spec.get("shape") or (1,)
                    if dtype == "float32":
                        frame[fname] = np.zeros(shape, dtype=np.float32)
                    elif dtype == "int8":
                        frame[fname] = np.zeros(shape, dtype=np.int8)
                    elif dtype == "int64":
                        frame[fname] = np.zeros(shape, dtype=np.int64)
                    elif dtype == "bool":
                        frame[fname] = np.zeros(shape, dtype=bool)
                    else:
                        frame[fname] = np.zeros(shape, dtype=np.float32)
                ds.add_frame(frame)
            ds.save_episode()
        ds.finalize()
        return ds

    def test_open_response_carries_reward_and_success_when_present(
        self, app_with_state, tmp_path, empty_lerobot_dataset_factory
    ):
        """A dataset that already has reward + success on disk must
        report both in features_schema — the banner uses this to decide
        whether to show 'missing default features'. If fs comes back
        empty or missing these keys, the banner false-positives."""
        _app, state = app_with_state
        ds = self._build_dataset_with(
            tmp_path / "ds",
            empty_lerobot_dataset_factory,
            extra_features={
                "reward": {"dtype": "float32", "shape": (1,), "names": None},
                "success": {"dtype": "int8", "shape": (1,), "names": None, "per_episode": True},
            },
        )
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        # Build the same DatasetInfo the API would return (without the HTTP
        # round-trip, since this test is about the response shape).
        info = datasets_module._dataset_info_from(dataset_id, ds)
        fs = info.features_schema

        assert "reward" in fs, f"reward missing from features_schema: {sorted(fs.keys())}"
        assert "success" in fs, f"success missing from features_schema: {sorted(fs.keys())}"
        assert fs["reward"].dtype == "float32"
        assert fs["success"].dtype == "int8"
        # success was declared per_episode=True in info.json — must round-trip.
        assert fs["success"].is_per_episode is True

    def test_list_datasets_endpoint_includes_features_schema(
        self, app_with_state, tmp_path, empty_lerobot_dataset_factory
    ):
        """GET /api/datasets returns the full DatasetInfo for each open
        dataset, including features_schema. The frontend's
        restoreOpenedDatasets / page-reload path calls this to rebuild
        window.datasets — if features_schema is missing here, the banner
        false-positives."""
        app, state = app_with_state
        ds = self._build_dataset_with(
            tmp_path / "ds",
            empty_lerobot_dataset_factory,
            extra_features={
                "reward": {"dtype": "float32", "shape": (1,), "names": None},
                "success": {"dtype": "int8", "shape": (1,), "names": None, "per_episode": True},
            },
        )
        dataset_id = str(ds.root)
        state.datasets[dataset_id] = ds

        async def get_list():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.get("/api/datasets")

        resp = asyncio.run(get_list())
        assert resp.status_code == 200
        items = resp.json()
        match = next((d for d in items if d["id"] == dataset_id), None)
        assert match is not None, f"dataset {dataset_id} missing from list response"
        assert "features_schema" in match, "list response missing features_schema field"
        fs = match["features_schema"]
        assert "reward" in fs, f"reward missing from list features_schema: {sorted(fs.keys())}"
        assert "success" in fs, f"success missing from list features_schema: {sorted(fs.keys())}"

    def test_features_legacy_list_and_features_schema_agree(
        self, app_with_state, tmp_path, empty_lerobot_dataset_factory
    ):
        """The legacy `features: list[str]` field and `features_schema:
        dict[str, FeatureSchema]` must contain the same set of feature
        names (modulo the subtask synthesis). If they ever drift, the
        frontend has two sources of truth that disagree."""
        _app, state = app_with_state
        ds = self._build_dataset_with(
            tmp_path / "ds",
            empty_lerobot_dataset_factory,
            extra_features={
                "reward": {"dtype": "float32", "shape": (1,), "names": None},
                "success": {"dtype": "int8", "shape": (1,), "names": None, "per_episode": True},
                "custom_score": {"dtype": "float32", "shape": (1,), "names": None},
            },
        )
        state.datasets[str(ds.root)] = ds

        info = datasets_module._dataset_info_from(str(ds.root), ds)
        legacy = set(info.features)
        schema = set(info.features_schema.keys())
        assert legacy == schema, (
            f"features list and features_schema disagree:\n"
            f"  only in features:        {sorted(legacy - schema)}\n"
            f"  only in features_schema: {sorted(schema - legacy)}"
        )


# ── Flags (bitset) columns ──────────────────────────────────────────


class TestPostFlagsColumn:
    """A ``flags`` list makes the column a bitset, and the contract -- not the
    dialog -- decides how it is stored."""

    def test_a_flags_column_is_created_with_its_vocabulary(self, app_with_state, opened_dataset):
        app, _state = app_with_state
        dataset_id, ds = opened_dataset
        resp = _post_json(
            app,
            f"/api/datasets/{dataset_id}/features",
            {"name": "quality", "dtype": "int64", "flags": ["blurry", "fumble"]},
        )
        assert resp.status_code == 200, resp.text
        spec = ds.meta.features["quality"]
        assert spec["flags"] == ["blurry", "fumble"]
        assert spec["dtype"] == "int64"
        # In-memory metadata carries the shape as a tuple; info.json is the
        # persisted contract, so check the bytes as well as the object.
        assert tuple(spec["shape"]) == (1,)
        persisted = json.loads((ds.root / "meta" / "info.json").read_text())["features"]["quality"]
        assert persisted == {
            "dtype": "int64",
            "shape": [1],
            "names": None,
            "per_episode": False,
            "flags": ["blurry", "fumble"],
        }

    def test_storage_fields_are_ignored_rather_than_trusted(self, app_with_state, opened_dataset):
        """The operator picked a kind of column, not a dtype. A client sending
        nonsense alongside the flags must not be able to produce a column the
        contract would reject."""
        app, _state = app_with_state
        dataset_id, ds = opened_dataset
        resp = _post_json(
            app,
            f"/api/datasets/{dataset_id}/features",
            {
                "name": "quality",
                "dtype": "float32",
                "shape": [7],
                "fill_value": 3.5,
                "flags": ["blurry"],
            },
        )
        assert resp.status_code == 200, resp.text
        spec = ds.meta.features["quality"]
        assert (spec["dtype"], tuple(spec["shape"])) == ("int64", (1,))
        assert all(int(ds[i]["quality"]) == 0 for i in range(len(ds))), "fill_value ignored too"

    def test_a_new_flags_column_starts_with_nothing_set(self, app_with_state, opened_dataset):
        app, _state = app_with_state
        dataset_id, ds = opened_dataset
        _post_json(
            app,
            f"/api/datasets/{dataset_id}/features",
            {"name": "quality", "dtype": "int64", "flags": ["blurry"]},
        )
        assert all(int(ds[i]["quality"]) == 0 for i in range(len(ds)))

    @pytest.mark.parametrize(
        ("flags", "expected"),
        [
            (["a", "a"], "repeats"),
            (["a", ""], "not a non-empty string"),
            ([f"f{i}" for i in range(64)], "holds 63"),
        ],
    )
    def test_an_unusable_vocabulary_is_refused(self, app_with_state, opened_dataset, flags, expected):
        """Refused against the vocabulary the operator just typed, rather than
        at the first write."""
        app, _state = app_with_state
        dataset_id, ds = opened_dataset
        resp = _post_json(
            app,
            f"/api/datasets/{dataset_id}/features",
            {"name": "quality", "dtype": "int64", "flags": flags},
        )
        assert resp.status_code == 400
        assert expected in resp.json()["detail"]
        assert "quality" not in ds.meta.features, "a refused column must not be created"

    def test_an_empty_flags_list_still_makes_an_ordinary_column(self, app_with_state, opened_dataset):
        """``flags: []`` is absence, not an empty bitset -- the dialog sends the
        key only for the flags kind, and JSON round trips can flatten None."""
        app, _state = app_with_state
        dataset_id, ds = opened_dataset
        resp = _post_json(
            app,
            f"/api/datasets/{dataset_id}/features",
            {"name": "score", "dtype": "float32", "flags": []},
        )
        assert resp.status_code == 200, resp.text
        assert ds.meta.features["score"]["dtype"] == "float32"
        assert "flags" not in ds.meta.features["score"]


# ── Growing and renaming a vocabulary ────────────────────────────────


def _patch_json(app, path: str, body=None):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.patch(path, json=body)

    return asyncio.run(run())


@pytest.fixture
def flagged_dataset(app_with_state, opened_dataset):
    """An opened dataset carrying a two-flag bitset, with frame 1 flagged
    ``fumble`` (bit 1) so value-preservation is observable."""
    app, _state = app_with_state
    dataset_id, ds = opened_dataset
    _post_json(
        app,
        f"/api/datasets/{dataset_id}/features",
        {"name": "quality", "dtype": "int64", "flags": ["blurry", "fumble"]},
    )
    return app, dataset_id, ds


class TestGrowVocabulary:
    def test_appending_takes_the_next_bit(self, flagged_dataset):
        app, dataset_id, ds = flagged_dataset
        resp = _post_json(app, f"/api/datasets/{dataset_id}/features/quality/flags", {"flag": "occluded"})
        assert resp.status_code == 200, resp.text
        assert ds.meta.features["quality"]["flags"] == ["blurry", "fumble", "occluded"]

    def test_appending_does_not_rewrite_the_data(self, flagged_dataset):
        """The property that makes this cheap: only info.json changes, so no
        stored value can be disturbed."""
        app, dataset_id, ds = flagged_dataset
        shards = sorted((ds.root / "data").rglob("*.parquet"))
        before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in shards}
        assert before, "expected at least one data shard"

        _post_json(app, f"/api/datasets/{dataset_id}/features/quality/flags", {"flag": "occluded"})

        for path, (mtime, payload) in before.items():
            assert path.stat().st_mtime_ns == mtime, f"{path.name} was rewritten"
            assert path.read_bytes() == payload

    def test_an_existing_flag_is_refused(self, flagged_dataset):
        app, dataset_id, ds = flagged_dataset
        resp = _post_json(app, f"/api/datasets/{dataset_id}/features/quality/flags", {"flag": "blurry"})
        assert resp.status_code == 400
        assert ds.meta.features["quality"]["flags"] == ["blurry", "fumble"]

    def test_growing_past_the_limit_is_refused(self, flagged_dataset):
        app, dataset_id, ds = flagged_dataset
        for i in range(61):  # 2 declared + 61 = 63, the limit
            resp = _post_json(app, f"/api/datasets/{dataset_id}/features/quality/flags", {"flag": f"x{i}"})
            assert resp.status_code == 200, resp.text
        assert len(ds.meta.features["quality"]["flags"]) == 63

        resp = _post_json(app, f"/api/datasets/{dataset_id}/features/quality/flags", {"flag": "one-too-many"})
        assert resp.status_code == 400
        assert "holds 63" in resp.json()["detail"]

    def test_a_column_that_is_not_a_bitset_is_refused(self, app_with_state, opened_dataset):
        app, _state = app_with_state
        dataset_id, _ds = opened_dataset
        resp = _post_json(app, f"/api/datasets/{dataset_id}/features/action/flags", {"flag": "blurry"})
        assert resp.status_code == 400
        assert "not a flags column" in resp.json()["detail"]


class TestRenameFlag:
    def test_renaming_keeps_the_bit(self, flagged_dataset):
        """Renames what the flag is called, not what any frame means."""
        app, dataset_id, ds = flagged_dataset
        resp = _patch_json(app, f"/api/datasets/{dataset_id}/features/quality/flags/1", {"flag": "mistimed"})
        assert resp.status_code == 200, resp.text
        assert ds.meta.features["quality"]["flags"] == ["blurry", "mistimed"]

    def test_renaming_does_not_rewrite_the_data(self, flagged_dataset):
        app, dataset_id, ds = flagged_dataset
        shards = sorted((ds.root / "data").rglob("*.parquet"))
        before = {p: p.read_bytes() for p in shards}
        _patch_json(app, f"/api/datasets/{dataset_id}/features/quality/flags/0", {"flag": "soft"})
        for path, payload in before.items():
            assert path.read_bytes() == payload

    def test_a_bit_outside_the_vocabulary_is_refused(self, flagged_dataset):
        app, dataset_id, _ds = flagged_dataset
        resp = _patch_json(app, f"/api/datasets/{dataset_id}/features/quality/flags/7", {"flag": "nope"})
        assert resp.status_code == 404
        assert "no bit 7" in resp.json()["detail"]

    def test_renaming_onto_another_flag_is_refused(self, flagged_dataset):
        """Two bits with one name cannot be told apart, and the round trip
        would stop being reversible."""
        app, dataset_id, ds = flagged_dataset
        resp = _patch_json(app, f"/api/datasets/{dataset_id}/features/quality/flags/1", {"flag": "blurry"})
        assert resp.status_code == 400
        assert ds.meta.features["quality"]["flags"] == ["blurry", "fumble"]

    def test_renaming_a_flag_to_itself_is_allowed(self, flagged_dataset):
        """A no-op edit should not be reported as a collision with itself."""
        app, dataset_id, ds = flagged_dataset
        resp = _patch_json(app, f"/api/datasets/{dataset_id}/features/quality/flags/0", {"flag": "blurry"})
        assert resp.status_code == 200, resp.text
        assert ds.meta.features["quality"]["flags"] == ["blurry", "fumble"]

    def test_an_unusable_new_name_is_refused(self, flagged_dataset):
        app, dataset_id, ds = flagged_dataset
        resp = _patch_json(app, f"/api/datasets/{dataset_id}/features/quality/flags/0", {"flag": "   "})
        assert resp.status_code == 400
        assert ds.meta.features["quality"]["flags"] == ["blurry", "fumble"]


class TestVocabularyPersistence:
    def test_a_grown_vocabulary_survives_a_reopen(self, flagged_dataset):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        app, dataset_id, ds = flagged_dataset
        _post_json(app, f"/api/datasets/{dataset_id}/features/quality/flags", {"flag": "occluded"})
        reopened = LeRobotDataset(repo_id=ds.repo_id, root=ds.root)
        assert reopened.meta.features["quality"]["flags"] == ["blurry", "fumble", "occluded"]

    def test_a_rename_survives_a_reopen(self, flagged_dataset):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        app, dataset_id, ds = flagged_dataset
        _patch_json(app, f"/api/datasets/{dataset_id}/features/quality/flags/1", {"flag": "late"})
        reopened = LeRobotDataset(repo_id=ds.repo_id, root=ds.root)
        assert reopened.meta.features["quality"]["flags"] == ["blurry", "late"]


def test_the_flag_vocabulary_reaches_the_frontend_schema(flagged_dataset):
    """``features_schema`` is what the browser renders from. Without ``flags``
    there, a flags column renders as a slider over the raw integer -- which is
    how this was found, and no backend test would have caught it."""
    app, dataset_id, ds = flagged_dataset
    # Built directly rather than through an endpoint: the response model is the
    # thing under test, and reaching it does not require a mounted route.
    schema = {
        name: spec.model_dump()
        for name, spec in datasets_module._build_features_schema(ds.meta.features).items()
    }
    assert schema["quality"]["flags"] == ["blurry", "fumble"]
    assert schema["action"]["flags"] is None, "a non-bitset column must not claim a vocabulary"


def test_the_rename_endpoint_is_reachable_from_the_frontend():
    """An endpoint with no caller is dead code in a PR. The append control was
    added and the rename one was not, so this pins that both exist.

    Asserted on the URL, the verb and the control rather than on one literal:
    the two vocabulary edits share a submit helper, so the verb is an argument
    now and the shape of the call is free to change again.
    """
    static = pathlib.Path(datasets_module.__file__).resolve().parents[2] / "gui" / "static"
    js = (static / "feature_editing.js").read_text()
    assert "/flags/${bit}" in js, "no frontend call reaches the rename endpoint"
    assert '"PATCH"' in js, "nothing sends the verb the rename endpoint answers"
    assert "flag-rename" in js, "no control invokes the rename"
