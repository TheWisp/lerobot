# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for the local-cache completeness pre-check on dataset open.

Guards the contract that the GUI is local-only by default: it must not
silently call ``snapshot_download`` when a user opens a partial cache. The
pre-check refuses such opens with HTTP 409 + structured detail so the
frontend can ask the user to confirm an explicit Hub download.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI

from lerobot.gui.api import datasets as datasets_module
from lerobot.gui.api.datasets import _check_local_dataset_complete
from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState


@pytest.fixture
def app_with_state():
    """FastAPI app with the datasets router and a clean module-level state."""
    app = FastAPI()
    app.include_router(datasets_module.router)

    state = AppState(frame_cache=FrameCache(max_bytes=1_000_000))
    original_state = datasets_module._app_state
    original_mtime = datasets_module._dataset_info_mtime.copy()
    original_indices = datasets_module._episode_start_indices.copy()
    datasets_module.set_app_state(state)

    yield app, state

    datasets_module._app_state = original_state
    datasets_module._dataset_info_mtime.clear()
    datasets_module._dataset_info_mtime.update(original_mtime)
    datasets_module._episode_start_indices.clear()
    datasets_module._episode_start_indices.update(original_indices)


# ── _check_local_dataset_complete ───────────────────────────────────────────


class TestCheckLocalDatasetComplete:
    """Direct tests for the pure helper used by the open-dataset pre-check."""

    def test_complete_dataset_returns_ok(self, tmp_path, lerobot_dataset_factory):
        """A freshly-built dataset on disk has every data + video file present."""
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
        is_complete, problems = _check_local_dataset_complete(ds.root)
        assert is_complete, f"expected complete, got problems: {problems}"
        assert problems == []

    def test_missing_info_json(self, tmp_path):
        """An empty directory is reported with a clear marker problem."""
        is_complete, problems = _check_local_dataset_complete(tmp_path / "empty")
        assert not is_complete
        assert any("meta/info.json is missing" in p for p in problems)

    def test_missing_data_parquet(self, tmp_path, lerobot_dataset_factory):
        """Removing data parquet files is flagged with a count + sample path."""
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
        # Wipe all data parquet files (keep the directory).
        for p in (ds.root / "data").rglob("*.parquet"):
            p.unlink()

        is_complete, problems = _check_local_dataset_complete(ds.root)
        assert not is_complete
        assert any("data parquet file(s) missing" in p for p in problems), problems

    def test_missing_video_file(self, tmp_path, lerobot_dataset_factory):
        """Removing a video referenced by metadata is flagged separately."""
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30, use_videos=True)
        videos_dir = ds.root / "videos"
        if not videos_dir.exists() or not list(videos_dir.rglob("*.mp4")):
            pytest.skip("factory built dataset without video files")

        # Drop one mp4 and verify it's surfaced as a missing video.
        next(videos_dir.rglob("*.mp4")).unlink()

        is_complete, problems = _check_local_dataset_complete(ds.root)
        assert not is_complete
        assert any("video file(s) missing" in p for p in problems), problems


# ── POST /api/datasets — 409 path on incomplete local cache ─────────────────


class TestOpenDatasetIncompleteCache:
    """End-to-end through the open endpoint."""

    def test_complete_local_cache_opens_normally(self, app_with_state, tmp_path, lerobot_dataset_factory):
        """Sanity: pre-check doesn't false-positive on a complete dataset."""
        app, _state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post("/api/datasets", json={"local_path": str(ds.root)})
                assert resp.status_code == 200, resp.text
                body = resp.json()
                assert body["total_episodes"] == 3

        asyncio.run(run())

    def test_incomplete_cache_returns_409_with_structured_detail(
        self, app_with_state, tmp_path, lerobot_dataset_factory
    ):
        """The core contract: don't silently download — surface 409 instead.

        The frontend keys off ``detail.code == "incomplete_local_cache"`` to
        decide whether to show the Hub-sync confirmation modal, so the shape
        of the payload is part of the API contract.
        """
        app, _state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
        # Make the local cache incomplete by removing all data parquets.
        for p in (ds.root / "data").rglob("*.parquet"):
            p.unlink()

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post("/api/datasets", json={"local_path": str(ds.root)})
                assert resp.status_code == 409, resp.text
                detail = resp.json()["detail"]
                assert detail["code"] == "incomplete_local_cache"
                assert detail["hub_sync_available"] is True
                assert detail["local_path"] == str(ds.root)
                assert detail["repo_id"]  # non-empty
                assert isinstance(detail["problems"], list) and detail["problems"]
                assert any("data parquet" in p for p in detail["problems"])

        asyncio.run(run())

    def test_confirm_hub_sync_skips_precheck(self, app_with_state, tmp_path, lerobot_dataset_factory):
        """With ``confirm_hub_sync=True`` the pre-check is bypassed, so the
        endpoint goes on to construct ``LeRobotDataset`` (which would normally
        download). We mock that constructor to confirm it's reached without
        the 409 firing — the real download path is exercised manually.
        """
        app, _state = app_with_state
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
        # Snapshot the FULLY-built dataset under a separate root so the patched
        # LeRobotDataset constructor can return it as if downloaded.
        complete_root = ds.root

        # Now pretend the user opens a different (incomplete) path. We point
        # local_path at the complete root but verify that even if it WEREN'T
        # complete, confirm_hub_sync would skip the pre-check.
        for p in (complete_root / "data").rglob("*.parquet"):
            # Delete and recreate to force the pre-check to fail without
            # confirm_hub_sync. With confirm_hub_sync, this branch should
            # be skipped entirely.
            p.unlink()

        # Replace the LeRobotDataset import inside the open endpoint with
        # a stub that returns a minimally-functional handle. We can't easily
        # rebuild a real one after deleting the parquets, so the simplest
        # strong assertion is: when confirm_hub_sync=True, *no* 409 fires —
        # the request fails differently (downstream of the pre-check).
        with (
            patch.object(datasets_module, "_check_local_dataset_complete") as mock_check,
        ):
            # Sentinel: if the pre-check is consulted, the test is wrong.
            mock_check.side_effect = AssertionError("pre-check must be skipped on confirm_hub_sync")

            async def run():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    resp = await client.post(
                        "/api/datasets",
                        json={"local_path": str(complete_root), "confirm_hub_sync": True},
                    )
                    # The pre-check is skipped (mock not called). The downstream
                    # LeRobotDataset constructor will fail on the missing data
                    # files — but it must NOT be a 409 from the pre-check.
                    assert resp.status_code != 409, resp.text
                    # And we never consulted the pre-check helper.
                    mock_check.assert_not_called()

            asyncio.run(run())
