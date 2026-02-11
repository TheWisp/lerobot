#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for virtual (non-destructive) dataset editing operations.

Virtual edits modify metadata and parquet data but do NOT re-encode video files.
This preserves video quality and is much faster than re-encoding.
"""

import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import load_episodes, load_info


@pytest.fixture
def video_dataset(tmp_path):
    """Create a copy of pusht dataset for testing with only 3 episodes."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    source_dataset = LeRobotDataset("lerobot/pusht", episodes=[0, 1, 2])

    test_root = tmp_path / "pusht_test"
    shutil.copytree(source_dataset.root, test_root)

    # Filter episodes parquet to only include the 3 episodes we loaded
    episodes_parquet_path = test_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if episodes_parquet_path.exists():
        table = pq.read_table(episodes_parquet_path)
        df = table.to_pandas()
        # Keep only first 3 episodes
        df_filtered = df[df["episode_index"].isin([0, 1, 2])].copy()
        pq.write_table(pa.Table.from_pandas(df_filtered, preserve_index=False), episodes_parquet_path)

    # Update info.json with correct totals
    info_path = test_root / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        info["total_episodes"] = 3
        # Calculate total frames from the 3 episodes
        total_frames = sum(ep["length"] for ep in source_dataset.meta.episodes)
        info["total_frames"] = total_frames
        info["splits"] = {"train": "0:3"}
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(test_root)

        dataset = LeRobotDataset("lerobot/pusht", root=test_root)

    # Ensure episodes are loaded
    if dataset.meta.episodes is None:
        dataset.meta.episodes = load_episodes(dataset.root)

    return dataset


class TestVirtualTrim:
    """Tests for virtual trim operation (no video re-encoding)."""

    def test_trim_updates_metadata(self, video_dataset):
        """Verify trim updates episode metadata correctly."""
        from lerobot.datasets.dataset_tools import trim_episode_virtual

        original_length = video_dataset.meta.episodes[0]["length"]
        original_total_frames = video_dataset.meta.total_frames

        # Trim 10 frames from start, 5 from end
        start_frame = 10
        end_frame = original_length - 5

        trim_episode_virtual(
            video_dataset,
            episode_index=0,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        # Reload metadata
        video_dataset.meta.episodes = load_episodes(video_dataset.root)
        video_dataset.meta.info = load_info(video_dataset.root)

        new_length = video_dataset.meta.episodes[0]["length"]
        expected_length = end_frame - start_frame

        assert new_length == expected_length
        assert video_dataset.meta.total_frames == original_total_frames - (original_length - expected_length)

    def test_trim_updates_video_timestamps(self, video_dataset):
        """Verify trim updates video from/to timestamps correctly."""
        from lerobot.datasets.dataset_tools import trim_episode_virtual

        video_key = video_dataset.meta.video_keys[0]
        original_from_ts = video_dataset.meta.episodes[0][f"videos/{video_key}/from_timestamp"]
        original_to_ts = video_dataset.meta.episodes[0][f"videos/{video_key}/to_timestamp"]
        original_length = video_dataset.meta.episodes[0]["length"]

        # Trim 10 frames from start
        start_frame = 10
        end_frame = original_length
        trim_start_s = start_frame / video_dataset.fps

        trim_episode_virtual(
            video_dataset,
            episode_index=0,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        video_dataset.meta.episodes = load_episodes(video_dataset.root)

        new_from_ts = video_dataset.meta.episodes[0][f"videos/{video_key}/from_timestamp"]
        new_to_ts = video_dataset.meta.episodes[0][f"videos/{video_key}/to_timestamp"]

        # from_timestamp should increase by trim amount
        assert abs(new_from_ts - (original_from_ts + trim_start_s)) < 0.001
        # to_timestamp should stay the same (we only trimmed start)
        assert abs(new_to_ts - original_to_ts) < 0.001

    def test_trim_does_not_affect_other_episodes_timestamps(self, video_dataset):
        """Verify trimming episode 0 does NOT change other episodes' timestamps."""
        from lerobot.datasets.dataset_tools import trim_episode_virtual

        video_key = video_dataset.meta.video_keys[0]

        # Save original timestamps for episodes 1 and 2
        ep1_from = video_dataset.meta.episodes[1][f"videos/{video_key}/from_timestamp"]
        ep1_to = video_dataset.meta.episodes[1][f"videos/{video_key}/to_timestamp"]
        ep2_from = video_dataset.meta.episodes[2][f"videos/{video_key}/from_timestamp"]
        ep2_to = video_dataset.meta.episodes[2][f"videos/{video_key}/to_timestamp"]

        original_length = video_dataset.meta.episodes[0]["length"]
        trim_episode_virtual(video_dataset, episode_index=0, start_frame=10, end_frame=original_length)

        video_dataset.meta.episodes = load_episodes(video_dataset.root)

        # Other episodes' timestamps should be UNCHANGED
        assert video_dataset.meta.episodes[1][f"videos/{video_key}/from_timestamp"] == ep1_from
        assert video_dataset.meta.episodes[1][f"videos/{video_key}/to_timestamp"] == ep1_to
        assert video_dataset.meta.episodes[2][f"videos/{video_key}/from_timestamp"] == ep2_from
        assert video_dataset.meta.episodes[2][f"videos/{video_key}/to_timestamp"] == ep2_to

    def test_trim_does_not_modify_video_file(self, video_dataset):
        """Verify virtual trim does NOT modify video files."""
        from lerobot.datasets.dataset_tools import trim_episode_virtual

        video_key = video_dataset.meta.video_keys[0]
        video_path = video_dataset.root / video_dataset.meta.get_video_file_path(0, video_key)

        # Get original video file stats
        original_size = video_path.stat().st_size
        original_mtime = video_path.stat().st_mtime

        original_length = video_dataset.meta.episodes[0]["length"]
        trim_episode_virtual(
            video_dataset,
            episode_index=0,
            start_frame=10,
            end_frame=original_length - 10,
        )

        # Video file should be UNCHANGED
        assert video_path.stat().st_size == original_size
        assert video_path.stat().st_mtime == original_mtime

    def test_trim_updates_parquet_data(self, video_dataset):
        """Verify trim removes correct rows from parquet data."""
        from lerobot.datasets.dataset_tools import trim_episode_virtual

        original_length = video_dataset.meta.episodes[0]["length"]
        original_total_rows = len(video_dataset.hf_dataset)

        start_frame = 5
        end_frame = original_length - 5
        frames_removed = original_length - (end_frame - start_frame)

        trim_episode_virtual(
            video_dataset,
            episode_index=0,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        # Reload dataset
        video_dataset.hf_dataset = video_dataset.load_hf_dataset()

        assert len(video_dataset.hf_dataset) == original_total_rows - frames_removed

    def test_trim_shifts_subsequent_episodes(self, video_dataset):
        """Verify trimming episode 0 shifts dataset indices of subsequent episodes."""
        from lerobot.datasets.dataset_tools import trim_episode_virtual

        original_ep1_from = video_dataset.meta.episodes[1]["dataset_from_index"]
        original_length = video_dataset.meta.episodes[0]["length"]

        start_frame = 10
        end_frame = original_length - 10
        frames_removed = original_length - (end_frame - start_frame)

        trim_episode_virtual(
            video_dataset,
            episode_index=0,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        video_dataset.meta.episodes = load_episodes(video_dataset.root)

        new_ep1_from = video_dataset.meta.episodes[1]["dataset_from_index"]
        assert new_ep1_from == original_ep1_from - frames_removed


class TestVirtualDelete:
    """Tests for virtual delete operation (no video re-encoding)."""

    def test_delete_removes_episode_metadata(self, video_dataset):
        """Verify delete removes episode from metadata."""
        from lerobot.datasets.dataset_tools import delete_episodes_virtual

        original_total_episodes = video_dataset.meta.total_episodes

        delete_episodes_virtual(video_dataset, episode_indices=[1])

        video_dataset.meta.episodes = load_episodes(video_dataset.root)
        video_dataset.meta.info = load_info(video_dataset.root)

        assert video_dataset.meta.total_episodes == original_total_episodes - 1
        assert len(video_dataset.meta.episodes) == original_total_episodes - 1

    def test_delete_renumbers_episodes(self, video_dataset):
        """Verify remaining episodes are renumbered correctly."""
        from lerobot.datasets.dataset_tools import delete_episodes_virtual

        # Delete episode 1 (middle episode)
        delete_episodes_virtual(video_dataset, episode_indices=[1])

        video_dataset.meta.episodes = load_episodes(video_dataset.root)

        # Episode indices should be 0, 1 (was 0, 2)
        episode_indices = [ep["episode_index"] for ep in video_dataset.meta.episodes]
        assert episode_indices == [0, 1]

    def test_delete_does_not_modify_video_files(self, video_dataset):
        """Verify virtual delete does NOT modify video files."""
        from lerobot.datasets.dataset_tools import delete_episodes_virtual

        video_key = video_dataset.meta.video_keys[0]

        # Get all video file stats before delete
        video_files = list((video_dataset.root / "videos" / video_key).rglob("*.mp4"))
        original_stats = {str(f): (f.stat().st_size, f.stat().st_mtime) for f in video_files}

        delete_episodes_virtual(video_dataset, episode_indices=[1])

        # All video files should be UNCHANGED
        for f in video_files:
            size, mtime = original_stats[str(f)]
            assert f.stat().st_size == size
            assert f.stat().st_mtime == mtime

    def test_delete_updates_parquet_data(self, video_dataset):
        """Verify delete removes correct rows from parquet data."""
        from lerobot.datasets.dataset_tools import delete_episodes_virtual

        ep1_length = video_dataset.meta.episodes[1]["length"]
        # Calculate original row count from episode metadata, not hf_dataset
        # (since the data parquet may contain more rows than selected episodes)
        original_total_rows = sum(ep["length"] for ep in video_dataset.meta.episodes)

        delete_episodes_virtual(video_dataset, episode_indices=[1])

        video_dataset.hf_dataset = video_dataset.load_hf_dataset()

        assert len(video_dataset.hf_dataset) == original_total_rows - ep1_length

    def test_delete_multiple_episodes(self, video_dataset):
        """Verify deleting multiple episodes works correctly."""
        from lerobot.datasets.dataset_tools import delete_episodes_virtual

        original_total_episodes = video_dataset.meta.total_episodes
        ep0_length = video_dataset.meta.episodes[0]["length"]
        ep2_length = video_dataset.meta.episodes[2]["length"]
        original_total_frames = video_dataset.meta.total_frames

        delete_episodes_virtual(video_dataset, episode_indices=[0, 2])

        video_dataset.meta.episodes = load_episodes(video_dataset.root)
        video_dataset.meta.info = load_info(video_dataset.root)

        assert video_dataset.meta.total_episodes == original_total_episodes - 2
        assert video_dataset.meta.total_frames == original_total_frames - ep0_length - ep2_length

    def test_delete_updates_dataset_indices(self, video_dataset):
        """Verify dataset_from_index and dataset_to_index are updated correctly."""
        from lerobot.datasets.dataset_tools import delete_episodes_virtual

        # Delete episode 0, episode 1 (was index 1) should become index 0
        # and its dataset_from_index should start at 0
        delete_episodes_virtual(video_dataset, episode_indices=[0])

        video_dataset.meta.episodes = load_episodes(video_dataset.root)

        # First episode should now start at index 0
        assert video_dataset.meta.episodes[0]["dataset_from_index"] == 0


class TestVirtualEditIntegration:
    """Integration tests for virtual edits."""

    def test_trim_then_delete(self, video_dataset):
        """Verify trim followed by delete works correctly."""
        from lerobot.datasets.dataset_tools import delete_episodes_virtual, trim_episode_virtual

        original_total_frames = video_dataset.meta.total_frames
        ep0_length = video_dataset.meta.episodes[0]["length"]
        ep1_length = video_dataset.meta.episodes[1]["length"]

        # Trim episode 0
        trim_episode_virtual(
            video_dataset,
            episode_index=0,
            start_frame=5,
            end_frame=ep0_length - 5,
        )

        # Reload
        video_dataset.meta.episodes = load_episodes(video_dataset.root)

        # Delete episode 1
        delete_episodes_virtual(video_dataset, episode_indices=[1])

        video_dataset.meta.episodes = load_episodes(video_dataset.root)
        video_dataset.meta.info = load_info(video_dataset.root)

        # Should have 2 episodes left
        assert video_dataset.meta.total_episodes == 2

        # Total frames = original - 10 (trim) - ep1_length (delete)
        expected_frames = original_total_frames - 10 - ep1_length
        assert video_dataset.meta.total_frames == expected_frames

    def test_dataset_still_loadable_after_edits(self, video_dataset):
        """Verify dataset can be reloaded after virtual edits."""
        from lerobot.datasets.dataset_tools import delete_episodes_virtual, trim_episode_virtual

        ep0_length = video_dataset.meta.episodes[0]["length"]

        trim_episode_virtual(
            video_dataset,
            episode_index=0,
            start_frame=5,
            end_frame=ep0_length - 5,
        )
        delete_episodes_virtual(video_dataset, episode_indices=[1])

        # Try to reload the dataset
        with (
            patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
            patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
        ):
            mock_get_safe_version.return_value = "v3.0"
            mock_snapshot_download.return_value = str(video_dataset.root)

            reloaded = LeRobotDataset("lerobot/pusht", root=video_dataset.root)

        assert reloaded.meta.total_episodes == 2
        assert len(reloaded) == reloaded.meta.total_frames
