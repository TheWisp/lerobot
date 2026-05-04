# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for ``dataset_tools.set_feature_values`` — the in-place
per-frame value editing primitive.

This is the LeRobot-level peer of :func:`modify_features`. The GUI's
"apply edits" path translates staged edits into a single call here;
notebooks / CLI tools can use it directly. These tests pin the
contract: half-open ``[from, to)`` ranges, atomic shard rewrites,
schema unchanged, episode stats recomputed, image/video features
rejected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lerobot.datasets.dataset_tools import (
    FeatureValueEdit,
    set_feature_values,
)


def _read_data_df(root: Path) -> pd.DataFrame:
    """Load all data parquet shards merged into a single DataFrame, sorted by global ``index``."""
    parts = []
    for shard in sorted((root / "data").glob("*/*.parquet")):
        parts.append(pd.read_parquet(shard))
    df = pd.concat(parts, ignore_index=True).sort_values("index").reset_index(drop=True)
    return df


def _read_episode_stats_min_max(root: Path, feature: str) -> dict[int, tuple[float, float]]:
    """Return ``{episode_index: (min, max)}`` for ``feature`` from the episodes parquet."""
    out: dict[int, tuple[float, float]] = {}
    for shard in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
        df = pd.read_parquet(shard)
        for _, row in df.iterrows():
            ep = int(row["episode_index"])
            mn = row.get(f"stats/{feature}/min")
            mx = row.get(f"stats/{feature}/max")
            mn = float(mn[0]) if hasattr(mn, "__len__") else float(mn)
            mx = float(mx[0]) if hasattr(mx, "__len__") else float(mx)
            out[ep] = (mn, mx)
    return out


# ── Validation ──────────────────────────────────────────────────────────────


class TestValidation:
    def test_unknown_feature_raises(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        with pytest.raises(ValueError, match="Unknown feature"):
            set_feature_values(ds, edits=[{"feature": "no_such", "from_index": 0, "to_index": 1, "value": 0}])

    def test_image_feature_rejected(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        img = next((n for n, ft in ds.meta.features.items() if ft["dtype"] in ("image", "video")), None)
        if img is None:
            pytest.skip("factory built dataset without image/video features")
        with pytest.raises(ValueError, match="not editable"):
            set_feature_values(ds, edits=[{"feature": img, "from_index": 0, "to_index": 1, "value": None}])

    def test_out_of_range_rejected(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        # action is reliably present.
        with pytest.raises(ValueError, match="Invalid range"):
            set_feature_values(
                ds,
                edits=[{"feature": "action", "from_index": -1, "to_index": 5, "value": 0.0}],
            )

    def test_inverted_range_rejected(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        with pytest.raises(ValueError, match="Invalid range"):
            set_feature_values(
                ds,
                edits=[{"feature": "action", "from_index": 10, "to_index": 5, "value": 0.0}],
            )

    def test_empty_edit_list_is_noop(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        # Should not raise; just returns.
        set_feature_values(ds, edits=[])

    def test_in_place_with_output_dir_rejected(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        with pytest.raises(ValueError, match="output_dir is not used"):
            set_feature_values(
                ds,
                edits=[{"feature": "action", "from_index": 0, "to_index": 1, "value": 0.0}],
                in_place=True,
                output_dir=tmp_path / "fork",
            )

    def test_fork_without_output_dir_rejected(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        with pytest.raises(ValueError, match="output_dir is required"):
            set_feature_values(
                ds,
                edits=[{"feature": "action", "from_index": 0, "to_index": 1, "value": 0.0}],
                in_place=False,
            )


# ── Core overwrite semantics ────────────────────────────────────────────────


class TestOverwrite:
    def test_vector_feature_overwrite_writes_to_target_range_only(self, tmp_path, lerobot_dataset_factory):
        """Action (vector) overwrite — confirm only the listed rows change."""
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
        action_feature = ds.meta.features["action"]
        action_dim = action_feature["shape"][0]

        before = _read_data_df(ds.root).copy()
        new_value = [0.42] * action_dim

        set_feature_values(
            ds,
            edits=[FeatureValueEdit(feature="action", from_index=5, to_index=12, value=new_value)],
        )

        after = _read_data_df(ds.root)

        # In range: every cell equals the new value (per-component).
        in_range = after[(after["index"] >= 5) & (after["index"] < 12)]
        for _, row in in_range.iterrows():
            cell = np.asarray(row["action"])
            assert cell.shape == (action_dim,)
            assert np.allclose(cell, 0.42)

        # Out of range: completely unchanged from the before snapshot.
        for idx in range(0, 5):
            row_before = before[before["index"] == idx].iloc[0]
            row_after = after[after["index"] == idx].iloc[0]
            assert np.allclose(np.asarray(row_after["action"]), np.asarray(row_before["action"]))
        for idx in range(12, len(before)):
            row_before = before[before["index"] == idx].iloc[0]
            row_after = after[after["index"] == idx].iloc[0]
            assert np.allclose(np.asarray(row_after["action"]), np.asarray(row_before["action"]))

    def test_half_open_boundary_excludes_to_index(self, tmp_path, lerobot_dataset_factory):
        """``[5, 10)`` writes frames 5..9, leaves frame 10 untouched."""
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        action_dim = ds.meta.features["action"]["shape"][0]

        before = _read_data_df(ds.root)
        marker = [-99.0] * action_dim
        original_at_10 = np.asarray(before[before["index"] == 10].iloc[0]["action"])

        set_feature_values(
            ds, edits=[{"feature": "action", "from_index": 5, "to_index": 10, "value": marker}]
        )

        after = _read_data_df(ds.root)

        # Frame 9 IS changed (last frame in half-open range).
        cell9 = np.asarray(after[after["index"] == 9].iloc[0]["action"])
        assert np.allclose(cell9, -99.0)
        # Frame 10 is NOT changed.
        cell10 = np.asarray(after[after["index"] == 10].iloc[0]["action"])
        assert np.allclose(cell10, original_at_10)

    def test_single_frame_edit(self, tmp_path, lerobot_dataset_factory):
        """``[N, N+1)`` is the canonical single-frame edit."""
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        action_dim = ds.meta.features["action"]["shape"][0]

        marker = [7.0] * action_dim
        set_feature_values(ds, edits=[{"feature": "action", "from_index": 7, "to_index": 8, "value": marker}])

        after = _read_data_df(ds.root)
        cell7 = np.asarray(after[after["index"] == 7].iloc[0]["action"])
        cell6 = np.asarray(after[after["index"] == 6].iloc[0]["action"])
        cell8 = np.asarray(after[after["index"] == 8].iloc[0]["action"])
        assert np.allclose(cell7, 7.0)
        assert not np.allclose(cell6, 7.0)
        assert not np.allclose(cell8, 7.0)

    def test_multiple_features_in_one_call(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        action_dim = ds.meta.features["action"]["shape"][0]

        # Pick a non-image, non-action numeric feature to also edit.
        secondary = next(
            (
                n
                for n, ft in ds.meta.features.items()
                if n != "action"
                and ft["dtype"] not in ("image", "video", "string")
                and not n.startswith("observation.images")
            ),
            None,
        )
        if secondary is None:
            pytest.skip("no secondary editable feature in factory dataset")

        sec_shape = ds.meta.features[secondary]["shape"]
        sec_value = (
            0.5
            if (not sec_shape or sec_shape == [1])
            else [0.5] * (sec_shape[0] if isinstance(sec_shape[0], int) else 1)
        )

        set_feature_values(
            ds,
            edits=[
                {"feature": "action", "from_index": 3, "to_index": 6, "value": [1.5] * action_dim},
                {"feature": secondary, "from_index": 10, "to_index": 12, "value": sec_value},
            ],
        )

        after = _read_data_df(ds.root)
        # Both ranges took effect.
        for idx in range(3, 6):
            assert np.allclose(np.asarray(after[after["index"] == idx].iloc[0]["action"]), 1.5)


# ── Schema / videos must NOT change ──────────────────────────────────────────


class TestSchemaUnchanged:
    def test_info_json_untouched(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        info_path = ds.root / "meta" / "info.json"
        before = info_path.read_bytes()

        set_feature_values(
            ds,
            edits=[
                {
                    "feature": "action",
                    "from_index": 0,
                    "to_index": 5,
                    "value": [0.0] * ds.meta.features["action"]["shape"][0],
                }
            ],
        )

        after = info_path.read_bytes()
        assert before == after, "info.json should never be rewritten by set_feature_values"

    def test_video_files_untouched(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20, use_videos=True)
        videos = list((ds.root / "videos").rglob("*.mp4")) if (ds.root / "videos").exists() else []
        if not videos:
            pytest.skip("factory built dataset without video files")
        before_mtimes = {p: p.stat().st_mtime_ns for p in videos}

        set_feature_values(
            ds,
            edits=[
                {
                    "feature": "action",
                    "from_index": 0,
                    "to_index": 5,
                    "value": [0.0] * ds.meta.features["action"]["shape"][0],
                }
            ],
        )

        for p, mtime in before_mtimes.items():
            assert p.stat().st_mtime_ns == mtime, f"video {p} was modified"


# ── Stats recomputation ─────────────────────────────────────────────────────


class TestStatsRecomputed:
    def test_action_stats_reflect_new_min_max(self, tmp_path, lerobot_dataset_factory):
        """Overwriting frames in episode 0 with a marker value pulls the
        per-episode action-min and action-max in that direction."""
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=3, total_frames=30)
        action_dim = ds.meta.features["action"]["shape"][0]

        # Plant a wildly out-of-bound marker value in episode 0 (frames 0..3).
        marker_min = -123.456
        set_feature_values(
            ds,
            edits=[
                {
                    "feature": "action",
                    "from_index": 0,
                    "to_index": 3,
                    "value": [marker_min] * action_dim,
                }
            ],
        )

        stats = _read_episode_stats_min_max(ds.root, "action")
        # Episode 0's new min must be at most the marker value.
        ep0_min, _ = stats[0]
        assert ep0_min <= marker_min + 1e-6, f"expected ep0 stats/action/min ≤ {marker_min}, got {ep0_min}"


# ── Fork-mode (in_place=False) ───────────────────────────────────────────────


class TestForkMode:
    def test_fork_writes_to_output_and_leaves_original_intact(self, tmp_path, lerobot_dataset_factory):
        ds = lerobot_dataset_factory(root=tmp_path / "ds", total_episodes=2, total_frames=20)
        action_dim = ds.meta.features["action"]["shape"][0]

        before_original = _read_data_df(ds.root).copy()
        fork_root = tmp_path / "fork"

        set_feature_values(
            ds,
            edits=[
                {
                    "feature": "action",
                    "from_index": 0,
                    "to_index": 5,
                    "value": [9.0] * action_dim,
                }
            ],
            in_place=False,
            output_dir=fork_root,
        )

        # Original is untouched.
        after_original = _read_data_df(ds.root)
        for i in range(len(before_original)):
            row_b = before_original.iloc[i]
            row_a = after_original.iloc[i]
            assert np.allclose(np.asarray(row_a["action"]), np.asarray(row_b["action"]))

        # Fork has the edits.
        forked = _read_data_df(fork_root)
        for idx in range(0, 5):
            assert np.allclose(np.asarray(forked[forked["index"] == idx].iloc[0]["action"]), 9.0)
