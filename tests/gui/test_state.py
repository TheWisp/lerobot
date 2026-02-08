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
"""Tests for the AppState class."""

import pytest

from lerobot.gui.frame_cache import FrameCache
from lerobot.gui.state import AppState, PendingEdit


@pytest.fixture
def app_state():
    """Create an AppState instance for testing."""
    return AppState(frame_cache=FrameCache(max_bytes=1_000_000))


class TestEditFiltering:
    """Tests for filtering edits by dataset."""

    def test_get_edits_for_dataset_filters_correctly(self, app_state):
        """Verify that get_edits_for_dataset returns only matching edits."""
        app_state.add_edit(PendingEdit("trim", "ds_a", 0, {"start_frame": 5, "end_frame": 10}))
        app_state.add_edit(PendingEdit("delete", "ds_b", 1))
        app_state.add_edit(PendingEdit("trim", "ds_a", 2, {"start_frame": 0, "end_frame": 5}))
        app_state.add_edit(PendingEdit("delete", "ds_a", 3))

        edits_a = app_state.get_edits_for_dataset("ds_a")

        assert len(edits_a) == 3
        assert all(e.dataset_id == "ds_a" for e in edits_a)

    def test_get_edits_for_dataset_empty_result(self, app_state):
        """Verify that filtering returns empty list for non-existent dataset."""
        app_state.add_edit(PendingEdit("trim", "ds_a", 0, {}))

        edits = app_state.get_edits_for_dataset("ds_nonexistent")

        assert edits == []

    def test_get_edits_for_dataset_preserves_order(self, app_state):
        """Verify that edits are returned in the order they were added."""
        app_state.add_edit(PendingEdit("trim", "ds_a", 0, {}))
        app_state.add_edit(PendingEdit("trim", "ds_a", 1, {}))
        app_state.add_edit(PendingEdit("trim", "ds_a", 2, {}))

        edits = app_state.get_edits_for_dataset("ds_a")

        assert [e.episode_index for e in edits] == [0, 1, 2]


class TestTrimReplacement:
    """Tests for trim replacement behavior."""

    def test_trim_replacement_logic(self, app_state):
        """Verify that the trim replacement logic works correctly.

        This tests the pattern used in the API where existing trims
        are removed before adding a new one for the same episode.
        """
        # Add initial trim
        app_state.add_edit(PendingEdit("trim", "ds_a", 0, {"start_frame": 0, "end_frame": 50}))
        app_state.add_edit(PendingEdit("delete", "ds_a", 1))  # Different edit type
        app_state.add_edit(PendingEdit("trim", "ds_a", 2, {"start_frame": 0, "end_frame": 30}))

        # Simulate replacement: remove existing trim for episode 0
        app_state.pending_edits = [
            e for e in app_state.pending_edits
            if not (e.dataset_id == "ds_a" and e.episode_index == 0 and e.edit_type == "trim")
        ]
        # Add new trim for episode 0
        app_state.add_edit(PendingEdit("trim", "ds_a", 0, {"start_frame": 10, "end_frame": 40}))

        # Verify only one trim for episode 0
        trims_ep0 = [
            e for e in app_state.get_edits_for_dataset("ds_a")
            if e.edit_type == "trim" and e.episode_index == 0
        ]
        assert len(trims_ep0) == 1
        assert trims_ep0[0].params["start_frame"] == 10
        assert trims_ep0[0].params["end_frame"] == 40

        # Verify other edits are untouched
        assert len(app_state.pending_edits) == 3  # delete + trim ep2 + new trim ep0

    def test_trim_replacement_different_episode_untouched(self, app_state):
        """Verify that trim replacement only affects the specific episode."""
        app_state.add_edit(PendingEdit("trim", "ds_a", 0, {"start_frame": 0, "end_frame": 50}))
        app_state.add_edit(PendingEdit("trim", "ds_a", 1, {"start_frame": 5, "end_frame": 45}))

        # Replace trim for episode 0 only
        app_state.pending_edits = [
            e for e in app_state.pending_edits
            if not (e.dataset_id == "ds_a" and e.episode_index == 0 and e.edit_type == "trim")
        ]
        app_state.add_edit(PendingEdit("trim", "ds_a", 0, {"start_frame": 10, "end_frame": 40}))

        # Episode 1's trim should be unchanged
        trims_ep1 = [
            e for e in app_state.pending_edits
            if e.episode_index == 1 and e.edit_type == "trim"
        ]
        assert len(trims_ep1) == 1
        assert trims_ep1[0].params["start_frame"] == 5


class TestBasicEditOperations:
    """Tests for basic edit add/remove/clear operations."""

    def test_add_edit(self, app_state):
        """Verify that add_edit appends to the list."""
        edit = PendingEdit("delete", "ds_a", 0)
        app_state.add_edit(edit)

        assert len(app_state.pending_edits) == 1
        assert app_state.pending_edits[0] == edit

    def test_remove_edit_valid_index(self, app_state):
        """Verify that remove_edit removes the correct entry."""
        app_state.add_edit(PendingEdit("delete", "ds_a", 0))
        app_state.add_edit(PendingEdit("delete", "ds_a", 1))
        app_state.add_edit(PendingEdit("delete", "ds_a", 2))

        app_state.remove_edit(1)

        assert len(app_state.pending_edits) == 2
        assert app_state.pending_edits[0].episode_index == 0
        assert app_state.pending_edits[1].episode_index == 2

    def test_remove_edit_invalid_index(self, app_state):
        """Verify that invalid index is a no-op."""
        app_state.add_edit(PendingEdit("delete", "ds_a", 0))

        app_state.remove_edit(5)  # Invalid index
        app_state.remove_edit(-1)  # Negative index

        assert len(app_state.pending_edits) == 1

    def test_clear_edits_all(self, app_state):
        """Verify that clear_edits without dataset clears all."""
        app_state.add_edit(PendingEdit("delete", "ds_a", 0))
        app_state.add_edit(PendingEdit("delete", "ds_b", 1))

        app_state.clear_edits()

        assert len(app_state.pending_edits) == 0

    def test_clear_edits_by_dataset(self, app_state):
        """Verify that clear_edits with dataset clears only that dataset."""
        app_state.add_edit(PendingEdit("delete", "ds_a", 0))
        app_state.add_edit(PendingEdit("delete", "ds_b", 1))
        app_state.add_edit(PendingEdit("trim", "ds_a", 2, {}))

        app_state.clear_edits("ds_a")

        assert len(app_state.pending_edits) == 1
        assert app_state.pending_edits[0].dataset_id == "ds_b"


class TestEpisodeStatusChecks:
    """Tests for is_episode_deleted and get_episode_trim."""

    def test_is_episode_deleted_true(self, app_state):
        """Verify is_episode_deleted returns True for deleted episodes."""
        app_state.add_edit(PendingEdit("delete", "ds_a", 5))

        assert app_state.is_episode_deleted("ds_a", 5) is True

    def test_is_episode_deleted_false(self, app_state):
        """Verify is_episode_deleted returns False for non-deleted episodes."""
        app_state.add_edit(PendingEdit("delete", "ds_a", 5))

        assert app_state.is_episode_deleted("ds_a", 0) is False
        assert app_state.is_episode_deleted("ds_b", 5) is False

    def test_is_episode_deleted_trim_not_counted(self, app_state):
        """Verify that trim edits don't count as deletions."""
        app_state.add_edit(PendingEdit("trim", "ds_a", 5, {}))

        assert app_state.is_episode_deleted("ds_a", 5) is False

    def test_get_episode_trim_exists(self, app_state):
        """Verify get_episode_trim returns the trim range."""
        app_state.add_edit(PendingEdit("trim", "ds_a", 0, {"start_frame": 10, "end_frame": 50}))

        result = app_state.get_episode_trim("ds_a", 0)

        assert result == (10, 50)

    def test_get_episode_trim_not_found(self, app_state):
        """Verify get_episode_trim returns None when no trim exists."""
        app_state.add_edit(PendingEdit("delete", "ds_a", 0))

        assert app_state.get_episode_trim("ds_a", 0) is None
        assert app_state.get_episode_trim("ds_a", 1) is None
