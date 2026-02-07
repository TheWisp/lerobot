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
"""Application state for the GUI server."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.gui.frame_cache import FrameCache


@dataclass
class PendingEdit:
    """A pending edit operation that hasn't been applied yet."""

    edit_type: Literal["delete", "trim"]
    dataset_id: str
    episode_index: int
    params: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AppState:
    """Global application state.

    Holds all opened datasets, frame cache, and pending edits.
    """

    frame_cache: "FrameCache"
    datasets: dict[str, "LeRobotDataset"] = field(default_factory=dict)
    pending_edits: list[PendingEdit] = field(default_factory=list)

    def add_edit(self, edit: PendingEdit) -> None:
        """Add a pending edit."""
        self.pending_edits.append(edit)

    def remove_edit(self, index: int) -> None:
        """Remove a pending edit by index."""
        if 0 <= index < len(self.pending_edits):
            self.pending_edits.pop(index)

    def clear_edits(self, dataset_id: str | None = None) -> None:
        """Clear pending edits, optionally filtered by dataset."""
        if dataset_id is None:
            self.pending_edits.clear()
        else:
            self.pending_edits = [e for e in self.pending_edits if e.dataset_id != dataset_id]

    def get_edits_for_dataset(self, dataset_id: str) -> list[PendingEdit]:
        """Get pending edits for a specific dataset."""
        return [e for e in self.pending_edits if e.dataset_id == dataset_id]

    def is_episode_deleted(self, dataset_id: str, episode_index: int) -> bool:
        """Check if an episode is marked for deletion."""
        return any(
            e.dataset_id == dataset_id and e.episode_index == episode_index and e.edit_type == "delete"
            for e in self.pending_edits
        )

    def get_episode_trim(self, dataset_id: str, episode_index: int) -> tuple[int, int] | None:
        """Get trim range for an episode if it exists."""
        for e in self.pending_edits:
            if e.dataset_id == dataset_id and e.episode_index == episode_index and e.edit_type == "trim":
                return (e.params.get("start_frame", 0), e.params.get("end_frame", 0))
        return None
