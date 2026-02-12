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

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.gui.frame_cache import FrameCache

logger = logging.getLogger(__name__)

# File names for persistence
EDITS_FILENAME = ".lerobot_gui_edits.json"
EDITS_FILE_VERSION = 1


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


# ============================================================================
# Persistence Functions
# ============================================================================


def _edit_to_dict(edit: PendingEdit) -> dict[str, Any]:
    """Convert a PendingEdit to a JSON-serializable dict."""
    return {
        "edit_type": edit.edit_type,
        "episode_index": edit.episode_index,
        "params": edit.params,
        "created_at": edit.created_at.isoformat(),
    }


def _dict_to_edit(d: dict[str, Any], dataset_id: str) -> PendingEdit:
    """Convert a dict back to a PendingEdit."""
    return PendingEdit(
        edit_type=d["edit_type"],
        dataset_id=dataset_id,
        episode_index=d["episode_index"],
        params=d.get("params", {}),
        created_at=datetime.fromisoformat(d["created_at"]) if "created_at" in d else datetime.now(),
    )


def save_edits_to_file(root: Path, edits: list[PendingEdit]) -> None:
    """Save pending edits to the dataset's edits file.

    Args:
        root: Dataset root directory.
        edits: List of pending edits to save.
    """
    edits_file = root / EDITS_FILENAME

    if not edits:
        # No edits - remove file if it exists
        if edits_file.exists():
            edits_file.unlink()
            logger.info(f"Removed empty edits file: {edits_file}")
        return

    data = {
        "version": EDITS_FILE_VERSION,
        "edits": [_edit_to_dict(e) for e in edits],
    }

    edits_file.write_text(json.dumps(data, indent=2))
    logger.info(f"Saved {len(edits)} edits to {edits_file}")


def load_edits_from_file(root: Path, dataset_id: str) -> list[PendingEdit]:
    """Load pending edits from the dataset's edits file.

    Args:
        root: Dataset root directory.
        dataset_id: Dataset ID to associate with loaded edits.

    Returns:
        List of pending edits, empty if file doesn't exist.
    """
    edits_file = root / EDITS_FILENAME

    if not edits_file.exists():
        return []

    try:
        data = json.loads(edits_file.read_text())
        version = data.get("version", 1)

        if version > EDITS_FILE_VERSION:
            logger.warning(f"Edits file version {version} is newer than supported {EDITS_FILE_VERSION}")

        edits = [_dict_to_edit(d, dataset_id) for d in data.get("edits", [])]
        logger.info(f"Loaded {len(edits)} edits from {edits_file}")
        return edits

    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to load edits from {edits_file}: {e}")
        return []


def clear_edits_file(root: Path) -> None:
    """Remove the edits file after edits are applied.

    Args:
        root: Dataset root directory.
    """
    edits_file = root / EDITS_FILENAME
    if edits_file.exists():
        edits_file.unlink()
        logger.info(f"Cleared edits file: {edits_file}")
