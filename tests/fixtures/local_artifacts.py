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
"""Committed dataset slices — so tests never download from the Hub.

Tests that called ``LeRobotDataset("lerobot/pusht", ...)`` fetched a real dataset at
fixture setup. That makes the suite depend on network availability and on HuggingFace
not rate-limiting the runner: CI failed with ``429 Too Many Requests`` on exactly
those tests, which says nothing about the code under test.

The slices under ``tests/artifacts/datasets/lerobot/`` carry the first three episodes
of each source dataset with every feature preserved, at a fraction of a megabyte, so
the same tests run offline and deterministically.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "datasets" / "lerobot"


def local_dataset(name: str, **kwargs):
    """Load a committed slice by artifact name (e.g. ``"pusht_slice"``).

    Pre: the artifact is present (git-lfs pulled). Post: a LeRobotDataset backed
    entirely by files in this repository — no Hub request is made, so a broken
    artifact fails loudly instead of silently downloading.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = ARTIFACT_DIR / name
    if not (root / "meta" / "info.json").exists():
        pytest.skip(f"{name} artifact missing — run `git lfs pull`")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HF_HUB_OFFLINE", "1")
        return LeRobotDataset(repo_id=f"lerobot/{name}", root=root, **kwargs)
