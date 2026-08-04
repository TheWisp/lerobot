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
"""``process_dataset`` end-to-end with an injected fake segmenter (no GPU/SAM3).

Per-region model: each object carries a ``treatment`` and ``background_treatment``
applies to everything else. Output counts (incl. variants), verbatim non-camera
data, per-region pixels, cancel mid-run. The pure per-region composite itself is
unit-tested in ``tests/gui/test_effects_composite.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.datasets import dataset_postprocess as pp
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class _FakeAdapter:
    """Segments a fixed central box as the "obj" foreground — a stand-in for SAM3."""

    def set_control(self, c):
        pass

    def set_camera(self, c):
        pass

    def reset(self):
        pass

    def segment(self, rgb):
        h, w = rgb.shape[:2]
        m = np.zeros((h, w), dtype=bool)
        m[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
        return {"obj": m}


@pytest.fixture
def src_dataset(tmp_path, empty_lerobot_dataset_factory):
    """A small image dataset: 2 episodes × 4 frames, one camera + state/action."""
    features = {
        "action": {"dtype": "float32", "shape": (3,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (3,), "names": None},
        "observation.images.cam": {"dtype": "image", "shape": (48, 64, 3), "names": None},
    }
    ds = empty_lerobot_dataset_factory(root=tmp_path / "src", features=features)
    for ep in range(2):
        for f in range(4):
            ds.add_frame(
                {
                    "action": np.array([0.1 * f, 0.0, 0.0], dtype=np.float32),
                    "observation.state": np.array([ep, f, 0.0], dtype=np.float32),
                    "observation.images.cam": np.full((48, 64, 3), 120, dtype=np.uint8),
                    "task": "pick",
                }
            )
        ds.save_episode()
    ds.finalize()
    return LeRobotDataset(repo_id=ds.repo_id, root=tmp_path / "src")


def test_process_dataset_treats_regions_and_preserves_non_camera_data(src_dataset, tmp_path):
    # Object "obj" kept as-is (none); background replaced with solid red.
    out = pp.process_dataset(
        src_dataset,
        out_repo_id="me/out",
        objects=[{"name": "obj", "sign": "+", "treatment": {"key": "none"}}],
        background_treatment={"key": "tint", "params": {"color": [255, 0, 0], "strength": 1.0}},
        out_root=tmp_path / "out",
        adapter=_FakeAdapter(),
    )
    assert out.episodes_written == 2 and out.frames_written == 8 and not out.cancelled

    res = LeRobotDataset(repo_id="me/out", root=tmp_path / "out")
    assert res.meta.total_episodes == 2 and res.meta.total_frames == 8
    # Non-camera data copied verbatim.
    for i in range(8):
        np.testing.assert_allclose(res[i]["action"].numpy(), src_dataset[i]["action"].numpy())
        np.testing.assert_allclose(
            res[i]["observation.state"].numpy(), src_dataset[i]["observation.state"].numpy()
        )
    # Camera pixels: object centre kept (real 120), background corner replaced (red).
    cam = (res[0]["observation.images.cam"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    assert tuple(cam[24, 32]) == pytest.approx((120, 120, 120), abs=3)
    assert tuple(cam[0, 0]) == pytest.approx((255, 0, 0), abs=3)


def test_process_dataset_variants_multiply_episodes(src_dataset, tmp_path):
    out = pp.process_dataset(
        src_dataset,
        out_repo_id="me/out",
        objects=[{"name": "obj"}],
        background_treatment={"key": "random"},
        variants=3,
        out_root=tmp_path / "out",
        adapter=_FakeAdapter(),
    )
    assert out.episodes_written == 6 and out.frames_written == 24
    res = LeRobotDataset(repo_id="me/out", root=tmp_path / "out")
    assert res.meta.total_episodes == 6 and res.meta.total_frames == 24


def test_process_dataset_cancel_midway_finalizes_partial(src_dataset, tmp_path):
    calls = {"n": 0}

    def cancel_after_5():
        calls["n"] += 1
        return calls["n"] > 5

    out = pp.process_dataset(
        src_dataset,
        out_repo_id="me/out",
        objects=[{"name": "obj"}],
        background_treatment={"key": "blur", "params": {"strength": 6}},
        out_root=tmp_path / "out",
        adapter=_FakeAdapter(),
        should_cancel=cancel_after_5,
    )
    assert out.cancelled
    assert out.frames_written < 8  # stopped early; a clean partial dataset remains
    LeRobotDataset(repo_id="me/out", root=tmp_path / "out")  # reads back without error


def test_copyable_feature_keys_excludes_task_and_defaults():
    # A dataset that materialises `task` (and `task_index`) as regular features — the
    # `task` string must NOT be copied to the output, else validate_frame (which strips
    # `task` before checking) reports it "missing" on every frame. Regression for a real
    # failure on `cylinder_ring_assembly_merged_raw`.
    features = {
        "action": {},
        "observation.state": {},
        "observation.images.top": {},
        "subtask": {},  # a benign extra feature — should pass through
        "task": {},  # special — must be excluded
        "task_index": {},  # DEFAULT — must be excluded
        "timestamp": {},  # DEFAULT — must be excluded
    }
    keys = pp._copyable_feature_keys(features)
    assert "task" not in keys and "task_index" not in keys and "timestamp" not in keys
    assert set(keys) == {"action", "observation.state", "observation.images.top", "subtask"}


def test_process_dataset_rejects_unknown_treatment(src_dataset, tmp_path):
    with pytest.raises(ValueError, match="unknown treatment"):
        pp.process_dataset(
            src_dataset,
            out_repo_id="me/out",
            objects=[{"name": "obj"}],
            background_treatment={"key": "does_not_exist"},
            out_root=tmp_path / "out",
            adapter=_FakeAdapter(),
        )


class _FakeBatchAdapter(_FakeAdapter):
    """Batch-capable stand-in (the SAM 3.1 shape): declares process_episode and
    records what the pipeline hands it; segment() still serves the per-frame loop."""

    def __init__(self):
        self.episodes_seen: list[dict[str, int]] = []

    def process_episode(self, frames_by_cam):
        assert all(len(v) > 0 for v in frames_by_cam.values())
        self.episodes_seen.append({k: len(v) for k, v in frames_by_cam.items()})


def test_process_dataset_batch_adapter_gets_whole_episodes(src_dataset, tmp_path):
    # A batch adapter is primed once per episode with EVERY edited camera's frames
    # (the fast native path), and the output is unchanged vs the per-frame contract.
    ad = _FakeBatchAdapter()
    out = pp.process_dataset(
        src_dataset,
        out_repo_id="me/out_batch",
        objects=[{"name": "obj", "sign": "+", "treatment": {"key": "none"}}],
        background_treatment={"key": "tint", "params": {"color": [255, 0, 0], "strength": 1.0}},
        out_root=tmp_path / "out_batch",
        adapter=ad,
    )
    assert out.episodes_written == 2 and out.frames_written == 8 and not out.cancelled
    assert ad.episodes_seen == [{"observation.images.cam": 4}] * 2
