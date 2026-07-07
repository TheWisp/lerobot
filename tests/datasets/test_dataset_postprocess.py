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
"""Tests for offline visual data editing (``dataset_postprocess``).

Covers the two halves that don't need a GPU/SAM3:
  * the pure effect/compositing functions (foreground protected, background
    replaced, global effects ignore the mask, alpha feathering);
  * ``process_dataset`` end-to-end with an injected fake segmenter — output
    counts (incl. variants), verbatim non-camera data, cancel mid-run.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.datasets import dataset_postprocess as pp
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class _FakeAdapter:
    """Segments a fixed central box as the foreground — a stand-in for SAM3."""

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


# ── Effect / compositing units ───────────────────────────────────────────────


def test_feathered_alpha_soft_edges():
    h, w = 48, 64
    m = np.zeros((h, w), dtype=bool)
    m[12:36, 16:48] = True
    alpha = pp._feathered_alpha([m], h, w)
    assert alpha.shape == (h, w)
    assert alpha.dtype == np.float32
    assert alpha.min() == 0.0 and alpha.max() <= 1.0
    assert alpha[24, 32] == pytest.approx(1.0, abs=1e-3)  # deep interior protected
    assert alpha[0, 0] == 0.0  # far background


def test_feathered_alpha_no_detection_is_all_background():
    alpha = pp._feathered_alpha([np.zeros((10, 10), dtype=bool)], 10, 10)
    assert alpha.max() == 0.0  # nothing detected -> whole frame is background


def test_bg_solid_protects_foreground_replaces_background():
    h, w = 48, 64
    rgb = np.full((h, w, 3), 120, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
    alpha = pp._feathered_alpha([mask], h, w)
    out = pp.apply_effect(rgb, alpha, "bg_solid", {"color": [255, 0, 0]}, {})
    assert tuple(out[h // 2, w // 2]) == pytest.approx((120, 120, 120), abs=2)  # fg kept
    assert tuple(out[0, 0]) == (255, 0, 0)  # bg replaced


def test_brightness_is_global_and_ignores_mask():
    rgb = np.full((8, 8, 3), 100, dtype=np.uint8)
    alpha = np.ones((8, 8), dtype=np.float32)  # would protect everything if respected
    out = pp.apply_effect(rgb, alpha, "brightness", {}, {"bright": 0.2, "contrast": 1.0})
    assert out.mean() > 120  # whole frame brightened despite the full mask


def test_sample_effect_random_color_is_deterministic_per_seed():
    a = pp.sample_effect("bg_random_color", {}, 8, 8, np.random.default_rng(0))
    b = pp.sample_effect("bg_random_color", {}, 8, 8, np.random.default_rng(0))
    assert a["color"] == b["color"]


# ── process_dataset end-to-end (fake segmenter) ──────────────────────────────


def test_process_dataset_preserves_non_camera_data(src_dataset, tmp_path):
    out = pp.process_dataset(
        src_dataset,
        out_repo_id="me/out",
        objects=[{"name": "obj", "color": [0, 255, 0], "sign": "+"}],
        effect="bg_solid",
        effect_params={"color": [255, 0, 0]},
        out_root=tmp_path / "out",
        adapter=_FakeAdapter(),
    )
    assert out.episodes_written == 2 and out.frames_written == 8 and not out.cancelled

    res = LeRobotDataset(repo_id="me/out", root=tmp_path / "out")
    assert res.meta.total_episodes == 2 and res.meta.total_frames == 8
    # Actions / states copied verbatim from the source.
    for i in range(8):
        np.testing.assert_allclose(res[i]["action"].numpy(), src_dataset[i]["action"].numpy())
        np.testing.assert_allclose(
            res[i]["observation.state"].numpy(), src_dataset[i]["observation.state"].numpy()
        )


def test_process_dataset_variants_multiply_episodes(src_dataset, tmp_path):
    out = pp.process_dataset(
        src_dataset,
        out_repo_id="me/out",
        objects=[{"name": "obj"}],
        effect="bg_random_color",
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
        effect="bg_solid",
        out_root=tmp_path / "out",
        adapter=_FakeAdapter(),
        should_cancel=cancel_after_5,
    )
    assert out.cancelled
    assert out.frames_written < 8  # stopped early; a clean partial dataset remains
    LeRobotDataset(repo_id="me/out", root=tmp_path / "out")  # reads back without error


def test_process_dataset_rejects_unknown_effect(src_dataset, tmp_path):
    with pytest.raises(ValueError, match="unknown effect"):
        pp.process_dataset(
            src_dataset,
            out_repo_id="me/out",
            objects=[{"name": "obj"}],
            effect="does_not_exist",
            out_root=tmp_path / "out",
            adapter=_FakeAdapter(),
        )
