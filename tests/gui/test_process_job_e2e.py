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
"""End-to-end data-editing job on a SYNTHETIC dataset — config in, dataset out.

Everything the shipped job does except load a model: the worker parses its
``LEROBOT_PROCESS_WORKER_CONFIG``, runs ``process_dataset``, streams progress to
the job's JSON file, and finalizes a real ``LeRobotDataset`` on disk. The
segmenter is faked (a colour match), which is what makes the test hermetic and
fast; every other stage is production code.

The source dataset is built here with ``image`` features rather than video, so
the pixel assertions are exact instead of codec-approximate.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset

H, W = 48, 64
BG_VALUE = 100  # flat background the treatment must overwrite
OBJ_COLOR = (220, 30, 30)  # the "object" the fake segmenter keys on; must survive
# The object must be large relative to the compositor's 5 px feather: on a small
# object the softened mask never reaches alpha 1.0, so a few percent of the treated
# background bleeds into its centre (shipped behaviour, not a test artifact).
OBJ, FRAMES_PER_EP, N_EPISODES = 24, 6, 2
EDITED_CAM, EXCLUDED_CAM = "observation.images.cam", "observation.images.side"


def _frame(step: int) -> np.ndarray:
    """Flat background with a solid square sliding right — a moving 'object'."""
    img = np.full((H, W, 3), BG_VALUE, dtype=np.uint8)
    x = 4 + step * 4
    img[H // 2 - OBJ // 2 : H // 2 + OBJ // 2, x : x + OBJ] = OBJ_COLOR
    return img


def _obj_center(step: int) -> tuple[int, int]:
    return H // 2, 4 + step * 4 + OBJ // 2


def _to_u8(t) -> np.ndarray:
    """LeRobotDataset hands back CHW float in [0,1]; round (never truncate — 220/255
    scaled back is 219.999…) and restore HWC uint8."""
    return np.rint(t.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)


class _ColorSegmenter:
    """Stands in for SAM3: returns the exact-colour pixels as the object's mask."""

    def __init__(self, *a, **k):
        self.controls = []

    def set_control(self, c):
        self.controls.append(c)

    def set_camera(self, cam):
        self._cam = cam

    def reset(self):
        pass

    def segment(self, rgb):
        return {"block": np.all(rgb == np.asarray(OBJ_COLOR, dtype=np.uint8), axis=-1)}

    def segment_many(self, frames_by_cam):
        return {cam: self.segment(rgb) for cam, rgb in frames_by_cam.items()}


@pytest.fixture
def synthetic_source(tmp_path, empty_lerobot_dataset_factory):
    features = {
        "action": {"dtype": "float32", "shape": (3,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (3,), "names": None},
        EDITED_CAM: {"dtype": "image", "shape": (H, W, 3), "names": None},
        EXCLUDED_CAM: {"dtype": "image", "shape": (H, W, 3), "names": None},
    }
    ds = empty_lerobot_dataset_factory(root=tmp_path / "src", features=features)
    for ep in range(N_EPISODES):
        for f in range(FRAMES_PER_EP):
            ds.add_frame(
                {
                    "action": np.array([0.1 * f, float(ep), 0.0], dtype=np.float32),
                    "observation.state": np.array([float(ep), float(f), 0.5], dtype=np.float32),
                    EDITED_CAM: _frame(f),
                    EXCLUDED_CAM: np.full((H, W, 3), 7, dtype=np.uint8),
                    "task": "slide the block",
                }
            )
        ds.save_episode()
    ds.finalize()
    return LeRobotDataset(repo_id=ds.repo_id, root=tmp_path / "src")


def _run_worker(tmp_path, monkeypatch, source, *, variants=1):
    """Drive the real worker entry point over a fake segmenter."""
    from lerobot.gui import process_worker
    from lerobot.gui.process_jobs import JOBS_DIR, ProcessJobConfig, ProcessJobPaths
    from lerobot.overlays import adapters as adapters_mod

    monkeypatch.setattr(adapters_mod, "build_adapter", lambda *a, **k: _ColorSegmenter())
    jobs_dir = tmp_path / "jobs"
    cfg = ProcessJobConfig(
        job_id="e2e-job",
        source_id=str(source.root),
        source_repo_id=source.repo_id,
        source_root=str(source.root),
        out_repo_id="claude/e2e_out",
        out_root=str(tmp_path / "out"),
        model="sam3_track",
        objects=[{"name": "block", "sign": "+", "treatment": {"key": "none"}}],
        background_treatment={"key": "random", "params": {}},
        apply_mode="per_episode",
        variants=variants,
        multi_instance=True,
        cameras=[EDITED_CAM],
        episodes=None,
        preview=False,
        jobs_dir=str(jobs_dir),
    )
    monkeypatch.setenv("LEROBOT_PROCESS_WORKER_CONFIG", cfg.to_json())
    rc = process_worker.main()
    progress = json.loads(ProcessJobPaths.for_job("e2e-job", jobs_dir).progress.read_text())
    assert JOBS_DIR is not None  # module import sanity; the job used our tmp jobs_dir
    return rc, progress, LeRobotDataset(repo_id="claude/e2e_out", root=tmp_path / "out")


def test_job_writes_a_valid_edited_dataset(tmp_path, monkeypatch, synthetic_source):
    rc, progress, out = _run_worker(tmp_path, monkeypatch, synthetic_source)

    assert rc == 0
    assert progress["status"] == "complete" and progress["stage"] == "done"
    assert progress["frames_done"] == N_EPISODES * FRAMES_PER_EP

    # Shape preserved: same episode/frame counts, same features.
    assert out.meta.total_episodes == N_EPISODES
    assert out.meta.total_frames == N_EPISODES * FRAMES_PER_EP
    assert set(synthetic_source.meta.features) == set(out.meta.features)

    for i in range(len(out)):
        src_item, out_item = synthetic_source[i], out[i]
        # Non-camera data is copied verbatim — augmentation touches pixels only.
        np.testing.assert_allclose(out_item["action"].numpy(), src_item["action"].numpy())
        np.testing.assert_allclose(
            out_item["observation.state"].numpy(), src_item["observation.state"].numpy()
        )
        assert out_item["task"] == src_item["task"]
        # A camera the user excluded passes through untouched.
        np.testing.assert_array_equal(out_item[EXCLUDED_CAM].numpy(), src_item[EXCLUDED_CAM].numpy())

    # Pixels: the object (treatment "none") survives; the background is replaced.
    for ep in range(N_EPISODES):
        base = int(out.meta.episodes["dataset_from_index"][ep])
        for f in range(FRAMES_PER_EP):
            img = _to_u8(out[base + f][EDITED_CAM])
            cy, cx = _obj_center(f)
            np.testing.assert_array_equal(
                img[cy, cx],
                np.asarray(OBJ_COLOR, np.uint8),
                err_msg=f"object pixel was altered at ep{ep} f{f}",
            )
            assert img[0, 0].tolist() != [BG_VALUE] * 3, f"background not treated at ep{ep} f{f}"


def test_variants_write_independent_copies(tmp_path, monkeypatch, synthetic_source):
    # variants=N multiplies the dataset: each source episode is written N times with
    # an independently drawn background, which is the whole point of the knob.
    _rc, _progress, out = _run_worker(tmp_path, monkeypatch, synthetic_source, variants=2)

    assert out.meta.total_episodes == N_EPISODES * 2
    assert out.meta.total_frames == N_EPISODES * FRAMES_PER_EP * 2

    def bg_of(ep_index):
        base = int(out.meta.episodes["dataset_from_index"][ep_index])
        return _to_u8(out[base][EDITED_CAM])[0, 0]

    # Episodes 0 and N_EPISODES are the two variants of the SAME source episode.
    assert bg_of(0).tolist() != bg_of(N_EPISODES).tolist(), "variants must differ"
    # ...and each variant keeps the object intact.
    for ep in (0, N_EPISODES):
        base = int(out.meta.episodes["dataset_from_index"][ep])
        img = _to_u8(out[base][EDITED_CAM])
        cy, cx = _obj_center(0)
        np.testing.assert_array_equal(img[cy, cx], np.asarray(OBJ_COLOR, np.uint8))
