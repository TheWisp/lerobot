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
"""Data editing on a photorealistic slice, with the REAL SAM3 — correctness + speed.

``test_process_job_e2e`` proves the plumbing with a fake segmenter, but it cannot
prove the *model* finds anything: its imagery (a solid square on flat grey) is not
something an open-vocabulary segmenter would ever key on. This test closes that gap
without needing a large dataset — it slices a handful of frames out of a PUBLIC
LeRobot dataset (real rendered robot footage), rewrites them as a tiny video-backed
dataset, and runs the shipped job over it with the real ``sam3_track`` adapter.

Small enough to stay quick, real enough that segmentation is meaningful. It also
prints the measured throughput and peak VRAM, so a regression in either shows up
here rather than on a six-hour production run.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

pytestmark = pytest.mark.requires_sam3_gpu

PUBLIC_REPO = "lerobot/aloha_sim_transfer_cube_human"  # public, real rendered footage
CAM = "observation.images.top"
SLICE_EPISODES, SLICE_FRAMES = 2, 12
PROMPT = "robot arm"


@pytest.fixture(scope="module")
def realistic_slice(tmp_path_factory):
    """A tiny video-backed dataset carved out of a public one — photorealistic
    content, trivial size. Skips (never downloads) when the source isn't cached."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    root = HF_LEROBOT_HOME / PUBLIC_REPO
    if not (root / "meta" / "info.json").exists():
        pytest.skip(f"{PUBLIC_REPO} not in the local HF cache; fetch it to run this test")

    src = LeRobotDataset(repo_id=PUBLIC_REPO, root=root)
    assert CAM in src.meta.camera_keys, f"{PUBLIC_REPO} lost {CAM}"
    out_root = Path(tmp_path_factory.mktemp("slice")) / "src"
    h, w = src.meta.features[CAM]["shape"][:2]
    dst = LeRobotDataset.create(
        repo_id="test/realistic_slice",
        fps=src.meta.fps,
        features={
            "action": {"dtype": "float32", "shape": (len(src[0]["action"]),), "names": None},
            CAM: {"dtype": "video", "shape": (h, w, 3), "names": None},
        },
        root=out_root,
        use_videos=True,
    )
    for ep in range(SLICE_EPISODES):
        start = int(src.meta.episodes["dataset_from_index"][ep])
        for f in range(SLICE_FRAMES):
            item = src[start + f]
            frame = item[CAM]
            rgb = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            dst.add_frame({"action": item["action"].numpy(), CAM: rgb, "task": "transfer cube"})
        dst.save_episode()
    dst.finalize()
    return LeRobotDataset(repo_id="test/realistic_slice", root=out_root)


def test_real_sam3_edits_a_photorealistic_slice(tmp_path, realistic_slice):
    import torch

    from lerobot.datasets.dataset_postprocess import process_dataset

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = process_dataset(
        realistic_slice,
        out_repo_id="test/realistic_slice_edited",
        model="sam3_track",
        resolution=672,
        objects=[{"name": PROMPT, "sign": "+", "treatment": {"key": "none"}}],
        background_treatment={"key": "random", "params": {}},
        cameras=[CAM],
        out_root=tmp_path / "edited",
        multi_instance=True,
    )
    wall = time.perf_counter() - t0
    n = SLICE_EPISODES * SLICE_FRAMES
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"\nreal SAM3 on {n} photorealistic frames: {wall:.1f}s total "
        f"({wall / n * 1000:.0f} ms/frame incl. model load), peak VRAM {peak_gb:.2f} GB"
    )

    assert not result.cancelled
    assert result.episodes_written == SLICE_EPISODES
    assert result.frames_written == n

    out = LeRobotDataset(repo_id="test/realistic_slice_edited", root=tmp_path / "edited")
    preserved = []
    for ep in range(SLICE_EPISODES):
        base = int(out.meta.episodes["dataset_from_index"][ep])
        for f in range(SLICE_FRAMES):
            a = (out[base + f][CAM].numpy() * 255).astype(np.int16)
            b = (realistic_slice[base + f][CAM].numpy() * 255).astype(np.int16)
            preserved.append(float((np.abs(a - b).max(axis=0) < 12).mean()))

    kept = float(np.mean(preserved))
    print(f"pixels preserved (the prompted object): {kept * 100:.1f}% of frame")
    # The real assertion: the mask is neither empty nor everything. An empty mask
    # would randomize the whole frame (kept ~ 0); a runaway mask would protect it
    # all (kept ~ 1). Either means segmentation is broken on real imagery even
    # though the plumbing "works".
    assert 0.02 < kept < 0.75, f"implausible preserved fraction {kept:.3f} — check segmentation"
    # VRAM is bounded by the rolling session rebuild, not creeping with frame count.
    assert peak_gb < 12, f"peak VRAM {peak_gb:.1f} GB is far above the ~3 GB this model needs"
