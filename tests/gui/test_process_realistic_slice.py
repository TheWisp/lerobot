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
without needing a large dataset — and without a network fetch, an HF-cache
dependency, or rebuilding a fixture on every run.

The source is a **committed** artifact: ``tests/artifacts/datasets/lerobot/
data_editing_slice``, a 2 x 12-frame video-backed slice of the public
``lerobot/aloha_sim_transfer_cube_human`` (196 KB; see its README for provenance).
Because the pixels are fixed, the measured preserved-fraction is a stable baseline
instead of something that drifts with a re-encode.

The only irreducible prerequisites are the ones that cannot be committed: a CUDA
device and the gated ``facebook/sam3`` weights. Both are declared by the
``requires_sam3_gpu`` marker and auto-skip.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset

pytestmark = pytest.mark.requires_sam3_gpu

ARTIFACT_REPO = "lerobot/data_editing_slice"
ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "datasets" / ARTIFACT_REPO
CAM = "observation.images.top"
SLICE_EPISODES, SLICE_FRAMES = 2, 12
PROMPT = "robot arm"


@pytest.fixture(scope="module")
def realistic_slice():
    """The committed fixture — no fetch, no rebuild.

    Order matters: check CUDA BEFORE constructing the dataset. LeRobotDataset falls
    back to a Hub download whenever the local read fails, so on a CPU-only runner an
    early load turned into a 401 against a repo_id that exists only on disk. And the
    load runs under HF_HUB_OFFLINE so a broken artifact can never become a silent
    network fetch — this fixture is local by construction, and should fail loudly if
    it isn't.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if not (ARTIFACT_ROOT / "meta" / "info.json").exists():
        pytest.skip(f"{ARTIFACT_REPO} artifact missing — run `git lfs pull`")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HF_HUB_OFFLINE", "1")
        ds = LeRobotDataset(repo_id=ARTIFACT_REPO, root=ARTIFACT_ROOT)
    assert ds.meta.total_frames == SLICE_EPISODES * SLICE_FRAMES, "fixture changed shape"
    assert CAM in ds.meta.camera_keys
    return ds


@pytest.fixture(scope="module")
def sam3_adapter():
    """The real segmenter. Skips (never fails) when CUDA or the gated weights are
    absent — those are the only parts that cannot be committed alongside the test."""
    from lerobot.overlays.adapters import build_adapter

    try:
        return build_adapter("sam3_track", device="cuda", resolution=672)
    except Exception as e:  # gated weights absent, or transformers too old
        pytest.skip(f"real SAM3 unavailable: {e}")


def test_real_sam3_edits_a_photorealistic_slice(tmp_path, realistic_slice, sam3_adapter):
    import torch

    from lerobot.datasets.dataset_postprocess import process_dataset

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = process_dataset(
        realistic_slice,
        out_repo_id="test/realistic_slice_edited",
        adapter=sam3_adapter,
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
        f"\nreal SAM3 on {n} photorealistic frames: {wall:.1f}s "
        f"({wall / n * 1000:.0f} ms/frame, model load excluded — it happens in the fixture), "
        f"peak VRAM {peak_gb:.2f} GB"
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
