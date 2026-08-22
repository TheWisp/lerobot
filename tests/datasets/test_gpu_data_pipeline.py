# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The GPU data path must fetch the frames it claims to fetch.

Testable without CUDA: torchcodec decodes on CPU and every op in the pipeline is
device-generic, so the *logic* is pinned here and only the device changes in
production.

GpuFrameSource rebuilds the global-index -> (video file, frame-within-file)
mapping from episode metadata instead of going through LeRobotDataset. An
off-by-one there is silent -- training would run on temporally shifted frames
and nothing would error -- so every frame it returns is compared against what
the dataset returns for the same index.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("torchcodec")

from lerobot.datasets.gpu_data_pipeline import GpuFrameSource  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from tests.fixtures.constants import DUMMY_REPO_ID  # noqa: E402


def test_every_fetched_frame_is_the_datasets_frame(tmp_path, lerobot_dataset_factory):
    """Pre: a video-backed dataset. Post: fetch(i) == dataset[i] for every i."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "gpu_pipeline_dataset",
        repo_id=DUMMY_REPO_ID,
        total_episodes=2,
        total_frames=40,
        use_videos=True,
    )
    camera = next(iter(dataset.meta.video_keys), None)
    if camera is None:
        pytest.skip("fixture produced no video keys")

    # uint8 on both sides: the source returns raw decoded pixels, so a float
    # dataset would compare 1.0 against 255 and fail for the wrong reason.
    dataset = LeRobotDataset(DUMMY_REPO_ID, root=dataset.root, return_uint8=True)
    source = GpuFrameSource(dataset, camera, device="cpu")
    rng = np.random.default_rng(5)
    indices = rng.integers(0, dataset.num_frames, 6).astype(np.int64)

    fetched = source.fetch(indices)

    for j, i in enumerate(indices):
        want = dataset[int(i)][camera]
        assert torch.equal(fetched[j].cpu(), want), f"frame mismatch at dataset index {int(i)}"


def test_a_silently_wrong_cuda_decode_is_rejected():
    """The probe must compare PIXELS, not just check that decoding returns.

    Regression for a measured incident: on Blackwell with torchcodec
    0.11.1+cu128, AV1 decodes on CUDA into a flat blue frame with no error
    raised. A training run consumed it for 800 steps and its loss curve was
    indistinguishable from the correct run's (0.2128 vs 0.2056 at step 800),
    so neither an exception nor the loss would ever have caught it. These are
    the actual observed frames' statistics.
    """
    from lerobot.datasets.gpu_data_pipeline import DECODE_TOLERANCE, decode_disagreement

    real = torch.randint(60, 90, (3, 64, 64), dtype=torch.uint8)
    flat = torch.zeros((3, 64, 64), dtype=torch.uint8)
    flat[2] = 255  # R=0, G~0, B=255 — the observed failure

    assert decode_disagreement(real, flat) > DECODE_TOLERANCE, "flat frame must be rejected"
    assert decode_disagreement(real, real.clone()) == 0.0, "identical decodes must be accepted"
    # A codec whose hardware path rounds slightly still passes.
    nearly = (real.float() + 0.4).to(torch.uint8)
    assert decode_disagreement(real, nearly) <= DECODE_TOLERANCE
