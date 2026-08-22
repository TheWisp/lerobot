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


def test_the_decode_gate_picks_the_conversion_and_rejects_a_constant_frame():
    """The GPU decode is admitted by comparing pixels, and the comparison also
    chooses the colour conversion instead of assuming one.

    Regression for a measured incident: torchcodec's CUDA decoder on this
    host returns a constant frame for every file, raising nothing, and a
    training run consumed it for 800 steps with a loss curve indistinguishable
    from the correct run's (0.2128 against 0.2056 at step 800). Neither an
    exception nor the loss catches that. The second half guards the subtler
    case: a wrong colour matrix is a perfectly normal-looking image that is
    quietly wrong everywhere (measured mean 11.3 levels against 0.91 for the
    right one).
    """
    from lerobot.datasets.gpu_data_pipeline import (
        DECODE_TOLERANCE_MEAN,
        NV12Conversion,
        nv12_to_rgb,
        select_conversion,
    )

    h, w = 32, 48
    gen = torch.Generator().manual_seed(7)
    plane = torch.randint(16, 240, (h * 3 // 2, w), dtype=torch.uint8, generator=gen)
    truth = NV12Conversion("bt601", limited_range=True)
    reference = nv12_to_rgb(plane, h, w, truth).float()

    mean, _, chosen = select_conversion(reference, plane, h, w)
    assert chosen == truth, f"picked {chosen} instead of {truth}"
    assert mean == 0.0

    # The observed failure: Y and chroma constant -> a flat frame.
    flat = torch.empty_like(plane)
    flat[:h] = 30
    flat[h:] = 255
    flat_mean, _, _ = select_conversion(reference, flat, h, w)
    assert flat_mean > DECODE_TOLERANCE_MEAN, "a constant frame must never be admitted"

    # A plausible-looking image under the wrong matrix must also be rejected.
    wrong = nv12_to_rgb(plane, h, w, NV12Conversion("bt709", limited_range=False)).float()
    assert (reference - wrong).abs().mean() > DECODE_TOLERANCE_MEAN
