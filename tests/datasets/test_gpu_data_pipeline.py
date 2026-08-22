# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The GPU data path must fetch the same frames and produce the same images.

Two contracts, both testable without CUDA (torchcodec decodes on CPU and every
op in the pipeline is device-generic, so the *logic* is pinned here and only
the device changes in production):

* **Frame identity.** GpuFrameSource rebuilds the global-index -> (video file,
  frame-within-file) mapping from episode metadata instead of going through
  LeRobotDataset. An off-by-one there is silent — training would run on
  temporally shifted frames and nothing would error — so every frame it
  returns is compared against what the dataset returns for the same index.

* **Image equality.** GpuImagePipeline.prepare must match the CPU path
  (decode -> SavedMaskCompositor -> resize) within the composite-equivalence
  contract. The comparison runs at full precision through both paths on the
  same fixture dataset the training-side mask tests already use.
"""

import json

import numpy as np
import pytest
import torch

pytest.importorskip("torchcodec")

from lerobot.datasets.gpu_data_pipeline import GpuFrameSource, GpuImagePipeline  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from tests.datasets.test_saved_masks_training import masked_dataset_root  # noqa: E402,F401

CAMERAS = ["observation.images.top", "observation.images.wrist"]


def _all_none_recipe(root) -> None:
    """The fixture's recipe tints an object; the GPU path's v1 gate refuses
    that. The gate is deliberate, so the fixture is narrowed to the supported
    class instead of the gate being widened for a test."""
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for feat in info["features"].values():
        if isinstance(feat, dict) and feat.get("mask_encoding") == "coco_rle":
            feat["mask_treatments"] = {k: {"key": "none", "params": {}} for k in feat["mask_labels"]}
            feat["mask_background"] = {"key": "blur", "params": {}}
    info_path.write_text(json.dumps(info, indent=2))


def test_every_fetched_frame_is_the_datasets_frame(masked_dataset_root):  # noqa: F811
    root, repo_id = masked_dataset_root
    ds = LeRobotDataset(repo_id, root=root, return_uint8=True)
    src = GpuFrameSource(ds, CAMERAS[0], device="cpu")
    rng = np.random.default_rng(5)
    idx = rng.integers(0, ds.num_frames, 6)
    got = src.fetch(idx.astype(np.int64))
    for j, i in enumerate(idx):
        want = ds[int(i)][CAMERAS[0]]
        assert torch.equal(got[j].cpu(), want), f"frame mismatch at dataset index {int(i)}"


def test_prepare_matches_the_cpu_path(masked_dataset_root):  # noqa: F811
    from torchvision.transforms import functional as tvf

    root, repo_id = masked_dataset_root
    _all_none_recipe(root)
    resize = (48, 48)

    cpu_ds = LeRobotDataset(repo_id, root=root, apply_saved_masks=True)
    pipe_ds = LeRobotDataset(repo_id, root=root)
    pipe = GpuImagePipeline(pipe_ds, CAMERAS, resize_to=resize, device="cpu")
    assert set(pipe.composites) == set(CAMERAS), "fixture masks both cameras; both must composite"

    rng = np.random.default_rng(9)
    idx = [int(i) for i in rng.integers(0, cpu_ds.num_frames, 5)]
    batch = {
        "index": torch.tensor(idx),
        **{pipe.mask_key[cam]: [pipe_ds.hf_dataset[i][pipe.mask_key[cam]] for i in idx] for cam in CAMERAS},
    }
    gpu_out = pipe.prepare(batch, timed=True)

    for cam in CAMERAS:
        want = torch.stack(
            [
                tvf.resize(
                    cpu_ds[i][cam], list(resize), interpolation=tvf.InterpolationMode.BILINEAR, antialias=True
                )
                for i in idx
            ]
        )
        got = gpu_out[cam].cpu()
        assert got.shape == want.shape and got.dtype == torch.float32
        # Both paths quantize the composite to uint8 before the resize, so the
        # composite-equivalence contract (≤2 levels) carries through the same
        # linear resize unchanged.
        diff = (got - want).abs().max().item() * 255.0
        assert diff <= 2.0 + 1e-3, f"{cam}: images diverged by {diff:.2f}/255"

    report = pipe.report()
    assert report.get("gpu_prep_steps_timed") == 1.0
    assert all(f"gpu_prep_{p}_ms" in report for p in pipe.PHASES)


def test_an_unmasked_camera_passes_through_untreated(masked_dataset_root):  # noqa: F811
    """Cameras without a mask column decode and resize only."""
    root, repo_id = masked_dataset_root
    ds = LeRobotDataset(repo_id, root=root)
    # strip the mask features so no compositor is built
    for key in [k for k in ds.meta.features if k.startswith("observation.masks.")]:
        del ds.meta.features[key]
    pipe = GpuImagePipeline(ds, CAMERAS[:1], resize_to=None, device="cpu")
    assert not pipe.composites
    out = pipe.prepare({"index": torch.tensor([0, 3])})
    img = out[CAMERAS[0]]
    assert img.dtype == torch.float32 and img.max() <= 1.0


def test_external_images_sample_skips_video_but_keeps_the_pipelines_inputs(masked_dataset_root):  # noqa: F811
    """The wrapper's GPU mode must not decode video, and must still deliver
    what the pipeline and the audit need: the global index and the RLE rows."""
    from lerobot.policies.hvla.s1.flow_matching.train import SAMPLING_AUDIT_KEY, FlowMatchingDataset

    root, repo_id = masked_dataset_root
    ds = LeRobotDataset(repo_id, root=root)
    wrapper = FlowMatchingDataset(
        ds,
        s2_latents=None,
        chunk_size=4,
        max_delay_seconds=0.0,
        fps=ds.fps,
        resize_to=(48, 48),
        image_keys=CAMERAS,
        action_feature_names=None,
        state_feature_names=[f"j{i}.pos" for i in range(6)],
        external_images=True,
    )
    sample = wrapper[2]
    assert not any(k.startswith("observation.images.") for k in sample), "video was decoded"
    assert "index" in sample
    assert SAMPLING_AUDIT_KEY in sample
    for cam in CAMERAS:
        key = cam.replace(".images.", ".masks.")
        assert key in sample, f"mask rows for {key} must ride through the batch"


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU decode needs CUDA")
def test_gpu_decode_reproduces_the_cpu_decoder(masked_dataset_root):  # noqa: F811
    """The shipped GPU decode must reproduce the CPU decoder's frames.

    The unit tests above pin the gate's logic on synthetic planes; this runs
    the real thing -- NVDEC, the calibrated conversion, the parallel
    file-splitting fetch -- against LeRobotDataset's own frames. Measured on
    the training datasets: mean 0.45, max 3 at 720p across ten cameras of
    three datasets, which is the 4:2:0 chroma round-trip.
    """
    pytest.importorskip("PyNvVideoCodec")
    from lerobot.datasets.gpu_data_pipeline import DECODE_TOLERANCE_MAX, DECODE_TOLERANCE_MEAN

    root, repo_id = masked_dataset_root
    ds = LeRobotDataset(repo_id, root=root, return_uint8=True)
    src = GpuFrameSource(ds, CAMERAS[0], device="cuda")
    idx = np.array([0, 5, 2, 9], dtype=np.int64)
    got = src.fetch(idx).cpu().float()
    for j, i in enumerate(idx):
        want = ds[int(i)][CAMERAS[0]].float()
        diff = (want - got[j]).abs()
        assert diff.mean() <= DECODE_TOLERANCE_MEAN, f"index {i}: mean {diff.mean():.2f}"
        assert diff.max() <= DECODE_TOLERANCE_MAX, f"index {i}: max {diff.max():.0f}"
