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

from lerobot.datasets.gpu_data_pipeline import GpuFrameSource, GpuImagePipeline  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from tests.fixtures.constants import DUMMY_REPO_ID, DUMMY_VIDEO_INFO  # noqa: E402


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU decode needs CUDA")
def test_gpu_decode_reproduces_the_cpu_decoder(tmp_path, lerobot_dataset_factory):
    """The shipped GPU decode must reproduce the CPU decoder's frames.

    The unit tests above pin the gate's logic on synthetic planes; this runs
    the real thing -- NVDEC, the calibrated conversion, the parallel
    file-splitting fetch -- against LeRobotDataset's own frames. Measured on
    the training datasets: mean 0.45, max 3 at 720p across ten cameras of
    three datasets, which is the 4:2:0 chroma round-trip.
    """
    pytest.importorskip("PyNvVideoCodec")
    from lerobot.datasets.gpu_data_pipeline import DECODE_TOLERANCE_MAX, DECODE_TOLERANCE_MEAN

    # NVDEC refuses to create a decoder below its minimum dimensions, and the
    # shared fixture's cameras are 64x96 -- well under it. Asking for a larger
    # frame is what makes this exercise the hardware decoder rather than skip.
    built = lerobot_dataset_factory(
        root=tmp_path / "gpu_decode_dataset",
        repo_id=DUMMY_REPO_ID,
        total_episodes=2,
        total_frames=40,
        use_videos=True,
        camera_features={
            "front": {
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
                "info": DUMMY_VIDEO_INFO,
            }
        },
    )
    camera = next(iter(built.meta.video_keys), None)
    if camera is None:
        pytest.skip("fixture produced no video keys")
    ds = LeRobotDataset(DUMMY_REPO_ID, root=built.root, return_uint8=True)
    src = GpuFrameSource(ds, camera, device="cuda")
    idx = np.array([0, 5, 2, 9], dtype=np.int64)
    got = src.fetch(idx).cpu().float()
    for j, i in enumerate(idx):
        want = ds[int(i)][camera].float()
        diff = (want - got[j]).abs()
        assert diff.mean() <= DECODE_TOLERANCE_MEAN, f"index {i}: mean {diff.mean():.2f}"
        assert diff.max() <= DECODE_TOLERANCE_MAX, f"index {i}: max {diff.max():.0f}"


def _cpu_path_images(root, repo_id, camera, indices, resize_to):
    """What the CPU data path hands the model for these indices."""
    from torchvision.transforms import functional as tvf

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root)
    out = []
    for i in indices:
        image = ds[int(i)][camera]
        out.append(
            tvf.resize(image, list(resize_to), interpolation=tvf.InterpolationMode.BILINEAR, antialias=True)
        )
    return torch.stack(out)


def test_the_two_data_paths_agree_on_the_pixels_they_hand_the_model(tmp_path, lerobot_dataset_factory):
    """The GPU path must produce what the CPU path produces, not merely something.

    Decode identity is already pinned above, but that is one stage. This runs
    the whole preparation -- fetch, float conversion, resize -- and compares it
    against what `FlowMatchingDataset` hands the model on the CPU path for the
    same indices.

    On the CPU device both paths decode through torchcodec, so the comparison is
    exact and any difference is the pipeline's own arithmetic: a wrong dtype
    scale, a resize on the wrong axis order, a frame served against the wrong
    index. That is the failure this is here to catch, and it needs no GPU.
    """
    built = lerobot_dataset_factory(
        root=tmp_path / "equivalence",
        repo_id=DUMMY_REPO_ID,
        total_episodes=2,
        total_frames=40,
        use_videos=True,
    )
    camera = next(iter(built.meta.video_keys), None)
    if camera is None:
        pytest.skip("fixture produced no video keys")
    resize_to = (32, 32)
    indices = np.array([0, 7, 3, 19, 11], dtype=np.int64)

    pipeline = GpuImagePipeline(built, [camera], resize_to=resize_to, device="cpu")
    got = pipeline.prepare({"index": torch.from_numpy(indices)})[camera]

    want = _cpu_path_images(built.root, DUMMY_REPO_ID, camera, indices, resize_to)

    assert got.shape == want.shape, f"{got.shape} != {want.shape}"
    assert got.dtype == want.dtype == torch.float32
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the GPU path needs CUDA")
def test_the_prefetcher_serves_every_batch_intact(tmp_path, lerobot_dataset_factory):
    """Preparation runs on another thread; what it hands over must still be right.

    The consumer reads tensors a second stream wrote. Without the event wait the
    reads race the writes, and without `record_stream` the allocator can recycle
    a buffer into the next preparation while this one is still being read.
    Neither failure raises -- both surface as wrong pixels -- so the check is
    that every delivered batch still equals what the pipeline computes for the
    indices that batch carries.
    """
    pytest.importorskip("PyNvVideoCodec")
    from torch.utils.data import DataLoader, Dataset

    from lerobot.datasets.gpu_data_pipeline import GpuBatchPrefetcher

    built = lerobot_dataset_factory(
        root=tmp_path / "prefetch",
        repo_id=DUMMY_REPO_ID,
        total_episodes=2,
        total_frames=40,
        use_videos=True,
        camera_features={
            "front": {
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
                "info": DUMMY_VIDEO_INFO,
            }
        },
    )
    camera = next(iter(built.meta.video_keys))

    class _Indices(Dataset):
        def __len__(self):
            return built.num_frames

        def __getitem__(self, i):
            return {"index": i}

    loader = DataLoader(_Indices(), batch_size=4, shuffle=False, num_workers=0)
    pipeline = GpuImagePipeline(built, [camera], resize_to=(64, 64), device="cuda")
    prefetcher = GpuBatchPrefetcher(loader, pipeline, "cuda", depth=2)

    reference = GpuImagePipeline(built, [camera], resize_to=(64, 64), device="cuda")
    for _ in range(6):
        batch = next(prefetcher)
        want = reference.prepare({"index": batch["index"]})[camera]
        torch.testing.assert_close(batch[camera], want, rtol=0, atol=0)
    prefetcher.close()


def test_timed_preparation_reports_every_phase(tmp_path, lerobot_dataset_factory):
    """`timed=True` is a separate code path, and it only runs in real training.

    The per-phase timings are sampled on every Nth batch, so nothing in the test
    suite reached them until this: a name error in the timed branch would have
    surfaced only after fifty steps of a real run.
    """
    built = lerobot_dataset_factory(
        root=tmp_path / "timed",
        repo_id=DUMMY_REPO_ID,
        total_episodes=2,
        total_frames=40,
        use_videos=True,
    )
    camera = next(iter(built.meta.video_keys), None)
    if camera is None:
        pytest.skip("fixture produced no video keys")

    pipeline = GpuImagePipeline(built, [camera], resize_to=(32, 32), device="cpu")
    assert pipeline.report() == {}, "nothing timed yet, so there is nothing to report"

    pipeline.prepare({"index": torch.arange(4)}, timed=True)

    report = pipeline.report()
    assert report["gpu_prep_steps_timed"] == 1.0
    for phase in pipeline.PHASES:
        assert f"gpu_prep_{phase}_ms" in report, f"{phase} missing from {sorted(report)}"
        assert report[f"gpu_prep_{phase}_ms"] >= 0.0


def test_chunked_conversion_is_bit_identical_to_converting_whole(tmp_path, lerobot_dataset_factory):
    """Converting in chunks must change memory, never pixels.

    The batch is promoted to float32 at native resolution before the resize
    shrinks it, so the conversion runs in chunks to bound that transient --
    measured at four 1280x720 cameras, batch 128, 4272 MiB against 2408 MiB for
    294 MiB of actual model input.

    The identity is asserted exactly rather than closely. Chunking's own failure
    modes are batch-axis mistakes -- a chunk written to the wrong slice, a short
    final chunk dropped, an off-by-one stride -- and every one of them survives a
    tolerance while corrupting the images the model trains on.
    """
    built = lerobot_dataset_factory(
        root=tmp_path / "chunking",
        repo_id=DUMMY_REPO_ID,
        total_episodes=2,
        total_frames=40,
        use_videos=True,
    )
    camera = next(iter(built.meta.video_keys), None)
    if camera is None:
        pytest.skip("fixture produced no video keys")
    resize_to = (32, 32)
    indices = np.array([0, 7, 3, 19, 11, 2, 14], dtype=np.int64)
    batch = {"index": torch.from_numpy(indices)}

    whole = GpuImagePipeline(built, [camera], resize_to=resize_to, device="cpu")
    whole.CONVERT_TRANSIENT_BYTES = 1 << 40  # the batch fits: one chunk
    want = whole.prepare(batch)[camera]

    chunked = GpuImagePipeline(built, [camera], resize_to=resize_to, device="cpu")
    chunked.CONVERT_TRANSIENT_BYTES = 1  # one frame per chunk: every seam exercised
    got = chunked.prepare(batch)[camera]

    assert want.shape == (len(indices), 3, *resize_to), want.shape
    assert got.shape == want.shape, f"{got.shape} != {want.shape}"
    assert torch.equal(got, want), (got - want).abs().max().item()


@pytest.mark.parametrize(
    ("codec", "why"),
    [("hevc", "segfaults on the final frame"), (None, "no codec recorded"), ("vp9", "never verified")],
)
def test_a_codec_the_decoder_cannot_survive_is_refused_before_it_is_opened(
    tmp_path, lerobot_dataset_factory, codec, why
):
    """The refusal has to happen before a decoder exists, not after it crashes.

    PyNvVideoCodec 2.2.2 segfaults producing the last frame of an HEVC file, and
    the process cannot catch that -- so the usual shape here, probe the dataset
    and fall back when the probe raises, cannot work. The codec is read from
    metadata and rejected before any decoder is constructed.

    A missing codec and an unverified one are refused on the same footing: this
    decoder returns wrong pixels rather than erroring on some inputs, so an
    unrecognised codec is not a safe default.
    """
    built = lerobot_dataset_factory(
        root=tmp_path / f"codec-{codec}",
        repo_id=DUMMY_REPO_ID,
        total_episodes=1,
        total_frames=12,
        use_videos=True,
    )
    camera = next(iter(built.meta.video_keys), None)
    if camera is None:
        pytest.skip("fixture produced no video keys")

    info = built.meta.features[camera].setdefault("info", {})
    if codec is None:
        info.pop("video.codec", None)
    else:
        info["video.codec"] = codec

    with pytest.raises(NotImplementedError, match="GPU data path decodes") as raised:
        GpuImagePipeline(built, [camera], resize_to=(32, 32), device="cpu")
    assert camera in str(raised.value), f"{why}: the message must name the camera"


def test_a_verified_codec_is_accepted(tmp_path, lerobot_dataset_factory):
    """The guard must not refuse everything -- that would pass the test above vacuously."""
    built = lerobot_dataset_factory(
        root=tmp_path / "codec-ok",
        repo_id=DUMMY_REPO_ID,
        total_episodes=1,
        total_frames=12,
        use_videos=True,
    )
    camera = next(iter(built.meta.video_keys), None)
    if camera is None:
        pytest.skip("fixture produced no video keys")
    built.meta.features[camera].setdefault("info", {})["video.codec"] = "h264"
    GpuImagePipeline(built, [camera], resize_to=(32, 32), device="cpu")
