# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Where a training batch's images come from.

A trainer should ask for its images once and stop knowing how they arrive. Before
this, the choice leaked into five separate places in `lerobot-train` -- the
worker count, the loader wrapper, the per-batch conversion, the log line, and the
dataset's own decoding flag -- each an `if` on the same condition, each a place
for the two paths to drift apart.

The two paths are alternatives, so they are two implementations of one interface
rather than a flag threaded through a function. That the CPU path is also an
object is the point: it gives the paths one place to be compared, and there is no
branch left in the trainer to get wrong.

Preconditions for both: the loader yields batches carrying `index`, and the
trainer calls `loader_workers` before building its DataLoader and `finish` on
every batch before the model sees it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, Protocol

import torch

logger = logging.getLogger(__name__)

# Dataloader workers on the GPU path. Not a tuning knob: with video decoding off
# a worker assembles parquet rows and an index, and one process outruns the
# training step by more than a hundredfold (1246 batches/s at batch 4 against
# 4.75 consumed; 292 at batch 64 against 2.28). Zero is not the answer either --
# it puts the loader back in the trainer's interpreter beside the producer
# thread, measured about 8% worse on the update.
GPU_PATH_WORKERS = 1


class ImageSource(Protocol):
    """How a trainer obtains the images for a batch."""

    def loader_workers(self, requested: int) -> int:
        """Worker count this source needs, given what the config asked for."""
        ...

    def iterate(self, loader) -> Iterator[dict[str, Any]]:
        """Endless batches from ``loader``, prepared as this source prepares them."""
        ...

    def finish(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Whatever remains to be done to a batch before the model sees it."""
        ...

    def telemetry(self) -> str:
        """A fragment for the training log naming this source and its costs."""
        ...


class LoaderImageSource:
    """Images decoded by the DataLoader's workers -- the path that always existed.

    Nothing here is new behaviour. It is the trainer's previous inline code,
    moved so that the choice of path has one shape rather than two.
    """

    def __init__(self, camera_keys: list[str]):
        self.camera_keys = list(camera_keys)

    def loader_workers(self, requested: int) -> int:
        return requested

    def iterate(self, loader) -> Iterator[dict[str, Any]]:
        while True:
            yield from loader

    def finish(self, batch: dict[str, Any]) -> dict[str, Any]:
        # The loader hands over uint8 to keep the worker-to-trainer copy small;
        # the scale to [0, 1] happens here, where the GPU path's own conversion
        # has already happened on the device.
        for key in self.camera_keys:
            if key in batch and batch[key].dtype == torch.uint8:
                batch[key] = batch[key].to(dtype=torch.float32) / 255.0
        # Postcondition: the model is handed floats. A camera the key list does
        # not cover would otherwise arrive as uint8 and train on values in
        # [0, 255], which no downstream check would notice.
        assert not any(v.dtype == torch.uint8 for v in batch.values() if isinstance(v, torch.Tensor)), (
            "a uint8 image survived the loader path's conversion"
        )
        return batch

    def telemetry(self) -> str:
        return " | data:cpu"


class DeviceImageSource:
    """Images decoded on NVDEC and prepared on the device.

    The batches this yields already carry float32 images, so ``finish`` has
    nothing to do -- the work happened on the producer thread while the model
    was busy with the previous batch.
    """

    def __init__(self, pipeline, device: str):
        self.pipeline = pipeline
        self.device = device

    def loader_workers(self, requested: int) -> int:
        assert requested >= 0, f"worker count cannot be negative, got {requested}"
        if requested != GPU_PATH_WORKERS:
            logger.info(
                "Data path: GPU — using %d dataloader worker instead of %d; "
                "workers no longer decode video on this path.",
                GPU_PATH_WORKERS,
                requested,
            )
        return GPU_PATH_WORKERS

    def iterate(self, loader) -> Iterator[dict[str, Any]]:
        assert loader is not None, "the device path needs a loader to draw indices from"
        from lerobot.datasets.gpu_data_pipeline import GpuBatchPrefetcher  # noqa: PLC0415

        # The prefetcher restarts the source itself, so it is not wrapped in a
        # cycle: doing both would restart an iterator that had already restarted.
        return iter(GpuBatchPrefetcher(loader, self.pipeline, device=self.device))

    def finish(self, batch: dict[str, Any]) -> dict[str, Any]:
        return batch

    def telemetry(self) -> str:
        # The report's keys carry their own units. The allocation figure is
        # dropped: `memory_allocated` is process-wide and this runs while the
        # model allocates on another thread, so the delta is the model's as much
        # as the pipeline's -- it read 21 GB for a pipeline holding about 200 MB.
        phases = self.pipeline.report()
        parts = " ".join(f"{k}:{v:.1f}" for k, v in sorted(phases.items()) if not k.endswith("_mb"))
        return f" | data:gpu {parts}".rstrip()


def _loader_only_features(dataset, camera_keys: list[str]) -> str | None:
    """Why this run's images must come from the loader, or None if they need not.

    Both cases below are things the loader does to camera frames on the way out
    that the device path does not do at all. Neither would announce itself: one
    raises deep in the reader, and the other changes a tensor's rank in a way the
    policy accepts and trains wrongly on.
    """
    if getattr(dataset, "image_transforms", None) is not None:
        # The reader applies transforms to every camera key after decoding. With
        # decoding off those keys are absent and it raises KeyError -- and were it
        # guarded, the augmentation would simply be skipped, which is worse.
        return "image transforms run in the dataloader and are not implemented on the GPU path"

    delta = getattr(getattr(dataset, "reader", None), "delta_indices", None) or {}
    stacked = sorted(k for k in camera_keys if len(delta.get(k, [0])) > 1)
    if stacked:
        # The loader stacks a frame history into (B, T, C, H, W); the device path
        # fetches one frame per camera and returns (B, C, H, W). Seven policies
        # ask for this, diffusion among them.
        return f"these cameras need a frame history the GPU path does not stack: {', '.join(stacked)}"
    return None


def resolve_image_source(
    choice: str,
    dataset,
    camera_keys: list[str],
    resize_to: tuple[int, int] | None,
    device: str,
) -> ImageSource:
    """The image source for this run. Never None: the CPU path is a source too.

    Turning the dataset's own decoding off belongs here rather than in the
    trainer, and must happen before any worker forks -- workers copy the flag,
    so a later change reaches the trainer and not them.
    """
    from lerobot.datasets.gpu_data_pipeline import resolve_gpu_pipeline  # noqa: PLC0415

    # Validated here, not left to the resolver below: the refusal that follows
    # returns before that runs, so a typo reached the loader path as if it had
    # been asked for -- but only on datasets that trip the refusal, which is the
    # worst way for a validation to be wrong.
    assert choice in ("auto", "cpu", "gpu"), f"unknown data path {choice!r}; expected auto, cpu or gpu"

    unsupported = _loader_only_features(dataset, camera_keys)
    if unsupported:
        if choice == "gpu":
            raise NotImplementedError(f"the GPU data path cannot serve this run: {unsupported}")
        logger.warning("Data path: CPU (GPU path unavailable - %s)", unsupported)
        return LoaderImageSource(camera_keys)

    pipeline = resolve_gpu_pipeline(choice, dataset, camera_keys, resize_to, device)
    if pipeline is None:
        return LoaderImageSource(camera_keys)
    # The handoff, and it is one decision with two halves: from here the GPU path
    # owns the frames AND the compositing, so the reader must stop doing both.
    # They were coupled only by convention across three modules, and the first
    # real run on this path died with `KeyError: masks.<camera>` because the
    # reader kept stripping rows the pipeline needed. Checked here, where the
    # handoff happens, rather than discovered in a dataloader worker.
    dataset.set_video_decoding(False)
    if pipeline.composites:
        assert dataset.delivers_mask_rows, (
            "the GPU path composites saved masks but the dataset still strips the "
            "mask rows; the pipeline would get KeyError on its first batch"
        )
    return DeviceImageSource(pipeline, device)
