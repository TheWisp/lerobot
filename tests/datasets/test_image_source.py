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
"""The two image sources, and the trainer's ignorance of which one it has.

Where a batch's images come from used to be five `if`s spread through
`lerobot-train`. Each was a place the paths could drift, and the drift would not
show up as a failure -- it would show up as one path quietly behaving unlike the
other. These pin the contract both sources owe, and that the trainer still
delegates rather than deciding.
"""

import inspect

import pytest
import torch

from lerobot.datasets.image_source import (
    GPU_PATH_WORKERS,
    DeviceImageSource,
    LoaderImageSource,
)


class _FakePipeline:
    def __init__(self, phases=None):
        self.phases = phases or {"gpu_prep_decode_ms": 12.0, "gpu_prep_alloc_mb": 21247.0}

    def report(self):
        return self.phases


def test_both_sources_offer_the_same_surface():
    """A method on one and not the other is a branch waiting to reappear."""
    loader = LoaderImageSource(["cam"])
    device = DeviceImageSource(_FakePipeline(), "cuda")
    for name in ("loader_workers", "iterate", "finish", "telemetry"):
        assert callable(getattr(loader, name)), f"LoaderImageSource lacks {name}"
        assert callable(getattr(device, name)), f"DeviceImageSource lacks {name}"
        assert inspect.signature(getattr(loader, name)) == inspect.signature(getattr(device, name)), (
            f"{name} differs between the sources"
        )


def test_the_loader_source_scales_uint8_and_leaves_float_alone():
    src = LoaderImageSource(["cam", "other"])
    batch = {
        "cam": torch.full((2, 3, 4, 4), 255, dtype=torch.uint8),
        "other": torch.full((2, 3, 4, 4), 0.5),
        "index": torch.arange(2),
    }
    out = src.finish(batch)
    assert out["cam"].dtype == torch.float32
    assert torch.equal(out["cam"], torch.ones_like(out["cam"])), "255 must scale to 1.0"
    assert out["other"].dtype == torch.float32
    assert torch.equal(out["other"], torch.full((2, 3, 4, 4), 0.5)), "float images must pass through"


def test_the_device_source_does_not_touch_a_batch_it_already_prepared():
    """Converting twice would halve every pixel, and nothing would raise."""
    src = DeviceImageSource(_FakePipeline(), "cuda")
    images = torch.rand(2, 3, 4, 4)
    out = src.finish({"cam": images.clone(), "index": torch.arange(2)})
    assert torch.equal(out["cam"], images)


def test_only_the_device_source_pins_the_worker_count():
    assert LoaderImageSource(["cam"]).loader_workers(8) == 8, "the loader path must keep its count"
    assert DeviceImageSource(_FakePipeline(), "cuda").loader_workers(8) == GPU_PATH_WORKERS


def test_the_device_source_never_reports_the_allocation_figure():
    """It is process-wide and measured while the model allocates: it read 21 GB
    for a pipeline holding about 200 MB. Reporting it is worse than silence."""
    line = DeviceImageSource(_FakePipeline(), "cuda").telemetry()
    assert "data:gpu" in line
    assert "gpu_prep_decode_ms" in line
    assert "alloc_mb" not in line, "an allocation delta must not reach the log"


def test_each_source_names_itself_in_the_log():
    """A run that fell back must not read like one that did not."""
    assert LoaderImageSource(["cam"]).telemetry().strip() == "| data:cpu"
    assert "data:gpu" in DeviceImageSource(_FakePipeline(), "cuda").telemetry()


def test_the_loader_source_iterates_endlessly():
    """The trainer runs to a step count, not to the end of an epoch."""
    src = LoaderImageSource(["cam"])
    it = src.iterate([{"index": torch.tensor([0])}, {"index": torch.tensor([1])}])
    seen = [next(it)["index"].item() for _ in range(5)]
    assert seen == [0, 1, 0, 1, 0], "the source must wrap rather than stop"


def test_the_trainer_delegates_rather_than_branching():
    """The point of the seam: no `if` in the trainer asks which path this is.

    This is the regression that matters. Re-adding one `if gpu_pipeline is None`
    is easy, reviews as harmless, and puts the drift back.
    """
    from lerobot.scripts import lerobot_train

    src = inspect.getsource(lerobot_train.train)
    for forbidden in ("gpu_pipeline", "GpuBatchPrefetcher", "GPU_PATH_WORKERS"):
        assert forbidden not in src, f"{forbidden} is back in the trainer; it belongs behind the source"
    assert src.count("image_source.") >= 4, "the trainer should delegate through the source"


# ── What the loader does that the device path does not ──────────────────────
#
# `auto` is the default, so every run on a CUDA host now reaches this decision.
# Both cases below would otherwise change what an existing, working run does:
# one by raising deep in the reader, the other by silently altering a tensor's
# rank. Refusing is the compatible answer; falling back costs nothing.


class _FakeReader:
    def __init__(self, delta_indices=None):
        self.delta_indices = delta_indices


class _FakeDataset:
    def __init__(self, image_transforms=None, delta_indices=None):
        self.image_transforms = image_transforms
        self.reader = _FakeReader(delta_indices)

    def set_video_decoding(self, enabled):  # pragma: no cover - must not be reached
        raise AssertionError("the dataset must not be reconfigured when the path is refused")


def test_a_run_with_image_transforms_keeps_the_loader():
    """The reader applies transforms to camera keys that decoding-off removes.

    Unguarded, so it raises KeyError rather than skipping -- and skipping would
    be worse, because the run would train without the augmentation it asked for.
    """
    from lerobot.datasets.image_source import resolve_image_source

    src = resolve_image_source(
        "auto", _FakeDataset(image_transforms=object()), ["observation.images.cam"], None, "cuda"
    )
    assert isinstance(src, LoaderImageSource)


def test_a_run_needing_a_frame_history_is_refused_the_device_path():
    """The loader stacks (B, T, C, H, W); the device path returns (B, C, H, W).

    diffusion, tdmpc, vqbet and four others ask for this. Nothing would raise:
    the policy would receive a tensor of the wrong rank and train on it.

    Asserted against `_loader_only_features` rather than the resolved source.
    Going through `resolve_image_source` with a stub dataset passes whether or
    not this check exists -- the GPU probe fails on the stub and `auto` falls
    back anyway -- so the first version of this test survived deleting the very
    branch it was written for.
    """
    from lerobot.datasets.image_source import _loader_only_features

    ds = _FakeDataset(delta_indices={"observation.images.cam": [-1, 0]})
    reason = _loader_only_features(ds, ["observation.images.cam"])
    assert reason is not None, "a frame history must send the run to the loader"
    assert "observation.images.cam" in reason, "the reason must name the camera"


def test_transforms_are_refused_at_the_same_level():
    from lerobot.datasets.image_source import _loader_only_features

    reason = _loader_only_features(_FakeDataset(image_transforms=object()), ["observation.images.cam"])
    assert reason is not None and "transform" in reason


def test_a_single_step_delta_is_not_mistaken_for_a_history():
    """[0] is one frame -- refusing it would send every ordinary run to the CPU."""
    from lerobot.datasets.image_source import _loader_only_features

    ds = _FakeDataset(delta_indices={"observation.images.cam": [0]})
    assert _loader_only_features(ds, ["observation.images.cam"]) is None
    assert _loader_only_features(_FakeDataset(), ["observation.images.cam"]) is None


def test_an_explicit_gpu_request_that_cannot_be_served_stops_the_run():
    """Silently honouring the loader here is how a benchmark measures the wrong path."""
    from lerobot.datasets.image_source import resolve_image_source

    with pytest.raises(NotImplementedError, match="cannot serve this run"):
        resolve_image_source(
            "gpu", _FakeDataset(image_transforms=object()), ["observation.images.cam"], None, "cuda"
        )


def test_an_unknown_path_is_rejected_even_when_the_run_needs_the_loader():
    """Validation must not depend on which branch the run happens to take.

    The refusal above returns before the resolver validates `choice`, so a typo
    was accepted as the CPU path -- but only on datasets that trip the refusal.
    A validation that is right for most inputs and silent for the rest is the
    worst shape available.
    """
    from lerobot.datasets.image_source import resolve_image_source

    for dataset in (
        _FakeDataset(image_transforms=object()),  # trips the refusal
        _FakeDataset(),  # does not
    ):
        with pytest.raises(AssertionError, match="unknown data path"):
            resolve_image_source("nvdec", dataset, ["observation.images.cam"], None, "cuda")


def test_the_loader_path_refuses_to_hand_over_a_uint8_image():
    """A camera the key list does not cover would train on values in [0, 255]."""
    src = LoaderImageSource(["cam"])  # 'other' deliberately absent from the list
    with pytest.raises(AssertionError, match="uint8 image survived"):
        src.finish(
            {
                "cam": torch.zeros(1, 3, 2, 2, dtype=torch.uint8),
                "other": torch.zeros(1, 3, 2, 2, dtype=torch.uint8),
            }
        )
