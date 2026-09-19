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
"""Letting something else decode the frames.

The GPU data path fetches a batch's frames by index, on device. For that to be
worth anything the DataLoader must not decode them as well -- otherwise every
frame is decoded twice and the run is slower than the path it replaced. These
pin the seam that turns loader-side decoding off, and the contract of the
resolver that decides whether to use it.
"""

import pytest
import torch

pytest.importorskip("torchcodec")

from lerobot.datasets.gpu_data_pipeline import resolve_gpu_pipeline  # noqa: E402
from tests.fixtures.constants import DUMMY_REPO_ID  # noqa: E402


def _cameras(dataset, item):
    """Camera keys present in an item, named by the dataset rather than guessed.

    The fixtures call their cameras `laptop` and `phone`, so a prefix match on
    `observation.images` finds nothing and the assertion passes for the wrong
    reason.
    """
    return sorted(k for k in dataset.meta.video_keys if k in item)


def test_decoding_off_drops_the_frames_but_keeps_the_index(tmp_path, lerobot_dataset_factory):
    """The index is what a external decoder fetches against, so it must survive."""
    built = lerobot_dataset_factory(
        root=tmp_path / "external",
        repo_id=DUMMY_REPO_ID,
        total_episodes=1,
        total_frames=12,
        use_videos=True,
    )
    if not built.meta.video_keys:
        pytest.skip("fixture produced no video keys")

    assert _cameras(built, built[0]), "the fixture must decode frames by default"

    built.set_video_decoding(False)
    item = built[0]
    assert _cameras(built, item) == [], "no camera should be decoded once decoding is off"
    assert "index" in item, "the global index must survive; it is what replaces the pixels"
    assert isinstance(item["index"], torch.Tensor)


def test_decoding_can_be_turned_back_on(tmp_path, lerobot_dataset_factory):
    """A one-way switch would make the flag unusable from a fixture or a notebook."""
    built = lerobot_dataset_factory(
        root=tmp_path / "external-back",
        repo_id=DUMMY_REPO_ID,
        total_episodes=1,
        total_frames=12,
        use_videos=True,
    )
    if not built.meta.video_keys:
        pytest.skip("fixture produced no video keys")
    before = _cameras(built, built[0])
    built.set_video_decoding(False)
    built.set_video_decoding(True)
    assert _cameras(built, built[0]) == before


def test_cpu_is_honoured_without_building_anything():
    assert resolve_gpu_pipeline("cpu", None, ["cam"], None, "cuda") is None


def test_auto_falls_back_when_the_path_cannot_be_had(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        assert resolve_gpu_pipeline("auto", None, ["cam"], None, "cpu") is None
    assert "not CUDA" in caplog.text, "the reason belongs in the log, not just the outcome"


def test_an_explicit_gpu_request_that_cannot_be_met_raises():
    with pytest.raises(NotImplementedError, match="not CUDA"):
        resolve_gpu_pipeline("gpu", None, ["cam"], None, "cpu")


def test_a_dataset_with_no_cameras_is_not_a_gpu_path_candidate():
    with pytest.raises(NotImplementedError, match="no camera features"):
        resolve_gpu_pipeline("gpu", None, [], None, "cuda")


def test_an_unknown_choice_is_rejected_rather_than_defaulted():
    with pytest.raises(ValueError, match="unknown data path"):
        resolve_gpu_pipeline("nvdec", None, ["cam"], None, "cuda")


def test_the_gpu_path_pins_the_loader_to_one_worker():
    """Not a tuning knob, so it is asserted rather than left to a default.

    The constant lives beside the source that applies it; the sources' own
    behaviour is covered in test_image_source.py. What this pins is the value.
    """
    from lerobot.datasets.image_source import GPU_PATH_WORKERS

    assert GPU_PATH_WORKERS == 1


def test_the_worker_count_reaches_the_loader_rather_than_the_config():
    """The pinned count must be what the DataLoader is built with.

    Reading `cfg.num_workers` at the DataLoader would silently ignore the pin,
    and nothing else in the run would notice: the extra workers would simply
    idle.
    """
    import inspect

    from lerobot.scripts import lerobot_train

    src = inspect.getsource(lerobot_train.train)
    loader = src[src.index("torch.utils.data.DataLoader(") :]
    loader = loader[: loader.index(")\n")]
    assert "num_workers=loader_workers" in loader, "the DataLoader must use the resolved count"
    assert "num_workers=cfg.num_workers" not in loader
