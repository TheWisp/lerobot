# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""HVLA consumes exactly the cameras it was given, and every camera when given none.

HVLA does not go through ``lerobot-train``: it has its own argparse entrypoint and
its own dataset class, so nothing in the library-side coverage says anything about
what it actually trains on. Both halves are checked here — the input contract
written into the config (and therefore into the checkpoint), and the sample the
model is handed at each step.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
from lerobot.policies.hvla.s1.flow_matching.train import (
    FlowMatchingDataset,
    configure_from_dataset_features,
)

CAMERAS = ("top_l", "top_r", "wrist")


def _features(*cameras: str) -> dict[str, dict]:
    features: dict[str, dict] = {
        "action": {"dtype": "float32", "shape": (6,), "names": [f"j{i}" for i in range(6)]},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": [f"j{i}" for i in range(6)]},
    }
    for name in cameras:
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (3, 64, 96),
            "names": ["channels", "height", "width"],
        }
    return features


def _configure(cameras: list[str] | None) -> FlowMatchingS1Config:
    config = FlowMatchingS1Config()
    configure_from_dataset_features(config, _features(*CAMERAS), resize_to=(224, 224), cameras=cameras)
    return config


# --------------------------------------------------------------------------
# The input contract the checkpoint carries
# --------------------------------------------------------------------------


def test_no_selection_takes_every_camera():
    assert list(_configure(None).image_features) == [f"observation.images.{c}" for c in CAMERAS]


def test_a_selection_takes_exactly_those_cameras():
    config = _configure(["top_l"])
    assert list(config.image_features) == ["observation.images.top_l"]


def test_a_selection_may_name_cameras_fully_or_briefly():
    brief = _configure(["top_l", "wrist"])
    full = _configure(["observation.images.top_l", "observation.images.wrist"])
    assert list(brief.image_features) == list(full.image_features)


def test_the_order_is_the_dataset_s_not_the_argument_s():
    """Feature order decides channel order downstream; argv must not set it."""
    assert list(_configure(["wrist", "top_l"]).image_features) == [
        "observation.images.top_l",
        "observation.images.wrist",
    ]


def test_an_unknown_camera_is_refused_with_the_shared_message():
    with pytest.raises(ValueError, match="Unknown camera"):
        _configure(["labtop"])


# --------------------------------------------------------------------------
# The sample the model is handed
# --------------------------------------------------------------------------


class _HFDataset:
    column_names = ["action", "observation.state", "episode_index"]

    def __init__(self) -> None:
        self._columns = {
            "action": [[float(i)] * 6 for i in range(4)],
            "observation.state": [[float(i)] * 6 for i in range(4)],
            "episode_index": [0, 0, 0, 0],
        }

    def __getitem__(self, key: str):
        return self._columns[key]


class _LeRobotDataset:
    """Yields every camera, as the real one does — the trainer must do the narrowing."""

    def __init__(self) -> None:
        self.hf_dataset = _HFDataset()

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = {
            "action": torch.tensor(self.hf_dataset["action"][index]),
            "observation.state": torch.tensor(self.hf_dataset["observation.state"][index]),
        }
        for name in CAMERAS:
            sample[f"observation.images.{name}"] = torch.zeros(3, 64, 96)
        return sample


def _sample(image_keys: list[str] | None, resize_to=(224, 224)) -> dict:
    dataset = FlowMatchingDataset(
        _LeRobotDataset(),
        s2_latents=np.zeros((4, 2048), dtype=np.float32),
        chunk_size=2,
        resize_to=resize_to,
        image_keys=image_keys,
        # This suite is about which cameras reach the batch. The state
        # position floor needs an ordered state contract to know which
        # positions are joints, which this fake does not model.
        state_position_std_floor=0.0,
    )
    return dataset[0]


def test_an_unselected_camera_never_reaches_the_batch():
    sample = _sample(["observation.images.top_l"])
    present = sorted(k for k in sample if k.startswith("observation.images."))
    assert present == ["observation.images.top_l"]


def test_every_camera_reaches_the_batch_when_none_is_selected():
    sample = _sample(None)
    present = sorted(k for k in sample if k.startswith("observation.images."))
    assert present == sorted(f"observation.images.{c}" for c in CAMERAS)


def test_the_drop_does_not_depend_on_resizing():
    """The unselected cameras are the reason /dev/shm ran out, resize or not.

    They were once dropped inside the resize branch, so a run without
    ``--resize-images`` collated three full-size frames per sample for nothing.
    """
    sample = _sample(["observation.images.top_l"], resize_to=None)
    present = sorted(k for k in sample if k.startswith("observation.images."))
    assert present == ["observation.images.top_l"]


def test_the_selected_camera_is_still_resized():
    """Narrowing must not cost the transform the surviving camera needs."""
    sample = _sample(["observation.images.top_l"])
    assert tuple(sample["observation.images.top_l"].shape[-2:]) == (224, 224)
