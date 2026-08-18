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

"""A training run may use a subset of the dataset's cameras.

Without this, every visual feature in the dataset became a model input, so
comparing "with this camera" against "without" meant building a second dataset.
A stereo camera contributing two near-identical eyes makes that the ordinary
case.

The selection lands in ``config.image_features``, which is what the checkpoint
stores and what inference reads to decide which robot cameras to request — so
these assertions are also about what a trained model will ask the robot for.
"""

import pytest

from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config
from lerobot.policies.hvla.s1.flow_matching.train import configure_from_dataset_features

CAMERAS = ("top_l", "top_r", "left_wrist", "right_wrist")


def _features() -> dict:
    features = {
        "action": {"shape": (16,), "dtype": "float32", "names": [f"j{i}.pos" for i in range(16)]},
        "observation.state": {
            "shape": (48,),
            "dtype": "float32",
            "names": [f"s{i}" for i in range(48)],
        },
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "shape": (720, 1280, 3),
            "dtype": "video",
            "names": ["height", "width", "channels"],
        }
    return features


def _configure(cameras=None) -> FlowMatchingS1Config:
    config = FlowMatchingS1Config()
    configure_from_dataset_features(config, _features(), resize_to=(224, 224), cameras=cameras)
    return config


def _selected(config) -> set[str]:
    return {k.removeprefix("observation.images.") for k in config.image_features}


def test_default_uses_every_camera():
    assert _selected(_configure()) == set(CAMERAS)


def test_selects_a_single_eye():
    # The reason this exists: one processed dataset serves left-only and
    # both-eyes runs without being rebuilt.
    assert _selected(_configure(["top_l"])) == {"top_l"}


def test_selects_a_subset():
    assert _selected(_configure(["top_l", "left_wrist"])) == {"top_l", "left_wrist"}


def test_accepts_fully_qualified_names():
    assert _selected(_configure(["observation.images.top_r"])) == {"top_r"}


def test_selection_order_follows_the_dataset_not_the_flag():
    # image_features ordering is the model's input ordering, so it must come
    # from the dataset rather than from how the operator typed the flag.
    a = list(_configure(["right_wrist", "top_l"]).image_features)
    b = list(_configure(["top_l", "right_wrist"]).image_features)
    assert a == b


def test_unknown_camera_is_rejected():
    # A typo would otherwise silently train on every camera instead.
    with pytest.raises(ValueError, match="does not have"):
        _configure(["top_left"])


def test_empty_selection_is_rejected():
    with pytest.raises(ValueError, match="no cameras"):
        _configure([])


def test_resize_still_applies_to_the_selection():
    config = _configure(["top_l"])
    assert config.image_resize_shape == (224, 224)
    assert config.image_features["observation.images.top_l"] == 224
