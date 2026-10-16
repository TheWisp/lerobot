#!/usr/bin/env python

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

from unittest.mock import MagicMock

import pytest

from lerobot.robots.bi_openarm_follower.bi_openarm_follower import BiOpenArmFollower


def test_right_connect_failure_rolls_back_left_arm():
    robot = object.__new__(BiOpenArmFollower)
    robot.left_arm = MagicMock(is_connected=False)
    robot.right_arm = MagicMock(is_connected=False)
    robot.right_arm.connect.side_effect = RuntimeError("right connect")

    with pytest.raises(RuntimeError, match="right connect"):
        robot.connect()

    robot.left_arm.connect.assert_called_once_with(False)
    robot.right_arm.connect.assert_called_once_with(False)
    robot.left_arm.disconnect.assert_called_once_with()


def test_camera_feature_names_follow_what_get_observation_publishes():
    """Top-level cameras keep their key; a camera an arm owns takes that arm's
    prefix, which is the name its features carry in the dataset."""
    robot = object.__new__(BiOpenArmFollower)
    robot._top_level_cam_keys = {"top", "top_l", "top_r", "top_depth"}
    robot.left_arm = MagicMock(cameras={"top": object(), "wrist": object()})
    robot.right_arm = MagicMock(cameras={"wrist": object()})
    # NB: `cameras` merges both dicts, so a shared per-arm name collapses there; the
    # map is keyed the same way and the right arm's entry wins, as the merge does.
    assert robot.camera_feature_names == {"top": "top", "wrist": "right_wrist"}
