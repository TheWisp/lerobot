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

"""Side-by-side stereo cameras publish one channel per eye.

Stereo USB devices have no single-eye capture mode, so a ZED Mini recorded
naively stores every frame twice over and halves the horizontal resolution a
policy ever sees. The failure is silent — nothing errors, the data is just
worse — which is why the contract is pinned here rather than left to a runtime
check.

The observation-features case matters beyond the robot: ObservationStream sizes
its shared-memory blocks from those features, so a channel that is emitted but
not declared would reach no viewer.
"""

import numpy as np
import pytest

from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.stereo import EYES, split_stereo_frame, stereo_channel_keys


def _stereo_frame(width: int = 2560, height: int = 720) -> np.ndarray:
    """A BGR frame whose halves are distinguishable: left blue, right red."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, : width // 2, 0] = 255
    frame[:, width // 2 :, 2] = 255
    return frame


def _camera(**kwargs) -> OpenCVCamera:
    kwargs.setdefault("width", 1280)
    kwargs.setdefault("height", 720)
    config = OpenCVCameraConfig(index_or_path=0, fps=30, color_mode=ColorMode.BGR, **kwargs)
    return OpenCVCamera(config)


class TestPrimitive:
    def test_splits_in_left_right_order(self):
        left, right = split_stereo_frame(_stereo_frame())
        assert left.shape == right.shape == (720, 1280, 3)
        assert (left[:, :, 0] == 255).all() and (right[:, :, 2] == 255).all()

    def test_halves_do_not_view_the_source(self):
        # The live camera hands these to a reader thread; a view would pin the
        # whole stereo frame for as long as either half is held.
        for half in split_stereo_frame(_stereo_frame()):
            assert half.base is None

    def test_odd_width_is_rejected(self):
        with pytest.raises(ValueError, match="even width"):
            split_stereo_frame(np.zeros((4, 7, 3), dtype=np.uint8))

    def test_channel_names_follow_the_camera_key(self):
        assert stereo_channel_keys("top") == ("top_l", "top_r")
        assert EYES == ("l", "r")


class TestCaptureDimensions:
    """`width`/`height` describe one eye; the device opens at twice the width."""

    def test_capture_width_is_doubled(self):
        cam = _camera(stereo_split=True)
        assert (cam.width, cam.height) == (1280, 720)
        assert (cam.capture_width, cam.capture_height) == (2560, 720)

    def test_unsplit_camera_is_unaffected(self):
        cam = _camera()
        assert (cam.capture_width, cam.capture_height) == (1280, 720)
        assert cam.latest_frame_right is None

    def test_doubling_lands_on_the_pre_rotation_axis(self):
        cam = _camera(stereo_split=True, rotation=Cv2Rotation.ROTATE_90)
        assert (cam.capture_width, cam.capture_height) == (1440, 1280)


class TestCameraChannels:
    def test_primary_channel_is_the_left_eye(self):
        # Callers that know nothing about stereo must still get one coherent image.
        out = _camera(stereo_split=True)._postprocess_image(_stereo_frame())
        assert out.shape == (720, 1280, 3)
        assert (out[:, :, 0] == 255).all()

    def test_both_eyes_come_from_one_capture(self):
        left, right = _camera(stereo_split=True)._postprocess_stereo(_stereo_frame())
        assert (left[:, :, 0] == 255).all()
        assert (right[:, :, 2] == 255).all()

    def test_right_eye_unavailable_without_the_option(self):
        with pytest.raises(RuntimeError, match="not configured with stereo_split"):
            _camera().read_latest_right()

    def test_split_precedes_rotation(self):
        cam = _camera(stereo_split=True, rotation=Cv2Rotation.ROTATE_90)
        left, right = cam._postprocess_stereo(_stereo_frame(1440, 1280))
        assert left.shape == (720, 1280, 3)
        assert (left[:, :, 0] == 255).all() and (right[:, :, 2] == 255).all()

    def test_colour_conversion_applies_to_both_eyes(self):
        config = OpenCVCameraConfig(
            index_or_path=0,
            fps=30,
            width=1280,
            height=720,
            color_mode=ColorMode.RGB,
            stereo_split=True,
        )
        left, right = OpenCVCamera(config)._postprocess_stereo(_stereo_frame())
        assert (left[:, :, 2] == 255).all()  # BGR blue -> RGB last channel
        assert (right[:, :, 0] == 255).all()

    def test_mismatched_frame_is_still_rejected(self):
        with pytest.raises(RuntimeError, match="do not match configured"):
            _camera(stereo_split=True)._postprocess_stereo(_stereo_frame(1280, 720))

    def test_unsplit_camera_keeps_the_whole_frame(self):
        # The state every dataset recorded so far is in.
        out = _camera(width=2560)._postprocess_image(_stereo_frame())
        assert out.shape == (720, 2560, 3)


class TestObservationFeatures:
    """A channel that is emitted but not declared reaches no viewer."""

    def test_features_declare_both_eyes_and_not_the_whole_frame(self):
        from lerobot.robots.openarm_follower.openarm_follower import OpenArmFollower

        class _Cfg:
            height, width, stereo_split, use_rgb, use_depth = 720, 1280, True, True, False

        follower = OpenArmFollower.__new__(OpenArmFollower)
        follower.cameras = {"top": object()}
        follower.config = type("C", (), {"cameras": {"top": _Cfg()}})()

        features = follower._cameras_ft
        assert set(features) == {"top_l", "top_r"}, "the undivided key must not be published"
        assert features["top_l"] == features["top_r"] == (720, 1280, 3)

    def test_non_stereo_camera_declares_one_key(self):
        from lerobot.robots.openarm_follower.openarm_follower import OpenArmFollower

        class _Cfg:
            height, width, stereo_split, use_rgb, use_depth = 600, 960, False, True, False

        follower = OpenArmFollower.__new__(OpenArmFollower)
        follower.cameras = {"left_wrist": object()}
        follower.config = type("C", (), {"cameras": {"left_wrist": _Cfg()}})()

        assert follower._cameras_ft == {"left_wrist": (600, 960, 3)}
