"""An external RGB-D import must preserve depth validity and source timing."""

import pytest

from lerobot.policies.flash.observation_dataset import camera_feature_keys, scene_from_camera


def test_observation_camera_names_come_from_metadata():
    features = {
        "observation.images.top": {"dtype": "video", "shape": [64, 64, 3]},
        "observation.images.range": {"dtype": "video", "shape": [64, 64, 1], "info": {"is_depth_map": True}},
    }
    assert camera_feature_keys(features, {}) == ("observation.images.top", "observation.images.range")
    with pytest.raises(ValueError, match="needs a recorded camera pose"):
        scene_from_camera({}, features, {"camera_motion_assumption": "moving"})
