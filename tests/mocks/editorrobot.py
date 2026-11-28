"""The real virtual motor driver plus synthetic images, for editor I/O tests."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.intrinsics import CameraIntrinsics
from lerobot.robots.config import RobotConfig
from lerobot.robots.virtual_bi_so107 import VirtualBiSO107Follower, VirtualBiSO107FollowerConfig


@RobotConfig.register_subclass("editor_test_robot")
@dataclass
class EditorRobotConfig(VirtualBiSO107FollowerConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


class EditorRobot(VirtualBiSO107Follower):
    config_class = EditorRobotConfig
    name = "editor_test_robot"

    def __init__(self, config):
        super().__init__(config)
        self.cameras = {
            name: StereoCamera(cam) if getattr(cam, "enable_infrared", False) else SimpleNamespace(config=cam)
            for name, cam in config.cameras.items()
        }

    @property
    def observation_features(self):
        return {**self._motors_ft, "scene": (64, 96, 3)}

    def get_observation(self):
        return {**super().get_observation(), "scene": np.full((64, 96, 3), 90, dtype=np.uint8)}


class StereoCamera:
    infrared_shape = (64, 96, 1)
    intrinsics = CameraIntrinsics(96, 64, 90, 90, 48, 32)
    depth_processing = {"spatial": False, "temporal": False, "hole_filling": False}

    def __init__(self, config):
        self.config = config
        self.sequence = 0

    def read_latest_stereo(self):
        self.sequence += 1
        value = self.sequence % 255
        return (
            np.full((64, 96, 3), value, np.uint8),
            np.full((64, 96, 1), 400 + value, np.uint16),
            np.full((64, 96), value, np.uint8),
            np.full((64, 96), value + 1, np.uint8),
            1000.123456789 + self.sequence * 0.103,
            self.sequence,
        )

    def stereo_calibration(self):
        return {
            "left_intrinsics": self.intrinsics.to_dict(),
            "right_intrinsics": self.intrinsics.to_dict(),
            "color_from_left": np.eye(4).tolist(),
            "baseline_m": 0.05,
        }


def calibrated_profile(tmp_path, identity="editor-test", *, stereo=False):
    import json

    geometry = tmp_path / f"{identity}.json"
    geometry.write_text(
        json.dumps(
            {
                "robot_id": identity,
                "cameras": {
                    "scene": {
                        "identity": {"type": "intelrealsense", "serial_number_or_name": "test"}
                        if stereo
                        else {"type": "opencv", "index_or_path": 0},
                        "T_base_cam": np.eye(4).tolist(),
                    }
                },
            }
        )
    )
    return {
        "type": "editor_test_robot",
        "fields": {"id": identity, "geometry_path": str(geometry)},
        "cameras": {
            "scene": {
                **(
                    {"type": "intelrealsense", "serial_number_or_name": "test"}
                    if stereo
                    else {"type": "opencv", "index_or_path": 0}
                ),
                "width": 96,
                "height": 64,
                "fps": 10,
            }
        },
    }
