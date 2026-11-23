"""Measured geometry must stay bound to its robot/camera and survive recording."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.datasets.camera_calibration import attach_robot_geometry
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.robots.geometry import RobotGeometry
from lerobot.robots.virtual_bi_so107 import VirtualBiSO107Follower, VirtualBiSO107FollowerConfig


@pytest.fixture
def measurements():
    return {
        "robot_id": "measured-arm",
        "tool_point": {"offset_m": [0, 0, 0.02]},
        "joint_zero_deg": {"elbow_flex": 1.2},
        "cameras": {
            "top": {
                "identity": {"type": "intelrealsense", "serial_number_or_name": "camera-a"},
                "T_base_cam": [[0, -1, 0, 0.1], [1, 0, 0, 0.2], [0, 0, 1, 0.3], [0, 0, 0, 1]],
            }
        },
    }


def test_robot_owns_geometry_separately_from_motor_calibration(tmp_path, measurements):
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(measurements))
    robot = VirtualBiSO107Follower(
        VirtualBiSO107FollowerConfig(
            id="measured-arm", geometry_path=path, calibration_dir=tmp_path / "motors"
        )
    )
    assert robot.calibration == {}
    assert not robot.is_connected
    config = measurements["cameras"]["top"]["identity"]
    pose = robot.geometry.camera_pose("top", config)
    # Check direction and units through the API an execution consumer uses.
    np.testing.assert_allclose(pose @ [0.01, 0.02, 0.03, 1], [0.08, 0.21, 0.33, 1])
    path.unlink()
    robot.geometry.to_dict()["cameras"].clear()
    np.testing.assert_array_equal(robot.geometry.camera_pose("top", config), pose)


def test_ordinary_robot_still_works_without_camera_geometry(tmp_path):
    robot = VirtualBiSO107Follower(VirtualBiSO107FollowerConfig(calibration_dir=tmp_path))
    assert robot.geometry is None
    robot.connect()
    try:
        robot.send_action({"left_elbow_flex.pos": 10.0})
        assert robot.get_observation()["left_elbow_flex.pos"] == 10.0
    finally:
        robot.disconnect()


@pytest.mark.parametrize("value", [None, "other-arm"])
def test_geometry_cannot_be_reused_for_a_different_robot(measurements, value):
    with pytest.raises(ValueError, match="different robot ID"):
        RobotGeometry(measurements, robot_id=value)


@pytest.mark.parametrize(
    "bad_transform",
    [
        np.eye(3).tolist(),
        np.diag([1, 1, 1, 0]).tolist(),
        np.diag([1, 1, -1, 1]).tolist(),
        np.diag([2, 1, 1, 1]).tolist(),
        np.diag([float("nan"), 1, 1, 1]).tolist(),
    ],
)
def test_invalid_rigid_transforms_are_rejected(measurements, bad_transform):
    measurements["cameras"]["top"]["T_base_cam"] = bad_transform
    with pytest.raises(ValueError, match="rigid 4x4"):
        RobotGeometry(measurements, robot_id="measured-arm")


@pytest.mark.parametrize(
    "name,changes",
    [
        ("wrist", {}),
        ("top", {"serial_number_or_name": "camera-b"}),
        ("top", {"type": "opencv"}),
        ("top", {"rotation": 90}),
    ],
)
def test_swapped_or_rotated_camera_is_rejected(measurements, name, changes):
    geometry = RobotGeometry(measurements, robot_id="measured-arm")
    config = {**measurements["cameras"]["top"]["identity"], **changes}
    with pytest.raises(ValueError):
        geometry.camera_pose(name, config)


def test_episode_keeps_geometry_without_original_file_and_rejects_changed_resume(tmp_path, measurements):
    dataset = LeRobotDataset.create(
        repo_id="test/geometry",
        root=tmp_path / "episode",
        fps=10,
        features={"action": {"dtype": "float32", "shape": (1,), "names": ["joint"]}},
        use_videos=False,
    )

    def unsupported():
        raise NotImplementedError

    robot = SimpleNamespace(
        geometry=RobotGeometry(measurements, robot_id="measured-arm"), cameras={}, cartesian_model=unsupported
    )
    attach_robot_geometry(dataset.meta, robot)
    dataset.add_frame({"task": "test", "action": np.array([1], dtype=np.float32)})
    dataset.save_episode()
    dataset.finalize()
    reopened = LeRobotDatasetMetadata("test/geometry", root=dataset.root)
    assert reopened.info.robot_geometry == measurements
    changed = copy.deepcopy(measurements)
    changed["cameras"]["top"]["T_base_cam"][0][3] += 0.01
    robot.geometry = RobotGeometry(changed, robot_id="measured-arm")
    with pytest.raises(ValueError, match="differs"):
        attach_robot_geometry(reopened, robot)
    robot.geometry = None
    with pytest.raises(ValueError, match="requires"):
        attach_robot_geometry(reopened, robot)
