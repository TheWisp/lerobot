"""Demo Actions display comes from all saved commands, without robot access."""

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from lerobot.datasets.camera_calibration import attach_robot_geometry
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.gui.api import datasets, flash_editor
from lerobot.policies.flash.demo_trajectory import read_demo_trajectory
from tests.policies.flash.test_demo_execution import ready_state, robot  # noqa: F401


@pytest.mark.asyncio
async def test_demo_actions_use_demonstrated_poses_and_saved_camera_on_dedicated_executor(
    tmp_path,
    robot,  # noqa: F811
    monkeypatch,
):
    from threading import current_thread

    model = robot.cartesian_model()
    state = ready_state(model)
    names = list(state)
    camera = "observation.images.camera"
    calibration = {"width": 16, "height": 16, "fx": 12, "fy": 12, "cx": 8, "cy": 8}
    features = {
        key: {"dtype": "float32", "shape": (7,), "names": names} for key in ("observation.state", "action")
    }
    features[camera] = {
        "dtype": "image",
        "shape": (16, 16, 3),
        "names": ["height", "width", "channels"],
        "info": {"camera": {"intrinsics": calibration}},
    }
    dataset = LeRobotDataset.create("test/trajectory", root=tmp_path / "demo", fps=30, features=features)
    attach_robot_geometry(dataset.meta, robot)
    observations, actions = [], []
    for angle in (5, 10, 20, 15, 0):
        observed = {**state, model.joint_names[0]: angle}
        action = {**observed, model.joint_names[1]: state[model.joint_names[1]] + 8}
        observations.append(observed)
        actions.append(action)
        dataset.add_frame(
            {
                "task": "move",
                "observation.state": np.array(list(observed.values()), np.float32),
                "action": np.array(list(action.values()), np.float32),
                camera: np.zeros((16, 16, 3), np.uint8),
            }
        )
    dataset.save_episode()
    dataset.finalize()
    dataset = LeRobotDataset("test/trajectory", root=dataset.root)
    robot.config.geometry_path.unlink()
    monkeypatch.setattr(datasets, "_app_state", SimpleNamespace(datasets={"demo": dataset}))
    original_column = dataset.episode_column
    threads = []

    def column(*args):
        threads.append(current_thread().name)
        return original_column(*args)

    monkeypatch.setattr(dataset, "episode_column", column)
    result = await flash_editor.demo_trajectory("demo", camera, 0)
    assert threads and all(name.startswith("flash-editor") for name in threads)
    assert [point["frame"] for point in result["points"]] == list(range(5))
    np.testing.assert_allclose([p["time_s"] for p in result["points"]], np.arange(5) / 30)
    camera_from_base = np.linalg.inv(robot.geometry.to_dict()["cameras"]["camera"]["T_base_cam"])
    expected = [(camera_from_base @ model.forward(observed))[:3, 3] for observed in observations]
    np.testing.assert_allclose([p["position"] for p in result["points"]], expected)
    assert not np.allclose(
        result["points"][0]["position"], (camera_from_base @ model.forward(actions[0]))[:3, 3]
    )
    np.testing.assert_allclose(result["spans"][0]["positions"][::4], expected)
    assert result["camera"] == {"width": 16, "height": 16, "intrinsics": [[12, 0, 8], [0, 12, 8], [0, 0, 1]]}
    robot.connect.assert_not_called()
    robot.send_action.assert_not_called()
    # Missing metadata cannot silently substitute today's calibration or guessed units.
    dataset.meta.info.robot_cartesian = None
    with pytest.raises(HTTPException) as error:
        await flash_editor.demo_trajectory("demo", camera, 0)
    assert error.value.status_code == 422


def test_camera_only_demo_has_no_robot_actions():
    with pytest.raises(KeyError):
        read_demo_trajectory(SimpleNamespace(meta=SimpleNamespace(info={})), 0, "observation.images.camera")
