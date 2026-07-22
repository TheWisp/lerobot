#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from unittest.mock import MagicMock, call, patch

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("deepdiff", reason="deepdiff is required (install lerobot[hardware])")

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.scripts.lerobot_calibrate import CalibrateConfig, calibrate
from lerobot.scripts.lerobot_record import RecordConfig, record
from lerobot.scripts.lerobot_replay import DatasetReplayConfig, ReplayConfig, replay
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.mocks.mock_robot import MockRobot, MockRobotConfig
from tests.mocks.mock_teleop import MockTeleopConfig


def test_calibrate():
    robot_cfg = MockRobotConfig()
    cfg = CalibrateConfig(robot=robot_cfg)
    calibrate(cfg)


def test_teleoperate():
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    cfg = TeleoperateConfig(
        robot=robot_cfg,
        teleop=teleop_cfg,
        teleop_time_s=0.1,
    )
    teleoperate(cfg)


def _teleoperate_config() -> TeleoperateConfig:
    return TeleoperateConfig(
        robot=MockRobotConfig(),
        teleop=MockTeleopConfig(),
        teleop_time_s=0.1,
    )


def test_teleoperate_cleans_up_when_robot_connect_fails():
    teleop = MagicMock()
    robot = MagicMock()
    robot.get_observation_processor_steps.return_value = []
    robot.connect.side_effect = RuntimeError("connect failed")

    with (
        patch(
            "lerobot.scripts.lerobot_teleoperate.make_teleoperator_from_config",
            return_value=teleop,
        ),
        patch("lerobot.scripts.lerobot_teleoperate.make_robot_from_config", return_value=robot),
        patch("lerobot.scripts.lerobot_teleoperate.setup_run_logging"),
        pytest.raises(RuntimeError, match="connect failed"),
    ):
        teleoperate(_teleoperate_config())

    teleop.connect.assert_called_once_with()
    robot.connect.assert_called_once_with()
    robot.attach_teleop.assert_called_once_with(None)
    teleop.disconnect.assert_called_once_with()
    robot.disconnect.assert_called_once_with()


def test_teleoperate_cleans_up_when_attach_fails():
    teleop = MagicMock()
    robot = MagicMock()
    robot.get_observation_processor_steps.return_value = []
    robot.attach_teleop.side_effect = [RuntimeError("attach failed"), None]

    with (
        patch(
            "lerobot.scripts.lerobot_teleoperate.make_teleoperator_from_config",
            return_value=teleop,
        ),
        patch("lerobot.scripts.lerobot_teleoperate.make_robot_from_config", return_value=robot),
        patch("lerobot.scripts.lerobot_teleoperate.setup_run_logging"),
        pytest.raises(RuntimeError, match="attach failed"),
    ):
        teleoperate(_teleoperate_config())

    assert robot.attach_teleop.call_args_list == [call(teleop), call(None)]
    teleop.disconnect.assert_called_once_with()
    robot.disconnect.assert_called_once_with()


def test_teleoperate_cleanup_steps_are_independent(caplog):
    teleop = MagicMock()
    robot = MagicMock()
    robot.get_observation_processor_steps.return_value = []

    def attach(value):
        if value is None:
            raise RuntimeError("detach failed")

    robot.attach_teleop.side_effect = attach
    teleop.disconnect.side_effect = RuntimeError("teleop disconnect failed")
    robot.disconnect.side_effect = RuntimeError("robot disconnect failed")

    with (
        patch(
            "lerobot.scripts.lerobot_teleoperate.make_teleoperator_from_config",
            return_value=teleop,
        ),
        patch("lerobot.scripts.lerobot_teleoperate.make_robot_from_config", return_value=robot),
        patch("lerobot.scripts.lerobot_teleoperate.setup_run_logging"),
        patch(
            "lerobot.scripts.lerobot_teleoperate.teleop_loop",
            side_effect=ValueError("loop failed"),
        ),
        pytest.raises(ValueError, match="loop failed"),
    ):
        teleoperate(_teleoperate_config())

    assert robot.attach_teleop.call_args_list == [call(teleop), call(None)]
    teleop.disconnect.assert_called_once_with()
    robot.disconnect.assert_called_once_with()
    assert "Failed to detach teleoperator from robot" in caplog.text
    assert "Failed to disconnect teleoperator" in caplog.text
    assert "Failed to disconnect robot" in caplog.text


def test_record_and_resume(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    dataset = record(cfg)

    assert dataset.fps == 30
    assert dataset.meta.total_episodes == dataset.num_episodes == 1
    assert dataset.meta.total_frames == dataset.num_frames == 3
    assert dataset.meta.total_tasks == 1

    cfg.resume = True
    # Mock the revision to prevent Hub calls during resume
    with (
        patch("lerobot.datasets.dataset_metadata.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.dataset_metadata.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record")
        dataset = record(cfg)

    assert dataset.meta.total_episodes == dataset.num_episodes == 2
    assert dataset.meta.total_frames == dataset.num_frames == 6
    assert dataset.meta.total_tasks == 1


def test_record_saves_action_returned_by_robot(tmp_path):
    robot_cfg = MockRobotConfig(random_values=False, static_values=[0.0, 0.0, 0.0])
    teleop_cfg = MockTeleopConfig(random_values=False, static_values=[25.0, 50.0, 75.0])
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record_sent_action",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    def clipped_action(_robot, action):
        return dict.fromkeys(action, 7.0)

    with patch.object(MockRobot, "send_action", autospec=True, side_effect=clipped_action):
        dataset = record(cfg)

    assert list(dataset.get_raw_item(0)["action"]) == pytest.approx([7.0, 7.0, 7.0])


def test_record_and_replay(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    record_dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record_and_replay",
        num_episodes=1,
        episode_time_s=0.1,
        push_to_hub=False,
    )
    record_cfg = RecordConfig(
        robot=robot_cfg,
        dataset=record_dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )
    replay_dataset_cfg = DatasetReplayConfig(
        repo_id=DUMMY_REPO_ID,
        episode=0,
        root=tmp_path / "record_and_replay",
    )
    replay_cfg = ReplayConfig(
        robot=robot_cfg,
        dataset=replay_dataset_cfg,
        play_sounds=False,
    )

    record(record_cfg)

    # Mock the revision to prevent Hub calls during replay
    with (
        patch("lerobot.datasets.dataset_metadata.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.dataset_metadata.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record_and_replay")
        replay(replay_cfg)
