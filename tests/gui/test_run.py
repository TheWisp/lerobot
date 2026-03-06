"""Tests for Run tab API: profile→CLI conversion and endpoint arg assembly."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from lerobot.gui.api.run import (
    RERUN_GRPC_PORT,
    RecordRequest,
    ReplayRequest,
    TeleoperateRequest,
    _display_args,
    _ensure_no_active_process,
    _profile_to_cli_args,
    _rerun_env,
    start_record,
    start_replay,
    start_teleoperate,
)


# ============================================================================
# _profile_to_cli_args
# ============================================================================


class TestProfileToCliArgs:
    """Tests for converting profile dicts to draccus CLI arguments."""

    def test_basic_type(self):
        profile = {"type": "bi_so107_follower"}
        args = _profile_to_cli_args(profile, "robot")
        assert args == ["--robot.type=bi_so107_follower"]

    def test_string_fields(self):
        profile = {"type": "koch_follower", "fields": {"port": "/dev/ttyACM0"}}
        args = _profile_to_cli_args(profile, "robot")
        assert "--robot.type=koch_follower" in args
        assert "--robot.port=/dev/ttyACM0" in args

    def test_bool_fields_lowercase(self):
        profile = {"type": "t", "fields": {"gripper_bounce": True, "enabled": False}}
        args = _profile_to_cli_args(profile, "robot")
        assert "--robot.gripper_bounce=true" in args
        assert "--robot.enabled=false" in args

    def test_none_fields_skipped(self):
        profile = {"type": "t", "fields": {"port": None, "baudrate": 115200}}
        args = _profile_to_cli_args(profile, "robot")
        assert len(args) == 2  # type + baudrate
        assert not any("port" in a for a in args)

    def test_numeric_fields(self):
        profile = {"type": "t", "fields": {"baudrate": 115200, "timeout": 1.5}}
        args = _profile_to_cli_args(profile, "robot")
        assert "--robot.baudrate=115200" in args
        assert "--robot.timeout=1.5" in args

    def test_cameras_json(self):
        cameras = {"cam_0": {"type": "opencv", "index": 0}}
        profile = {"type": "t", "cameras": cameras}
        args = _profile_to_cli_args(profile, "robot")
        cam_arg = next(a for a in args if "cameras" in a)
        assert cam_arg == f"--robot.cameras={json.dumps(cameras)}"

    def test_empty_cameras_omitted(self):
        profile = {"type": "t", "cameras": {}}
        args = _profile_to_cli_args(profile, "robot")
        assert not any("cameras" in a for a in args)

    def test_no_cameras_key(self):
        profile = {"type": "t"}
        args = _profile_to_cli_args(profile, "robot")
        assert not any("cameras" in a for a in args)

    def test_no_fields_key(self):
        profile = {"type": "t"}
        args = _profile_to_cli_args(profile, "robot")
        assert args == ["--robot.type=t"]

    def test_teleop_prefix(self):
        profile = {"type": "so_leader", "fields": {"port": "/dev/ttyACM1"}}
        args = _profile_to_cli_args(profile, "teleop")
        assert "--teleop.type=so_leader" in args
        assert "--teleop.port=/dev/ttyACM1" in args

    def test_multiple_fields_ordering(self):
        profile = {"type": "t", "fields": {"a": 1, "b": 2, "c": 3}}
        args = _profile_to_cli_args(profile, "robot")
        # Type always first
        assert args[0] == "--robot.type=t"
        # All fields present
        assert len(args) == 4

    def test_mixed_field_types(self):
        profile = {
            "type": "bi_so107_follower",
            "fields": {
                "port": "/dev/ttyACM0",
                "baudrate": 1000000,
                "gripper_bounce": True,
                "unused": None,
            },
            "cameras": {"cam_0": {"type": "opencv", "index": 0}},
        }
        args = _profile_to_cli_args(profile, "robot")
        assert "--robot.type=bi_so107_follower" in args
        assert "--robot.port=/dev/ttyACM0" in args
        assert "--robot.baudrate=1000000" in args
        assert "--robot.gripper_bounce=true" in args
        assert not any("unused" in a for a in args)
        assert any("cameras" in a for a in args)

    def test_include_cameras_false(self):
        cameras = {"cam_0": {"type": "opencv", "index": 0}}
        profile = {"type": "t", "fields": {"port": "/dev/ttyACM0"}, "cameras": cameras}
        args = _profile_to_cli_args(profile, "robot", include_cameras=False)
        assert "--robot.port=/dev/ttyACM0" in args
        assert not any("cameras" in a for a in args)


# ============================================================================
# _display_args
# ============================================================================


class TestDisplayArgs:
    """Tests for Rerun display arg generation."""

    def test_returns_args_when_rerun_started(self):
        with patch("lerobot.gui.api.run._rerun_started", True):
            args = _display_args()
        assert "--display_data=true" in args
        assert "--display_compressed_images=true" in args

    def test_returns_empty_when_rerun_not_started(self):
        with patch("lerobot.gui.api.run._rerun_started", False):
            args = _display_args()
        assert args == []

    def test_rerun_env_when_started(self):
        with patch("lerobot.gui.api.run._rerun_started", True):
            env = _rerun_env()
        assert env == {"LEROBOT_RERUN_SERVE_PORT": str(RERUN_GRPC_PORT)}

    def test_rerun_env_when_not_started(self):
        with patch("lerobot.gui.api.run._rerun_started", False):
            env = _rerun_env()
        assert env == {}


# ============================================================================
# _ensure_no_active_process
# ============================================================================


class TestEnsureNoActiveProcess:
    """Tests for the active process guard."""

    def test_no_process_passes(self):
        with patch("lerobot.gui.api.run._active_process", None):
            _ensure_no_active_process()  # Should not raise

    def test_exited_process_passes(self):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0  # Process has exited
        mock_proc.pid = 123
        with patch("lerobot.gui.api.run._active_process", mock_proc):
            _ensure_no_active_process()  # Should not raise

    def test_running_process_raises_409(self):
        mock_proc = AsyncMock()
        mock_proc.returncode = None  # Still running
        mock_proc.pid = 456
        with (
            patch("lerobot.gui.api.run._active_process", mock_proc),
            patch("lerobot.gui.api.run._active_command", "teleoperate"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                _ensure_no_active_process()
            assert exc_info.value.status_code == 409


# ============================================================================
# Endpoint arg assembly
# ============================================================================

_ROBOT = {
    "type": "bi_so107_follower",
    "fields": {"port": "/dev/ttyACM0"},
    "cameras": {"cam_0": {"type": "opencv", "index_or_path": "/dev/video0"}},
}
_TELEOP = {"type": "so_leader", "fields": {"port": "/dev/ttyACM1"}}


def _make_fake_launch(captured_args):
    """Create a fake _launch_subprocess that captures args and sets _active_process."""
    import lerobot.gui.api.run as run_module

    async def fake_launch(args, command, config):
        captured_args.extend(args)
        # The endpoint reads _active_process.pid after launch, so set a mock
        mock_proc = AsyncMock()
        mock_proc.pid = 9999
        run_module._active_process = mock_proc

    return fake_launch


class TestTeleoperateEndpoint:
    """Tests for the teleoperate endpoint's CLI arg assembly."""

    def test_teleoperate_args(self):
        captured_args = []

        async def run():
            req = TeleoperateRequest(robot=_ROBOT, teleop=_TELEOP, fps=45)
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
                patch("lerobot.gui.api.run._rerun_started", False),
            ):
                await start_teleoperate(req)

        asyncio.run(run())

        assert captured_args[0] == "lerobot-teleoperate"
        assert "--robot.type=bi_so107_follower" in captured_args
        assert "--robot.port=/dev/ttyACM0" in captured_args
        assert "--teleop.type=so_leader" in captured_args
        assert "--teleop.port=/dev/ttyACM1" in captured_args
        assert "--fps=45" in captured_args

    def test_teleoperate_includes_display_args(self):
        captured_args = []

        async def run():
            req = TeleoperateRequest(robot=_ROBOT, teleop=_TELEOP)
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
                patch("lerobot.gui.api.run._rerun_started", True),
            ):
                await start_teleoperate(req)

        asyncio.run(run())

        assert "--display_data=true" in captured_args


class TestRecordEndpoint:
    """Tests for the record endpoint's CLI arg assembly."""

    def test_record_args(self):
        captured_args = []

        async def run():
            req = RecordRequest(
                robot=_ROBOT,
                teleop=_TELEOP,
                repo_id="user/my_dataset",
                single_task="pick up the cube",
                fps=30,
                episode_time_s=120,
                reset_time_s=30,
                num_episodes=10,
                video=True,
                vcodec="libsvtav1",
                play_sounds=False,
                resume=False,
            )
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
                patch("lerobot.gui.api.run._rerun_started", False),
            ):
                await start_record(req)

        asyncio.run(run())

        assert captured_args[0] == "lerobot-record"
        assert "--robot.type=bi_so107_follower" in captured_args
        assert "--teleop.type=so_leader" in captured_args
        assert "--dataset.repo_id=user/my_dataset" in captured_args
        assert "--dataset.single_task=pick up the cube" in captured_args
        assert "--dataset.fps=30" in captured_args
        assert "--dataset.episode_time_s=120.0" in captured_args
        assert "--dataset.reset_time_s=30.0" in captured_args
        assert "--dataset.num_episodes=10" in captured_args
        assert "--dataset.video=true" in captured_args
        assert "--dataset.push_to_hub=false" in captured_args
        assert "--dataset.vcodec=libsvtav1" in captured_args
        assert "--play_sounds=false" in captured_args
        # resume=False should not add --resume
        assert "--resume=true" not in captured_args
        # No root provided — should not have --dataset.root
        assert not any("--dataset.root" in a for a in captured_args)

    def test_record_with_root(self):
        captured_args = []

        async def run():
            req = RecordRequest(
                robot=_ROBOT,
                teleop=_TELEOP,
                repo_id="user/my_dataset",
                root="/home/user/.cache/huggingface/lerobot/user/my_dataset",
                single_task="pick up the cube",
                resume=True,
            )
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
                patch("lerobot.gui.api.run._rerun_started", False),
            ):
                await start_record(req)

        asyncio.run(run())

        assert "--dataset.root=/home/user/.cache/huggingface/lerobot/user/my_dataset" in captured_args
        assert "--resume=true" in captured_args

    def test_record_resume_flag(self):
        captured_args = []

        async def run():
            req = RecordRequest(
                robot=_ROBOT,
                teleop=_TELEOP,
                repo_id="user/ds",
                single_task="task",
                resume=True,
            )
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
                patch("lerobot.gui.api.run._rerun_started", False),
            ):
                await start_record(req)

        asyncio.run(run())

        assert "--resume=true" in captured_args

    def test_record_video_false(self):
        captured_args = []

        async def run():
            req = RecordRequest(
                robot=_ROBOT,
                teleop=_TELEOP,
                repo_id="user/ds",
                single_task="task",
                video=False,
            )
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
                patch("lerobot.gui.api.run._rerun_started", False),
            ):
                await start_record(req)

        asyncio.run(run())

        assert "--dataset.video=false" in captured_args


class TestUnifiedFps:
    """The GUI uses a single FPS field for both teleop-only and record modes.

    Verify that the same fps value routes to --fps (teleoperate) or
    --dataset.fps (record) depending on the endpoint.
    """

    def test_teleop_uses_fps_flag(self):
        captured_args = []

        async def run():
            req = TeleoperateRequest(robot=_ROBOT, teleop=_TELEOP, fps=25)
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
                patch("lerobot.gui.api.run._rerun_started", False),
            ):
                await start_teleoperate(req)

        asyncio.run(run())

        assert "--fps=25" in captured_args
        assert not any("--dataset.fps" in a for a in captured_args)

    def test_record_uses_dataset_fps_flag(self):
        captured_args = []

        async def run():
            req = RecordRequest(
                robot=_ROBOT, teleop=_TELEOP,
                repo_id="user/ds", single_task="task", fps=25,
            )
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
                patch("lerobot.gui.api.run._rerun_started", False),
            ):
                await start_record(req)

        asyncio.run(run())

        assert "--dataset.fps=25" in captured_args
        assert not any(a == "--fps=25" for a in captured_args)


class TestReplayEndpoint:
    """Tests for the replay endpoint's CLI arg assembly."""

    def test_replay_args(self):
        captured_args = []

        async def run():
            req = ReplayRequest(
                robot=_ROBOT,
                repo_id="user/my_dataset",
                episode=5,
                fps=60,
            )
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
                patch("lerobot.gui.api.run._rerun_started", False),
            ):
                await start_replay(req)

        asyncio.run(run())

        assert captured_args[0] == "lerobot-replay"
        assert "--robot.type=bi_so107_follower" in captured_args
        assert "--robot.port=/dev/ttyACM0" in captured_args
        assert "--dataset.repo_id=user/my_dataset" in captured_args
        assert "--dataset.episode=5" in captured_args
        assert "--dataset.fps=60" in captured_args
        # Replay should not have teleop args or cameras
        assert not any("teleop" in a for a in captured_args)
        assert not any("cameras" in a for a in captured_args)
        # No root provided
        assert not any("--dataset.root" in a for a in captured_args)

    def test_replay_with_root(self):
        captured_args = []

        async def run():
            req = ReplayRequest(
                robot=_ROBOT,
                repo_id="user/my_dataset",
                root="/tmp/datasets/user/my_dataset",
                episode=3,
                fps=30,
            )
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
                patch("lerobot.gui.api.run._rerun_started", False),
            ):
                await start_replay(req)

        asyncio.run(run())

        assert "--dataset.root=/tmp/datasets/user/my_dataset" in captured_args
        assert "--dataset.repo_id=user/my_dataset" in captured_args
        assert "--dataset.episode=3" in captured_args
