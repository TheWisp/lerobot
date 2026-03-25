"""Tests for Run tab API: profile→CLI conversion and endpoint arg assembly."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from lerobot.gui.api.run import (
    RecordRequest,
    ReplayRequest,
    TeleoperateRequest,
    _ensure_no_active_process,
    _profile_to_cli_args,
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
                "left_arm_port": "/dev/ttyACM0",
                "right_arm_port": "/dev/ttyACM1",
                "left_arm_disable_torque_on_disconnect": True,
                "unused": None,
            },
            "cameras": {"cam_0": {"type": "opencv", "index": 0}},
        }
        args = _profile_to_cli_args(profile, "robot")
        assert "--robot.type=bi_so107_follower" in args
        assert "--robot.left_arm_port=/dev/ttyACM0" in args
        assert "--robot.right_arm_port=/dev/ttyACM1" in args
        assert "--robot.left_arm_disable_torque_on_disconnect=true" in args
        assert not any("unused" in a for a in args)
        assert any("cameras" in a for a in args)

    def test_unknown_fields_filtered_with_warning(self):
        """Fields not in the config class are silently skipped."""
        profile = {
            "type": "bi_so107_follower",
            "fields": {
                "left_arm_port": "/dev/ttyACM0",
                "this_field_will_never_exist_xyz": True,
            },
        }
        args = _profile_to_cli_args(profile, "robot")
        assert "--robot.left_arm_port=/dev/ttyACM0" in args
        assert not any("this_field_will_never_exist_xyz" in a for a in args)

    def test_include_cameras_false(self):
        cameras = {"cam_0": {"type": "opencv", "index": 0}}
        profile = {"type": "t", "fields": {"port": "/dev/ttyACM0"}, "cameras": cameras}
        args = _profile_to_cli_args(profile, "robot", include_cameras=False)
        assert "--robot.port=/dev/ttyACM0" in args
        assert not any("cameras" in a for a in args)


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
    "type": "so107_follower",
    "fields": {"port": "/dev/ttyACM0"},
    "cameras": {"cam_0": {"type": "opencv", "index_or_path": "/dev/video0"}},
}
_TELEOP = {"type": "so107_leader", "fields": {"port": "/dev/ttyACM1"}}


def _make_fake_launch(captured_args):
    """Create a fake _launch_subprocess that captures args and sets _active_process."""
    import lerobot.gui.api.run as run_module

    async def fake_launch(args, command, config, extra_env=None):
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
            ):
                await start_teleoperate(req)

        asyncio.run(run())

        assert captured_args[0] == "lerobot-teleoperate"
        assert "--robot.type=so107_follower" in captured_args
        assert "--robot.port=/dev/ttyACM0" in captured_args
        assert "--teleop.type=so107_leader" in captured_args
        assert "--teleop.port=/dev/ttyACM1" in captured_args
        assert "--fps=45" in captured_args


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
            ):
                await start_record(req)

        asyncio.run(run())

        assert captured_args[0] == "lerobot-record"
        assert "--robot.type=so107_follower" in captured_args
        assert "--teleop.type=so107_leader" in captured_args
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
            ):
                await start_record(req)

        asyncio.run(run())

        assert "--dataset.video=false" in captured_args

    def test_record_with_policy_path(self):
        captured_args = []

        async def run():
            req = RecordRequest(
                robot=_ROBOT,
                teleop=_TELEOP,
                repo_id="eval/eval_my_policy",
                single_task="pick up socket",
                policy_path="/home/user/outputs/act/checkpoints/last/pretrained_model",
            )
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
            ):
                await start_record(req)

        asyncio.run(run())

        assert "--policy.path=/home/user/outputs/act/checkpoints/last/pretrained_model" in captured_args
        assert "--teleop.type=so107_leader" in captured_args

    def test_record_policy_only_no_teleop(self):
        captured_args = []

        async def run():
            req = RecordRequest(
                robot=_ROBOT,
                teleop=None,
                repo_id="eval/eval_my_policy",
                single_task="pick up socket",
                policy_path="/home/user/outputs/act/checkpoints/last/pretrained_model",
            )
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
            ):
                await start_record(req)

        asyncio.run(run())

        assert "--policy.path=/home/user/outputs/act/checkpoints/last/pretrained_model" in captured_args
        assert not any("--teleop." in a for a in captured_args)

    def test_record_no_teleop_no_policy_raises(self):
        async def run():
            req = RecordRequest(
                robot=_ROBOT,
                teleop=None,
                repo_id="eval/eval_my_policy",
                single_task="task",
            )
            with (
                patch("lerobot.gui.api.run._active_process", None),
            ):
                await start_record(req)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(run())
        assert exc_info.value.status_code == 400


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
            ):
                await start_replay(req)

        asyncio.run(run())

        assert captured_args[0] == "lerobot-replay"
        assert "--robot.type=so107_follower" in captured_args
        assert "--robot.port=/dev/ttyACM0" in captured_args
        assert "--dataset.repo_id=user/my_dataset" in captured_args
        assert "--dataset.episode=5" in captured_args
        assert "--dataset.fps=60" in captured_args
        # Replay should not have teleop args
        assert not any("teleop" in a for a in captured_args)
        # Replay now includes cameras (for obs-stream live viewer)
        assert any("cameras" in a for a in captured_args)
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
            ):
                await start_replay(req)

        asyncio.run(run())

        assert "--dataset.root=/tmp/datasets/user/my_dataset" in captured_args
        assert "--dataset.repo_id=user/my_dataset" in captured_args
        assert "--dataset.episode=3" in captured_args


# ============================================================================
# Stale obs reader cleanup on launch
# ============================================================================


class TestObsReaderCleanupOnLaunch:
    """_launch_subprocess must close the obs reader so the GUI doesn't serve
    stale frames from a previous session's shared memory segments."""

    def test_launch_clears_obs_reader(self):
        """Launching any workflow closes existing obs reader."""
        import lerobot.gui.api.run as run_module

        close_called = False
        original_close = run_module._close_obs_reader

        def tracking_close():
            nonlocal close_called
            close_called = True
            original_close()

        captured_args = []

        async def run():
            req = TeleoperateRequest(robot=_ROBOT, teleop=_TELEOP, fps=30)
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._close_obs_reader", tracking_close),
                patch(
                    "lerobot.gui.api.run._launch_subprocess",
                    wraps=run_module._launch_subprocess,
                ) as mock_launch,
                patch("asyncio.create_subprocess_exec") as mock_exec,
            ):
                mock_proc = AsyncMock()
                mock_proc.pid = 9999
                mock_proc.stdout = AsyncMock()
                mock_proc.stderr = AsyncMock()
                mock_exec.return_value = mock_proc
                await start_teleoperate(req)

        asyncio.run(run())
        assert close_called, "_close_obs_reader must be called during launch"

    def test_inode_based_reader_reattach(self):
        """_get_obs_reader detects inode change and re-attaches."""
        import lerobot.gui.api.run as run_module

        # Simulate: reader attached with inode 1000, then stream recreated with inode 2000
        mock_reader = AsyncMock()
        mock_reader.close = lambda: None

        run_module._obs_reader = mock_reader
        run_module._obs_reader_meta_ino = 1000

        with (
            patch("os.stat") as mock_stat,
            patch(
                "lerobot.robots.obs_stream.ObservationStreamReader",
            ) as MockReader,
        ):
            # Current inode is different — stream was recreated
            mock_stat.return_value.st_ino = 2000
            new_reader = AsyncMock()
            new_reader.obs_scalar_keys = ["j.pos"]
            new_reader.image_keys = {}
            MockReader.return_value = new_reader

            result = run_module._get_obs_reader()

            assert result is new_reader, "Should return newly created reader"
            assert run_module._obs_reader is new_reader
            assert run_module._obs_reader_meta_ino == 2000

        # Cleanup
        run_module._obs_reader = None
        run_module._obs_reader_meta_ino = None

    def test_same_inode_keeps_reader(self):
        """_get_obs_reader returns cached reader when inode hasn't changed."""
        import lerobot.gui.api.run as run_module

        mock_reader = AsyncMock()
        run_module._obs_reader = mock_reader
        run_module._obs_reader_meta_ino = 1000

        with patch("os.stat") as mock_stat:
            mock_stat.return_value.st_ino = 1000  # Same inode
            result = run_module._get_obs_reader()
            assert result is mock_reader, "Should return cached reader"

        # Cleanup
        run_module._obs_reader = None
        run_module._obs_reader_meta_ino = None

    def test_missing_shm_clears_reader(self):
        """_get_obs_reader clears reader if /dev/shm file disappears."""
        import lerobot.gui.api.run as run_module

        mock_reader = AsyncMock()
        mock_reader.close = lambda: None
        run_module._obs_reader = mock_reader
        run_module._obs_reader_meta_ino = 1000

        with patch("os.stat", side_effect=FileNotFoundError):
            result = run_module._get_obs_reader()
            assert result is None
            assert run_module._obs_reader is None

        # Cleanup
        run_module._obs_reader_meta_ino = None
