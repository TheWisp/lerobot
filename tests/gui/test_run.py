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
                robot=_ROBOT,
                teleop=_TELEOP,
                repo_id="user/ds",
                single_task="task",
                fps=25,
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
        # Replay must NOT pass --dataset.fps: the upstream lerobot-replay loop
        # paces by `dataset.fps` (read from the loaded dataset) regardless,
        # so a CLI-supplied value would be misleading dead config.
        assert not any(a.startswith("--dataset.fps") for a in captured_args)
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

        async def run():
            req = TeleoperateRequest(robot=_ROBOT, teleop=_TELEOP, fps=30)
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._close_obs_reader", tracking_close),
                patch(
                    "lerobot.gui.api.run._launch_subprocess",
                    wraps=run_module._launch_subprocess,
                ),
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
            ) as mock_reader_cls,
        ):
            # Current inode is different — stream was recreated
            mock_stat.return_value.st_ino = 2000
            new_reader = AsyncMock()
            new_reader.obs_scalar_keys = ["j.pos"]
            new_reader.image_keys = {}
            mock_reader_cls.return_value = new_reader

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


# ============================================================================
# GET /api/run/rlt-config — slider seed endpoint
# ============================================================================


class TestGetRltConfig:
    """The GUI sliders seed from this endpoint on init so they reflect what
    the training process actually has applied, not HTML hardcoded defaults.
    """

    @pytest.fixture
    def reset_active(self):
        import lerobot.gui.api.run as run_module

        prev = run_module._active_config
        run_module._active_config = None
        yield run_module
        run_module._active_config = prev

    def test_no_active_session_returns_defaults(self, reset_active):
        """Before any RLT session starts, endpoint returns RLTConfig defaults.
        Without this the GUI would crash trying to read undefined state."""
        from lerobot.gui.api.run import get_rlt_config
        from lerobot.policies.hvla.rlt.config import RLTConfig

        cfg = asyncio.run(get_rlt_config())
        defaults = RLTConfig()
        assert cfg["beta"] == defaults.beta
        assert cfg["exploration_sigma"] == defaults.exploration_sigma
        assert cfg["target_sigma"] == defaults.target_sigma
        assert cfg["dump_chunks"] is False

    def test_reads_current_override_values(self, reset_active, tmp_path):
        """The whole point: the value on disk must round-trip to the GUI.
        Before this endpoint existed, the slider hardcoded 0.02 regardless
        of what the training process was using (the bug we're fixing)."""
        from lerobot.gui.api.run import get_rlt_config

        (tmp_path / "rlt_overrides.json").write_text(
            json.dumps({"beta": 0.5, "exploration_sigma": 0.0, "dump_chunks": True})
        )
        reset_active._active_config = {"rlt_output_dir": str(tmp_path)}
        cfg = asyncio.run(get_rlt_config())
        assert cfg["beta"] == 0.5
        assert cfg["exploration_sigma"] == 0.0
        assert cfg["dump_chunks"] is True

    def test_legacy_actor_sigma_key_is_accepted(self, reset_active, tmp_path):
        """Older GUI builds wrote ``actor_sigma``. Accept as synonym for
        ``exploration_sigma`` so mid-session upgrades don't drop the value."""
        from lerobot.gui.api.run import get_rlt_config

        (tmp_path / "rlt_overrides.json").write_text(json.dumps({"beta": 0.5, "actor_sigma": 0.0}))
        reset_active._active_config = {"rlt_output_dir": str(tmp_path)}
        cfg = asyncio.run(get_rlt_config())
        assert cfg["exploration_sigma"] == 0.0

    def test_partial_override_fills_missing_keys_from_defaults(self, reset_active, tmp_path):
        """Old override files may predate some keys. Missing keys fall back
        to config defaults so the GUI always has a value to show."""
        from lerobot.gui.api.run import get_rlt_config
        from lerobot.policies.hvla.rlt.config import RLTConfig

        (tmp_path / "rlt_overrides.json").write_text(json.dumps({"beta": 0.3}))
        reset_active._active_config = {"rlt_output_dir": str(tmp_path)}
        cfg = asyncio.run(get_rlt_config())
        assert cfg["beta"] == 0.3
        assert cfg["exploration_sigma"] == RLTConfig().exploration_sigma
        assert cfg["dump_chunks"] is False

    def test_missing_file_returns_defaults(self, reset_active, tmp_path):
        """Fresh RLT session hasn't written the file yet — don't 404."""
        from lerobot.gui.api.run import get_rlt_config
        from lerobot.policies.hvla.rlt.config import RLTConfig

        reset_active._active_config = {"rlt_output_dir": str(tmp_path)}
        cfg = asyncio.run(get_rlt_config())
        assert cfg["beta"] == RLTConfig().beta

    def test_malformed_file_returns_defaults(self, reset_active, tmp_path):
        """Never break the GUI over a corrupt override file."""
        from lerobot.gui.api.run import get_rlt_config
        from lerobot.policies.hvla.rlt.config import RLTConfig

        (tmp_path / "rlt_overrides.json").write_text("not json {{{")
        reset_active._active_config = {"rlt_output_dir": str(tmp_path)}
        cfg = asyncio.run(get_rlt_config())
        assert cfg["beta"] == RLTConfig().beta


# ============================================================================
# RLT mode requires the RL Token Encoder. Caught at the API boundary so a
# missing field never makes it to the subprocess (where it would crash deep
# inside actor.pt state_dict load with a size mismatch). See incident
# 9bf49910f / launch attempt of rlt_online_v2_widened on 2026-04-27.
# ============================================================================


class TestHvlaRltTokenRequired:
    """rlt_mode=True without rlt_token_checkpoint must be rejected at the
    API boundary with a helpful 400, not silently passed through."""

    def _make_request_kwargs(self, **overrides):
        """Minimum HVLARunRequest kwargs to exercise the validation path.
        Real launches need more; the rejection fires before any of it
        matters."""
        base = {
            "robot": _ROBOT,
            "s1_checkpoint": "/tmp/fake_s1",
            "task": "assemble cylinder into ring",
            "fps": 30,
            "rlt_mode": True,
        }
        base.update(overrides)
        return base

    def test_missing_token_checkpoint_raises_400(self):
        from lerobot.gui.api.run import HVLARunRequest, start_hvla

        req = HVLARunRequest(**self._make_request_kwargs())
        with patch("lerobot.gui.api.run._active_process", None), pytest.raises(HTTPException) as excinfo:
            asyncio.run(start_hvla(req))
        assert excinfo.value.status_code == 400
        # Must be actionable — name the field and the recipe
        msg = str(excinfo.value.detail)
        assert "rlt_token_checkpoint" in msg
        assert "RL Token Encoder" in msg

    def test_blank_token_checkpoint_raises_400(self):
        """Empty string must be treated the same as None — JS may send
        '' from a never-filled field."""
        from lerobot.gui.api.run import HVLARunRequest, start_hvla

        req = HVLARunRequest(
            **self._make_request_kwargs(
                rlt_token_checkpoint="   "  # whitespace only
            )
        )
        with patch("lerobot.gui.api.run._active_process", None), pytest.raises(HTTPException) as excinfo:
            asyncio.run(start_hvla(req))
        assert excinfo.value.status_code == 400

    def test_rlt_disabled_does_not_require_token(self):
        """When rlt_mode=False, the token field is irrelevant — must
        not be enforced (regression guard so the validation doesn't
        leak into normal HVLA runs)."""
        from lerobot.gui.api.run import HVLARunRequest, start_hvla

        captured_args = []
        req = HVLARunRequest(
            **self._make_request_kwargs(
                rlt_mode=False,
                rlt_token_checkpoint=None,
            )
        )
        with (
            patch("lerobot.gui.api.run._active_process", None),
            patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
        ):
            asyncio.run(start_hvla(req))
        # Reached the launch path, no rejection
        assert any("--rlt-mode" not in a for a in captured_args)
        assert "--rlt-mode" not in captured_args


class TestTimeoutExcsRegression:
    """Regression guard for the asyncio.TimeoutError vs builtin TimeoutError
    Python-3.10 mismatch.

    On Python 3.10 (the lerobot conda env's actual interpreter, despite
    pyproject's ``requires-python = ">=3.12"``) the two classes are
    *distinct* and not even in the same hierarchy. Pyupgrade and ruff
    UP041 rewrite any literal ``asyncio.TimeoutError`` token in an except
    clause to ``TimeoutError`` under py311+ targets, which silently
    breaks the SSE keepalive on the actual runtime — every 2s wait_for
    timeout escapes and crashes the response generator.

    These tests fail loudly if anyone simplifies _TIMEOUT_EXCS away."""

    def test_constant_includes_asyncio_timeout(self):
        from lerobot.gui.api.run import _TIMEOUT_EXCS

        assert TimeoutError in _TIMEOUT_EXCS
        assert asyncio.TimeoutError in _TIMEOUT_EXCS

    def test_catches_what_wait_for_raises(self):
        """End-to-end: asyncio.wait_for timing out is caught by the tuple."""
        from lerobot.gui.api.run import _TIMEOUT_EXCS

        async def go():
            try:
                await asyncio.wait_for(asyncio.sleep(10), timeout=0.01)
            except _TIMEOUT_EXCS:
                return "caught"
            return "missed"

        assert asyncio.run(go()) == "caught"
# ============================================================================
# Sim env dispatch (gym-hil via gym_manipulator)
# ============================================================================

_ENV_KEYBOARD = {
    "type": "gym_manipulator",
    "name": "test_kb",
    "fields": {
        "name": "gym_hil",
        "task": "PandaPickCubeKeyboard-v0",
        "fps": 10,
        "device": "cuda",
    },
}
_ENV_GAMEPAD = {
    "type": "gym_manipulator",
    "name": "test_gp",
    "fields": {
        "name": "gym_hil",
        "task": "PandaPickCubeGamepad-v0",
        "fps": 10,
        "device": "cuda",
    },
}


class TestEnsureOneSource:
    """`_ensure_one_source` enforces robot XOR env at request boundary."""

    def test_robot_only(self):
        from lerobot.gui.api.run import _ensure_one_source

        assert _ensure_one_source({"type": "x"}, None) == "robot"

    def test_env_only(self):
        from lerobot.gui.api.run import _ensure_one_source

        assert _ensure_one_source(None, {"type": "gym_manipulator"}) == "env"

    def test_both_set_rejected(self):
        from lerobot.gui.api.run import _ensure_one_source

        with pytest.raises(HTTPException) as exc:
            _ensure_one_source({"type": "x"}, {"type": "gym_manipulator"})
        assert exc.value.status_code == 400
        assert "not both" in str(exc.value.detail)

    def test_neither_set_rejected(self):
        from lerobot.gui.api.run import _ensure_one_source

        with pytest.raises(HTTPException) as exc:
            _ensure_one_source(None, None)
        assert exc.value.status_code == 400


class TestEnvProfileToGymManipulatorCfg:
    """`_env_profile_to_gym_manipulator_cfg` shape, defaults, and the
    discriminator-stripping behaviour that makes draccus accept the env dict."""

    def test_basic_teleop_no_dataset(self):
        from lerobot.gui.api.run import _env_profile_to_gym_manipulator_cfg

        cfg = _env_profile_to_gym_manipulator_cfg(_ENV_KEYBOARD, mode=None)
        assert cfg["mode"] is None
        assert cfg["device"] == "cuda"
        assert cfg["env"]["task"] == "PandaPickCubeKeyboard-v0"
        assert cfg["env"]["fps"] == 10
        # Placeholder dataset for non-recording mode (gym_manipulator's
        # GymManipulatorConfig.dataset is required even when mode is None).
        assert cfg["dataset"]["repo_id"].startswith("_unused")

    def test_record_mode_dataset_propagated(self):
        from lerobot.gui.api.run import _env_profile_to_gym_manipulator_cfg

        ds = {
            "repo_id": "user/sim_demo",
            "task": "pick_cube",
            "num_episodes_to_record": 5,
            "push_to_hub": False,
            "resume": False,
        }
        cfg = _env_profile_to_gym_manipulator_cfg(_ENV_KEYBOARD, mode="record", dataset=ds)
        assert cfg["mode"] == "record"
        assert cfg["dataset"] == ds

    def test_type_discriminator_stripped(self):
        """GymManipulatorConfig.env is typed as the concrete
        HILSerlRobotEnvConfig, so draccus would reject a `type` field
        inside it. The helper must strip the registry discriminator."""
        from lerobot.gui.api.run import _env_profile_to_gym_manipulator_cfg

        cfg = _env_profile_to_gym_manipulator_cfg(_ENV_KEYBOARD, mode=None)
        assert "type" not in cfg["env"]

    def test_device_lifted_to_top_level(self):
        """`device` lives at the top of GymManipulatorConfig, not inside env."""
        from lerobot.gui.api.run import _env_profile_to_gym_manipulator_cfg

        profile = {**_ENV_KEYBOARD, "fields": {**_ENV_KEYBOARD["fields"], "device": "cpu"}}
        cfg = _env_profile_to_gym_manipulator_cfg(profile, mode=None)
        assert cfg["device"] == "cpu"
        assert "device" not in cfg["env"]

    def test_default_device_cuda_when_absent(self):
        from lerobot.gui.api.run import _env_profile_to_gym_manipulator_cfg

        profile = {
            "type": "gym_manipulator",
            "name": "x",
            "fields": {"task": "PandaPickCubeBase-v0"},
        }  # no device
        cfg = _env_profile_to_gym_manipulator_cfg(profile, mode=None)
        assert cfg["device"] == "cuda"

    def test_non_gym_manipulator_type_rejected(self):
        """Other env types (aloha, libero, ...) are not wired to the
        gym_manipulator binary yet — the helper must refuse them with a
        400 instead of silently producing a config that won't parse."""
        from lerobot.gui.api.run import _env_profile_to_gym_manipulator_cfg

        profile = {"type": "libero", "name": "x", "fields": {}}
        with pytest.raises(HTTPException) as exc:
            _env_profile_to_gym_manipulator_cfg(profile, mode=None)
        assert exc.value.status_code == 400


class TestCheckEnvInputCompat:
    """`_check_env_input_compat` refuses Gamepad tasks when no controller is
    detected, so users don't see gym-hil's silent rc=0 exit as a 'crash'."""

    def test_gamepad_task_no_controller_rejected(self):
        from lerobot.gui.api.run import _check_env_input_compat

        with patch("lerobot.gui.api.run._detect_linux_gamepad", return_value=False):
            with pytest.raises(HTTPException) as exc:
                _check_env_input_compat(_ENV_GAMEPAD)
            assert exc.value.status_code == 400
            msg = str(exc.value.detail)
            assert "Gamepad" in msg
            assert "Keyboard" in msg  # must point user at the fallback task

    def test_gamepad_task_with_controller_passes(self):
        from lerobot.gui.api.run import _check_env_input_compat

        with patch("lerobot.gui.api.run._detect_linux_gamepad", return_value=True):
            _check_env_input_compat(_ENV_GAMEPAD)  # should not raise

    def test_keyboard_task_always_passes(self):
        """Keyboard tasks have no hardware dependency — must work even
        when no gamepad is present."""
        from lerobot.gui.api.run import _check_env_input_compat

        with patch("lerobot.gui.api.run._detect_linux_gamepad", return_value=False):
            _check_env_input_compat(_ENV_KEYBOARD)  # should not raise

    def test_base_task_always_passes(self):
        """The autonomous-only Base variant has no input device at all."""
        from lerobot.gui.api.run import _check_env_input_compat

        profile = {**_ENV_KEYBOARD, "fields": {**_ENV_KEYBOARD["fields"], "task": "PandaPickCubeBase-v0"}}
        with patch("lerobot.gui.api.run._detect_linux_gamepad", return_value=False):
            _check_env_input_compat(profile)


class TestSimDispatchEndToEnd:
    """End-to-end: a request with an `env` field dispatches to the
    gym_manipulator binary, not the lerobot-* scripts."""

    def test_teleoperate_env_dispatches_to_gym_manipulator(self):
        captured_args = []

        async def go():
            req = TeleoperateRequest(env=_ENV_KEYBOARD, fps=10)
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
                patch("lerobot.gui.api.run._detect_linux_gamepad", return_value=True),
            ):
                await start_teleoperate(req)

        asyncio.run(go())
        # Dispatch is to gym_manipulator, NOT lerobot-teleoperate.
        joined = " ".join(captured_args)
        assert "lerobot.rl.gym_manipulator" in joined
        assert "lerobot-teleoperate" not in joined
        # Config path passed via --config_path
        assert any("--config_path=" in a for a in captured_args)

    def test_teleoperate_env_no_robot_args(self):
        """Sim path must not pass any --robot.* args (they would fail draccus)."""
        captured_args = []

        async def go():
            req = TeleoperateRequest(env=_ENV_KEYBOARD, fps=10)
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
                patch("lerobot.gui.api.run._detect_linux_gamepad", return_value=True),
            ):
                await start_teleoperate(req)

        asyncio.run(go())
        assert not any("--robot." in a for a in captured_args)
        assert not any("--teleop." in a for a in captured_args)

    def test_record_env_writes_dataset_into_config(self):
        """Sim record passes dataset.* via the config file, not CLI flags
        like the real-robot path. Verify the JSON written to the temp file
        contains the dataset block."""
        import json as _json
        from pathlib import Path

        captured_args = []

        async def go():
            req = RecordRequest(
                env=_ENV_KEYBOARD,
                repo_id="user/sim_demo",
                single_task="pick the cube",
                num_episodes=3,
            )
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
                patch("lerobot.gui.api.run._detect_linux_gamepad", return_value=True),
            ):
                await start_record(req)

        asyncio.run(go())
        # Find the --config_path arg, read the JSON it points to
        cfg_arg = next(a for a in captured_args if a.startswith("--config_path="))
        cfg_path = cfg_arg.split("=", 1)[1]
        cfg = _json.loads(Path(cfg_path).read_text())
        assert cfg["mode"] == "record"
        assert cfg["dataset"]["repo_id"] == "user/sim_demo"
        assert cfg["dataset"]["num_episodes_to_record"] == 3

    def test_replay_env_passes_replay_episode(self):
        """Sim replay sets dataset.replay_episode (gym_manipulator's
        DatasetConfig field), not --dataset.episode like the real-robot path."""
        import json as _json
        from pathlib import Path

        captured_args = []

        async def go():
            req = ReplayRequest(env=_ENV_KEYBOARD, repo_id="user/sim_demo", episode=2)
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
                patch("lerobot.gui.api.run._detect_linux_gamepad", return_value=True),
            ):
                await start_replay(req)

        asyncio.run(go())
        cfg_arg = next(a for a in captured_args if a.startswith("--config_path="))
        cfg = _json.loads(Path(cfg_arg.split("=", 1)[1]).read_text())
        assert cfg["mode"] == "replay"
        assert cfg["dataset"]["replay_episode"] == 2

    def test_real_robot_path_still_works(self):
        """Regression guard: existing real-robot dispatch is untouched
        when the request omits `env` and provides `robot`+`teleop`."""
        captured_args = []

        async def go():
            req = TeleoperateRequest(robot=_ROBOT, teleop=_TELEOP, fps=30)
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._launch_subprocess", _make_fake_launch(captured_args)),
            ):
                await start_teleoperate(req)

        asyncio.run(go())
        assert captured_args[0] == "lerobot-teleoperate"
        assert not any("gym_manipulator" in a for a in captured_args)

    def test_gamepad_task_without_controller_rejected_at_endpoint(self):
        """Defensive: even if frontend skips the env-editor warning, the
        teleop endpoint refuses gamepad tasks when no controller present."""

        async def go():
            req = TeleoperateRequest(env=_ENV_GAMEPAD, fps=10)
            with (
                patch("lerobot.gui.api.run._active_process", None),
                patch("lerobot.gui.api.run._detect_linux_gamepad", return_value=False),
            ):
                with pytest.raises(HTTPException) as exc:
                    await start_teleoperate(req)
                assert exc.value.status_code == 400
                assert "Keyboard" in str(exc.value.detail)

        asyncio.run(go())
