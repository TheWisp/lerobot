"""The GUI builds CLI argv for the `lerobot-*` scripts; this checks it still parses.

The GUI is a *caller* of those scripts across a process boundary, so nothing --
not imports, not type checking, not the scripts' own tests -- connects the flag
names it emits to the config dataclasses that receive them. When an upstream
merge renamed ``DatasetRecordConfig.vcodec`` to ``rgb_encoder.vcodec``, the GUI
kept emitting ``--dataset.vcodec``, every test stayed green, and recording died
at launch with "unrecognized arguments" the first time a human tried it.

So drive each launch endpoint, capture the argv it really produces, and hand it
to the same ``draccus.parse`` the script's own entry point uses. Parsing only --
no subprocess, no hardware. Any renamed, removed, re-nested, or retyped field on
either side of the boundary fails here instead of in front of the user.

Precondition: the robot/teleop profiles below name registered types with
port-shaped fields. Parsing constructs configs but never opens a device.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import draccus
import pytest

from lerobot.configs import parser
from lerobot.gui.api.run import (
    RecordRequest,
    ReplayRequest,
    TeleoperateRequest,
    start_record,
    start_replay,
    start_teleoperate,
)

# Two-arm profile: exercises the nested per-arm flags as well as the top-level ones.
ROBOT_PROFILE = {
    "type": "bi_so107_follower",
    "fields": {
        "id": "test-follower",
        "left_arm_port": "/dev/ttyACM0",
        "right_arm_port": "/dev/ttyACM1",
    },
}
TELEOP_PROFILE = {
    "type": "bi_so107_leader",
    "fields": {
        "id": "test-leader",
        "left_arm_port": "/dev/ttyACM2",
        "right_arm_port": "/dev/ttyACM3",
    },
}


async def _capture_argv(endpoint, request) -> list[str]:
    """Run a launch endpoint with the subprocess stubbed out, returning its argv."""
    captured: list[str] = []

    async def fake_launch(args, **kwargs):
        captured.extend(args)

    with (
        patch("lerobot.gui.api.run._launch_subprocess", side_effect=fake_launch),
        patch("lerobot.gui.api.run._ensure_no_active_process"),
        patch("lerobot.gui.api.run._active_process", new=AsyncMock(pid=4242)),
    ):
        await endpoint(request)

    assert captured, "endpoint launched nothing"
    return captured


def _parse(config_cls, argv: list[str]) -> object:
    """Parse argv exactly the way the script's @parser.wrap entry point does.

    wrap() is not a thin shim over draccus.parse: it strips ``.path`` arguments
    (they are resolved later, from the Hub or disk) before parsing. Skipping that
    step here would fail every policy-mode launch on a flag the real CLI accepts,
    so mirror it -- a guardrail that does not match the production path is worse
    than none.
    """
    cli_args = argv[1:]  # argv[0] is the program
    if parser.has_method(config_cls, "__get_path_fields__"):
        cli_args = parser.filter_path_args(config_cls.__get_path_fields__(), cli_args)
    # Config __post_init__ hooks re-read sys.argv to recover the stripped
    # ``.path`` values, so the real argv has to be in place during the parse.
    with patch.object(sys, "argv", list(argv)):
        return draccus.parse(config_class=config_cls, args=cli_args)


@pytest.mark.asyncio
async def test_teleoperate_argv_parses():
    from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig

    argv = await _capture_argv(
        start_teleoperate,
        TeleoperateRequest(robot=ROBOT_PROFILE, teleop=TELEOP_PROFILE, fps=30),
    )
    assert argv[0] == "lerobot-teleoperate"
    _parse(TeleoperateConfig, argv)


@pytest.mark.asyncio
async def test_record_argv_parses():
    """Regression: the merge re-nested vcodec and the GUI kept emitting the flat flag."""
    from lerobot.scripts.lerobot_record import RecordConfig

    argv = await _capture_argv(
        start_record,
        RecordRequest(
            robot=ROBOT_PROFILE,
            teleop=TELEOP_PROFILE,
            repo_id="test/contract",
            single_task="Do the thing.",
            num_episodes=1,
        ),
    )
    assert argv[0] == "lerobot-record"
    cfg = _parse(RecordConfig, argv)
    # The codec must actually reach the encoder, not just parse into some field.
    assert cfg.dataset.rgb_encoder.vcodec == "libsvtav1"


@pytest.mark.asyncio
async def test_record_argv_parses_in_policy_mode():
    """The policy branch emits a different flag set than the teleop branch."""
    from lerobot.scripts.lerobot_record import RecordConfig

    argv = await _capture_argv(
        start_record,
        RecordRequest(
            robot=ROBOT_PROFILE,
            policy_path="lerobot/pi0",
            repo_id="test/contract-policy",
            single_task="Do the thing.",
            num_episodes=1,
            resume=True,
            root="/tmp/contract-root",
            intervention_repo_id="test/intervention",
        ),
    )
    # Resolving the checkpoint itself would hit the Hub; this test is about the
    # flag surface, so stand in for the fetch and let everything else run real.
    with patch(
        "lerobot.scripts.lerobot_record.PreTrainedConfig.from_pretrained",
        return_value=SimpleNamespace(pretrained_path=None),
    ):
        _parse(RecordConfig, argv)


@pytest.mark.asyncio
async def test_replay_argv_parses():
    from lerobot.scripts.lerobot_replay import ReplayConfig

    argv = await _capture_argv(
        start_replay,
        ReplayRequest(robot=ROBOT_PROFILE, repo_id="test/contract", episode=0),
    )
    assert argv[0] == "lerobot-replay"
    _parse(ReplayConfig, argv)


def test_train_recipe_argv_parses():
    """The training tab is the other live CLI surface, with the same exposure.

    It composes ``docker run … lerobot-train --key=value``; the flags after
    ``lerobot-train`` face ``TrainPipelineConfig`` across the same untyped gap.
    """
    import time
    from pathlib import Path

    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.gui.training.recipes import build_lerobot_train_command
    from lerobot.gui.training.runs import Run, RunPaths, RunState

    run = Run(
        run_id="contract",
        host_id="this-server",
        recipe_name="act-default",
        dataset_id="test/contract",
        args={
            "dataset.repo_id": "test/contract",
            "policy.type": "act",
            "policy.device": "cpu",
            "steps": 100,
            "batch_size": 2,
        },
        state=RunState.PENDING,
        created_at=time.time(),
    )
    argv, _env = build_lerobot_train_command(run, RunPaths(root=Path("/tmp/contract-run"), run_id="contract"))

    assert "lerobot-train" in argv, "recipe stopped emitting a train command"
    train_flags = argv[argv.index("lerobot-train") :]
    _parse(TrainPipelineConfig, train_flags)


@pytest.mark.asyncio
async def test_a_bogus_flag_would_be_caught():
    """Guard the guard: prove parsing actually rejects an unknown flag.

    Without this, a parse that silently tolerated extras would make every test
    above vacuous -- which is precisely how the vcodec break went unnoticed.
    """
    from lerobot.scripts.lerobot_record import RecordConfig

    argv = await _capture_argv(
        start_record,
        RecordRequest(
            robot=ROBOT_PROFILE,
            teleop=TELEOP_PROFILE,
            repo_id="test/contract",
            single_task="Do the thing.",
            num_episodes=1,
        ),
    )
    with pytest.raises(SystemExit):
        _parse(RecordConfig, [*argv, "--dataset.no_such_field=1"])
