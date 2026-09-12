"""The HVLA checkbox must reach the trainer with the same meaning."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from lerobot.gui.api.training import list_policies
from lerobot.gui.training.recipes import HVLA_FLOW_S1_RECIPE, _build_hvla_flow_s1_command
from lerobot.gui.training.runs import Run, RunPaths, RunState
from lerobot.policies.hvla.s1.flow_matching.train import build_arg_parser


def field():
    catalog = list_policies()
    hvla = next(p for p in catalog if p.get("recipe") == HVLA_FLOW_S1_RECIPE)
    for policy in catalog:
        if policy is not hvla:
            assert "state_dropout" not in {f["name"] for f in policy["fields"]}
    return next(f for f in hvla["fields"] if f["name"] == "state_dropout")


@pytest.mark.parametrize("checked", [False, True])
def test_checkbox_to_recipe_to_trainer(tmp_path, checked):
    f = field()
    assert f["type"] == "bool" and f["default"] is False
    assert not f.get("advanced")
    run = Run(
        run_id="state-dropout-test",
        host_id="this-server",
        recipe_name="hvla",
        dataset_id="test/local",
        state=RunState.PENDING,
        created_at=0,
        args={"__recipe__": HVLA_FLOW_S1_RECIPE, "dataset_repo_id": "test/local", f["name"]: checked},
    )
    command, _ = _build_hvla_flow_s1_command(run, RunPaths.for_run(run.run_id, runs_dir=tmp_path))
    module_at = command.index("lerobot.policies.hvla.s1.flow_matching.train")
    args = build_arg_parser().parse_args(command[module_at + 1 :])
    assert args.state_dropout_p == (0.2 if checked else 0)
    assert args.dropout == 0.1  # existing action-expert dropout is independent
    assert args.ball_view is False and args.ball_aux is False


def test_real_frontend_renders_and_reads_checkbox():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the actual training.js")
    result = subprocess.run(
        [node, str(Path(__file__).with_suffix(".test.js"))],
        input=json.dumps(field()),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
