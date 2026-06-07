# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the training-recipe builder.

Pure unit tests — verify argv composition for both the fake fallback and
the real ``docker run`` recipe. No docker invocation; the real-recipe path
is validated end-to-end via the live smoke (see scripts/training/README.md
"2026-06-07 verification").
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from lerobot.gui.training_recipes import (
    CONTAINER_HF_CACHE,
    CONTAINER_OUTPUT_SUBDIR,
    CONTAINER_RUNS_MOUNT,
    DEFAULT_IMAGE,
    FAKE_RECIPE_MARKER,
    build_lerobot_train_command,
    docker_available,
    is_fake_recipe,
    output_subdir_in_run,
)
from lerobot.gui.training_runs import Run, RunPaths, RunState, new_run_id


def _make_run(args: dict) -> Run:
    return Run(
        run_id=new_run_id(),
        host_id="this-server",
        recipe_name="act-default",
        dataset_id="lerobot/pusht",
        args=args,
        state=RunState.PENDING,
        created_at=time.time(),
    )


# ── Fake recipe fallback ──────────────────────────────────────────────────────


def test_is_fake_recipe_detects_marker() -> None:
    assert is_fake_recipe(_make_run({"__recipe__": FAKE_RECIPE_MARKER}))
    assert not is_fake_recipe(_make_run({}))
    assert not is_fake_recipe(_make_run({"policy.type": "act"}))


def test_output_subdir_for_fake_is_empty() -> None:
    assert output_subdir_in_run(_make_run({"__recipe__": FAKE_RECIPE_MARKER})) == ""


def test_output_subdir_for_docker_is_output() -> None:
    assert output_subdir_in_run(_make_run({"policy.type": "act"})) == CONTAINER_OUTPUT_SUBDIR


def test_fake_recipe_emits_python_runner_argv(tmp_path: Path) -> None:
    paths = RunPaths.for_run("test", runs_dir=tmp_path)
    paths.ensure_exists()
    run = _make_run({"__recipe__": FAKE_RECIPE_MARKER, "num_steps": 5, "save_every": 2})
    cmd, env = build_lerobot_train_command(run, paths)
    # Skips meta marker; emits the rest as kebab-cased --flag value
    assert cmd[1:4] == ["-m", "lerobot.gui.training_runner", "--run-dir"]
    assert "--num-steps" in cmd
    assert "5" in cmd
    assert "--save-every" in cmd
    assert "2" in cmd
    assert env == {}
    # Meta marker doesn't leak into the argv
    assert "--__recipe__" not in cmd
    assert FAKE_RECIPE_MARKER not in cmd


# ── Docker recipe ─────────────────────────────────────────────────────────────


def _docker_cmd(run: Run, paths: RunPaths) -> list[str]:
    cmd, _ = build_lerobot_train_command(run, paths)
    return cmd


def test_docker_recipe_command_shape(tmp_path: Path) -> None:
    paths = RunPaths.for_run("abc123", runs_dir=tmp_path)
    paths.ensure_exists()
    run = _make_run({"policy.type": "act", "dataset.repo_id": "lerobot/pusht", "steps": 5})
    cmd = _docker_cmd(run, paths)
    # docker run prefix
    assert cmd[0:2] == ["docker", "run"]
    # GPU passthrough
    assert "--gpus" in cmd and "all" in cmd
    # User UID/GID (whatever the test process is running as)
    user_idx = cmd.index("--user")
    assert cmd[user_idx + 1] == f"{os.getuid()}:{os.getgid()}"
    # Image is positional, just before "lerobot-train"
    assert "lerobot-train" in cmd
    train_idx = cmd.index("lerobot-train")
    assert cmd[train_idx - 1] == DEFAULT_IMAGE


def test_docker_recipe_bind_mounts(tmp_path: Path) -> None:
    paths = RunPaths.for_run("abc123", runs_dir=tmp_path)
    paths.ensure_exists()
    run = _make_run({"policy.type": "act"})
    cmd = _docker_cmd(run, paths)
    # HF cache mount
    hf_host = os.path.expanduser("~/.cache/huggingface")
    assert f"{hf_host}:{CONTAINER_HF_CACHE}" in cmd
    # Run dir mount
    assert f"{paths.root}:{CONTAINER_RUNS_MOUNT}" in cmd


def test_docker_recipe_forces_safety_flags(tmp_path: Path) -> None:
    paths = RunPaths.for_run("xyz", runs_dir=tmp_path)
    run = _make_run({"policy.type": "act"})
    cmd = _docker_cmd(run, paths)
    # All three workflow-identified must-have flags
    assert "--policy.push_to_hub=false" in cmd
    assert f"--policy.repo_id=local/{run.run_id}" in cmd
    assert "--wandb.enable=false" in cmd
    assert "--save_checkpoint=true" in cmd
    # output_dir locked to /runs/<subdir> (host can read via bind-mount)
    assert f"--output_dir={CONTAINER_RUNS_MOUNT}/{CONTAINER_OUTPUT_SUBDIR}" in cmd


def test_docker_recipe_user_flag_wins_for_non_critical_overrides(tmp_path: Path) -> None:
    """If the user sets push_to_hub=true explicitly we let them (might want
    to push to a personal repo). But output_dir is always forced inside the
    bind-mount — that's a correctness invariant, not a preference."""
    paths = RunPaths.for_run("xyz", runs_dir=tmp_path)
    run = _make_run(
        {
            "policy.type": "act",
            "policy.push_to_hub": True,  # user wants to push
            "output_dir": "/somewhere/else",  # user wants different output dir
        }
    )
    cmd = _docker_cmd(run, paths)
    # User's push_to_hub=true wins
    assert "--policy.push_to_hub=true" in cmd
    assert "--policy.push_to_hub=false" not in cmd
    # But output_dir is always forced (silent override)
    assert f"--output_dir={CONTAINER_RUNS_MOUNT}/{CONTAINER_OUTPUT_SUBDIR}" in cmd
    assert "--output_dir=/somewhere/else" not in cmd


def test_docker_recipe_translates_arg_types(tmp_path: Path) -> None:
    paths = RunPaths.for_run("xyz", runs_dir=tmp_path)
    run = _make_run(
        {
            "policy.type": "act",
            "policy.use_vae": False,  # bool → "false"
            "policy.optimizer_betas": [0.9, 0.999],  # list → "[0.9,0.999]"
            "policy.chunk_size": 100,  # int → "100"
        }
    )
    cmd = _docker_cmd(run, paths)
    assert "--policy.use_vae=false" in cmd
    assert "--policy.optimizer_betas=[0.9,0.999]" in cmd
    assert "--policy.chunk_size=100" in cmd


def test_docker_recipe_image_override(tmp_path: Path) -> None:
    paths = RunPaths.for_run("xyz", runs_dir=tmp_path)
    run = _make_run({"policy.type": "act", "__image__": "myreg/my-custom:dev"})
    cmd = _docker_cmd(run, paths)
    assert "myreg/my-custom:dev" in cmd
    assert DEFAULT_IMAGE not in cmd
    # Meta marker doesn't become a flag
    assert "--__image__=myreg/my-custom:dev" not in cmd


def test_docker_recipe_drops_meta_markers(tmp_path: Path) -> None:
    """Any double-underscore-prefixed arg is for the recipe builder, not the
    train CLI."""
    paths = RunPaths.for_run("xyz", runs_dir=tmp_path)
    run = _make_run({"policy.type": "act", "__some_marker__": "x", "__another__": True})
    cmd = _docker_cmd(run, paths)
    for tok in cmd:
        assert "__" not in tok or tok.endswith(".pyc"), f"unexpected meta token: {tok}"


# ── docker_available probe ────────────────────────────────────────────────────


def test_docker_available_truthy_when_docker_on_path() -> None:
    # On this dev host, docker is installed (verified by C0 smoke)
    # In CI / sandboxes without docker, this would return False.
    # We don't strictly assert True — but assert it returns a bool.
    result = docker_available()
    assert isinstance(result, bool)


def test_docker_available_false_without_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lerobot.gui.training_recipes.shutil.which", lambda _: None)
    assert docker_available() is False
