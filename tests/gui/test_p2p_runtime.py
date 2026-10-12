# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""A changed dependency must not silently become the live tracker."""

import importlib.util
import json
import subprocess
import sys

import pytest

from lerobot import p2p_runtime as runtime


@pytest.fixture
def source(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    runtime.git(repo, "config", "user.name", "Test")
    runtime.git(repo, "config", "user.email", "test@example.invalid")
    (repo / "pose.py").write_text("fixed = True\n")
    (repo / ".gitignore").write_text("/checkpoints\n")
    runtime.git(repo, "add", ".")
    runtime.git(repo, "commit", "-qm", "baseline")
    return repo, {"schema": 1, "repository": str(repo), "commit": runtime.git(repo, "rev-parse", "HEAD")}


def test_exact_clean_commit_is_required(source):
    repo, lock = source
    assert runtime.verify_checkout(repo, lock) == lock["commit"]
    with pytest.raises(RuntimeError, match="does not match pinned"):
        runtime.verify_checkout(repo, {**lock, "commit": "0" * 40})


@pytest.mark.parametrize("path", ["pose.py", "new_module.py"])
def test_tracked_and_untracked_source_changes_are_rejected(source, path):
    repo, lock = source
    (repo / path).write_text("fixed = False\n")
    with pytest.raises(RuntimeError, match="local changes"):
        runtime.verify_checkout(repo, lock)


def test_setup_pins_remote_source_and_never_overwrites_a_modified_checkout(source, tmp_path):
    spec = importlib.util.spec_from_file_location("setup_p2p", runtime.ROOT / "benchmarks/setup_p2p.py")
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    _, lock = source
    assets = tmp_path / "assets"
    assets.mkdir()
    checkout = setup.prepare(lock, tmp_path / "cache", assets)
    assert runtime.verify_checkout(checkout, lock) == lock["commit"]
    assert (checkout / "checkpoints").resolve() == assets
    assert setup.prepare(lock, tmp_path / "cache", assets) == checkout
    (checkout / "pose.py").write_text("my experiment\n")
    with pytest.raises(RuntimeError, match="local changes"):
        setup.prepare(lock, tmp_path / "cache", assets)
    assert (checkout / "pose.py").read_text() == "my experiment\n"


def test_bridge_refuses_unknown_commit_before_loading_models(source, tmp_path):
    repo, _ = source
    result = subprocess.run(
        [
            sys.executable,
            str(runtime.ROOT / "benchmarks/p2p_bridge.py"),
            "--repo",
            str(repo),
            "--config",
            str(tmp_path / "unused.yaml"),
        ],
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode != 0
    assert "does not match pinned" in result.stderr
    assert not result.stdout, "No ready message may precede verification"


def test_manifest_requires_an_immutable_commit(tmp_path):
    path = tmp_path / "lock.json"
    path.write_text(json.dumps({"schema": 1, "repository": "example", "commit": "main"}))
    with pytest.raises(ValueError, match="Invalid P2P commit"):
        runtime.load_lock(path)


def test_override_changes_location_but_cannot_bypass_commit_validation(source, monkeypatch):
    repo, lock = source
    monkeypatch.setenv("LEROBOT_P2P_REPO", str(repo))
    assert runtime.checkout_path() == repo
    with pytest.raises(RuntimeError, match="does not match pinned"):
        runtime.verify_checkout(runtime.checkout_path(), {**lock, "commit": "0" * 40})
