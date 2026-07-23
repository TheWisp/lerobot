"""Tests for the training image status endpoint helpers."""

import lerobot.gui.api.training as training_mod


def test_repo_root_found_in_checkout():
    """This repo IS a git checkout, so the dev-machine path must resolve."""
    root = training_mod._repo_root()
    assert root is not None
    assert (root / ".git").exists()


def test_image_status_without_git_checkout(monkeypatch):
    """pip-installed GUI (no .git): git section must be None so the frontend
    hides freshness — there is no local history to compare against."""
    monkeypatch.setattr(training_mod, "_repo_root", lambda: None)
    monkeypatch.setattr(training_mod, "_local_image_created", lambda tag: None)
    status = training_mod.get_image_status()
    assert status["git"] is None
    assert status["image"]  # tag still reported


def test_image_status_unknown_image_commit(monkeypatch):
    """Image sha not in local history (e.g. deleted branch): commits_behind
    must be None, not a bogus number."""
    monkeypatch.setattr(training_mod, "_local_image_created", lambda tag: None)
    status = training_mod.get_image_status()
    # The current DEFAULT_IMAGE tag's sha (e6bf147) is not in this repo.
    assert status["git"]["image_commit"] is not None
    assert status["git"]["commits_behind"] is None
    assert status["git"]["branch"] is not None


def test_image_status_known_image_commit(monkeypatch):
    """When the tag's sha IS in local history, compute commits-behind."""
    import subprocess

    head = subprocess.run(
        ["git", "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    monkeypatch.setattr(
        training_mod, "DEFAULT_IMAGE", f"ghcr.io/x/lerobot-training:some-branch-{head}", raising=False
    )
    # DEFAULT_IMAGE is imported lazily inside get_image_status; patch recipes instead.
    from lerobot.gui.training import recipes

    monkeypatch.setattr(recipes, "DEFAULT_IMAGE", f"ghcr.io/x/lerobot-training:some-branch-{head}")
    monkeypatch.setattr(training_mod, "_local_image_created", lambda tag: None)
    status = training_mod.get_image_status()
    assert status["git"]["image_commit"] == head
    assert status["git"]["commits_behind"] == 0
    assert status["git"]["image_commit_date"] is not None
