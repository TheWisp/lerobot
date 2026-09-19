"""Tests for the training image status endpoint helpers."""

import asyncio

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
    """Image sha not in local history and no local image to recover it from:
    commits_behind must be None, not a bogus number."""
    from lerobot.gui.training import recipes

    # A tag whose sha cannot exist in history, and no local image (no OCI
    # revision label to fetch by).
    monkeypatch.setattr(recipes, "DEFAULT_IMAGE", "ghcr.io/x/lerobot-training:gone-branch-deadbeef")
    monkeypatch.setattr(training_mod, "_docker_image_inspect", lambda tag: {})
    monkeypatch.setattr(training_mod, "_local_image_created", lambda tag: None)
    status = training_mod.get_image_status()
    assert status["git"]["image_commit"] == "deadbeef"
    assert status["git"]["commits_behind"] is None


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


def test_build_reuses_the_layer_cache_by_default():
    """The default path must not pass --no-cache.

    The Dockerfile split only pays off if the cache is actually consulted, so
    the absence of this flag is part of the performance contract, not a detail.
    """
    argv = training_mod._image_build_argv()
    assert argv[:2] == ["docker", "build"]
    assert "--no-cache" not in argv
    assert argv[-1] == "."


def test_full_rebuild_opts_out_of_the_cache_explicitly():
    argv = training_mod._image_build_argv(force_full_rebuild=True)
    assert argv[:3] == ["docker", "build", "--no-cache"]
    assert argv.count("--no-cache") == 1


def test_build_argv_keeps_the_provenance_label():
    """The OCI revision label is how the GUI reports which commit an image was
    built from; routing the argv through a helper must not drop it."""
    argv = training_mod._image_build_argv(label_args=["--label", "org.opencontainers.image.revision=abc"])
    assert "--label" in argv
    assert "org.opencontainers.image.revision=abc" in argv
    assert argv.index("--label") < argv.index("-t")


def test_build_image_endpoint_forwards_the_full_rebuild_choice(monkeypatch, tmp_path):
    """The JSON option must cross the endpoint and reach the build task."""
    from lerobot.gui.training import recipes

    received = []

    async def fake_build(repo_root, force_full_rebuild=False):
        received.append((repo_root, force_full_rebuild))

    monkeypatch.setattr(recipes, "docker_available", lambda: True)
    monkeypatch.setattr(training_mod, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(training_mod, "_run_image_build", fake_build)
    training_mod._build_task = None

    async def exercise_endpoint():
        response = await training_mod.build_image(training_mod.BuildImageRequest(force_full_rebuild=True))
        await training_mod._build_task
        return response

    try:
        response = asyncio.run(exercise_endpoint())
    finally:
        training_mod._build_task = None

    assert response["force_full_rebuild"] is True
    assert received == [(tmp_path, True)]


def test_build_image_endpoint_defaults_to_incremental(monkeypatch, tmp_path):
    """An older client that sends no body must not trigger a full rebuild."""
    from lerobot.gui.training import recipes

    received = []

    async def fake_build(repo_root, force_full_rebuild=False):
        received.append((repo_root, force_full_rebuild))

    monkeypatch.setattr(recipes, "docker_available", lambda: True)
    monkeypatch.setattr(training_mod, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(training_mod, "_run_image_build", fake_build)
    training_mod._build_task = None

    async def exercise_endpoint():
        response = await training_mod.build_image(None)
        await training_mod._build_task
        return response

    try:
        response = asyncio.run(exercise_endpoint())
    finally:
        training_mod._build_task = None

    assert response["force_full_rebuild"] is False
    assert received == [(tmp_path, False)]
