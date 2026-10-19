# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Resolve and verify the experimental P2P dependency before loading models."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.parse

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmarks/p2p.lock.json"


def git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE).strip()


def load_lock(path: pathlib.Path = MANIFEST) -> dict:
    lock = json.loads(path.read_text())
    if lock.get("schema") != 1 or not re.fullmatch(r"[0-9a-f]{40}", lock.get("commit", "")):
        raise ValueError(f"Invalid P2P commit lock: {path}")
    if not lock.get("repository"):
        raise ValueError(f"Missing P2P repository: {path}")
    return lock


def python_path() -> str:
    """The interpreter of P2P's own environment, which runs its models (the point groups' view among them)."""
    return os.environ.get(
        "LEROBOT_P2P_PYTHON", str(pathlib.Path.home() / ".cache/point2pose/venv/bin/python")
    )


def checkout_path() -> pathlib.Path:
    override = os.environ.get("LEROBOT_P2P_REPO")
    if override:
        return pathlib.Path(override).expanduser()
    return pathlib.Path.home() / ".cache/point2pose/checkouts" / load_lock()["commit"]


def verify_checkout(repo: pathlib.Path, lock: dict | None = None) -> str:
    """Reject an unknown or edited implementation instead of skipping a repair."""
    lock = load_lock() if lock is None else lock
    try:
        commit = git(repo, "rev-parse", "HEAD")
        if commit != lock["commit"]:
            raise RuntimeError(f"P2P revision {commit} does not match pinned {lock['commit']}")
        changes = git(repo, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none")
        if changes:
            raise RuntimeError(f"P2P checkout has local changes: {changes}")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"P2P checkout unavailable: {repo}; run python benchmarks/setup_p2p.py") from exc
    return commit


def environment_snapshot() -> dict:
    """Record package versions and editable source revisions, including local patches.

    No paths or credentials from package URLs are included in the record.
    This snapshots the existing environment; it does not install anything.
    """
    packages, editable = {}, {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        packages[name] = dist.version
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        if direct.get("dir_info", {}).get("editable"):
            url = urllib.parse.urlsplit(direct.get("url", ""))
            if url.scheme != "file":
                continue
            repo = pathlib.Path(urllib.parse.unquote(url.path))
            try:
                editable[name] = {
                    "commit": git(repo, "rev-parse", "HEAD"),
                    "patch": git(repo, "diff", "HEAD", "--"),
                    "untracked": git(repo, "ls-files", "--others", "--exclude-standard"),
                }
            except (OSError, subprocess.CalledProcessError):
                editable[name] = {"unversioned": True}
    return {
        "python": sys.version.split()[0],
        "packages": dict(sorted(packages.items())),
        "editable": editable,
    }


def provenance(repo: pathlib.Path, config: pathlib.Path, environment: dict) -> dict:
    commit = verify_checkout(repo)
    return {
        "p2p_commit": commit,
        "lerobot_commit": git(ROOT, "rev-parse", "HEAD"),
        "lerobot_dirty": bool(git(ROOT, "status", "--porcelain", "--untracked-files=all")),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "bridge_sha256": hashlib.sha256((ROOT / "benchmarks/p2p_bridge.py").read_bytes()).hexdigest(),
        "environment_sha256": hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest(),
    }
