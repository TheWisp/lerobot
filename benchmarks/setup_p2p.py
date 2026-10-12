#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Fetch the pinned P2P source into a new checkout; preserve the running checkout and environment."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lerobot.p2p_runtime import environment_snapshot, load_lock, verify_checkout  # noqa: E402


def prepare(lock: dict, cache: pathlib.Path, assets: pathlib.Path | None = None) -> pathlib.Path:
    target = cache / lock["commit"]
    if target.exists():
        verify_checkout(target, lock)
        return target
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".p2p-", dir=cache) as staging:
        repo = pathlib.Path(staging) / "repo"
        subprocess.run(["git", "clone", "--no-checkout", lock["repository"], str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", lock["commit"]], check=True)
        if assets is not None:
            if not assets.is_dir():
                raise RuntimeError(f"Checkpoint directory missing: {assets}")
            (repo / "checkpoints").symlink_to(assets.resolve(), target_is_directory=True)
        verify_checkout(repo, lock)
        repo.rename(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=pathlib.Path, default=pathlib.Path.home() / ".cache/point2pose/checkouts"
    )
    parser.add_argument(
        "--assets-from",
        type=pathlib.Path,
        default=pathlib.Path.home() / ".cache/point2pose/point-to-pose/checkpoints",
    )
    parser.add_argument(
        "--check", action="store_true", help="Verify existing source without fetching or modifying it"
    )
    parser.add_argument(
        "--snapshot-environment", type=pathlib.Path, help="Write the calling Python's package/source snapshot"
    )
    args = parser.parse_args()
    lock = load_lock()
    target = args.cache / lock["commit"]
    if args.check:
        verify_checkout(target, lock)
    else:
        target = prepare(lock, args.cache, args.assets_from)
    if args.snapshot_environment:
        args.snapshot_environment.write_text(
            json.dumps(environment_snapshot(), indent=2, sort_keys=True) + "\n"
        )
    print(target)


if __name__ == "__main__":
    main()
