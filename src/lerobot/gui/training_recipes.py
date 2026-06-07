# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Recipe builder — composes the `docker run … lerobot-train …` argv from a Run.

Image-everywhere (DESIGN.md § Unified execution): every training run is
``docker run training-image lerobot-train ...``, on every host mode. The
recipe builder produces that argv (plus an env dict) from a Run's args dict.

Run.args convention (flat dict, dotted keys):

    {
        "policy.type": "act",
        "policy.chunk_size": 100,
        "dataset.repo_id": "lerobot/pusht",
        "steps": 5,
        "batch_size": 8,
        "save_freq": 5,
    }

Each entry becomes a ``--key=value`` flag on the lerobot-train command line.

Forced flags (these are always emitted, regardless of what's in args; they
defend against the verified-by-smoke gotchas):

  --policy.push_to_hub=false       — lerobot-train requires push_to_hub OR
                                     a valid repo_id; we disable Hub push.
  --policy.repo_id=local/<run_id>  — required even when push_to_hub=false
                                     (validate() at end of run checks it).
  --wandb.enable=false             — no tracking in v1.
  --save_checkpoint=true           — explicit; checkpoints are how we close
                                     the felt loop in C3.
  --output_dir=/runs/output        — fixed subdir of the bind-mount. Must
                                     not pre-exist (lerobot-train refuses
                                     to overwrite). Bind-mount target /runs
                                     already exists; /runs/output does not.

For backwards-compat, ``Run.args["__recipe__"] == "__fake__"`` returns the
old python-only fake-training-runner command. Used by the orchestrator's
existing unit tests to keep them fast (no docker dependency).
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Any

from lerobot.gui.training_runs import Run, RunPaths

# Pinned image tag — bumped explicitly via PR. ``latest`` is only published
# on main; per-branch builds publish ``<branch>-<sha>``. This default points
# at the latest verified-by-smoke build. Override per-run via
# Run.args["__image__"].
DEFAULT_IMAGE = "ghcr.io/thewisp/lerobot-training:feat-gui-training-deploy-proto-2808d5e"

# Marker that selects the fake-training runner instead of real lerobot-train.
# Used by orchestrator unit tests so they don't depend on docker.
FAKE_RECIPE_MARKER = "__fake__"

# Inside-container paths. The bind-mounts in the docker command line map
# host paths to these.
CONTAINER_RUNS_MOUNT = "/runs"
CONTAINER_OUTPUT_SUBDIR = "output"  # /runs/output — lerobot-train writes here
CONTAINER_HF_CACHE = "/home/user_lerobot/.cache/huggingface"


def is_fake_recipe(run: Run) -> bool:
    """Whether this Run uses the fake-training fallback path (no docker)."""
    return run.args.get("__recipe__") == FAKE_RECIPE_MARKER


def output_subdir_in_run(run: Run) -> str:
    """Per-run output subdir name relative to the run's root.

    The orchestrator uses this to know where lerobot-train wrote checkpoints
    (host-side, via the bind-mount): ``paths.root / output_subdir_in_run(run)``.

    Fake recipe writes to ``paths.root`` directly (no subdir) for backwards
    compat with the existing unit tests.
    """
    return "" if is_fake_recipe(run) else CONTAINER_OUTPUT_SUBDIR


def build_lerobot_train_command(run: Run, paths: RunPaths) -> tuple[list[str], dict[str, str]]:
    """Return ``(command_argv, extra_env)`` for a Run.

    For docker recipes: composes ``docker run --gpus all --user UID:GID
    -v HF:HF -v paths.root:/runs IMAGE lerobot-train --key=value ...`` with
    all the forced flags above.

    For the fake recipe (``__recipe__=__fake__``): returns the legacy
    ``python -m lerobot.gui.training_runner`` argv.

    Pre: ``paths.root`` exists (created by RunPaths.ensure_exists()).
    Post: returned argv is ready for ``subprocess.Popen``; env dict should
    be merged on top of ``os.environ``.
    """
    if is_fake_recipe(run):
        return _build_fake_command(run, paths)
    return _build_docker_command(run, paths)


def docker_available() -> bool:
    """Cheap probe used at run start to fail loudly if the docker recipe
    is requested but docker isn't installed."""
    return shutil.which("docker") is not None


# ── Fake-training fallback ────────────────────────────────────────────────────


def _build_fake_command(run: Run, paths: RunPaths) -> tuple[list[str], dict[str, str]]:
    cmd = [sys.executable, "-m", "lerobot.gui.training_runner", "--run-dir", str(paths.root)]
    for k, v in run.args.items():
        if k.startswith("__"):
            continue  # skip meta markers
        cmd.extend([f"--{k.replace('_', '-')}", _fmt_arg(v)])
    return cmd, {}


# ── Docker recipe ─────────────────────────────────────────────────────────────


# Flags the recipe builder always forces — see the module docstring.
_FORCED_FLAGS: dict[str, str] = {
    "policy.push_to_hub": "false",
    "wandb.enable": "false",
    "save_checkpoint": "true",
    "output_dir": f"{CONTAINER_RUNS_MOUNT}/{CONTAINER_OUTPUT_SUBDIR}",
}


def _build_docker_command(run: Run, paths: RunPaths) -> tuple[list[str], dict[str, str]]:
    image = run.args.get("__image__") or DEFAULT_IMAGE
    hf_cache_host = os.path.expanduser("~/.cache/huggingface")

    # Translate Run.args → lerobot-train --key=value flags
    train_args: list[str] = []
    seen: set[str] = set()
    # User-supplied flags first
    for k, v in run.args.items():
        if k.startswith("__"):
            continue
        if k in _FORCED_FLAGS:
            # User explicitly set a flag we'd otherwise force — let them win
            # for everything EXCEPT output_dir (which has to live inside the
            # bind-mount or the host can't read the checkpoints).
            if k == "output_dir":
                continue
            train_args.append(f"--{k}={_fmt_arg(v)}")
            seen.add(k)
            continue
        train_args.append(f"--{k}={_fmt_arg(v)}")
        seen.add(k)

    # Forced flags — emit only if user didn't already provide
    forced = dict(_FORCED_FLAGS)
    forced["policy.repo_id"] = f"local/{run.run_id}"
    for k, v in forced.items():
        if k in seen:
            continue
        train_args.append(f"--{k}={v}")

    docker_argv = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{hf_cache_host}:{CONTAINER_HF_CACHE}",
        "-v",
        f"{paths.root}:{CONTAINER_RUNS_MOUNT}",
        image,
        "lerobot-train",
        *train_args,
    ]
    return docker_argv, {}


def _fmt_arg(v: Any) -> str:
    """Format a Python value for lerobot-train's draccus CLI parser.

    Booleans → 'true' / 'false' (draccus accepts both).
    Lists / tuples → '[a,b,c]' (draccus syntax).
    Everything else → str().
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_fmt_arg(x) for x in v) + "]"
    return str(v)
