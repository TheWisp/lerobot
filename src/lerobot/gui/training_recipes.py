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

# Marker that selects the HVLA Flow Matching S1 training script instead of
# lerobot-train. HVLA isn't registered with lerobot-train's draccus policy
# registry — it has its own argparse-based train script with a different CLI
# shape (dashed --key value rather than dotted --key=value). Set via
# Run.args["__recipe__"]; the form does this when "HVLA Flow Matching S1"
# is picked.
HVLA_FLOW_S1_RECIPE = "hvla_flow_s1"

# Inside-container entrypoint for the HVLA Flow Matching S1 trainer.
HVLA_FLOW_S1_ENTRYPOINT = ["python", "-u", "-m", "lerobot.policies.hvla.s1.flow_matching.train"]

# HVLA argparse uses dashed-kebab CLI flag names. Map from the form's
# field key → CLI flag name. (Form keys are clean snake_case so they can
# be re-used by other recipes; per-recipe translation lives here.)
HVLA_FLOW_S1_FIELD_TO_FLAG: dict[str, str] = {
    "dataset_repo_id": "--dataset-repo-id",
    "output_dir": "--output-dir",
    "steps": "--steps",
    "batch_size": "--batch-size",
    "save_freq": "--save-freq",
    "num_workers": "--num-workers",
    "device": "--device",
    "chunk_size": "--chunk-size",
    "num_inference_steps": "--num-inference-steps",
    "rtc_max_delay": "--rtc-max-delay",
    "rtc_drop_prob": "--rtc-drop-prob",
    "max_delay": "--max-delay",
    "resize_images": "--resize-images",
    "hidden_dim": "--hidden-dim",
    "num_decoder_layers": "--num-decoder-layers",
    "s2_latent_path": "--s2-latent-path",  # OMIT to train without S2
}

# Inside-container paths. The bind-mounts in the docker command line map
# host paths to these.
CONTAINER_RUNS_MOUNT = "/runs"
CONTAINER_OUTPUT_SUBDIR = "output"  # /runs/output — lerobot-train writes here
CONTAINER_HF_CACHE = "/home/user_lerobot/.cache/huggingface"


def is_fake_recipe(run: Run) -> bool:
    """Whether this Run uses the fake-training fallback path (no docker)."""
    return run.args.get("__recipe__") == FAKE_RECIPE_MARKER


def is_hvla_flow_s1_recipe(run: Run) -> bool:
    """Whether this Run uses the HVLA Flow Matching S1 training script
    instead of lerobot-train."""
    return run.args.get("__recipe__") == HVLA_FLOW_S1_RECIPE


def output_subdir_in_run(run: Run) -> str:
    """Per-run output subdir name relative to the run's root.

    The orchestrator uses this to know where the worker wrote checkpoints
    (host-side, via the bind-mount): ``paths.root / output_subdir_in_run(run)``.

    Fake recipe writes to ``paths.root`` directly (no subdir) for backwards
    compat with the existing unit tests.

    Real (lerobot-train OR HVLA flow_matching) recipes write to
    ``paths.root / output / ...`` — both honor ``--output-dir /runs/output``
    (HVLA uses dashed form, lerobot-train dotted; same result on disk).
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
    if is_hvla_flow_s1_recipe(run):
        return _build_hvla_flow_s1_command(run, paths)
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


# ── HVLA Flow Matching S1 recipe ──────────────────────────────────────────────


def _build_hvla_flow_s1_command(run: Run, paths: RunPaths) -> tuple[list[str], dict[str, str]]:
    """Compose a `docker run … python -m lerobot.policies.hvla.s1.flow_matching.train …`
    argv for the HVLA Flow Matching S1 training script.

    Differs from the lerobot-train recipe in three ways:
      1. Different entrypoint inside the container.
      2. Dashed-kebab argparse CLI (--key value, space-separated), not
         draccus's --key=value dotted-dataclass form.
      3. None of lerobot-train's safety flags (--policy.push_to_hub etc.)
         apply — HVLA's argparse rejects them.

    S2 conditioning: HVLA trains WITHOUT S2 iff ``--s2-latent-path`` is
    omitted from the CLI. The form leaves it out by default; user can
    opt in via Run.args["s2_latent_path"]="..." (NOT supported via the
    form today; would need an upstream extension).
    """
    image = run.args.get("__image__") or DEFAULT_IMAGE
    hf_cache_host = os.path.expanduser("~/.cache/huggingface")

    # Translate the form's flat snake_case args dict → HVLA's dashed CLI flags.
    train_args: list[str] = []
    for k, v in run.args.items():
        if k.startswith("__"):
            continue
        flag = HVLA_FLOW_S1_FIELD_TO_FLAG.get(k)
        if flag is None:
            # Skip unknown keys — HVLA argparse would error on them. Logged
            # at the orchestrator level if we ever want to surface a warning.
            continue
        # Bool / None / list handling: HVLA argparse expects "true"/"false"
        # for bools (same as draccus); list args aren't part of the schema.
        if v is None:
            continue
        train_args.extend([flag, _fmt_arg(v)])

    # Forced: output-dir always lives inside the bind-mount (host needs to
    # read checkpoints back), and we always omit --s2-latent-path so the
    # S1-without-S2 path is taken — that's the prototype's scope per
    # DESIGN.md (S2 latents extraction is a separate workflow not wired
    # to the GUI yet).
    forced_output_dir = f"{CONTAINER_RUNS_MOUNT}/{CONTAINER_OUTPUT_SUBDIR}"
    if "--output-dir" not in train_args:
        train_args.extend(["--output-dir", forced_output_dir])
    else:
        # User-provided output-dir — refuse to honor it; we MUST control
        # the path so checkpoints land where the host's manifest scanner
        # looks. Replace silently.
        idx = train_args.index("--output-dir")
        train_args[idx + 1] = forced_output_dir
    # Strip any --s2-latent-path the user shoved in via meta marker — until
    # the GUI properly supports the S2 conditioning workflow, we always
    # train S1-only.
    if "--s2-latent-path" in train_args:
        idx = train_args.index("--s2-latent-path")
        del train_args[idx : idx + 2]

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
        *HVLA_FLOW_S1_ENTRYPOINT,
        *train_args,
    ]
    return docker_argv, {}
