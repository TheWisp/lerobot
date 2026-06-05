# Training-deploy prototype

Demonstrates the "deploy an image, then run trainings" flow end-to-end on localhost. The same mechanism extends to remote SSH hosts later.

> **Looking for the full design + progress tracker?** See [`DESIGN.md`](DESIGN.md) in this directory — phased plan, architecture, cost-optimization, known gaps. This README focuses on _how to use_ the bash scripts today.

## Why two scripts (and not one)

Deploy and training launch are distinct operations:

|                                | Frequency                                        | What it does                                                                      |
| ------------------------------ | ------------------------------------------------ | --------------------------------------------------------------------------------- |
| **`setup_host.sh`** (deploy)   | Once per host, or after a new image is published | Verifies prerequisites, pulls (or builds) the training image, confirms GPU access |
| **`run_training.sh`** (launch) | Many times                                       | Spawns a fresh container with the training command; one container per run         |

This matches every production training stack (Modal `deploy` vs `run`, SageMaker image-registration vs `create-training-job`, Kubeflow operator vs `TFJob` CRD, etc.), and matches the eventual GUI flow where adding a host is separate from clicking Start.

## Image is published from our fork automatically

CI builds [`docker/Dockerfile.training`](../../docker/Dockerfile.training) on every push to `main` and on each release, pushing to GHCR:

```
ghcr.io/thewisp/lerobot-training:latest      ← tracks main
ghcr.io/thewisp/lerobot-training:main-<sha>  ← per-commit, immutable
ghcr.io/thewisp/lerobot-training:<release>   ← release tags
```

Workflow: [`.github/workflows/docker_publish_fork_training.yml`](../../.github/workflows/docker_publish_fork_training.yml). Uses the auto-provisioned `GITHUB_TOKEN` — no secret management.

We don't reuse upstream's `huggingface/lerobot-gpu:latest` because our fork has diverged in both code and `pyproject.toml`/`uv.lock` (200+ files, 170k+ lines). The upstream image's installed packages won't satisfy our code's imports.

**First-time setup (one click in the GitHub UI after CI's first publish):**

Go to `https://github.com/users/thewisp/packages/container/lerobot-training/settings` → **Change visibility** → **Public**. After that, `docker pull` works without auth.

## One-time host setup

```bash
# Install Docker Engine + nvidia-container-toolkit (idempotent, ~5 min)
sudo bash scripts/training/install_prereqs.sh
# Log out and back in (or `newgrp docker`) so docker group takes effect.
```

## Deploy

```bash
bash scripts/training/setup_host.sh
```

Default: pulls `ghcr.io/thewisp/lerobot-training:latest`. ~30s on a fresh host, instant when already cached.

If the pull fails (image not yet published, or GHCR package is still private), the script automatically falls back to building locally from `docker/Dockerfile.training` (~5 min first build). To skip the pull attempt entirely:

```bash
bash scripts/training/setup_host.sh --build-local
```

Other options:

```bash
bash scripts/training/setup_host.sh --tag=main-abc1234   # pin to a specific commit
bash scripts/training/setup_host.sh --image=foo:tag      # any image you specify
```

## Run a training

Default invocation runs a 10-step ACT smoke on `lerobot/pusht`:

```bash
bash scripts/training/run_training.sh
```

To run a real workload (anything after `--` is forwarded into the container verbatim):

```bash
bash scripts/training/run_training.sh -- \
    lerobot-train \
        --policy.type=act \
        --dataset.repo_id=thewisp/pick_place_black_king_jan_11 \
        --batch_size=8 --steps=10000 --save_freq=1000 \
        --output_dir=/lerobot/outputs/act_pick_place \
        --job_name=act_pick_place \
        --policy.device=cuda \
        --wandb.enable=true
```

## Ad-hoc: run with your live local source (without rebuilding the image)

When you're iterating on lerobot code, rebuilding the image on every edit is too slow. `--bind-local` overlays your current `src/lerobot/` on top of the image's installed source. Because the image installs lerobot in editable mode (`pip install -e .`), `import lerobot.X.Y` resolves through the bind-mount and your live edits run immediately:

```bash
bash scripts/training/run_training.sh --bind-local -- \
    lerobot-train --policy.type=act --dataset.repo_id=lerobot/pusht --steps=10
```

What's reused from the image: dependencies (torch, transformers, ffmpeg, etc.), CUDA libs, system packages.
What comes from your working tree: everything under `src/lerobot/`.

**Caveat:** if you edit `pyproject.toml` or `uv.lock`, re-run `setup_host.sh --build-local` to rebuild — bind-mount doesn't reach pip-installed deps. (Or wait for CI to publish a new `:latest` after you push.)

## Architecture in one diagram

```
host                          container (--rm per run)
─────                         ──────────────────────────
src/lerobot/   ───────────►  /lerobot/src/lerobot   (if --bind-local)
~/.cache/hf    ───────────►  /home/user_lerobot/.cache/huggingface
$PWD/outputs   ───────────►  /lerobot/outputs

                              tini (PID 1, via docker run --init)
                                  │
                                  ▼
                              lerobot-train  (your command)
                                  │
                                  ▼
                              CUDA via --gpus all (~0% overhead)
```

`docker run --init` injects Docker's own minimal init (effectively tini) as PID 1 — required so SIGTERM reaches the Python process on `docker stop` or spot preemption.

`--rm` cleans the container on exit; checkpoints survive via the `$PWD/outputs` bind-mount.

`--network host` skips Docker's bridge for HF Hub traffic (faster, no isolation cost on localhost).

## What this proves out for the eventual GUI

- **Image is the single deployment artifact** carrying every Python + CUDA + system dep. Published automatically from our fork via CI.
- **Deploy is decoupled from training launch.** Host profile setup happens once; trainings launch many times against it.
- **Bind-mount path exists for dev iteration** so devs aren't blocked on image rebuilds for code-only changes.
- **`docker run --init` covers signal handling** for clean cancel + spot preemption.
- **HF Hub is both source (datasets) and sink (checkpoints)** via the shared host cache.

## Known gaps + workarounds (discovered during 2026-06-04 verification on Nebius preemptible H100)

The smoke test passed end-to-end (image pulled → GPU pass-through → 10-step ACT training → checkpoint persisted to host bind-mount), but every workaround applied during that run is a real gap the deploy story needs to address. Tracked here so the next person doesn't have to rediscover.

| Gap                                                                                                                                                                                                       | Workaround used                                                                                       | Fix needed                                                                                                                                   |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `lerobot-train` requires `policy.repo_id` and tries to push to HF Hub at the end of every run, regardless of `--wandb.enable`. Unauthenticated runs crash with 401 after training completes successfully. | Pass a placeholder `--policy.repo_id=` and accept the trailing 401.                                   | Add a `--push_to_hub=false` flag to `lerobot-train`, OR have `run_training.sh` auto-detect missing HF_TOKEN and warn + skip.                 |
| HF cache bind-mount fails with `PermissionError` on `/home/user_lerobot/.cache/huggingface/lerobot` — host and container UIDs nominally match but cache subdirs are owner-restricted.                     | Drop the HF cache bind-mount; let the container's cache be ephemeral. Dataset re-downloads every run. | Either build the image with a configurable UID, use a named docker volume, or pre-`chown` the host cache dir during `setup_host.sh`.         |
| `outputs` bind-mount initially read-only to the container's `user_lerobot`.                                                                                                                               | `sudo chmod 777 ~/outputs` before `docker run`.                                                       | Same UID issue as above. `run_training.sh` should set this up automatically.                                                                 |
| `nvidia-container-toolkit` is a **held package** on Nebius's `ubuntu24.04-cuda13.0` image. Plain `apt-get install -y` errors out.                                                                         | `sudo apt-mark unhold nvidia-container-toolkit ...` then install with `--allow-change-held-packages`. | `install_prereqs.sh` must handle held packages.                                                                                              |
| `curl ... \| sudo gpg --dearmor ...` fails over SSH with `cannot open '/dev/tty'`.                                                                                                                        | Download keyring to `/tmp` first, then `gpg --batch --yes --dearmor`.                                 | `install_prereqs.sh` uses the broken pipe pattern; needs the download-then-dearmor variant for non-interactive runs (CI, remote drivers).    |
| `feit` is not in the `docker` group on Nebius (cloud-init's `sudo:` field adds a sudoers entry without changing groups). Every `docker` command needs `sudo`.                                             | Prefixed every `docker` command with `sudo`.                                                          | Either: add `groups: [docker]` to cloud-init users block, OR have `setup_host.sh` detect missing membership and fall back to sudo.           |
| Tini-PID-1 warning: `Dockerfile.training` sets `ENTRYPOINT ["tini", "--"]` AND `docker run --init` injects Docker's own tini → nested tinis, inner one warns about zombie reaping.                        | Ignored (harmless for the smoke).                                                                     | Pick one: drop tini from Dockerfile.training, rely on `--init` from the run command.                                                         |
| `run_training.sh` reads the host's HF_TOKEN locally but never forwards it to the remote SSH host. The smoke didn't need it; real runs that push to Hub will.                                              | n/a for this run.                                                                                     | Forward HF_TOKEN as an env var to the remote container.                                                                                      |
| `--network host` used. Convenient for HF Hub bandwidth but breaks container network isolation.                                                                                                            | Kept. Fine for a training-only pod, but explicit decision.                                            | Document for production deployment.                                                                                                          |
| No way to start/stop/check the Nebius VM from this conversation — VM lifecycle requires the Nebius web console.                                                                                           | Manual via Nebius UI.                                                                                 | Install `nebius` CLI on the dev workstation, OR drive via Terraform from this repo, so an agent (or automated test) can manage VM lifecycle. |

Phase 2 (the SSH backend) reuses all of this — replacing the local `docker run` with `ssh ... 'docker run ...'`, and adding tmux for session survival and a poll loop for status.
