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

## Policies in the GUI: what gets shown and how to add one

### How the picker is populated

The Training form's Policy dropdown is **hand-curated**, not auto-introspected. Every policy that the GUI knows how to launch lives in a single dict in [`src/lerobot/gui/static/training.js`](../../src/lerobot/gui/static/training.js):

```js
const POLICY_FORMS = {
  act: {
    label: "ACT (Action Chunking Transformer)",
    // no `recipe` → defaults to lerobot-train (draccus CLI)
    fields: [
      {
        key: "policy.chunk_size",
        label: "Chunk size",
        type: "int",
        default: 100,
      },
      { key: "policy.use_vae", label: "Use VAE", type: "bool", default: true },
      // ...
    ],
  },
  diffusion: {
    /* same shape */
  },
  hvla_flow_s1: {
    label: "HVLA Flow Matching S1 (no S2)",
    recipe: "hvla_flow_s1", // routes to a non-lerobot-train recipe builder
    fields: [
      { key: "chunk_size", label: "Chunk size", type: "int", default: 50 },
      {
        key: "num_inference_steps",
        label: "Inference steps",
        type: "int",
        default: 15,
      },
      // snake_case keys for HVLA's argparse
    ],
  },
};
```

`Object.entries(POLICY_FORMS)` populates the dropdown. The form generator (`trainingRenderPolicyFields(policyType)`) emits an `<input>` / `<select>` / checkbox per field; the submit handler (`trainingSubmitStart`) collects values into a flat `args` dict that the backend recipe builder turns into argv.

**Why hand-curated?** lerobot's `PreTrainedConfig.ChoiceRegistry` already knows every registered policy and each is an annotated dataclass — auto-introspection is feasible. We're not doing it yet because:

- For 3 policies the hand-curation is ~80 LOC; the generic dataclass-to-form renderer + a backend introspection endpoint would be ~800 LOC.
- Choosing which fields to expose is a curation decision anyway — most config classes have 30+ fields, only ~6 are useful as form inputs. A renderer that surfaces them all is worse UX than 6 hand-picked ones.

There's a comment in `training.js` flagging **policy #4 as the trigger point** — that's when the hand-curation cost likely crosses the introspection cost.

### Adding a new policy: checklist

Concrete steps, scaled to whether the policy runs under `lerobot-train` or has its own trainer.

**A. If the policy is registered with `lerobot-train` (uses draccus + `PreTrainedConfig`)** — most upstream policies. ACT and Diffusion are this shape.

1. Verify the image installs the policy's extras. If the policy's `pyproject.toml` entry under `[project.optional-dependencies]` lists something the image doesn't already have, add it to the `uv sync --extra ...` line in [`docker/Dockerfile.training`](../../docker/Dockerfile.training). (This is the gap that bit Diffusion until C5 — the form picker exposed it before the image had `diffusers`.)
2. Add an entry to `POLICY_FORMS` in [`training.js`](../../src/lerobot/gui/static/training.js). Copy defaults verbatim from the policy's `configuration_<policy>.py`. Use dotted keys (`policy.<field>`) — that's the draccus CLI shape.
3. Bump the cache-buster (`training.js?v=N+1`) in [`index.html`](../../src/lerobot/gui/static/index.html).
4. Done. The recipe builder forwards every `policy.*` flag verbatim to `lerobot-train`, which finds the policy class via its registry.

**B. If the policy has its own trainer script** (HVLA, anything not draccus-based).

1. Steps 1–3 from A — same image-extras + form entry + cache-buster.
2. Add `recipe: "<marker>"` to the `POLICY_FORMS` entry so the frontend tags submissions with `__recipe__=<marker>`.
3. In [`src/lerobot/gui/training/recipes.py`](../../src/lerobot/gui/training/recipes.py), add:
   - a recipe marker constant (`MY_RECIPE = "my_recipe"`),
   - an entrypoint list (`MY_ENTRYPOINT = ["python", "-u", "-m", "my.module.train"]`),
   - a field-to-flag map (`MY_FIELD_TO_FLAG = { "chunk_size": "--chunk-size", ... }`) if the CLI shape differs from draccus's,
   - a `_build_my_command(run, paths)` function that composes the `docker run` argv,
   - a routing branch in `build_lerobot_train_command(run, paths)`:
     ```python
     if is_my_recipe(run):
         return _build_my_command(run, paths)
     ```
4. If your trainer writes checkpoints under a non-standard layout (HVLA uses `checkpoint-<N>/` instead of `<NNNNNN>/`), extend the regex in `Orchestrator._iter_checkpoint_dirs` — already accepts both today.
5. If `output_dir` semantics differ (HVLA puts checkpoints directly under `output/checkpoints/`; lerobot-train uses `output/checkpoints/<NNNNNN>/`), revisit `output_subdir_in_run()`.

**For both A and B**: add tests.

- Recipe-builder unit tests in [`tests/gui/training/test_recipes.py`](../../tests/gui/training/test_recipes.py) assert the docker argv shape, that the right flags get forwarded, and that user-supplied flags don't override safety flags (`output_dir`, etc.).
- A live form-driven smoke ([`scripts/training/screenshots/`](screenshots/) has the working ones) — start a run with `steps=10, save_freq=5`, wait for completion, verify 2 checkpoints land with sha256s matching what the orchestrator wrote.

### How hyperparameters round-trip

1. **Form submit** → `args` dict built from POLICY_FORMS fields + the shared TRAINING_FIELDS (steps/batch_size/save_freq) + `dataset.repo_id` (lerobot-train) or `dataset_repo_id` (HVLA) + `__recipe__` marker if non-default.
2. **POST `/api/training/runs`** with the args dict.
3. **Recipe builder** turns args → argv. For lerobot-train: `args["policy.chunk_size"]=100` becomes `--policy.chunk_size=100`. For HVLA: `args["chunk_size"]=50` becomes `--chunk-size 50` (via `HVLA_FLOW_S1_FIELD_TO_FLAG`).
4. **Orchestrator** spawns `docker run … <image> <entrypoint> <argv>`.
5. **Detail view's Configuration card** re-reads `run.args` and renders it as a key/value table, so the user can see exactly what was passed and click "Run with same config" to re-launch with edits.

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

## 2026-06-07 verification: workstation docker subprocess transport

Validated the full image-everywhere chain on a fresh RTX 5090 Ubuntu 24.04 workstation (no Docker installed). Steps that worked end-to-end (run as the GUI server's host user, not root):

```bash
# 1. Prereqs (one-time)
sudo bash scripts/training/install_prereqs.sh
newgrp docker  # or log out / in

# 2. Pull the image (per-commit tag; latest only published on main)
docker pull ghcr.io/thewisp/lerobot-training:feat-gui-training-deploy-proto-2808d5e

# 3. Real smoke — 5 steps of ACT on lerobot/pusht, GPU + bind-mounts
mkdir -p $HOME/.cache/lerobot/smoke_runs  # must be host-user-owned (see gotcha below)
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -v $HOME/.cache/huggingface:/home/user_lerobot/.cache/huggingface \
  -v $HOME/.cache/lerobot/smoke_runs:/runs \
  ghcr.io/thewisp/lerobot-training:feat-gui-training-deploy-proto-2808d5e \
  lerobot-train \
    --policy.type=act --dataset.repo_id=lerobot/pusht \
    --steps=5 --batch_size=2 --save_freq=5 \
    --policy.push_to_hub=false --policy.repo_id=local/smoke --wandb.enable=false \
    --output_dir=/runs/run_$(date +%s)
```

Confirmed working: `--gpus all` GPU passthrough, lerobot/pusht auto-download, ResNet18 backbone download, ACT policy training (~6 steps/sec on 5090), checkpoint persisted to host at `~/.cache/lerobot/smoke_runs/run_<ts>/checkpoints/000005/pretrained_model/model.safetensors` (~207 MB).

### Additional gotchas surfaced (extend the table above)

| Gap                                                                                                                                                                                                                                                                                               | Workaround used                                                                                         | Fix needed                                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lerobot-train` refuses to start if `--output_dir` already exists (FileExistsError). Combined with docker's bind-mount-auto-creates-dirs behavior, the naïve `mkdir + bind-mount + output_dir=<mount>` flow fails — docker creates the mount dir, then lerobot-train sees it as "already exists". | Bind-mount the parent dir; pass a `--output_dir=<mount>/run_<timestamp>` subdir that doesn't yet exist. | Surface as a setup hint in `run_training.sh` / the orchestrator's recipe builder. The subdir approach is the right pattern long-term anyway (one mount, many runs). |
| Docker's bind-mount auto-creates the host path as root if it doesn't exist. If the host path is then root-owned, the container's UID-1000 user can't write into it, even with `--user $(id -u):$(id -g)`.                                                                                         | Pre-create the mount target via `mkdir -p` as the host user **before** running docker.                  | The orchestrator's spawn step must ensure the runs-dir parent exists with the right ownership before invoking docker. Document it as a setup invariant.             |
| The per-branch image tag follows the pattern `<branch-with-slash-replaced-by-dash>-<sha>`, not `:latest`. `:latest` is only published when merged to main.                                                                                                                                        | Use the per-commit tag explicitly (or use `:main-<sha>` once merged).                                   | Document this in the recipe builder + the run-config UI.                                                                                                            |
