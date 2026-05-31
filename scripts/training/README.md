# Training-deploy prototype

Demonstrates the "deploy an image, run training" path end-to-end on
localhost. The same mechanism will extend to a remote SSH host in
phase 2 (see `gui/docs/model_training.md`, draft pending).

## What's here

| File                               | Purpose                                                                                            |
| ---------------------------------- | -------------------------------------------------------------------------------------------------- |
| `../../docker/Dockerfile.training` | Image: CUDA + Python + uv + lerobot, training extras only, `tini` as PID 1 for SIGTERM-clean exec. |
| `install_prereqs.sh`               | One-shot installer for Docker Engine + nvidia-container-toolkit (Ubuntu 22.04 / 24.04).            |
| `deploy_local.sh`                  | Builds the image if needed, runs a container with `--gpus all`, executes a training command.       |

## One-time setup

```bash
# Install Docker + nvidia-container-toolkit (idempotent; safe to re-run)
sudo bash scripts/training/install_prereqs.sh

# Log out and back in (or `newgrp docker`) so the docker group takes effect.
```

## Run a training

The default invocation runs a 10-step ACT smoke test on `lerobot/pusht`:

```bash
bash scripts/training/deploy_local.sh
```

First run builds the image (~3–5 min for the dep layer; cached afterwards).
Subsequent runs skip the build unless `pyproject.toml` has changed.

To run an actual training, forward the command directly:

```bash
bash scripts/training/deploy_local.sh \
    lerobot-train \
        --policy.type=act \
        --dataset.repo_id=thewisp/pick_place_black_king_jan_11 \
        --batch_size=8 --steps=10000 --save_freq=1000 \
        --output_dir=/lerobot/outputs/act_pick_place \
        --job_name=act_pick_place \
        --policy.device=cuda \
        --wandb.enable=true
```

Anything after `deploy_local.sh` is forwarded verbatim into the container.

## What the deploy script does, in order

1. **Probe prerequisites** — checks `docker`, `nvidia-smi`, docker daemon
   reachable, and that `--gpus all` actually works inside a container.
   Prints a clear remediation message + exits if anything is missing,
   no silent fallback.
2. **Build the image if needed** — builds `lerobot-training:dev` from
   `docker/Dockerfile.training`. Skips the build if the image is newer
   than `pyproject.toml`.
3. **Run the container** with:
   - `--gpus all` for GPU pass-through (effectively zero overhead vs bare metal)
   - `--rm` so the container disappears on exit
   - `-v ~/.cache/huggingface:/home/user_lerobot/.cache/huggingface` so
     HF datasets, models, and the auth token are shared with the host
   - `-v $REPO/outputs:/lerobot/outputs` so checkpoints survive container exit
   - `-e HF_TOKEN=...` (if a host-side token exists) so the container
     can push to Hub without needing a separate `huggingface-cli login`
4. **Forward the training command** to the container as the entrypoint
   argument. `tini` as PID 1 ensures SIGTERM reaches the Python process.

## Design notes

- **No bare-pip install path here.** The image is the only deployment
  unit; if Docker isn't present, the script tells you what to install.
  This mirrors what we expect for the cloud-SSH backend in phase 2.
- **HF cache is shared with the host.** Datasets aren't re-downloaded
  between container runs. A 50 GB dataset takes one hit; reruns are
  instant. (On a cloud pod this same role is filled by the
  provider's persistent volume mounted at the same path.)
- **Network = host.** Avoids Docker's bridge overhead for HF Hub
  traffic; on localhost there's no isolation benefit to keeping the
  container on a separate network.
- **tini as PID 1.** Required for SIGTERM to actually reach the Python
  process — see preemption research in `gui/docs/`. Without it, `bash
-c "python ..."` would eat the signal.
