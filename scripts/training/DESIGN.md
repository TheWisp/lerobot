# Model Training: cloud-GPU training pipeline integrated into the GUI

End-to-end design for taking a user from "I want to train on a cloud GPU" to "trained model on HF Hub", driven from the LeRobot GUI rather than the CLI.

Companion to [`gui/docs/hub_transfers.md`](../../src/lerobot/gui/docs/hub_transfers.md). Training reuses Hub Transfers' worker-IPC + polling + Hub-upload patterns wholesale.

---

## Status as of 2026-06-05

Branch: [`feat/gui-training-deploy-proto`](https://github.com/TheWisp/lerobot/pull/new/feat/gui-training-deploy-proto)

### Phase 0 — Bash prototype ✅

|                                                                            | Commit                    | Status                                                                        |
| -------------------------------------------------------------------------- | ------------------------- | ----------------------------------------------------------------------------- |
| Docker image (`Dockerfile.training`) + GHCR auto-publish workflow          | `cc5989446`               | ✅ Built, published, smoke-verified                                           |
| Bash scaffolding: `install_prereqs.sh`, `setup_host.sh`, `run_training.sh` | `d69c0341d` → `98436d0e3` | ✅ All 10 gaps from first remote-VM smoke fixed                               |
| First end-to-end smoke test on Nebius preemptible H100                     | (2026-06-04)              | ✅ Image pulled, GPU pass-through, 10-step ACT training, checkpoint persisted |

**Today's working pipeline (CLI):**

```bash
sudo bash scripts/training/install_prereqs.sh
bash scripts/training/setup_host.sh        # pulls ghcr.io/thewisp/lerobot-training:latest
bash scripts/training/run_training.sh -- lerobot-train --policy.type=act ...
```

### Phase 1 — Python worker scaffold ✅

|                                                                                                                                                            | Commit      | Status                            |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | --------------------------------- |
| `src/lerobot/gui/training_jobs.py` — types boundary (JobConfig, JobState, HostProfile, PollScheduler, classify_ssh_error, atomic_write_json, append_event) | `617be0517` | ✅                                |
| `src/lerobot/gui/training_worker.py` — SshConnection, poll_once, run_polling_loop, events.jsonl, incremental log tail                                      | `617be0517` | ✅ Skeleton (spawn step deferred) |
| `tests/gui/test_training_jobs.py` — 37 tests including full PollScheduler timeline, error classification matrix, monotonicity                              | `617be0517` | ✅ All passing                    |

### Phase 2 — Worker spawn step + API endpoints 🚧

|                                                                                                                 | Status  |
| --------------------------------------------------------------------------------------------------------------- | ------- |
| Wire `docker pull` + `tmux new-session -d ... docker run --gpus all lerobot-train ...` into the worker's main() | Pending |
| Forward HF_TOKEN, mount HF cache + outputs, bind-local src/lerobot when set                                     | Pending |
| `POST /api/training/runs/start` endpoint (mirror of `/hub/upload` from Hub Transfers)                           | Pending |
| `GET /api/training/runs`, `GET /api/training/runs/{id}` — list + detail with merged worker progress             | Pending |
| `POST /api/training/runs/{id}/cancel` — SIGTERM the worker → SSH kill tmux session on pod                       | Pending |
| `POST /api/training/runs/{id}/dismiss` — drop terminal run from registry                                        | Pending |
| Integration tests with mocked Popen + mocked SSH (parallel of `tests/gui/test_hub_endpoints.py`)                | Pending |

### Phase 3 — Host profile CRUD + Add-host dialog 🚧

|                                                                                               | Status                                                      |
| --------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `src/lerobot/gui/api/training_hosts.py` — CRUD endpoints (mirrors `robot.py` profile pattern) | Pending                                                     |
| Host profile storage: `~/.config/lerobot/training_hosts/<name>.json`                          | Already structured in `training_jobs.HostProfile.save/load` |
| `POST /api/training/hosts/test` — SSH probe (nvidia-smi + docker + uv detection)              | Pending                                                     |
| Frontend "Add training host" modal: paste-SSH-command + name + kind + workdir                 | Pending                                                     |
| Test-connection action with capability auto-detect                                            | Pending                                                     |
| Saved-hosts list + edit/delete UI in Model tab settings                                       | Pending                                                     |

### Phase 4 — Frontend Model tab 🚧

|                                                                                            | Status  |
| ------------------------------------------------------------------------------------------ | ------- |
| Model tab → "Training hosts" subpanel                                                      | Pending |
| "+ Start training" form: dataset picker, recipe dropdown, host dropdown, hyperparam editor | Pending |
| Active-run card: loss sparkline + GPU util + current step + ETA + Cancel button            | Pending |
| Collapsible raw-log tail panel (last 500 lines, downloadable full log)                     | Pending |
| Connection-history view (events.jsonl rendered as timeline)                                | Pending |
| Run history list + filter + dismiss                                                        | Pending |

### Phase 5 — Auto-push checkpoints + recovery 🚧

|                                                                                                          | Status                                 |
| -------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| Modify or wrap `lerobot-train` to push checkpoint to HF Hub every N steps in a background thread         | Pending                                |
| Reuses Hub Transfers' upload pipeline (atomic, Xet-deduped, resumable)                                   | Pending                                |
| Detect preemption: 3 consecutive poll failures → "disconnected"; 10 → "lost contact"                     | Implemented in PollScheduler (Phase 1) |
| "Resume" action: pre-fills new run with `--resume --config_path=hf://<repo>/<sha>`                       | Pending                                |
| End-to-end test: kill remote VM mid-training, verify resume from Hub checkpoint produces same loss curve | Pending                                |

### Phase 6 — Recipes + multi-host polish 🚧

|                                                                                                  | Status  |
| ------------------------------------------------------------------------------------------------ | ------- |
| Built-in recipe catalog: ACT-default, Diffusion-default, SmolVLA-default, PI05-LoRA              | Pending |
| User-saved recipes: `~/.config/lerobot/recipes/<name>.json` (mirrors robot/host profile pattern) | Pending |
| Multiple-host support in Start form (dropdown)                                                   | Pending |
| "Open in W&B" link from completed runs                                                           | Pending |
| "Use in Run tab" handle: `from_pretrained(hf://...)` for the robot side                          | Pending |

---

## Known gaps to fix before v1 (carried forward from the prototype README)

Discovered during the 2026-06-04 verification on Nebius preemptible H100. The Phase 0 commit fixed the bash-script ones; the others are scoped for later phases.

| Gap                                                                                                                                        | Phase to fix              | Current state                         |
| ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------- | ------------------------------------- |
| `install_prereqs.sh` broke over SSH: `gpg --dearmor` no-tty, held-package not unhold                                                       | Phase 0 ✅                | Fixed                                 |
| `run_training.sh` couldn't fall back to `sudo docker` on fresh cloud-init VMs                                                              | Phase 0 ✅                | Fixed                                 |
| HF_TOKEN not forwarded from host to remote container                                                                                       | Phase 0 ✅                | Fixed                                 |
| Nested-tini warning (image ENTRYPOINT + docker --init)                                                                                     | Phase 0 ✅                | Fixed (dropped image ENTRYPOINT)      |
| `lerobot-train` validates `--policy.repo_id` even with wandb disabled, then tries Hub push at end-of-run (crashes with 401 if no HF_TOKEN) | Phase 5 (or upstream fix) | Documented; smoke uses placeholder    |
| HF cache bind-mount fails with PermissionError (UID mismatch host/container)                                                               | Phase 2                   | Use named docker volume in spawn step |
| Spawn step on remote pod                                                                                                                   | Phase 2                   | Not started                           |
| Manage Nebius VM lifecycle from chat                                                                                                       | (out of scope for now)    | Use Nebius web console                |

---

## Cost-optimization checklist

### The lesson from 2026-06-05's $2.94 bill: disks bill continuously, not "while VM is running"

77% of that bill was disk, not compute. The math turned out to be straightforward, but the surprising part is _how_ disk billing works on Nebius (and most clouds):

> **A disk bills from the moment it's created to the moment it's deleted. The VM's state (running / stopped / deleted) does not affect disk billing.**

Stopping a VM stops the per-second compute charge. The disk continues at its $/GiB/month rate. A 640 GB SSD costs about **$2/day just sitting there**, whether or not anything's using it.

What the 19,440 GiB-hours actually broke down as:

```
  50 GiB (maroon-panda-instance-8 disk) × ~72 hours from creation =  3,600 GiB-hours
 640 GiB (main-training-preemptible disk) × ~25 hours from creation = 16,000 GiB-hours
                                                                  ───────────────────
                                                                       ≈ 19,600
```

Both VMs were Stopped most of those hours. Both disks were still alive in `Compute → Disks`. Both kept billing.

### Hygiene rules to follow

- **Right-size the boot disk on VM creation.** 50–100 GiB is enough for OS + Docker + a couple of cached datasets for almost all LeRobot training. The Nebius web form default is enormous (~1 TB) — always override.
- **Stopping a VM ≠ no disk charge.** If you're done with a VM for the day, **delete the disk too**. Use _"Delete instance + delete attached disks"_ in the console (or `terraform destroy` _and then_ verify in `Compute → Disks` that no orphans remain).
- **`terraform destroy` only deletes resources Terraform owns.** If you declared the disk as a separate `nebius_compute_v1_disk` resource, Terraform _will_ delete it. If you declared it inline on the VM and later restructured, it might survive. **Always check `Compute → Disks` after teardown.**
- **Two-tier disk strategy** for sustained use:
  - Small boot disk (50–100 GiB) for OS + Docker
  - Optional larger data disk attached only when actually training
  - Detach + delete the data disk between sessions; boot disk preserves the environment cheaply
- **Public IP allocations also bill.** Available / unattached public IPs cost ~$0.01/hr (~$0.24/day). Cheap by themselves, but the same hygiene applies: VPC → IP addresses → delete anything you no longer need.
- **`setup_host.sh` should warn on >256 GB boot disks.** Not yet implemented; follow-up.

### Order-of-magnitude reference for Nebius eu-north1 (as of 2026-06)

| Resource              | Cost                 | What this means in practice                         |
| --------------------- | -------------------- | --------------------------------------------------- |
| Network SSD           | ~$0.10 / GiB / month | A 100 GB disk is ~$10/month, regardless of VM state |
| L40S preemptible      | ~$0.90 / GPU-hour    | $0.075 for a 5-min smoke; $22 for a 24-hr run       |
| H100 preemptible      | ~$2.13 / GPU-hour    | $0.18 for a 5-min smoke; $51 for a 24-hr run        |
| Public IPv4 (dynamic) | ~$0.01 / hour        | Negligible, but adds up if left orphaned            |

**Rule of thumb:** for occasional training (a few runs/week), keep no Nebius resources between sessions. Re-cloud-init costs ~3 minutes; the $5–10/week in idle-disk fees saved is worth it.

---

## Architectural design (reference)

The pieces below were designed in long-form during the May-June 2026 design conversation. This section is a structured summary of the decisions. Where the decision is already implemented, it links to the commit.

### Three-tier deployment chain

```
Tier 0  Dev's browser/editor
                │ (via browser)
                ▼
Tier 1  GUI server's host (this is where lerobot-train would be invoked locally)
                │ (via SSH)
                ▼
Tier 2  Cloud GPU pod (this is where training actually runs)
                │ (via HF Hub API)
                ▼
              HF Hub (durable storage for checkpoints + datasets)
```

The user interacts with Tier 0; the GUI runs on Tier 1; training happens on Tier 2. Tier 1 may collapse with Tier 0 today (dev laptop runs both), but the design treats them as independent.

### One image, three transport modes

- **Image**: `ghcr.io/thewisp/lerobot-training:<tag>` built from our fork on every push to main + tagged releases (`docker_publish_fork_training.yml`).
- **Code transport** (overlay on top of the image's installed source):
  - **Release** — image's pre-installed lerobot, no overlay. Default for non-dev users.
  - **Git SHA** — `git fetch && checkout <sha>` if user's tree is clean + pushed. Repro from version control.
  - **Rsync** — rsync `src/lerobot/` from local working tree if dirty. Repro warning shown.

### Session survival (the laptop-closes-the-lid case)

Training launches inside `tmux new-session -d` on the pod with `exec python -m lerobot.gui.train_worker`. The GUI server never holds an open SSH pipe to training. Polling is via fresh `ssh ... 'cat progress.json'` every 5s, reusing TCP via `ControlMaster` / `ControlPersist`.

When the laptop closes:

- GUI server pauses, polls stop
- Training on pod keeps running (tmux detaches it from the SSH session)
- Auto-push-to-Hub from inside the pod keeps running
- Laptop wakes → polls resume → card shows current step

### Connection resilience

`PollScheduler` (implemented in Phase 1):

- Exponential backoff: 5/10/20/40/60s cap, 10 attempts, ~5 min total before give-up
- Transient errors (timeout, connection refused) → retry with backoff
- Permanent errors (auth failed, bad host key) → immediate give-up via `permanent=True` shortcut

Three log surfaces:

- `progress.json` — atomically-rewritten snapshot of current state
- `events.jsonl` — append-only audit log of state transitions (connect, fail, retry, lost_contact)
- `stderr.log` — continuously mirrored from pod via byte-offset incremental tail

### Recovery from preemption

Primary mechanism: **periodic checkpoint to local pod disk + async background upload to HF Hub**. The grace window from `SIGTERM → SIGKILL` is too short for big models (~7 GB for pi05) at typical bandwidth, so we don't try to flush on signal — the most-recently-uploaded Hub checkpoint is what survives.

Cadence trade-off: every ~60–180s of training time for preemptible pods (vs ~10 min default for on-demand). Smaller per-checkpoint cost vs lower work-loss-on-eviction.

Resume flow: failed run's last Hub checkpoint URL becomes the `--resume --config_path=` on a fresh pod. One click in the GUI.

### Host profile pattern

Storage: `~/.config/lerobot/training_hosts/<name>.json` — one JSON per host, exactly the same shape as robot profiles in `~/.config/lerobot/robot_profiles/`. CRUD endpoints mirror `robot.py`. Frontend modals mirror robot-profile UI.

Test-connection on save: SSH probe runs `nvidia-smi`, `docker --version`, `which uv`, captures capabilities into the profile.

### What we don't try to do

- **Pod provisioning.** User launches the pod on Nebius/RunPod/etc. themselves; we accept an SSH endpoint.
- **In-GUI HF login.** Delegated to `huggingface-cli login` (out-of-band terminal flow).
- **Multi-GPU / multi-node training.** Out of scope for v1.
- **Image-everywhere for local dev.** Local CUDA uses the bash-subprocess path against the dev's venv. The image is for remote pods.
- **Browser-coupled providers** (Colab Free/Pro/Pro+, Kaggle interactive). They kill on user-side idle regardless of GPU activity — no design from our side defeats that. Documented as out-of-scope.
