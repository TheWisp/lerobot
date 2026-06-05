# Model Training: cloud-GPU training pipeline integrated into the GUI

End-to-end design for taking a user from "I want to train on a cloud GPU" to "trained model on HF Hub", driven from the LeRobot GUI rather than the CLI.

Companion to [`gui/docs/hub_transfers.md`](../../src/lerobot/gui/docs/hub_transfers.md). Training reuses Hub Transfers' worker-IPC + polling + Hub-upload patterns wholesale.

---

## Vocabulary

Two host-management modes the GUI supports. Pinned terminology so the rest of this doc + code + UI copy stays consistent:

| Term                | Meaning                                                                                                                                                                                                                                                                                       | When you'd pick it                                                                                                                                                        |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Persistent host** | User provisions the VM themselves (on Nebius / RunPod / their lab box / whatever). User pastes the SSH command into the GUI's "Add training host" dialog. GUI never creates or destroys VMs; only connects, runs trainings, and disconnects. The VM lives until the user manually deletes it. | Always-on lab box, dev VM you SSH into for other reasons, debugging session, vendor where we don't have an integration                                                    |
| **Ephemeral host**  | GUI creates the VM on the fly when training starts and destroys it (VM + disk + public IP) when training finishes. Lifecycle is automatic; no idle resources between sessions. Requires the GUI to hold a vendor API token.                                                                   | The default for most users — cost-disciplined, no "I forgot to delete the disk" surprises, matches what Modal / SageMaker / Anyscale already do for training-as-a-service |

The terms are deliberately not "on-demand vs spot" — that pairing already means something else in cloud-vendor pricing pages (on-demand = guaranteed duration, spot = preemptible). Persistent vs Ephemeral is about **VM lifecycle ownership**, orthogonal to whether the underlying instance is preemptible or not. An Ephemeral host CAN use a preemptible instance underneath (and probably will, for cost).

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

### Phase 2 — Worker spawn step + API endpoints 🚧 (Persistent-only scope)

Concrete-case-first: this phase wires the SSH-into-an-existing-host path end-to-end. The provider abstraction emerges from generalizing this in Phase 2.5 once the concrete case is solid.

|                                                                                                                 | Status  |
| --------------------------------------------------------------------------------------------------------------- | ------- |
| Wire `docker pull` + `tmux new-session -d ... docker run --gpus all lerobot-train ...` into the worker's main() | Pending |
| Forward HF_TOKEN, mount HF cache + outputs, bind-local src/lerobot when set                                     | Pending |
| `POST /api/training/runs/start` endpoint (mirror of `/hub/upload` from Hub Transfers)                           | Pending |
| `GET /api/training/runs`, `GET /api/training/runs/{id}` — list + detail with merged worker progress             | Pending |
| `POST /api/training/runs/{id}/cancel` — SIGTERM the worker → SSH kill tmux session on pod                       | Pending |
| `POST /api/training/runs/{id}/dismiss` — drop terminal run from registry                                        | Pending |
| Integration tests with mocked Popen + mocked SSH (parallel of `tests/gui/test_hub_endpoints.py`)                | Pending |

### Phase 2.5 — HostProvider protocol + Persistent implementation 🚧

Once Phase 2 works against one concrete case, the abstraction falls out. Belongs **after** Phase 2, not before — Rule of Three: write the concrete case first, then generalize when a second case (Ephemeral / RunPod) is on the way.

|                                                                                                                                                         | Status  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `src/lerobot/gui/training_providers/protocol.py` — `HostProvider` Protocol + `SpawnSpec`, `HostHandle`, `CostSnapshot` dataclasses                      | Pending |
| `src/lerobot/gui/training_providers/persistent.py` — wraps existing HostProfile, `spawn()` raises `NotImplementedError`, `destroy()` is a no-op         | Pending |
| Refactor Phase 2's spawn step to consume a `HostHandle`, not a `HostProfile` directly                                                                   | Pending |
| Protocol omits `stop()` / `start()` (Lambda has neither — every off-switch is `terminate`); omits SSH runtime (lives above protocol in `SshConnection`) | Pending |
| Tests: `runtime_checkable` Protocol conformance + `PersistentSshProvider` no-op cost / no-op destroy / `verify_destroyed` → True                        | Pending |

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

### Phase 6.5 — Ephemeral mode + RunPod provider 🚧

The first Ephemeral provider. RunPod is chosen over Nebius for v1 because:

- `--terminate-after <ttl>` is a built-in dead-man switch at pod-create time (Nebius requires us to implement TTL with a scheduled async delete op).
- Network volumes are a separate billable object with explicit lifecycle ("network volumes survive pod termination, billed monthly regardless of any pod") — the correct primitive for persistent dataset cache.
- Python SDK is a thin GraphQL wrapper, actively maintained.
- Per-second billing.
- Real SSH on a mapped TCP port; root container.

Three guardrails on every Ephemeral spawn, baked into the protocol:

1. **Hard TTL** (`--terminate-after` at the vendor layer) — REQUIRED on every `SpawnSpec`; provider rejects spawn without it. Default 24h, configurable in settings.
2. **Spend cap per run** — computed from `(gpu_hourly × ttl_hours) + (disk_size × storage_rate × ttl_hours/720)` before spawn. Refuse to launch above the user-configured ceiling (default $25/run).
3. **Destroy verification** — after the run terminates, re-list resources in the user's account; refuse to mark the run "complete" until `verify_destroyed()` returns True. Automates the "always check Compute → Disks after teardown" hygiene rule.

|                                                                                                                                                 | Status  |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `src/lerobot/gui/training_providers/runpod.py` — `spawn` calls `runpod.create_pod(terminate_after=...)`, `destroy` calls `runpod.terminate_pod` | Pending |
| `spend_cap_usd` field on the Start-training form; estimate displayed before launch                                                              | Pending |
| Destroy-verification UI: run badge stays yellow ("verifying cleanup") until `verify_destroyed()` returns True                                   | Pending |
| Settings: API-key storage at `~/.config/lerobot/provider_credentials/runpod.json`, encrypted at rest (OS keyring or `cryptography.fernet`)      | Pending |
| Default-mode toggle in Settings: Ephemeral (default) vs Persistent                                                                              | Pending |
| End-to-end test: synthetic 10-step training run that spawns + destroys a RunPod pod via the protocol; verify_destroyed at end                   | Pending |

### Phase 7 — Nebius provider 🚧

Defer Nebius until Phase 7 even though it's the platform we've already paid on. Reasoning: Nebius's idempotency is undocumented (no `--idempotency-key`; the recommended pattern is `get-by-name` first, create-if-404), the SDK is 0.3.x with thin compute examples, and the disk-lifecycle gotcha needs to be encoded as defaults — landing it as the v1 reference would compete for design attention with the protocol itself.

|                                                                                                                                                                | Status  |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `training_providers/nebius.py` — `spawn` declares disk **inline** on the VM (cascade delete); never use standalone disks unless user explicitly opts in        | Pending |
| Warn at SpawnSpec validation if `disk_gib > 256` (most LeRobot training fits in <100; defends against the 1.28-TB-default trap)                                | Pending |
| Warn on static public-IP allocation: _"If you stop a VM that has a static IP address, the address will not return to the range"_ — surface this in the UI      | Pending |
| Idempotency wrapper: `get_instance_by_name(name)` before `create_instance(name)`. Server has no `--idempotency-key`.                                           | Pending |
| `verify_destroyed` does the "Compute → Disks + VPC → IP addresses" re-check the cost-optimization section currently asks the user to do manually               | Pending |
| End-to-end test against real Nebius (gated by env var, like Hub Transfers' `LEROBOT_HUB_LIVE=1`): 10-step training + cleanup, asserts $0 of leftover resources | Pending |

### Phase 8+ — Modal / Lambda (separate code paths) 🚧

These don't fit the SshConnection model and aren't a priority for v1.

- **Modal** — fundamentally different shape (no VM, no SSH). A separate code path that ignores `SshConnection` entirely and just calls `modal.Function.from_name(...).remote()`. High UX ceiling once it works. Worth doing once the protocol matures, but as a "Modal backend" alongside the SshConnection-based providers, not inside the same protocol.
- **Lambda Cloud** — clean REST, but no `stop` — every off-switch is `terminate`. Actually fits Ephemeral well (no way to leave a stopped VM bleeding compute; only the separate Filesystem product persists). Add once the protocol is stable.

### Skipped vendors

- **Vast.ai** — marketplace variance ("advertised 1500 Mbps NIC sometimes delivers ~100 Mbps", volumes pinned to a single physical host so 'persistent' isn't really persistent, no SLA). Power users can still use it through the Persistent SSH path; we just won't market it as a managed integration.
- **Paperspace** — SDK story is broken (the `gradient` PyPI package's old surface is broken; the new `digitalocean/gradient-python` SDK does not manage Paperspace machines), auto-shutdown is console-only (not API/CLI), which defeats GUI-driven lifecycle. Defer indefinitely.
- **Colab / Kaggle interactive** — browser-coupled lifecycle (already out-of-scope below).

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

These rules apply to **Persistent-host mode** (user owns the VM lifecycle) and to **Phase 0 / 1 / 2 / 2.5** while we're still bash-driven. The whole point of **Phase 6.5+ Ephemeral mode** is to automate these rules — `destroy()` + `verify_destroyed()` are the protocol's enforcement of items 2 and 3 below.

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

### HostProvider protocol (Phase 2.5)

Small surface, deliberately narrow. The SSH runtime (`SshConnection`, polling loop, log tail) lives **above** the protocol, not inside it — the provider's only job is to hand back a usable SSH endpoint and to make it (and its disk and its public IP) go away later. Same code path drives Persistent and Ephemeral; the discriminant is which provider is plugged in.

```python
# src/lerobot/gui/training_providers/protocol.py
from dataclasses import dataclass
from typing import Protocol, Literal, runtime_checkable

GpuKind = Literal["RTX_4090", "L40S", "A100_80", "H100", "H200", "B200"]
ProviderId = Literal["persistent", "runpod", "nebius", "lambda", "modal"]


@dataclass(frozen=True, kw_only=True)
class SpawnSpec:
    """What the user asked for. Vendor-neutral."""
    gpu: GpuKind
    gpu_count: int = 1
    preemptible: bool = True             # Ephemeral defaults to spot/interruptible
    disk_gib: int = 100                  # boot + container disk
    image: str                           # docker ref the training container pulls
    region_hint: str | None = None       # provider may ignore; e.g. "us", "eu"
    ttl_seconds: int                     # hard kill — REQUIRED, no default
    estimated_cost_ceiling_usd: float    # refuse to spawn above this


@dataclass(frozen=True, kw_only=True)
class HostHandle:
    """Live handle the SSH path uses. All vendors converge on this shape."""
    provider: ProviderId
    provider_resource_id: str            # vendor's pod/instance id, for destroy()
    ssh_host: str
    ssh_port: int
    ssh_user: str = "root"
    ssh_key_path: str                    # identity key; provider may have generated it
    persistent_volume_id: str | None = None  # storage that survives the handle
    region: str
    expires_at_unix: int                 # provider's hard-TTL deadline — never None


@dataclass(frozen=True, kw_only=True)
class CostSnapshot:
    compute_hourly_usd: float
    storage_monthly_usd_per_gib: float
    accrued_usd_estimate: float          # since spawn; provider-side or local clock


@runtime_checkable
class HostProvider(Protocol):
    id: ProviderId
    display_name: str

    def estimate_cost(self, spec: SpawnSpec) -> CostSnapshot: ...
    def spawn(self, spec: SpawnSpec) -> HostHandle: ...
    def destroy(self, handle: HostHandle) -> None: ...
    def verify_destroyed(self, handle: HostHandle) -> bool: ...
    def current_cost(self, handle: HostHandle) -> CostSnapshot: ...
```

**What's deliberately NOT in the protocol:**

- **`stop()` / `start()`.** Ephemeral never stops, it destroys. Lambda doesn't even have stop. Including stop would force a fiction onto vendors that don't support it, and (worse) hand the user a footgun — a stopped VM keeps billing disk.
- **`ssh_run(cmd)` / `ssh_tail(log)`.** The existing `SshConnection` + `PollScheduler` are vendor-agnostic and live above the provider. The provider's responsibility ends at "here's a working SSH endpoint."
- **Auth flow.** Each provider has its own credential storage (`~/.nebius/`, `~/.modal.toml`, RunPod env var). The provider implementation reads its own config; the GUI's API-key storage is a separate concern.
- **Upload/download of artifacts.** Checkpoints go to HF Hub; no provider needs to push/pull files.
- **Preemption notification.** Handled by `PollScheduler` (3 fails → disconnected, 10 → lost contact). Provider doesn't emit events.

**Persistent host as a degenerate provider:**

```python
class PersistentSshProvider:
    id = "persistent"
    display_name = "BYO SSH endpoint"

    def estimate_cost(self, spec):    return CostSnapshot(0.0, 0.0, 0.0)
    def spawn(self, spec):            raise NotImplementedError("Persistent hosts are added, not spawned")
    def destroy(self, handle):        pass            # user owns the VM
    def verify_destroyed(self, handle): return True   # always
    def current_cost(self, handle):   return CostSnapshot(0.0, 0.0, 0.0)
```

Run-driving code only ever talks to a `HostProvider` + a `HostHandle`. The two modes differ only in which provider is plugged in. Phase 3's HostProfile becomes "the way the Persistent provider gets its HostHandle." The first real spawn lands in Phase 6.5 (RunPod).

### What we don't try to do

- **Pod provisioning outside the HostProvider protocol.** Ephemeral mode (Phase 6.5+) provisions through a vetted vendor SDK with hard TTL, spend cap, and destroy-verification. The GUI does NOT embed vendor consoles, manage payment methods, or spawn without a TTL + spend cap. Persistent mode still accepts a user-provided SSH endpoint for vendors we don't (yet) wrap.
- **In-GUI HF login.** Delegated to `huggingface-cli login` (out-of-band terminal flow).
- **Multi-GPU / multi-node training.** Out of scope for v1.
- **Image-everywhere for local dev.** Local CUDA uses the bash-subprocess path against the dev's venv. The image is for remote pods.
- **Browser-coupled providers** (Colab Free/Pro/Pro+, Kaggle interactive). They kill on user-side idle regardless of GPU activity — no design from our side defeats that. Documented as out-of-scope.
- **Vendor marketplaces with quality variance** (Vast.ai). Power users can still use them through the Persistent SSH path; we just don't market them as managed integrations.
