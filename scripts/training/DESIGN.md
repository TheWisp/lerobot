# Model training — architecture

End-to-end design for taking a user from "I want to train a policy" to "trained model on HF Hub", driven from the LeRobot GUI.

Companion to [`gui/docs/hub_transfers.md`](../../src/lerobot/gui/docs/hub_transfers.md) — training reuses Hub Transfers' worker-IPC + polling + Hub-upload patterns wholesale.

---

## Use cases

Three host modes, in increasing complexity. Each builds on the last.

### A. Local GPU (default for workstations)

**Who:** developer with a GPU in their own machine.

**Setup:**

```
huggingface-cli login
lerobot-gui                                 # binds 127.0.0.1
```

**Flow:** open browser → pick dataset + recipe + hyperparams → click **Start** → training runs as a subprocess on the same machine → checkpoints push to HF Hub.

No SSH, no cloud, no provider. The existing bash-subprocess training path the GUI already has.

### B. Persistent SSH host (always-on lab box)

**Who:** developer with a remote machine they manage themselves — lab server, university cluster, leased EC2.

**Setup:**

```
huggingface-cli login                       # on user's laptop
lerobot-gui                                 # local or LAN
# in GUI: Add training host → paste SSH command → name → save
```

**Flow:** GUI's "Add training host" dialog accepts an `ssh user@host` command; GUI tests reachability + capabilities (`nvidia-smi`, `docker`, `uv`); training runs over SSH inside `tmux` on that host; GUI polls progress via incremental file tails.

The GUI never creates or destroys the VM — that's the user's responsibility, as is paying for it.

### C. Ephemeral cloud host (auto-managed VM)

**Who:** developer who wants H100s on demand without touching a cloud console.

**Setup** (one-time per laptop):

```
nebius profile create
huggingface-cli login
lerobot auth-helper start --origin http://lab-server:8000   # only if the GUI is on a LAN box
```

**Flow:** GUI's "Start training" dialog has a GPU dropdown + spend cap + TTL fields. On Start: provider's `spawn()` shells out to the vendor CLI with the right SKU, disk size, server-side TTL; waits for SSH reachability; runs training inside tmux as in case B; on completion or TTL hit, calls `destroy()` and `verify_destroyed()` to ensure no orphan disks/IPs.

The provider abstraction (below) lets us add other vendors without changing the rest of the GUI.

---

## Vocabulary

| Term                | Meaning                                                                                                                                                                          | Setup cost                           |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| **Local host**      | Training runs as a subprocess of the GUI server on the same machine. No SSH, no provider, no cloud.                                                                              | Zero beyond `huggingface-cli login`. |
| **Persistent host** | User owns the VM (lab box, RunPod, leased EC2 — anything reachable by SSH). User pastes SSH command into GUI. GUI connects, trains, disconnects; VM lives until user deletes it. | One-time SSH command paste.          |
| **Ephemeral host**  | GUI creates the VM on training start, destroys it on completion. Lifecycle automatic. Requires per-user cloud credential.                                                        | One-time `nebius profile create`.    |

"Persistent" vs "Ephemeral" is about lifecycle ownership, orthogonal to spot/preemptible pricing. An Ephemeral host typically uses a preemptible instance underneath.

---

## Architecture

### Deployment tiers

```
Tier 0   User's browser (laptop / tablet)
                │
                ▼
Tier 1   GUI server (workstation in local mode, LAN box in shared mode)
                │   subprocess  (Local mode)
                │   SSH         (Persistent / Ephemeral modes)
                ▼
Tier 2   Training host (same machine, lab box, or cloud VM)
                │
                ▼
              HF Hub (datasets + checkpoints + final model)
```

Tier 1 may collapse with Tier 0 (workstation GPU case). The design treats them as independent.

### HostProvider protocol

The bridge between the GUI's orchestration code and vendor-specific VM lifecycle. Same code path drives Persistent and Ephemeral; the discriminant is which provider is plugged in. Local mode bypasses the provider entirely.

```python
@dataclass(frozen=True, kw_only=True)
class SpawnSpec:                                      # vendor-neutral request
    gpu: GpuKind
    gpu_count: int = 1
    preemptible: bool = True
    disk_gib: int = 100
    image: str
    region_hint: str | None = None
    ttl_seconds: int                                  # hard kill — REQUIRED
    estimated_cost_ceiling_usd: float                 # refuse to spawn above this


@dataclass(frozen=True, kw_only=True)
class HostHandle:                                     # what the SSH path consumes
    provider: ProviderId
    provider_resource_id: str                         # vendor's VM/pod id
    ssh_host: str
    ssh_port: int
    ssh_user: str = "root"
    ssh_key_path: str
    persistent_volume_id: str | None = None
    region: str
    expires_at_unix: int                              # provider's hard-TTL deadline


@dataclass(frozen=True, kw_only=True)
class CostSnapshot:
    compute_hourly_usd: float
    storage_monthly_usd_per_gib: float
    accrued_usd_estimate: float                       # since spawn (Ephemeral) or 0 (Persistent)


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

**Deliberately omitted:**

- `stop()` / `start()` — not every vendor supports stop (Lambda doesn't); a stopped VM still bills its disk. Every off-switch is `terminate`.
- `ssh_run()` / `ssh_tail()` — SSH runtime is vendor-agnostic and lives above the provider.
- Auth flow — handled by the credential helper (below); credentials injected per-request as a provider constructor arg.
- Upload/download — checkpoints flow through HF Hub; providers don't move files.

**Persistent as a degenerate provider:** `spawn()` raises (user supplied the SSH endpoint by hand); `destroy()` is a no-op; cost is zero.

### SSH transport + log surfaces

Training launches inside `tmux new-session -d` on the host with `exec python -m lerobot.gui.train_worker`. The GUI server never holds an open SSH pipe to training. Polling is via short-lived `ssh … 'cat progress.json'` every 5s, reusing TCP via `ControlMaster` / `ControlPersist`.

Three log surfaces:

- **`progress.json`** — atomically-rewritten snapshot (loss, step, ETA, GPU util)
- **`events.jsonl`** — append-only audit log of state transitions (connect, fail, retry, lost_contact)
- **`stderr.log`** — byte-offset incremental tail mirrored from the host

### Session survival

The laptop-closes-the-lid case (Persistent / Ephemeral):

- GUI server pauses, polls stop
- Training on the host keeps running (tmux detached from SSH)
- Auto-push-to-Hub from inside the training script keeps running
- Laptop wakes → polls resume → card shows current step

Browser-close in Ephemeral mode is the same property: training continues via the SSH key (no provider token needed); on resume, the GUI re-fetches tokens from the helper.

### Connection resilience

`PollScheduler` policy:

- Exponential backoff: 5/10/20/40/60s cap, 10 attempts, ~5 min before give-up
- Transient errors (timeout, connection refused) → retry with backoff
- Permanent errors (auth failed, bad host key) → immediate give-up
- Transitions surface as `events.jsonl` entries + a connection-quality indicator in the UI

### Recovery from preemption

Primary mechanism: **periodic checkpoint to host disk + async background upload to HF Hub**. The grace window from `SIGTERM → SIGKILL` is too short for big models (~7 GB at typical bandwidth) to flush on signal, so the most-recent Hub checkpoint is what survives.

Cadence: every ~60–180s of training time for preemptible hosts (vs ~10 min default for on-demand). Tunable per recipe.

### Authentication

When the backend is shared (lab LAN), creds must be per-user, must not leak between users, and must not persist on the backend. Solution: the user's laptop is the sole source of truth; the backend handles tokens only in request scope.

```
[User's laptop]                                  [Lab server]
~/.nebius/  ~/.cache/huggingface/   ◄── lerobot-auth-helper (127.0.0.1:39847)
                                                ▲
                                                │ GET /tokens (Origin + X-Helper-Token)
                                            Browser ─── X-Nebius-Auth / X-HF-Auth ──► GUI backend
                                                                                     (request-scope, no storage)
```

**Local-host mode** (GUI binds `127.0.0.1`): helper auto-launched as a child of the GUI server with auto-pairing. No setup beyond `huggingface-cli login`.

**Shared-backend mode** (LAN): each user runs the helper on their own laptop; pastes the helper's pairing token into GUI Settings once. After that, tokens rotate transparently.

**Security properties:**

- Long-lived creds stay on the user's laptop; backend has zero credential persistence
- Helper bound to `127.0.0.1` only; rejects requests missing matching `Origin` AND `X-Helper-Token`
- Compromised backend worst case: ≤12h Nebius tokens for currently-connected users; no refresh creds, no long-lived secrets

**HF caveat.** `~/.cache/huggingface/token` is non-expiring. Subprocess isolation (Hub Transfers already uses `hub_worker.py`, a separate process) confines it to that subprocess's RAM. Users wanting tighter scope: switch to a fine-grained HF token with explicit expiry on huggingface.co — no code change needed.

### Cost discipline

**Disks bill from creation to deletion, regardless of VM state.** A stopped VM stops the per-second compute charge; the attached disk continues at $/GiB/month. A 640 GB SSD costs ~$2/day just sitting there.

Three structural defenses baked into Ephemeral mode:

1. **Inline-managed boot disk only.** Disk lifecycle is cascaded with the VM; no standalone disks unless the user opts in.
2. **Disk-size warning.** Nebius's web-form default is ~1 TB; we warn above 256 GiB (most LeRobot training fits in <100 GB).
3. **Hard server-side TTL.** Set at spawn via the vendor's scheduled-delete API; the VM dies even if the GUI server dies.

Order-of-magnitude reference (Nebius eu-north1, mid-2026):

| Resource              | Cost                 | Practical meaning                              |
| --------------------- | -------------------- | ---------------------------------------------- |
| Network SSD           | ~$0.10 / GiB / month | 100 GB disk ≈ $10/month regardless of VM state |
| L40S preemptible      | ~$0.90 / GPU-hour    | $0.075 for a 5-min smoke; $22 for a 24-hr run  |
| H100 preemptible      | ~$2.13 / GPU-hour    | $0.18 for a 5-min smoke; $51 for a 24-hr run   |
| Public IPv4 (dynamic) | ~$0.01 / hour        | Negligible alone, accumulates if left orphaned |

For Persistent hosts the GUI can't enforce these defenses; the user owns the VM. Rule of thumb: for occasional training, delete VM + disk between sessions — re-cloud-init costs ~3 minutes vs $5–10/week in idle-disk fees.

---

## What we don't try to do

- **Pod provisioning outside the HostProvider protocol.** Ephemeral mode goes through a vetted vendor SDK with hard TTL, spend cap, and destroy-verification. The GUI does NOT embed vendor consoles or manage payment methods.
- **Multi-GPU / multi-node training.** Out of scope for v1.
- **Image-everywhere for local dev.** Local mode uses the bash-subprocess path against the dev's venv; the docker image is for remote hosts.
- **Browser-coupled providers** (Colab, Kaggle interactive). They kill on user-side idle regardless of GPU activity; no design from our side defeats that.
- **Vendor marketplaces with quality variance** (Vast.ai). Power users can still use them through the Persistent SSH path; we just don't market them as managed integrations.
- **In-GUI credential entry.** No "paste your API key" forms. Credentials live in the user's local CLI state; the helper bridges browser ↔ local state.
