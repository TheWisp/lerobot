# Model training — architecture

End-to-end design for taking a user from "I want to train a policy" to "trained model in a checkpoint store", driven from the LeRobot GUI.

Companion to [`gui/docs/hub_transfers.md`](../../src/lerobot/gui/docs/hub_transfers.md) — training reuses Hub Transfers' worker-IPC + polling + upload patterns wholesale.

---

## Use cases

Two host modes by lifecycle ownership. The workstation-GPU case is the easy slice of Persistent.

### A. Workstation (auto-registered Persistent host)

**Who:** developer with a GPU in the same machine that runs the GUI server.

**Setup:**

```
huggingface-cli login
lerobot-gui                                 # binds 127.0.0.1; helper auto-launched
```

On first start the GUI runs `nvidia-smi` locally; if a GPU is found, a host called "This server" appears in the dropdown. No further setup.

**Flow:** open browser → pick training host → pick dataset + recipe + hyperparams → click **Start** → training runs on that host → checkpoints push to the configured store.

### B. User-added Persistent host (lab box, leased VM, user's laptop)

**Who:** developer with a remote machine they manage themselves — lab server, university cluster, leased EC2, or their own laptop reachable from the GUI server.

**Setup:**

```
huggingface-cli login                       # on user's laptop
lerobot-gui                                 # local or LAN
# in GUI: Add training host → paste SSH command → name → save
```

**Flow:** "Add training host" dialog accepts `ssh user@host`; GUI tests reachability + capabilities (`nvidia-smi`, `docker`, `uv`); training runs over SSH inside `tmux` on that host; GUI polls progress via incremental file tails.

The GUI never creates or destroys the VM — that's the user's responsibility, as is paying for it.

### C. Ephemeral cloud host (auto-managed VM)

**Who:** developer who wants H100s on demand without touching a cloud console.

**Setup** (one-time per laptop):

```
<vendor> profile create                                            # whichever vendor's CLI
huggingface-cli login
lerobot auth-helper start --origin http://lab-server:8000          # only if the GUI is on a LAN box
```

**Flow:** "Start training" dialog has a GPU dropdown + spend cap + TTL fields. On Start: the provider's `spawn()` provisions the VM with the right SKU, disk size, and server-side TTL; waits for SSH reachability; runs training inside tmux as in case B; on completion or TTL hit, calls `destroy()` and `verify_destroyed()` to ensure no orphan disks/IPs.

The provider abstraction (below) lets us add new vendors without changing the rest of the GUI.

---

## Vocabulary

| Term                 | Meaning                                                                                                                                          |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Training host**    | Any machine where training can run. May be the GUI server's own box, a user-added remote box, or an auto-spawned cloud VM.                       |
| **Persistent host**  | User-managed lifecycle. GUI connects; doesn't create or destroy. Default for both the workstation case (auto-registered) and any user-added box. |
| **Ephemeral host**   | Provider-managed lifecycle. GUI creates the VM on training start, destroys on completion.                                                        |
| **Checkpoint store** | Where checkpoints land. May be HF Hub, a vendor-local object store, or anywhere else. Decoupled from the training host.                          |

"Persistent" vs "Ephemeral" is about lifecycle ownership, orthogonal to spot/preemptible pricing. An Ephemeral host typically uses a preemptible instance underneath.

---

## Architecture

### Deployment tiers

```
Tier 0   User's browser (laptop / tablet)
                │
                ▼
Tier 1   GUI server (workstation or LAN box)
                │   transport: subprocess  (host is the GUI server itself)
                │   transport: SSH         (host is anywhere else)
                ▼
Tier 2   Training host (same machine, lab box, or cloud VM)
                │
                ▼
       Checkpoint store (HF Hub, vendor object store, …)
```

Tier 1 may collapse with Tier 0 (workstation case). The design treats them as independent.

### HostProvider protocol

The bridge between the GUI's orchestration code and vendor-specific VM lifecycle. Same code path drives Persistent and Ephemeral; the discriminant is which provider is plugged in.

```python
@dataclass(frozen=True, kw_only=True)
class SpawnSpec:                                      # vendor-neutral request
    gpu: GpuKind
    gpu_count: int = 1                                # v1: always 1; field reserved for later
    preemptible: bool = True
    disk_gib: int = 100
    image: str
    region_hint: str | None = None
    ttl_seconds: int                                  # hard kill — REQUIRED
    estimated_cost_ceiling_usd: float                 # refuse to spawn above this


@dataclass(frozen=True, kw_only=True)
class SshTransport:
    host: str
    port: int = 22
    user: str = "root"
    key_path: str

@dataclass(frozen=True, kw_only=True)
class SubprocessTransport:                            # for the workstation case
    cwd: Path

HostTransport = SshTransport | SubprocessTransport


@dataclass(frozen=True, kw_only=True)
class HostHandle:                                     # what the run-driving code consumes
    provider: ProviderId
    provider_resource_id: str                         # vendor's VM/pod id (or "local")
    transport: HostTransport
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

- `stop()` / `start()` — not every vendor supports stop; a stopped VM still bills its disk. Every off-switch is `terminate`.
- `ssh_run()` / `ssh_tail()` — transport runtime is vendor-agnostic and lives above the provider.
- Auth flow — handled by the credential helper (below); credentials injected per-request as a provider constructor arg.
- Upload/download — checkpoints flow through `CheckpointStore` (next section); providers don't move files.

**Persistent as a degenerate provider:** `spawn()` raises (the host already exists); `destroy()` is a no-op; cost is zero. The auto-registered workstation host uses `SubprocessTransport`; user-added hosts use `SshTransport`.

### CheckpointStore protocol

Parallel to `HostProvider`, but for where checkpoints land. Decouples storage choice from compute choice.

```python
@runtime_checkable
class CheckpointStore(Protocol):
    id: StoreId                                       # "hf_hub", "nebius_s3", ...
    display_name: str
    pricing_info_url: str | None                      # link to vendor's pricing page; UI surfaces this

    def push(self, run_id: str, step: int, local_path: Path) -> StorageUri: ...
    def pull(self, uri: StorageUri, dest: Path) -> None: ...
    def list_checkpoints(self, run_id: str) -> list[StorageEntry]: ...
```

A training run can use two stores:

- **scratch store** — frequent checkpoints during training. Defaults to whatever the host's vendor-local store is (e.g., the vendor's own S3-compatible bucket); falls back to the publish store if no vendor-local option is configured.
- **publish store** — final model destination. HF Hub by default.

Why the split: scratch checkpoints can be hundreds of GiB-writes per run, and egress to a cross-network store can dominate compute cost. Putting scratch on a vendor-local store (where egress is typically free or near-free) avoids that. The publish store gets a single final upload.

**On surfacing cost.** We do NOT predict dollar amounts — vendor pricing drifts and we have no reliable feed for it. The store exposes `pricing_info_url`; the host-config UI shows a "💰 See pricing" link per host so the user reads current rates themselves.

### SSH / subprocess transport + log surfaces

Training launches inside `tmux new-session -d` on the host (SSH transport) or under a managed subprocess (`SubprocessTransport`, workstation case), invoking `python -m lerobot.gui.train_worker`. The GUI server never holds an open pipe to training. Polling is via short-lived reads of `progress.json` and byte-offset tails of `stderr.log` every 5s (TCP reused via `ControlMaster` / `ControlPersist` for SSH).

Three log surfaces, identical across both transports:

- **`progress.json`** — atomically-rewritten snapshot (loss, step, ETA, GPU util)
- **`events.jsonl`** — append-only audit log of state transitions (connect, fail, retry, lost_contact)
- **`stderr.log`** — byte-offset incremental tail mirrored from the host

### Session survival

The laptop-closes-the-lid case (any host mode):

- GUI server pauses, polls stop
- Training on the host keeps running (tmux detached / subprocess unaffected)
- Auto-push from inside the training script keeps running
- Laptop wakes → polls resume → card shows current step

Browser-close in Ephemeral mode is the same property: training continues via the SSH key (no provider token needed); on resume, the GUI re-fetches tokens from the helper.

### Connection resilience

`PollScheduler` policy:

- Exponential backoff: 5/10/20/40/60s cap, 10 attempts, ~5 min before give-up
- Transient errors (timeout, connection refused) → retry with backoff
- Permanent errors (auth failed, bad host key) → immediate give-up
- Transitions surface as `events.jsonl` entries + a connection-quality indicator in the UI

### Recovery from preemption

Primary mechanism: **periodic checkpoint to host disk + async background upload to the scratch store**. The grace window from `SIGTERM → SIGKILL` is too short for big models (~7 GB at typical bandwidth) to flush on signal, so the most-recent uploaded checkpoint is what survives.

Cadence: every ~60–180s of training time for preemptible hosts (vs ~10 min default for on-demand). Tunable per recipe.

### Authentication

When the backend is shared (lab LAN), creds must be per-user, must not leak between users, and must not persist on the backend. Solution: the user's laptop is the sole source of truth; the backend handles tokens only in request scope.

```
[User's laptop]                                  [Lab server]
~/.nebius/  ~/.cache/huggingface/   ◄── lerobot-auth-helper (127.0.0.1:39847)
                                                ▲
                                                │ GET /tokens (Origin + X-Helper-Token)
                                            Browser ─── X-<Vendor>-Auth / X-HF-Auth ──► GUI backend
                                                                                       (request-scope, no storage)
```

**Workstation case** (GUI binds `127.0.0.1`): helper auto-launched as a child of the GUI server with auto-pairing. No setup beyond `huggingface-cli login`. The user runs ONE command total: `lerobot-gui`.

**Shared-LAN case**: GUI server runs only `lerobot-gui` — no second service. Each user runs `lerobot auth-helper start --origin <gui-url>` ONCE on their own laptop; pairing token pasted into GUI Settings once. After that, tokens rotate transparently.

**Security properties:**

- Long-lived creds stay on the user's laptop; backend has zero credential persistence
- Helper bound to `127.0.0.1` only; rejects requests missing matching `Origin` AND `X-Helper-Token`
- Compromised backend worst case: short-lived provider tokens (typically ≤12h) for currently-connected users; no refresh creds, no long-lived secrets

**HF caveat.** `~/.cache/huggingface/token` is non-expiring. Subprocess isolation (Hub Transfers already uses `hub_worker.py`, a separate process) confines it to that subprocess's RAM. Users wanting tighter scope: switch to a fine-grained HF token with explicit expiry on huggingface.co — no code change needed.

### Cost discipline

Vendor-neutral properties to defend against:

**Disks bill from creation to deletion, regardless of VM state.** A stopped VM stops compute; the attached disk continues at $/GiB/month. A 640 GB SSD costs ~$2/day just sitting there on most clouds.

Three structural defenses baked into Ephemeral mode:

1. **Inline-managed boot disk only.** Disk lifecycle cascades with the VM; no standalone disks unless the user opts in.
2. **Disk-size warning.** Many vendor web forms default to enormous disks (1 TB+); we warn above 256 GiB. Most LeRobot training fits in <100 GB.
3. **Hard server-side TTL.** Set at spawn via the vendor's scheduled-delete API; the VM dies even if the GUI server dies.

Egress for checkpoints is the other big variable; see `CheckpointStore` above for the scratch/publish split that lets users put scratch on intra-vendor storage when available.

For Persistent hosts the GUI can't enforce these defenses; the user owns the VM. Rule of thumb: for occasional training, delete VM + disk between sessions — re-cloud-init takes a few minutes vs continuous disk billing.

---

## What we don't try to do

- **Pod provisioning outside the HostProvider protocol.** Ephemeral mode goes through a vetted vendor SDK with hard TTL, spend cap, and destroy-verification. The GUI does NOT embed vendor consoles or manage payment methods.
- **Multi-GPU / multi-node training.** Out of scope for v1. `gpu_count` is in the schema so the UI can grow into it later.
- **Browser-coupled providers** (Colab, Kaggle interactive). They kill on user-side idle regardless of GPU activity; no design from our side defeats that.
- **Vendor marketplaces with quality variance** (Vast.ai). Power users can still use them through the Persistent SSH path; we just don't market them as managed integrations.
- **In-GUI credential entry.** No "paste your API key" forms. Credentials live in the user's local CLI state; the helper bridges browser ↔ local state.
- **Concrete cost predictions.** No "this run will cost $X" claims. The GUI surfaces a pricing-page link per host and store; vendor pricing changes too often for us to be accurate.

---

## Future enhancements (parked — must not be blocked by v1)

- **Auto-discovery of the user's machine** via the auth helper's `/system-info` endpoint. Would add the user's laptop as a candidate Persistent host if reachable via SSH from the GUI server. Architectural hook: a `HostDiscoverer` interface; today's only impl reads `nvidia-smi` on the GUI server.
- **mDNS-based LAN host advertisement.** Each opted-in GPU server runs a tiny advert daemon; clients see them on the LAN. Same `HostDiscoverer` interface. Passive advertisement (opt-in by the host owner) is the right pattern; active port-scanning would be an antipattern.
- **Helper install-script / auto-start unit.** v1: each user runs `lerobot auth-helper start` on their laptop. v2: one-time install creates the launch-agent / systemd-user unit so it auto-starts at login.
- **HF OAuth in browser.** Right answer if HF's long-lived-token model becomes a hard problem. Helper architecture doesn't change — only the HF endpoint inside it.
- **Multi-GPU support.** `gpu_count` field already present.
- **Post-run billing reconciliation.** When vendor billing APIs are available (Nebius's `nebius billing`, AWS Cost Explorer, etc.), surface estimated vs actual cost after a run completes.
- **Additional `CheckpointStore` implementations.** S3, GCS, local NAS over rsync. HF Hub is the v1 default + publish target.
