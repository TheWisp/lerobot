# Model training — architecture

End-to-end design for taking a user from "I want to train a policy" to "trained model in a checkpoint store", driven from the LeRobot GUI.

Companion to [`gui/docs/hub_transfers.md`](../../src/lerobot/gui/docs/hub_transfers.md) — training reuses Hub Transfers' worker-IPC + polling + upload patterns wholesale.

---

## Use cases

Two host modes by lifecycle ownership. The workstation-GPU case is the easy slice of Persistent.

### A. Workstation (auto-registered Persistent host)

A developer with a GPU on the same machine that runs the GUI server. On first start, the GUI detects the local GPU and auto-registers it as a host called "This server". The developer picks dataset + recipe + hyperparams in the browser and starts training; the run happens as a subprocess on the same machine; checkpoints push to the configured store.

### B. User-added Persistent host (lab box, leased VM, user's laptop)

A developer with a remote machine they manage themselves — lab server, university cluster, leased VM, or their own laptop reachable from the GUI server. They paste an SSH command into the GUI's "Add training host" dialog; the GUI probes reachability and capabilities (GPU, container runtime, package manager) and saves the profile. Training runs over SSH inside a detached session on that host; the GUI never creates or destroys the VM — that's the user's responsibility, as is paying for it.

### C. Ephemeral cloud host (auto-managed VM)

A developer who wants on-demand cloud GPUs without touching the vendor console. They configure a host profile with a GPU class, region hint, spend cap, and TTL. On training start, the provider spawns a VM with those parameters; the GUI runs training as in case B; on completion or TTL hit, the provider destroys the VM and verifies no orphan disks/IPs remain.

The provider abstraction lets us add new vendors without changing the rest of the GUI.

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

### Components

Four components, each with a defined responsibility:

```
  Browser
    │ HTTP/WS
    ▼
  GUI server
    │ SSH or subprocess
    ▼
  Training host
    │ HTTPS
    ▼
  Checkpoint store
```

- **Browser** — the user's interface. Holds no credentials directly; reads UI state from the GUI server and pulls short-lived tokens from the local credential helper on demand.
- **GUI server** — orchestrates training. Discovers local GPU at startup, holds host profiles + run state, talks to providers and stores. Never holds long-lived credentials.
- **Training host** — where the training process runs. The GUI server's own box (subprocess transport) or any machine reachable by SSH (SSH transport). The choice of transport is the only thing that distinguishes the workstation case from a remote case.
- **Checkpoint store** — where checkpoints land. Decoupled from the training host so users can pair cheap intra-vendor storage with remote training.

The browser may run on the same machine as the GUI server (workstation) or on a different one (LAN). The training host may collapse with the GUI server, be a user-managed remote box, or be a vendor-spawned VM. The checkpoint store is always separate.

### HostProvider

A vendor adapter for VM lifecycle. Each provider implementation handles one vendor.

**What it provides:**

- **estimate_cost** — a pure function from a spawn spec to a projected cost; no side effects. The caller uses this to reject expensive specs before any provisioning.
- **spawn** — provision a VM matching the spec, attach disk, allocate public IP, wait until reachable. The provider MUST configure a server-side TTL so the VM dies even if the GUI server loses control of it.
- **destroy** — idempotently destroy the VM, attached boot disk, and public IP allocation. Does not touch any persistent volume that's lifecycled separately.
- **verify_destroyed** — query the vendor to confirm no billable resource identifying this handle remains. Automates the "always check the console for orphans after teardown" hygiene rule.
- **current_cost** — best-effort accrued cost for UI display.

**What a spawn spec carries** (vendor-neutral):
GPU class, GPU count (v1: always 1), preemptibility, boot disk size, container image, region hint, hard TTL, and a cost ceiling above which spawn refuses. The provider's `estimate_cost` MUST agree with the ceiling before `spawn` is called.

**What a host handle carries:**
provider id, vendor's resource id, a transport (SSH or subprocess), an optional persistent volume id, region, and the provider's hard-TTL deadline as a wall-clock time.

**Deliberately omitted:**

- **Start / stop.** Not every vendor supports stop; a stopped VM still bills its disk. Every off-switch is destroy.
- **SSH command execution and log tailing.** Vendor-agnostic; live in the transport layer above the provider.
- **Auth flow.** Handled by the credential helper (below); credentials injected per-request as a provider constructor arg.
- **Upload / download.** Checkpoints flow through CheckpointStore; providers don't move files.

The Persistent provider is a degenerate implementation: spawn raises (the host already exists), destroy is a no-op, cost is zero. The auto-registered workstation host has subprocess transport; user-added hosts have SSH transport.

### CheckpointStore

Parallel to HostProvider but for where checkpoints land. Each implementation handles one storage destination.

**What it provides:**

- **push** — upload a local checkpoint to the store, returning a stable URI.
- **pull** — fetch a stored checkpoint to a local path (used on resume).
- **list_checkpoints** — enumerate checkpoints for a given run id; the UI uses this to surface "latest" and let the user pick a different one.
- **pricing_info_url** — a link to the vendor's current pricing page. The UI surfaces this so users can read live rates themselves; we do not claim to know prices that drift.

A training run can use two stores:

- **scratch store** — frequent checkpoints during training. Defaults to whatever the host's vendor-local store is (e.g., the vendor's S3-compatible bucket); falls back to the publish store if no vendor-local option is configured.
- **publish store** — final model destination. HF Hub by default.

Why the split: scratch checkpoints can be hundreds of GiB-writes per run, and egress to a cross-network store can dominate compute cost. Putting scratch on a vendor-local store (typically free or near-free egress within the same network) avoids that. The publish store gets a single final upload of the trained model.

### Transport, log surfaces

Training launches in a detached session on the host (tmux for SSH transport, managed subprocess for subprocess transport), invoking a worker process. The GUI server never holds an open pipe to training. Polling is via short-lived reads of three log surfaces, identical across both transports:

- **progress.json** — atomically-rewritten snapshot (loss, step, ETA, GPU util)
- **events.jsonl** — append-only audit log of state transitions (connect, fail, retry, lost_contact)
- **stderr.log** — byte-offset incremental tail

### Session survival

The user's browser or laptop closing is a non-event. Polling stops; training on the host keeps running (detached from the session); auto-push from inside the training script keeps running. On wake, polling resumes and the run card shows current step.

Browser-close in Ephemeral mode is the same property: training continues via the SSH key (no provider token needed); on resume, the GUI re-fetches tokens from the helper.

### Connection resilience

The poll scheduler uses exponential backoff (5/10/20/40/60s capped, 10 attempts, ~5 min before give-up). Transient errors retry with backoff; permanent errors (auth failed, bad host key) give up immediately. Transitions surface as `events.jsonl` entries and a connection-quality indicator in the UI.

### Recovery from preemption

The primary mechanism is periodic checkpointing to host disk with async background upload to the scratch store. The grace window between SIGTERM and SIGKILL is too short for big models to flush on signal, so the most recently uploaded checkpoint is what survives. Cadence defaults to every 60–180s of training time for preemptible hosts and ~10 minutes for on-demand; tunable per recipe.

### Authentication

When the GUI backend is shared (LAN deployment), credentials must be per-user, must not leak between users, and must not persist on the backend.

**The model:**

- Long-lived credentials live on the user's laptop in the standard CLI tool locations (vendor profile directory, HF cache).
- A small helper daemon on the user's laptop mints short-lived tokens on demand — for vendors with IAM via long-lived service-account keys, by shelling out to the vendor CLI; for HF, by reading the cached token.
- The browser fetches fresh tokens from the helper on demand (loopback only, origin-checked, plus a pairing token). It includes them in vendor-specific headers on every request to the GUI server.
- The GUI server reads token headers per request, constructs the provider with credentials in scope, makes the call, discards.

**Workstation case** (GUI binds the loopback interface): the helper is auto-launched as a child of the GUI server with auto-pairing. The GUI server runs only its own process; nothing else to manage.

**Shared-LAN case**: the GUI server runs only its own process — no second service. Each user runs the helper once on their own laptop. The pairing token is pasted into GUI Settings once; from then on, tokens rotate transparently.

**Security properties:**

- Long-lived creds stay on the user's laptop; backend has zero credential persistence.
- Helper bound to loopback only; rejects requests missing matching Origin and pairing token.
- Compromised backend worst case: short-lived tokens (typically ≤12h) for currently-connected users; no refresh creds, no long-lived secrets.

**HF caveat.** The HF token is non-expiring. Subprocess isolation (Hub Transfers already uses a separate worker process) confines it to that subprocess's RAM. Users wanting tighter scope can switch to a fine-grained HF token with explicit expiry — no code change needed.

### Cost discipline

Vendor-neutral properties to defend against:

**Disks bill from creation to deletion, regardless of VM state.** A stopped VM stops compute; the attached disk continues at $/GiB/month. A large attached disk left running can cost more than the GPU did.

Three structural defenses baked into Ephemeral mode:

1. **Inline-managed boot disk only.** Disk lifecycle cascades with the VM; no standalone disks unless the user opts in.
2. **Disk-size warning.** Many vendor web forms default to enormous boot disks (1 TB+); the GUI warns above 256 GiB. Most LeRobot training fits in <100 GB.
3. **Hard server-side TTL.** Set at spawn via the vendor's scheduled-delete mechanism; the VM dies even if the GUI server dies.

Egress for checkpoints is the other big variable; see CheckpointStore above for the scratch/publish split that lets users put scratch on intra-vendor storage when available.

For Persistent hosts the GUI cannot enforce these defenses; the user owns the VM. Rule of thumb: for occasional training, delete VM + disk between sessions.

---

## What we don't try to do

- **Pod provisioning outside the HostProvider protocol.** Ephemeral mode goes through a vetted vendor SDK with hard TTL, spend cap, and destroy-verification. The GUI does NOT embed vendor consoles or manage payment methods.
- **Multi-GPU / multi-node training.** Out of scope for v1. The `gpu_count` field is in the schema so the UI can grow into it later.
- **Browser-coupled providers** (Colab, Kaggle interactive). They kill on user-side idle regardless of GPU activity; no design from our side defeats that.
- **Vendor marketplaces with quality variance** (Vast.ai). Power users can still use them through the Persistent SSH path; we just don't market them as managed integrations.
- **In-GUI credential entry.** No "paste your API key" forms. Credentials live in the user's local CLI state; the helper bridges browser ↔ local state.
- **Concrete cost predictions.** No "this run will cost $X" claims. The GUI surfaces pricing-page links per host and store; vendor pricing changes too often for us to be accurate.

---

## Future enhancements (parked — must not be blocked by v1)

- **Auto-discovery of the user's machine** via the auth helper's system-info endpoint. Would add the user's laptop as a candidate Persistent host if reachable via SSH from the GUI server. Architectural hook: a `HostDiscoverer` interface; today's only impl reads `nvidia-smi` on the GUI server.
- **mDNS-based LAN host advertisement.** Each opted-in GPU server runs a tiny advert daemon; clients see them on the LAN. Same `HostDiscoverer` interface. Passive advertisement (opt-in by the host owner) is the right pattern; active port-scanning would be an antipattern.
- **Helper install-script / auto-start unit.** v1: each user runs the helper on their laptop. v2: one-time install creates a launch-agent / systemd-user unit so it auto-starts at login.
- **HF OAuth in browser.** Right answer if HF's long-lived-token model becomes a hard problem. Helper architecture doesn't change — only the HF endpoint inside it.
- **Multi-GPU support.** `gpu_count` field already present.
- **Post-run billing reconciliation.** When vendor billing APIs are available, surface estimated vs actual cost after a run completes.
- **Additional CheckpointStore implementations.** S3, GCS, local NAS over rsync. HF Hub is the v1 default + publish target.

---

## Setup reference

Commands needed per deployment shape. The GUI server never runs more than `lerobot-gui`.

| Deployment                                  | On the GUI server | On each user's laptop                                                                                                                                  |
| ------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Workstation** (GUI + GPU on same machine) | `lerobot-gui`     | same machine — nothing extra (auth helper is a child of `lerobot-gui`); user already ran `huggingface-cli login` to get HF token in the standard cache |
| **Shared LAN — Persistent hosts only**      | `lerobot-gui`     | `huggingface-cli login`; `lerobot auth-helper start --origin <gui-url>`; paste pairing token into Settings once                                        |
| **Shared LAN — with Ephemeral cloud**       | `lerobot-gui`     | as above, plus `<vendor> profile create` once per cloud account                                                                                        |
