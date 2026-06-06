# Model training — architecture

End-to-end design for taking a user from "I want to train a policy" to "trained model in a checkpoint store", driven from the LeRobot GUI.

---

## Modes

| Mode                                         | Who it's for                                                                                                                                                                             | Lifecycle                                                                       |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Workstation** (auto-registered Persistent) | Developer with a GPU on the same machine that runs the GUI server. GUI detects on first start and registers the host as "This server".                                                   | User-managed — the user owns the machine.                                       |
| **User-added Persistent host**               | Developer with a remote SSH-reachable box — lab server, university cluster, leased VM, or their own laptop reachable from the GUI server. They paste an SSH command into the GUI dialog. | User-managed — the GUI never creates or destroys the VM.                        |
| **Ephemeral cloud host**                     | Developer who wants on-demand cloud GPUs without touching the vendor console. They configure a host profile with a GPU class, region hint, and spend cap.                                | Provider-managed — GUI spawns the VM on training start, destroys on completion. |

"Persistent" vs "Ephemeral" is about lifecycle ownership, orthogonal to spot/preemptible pricing. An Ephemeral host typically uses a preemptible instance underneath.

A **training host** is any of the above. A **checkpoint store** is where checkpoints land, configured separately from the host (see Architecture).

---

## Architecture

### Components

```
  Browser
    │ HTTP/WS
    ▼
  GUI server
    │ SSH or subprocess
    ▼
  Training host  ──HTTPS──►  Checkpoint store
```

- **Browser** — the user's interface. Holds no credentials directly; pulls short-lived tokens from the local credential helper on demand and forwards them to the GUI server in request headers.
- **GUI server** — orchestrates training. Discovers a local GPU at startup, holds host profiles + run state, talks to providers. Never holds long-lived credentials.
- **Training host** — where the training process runs. The GUI server's own box (subprocess transport) or any machine reachable by SSH (SSH transport). The choice of transport is the only thing that distinguishes the workstation case from a remote case.
- **Checkpoint store** — where checkpoints land. The training process on the host uploads directly to the store; the GUI server doesn't shuttle bytes through itself.

### HostProvider

A vendor adapter for VM lifecycle. Each implementation handles one vendor.

**What it provides:**

- **spawn** — provision a VM matching the spec, attach disk, allocate public IP, wait until reachable.
- **destroy** — idempotently destroy the VM, attached boot disk, and public IP allocation. Does not touch any persistent volume that's lifecycled separately.
- **verify_destroyed** — query the vendor to confirm no billable resource identifying this handle remains. Automates the "always check the console for orphans after teardown" hygiene rule.

**What a spawn spec carries** (vendor-neutral):
GPU class, GPU count (v1: always 1), preemptibility, boot disk size, container image, region hint, and a cost ceiling above which spawn refuses.

**What a host handle carries:**
provider id, vendor's resource id, a transport (SSH or subprocess), an optional persistent volume id, region.

**Deliberately omitted:**

- **Start / stop.** Not every vendor supports stop; a stopped VM still bills its disk. Every off-switch is destroy.
- **SSH command execution and log tailing.** Vendor-agnostic; live in the transport layer above the provider.
- **Auth flow.** Handled by the credential helper (below); credentials injected per-request as a provider constructor arg.
- **Cost reporting.** v1 doesn't surface running cost. Users read the vendor's pricing page (linked in the host config UI).

The Persistent provider is a degenerate implementation: spawn raises (the host already exists), destroy is a no-op. The auto-registered workstation host has subprocess transport; user-added hosts have SSH transport.

### CheckpointStore

Where the trained model lands. Each implementation handles one storage destination; the training process on the host uploads to the store using credentials passed through env vars at spawn time.

**What it provides:**

- **push** — upload a local checkpoint, returning a stable URI (a Hub URL, an S3 key, a NAS path).
- **pull** — fetch a stored checkpoint to a local path (used on resume from a prior run's checkpoint).
- **list_checkpoints** — enumerate checkpoints for a given run id; the UI uses this to show what's been uploaded.
- **pricing_info_url** — a link to the vendor's pricing page (egress / storage). The UI surfaces this; we don't claim to know prices that drift.

**Adding and configuring a store:**

Stores are configured in a Stores section of GUI Settings, parallel to host profiles. Each store type has its own setup:

- **HF Hub** (v1 default): needs the user's HF token (already available via the credential helper). The user picks a target repo or namespace prefix to auto-create per run. No additional setup.
- _(future)_ **Vendor-local object store** (S3-compatible bucket on the same cloud vendor as the training host): would need the vendor's storage credentials, typically the same auth path as the compute provider.
- _(future)_ **External S3 / GCS**: needs separate cloud credentials; would extend the helper with a new endpoint.
- _(future)_ **Local NAS over rsync / SFTP**: needs a target host + path + SSH key.

**Surfacing the result:**

When training completes, the GUI shows the store URI as a clickable link — for HF Hub, a direct link to the repo page; for an S3 bucket, the bucket URL; for a NAS path, the file path. The run history persists the URI so the user can find it later. Resume-from-checkpoint uses the same URI to pull the latest checkpoint onto a new host.

**Scratch vs publish stores** (future): a separate scratch store for frequent in-training checkpoints could be paired with a publish store for the final model — useful when scratch lives on intra-vendor storage (cheap egress) and publish lives on HF Hub (shareable). v1 uses a single store throughout, which is fine for HF Hub (uploads are free under normal quotas).

### Transport, log surfaces

Training launches in a detached session on the host (tmux for SSH transport, managed subprocess for subprocess transport), invoking a worker process. The GUI server never holds an open pipe to training. Polling is via short-lived reads of three log surfaces, identical across both transports:

- **progress.json** — atomically-rewritten snapshot (loss, step, ETA, GPU util)
- **events.jsonl** — append-only audit log of state transitions (connect, fail, retry, lost_contact)
- **stderr.log** — byte-offset incremental tail

### Session survival

What survives what, by deployment shape:

| Event                                          | LAN deployment (GUI server on its own box)                                                                                  | Workstation deployment (GUI server is the user's laptop)                                                                               |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| User closes browser                            | Non-event. GUI server keeps polling; training continues. On reopen the frontend reads current state from the server.        | Non-event by itself; typically coincident with closing the laptop (next row).                                                          |
| Laptop sleeps                                  | Non-event for the GUI server (different machine).                                                                           | GUI server suspends with the laptop; polling stops. Training subprocess survives or not depending on OS sleep — outside GUI's control. |
| GUI server restarts                            | Polling pauses. On restart, GUI server reads persisted run state and resumes polling. Training in tmux survives (detached). | Same — GUI picks up where it left off; tmux-based training survives.                                                                   |
| Training host loses contact (preemption, etc.) | Detected by the poll scheduler; UI marks the run unhealthy. See Health.                                                     | Same.                                                                                                                                  |

### Connection resilience

The poll scheduler uses exponential backoff (5/10/20/40/60s capped, 10 attempts, ~5 min before give-up). Transient errors retry with backoff; permanent errors (auth failed, bad host key) give up immediately. Transitions surface as `events.jsonl` entries and a connection-quality indicator in the UI.

### Recovery from preemption

The primary mechanism is periodic checkpointing to host disk with async background upload to the store. The grace window between SIGTERM and SIGKILL is too short for big models to flush on signal, so the most recently uploaded checkpoint is what survives. Cadence defaults to every 60–180s of training time for preemptible hosts and ~10 minutes for on-demand; tunable per recipe.

### Health

The GUI server monitors each running training for liveness:

- **Liveness signals:** heartbeat in `progress.json` (updated each step), non-zero training step rate over a rolling window, non-zero GPU utilization (catches stuck-on-data-loading).
- **Unhealthy:** the UI marks the run; the user can attempt resume or destroy.
- **Completion:** GUI calls `provider.destroy()` with the most recently cached provider token. If that token has expired and no one is connected, the run is marked "awaiting destroy" and destroyed on the next browser session.
- **Backstop:** spawn configures a vendor-side scheduled delete (default 24h, extended on healthy progress) so a VM eventually dies even if the GUI server permanently disappears. Not a user-facing config — an implementation safety net.

Forgotten-pod failure mode: if the user spawns an Ephemeral run and permanently disappears, the vendor scheduled-delete catches it within 24h. Vendor billing alerts are the final safety net.

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

**HF caveat.** The HF token is non-expiring. Subprocess isolation (training scripts run as separate processes) confines it to that subprocess's RAM. Users wanting tighter scope can switch to a fine-grained HF token with explicit expiry — no code change needed.

### Cost discipline

Vendor-neutral properties to defend against:

**Disks bill from creation to deletion, regardless of VM state.** A stopped VM stops compute; the attached disk continues at $/GiB/month. A large attached disk left running can cost more than the GPU did.

Three structural defenses baked into Ephemeral mode:

1. **Inline-managed boot disk only.** Disk lifecycle cascades with the VM; no standalone disks unless the user opts in.
2. **Disk-size warning.** Many vendor web forms default to enormous boot disks (1 TB+); the GUI warns above 256 GiB. Most LeRobot training fits in <100 GB.
3. **Auto-destroy on completion + backstop scheduled-delete.** See Health.

Egress for checkpoints is the other big variable; v1 uses HF Hub uploads (free under normal quotas). The future scratch/publish split (see CheckpointStore) would let users put frequent checkpoints on intra-vendor storage when integrating storage destinations with paid egress.

For Persistent hosts the GUI cannot enforce these defenses; the user owns the VM. Rule of thumb: for occasional training, delete VM + disk between sessions.

---

## What we don't try to do

- **Pod provisioning outside the HostProvider protocol.** Ephemeral mode goes through a vetted vendor SDK with spend cap and destroy-verification. The GUI does NOT embed vendor consoles or manage payment methods.
- **Multi-GPU / multi-node training.** Out of scope for v1. The `gpu_count` field is in the schema so the UI can grow into it later.
- **Browser-coupled providers** (Colab, Kaggle interactive). They kill on user-side idle regardless of GPU activity; no design from our side defeats that.
- **Vendor marketplaces with quality variance** (Vast.ai). Power users can still use them through the Persistent SSH path; we just don't market them as managed integrations.
- **In-GUI credential entry.** No "paste your API key" forms. Credentials live in the user's local CLI state; the helper bridges browser ↔ local state.
- **Concrete cost predictions.** No "this run will cost $X" claims. The GUI surfaces pricing-page links per host and store; vendor pricing changes too often for us to be accurate.

---

## Future enhancements (parked — must not be blocked by v1)

- **Auto-discovery of the user's machine** via the auth helper's system-info endpoint. Would add the user's laptop as a candidate Persistent host if reachable via SSH from the GUI server.
- **mDNS-based LAN host advertisement.** Each opted-in GPU server runs a tiny advert daemon; clients see them on the LAN. Passive advertisement (opt-in by the host owner) is the right pattern; active port-scanning would be an antipattern.
- **Helper install-script / auto-start unit.** v1: each user runs the helper on their laptop. v2: one-time install creates a launch-agent / systemd-user unit so it auto-starts at login.
- **HF OAuth in browser.** Right answer if HF's long-lived-token model becomes a hard problem.
- **Multi-GPU support.** `gpu_count` field already present.
- **Post-run billing reconciliation.** When vendor billing APIs are available, surface estimated vs actual cost after a run completes.
- **Additional `CheckpointStore` implementations** (vendor S3, external S3 / GCS, local NAS) with a **scratch-vs-publish split** to avoid egress when integrating paid-egress destinations. HF Hub is the v1 default + publish target.

---

## Setup reference

Commands needed per deployment shape. The GUI server never runs more than `lerobot-gui`.

| Deployment                                  | On the GUI server | On each user's laptop                                                                                                                                  |
| ------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Workstation** (GUI + GPU on same machine) | `lerobot-gui`     | same machine — nothing extra (auth helper is a child of `lerobot-gui`); user already ran `huggingface-cli login` to get HF token in the standard cache |
| **Shared LAN — Persistent hosts only**      | `lerobot-gui`     | `huggingface-cli login`; `lerobot auth-helper start --origin <gui-url>`; paste pairing token into Settings once                                        |
| **Shared LAN — with Ephemeral cloud**       | `lerobot-gui`     | as above, plus `<vendor> profile create` once per cloud account                                                                                        |
