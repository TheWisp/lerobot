# Model training — architecture

End-to-end design for taking a user from "I want to train a policy" to "trained model in the Models tab (and optionally on Hugging Face)", driven from the LeRobot GUI.

---

## Modes

| Mode                                         | Who it's for                                                                                                                                                                             | Lifecycle                                                                       |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Workstation** (auto-registered Persistent) | Developer with a GPU on the same machine that runs the GUI server. GUI detects on first start and registers the host as "This server".                                                   | User-managed — the user owns the machine.                                       |
| **User-added Persistent host**               | Developer with a remote SSH-reachable box — lab server, university cluster, leased VM, or their own laptop reachable from the GUI server. They paste an SSH command into the GUI dialog. | User-managed — the GUI never creates or destroys the VM.                        |
| **Ephemeral cloud host**                     | Developer who wants on-demand cloud GPUs without touching the vendor console. They configure a host profile with a GPU class, region hint, and spend cap.                                | Provider-managed — GUI spawns the VM on training start, destroys on completion. |

"Persistent" vs "Ephemeral" is about lifecycle ownership, orthogonal to spot/preemptible pricing. An Ephemeral host typically uses a preemptible instance underneath.

A **training host** is any of the above. Trained models land in the existing **Models tab**, and from there the user can publish them to external destinations (Hugging Face Hub today).

---

## Architecture

### Components

```
  Browser
    │ HTTP/WS
    ▼
  GUI server  ──HTTPS──►  External destinations (HF Hub, …)
    │ SSH or subprocess
    ▼
  Training host
```

- **Browser** — the user's interface. Holds no credentials directly; pulls short-lived tokens from the local credential helper on demand and forwards them to the GUI server in request headers.
- **GUI server** — orchestrates training and owns model storage. Discovers a local GPU at startup; holds host profiles + run state + the canonical checkpoint store; talks to providers. Never holds long-lived credentials.
- **Training host** — where the training process runs. The GUI server's own box (subprocess transport) or any machine reachable by SSH (SSH transport). Writes checkpoints to local disk; does NOT talk to external destinations.
- **External destinations** — places the user can publish models to (HF Hub today; S3, GCS, NAS later). Pushed from the GUI server only, never from the training host, never automatically without user opt-in.

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

### Checkpoints

The training process writes periodic checkpoints to the host's local disk. The GUI server pulls each new checkpoint back over the existing transport (SCP for SSH, direct file copy for subprocess) into its own storage at `~/.cache/lerobot/runs/<run_id>/<step>/`. The host's local copy is cleaned up after a successful pull.

This means:

- **The training pod doesn't need external credentials.** No HF token, no S3 keys, no per-vendor auth on the pod. The pod only writes to its local disk; the GUI server reads back over the SSH/subprocess channel it already has.
- **Pod egress cost is the same.** Bytes still leave the pod's network; they go to the GUI server instead of HF Hub. The pod-side egress is identical to a direct-to-HF push.
- **The Models tab is the canonical surface.** A pulled checkpoint appears there immediately — before any external upload.
- **External publication is a separate Models-tab action.** The user clicks "Push to Hugging Face" on a model row; the GUI server uses the user's HF token (from the helper) to push. Can be set to automatic per run or per recipe.

For workstation deployment, training writes directly to the GUI server's disk (subprocess transport) — there's no pull step; the checkpoint is already in the right place.

**Trade-off.** For Ephemeral pods in the cloud with the GUI server on a LAN box, the pod→GUI pull may be slower than a hypothetical direct pod→HF push (LAN bandwidth, intercontinental latency). The cost on the pod side is the same either way; the difference is wall-clock to durability. Acceptable for v1; future enhancement options include direct pod→external streaming or a vendor-local intermediate.

**Disk discipline.** GUI server applies a retention policy per run (default: keep latest N checkpoints, drop older). For large models on disk-constrained LAN boxes, the user can configure the retention count or push-and-evict to an external destination automatically.

### Transport, log surfaces

Training launches in a detached session on the host (tmux for SSH transport, managed subprocess for subprocess transport), invoking a worker process. The GUI server never holds an open pipe to training. Polling is via short-lived reads of three log surfaces, identical across both transports:

- **progress.json** — atomically-rewritten snapshot (loss, step, ETA, GPU util)
- **events.jsonl** — append-only audit log of state transitions (connect, fail, retry, lost_contact)
- **stderr.log** — byte-offset incremental tail

The same poll loop that reads these also scans the host's checkpoint directory and SCPs new files.

### Session survival

What survives what, by deployment shape:

| Event                                          | LAN deployment (GUI server on its own box)                                                                                   | Workstation deployment (GUI server is the user's laptop)                                                                               |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| User closes browser                            | Non-event. GUI server keeps polling and pulling checkpoints; training continues. On reopen the frontend reads current state. | Non-event by itself; typically coincident with closing the laptop (next row).                                                          |
| Laptop sleeps                                  | Non-event for the GUI server (different machine).                                                                            | GUI server suspends with the laptop; polling stops. Training subprocess survives or not depending on OS sleep — outside GUI's control. |
| GUI server restarts                            | Polling pauses. On restart, GUI server reads persisted run state and resumes polling. Training in tmux survives (detached).  | Same — GUI picks up where it left off; tmux-based training survives.                                                                   |
| Training host loses contact (preemption, etc.) | Detected by the poll scheduler; UI marks the run unhealthy. See Health.                                                      | Same.                                                                                                                                  |

### Connection resilience

The poll scheduler uses exponential backoff (5/10/20/40/60s capped, 10 attempts, ~5 min before give-up). Transient errors retry with backoff; permanent errors (auth failed, bad host key) give up immediately. Transitions surface as `events.jsonl` entries and a connection-quality indicator in the UI.

### Recovery from preemption

The primary mechanism is periodic checkpointing to host disk with async background pull by the GUI server. The grace window between SIGTERM and SIGKILL is too short for big models to flush on signal, so the most recently pulled checkpoint is what survives. Cadence defaults to every 60–180s of training time for preemptible hosts and ~10 minutes for on-demand; tunable per recipe.

If the user enabled "auto-push to HF" for the run, the GUI server pushes each pulled checkpoint to HF Hub asynchronously — a third copy, made out of band of training.

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
- The GUI server reads token headers per request, uses the token in scope (constructs a provider for spawn/destroy, or invokes the HF client for a Push-to-HF action), discards.

**Workstation case** (GUI binds the loopback interface): the helper is auto-launched as a child of the GUI server with auto-pairing. The GUI server runs only its own process; nothing else to manage.

**Shared-LAN case**: the GUI server runs only its own process — no second service. Each user runs the helper once on their own laptop. The pairing token is pasted into GUI Settings once; from then on, tokens rotate transparently.

**Security properties:**

- Long-lived creds stay on the user's laptop; backend has zero credential persistence.
- Helper bound to loopback only; rejects requests missing matching Origin and pairing token.
- Compromised backend worst case: short-lived tokens (typically ≤12h) for currently-connected users; no refresh creds, no long-lived secrets.
- **Training pods never receive HF tokens or other external credentials.** Pulled checkpoints flow over the existing SSH/subprocess channel; external pushes happen from the GUI server with the user's token only when they click Push.

**HF caveat.** The HF token is non-expiring. Push actions invoke an HF subprocess (the existing Hub Transfers worker pattern); the token lives in that subprocess's RAM and is GC'd with it. Users wanting tighter scope can switch to a fine-grained HF token with explicit expiry — no code change needed.

### Cost discipline

Vendor-neutral properties to defend against:

**Disks bill from creation to deletion, regardless of VM state.** A stopped VM stops compute; the attached disk continues at $/GiB/month. A large attached disk left running can cost more than the GPU did.

Three structural defenses baked into Ephemeral mode:

1. **Inline-managed boot disk only.** Disk lifecycle cascades with the VM; no standalone disks unless the user opts in.
2. **Disk-size warning.** Many vendor web forms default to enormous boot disks (1 TB+); the GUI warns above 256 GiB. Most LeRobot training fits in <100 GB.
3. **Auto-destroy on completion + backstop scheduled-delete.** See Health.

Egress for checkpoints is determined by the pod-to-GUI-server hop (see Checkpoints). The cost on the pod side is the same whether checkpoints go to the GUI server or to HF Hub directly. Pushing from the GUI server to HF is on the GUI server's network, which is typically free or not the bottleneck.

For Persistent hosts the GUI cannot enforce these defenses; the user owns the VM. Rule of thumb: for occasional training, delete VM + disk between sessions.

---

## What we don't try to do

- **Pod provisioning outside the HostProvider protocol.** Ephemeral mode goes through a vetted vendor SDK with spend cap and destroy-verification. The GUI does NOT embed vendor consoles or manage payment methods.
- **Multi-GPU / multi-node training.** Out of scope for v1. The `gpu_count` field is in the schema so the UI can grow into it later.
- **Browser-coupled providers** (Colab, Kaggle interactive). They kill on user-side idle regardless of GPU activity; no design from our side defeats that.
- **Vendor marketplaces with quality variance** (Vast.ai). Power users can still use them through the Persistent SSH path; we just don't market them as managed integrations.
- **In-GUI credential entry.** No "paste your API key" forms. Credentials live in the user's local CLI state; the helper bridges browser ↔ local state.
- **Concrete cost predictions.** No "this run will cost $X" claims. The GUI surfaces pricing-page links per host; vendor pricing changes too often for us to be accurate.
- **Direct pod-to-external-store uploads.** The pod uploads only to the GUI server. Any external destination (HF Hub today) is pushed from the GUI server. Simpler config; one credential boundary, not N.

---

## Future enhancements (parked — must not be blocked by v1)

- **Auto-discovery of the user's machine** via the auth helper's system-info endpoint. Would add the user's laptop as a candidate Persistent host if reachable via SSH from the GUI server.
- **mDNS-based LAN host advertisement.** Each opted-in GPU server runs a tiny advert daemon; clients see them on the LAN. Passive advertisement (opt-in by the host owner) is the right pattern; active port-scanning would be an antipattern.
- **Helper install-script / auto-start unit.** v1: each user runs the helper on their laptop. v2: one-time install creates a launch-agent / systemd-user unit so it auto-starts at login.
- **HF OAuth in browser.** Right answer if HF's long-lived-token model becomes a hard problem.
- **Multi-GPU support.** `gpu_count` field already present.
- **Post-run billing reconciliation.** When vendor billing APIs are available, surface estimated vs actual cost after a run completes.
- **Additional external push destinations from the Models tab** — S3, GCS, vendor-local object stores, local NAS over rsync. Same pattern: pushed from the GUI server using user-supplied credentials.
- **Direct pod-to-external streaming option** for very large models where pod→GUI pull is bandwidth-limited. Adds a second code path; only worth doing if the pull-through-GUI model proves a bottleneck.

---

## Setup reference

Commands needed per deployment shape. The GUI server never runs more than `lerobot-gui`.

| Deployment                                  | On the GUI server | On each user's laptop                                                                                                                                  |
| ------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Workstation** (GUI + GPU on same machine) | `lerobot-gui`     | same machine — nothing extra (auth helper is a child of `lerobot-gui`); user already ran `huggingface-cli login` to get HF token in the standard cache |
| **Shared LAN — Persistent hosts only**      | `lerobot-gui`     | `huggingface-cli login`; `lerobot auth-helper start --origin <gui-url>`; paste pairing token into Settings once                                        |
| **Shared LAN — with Ephemeral cloud**       | `lerobot-gui`     | as above, plus `<vendor> profile create` once per cloud account                                                                                        |
