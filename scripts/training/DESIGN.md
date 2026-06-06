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
- **External destinations** — places the user can publish models to (HF Hub today; S3, GCS, NAS later). Pushed from the GUI server only, never from the training host. Push is a user-initiated Models-tab action in v1; auto-push per run is a future enhancement.

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

The training process writes periodic checkpoints to the host's local disk **and** appends a line to a `checkpoints.jsonl` manifest after each one is fully written (atomic append). The manifest is the signal — the GUI server's poll loop reads it alongside `progress.json`, sees new entries, and brings the named files back.

Cadence is set by the training recipe (`save_steps` or equivalent); the GUI does not impose one. The cadence the user picks shapes the recovery story — tighter for preemptible hosts, looser for on-demand.

Each manifest line includes a `sha256` checksum of the checkpoint file. The GUI server verifies after pull; a mismatch surfaces in the UI with a retry option (catches partial writes, network corruption, disk faults).

How "bringing the files back" works, by transport:

- **Subprocess transport** (workstation): the files are already on the GUI server's disk. No copy. The GUI server opens them in place; the manifest still drives indexing into the Models tab.
- **SSH transport** (Persistent / Ephemeral): the GUI server SCPs the named files back to `~/.cache/lerobot/runs/<run_id>/<step>/` over the existing SSH connection. The host's local copy is left in place — for Ephemeral hosts it's discarded with the VM; for Persistent hosts the user manages their own cleanup as they normally would.

**This means:**

- **The training pod's external credential surface is minimal.** The pod receives the HF token (env var at spawn time, scoped to that run) for dataset download from HF Hub — and nothing else. No vendor cloud credentials, no checkpoint-store credentials, no S3 keys. Pod compromise leaks at most the user's HF token, not their cloud account.
- **Pod egress cost is the same.** Bytes leave the pod's network either way; they go to the GUI server instead of an external store. The pod-side cost is identical to a direct-to-HF push.
- **The Models tab is the canonical surface.** A pulled checkpoint appears there immediately — before any external upload.
- **External publication is a separate Models-tab action** (v1: user-initiated; auto-push is a future enhancement).

**Trade-off.** The pull-based design is simpler than the alternative (pod pushes to a central artifact store): no per-store credentials on the pod, one credential boundary, no artifact-store infrastructure. The cost: the GUI server's network and disk are the bottleneck. Acceptable at our scale. At 10+ concurrent runs with very large models, a push-to-shared-store model would be the right rethink — but that day is far off.

For Ephemeral pods in the cloud with the GUI server on a LAN box, pod→GUI may also be slower than a hypothetical direct pod→external push (LAN bandwidth, intercontinental latency). Pod-side cost is the same; the difference is wall-clock to durability.

**Resume from checkpoint.** The Models tab lets the user pick a checkpoint and start a new run resuming from it:

1. User picks a checkpoint and a host (any of the three modes).
2. GUI server spawns the host (Ephemeral) or selects an idle one (Persistent / workstation).
3. Before launching training, the GUI server pushes the chosen checkpoint to the host's local disk (SCP for SSH, file copy for subprocess).
4. The training worker reads a standard `--resume_from_checkpoint <path>` argument and continues from that step.

The new run gets its own run id; the resumed-from checkpoint is recorded in the run metadata so the lineage is queryable.

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

The primary mechanism is periodic checkpointing to host disk with async background pull by the GUI server. The grace window between SIGTERM and SIGKILL is too short for big models to flush on signal, so the most recently pulled checkpoint is what survives.

Cadence is set by the training recipe (`save_steps` or equivalent), not by the orchestration layer. Time-to-recovery is a function of step rate × `save_steps` × model size — properties of the recipe, not knobs the GUI imposes. Users picking preemptible hosts should choose a tighter `save_steps` than they would for on-demand.

### Health

Liveness comes from two independent signals over the channel we already have. Neither relies on parsing stderr.

- **Process probe** — over SSH transport, `pgrep` against the worker's pid file on the host. Over subprocess transport, `subprocess.poll()`. Definitive answer to "is the training process alive?"
- **Progress freshness** — `progress.json` is atomically rewritten on every training step and includes a timestamp. If the timestamp hasn't advanced over a rolling window (e.g., 5 min), training is stuck.

A run is unhealthy when the process is dead (crashed) OR the progress timestamp is stale (stuck). The UI marks the run; the user can attempt resume or destroy.

Why not Wandb-style HTTP heartbeats from the training script? Those require the pod to reach the GUI server, which often fails (NAT, firewall, no inbound on the lab box). Our setup is the inverse — the GUI server reaches the pod — so we use the channel we already have. The combination of process probe + atomic progress file gives the same coverage without the extra reachability requirement.

**Completion signal.** The worker writes a final line to `events.jsonl` before exiting:

- `{type: "completed_naturally", exit_code: 0, final_step: N}` — recipe's step limit reached
- `{type: "aborted_by_user", final_step: N}` — user clicked Stop
- `{type: "crashed", exit_code: ..., error: "..."}` — unexpected exit

The GUI reads this — not "process is gone" — as the canonical completion signal. Distinguishing the three cases lets the UI label the outcome correctly and lets Ephemeral destroy proceed with the right framing.

**User-initiated stop.** The Stop button sends SIGTERM to the training process; the worker has a brief window to write its final `events.jsonl` line and exit cleanly. The host is then destroyed (Ephemeral) or freed (Persistent). The most recent checkpoint — already pulled to the GUI server — is the saved model; no special "save before stop" handshake is needed. This intentionally treats user-initiated stop as "the checkpoint mechanism already did the work" rather than introducing a separate save-and-stop path.

**Auto-destroy on completion:** the GUI calls `provider.destroy()` with the most recently cached provider token. If the token has expired and no one is connected, the run is marked "awaiting destroy" and destroyed on the next browser session.

**Backstop for forgotten Ephemeral pods:** spawn configures a vendor-side scheduled delete (default 24h, extended on healthy progress) so a VM eventually dies even if the GUI server permanently disappears. Not a user-facing config — an implementation safety net. Vendor billing alerts are the final safety net.

### Concurrency

Multiple ways to accidentally start the same training twice — double-click on Start, page refresh during start, tab restored on browser reopen, two browser tabs open. Defenses, layered:

- **Idempotency key on the Start request.** Browser generates a UUID before clicking Start; the GUI server treats it as a unique key. A duplicate request with the same key returns the existing run id instead of creating a new one. Survives network retries and double-clicks.
- **Per-host single-active-run lock.** Each host profile carries an atomic active-run pointer. A Start request that targets a host while it's busy is refused with "Host has an active run." v1's one-GPU-per-host model makes this exactly one slot.
- **Server-side state machine.** Run state transitions (`queued → spawning → running → completing → done`) are gated; you cannot transition `running → queued`, so a stale duplicate cannot accidentally re-run.
- **Tmux / subprocess session naming on the pod.** The session name embeds the run id. If the spawn path somehow raced, the second `tmux new-session` would fail rather than launch a parallel process.
- **On-pod lock file** as a final assertion. Spawn writes a `lock.json` on the host with the run id and spawning GUI-server identifier; a second spawn finds it and refuses with an assert. Catches the rare case where the server-side lock was lost (GUI server restarted mid-spawn, etc.).

Can you legitimately run multiple trainings on the same Persistent host? Yes — but you'd register the host twice as separate logical hosts (e.g., "lab-server / gpu0" and "lab-server / gpu1"), each pinned to one GPU. v1 doesn't surface this; v2 multi-GPU adds the slot abstraction explicitly. Until then, one run per host.

### Robustness (v2 enhancement, noted now to avoid corner-painting)

The GUI server is the checkpoint store; if its disk fills, pulls would fail. v1 relies on the user provisioning enough disk for their typical run × retention count. v2 should add:

- **Pre-flight check.** Refuse a new run if free disk on the GUI server can't hold at least one expected checkpoint (estimated from model parameter count × dtype size + optimizer-state multiplier — open to a simpler heuristic).
- **During-run back-pressure.** When free disk drops below a threshold: surface in the UI, tighten retention to keep only the latest checkpoint, but **do not** destroy the Ephemeral host (data preservation > tidy cleanup).
- **Worst-case escalation.** If pulls would fail, pause new pulls and alert the user. The training continues on the host; the user has a window to free disk on the GUI server before checkpoints are lost.

Architectural property to preserve in v1: the manifest + pull mechanism makes adding back-pressure later a local change, not a refactor.

### Authentication

When the GUI backend is shared (LAN deployment), credentials must be per-user, must not leak between users, and must not persist on the backend.

**The model:**

- Long-lived credentials live on the user's laptop in the standard CLI tool locations (vendor profile directory, HF cache).
- A small helper daemon on the user's laptop mints short-lived tokens on demand. The helper is vendor-agnostic; what varies is what each vendor's local CLI does to produce the token:
  - **OAuth-capable vendors** (HF Hub, GCP, Azure, GitHub) — the user runs the vendor's `… login` once; the CLI handles the OAuth flow and stores refresh credentials locally. The helper invokes `… print-access-token` (or reads the cached token for HF) to surface a fresh access token.
  - **Service-account-key vendors** (Nebius, RunPod, Lambda) — the user runs the vendor's `… profile create` to install a long-lived service-account key locally; the CLI exchanges that key for a short-lived IAM token via the vendor's OAuth 2.0 token-exchange endpoint (RFC 8693). The helper invokes the exchange.
- The browser fetches fresh tokens from the helper on demand (loopback only, origin-checked, plus a pairing token). It includes them in vendor-specific headers on every request to the GUI server.
- The GUI server reads token headers per request, uses the token in scope (constructs a provider for spawn/destroy, or invokes the HF client for a Push-to-HF action), discards.

The OAuth vs service-account distinction is opaque to the GUI server; the helper exposes one interface (`GET /tokens` → short-lived bearer tokens) regardless of how the underlying vendor CLI obtained them.

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

### Dependencies

Three places dependencies land, with different rules for each.

**1. User's laptop and GUI server: vendor CLIs (installed out of band).**

| Component     | What's needed                                                 | Why                                                                     |
| ------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------- |
| User's laptop | Vendor CLI for any provider they want to use (e.g., `nebius`) | The auth helper shells out to it to mint short-lived tokens             |
| GUI server    | Same vendor CLI for each enabled provider                     | The provider's `spawn` / `destroy` / `verify_destroyed` shell out to it |
| Both          | `huggingface-cli`                                             | Ships with `huggingface-hub`, already a LeRobot Python dependency       |

We do NOT auto-install vendor CLIs. We auto-**detect** them. At startup the GUI server checks which CLIs are present; providers without their CLI installed are disabled in the UI with a clear error pointing at the vendor's install instructions (an "Open install guide" link). Other providers stay fully enabled; the user is never blocked from the modes that work.

Why no auto-install: vendor CLIs are signed binaries hosted by the vendor, often need sudo, and users have preferred install methods (apt, Homebrew, manual). Programmatically running `curl … | bash` from the GUI is a security smell we don't want LeRobot to be the one doing. The "system-installed CLI + we detect" pattern is what every cloud SDK does (kubectl, terraform, helm, the cloud vendors' own SDKs).

**2. Training image: vendor-neutral.**

The training Docker image contains:

- CUDA base + Python + LeRobot training stack
- `huggingface_hub` (Python; used for dataset download)
- `tmux` (detached session for survival)
- `sshd` (GUI server SSHes in)

What it does NOT contain: any vendor cloud CLI or SDK (no Nebius, no RunPod, no AWS, no GCP). The same image runs on a workstation, a lab box, or any cloud vendor's VM. This is a consequence of the pull-based checkpoint design — the pod doesn't talk to cloud APIs directly.

The pod does receive the user's HF token as an env var at spawn time (for dataset download from HF Hub; see Checkpoints). That's the only external credential the pod sees; no vendor cloud credentials ever reach it.

**3. Future enhancement: drop the GUI-server CLI requirement.**

Provider implementations could call vendor REST/gRPC APIs directly instead of shelling out to the CLI. The CLI on the user's laptop is still needed for auth; the GUI server could become CLI-free if each provider talks directly to the vendor's API. Reduces install friction for the GUI-server admin; some code complexity in exchange. Worth it once 2+ vendors are integrated.

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
- **Auto-push to external destination per run / per recipe.** v1: user clicks Push in the Models tab. v2: a "publish on completion" toggle in the run config so successful runs are pushed without extra clicks.
- **Native experiment tracking visualization.** v1 relies on the training recipe's own W&B / TensorBoard integration; output streams as part of `stderr.log` and is visible in the UI's log panel (the W&B run URL is one click away). v2 could surface loss curves, hyperparameter diffs, and cross-run comparison natively, or deep-link to the user's W&B project.
- **GUI-server disk pre-flight + adaptive retention under pressure** (see Robustness).
- **Additional external push destinations from the Models tab** — S3, GCS, vendor-local object stores, local NAS over rsync. Same pattern: pushed from the GUI server using user-supplied credentials.
- **Direct pod-to-external streaming option** for very large models where pod→GUI pull is bandwidth-limited. Adds a second code path; only worth doing if the pull-through-GUI model proves a bottleneck.
- **Multi-GPU host slots.** Register a Persistent host as N logical hosts pinned to GPU 0..N-1; one run per slot.

---

## Setup reference

Commands needed per deployment shape. The GUI server never runs more than `lerobot-gui`.

| Deployment                                  | On the GUI server | On each user's laptop                                                                                                                                  |
| ------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Workstation** (GUI + GPU on same machine) | `lerobot-gui`     | same machine — nothing extra (auth helper is a child of `lerobot-gui`); user already ran `huggingface-cli login` to get HF token in the standard cache |
| **Shared LAN — Persistent hosts only**      | `lerobot-gui`     | `huggingface-cli login`; `lerobot auth-helper start --origin <gui-url>`; paste pairing token into Settings once                                        |
| **Shared LAN — with Ephemeral cloud**       | `lerobot-gui`     | as above, plus `<vendor> profile create` once per cloud account                                                                                        |
