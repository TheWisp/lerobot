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

### Unified execution: training always runs in the image

The training process always runs inside the same Docker image, on every host
mode — workstation, user-added Persistent, Ephemeral cloud. This is the
forcing function that keeps the pipeline honest: what runs locally on the
developer's box is bit-for-bit what runs on a remote pod, with the same
Python, the same CUDA, the same `lerobot` install. Without this, "works on
my workstation" silently diverges from "works on Nebius" and surprises
land at the worst moment.

Operationally that means even subprocess transport is `docker run … training-image`,
not a bare `python -m`. Optimizations that skip the bytes-on-the-wire steps
are fine (mount the HF dataset cache instead of staging files; mount the
checkpoint dir instead of SCPing back) — but the runtime environment is
identical to a remote pod.

The single image source of truth is `ghcr.io/thewisp/lerobot-training:<tag>`;
see Dependencies for what it does and does not contain.

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

- **Browser** — the user's interface. Holds short-lived tokens in localStorage (user-pasted in v1; auto-fetched from a local helper in v2) and forwards them to the GUI server in vendor-specific request headers.
- **GUI server** — orchestrates training and owns model storage. Discovers a local GPU at startup; holds host profiles + run state + the canonical checkpoint store; talks to providers. Never holds long-lived credentials.
- **Training host** — where the training container runs. Either the GUI server's own box (subprocess transport: `docker run` locally) or any machine reachable by SSH (SSH transport: `docker run` over SSH inside `tmux`). Writes checkpoints to local disk; does NOT talk to external destinations.
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
- **Auth flow.** See Authentication below; credentials injected per-request as a provider constructor arg from the request's headers.
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

- **The training pod has zero external credentials.** No HF token, no vendor cloud token, no S3 keys. Dataset is staged to the pod by the GUI server before training starts (see below); the pod runs with `HF_HUB_OFFLINE=1` against local files. Pod compromise leaks nothing externally exploitable.
- **Pod egress cost is the same.** Bytes leave the pod's network either way; they go to the GUI server instead of an external store. The pod-side cost is identical to a direct-to-HF push.
- **The Models tab is the canonical surface.** A pulled checkpoint appears there immediately — before any external upload.
- **External publication is a separate Models-tab action** (v1: user-initiated; auto-push is a future enhancement).

**Trade-off.** The pull-based design is simpler than the alternative (pod pushes to a central artifact store): no per-store credentials on the pod, one credential boundary, no artifact-store infrastructure. The cost: the GUI server's network and disk are the bottleneck. Acceptable at our scale. At 10+ concurrent runs with very large models, a push-to-shared-store model would be the right rethink — but that day is far off.

For Ephemeral pods in the cloud with the GUI server on a LAN box, pod→GUI may also be slower than a hypothetical direct pod→external push (LAN bandwidth, intercontinental latency). Pod-side cost is the same; the difference is wall-clock to durability.

**Dataset staging.** Always done in two steps: GUI server ensures the dataset is in its own local HF cache (downloading from HF Hub via the GUI server's existing HF auth if missed), then makes that cache available inside the container with `HF_HUB_OFFLINE=1`. For SSH transport (remote host), the GUI server SCPs the dataset to the pod under `/data/<dataset_id>/` first, then mounts that into the container; for subprocess transport (workstation), the cache directory is mounted directly into the container — no file copy needed. The training script's view of the dataset is identical in both cases (local files + offline mode); only the path to the bytes changes.

**Resume from checkpoint.** The Models tab lets the user pick a checkpoint and start a new run resuming from it:

1. User picks a checkpoint and a host (any of the three modes).
2. GUI server spawns the host (Ephemeral) or selects an idle one (Persistent / workstation).
3. Before launching training, the GUI server pushes the chosen checkpoint to the host's local disk (SCP for SSH, file copy for subprocess).
4. The training worker reads a standard `--resume_from_checkpoint <path>` argument and continues from that step.

The new run gets its own run id; the resumed-from checkpoint is recorded in the run metadata so the lineage is queryable.

### Transport, log surfaces

Training launches as `docker run … training-image` in a detached session on the host (tmux for SSH transport; managed subprocess for subprocess transport). The container invokes the worker process inside. The GUI server never holds an open pipe to training. Polling is via short-lived reads of three log surfaces, identical across both transports:

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

v1 scope is narrow: the only new credential the GUI server takes per-user is the **vendor IAM token for Ephemeral mode**. HF auth is left to the existing GUI behavior unchanged; the training pod sees no credentials at all.

**v1 paste-token model (vendor IAM only):**

- Long-lived vendor credentials live on the user's laptop (vendor CLI profile).
- The user pastes a short-lived vendor IAM token into the GUI's Settings page: obtained via the vendor's CLI (e.g., `nebius iam get-access-token --duration=12h`). Short-lived, paste again when it expires.
- The browser stores it in localStorage and includes it in a vendor-specific header (`X-Nebius-Authorization`) on each request to the GUI server.
- The GUI server reads the header per request, uses the token in scope (constructs a provider for spawn/destroy), discards. No persistence.

**Workstation case**: the user can optionally bypass the paste via a "use my local CLI auth" toggle that has the vendor SDK read its default profile directly — a convenience for single-user-on-own-machine.

**Shared-LAN case**: each user pastes their own vendor IAM token. Backend never sees another user's tokens.

**UX softening for the daily Nebius re-paste:**

- Settings shows the current token's expiry as a wall-clock time.
- 30-min-before-expiry banner reminds the user.
- "Refresh Nebius token" button copies the exact CLI command to the clipboard.
- On 401-from-vendor mid-request, the GUI prompts re-paste without losing page state.

**HF in v1 — inherited, unchanged.** The GUI server already reads `~/.cache/huggingface/token` for its existing Hub features (Hub Transfers, model push, etc.). The training pipeline reuses this for any HF access it needs (mainly: ensuring the chosen dataset is in the GUI server's local cache before staging it to the pod — see Dataset staging in Checkpoints). The pod itself receives **no** HF token; it runs with `HF_HUB_OFFLINE=1` against the staged dataset files.

This means the LAN-deployment shared-HF-identity limitation present in today's GUI is inherited: all users on a shared lab server see one HF identity. **Not a regression — same as today.** v2 fixes per-user HF auth (paste or helper) properly.

**Security properties:**

- Vendor cloud credentials never on backend; backend has zero credential persistence for them.
- Per-user vendor isolation in LAN: each user's browser holds its own localStorage; backend never mixes them.
- Compromised backend worst case: short-lived vendor tokens (typically ≤12h) for currently-connected users. No refresh creds, no long-lived vendor secrets.
- **Training pods receive no external credentials.** No HF token, no vendor cloud token. Pod compromise leaks nothing externally exploitable.

**v2: credential helper + per-user HF auth.** A small daemon on each user's laptop mints tokens on demand (eliminating the manual paste and the daily Nebius re-paste), AND a parallel per-user HF auth flow replaces the inherited workstation-mode assumption. The architecture doesn't change for vendor auth; HF auth becomes per-user. (See Future enhancements.)

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

**1. User's laptop: vendor CLI** (installed out of band, one-time per vendor).

The auth helper shells out to the vendor CLI to mint short-lived tokens. The user already has the vendor CLI because they set up their cloud account with it; the helper just leverages what's there.

| Vendor       | What the user installs on their laptop                          |
| ------------ | --------------------------------------------------------------- |
| Hugging Face | `huggingface-cli` (ships with `huggingface-hub` Python package) |
| Nebius       | `nebius` CLI (via Nebius's install script)                      |
| RunPod       | `runpodctl`                                                     |
| AWS          | `aws` CLI                                                       |
| GCP          | `gcloud` CLI                                                    |
| …            | per vendor docs                                                 |

If a vendor CLI is missing on the laptop, the helper's `/tokens` response for that vendor errors with a clear "install `<vendor>` CLI" message. Other vendors keep working.

**2. GUI server: vendor Python SDKs** (distributed as LeRobot optional extras).

The GUI server's provider implementations call vendor Python SDKs — NOT the vendor CLI. Enable a vendor by installing the corresponding extra:

```
uv sync --extra nebius                  # or: pip install "lerobot[nebius]"
uv sync --extra nebius --extra runpod   # multi-vendor admins
```

In `pyproject.toml`:

```toml
[project.optional-dependencies]
nebius = ["nebius>=x.y.z"]
runpod = ["runpod>=x.y.z"]
# huggingface-hub is in the base install
```

This is the **same pattern LeRobot already uses for hardware** (`lerobot[aloha]`, `lerobot[feetech]`, `lerobot[dynamixel]`). Pip-installable, no curl-bash, no system binaries we'd vouch for, standard Python packaging.

At startup the GUI server does a soft import (`try: import nebius`); installed providers are enabled, missing ones are disabled in the UI with a clear message — _"To enable Nebius training, run `uv sync --extra nebius` on the GUI server"_. Other providers stay fully usable; the user is never blocked from the modes that work.

The provider implementation uses the Python SDK with the per-request bearer token from the helper. The SDK doesn't manage its own auth state; we hand it the token.

**Fallback for vendors without a usable Python SDK:** call the vendor REST/gRPC API directly via `httpx`. Same effort either way; most SDKs are thin wrappers around REST. Documented as the implementation option for any provider where the SDK is missing, abandoned, or unwieldy.

Why CLI on the laptop and SDK on the GUI server? Different needs. The laptop already has the CLI from the user's normal account workflow; reusing it avoids a second install. The GUI server needs reliable scripted access under our control, where a pip-installable SDK fits LeRobot's existing extras pattern.

**3. Training image: vendor-neutral, runs on every host.**

The training Docker image contains:

- CUDA base + Python + LeRobot training stack
- `huggingface_hub` (Python; used for dataset download)

What it does NOT contain: any vendor cloud CLI or SDK (no Nebius, no RunPod, no AWS, no GCP). Same image runs on a workstation, a lab box, or any cloud vendor's VM — see "Unified execution" above for why this matters. It also doesn't need `tmux` or `sshd` inside; those live on the host (the SSH transport uses tmux on the host to keep the `docker run` detached, then the GUI server SSHes back in for short-lived `cat` reads of the mounted log files).

The container receives no external credentials. Dataset cache + checkpoint dir + run dir are mounted in; the worker runs with `HF_HUB_OFFLINE=1` and writes to the run dir. See Checkpoints → Dataset staging.

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

- **Per-user HF auth in the GUI server.** v1 inherits the existing GUI's workstation-mode assumption (one shared HF identity per GUI server). v2 fixes this with either a paste flow (parallel to v1 vendor paste) or via the credential helper below.
- **Credential helper on each user's laptop** to eliminate the manual token paste and the daily Nebius re-paste. A small loopback daemon that mints vendor tokens (and an HF token, replacing the v2 paste) by shelling out to the user's existing vendor CLI / reading the HF cache; browser fetches transparently with Origin + pairing-token checks. Same request headers as v1, same backend behavior — only the token source shifts from manual paste to auto-mint. The biggest UX win after v1 lands.
- **Auto-discovery of the user's machine** (depends on the helper above) via its system-info endpoint. Would add the user's laptop as a candidate Persistent host if reachable via SSH from the GUI server.
- **mDNS-based LAN host advertisement.** Each opted-in GPU server runs a tiny advert daemon; clients see them on the LAN. Passive advertisement (opt-in by the host owner) is the right pattern; active port-scanning would be an antipattern.
- **Helper auto-start unit** (depends on the helper above). One-time install creates a launch-agent / systemd-user unit so it auto-starts at login.
- **HF OAuth in browser.** Right answer if even the one-time HF token paste becomes unwanted friction.
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

Commands needed per deployment shape. The GUI server only ever runs `lerobot-gui`.

| Deployment                                  | On the GUI server                          | On each user's laptop / per browser                                                                                                                                                                   |
| ------------------------------------------- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Workstation** (GUI + GPU on same machine) | `lerobot-gui`                              | Nothing for local training. For Ephemeral cloud: toggle "use my local CLI auth" in Settings (single-user-on-own-machine convenience) or follow the Ephemeral row below.                               |
| **Shared LAN — Persistent hosts only**      | `lerobot-gui`                              | Nothing — training auth path is HF-free in v1.                                                                                                                                                        |
| **Shared LAN — with Ephemeral cloud**       | `lerobot-gui` + `uv sync --extra <vendor>` | Install the vendor CLI (`nebius` etc.); run `<vendor> profile create` once; then `nebius iam get-access-token --duration=12h` → paste into Settings → re-paste every ~12h (clipboard button assists). |

HF auth follows the existing GUI's behavior in v1 (GUI server uses its own `~/.cache/huggingface/token`; LAN deployments share that one identity — same as today's GUI). Per-user HF and the no-paste vendor flow both land in v2.
