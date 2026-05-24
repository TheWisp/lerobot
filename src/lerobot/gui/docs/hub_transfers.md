# Hub Transfers Design

Background upload and download of LeRobot datasets to and from the HuggingFace Hub, surfaced through the GUI's Transfers tray. Optimized for **wall-clock throughput** on real workloads, which range from a few-MB sandbox dataset to a multi-hundred-GB long-horizon multi-camera recording session.

The shape: each transfer runs in its own subprocess; the GUI server holds the source-of-truth job registry that every connected frontend reads; the actual byte transfer uses HuggingFace's optimized helpers (`upload_folder`, `upload_large_folder`, `snapshot_download`) so we inherit Xet content-addressed dedupe, per-file resumability, etag skip, and parallel HTTP — none of which are worth re-implementing.

---

## Goals

### Functional

- **F1 — Background, multi-tab consistent, close-tab-safe.** A transfer keeps running even after the user closes the browser tab or opens the GUI on a different device. The GUI server is the source of truth for job state; every frontend reads the same `/api/datasets/hub/jobs` and sees identical state. Killing the GUI server **does** terminate in-flight subprocesses — that's the orphan-subprocess problem tracked separately, out of scope here.
- **F2 — Throughput is the primary cost metric.** Wall-clock time dominates user pain on long uploads. Every design decision below trades against this metric; UX polish is secondary.
- **F3 — Skip already-uploaded files and chunks.** Three dedupe layers, all server-driven, all native to `upload_folder` / `upload_large_folder`. See [Upload skip behavior](#upload-skip-behavior) below for what's skipped and at what granularity per file type. Re-uploading an identical dataset transfers ~0 bytes. Re-uploading after editing one chunk of one file transfers ~one chunk.
- **F4 — Skip already-downloaded files.** Per-file etag matching in `snapshot_download` / `hf_hub_download`. Re-downloading a synced dataset is near-instant.
- **F5 — Elegant error handling.** Lean on HF's built-in per-file retries first; surface terminal failures clearly; expose a Retry button that exploits HF's resume primitives so retrying a failed transfer is cheap.
- **F6 — Auto-mode selection by dataset size, warned in advance.** Datasets above a configured byte threshold get the resumable upload path; users see which mode will run before they click Upload, with the trade-off spelled out.

### Non-functional

- **N1 — UX is acceptable but secondary.** Coarse milestones from HF's text output are fine; we don't need byte-precise bars. The tray must show "running" / "complete" / "failed" / "cancelled" honestly; perfection isn't required.
- **N2 — Cancelable.** Background transfers must be killable from the GUI. Subprocess + `SIGTERM` is the mechanism.
- **N3 — Crash-isolated.** A failed HF library call, an OOM during a multi-GB download, a stuck commit RPC — none of these can take the GUI server down with them.

## Non-goals

- **Byte-level progress bars.** HF 1.x exposes no programmatic callback on `upload_folder` / `upload_large_folder` — only stderr / stdout text. We do not build a parallel byte-tracking layer (custom `ProgressFile`-wrapped `create_commit`, monkey-patched `XetProgressReporter`, etc.) because doing so costs throughput, complexity, and a coupling to private HF internals. The 60-second milestone text from `upload_large_folder` is the level of detail we operate at.
- **Atomic-on-large uploads.** Datasets large enough to need resumability use `upload_large_folder`, which writes multiple commits as it goes. The repo HEAD walks through partial states during the upload. This is documented and accepted (see F6, and the warning shown at modal open) — the alternative is "atomic but a network hiccup at hour 4 of 5 wastes everything," which is worse on the F2 axis.
- **Push-to-branch-then-merge for atomic semantics on large datasets.** Possible (push to a feature branch, merge to main at the end), but introduces branch lifecycle, partial-merge cleanup, and PR mechanics. Open question for later if a concrete consumer of the repo's intermediate state shows up.
- **Cross-process orphan-subprocess management.** Killing the GUI server should cleanly terminate in-flight transfers via `PR_SET_PDEATHSIG` in the worker. This belongs to the broader orphan-subprocess infrastructure already on [gui/TODO.md](../TODO.md) (Run-tab section) and applies equally to `lerobot-teleoperate` and `lerobot-record`. We use the registry that work lands, not duplicate it here.

---

## Architecture

Three pieces:

1. **`HubJobState`** on the GUI server — one entry per in-flight or recently-finished transfer. The single source of truth that every frontend polls via `GET /api/datasets/hub/jobs`. Lives in `AppState.hub_jobs`, keyed by job_id, garbage-collected 30 minutes after the transfer reaches a terminal state.
2. **Worker subprocess** per transfer — spawned with `start_new_session=True`, communicates with the GUI server via a JSON progress file on disk. Owns the actual `huggingface_hub` call. SIGTERM by the GUI server is the cancel mechanism.
3. **JSONL analytics** — one line appended to `~/.config/lerobot/gui/hub_transfers.jsonl` per terminal outcome (complete / failed / cancelled). The mining target for tuning the F6 threshold over time and for spotting failure patterns in real workloads.

```
┌──────────────┐  fetch /hub/jobs                ┌──────────────────┐
│  Frontend A  │ ───────────────────────────────▶│                  │
└──────────────┘                                 │                  │  spawn(SIGTERM)
                                                 │  GUI server      │ ────────────▶ ┌────────┐
┌──────────────┐  fetch /hub/jobs                │  (FastAPI)       │               │ worker │
│  Frontend B  │ ───────────────────────────────▶│                  │ ◀──────────── │ subproc│
└──────────────┘                                 │  AppState        │  reads JSON   └────────┘
                                                 │  .hub_jobs[*]    │  progress         │
                                                 └──────────────────┘                   │
                                                                                  ┌─────▼─────────────────────┐
                                                                                  │ huggingface_hub helpers   │
                                                                                  │   upload_folder           │
                                                                                  │   upload_large_folder     │
                                                                                  │   snapshot_download       │
                                                                                  └───────────────────────────┘
```

### Why subprocess

Two properties only the process boundary buys cleanly:

- **Cancel without thread-pool wrestling.** `snapshot_download(max_workers=8)` spins an internal thread pool we cannot interrupt cleanly from inside the same process. `os.killpg(pid, SIGTERM)` kills the whole tree in one call.
- **Crash isolation.** HF's library can raise unexpected exceptions, hit OOM during very large transfers, or get stuck in a commit RPC. Inside the GUI server's event loop those failures cascade; behind a process boundary they're just a job that flipped to `failed` with a captured error string.

The cost we pay is ~1s subprocess startup per transfer (Python init + HF imports). At any realistic dataset size this is lost in the noise; on a hand-typed `lerobot-record` push it's the dominant cost, which we accept.

### Why HuggingFace's helpers, not a hand-rolled per-file loop

A prior iteration of this work (closed) used a `for file in files: upload_file(...)` / `hf_hub_download(...)` loop to get clean cancel-between-files and a custom progress bar. Measured cost (random bytes, no Xet dedupe, fresh HF cache):

| Direction | Dataset                  | per-file loop | HF helper                     | Slowdown |
| --------- | ------------------------ | ------------- | ----------------------------- | -------- |
| Download  | pusht (8 files / 7.5 MB) | 5.18 s        | 1.25 s (`snapshot_download`)  | **4.1×** |
| Download  | svla (28 files / 83 MB)  | 12.48 s       | 10.69 s (`snapshot_download`) | **1.2×** |
| Upload    | 8 files × 1 MB (8 MB)    | 18.25 s       | 6.01 s (`upload_folder`)      | **3.0×** |
| Upload    | 28 files × 3 MB (84 MB)  | 90.78 s       | 42.58 s (`upload_folder`)     | **2.1×** |

The slowdown shrinks as bytes-transferred dominates over per-file protocol overhead — but it never disappears, and on smaller datasets it is felt. Against F2 this is disqualifying. The per-file loop also forfeits Xet's cross-file chunk dedupe and lacks the internal retry behavior the helpers ship with — so it loses on F3 and F5 too.

The helpers are the right primitives. The remaining design question is which helper, when.

---

## Upload modes

Two modes corresponding to two operating regimes:

| Mode          | HF call                     | Atomicity                                                                                    | Resumability                                                                                                                                             | Best for                                               |
| ------------- | --------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **Atomic**    | `HfApi.upload_folder`       | Single commit. Repo HEAD never partial.                                                      | None — a mid-transfer failure means re-uploading from scratch (Xet makes the retry cheap, but the workers and time are spent).                           | Small to medium datasets where re-upload is acceptable |
| **Resumable** | `HfApi.upload_large_folder` | N commits, paced by ~5 min or ~150 files per commit. Repo HEAD walks through partial states. | Full: per-file state in `<folder>/.cache/.huggingface/` survives interruption. Hours-long upload + a network drop = resume from the last completed file. | Large datasets where atomic restart is unacceptable    |

### Mode resolution

A function of `total_bytes` computed over the files actually being uploaded (after `ignore_patterns`). Threshold: **5 GB**.

```
total_bytes < 5 GB  →  Atomic
total_bytes ≥ 5 GB  →  Resumable
```

Resolved when the user opens the Upload modal, not when they click Upload. The modal renders the selected mode with an inline explanation of the trade-off so the user sees what will happen before committing:

```
Mode: [Atomic — single commit, recommended for this 1.2 GB dataset ▾]
```

For datasets at or above the threshold:

```
Mode: [Resumable — multi-commit, recommended for this 47.3 GB dataset ▾]
ⓘ This dataset is large enough to benefit from resumable uploads.
  Trade-off: the upload lands as several commits on the Hub rather
  than one. A network drop mid-upload resumes from the last commit
  instead of restarting.
```

The mode is a dropdown — a power user uploading 300 MB who wants to test resumable behavior, or uploading 8 GB but wanting strict atomicity, can override. The default reflects the F2 trade-off automatically.

### Why 5 GB

A guess, tuned by analytics. Anchored by:

- At ~10 MB/s sustained, 5 GB is ~8 minutes — short enough that a full restart on failure is annoying-but-OK; longer transfers cross the threshold where resumability dominates atomicity in expected wall-clock cost.
- Below 5 GB the `upload_large_folder` startup cost (worker spawn, file hashing, pre-upload setup) is not a free win — for a fast atomic upload there's nothing to save it from.
- The JSONL analytics record the actual time-to-complete vs mode + size, so we can revisit this threshold against real data.

If a user has a 4.9 GB dataset on a flaky LTE connection, they can flip to Resumable manually. If a user has a 6 GB dataset on a stable LAN and prefers atomic semantics, they can flip to Atomic manually. The default is what's right for the typical case at each size.

### Upload skip behavior

F3 — "skip already-uploaded files and chunks" — is satisfied for free by both upload modes because HuggingFace classifies each file into one of three storage backends server-side and applies the appropriate dedupe layer. We do not implement skip logic ourselves; we rely on the server's pre-upload negotiation and let it tell us what bytes it already has.

| Storage backend             | Picks up                                                                                                                        | Granularity of skip                                                               | Mechanism                                                                                                                                                                                    |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Xet** (content-addressed) | Most large binary files in modern datasets repos: parquet, mp4, large arrays. Chosen by the server based on repo configuration. | **Chunk-level.** Content-addressed; the server stores chunks keyed by their hash. | Client splits files into chunks, sends `(chunk_hash, size)` tuples to the server; server replies which chunks it already has. Client uploads only the missing ones.                          |
| **LFS**                     | Files configured for Git-LFS (legacy on most newer dataset repos, but still possible).                                          | **Whole-file.** SHA-256 of the full file.                                         | Client sends the file's SHA-256 in a pre-upload request; server returns "have it, skip" or "need it, upload here." No mid-file dedupe — a 1-byte change forces a full re-upload of the file. |
| **Regular**                 | Small text files: `README.md`, `meta/info.json`, `.gitattributes`.                                                              | **Whole-file**, but transparent because the files are tiny.                       | Standard Git blob storage.                                                                                                                                                                   |

Both `upload_folder` and `upload_large_folder` invoke this negotiation as a pre-flight step (`_fetch_upload_modes` in `huggingface_hub._commit_api`, called for every batch of files before any bytes are transferred). The server's response includes a `should_ignore` flag per file when the content is fully known, and a list of needed Xet chunks for files where only some chunks are new.

Concrete behavior on three real scenarios:

1. **Re-uploading identical content.** Every Xet chunk hash matches; every LFS file's SHA-256 matches; every regular blob's hash matches. Server says "I have all of this." Total bytes transferred: **0** (plus negotiation chatter on the order of a few KB). This is the case the v1 benchmark accidentally measured before we fixed it — the `0.00 B/s New Data Upload` in the upload_folder logs was the dedupe in action.
2. **Re-uploading after editing one parquet's contents.** Xet rehashes the parquet locally, finds most chunks identical to the existing ones, asks the server about the rest. Server returns the list of changed-chunk hashes. Client uploads only those. Total bytes transferred ≈ size of the edited region, not size of the whole parquet.
3. **Re-uploading after replacing a video file.** Xet chunks differ entirely; server has none of them. Client uploads the full new file. This is the worst-case for chunk dedupe and is exactly the right behavior — the file genuinely changed.

What this means for our two modes:

- **Atomic mode (`upload_folder`).** Single pre-upload negotiation covering all files in one batch. Optimal for the case where the user pushes a dataset they've never pushed before, or pushes after editing a small slice — both happen in one network round-trip of negotiation followed by transfer of the actual delta.
- **Resumable mode (`upload_large_folder`).** Pre-upload negotiation runs across multiple workers, each handling a slice of files. Per-file hash + pre-upload state is cached in `<folder>/.cache/.huggingface/` so that an interrupted run skips not only the bytes the server already has but also the local hashing work. A resumed upload reads the cache, sees the work it had already negotiated, and picks up at the next unfinished file.

The takeaway: we get file-level skip for free on all file types, chunk-level skip for free on Xet-classified files (which covers the bulk of LeRobot dataset bytes — parquet + mp4), and the work of building or maintaining that machinery is on HuggingFace's side, not ours.

---

## Download mode

One mode: `snapshot_download(repo_id, repo_type, local_dir=dataset.root, max_workers=8)` directly into the dataset root. No staging directory, no atomic swap.

The temptation to write into `/tmp/<job_id>/` and atomically rename into `dataset.root` is real — "atomic local state on download" sounds responsible. It is the wrong choice here for three independent reasons:

1. **Loses the etag-skip optimization (F4).** HF's etag skip compares against files in `local_dir`. A fresh temp directory has no etag matches, so a "refresh" download has to re-fetch every file end-to-end, even if most are unchanged. Re-downloading an 83 GB dataset because four parquet files changed is the wrong trade.
2. **Clobbers local-only files.** A user who added an episode locally that's not on remote loses it when the temp directory is renamed over `dataset.root`. `snapshot_download` directly into `dataset.root` preserves unknown files.
3. **Adds no real atomicity.** HF already gives us per-file resumability via `.incomplete` cache files + `Range:` requests. A mid-file network drop resumes from byte offset; a missing file gets re-fetched. From the user's perspective, "click Download, network failed at 60%, click Download again, it finishes in seconds" — the partial state is self-healing, not a corruption.

The interaction with our open-dataset precheck closes the last gap. A download interrupted mid-run leaves `dataset.root` partially populated; if the user closes and reopens the dataset before retrying, the precheck returns HTTP 409 with `code=incomplete_local_cache` and the GUI prompts to resume. That mechanism already exists.

`max_workers=8` is HF's default; we don't tune it. The download throughput is bandwidth-bound for any realistic dataset; more workers don't help once the pipe is full.

---

## Worker IPC

The subprocess is a thin Python script:

```
python -m lerobot.gui.hub_worker
    --direction upload|download
    --repo-id thewisp/cylinder_ring_assembly
    --root /home/feit/.cache/.../cylinder_ring_assembly
    --mode atomic|resumable                       # uploads only
    --job-id <uuid>
    --progress-path /var/tmp/lerobot/hub/<job_id>.json
```

It runs the appropriate `huggingface_hub` call. Two pieces of plumbing:

### Progress JSON file

The worker writes `progress.json` every 500 ms while running:

```json
{
  "status": "running",
  "stage": "uploading",
  "started_at": 1779700000.123,
  "milestone": "Pre-uploading LFS files (12 / 47)",
  "milestone_at": 1779700087.456,
  "files_total": 47,
  "files_done_estimate": 12,
  "bytes_total": 7686801234,
  "bytes_done_estimate": null
}
```

The `_estimate` suffix is honest: the GUI displays whatever the worker can extract from HF's text output. For `upload_folder`, that's "uploading"-ish until completion; for `upload_large_folder`, the `print_report` text gives file counts every 60 s; for `snapshot_download`, parallel tqdm bars give per-file progress that the worker aggregates into a single "X of N files" number. We do not claim byte precision (see N1).

On the GUI server side, `GET /api/datasets/hub/jobs` reads each running job's progress file once per request and merges the contents into the `HubJobState` it returns. The file is the IPC; `HubJobState` is the projection clients see.

### Stdout / stderr capture

The worker redirects HF's stderr to a per-job log file under the same directory. Two purposes:

- **Debug.** The full HF output is preserved for post-mortem on a failed transfer.
- **Milestone extraction.** Lines matching certain patterns (e.g. `Fetching N files`, `Pre-uploading file ...`, `Committing ...`) become `milestone` updates in the progress JSON. Fragile but acceptable: the format is stable across patch versions of `huggingface_hub`, the parsing is best-effort, and the fallback (`milestone = "running"`) is fine.

### Cancel

`Transfers.cancel(job_id)` on the GUI calls `POST /hub/progress/{job_id}/cancel`. The server `os.killpg(worker.pid, SIGTERM)`s the subprocess group. The worker has a SIGTERM handler that:

1. Writes `status: cancelled` + the partial counters into `progress.json`.
2. For `upload_large_folder`, the local `.cache/.huggingface/` state is left intact — a subsequent retry resumes.
3. Exits.

SIGKILL escalation after a 5-second grace period covers the case where the HF call is stuck and ignores SIGTERM.

---

## Error model

HF's helpers retry internally on transient 5xx and connection errors with exponential backoff. The worker treats their final exception as terminal: it writes `status: failed, error: <stringified exception>` and exits.

What the GUI does with terminal failures:

| Surface             | Behavior                                                                                   |
| ------------------- | ------------------------------------------------------------------------------------------ |
| Transfers tray card | Red-bordered, shows the error string verbatim. Per-card **Retry** and dismiss (×) buttons. |
| Toast               | One-shot "<Direction> failed: <error>" with 8-second visibility.                           |
| Analytics line      | JSONL entry with `outcome: "failed"`, `error`, `duration_s`, all counters as last known.   |

**Retry** is a fresh worker spawn with identical args. For `upload_folder`, Xet dedupe means retransmitting already-uploaded chunks is free. For `upload_large_folder`, the local `.cache/.huggingface/` survives the failed run and the retry picks up at the last completed file. For `snapshot_download`, the `.incomplete` cache + etag skip makes the retry resume from the last partial file. We don't build retry logic on top — HF's resume primitives are the retry mechanism; we just re-invoke them.

### What "elegant" means specifically (E1–E4)

- **E1.** The error string in the tray is the HF exception message, not "Failed". Users debug from the GUI.
- **E2.** A Retry button on every failed card spawns a fresh worker with identical args; resume is automatic via HF's primitives.
- **E3.** Ambiguous-result uploads (network drop after commit RPC sent but before client got the 200) are resolved by Retry: Xet sees the chunks already exist, and the commit is either a no-op or applies the same change. No user investigation required.
- **E4.** Every terminal outcome writes a JSONL line with enough detail to retry from CLI if the GUI is unavailable: `repo_id`, `root`, `mode`, last `milestone`. A debug command can read the JSONL and replay.

---

## UX surface

The visible surface is the Transfers tray in the tab bar — already shipped in the close-PR-15 iteration of this work. The pieces:

- **Indicator pill** in the tab bar, hidden when no jobs exist, cyan + pulsing when at least one job is active, neutral when only terminal entries remain.
- **Popover** anchored under the pill, listing one card per active or recently-terminal job. Each card shows direction, clickable repo link, status, milestone text from the worker, and a per-card cancel-or-dismiss button.
- **Modal additions** to the existing Hub Upload modal: the mode dropdown described in [Upload modes](#upload-modes), with the inline trade-off explanation rendered live based on the auto-resolved mode.

No new top-level UI; this is an evolution of the existing tray, not a new surface.

---

## Analytics

One JSONL file at `~/.config/lerobot/gui/hub_transfers.jsonl`, one line appended per terminal outcome. Persistent across sessions; never pruned automatically.

Schema:

```json
{
  "ts": 1779700123.456,
  "direction": "upload",
  "repo_id": "thewisp/cylinder_ring_assembly",
  "dataset_id": "/home/feit/.cache/.../cylinder_ring_assembly",

  "mode": "resumable",
  "user_mode_choice": "auto",

  "started_at": 1779696523.111,
  "finished_at": 1779700123.456,
  "duration_s": 3600.345,

  "files_total": 412,
  "files_done": 412,
  "bytes_total": 47800000000,
  "bytes_transferred_estimate": 31200000000,

  "commits": 27,
  "resumed_from_cache": false,

  "outcome": "complete",
  "error": null
}
```

What we mine it for:

- **Threshold validation (F6).** If `outcome=failed` correlates strongly with `mode=atomic AND bytes_total > 2 GB`, the 5 GB threshold is too high. If `mode=resumable AND bytes_total < 1 GB AND duration_s < 60 s`, the threshold is too low (we're paying `upload_large_folder` startup for no benefit). Adjust and re-deploy.
- **Failure clustering.** Group `error` strings by direction + mode. Cluster what's actually breaking.
- **Throughput-over-time.** Plot `bytes_total / duration_s` over weeks. A sudden drop in throughput is a network or HF-side regression we can localize.
- **Xet dedupe ratio.** `bytes_transferred_estimate / bytes_total` over time. If this drops, something changed in how we ignore-pattern or how Xet sees our data.
- **Cancel rate.** Fraction of jobs with `outcome=cancelled`, segmented by `direction` and `duration_at_cancel`. Tells us whether users are aborting because something looks stuck.

The format is plain JSONL, greppable from a shell, parseable with one-line Python. No tooling is required for the file to be useful; a Stats view in the GUI can come later if a question warrants it.

---

## Open questions

- **Threshold value.** 5 GB is a starting guess. The JSONL is how we validate it. Worth revisiting after a few real workloads cross both sides of the line.
- **Push-to-branch-then-merge for atomic-on-large.** If a downstream consumer ever reads the repo HEAD while a `upload_large_folder` is in flight and sees a partial state, we'd want the atomic property without losing resumability. Mechanism would be `upload_large_folder(revision="some-branch")` then a final `merge` once done. Not designed here; flagged as future work if the need is concrete.
- **Per-worker concurrency limit.** Two parallel transfers on different datasets currently run two subprocesses competing for the same uplink/downlink. At realistic bandwidths this is fine; at gigabit-plus with many small files in parallel transfers, we may want a global N=2 cap to avoid head-of-line stalls. Defer until a measurement says it matters.
- **Mode parity for downloads.** Today downloads use one mode (`snapshot_download` into `dataset.root`). If a future workload needs explicit checkpointing of in-progress downloads beyond HF's per-file `.incomplete`, we'd add a resumable-download mode mirroring uploads. Not currently needed.
- **Retry budget.** A retry that fails should probably surface differently from a first-attempt failure (e.g. "Failed 3× in a row — check connection" instead of just the latest error). Not designed; small future addition.
