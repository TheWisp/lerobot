# GUI TODO

## Data Tab

- [High] **Warning/error panel**: dataset verification errors and warnings are currently buried in server log text. Add a visible warning panel (banner or sidebar) that surfaces verification results when a dataset is opened — errors as red, warnings as yellow. Users must not miss data integrity issues.
- [High] **Open local dataset by path**: opening a copied/renamed local dataset fails because `LeRobotDataset.__init__` tries to reach HuggingFace Hub when the folder name doesn't match a cached `owner/name` repo_id. Spaces in folder names also rejected. Need to bypass Hub entirely for local-only datasets.
- [ ] Parquet data display (action/state charts in timeline) — superseded by Feature Editing (see below); action/state co-display alongside cameras tracked as a follow-up there
- [ ] Monitor local dataset changes — auto-refresh UI when new episodes recorded while GUI is open
- [ ] Duplicate episode
- [ ] Copy/move episodes between datasets
- [ ] Reorder episodes
- [ ] Create new dataset from UI
- [ ] Drag-drop dataset opening
- [High] **Language annotations (v3.1) — GUI parity**: [docs/stories/language_annotations.md](docs/stories/language_annotations.md)
- [ ] Undo/redo — explicitly punted from Feature Editing V1 (per-chip removal in edits-bar covers most "oops" cases). Real undo across Saves needs pre-edit value capture or a Git-like history.

### Data Editing / Augmentation (segment + effect → new dataset)

Shipped (prototype): a "Process dataset…" button in the data-tab overlay panel
opens a menu that reuses the SAM3-segmented objects as the protected foreground,
applies a background/global effect to every frame, and writes an augmented copy
as a new LeRobotDataset via an async worker job (modelled on the Hub-transfer
tray). See [docs/data_editing.md](docs/data_editing.md). Effects:
background random-colour / random-texture / solid / blur (foreground feathered) +
global brightness-contrast. Randomized effects sample per-episode only.

Staging (done): tune criteria live in the overlay → **Preview this episode**
(ephemeral single-episode run, auto-opened) → **Process all episodes**. The menu
shows measured wall-clock estimates for both (segmentation dominates at ~90
ms/frame/cam; full dataset = tens of min, one episode = seconds).

Follow-ups:

- [ ] **Background Replace from a texture/photo library** — the highest-impact
      effect per GreenAug/RoboEngine (random _texture_ backgrounds beat solid colour
      and beat generative). Needs a source-folder picker + per-episode image choice.
- [ ] **Live effect preview on the scrubbed frame** — the overlay already draws
      masks per-frame; render the chosen _effect_ (not just contours) as the overlay
      so the composited look updates as you scrub, before even the episode preview.
      Near-free (reuses the warm overlay worker's masks + ~9 ms effect apply).
- [ ] **Warm-model reuse across preview → commit** — today each run reloads SAM3
      (~6 s) and preview tears down the live overlay. A persistent worker with a
      command channel (like the overlay worker) would let preview and commit share a
      loaded model and skip the reload.
- [ ] **Hue / colour-shift effect** — modest ±deg range (keep segmented target
      objects recognizable); deferred to keep the v1 menu small.
- [ ] **Multi-instance foreground** — SAM3 locks one instance per concept, so a
      two-arm scene only protects one arm unless the user adds a second object row.
      Consider auto-expanding "robot arm" to all detected instances.

### Feature Editing (per-frame view + edit)

See [docs/feature_editing.md](docs/feature_editing.md) for the full design.

V1: schema-driven, drag-to-select-range + Inspector typed editing. Lays the foundation for RECAP-style labeling (`reward`, `success`, `subtask`).

V1 phases (A1–B6) and the schema-add layer (see [docs/add_feature.md](docs/add_feature.md)) are done as of 2026-05-06. Phase C1 (resize handles) and follow-ups remain.

- [x] Phase A1: schema in `DatasetInfo` (extend the existing dataset-open response with full features dict — dtype, shape, names)
- [x] Phase A2: per-frame feature values endpoint + Inspector dataset-summary empty state (schema-driven renderer registry)
- [x] Phase A3: episode feature-series endpoint + timeline rows (line / band / stripe per dtype)
- [x] Phase B1: vertical-slice selection + click-to-seek-and-select (click inside trim → playhead + selection `[N, N+1)`; drag inside trim → seek + range; playhead-thumb drag scrubs without re-selecting; click outside trim → no-op; Esc clears)
- [x] Phase B2: Inspector edit widgets (checkbox / slider+number / dropdown for `subtask_index` / text / row-of-N inputs / "Edit as JSON…"); auto-staging on change with `● pending` indicator
- [x] Phase B3: stage `feature_set` edits in `PendingEdit` pipeline; group-by-`(feature, episode)` rendering with expand/collapse
- [x] Phase B4: "Show pending edits" toggle (current vs post-Save data overlay on timeline)
- [x] Phase B5: validation + safety rails (block edits on `DEFAULT_FEATURES` / `action` / `observation.*` / image / video; >10k frames confirmation)
- [x] Phase B6: **new `set_feature_values()` API** in `dataset_tools.py` (peer to `modify_features`) — in-place parquet rewrite of value cells, stats recomputation, `subtasks.parquet` updates, `finalize()`; GUI's `_apply_feature_set_edits()` translates staged edits into one call
- [x] **Schema-add layer (2026-05-06)**: `dataset_tools.add_features_inplace()` + GUI banner offering to add MUST-have `reward`/`success` defaults + generic "+ Add feature" dialog for custom columns + `success` tri-state widget + declared-`per_episode`-wins-over-inference + orphan-`.tmp` sweep on dataset open. Single-tab UX only — multi-tab cross-update is a follow-up.
- [ ] Phase C1: two mouse-draggable resize handles (vertical Inspector ↔ main, horizontal cameras ↔ timeline)

Follow-ups (post-V1, listed in design doc):

- [ ] **WS broadcast for `dataset.schema_changed`** — multi-tab cross-update for the schema-add path. Today's frontend updates `window.datasets[id]` from the POST response only (single-tab); other tabs/windows looking at the same dataset don't auto-refresh. Needs a connection registry in `AppState` and a per-dataset `broadcast_to_dataset(id, msg)` helper. Skipped during the schema-add build because no broadcast infra existed.
- [ ] **`add_features_inplace` UX polish** — confirmation modal for >100k frames (per spec); pre-canned "Common features" preset list in the Add Feature dialog (success/reward/subtask/quality_score) so common cases are one click; tri-state colored band rendering on the timeline for pinned per-episode int8 features (today: line plot fallback).
- [P1/P2] **Stats viewer** — surface `meta/episodes/*.parquet` `stats/*` columns: per-feature min/max/mean/std/quantiles, per-episode and per-dataset, read-only
- [ ] In-place segment editing: boundary drag, double-click rename, click-empty-and-type, split / merge
- [ ] Row context menus (Rename, Delete, Split here, Merge with next)
- [ ] Multi-range selection
- [ ] Loop-on-selection during playback
- [ ] Curve editor for continuous features (reward, value, advantage)
- [ ] **Multi-dim numeric vector visualization** for `action` / `observation.*` — current state: up to 32 components rendered as overlaid lines with a 16-color palette (post-PR3), but for a 14-DOF action the lines cluster into a dense band and the components aren't individually readable. Decided direction: **expandable per-dim rows** — chevron toggle on the row label; collapsed (default) shows the overlaid view, expanded shows a stack of mini-rows (~16px each), one per component, with `ft.names[i]` labels (or `[i]` fallback). State persisted per-dataset-session via `featureRowState.expanded` (already declared, not yet wired).
- [ ] Schema mutations: add / remove / rename features via `modify_features`
- [ ] URDF / 3D trajectory views
  - [x] ~~**EE-trajectory line as an always-live alternative.**~~ Done — landed as a 5 mm cyan tube along the FK'd EE positions of multi-frame sources. Surfaces immediately in dataset/replay mode (the next `horizon` rows of `action` are always available); also surfaces in any live mode that publishes chunks.
  - [ ] **Chunk-source playback in the URDF viewer.** Now that the trajectory tube ships the path-in-space dimension, the remaining work is the time dimension: step through the captured chunk's frames at the producer's `fps`. Needs (a) producer-side pause/snapshot so a captured chunk doesn't fight live motion, (b) scrubber UI in the URDF tile.
  - [ ] **Chunk-publishing for live policy modes.** Today an ACT/diffusion/HVLA policy writes one action per step through `robot.send_action`; the trajectory tube only lights up when the source returns N > 1 frames. To light it up live, the policy needs to publish its chunk on a small shared-memory channel; the renderer reads it unchanged.
  - [ ] **Multi-source overlay UI.** Today the URDF tile has a single "show action" toggle. When meta.sources lists a second overlay candidate (e.g. "prediction" from a debug model), the toggle needs to become a multi-pick (or dropdown). The backend API (`?source=` parameterised, meta lists names) is already shaped for this.
  - [ ] **Layered correctness viz: solid = real, ghost = desired-now, tube = desired-future.** Today the URDF viz can't distinguish IK accuracy from control accuracy. "The arm is off by 2 cm" gets blamed on IK by default when it's usually the motor PID's steady-state lag under gravity load (e.g. shoulder_lift sag on SO-107). Three layers, each a different error source:
    1. **Solid body** — FK of `Present_Position`. The actual physical state.
    2. **Translucent ghost arm** — FK of `Goal_Position`. Where the controller wants the arm to be RIGHT NOW. The solid-vs-ghost gap is per-joint tracking error (sag, friction, load); when tracking is clean the ghost collapses onto the solid and disappears.
    3. **Cyan EE-trajectory tube** (already shipped) — FK of the next N frames of `action`. Where the controller wants the arm to be SOON.

    The chain reads as "this used to be true → should be true now → should be true soon," and the tube starts at the ghost EE so the geometry is visually continuous. Couple with a per-arm strip in the latency panel that decomposes the error sources numerically: `IK: 0.2 mm / 0.1°  ·  tracking: 12 mm / 3.5° (sag: -8 mm Z, mostly shoulder_lift)` so the user knows which layer owns the deviation.

    Decisions before implementation: (a) show ghost when the clutch is **disengaged** (helpful — "we'll resume tracking from here") or hide it (avoids confusion since Goal_Position is held)? (b) ghost color — desaturated arm color vs single neutral shade — must not compete with the cyan tube; (c) the "sag" callout — useful when the gap exceeds N° but easy to over-clutter, probably gated behind a toggle; (d) live (informs ongoing teleop, doubles per-frame URDF work + needs a `Present_Position` subscription) vs dataset-playback-only (cheap, no hot-loop cost) vs both. Recommended first pass: replay-only ghost arm, no callout, single neutral color — minimum surface to validate the mental model before paying the live-rendering cost.

- [Mid] **Compact per-episode-feature storage** (tech debt): per-episode features are currently marked via `info.json` but stored by replicating the same value across every frame in `data/*.parquet` — wasteful for what is logically one value per episode. Find a more compact representation (e.g. dedicated columns in `meta/episodes/*.parquet`, or sparse encoding in `data/*.parquet`) while staying backward-compatible with readers that expect the per-frame layout.
- [Mid] **Remove implicit "episode-wide if uniform" detection** (tech debt, depends on the compact storage above): today a feature is recognized as episode-wide if all its frame values within an episode are identical. That heuristic is fragile (one stray edit makes it look per-frame again) and only exists because we lacked an explicit format. Once compact storage is in place, drop the heuristic in favor of the explicit `episode_features` declaration.
- [ ] Episode-list shortcuts: tri-state `success` checkbox column with bulk-set; inline-editable `task` per episode
- [ ] First-class subtask writer at recording time (`add_frame(subtask=...)` analog to `add_frame(task=...)`); today only the read hook + offline annotation Space exist. Must write a v3.1 `language_persistent` row, not the fork's `subtask_index` — see [docs/stories/language_annotations.md](docs/stories/language_annotations.md).

### Per-Episode Quality Badges (Latency)

`lerobot_record` now writes `meta/episodes_health.jsonl` (one line per
episode) and `meta/recording_health.json` (session summary) into the
dataset's meta dir. Each line carries `healthy`, `issues`, `overrun_ratio`,
`loop_dt_ms` percentiles, and per-camera staleness. The data panel
doesn't yet read this — wire it up:

- [ ] **Render per-episode quality badge** in the episode list (yellow/red dot next to bad episodes; tooltip shows the issue messages from `episodes_health.jsonl`).
- [ ] **Dataset-level health banner** when `recording_health.json` reports `healthy: false` — surfaces "this run had X% overrun, Y issues" on dataset open, so users can decide to keep / discard / fix before training.
- [ ] **Filter / bulk-select bad episodes** for deletion via the existing data-panel deletion flow.
- [ ] **Backend endpoint** `GET /api/datasets/{id}/health` returning the parsed contents of both files (or empty when the files aren't present — older recordings).
- [ ] Schema docs in [latency_monitoring.md](../docs/latency_monitoring.md) — already covers the file format under "Persistent per-episode quality metadata".

Verdict thresholds live in `src/lerobot/utils/latency/recording_health.py::DEFAULT_VERDICT_THRESHOLDS` — surface them somewhere the user can tweak per-dataset if needed (probably not in V1).

### Control-Lag Analyzer

Cross-correlation of `action[:, j]` vs `observation.state[:, j]` per joint gives the effective control lag for free, from any existing dataset — no extra recording, no extra hardware. Prototype lives at `/tmp/analyze_control_lag.py`; ran clean on three SO-107 datasets (199K / 179K / 137K frames each) and converged on a uniform 3–4 frame baseline (~100–133 ms at 30 Hz). The fact that the lag is the same across all 14 joints — not a gradient from heavy shoulders to light gripper — points at the sampling rate + default PID gains as the systemic floor rather than per-joint mechanics.

Worth landing as a proper tool:

- [ ] **Promote the analyzer** out of `/tmp` to `src/lerobot/utils/analyzers/control_lag.py` with a `lerobot-analyze-control-lag <dataset_dir>` CLI entry (or as a function callable from the data panel backend).
- [ ] **Integrate into `meta/recording_health.json`**: append a `control_lag` block with `{joint_name: {lag_frames, lag_ms, corr, n_moving}}` so the verdict has a measured number rather than just "did the loop hit its budget." Computed once at end of recording from the freshly-written parquet, no live overhead.
- [ ] **Data panel visualisation**: a small per-joint bar chart on the dataset summary, color-coded against expected baseline (~4 frames at 30 Hz for hobby servos). Outliers (e.g. one joint at 7+ frames) flag specific hardware issues — backlash, retries, loose mount.
- [ ] **Gripper handling**: gripper position has nonlinear contact dynamics so its lag correlation is ~0.94 vs ~0.99 for other joints. Either filter to no-contact segments before correlating, or flag the gripper number as low-confidence in the UI.
- [ ] **Cross-dataset comparison view**: same set of joints across N datasets to spot drift (servo wear, firmware change, mount loosening). Lives naturally next to the per-episode-quality badges work above — same data-panel concept.

Caveat: this measures _tracking lag_ (action-to-state convergence during continuous motion), which is a strict superset of the pure command-to-actuation dead time discussed in the latency design doc. The rig-based C2A calibration (IMU + MCU master clock) is still the only way to isolate the few-ms servo dead time itself — but for "is the control loop tracking?" / "did this hardware regress?", cross-correlation is enough and free.

### Dataset Merge

See [docs/data_tab.md](docs/data_tab.md) for full design.

- [ ] Merge dialog: select target (new dataset), select source datasets from opened datasets
- [ ] Episode selection: checkbox per episode in each source (start with whole-dataset, add per-episode later)
- [ ] Pre-merge validation panel (FPS, robot_type, features — green/red checks)
- [ ] Post-merge integrity verification (video timestamps, parquet row counts, contiguous indices)
- [ ] Backend: `/api/edits/merge` endpoint wrapping `merge_datasets()` from `dataset_tools.py`
- [ ] Test chained merges (A+B->C, C+D->E) — existing regression test covers meta file mapping, but need end-to-end chain test

### HuggingFace Hub Sync

Background Transfers tray + subprocess-worker pipeline landed in PR #15
(merged 2026-05-26). Design: [docs/hub_transfers.md](docs/hub_transfers.md).

- [x] `GET /api/hub/auth-status` — check login state
- [x] **Keep synchronous Hub probes off uvicorn's event loop** (2026-07-28).
      `auth-status`, repo-info, diff metadata, and upload/download preflight now run
      in worker threads. Regression coverage deliberately stalls `HfApi.whoami()`
      and proves an unrelated dataset request still completes. This is containment,
      not a complete outage policy: the caller remains pending and the worker
      thread remains occupied until the underlying HTTP call returns.
- [x] `POST /api/datasets/{id}/hub/download` — pull from Hub (with completeness check + confirm_force override)
- [x] `POST /api/datasets/{id}/hub/upload` — push to Hub via PR (atomic merge to main)
- [x] Frontend: auth indicator in header, Hub Upload / Hub Download per-dataset entry points
- [x] **Background Transfers tray** — pill in top-right, popover with one card per job, real progress, Cancel / Retry / Discard / Hide actions. Multi-tab consistent (server holds state), workers survive server restart, multi-dataset parallel.
- [High] **Bound and coalesce the startup auth probe.** `checkHubAuth()` runs
  once per full browser document load. When Hub networking is blackholed, the
  GUI now remains usable but the header indicator stays blank until the kernel
  TCP timeout, and every reload can consume another default-executor thread.
  Show `HF: checking…` immediately; distinguish `unavailable` from `not logged
in`; cache/single-flight concurrent probes; and configure a short timeout in
  the Hub HTTP client itself. `asyncio.wait_for(asyncio.to_thread(...))` alone
  is insufficient because cancelling the await does not stop the already
  running worker thread.
- [High] **Offload remote dataset open/download construction.**
  `POST /api/datasets` still constructs `LeRobotDataset(repo_id)` inline, and
  confirmed incomplete-local-cache opens can also enter `snapshot_download`.
  A stalled Hub therefore still freezes the whole GUI through this separate
  route. Extract the synchronous load/verify work into a worker call, then
  publish the returned dataset into `AppState` on the event-loop thread. Avoid
  racing the process-global `datasets.disable_caching()` toggle across
  concurrent opens.
- [High] **Finish the Hub request-path audit.** Upload retry discovery still
  calls `get_discussion_details()` inline, and failed/cancelled-job dismissal
  calls both `get_discussion_details()` and `change_discussion_status()` inline.
  Offload only the network portions; keep `AppState`/PR-ownership mutations on
  the event-loop thread so Retry and Discard cannot race.
- [ ] **Stale-PR sweep** ([open question in design doc](docs/hub_transfers.md#open-questions)) — failed uploads leave draft PRs; surface them on Upload-modal-open so the user sees stale attempts.
- [ ] **Re-enable `super_squash_history`** ([open question](docs/hub_transfers.md#open-questions)) — currently disabled; main accumulates N commits per upload (cosmetic only — atomicity and throughput unaffected). Need correct HF API usage or post-merge squash-on-main.
- [ ] **Retry budget UX** — third-strike retries should surface differently ("Failed 3× — check connection") rather than repeating the latest error.
- [ ] **`POST /api/hub/login`** — currently delegated to `huggingface-cli login` (out-of-band terminal flow). Add an in-GUI login form only if the standalone GUI runtime (no terminal) is supported.

### Async Request-Path Blocking Audit

The Hub outage exposed a general invariant: an `async def` FastAPI handler
still runs ordinary Python calls on uvicorn's event-loop thread. Any synchronous
network, subprocess, device, filesystem, Parquet, or CPU-heavy call inside it
can stall static files, websockets, SSE, and every other endpoint. Converting a
handler to plain `def` lets FastAPI run the whole handler in its thread pool;
handlers that need async locks/state should instead offload the blocking slice
and perform shared-state mutations back on the event-loop thread.

- [High] **Work off the 39 baselined event-loop violations.** `scripts/lint/no_blocking_in_async_baseline.txt` accepts the debt that existed when the lint landed; the ratchet only stops it growing. Counts should only ever go down, and `--update-baseline` after a migration is part of the change. Breakdown as of landing:
  - **32 × shared-pool** — `run_in_executor(None, …)` / `asyncio.to_thread(…)`, worst in `api/robot.py` (14) and `api/datasets.py` (9). Mechanical: give each class of work its own bounded `ThreadPoolExecutor`, following `_decode_executor` / `_prefetch_executor`. Low risk, can be done file by file.
  - **3 × blocking-call** — `LeRobotDataset(...)` constructed inline in async handlers at `api/datasets.py` 1512, 1545, 3270. **1545 is the dangerous one**: a bare `repo_id`, so it resolves against the Hub on the event loop. Covered in more detail by the "offload remote dataset open" item above; the other two are local (`root=` / `local_files_only=True`) and merely slow.
  - **4 × blocking-via-helper** — `_check_local_dataset_complete` (reaches `LeRobotDatasetMetadata`), `overlays.py` ×2 (`nvidia-smi pmon`, ~30 ms and cached at 1 Hz — probably an annotation rather than a fix), `bug_reports.py` (git sha).

  Note the counts undercount: the name-denylist half of the lint only sees what someone listed. Do not read a shrinking baseline as approaching zero blocking.

- [High] **Offload dataset mutation pipelines while retaining dataset locks.**
  Schema add/default migration/remove and Apply Edits synchronously rewrite
  Parquet shards, recompute stats, reload metadata, and verify the dataset;
  large datasets can hold the event loop for minutes. Keep the async
  per-dataset lock, execute the disk pipeline in a worker, then invalidate and
  replace shared state on the loop.
- [Med] **Move camera preview read + JPEG encode off-loop.** The nominally
  `async_read()` camera API is synchronous and may wait 200 ms; the frontend
  calls `/api/robot/camera-frame/{index}` at 10 Hz per camera. Offload the
  combined wait/encode operation or switch the endpoint to a nonblocking
  `read_latest()` snapshot with explicit stale-frame behavior.
- [Med] **Remove `nvidia-smi` from overlay status request paths.** Live/data
  status is polled every 500 ms, while `_proc_sm()` can synchronously run
  `nvidia-smi pmon` with a 3 second timeout. Sample GPU utilization in one
  background task and serve its cached value.
- [Med] **Offload remaining dataset read hot paths.** Cold `DatasetInfo`
  construction scans Parquet shards for per-episode-feature inference;
  frame-feature reads use `dataset[i]`; URDF trajectory and feature-series
  endpoints synchronously call `pd.read_parquet`; Hub diff recursively scans
  the local tree after its network call. Cache or thread these reads, especially
  the endpoints driven by scrubbing/polling.
- [Low] **Move bounded diagnostics/file scans out of async handlers.** Bug
  report submission runs two Git subprocesses and may decode/write a 16 MiB
  screenshot; model checkpoint inspection and full overlay-log reads can also
  become slow on large trees/files. Prefer synchronous route declarations or
  focused worker calls.

## Model Tab

See [docs/model_tab.md](docs/model_tab.md) for the (older) browse/inspect design and [training/DESIGN.md](training/DESIGN.md) for the cloud-GPU training pipeline + deployment design (the active workstream as of 2026-06).

- [ ] Phase 1: Browse & Inspect — source scanning, list/info/config endpoints, source tree, detail panel
- [ ] Phase 2: Training — subprocess launch/stop/status, training form, terminal output, resume
- [ ] Phase 3: Run Tab Integration — "Use in Run tab" passes checkpoint path to policy workflow
- [ ] Phase 4: Metrics & Polish — WandB embed, training curves, model comparison, HF Hub model sync
- [High] **Aggregate resumed attempts under one logical training job.** Keep each process invocation
  as an immutable attempt with its own host/image/config, timestamps, logs, terminal reason, and output
  directory—the isolation protects the last known-good checkpoint and preserves failure evidence—but
  stop exposing every attempt as an unrelated top-level run. Add a stable logical training ID plus
  parent/attempt metadata; group attempts in the API and sidebar; show an attempt timeline/selector;
  derive the job status from the latest attempt; combine active elapsed time; and merge metrics and
  checkpoints with visible attempt boundaries (when steps overlap after rollback, the newer attempt
  owns its displayed range, but the older attempt remains available). A rollback must branch from a
  read-only source checkpoint into a new attempt namespace; never overwrite or proactively delete a
  newer source checkpoint merely because the new attempt reaches the same step. Migrate legacy
  records as single-attempt jobs and define delete-job versus delete-attempt behavior.

  Keep the user-facing state model to exactly three outcomes: **completed** (successful target or
  explicit early-stop criterion), **stopped** (intentional or accidental termination), and **failed**
  (an unrecoverable error). Show the detailed stop/failure reason and checkpoint resumability
  separately rather than multiplying badge states. Every state badge needs one floating explanation
  available on pointer hover, keyboard focus, and touch/click—not hover alone.

### Cloud-GPU training pipeline

Active workstream tracked in [`training/DESIGN.md`](training/DESIGN.md). Phased status (Phase 0 + 1 done; Phases 2–6 pending — spawn step, API endpoints, host-profile UI, Model-tab frontend, auto-push-to-Hub, recipes). The bash CLI surface (`scripts/training/{install_prereqs,setup_host,run_training}.sh`) is the working prototype today; Python worker scaffold at `src/lerobot/gui/training_{jobs,worker}.py` is staged for the GUI integration.

## Model Debugger (Single-Frame Inference)

- [High] **Single-frame model debugger**: Run any model (S1, S2, Pi0, ACT, etc.) on a single dataset frame and inspect outputs. Goal: understand what a model "sees" at a specific moment, compare outputs across frames to diagnose bugs (gripper drops, wrong subtask predictions, action divergence).

  **Core features:**
  - Select a frame from any opened dataset (episode + frame index, or scrub to it in playback)
  - Select a model checkpoint (from Models tab)
  - Run inference -> show outputs:
    - **VLM (S2)**: decoded subtask text, latent vector norm/stats, token probabilities
    - **Action policy (S1, ACT, Pi0)**: action chunk as trajectory plot (per-joint over horizon), key stats (max delta, gripper values)
    - **Any model**: raw tensor shapes, norms, timing
  - Show input images (all cameras) alongside outputs

  **Compare mode:**
  - Pin frame A's output, select frame B -> side-by-side diff
  - Highlight which outputs changed and by how much
  - Use case: "frame 100 gives smooth actions, frame 105 gives a gripper drop — what changed?"

  **Episode sweep:**
  - Run model on every N-th frame of an episode, plot outputs over time
  - Visualize subtask transitions, latent norm trajectory, action variance
  - Overlay ground-truth subtask labels from dataset

  **Integration points:**
  - Data tab: right-click frame -> "Debug with model"
  - Models tab: "Test on frame" button on checkpoint detail
  - Saved dumps (`/tmp/hvla_drops/`): load images + state from dump directory

  **Live teleop probe mode:**
  - Teleop the robot freely while model(s) predict in real-time — predictions are displayed but NOT executed
  - Use cases:
    - **Subtask discovery**: teleop through a task, watch S2 subtask label change live
    - **Action preview**: freeze at a pose, see the predicted action chunk trajectory
    - **Confidence mapping**: teleop slowly through workspace, display model uncertainty
    - **Multi-model comparison**: run two checkpoints side-by-side on same live observation
    - **Data collection guidance**: while teleoping, see what the current model predicts
  - Implementation: teleop process writes obs to shared memory, separate model process reads and predicts, GUI displays predictions

  **Backend:**
  - `/api/debug/run-frame` endpoint: accepts model path + dataset + frame index (or image paths)
  - `/api/debug/live-probe` endpoint: start/stop live prediction alongside teleop
  - Lazy model loading with caching (don't reload for consecutive frames)
  - Returns structured JSON (predictions, stats, timing)

## Run Tab

- [High] **Sticky last-used Run form selections.** Long-standing pain point. The Robot dropdown (and the rest of the Run form) resets to defaults on every page load, and the default is whatever profile sorts first alphabetically — today that is `virtual_so107_follower`, which sorts before `white`. In two separate real sessions the operator launched against the virtual robot without noticing, burning a run and a debugging round each time. The Dataset selector no longer moves on its own (fixed — only the user changes it), but nothing remembers the user's choices across reloads. Persist the last-used selections (robot, teleop, FPS, dataset mode, task) — server-side beats localStorage so it survives browser changes; a small `run_form_defaults.json` next to the profile store would do. A hardware profile must never be silently substituted with the virtual one, and vice versa.
- [High] **Stranded hotkeys on GUI-launched runs — migrate all three consumers onto the stdin channel (recipe and the step-by-step generalization roadmap — per-kind vocabulary registry, subscribe hook, declarative frontend key registry: `lerobot/gui/CONTROLS.md`).** Inventory: RLT rewards `R`/`LEFT` (s1*process, wraps the pynput-only `on_press`), `so_leader` `SPACE` intervention toggle, and HVLA S2 standalone subtask injection `1`-`9`/`SPACE`. All three rode the retired local-keyboard path. **RLT detail — a knowing regression on X11:** The stdin control channel carries `exit_early` / `rerecord_episode` / `stop_recording` — episode \_flow*. RLT's terminal-reward keys are a different vocabulary: `R` signals SUCCESS (+1) and LEFT signals ABORT (`cfg.abort_reward`, default −1.0), and `s1_process` delivers them by wrapping `listener.on_press` — a `pynput` attribute. Before the control channel landed, this genuinely worked on X11 setups (pynput's global grab reaches a GUI-launched subprocess there); it was silently broken only on Wayland and under the terminal backend. The GUI now sets `LEROBOT_KEYBOARD_LISTENER=0` for subprocesses it launches — accepted deliberately so stdin is the single control source and Wayland works at all — which removes the working X11 path too: no keyboard backend, no reward path, every trajectory lands with the default terminal reward (the run itself is unaffected, and `s1_process` warns instead of crashing). Migration is the fix: extend the stdin vocabulary with the reward commands (both ends: `_CONTROL_COMMANDS` in `gui/api/run.py` and `_STDIN_COMMAND_TO_CONTROL` in `keyboard_input.py`, which a test pins as equal) and give `s1_process` a supported hook instead of monkeypatching `on_press`. Must land before the next GUI-driven RLT data-collection session on this (X11) workstation.

- [Low] Text output freezes after a while — teleoperate uses ANSI cursor-up in piped stdout
- [Low] Rerun web viewer has ~200ms visual lag (Rerun 0.26 limitation)
- [Low] Replay FPS setting doesn't seem to affect playback speed — remove if not useful
- [Low] **SIGKILL during HVLA shutdown leaks semaphores**. When stopping S1 inference (or any HVLA run) via the GUI's stop button, the 5s SIGTERM→SIGKILL grace window in `gui/api/run.py` sometimes isn't enough — visible as `resource_tracker: There appear to be N leaked semaphore objects` and `Process exited with code -9`. Cosmetic only (kernel reclaims on process death) but noisy. **Don't extend the timeout blindly** — 15s feels like an eternity from the GUI when the user wants to relaunch. First instrument the shutdown: log per-phase elapsed times for soft-land, teleop disconnect, robot disconnect (+ per-camera if accessible), shm cleanup. Get measured numbers from a few runs, then decide whether the fix is "kill the slow phase", "parallelize cameras", "release shm first", or just "the timeout is fine, the warning is harmless".
- [High] **Foolproof shutdown-cleanup registry.** Today the server's shutdown hook is a hand-maintained list of cleanup calls (`_stop_debug_process`, `_terminate_active_process`, `cleanup_stale_streams`, `cleanup_in_process_resources`, `shutdown_prefetch_executor`). Adding a new module-level resource — preview cameras, recording robots, prefetch executor, anything yet to come — means remembering to wire it into `server.py::shutdown_event`. Easy to forget, and the omission is silent (the OS reclaims FDs on process death so cleanup never visibly fails). Shape options:
  1. **Registry pattern.** Add `lerobot.gui.shutdown.register(callback)`; each module that opens a long-lived resource calls it at import / first-use. `shutdown_event` iterates the registry. Symmetric, discoverable; the resource owner is co-located with its cleanup. Easy to add.
  2. **AsyncExitStack on the FastAPI lifespan.** Migrate `@app.on_event("startup"/"shutdown")` to the modern `lifespan` async-context-manager API and have each module yield its `AsyncExitStack.callback(cleanup_fn)`. More idiomatic FastAPI, gets us off the deprecated `on_event` decorator (already a `DeprecationWarning` in the test output).
  3. **Resource ownership in `AppState`.** Every long-lived resource becomes an attribute of `AppState`; `AppState.close()` cascades cleanup. Most invasive but makes the lifecycle most explicit.

  Option 2 is the natural endpoint (we'll have to migrate off `on_event` eventually anyway). Option 1 is the quick win. Either way, the goal is: a contributor adding a new background thread / FD-owning resource doesn't need to know about `server.py`.

  **Orphan-subprocess failure mode (observed 2026-05-13).** A specific incarnation of this bug that bit hard: the GUI launches `lerobot-teleoperate` / `lerobot-record` / `lerobot-replay` subprocesses via `asyncio.create_subprocess_exec(..., start_new_session=True)`. The `start_new_session=True` is intentional — it detaches the child's process group so a Ctrl+C in the user's GUI terminal doesn't kill the workload mid-recording — but it also means **any abnormal GUI exit leaves the subprocess running**. Concrete case found: a `lerobot-teleoperate` subprocess was still alive almost 2 hours after the GUI server process that spawned it had exited, holding all four ttyACM ports and the cameras, blocking any subsequent launch. Only the graceful `shutdown_event` path calls `_terminate_active_process`; SIGKILL on the GUI (or any path that skips FastAPI lifespan) is silent and the subprocess survives indefinitely. Required to fix:
  - Whatever cleanup-registry mechanism we land must wrap the active subprocess so a parent-death detector (Linux: `prctl(PR_SET_PDEATHSIG)`, set in the child before exec) can SIGTERM it when the GUI's process tree dies — independent of the lifespan hook firing.
  - On GUI startup, scan for prior-launch zombies (e.g. by writing the subprocess PID into `~/.cache/huggingface/lerobot/gui/active_process.json` at launch and reading + killing on next startup if still alive and the heartbeat is stale).
  - The shutdown registry's iteration should be best-effort under exception, so a hung cleanup callback can't block the others.

- [Critical] **Umbrella-service process + resource registry (the fleet is NOT 1:1).** The GUI parent owns a _fleet_ of subprocesses at once — teleop, record, the overlay standalone, the S2 standalone, training jobs — each holding shm (obs stream, overlay buffer), serial ports, cameras, and GPU. The `active_process.json` idea above is itself a 1:1 assumption that breaks the moment two run concurrently, and today every resource is reclaimed by a separate ad-hoc mechanism (per-child `PR_SET_PDEATHSIG`, the blunt `cleanup_stale_streams` "only safe when nothing's running" sweep, a per-block write-timestamp). Each gap has been a real bug: an orphaned overlay standalone pegging the GPU for hours; a leaked obs stream the next standalone latched onto and served as frozen "live". Land ONE primitive: a **registry of the umbrella's legitimate processes + the resources each owns, keyed by a session token** (e.g. GUI PID + start-time, stamped into every child's env and every shm/meta header). One mechanism then answers everything:
  - **Query** — "all legit live processes/resources of _this_ session" (and, by difference against the OS, the orphans).
  - **Cleanup** — kill every live registered child on shutdown; supersedes the hand-maintained list and the per-child PDEATHSIG patch.
  - **Liveness / "off vs idle vs stale"** — a consumer (e.g. the overlay standalone reading the obs stream) asks "is the writer a process in this session, and alive?": alive + current-session → running but not advancing (teleop/playback paused); PID dead or foreign/stale session token → orphaned/stale stream. This is the only real way to distinguish "no publisher" from "idle publisher" from "stale stream" instead of guessing (see the overlays obs-stream warning, which currently hedges "is teleop publishing?").
  - **Orphan reclamation on startup** — any resource whose session token isn't current is a prior session's leftover → reclaim it specifically, instead of the all-or-nothing sweep.

  Extends the `[High]` shutdown-cleanup-registry above (replacing its single-process assumption) and closes the overlays obs-stream liveness gap.
  - **What remains vs. what's covered:** this is a PRE-EXISTING infrastructure limitation (single-robot shm design), not a bug introduced by any PR. The consumer-side slice is already handled (the worker re-attaches on the obs-stream meta-segment inode and reports "publisher gone" vs "no input frames"). Still open — the producer side: stamp `{writer_pid, session_token}` into the stream's meta header so a reader can tell **paused** from **writer-died-without-cleanup** (same inode in both cases today).

- [Med] **Remote/low-bandwidth mode (GUI over Tailscale is slow; RDP is smooth).** Measured 2026-07-22: a MacBook on a DIRECT WireGuard path (no DERP relay) received 356 MB of GUI traffic in one session and everything felt sluggish, while RDP to the same box was smooth — because over RDP the browser is local (all GUI traffic on loopback; only compressed screen deltas cross the WAN), whereas over Tailscale EVERY per-tile image poll + 0.5-2s status poll crosses the WAN: RTT multiplies each poll, payloads saturate the uplink, and the browser's ~6-conn-per-origin limit queues the rest. The GUI's polling design is localhost-first (see the RLT metrics item below — "fine on localhost, wasteful otherwise" — same disease). Cheap first step: quality-capped JPEG tiles with optional downscale (`?w=480&q=60`) + longer poll intervals for non-localhost clients (~10x traffic cut). Real fix: push, not poll — SSE/WebSocket for status/metrics, MJPEG (or WebRTC) for camera streams. Until then: use RDP for interactive remote work.
- [High] **RLT metrics pipeline is wasteful end-to-end**. Two compounding issues: (1) Training subprocess rewrites the entire `metrics.json` (~150KB for 5000 points × 8 series) on every save, even when only a few new points were appended — atomic write via `.tmp` + `os.replace` on each episode end and every 100 inference steps. (2) GUI polls `/api/run/rlt-metrics` every 2s and gets the same full snapshot back, of which ~99.6% is unchanged from the previous poll. Fine on localhost, wasteful otherwise. Options: append-only JSONL or shared memory on the write side; SSE push of only new points or cursor-based polling (`?since_step=N`) on the read side. Frontend maintains a local buffer.
- [Mid] **RLT dashboard chart smoothing**. Per-inference series like `actor_deltas` have per-sample noise comparable to the actual trend (e.g. δ raw σ ≈ 0.015 per sample vs a real β-driven shift of 0.018 — z=17.8 over 500 samples but invisible in any single-sample view). The dashboard plots raw values, so genuine learning signals get hidden. Add a smoothing toggle / moving-average overlay (window picker: raw / 50 / 200 / 500), or always show both the raw line (light) and a smoothed line (bold). Same applies to `q_values_*`, `critic_losses`, `actor_q_term`, `actor_bc_term` — all have per-grad-step noise. Cheap if the smoothing happens in JS over the buffered series the dashboard already pulls.
- [Mid] **Backend-driven Launch validation schema**. The Launch button's required-field rules currently live in JS-side `_WORKFLOW_VALIDATORS` and `_POLICY_VALIDATORS` registries in `gui/static/run.js`. Each policy_type's Pydantic/Draccus config already declares its required fields in Python — the JS registry has to be updated by hand whenever those declarations change, and silent drift between the two is invisible until a user clicks Launch and gets a backend 400. Replace with a `GET /api/policy-schemas/{policy_type}` (and similar for workflows) returning e.g. `{required: ["task"], required_when: {"rlt_token_checkpoint": "rlt_mode"}}`; the frontend consumes that verbatim, so adding a new policy means only one Python edit. Until then, any change to a policy's required fields must be reflected in both places.
- [Mid] **Audit `torch.load()` for `weights_only=True`** (bandit B614, currently in global skips). Call sites: `src/lerobot/policies/act_vlm/modeling_act_vlm.py`, `src/lerobot/policies/hvla/s1/flow_matching/model.py`, `src/lerobot/policies/hvla/s1_process.py`. Since PyTorch 2.6 the default flipped to `weights_only=True`; our checkpoints predate that and contain non-tensor pickled metadata, so wholesale flip would break loaders. Plan: per-site, switch to `weights_only=True` and migrate any non-tensor state to a sidecar JSON / safetensors. Remove B614 from `pyproject.toml` `[tool.bandit].skips` once the audit completes.
- [Mid] **Predictive follower amplifies Quest VR hand jitter — feel A/B.**
  Real-arm Quest-VR session 2026-05-30 (right shoulder_lift freshly
  swapped, calibration preserved): the **plain** `bi_so107_follower`
  logged 67 IK-warning events (66 on the right arm) but felt OK to
  drive. The **predictive** `bi_so107_follower_predictive` with VR
  logged 0 IK warnings — strictly cleaner targets — yet felt harder to
  control subjectively, with the arm appearing to overshoot small hand
  motions. Hypothesis: the predictive controller's 80 ms lookahead
  multiplied by the small high-frequency jitter that the VR hand
  carries (controller pose is a noisy 6-DoF estimate, not a smoothed
  leader-arm encoder) produces velocity-corrected setpoints that
  amplify rather than dampen hand tremor. Plain-follower's leader-pose
  is a low-pass-filtered motor encoder so the equivalent jitter never
  reaches the IK. To pursue: (1) record matched VR sessions on both
  followers + replay through both with a tremor-injected leader stream
  to confirm the predictive path amplifies frequencies the encoder
  path attenuates; (2) if confirmed, add a VR-specific
  `velocity_lowpass_hz` (or disable corrector_alpha) just on the
  `BimanualCartesianIKAdapter` -> predictive path, leaving leader-arm
  teleop unaffected. The 66/1 right-vs-left IK warning split on plain
  also needs explanation — could be the swapped motor's tighter range
  limits clipping IK output, or VR-hand resting orientation that
  drives the right arm closer to a singularity than the left. Worth
  splitting the two questions before tuning.

### Recording Loop Performance

**Measured baseline** — 2-episode bi_so107_follower record run with trajectory_replay, 4 cameras at 1280×720 @ 30 fps, instrumented build with `attach_pipeline_step_spans` + temporary per-encoder probes (2026-05-11, 729 iterations, snapshot in `outputs/record_instr/latency_snapshot.json`):

| stage           |    p50 ms |    p95 ms | p95/p50 | attribution (measured)                               |
| --------------- | --------: | --------: | ------: | ---------------------------------------------------- |
| loop_dt         |     23.76 |     27.98 |    1.2x | overrun 0.69%                                        |
| get_observation |      2.31 |      8.72 |    3.8x | motor reads + cached cam reads                       |
| process_obs     | **11.54** | **19.86** |    1.7x | **99.8% DepthEdgeOverlayProcessorStep**              |
| process_action  |      0.02 |      0.02 |       — | file-backed leader                                   |
| action_send     |      0.25 |      0.31 |    1.2x | sync_write to follower bus                           |
| dataset_write   |  **4.56** | **11.91** |    2.6x | **4 × per-camera `push_frame` (∑ medians ≈ 4.4 ms)** |

Per-camera `OurStreamingVideoEncoder.push_frame()` (inside `dataset_write`):

| camera                               | p50 ms | p95 ms | p95/p50 |
| ------------------------------------ | -----: | -----: | ------: |
| front (1280×720 OpenCV)              |   0.81 |   6.36 |    7.8x |
| left_wrist (1280×720 OpenCV)         |   1.15 |   7.70 |    6.7x |
| right_wrist (1280×720 OpenCV)        |   1.15 |   6.80 |    5.9x |
| top (1280×720 RealSense color+depth) |   1.26 |   8.16 |    6.5x |

**What the data invalidated about the earlier static-analysis pass:**

- ❌ **"DepthEdge ≈ 7 ms of 17.7 ms process_obs, with 8-10 ms unexplained gap."** Wrong. DepthEdge IS process_obs — 99.8% of it (11.52 / 11.54). No gap. Optimizing depth-edge captures the whole stage win, not a partial one.
- ❌ **"obs_stream writer adds ~0.84 ms per frame."** Doesn't fire from CLI launches — the step is gated on `LEROBOT_OBS_STREAM=1`, only set by the GUI runner. Only relevant under GUI-launched recordings.
- ❌ **"dataset_write p95 spike = `feed_frame()` timeout on the batch streaming encoder ([video_utils.py:828](../../datasets/video_utils.py))."** Wrong path. Default config (`streaming_encoding=False`, `record_images=True`, `batch_encoding_size=1`) goes through the per-camera `OurStreamingVideoEncoder.push_frame()` ([dataset_writer.py:278-280](../../datasets/dataset_writer.py)), not the batch `_streaming_encoder.feed_frame()`. The supposedly-non-blocking `push_frame` is bursting 6-8x p95/p50 anyway — different root cause.
- ✅ **"Off-loop `dataset.add_frame()` removes 4.5 / 12 ms from the critical path."** Confirmed.

**Actionable optimizations, ordered by measured impact ÷ effort:**

- [x] **Cut `DepthEdgeOverlayProcessorStep` cost.** Done in two steps (2026-05-11):
  1. **Inside the algorithm:** cProfile found 70% of the cost was `np.percentile(valid_gradients, 95)` on a ~900k-sample array (full sort, O(N log N)). Replaced with `np.partition([idx_lo, idx_hi])` + the same linear-interpolation formula np.percentile uses (O(N), numerically equivalent). Also swapped `np.sqrt(gx**2 + gy**2)` for `cv2.magnitude(gx, gy)`. Output bit-identical against a golden frame; in-loop savings ~2 ms p50 (run-to-run noise dominated the headline number).
  2. **Move it off the control loop entirely:** Added an optional `post_grab_processor` attribute on `RealSenseCamera` invoked inside `_read_loop` after color+depth are ready. The grab thread consumes depth and caches the overlay-RGB; the control thread reads `latest_color_frame` like a normal camera. `BiSO107Follower.__init__` installs `DepthEdgeOverlayProcessorStep` on each RealSense camera with `use_depth=True`. Empty `get_observation_processor_steps()`. Same algorithm, same bit-identical output, but the control loop never pays the cost.
     Measured: process_obs p50 9.8 → **0.01 ms** (entire stage gone from the loop). Loop p50 12.7 → **2.6 ms** (-79% this step, -89% from the pre-optimization baseline of 24 ms). Bonus: `top` camera staleness 11 → 6.8 ms p50 because the loop now iterates ~5× faster than the camera produces frames, so the consumer always grabs the freshest possible cached frame.
- [x] **Off-loop the CPU work hidden inside `OurStreamingVideoEncoder.push_frame()`.** Done (2026-05-11). The function claimed "Never blocks the caller" but called `_reservoir_sample()` synchronously, which allocated a 345 KB float64 array per call per camera (for 4 reductions on the running stats). Moved the entire reservoir + running-stats work into `_encoding_loop()` (which is already a background thread). Switched the reservoir's `random.randint` to a per-encoder `random.Random()` so concurrent encoder threads don't fight over the global random state. Stats output verified mathematically identical via a synthetic-frames correctness test. Measured on a real recording: `dataset_write` p50 4.56 ms → 0.07 ms (-98%); p95 11.91 ms → 0.09 ms (-99%); push_frame standalone benchmark 1.1 ms → 0.5 µs (-2000×). Bonus: `get_observation` p95 also dropped from 8.72 ms → 2.51 ms because the eliminated float64 allocations were the GIL contention source for camera cache reads. Headline: loop_dt p50 17.88 ms → 12.68 ms (the off-loop save alone was -5.2 ms, half from `dataset_write` and half from reduced GIL contention).
- [x] **Investigate per-camera `push_frame` 6-8x p95/p50 ratios.** Resolved by moving `_reservoir_sample` off the caller thread (2026-05-11). Post-fix push_frame p50 dropped from 1.1 ms to 0.5 µs — the ratio is now irrelevant because both percentiles are sub-microsecond.
- [Low] **`get_observation` p95 stays ~3.8× p50 even after the loop is otherwise gutted.** Measured 2026-05-11 post-refactor: p50 2.28 ms, p95 8.73 ms. Per-span attribution rules out the obvious suspects — `motor_read_*` p95/p50 ratio is ~1.2× (stable, hardware-bound) and the `camera_read_*` spans are all <0.1 ms p95 (the lock-protected cache read is effectively free). The remaining ~6 ms tail must be in the **gaps between spans** in [bi_so107_follower.py:get_observation](../../robots/bi_so107_follower/bi_so107_follower.py) — Python dict updates between motor*read_right exit and the camera loop, the for-loop dispatch itself, or GIL handoffs while the grab thread is mid-frame on depth-edge processing (~10 ms CPU work per RealSense frame, cv2 releases the GIL but Python wrapper code doesn't). Diagnostic: add a `latency_session.add_span("obs_gap*<n>", t0, t1)` between the existing spans to localise where the gap lives. Acting on it is only worth it when pushing past 60 Hz (16.7 ms budget) — at 30 Hz the p95 fits comfortably and a real fix likely involves moving more work to the grab thread or replacing the dict-merge bookkeeping.
- [Low] **`AsyncImageWriter` non-blocking + drop-frame fallback.** [datasets/image_writer.py:174-180](../../datasets/image_writer.py) — `queue.put()` defaults to blocking with infinite timeout; switch to `put(block=False)` with logging on `queue.Full`. Not active in our per-camera config (image writer bypassed for video keys), but defensive in case a future config path reaches it.

**Already-measured non-issues:**

- `motor_read_left/right` are stable (~1.2-1.4x p95/p50). No work to do here.
- `process_action`, `action_send` together cost 0.27 ms p50. Not worth touching.
- `IdentityProcessorStep` in the default pipeline costs 0 ms — confirmed no-op.

### Latency Panel UX

- [High] **Residual teleop jitter at fast speed**: the 1 Hz writer-thread refactor cut the loop-side spike from 22 ms to ~1 ms, but the user still feels small jitter under fast movement. Suspected GIL contention: the background writer thread holds the GIL during snapshot computation (transient aggregator construction, dict building, JSON dump), and the loop thread waits for the GIL on each iteration that overlaps the writer's work. To verify: time the loop thread's iteration during the second the writer runs vs during a quiet second; difference > 0 confirms GIL hit. Mitigations to try in order: (a) reduce writer-thread compute (skip `iterations` / `aggregate_iteration` when no GUI is polling? compute fewer percentiles?), (b) drop snapshot rate to 0.5 Hz, (c) move the writer to a subprocess (`multiprocessing.Process`) which doesn't share the GIL — heavyweight but the only way to truly isolate Python work from the loop.
- [Mid] **Surface action-to-state latency**: the panel shows camera staleness, loop timing, and model/inference stages (Gantt) but not the action-to-state latency — the physical lag from issuing a `Goal_Position` command to the motor actually reaching it. `action_send_ms` is bus-TX only ("not motor motion"), and `latency_monitoring.md` deliberately scoped tracking dynamics out of "latency". The predictive controller already measures this for free: its adaptive lookahead `L` is derived from a ~2 s cross-correlation of intent vs observed state, so for `*_follower_predictive` robots the number exists at runtime and just needs plumbing into the latency snapshot + a panel card. For non-predictive robots there is no live xcorr — either compute a rolling per-joint action-vs-state xcorr in the latency aggregator, or leave the card predictive-only with an "n/a" state. Offline counterpart is the "Control-Lag Analyzer" section above (dataset xcorr); this is the live-panel version.

### Safe Trajectory Probe

Goal: a tiny, pre-baked, motion-only "probe trajectory" that any robot can execute on demand to produce a quality report — joint-space, conservative speed limits, stays inside a known safe envelope, no Cartesian collision math, no human teleop required.

Use cases:

- **Real action_send_ms measurement** under bus load — currently 0 in our dry-run tests because send_action is no-op'd.
- **Tracking lag measurement** without needing an operator at the leader arm; replaces the cross-correlation analyzer for hardware-side characterisation.
- **Reproducible hardware benchmarks** — same trajectory across different robots / different days, compare per-joint lag, peak loop_dt, encoder behaviour under predictable motion.
- **Cross-camera sync verification** when the arm IS moving (camera staleness against a known-moving target).
- **Per-joint motor health checks** — does each motor respond? backlash, thermal envelope.

Design constraints:

- [ ] Joint-space waypoints, not Cartesian — sidesteps IK / collision concerns.
- [ ] Per-robot **safe envelope** config: per-joint min/max angles, max velocity, max acceleration. Stored next to the robot profile.
- [ ] Smooth profile (cubic spline / trapezoidal velocity) — no step inputs, no near-limit positions.
- [ ] Hand-recorded once with the leader arm — the operator records the safe envelope trajectory, the file becomes the reference. Stored alongside the robot config (e.g. `safe_probe.parquet`).
- [ ] **Not** the same as Replay (which is for dataset playback). This lives in `src/lerobot/scripts/lerobot_probe.py` and emits a structured report.

Output: `<robot_id>_probe_report.json` with per-joint tracking lag, peak loop_dt, overrun ratio, per-camera staleness during motion, per-joint backlash estimate (max(state - action) reversing direction), and a pass/fail verdict against the robot's expected baseline. Drop-in replacement for the `meta/recording_health.json` format so the data panel can render it the same way.

Initially: build for `bi_so107_follower` (the active hardware), generalise later. Build only when there's a concrete use case forcing the hand — for now this TODO captures the design.

### Dataset Debugging Overlay

Live overlay during teleop/record showing how the current state compares to the dataset — helps the user identify gaps in data coverage and fill them efficiently.

- [ ] **Live coverage indicator**: compare current observation against the dataset (either a selected reference dataset or the one currently being recorded). Show as an overlay badge (like the S2 subtask overlay) indicating how "novel" the current state is relative to existing data.
- [ ] **Growing coverage feedback**: as the user records more episodes, the feedback should reflect that more cases are covered — "you've seen states like this N times" or a heatmap-style confidence.
- [ ] **Define similarity metric**: what does "similar" mean? Options to explore:
  - Joint state L2 distance (cheap, ignores visual context)
  - Image embedding distance (e.g. DINO/CLIP features, captures visual similarity)
  - S2 latent distance (if debug model is loaded — reuses existing infrastructure)
  - Hybrid: state distance + image embedding distance
- [ ] **Nearest-neighbor lookup**: build an index (e.g. FAISS) over dataset observations, query with current obs each frame. Display distance + closest episode/frame reference.
- [ ] **TODO: hardcoded vs generic**: start with a simple metric (joint state L2 or S2 latent distance), add a comment that this will be generalized later (same pattern as model debugger overlay).

## Overlays

The observation-processing panel (data + live), SAM3 first. See `src/lerobot/gui/docs/overlays.md`.

- [Med] **Migrate the S2 model-output debugger into the Overlays framework.** The panel is meant to be the single debug-model surface, but today it's perception-only (`_STEPS` = SAM3); HVLA S2 still lives in the separate "Debug model" / "Model Output" path (`/api/run/debug/{load,output,subtask}` → `policies/hvla/s2_standalone`, the bottom-right terminal + the subtask camera overlay). Fold it in: let an overlay model declare an output schema (text vs image) so S2's subtask text renders through the same panel (tile overlay and/or the panel log), then retire the old path. The hardcoding to generalize is marked by the three `TODO: generalize … model output schema` comments in `api/run.py`, `static/run.js`, `static/index.html`.
- [Low] **Multi-instance masking per concept.** SAM3's seed takes the single _largest_ detected instance per concept name (`_detect` → `max(masks, key=area)` in `overlays/adapters.py`), so e.g. "meat" masks the one clearest cube, not all of them on the plate. Seed every detected instance (one tracker obj-id each) so a concept masks all of its instances. Future optimization — the single-instance path is correct, just incomplete.
- [Low] **No-prompt / default detection.** The live overlay won't launch until at least one concept is named (`namedObjects().length > 0` gate in `static/overlays.js`), surfaced as a red-bordered required field. The earlier debug-vision (grounding-DINO) detected all object-alike things with no prompt; SAM3 here needs an explicit concept. Follow-up: let the step start with an implicit default (detect `"object"`, or a generic-objectness pass) so it runs without naming, restoring that behaviour — make it a per-step capability flag (concept-required vs prompt-optional vs no-objects like depth) rather than the current unconditional gate. Accepted limitation for now; see `docs/overlays.md` → "What the mechanism implies for the controls".
- [Low] **Run-tab panel polish**: resizable panel, auto-unload idle timer (free VRAM after the keep-warm window), and a full in-panel log viewer (v1 is the badge + a roomy alert).
- [Med] **Multiplex the observation stream (run ↔ data conflict).** `lerobot_obs_*` is a singleton (fixed shm names), so the run subprocess (teleop/policy/record) and the data-tab overlay publisher can't both own it. The GUI arbitrates one-writer-at-a-time — `is_run_active()` refusal + a `stop_data_publisher()` teardown on run-start (`api/run.py` / `api/overlays.py`) — so a live-run overlay and a data-tab overlay can't be active at once. Pre-existing debt in the single-robot stream design, exposed by the data publisher. Proper fix: namespace the stream per context (e.g. `lerobot_obs_<ctx>_*`) or a small stream registry so multiple producers/consumers coexist. See `docs/overlays.md` → Architecture (known limitation).

- [Low] **Empty data-camera selection mislabels the worker log.** With zero cameras selected, the publisher sends `_data_pub_cameras or None`, so `[]` collapses to `None` and the worker's `_resolve_active(None)` falls back to all cameras — the log's `active=[...]` then lists every camera even though the worker correctly idles (no frames are published). Cosmetic: the idle behaviour is right, only the label is misleading. A faithful fix needs the worker to tell an explicit-empty selection (none) from an absent filter (all), which `_resolve_active` deliberately conflates (`[]`/`None` → all — shared with the live path + a unit test), so it's not a one-liner.

- [Low] **Module move follow-ups (`debug_vision` → `lerobot/overlays/`, done #41).** The overlay worker / adapters / IPC / state-machine moved out of `policies/` to a neutral top-level `lerobot/overlays/` (it is not a policy; both the GUI and policies depend on it, so it can't live under `gui/` either). Remaining tidy-ups, none blocking: (1) the overlay IPC still imports `SharedBlock` from `policies/hvla/ipc` — a general module reaching into one specific policy; promote it to a neutral shm primitive (e.g. `lerobot/overlays/shm.py` or a shared `lerobot/ipc/`). (2) The `debug-vision` pyproject extra is now misnamed — rename to `overlays` (touches pyproject + install docs + CI; left as-is to keep the move minimal). (3) Consolidate `aux_ipc.py` + `overlay_ipc.py` → one `ipc.py` — deferred because `aux_ipc.py` is a #42 file and `overlay_ipc.py` is #41, so the merge spans the stack; do it after both land.

#### Policy saliency (model-internal overlay) — async + raw-capture

The gradient/rollout saliency runs **synchronously in the S1 inference thread** (not the control loop), demand-gated + every Nth inference; the badge shows the live cost (`sal N ms`). Measured (`flow_s1_no_s2` ckpt-40000, vits14, 4 cams @224): gradient ~30–36 ms · rollout 23 ms · one 15-step inference 36 ms, with a ~50× chunk budget margin — cost is a non-issue; these follow-ups are about capability, not perf. Open items:

- [Big] **Attention map for ANY policy (currently S1-only compute).** The seam is already generic (worker/GUI/aux/`SaliencyPublisher`); what's missing is (a) a generic gradient compute and (b) a publisher hook in the standard record loop. Surveyed 2026-07-22 (9 families, file-level):
  - **M1 — generic compute** `overlays/saliency.py::compute_input_saliency_generic(policy, batch, preprocessor)`: clone image leaves with `requires_grad`, run the policy's _stateless chunk-level differentiable path_, backprop the first-future-action norm, area-pool. Per-family shim registry names that path — **easy tier** (differentiable continuous chunk method exists): ACT (`predict_action_chunk.__wrapped__` / `policy.model(batch)`), ACT-VLM (same + the internal frozen-DINO `no_grad` needs the S1-style unfreeze), SmolVLA (`_get_action_chunk` after `_prepare_batch`), Diffusion (`diffusion.generate_actions` directly — its `predict_action_chunk` reads from internal queues so grad-clones passed in yield `grad=None`; drop `num_inference_steps` to ~4–10 for the pass).
  - **M2 — record-loop hook**: construct the publisher in `lerobot_record.record()` when `cfg.policy` is set; `publish(observation_frame)` at the two `predict_action` sites in `record_loop` (the frame already has `observation.images.<cam>` keys, CPU HWC uint8 — the publisher's expected shape). The compute applies the checkpoint preprocessor itself. Note: per-STEP hook (chunked policies pop queues between real forwards) — the publisher's `every` cadence throttles; the compute is stateless chunk-level so it does a fresh pass regardless.
  - **M3 (on demand) — medium tier**: pi0 (strip two stacked `no_grad`s), pi05 (a `copy.deepcopy` of the KV cache inside `denoise_step` blocks grad — needs a small bypass), GR00T (inner `get_action.__wrapped__`), multi-task-DiT (queue-overwrite bypass + weave grads into the n_obs_steps history stack).
  - **Non-goals**: pi0_fast gradient (discrete autoregressive tokens — blocked at argmax); rollout on non-transformer/non-pixel-attention archs (diffusion has no attention; mtDiT attends only over action tokens). Dispatch stays `getattr(policy, "compute_input_saliency", None) or GENERIC_REGISTRY.get(type)` — S1 keeps its bespoke path.
- [Med] **Async the compute off the inference thread** — a background thread + dedicated CUDA stream so the inference thread isn't stalled. The policy is small, so a side stream likely overlaps with inference (near-free); rollout (forward-only) backgrounds cleanly, the gradient touches autograd + the `freeze_backbone` toggle (shared mutable state) so it needs a lock. NOTE: one GPU — async removes the stall/jitter, not the GPU contention.
- [Big] **Capture raw saliency data at run for deeper OFFLINE sync analysis** (user direction 2026-06-28). Promote the throwaway `.pt` batch dump (the offline-lab prototype) into a first-class run-time capture — raw attention tensors / activations / preprocessed batches — written as part of the dataset or a new replay format, so any attribution method (rollout · Chefer-MM · AttnLRP · gradient · future) can be run + compared offline without re-running the robot.
- [Low] **Borrow from `cijerezg/lerobot`'s `docs/recap/metrics.md` attention suite** (reviewed 2026-06-30). Theirs (π₀.₅, a unified VLA) is RAW single-layer (0/9/17) cross-attention, per-head→mean, mean over action queries, probed at one diffusion timestep — an OFFLINE dataset probe. Ours (`compute_attention_rollout`) is more principled: residual-aware ROLLOUT (Â=½A+½I) across the obs_encoder layers ∘ the decoder cross-attn — REQUIRED by our SPLIT obs_encoder (it mixes patches before the decoder sees them; theirs is unified, so a single layer suffices). Verified our rollout math is correct (Abnar-Zuidema; not a bug). Worth borrowing: (1) **query-typed maps**, esp. a SUBTASK-attention map for HVLA (where the subtask token looks); (2) a **spatial-memorization** metric — saliency/attention hotspot persisting across DIFFERENT episodes regardless of scene = a memorization/corruption detector (cheap once we dump grids; pairs with the [Big] above); (3) per-head panels + bicubic upsample (vs our area-pool). Note: their "Action Drift Jacobian" is attention-gradient (d action / d attention); our `compute_input_saliency` is input-gradient (d action / d pixels) — ours is the more direct causal signal.
- [Low] **Surface a "warming up" state for the saliency overlay** so it doesn't read as frozen. The first-ever `rollout` pass in a run pays a ~7 s one-time setup (it installs the obs_encoder forward hooks + monkeypatches SDPA), during which the overlay holds the last frame; the policy's `torch.compile` warm-up (~1 s) and the first gradient pass also delay first paint. Verified live 2026-06-28: switches are otherwise instant (gradient↔rollout both directions), but the first rollout looks "stuck". Mark the overlay badge "warming up (~N s)" on method-switch / first-paint — pairs with the named idle states (live / paused / publisher-gone) so a warming overlay isn't mistaken for the (now-swept) stale-stream freeze.
- [Med] **Multi-camera worker first-draw is slow (~50 s for 4 cameras vs ~11 s for 1).** Measured live 2026-06-28: `policy_saliency` now defaults to ALL cameras (a `fast` model), so the worker launches `--cameras front left_wrist right_wrist top` and the first overlay didn't paint until ~55 s (the frontend `started` flips true only once the worker reports `active`); a single-camera launch painted at ~11 s. The cost scales far worse than 4×, pointing at per-camera startup work blocking the first `active` report (per-camera obs-stream attach, the first colorize, or the `active` gate waiting on _all_ cameras). Don't guess — instrument the worker spawn (per-phase elapsed: import · obs-stream attach per camera · first colorize per camera · first `active` report), get measured numbers, then pick the fix (parallelize per-camera attach · report `active` on the FIRST camera ready · or draw each tile as it becomes available). The "warming up" badge above currently masks it.
- [Low] **`policy_saliency` panel shows "name an object to start".** The live status text is the generic object-prompt (`hasObj`-gated, overlays.js ~line 359), but `policy_saliency` declares no `objects` control (`requiresObjects()` is false, so it isn't actually gated) — the text is just misleading. Show a model-appropriate prompt (or nothing) when the step has no `objects` control.

#### Performance (perf audit 2026-06-27)

- [Med] **Back off re-detection on cameras that can't lock a concept.** A camera with no track re-runs the full ~200 ms detector every frame — e.g. "plate" isn't visible on the front view, so front pays ~200 ms/frame while top tracks at ~65 ms, dragging the 2-camera case to ~4 fps. After K consecutive failed detects, drop that camera's re-detect cadence (retry every Nth frame). Manual workaround today: deselect the camera (the camera-selection plumbing). Adapter seed-logic change; trade-off is a few-frame delay to lock if the concept later appears there.
- [Med] **Batch cameras through the GPU** instead of the serial per-camera loop (N× latency). A single 720p inference underutilizes the GPU, so a batch is ~1.5× not N×. The detector seed batches cleanly; the tracker propagation is per-camera session state, so batching it needs work inside the SAM3 video predictor. Highest ceiling, highest complexity.
- [Low] **Stop decoding each data frame twice.** The camera-tile display (`get_frame`) and the overlay publish (`ds[idx]` in `data_publish`) decode the same item separately; a 4-camera dataset with one active camera also decodes all four. Decode once / only the active cameras so the publish stops capping the effective rate on multi-camera video.
- [Low] **Inference-resolution knob** — downscale the frame fed to the model (coarser mask, faster inference); a quality/speed dial for "many cameras live".
- [Low] **`infer/s` metric clarity** — it is cycles/s (one pass over the active cameras) while `compute Xms` is per-camera, so they do not multiply to 1000 (invites "why does 124 ms make 4 fps?"). Report per-camera rate + cycle time, or label it cycles/s. Tiny adjacent wins: WebP instead of PNG overlays; batch the per-camera overlay fetch into one response.
- [Low] **Audit follow-ups (2026-06-27 overlay audit).** Frontend fetch `.catch(() => {})` should `console.error` on the critical endpoints (data/configure, live/start) instead of swallowing; `_get_live_reader` / `_read_status` should log their _expected_ attach failures at DEBUG rather than suppressing silently.

## Robot Tab

- [Low] UX consistency pass: ensure consistent button coloring/hierarchy across views
- [Low] ~1s latency when first opening the Robot tab while loading profiles
- [ ] **Surface `/api/robot/recover` from runtime error states (deeper integration).** A dedicated "Recover" button is shipped next to the rest-position buttons (`renderRestPositionSection` in `robot.js`), and `recoverRobot()` chains into `/move-to-rest-position` when the report is clean and a rest position is on file (with doubled `duration_s` if anything was recovered). What's missing: rendering an inline "Try recovery" link next to the error toast when teleop / record / replay / move-to-rest fails because of a wedged motor chain (e.g. `FeetechMotorsBus motor check failed ... Missing motor IDs`). Today these errors are printed-only (`logger.exception` + raw string in the response body), so users have to read the server log to know what's wrong, and there's no path from "I see an error" to "click to attempt recovery". Prereq: consolidate error handling — promote the relevant `RuntimeError` / `ConnectionError` from the bus and robot layers into a small set of typed exceptions (e.g. `MotorChainWedged`, `PortBusy`, `MissingMotor`) that the FastAPI layer maps to structured error bodies (`{"error_code": "...", "detail": "...", "remediation_hint": "recover"}`). The frontend can then render the appropriate hint generically instead of string-matching error messages.

### Guided Calibration Wizard

LeRobot's stock calibration is rudimentary — `robot.calibrate()` is blocking
`input()` / `print()` prompts, no visual reference for the "middle of range"
homing pose, no verification. Replace it with a GUI-driven, visually-guided
wizard. Designed as three phases; ship phase 1 first as its own PR.

- [ ] **Phase 1 — motor calibration, any robot, no URDF.** (1) Homing: show a
      mid-range reference pose, the user matches the physical arm to it, capture.
      (2) Ranges: GUI-driven range-of-motion recording. Reuse the motor-bus
      primitives directly (`set_half_turn_homings`, `record_ranges_of_motion`,
      `MotorCalibration`, `write_calibration`) — NOT the `calibrate()` wrapper,
      which is CLI-`input()`-bound. Emit the same calibration files so the tool
      complements `lerobot-calibrate`. No camera pane — the human is at the robot.
- [ ] **Phase 2 — URDF verification + live layer-2 tuning.** For robots with a
      vendored `*_description` URDF: verify the URDF against the live observed
      pose, and let the user tune the per-joint `(sign, offset_deg)` layer-2
      alignment interactively instead of hand-editing `joint_alignment.py`. Only
      CAD-exported URDFs (e.g. SO-107) need this; a calibration-aligned URDF
      (SO-101) is identity. Depends on the URDF state visualization.
- [ ] **Phase 3 — kinematic calibration (research).** Beyond joint zero/range:
      link-length / mounting-offset estimation. Scope TBD.

### Motor Diagnostics (per-arm health panel)

When a motor on a real arm misbehaves the user currently has to drop into
ad-hoc Python (e.g. `scripts/diag_motor_2.py`, `scripts/scan_motor_variants.py`)
to figure out _why_. Surface the same information in the Robot tab as a
per-arm diagnostic panel so the diagnosis flow is "click → see structured
report" instead of "open a terminal, write a probe, paste output to a
collaborator." Captured after a real session where a dying right-arm motor
took several rounds of CLI work to isolate.

**Working CLI today:** the diagnostic + migration flows already exist as
`scripts/motor_tools.py` on the `feat/motor-tools` branch
(`inventory` / `health` / `read` / `write` / `set-id` subcommands) plus
`scripts/MOTOR_MIGRATION.md` for the end-to-end swap-preserving
procedure. The work below is to lift those flows into the GUI; the CLI
is the spec.

- [ ] **Per-motor health probe.** For each motor on the selected robot
      profile: ping, then read `Torque_Enable`, `Goal_Position`,
      `Present_Position`, `Present_Current`, `Moving`, `Operating_Mode`,
      `Lock`, `Min_Position_Limit`, `Max_Position_Limit`, `Max_Torque_Limit`,
      `Torque_Limit`, `P_Coefficient`, plus a tiny commanded-move test
      (both directions) with a watch loop. Diagnostic verdict ladder:
      _bus failure_ (ping fails) → _MCU rejects writes_ (Torque*Enable
      doesn't latch) → \_driver dead* (Goal/Present diverge but Current=0)
      → _mechanical jam_ (Current spikes, Position doesn't change) →
      _misconfigured_ (Operating_Mode / Lock / P=0). Sounds-like-buttons
      next to each motor row.
- [ ] **Motor registry / per-arm inventory.** Read
      `Firmware_Major/Minor_Version`, `Max_Voltage_Limit`, `Present_Voltage`,
      `Model_Number` for every motor on every connected port and surface
      them in a per-arm table. Caveats discovered while building the CLI
      version: STS3215 variants (7.4V vs 12V; 1:147, 1:191, 1:345 gear
      ratios) all share `Model_Number=777` and identical firmware
      (3.10). **`Max_Voltage_Limit` is a writable EEPROM field**, so a
      heuristic that calls 8.0V "7.4V variant" and 12.0V "12V variant"
      breaks the moment anyone writes that register (real example:
      shoulder*lift motors on both leaders of `white` read 8.0V while
      the rest of the same arm read 12.0V — manual rewrites, not actual
      variant difference). **Gear ratio isn't readable from firmware at
      all** — the MCU only sees the encoder. So the registry is honest
      diagnostic info but **must not pretend to identify variant or gear
      ratio**; ground truth is the printed label on the motor housing.
      A useful UI gives the user a place to \_record* the variant they
      see on the label, persisting alongside the calibration so future
      sessions remember which motor is where.
- [ ] **Swap-preserving recalibration helper.** When a single motor needs
      replacement, the standard `calibrate()` flow throws away the rest
      of the arm's calibration and re-measures _everything_ via human
      eyeballing the canonical pose (±2-5° per joint of drift versus old
      datasets). A surgical tool would let the user (1) before pulling
      the dying motor, capture `raw_old` and `homing_offset_old` at any
      held pose; (2) pause for the swap; (3) re-read `raw_new` at the
      same physical pose; (4) compute `homing_offset_new = raw_new -
(raw_old - homing_offset_old)` and update only that one motor's
      entry in the calibration JSON. Preserves the rest of the arm's
      old calibration exactly, so existing datasets, FK, and IK frames
      stay valid without recalibrating six other joints. Critical detail:
      the captured-pose step needs a visual / mechanical reference the
      user can reproduce post-swap (mechanical end-stop, marked tape, or
      lining the broken arm against the working twin) — surface those
      options in the wizard. **Also write `Min_Position_Limit` /
      `Max_Position_Limit` to the new motor's EEPROM**, not just
      `Homing_Offset` — a fresh motor harvested from a different role
      carries stale limit registers that silently clip `Goal_Position`
      to a narrow window unrelated to your arm. The CLI counterpart
      (`scripts/motor_tools.py write … --min-position-limit=… \
      --max-position-limit=…`) hit this exactly: first attempt drove to
      809 ticks then froze at 958 because the new motor had been a
      gripper on its previous arm with max=952.

## Architecture

- [High] **Detect event-loop blocking at runtime, not by listing call names.** `scripts/lint/no_blocking_in_async.py` matches a denylist of blocking calls (`requests`, `subprocess`, `whoami`, …). That can only catch what someone thought to list, and the danger is not a name — it is reaching a syscall that can block indefinitely, which any abstraction can hide. Demonstrated: three `LeRobotDataset(...)` constructions run inline inside `async def` handlers and the checker saw none of them, including `datasets.py:1545` which passes a bare `repo_id` and therefore resolves against the Hub _on the event loop_ — the exact hang that motivated the lint. Adding those two names to the table patches the instance, not the class. The fix is to measure the symptom: `loop.set_debug(True)` with `loop.slow_callback_duration` already logs any callback that hogs the loop however the blocking is spelled, and a watchdog task (sleep 100 ms, measure drift, dump `sys._current_frames()` past a threshold) names the culprit. Then make it fail rather than log: the existing `e2e_flow` GUI tests drive real endpoints, so running them with asyncio debug on and asserting zero slow callbacks turns the suite we already have into a complete blocking detector. Keep the static lint as a fast pre-commit signal — the shared-pool rule is a genuine static property a runtime check cannot give — but stop treating it as the safety net.

- [Medium] **SmolVLA cannot be fine-tuned from `smolvla_base` onto a robot with wider telemetry.** The model-compat paving work made SmolVLA's `max_state_dim` / `max_action_dim` derive from the dataset, which fixes training _from scratch_ — a 48-dim OpenArm 2.0 state correctly grows `max_state_dim` 32 → 48. Fine-tuning from the published base checkpoint still fails: the config grows, the model is built 48-wide, and then the base's 32-wide `state_proj` refuses to load (`size mismatch for model.state_proj.weight: [960, 32] vs [960, 48]`, reproduced via `make_policy` with `pretrained_path="lerobot/smolvla_base"`). This was equally impossible before — the old code would have mis-padded instead — so it is a standing limitation, not a regression. Fixing it means deliberately re-initialising the projection layer (and the matching action head) when the dataset's width exceeds the checkpoint's, and deciding what that means for fine-tuning: a re-initialised state projection discards the base model's proprioception encoding, so it should be an explicit opt-in with a loud log line, not a silent fallback. Until then, SmolVLA on a >32-dim-state robot means training from scratch.

- [High] **Shrink the fork's divergence in `scripts/lerobot_record.py`.** Upstream's is 550 lines; the fork's is 1326 — **+886 / −110** against upstream, the single largest modification the fork makes to an upstream file. Every sync pays for it: this file produced the most conflict hunks in the 2026-07 merge, it is where the `record_images` type mismatch hid, and it is where `stamp_repo_id()` was lost as collateral while resolving an unrelated region three lines away. The additions are real features — policy-in-record, `LatencySession`, episode health metadata, intervention datasets, `rename_map` — but they live inline in a file upstream also rewrites. The goal is to make this file close to a thin fork of upstream's: move each addition behind a seam the fork owns (a subclassed config, a processor step, a strategy object, a separate module `record()` imports), so upstream's version can be taken almost verbatim. Worth costing before the next sync rather than after. Related: upstream split policy deployment into `lerobot-rollout`, which the fork does not use — deciding whether to adopt that split would remove a large share of the divergence on its own.

- [High] **Two incompatible subtask models coexist in the tree** — the fork's `subtask_index` + `meta/subtasks.parquet` and upstream's `language_persistent`. Every sync re-litigates it. See [docs/stories/language_annotations.md](docs/stories/language_annotations.md).

- [**Critical**] **Reactive UI state management**: the current imperative DOM manipulation (innerHTML + manual toggle/refresh calls scattered across functions) is fundamentally broken. Field state (disabled, visible, selected) must be called at every possible code path that reveals the element, leading to endless monkey-patching. Migrate to a reactive pattern where UI state is derived from data (React, Preact, or even a minimal reactive store + render loop). This blocks every new UI feature.
- [High] **Coherent GUI-wide persistence policy.** Persistence is currently ad-hoc, per-feature: opened datasets live in `opened.json`; model sources in `model_sources.json`; robot/teleop profiles under `~/.config/lerobot/`; per-feature view state in scattered LocalStorage keys; **launch settings (Run tab, Model Debugger, RLT mode, episode counts, robot/teleop profile choice, etc.) aren't persisted at all and reset to defaults every session**. The result is two distinct failure modes for users:
  1. **Silent default surprise.** A toggle that controls destructive behaviour (e.g. RLT `--rlt-deploy` vs train mode) reverts to its default between sessions. The user reasonably assumes their last choice is still in effect, launches, and only realises mid-run (or later, reading logs) that the wrong mode ran. Real example: 2026-05-14, an HVLA + RLT rollout intended to be DEPLOY mode launched in TRAIN mode because `--rlt-deploy` wasn't toggled in the GUI — gradient updates ran for an entire evaluation session, contaminating the checkpoint.
  2. **Stale-choice surprise.** A persistent dropdown (robot profile, model checkpoint path) retains a previous selection from a different workflow; user launches without noticing. Reported once at least.

  These are symptoms of a missing GUI-wide policy. Worth thinking through, not just patching per tab.

  Concrete questions to answer in a short design doc before implementation:
  - **What to persist?** Per-tab session state (current selections, expanded/collapsed view), per-workflow last-used parameters (last train args, last record args, last debug-model checkpoint), or both? Different lifetimes — session state is per-browser; workflow state is per-user.
  - **Where to persist?** Server-side JSON in `~/.config/lerobot/gui/` (survives browser changes, can be backed up, single source of truth) vs LocalStorage (zero round-trip, per-browser). Probably server for workflow state, LocalStorage only for UI affordances (which panel was expanded, scroll position).
  - **Reset semantics.** Some launches must NOT inherit prior state (training run on dataset A → next launch shouldn't auto-pick A; user might be deliberately switching tasks). Need an explicit "use last config" vs "fresh defaults" affordance, and the default should be the safer of the two.
  - **Visibility / freshness.** When the GUI restores 2-week-old launch flags, the user should be told. Show a freshness banner ("Last used 14 days ago — review settings before launching") or surface a diff against a documented default.
  - **Audit trail.** Currently the only record of what was launched is the GUI server log. Make the resolved CLI / config visible in the UI before submission — both as a confirmation step and as a record. The Launch button modal could show the constructed command in a copyable block.
  - **Safe defaults for destructive toggles.** Things like RLT train mode, dataset deletion, push-to-Hub: defaults should fail closed (require explicit opt-in each session), regardless of persistence.

  Likely deliverable: a `GuiPersistenceStore` server-side abstraction (single JSON file per concern, atomic write, schema-versioned for migrations) + a frontend wrapper that hydrates form fields on tab mount + a "Launch summary" modal that surfaces resolved settings before submission. Touches every tab — but the consolidation is the win.

  Scaling concern: as platform features grow (HVLA, RLT, intervention, eval suites, dataset tools), the surface of "settings that could surprise the user if not persisted (or persisted incorrectly)" grows linearly. Doing this once at the policy level is cheaper than retrofitting per-tab.

- [Mid] **Config grouping / categories** (to be designed): the number of configurable settings keeps growing (Run tab workflow params, Robot tab profile fields, RLT toggles, debug-model knobs, etc.) and they're currently rendered as one flat form per tab. Need a grouping/category abstraction so related fields cluster visually — collapsible sections, tabs-within-tab, "Advanced" reveal, or schema-driven groups declared alongside the Pydantic/Draccus config. Design open: where the grouping metadata lives (Python config decorators? frontend JSON?) and how collapsed state persists (ties to the [Persistence policy](#) item above).
- [Mid] Cross-tab data synchronization — replace point-to-point refresh calls with pub/sub event bus
- [High] Extract frontend to separate files (then optionally migrate to React/Vite)
- [Mid] FastAPI dependency injection for AppState — replace global `_app_state` with `Depends(get_app_state)`
- [Mid] Consolidate module-level caches (`_episode_start_indices`, `_dataset_info_mtime`) into AppState
- [High] **Unified intervention contract — any teleop in any flow.** Three call sites assume a human-intervention surface on the teleop: [`s1_process.py` (HVLA + RLT)](../policies/hvla/s1_process.py), [`lerobot_record.py` (regular record with policy)](../scripts/lerobot_record.py), and [`hil_processor.py` (HIL serl gym)](../processor/hil_processor.py). The "contract" today is **five methods that every teleop re-implements independently**:
  - `intervention_enabled: bool` field in the config (HVLA forces it `True` at launch — a teleop config that doesn't declare the field crashes at draccus decode time)
  - `get_teleop_events() → {IS_INTERVENTION: bool, TERMINATE_EPISODE: bool, SUCCESS: bool, RERECORD_EPISODE: bool}`
  - `reset_intervention()` at episode boundary
  - `disable_torque()` / `enable_torque()` for the policy↔teleop swap (only meaningful for physical leaders; non-physical teleops like Quest VR have nothing to do here, but the call sites guard with `hasattr` so this is more annoyance than blocker)
  - `set_intervention_transition_lock(bool)` to gate user toggles while the swap is mid-flight (servo sync, torque enable, etc.)

  Every leader teleop (`so_leader`, `bi_so107_leader`, the two highrate variants, `keyboard`, `gamepad`) re-implements the toggle plumbing — each spins up its own `pynput` SPACE listener, owns its own `_intervention_active` / `_intervention_transition_lock` / `_intervention_debounce_s` state, duplicates the `_try_toggle_intervention` decision function. There's no shared mixin, wrapper, or protocol. A new teleop (Quest VR landed 2026-05-31 with **zero** intervention surface) doesn't plug in — it can't be used as the human source for HVLA RLT, the record-loop's intervention path, or HIL gym, until five methods + one config field are added by hand.

  There's a second, deeper gap: **`teleop.get_action()` is assumed to return joint commands**. A Cartesian teleop (Quest VR) returns Cartesian EE deltas until a Cartesian-IK robot wires it up via `robot.attach_teleop(teleop)`. HVLA / record never call `attach_teleop`, so even if Quest VR sprouted the five intervention methods tomorrow, the joint dict the intervention path needs wouldn't materialise. Today the joint-emitting teleop is implicit in the leader's hardware; for any non-leader teleop the call site has to opt in.

  Proposed shape (design before code):
  - **Shared toggle abstraction** — extract the SPACE-listener + `_intervention_active` + debounce + transition-lock state into a small `InterventionState` class (or `InterventionMixin`) the teleop composes in. The **toggle source is pluggable**: keyboard SPACE for leaders, gamepad button for `gamepad`, **Quest controller face button B/Y at WebXR index 5** for VR (haptic-feedback path already exists to signal "intervention engaged" without the operator looking outside the headset).
  - **Joint-emission entry point** — HVLA / record / HIL-gym call `robot.attach_teleop(teleop)` after `teleop.connect()`. Robots that don't need it (leaders that already emit joints) keep today's no-op `attach_teleop`. Robots with Cartesian-IK paths (`BiSO107Follower`, `BiSO107FollowerPredictive`, `VirtualBiSO107Follower`) install the IK transform transparently. Same machinery `lerobot_teleoperate` already uses for the Cartesian path; intervention paths just have to call it.
  - **Torque ops are optional** — keep the `hasattr` guards in the call sites, but document explicitly that `enable_torque`/`disable_torque` are leader-only. Non-physical teleops (VR) implement neither.

  Until this lands, every new teleop pays the duplicated-plumbing tax, and the immediate Quest VR ask "use VR for HVLA intervention" requires per-teleop hacks rather than a clean opt-in.

  Open design questions before any code:
  - Mixin vs wrapper class — mixin is less invasive, wrapper composes more cleanly and survives multiple inheritance. Wrapper probably wins.
  - Where does the toggle binding live — config field (per-teleop, declared in the config dataclass) vs hardcoded in each teleop class? Config field is more flexible but inflates each config; per-class with a class-level default is probably the right shape.
  - Bimanual leaders attach the intervention state to one arm only (the left arm owns the SPACE listener). Quest VR is intrinsically bimanual — what does intervention mean when only one Quest controller is engaged? Likely: any-arm-engaged = intervention is active. Document the per-teleop semantics.
  - Bidirectional torque sync on leaders (servo to follower at intervention release) is leader-specific. For Quest VR the equivalent is "re-anchor the engage snapshot at intervention start" (which the existing reset-button path already does — same primitive). Both can be expressed as a single `on_intervention_transition(direction)` hook the teleop implements.

- [High] **Decouple data loaders from UI state**: today, three loaders gate data fetching on UI flags:
  - `model.js::loadModelSources` only scans sources where `s.expanded` is true
  - `app.js::loadSources` (datasets) — same shape
  - `robot.js`'s `robotTabInitialized` / `teleopTabInitialized` one-shot guards force consumers like `run.js:23-35` to inline-fetch profiles as a workaround

  Symptom: cross-tab consumers (e.g. Run -> Policy dropdown) get stuck on "Loading models..." when no Models-tab source is expanded. Each new consumer keeps inventing "ensure-loaded" workarounds because the data layer leaks UI state.

  Refactor: each owner module exposes an idempotent `ensureXxxLoaded()` that fetches and caches without consulting UI state. Tab init and other tabs both call the same function — the data layer never references `expanded` / `tabInitialized` flags. The `expanded` flag remains in `~/.config/lerobot/model_sources.json` purely as UI persistence (governs render shape, never data fetches).

  Sites:
  - `model.js`: introduce `ensureAllModelsLoaded()`. `modelTabInit` calls it. Drop the `expanded`-gated scan inside `loadModelSources`. `run.js::_ensureModelDataLoaded` becomes a one-liner.
  - `app.js`: introduce `ensureAllSourceDatasetsLoaded()`, same pattern.
  - `robot.js`: `ensureRobotProfilesLoaded()` / `ensureTeleopProfilesLoaded()`. Remove the inline fetches in `run.js:23-35`.

  Invariant after refactor: `s.expanded` references appear only in render/toggle code. Periodic check:

  ```bash
  grep -nE "if.*\.expanded" src/lerobot/gui/static/*.js | grep -vE "render|toggle|html"
  # Should match nothing.
  ```

## Bug Reporting

The "Report bug" button in the tab-bar (top right) stores a report **on the GUI server** under `~/.cache/lerobot/bug_reports/<timestamp>_<slug>/` with `report.json` + `screenshot.png`. Backend: `gui/api/bug_reports.py`. Frontend: `gui/static/bug_report.js`. Nothing is sent off the server.

- [Mid] **Upload to GitHub issue**: optional "Upload" button on a saved report. Use `gh issue create --repo <owner>/<repo> --title ... --body-file report.json --attach screenshot.png` (piggy-backs on `gh auth login` configured on the GUI server). If `gh` is missing or unauthenticated, the button is disabled with a tooltip. Add `GET /api/bug_reports/{id}/gh-status` to surface readiness, and `POST /api/bug_reports/{id}/upload-gh` to do the upload. Server-side save stays the default; upload is opt-in per report.
- [Low] **Vendor html2canvas**: today the client lazy-loads html2canvas from jsdelivr. Air-gapped / restricted-network setups need a vendored copy at `gui/static/vendor/html2canvas.min.js` with the loader preferring the local path. ~150 KB.
- [Low] **Recent reports panel**: small "View past reports" section that lists what `GET /api/bug_reports` returns. On localhost the server path is also the user's path, so a "copy path" affordance is enough; once hosted, the panel should offer "download report" (zip the directory) instead, since the server filesystem won't be the user's.
- [Low] **Camera/video tiles in screenshots**: html2canvas renders `<video>` and `<canvas>` elements blank or as poster frames. For reports about a live camera glitch this drops the most useful pixels. Workaround: also snapshot each visible `<video>` / `<canvas>` via its own `.captureStream()` / `getContext('2d').getImageData()` and stitch them into the report directory as separate PNGs.
- [Low] **Per-user scoping when hosted**: today every report lands in one shared directory keyed by timestamp+slug. Fine for a single-operator localhost setup; on a hosted deployment we'd want `<user>/<timestamp>_<slug>/` (or a small SQLite index) so triage can filter by submitter and one user can't see another's reports. Postpone until there's a real auth story.

## UX

- [Mid] **Repository skill for the GUI's visual and interaction language.** Add `.agents/skills/gui-style/SKILL.md` so agents working on the frontend first reuse the conventions already established in the GUI instead of inventing a new treatment per feature. The skill should point to canonical implementations and cover semantic color/severity (reserve alarming red for genuinely dangerous or destructive states), card overlays and empty/loading/disabled/error states, typography and spacing, button variants, accessible tooltip behavior, and static-asset cache busting. In particular, do not use `cursor: help` as an implicit question-mark tooltip affordance; follow the GUI's existing hover/focus/touch patterns and keep the full diagnostic accessible without making it the card's dominant visual. Require a before/after screenshot check for material UI changes. Keep detailed examples in the skill's `references/` directory so the core instructions stay concise and can evolve with the source UI rather than becoming a frozen parallel design system.
- [Mid] Cross-reference navigation: clickable links from dataset/model/robot references to their tab (generic utility, not one-off per instance)
- [Mid] **Demote the "AI setup" top-bar link into a settings/gear (⚙) menu.** Currently it sits next to Transfers and Report bug — both session-frequent — while AI setup is a per-device, one-off ceremony. Visual prominence mismatches usage. A small ⚙ icon opening a dropdown ("AI setup", and future admin links like token rotation / system info / log viewer / the WS-auth toggle if that audit followup lands) de-emphasizes one-offs without hiding them, and matches conventions users already know (VSCode, Slack, GitHub). No new tab required.
- [Mid] **Unify notification UX.** The GUI currently has two parallel notification surfaces: the existing toast bar (`showToast()` in app.js, top-right, used for "dataset opened", "save failed", etc.) and the browser's OS-level Notification API (used by the MCP bridge tool `notify_user` for backgrounded-tab pings). They look and feel different and don't share routing. Slack-style pattern is the target: in-tab events → toast; background-tab events → OS notification with the same content. A small notification dispatcher would let both `showToast` callers and `notify_user` share one surface and one escalation policy. Until this lands, `notify_user` keeps the OS-notification path (it's the only thing that refocuses a background tab) but isn't surfaced in the e2e demo because it's not visually integrated with the rest of the GUI.

## AI-native (MCP)

- [High] **Bridge WS subscriber auth.** Today `/api/bridge/ws` accepts any LAN connection — fine under LAN-trust + single-operator, but a wildcard subscriber can tail every AI command the operator's AI fires (every `navigate_to`, `notify_user`, `highlight_in_viewer`). Two design choices to make: (a) require a bearer on WS upgrade (and give GUI tabs a way to obtain one — currently they have none), or (b) default tabs to a per-tab UUID target so the wildcard subscription becomes an explicit operator action. Plan doc's "Open decisions" item 2 captures the choice. Audit caught this pre-merge and accepted the risk for the LAN-trust + single-operator posture this PR ships under.
- [High] **Surface MCP comments in the GUI.** The MCP `tag_episode` tool writes to a sidecar SQLite (`_mcp_annotations.sqlite`) that's keyed by `(repo_id, episode_id, key)`. Today only AI tools read them back via `get_episode_tags`; humans see nothing. A small Annotations panel in the Data tab (next to Inspector, or as an Inspector tab) would close decision #7 in the plan doc — "comments are eventually surfaced in the GUI too." Spec: per-episode list of `(key, value, set_at)` rows, clickable to filter / highlight, with a manual "Add comment" affordance so it's not AI-only. Plan doc captures the intent.
- [Mid] **Filter UI for the dataset tree → re-register `set_filter`.** The MCP `set_filter` tool was dropped from v1 because no GUI viewer has a real filter input. When the data sidebar grows a search/criteria field (likely with the Annotations panel above), re-register `set_filter` in `mcp/bridge_tools.py` and re-wire the `lerobot-bridge:filter` consumer in `bridge_consumers.js` (the listener was stubbed out, comment left as a marker). The wire shape (`"filter"` in `SUPPORTED_COMMAND_TYPES`) is already stable.
- [Mid] **Bridge tool surface — edit + operate tiers.** This PR ships the read + comment + bridge surface (~12 tools). The plan doc's "Tools (design surface)" table describes the full ~52-tool target. Next chunks, one PR per domain: Dataset edits (`propose_*` + `apply_pending_edits` mirroring the GUI's PendingEdit pipeline), Hub (auth-status / repo-info / `start_upload` / job-progress), Models (sources + debug-model load/unload), Robots (profile reads first, then writes, then motor ops), Run (status reads + `stop_current_run` first, then `start_*`). Each domain needs a shared-business-logic refactor between its FastAPI route and the MCP wrapper — don't auto-bind.
- [Low] **Surface `list_tagged_episodes` MCP tool.** The underlying storage method (`AnnotationStore.list_tagged_episodes`) already exists; only the MCP wrapper at `mcp/server.py` is missing. ~3 lines + a test. Bundle with whichever larger PR touches the dataset-read surface next.
- [Mid] **Add `get_feature_series` MCP tool.** Per-frame time series for a chosen feature (e.g. action / state / reward) across an episode. The GUI's `/api/datasets/{repo_id}/episodes/{episode_idx}/feature-series` route already does this; the MCP version needs the same business logic refactored out so both surfaces share it (NOT another HTTP self-call from MCP — that's the in-process anti-pattern). Estimate is larger (~30 lines + extraction) than the `list_tagged_episodes` wrapper above.
- [Low] **Retro transcript log for the original PR #20 tools.** The dataset-edit PR (#22) established the "transcript log alongside screenshots" standard documented in `mcp/README.md`. The PR #20 tools (list_datasets, get_dataset_info, list_episodes, get_episode_summary, get_frame, get_episode_tags, tag_episode, delete_episode_tag, lerobot_whoami, navigate_to, notify_user, highlight_in_viewer) have `demo_e2e.md` (happy-path 9-call demo) but no structured error-case transcript. Capture an e2e transcript analogous to `dataset_edit_transcript.md`, or fold the missing error coverage into the existing demo. Not blocking, but closes the standard-uniformity gap.
- [Low] **Consider splitting `navigate_to` into typed sub-tools.** Today `navigate_to(view: str, params: dict)` takes a view-dependent dict whose shape depends on `view`. Documented in the docstring but the AI has to read carefully. Alternatives: `navigate_to_episode(repo_id, episode_id)`, `navigate_to_dataset(repo_id)`, `navigate_to_tab(tab)`. Trade-off: more tools to maintain, but each has an unambiguous schema. Audit considered; left as-is for v1 — revisit if AI agents trip over the dict-params shape in real use.

## Agentic frontend observation

Goal: an AI agent on the operator's machine can both _drive_ the lerobot GUI (existing bridge MCP) and _observe_ it (screenshots, recordings, DOM state) via natural language, without per-task bespoke Python scripts. Pattern is Bash + skill markdown + small single-purpose CLIs — no new MCP server. The agent orchestrates; the CLIs each do one thing.

- [High] **`scripts/gui/capture_window.py` — capture the operator's open browser tab.** The blocking gap. Today `screenshot_gui.py` spawns its own Chrome at a fixed position; for interactive flows the operator already has the GUI open. New util: `find_lerobot_window()` returns `{x, y, w, h}` via `xdotool search --name lerobot` (Linux) / `osascript` (macOS) / `EnumWindows` (Windows), then `ffmpeg x11grab` against that geometry, write to `~/.lerobot/captures/<ts>.png`. ~80 lines. Without this nothing else in this section works.
- [High] **Skill: `.claude/skills/capture-lerobot-frontend.md`.** Tells the AI "to see the GUI, run `scripts/gui/capture_window.py` and Read the PNG it prints." Tiny markdown. Pairs with the script above; the skill IS the agentic orchestration layer (no MCP needed for synchronous capture — Bash + Read tool already cover it).
- [Mid] **`scripts/gui/record_window.py` — timed video capture.** Same `find_lerobot_window` underneath; ffmpeg x11grab with `-t <duration>` writes MP4 to `~/.lerobot/captures/<job_id>.mp4`. PID-file lifecycle for `--start` / `--stop <job_id>` so an agent can record across other tool calls.
- [Mid] **`scripts/gui/gui_compose.py` — post-process toolkit.** Side-by-side composer, crop-to-window, extract-frame-at-time. Each is a 5-line ffmpeg/Pillow wrapper, but canonical CLIs mean skill markdown reads "for before/after, call `gui_compose.py side_by_side a.png b.png out.png`" instead of teaching ffmpeg flags every conversation.
- [Mid] **Bridge symbolic-observation extension.** Adds `get_frontend_state()`, `get_console_logs()`, `query_dom(selector)` to the existing bridge — browser tab answers via injected JS over the WS. No new infrastructure; same channel that already carries `navigate_to`. Closes "did the bridge command actually render?" without needing pixel capture. Pairs with the bridge WS subscriber auth work above.

Explicitly out of scope (ruled out in the design discussion before merge of PR #20): bundling general screen-capture / clipboard / OS-input as MCP tools; shipping a browser extension; persistent screen-share permission via `getDisplayMedia`. The OS-level capture path (ffmpeg x11grab, screencapture, BitBlt) inherits the operator's existing user permissions and needs zero per-session consent on Linux X11, a one-time settings toggle on macOS.

## Dataset Tools

- [Mid] Consolidate `_keep_episodes_from_video_by_time` (time-based) with `_keep_episodes_from_video_with_av` (frame-based, upstream) in `dataset_tools.py`. Migrate trim callers to frame indices.
- [Mid] Consolidate streaming video encoders: our `video_encoder.py` vs upstream's `video_utils.py`. Upstream's is more mature (HW encoders, frame dropping). Consider migrating.
- [High] **Move per-frame stats work off the loop thread** in `StreamingVideoEncoder.push_frame`. Measured cost: ~4 ms p50 / 5 ms p95 per `dataset.add_frame()` call (4 cameras × 720p, synthetic random images on a fast machine) — see `/tmp/encoder_benchmark.py`. That's ~12% of a 30 Hz budget consumed by work that does NOT need to happen synchronously: `_reservoir_sample` (downsample copy + running per-channel mean/std/min/max accumulation) and HWC/CHW + dtype normalisation all run on the caller's thread inside `push_frame`. The actual encoding already runs in the background thread; the stats work is the leftover synchronous tax. Real cameras compress better than random noise, so production numbers may be lower, but the synchronous portion is the same. Risk surface: at higher control rates (60 Hz halves the budget to 16 ms) or on weaker CPUs, this 4 ms becomes a real fraction of the iteration time and can push the loop into overrun under load. Plan: (a) push the raw frame onto the queue first (cheap), (b) let the encoder thread do the reservoir sampling + running stats off-loop. The queue is already unbounded so backpressure isn't a concern; only the memory of in-flight frames grows during a backlog, which is acceptable. Sites: `src/lerobot/datasets/video_encoder.py::push_frame` and `::_reservoir_sample`.
- [High] **Duplicate detection within dataset**: detect near-duplicate episodes during dataset opening and before merging. Prevents wasted training compute on redundant data. Could use joint state trajectory similarity or image embedding distance.
- ~~**Subtask labeling in GUI**~~ — delivered by Feature Editing V1 ([docs/feature_editing.md](docs/feature_editing.md)): drag-select a frame range on the subtask row → type the label in the Inspector → Apply.
- ~~**Subtask format**~~ — superseded by upstream's v3.1 language columns; see [docs/stories/language_annotations.md](docs/stories/language_annotations.md).

## Overlays / Data Editing

- [ ] **Root-cause the camera-batching regression** (highest-value perf item). Batching
      is the only lever that reduces kernel launches per frame, and it measured ~1.3x —
      but it degraded tracking (an object held 199/199 frames serial fell to 176/199
      batched; another 89/199 -> 4/199 on merged_raw ep157). The long-assumed cause was
      "batched kernels diverge numerically"; that is now DISPROVEN — `trk.get_image_features`
      is bit-identical batched vs serial (max |diff| exactly 0.0, for two cameras stacked
      and for a duplicated frame). Prime suspect is the prime-cache seeding: which frame
      index each session's pre-computed features get filed under. If that is an indexing
      bug, batching becomes a clean ~1.3x with no quality cost.
- [ ] **Try CUDA graphs** — `torch.compile(mode="reduce-overhead")` on the vision encoders.
      Profiling says the workload is LAUNCH-BOUND, not compute-bound: ~1,233 kernel
      launches and 4 blocking `cudaStreamSynchronize` (8.17 ms) per frame, ~15-20 ms of
      actual kernel time inside a 26.5 ms call. Graph capture replays that launch sequence
      as one submission, which is the textbook fix. NOTE: default-mode torch.compile was
      already tested and rejected (1.12x for a 24 s compile) — it does not capture graphs,
      so that result says nothing about this one. Capture may fail: graphs need static
      shapes and no syncs inside the captured region, and this model has four.
- [ ] Remove the 4 per-frame syncs if possible — deferring mask->CPU by one frame would
      stop draining the pipeline every frame. Cheaper than graph capture, same target.

- [ ] (LOW PRIORITY — only pays off at `variants` > 1, and the practical workflow is
      `variants=1`: extra variants cost disk for near-duplicate data, and randomizing
      across a decently sized dataset at episode level gives the same diversity.)
      Hoist segmentation out of the `variants` loop. Measured (pick_ball ep0, 241
      frames, 2 cams, sam3_track@672): segmentation is **77% of a job's wall time**
      (15.9 s of 20.7 s), and `variants` is the OUTERMOST loop — so N randomized
      copies re-segment every frame N times for identical masks (masks depend on the
      frames and the object list, not on the random draws). Cache the episode's masks
      (bit-packed or RLE to bound RAM) and composite N times: extra variants become
      nearly free instead of costing N x the GPU work.
- [ ] Overlap the CPU and GPU stages of a processing job. The loop is strictly serial
      per frame — decode (CPU) -> segment (GPU) -> composite (CPU) -> buffer, then
      encode per episode (CPU) — so the GPU idles through ~22% of wall time
      (composite 13%, decode 5%, encode 4%) and measured utilization is mean 65%,
      max 79% (never higher, even mid-segmentation; note nvidia-smi "utilization"
      only means a kernel was running, so true compute efficiency is lower).
      Prefetch decode in a thread and composite frame N-1 while the GPU segments
      frame N: ~1.25x, no quality risk.
- [x] ~~torch.compile the detector~~ — MEASURED AND REJECTED (2026-08-04). The old
      "1.34x" was a micro-benchmark of the detector forward alone; it does not
      translate, because `sam3_track` throttles detection and the per-frame cost is
      the TRACKER. Compiling both 454M-param vision encoders end to end: **1.12x**
      steady state (33.4 -> 29.7 ms/frame), **24.4 s** one-time compile, so
      **break-even at ~6,600 frames** (~26 episodes). On a 274k-frame job that is a
      ~4% saving; on anything smaller it is a net loss, and it would add 24 s before
      the live preview's first mask. It also perturbs numerics (mean IoU 0.9998 vs
      uncompiled, only 19/80 masks identical) — the same class of drift that made
      camera batching collapse holds on a real dataset. Not worth 4%.
- [ ] Encode WYSIWYG composite overlays as JPEG/WebP instead of PNG. Composites are
      full-frame photo-like images, so PNG is the pathological codec (~300-800 KB each);
      2 cams x ~10/s saturates a Wi-Fi link. The completion-gated loader keeps remote
      tiles updating under that pressure, but a 5-10x smaller encoding would restore
      near-full overlay refresh remotely. Keep PNG for contour/chrome overlays (alpha).
- [ ] Random-texture bank: GreenAug's best background randomization is random REAL
      texture images (87% success vs 66% Perlin / 65% solid; arXiv:2407.07868 Table 4)
      — a local bank of high-entropy texture photos sampled per episode, offered next
      to the per-pixel static default (which follows the paper's entropy trend but is
      untested there and gets low-passed by video codecs on commit).
- [ ] Overlap resolution upgrade: per-pixel mask-logit argmax when object masks
      dispute a pixel, replacing the smallest-mask-wins containment heuristic
      (effects.build_and_sample_regions). Needs per-concept logits plumbed through the
      adapter contract. Prior art: confidence-sorted greedy merging (Kirillov et al.
      arXiv:1801.00868), logit argmax (EfficientPS arXiv:2004.02307), learned occlusion
      order (Lazarow et al. CVPR 2020). Only worth it if tint seams at true occlusion
      boundaries (e.g. fingers wrapping a held object) become visible in practice.

## HVLA / Policy Evaluation

- [**Critical / Safety**] **State-to-first-action gap at inference start.** At t=0 of a policy rollout, the robot sits at some state `s₀` and the policy emits `chunk[0..H]`. If `chunk[0]` is far from `s₀` — e.g. the policy was conditioned on an observation where the gripper was already in pre-grasp pose, but the operator left the arm in a random rest position — the motor goal is set to a position several joints' degrees away from where the arm currently is, and the motor moves at full available speed to close the gap. **This is a real safety hazard**: the arm can crash into the workspace, the operator, or other arms before anyone reacts. Fix shapes, roughly cheap → safe:
  1. **Pre-position step before policy runs.** Reuse the existing `move_to_rest_position` flow / safe-trajectory ramp: drive the robot from `s₀` to `chunk[0]` over `ramp_duration_s` (default 2-3 s, scale with joint-space distance). Only then engage the policy loop. Mirrors what `lerobot_record.py` already does for the auto-reset phase between episodes — same code path, just driven by the first policy action instead of `start_pose`. Smallest blast radius.
  2. **Velocity-limited safe-step controller in `send_action`.** Cap the per-step `Goal_Position` delta to `max_velocity_deg_per_step` regardless of what the policy asks for. The motor can't be commanded to teleport. Protects against bad policy outputs across the entire episode, not just the first frame.
  3. **Pre-flight check.** Refuse to start the rollout if `||chunk[0] − s₀||₂ > safety_threshold`. Show the operator a "robot is too far from policy's expected starting pose; reset arm first" message. Belongs in the GUI before `/api/run/start_hvla` flips to "running".
     Highest priority: ship #1 (reuses existing infrastructure, immediate). Track #2 as the durable safety net since it protects against more than just startup. #3 is the user-facing complement. Design tracked separately before any code lands.
- [High] **Train S1 with intervention data**: merge `thewisp/intervention_cylinder_ring_assembly` corrections into training dataset and retrain S1 to learn from human corrections (DAgger-style).
- [Mid] **Optional success dataset**: when running HVLA with intervention, optionally save episodes where the user advanced without ever intervening to a separate "success" dataset. Requires proper `clear_episode_buffer` lifecycle and video encoder management — refactor `s1_process.py` recording logic first.
- [Mid] **Refactor s1_process.py recording**: the episode lifecycle (dataset creation, encoder start/stop/discard, intervention buffer management) has grown organically. Align with `lerobot_record.py`'s structure before adding more features.

## Testing

- [ ] API datasets tests (`tests/gui/test_api_datasets.py`) — TestClient + mock dataset fixtures
- [ ] API edits tests (`tests/gui/test_api_edits.py`)
- [Low] Playback WebSocket tests (`tests/gui/test_api_playback.py`)

## Hardware

- [Low] Use stable Linux device paths (`/dev/serial/by-id/`, `/dev/v4l/by-id/`, `/dev/v4l/by-path/`) instead of volatile `/dev/ttyACM*` and `/dev/video*` in profiles. GUI Robot tab could auto-detect and offer these in port selection.

## Python 3.12+ Compatibility

- [Mid] Remove Python < 3.12 workarounds once we drop 3.10/3.11 support. Upstream lerobot now requires 3.12+. Our fork pins 3.10 compatibility via:
  - `datasets/utils.py`: `class Backtrackable(Generic[T])` -> native `class Backtrackable[T]:`
  - `motors/motors_bus.py`: `NameOrID = Union[str, int]` -> native `type NameOrID = str | int`
  - `utils/io_utils.py`: module-level `T = TypeVar(...)` -> native `def foo[T: Bound](...)`
  - `processor/pipeline.py`: `Generic[TInput, TOutput]` -> native type params
  - `policies/pretrained.py`: conditional `Unpack` import -> direct `from typing import Unpack`
  - Multiple modeling files: `from typing_extensions import Unpack` -> `from typing import Unpack`

## Workflow

To work on this TODO autonomously:

1. Read the TODO.md
2. Find the next incomplete task and implement it
3. Commit your changes
4. Update TODO with what you did
   ONLY DO ONE TASK AT A TIME.
