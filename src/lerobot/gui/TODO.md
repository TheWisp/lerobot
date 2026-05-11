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
- [ ] Undo/redo — explicitly punted from Feature Editing V1 (per-chip removal in edits-bar covers most "oops" cases). Real undo across Saves needs pre-edit value capture or a Git-like history.

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
- [Mid] **Compact per-episode-feature storage** (tech debt): per-episode features are currently marked via `info.json` but stored by replicating the same value across every frame in `data/*.parquet` — wasteful for what is logically one value per episode. Find a more compact representation (e.g. dedicated columns in `meta/episodes/*.parquet`, or sparse encoding in `data/*.parquet`) while staying backward-compatible with readers that expect the per-frame layout.
- [Mid] **Remove implicit "episode-wide if uniform" detection** (tech debt, depends on the compact storage above): today a feature is recognized as episode-wide if all its frame values within an episode are identical. That heuristic is fragile (one stray edit makes it look per-frame again) and only exists because we lacked an explicit format. Once compact storage is in place, drop the heuristic in favor of the explicit `episode_features` declaration.
- [ ] Episode-list shortcuts: tri-state `success` checkbox column with bulk-set; inline-editable `task` per episode
- [ ] First-class `subtask_index` writer at recording time (`add_frame(subtask=...)` analog to `add_frame(task=...)`); today only the read hook + offline annotation Space exist

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

- [ ] `GET /api/hub/auth-status` — check login state
- [ ] `POST /api/hub/login` — store HF token
- [ ] `POST /api/datasets/{id}/hub/download` — pull from Hub (overwrites local, with confirmation)
- [ ] `POST /api/datasets/{id}/hub/upload` — push to Hub (overwrites remote, with confirmation)
- [ ] Frontend: auth indicator in header, download/upload buttons per dataset
- [High] **Hub progress bar (uploads + downloads)**: every `snapshot_download` / `upload_folder` call writes a tqdm bar to the server stderr that the GUI never sees. The Hub modal (upload / download / open-sync) just shows a static "Uploading…" / "Downloading…" status while the request blocks for minutes. Wire real progress for all Hub transfers: hook `huggingface_hub.utils.tqdm` into a per-request callback that pushes byte counts + current filename into a shared progress dict, expose via SSE or a poll endpoint, and render a progress bar in the modal status area. Affects all three Hub modal modes; most visible for the open-sync flow because users see it before they have any sense of dataset size.

## Model Tab

See [docs/model_tab.md](docs/model_tab.md) for full design.

- [ ] Phase 1: Browse & Inspect — source scanning, list/info/config endpoints, source tree, detail panel
- [ ] Phase 2: Training — subprocess launch/stop/status, training form, terminal output, resume
- [ ] Phase 3: Run Tab Integration — "Use in Run tab" passes checkpoint path to policy workflow
- [ ] Phase 4: Metrics & Polish — WandB embed, training curves, model comparison, HF Hub model sync

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

- [Low] Text output freezes after a while — teleoperate uses ANSI cursor-up in piped stdout
- [Low] Rerun web viewer has ~200ms visual lag (Rerun 0.26 limitation)
- [Low] Replay FPS setting doesn't seem to affect playback speed — remove if not useful
- [Low] **SIGKILL during HVLA shutdown leaks semaphores**. When stopping S1 inference (or any HVLA run) via the GUI's stop button, the 5s SIGTERM→SIGKILL grace window in `gui/api/run.py` sometimes isn't enough — visible as `resource_tracker: There appear to be N leaked semaphore objects` and `Process exited with code -9`. Cosmetic only (kernel reclaims on process death) but noisy. **Don't extend the timeout blindly** — 15s feels like an eternity from the GUI when the user wants to relaunch. First instrument the shutdown: log per-phase elapsed times for soft-land, teleop disconnect, robot disconnect (+ per-camera if accessible), shm cleanup. Get measured numbers from a few runs, then decide whether the fix is "kill the slow phase", "parallelize cameras", "release shm first", or just "the timeout is fine, the warning is harmless".
- [High] **RLT metrics pipeline is wasteful end-to-end**. Two compounding issues: (1) Training subprocess rewrites the entire `metrics.json` (~150KB for 5000 points × 8 series) on every save, even when only a few new points were appended — atomic write via `.tmp` + `os.replace` on each episode end and every 100 inference steps. (2) GUI polls `/api/run/rlt-metrics` every 2s and gets the same full snapshot back, of which ~99.6% is unchanged from the previous poll. Fine on localhost, wasteful otherwise. Options: append-only JSONL or shared memory on the write side; SSE push of only new points or cursor-based polling (`?since_step=N`) on the read side. Frontend maintains a local buffer.
- [Mid] **RLT dashboard chart smoothing**. Per-inference series like `actor_deltas` have per-sample noise comparable to the actual trend (e.g. δ raw σ ≈ 0.015 per sample vs a real β-driven shift of 0.018 — z=17.8 over 500 samples but invisible in any single-sample view). The dashboard plots raw values, so genuine learning signals get hidden. Add a smoothing toggle / moving-average overlay (window picker: raw / 50 / 200 / 500), or always show both the raw line (light) and a smoothed line (bold). Same applies to `q_values_*`, `critic_losses`, `actor_q_term`, `actor_bc_term` — all have per-grad-step noise. Cheap if the smoothing happens in JS over the buffered series the dashboard already pulls.
- [Mid] **Backend-driven Launch validation schema**. The Launch button's required-field rules currently live in JS-side `_WORKFLOW_VALIDATORS` and `_POLICY_VALIDATORS` registries in `gui/static/run.js`. Each policy_type's Pydantic/Draccus config already declares its required fields in Python — the JS registry has to be updated by hand whenever those declarations change, and silent drift between the two is invisible until a user clicks Launch and gets a backend 400. Replace with a `GET /api/policy-schemas/{policy_type}` (and similar for workflows) returning e.g. `{required: ["task"], required_when: {"rlt_token_checkpoint": "rlt_mode"}}`; the frontend consumes that verbatim, so adding a new policy means only one Python edit. Until then, any change to a policy's required fields must be reflected in both places.
- [Mid] **Audit `torch.load()` for `weights_only=True`** (bandit B614, currently in global skips). Call sites: `src/lerobot/policies/act_vlm/modeling_act_vlm.py`, `src/lerobot/policies/hvla/s1/flow_matching/model.py`, `src/lerobot/policies/hvla/s1_process.py`. Since PyTorch 2.6 the default flipped to `weights_only=True`; our checkpoints predate that and contain non-tensor pickled metadata, so wholesale flip would break loaders. Plan: per-site, switch to `weights_only=True` and migrate any non-tensor state to a sidecar JSON / safetensors. Remove B614 from `pyproject.toml` `[tool.bandit].skips` once the audit completes.

### Recording Loop Performance

Baseline from a 2-episode bi_so107_follower record run with trajectory_replay, 4 cameras at 1280×720 @ 30 fps (2026-05-11, snapshot in `outputs/record/latency_snapshot.json`): loop p50=24.9 ms / p95=28.2 ms / p99=30.8 ms, overrun 0.55%. Healthy at 30 fps but the budget breakdown surfaces three places worth attention:

- [High] **`dataset_write` p95 is encoder backpressure, not random burst.** Stage measures p50=4.5 ms / p95=12.3 ms (2.7× ratio). Root cause: [datasets/video_utils.py:828](../../datasets/video_utils.py) — streaming encoder's `feed_frame()` does `queue.put(..., timeout=0.1)`. When libsvtav1 can't keep pace with 4 cameras at 30 fps (default preset 12 mapped to 10, fairly heavy), the queue (default maxsize 30 ≈ 1 s) fills and the control thread blocks waiting for a slot. Cheap mitigations: (a) bump `encoder_queue_maxsize`; (b) `--dataset.vcodec=auto` to pick hardware encoder (h264_nvenc); (c) switch to the per-camera `OurStreamingVideoEncoder` path at [datasets/dataset_writer.py:278-280](../../datasets/dataset_writer.py) which uses non-blocking `push_frame()`.
- [Mid] **`process_obs` is dominated by DepthEdgeOverlayProcessorStep** (~7 ms p50 on a single RealSense depth stream). Defined in [robots/bi_so107_follower/bi_so107_follower.py:225-247](../../robots/bi_so107_follower/bi_so107_follower.py) — Sobel gradients + percentile threshold + dilate + applyColorMap + blend on 1280×720. Make it opt-out via robot config when the policy isn't trained on the overlay; instant ~7 ms reclaim. Worth a config flag like `depth_edge_overlay: bool = True` so it's explicit which runs need it.
- [Mid] **Unexplained ~8-10 ms remaining in `process_obs`.** After accounting for the depth-edge step (~7 ms) and obs-stream writer (~0.84 ms for 4 cameras × `tobytes()` + shm copy), the measured 17.7 ms p50 leaves ~8-10 ms unaccounted for. Likely Python/cv2/malloc per-frame overhead, but worth confirming with finer-grained spans before optimizing. Add a few `latency_session.span("...")` blocks inside the processor-pipeline loop to attribute the gap to specific steps (or to inter-step bookkeeping).

### GUI Resource & Async Hygiene

- [High] **GUI server transiently holds `/dev/video*` FDs and blocks CLI teleop — root cause found.** [gui/api/robot.py:418-431, 456-468](../../gui/api/robot.py): `_detect_and_open_cameras()` calls `camera.connect(warmup=True)` then appends to `_preview_cameras`. The surrounding `except` clause only logs — if any exception fires after `connect()` succeeds but before the append (e.g. an unrelated error mid-iteration), the camera handle is abandoned and `/dev/videoN` stays open. Reproduced 2026-05-11: `fuser /dev/video10` named the GUI server PID directly. Fix shape: tight try/finally around each per-camera section so an exception triggers `disconnect()` before bubbling up. Plus a defensive `_close_preview_cameras()` call in any catch-all error path.
- [Mid] **Blocking `subprocess.Popen` in async FastAPI handler.** [gui/api/robot.py:667](../../gui/api/robot.py) — `open_in_file_manager` runs `subprocess.Popen(["xdg-open", ...])` directly from `async def`. If `xdg-open` hangs (slow systems, locked-up desktop env), the FastAPI event loop stalls and all concurrent HTTP requests block. Wrap in `loop.run_in_executor(...)` — small surgery, preventive (no observed hang yet).

### Trajectory-Replay Follow-Ups

- [Critical] **`trajectory_replay` special-casing leaks into core scripts.** The current implementation works but violates the Teleoperator abstraction: `lerobot_teleoperate.py` and `lerobot_record.py` both contain code paths that exist only to support file-backed teleops. Specifically:
  - `teleop_loop` (teleoperate) checks `getattr(teleop, "is_exhausted", False)` to break.
  - `record_loop` does the same check.
  - `run_reset_phase()` has a whole branch keyed on `hasattr(teleop, "start_pose")` that skips `record_loop`, calls `move_to_rest_position(robot, teleop.start_pose, …)`, and reconnects the teleop.
  - The initial reset is forced for episodic teleops regardless of `--start_with_reset`, and the final-episode reset is no longer skipped for them.
    The original vision was a drop-in: register the new teleop type, point `--teleop.type=trajectory_replay` at it, and have the existing scripts treat it exactly like a serial-backed leader. We're not there. Options to evaluate, roughly cheap → expensive:
  1. **Extend the `Teleoperator` base** with optional lifecycle hooks (no-op defaults): `is_exhausted`, `on_episode_end(robot)` (drives the robot to a known pose + re-arms). Scripts always call them — no `hasattr` dances, no per-type branches. Smallest blast radius, makes the contract explicit. Still couples episode semantics to the Teleoperator class.
  2. **Introduce an `EpisodeController` abstraction** separate from `Teleoperator`. Today's behavior becomes a `DurationEpisodeController`; trajectory replay uses a `TrajectoryEpisodeController`. Clean separation of "where actions come from" vs. "when episodes end and how to reset." A new framework concept — more upfront design.
  3. **Run trajectory replay as a real subprocess that speaks the existing leader protocol** (shared memory or local socket exposing the same API as a serial leader). Scripts see "just another leader" — truly process-agnostic. IPC complexity, harder to debug.
  4. **Loop the trajectory automatically and trust `--episode_time_s` / `--reset_time_s` to govern boundaries.** Zero script changes; the user picks `episode_time_s == trajectory_duration` for clean episode boundaries. Loses exact frame-level alignment and programmatic reset.
     Option 1 is the right "fix what we have" path; option 2 is the right "build for the future" path. Discuss before refactoring.
- [Mid] **Multiple trajectories per run.** Today `--teleop.trajectory_path` takes one file and every episode replays it identically. Useful next step: support a list / directory and rotate through trajectories (round-robin or random) so a multi-episode dataset covers a variety of motions automatically. Likely shape: change `trajectory_path` to accept a directory (rotates) or a JSON-list-of-paths (explicit order). Tracked separately because the right rotation semantics depend on the dataset-iteration flow the user is designing.
- [Mid] **Configurable auto-reset duration.** Currently hard-coded to `_auto_reset_duration_s = 3.0` in `lerobot_record.py`. Promote to a CLI flag (e.g. `--auto_reset_duration_s=3.0`) once a use case forces it.

### Run Log Files & Crash Visibility

- [High] **Promote teleop's log-file + excepthook helper into a reusable utility.** `lerobot-teleoperate` now writes `outputs/teleop/teleop_<ts>.log` and routes uncaught exceptions (main thread + background threads) through `logging.error`. The other CLI scripts have NO per-run log file and NO excepthooks — `init_logging()` is called bare, so any crash goes to stderr only with no post-mortem. Promote `_install_exception_logging()` + the log-path resolution from `lerobot_teleoperate.py` into `lerobot.utils.utils` as a small `setup_run_logging(output_dir, run_name)` helper. Then apply to the scripts in priority order:
  1. **lerobot-record** — highest impact: hardware crash mid-recording today produces no log, and the beep `subprocess.Popen` calls at [scripts/lerobot_record.py:709, 730](../../scripts/lerobot_record.py) have no cleanup if the script crashes mid-episode (process leaks until the parent exits).
  2. **lerobot-train** — multi-process accelerate; worker failures go to stderr only. Add per-run log file in the training output dir; consider per-process suffixes for distributed debugging.
  3. **lerobot-eval** — daemon prefetch thread at [scripts/lerobot_eval.py:784](../../scripts/lerobot_eval.py) isn't `join()`-ed on the success path; could leave a GPU prefetch process orphaned. Fix in the same pass.
  4. **lerobot-replay** — minimal; just the log file + excepthook (no thread issues).
  5. **lerobot-calibrate**, **find-cameras**, **find-port** — lower urgency; manual one-off tools where the user is in front of the console anyway.

### Latency Panel UX

- [High] **Duplicate track rendering**: `LATENCY_SOURCES` has both `teleop` and `record` keys pointing at the same `outputs/teleop` directory, so when teleop runs, `/api/run/latency-sources` reports both as fresh (they both find the same snapshot file) and the dashboard renders the panel twice. Fix: give record its own output dir (e.g. `outputs/record`) and update the GUI's `--latency_output_dir` for start_record + the LATENCY_SOURCES registry to match. Alternatively, dedupe by snapshot path in `list_latency_sources`.
- [High] **Residual teleop jitter at fast speed**: the 1 Hz writer-thread refactor cut the loop-side spike from 22 ms to ~1 ms, but the user still feels small jitter under fast movement. Suspected GIL contention: the background writer thread holds the GIL during snapshot computation (transient aggregator construction, dict building, JSON dump), and the loop thread waits for the GIL on each iteration that overlaps the writer's work. To verify: time the loop thread's iteration during the second the writer runs vs during a quiet second; difference > 0 confirms GIL hit. Mitigations to try in order: (a) reduce writer-thread compute (skip `iterations` / `aggregate_iteration` when no GUI is polling? compute fewer percentiles?), (b) drop snapshot rate to 0.5 Hz, (c) move the writer to a subprocess (`multiprocessing.Process`) which doesn't share the GIL — heavyweight but the only way to truly isolate Python work from the loop.
- [Mid] **Gantt bar height ~1.5x**: per-iteration timeline bars are too short to hold the label text without clipping. Bump the row height (`rowH` calculation in `_drawGantt`) by ~50%.
- [Mid] **X-axis label overlap at edges**: extent labels at the corners (`{minMs} ms` / `{maxMs} ms`) overlap with the nearest tick when the tick happens to land near the corner. Either always-skip ticks within ~30 px of the corners, or always-render the corner labels and skip the conflicting ticks.
- [Mid] **Top-row metrics: only loop + overrun**: current cards (loop, get_observation, action_send, overrun) mix one umbrella stat with two partial stages and one rate — confusing because (a) loop already includes get_observation + action_send + other unshown stages like process_obs, (b) overrun is conceptually a different category. Replace with just `loop` and `overrun` as the headline row; move per-stage breakdowns to the Gantt + a separate compact stage table if useful.
- [Mid] **camera_read_strategy as dropdown in robot config UI**: today it's an open text field. Two valid values (`latest`, `wait_for_new`) — render as a `<select>` so users don't typo and silently get fallback behaviour. Same pattern as other enum-ish robot fields. May require a small schema-driven render hint (e.g. metadata in the dataclass field) so this generalises beyond bi_so107.
- [Discussion] **Color thresholds**: live panel shows 21.1 ms as yellow at a 30 Hz target (33.3 ms budget). Current rule fires yellow at ≥ 70% of budget; that's 23.3 ms, so 21 ms shouldn't be yellow — likely the comparison is against p95 (which IS > 23.3 if p50 is 21). Worth reviewing whether yellow should fire on p50 or p95, and whether the 70% threshold is right.

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

## Robot Tab

- [Low] UX consistency pass: ensure consistent button coloring/hierarchy across views
- [Low] ~1s latency when first opening the Robot tab while loading profiles
- [ ] Guided calibration wizard (adapt `robot.calibrate()` for GUI-driven step-by-step flow)
- [ ] **Surface `/api/robot/recover` from runtime error states (deeper integration).** A dedicated "Recover" button is shipped next to the rest-position buttons (`renderRestPositionSection` in `robot.js`), and `recoverRobot()` chains into `/move-to-rest-position` when the report is clean and a rest position is on file (with doubled `duration_s` if anything was recovered). What's missing: rendering an inline "Try recovery" link next to the error toast when teleop / record / replay / move-to-rest fails because of a wedged motor chain (e.g. `FeetechMotorsBus motor check failed ... Missing motor IDs`). Today these errors are printed-only (`logger.exception` + raw string in the response body), so users have to read the server log to know what's wrong, and there's no path from "I see an error" to "click to attempt recovery". Prereq: consolidate error handling — promote the relevant `RuntimeError` / `ConnectionError` from the bus and robot layers into a small set of typed exceptions (e.g. `MotorChainWedged`, `PortBusy`, `MissingMotor`) that the FastAPI layer maps to structured error bodies (`{"error_code": "...", "detail": "...", "remediation_hint": "recover"}`). The frontend can then render the appropriate hint generically instead of string-matching error messages.

## Architecture

- [**Critical**] **Reactive UI state management**: the current imperative DOM manipulation (innerHTML + manual toggle/refresh calls scattered across functions) is fundamentally broken. Field state (disabled, visible, selected) must be called at every possible code path that reveals the element, leading to endless monkey-patching. Migrate to a reactive pattern where UI state is derived from data (React, Preact, or even a minimal reactive store + render loop). This blocks every new UI feature.
- [Mid] Cross-tab data synchronization — replace point-to-point refresh calls with pub/sub event bus
- [High] Extract frontend to separate files (then optionally migrate to React/Vite)
- [Mid] FastAPI dependency injection for AppState — replace global `_app_state` with `Depends(get_app_state)`
- [Mid] Consolidate module-level caches (`_episode_start_indices`, `_dataset_info_mtime`) into AppState
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

## UX

- [Mid] Cross-reference navigation: clickable links from dataset/model/robot references to their tab (generic utility, not one-off per instance)

## Dataset Tools

- [Mid] Consolidate `_keep_episodes_from_video_by_time` (time-based) with `_keep_episodes_from_video_with_av` (frame-based, upstream) in `dataset_tools.py`. Migrate trim callers to frame indices.
- [Mid] Consolidate streaming video encoders: our `video_encoder.py` vs upstream's `video_utils.py`. Upstream's is more mature (HW encoders, frame dropping). Consider migrating.
- [High] **Move per-frame stats work off the loop thread** in `StreamingVideoEncoder.push_frame`. Measured cost: ~4 ms p50 / 5 ms p95 per `dataset.add_frame()` call (4 cameras × 720p, synthetic random images on a fast machine) — see `/tmp/encoder_benchmark.py`. That's ~12% of a 30 Hz budget consumed by work that does NOT need to happen synchronously: `_reservoir_sample` (downsample copy + running per-channel mean/std/min/max accumulation) and HWC/CHW + dtype normalisation all run on the caller's thread inside `push_frame`. The actual encoding already runs in the background thread; the stats work is the leftover synchronous tax. Real cameras compress better than random noise, so production numbers may be lower, but the synchronous portion is the same. Risk surface: at higher control rates (60 Hz halves the budget to 16 ms) or on weaker CPUs, this 4 ms becomes a real fraction of the iteration time and can push the loop into overrun under load. Plan: (a) push the raw frame onto the queue first (cheap), (b) let the encoder thread do the reservoir sampling + running stats off-loop. The queue is already unbounded so backpressure isn't a concern; only the memory of in-flight frames grows during a backlog, which is acceptable. Sites: `src/lerobot/datasets/video_encoder.py::push_frame` and `::_reservoir_sample`.
- [High] **Duplicate detection within dataset**: detect near-duplicate episodes during dataset opening and before merging. Prevents wasted training compute on redundant data. Could use joint state trajectory similarity or image embedding distance.
- [Mid] **Subtask labeling in GUI** — **superseded by [docs/feature_editing.md](docs/feature_editing.md)**. V1 of Feature Editing delivers exactly this: drag-select a frame range on the subtask row → type the label in the Inspector → Apply.
- [Mid] **Subtask format**: conform subtask column to LeRobot 3.0 format + OpenPI changes. Currently uses raw string column; may need task_index remapping.

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
