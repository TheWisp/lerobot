# GUI TODO

## Data Tab

- [High] **Warning/error panel**: dataset verification errors and warnings are currently buried in server log text. Add a visible warning panel (banner or sidebar) that surfaces verification results when a dataset is opened — errors as red, warnings as yellow. Users must not miss data integrity issues.
- [Mid] **Open local dataset by path** — partial. Folder names with spaces / non-alphanumerics work for complete caches (handled by the synthesized "<parent>/<name>" repo_id + local-first metadata load). Incomplete caches with such names previously surfaced a confusing `"Repo id must use alphanumeric chars…"` from `huggingface_hub`; the pre-check in `_check_local_dataset_complete` now probes required meta files (`tasks.parquet`, `episodes/`) directly and surfaces the real diagnosis (fixed 2026-05-12). Remaining ask: a true `local_only=True` flag on `LeRobotDataset` that disables every Hub fallback (e.g. for version-tag resolution on incompatible-version datasets) — only worth the lift when a concrete failure mode forces it.
- [ ] Parquet data display (action/state charts in timeline) — superseded by Feature Editing (see below); action/state co-display alongside cameras tracked as a follow-up there
- [Mid] **Opening a new dataset from sources doesn't switch to it**: after using "Open" on a source folder to load a new dataset, the GUI stays on whatever dataset was previously selected instead of focusing the just-opened one. Should auto-select the new dataset (and likely make it the active tab in the dataset list).
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

### Latency Panel

- [Mid] **Fixed panel height not adaptive to content**: panel renders taller than the default panel height on at least one setup, leaving empty space (or clipping). Make the height fit its actual content instead of a hard-coded value.
- [Mid] **Timeline span is dominated by max**: a single extreme sample stretches the y-axis so the median band collapses into a narrow stripe in the middle, defeating the point of the view. Options to explore: clip to a percentile (e.g. p95 / p99) with an out-of-range indicator, or switch to a log scale, or split into separate "typical" and "tail" views.
- [Mid] **Distorted / unreadable labels in loop topline**: the current-value text on the right side renders very narrow / squished; the bottom-left label is also distorted and unreadable on top of the colored background. Investigate the CSS/SVG sizing (likely `transform: scale`, `text-anchor`, or `width` overconstraint) and fix the contrast on the bottom-left label.

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

## Bug Reporting

The "Report bug" button in the tab-bar (top right) stores a report **on the GUI server** under `~/.cache/lerobot/bug_reports/<timestamp>_<slug>/` with `report.json` + `screenshot.png`. Backend: `gui/api/bug_reports.py`. Frontend: `gui/static/bug_report.js`. Nothing is sent off the server.

- [Mid] **Upload to GitHub issue**: optional "Upload" button on a saved report. Use `gh issue create --repo <owner>/<repo> --title ... --body-file report.json --attach screenshot.png` (piggy-backs on `gh auth login` configured on the GUI server). If `gh` is missing or unauthenticated, the button is disabled with a tooltip. Add `GET /api/bug_reports/{id}/gh-status` to surface readiness, and `POST /api/bug_reports/{id}/upload-gh` to do the upload. Server-side save stays the default; upload is opt-in per report.
- [Low] **Vendor html2canvas**: today the client lazy-loads html2canvas from jsdelivr. Air-gapped / restricted-network setups need a vendored copy at `gui/static/vendor/html2canvas.min.js` with the loader preferring the local path. ~150 KB.
- [Low] **Recent reports panel**: small "View past reports" section that lists what `GET /api/bug_reports` returns. On localhost the server path is also the user's path, so a "copy path" affordance is enough; once hosted, the panel should offer "download report" (zip the directory) instead, since the server filesystem won't be the user's.
- [Low] **Camera/video tiles in screenshots**: html2canvas renders `<video>` and `<canvas>` elements blank or as poster frames. For reports about a live camera glitch this drops the most useful pixels. Workaround: also snapshot each visible `<video>` / `<canvas>` via its own `.captureStream()` / `getContext('2d').getImageData()` and stitch them into the report directory as separate PNGs.
- [Low] **Per-user scoping when hosted**: today every report lands in one shared directory keyed by timestamp+slug. Fine for a single-operator localhost setup; on a hosted deployment we'd want `<user>/<timestamp>_<slug>/` (or a small SQLite index) so triage can filter by submitter and one user can't see another's reports. Postpone until there's a real auth story.

## UX

- [Mid] Cross-reference navigation: clickable links from dataset/model/robot references to their tab (generic utility, not one-off per instance)
- [High] **Dialog consistency pass**: today the GUI mixes two dialog styles. Native browser dialogs (`window.confirm` / `window.prompt` / `window.alert`) are used in places like Data tab's "add new source folder", Robot tab's recover, and Data tab's add-new-feature confirmation; custom centered modal dialogs are used in places like the add-new-feature UI itself, HF upload/download, and Hub sync. Need a design pass on a single dialog experience. Open questions:
  - Should everything migrate to the custom modal, or are there cases where native dialogs are preferable (blocking, focus trap, escape key behavior)?
  - For multi-step flows (e.g. add source → confirm → result), should each step replace the previous dialog, or stack on top of each other (and if stacking, what's the back/cancel semantics)?
  - Need a small dialog component API that handles confirm / prompt / multi-step out of the box so consumers stop reaching for `window.confirm` ad-hoc.

## Dataset Tools

- [Mid] Consolidate `_keep_episodes_from_video_by_time` (time-based) with `_keep_episodes_from_video_with_av` (frame-based, upstream) in `dataset_tools.py`. Migrate trim callers to frame indices.
- [Mid] Consolidate streaming video encoders: our `video_encoder.py` vs upstream's `video_utils.py`. Upstream's is more mature (HW encoders, frame dropping). Consider migrating.
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
