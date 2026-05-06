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
- [ ] **Multi-dim numeric vector visualization** for `action` / `observation.*` — design between stacked-rows / overlaid-curves / collapsed-expandable
- [ ] Schema mutations: add / remove / rename features via `modify_features`
- [ ] URDF / 3D trajectory views
- [ ] True first-class per-episode features (format extension to `info.json` `episode_features` + `episodes.parquet` writer)
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
- [High] **Duplicate detection within dataset**: detect near-duplicate episodes during dataset opening and before merging. Prevents wasted training compute on redundant data. Could use joint state trajectory similarity or image embedding distance.
- [Mid] **Subtask labeling in GUI** — **superseded by [docs/feature_editing.md](docs/feature_editing.md)**. V1 of Feature Editing delivers exactly this: drag-select a frame range on the subtask row → type the label in the Inspector → Apply.
- [Mid] **Subtask format**: conform subtask column to LeRobot 3.0 format + OpenPI changes. Currently uses raw string column; may need task_index remapping.

## HVLA / Policy Evaluation

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
