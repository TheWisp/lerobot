# GUI TODO

## Data Tab

- [High] **Warning/error panel**: dataset verification errors and warnings are currently buried in server log text. Add a visible warning panel (banner or sidebar) that surfaces verification results when a dataset is opened — errors as red, warnings as yellow. Users must not miss data integrity issues.
- [High] **Open local dataset by path**: opening a copied/renamed local dataset fails because `LeRobotDataset.__init__` tries to reach HuggingFace Hub when the folder name doesn't match a cached `owner/name` repo_id. Spaces in folder names also rejected. Need to bypass Hub entirely for local-only datasets.
- [ ] Parquet data display (action/state charts in timeline)
- [ ] Monitor local dataset changes — auto-refresh UI when new episodes recorded while GUI is open
- [ ] Duplicate episode
- [ ] Copy/move episodes between datasets
- [ ] Reorder episodes
- [ ] Create new dataset from UI
- [ ] Drag-drop dataset opening
- [ ] Undo/redo

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

## UX

- [Mid] Cross-reference navigation: clickable links from dataset/model/robot references to their tab (generic utility, not one-off per instance)

## Dataset Tools

- [Mid] Consolidate `_keep_episodes_from_video_by_time` (time-based) with `_keep_episodes_from_video_with_av` (frame-based, upstream) in `dataset_tools.py`. Migrate trim callers to frame indices.
- [Mid] Consolidate streaming video encoders: our `video_encoder.py` vs upstream's `video_utils.py`. Upstream's is more mature (HW encoders, frame dropping). Consider migrating.
- [High] **Duplicate detection within dataset**: detect near-duplicate episodes during dataset opening and before merging. Prevents wasted training compute on redundant data. Could use joint state trajectory similarity or image embedding distance.
- [Mid] **Subtask labeling in GUI**: allow labeling subtask boundaries within an episode from the Data tab timeline. Click/drag to mark frame ranges with subtask names.
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
