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

### Control-Lag Analyzer & Feetech PID Sweep (2026-05-11)

Cross-correlation of `action[:, j]` vs `observation.state[:, j]` per joint measures the effective tracking lag. Analyzer landed at [scripts/analyze_control_lag.py](../../../scripts/analyze_control_lag.py). Filters joints by correlation strength (≥0.95) to avoid spurious lag readings on near-static joints.

**Parameter sweep on bi_so107_follower (white profile), trajectory_replay of `white.trajectory.json`, 1 episode each:**

| Config                                                                        |   Mean lag |      Median |    σ_state | Verdict                         |
| ----------------------------------------------------------------------------- | ---------: | ----------: | ---------: | ------------------------------- |
| **P=16 (current main)**                                                       |   120.0 ms |    133.3 ms |     0.0092 | baseline                        |
| P=16 + dz=0 + startup_force=0                                                 |   123.3 ms |    133.3 ms |     0.0086 | dz alone: no effect on lag      |
| P=32 + dz=0 + startup_force=0 (`feature/motor-sensitivity-fix` branch bundle) |    86.7 ms |    100.0 ms |     0.0192 | -33 ms, σ 2x                    |
| **P=48 (dz default, I=0, D=32)**                                              | **~67 ms** | **66.7 ms** | **0.0133** | **WINNER: -53 ms, σ 1.4x**      |
| P=64                                                                          |    72.2 ms |     66.7 ms |     0.0300 | over-tuned: σ 3.3x for ~no gain |
| P=48 + D=16                                                                   |    72.7 ms |     66.7 ms |     0.0139 | D doesn't help in smooth motion |
| P=48 + D=0                                                                    |    72.7 ms |     66.7 ms |     0.0159 | D doesn't help                  |
| P=48 + I=8                                                                    |    63.3 ms |     66.7 ms |     0.0610 | I causes windup, σ explodes 5x  |
| P=48 + Goal_Velocity=4000                                                     |    70.0 ms |     66.7 ms |     0.0207 | velocity ceiling: no effect     |

**Decoupled send/read rate probe (2026-05-11 follow-up):** `scripts/probe_motor_send_rate.py` connects motors only (no cameras, no dataset) and sends interpolated `goal_position` at a high rate while reading `present_position` at a separate, lower rate. With P=48 and 172 Hz effective send / 30 Hz read, mean lag drops to **60 ms** (from the 67 ms P=48 baseline). The previous "200 Hz coupled" attempt failed due to bus saturation (both sync_read and sync_write at the same high rate); decoupling lets the bus comfortably sustain 170+ Hz of sends. The architectural takeaway: **for trajectory_replay / open-loop replay, only the send rate matters; state reads can stay at observation rate.** This is the same multi-rate concept that would underlie a TimedActionQueue for chunked-policy execution. The 7 ms gain over plain P=48 is the predicted `T/2` reduction from sampling-related lag; the rest of the floor is the motor's intrinsic τ.

**Velocity feed-forward (Goal_Velocity per iteration) probed and rejected (2026-05-11):** Hypothesized that writing `Goal_Velocity` (register 46) alongside `Goal_Position` would give the motor explicit velocity-target information and reduce tracking lag via the cascaded-position-plus-velocity-loop idea. Tested in the probe script with `--velocity-ff --vff-scale {10, 200}`. Result: lag **regresses** to 121-243 ms (vs 63 ms baseline at P=48). The Feetech `Goal_Velocity` register is a velocity CEILING for the firmware's internal position-move profile, not a velocity feed-forward signal — writing it per iteration either caps the motor speed (small scale) or disrupts the position controller state (large scale). Conclusion: **Goal_Velocity has no useful per-iteration role on STS3215**, regardless of scale. Velocity feed-forward in the control-theory sense isn't exposed by this firmware. Hardware change required (e.g., Dynamixel current-control mode) to get a real velocity FF channel.

**Findings:**

- **P_Coefficient is the only lever that matters.** Bumping from 16 to 48 cuts mean lag from 120 → 67 ms (-44%) and improves the gripper from "barely tracking small commands" to "actually tracking" — at P=16 the gripper IGNORES fine motion because its commanded action stays within the firmware dead zone + the low gain can't overcome static friction.
- **Dead-zone register changes do nothing for lag.** The `feature/motor-sensitivity-fix` branch's bundle attributed its win to "dead zones + startup force + P bump," but dz=0 alone gives 123 ms (statistically identical to baseline 120 ms). The bundle's gain came entirely from P=32. Dead zones affect _accuracy_ around target (fine motion in/out of the deadband) more than tracking lag itself.
- **P=64 is over-tuned.** Marginal lag gain (-3 ms vs P=48), σ jumps to 0.030 (3.3× the P=16 baseline) — the original "P=16 to avoid shakiness" comment becomes valid past P=48.
- **D doesn't help in smooth-motion regime.** D=32, 16, 0 all give ~67 ms — D dampens response to velocity changes, but human-recorded trajectories don't have abrupt step-like inputs that D would suppress. D is dead weight here. (Would matter on step inputs / sharp direction reversals.)
- **I causes windup.** I=8 nominally shaved 4 ms off the lag, but σ jumped 5× — `left_shoulder_lift` σ went from 0.013 to 0.257. I integrates error during smooth tracking, builds up demand, overshoots and oscillates. Hard pass.
- **Goal_Velocity (cascaded velocity feed-forward) doesn't help either.** Setting register 46 to 4000 (~280°/s ceiling) during configure gave 70 ms — within noise of plain P=48. Either the motor already uses max velocity by default, or velocity isn't the binding constraint (acceleration is, and that's already maxed at 254).
- **Conclusion: P=48 is the software ceiling.** The remaining 67 ms is the physical floor — 1 frame (33 ms) of structural read-after-write delay + motor PID/dynamics floor + mechanical (inertia, backlash, friction).

**Recommended production change** (deferred — needs user decision because of training data implications):

1. Port the `feature/motor-sensitivity-fix` branch's `p_coefficient` config field onto this branch (without the dead-zone fields, which don't help). Default to **16** to preserve existing behavior — DON'T silently break existing policies trained against P=16's slower tracking.
2. Document P=48 as the recommended value for new datasets / new users. Datasets recorded at different P values are technically incompatible (the action→state response curve differs).
3. Long-term: when the team agrees, switch the default to 48 and retrain affected policies. The gripper tracking fix is a correctness improvement, not just a latency one — old datasets where fine-motion gripper commands were silently dropped should be flagged.

Datasets used: `~/.cache/huggingface/lerobot/thewisp/{p16_baseline,p16_dz0,pid_bundle,p48,p64}_*`. Re-runnable any time the so_follower.py P value is changed.

Caveat: this measures _tracking lag_ (action-to-state convergence during continuous motion), which is a strict superset of the pure command-to-actuation dead time discussed in the latency design doc. The rig-based C2A calibration (IMU + MCU master clock) is still the only way to isolate the few-ms servo dead time itself — but for "is the control loop tracking?" / "did this hardware regress?", cross-correlation is enough and free.

Outstanding work:

- [ ] **Make P_Coefficient configurable** (port `feature/motor-sensitivity-fix` config-field pattern), defaulting to 16 for backwards compat.
- [ ] **Integrate analyzer output into `meta/recording_health.json`** as a `control_lag` block so each recording carries its measured tracking lag in metadata.
- [ ] **Per-motor PID tuning** if needed: the gripper has different torque limits and might want different P than the major joints.

Untested but unlikely to help (documented for completeness):

- **Operating_Mode change** (currently POSITION). Velocity / position+current modes are different control schemes, riskier to swap blind. Not worth without a specific failure mode pointing here.
- **Per-iteration `Goal_Velocity` sync_write** alongside `Goal_Position`. Different from setting once at configure (which we tested) — would feed velocity targets every frame. Would double bus traffic per iteration. Worth trying only if someone needs to push past P=48 ceiling.
- **Max_Torque_Limit / Protection_Current bump** on the gripper specifically (currently 500/250, half of other motors). Could give gripper more force at the cost of overload risk. Only worth if gripper-specific lag is the bottleneck for a specific task.

### Predictive Lookahead Teleop (2026-05-11)

**Goal**: cut the action-to-state lag operators feel during live bimanual teleop by predicting where the leader will be at `t + L` and sending that as the follower's command, instead of `leader(t)` directly. Validated on the bi_so107 hardware with a cylinder-insertion task.

**Architecture** (prototyped in `scripts/proto_decoupled_teleop.py`, not yet productionized):

- **Decoupled control / observation threads.** Control thread runs at ~200 Hz: read leader → compute predicted action → `sync_write` Goal_Position to the follower. Observation thread runs at ~30 Hz: `sync_read` follower state for adaptive measurement (and, in future, dataset writes). A shared `bus_lock` serializes follower-bus access. The leader is on separate USB ports so its reads don't contend.
- **Velocity estimator: linear least-squares slope** over the last ~70 ms of leader reads (≈14 samples at 200 Hz). Alternatives tested and rejected: 2-point forward difference (noisier estimator, similar action smoothness on smooth data, worse on noisy live data); quadratic LSQ (captures acceleration but amplifies sensor noise into command jitter — `c·L²` term is the noise multiplier).
- **Predictor-corrector for command smoothness.** Each tick:
  - `raw_shifted = leader[t] + v_leader · L`
  - `v_action = LSQ slope of recent action history` (smoother than `v_leader` because actions are already filtered)
  - `predictor = a(n-1) + v_action · dt`
  - `action[t] = α · raw_shifted + (1 − α) · predictor`
  - α=0.3 (30 % fresh measurement + 70 % advance-from-previous-action). Kalman-flavored — the predictor propagates the smoothed action trajectory forward one tick. Cuts action excess jerk ~50 % vs the bare `leader + v_leader · L`.
- **Adaptive lookahead** with operator-comfort cap. Symmetric cross-correlation over a rolling 3-second window of (leader, state) measures the residual lag. Update rule: `L ← L + α_la · (residual − read_bias)`. Convergence is a true fixed point; the hard cap (operator-feel-tunable) is the only thing standing between adaptive and runaway when measurements are noisy.

**Production settings, validated by operator on cylinder-insertion task**:

```
P_Coefficient        = 16  (stock, no override)
velocity_method      = linear LSQ
velocity_window_ms   = 70
corrector_alpha      = 0.3
max_lookahead_ms     = 110  ← operator-feel cap (per-arm tunable)
control_fps          = 200, obs_fps = 30
```

**Live A/B results** (25-second cylinder-insertion teleop, P=16/32/48 with same corrector + linear LSQ; `state-vs-leader lag` = signed cross-correlation, `plateau jitter` = `mean|state″|` during quiet-leader rows):

| Metric                   |                P=16 (cap 110) |    P=32 (cap 90) |      P=48 (cap 90) |
| ------------------------ | ----------------------------: | ---------------: | -----------------: |
| state-vs-leader lag      |                        +36 ms |           +17 ms |             +13 ms |
| Fidelity RMSE            |                          3.10 |             2.20 |               2.25 |
| action excess jerk       |                      −0.00550 |         −0.00667 |           −0.00734 |
| state excess jerk        |                      −0.01579 |         −0.01363 |           −0.01202 |
| wrist plateau jitter avg |                       ~0.0035 |    ~0.0035–0.006 | **~0.0088–0.0105** |
| operator feel            | **acceptable, slight jitter** | a bit more shaky |   more shaky still |

P=16 is the production winner. Higher P buys ~20 ms less lag but visibly amplifies wrist tremor — confirms the upstream "Default 32, set to 16 to avoid shakiness" rationale at multiple operating points (coarse 30 Hz commands AND smooth 200 Hz + corrector). The wrists are the perceptual ceiling; shoulders/elbows would tolerate P=48 fine.

**Things tested and ruled out** (documented to avoid re-discovery):

- **Quadratic LSQ velocity.** Captures acceleration on deterministic data (trajectory_blind ran great), but on noisy live leader the `c·L²` term amplifies sensor noise into a 3-12× jitter increase in the command stream. Linear LSQ is the right sweet spot. See synthetic test in script doc.
- **Uniform P=48.** Best fidelity (0.81 RMSE on trajectory_blind, ~0 ms residual lag) AND best cross-corr — but wrist amplification dominates the operator feel. Quadratic+P=48 was the "extremely shaky" combination.
- **`max_lookahead_ms` past ~110.** Trying `cap=130` made adaptive saturate and the operator reported overshoot. The cap is the operator-perceptual jitter ceiling, not a safety bound. Treat as a tunable knob.
- **Pure leader-history velocity for the predictor.** What was originally implemented as the "corrector" — using `v_leader · dt` instead of `v_action · dt` in the predictor step. The corrector still works, but the smoothness benefit is smaller because both arms (raw_shifted and predictor) carry the same noisy `v_leader`. Action-history velocity is the right answer for the predictor.

**Open follow-ups**:

- [ ] **Productionize into `lerobot-teleoperate`.** Move the runner logic out of the prototype script and into the proper teleop entry point. Adds two CLI knobs: `--lookahead-ms` (or `--max-lookahead-ms` for adaptive), `--corrector-alpha`. Defaults match the validated production settings above.
- [ ] **Carry over to `lerobot-record`.** Same control architecture applies; the planning thread additionally writes dataset frames + the existing camera reads. Should improve recording quality (less drift between leader and follower → cleaner action-state pairs in the dataset).
- [ ] **Per-joint P tuning.** Wrist amplification is the only thing keeping P=48 from being the right answer for the big joints. A config that runs P=48 on shoulders/elbows + P=16 on wrists/gripper would plausibly capture the responsiveness without the shake. Requires `SOFollower.configure()` to take a per-motor map. Untested.
- [ ] **Per-arm operator-feel cap** as a calibrated config field. Default 110 ms is fine for the bi_so107 white profile; other arms with different motor τ or different operators may want different. Belongs in the robot's JSON config.
- [ ] **Save raw (action, state, leader) tensors to `.npz`** during runs. Currently the prototype only prints aggregated metrics; raw arrays would let us post-process arbitrarily (different warmup, different per-regime splits, frequency analysis). Trivial extension.

**Notes on metric design** (some hard-won):

- **`fidelity RMSE` = state vs leader at matched timestamps.** No phase shift. Sign cancellation impossible (RMSE).
- **`state-vs-leader lag`** = signed cross-correlation lag. The "by how many ms is follower behind leader at best alignment" metric. Negative = follower briefly leads.
- **`excess jerk`** = `mean|x″| − mean|leader″|`. Subtracts the leader's own natural high-freq content; what remains is the algorithm/motor contribution. **Invariant across operator sessions** — comparable session-to-session in a way raw jerk is not.
- **`plateau jitter`** = `mean|state″|` computed only on rows where the leader is quasi-stationary (velocity in the bottom 30% per-joint). This is the metric closest to "is the motor shaking when I'm holding still" — the regime where sensor noise + algorithm noise is most visible to the operator. **Averaging jerk over the whole run dilutes the plateau regime under the sweep regime; the regime-aware split is what actually maps to operator feel.**
- **First 1 second of every run is dropped from analysis.** Startup transient (follower position vs leader position vs adaptive's initial guess) is wild and pollutes every metric. The lookahead-convergence trace is reported unfiltered so we can see how it climbed.

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
- [High] **Foolproof shutdown-cleanup registry.** Today the server's shutdown hook is a hand-maintained list of cleanup calls (`_stop_debug_process`, `_terminate_active_process`, `cleanup_stale_streams`, `cleanup_in_process_resources`, `shutdown_prefetch_executor`). Adding a new module-level resource — preview cameras, recording robots, prefetch executor, anything yet to come — means remembering to wire it into `server.py::shutdown_event`. Easy to forget, and the omission is silent (the OS reclaims FDs on process death so cleanup never visibly fails). Shape options:
  1. **Registry pattern.** Add `lerobot.gui.shutdown.register(callback)`; each module that opens a long-lived resource calls it at import / first-use. `shutdown_event` iterates the registry. Symmetric, discoverable; the resource owner is co-located with its cleanup. Easy to add.
  2. **AsyncExitStack on the FastAPI lifespan.** Migrate `@app.on_event("startup"/"shutdown")` to the modern `lifespan` async-context-manager API and have each module yield its `AsyncExitStack.callback(cleanup_fn)`. More idiomatic FastAPI, gets us off the deprecated `on_event` decorator (already a `DeprecationWarning` in the test output).
  3. **Resource ownership in `AppState`.** Every long-lived resource becomes an attribute of `AppState`; `AppState.close()` cascades cleanup. Most invasive but makes the lifecycle most explicit.

  Option 2 is the natural endpoint (we'll have to migrate off `on_event` eventually anyway). Option 1 is the quick win. Either way, the goal is: a contributor adding a new background thread / FD-owning resource doesn't need to know about `server.py`.

  **Orphan-subprocess failure mode (observed 2026-05-13).** A specific incarnation of this bug that bit hard: the GUI launches `lerobot-teleoperate` / `lerobot-record` / `lerobot-replay` subprocesses via `asyncio.create_subprocess_exec(..., start_new_session=True)`. The `start_new_session=True` is intentional — it detaches the child's process group so a Ctrl+C in the user's GUI terminal doesn't kill the workload mid-recording — but it also means **any abnormal GUI exit leaves the subprocess running**. Concrete case found: a `lerobot-teleoperate` subprocess was still alive almost 2 hours after the GUI server process that spawned it had exited, holding all four ttyACM ports and the cameras, blocking any subsequent launch. Only the graceful `shutdown_event` path calls `_terminate_active_process`; SIGKILL on the GUI (or any path that skips FastAPI lifespan) is silent and the subprocess survives indefinitely. Required to fix:
  - Whatever cleanup-registry mechanism we land must wrap the active subprocess so a parent-death detector (Linux: `prctl(PR_SET_PDEATHSIG)`, set in the child before exec) can SIGTERM it when the GUI's process tree dies — independent of the lifespan hook firing.
  - On GUI startup, scan for prior-launch zombies (e.g. by writing the subprocess PID into `~/.cache/huggingface/lerobot/gui/active_process.json` at launch and reading + killing on next startup if still alive and the heartbeat is stale).
  - The shutdown registry's iteration should be best-effort under exception, so a hung cleanup callback can't block the others.

- [High] **RLT metrics pipeline is wasteful end-to-end**. Two compounding issues: (1) Training subprocess rewrites the entire `metrics.json` (~150KB for 5000 points × 8 series) on every save, even when only a few new points were appended — atomic write via `.tmp` + `os.replace` on each episode end and every 100 inference steps. (2) GUI polls `/api/run/rlt-metrics` every 2s and gets the same full snapshot back, of which ~99.6% is unchanged from the previous poll. Fine on localhost, wasteful otherwise. Options: append-only JSONL or shared memory on the write side; SSE push of only new points or cursor-based polling (`?since_step=N`) on the read side. Frontend maintains a local buffer.
- [Mid] **RLT dashboard chart smoothing**. Per-inference series like `actor_deltas` have per-sample noise comparable to the actual trend (e.g. δ raw σ ≈ 0.015 per sample vs a real β-driven shift of 0.018 — z=17.8 over 500 samples but invisible in any single-sample view). The dashboard plots raw values, so genuine learning signals get hidden. Add a smoothing toggle / moving-average overlay (window picker: raw / 50 / 200 / 500), or always show both the raw line (light) and a smoothed line (bold). Same applies to `q_values_*`, `critic_losses`, `actor_q_term`, `actor_bc_term` — all have per-grad-step noise. Cheap if the smoothing happens in JS over the buffered series the dashboard already pulls.
- [Mid] **Backend-driven Launch validation schema**. The Launch button's required-field rules currently live in JS-side `_WORKFLOW_VALIDATORS` and `_POLICY_VALIDATORS` registries in `gui/static/run.js`. Each policy_type's Pydantic/Draccus config already declares its required fields in Python — the JS registry has to be updated by hand whenever those declarations change, and silent drift between the two is invisible until a user clicks Launch and gets a backend 400. Replace with a `GET /api/policy-schemas/{policy_type}` (and similar for workflows) returning e.g. `{required: ["task"], required_when: {"rlt_token_checkpoint": "rlt_mode"}}`; the frontend consumes that verbatim, so adding a new policy means only one Python edit. Until then, any change to a policy's required fields must be reflected in both places.
- [Mid] **Audit `torch.load()` for `weights_only=True`** (bandit B614, currently in global skips). Call sites: `src/lerobot/policies/act_vlm/modeling_act_vlm.py`, `src/lerobot/policies/hvla/s1/flow_matching/model.py`, `src/lerobot/policies/hvla/s1_process.py`. Since PyTorch 2.6 the default flipped to `weights_only=True`; our checkpoints predate that and contain non-tensor pickled metadata, so wholesale flip would break loaders. Plan: per-site, switch to `weights_only=True` and migrate any non-tensor state to a sidecar JSON / safetensors. Remove B614 from `pyproject.toml` `[tool.bandit].skips` once the audit completes.

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
### GUI Camera FD Leak

Baseline from a 2-episode bi_so107_follower record run with trajectory_replay, 4 cameras at 1280×720 @ 30 fps (2026-05-11, snapshot in `outputs/record/latency_snapshot.json`): loop p50=24.9 ms / p95=28.2 ms / p99=30.8 ms, overrun 0.55%. Healthy at 30 fps but the budget breakdown surfaces three places worth attention:
Baseline from a 2-episode bi_so107_follower record run with trajectory_replay, 4 cameras at 1280×720 @ 30 fps (2026-05-11, snapshot in `outputs/record/latency_snapshot.json`): loop p50=24.9 ms / p95=28.2 ms / p99=30.8 ms, overrun 0.55%. Healthy at 30 fps but ~75% of the 33.3 ms budget is used; concrete optimizations below are ordered by (impact ÷ effort). Doing items 1-4 takes loop p50 from ~25 ms to ~18-19 ms.

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

- [High] **Cut `DepthEdgeOverlayProcessorStep` cost (the whole 11.5 ms p50).** Step is unconditional in [robots/bi_so107_follower/bi_so107_follower.py:225-247](../../robots/bi_so107_follower/bi_so107_follower.py); algorithm in [processor/depth_edge_processor.py](../../processor/depth_edge_processor.py). Three stackable wins, each independently profilable:
  1. Downsample 4× before Sobel + Gaussian, upscale mask back via `cv2.resize(INTER_LINEAR)`.
  2. Skip alternate frames (reuse last mask). The overlay is for human-eye debugging, not policy input.
  3. Cache the percentile threshold every N frames.
     Per-step savings need post-change measurement; combined target: process_obs under 4 ms. Plus a `depth_edge_overlay: bool = True` config flag for full opt-out when the policy doesn't consume the overlay.
- [High] **Off-loop `dataset.add_frame()`.** Removes the measured 4.5 ms p50 / 12 ms p95 from the critical path. `add_frame()` in [datasets/dataset_writer.py:229-297](../../datasets/dataset_writer.py) writes to `episode_buffer` which only needs to be consistent at `save_episode()` time. Wrap in a thread-safe background queue + worker; copy the frame dict before enqueueing; preserve ordering for `save_episode()`. Medium effort.
- [Mid] **Investigate per-camera `push_frame` 6-8x p95/p50 ratios.** All four `OurStreamingVideoEncoder` per-camera pushes burst together. Documented as non-blocking but evidence says otherwise. Hypotheses: encoder thread holds the GIL during encode (`libsvtav1` is CPU-bound) and blocks the control thread mid-push; or per-frame shared-memory copy contends for memory bandwidth. Probe by timing the `push_frame()` body internally vs. the queue.put; once attributed, fix is likely either to make the GIL-release tighter or to switch the encoder to a subprocess.
- [Mid] **Investigate `get_observation` 3.8x p95/p50.** Stable on motor reads (~1.2-1.4x), so the tail comes from `cam.read_latest()` despite the cache. Possible cause: camera grab thread holds the cache lock during `numpy.copy()` into the cache slot. Cheap follow-up: per-camera consume-time span around `cam.read_latest()` in `bi_so107_follower.get_observation`.
- [Low] **`AsyncImageWriter` non-blocking + drop-frame fallback.** [datasets/image_writer.py:174-180](../../datasets/image_writer.py) — `queue.put()` defaults to blocking with infinite timeout; switch to `put(block=False)` with logging on `queue.Full`. Not active in our per-camera config (image writer bypassed for video keys), but defensive in case a future config path reaches it.

**Already-measured non-issues:**

- `motor_read_left/right` are stable (~1.2-1.4x p95/p50). No work to do here.
- `process_action`, `action_send` together cost 0.27 ms p50. Not worth touching.
- `IdentityProcessorStep` in the default pipeline costs 0 ms — confirmed no-op.

### GUI Resource & Async Hygiene

- [x] ~~**GUI server transiently holds `/dev/video*` FDs and blocks CLI teleop.**~~ Fixed 2026-05-12 in `_detect_and_open_cameras()`: each per-camera section now uses an ownership-transfer try/finally (`camera = None` reset only after a successful append) so any error between `connect()` and successful registration triggers `disconnect()`. Regression covered by `tests/gui/test_camera_preview_lifecycle.py` (connect-failure, partial-failure, happy-path).
- [x] ~~**Blocking `subprocess.Popen` in async FastAPI handler.**~~ Fixed 2026-05-12 across all three `/open-in-files` endpoints (`gui/api/robot.py`, `gui/api/datasets.py`, `gui/api/models.py`): the `xdg-open` spawn now runs via `loop.run_in_executor(None, _spawn)` so a slow fork/exec cannot stall the FastAPI event loop. `visualize_episode` (`gui/api/datasets.py`) still launches its Rerun subprocess synchronously — separate follow-up if a hang is ever observed there.

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
- [x] ~~**Configurable auto-reset duration.**~~ Promoted 2026-05-12 from the inline `_auto_reset_duration_s` to a `RecordConfig.auto_reset_duration_s` field (default 3.0 s); plumbed into `move_to_rest_position` during the reset phase for file-backed teleops.

### Run Log Files & Crash Visibility

- [High] **Promote teleop's log-file + excepthook helper into a reusable utility.** `setup_run_logging(output_dir, run_name)` lives in [utils/utils.py](../../utils/utils.py); creates `<output_dir>/<run_name>_<ts>.log`, calls `init_logging`, and installs main-thread + thread-pool excepthooks. Covered by `tests/utils/test_setup_run_logging.py`. Adopted by `lerobot-teleoperate` and `lerobot-record` (2026-05-12). Remaining scripts to migrate, in priority order:
  1. [x] ~~**lerobot-record**: switched to `setup_run_logging` + per-run log file under `outputs/record/`. The `latency_output_dir` default also moved from `"outputs/teleop"` to `"outputs/record"` so the GUI dashboard treats the two workflows as distinct source keys.~~ Beep `subprocess.Popen` cleanup still open.
  2. **lerobot-train** — multi-process accelerate; worker failures go to stderr only. Add per-run log file in the training output dir; consider per-process suffixes for distributed debugging.
  3. **lerobot-eval** — daemon prefetch thread at [scripts/lerobot_eval.py:784](../../scripts/lerobot_eval.py) isn't `join()`-ed on the success path; could leave a GPU prefetch process orphaned. Fix in the same pass.
  4. **lerobot-replay** — minimal; just the log file + excepthook (no thread issues).
  5. **lerobot-calibrate**, **find-cameras**, **find-port** — lower urgency; manual one-off tools where the user is in front of the console anyway.

### Latency Panel UX

- [x] ~~**Duplicate track rendering** (fixed by `e935ce925`): teleop and record now write to separate `outputs/teleop` and `outputs/record` directories; `LATENCY_SOURCES` maps them to those distinct paths, so each source key reports `fresh` only when its own writer is publishing.~~
- [High] **Residual teleop jitter at fast speed**: the 1 Hz writer-thread refactor cut the loop-side spike from 22 ms to ~1 ms, but the user still feels small jitter under fast movement. Suspected GIL contention: the background writer thread holds the GIL during snapshot computation (transient aggregator construction, dict building, JSON dump), and the loop thread waits for the GIL on each iteration that overlaps the writer's work. To verify: time the loop thread's iteration during the second the writer runs vs during a quiet second; difference > 0 confirms GIL hit. Mitigations to try in order: (a) reduce writer-thread compute (skip `iterations` / `aggregate_iteration` when no GUI is polling? compute fewer percentiles?), (b) drop snapshot rate to 0.5 Hz, (c) move the writer to a subprocess (`multiprocessing.Process`) which doesn't share the GIL — heavyweight but the only way to truly isolate Python work from the loop.
- [x] ~~**Gantt bar height ~1.5x** (fixed by `e935ce925`): `rowH` clamp bumped 20→30 max / 10→15 min in `_drawGantt`, headroom for the 10 px label + 4 px bar margin.~~
- [x] ~~**X-axis label overlap at edges** (fixed by `e935ce925`): tick rendering in `_drawGantt` now skips any tick whose x falls within `cornerLabelHalfWidth = 32 px` of either edge, and the corner extent labels always render — no more "-40 -40ms" overlaps.~~
- [Mid] **Top-row metrics: only loop + overrun**: current cards (loop, get_observation, action_send, overrun) mix one umbrella stat with two partial stages and one rate — confusing because (a) loop already includes get_observation + action_send + other unshown stages like process_obs, (b) overrun is conceptually a different category. Replace with just `loop` and `overrun` as the headline row; move per-stage breakdowns to the Gantt + a separate compact stage table if useful.
- [Mid] **camera_read_strategy as dropdown in robot config UI**: today it's an open text field. Two valid values (`latest`, `wait_for_new`) — render as a `<select>` so users don't typo and silently get fallback behaviour. Same pattern as other enum-ish robot fields. May require a small schema-driven render hint (e.g. metadata in the dataclass field) so this generalises beyond bi_so107.
- [Discussion] **Color thresholds**: live panel shows 21.1 ms as yellow at a 30 Hz target (33.3 ms budget). Current rule fires yellow at ≥ 70% of budget; that's 23.3 ms, so 21 ms shouldn't be yellow — likely the comparison is against p95 (which IS > 23.3 if p50 is 21). Worth reviewing whether yellow should fire on p50 or p95, and whether the 70% threshold is right.
- [Low] **Revisit Gantt vs flame graph**: today the Gantt renders nested spans (e.g. `motor_read_left` inside `get_observation`) as overlapping bars on the same row. First-time viewers read the overlap as parallelism — "is `get_observation` on another thread?" — when actually they're sequential on the same thread, just nested via `with span():`. A flame-graph style (parent on top, children stacked beneath, vertical = call depth) would make the nesting obvious without legend reading. Single-iteration view benefits the most; aggregate now also preserves nesting after the layout fix so the visualisation upgrade applies to both modes.

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
