### PR #4 [MERGED] Latency monitoring framework for teleop / record / HVLA loops + supporting perf

- (embed) Latency monitoring dashboard — multi-track Gantt with per-stage spans (including BiSO107's per-arm motor reads and per-camera sub-spans), loop-health card with p50/p95 and overrun ratio, and per-camera staleness footer -> https://github.com/TheWisp/lerobot/raw/pr/latency-monitoring/src/lerobot/gui/docs/images/latency_dashboard.png

### PR #7 [MERGED] Lookahead motor controller: chunk-aware exact lookup + rate-invariant predictive extrapolation

- (embed) Lookahead controller demo — bi_so107 with stateful_lp + L=150 ms -> https://img.youtube.com/vi/Ttw2Oi1WeX4/0.jpg
- (embed) stateful_lp final composite -> https://raw.githubusercontent.com/TheWisp/lerobot/2303f6dc7a0b9825496bbbdcc177034e24ad9669/docs/predictive_lookahead/predictive_lookahead.png
- (embed) stateful_lp animated walkthrough -> https://raw.githubusercontent.com/TheWisp/lerobot/2303f6dc7a0b9825496bbbdcc177034e24ad9669/docs/predictive_lookahead/predictive_lookahead.gif

### PR #8 [MERGED] gui: URDF state visualization

- (embed) URDF visualizer -> https://raw.githubusercontent.com/TheWisp/lerobot/feat/urdf-viz/src/lerobot/gui/docs/images/urdf_viz.png

### PR #9 [closed] kinematics: Pink-based IK with posture regularization

- (embed) so101 circle 30mm -> https://raw.githubusercontent.com/TheWisp/lerobot/proof/ik-e2e/proof/ik_so101_circle-30mm.gif
- (embed) so107 circle 30mm -> https://raw.githubusercontent.com/TheWisp/lerobot/proof/ik-e2e/proof/ik_so107_circle-30mm.gif
- (embed) so101 circle 60mm -> https://raw.githubusercontent.com/TheWisp/lerobot/proof/ik-e2e/proof/ik_so101_circle-60mm.gif
- (embed) so107 circle 60mm -> https://raw.githubusercontent.com/TheWisp/lerobot/proof/ik-e2e/proof/ik_so107_circle-60mm.gif
- (embed) so101 square 50mm -> https://raw.githubusercontent.com/TheWisp/lerobot/proof/ik-e2e/proof/ik_so101_square-50mm.gif
- (embed) so107 square 50mm -> https://raw.githubusercontent.com/TheWisp/lerobot/proof/ik-e2e/proof/ik_so107_square-50mm.gif
- (embed) Pink IK trade-off -> https://raw.githubusercontent.com/TheWisp/lerobot/feat/pink-ik/src/lerobot/model/docs/pink_ik_tradeoff.png

### PR #10 [closed] [Deprecated — too large, splitting follow-ups] teleoperators: Cartesian VR teleop for bimanual SO-107 (Quest)

- (embed) Trajectory closure -> https://raw.githubusercontent.com/TheWisp/lerobot/feat/quest-vr-teleop/src/lerobot/teleoperators/quest_vr/docs/cartesian_ik_trajectory.png
- (embed) Hardware trajectory traces -> https://raw.githubusercontent.com/TheWisp/lerobot/feat/quest-vr-teleop/src/lerobot/teleoperators/quest_vr/docs/hardware/trajectory_traces.png
- (embed) Static-hold per-motor -> https://raw.githubusercontent.com/TheWisp/lerobot/feat/quest-vr-teleop/src/lerobot/teleoperators/quest_vr/docs/hardware/static_hold.png

### PR #11 [closed] feat(gui/hub): live progress + cancel for HF upload/download

- (embed) 02_download_in_progress.png -> https://raw.githubusercontent.com/TheWisp/lerobot/screenshots-hub-progress/02_download_in_progress.png
- (embed) 05_upload_in_progress.png -> https://raw.githubusercontent.com/TheWisp/lerobot/screenshots-hub-progress/05_upload_in_progress.png
- (link) Both toasts after completion -> https://raw.githubusercontent.com/TheWisp/lerobot/screenshots-hub-progress/06_upload_complete.png

### PR #12 [closed] gui: action-overlay ghost on URDF viz; data tab gets a URDF tile

- (embed) off -> https://raw.githubusercontent.com/TheWisp/lerobot/feat/urdf-viz-action-overlay/src/lerobot/gui/docs/images/urdf_viz_data_tab_off.png
- (embed) on -> https://raw.githubusercontent.com/TheWisp/lerobot/feat/urdf-viz-action-overlay/src/lerobot/gui/docs/images/urdf_viz_data_tab_on.png

### PR #14 [MERGED] gui: action-overlay ghost on URDF viz; data tab gets a URDF tile

- (embed) off -> https://raw.githubusercontent.com/TheWisp/lerobot/feat/urdf-viz-action-overlay/src/lerobot/gui/docs/images/urdf_viz_data_tab_off.png
- (embed) on -> https://raw.githubusercontent.com/TheWisp/lerobot/feat/urdf-viz-action-overlay/src/lerobot/gui/docs/images/urdf_viz_data_tab_on.png

### PR #15 [MERGED] feat(gui/hub): background Transfers tray + subprocess-worker pipeline (design + impl)

- (embed) A_idle.png -> https://raw.githubusercontent.com/TheWisp/lerobot/screenshots-hub-progress/impl_v2/A_idle.png
- (embed) B_modal_ready.png -> https://raw.githubusercontent.com/TheWisp/lerobot/screenshots-hub-progress/impl_v2/B_modal_ready.png
- (embed) C_active_inflight.png -> https://raw.githubusercontent.com/TheWisp/lerobot/screenshots-hub-progress/impl_v2/C_active_inflight.png
- (embed) D_complete_with_hide.png -> https://raw.githubusercontent.com/TheWisp/lerobot/screenshots-hub-progress/impl_v2/D_complete_with_hide.png
- (embed) E_failed_retry_discard.png -> https://raw.githubusercontent.com/TheWisp/lerobot/screenshots-hub-progress/impl_v2/E_failed_retry_discard.png

### PR #16 [MERGED] gui(urdf-viz): EE-trajectory tube for multi-frame sources

- (embed) Trajectory progression -> https://raw.githubusercontent.com/TheWisp/lerobot/feat/urdf-viz-trajectory/src/lerobot/gui/docs/images/urdf_viz_data_tab_trajectory.png

### PR #17 [MERGED] robots: Cartesian-EE teleop + virtual robot for end-to-end GUI verification

- (embed) Virtual robot record-then-replay -> https://raw.githubusercontent.com/TheWisp/lerobot/feat/cartesian-ee-virtual-robot/src/lerobot/robots/virtual_bi_so107/docs/virtual_bi_so107_demo.gif
- (embed) IK trajectory closure -> https://raw.githubusercontent.com/TheWisp/lerobot/feat/cartesian-ee-virtual-robot/src/lerobot/robots/virtual_bi_so107/docs/cartesian_ik_trajectory.png
- (embed) robot config form with ik_posture_cost + ik_max_iters -> https://raw.githubusercontent.com/TheWisp/lerobot/feat/cartesian-ee-virtual-robot/src/lerobot/robots/virtual_bi_so107/docs/robot_form_ik_config.png

### PR #18 [MERGED] teleop: Quest 3 WebXR Cartesian teleop for SO-107 (quest_vr)

- (embed) Quest VR → bi_so107 real-hardware demo -> https://img.youtube.com/vi/KSwNev5JRIc/hqdefault.jpg
- (youtube) KSwNev5JRIc -> https://youtu.be/KSwNev5JRIc
- (embed) quest_vr teleop form -> https://raw.githubusercontent.com/TheWisp/lerobot/feat/quest-vr/src/lerobot/teleoperators/quest_vr/docs/teleop_form.png
- (embed) Quest VR → virtual_bi_so107 demo -> https://img.youtube.com/vi/C5pX30HpgeI/hqdefault.jpg

### PR #20 [MERGED] AI-native: MCP server + GUI bridge (unified single-process)

- (embed) /ai_setup default -> https://raw.githubusercontent.com/TheWisp/lerobot/90119bce47287856d29123151b2eedbb5ec08385/src/lerobot/gui/docs/images/ai_setup_collapsed.png
- (embed) /ai_setup expanded -> https://raw.githubusercontent.com/TheWisp/lerobot/90119bce47287856d29123151b2eedbb5ec08385/src/lerobot/gui/docs/images/ai_setup_expanded.png
- (embed) Codex querying lerobot MCP -> https://raw.githubusercontent.com/TheWisp/lerobot/90119bce47287856d29123151b2eedbb5ec08385/src/lerobot/gui/docs/images/codex_count_datasets.png
- (embed) navigate_to tab -> https://raw.githubusercontent.com/TheWisp/lerobot/90119bce47287856d29123151b2eedbb5ec08385/src/lerobot/mcp/docs/proofs/feat_navigate_tab.png
- (embed) navigate_to dataset -> https://raw.githubusercontent.com/TheWisp/lerobot/90119bce47287856d29123151b2eedbb5ec08385/src/lerobot/mcp/docs/proofs/feat_navigate_dataset.png
- (embed) navigate_to episode -> https://raw.githubusercontent.com/TheWisp/lerobot/90119bce47287856d29123151b2eedbb5ec08385/src/lerobot/mcp/docs/proofs/feat_navigate_episode.png
- (embed) notify_user -> https://raw.githubusercontent.com/TheWisp/lerobot/90119bce47287856d29123151b2eedbb5ec08385/src/lerobot/mcp/docs/proofs/feat_notify_user.png
- (embed) highlight_in_viewer -> https://raw.githubusercontent.com/TheWisp/lerobot/90119bce47287856d29123151b2eedbb5ec08385/src/lerobot/mcp/docs/proofs/feat_highlight_viewer.png
- (embed) mcp e2e demo -> https://raw.githubusercontent.com/TheWisp/lerobot/90119bce47287856d29123151b2eedbb5ec08385/src/lerobot/mcp/docs/demo_e2e.gif
- (link) `src/lerobot/mcp/docs/demo_e2e.mp4` -> https://media.githubusercontent.com/media/TheWisp/lerobot/90119bce47287856d29123151b2eedbb5ec08385/src/lerobot/mcp/docs/demo_e2e.mp4
- (embed) ai_setup default collapsed -> https://raw.githubusercontent.com/TheWisp/lerobot/51175d5c1edfac5b7ce02b9aa727e63f6f990d9a/src/lerobot/gui/docs/images/ai_setup_collapsed.png
- (embed) ai_setup snippets expanded -> https://raw.githubusercontent.com/TheWisp/lerobot/51175d5c1edfac5b7ce02b9aa727e63f6f990d9a/src/lerobot/gui/docs/images/ai_setup_expanded.png

### PR #22 [MERGED] feat(mcp): dataset edit MCP tools — propose / apply / discard

- (embed) propose_delete -> https://raw.githubusercontent.com/TheWisp/lerobot/a38098fcc661b2d367eb22aa1887f88ea1b0316a/src/lerobot/mcp/docs/proofs/feat_propose_delete.png
- (embed) propose_trim -> https://raw.githubusercontent.com/TheWisp/lerobot/a38098fcc661b2d367eb22aa1887f88ea1b0316a/src/lerobot/mcp/docs/proofs/feat_propose_trim.png
- (embed) discard_pending -> https://raw.githubusercontent.com/TheWisp/lerobot/a38098fcc661b2d367eb22aa1887f88ea1b0316a/src/lerobot/mcp/docs/proofs/feat_discard_pending.png

### PR #34 [MERGED] fix(gui): faithfully diagnose unopenable local datasets

- (embed) Metadata inconsistent dialog -> https://raw.githubusercontent.com/TheWisp/lerobot/38cf914245b5ac4ae4af58b59344e944be8d503a/src/lerobot/gui/docs/images/open_dataset_metadata_inconsistent.png
- (embed) Missing files, not on Hub -> https://raw.githubusercontent.com/TheWisp/lerobot/38cf914245b5ac4ae4af58b59344e944be8d503a/src/lerobot/gui/docs/images/open_dataset_missing_files_not_on_hub.png
- (embed) Missing files, on Hub -> https://raw.githubusercontent.com/TheWisp/lerobot/38cf914245b5ac4ae4af58b59344e944be8d503a/src/lerobot/gui/docs/images/open_dataset_missing_files_on_hub.png

### PR #35 [MERGED] refactor(gui): consolidate chart primitives into one charts.js (hover everywhere)

- (embed) rlt panel -> https://raw.githubusercontent.com/TheWisp/lerobot/e9caf15401b10b64f94dcc7787c480b268101e16/scripts/gui/screenshots/charts-rlt-panel.png
- (embed) rlt hover -> https://raw.githubusercontent.com/TheWisp/lerobot/e9caf15401b10b64f94dcc7787c480b268101e16/scripts/gui/screenshots/charts-rlt-panel-hover.png
- (embed) latency cards -> https://raw.githubusercontent.com/TheWisp/lerobot/7d2556ac4c02088f98e9385e84705a9bc895ec1f/scripts/gui/screenshots/charts-latency-card.png
- (embed) latency hover -> https://raw.githubusercontent.com/TheWisp/lerobot/7d2556ac4c02088f98e9385e84705a9bc895ec1f/scripts/gui/screenshots/charts-latency-card-hover.png
