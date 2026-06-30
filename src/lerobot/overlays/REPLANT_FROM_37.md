# Replanting PR #37 (`proto/gui-debug-vision`) onto main

#37 is the pre-#41 debug-vision prototype. #41 landed and productionized the overlay infra + SAM3 into
`lerobot/overlays/`, so we **replant** #37's unique content here (branch `proto/debug-vision-rebased`,
off main) rather than a literal `git rebase` — #37 is 10+ commits behind, pre-move, with a 412-line
`run.js` + 211-line `run.py` divergence, i.e. a conflict swamp.

## Dedup — already in `lerobot/overlays/` (do NOT re-port)

- The overlay infra: `SharedOverlayBuffer`, the standalone loop, the `DebugVisionAdapter` base, the control IPC.
- The SAM3 track adapter — #41 renamed #37's `sam3_video` → `sam3_track` (`Sam3TrackByDetectionAdapter`); logic identical.
- #41 is a superset: adds `OverlayStatus`, `aux_ipc`, the `overlay_state` machine, inode-reattach in standalone, and the `policy_attention` / `policy_saliency` adapters.
- #41 **dropped** the `amodal` path (FoundationPose mesh occlusion) from the SAM3 adapter — re-add it in increment 3.

## Increments (replant order)

- [x] **1. obs_stream depth** — RealSense uint16 depth blocks + `read_depth` (commit `f759b1b96`). Foundational for FoundationPose.
- [ ] **2. The 6 unique debug-vision adapters** → `overlays/adapters.py` + the `ADAPTERS` registry + the panel model-schema (controls / load_cost in `gui/api/overlays.py`):
      `grounding_dino` (open-vocab boxes), `dino_features` (DINOv2 PCA heatmap), `depth_anything` (mono depth), `sam2_mask` (click-point), `sam3` (static text mask), `cotracker3` (point tracks). **Skip** the legacy OOM-prone `sam3_video_concept`. Manual port — `adapters.py` is heavily diverged, so it can't be diff-applied like obs_stream.
- [ ] **3. FoundationPose** — `foundationpose_{client,ipc,worker}.py` → `lerobot/overlays/` (was `policies/debug_vision/`, now gone). The worker stays in the isolated `~/.cache/sam3d/venv` (torch 2.8 ABI). Re-add the `amodal` toggle to the SAM3 adapter (spawn/kill the FP worker, composite its RGBA). Own `lerobot_fp_*` shm IPC (pure stdlib — no lerobot imports in the worker).
- [ ] **4. scan3d** — `gui/api/scan3d.py` + `scan3d_worker.py` + the 3D-mesh display in `urdf_viz.html` (vendored GLTFLoader). SAM3 mask + SAM 3D Objects → GLB → URDF viewer, scale anchored to metric depth.
- [ ] **5. pyproject** debug-vision extras + `uv.lock`.

## Compat delta (for every ported piece)

- Module path `policies.debug_vision` → `lerobot.overlays`; standalone entry `-m lerobot.overlays.standalone`.
- The adapter registry shape + the panel model-schema (each model declares its own controls — objects / none / click-point / grid).
- IPC class names — main's `SharedOverlayBuffer` is a superset, so the unique adapters need no IPC change.

## Part B — 3D tracking (the goal these increments enable)

Increments 1 + 3 + 4 **are** the 3D pipeline. Recommended build:
`SAM3 mask → multi-view RGBD pre-scan (object on a plate / held + rotated, top RealSense) → fused
complete mesh → FoundationPose live 6-DoF tracking → URDF-viewer display`.
FoundationPose's two-clock pattern (geometry once, pose live) is the backbone. The "refine-from-any-mask
gradually" ideal layers on later (re-fuse the FoundationPose-tracked views into the mesh). Pre-scan first
because single-view SAM3D hallucinates occluded geometry (the hollow-tube failure) — see the do-as-i-do notes.
