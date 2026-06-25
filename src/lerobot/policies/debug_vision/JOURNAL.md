# Perception-networks journal

Findings log for the debug-vision perception stack (SAM3 / Grounding DINO / DINOv2 /
DepthAnything / FoundationPose). Chronological; newest first. For forward-looking tasks
see `../../gui/TODO.md` (Debug-vision architecture section).

---

## 2026-06-25 — FoundationPose amodal: the "edge-on desync" was a wrong mesh, not tracking

**Symptom.** The amodal 3D overlay tracked the green ring fine face-on but diverged badly
when the ring went on its side (edge-on), and the pose gizmo kept spinning.

**Root cause #1 — the tracking mesh was wrong.** FoundationPose was tracking against a
single-view **SAM 3D Objects** reconstruction (`scan3d/object.glb`) made from a bad input:
the ring was clipped at the frame edge, oblique, and had a gripper finger through the hole.
SAM 3D is single-image and its geometry prior is **MoGe (monocular depth guess)** — it never
saw our RealSense depth — so it reconstructed a **fat rounded donut, ~2× too thick**
(axial/outer 0.63 vs the real 0.31). Face-on the silhouette matched; edge-on the fat tube
diverged. **FoundationPose itself was tracking correctly the whole time.**

**Root cause #2 — the object isn't a donut, it's a short tube.** Hand-measured ground truth:
an annular cylinder, **outer ⌀48 mm, hole ⌀23 mm, height 19 mm**. A torus is rounded edge-on;
a tube is a rectangle edge-on — that's the shape that was missing.

**Fix (shipped).** `foundationpose_worker.py` now generates the tube **parametrically** by
default (`RING_OUTER_DIA/HOLE_DIA/HEIGHT`, `trimesh.creation.annulus`); `--mesh` overrides.
`anchor_scale` re-sizes to the depth at register, so absolute size self-corrects — the
**ratios** are what matter. Overlay redesigned: white silhouette (outer + hole contours) +
faint front-face wireframe + pose gizmo + cyan SAM-mask outline. (The old solid fill made
amodal mesh-growth *under occlusion* look like a desync — it isn't: the mesh extending past
the SAM mask **is** amodal working.)

**Spinning gizmo = symmetry.** A tube's rotation about its axis is unobservable. Declared to
FoundationPose via `symmetry_tfs` (continuous about mesh +Z + 180° end-flip), and the gizmo's
perpendicular axes are derived from a fixed reference so it can't spin by construction.

**Live timings (RTX 5090, current pipeline).**
| stage | latency | note |
|---|---|---|
| SAM tracker / frame | 57 ms | ~18 fps — **sets the loop rate** |
| FP `track_one` / frame (incl. overlay) | 18 ms | sidecar, non-blocking → overlay lags <1 frame |
| FP `register` | **3.4 s** | one-time lock / on re-register (the freeze) |

**Register is a global ORIENTATION search, not an image search.** Position comes from the
SAM mask centroid + depth (already known). Register builds ~240 rotation hypotheses
(`make_rotation_grid` 40 views × 6 in-plane), refines *each* 5× through the refiner net, and
scores them. That's the 3.4 s. `track_one` is 18 ms because it refines one known pose, 2 iter.

**Re-register behavior (heavy-occlusion relevant).** Triggered by *divergence*
(`cover < 0.30`), throttled (every 20 frames), and only from a clean mask. Object disappears
→ holds the last overlay (no re-register); reappears near → cheap `TRACK`; reappears after
moving while hidden → one 3.4 s register once it's cleanly visible.

**Open research (deferred — not production).**
- **SAM-guided register:** PCA the masked-depth point cloud → object axis → seed register at
  that orientation, skip the ~240-hypothesis grid. For the symmetric tube this could take
  register from 3.4 s toward tracking speed. Biggest win for the occlusion freeze.
- **Cheap re-center:** on divergence, reset only translation (SAM centroid + depth), keep
  orientation, resume `TRACK` — full register only on a genuine tumble.
- **FoundationPose model-free** (`~/.cache/foundationpose/bundlesdf/nerf_runner.py`): multi-view
  RGB-D → neural object field → mesh, for objects you can't put a ruler on.

**Methodology lessons.**
- For a hand-measurable object, **ruler ground truth beats any single-image neural
  reconstruction**. Don't reach for SAM 3D / NeRF when calipers will do.
- The offline replay harness is **blocking** (send frame, wait) — not representative of the
  live **non-blocking** sidecar rate. Measure live latencies separately (done above).
- A capture bug cost real time: a tight `read_color_and_aligned_depth` loop returned **stale
  duplicate frames** (only 16 unique of 385). Fix: single read path + hard `.copy()` (the
  background thread reuses its buffer) + a self-check on unique-frame count.
