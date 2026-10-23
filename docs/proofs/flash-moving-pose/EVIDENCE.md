# Moving-object pose check — 2026-10-23

[Watch the phone-friendly gizmo video](https://github.com/TheWisp/lerobot/releases/download/flash-visible-tracking-20261023/FLASH_moving_objects_pose_gizmo.mp4) · 27 s, 1080 × 1440, H.264. The [earlier triangle view](https://github.com/TheWisp/lerobot/releases/download/flash-visible-tracking-20261023/FLASH_moving_objects_pose_check.mp4) remains available.

![Dowel rotation after the final local reseed](cover.jpg)

This follow-up deliberately shows objects that move: the dowel during a continuous
gripper-rotation sequence in the insertion demo, the yellow block pushed on the
tray, and a gamepad carried by the gripper. The earlier
[montage](../flash-visible-multiscene/EVIDENCE.md) included easier, nearly
stationary objects and excluded the yellow block because its depth check was
weaker. That selection made the tracker look more reliable than the evidence
supports.

In the video, the left close-up shows an inferred rigid fit from all available
mask/depth-supported TrackEverything IDs, including A/B/C when supported. Its
red X, green Y and blue Z axes start at landmark A: X follows A→B, Y follows
the component of A→C perpendicular to X, and Z is their cross product. The
right close-up shows the same three raw
point IDs without the fit. **A/B/C are display probes, not the three inputs to
the left-hand fit.** The white outline is the current SAM3 mask. Hollow raw
markers no longer pass the mask/depth support check. The axes describe a
local frame only _within one tracking segment_. After a reseed, A/B/C are
newly chosen points and their physical identity is **not** transferred.
The headings show lost tracking, unverified fits and unlinked reseeds rather
than silently filling those intervals. Each source frame is shown twice at
30 fps; this is half-speed playback, not measured live tracking speed.

| Moving object                            | Source and interval                                                                | Observation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ---------------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dowel during gripper rotation            | Insertion demo frames 120–370, shown without cuts                                  | Three local fits cover 120–172, 217–259 and 267–370. No pose is available for 173–216 or 260–266. At the end of the first interval and start of the third, all three selected raw anchors lack visible-surface support, so those fits are marked unverified. After visibility returns, the third fit follows the rotating dowel locally; this does not connect its material frame to the first two intervals.                                                                                                                                                         |
| Yellow Lego block moving beside the ring | Insertion rollout `20261016_112149`, 75 available frames                           | A fit exists in all 75 frames and follows the moving block in the image. The projected source surface agrees with both mask and depth for 87.7% of its samples in the median frame, falling to 57.3% in the worst frame. The fraction projected implausibly in front of depth reaches 22.8%. Some raw IDs leave the block for the gripper; the fitted pose remaining near the block is not proof of exact orientation or material identity.                                                                                                                           |
| Held gamepad                             | Pick-and-place rollout `20261012_194250`, source frames 160–240 (staged as 90–170) | One pass seeded before the carry loses usable tracking through occlusion. Three separate local passes in staged frames 90–119, 120–144 and 145–170 show recovery when surface becomes visible. They do **not** establish one continuous object frame. In the 120–145 overlap of two passes, the transform between their frames changes by 41.3 mm median (77.5 mm maximum) and 10° median (23.5° maximum) relative to its initial value. In the later 145–170 overlap, the corresponding changes are 27.9 mm median (33.2 mm maximum) and 4.5° median (5.9° maximum). |

The yellow-block figures measure rendered source-surface consistency, **not**
independently measured 6D pose error. The depth conflict can arise from a bad
fit, a changed visible face, or measurement artifacts; these data do not yet
separate them. The gamepad overlap test is stronger evidence that the current
reseeding procedure does not preserve a stable frame: correctly registered
rigid fits would have an approximately constant transform between their local
frames during overlap. The video shows reseeding as it currently works—start a
new local frame—rather than suggesting that the transition is solved.

### Drift audit after the gizmo review

The recorded measured gripper poses provide an independent rigid-grasp motion
prediction for the dowel. Camera capture times were aligned to the recorded
motor timestamps, and the saved camera-to-base calibration converted the fitted
camera-frame landmarks into the gripper frame. Frame 280 starts after the
unverified reseed and has all three selected raw IDs on the measured surface.
Over demo frames 280–370, the
gripper moved 40.8 mm and rotated 61.2°. If A were rigidly attached to it, its
gripper-frame position would be constant; instead, the inferred A changed by
17.9 mm from first to last. The complete A/B/C gizmo changed by 31.3°, but its
X direction changed by only 4.9°: much of the angular disagreement is twist
around the dowel's long direction. Shifting the camera-to-motor alignment by
up to 100 ms in either direction still left a 13.9–25.1 mm first-to-last A
discrepancy. These are **gripper-relative discrepancies**, not ground-truth
object-pose errors: actual slip at the grasp remains possible.

The point tracks themselves also fail a rigidity check that does not depend on
the gripper. Among 18 dowel IDs passing both frame 280 and frame 330 mask/depth
checks, pairwise 3D distances changed by 3.9 mm median and 9.2 mm at the 90th
percentile. A rigid object's material-point separations cannot change. The
current support check only verifies that a predicted point projects inside the
SAM3 mask and within 10 mm of aligned depth; it cannot establish that the ID
still marks the same piece of surface. Adjacent fits accept local agreements
within 5 mm and compose them, so gradual correspondence drift can accumulate
even when every link has enough inliers. This proves some tracking/depth error,
but does not allocate the whole gripper discrepancy between that error and
possible slip.

The fit is not simply dropping A away from its supporting cloud. For the 13
dowel IDs supported at both frames 280 and 370, A-to-ID distances changed by
2.0 mm median and 3.2 mm at the 90th percentile; only five of those IDs remain
supported in every intervening frame. Thus A largely follows the selected
point group while the group as a whole moves relative to the gripper. The
17.9 mm gripper discrepancy could be actual movement in the grasp, coherent
correspondence drift, or both. Accumulated adjacent-fit error is a mechanism
to test, not a demonstrated explanation for that full discrepancy.

The yellow block makes the distinction visible. At frame 58, raw display ID A
has migrated onto the held dowel beside the gripper and is rejected by the
yellow-block mask/depth check; C is also rejected. Only B of the chosen three
still passes. Their raw positions therefore must not be read as an observed
object frame. The left panel nevertheless has a rigid fit from the available
point bank: 14 inliers among 32 surface-supported points at that frame. The
accepted pose is not an independently verified six-axis orientation. Even
among the 31 IDs passing the surface check in both frames 0 and 58, pairwise
separations changed by 3.8 mm median and 7.9 mm at the 90th percentile. Across
the 75-frame interval, its worst projected source-surface agreement with mask
**and** depth is 57.3%, while foreground-depth conflict reaches 22.8%.
The original A/B/C selection spans a non-degenerate triangle (A–B 53.2 mm,
perpendicular C offset 31.7 mm); its later failure is loss of correspondence,
not an initially collinear gizmo. Replacing these three display landmarks
would change the drawing, but not the current all-point fit. Changing the
number and spatial coverage of _tracked query points_ can affect that fit;
count alone cannot turn a wrong identity or an underconstrained visible face
into a valid material frame.

TrackEverything is evaluated offline and may use future frames within its
inference window. This experiment does not establish a causal or real-time
tracker. The script that renders the saved results is
[`benchmarks/flash_moving_pose_video.py`](../../../benchmarks/flash_moving_pose_video.py).
Its inputs are local recordings and saved tracker outputs on the rig; they are
not committed to git. The dowel local-fit result is
`/tmp/flash-moving-check/dowel_rotation_segments.npz`; its segment boundaries
are in the adjacent JSON. The yellow scene is
`/tmp/flash-multivisible/ring_yellow/scene.npz`, and the gamepad local runs are
`/tmp/flash-moving-check/held_gamepad/reseed{090,120,145}.npz`.

### Implementation isolation and deterministic-fit correction

The video above predates this correction. The audit separates three effects:

1. **The supported raw points are not rigid.** Across dowel frames 280–370,
   pairwise distances between the 13 IDs supported at both endpoints change by
   4.73 mm median / 9.16 mm p90. Even the five continuously supported IDs deform:
   6.89 mm median pair-distance change. This occurs before any landmark fit.
   Replacing predicted depth with the measured depth at the same projected
   pixel reduces the 13-ID figure only to 4.12 mm. On the yellow block it
   worsens the equivalent figure from 3.00 to 3.36 mm. Simply snapping depth
   does not establish correct material correspondence.
2. **The point-bank fitter injected new randomness on every frame.**
   `VisibleRigidTracker.update` passed its incrementing frame count as the
   RANSAC seed. On the _identical_ yellow-block frame-68 coordinates and map,
   75 different seeds produced 13–18 inliers and orientations up to 61.58°
   apart. No image, depth, point or object movement was involved. It now uses
   registration's deterministic sampling, so unchanged inputs give unchanged
   output. This removes artificial switching, not the underlying ambiguity.
3. **Surface refinement can discard part of the point consensus.** At yellow frame
   50, ICP replaced 24 accepted material-point matches with 12 and changed
   orientation by 19.37° relative to the point-only fit. The consensus check
   requires the replacement to satisfy `min_inliers`. Requiring every old
   inlier to remain within tolerance was tested, but **not retained**: the
   rule can protect drifting material tracks at the expense of better
   measured surface agreement. Neither source proves the full material frame.

The frozen-frame regression includes both a no-motion case and a translated
case; determinism must not freeze real motion. Restoring the frame-dependent
seed makes that regression fail. No new fitting tolerance was introduced.
The existing surface test called 13→12 inliers "keeps point consensus"; it
establishes that a minimum consensus survives, **not that the original material
IDs or orientation survive**. That distinction is now explicit in its name.

The yellow recording still exposes unresolved disagreement among point IDs.
With the old per-frame sampling and no surface correction, frame 68→69 changes
the absolute point-bank orientation by 52.15° while an adjacent-frame fit to the raw points rotates
only 0.34°. Only eight of the 16 accepted IDs are common to the two absolute
fits. Each frame can select a different locally agreeing subset of a deformed
map. Chaining the adjacent fits is smoother here, but the dowel audit shows
that chaining does not establish long-term material accuracy either.

Deterministic sampling was replayed on the same 75 yellow-block frames. Median
orientation step falls from 3.03° to 2.82°, p90 from 28.61° to 15.44°, and the
worst from 59.26° to 52.32°. All 75 frames still produce a pose. Median
surface agreement remains similar (87.72% → 87.44%); median foreground-depth
conflict improves from 6.28% to 5.67%, though its worst frame worsens from
22.83% to 24.78%. The step metrics measure frame-to-frame discontinuities,
**not independently measured pose errors**.
Large jumps remain; this is not a solved rigid-tracking system.

The [rejected stricter surface experiment](yellow-strict-consensus-experiment.json)
reduced median orientation step further, to 1.79°, but reduced median
mask/depth agreement to 75.67% and raised median foreground conflict to 11.06%.
Smoother output alone was insufficient evidence to keep it. Surface refinement
therefore retains its previous behavior; source disagreement remains unresolved.

There are two further implementation limitations, left explicit rather than
covered by new thresholds:

- SAM masks seed and post-filter query tracks; they are **not inputs to
  TrackEverything's inference**. The filter verifies surface membership and
  depth proximity, not persistent identity. Invalid IDs are withheld from the
  fit, not removed from model memory, and may be admitted again later.
- Growth selects the globally largest uncovered later mask and then only
  considers frames after that birth. Thus the old dowel queries born at frame
  100 get no new companions until frame 361. A new offline inference seeded at
  frame 280 increased the minimum supported count over 280–370 from 13 to 41
  and reduced endpoint pair deformation from 4.73 to 3.05 mm, but the same A
  still disagreed with rigid-gripper prediction by 17.91 mm versus 17.87 mm.
  Coverage improved; the gripper-relative discrepancy remains unassigned
  between tracking error and an inaccurate rigid-grasp assumption.

The wrapper's query order `[frame, x, y]`, metric-depth input and normalized
3D output conversion agree with the installed upstream implementation. The
checks above do not prove that upstream tracking is optimal. They establish
that downstream randomized selection and refinement can add instability, and
that the remaining input tracks fail rigidity before that processing.

Reproduce the saved-input audit using
[`flash_visible_drift_audit.py`](../../../benchmarks/flash_visible_drift_audit.py).
The [dowel report](dowel-drift-audit.json) and [yellow report](yellow-drift-audit.json)
record the source paths, frame spans and existing tolerances. For either report,
pass its `scene`, `recording`, `mask_dir`, `role`, and frame limits as CLI
arguments, plus `--min-inliers 12 --trials 120 --output /tmp/audit.json`.
Add `--compare-surface` for the full yellow scene to compare saved poses with
the current fitter, and `--consensus-frame 68 --seeds 75` to repeat the frozen
point-bank experiment. The frozen frame's original coordinates are also saved
in `tests/policies/flash/fixtures/yellow_frozen_consensus_20261016.npz`.
Source images and tracker inference outputs remain local; no demo poses were overwritten or robot run
launched during this audit.
