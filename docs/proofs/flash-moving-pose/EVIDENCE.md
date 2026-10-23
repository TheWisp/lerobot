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
