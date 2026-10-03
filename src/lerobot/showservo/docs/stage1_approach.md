# Stage 1: the calibrated approach

One demonstration teaches where the gripper should be relative to an object.
Later, with the object somewhere else on the table, the arm goes there. This
document is the design for doing that with the calibrated arm and the top
camera, closed on the object and open on the tool, and for measuring how well
it works before anything is built on top of it.

## Requirements

- One demonstration: the operator names the object, the camera captures it,
  the operator jogs the fingertip to the pre-grasp (and later the grasp) and
  marks it. Nothing else is recorded.
- At run time the object may be translated and turned anywhere on the table,
  among other things, and may move slowly during the approach.
- The arm ends at the taught pose relative to the object, within what a grasp
  tolerates: a few millimetres and a few degrees.
- Every run reports its evidence: how the object was found, how sure the find
  is, and how far the arm ended from where it should be.

## Observations (2026-09-20, white left arm, RealSense D435 top camera)

- With the gravity feed-forward, the fingertip measured from the wrist link,
  joint zeros corrected by 5.3, 7.0 and -2.8 degrees (lift, elbow, wrist
  flex) and the camera pose fitted against touched marker corners, the
  camera-to-fingertip chain lands within a few millimetres: 4.3 mm rms on the
  calibration corners, and "a few millimetres high at the furthest corner" by
  eye on the go-to-marker test.
- A pre-grasp taught on a plain green ring, transported by the ring's depth
  blob (centroid shift) after the ring was moved, landed correctly.
- The ring gives three SIFT keypoints: uniform surfaces have no texture to
  match. Shape from depth found it; texture would not have.
- The residual that no calibration removes is the servo stiction band, plus
  or minus 2 to 3 mm, which flips with the direction of travel.
- The earlier package (`lerobot.showservo`) has designation by SAM3 concept,
  DINO patch features, a KLT tracker with a forward-backward gate, and a
  certificate that says when a fit is real. Its executor assumed the arm's
  model could not be trusted; that assumption no longer holds.

## Observations (2026-10-01, same rig, nothing moved between finds)

Twenty captures of one static, cluttered tray; every find compared with the
first capture. Numbers in appendix A.

- The certified DINO fit repeats the object's centre within 0.2 mm on a cube
  and 0.6 mm on a ring and a cylinder. Its rotation repeats within a degree or
  two, but about an axis that is effectively random: 60 to 80 degrees from
  the face normal. A large turn measured about such an axis is the sideways
  gripper of 2026-09-21. The fit's turn is usable; its axis is not.
- The face normal from the depth cloud repeats within 1.5 to 2.6 degrees on
  average and 4.8 degrees at worst. Under the table prior a measured tilt
  below that is noise and is dropped; above it the object has really tipped.
- One plane fitted to the whole mask found no usable face on a cube, a
  cylinder or a plain yellow block, because their masks include side faces.
  A consensus plane, with a test that it holds at least twice the points of
  the next plane, found the top face of all four objects. The face is the
  largest plane in the mask, not the mask's mean plane.
- Two of twenty finds of the yellow object were certified on eight and nine
  inliers of a 399-point card and would have moved the gripper 100 mm. A
  certificate needs a floor relative to the card, not an absolute handful.
- A sticker corner touched at calibration re-detects within 0.5 px eleven
  days later. The stickers can stand in for the camera-moved check.
- The depth-only shape find on a plain object repeats within 0.8 mm and 0.2
  degrees in the dark; per-pixel depth on a top face varies 0.2 mm between
  medians of five frames.

## Observations (2026-10-02, model-free live tracking: what others do)

The question was whether the hand-built tracker (designation, DINO patch
matching, a rigid fit, a growing card) is the wrong shape, and what the
current work on live 6-DoF tracking of an unknown surface looks like. Of the
published systems, four are relevant and one is runnable:

- Point2Pose (MIT, ECCV 2026, BSD-3, code public): model-free RGB-D tracking
  of several unknown rigid objects from one to three clicked points. Its
  correspondences come from a learned long-range 2D point tracker
  (BootsTAPIR by default; TAPNext++, Track-On2, LiteTracker and CoTracker3
  are drop-ins), not from per-frame matching; the pose is a sequential
  RANSAC and SVD against points it keeps across frames, with a TSDF of the
  object built while tracking. A fully occluded object is re-localised the
  frame it reappears because the tracks persist. Known weakness per the
  authors: textureless surfaces.
- TrackEverything (arXiv 2609.30222, no code): de-duplicated 3D point tracks
  in world coordinates with a static/dynamic split. The idea to take is the
  persistent, world-anchored track, which Point2Pose already has in object
  coordinates.
- BundleSDF, NeuralFeels, 6DOPE-GS, UA-Pose: refine a neural model of the
  object from partial views over time; all need seconds per frame or an
  offline pass and are not candidates for the loop.
- CoTracker3 online, SpatialTrackerV2, Track-On2: the point trackers
  themselves; usable inside Point2Pose's interface rather than beside it.

Point2Pose runs on the rig in its own environment (`~/.cache/point2pose`,
Python 3.11 and a cu128 torch for the 5090 instead of the authors' cu121
pin; the SAM2 fork's CUDA extension is built out because the system `nvcc`
predates Blackwell) and is the live tracker's fifth algorithm, `p2p`
(`benchmarks/p2p_bridge.py`, NPZ over a pipe). The tracker built in this
repository is called **PatchFit** from here on: SAM3 designates the object by
name, DINO patch descriptors are matched frame by frame, a RANSAC rigid fit
gives the pose, and a growing card holds the object's known points. Measured
against PatchFit on a
20 s recording of the static tray (342 frames, cube at the teach spot, the
same synthetic occluder painted into both; `captures/.../seq_static20`):

|                                             | PatchFit (DINO window)      | Point2Pose, authors' live config | Point2Pose, 60 points |
| ------------------------------------------- | --------------------------- | -------------------------------- | --------------------- |
| at rest, centre vs frame 0, mean / max      | 0.25 / 0.59 mm              | 0.92 / 2.03 mm                   | 0.88 / 1.88 mm        |
| at rest, rotation vs frame 0, mean / max    | 1.6 / 4.0 deg               | 2.6 / 6.3 deg                    | 2.3 / 4.8 deg         |
| half the cube hidden for 120 frames, centre | 0.39 / 1.06 mm              | 2.37 / 6.40 mm                   | 1.00 / 3.00 mm        |
| half hidden, rotation                       | 2.4 / 5.9 deg               | 7.0 / 20.5 deg                   | 3.0 / 10.0 deg        |
| whole cube hidden for 61 frames             | lost 61, back within 0.6 mm | lost 61, back within 1 mm        |                       |
| worker time per frame, median               | 62 ms                       | 71 ms                            | 71 ms                 |

Live on the same spot both run at 13.5 fps; the centre jitter is 0.2 to 0.5
mm for PatchFit and 0.8 to 1.5 mm for Point2Pose. So at rest and under a static
partial occluder the hand-built tracker is the tighter of the two, and the
instability seen live is not the matching: the same code over the same
camera's frames held every one of 342 frames. What Point2Pose brings is the
persistent track: it never reports an object lost while it is in view, and
it does not need a designation to re-acquire. Whether that wins once the
object moves, turns and is occluded by a hand is not measured; the two 20
and 90 s recordings made for it had nothing moving.

## Observations (2026-10-02, evening: both trackers against a ground truth)

YCBInEOAT (BundleTrack's benchmark: YCB objects manipulated by a robot arm
under a static RGB-D camera, 6-DoF poses annotated per frame, the set
Point2Pose reports on) is the first measurement of either tracker against a
truth under motion, turning and occlusion. Protocol as in the papers: the
tracker's motion since frame 0 is applied to the true pose of frame 0 and
scored by ADD-S against the YCB mesh, AUC over 0 to 10 cm; a frame the
tracker did not certify holds the last certified pose. Both trackers start
from the dataset's mask of frame 0; PatchFit re-designates by a SAM3 concept
("yellow box" for the sugar box, which "sugar box" never found). ADD-S AUC in
percent, with certified frames; the "yalehand" videos turn the object inside
a soft hand that covers most of it.

| video                            | motion                                         | PatchFit, as committed                   | Point2Pose, demo config | Point2Pose, published config + SAM2 | paper |
| -------------------------------- | ---------------------------------------------- | ---------------------------------------- | ----------------------- | ----------------------------------- | ----- |
| mustard0, 737 frames             | picked, lifted, turned 90 degrees, set upright | 89.2 (646), memory 84.6 (645)            | 94.1 (737)              | 95.3 (736)                          | 95.3  |
| mustard_easy_00_02, 689          | picked and placed                              | 91.7 (689)                               | 73.5 (684)              | 91.9 (678)                          | 95.7  |
| cracker_box_reorient, 375        | lifted and stood up                            | 89.9 (371)                               | 90.1 (375)              | 93.1 (375)                          | 96.4  |
| sugar_box1, 907                  | picked, turned, placed                         | 82.7 (829)                               | 92.4 (868)              | 95.3 (907)                          | 94.3  |
| bleach0, 663                     | pick and place                                 | 53.8 (248)                               | 25.5 (269)              | 31.0 (296)                          | 82.9  |
| bleach_hard_00_03_chaitanya, 441 | pick and place                                 | 91.3 (441)                               | 69.8 (298)              | 92.0 (438)                          | 93.9  |
| tomato_soup_can_yalehand0, 1308  | turned inside a soft hand                      | 57.4 (816), detect each frame 51.9 (172) | 74.3 (1308)             | 87.1 (1266)                         | 95.5  |
| cracker_box_yalehand0, 1327      | turned inside a soft hand                      | 87.6 (1254)                              | 40.7 (529)              | 93.5 (1299)                         | 92.5  |
| sugar_box_yalehand0, 1002        | turned inside a soft hand                      | 87.3 (983)                               | 40.4 (279)              | 86.6 (999)                          | 87.6  |
| mean                             |                                                | 81.2                                     | 66.8                    | 85.1                                | 92.7  |

The paper's mean rests on its full configuration (cluster RANSAC with TSDF
refinement, a 20-frame local graph, 25 points a keyframe at 480 px) and, for
the masks, the dataset's own; its live configuration (SAM2 from the first
mask, the simple register, 30 points a keyframe) reproduces it on the easy
videos and loses the object under heavy occlusion. The authors' configuration
with SAM2 holds there (cracker in hand 93.5, tomato 87.1) at two to three times the cost per frame, 0.4 to 1.5 s offline under contention; standalone on the rig's own 848 by 480 frames with the GPU idle, 44 ms a frame for the live configuration and 99 ms for the authors', so the robust one would still run near 10 fps in the loop. bleach0 defeats both (31.0). A middle configuration (their register, criterion and
sampler at the live tracker's resolution, no local graph) gained little
(tomato 78.1) for twice the cost.

What the mustard video showed about PatchFit, and what changed, in the order it
was measured:

- Through the pick and the lift both of our modes held the bottle. As it was
  set upright the fit flipped by 160 to 178 degrees and certified on a hundred
  matches: the bottle's two sides look alike to the descriptors and nothing
  tied one frame to the one before. Point2Pose never flips because its
  correspondences persist across frames. The rigid fit now takes the recent
  certified motions as references and never selects a candidate beyond what a
  rigid body could have reached from each: 30 degrees plus 60 degrees a second
  of elapsed time, which reaches 180 at 2.5 s and drops the reference. A bound
  against the last frame alone was walked round in four certified steps of
  under 50 degrees; against every recent frame no frame of 737 flipped, with
  91 held as occluded instead (ADD-S AUC 75.3 without the bound and 89.2 with
  it, at the same growth cadence; the window mode went from 79.5 to 85.2 and
  still flipped in 68 frames, since its window follows whatever pose it last
  certified). The bound stops a jump, not a slide: with the designation from
  memory the same bottle still walked from 43 to 115 degrees in certified
  steps of under 30, each within reach of the frame before. Per-frame
  matching on a self-similar surface has a continuum of locally supported
  wrong poses, and kinematics alone cannot tell a slide from a brisk turn.
- The card grew once per wall-clock second, which made an offline replay
  nondeterministic and, at the live loop's rate, too slow for an object being
  turned: once a second of video the card stopped at 743 points and the fit
  decayed to nine inliers. Growth now runs on the frames' clock, every 0.3 s
  or whenever the object has turned 10 degrees since the last growth.
- The descriptor crop follows the designation's bounding box, so a
  designation that shows only part of the object shifts the patch scale and
  the card stops matching. The acquisition crop is now sized from the card's
  own extent and the designation's depth and only placed by the designation.
- SAM3 designated the soup can inside the soft hand in one frame of ten at
  any inference resolution, so detection by name each frame is blind there.
  The designation is now carried forward from the segmenter's memory between
  frames of a track, detecting by name only to seed and to recover, as SAM2
  does inside Point2Pose: the can is certified in 816 frames instead of 172,
  and the two other in-hand videos, run only this way, score 87.6 and 87.3
  where Point2Pose's live configuration scores 40. On the mustard bottle the
  memory's masks cost 4.6 points through the slide above; one constant
  (`DESIGNATE_FROM_MEMORY`) chooses.

What this says for the bench (completed 2026-10-03): Point2Pose's published configuration averages 85.1 ADD-S AUC over the nine videos, PatchFit 81.2 and Point2Pose's demo configuration 66.8; the published one beats PatchFit on seven videos, ties the sugar box in a hand and loses only bleach0, which SAM2 loses for everyone. It is the tracker now, at 99 ms a frame on the rig's frames; the DINO modes stay in the menu as comparisons. The published configuration has no local graph: what it adds over the demo one is cluster RANSAC refined against the TSDF it builds, stricter point sampling, keyframes every 10 degrees and the point tracker's full refinement at 480 px. PatchFit's remaining failure is the slide on a self-similar surface, which no per-frame matcher can rule out by kinematics; Point2Pose's is the roll of a thin object about its own axis under sparse points.

## Observations (2026-10-03, the server's pose policy against ground truth)

The benchmark above scored the tracker's raw fit. The live view never showed
that: the server composed a pose on top of it — the resting prior (a turn
about the measured surface normal), the axis from the object's face, and
turn rules that chose between the fit's turn, the depth footprint's and, for
a morning, a long axis's. Replaying the same nine runs through that policy,
ADD-S AUC:

| video                       | PatchFit raw fit | prior on, rules on (the live default until today) | prior on, rules off | prior off, rules on | prior off, rules off |
| --------------------------- | ---------------- | ------------------------------------------------- | ------------------- | ------------------- | -------------------- |
| mustard0                    | 84.6             | 59.3                                              | 50.8                | 77.2                | 84.5                 |
| mustard_easy_00_02          | 91.7             | 55.5                                              | 56.6                | 83.2                | 77.6                 |
| cracker_box_reorient        | 88.0             | 53.4                                              | 53.4                | 63.5                | 63.5                 |
| sugar_box1                  | 82.7             | 27.5                                              | 27.1                | 82.7                | 82.7                 |
| bleach0                     | 53.8             | 22.1                                              | 21.7                | 53.8                | 53.8                 |
| bleach_hard_00_03_chaitanya | 91.3             | 11.2                                              | 11.2                | 53.3                | 53.4                 |
| tomato_soup_can_yalehand0   | 57.4             | 35.4                                              | 35.5                | 57.4                | 57.4                 |
| cracker_box_yalehand0       | 87.6             | 67.9                                              | 68.1                | 65.7                | 74.6                 |
| sugar_box_yalehand0         | 87.3             | 64.5                                              | 64.5                | 64.4                | 62.3                 |
| mean                        | 80.5             | 44.1                                              | 43.2                | 66.8                | 67.8                 |

Every video lifts its object, so the prior is wrong there by construction;
but with the prior off the face axis and the turn rules still never beat
the fit on a single video. On the videos' at-rest openings, the object
still on the table before the gripper arrives, the prior's own regime, mean
ADD in mm and median rotation error in degrees, raw fit against the live
default: cracker_box_reorient 10.4 / 5.1 against 9.4 / 6.2; mustard0 1.4 /
1.7 against 5.8 / 2.8; sugar_box1 1.8 / 0.9 against 1.9 / 1.6; mustard_easy
1.1 / 0.9 against 1.1 / 0.3; bleach_hard 5.9 / 0.8 against 6.5 / 0.6;
bleach0 3.1 / 1.7 against 65.8 / 115.3; tomato can 1.7 / 1.9 against 16.1 /
12.7. A wash on four, a slight gain on one, and twice the prior turned a
still object. No video has an object slid or turned while kept on the
table, so that case rests on these two ends.

The pose is therefore the fit, for every algorithm; the face tilt is
reported; the turn rules are deleted; the resting prior is an opt-in, off by
default and marked unvalidated. The live path replayed over the nine runs
reproduces the raw-fit column exactly. What this does not change: the fit's
own weaknesses, the slide on a self-similar surface and the roll a line of
points cannot pin down (a USB stick under Point2Pose rolled 140 degrees
about itself between frames while lying still), which no composition fixes
and a dense, model-based registration would.

## Observations (2026-10-03, evening: the first act on the real arm)

The guided flow ran end to end on the gamepad: taught by a click, tracked by
Point2Pose, a leader demo of 613 samples over 20.5 s (the object seen in 58 %
of them: the arm covers it from 5.8 s to 14.3 s), saved, the object then
moved by hand, Act at half speed. The tracker reported the move as 60.5 mm
and 22.3°, 22.4° of it a turn about vertical. The operator stopped the act
during "to the start": the arm "twisted itself to the side".

What the arm did was right by the formula and wrong in principle:

- The arm stopped 1.7 mm from the transported start pose, 22.6° about the
  vertical from the demo's start. The target was exactly the carried pose.
- Its joints were nothing like the demo's: shoulder pan 59° against 0°,
  forearm roll −49° against 5°, wrist roll 95° against −4°. Simulated offline
  with the jog's own kinematics and calibration (Pink's QP, the same walk),
  the walk from every starting configuration tried (the demo's start, its
  end, the ready pose, a raised pose) ends in those same joints: the pose
  requires them; no branch was chosen badly.
- The reason is the demo's start pose: the gripper was tilted 42° from
  vertical and 80 mm from the grasp. Carried by a 22° turn about the object,
  a tilted pose far from the turn's axis demands the forearm roll. The grasp
  itself was vertical (its approach axis 6° from plumb) and transports
  cleanly: solved from the demo's own joints it needs 4° more pan and 17°
  more wrist roll.
- The replay walked Cartesian targets through the differential IK from
  wherever the arm stood; the recorded joints were written but never read.

Conclusions, each the operator's point before the numbers were in:

1. Not every point of a demonstration is relative to the object. For a
   pickup the operator marks the pre-grasp point or points and the end of the
   grasp. The arm goes to each pre-grasp in a straight line, starting from
   wherever it stands, then replays the demo from the last pre-grasp to the
   grasp's end exactly, moved and turned with the object, and holds. The start
   of the recording is never replayed.
2. A drop-off is a later stage, relative to the desk or to another object,
   and is not built. Moving from one stage to the next needs vision, and later
   touch, to confirm the grasp; the demo's clock cannot.
3. The act is planned in joint space before anything moves: every sample is
   solved by IK from the one before, starting at the arm's present joints, the
   grasp's samples seeded with the demo's own joint change. A point out of
   reach, a sample below the table, or a jump between samples refuses the act
   by name. The plan streams as joint targets, as the leader handover does.

A first editor the same evening marked moments anchored to the object or to
the world and blended the correction between them, with marks suggested from
the gripper channel. The operator could not tell what an anchor applied to,
and the suggestions read the gripper backwards: on this arm a higher reading
is more closed, so the most-open moment was taken for the grasp. Both are gone.

## The transport

One demonstration fixes one invariant: the fingertip's pose relative to the
object. At run time the object's rigid motion carries that pose:

```
goal = new_centre + turn · (marked_gripper − old_centre)
```

with the goal's orientation the marked orientation rotated by the same turn.
The four terms are in the base frame: the marked gripper pose from forward
kinematics, the object's centre at teach and at find from the camera through
the calibration, and the turn measured in the camera frame and conjugated
into the base frame (`R_base = R_bc · R_cam · R_bcᵀ`). No object frame is
defined: the motion `T_new · T_old⁻¹` is the same for any frame glued to the
object, which is why the camera measures a motion and not a pose.

The error budget follows from the formula (appendix B). Anything constant
cancels between teach and find for a pure translation. What grows is
anything multiplied by the object's displacement (the calibration's rotation
error) or by the lever arm from the object's centre to the fingertip (the
turn's error, in angle and in axis).

**Superseded in part (2026-10-03).** The formula stands for the moments that
are relative to the object. Applied to a whole demonstration it carries the
recording's start and its drop-off along with the grasp, and a tilted start
pose far from the turn's axis then demands an arm configuration the human
never showed (observations above). Which moments the transport applies to is
the operator's mark on the demo; between marks the correction blends.

## Constraints

- The general case is a rigid motion in six degrees of freedom: the object
  may end up on a block, tipped over, or held. The table plane is a prior the
  find may use when the object is on it, never an assumption the design
  depends on: a plain object on the plane is fitted with three degrees of
  freedom because that is what its depth can support; anything the depth and
  descriptors can pin down in six is fitted in six.
- The camera does not move between calibration and run. If it does, the
  calibration is stale and the run must say so rather than proceed.
- The find must never be silently wrong: an abstention with a reason is
  always preferred to a confident bad transform.
- Same instance for now: the object at run time is the object that was taught.

## Architecture

### Teach

1. **Designate** by SAM3 concept ("green ring"); a drawn box is the fallback
   when no GPU is available. The mask is the object.
2. **Model** the object as every depth point under the mask (not a few
   keypoints), plus SIFT and DINO descriptors where the surface has any, plus
   its height and footprint above the table plane. The plane is fitted to the
   environment in a ring around the object.
3. **Mark** the pre-grasp: fingertip pose from FK in the base frame, and the
   wrist image at that moment (for stage 2). A second mark for the grasp
   records the descent as a move relative to the pre-grasp.

### Find

1. Same designation on the new frame.
2. **Register on the whole cloud**, six degrees of freedom when the evidence
   supports it (descriptor matches, a footprint with a direction, a height
   profile), and with the plane as a prior when the object sits on the table
   and the evidence is thin: a plain object on the plane gives a stable
   three-degree answer where a free six-degree fit would wander. The fit
   reports which it used. A round footprint has no measurable turn and gets
   none.
   **Superseded 2026-10-03.** The plane prior and every turn rule were
   replayed against ground truth and lost to the raw fit on all nine
   YCBInEOAT videos (observations below); the pose is now the fit, and the
   prior an opt-in nothing has shown to help.
3. **Use the environment twice.** Background points must fit no motion, which
   certifies that the camera has not moved; any point that moves with the
   background rather than the object is evicted from the object, which is how
   clutter and a loose designation are handled. The cheap form of the first
   half is the calibration stickers: their pixel positions at calibration
   time are known, and a find whose frame shows them elsewhere refuses to go.
4. **The turn's axis from the face.** The fit's rotation angle is trustworthy
   and its axis is not (observations of 2026-10-01). The object's face toward
   the camera, the largest plane in its depth cloud, is measured at teach and
   at find; the motion is the smallest rotation taking the taught face onto
   the found one, followed by the fit's turn about the found normal. Under the
   table prior the face is reported but not applied: the table's normal is
   exact, from the calibration or from the depth around the object, and a
   measured face tilt on a resting object is the face's own noise. An earlier
   version dropped only tilts inside a deadband and applied larger ones; a
   rounded object's face wandered 18 degrees with nothing moving (appendix A,
   live sweep), which that version would have passed to the gripper.
   **Superseded 2026-10-03.** Replayed against ground truth, taking the axis
   from the face cost up to 35 points of ADD-S AUC and never gained; the face
   tilt is reported, the fit's own rotation is used.
5. **Certificate** with every find: inlier count, rms, similarity scale (a
   rigid object keeps its size; a scale off 1 flags a depth fault), the
   background check, and the taught-vs-found height and footprint.

### Track

Point2Pose is the tracker: started on the teach frame with SAM3's mask so
its first pose is the teach pose, carrying the mask forward with SAM2's
memory, tracking points with BootsTAPIR, and registering each frame by
cluster RANSAC refined against the TSDF it builds of the object. It runs in
its own process and steps on every frame. Four comparison algorithms stay
behind the same state machine (acquiring, tracking, occluded, lost): SAM3
and DINO every frame; DINO in a window around the last pose, SAM3 only to
acquire; KLT on the matched points; depth only. The transported pre-grasp
updates live; the jog's bounded walk follows it. A slowly moving object is
the same loop.

An experimental second Point2Pose mode, "dense", keeps the published
configuration but registers each frame by the whole visible depth surface
against the TSDF (every masked depth pixel, subsampled, fitted by Gauss-Newton
on the signed distance), seeded by the sparse answer and by the previous pose;
the previous pose alone seeds it when the tracked points cannot carry a fit, so
a hand over the tracked corners does not lose a visible object; and the
surface's agreement is what the map-growth gates judge a frame by, so a side
seen at a wrong pose is not fused into the model. The register lives on the
`lerobot-dense` branch of the Point2Pose checkout; the mode is benchmarked on
the nine videos before it is trusted on the bench.

### Execute

The pre-grasp and the grasp, as the operator marked them on the demo. From
the arm's present pose, a straight line to each pre-grasp in turn at the
walk's speed, the gripper first walking to that point's opening where the arm
stands; then, when a grasp end is marked, the demo from the last pre-grasp to
it, sample for sample on the demo's clock, with the recorded gripper command.
Every pose is the demo's carried by the object's motion. The speed scales
both. Before anything moves every sample is solved by IK, to 0.5 mm, from the
one before; the act refuses, naming the point, when one cannot be reached
within 3 mm and 3°, when a sample would go lower than the table floor or than
the demo itself went there, or when two samples would need a joint jump over
10°. The plan streams as joint targets; at the end the Cartesian walk takes
over where the arm stopped, still commanding the grasp's closing. The jog's
guards apply throughout: a joint that falls 25 degrees behind freezes the arm,
a motor over 60 C freezes it. The editor says, point by point, whether the
plan is reachable as the object lies now.

### Evaluate

Ten placements per object: translations, turns including a half-turn for a
round object, one with clutter beside it, one with the object nudged during
the approach. Each trial logs the transported motion, the certificate, the
arrival gap from the encoders, the wrist image at arrival, and the operator's
verdict. The miss is measured in the wrist image: the object's offset between
the taught wrist image (recorded at Mark) and the wrist image at arrival,
converted to millimetres with the taught standoff. The top camera cannot
measure it, since the arm occludes the object at the pre-grasp. The outcome
is a table.

### Tab

A new Approach tab holds everything from this design; the Servo tab keeps
the earlier pipeline as it was.

```
Approach
 ├─ Camera     the RealSense session
 ├─ Jog        one arm: gizmo (T move, R rotate), gripper (W close, E open), ready, park, recover,
 │             leader drives (demo mode), record demo
 ├─ Touch calibration   fingertip, camera, joint zeros
 ├─ Teach      designate (concept | box) → mark pre-grasp / mark grasp, or keyframes from the demo
 ├─ Run        find (certificate) · track live (four algorithms, arm follows) · go · run grasp · stop
 └─ Trials     one row per run: motion, certificate, what the arm was told, how it ended, verdict
```

## Decisions

- Designation by SAM3 concept by default (targets are clear, named objects);
  the drawn box stays as the no-GPU fallback.
- Six degrees of freedom is the general case, and the pose is the tracker's
  own rigid fit. The table plane is an opt-in prior, off by default: replayed
  against ground truth it never beat the fit (2026-10-03).
- Success is measured in the wrist image against the taught wrist image,
  plus the operator's verdict; no marker on the gripper, and nothing from the
  top camera after arrival, which the arm occludes.
- Everything new lives in its own Approach tab: camera session, jog with the
  gripper, touch calibration, pre-grasp. The Servo tab keeps the earlier
  pipeline unchanged.

## Beyond stage 1: a held object

The stage after this one moves an arbitrarily held object against a target
with one demonstration. During a firm grasp the held object is glued to the
gripper, so its demo trajectory is the gripper's own, from forward
kinematics; vision owes two motions:

```
goal(t) = target_motion · demo_gripper_pose(t) · regrasp⁻¹
```

The target's motion acts on the world side, as in stage 1. The regrasp
correction acts on the tool side and is the change in where the object sits
in the hand between the demo and now. The held object's model comes from the
depth points that follow the gripper's known motion, minus the gripper's own
silhouette rendered from the URDF, accumulated in gripper coordinates while
the wrist turns the object in view; the regrasp term is the registration of
the demo's in-hand model against the current one, and the object's tip can
be measured by touch with the tool-point solver on an arm that senses
contact. The far end's error is the regrasp rotation error times the
grasp-to-tip lever arm, so contact tasks close the loop on the relation
between held object and target in a camera that sees both, where calibration
and kinematic errors cancel to first order, and on force where the arm has
it: OpenArm2's quasi-direct-drive motors report torque from current, the
SO-107 does not. None of this stage is built.

## What is built and what is not

Built (2026-09-20): the calibrated executor (`lerobot.gui.api.jog`), the touch
calibrations and joint-zero refinement (`lerobot.gui.api.calib`), and a
pre-grasp teach-and-transport with a drawn box, SIFT for textured objects and
a depth blob for plain ones (`lerobot.gui.api.pregrasp`).

Built (2026-09-21): designation by SAM3 concept with DINO patch features and
the bench's certified 3D rigid fit, as the Teach/Find default, in a worker
process (`benchmarks/pregrasp_worker.py`) the GUI spawns and feeds by a
long-polled job queue (`/api/pregrasp/worker/*`); the drawn box with SIFT or
depth shape remains the no-GPU fallback; the gripper opening is part of the
taught pose.

Built (2026-10-01): the turn's axis from the object's face when the table
prior is off (a consensus plane in the worker, composed with the fit's turn
on the server) and from the table normal under it; a trust floor on
certified finds relative to the card; the camera-moved check from the
calibration stickers, refused at Go; the jaw line and approach arrow on the
teach and find images; the base-frame turn and lean readout; live tracking
of the taught object with four switchable algorithms (SAM3 and DINO every
frame, DINO in a window, KLT on the matched points, depth only), states
acquiring, tracking, occluded and lost, and the arm following the live pose.

Built (2026-10-03, night): the guided one-button flow on the Approach tab;
the demo editor (playback, fingertip path, gripper strip, pre-grasp points and
the grasp's end, saved beside the demo as `keypoints.json`); the act that goes
straight to the pre-grasps and replays the grasp 1:1, planned in joint space
and streamed as joint targets (`jog` mode `joints`). **NOT IMPLEMENTED.** The
drop-off and any stage after the grasp; confirming the grasp before moving on;
a second object as a frame (needs its own tracker); re-anchoring a loaded demo
when the object is re-taught by a click after the load (the demo's reference
is the teach it was recorded against); a successful act on the real arm under
this flow has not happened yet.

Live sweep (2026-10-01, window algorithm, eight objects, nothing moving,
about 30 frames each): every object stayed in the tracking state; centre
jitter 0.3 to 0.8 mm on the cube, ring, scissors and tape roll, 1.7 mm on
the cylinder, 3 mm on the rounded yellow object; face tilt noise 2 to 4
degrees on flat tops, up to 13 degrees on the hand and 18 on the yellow
object. The worker's time per frame was 50 to 70 ms, the loop 13 to 24 fps.

Built (2026-10-02): the resting prior as the surface the object rests on,
fitted to the depth around it in both frames and composed like the face
path, so a tilted table or a ramp is a measured normal and not an
assumption; the leader-arm demo (the jog's leader mode, the recording, the
keyframes from the gripper signal); the grasp keyframe by gizmo; the grasp
run (hover, pre-grasp, grasp, close until the gripper's reading holds
still, lift, following the live pose); the trials table with the
operator's verdict. The leader mode, the demo and the run had not yet
moved the real arm when this was written.

Built (2026-10-02, later): Point2Pose as the live tracker's `p2p` algorithm
through a pipe to its own environment, the sequence recorder
(`POST /api/showservo/record`, the layout the point-tracking benchmarks
read), and the offline comparison above. **NOT IMPLEMENTED:** the comparison
on a moving, hand-occluded object; the recordings for it exist only as
static scenes so far.

Changed (2026-10-03): the pose is the tracker's raw fit; the server's
turn rules are gone and the resting prior is an opt-in; Point2Pose is
anchored on the teach frame and kept across mode switches; the live view
draws the accumulated model and an object-frame triad. Point2Pose in its published configuration is the default tracker; the DINO modes are listed as comparisons. A click on the camera view teaches the object under it, SAM3 prompted by the point instead of by a name.

**NOT IMPLEMENTED:** whole-cloud registration (the feature path fits the
card's points in six degrees of freedom; the box path uses a centroid shift
plus a footprint turn for plain objects); the background no-motion check
and eviction (the sticker check covers a moved camera, not a loose
designation); canonicalising a demo's keyframes against the object's
tracked pose at the time, so a nudged object corrupts the demo; the wrist
image at mark; the stage-2 wrist servo.

## Appendix A: no-move repeatability (2026-10-01)

Twenty captures of one static tray, each the median of five depth frames,
light on, both arms parked in the corners of the view; each find against the
first capture, on branch `proto/show-and-servo` with the consensus face fit.
Translation is the displacement of the object's centre; the raw rotation is
the fit's; the composed values are after the face composition.

| object        | card points | inliers                  | raw centre shift, mean / max                        | raw rotation, mean / max | raw axis from face normal | composed turn, std / max | face tilt, mean / max |
| ------------- | ----------- | ------------------------ | --------------------------------------------------- | ------------------------ | ------------------------- | ------------------------ | --------------------- |
| green cube    | 376         | 298 of 300               | 0.08 / 0.16 mm                                      | 0.8° / 1.5°              | 81°                       | 0.12° / 0.43°            | 2.6° / 4.2°           |
| green ring    | 180         | 129 of 130               | 0.27 / 0.55 mm                                      | 1.5° / 4.2°              | 71°                       | 0.44° / 0.96°            | 1.5° / 4.6°           |
| blue cylinder | 383         | 277 of 285               | 0.14 / 0.62 mm                                      | 0.2° / 0.4°              | 76°                       | 0.05° / 0.11°            | 2.6° / 3.6°           |
| yellow object | 399         | 17 of 19 finds certified | two certified finds on 8 and 9 inliers: 100 mm, 45° |                          |                           |                          |                       |

Whole-mask plane fit, same scenes: face usable on the ring only (planarity
0.89); cube 0.59, cylinder 0.41, yellow 0.39. Consensus fit: all four usable,
planarity 0.57 to 0.87, dominance 2.3 to 6.8.

Depth-only, in the dark, on a plain 35 mm object: shape find translation
0.76 mm mean, 3.9 mm max over 20; footprint yaw 0.22° std, 1° max; per-pixel
depth 0.21 mm median between captures.

Sticker corner touched at calibration on 2026-09-20, re-detected
2026-10-01: 0.5 px away.

## Appendix B: error budget

For `goal = new_centre + turn · (marked_gripper − old_centre)`, on this rig.

| term                 | error source                         | how it reaches the goal                                                                                                                                                  | measured                                      |
| -------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------- |
| marked gripper       | forward kinematics at Mark           | cancels except for the difference in model error between the mark pose and the goal pose                                                                                 | 4.3 mm rms across the tray after refinement   |
| the two centres      | depth, mask coverage, calibration    | a constant offset cancels for a pure translation and reaches twice itself for a half turn; a mask covering a different part of the object shifts the centre by that much | 0.1 to 0.6 mm with nothing moved (appendix A) |
| turn, angle          | feature matches                      | error angle times the lever arm from the object's centre to the fingertip, plus the same angle on the jaws                                                               | 0.05° to 0.44° std with nothing moved         |
| turn, axis           | the visible cloud                    | an axis off by α on a turn θ is a rotation error of about 2·sin(θ/2)·α; 30° on a 90° turn was 42°                                                                        | face normal 1.5° to 2.6° mean, 4.8° max       |
| calibration rotation | camera orientation in the base frame | grows with displacement (1° per 57 mm of travel) and with the turn by the same rule                                                                                      | within the 4.3 mm corner residual             |
| execution            | servo stiction                       | adds directly, direction dependent                                                                                                                                       | ±2 to 3 mm on the go-to-marker test           |
| fingertip model      | the tool point                       | adds directly                                                                                                                                                            | 3.9 mm rms on the touches                     |
