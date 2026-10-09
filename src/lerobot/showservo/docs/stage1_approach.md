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

The first act under this design lifted the gamepad (marks at 6.1 s and 13.1 s,
half speed, the gamepad 29 mm from where it was taught and turned 10°). The
second was refused before moving: the replayed grasp would have pressed 4 mm
deeper into the tray than the demo did. The tracker kept reporting tilt and
height for a gamepad lying flat: 1.2° of tilt untouched during the demo, 10.1°
before the first act, and 4 mm low before the second.

A projection onto the tray's plane, taking only the turn and the slide from the
tracker, let the next four acts lift the gamepad (turns up to 47°, shifts up to
91 mm). Then it failed through its own assumption: a tray plane fitted at a
find, with the arm in view, came out 87° off; averaged with the teach-time
plane it planned a grasp 16 mm into the tray. It was removed on 2026-10-04 by
the operator's rule: no assumptions about planes or particular objects, because
every such rule works against the system. The act uses the tracker's fit as it
is. A tracker height error shows again as a refusal, and the remedy is a more
accurate tracker, not a rule.

Three acts in a row were aborted at the pre-grasp with "the object was still
moving" while it sat untouched. The tracker had lost the gamepad before the
acts began, so they ran on a stale pose and compared the noisy views taken as
the arm covered it. Measured on the demo, consecutive views of the gamepad
untouched in clear view move the marked grasp by 0.55 mm at most, partly
covered by the arm by up to 1.3 mm, and while it reappears after the grasp by up
to 4.4 mm. The act now refuses to start while the tracker does not see the
object, and at the pre-grasp it no longer asks the last views before a loss to
agree.

A first editor the same evening marked moments anchored to the object or to
the world and blended the correction between them, with marks suggested from
the gripper channel. The operator could not tell what an anchor applied to,
and the suggestions read the gripper backwards: on this arm a higher reading
is more closed, so the most-open moment was taken for the grasp. Both are gone.

## Observations (2026-10-04): demo first, objects designated afterwards

Teaching before the demo existed only so the tracker could record the object
during it. The demo now records the camera's colour and depth instead, and the
objects that matter are designated on its playback: a click on any frame, SAM3
segments what is under it, and Point2Pose tracks it from that frame to the end
and, started afresh on the same frame, back to the start. At the act, one live
click finds the object by registering the live view against the demo's view of it
once, with no tracking between them, and the live tracker follows it from there.

Measured with the worker's own code on YCBInEOAT videos converted to the demo
recorder's layout (ADD-S AUC against truth, every frame). The bench first put the
click at the truth mask's pixel nearest the mask's centre. Where something covers
the object's centre, that pixel lies on the cover's edge: the rod holding the
sugar box at its frame 453, a gripper finger on the cracker box at its frame 187.
A person clicks the middle of a visible face, so those frames were run again with
the click at the mask pixel farthest from the mask's border.

| Video                | Clicked frame          | Click landed on                | Track through the recording | Click mask against truth (IoU) |
| -------------------- | ---------------------- | ------------------------------ | --------------------------- | ------------------------------ |
| mustard0             | 0                      | the bottle                     | 95.6                        | 0.74                           |
| mustard0             | 368, tracked both ways | the bottle                     | 96.7                        | 0.57                           |
| sugar_box1           | 0                      | the box                        | 96.1                        | 0.72                           |
| sugar_box1           | 453, tracked both ways | the edge of the rod holding it | 46.6                        | 0.002                          |
| sugar_box1           | 453, tracked both ways | the box's face                 | 96.0                        | 0.69                           |
| cracker_box_reorient | 0, both click rules    | the printed face               | 66.7, 57.0                  | 0.07, 0.07                     |
| cracker_box_reorient | 187, tracked both ways | the edge of a gripper finger   | 57.0                        | 0.04                           |
| cracker_box_reorient | 187, tracked both ways | the box's face                 | 93.4                        | 0.81                           |

Point2Pose started from the true mask scored 95.3 on mustard0 and sugar_box1 and
93.1 on cracker_box_reorient. From a click on the object's visible face the
tracking matches it on five of the six clicked frames. The sixth, the cracker box
at frame 0, fails because SAM3 segments a printed patch of the box under the
click, not the box; a click on the edge of whatever covers the object segments
the cover. Both failures show in the click's mask before any tracking runs.

The one-shot find, from the clicked view to every tenth frame by a click there,
pooled over the five clicked frames whose click mask matched the object, binned
by how far the object had turned:

| Turn since the clicked view | Finds | Certified | Within 10 mm ADD-S | Median ADD-S, per clicked frame |
| --------------------------- | ----- | --------- | ------------------ | ------------------------------- |
| under 15 degrees            | 147   | 134       | 134                | 1.3 to 4.7 mm                   |
| 15 to 30 degrees            | 9     | 9         | 9                  | 2.4 to 6.3 mm                   |
| 30 to 60 degrees            | 10    | 6         | 6                  | 3.9 to 4.5 mm                   |
| 60 to 90 degrees            | 72    | 50        | 4                  | 8.7 to 42.8 mm, mostly flipped  |
| over 90 degrees             | 128   | 59        | 5                  | 44 to 99 mm, flipped            |

The find holds within about 30 degrees of the demo's view: every certified find
there landed within 10 mm, and the rest were refused. Past 60 degrees it mostly
certifies a wrong answer, turned about 100 degrees from the truth, which an act
would carry out with confidence; the Act step therefore shows how far the
object was found turned. A Point2Pose pipeline started afresh in the same process
does not hand back all its GPU memory: five tracks in one process ran out of
memory, so the recorded-stream tracking gets a new process for every object.

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

The pre-grasp and the grasp, as the operator marked them on the demo. The
whole act is planned and judged first, from the arm's present joints. Then the
jog's walk takes the arm to each pre-grasp in turn, a straight line at the
walk's speed with the gripper first set to that point's opening, re-aimed at
every new tracker view, so the line bends toward an object that is moved. At
the last pre-grasp the arm waits until the object holds still: two consecutive
tracker views that move the grasp's fingertip path by less than the act's own
reach tolerance, a difference the act could not carry out anyway. When the
tracker stops seeing the object, the gripper covering it, the latest view it had
is used. The act does not start while the tracker does not see the object.
The grasp is then planned from where the arm stands with that pose, and the
demo from the last pre-grasp to the grasp end is streamed sample for sample on
the demo's clock with the recorded gripper command. Nothing in this is tuned:
following has no threshold, arrival is the existing stiction band and reach
tolerance, stillness is the reach tolerance, and waiting gives up after the
existing step timeout. Without tracking, the act runs from the view it started
with. The grasp itself is not followed: under the gripper the tracker loses the
object, as it did for most of the demo's grasp.
Every pose is the demo's carried by the tracker's fit of the object's motion,
as it is. The speed scales the lines and the grasp. Before anything moves every sample is solved by IK, to 0.5 mm, from the
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
**Superseded in part 2026-10-07.** The place onto another object (next section)
uses this formula, and measures the regrasp term by finds of the held object's
view in the demo, with the arm still, instead of an in-hand model; the in-hand
model, the tip by touch and the closed loop are not built.

## The place: the held object onto another

The picked object never sits in the gripper exactly as it did in the demo: the
grasp lands a few millimetres off or at another angle, and the object can shift
when it is lifted. Replaying the demo's place relative to the target alone puts
the object off by that difference, so the place must see the held object, at
least in part; a person placing an object watches it too.

Every place sample is the demo's fingertip pose moved by the target's motion and
corrected on the gripper's side by the change in the hold (the formula above):

```
G*(t) = (B_now . B_demo^-1) . G_demo(t) . H_demo . H_now^-1      H = G^-1 . A
```

G is the fingertip from forward kinematics on the observed joints, A the held
object's motion from its view in the demo, B the target's. The objects' own frames
cancel, so the finds' relative motions are enough. Held as in the demo, the
correction is the identity and the place is the demo carried by the target.

Observations (2026-10-07):

- **The hold is measured with the arm still.** On the gamepad demo
  (`demo_20261006_065853`), the demo track's hold at the object's centre agreed
  with itself to 0.6 deg and 0.1 mm (median) over 119 frames with the arm still,
  and was off by a median of 19 mm (90th percentile 28 mm) over the 47 frames it
  moved. No offset between camera frames and joint samples brought the moving
  frames onto the still hold.
- **The hold is measured by finds, in the demo and live alike.** One-shot finds of
  the demo's view on the held gamepad's still frames were all strong (28 to 36% of
  the card's 400 points) and agreed to 1.0 deg and 0.4 mm. Their hold sat 0.9 mm
  and 3.0 deg from the track's, and both fit the depth equally (1.1 to 1.2 mm
  residual, the same outline overlap). The live tracker lost the gamepad during
  the grasp in 3 of the 15 gamepad acts that lifted it; a find does not depend on
  the track surviving the grasp, and the same method on both sides keeps what it
  gets wrong in common.
- **The gripper tells a grasp from a miss.** Over the trials, the 25 acts that
  lifted their object (cube, dowel, gamepad) ended 1.9 to 7.0 units short of the
  closing command, and 8 of the 9 marked missed 0.1 to 0.4 short; the ninth, 4.8
  short, is the act whose snapshot shows the cube held.
  **Superseded 2026-10-07:** the pick-and-place demo squeezed the gamepad only 1.23
  short at its firm grip, and its acts held it 0.89 to 1.09 short, so the line first
  drawn at 1.0 stopped two acts that held it as misses. Over all 59 recorded acts
  that closed the gripper, closing on nothing stopped -0.06 to 0.48 short (9 marked
  missed, 3 whose frames show the cube left on the table) and closing on the object
  0.89 to 10.56; the line is now 0.7.
- **The frames after the release do not show the hold.** The operator may drop
  the object onto the target. The place's goal is the held object relative to the
  target at the release, still gripped.
- **The grip is firm before the grasp ends, and the view is best there.** On the
  stacking demo (`demo_20261007_091543`, the gamepad set on a cube) the gripper's
  reading stopped 4.8 short of its command at 8.3 s; the arm stood still until the
  lift at 8.8 s, and the grasp's end mark, after the lift, came at 9.2 s. On the
  frames between the grip and the lift every find of the gamepad's table view was
  strong (99 to 124 of 400 points); carried, with the gripper's body between the
  camera and the gamepad, 2 of 7 still frames were. Those finds also put the
  gamepad 9.5 to 10.1 mm and 6.5 to 9.3 deg from where it had lain: the closing
  fingers moved it, so its resting pose is not its pose in the grip. During the
  lift the reading went from 90.3 to 91.3, so the lift moved something too.

Design:

- Pre-place and place end come after the grasp end, like pre-grasp and grasp end:
  straight lines to each pre-place, then the demo from the last one to the place
  end, release included. The pre-grasps and the grasp end follow the object
  picked; the pre-places and the place end follow the one it goes onto, chosen on
  its own in the editor.
- The target is found by a locate, a find of the demo's view that teaches nothing
  and leaves the live track on the object picked: by the operator's click, then at
  every act where it was last found. Its motion is the locate times the inverse of
  where the demo's track had it on its pose frame. A locate made against another
  demo or another designation of the object is not used.
- Each object's demo pose is read on a frame of its own, not on a motion mark: the
  frame the operator sets on its row ("pose here"), else the frame it was clicked
  on, when that comes no later than its stage's replay begins (the last pre-grasp
  for the object picked, the last pre-place for the one placed onto), else the last
  frame it was seen by its stage's first mark. On the frame it was clicked on, its
  view is the one every find matches against, so the pose there is exact. Reading
  it at the first pre-place had made a clear view of a small target a waypoint:
  on the stacking demo of 2026-10-07 the cube is in full view only until the arm
  comes within about 13 s, while the pre-place belongs at the hover over it, 14 s.
  A pose read on a frame where the object is hidden is refused.
- The firm grip is where the gripper's reading stops, within 0.3 units over
  0.1 s, more than 0.7 units short of its command, after the closing has begun:
  separate from the grasp's end mark, which says where the replayed motion ends.
- The demo's hold is measured in two windows: at the grip, from the firm grip
  until the arm moves again; and while carried, from the grasp end to the last
  pre-place or the end of the pause it is marked in. In each, finds on the demo's
  own frames where the fingertip moved slower than 5 mm/s, nearest the window's
  end first, at least three frames apart. A hold is the mean of at least three
  views that agree within 5 mm at the object's centre and 5 deg; every still view
  of the gamepad on the first gamepad demo stayed inside both, at most 0.6 mm and
  3.8 deg from the rest. Each is measured once per marks and calibration, and so
  is a failure the worker measured; one it did not answer is asked again.
- The act's grasp replay pauses where the demo's grip became firm. There the
  gripper must stop more than 1.0 unit short of its command, read once it has
  stopped closing (a demo whose own grasp stopped no further short cannot be
  checked, and is not), and, once the arm stands still, finds on fresh frames give
  the live hold at the grip; then the lift goes on. After the lift the gripper is
  checked again, for a drop.
  **Superseded 2026-10-07:** the replay no longer pauses; the operator saw it stop
  the demo's motion to do work. From the stream's sample where the demo's grip
  became firm, a watcher beside the stream waits for the gripper's reading to stop,
  checks it there (now more than 0.7 units short; a miss halts the stream at once),
  reads the fingertip then for the grasp pose, and, while the arm stands still as
  the demo's did, takes views 0.1 s apart that are read while the act goes on. A
  replay whose arm does not stand still for three views gets no live hold at the
  grip. The finds an act needs (the place's object, the picked object's fresh
  track, the demo's holds) likewise run beside the arm from the start and are
  awaited at the last pre-grasp.
- The walk carries the object to each pre-place with the grasp's closing held. At
  the last, when the demo's carry hold was measured, the arm stands still and the
  live hold is measured again the same way. That pre-place pair, which sees any
  shift the lift caused, sets the correction when both sides were measured;
  otherwise the pair at the grip does. The act records both, which one it used,
  and how far apart they are. Live views are clicked where the demo's hold puts
  the object and, after a view that fails, where the live track has it. The arm
  goes to the pre-place corrected for the hold, and the place is planned from
  there.
- When no view of the held object can be used, at the grip or carried, the hold
  is the grasp pose: the grasp is replayed aimed by the object's estimated pose,
  so the hold is the demo's, changed by how far the arm actually landed from that
  aim at the firm grip, by the joint angles. On the pick-and-place demo of
  2026-10-07 the gripper's body hid the gamepad at the grip and it hung tilted
  while carried, so no view matched and only this path was left. It cannot see
  what the closing fingers or the lift did to the object; whenever a view is used
  as well, the act records how far it lies from the grasp pose, which is how that
  gets measured. Without a firm grip in the demo, and without a view, the act
  stops.

Alternatives: the hold from the frames after the release, which a drop moves;
the hold from the track while carrying, off by a median of 19 mm while the arm
moves; the in-hand model of the section above, more general (the object's far end,
contact tasks) but needing the gripper rendered out and a wrist motion the demo
did not show, and not needed while a find sees enough of the held object; a closed
loop on the held object relative to the target in one image, which cancels
calibration and kinematic errors to first order but needs both measured at the
pre-place, where the held object covers the target.

What it costs: a locate before each act; the demo's hold the first time, a few
finds; five finds while the arm holds the object at the last pre-place; and a demo
that holds the object still somewhere between the grasp and the place.

Evaluate: ten placements with both objects moved and turned within the find's
range; the place error measured in the top camera once the arm has withdrawn,
against where the demo left the object on the target.

**NOT IMPLEMENTED.** Following the target if it is moved during the act (it is
found once, at the start); a place relative to the world, a drop-off; the closed
loop on the relative pose.

### The landing: which places count as the same

Observations (2026-10-08, the pick_place demo, the cube found turned 38° from the
demo):

- An act stopped at its first pre-place: the walk asked wrist_flex for -101.2°.
  Its servo's calibrated range ends at 93.3°; it stopped at -93.1° and the
  fingertip stood 22 mm short for 20 s.
- The plan of the same line, solved before anything moved, had the wrist at
  -89.6°: the pose was reachable. Replayed offline from the arm's joints when the
  carry began, the walk's own solve ends on the act's commanded joints to the tenth
  of a degree: it solves its target tick by tick from where the arm stands and
  lands in another arm configuration than the one the plan checked. With the
  servos' ranges as its limits, the same walk stops at -93.3°, 13 mm off: it does
  not find the plan's configuration either.
- The demo's own place had the wrist at -83.5°, the gripper 83° from vertical:
  little room. Carried with a turned cube, the landing asks more of the wrist; how
  much of that is the turn and how much the find's 7° of tilt (the cube lies flat)
  is not separated.
- Whether a place may land turned is the operator's intention, not the object's
  shape. Something set on a cube by its middle may land turned any way about that
  middle; edges laid along its faces, at any quarter turn; a key into its lock,
  only as shown. The cube's declared symmetry (order 4) already folded the find's
  turn to the least of its four, 38°; nothing tried the others.

Design:

- Each place carries a landing rule, set in the demo editor beside the object it
  goes onto and kept with the marks: exact, as shown (the default); any of that
  object's symmetric turns; or any turn about its middle. The turns are about the
  vertical through the middle of its top, where the demo saw it.
- The act's first plan ranks the turns the rule allows (every 5° for any turn).
  Each is judged on the pre-places and on the place every 0.25 s, solved from the
  demo's own joints there with no joint past its servo's calibrated range; a turn
  is dropped when a sample is out of reach as solved, and the rest are ranked by
  how near their joints stay to the demo's. The cheapest that plans in full is
  taken, and the carry, the hold's correction and the place all aim by it.
- No plan asks a joint past its servo's calibrated range. A sample that would is
  solved with the joint held at its range and the others making up what they
  can; the plan is refused only when that leaves the sample out of reach, naming
  the joint, the stage and how far it would have to go. (An act of 2026-10-08
  stopped after its grasp because the place needed the wrist 1° past its range.)
- The carry streams joints planned from where the arm stands after the grasp, the
  grasp's closing held, as the grasp and the place stream theirs: what was checked
  is what runs. The correction for the hold at the last pre-place is still a short
  walk.

Replayed offline on that act, the plan with any turn allowed takes 15° and keeps
the wrist within -85.8° (as shown, -92.2°); ranking the turns took 0.9 s.

**NOT IMPLEMENTED.** A landing free in other ways (anywhere on a surface, at any
height); choosing the turn again when the hold's correction moves the place (it is
chosen once, in the first plan); the walk's own solve kept within the servos'
ranges (the jog's walk still solves within the model's limits).

## The point groups: an object's pose while its points are hidden

**Observations (2026-10-08).** The act loses an object's pose as soon as the
gripper covers part of it. The tracked points under the gripper latch onto the
arm and drift with it while the fit still looks confident, so the place object's
track is followed only while 97% of its points are seen (the trust gate) and a
covered object keeps the pose it was last seen at. The gamepad carried to the
cube has no pose at all: the hold is the grasp command. The reset's own detector
failed the same evening for a simpler reason, a cable across the objects.

Holding the last pose is right only while nothing moves: not the object, not what
it rests on, not the camera. A wrist camera, a moved tray, a bumped cube or a
stacked object carried with the one below all break it.

**The design (2026-10-09).** Every tracked point belongs to a rigid group or to
none, and the groups are flat: no chain of parents, which a partition already
expresses (a cube on the tray is the cube's points and the tray's in one group;
lifting the cube splits them). Two loops:

1. _Membership._ A group has one rigid motion per frame, fitted robustly over its
   visible members from where each joined, in the group's frame. A member whose
   residual stays out for a few frames leaves. A free point joins the group whose
   motion explains its last frames (carried back into that group's frame, it held
   still), and may not rejoin the group it just left for a while, since a slow
   slide is within the join test's noise over a short window. Free points that no
   group explained and that moved rigidly together found a group. Groups that
   moved alike for long enough merge, judged where the smaller one's points are,
   so a small body's fit turning within its noise moves nothing.
2. _Pose._ An object is a set of tracks and a frame. Its pose is its group's
   motion composed with its anchor in the group's frame, re-measured from its own
   points while enough are seen in that group and carried by the group's other
   members while they are not. Points borrowed from whatever it rests on, for as
   long as it rests on it; a hidden object that moves relative to its group is
   caught when it reappears, the same blind spot as today but relative to the
   group instead of the camera.

All of it is 3D in the camera frame: a turn out of the image plane is a rigid
motion, not a disagreement. Nothing is assumed static; the tray is a group like
any other, so a tray or a camera that moves carries what rests on it. The
borrowed points are per session, the demo's from the demo's clutter and the
act's from the act's; only the object crosses between them, through the find.
An object held in the gripper is the same mechanism with the gripper as a
tracked body, left for later.

Where the two formulations differ, "move relative to the borrowed points" and
"track the pose from them, then move relative to the pose": nowhere while the
object stays in its group, and the pose form is the one built, since the act
already moves relative to a pose. One cost stays: a fit's rotation error becomes
a translation error at the object through the lever arm from the fit's centroid,
so borrowed points near the object count for more than many far away.

**Measured.** Seven synthetic cases at RealSense noise (1.5 mm an axis): a still
scene is one group; a sliding body leaves and takes its pose; a hidden cube
follows the tray it rests on through a 15° and 50 mm move, where its last pose
would be 50 mm off; a stopped body merges back without its pose jumping; drifting
points leave without moving the group; a late track joins the group that
explains it; under 4 mm of hidden-pose error over 40 frames of slow drift. On the
2026-10-08 stacking recording, 50 frames: one group of 464 tracks, both objects'
own fits 0.4 and 0.6 mm from the group's estimate.

**Built (2026-10-09).** TAPIR alone (37 ms for 300 points with two refinement
passes) and the groups in one process, live from the camera with an MJPEG view
(`benchmarks/group_live.py`), 7.7 fps. The borrowed points are corners taken per
image cell, nearest the target first, on the rim, the markers and the clutter,
and none on the smooth tray, where a tracker drifts; depth is read as a window's
median and not on a depth edge. A point earns its weight in the fit over its
first second of holding its place and is retired after leaving three groups;
when fewer than half an object's borrowed points still stand, new corners near
it replace them. On the whole stacking act replayed: the cube is carried by the
tray's points from the moment the gamepad covers it to the end, z 402 ± 1 mm;
the placed gamepad is re-placed by its own points 35 mm higher, on the cube; the
gamepad in the gripper is 75-130 mm off when it reappears, the held case this
leaves out. **NOT IMPLEMENTED:** the act's place aimed by the group pose; the
act still runs on the trust gate.

**Measured with a moving carrier (2026-10-09, recording `groups_20261009_132316`,
81 s, replayed with four objects designated on its first frame).** The tray was
slid 97 mm (30.8–32.5 s), turned 7° (41–45 s), a cube pushed by hand 137 mm
(49–53 s), then a sheet of paper laid over the tape roll while the tray was slid
back 64 mm and un-turned (59.8–75.1 s, 250 frames hidden). The roll's outline
was carried by the tray's points: when the paper came off it stood 6.9 mm from
the roll's own points, where a held pose was 130.8 mm off. The pushed cube rode
in a group of its own (24–39 points) for 2 s and merged back when it stopped;
the still desk and arm points split off (217 points) while the tray slid and
merged back after. 794 tracks, 96 ms a frame for TAPIR at that count. Two
defects this recording found, both fixed: objects seeded next to each other
borrowed the same corners (copies of one track), and a RANSAC consensus made of
such copies crashed the refit; it is now an abstention.

For the eye, the world is the stillest group, the one whose members moved
least over the last 15 frames (not the biggest: a tray carrying most of the
tracks is the thing that moves), and the choice sticks until another group
has moved less than half as far; the world is drawn as the camera saw it,
white dots. A group that moves differently is coloured, its dots and the
smooth surface around them out to 28 px, never across a depth step, so
nothing far from a measurement is painted, and a label holds two of three
frames before it shows. A track hidden three frames is drawn hollow where its
group puts it; a field track hidden 150 frames is retired, an object's own
never. The field is re-seeded every 30 frames up to its budget, hidden tracks
counting as absent, so a sheet of paper laid on the tray gets corners of its
own and is a body the moment it moves.

The Approach tab's Groups panel runs the view: Start spawns it on the camera
(the tracker answers after 4 s) and records from its first frame, colour, depth,
times and intrinsics in the camera recordings' layout plus `groups.jsonl`, the
groups' state per frame, under `demos/.recordings/groups_<stamp>`; Finish
closes the recording and stops the view, the camera's recording and the GPU
with it. Without objects it tracks the stable corners nearest the middle of the
view: 200 points at 12.9 fps, 247 frames in 52 MB. The same script replays a
recording offline (`--recording DIR --out OUT [--object name=u,v]`), where the
objects can be given after the fact: this is how a tray push the operator
records is to be measured.

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

Built (2026-10-04): demo first. The demo records the camera's colour and depth;
the editor designates objects by a click on the playback and tracks them through
the recording (`objects.npz` beside the demo); marks name the object they are for;
at the act one live click finds that object from the demo's view of it, and the
act's motion is the live track times that find times the inverse of where the
demo's track had the object at the first pre-grasp. The guided row records first,
then sends the operator to the editor, then asks for the live click. The robot,
arm and leader choices come from the saved profiles and the last ones used.
**NOT IMPLEMENTED.** A find beyond about 30 degrees of the demo's view: the demo's
own track sees other sides of the object while it is carried, and those views are
the candidate extension; showing the click's mask before tracking it, since SAM3
can take a part of the object or whatever covers it; the place stage, with its key frames relative to
another object or the world. **Superseded 2026-10-07:** the place onto another object is built (below);
relative to the world it is not.

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
**Superseded 2026-10-07:** the place onto another object, the grasp check and the
second object as a frame are built (below), the second object found by a locate
rather than a tracker of its own; the drop-off is not.

Changed (2026-10-07): an object's demo pose is read on its own frame, set on its
row or else the frame it was clicked on, instead of by the first pre-grasp or
pre-place (the place section above). **Superseded:** "the act's motion is the live
track times that find times the inverse of where the demo's track had the object at
the first pre-grasp" (2026-10-04, above) holds only when the object was clicked
after its last pre-grasp and no pose is set for it.

Built (2026-10-07): the place onto another object, as designed above: pre-place
and place-end marks, each stage's object chosen in the editor; the locate job in
the worker and `POST /api/pregrasp/locate`; the demo's hold and the live hold by
finds with the arm still; the grasp check; the carry, the corrected pre-place and
the place streamed like the grasp; the target's find on the live view; the act's
measurements in its trial row. **NOT IMPLEMENTED.** A place on the real arm under
this flow has not happened yet.
**Superseded 2026-10-07:** the pick-and-place demo's acts, with the hold taken from
the grasp pose, completed four places; the frames after each release show the
gamepad on the cube in three and on its edge in one. In that one the cube's find,
strong at 237 of 369 points, put the cube 27 mm from where the act's first frame
shows it; across the six acts of that demo the others were 2 to 4 mm off (the
found surface's centre against the cube's coloured blob on the depth). The
gamepad's live track also saw it shift 6 to 16 mm in the gripper after the lift
there, against about 4 mm in the others. A strong find is no proof of position.

Built (2026-10-07, later): injected errors, to test what the place absorbs. Under
the Act row, "inject an error" takes a move (mm) and a turn (degrees) in the arm's
base frame, into one of two places. Into the aim: the approach and the grasp are
carried off by the error, turned about where the fingertip grips; the act knows
where it aimed, so the grasp pose measures the miss and the place takes it out.
Into the find: the object's found pose is wrong by the error, turned about the
object's centre; the act believes it, so without a view of the held object the
place is off by the error. With "correct for the hold" off the place replays the
demo against the target uncorrected, as a baseline. The error is capped at 30 mm
and 20 degrees, the act's reach and table checks still apply, and the trial row
records it.

Built (2026-10-07, later): the object a place goes onto is drawn on the live view
where its last find put it: its surface from the demo's view, carried by the find's
motion, outlined in orange with its frame and name, beside the tracked object's
magenta mask and yellow cloud. A find that missed shows as an outline off the real
object. **NOT IMPLEMENTED:** tracking the place's object. It is found when the act
starts and not followed, so the outline stays where that find put it if the object
is moved, and the place aims there.
**Superseded 2026-10-07:** tracked since, below.

Built (2026-10-07, later): the place's object is tracked in the picked object's own
Point2Pose session. Point2Pose follows several objects in one session (one SAM2 video
segmenter with a mask per object, one point tracker, a pose per object each step); the
bridge started every session with one mask and read back only the first object. A find
of the place's object (a click, or an act's) has it join that session; each tracked
frame of the picked object comes back with the place object's share, which moves its
find, so the outline and the act's aim follow it, the pre-places and the place re-aimed
during the carry. Point2Pose fixes its objects when a session starts, so an object
found anew starts the session over with every object's newest mask; each object's
motion carries across (its pose in the new session times its motion at the restart),
and a restart costs what a teach does, 6.6 to 6.8 s on the rig's worker log. At an
act's start the place's object is found first (0.1 s), then the picked object's fresh
find starts the session with both: one restart, not two. Where each object was last
seen (a click point, from its finds and tracked frames, at most every 2 s) is kept
beside the demo, and a load finds them there again, the session started once with
both, so a restart of the server needs no clicks while the objects stay put.

Measured on the pick-and-place recording (frames 209 to 335, both objects in view,
read from disk): one session following the gamepad and the cube stepped in a median
236 ms; the cube's centre stayed a median 0.9 mm (90% 1.6 mm, 125 frames) from its own
track made alone. **NOT IMPLEMENTED:** a restart that carries an object held in the
gripper. Restarted at frame 270 with the gamepad mid-lift, carried by its last SAM2
mask, its track was 171 mm off within 20 frames while the fingertip moved about
60 mm. No flow restarts mid-carry (an act finds both before its grasp), but a click
on the place's object while the picked one is held would.

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

## Appendix C: assumptions and shortcuts

What the approach and the place take for granted to work on this rig today. Each
row is a place where the system is narrower than the problem: kept here so each can
be revisited, and generalized, relaxed or measured. "Measured" is the evidence the
shortcut rests on, where there is any; a blank means nothing has tested it.

**Finding and tracking**

| Shortcut                                                                                                                                                                                                                                                                                                                                                            | Assumes                                                                                                                                                             | Measured                                                                                                                                                                                                                                                                                           | Where                                        |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| A find matches the demo's view of the object: DINO patches and one rigid fit                                                                                                                                                                                                                                                                                        | The object lies within about 30° of how it lay in the demo                                                                                                          | Gamepad, card of 400: 116-310 points matched within 5.5° and 1.6 mm; 22-75 points at 8.8-33.6° off                                                                                                                                                                                                 | `_find_reference`                            |
| A find's pose is the demo view's surface laid on the live depth: ICP from the matches' pose turned about the live surface's main plane normal every 30°, the lowest capped cost winning; the matches only place it and grade it                                                                                                                                     | The object's largest visible face is the one its turn is about                                                                                                      | Lime cube turned 41°, 104 still frames: turn off its top face a median 6° (matches: 36°, worst 45°); gamepad turned 34°: 3-4° off its long axis (matches: 8°); the demo's still frames as before                                                                                                   | `pose.fit_surface`                           |
| A find matching under 15% of the demo view's points is refused as weak                                                                                                                                                                                                                                                                                              | That share tracks the pose error for every object                                                                                                                   | 14 gamepad finds and the cube's                                                                                                                                                                                                                                                                    | `core.find_strength`                         |
| Each act clicks where the object was last seen                                                                                                                                                                                                                                                                                                                      | It has not moved far since                                                                                                                                          |                                                                                                                                                                                                                                                                                                    | `_find_afresh`, `_locate_afresh`             |
| A fresh Point2Pose process for every find                                                                                                                                                                                                                                                                                                                           |                                                                                                                                                                     | Finds in one process filled the GPU after a few acts; an hours-old track tipped the cube 16-32°                                                                                                                                                                                                    | `P2PBridge.init`                             |
| The target is found again as the act starts, beside the arm, and followed by its share of the picked object's session from then on                                                                                                                                                                                                                                  | Its share is trusted when Point2Pose has it and enough of the tracks it began with are seen, the picked object's test against its starting tracks instead of a card | One session, the cube a median 0.9 mm from its own track alone                                                                                                                                                                                                                                     | `_apply_others`                              |
| Point2Pose keeps a pose jump past 3 cm or 15° only with 10 inliers, 15% of them and 12 in a single cluster (its own HO3D values), not its YCB ones (6, 5%, 8)                                                                                                                                                                                                       | A jump with less support than that is the arm over the object, not the object moving                                                                                | Replay of act 20261007_222123, the gripper over a still cube: four trusted poses 138-144 mm off with the YCB values, none with these; it trusts 16 frames instead of 24                                                                                                                            | `benchmarks/p2p_rig.yaml`                    |
| The place object's track moves it only while at least 97% of its tracked points are seen; the page's slider sets the share                                                                                                                                                                                                                                          | A frame with fewer of its points seen is covered in part, by the arm or what it carries, and its pose has drifted onto what covers it                               | Replayed on five recorded acts: the place's target stayed 1.2-3.9 mm from where the cube lay (one act 9 mm for one frame, then 3) against 3.7-163 mm following every trusted frame; the few uncovered frames under 97% (58-94% seen) are held too, which costs nothing while the object lies still | `pregrasp.TRUST_SHARE_DEFAULT`               |
| An object declared rotationally symmetric (order n about the axis it rests on, set per object in the demo editor) has its find's turn folded to the least of the n turns it reads the same in; the pick_place demo declares the gamepad 2 (which end is which is not kept: right for stacking it, wrong for a task that needs the D-pad on one side) and the cube 4 | The resting axis is the live surface's main plane normal; turned by 360/n the object looks and acts the same                                                        | The gamepad, taught on one still scene: 1 of 4 finds 173 deg off undeclared, 8 of 8 at 5.6-8.5 deg declared 2; on 12 real frames the surface fit preferred it end to end every time                                                                                                                | `pose.fold_turn`, `/demo/objects/symmetry`   |
| A session restart carries each object it follows by its newest SAM2 mask                                                                                                                                                                                                                                                                                            | That mask is still on the object                                                                                                                                    | A held gamepad carried so lost its track (171 mm off in 20 frames)                                                                                                                                                                                                                                 | `Scene.start`                                |
| At an act's start the place's object is found before the picked one, whose find starts the session with both                                                                                                                                                                                                                                                        | The place's object does not move between the two finds                                                                                                              |                                                                                                                                                                                                                                                                                                    | `_act_task`                                  |
| After an act's re-find, the find's own view is the object's pose until the restarted track sees it                                                                                                                                                                                                                                                                  | The object has not moved since the find; the arm, on its way, may hide it from the camera                                                                           | Act of 16:12: the gamepad under the hovering gripper was never seen again, and waiting for it stalled the act 20 s                                                                                                                                                                                 | `_act_task`                                  |
| After a stream, an arm stopped within 6 deg of its last target has arrived                                                                                                                                                                                                                                                                                          | The shortfall is the servos held short under load, not a collision                                                                                                  | Five pick-and-place acts: worst joint 1.2-3.4 deg off right after the lift; the 3.4 held for 20 s                                                                                                                                                                                                  | `_act_task`                                  |
| After a load, each object is looked for where it was last seen                                                                                                                                                                                                                                                                                                      | It has not moved since                                                                                                                                              |                                                                                                                                                                                                                                                                                                    | `_refind_last_seen`                          |
| The arm walks toward the first pre-grasp on the track the act began with while that object's find starts over                                                                                                                                                                                                                                                       | That track is near enough for the walk; the fresh one is in by the last pre-grasp, where it is awaited                                                              |                                                                                                                                                                                                                                                                                                    | `_act_task`                                  |
| A locate is used only against the demo view it matched                                                                                                                                                                                                                                                                                                              |                                                                                                                                                                     |                                                                                                                                                                                                                                                                                                    | `_located`                                   |
| The point groups' replay takes each object as the raised body under a click on the start frame, within 12 mm of the click's height                                                                                                                                                                                                                                  | A stacked object parts from what it stands on by its height                                                                                                         | The gamepad and the cube on the 2026-10-08 recording, 6592 and 2026 px                                                                                                                                                                                                                             | `group_replay.object_mask`                   |
| The replay's borrowed points are the tray's surface (within 12 mm of its RANSAC plane) around the objects, in six image tiles, each a Point2Pose body of 60-150 sampled points                                                                                                                                                                                      | The tray and the low clutter on it are what the objects rest on; a body per tile keeps Point2Pose's sampler and SAM2 to a region each                               | 464 tracks over the tray; Point2Pose 1.3 s a frame with eight bodies                                                                                                                                                                                                                               | `group_replay.tray_tiles`, `p2p_groups.yaml` |
| A point leaves its group after 3 frames over 10 mm, joins within 6 mm RMS over 4 frames, may not rejoin for 12 frames; 16 points found a group; groups merge after 15 frames (a second) within 4 mm and 5°                                                                                                                                                          | Those are RealSense noise bands at half a metre and the frame rate's time scales                                                                                    | Seven synthetic cases pass at 1.5 mm noise; on the 2026-10-09 tray push, 8-point groups were hands and drift and a 60-frame merge left the desk coloured 4 s after the tray stopped                                                                                                                | `groups.GroupTracker` defaults               |

**The demo's object poses**

| Shortcut                                                                            | Assumes                                                                                                                               | Measured                                                                                                                                                                                            | Where                    |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| An object's demo pose is read on the frame it was clicked on, unless set on its row | It does not move between that frame and its stage: until the grasp for the object picked, until something is set on it for the target | Stacking demo, 2026-10-07: the operator bumped things; the cube's track turned 10-34 deg and shifted 1-5 mm from 10.7 s as the gripper came near, so the place was aimed at where the cube had been | `_pose_choice`           |
| A set pose comes no later than its stage's last pre-grasp or pre-place              | Nothing touches the object before its stage replays                                                                                   |                                                                                                                                                                                                     | `core.keypoints_problem` |

**The hold: the held object's pose in the gripper**

| Shortcut                                                                                                                                                  | Assumes                                                                                                                    | Measured                                                                                                                                                                                                                               | Where                      |
| --------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| The grip is rigid while closed                                                                                                                            | The object neither slides nor turns between the fingers                                                                    | On the stacking demo the gripper's reading stayed at 90.3-91.3 through the carry; turning between the fingers is unmeasured                                                                                                            | `_demo_hold`, `_live_hold` |
| The gripper's pose in every frame comes from the joint angles and the camera calibration                                                                  | Both are exact at every arm pose                                                                                           | Unmeasured; the stacking demo's still frames disagreed on the hold by up to 13°, and nothing separates calibration error from a change of grip                                                                                         | every hold                 |
| The hold is measured only with the arm still, fingertip under 5 mm/s                                                                                      | Moving frames cannot be used                                                                                               | Gamepad demo's track: 0.6° and 0.1 mm over still frames, a median of 19 mm over moving ones                                                                                                                                            | `core.HOLD_STILL_M_S`      |
| The hold is measured by finds of the object's view on the table, at the firm grip and while carried                                                       | The held object, partly covered, still looks enough like it did on the table                                               | Stacking demo: at the grip 4 of 4 still frames strong (106-120 of 400), the hold within 0.3 mm and 2.0 deg; carried, 2 of 7. Pick-and-place demo: at the grip the gripper's body hid it; carried, tilted beyond a find's reach, 0 of 7 | `_demo_hold`, `_live_hold` |
| Hold views must agree within 5 mm at the object's centre and 5°; at least 3 of up to 5 (7 tries)                                                          | These bounds separate a bad find from a good one                                                                           | One demo's still views stayed within 0.6 mm and 3.8°                                                                                                                                                                                   | `core.average_hold`        |
| Each live hold comes from one arm pose: the grip, and the last pre-place                                                                                  | One partial view of the held object is enough; more frames from the same pose only average noise                           |                                                                                                                                                                                                                                        | `_live_hold`               |
| The firm grip is where the gripper's reading stops, within 0.3 units over 0.1 s, more than 0.7 units short of its command                                 | Only the object stops the fingers there                                                                                    | Stacking demo: found at 8.26 s, the reading 90.3 against a command of 95.1; the grasp's end mark at 9.16 s                                                                                                                             | `core.firm_grip`           |
| Without a pre-place measurement on both sides, the hold at the grip is the hold at the place                                                              | The lift does not move the object in the fingers                                                                           | Stacking demo: during the lift the reading went from 90.3 to 91.3; how far the gamepad moved is unmeasured                                                                                                                             | `_act_task`                |
| With no usable view of the held object, the hold is the grasp pose: the demo's, changed by how far the arm landed from its planned grasp at the firm grip | The closing fingers and the lift do not move the object against where the grasp was aimed                                  | Stacking demo: the closing moved the gamepad 9.5-10.1 mm and 6.5-9.3 deg from where it lay; acts record seen_vs_grasp_pose whenever a view is used too                                                                                 | `_grasp_pose_fix`          |
| The grip is read beside the stream: the fingertip when the gripper's reading stops, and views 0.1 s apart only while the arm stands still                 | The fingertip then is where the object was gripped; the replay stands still where the demo's arm did, for at least 3 views |                                                                                                                                                                                                                                        | `_act_task`                |
| A failure of the demo's hold that the worker measured is kept with the demo, like a hold                                                                  | The same frames give the same answer while the marks and calibration stay                                                  |                                                                                                                                                                                                                                        | `_demo_hold`               |
| The gripper's pixels are not removed; the object's mask is SAM3's cut at a click on it                                                                    | The click lands on the object, not on a finger; if a find fails there, the live track's mask is the next click             |                                                                                                                                                                                                                                        | `_held_click`              |
| The place's goal is the held object's pose relative to the target at the release, still gripped                                                           | The release, or a drop, plays out as in the demo                                                                           |                                                                                                                                                                                                                                        | `core.plan_place`          |

**The grasp check**

| Shortcut                                                                                    | Assumes                                                              | Measured                                                               | Where             |
| ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------- | ----------------- |
| Held when the gripper stops more than 0.7 units short of its closing command                | The object stops the fingers; closing on nothing reaches the command | 59 acts: closing on nothing -0.06-0.48 short, on the object 0.89-10.56 | `core.grasp_held` |
| Skipped when the demo's own grasp stopped no further short than that                        | The demo squeezed the object                                         |                                                                        | `core.grasp_held` |
| The gripper is read once it moves less than 0.3 units in 0.1 s, at most 1 s after the grasp | It has finished closing by then                                      |                                                                        | `_act_task`       |

**Motion**

| Shortcut                                                                                                                                                                                                                                                          | Assumes                                                                                                             | Measured                                                                                                                                                                                                                                                                                                                                                             | Where                                                                |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Straight lines to every pre-grasp and pre-place, with no collision check                                                                                                                                                                                          | Nothing is in the way, the held object included                                                                     |                                                                                                                                                                                                                                                                                                                                                                      | `core.plan_pregrasp_grasp`, `core.plan_place`                        |
| The grasp and the place are replayed sample for sample, with no force or contact sensing                                                                                                                                                                          | Contact goes as in the demo                                                                                         | The SO-107 reports no torque                                                                                                                                                                                                                                                                                                                                         | `_act_task`                                                          |
| A stream is played back by the jog's loop on its own clock: every sample in order, one per tick, none before its time                                                                                                                                             | The order of the demo's samples matters more than their timing: a slow tick stretches the replay                    | Act of 2026-10-08 17:02, before this: a 0.53 s stall sent the fingers' opening and the lift away as one command and the gamepad was pulled over                                                                                                                                                                                                                      | `jog.play_joints`                                                    |
| Nothing CPU-bound runs in the server while an arm moves; the reach preview searches a landing once per find of the target and not at all during an act                                                                                                            | The jog loop shares the interpreter with the server: a second of solving held its serial reads up a second          | Arm at rest, landing "turn", editor open: 0.9-1.1 s stalls every 3 s; after: none over 6 ms with the camera off                                                                                                                                                                                                                                                      | `demo_reach`                                                         |
| A target is reached within 4 mm and 3°; a walk has also arrived once stopped with its commanded joints on the target and every arm joint within 6° of them                                                                                                        | That is the servo's stiction band; a walk held short by the servos is not closed by waiting, as after a stream      | With the settle trim off, 2026-10-08: an act stood 4.9 mm off its pre-place for 20 s, every joint within 0.8° of its command, and gave up; a reset's descent, commanded onto its target (0.1 mm), stood 6.6 mm off for 60 s with shoulder_lift 0.9° and elbow_flex 1.1° short                                                                                        | `ACT_ARRIVE_M`, `core.ACT_REACH_TOL_DEG`, `walk_to`                  |
| No sample may go below the table floor, or below the demo's own height there                                                                                                                                                                                      | The work happens on a flat table                                                                                    | Floor from the touch calibration                                                                                                                                                                                                                                                                                                                                     | `_plan_act`                                                          |
| Every IK solve (the walk's, the plans', the reach preview's) is bounded by the servos' calibrated ranges: the model's limits are narrowed to them when an arm connects; an act's plans hold a joint at its range, refusing only a sample that leaves out of reach | A joint within its calibrated range is free to move there; the servo's range is where it stops whatever it is asked | The model's limits are 5-21° wider than the servos' calibrated ranges, and the wrist roll stops at +97.5°; the dowel's wrist overloads were against that stop; before the narrowing an act's walk asked the wrist for -101° against 93.3° (2026-10-08); an act's place needing the wrist at -94° against -93° was refused outright before holding (2026-10-08 19:32) | `jog._limit_to_servos`, `core.solve_plan_joints`, `jog.servo_ranges` |
| The carry to the pre-places streams joints planned from where the arm stands after the grasp                                                                                                                                                                      | The target does not move during the carry; the hold's correction at the last pre-place aims at where it is then     | The act of 2026-10-08 15:46: the walk's solve and the plan's reached the same pose in arm configurations 18° apart at the elbow                                                                                                                                                                                                                                      | `_act_task`                                                          |
| The walk takes any joint solve, however far from its target                                                                                                                                                                                                       |                                                                                                                     | One walk went 100 mm off its target after a jumped pose                                                                                                                                                                                                                                                                                                              | `jog`                                                                |
| The walk caps the fingertip's speed, not the joints'                                                                                                                                                                                                              |                                                                                                                     | At speed 1.5-2 the first move from the folded pose froze the arm                                                                                                                                                                                                                                                                                                     | `jog`                                                                |
| Settle trim (**off by default since 2026-10-08**): a stopped joint's goal is nudged by up to 3°, grown only below 75% of its overload level and released above 87.5%                                                                                              | That a joint stopped short is held by friction a small push frees                                                   | Fingertip error at 12 still targets from up to 10.7 mm to 0.8-3.5 mm; but at (-60, -200, 60) mm it tripled the shoulder's load (39.2% against 11.2% with P 32 alone, 48.8% with the feed-forward too, 10.4% with the feed-forward alone) for a worse fingertip (4.6 mm against 1.4); resets overheated the shoulder past 50 °C in minutes                            | `jog.TRIM_*`, `jog._Jog.settle_on`                                   |
| A joint 25° behind its command, or a motor over 60 °C, freezes the arm                                                                                                                                                                                            |                                                                                                                     |                                                                                                                                                                                                                                                                                                                                                                      | `jog.DIVERGE_DEG`, `jog.MAX_TEMP_C`                                  |

**The landing**

| Shortcut                                                                                                                                                                                           | Assumes                                                                                       | Measured                                                                                          | Where                                         |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| A landing turns about the vertical through the middle of the target's top: its surface within 4 mm of its highest points (by the 95th percentile), where the demo saw it on the place's pose frame | The target rests upright and the camera sees its whole top                                    |                                                                                                   | `_landing_centre`                             |
| Any turn is tried every 5°; the cheapest 4 are planned in full                                                                                                                                     | The cost changes smoothly with the turn; the best is within 2.5° of the grid                  |                                                                                                   | `core.LANDING_STEP_DEG`, `core.LANDING_TRIES` |
| A turn is judged on the pre-places and on the place every 0.25 s, by the mean square of its joints' distance from the demo's, every joint alike                                                    | The demo's arm configuration is a good one to stay near; a degree of any joint costs the same | Act of 2026-10-08 15:46, replayed: as shown the wrist reached -92.2°, the turn taken (15°) -85.8° | `core.rank_landings`, `_landing_samples`      |
| The turn is chosen once, in the act's first plan                                                                                                                                                   | The hold's correction and the target's later track do not change which turn is best           |                                                                                                   | `_act_task`                                   |

**The editor**

| Shortcut                                                                              | Assumes                                | Measured                                            | Where                    |
| ------------------------------------------------------------------------------------- | -------------------------------------- | --------------------------------------------------- | ------------------------ |
| One object picked and one placed onto per demo                                        |                                        |                                                     | `core.keypoints_problem` |
| With objects clicked on the recording, a teach from before the demo is no Pick choice | The operator means the clicked objects | A leftover teach captured the stacking demo's marks | `deObjShow`              |
