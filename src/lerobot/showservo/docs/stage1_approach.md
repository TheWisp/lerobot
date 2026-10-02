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
5. **Certificate** with every find: inlier count, rms, similarity scale (a
   rigid object keeps its size; a scale off 1 flags a depth fault), the
   background check, and the taught-vs-found height and footprint.

### Track

KLT on the object's points at ten hertz between finds, re-finding when the
track's certificate fails. The transported pre-grasp updates live; the jog's
bounded walk follows it. A slowly moving object is the same loop.

### Execute

Go to a hover above the transported pre-grasp, then to it. The jog's guards
apply: the reference pauses when the IK holds a tick, a joint that falls 25
degrees behind freezes the arm, a motor over 60 C freezes it. Before Go the
operator can see what the arm will do: the jaw line and approach arrow of
the transported pose drawn on the find image, and the readout of how far the
gripper will turn about vertical and lean, both in the base frame.

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
- Six degrees of freedom is the general case; the table plane is a prior the
  find uses when the object is on it and the evidence is thin, never an
  assumption.
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
