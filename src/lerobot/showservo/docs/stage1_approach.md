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

## Constraints

- The object sits on the table. Between teach and run it slides and turns on
  the plane: two translations and one rotation about the table normal. It does
  not lift or tilt. (An object placed on a block or tipped over is out of
  scope for stage 1.)
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
2. **Register on the whole cloud with the plane as a prior**: three degrees of
   freedom, so a plain object or a handful of noisy points still gives a
   stable answer. Descriptor matches, where they exist, pin the turn; a round
   footprint has no measurable turn and gets none.
3. **Use the environment twice.** Background points must fit no motion, which
   certifies that the camera has not moved; any point that moves with the
   background rather than the object is evicted from the object, which is how
   clutter and a loose designation are handled.
4. **Certificate** with every find: inlier count, rms, similarity scale (a
   rigid object keeps its size; a scale off 1 flags a depth fault), the
   background check, and the taught-vs-found height and footprint.

### Track

KLT on the object's points at ten hertz between finds, re-finding when the
track's certificate fails. The transported pre-grasp updates live; the jog's
bounded walk follows it. A slowly moving object is the same loop.

### Execute

Go to a hover above the transported pre-grasp, then to it. The jog's guards
apply: the reference pauses when the IK holds a tick, a joint that falls 25
degrees behind freezes the arm, a motor over 60 C freezes it.

### Evaluate

Ten placements per object: translations, turns including a half-turn for a
round object, one with clutter beside it, one with the object nudged during
the approach. Each trial logs the transported motion, the certificate, the
arrival gap from the encoders, the object-to-fingertip offset re-measured by
the top camera after arrival, a wrist photo, and the operator's verdict. The
outcome is a table. The top-camera re-measurement shares the calibration with
the transport, so it is a consistency check, not an independent truth; the
operator's verdict and the wrist photo are the independent evidence.

### Tab

The Servo tab becomes Approach. The M1 servo and live-fit panels are
superseded by this design and hidden; their code stays until the evaluation
says whether a stage 2 (closed on the tool) is needed at all.

```
Approach
 ├─ Rig        camera session · jog (with gripper) · touch calibration
 ├─ Teach      designate (concept | box) → capture → mark pre-grasp → mark grasp
 ├─ Run        find (certificate) · track on/off · go hover / go · stop
 └─ Trials     one row per run: motion, certificate, gap, miss, verdict
```

## Decisions

- Designation by SAM3 concept by default (targets are clear, named objects);
  the drawn box stays as the no-GPU fallback.
- On-the-plane motion for stage 1.
- Success is measured by the top camera's re-measurement plus the operator's
  verdict and a wrist photo; no marker on the gripper.
- Rename the tab; hide the superseded panels; keep their code.

## What is built and what is not

Built (2026-09-20): the calibrated executor (`lerobot.gui.api.jog`), the touch
calibrations and joint-zero refinement (`lerobot.gui.api.calib`), and a
pre-grasp teach-and-transport with a drawn box, SIFT for textured objects and
a depth blob for plain ones (`lerobot.gui.api.pregrasp`).

**NOT IMPLEMENTED:** designation by SAM3 concept in the pre-grasp path (the
subprocess machinery exists in `benchmarks/showservo_real.py` and the bind
endpoint; it is not yet wired to teach/find); whole-cloud registration with
the plane prior (find uses a centroid shift for plain objects and a 6-DoF
Kabsch on keypoints for textured ones); the background no-motion check and
eviction; tracking between finds; the grasp mark and descent; the trials
table; the wrist image at mark.
