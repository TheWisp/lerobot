# Show-and-Servo (v0 prototype)

Few-demo manipulation with **no learned policy**. A demonstration is mined for
_boundary conditions_ — keyframe relations between tracked visual features — and its
motion is thrown away. Motion is regenerated at runtime by a closed-loop visual servo
that measures **both ends of the relation in one camera**, so hand-eye calibration,
FK and grasp-offset errors appear in both terms and cancel in the difference.

This package is the **runtime** (§3 of the spec). The offline compiler and the GUI
recorder (§4) are not built yet — see [What is not built](#what-is-not-built).

## Layout

| module        | role                                                         | spec |
| ------------- | ------------------------------------------------------------ | ---- |
| `card.py`     | the only task-specific artifact: stages, teams, goal, budget | §3.1 |
| `binder.py`   | demo→live correspondence, once per stage, with a certificate | §3.2 |
| `tracker.py`  | KLT + forward-backward gate + Shi-Tomasi replenishment       | §3.3 |
| `grouping.py` | per-team robust Sim2 fit, eviction, fission/defission        | §3.3 |
| `servo.py`    | the difference, the PI law, the measured Jacobian, DLS       | §3.4 |
| `monitor.py`  | stage state machine, retry ladder, certificate log           | §3.5 |

Registration geometry (`Sim2`, Umeyama, mutual matching) is **reused** from
`lerobot/fewshot`, which already measured it — this package adds no second copy.

## Four decisions worth knowing

**The goal is stored as pixels, not as a transform.** A stage records where the
held end sat _in the taught image_, next to the taught target constellation. At
runtime the target team's own taught→live fit transports those positions into the
live frame. Move the target and the goal moves with it, with no frame convention for
the runtime to agree on and nothing about the surrounding scene baked in
(invariant 3). The servo error is then a comparison of two fits evaluated at the same
taught points, which is also what makes it robust to individual points dying: the
durable quantity is the **fit**, not any point.

**The gripper is not in the card.** An empty `held` team is how a card says "the
moving end is the robot itself" — every D1 stage before the grasp. The gripper's
appearance is a property of the _rig_, so the runtime supplies it; storing it in a
card would bake the robot into a task description. This is what §3.4's
`held_or_gripper` resolves to, and `Stage.held_end` is the whole of the routing.

**The Jacobian is measured, not derived — a deliberate deviation from §3.4.** DLS
differential IK normally means the FK Jacobian. This rig's encoders are good but its
kinematic _model_ is not (see `lerobot/fewshot/README.md`), and an FK Jacobian would
inject exactly the error the same-lens difference just cancelled — plus it needs
hand-eye calibration and a per-point depth, and §6 rules depth out of v0. So the
mapping from joint deltas to image motion is bootstrapped by a short probe at stage
start and then Broyden-updated from the motion the loop is already producing. DLS is
unchanged; an empirically estimated Jacobian is precisely the ill-conditioned case
damping exists for. Cost: one probe per stage. Gain: no calibration exists to drift.

**Sim2 only, no homography.** A homography needs four points and, on the ~8 noisy
points a team actually carries, spends its extra freedom modelling noise as
perspective. Out-of-plane cases are honestly out of v0's scope rather than badly
covered.

## What is proven

`tests/showservo/` — 107 tests, no hardware, no network, ~0.6 s.

- **The loop closes on a plant whose Jacobian was seeded 40% wrong**, from four
  displacement quadrants. This is M1 in miniature and the evidence for the deviation
  above; Broyden recovers the true Jacobian to <5% of the initial error from ordinary
  servo steps.
- **Abstention, everywhere.** A failed team fit, a below-gate bind, a two-point team,
  a blank frame and an unseen scene all produce `ok=False` with a reason — never a
  zero error, which would be indistinguishable from "already there".
- **Fission is not fooled by the two cases that matter**: a still gripper hovering
  over the object (the hypotheses are indistinguishable — the monitor abstains rather
  than voting on noise), and a gripper travelling away from an object that did not
  come with it (a failed grasp, not a grasp).
- **The state machine is total.** Every (state, event) pair is spelled out and
  checked at import; every event is fed in every state without raising; terminal
  states absorb late events; every abort carries a failure class, and a timeout while
  waiting on fission is classed `grasp`, not `timeout`.
- **The runtime holds no task knowledge** (`test_task_agnostic.py`). One driver runs a
  pick-shaped stage (no held team, fission) and an insertion-shaped one (held team,
  push-test) without naming either, and a third never-before-seen shape
  (`pose_hold`) runs by adding a row to a detector registry, not a branch.

## What is measured but not proven

Nothing yet — every number above is synthetic. The perception tiers this package
_delegates_ to were measured in `lerobot/fewshot` (0–1.7° registration error over a
full rotation sweep with an examined bank; 0.80 transported-mask IoU on a real
re-placement), but no Show-and-Servo path has seen a camera.

One implementation fact came out of a test rather than a rig, and is pinned there:
**SIFT descriptors must come from SIFT's own detector on both sides.** Describing
externally chosen pixel coordinates (Shi-Tomasi corners) and matching those against
`detectAndCompute` output yields _zero_ mutual matches — the descriptor is defined
relative to the scale and orientation the detector assigned. `sift_keypoints()` is
the single path both the compiler and the binder must use.

## What is not built

- **Recorder and offline compiler (§4)** — the LeRobot GUI recording path, stage
  cuts, keypoint extraction/filtering, multi-demo correspondence, review screen. The
  card schema is settled first on purpose: it is the contract the compiler writes to.
- **The hardware loop** — nothing here talks to a robot or a camera. The driver in
  `test_task_agnostic.py::run_stage` is the shape it should take; it needs the
  recorder before it can be pointed at anything real.
- **SAM3 designation into binding** — `DinoBinder` takes a mask but nothing produces
  one yet; the in-house SAM3 adapters live in `lerobot/overlays/adapters.py`.
- **Everything in §5 from M0 onward.** No bench numbers, no naked servo, no D1.

## When the 2D goal transport is valid — the sharpest limit in v0

The goal is transported into the live frame through the **target** team's fit. That is
exact only when both hold:

1. the camera's optical axis is roughly perpendicular to the plane the object moves in;
2. the held team's features are **coplanar** with the target team's features.

Condition 1 is the well-known one and turns out to be the _minor_ one. Condition 2 is
what bites. Measured in `test_goal_transport_geometry.py` against a full pinhole
projection:

| situation                                         | goal error |
| ------------------------------------------------- | ---------- |
| perpendicular camera, coplanar teams, any yaw     | 0 (exact)  |
| 10° camera tilt, coplanar teams                   | 0.3 mm     |
| perpendicular camera, held features 8 cm too high | **16 mm**  |

The closed form for the height term, at camera height `H` and object displacement `d`:

```
|error| = |d| · (z_target − z_held) / (H − z_target)
```

Three consequences worth internalising:

- **In-plane rotation is not the fragile part.** Under a perpendicular camera a yaw
  about the table normal becomes an image rotation of the same angle, exactly, at any
  angle including 180°. Rotation needs no extra machinery.
- **A closed loop does not repair a wrong goal.** It repairs a wrong _Jacobian_. The
  loop drives whatever error it is handed to zero, so a bad setpoint produces
  confident convergence to the wrong place, logged as a success.
- **Track the fingertips.** At the grasp instant they sit on the object's own surface,
  which makes condition 2 true by construction and drives the dominant term to zero.

Beyond that, larger tilts are _caught_ rather than absorbed: at a 2 px consensus band
a ≥10° tilt makes the target fit fail to assemble inliers and abstain, so the servo
stops instead of converging on a biased transform. That protection is the **gate**,
not the geometry — widen the band and the bias comes back silently.

## Honest limits

Planar (SE(2)) relations only. The error is a mean over held points, so it corrects
translation and rotation of the constellation but not out-of-plane tilt. The retry
ladder's rungs are _decided_ here and _executed_ nowhere — the executor still owes
the idempotence property every rung assumes (back off to a known pose first). The
convergence certificate detects a stalled loop but cannot distinguish "stuck on
geometry" from "stuck on contact"; §3.5's joint-divergence contact detector needs bus
`present_position`, which arrives with the recorder.
