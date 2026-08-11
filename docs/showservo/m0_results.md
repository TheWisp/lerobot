# M0 — input-quality gate on real video

`thewisp/intervention_cylinder_ring_assembly`, 34 episodes, 30 fps, `top` camera.
Reproduce with `uv run python benchmarks/showservo_m0.py --repo-id <id> --camera top`.

No robot, no servo, no motion. M0 asks only whether the measurements the servo depends
on are trustworthy enough to be worth servoing on.

## Tracker point survival (KLT + forward-backward gate)

| frames    |    5 |   10 |   20 |   40 |   80 |  150 |
| --------- | ---: | ---: | ---: | ---: | ---: | ---: |
| seconds   |  0.2 |  0.3 |  0.7 |  1.3 |  2.7 |  5.0 |
| surviving | 0.62 | 0.49 | 0.45 | 0.40 | 0.19 | 0.07 |

**Half-life is about 10 frames — a third of a second.** A third of the team is gone in
0.2 s and 93% is gone by 5 s.

Same run on the `front` camera, for comparison:

| frames            |    5 |   10 |   20 |   40 |   80 |  150 |
| ----------------- | ---: | ---: | ---: | ---: | ---: | ---: |
| surviving (front) | 0.69 | 0.62 | 0.56 | 0.54 | 0.36 | 0.27 |
| surviving (top)   | 0.62 | 0.49 | 0.45 | 0.40 | 0.19 | 0.07 |

`front` looks roughly four times better at 5 s — and that comparison **cannot be
trusted**, for the same reason the cross-episode row cannot. With a centre-of-frame ROI,
"survival" partly measures how much static background happens to sit in the middle of
each view. A point on the table survives indefinitely and tells us nothing; a point on
the object is what matters. `front` may simply frame more static scene.

`front` is also _worse_ where it counts: only 75% of episodes bind at zero offset (one
episode yields too few features to clear the gate at all), against 100% on `top`. So
neither camera is cleanly better, and both numbers are confounded by the ROI.

The two views agreeing on nothing except "KLT decays fast" is itself the finding:
**designation has to be wired before any M0 number is interpretable.**

This is the headline result, and it is much worse than the rendered bench suggested
(where teams survived tens of frames). Renders have no motion blur, no sensor noise and
no rolling shutter; this is what the flattery was worth.

Consequences, in order of how much they change the plan:

- With the ~8 points/team §4 specifies, a team drops below the 4-point fit minimum in
  well under a second. **Replenishment is not a top-up, it is the main supply.**
- Re-bind cadence has to be ~1 s, not once per stage. "Semantics once per stage,
  geometry every frame" survives as a principle but the constant is far smaller than
  assumed.
- This is the concrete case for the learned tracker tier (CoTracker3 / TAPNext), which
  re-acquires points after occlusion where KLT cannot. It sits behind the same
  `init`/`step` interface, so the swap costs nothing architecturally.

## Bind inlier rate vs. frames since teaching (same episode)

| offset (frames) |    0 |    5 |   15 |   30 |   60 |  120 |  240 |
| --------------- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seconds         |  0.0 |  0.2 |  0.5 |  1.0 |  2.0 |  4.0 |  8.0 |
| certified       | 100% | 100% | 100% | 100% | 100% |  25% |   0% |
| inliers         | 71.0 | 24.0 | 13.8 | 10.5 |  8.5 |  3.2 |  2.2 |
| inlier ratio    | 1.00 | 0.87 | 0.67 | 0.55 | 0.55 | 0.25 | 0.40 |

Binding degrades gracefully and holds its certificate out to ~2 s of scene change,
then collapses. Note the failure is _certified_ as a failure — the gate refuses at
120 frames rather than returning a confident wrong transform, which is the behaviour
the retry ladder is built on.

## Bind across episodes

| episode   |   1 |   2 |   3 |   4 |   5 |   6 |
| --------- | --: | --: | --: | --: | --: | --: |
| certified |  no |  no |  no |  no |  no |  no |
| inliers   |   4 |   0 |   0 |   0 |   0 |   0 |

**Do not read this as a verdict on the design.** The bench uses a fixed centre-of-frame
ROI as a crude stand-in for SAM3 designation, so between episodes the "constellation"
is mostly background and arm, which legitimately does not correspond after a scene
reset. This measures _SIFT over a fixed image region_, which is not what a card is.

The comparable measurement done properly — DINOv2 features on a SAM mask of the actual
object, episode 3 vs episode 12 of the same task — reached 54/92 inliers and 0.80
transported-mask IoU (see `src/lerobot/fewshot/README.md`). The gap between that and
0/6 here is the value of designation, not evidence against cross-episode binding.

## What this changes

1. **Wire designation before trusting any cross-episode number.** The centre ROI has to
   go; SAM3 already exists in-house.
2. **Continuous replenishment, and treat the learned tracker tier as likely required
   rather than an optional upgrade.** A 0.33 s half-life is the strongest argument yet
   for it.
3. The sim bench should be re-read with this in mind: its convergence in ~12 frames was
   comfortably inside KLT's real half-life, so it never tested the regime that matters.
