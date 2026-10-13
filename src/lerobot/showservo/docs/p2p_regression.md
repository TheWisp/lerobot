# Point2Pose regression evidence

A code-contract test and a tracking-accuracy test answer different questions.
The SDF return fix passes its contract tests: its pose agrees with its residuals
and fallback behavior. The full robot replay nevertheless develops worse poses
under occlusion. The fix remains experimental; the unit-test result is not an
approval for physical stacking.

**October 13 rollback:** the live dependency selection was restored to baseline
`c3d2068d7aa12945c146206dea315bf1379a561c` after these regressions. Candidate
`4abb818f` remains on the fork's experimental branches for offline investigation.
The baseline also has known failures; restoring it removes this experimental
behavior change without claiming that the original stacking problem is solved.

## Evaluation before changing the runtime

Keep the small contract tests, then compare complete recorded sequences against
the previous implementation. Use the same RGB-D, initialization, configuration,
model weights, dependencies and random seed. Save every emitted pose, including
lost frames. Do not remove failures, truncate difficult sequences, restart at an
occlusion, or score only the frames declared successful. Record source commits,
bridge/configuration hashes, input hashes and the frame-to-ground-truth pairing.

Run every available sequence in the reference suite. The local YCBInEOAT suite
contains nine sequences and 7,449 frames. It has ground-truth poses and meshes.
Our integration protocol supplies only the first ground-truth mask and uses
SAM2 for subsequent masks, as the live tracker does. This differs from a paper
evaluation supplied with ground-truth masks throughout. Model-free motions are
applied to the initial ground-truth pose before scoring full-mesh ADD and ADD-S.
Images and poses pair by sorted ordinal, matching upstream `YcbineoatReader`;
most sequences use timestamp filenames for images and frame indices for poses.

Keep our multi-object, held-object and failed stacking recordings alongside
the reference suite. They exercise deployment conditions absent from the nine
single-object videos. They do not have six-dimensional ground truth: label
stationary-object drift, surface disagreements and rotation magnitudes as such.
An example video without depth, calibration or ground-truth poses cannot supply
a quantitative six-dimensional pose benchmark by itself.

Compare each sequence separately. Report accuracy distributions, large errors
declared not lost, lost intervals and recovery, as well as average ADD/ADD-S
scores. A better aggregate must not hide a worse sequence or a rare large error.
The tracker saying `lost:false` is distinct from the robot controller accepting
that pose. A bad estimate must not be described as a physical action unless the
controller's recorded decision establishes that.

Fixes should be developed on one set of recordings and checked on others that
did not inform their design. Retain newly discovered failures permanently. Use
repeated baseline runs to measure numerical variability before choosing allowed
changes; set accuracy and availability budgets before inspecting a candidate's
results. Include synthetic known motions and controlled occlusions where useful,
but do not substitute them for real moving-object sequences.

## Repeatable comparison

`benchmarks/compare_p2p_replays.py` checks a complete paired suite. It requires
`suite.json` with all sequence names and expected frame counts, and each run's
`manifest.json` and `errors.npz`. Missing sequences, incomplete input identity,
different inputs/configuration/seeds, wrong frame counts and non-finite errors
cannot pass. The caller explicitly supplies budgets; they are not hardcoded
object-specific rules or robot-control thresholds.

For example, the following is a strict screening comparison with zero allowance
for degraded scores, tail error or availability. The 50 mm limit labels large
ADD errors for this report; it is **not** a permitted stacking error:

```sh
python benchmarks/compare_p2p_replays.py /path/to/ycb_regression \
  --bad-add-mm 50 --auc-drop-pp 0 --p95-rise-mm 0 --extra-lost-frames 0
```

The command exits nonzero if any sequence regresses in ADD/ADD-S AUC,
95th-percentile ADD, high-error frames reported not lost, or lost-frame count
beyond the supplied allowances. It scores the emitted trajectory through lost
frames, so marking bad estimates lost cannot remove their errors from the score.
Tests verify that a rare accepted failure is caught even when both average and
95th-percentile accuracy improve. This is an offline comparison command, not an
automatic hardware interlock or a claim of zero regressions on unseen scenes.

The October 13 experiment archive retains the actual runner, seeds, input
hashes, full poses, masks protocol, per-frame errors, bridge logs and source
identity. The source checkout and live tracker are not edited by diagnostic
ablations. Any proposed estimator change needs its own complete comparison;
improvement on one frozen-frame probe is insufficient.

Only then proceed to supervised physical trials with reset scenes, varied grasps
and partial visibility. Keep the estimator experimental if a regression remains
unexplained. Neither a finite benchmark nor successful physical trials can
guarantee zero future failures.

## October 13 paired YCBInEOAT comparison

All nine local sequences completed: 7,449 frames for each implementation.
Baseline: P2P `c3d2068` with the LeRobot bridge from `c3c2cf633`, before the
return-value repair. Candidate: then-pinned P2P `4abb818f`. The bridge, configuration
and input identities are archived per run; no new estimator changes were made
during this comparison. The table reports ADD AUC over 0–10 cm (higher is better),
95th-percentile ADD in millimetres, and frames above 50 mm ADD with `lost:false`.
The latter is a P2P flag, not a physical controller decision.

| Sequence                    | ADD AUC, old → fixed | P95 ADD mm, old → fixed | Not lost above 50 mm, old → fixed |
| --------------------------- | -------------------: | ----------------------: | --------------------------------: |
| bleach0                     |        22.89 → 23.94 |           268.8 → 574.5 |                          401 → 29 |
| bleach_hard_00_03_chaitanya |        81.70 → 82.25 |             45.5 → 42.7 |                           17 → 16 |
| cracker_box_reorient        |        87.75 → 86.05 |             21.1 → 22.1 |                             0 → 3 |
| cracker_box_yalehand0       |        85.65 → 82.49 |           115.3 → 127.6 |                            1 → 86 |
| mustard0                    |        89.70 → 91.35 |             14.8 → 12.4 |                             0 → 0 |
| mustard_easy_00_02          |        76.13 → 85.95 |             74.6 → 35.4 |                          164 → 15 |
| sugar_box1                  |        88.42 → 89.03 |             55.8 → 81.7 |                            5 → 12 |
| sugar_box_yalehand0         |        73.23 → 72.00 |             36.2 → 39.0 |                             3 → 5 |
| tomato_soup_can_yalehand0   |        75.89 → 75.35 |             50.6 → 62.3 |                          44 → 117 |

Average ADD accuracy improves on five sequences and worsens on four; five
sequences have more large errors reported not lost. The strict comparison above
fails. Only `bleach_hard_00_03_chaitanya` passes every zero-regression screening
criterion. Some other flags reflect small availability changes (mustard lost
frames: 10 → 13 and 8 → 12), while others are substantial accuracy regressions.
`bleach0` is poor under both versions: fewer high-error not-lost flags coexist
with a much longer loss of tracking (106 → 478 lost frames) and worse tail error.
Neither an aggregate improvement nor a reduction in false-confidence count alone
would justify deployment.

This result does not undo the code-contract diagnosis. It establishes that
returning the accepted refinement changes the full estimator in ways that need
further work before physical validation. The candidate remains experimental and
has been removed from the active pin; no physical stacking trial followed these
results.

A repeat of the complete 375-frame baseline cracker-box reorientation run
produces bit-identical pose matrices (maximum element difference zero). The
regression on that sequence is therefore reproducible in this environment,
rather than established only by comparing two unrepeated noisy runs. This
repeatability check does not prove determinism for every sequence or machine.

## Rollback verification

With the dependency pin restored to `c3d2068`, the current bridge was replayed
over all 129 frames (2–130) of act `20261012_205715`, using the same inputs,
configuration and seed 17 as the archived pre-fix baseline. Both objects' pose
matrices are bit-identical at every frame (maximum element difference zero),
and their lost flags match. The pin check rejects the corrected `4abb818f`
checkout. All 25 targeted dependency, bridge and replay-comparison tests pass.
The experiment archive retains `replay_rollback_seed17/` and
`rollback_verification.json`. This verifies restoration of the baseline, not an
improvement in its known tracking failures; no physical trial was run.

## Consensus-refit experiment, October 13

The separate clear-view investigation traced an unstable depth-edge
correspondence and a sparse pose still influenced by a point discarded after
the final refit. Candidate `583bf9d` repeats the refit when consensus changes,
starting from `c3d2068`; it does **not** include the earlier SDF return repair.
See [the diagnosis and unit-test limits](p2p_registration.md#clear-view-depth-edge-investigation-october-13).

The candidate improves the full 129-frame robot replay but fails the complete
reference suite. All nine candidate sequences finished, totaling 7,449 frames,
paired with the archived baseline runs above. The comparator verifies matching
RGB-D, annotated poses, initialization mask, intrinsics, configuration and seed 17. Candidate provenance records the exact clean fork, LeRobot `1a1a00e56`,
the normal bridge hash and the additional experiment-bridge hash. The latter
selects the separate checkout in that process only and retains the historical
vectorized hypothesis search; the live dependency lock was never changed.

| Sequence                    | ADD AUC, baseline → candidate |    P95 ADD mm | Not lost above 50 mm | Lost frames |
| --------------------------- | ----------------------------: | ------------: | -------------------: | ----------: |
| bleach0                     |                 22.89 → 27.67 | 268.8 → 354.2 |              401 → 0 |   106 → 515 |
| bleach_hard_00_03_chaitanya |                 81.70 → 65.98 |  45.5 → 325.5 |               17 → 2 |     3 → 149 |
| cracker_box_reorient        |                 87.75 → 87.59 |   21.1 → 21.0 |                0 → 0 |       5 → 4 |
| cracker_box_yalehand0       |                 85.65 → 86.85 |  115.3 → 26.9 |               1 → 16 |     78 → 37 |
| mustard0                    |                 89.70 → 91.55 |   14.8 → 11.7 |                0 → 2 |      10 → 4 |
| mustard_easy_00_02          |                 76.13 → 75.22 |   74.6 → 79.7 |            164 → 157 |      8 → 14 |
| sugar_box1                  |                 88.42 → 88.55 |   55.8 → 55.0 |                5 → 7 |     55 → 49 |
| sugar_box_yalehand0         |                 73.23 → 71.19 |   36.2 → 37.7 |                3 → 1 |     10 → 37 |
| tomato_soup_can_yalehand0   |                 75.89 → 73.88 |   50.6 → 56.0 |              44 → 90 |   158 → 120 |

ADD AUC improves on four sequences and worsens on five; four have more large
errors reported not lost. Every sequence fails at least one zero-regression
screening criterion. Some changes are small, but the bleach hard sequence has
a substantial accuracy and availability regression. Marking more frames lost
does not solve that regression, since all emitted poses remain in the score.
The 50 mm reporting limit remains unrelated to acceptable stacking error.

The comparator exits 1 with `passed:false`; its complete ADD/ADD-S results are
archived in `p2p-clearview/ycb_comparison.json`. The candidate remains on the
experimental fork branch and was never promoted. Passing the 20 native unit
tests does not override this failed tracking comparison. The exact mechanism
of the candidate's broader regressions has not yet been isolated.

The archive also retains the runner and an aborted initialization attempt:
metadata writing failed before the first sequence, was repaired, then the
entire suite ran to completion. No completed sequence was truncated or omitted.
The baseline tracing observer was separately verified bit-identical for both
objects across the 129-frame robot recording. All work here was offline; the
live worker continued on baseline `c3d2068`, with follow and act off and the arm
disconnected.
