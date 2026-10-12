# Point2Pose regression evidence

A code-contract test and a tracking-accuracy test answer different questions.
The SDF return fix passes its contract tests: its pose agrees with its residuals
and fallback behavior. The full robot replay nevertheless develops worse poses
under occlusion. The fix remains experimental; the unit-test result is not an
approval for physical stacking.

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
return-value repair. Candidate: pinned P2P `4abb818f`. The bridge, configuration
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
further work before physical validation. The pinned implementation remains
experimental; no physical stacking trial followed these results.

A repeat of the complete 375-frame baseline cracker-box reorientation run
produces bit-identical pose matrices (maximum element difference zero). The
regression on that sequence is therefore reproducible in this environment,
rather than established only by comparing two unrepeated noisy runs. This
repeatability check does not prove determinism for every sequence or machine.
