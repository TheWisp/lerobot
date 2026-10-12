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
