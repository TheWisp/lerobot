# HVLA TODO

## S1 state normalization

Background: a joint held still for a whole recording gets a training std at
torch's `1e-6` numerical floor, and dividing by that amplifies differences below
sensor resolution. Measured on `GPU/0803_20260803_174402`: `left_joint_3.pos`
has mean 0.9508 and std 1.000e-06 across all 40,101 frames, and a rig reading of
0.732 — 0.22 degrees away — normalized to 218,569 sigma. A channel that size
dominates the first linear layer and corrupts the predicted actions for every
joint, including the ones the task uses. Two mitigations shipped: a default
`state_position_std_floor` of 0.5, and a `±10` clamp on the normalized state
applied in both `FlowMatchingDataset` and the policy.

- [High] **Revisit the `±10` clamp bound against measured data, not intuition.**
      The clamp is not free and the first version of it shipped with a comment
      claiming it was. Across `GPU/0803_20260803_174402`, 0.85% of training
      frames have at least one feature past 10 sigma. Most are degenerate
      channels, which is the intent, but real motion reaches it too:
      `right_joint_7.vel` peaks at 20.5 sigma and `right_joint_3.vel` at 18.0,
      on joints whose native std is 14–17 deg/s. Those transients are now
      truncated on both sides of training, so it is a consistent transform
      rather than a skew — but it does discard the fastest motion in the
      dataset. The bound is a tradeoff between "how much real transient do we
      keep" and "how large may a degenerate channel get", and the right number
      should come from a sweep on rollout quality, not from the round figure
      that was picked first. `tests/hvla/test_normalized_state_clamp.py::test_the_clamp_is_not_free`
      pins the measured peak so this cannot quietly revert to being assumed free.

- [High] **Evaluate QUANTILES against MEAN_STD for S1.** The clamp and the floor
      bound this failure; quantile normalization avoids it. Measured on the same
      data, same frames:

      | | train max | train frames clipped | rig frame |
      |---|---|---|---|
      | MEAN_STD | 23.1σ | 0.85% | 10,300 |
      | QUANTILES | 9.0 | 0.00% | 5.4 |

      Quantile never reaches its own clamp on training data and needs no
      mitigation on the frame that broke inference. The reason is the
      denominator guard: `normalize_processor.py` substitutes `1.0` whenever
      `q99 - q01 < 0.1`, a floor in dataset-native units, where mean/std's only
      guard is `clamp(min=1e-6)` — a divide-by-zero guard, not a scale. 10 of 48
      channels here trip the substitution. Secondarily, q01/q99 ignores tails,
      so one glitch frame cannot set the scale for a whole channel.

      Cost: everything past q99 compresses, so tail resolution is lost — which
      is the same tradeoff as the clamp, made continuously rather than at a
      cliff. Prerequisite is q01/q99 in dataset stats, and both OpenArm2
      datasets already carry `q01, q10, q50, q90, q99`, so the switch is cheap
      to try. Needs a paired run (same seed, same steps) before adopting.

- [Med] **Reuse `lerobot.processor` normalization instead of hand-rolling it.**
      `FlowMatchingDataset` implements its own mean/std normalization and the
      policy re-implements the matching inference-side transform, which is why
      the two could diverge and why the clamp had to be added to both by hand
      with a test to keep them in step. Upstream `normalize_processor.py`
      already provides `MEAN_STD`, `QUANTILES` and `QUANTILE10` behind
      `NormalizationMode`, with the degenerate-denominator substitution and a
      `clamp(-10, 10)` already in the quantile path — a comment there cites
      "nearly-stationary joints", so upstream hit this exact failure and landed
      the same two answers independently, including the same bound.

      Moving S1 onto `PolicyProcessorPipeline` would delete the duplicated
      transform, make the normalization mode a config choice rather than a
      rewrite, and let S1 inherit fixes made for the other policies. It is the
      larger change of the three: S1's trainer deliberately bypasses the
      processor pipeline today, so this is a structural migration and needs its
      own design pass. Do the quantile evaluation above first — if quantile
      wins, that is the strongest argument for the migration, since it is a
      one-line mode change afterwards rather than another hand-rolled path.
