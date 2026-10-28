# Held visual support on the saved FLASH demos

The shared held-attachment updater compares each direct object pose and the
gripper-propagated pose against the **same** aligned RGB-D frame and object mask.
It projects the fixed reference surface into that frame, counting projected
pixels within the mask and measuring the 90th-percentile depth residual there.
These numbers describe visible-surface agreement; they are **not** ground-truth
error of the hidden tip or of the full 6D pose.

Re-preparing the existing demos with unchanged tracking and attachment decisions
produced these local candidates:

| Demo           | Checked held frames | Accepted visual corrections | Accepted with visual depth p90 above configured 5 mm | Accepted although gripper projection had both better mask coverage and lower depth p90 |
| -------------- | ------------------: | --------------------------: | ---------------------------------------------------: | -------------------------------------------------------------------------------------: |
| Dowel / ring   |                  36 |                          34 |                                                   23 |                                                                                     13 |
| Gamepad / cube |                  23 |                          20 |                                                    3 |                                                                                      6 |

The bundles are under the demos' local `flash/v1/prepared/` directories as
`flash-dowel-ring-20261028-held-visual-v10` and
`flash-gamepad-cube-20261028-held-visual-v10`. Their
`held-attachment-trace.npz` files contain per-frame `visual_evidence` entries.
Both v10 `skill.json` files are identical to their v9 counterparts after
normalizing the candidate name in reference IDs; this diagnostic did not
change the extracted action or attachment decisions.
The visual and gripper proposals on the dowel's accepted frames differ by a
median of 0.92 mm at the object center and 1.0 degrees in rotation, with maxima
of 14.05 mm and 27.43 degrees. Those are **disagreement between estimates**,
not accuracy against the real object. The dowel's median visual depth p90 over
accepted frames is 7.45 mm.

The apparent improvement over the earlier dowel preparation is not caused by
a different example or a dowel-specific setting. The v6 and v9 preparations
have byte-identical `annotation.json`, `keypoints.json`, and `models.npz`; the
pose-selection implementation and held comparison changed. In the saved skill,
v6 has 102 conflicted held samples and v9 has none. This is still only a
same-recording replay result. The equally controlled gamepad comparison (also
identical annotation, keypoints, and model) retains a late visual/gripper
disagreement; v9 labels its 27 affected samples ambiguous rather than
confidently corrected. A `vision_corrected_prediction` sample can simply
propagate one earlier accepted correction, so its label is not independent
evidence on every frame. Silently opening a different prepared version hid
these limitations; the viewer now opens the newest candidate for review, but
every prepared version remains a separate, selectable artifact.

In particular, v6's dowel conflict begins at camera frame 271 and persists
through frame 475. In v9 it begins at the same frame but resolves at frame 291,
after another direct view. That single changed decision explains the large
drop in conflicted carry samples. The v6-to-v9 pass also changed pose-trace
selection (version 5 to 6), so the before/after does not isolate the image
scoring change alone. It cannot establish that all held poses are correct.

This exposes a specific decision gap: a _small_ visual correction is currently
accepted without requiring its visible surface to fit the current image. The
new comparison records the gap but intentionally leaves control unchanged.
Rejecting all 23 over-tolerance dowel corrections would also risk retaining a
bad gripper prediction: the reference is built from one early view and can fit
poorly from a new angle even when the pose is good. The next experiment should
compare reference coverage and both proposals by phase, then test a broader
view-conditioned reference on held-out frames before changing the acceptance
rule. A physical accuracy claim needs independently checked contact landmarks
or another ground-truth measurement.

## Counterfactual correction rules

The replay harness [ablate.py](ablate.py) reruns the shared attachment updater
from the saved pose trace and aligned masks. Its unmodified baseline reproduces
the recorded attachment matrices, conflict flags and correction flags exactly
for both demos. The alternative rules withhold an accepted correction when the
gripper projection dominates on both mask coverage and depth residual, when
that dominance also passes the configured depth tolerance, or whenever the
visual residual exceeds that tolerance. The harness saves the
[summary](ablation_results.json) and [per-frame decisions](ablation_results_frames.csv).

| Demo           | Rule                                           | Conflicted camera frames during annotated hold | Accepted direct corrections | Median visible-surface depth p90 |
| -------------- | ---------------------------------------------- | ---------------------------------------------: | --------------------------: | -------------------------------: |
| Dowel / ring   | Current baseline                               |                                             20 |                          34 |                          7.45 mm |
| Dowel / ring   | Withhold gripper-dominated correction          |                                            205 |                          17 |                          7.01 mm |
| Dowel / ring   | Withhold only when gripper is within tolerance |                                            205 |                          27 |                          7.64 mm |
| Dowel / ring   | Require visual residual within tolerance       |                                            244 |                          11 |                         19.24 mm |
| Gamepad / cube | Current baseline                               |                                              6 |                          20 |                          3.49 mm |
| Gamepad / cube | Withhold gripper-dominated correction          |                                              6 |                          13 |                          3.49 mm |
| Gamepad / cube | Withhold only when gripper is within tolerance |                                              6 |                          15 |                          3.49 mm |
| Gamepad / cube | Require visual residual within tolerance       |                                              6 |                          17 |                          3.49 mm |

The dowel's lower image residual under the first alternative is misleading:
withholding five early small corrections changes the gripper prior enough that
the frame-291 visual pose is no longer decisively favored, so the attachment
remains conflicted until much later. The blanket tolerance rule even rejects
that frame's 5.85 mm residual although the gripper prediction is off the same
image by 74.98 mm; its final attachment differs from the baseline by 272 mm.
These are counterfactual **software** replays, not alternate physical robot
outcomes. The harness rewinds accepted updater state for each candidate gate;
it has not been integrated into the runtime. It demonstrates that a single
instantaneous score or a stricter threshold is not a supported fix. We need
better view coverage and temporal evidence, then a benchmark with independent
task-point/axis checks before changing the active policy.
