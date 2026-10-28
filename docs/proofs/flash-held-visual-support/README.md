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
