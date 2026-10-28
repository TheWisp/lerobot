# Held-pose ambiguity in the gamepad demo

The direct RGB-D fit proposes a roughly 177–179° gamepad flip at recorded frames 429–430. The fixed object model and same-frame mask/depth do not choose it over the current gripper-carried pose:

| Frame | Visual fit: supported pixels, p90 depth error | Gripper prediction: supported pixels, p90 depth error |
| ----- | --------------------------------------------- | ----------------------------------------------------- |
| 429   | 100, 2.15 mm                                  | 115, 3.45 mm                                          |
| 430   | 87, 2.25 mm                                   | 112, 3.65 mm                                          |

Both depth errors are under the configured 5 mm fit tolerance. The visual proposal fits its supported pixels more closely; the gripper prediction explains more pixels. Neither dominates, so this is ambiguity, not evidence of a 180° slip. The attachment updater now retains a previously usable gripper prediction in this case, records the rejected visual proposal, and labels the prediction `ambiguous_visual_prediction`. It still withdraws the attachment if the gripper pose lacks sufficient same-frame support or exceeds that depth tolerance. A visual proposal that dominates remains pending until a later compatible image confirms a large correction.

The [gamepad contact sheet](pick_place.jpg) shows the retained orange prediction and independent cyan fit. The v7 replay had 47 conflicted held frames; v9 has six early conflicted frames, 41 late ambiguous-visual frames, and 139 vision-corrected frames. The prepared skill has 27 ambiguous-visual samples and no conflicted held samples. [Insertion](insertion.jpg) is unchanged: all 223 checked carry frames use a visual-corrected gripper prediction, with no ambiguous or conflicted carry frames.

Both v9 candidate bundles reused the same pose-trace selections as v7 because only held-attachment arbitration changed. These are offline replay results, not physical evidence that either hidden contact point is accurate. The ring's source pose remains predicted rather than directly observed.
