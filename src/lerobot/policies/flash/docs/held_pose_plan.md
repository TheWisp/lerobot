# Held-object pose under occlusion: evidence and next experiment

Audited 2026-10-20. This is a plan for the shared object estimator in
[the FLASH architecture](architecture.md#common-object-estimation-and-relationships),
not a dowel-specific controller rule. The local recordings cited here are
ordinary LeRobot episodes under
`~/.cache/huggingface/lerobot/demos/flash_insertion_trial_20261020_020-*`;
they are not committed to git. The audit uses the latest 19 local attempts,
17:22:16 through 20:33:10, rather than selecting only completed motions.

**28 October follow-up:** [Two independent recorded-attempt replays](../../../../../docs/proofs/flash-held-trial-holdout/README.md) now rerun the current surface backend from RGB-D and measured joints, retaining rejected proposals and masks. A rejected visible-patch direction is substantially closer to measured depth than the gripper prediction late in one attempt; the two are about equally supported late in the other. The shared proposal now records the measured patch direction, three spreads and point count even when a complete pose is rejected. This is a shadow measurement, not a change to the selected pose or held attachment. The group tracker, inference timing and physical outcomes were not rerun; no full-pose accuracy claim follows from the depth-direction comparison. The next experiment must calibrate when that direction is actually observable and evaluate a partial update against future frames and task landmarks before control uses it.

## What the attempts establish

| Evidence                                                 | Finding                                                                                                                                                                                                                                                       | Limit of the finding                                                                                                                                                                          |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| All 19 episode traces                                    | Six issued `attach-held`; thirteen stopped earlier. The earlier failures include object-pose loss, changing approach goals, IK, and motor/queue errors.                                                                                                       | Held-pose work cannot by itself fix every attempt. Keep those failure categories separate in evaluation.                                                                                      |
| 19:11:55, 19:51:34, 20:18:38, 20:33:10 tracker traces    | All four record a gripper-based held prediction and an instance-mask stream. None records an independently observed held pose or a post-grasp attachment correction.                                                                                          | A mask is evidence of visible pixels, not automatically a registered six-axis pose.                                                                                                           |
| 20:18:38 and 20:33:10 camera and motor frames            | Both grasp and carry. The former is visually upright in the ring but software interrupts on the following retreat; the latter completes the command sequence, enters the ring tilted, then falls after release. Neither has an independent success milestone. | The different physical results do not isolate one causal variable.                                                                                                                            |
| RGB-D diagnostic of the two latest grasps                | The two exposed dowel ends show an axis change of roughly 4–5° while the measured gripper continues closing after `attach-held`. The gripper-based axis differs by roughly 8–10° after lift.                                                                  | This color-assisted offline diagnostic is not a generic tracker or full six-axis ground truth. The hidden contact region and dowel's axial symmetry remain unobserved.                        |
| Prepared `insertion-migrated-v10-preferred-hover` bundle | Each object has one reference snippet at demo frame 100; the dowel surface used by the replay contains 318 sampled points.                                                                                                                                    | A single tray-side surface does not cover all held viewpoints. Adding views requires **registered** object frames, not copying predicted gripper poses as truth.                              |
| Stored masks and RGB-D at held frames                    | Masks often cover one exposed end during grasp/lift and more of the shaft near insertion. The current shape-only fallback rejects three replayed held frames from the two latest runs.                                                                        | The replay reconstructed masks from stored contours, so this is a diagnostic, not an exact reproduction of every worker input. The live backend also recorded `lost` proposals on held views. |

## The source demonstration has the same held-pose problem

The prepared `insertion-migrated-v10-preferred-hover` skill does not contain an
independently observed held pose. Its orange replay outline is the frame-130
pre-occlusion pose propagated by the recorded gripper after the annotated
`attach-held` event at arm sample 228 (7.638 s). At that event, the measured
gripper was 88.34° while the command was 97.42°; it did not settle near
97.81° until sample 235 (7.872 s). Closure and possible object motion thus
continued after the attachment assumption was made. The camera occludes the
contact, so the exact source of the resulting slip or offset is not directly
identified.

An offline sequential replay of the current SAM3 → DINO → RGB-D path through
demo frame 450 separates segmentation from pose registration. Its per-frame
results are saved locally at
`~/.cache/huggingface/lerobot/demos/insertion/flash/v1/analysis/demo_mask_pose_audit_current_backend.csv`.
On every frame
251–450 SAM3 returned a dowel mask (median 3,051 pixels), but the matcher
found a median of only five distinct reference correspondences and returned
no accepted pose. At frame 400 the mask had 3,423 pixels, but only seven
correspondences passed the configured similarity floor, below the 12 needed
for a rigid fit. A fresh mask at that frame also failed the current whole-shape
fallback. Reducing the appearance floor in a diagnostic produced up to 114
matches, but the candidate still disagreed with the segmented depth surface;
loosening the floor alone did not recover a supported pose. The prepared
reference has only one tray-side view at frame 100, while the object is later
seen upright and from another side.

The saved prediction is visibly wrong in clear held views. A color-assisted
RGB-D check at frame 400 estimates roughly 27 image pixels of centre offset
and 13° of axis disagreement between the projected prediction and the visible
dowel. Those are diagnostic estimates, not six-axis ground truth; the sampled
reference surface is incomplete. At frame 130, by comparison, 94% of its
projected points agree with measured depth within 10 mm. The initial pose
therefore looks plausible, but its rigid carry assumption is not verified.

The demo replay also had a separate display-time error on generated connections:
it drew the saved **end** held pose on every earlier camera frame, even though
the recorded gripper was still moving toward that endpoint. At frame 300 this
put the projected sampled-surface centre about 30 pixels from the frame-time
gripper-projected location. Replay now projects the same unverified grasp
attachment from the recorded gripper at that camera frame. This only removes
the future-endpoint display artifact; the frame-400 contact interval uses a
dense sample within about 12 ms of the camera frame, and its measured
27-pixel/13° discrepancy remains an estimation problem.

The demo viewer cannot draw preparation-time masks because that history was
not stored in the skill. A missing mask overlay is not evidence that SAM3 lost
the object. This audit establishes that the _demo relation_ used for insertion
is itself uncertain, so correcting only the live attachment cannot make the
transferred relation accurate. Future preparation needs to retain mask and
fit-failure evidence, update the object-to-gripper relation after closure from
independent partial observations, and reject a precise contact relation when
its task-relevant pose remains unverified. These requirements apply to any
held rigid object, not just this dowel.

The attachment lifecycle currently seeds `T_E_H` from a pregrasp object pose
and the **observed arm joints at the tick that dispatches** `attach-held`.
Dispatch does not mean the gripper has reached its commanded closure. In the
20:33:10 run, observed gripper position is about 90.7 at that event while its
command is about 97.4; it continues toward the command as the visible object
axis changes. This is a meaningful prior, not a measured settled grasp.
`HeldAttachmentUpdater` would accept a source-time-aligned independent pose,
but none is supplied by the current RGB-D backend in these four recordings.
The orange replay outline therefore keeps the same gripper-to-object
assumption. Its polygon also depicts the stored sampled surface, so polygon
edge error alone is not a pose-error measurement.

The present appearance path needs enough matched, depth-backed model points.
Its shape fallback compares **all** reference points against the live masked
surface in both directions and requires all passing object-frame orientations
to agree. That is appropriately conservative for a fully visible object but
often impossible when only one end is visible. Relaxing the residual cutoff
alone would invite sliding along a uniform shaft or false end swaps; it would
not make hidden degrees of freedom observable. The present backend publishes
either a fully supported `PoseEstimate` or `lost`, and the attachment updater
accepts only a fully observed pose. Thus even a useful partial axis constraint
has no path to the attachment today. The architecture's `EvidenceProposal`
contract must become a real interface before partial fitting can change
control.

## Estimation rule

Maintain a timestamped object belief, not a winner among named trackers. A
gripper attachment and an environment group predict motion conditional on a
relationship. Aligned RGB-D surface, mask boundary, free space, feature
correspondences and causal point motion provide measurement constraints where
they actually have support. A source may constrain a tip position or an axis
while leaving axial rotation or a hidden endpoint unresolved. Measurements
derived from the same camera pixels are correlated and must not count as
independent votes.

For each camera frame, estimate a set of plausible poses or a pose with an
observable subspace and task-point bounds. Evaluate candidates at that
frame's capture time, using measured arm history for the attachment prior.
When a slow result arrives, insert it at its source time and replay subsequent
motion before publication; never compare a new image to today's gripper pose.
The objective should combine independently calibrated terms:

- identity/correspondence consistency against registered reference snippets;
- robust depth and silhouette agreement on the **predicted visible** model
  surface, plus penalties for model surface in measured free space;
- point-motion and gripper/group motion residuals with uncertainty that grows
  when support disappears or contact may cause slip;
- physical object symmetry, retaining equivalent poses rather than choosing
  a convenient but unsupported orientation.

Use sensor noise and held-out residual calibration to scale these terms;
record each term and its support. Observability follows the measurement
Jacobian and the spatial distribution of visible points, not the number of
mask pixels alone. A single cap can support position without certifying the
shaft axis. A partial side view may support axis tilt but not translation
along a uniform shaft. A full-pose type must not turn these into fictitious
certainty. If two supported hypotheses remain, keep both or report the
task-relevant dimension unknown. The ring's environment-group estimate and
the dowel's gripper prediction have different relationship assumptions;
neither automatically overrides visible depth.

## Experiments and implementation order

1. **Create a fixed replay benchmark.** Preserve every source proposal,
   `lost` reason, mask and support IDs, fit diagnostics, camera capture time,
   measured gripper state, and selected belief in the episode extension.
   Align images, depth, motor history and proposals by source time. Annotate
   visible tips/axes and task landmarks at pregrasp, settled grasp, lift,
   carry and pre-release in the four held recordings; mark occluded labels
   unknown. Include the thirteen pre-attach attempts as false-acceptance and
   recovery tests. Hold out complete attempts when selecting parameters.
2. **Build a prior-conditioned partial measurement producer against the
   existing one-view reference first.** Start fits around gripper/group and
   appearance hypotheses, but permit independent relocalization. Render
   which model points should be visible in the source image; fit their
   measured subset with robust depth, boundary and free-space terms. Report
   supported pose directions, multiple symmetry-equivalent solutions, and
   residuals. Do not require the whole reference cloud to be present. Do not
   turn a low one-way nearest-neighbor residual into a full pose observation.
   This first ablation shows what partial fitting alone buys, and where the
   reference is missing a visible face.
3. **Expand reference coverage without assuming perfect attachment.**
   Register candidate demo/live snippets of the same physical instance into
   one stable object frame using RGB-D correspondences and a small pose graph.
   Use gripper motion as a noisy edge, not the registration truth. Reject or
   retain alternative alignments when the view is ambiguous. Compare one
   snippet, additional demo snippets, and verified live snippets separately
   with the same partial fitter.
4. **Fuse measurements in the one scene estimator.** Keep the existing
   source-time replay boundary. Update object and `T_E_H` constraints from
   accepted partial evidence during and after closure, using the arm pose at
   each image time. Track attachment uncertainty through later contact; a
   closing command or a clock timeout is not proof of a rigid grasp. Run the
   new producer in shadow mode first, recording proposed corrections without
   changing robot goals.
5. **Activate task-relevant control only after replay validation.** Transport
   may proceed under a wide held-pose belief; insertion needs bounds on the
   dowel tip/axis relative to the ring. If a required dimension is unknown,
   choose a safe view-changing motion when one exists, or report insufficient
   evidence. The 30 Hz motor loop keeps consuming its queue while asynchronous
   perception and planning revise future commands. Release verification is a
   separate milestone, not the completion of a gripper command.

The shadow comparison should report: fraction of frames with a usable
task-point/axis bound; false confident corrections; future-frame RGB-D
reprojection and free-space violations; manually labeled tip/axis error on
visible frames; post-grasp attachment updates; source-to-publication age;
perception p50/p95/p99; 30 Hz motor-tick gaps; and physical outcomes after
activation. A method that returns fewer full poses but fewer wrong contact
poses is preferable. No single residual or successful video is ground truth.

## Trade-offs to decide with the benchmark

| Choice                                       | Benefit                                                        | Cost / failure mode                                                              | Decision test                                                                          |
| -------------------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Full appearance matches plus whole-shape fit | High specificity on clear views; known baseline                | Abstains under occlusion and unseen reference faces                              | Compare held-view coverage and false accepts at equal frame budget.                    |
| Prior-conditioned partial geometry           | Uses visible cap/shaft with one camera; can correct attachment | Local minimum, axial slide, wrong mask, correlated depth bias                    | Require future-view prediction and free-space checks, not current-frame fit alone.     |
| Causal point tracks                          | Cheap motion between expensive updates                         | Plain surfaces and gripper crossings lose identity; motion is not absolute pose  | Ablate on grasp/carry segments and record support-point survival.                      |
| Gripper/group prediction                     | Timely during full occlusion; no perception stall              | Slip or a false support relation preserves an error                              | Compare predicted versus later visible task points, especially across closure/contact. |
| More registered views or learned shape       | Better visible-surface coverage and retrieval                  | Registration error, storage/compute, possible train/test leakage                 | Add views incrementally and test on held-out attempts and layouts.                     |
| Multiple hypotheses / uncertainty            | Honest symmetry and occlusion handling                         | More state and planner integration                                               | Measure wrong confident insertion versus safe abstention and recovery rate.            |
| Active observation                           | Can resolve an otherwise ambiguous axis                        | Motion time, reach/collision constraints; no benefit if it repeats the same view | Require predicted information gain for a task-relevant dimension.                      |

This sequence implements the existing architecture's shared measurement path.
It does not introduce an object-name rule, a new age-based truth threshold, or
a separate held-object tracker whose result bypasses scene evidence.
