# Held-object pose: independent recorded-trial check

The newer dowel preparation does **not** establish that held-object estimation is solved. Replaying the current FLASH surface backend and gripper attachment updater on a separate, unverified 20 October insertion trial exposes a real failure: later in the carry, a visible, plausible dowel direction is rejected as a _full rigid pose_, leaving a less accurate gripper prediction in control. This is an offline counterfactual, not a physical retest or ground-truth 6D pose measurement. The current runtime policy is unchanged.

![Measured dowel against rejected vision and gripper fallback](axes_comparison.png)

At recorded camera frame 481, the rejected vision candidate's long axis differs from the tracked visible surface's dominant 3D direction by 9.3°, versus 18.7° for the gripper prediction. At frame 565, those differences are 9.7° and 33.6°. The white outline is the recorded SAM mask. The measured surface uses the tracker's cleaned, spatially sampled, connected depth points. This checks the object's **direction**, not which end is which, its roll, its tip position, or its full pose. The 2D image overlay alone is not the 3D measurement.

## Reproduction and scope

- Trial: `~/.cache/huggingface/lerobot/demos/flash_insertion_trial_20261020_020-20261020-203310-655401`, episode 0 (646 motor rows, 553 unique RGB-D frames). Its saved outcome is `unknown / unverified outcome`.
- Independent second attempt: `~/.cache/huggingface/lerobot/demos/flash_insertion_trial_20261020_020-20261020-201838-054325`, episode 0 (534 unique RGB-D frames). Its compressed results, masks and axis audit are in `second-trial/`.
- Current prepared bundle: `~/.cache/huggingface/lerobot/demos/insertion/flash/v1/prepared/flash-dowel-ring-20261028-depth-audit-v11/bundle.json`. No recorded trial pose proposals were fed to the backend.
- `replay_trial.py` deduplicates aligned RGB-D frames, re-runs current segmentation and surface fitting, and seeds/updates the gripper attachment using source-time-aligned saved joint states. `result.json.gz` retains accepted estimates **and rejected raw candidates**; `masks.npz` retains the per-frame masks. `audit_axis.py` compares both hypotheses with the principal direction of the tracker's measured surface, and separately with all clean mask-depth pixels. `axis_audit_all_held.json.gz` contains this comparison for the entire held interval. Reproduction needs the local trial, bundle, calibration, and model weights; these are not copied into git.
- The trial's original tracker and inference scheduling are not reproduced. The group tracker was not rerun; this checks the current vision and gripper branches, not the full live engine. Replaying 553 frames took about 109 seconds, so this run says nothing about live frame rate.

From the repository root, with the FLASH native benchmark environment installed:

```bash
PYTHONPATH=src /tmp/flash-native-benchmark-venv/bin/python docs/proofs/flash-held-trial-holdout/replay_trial.py \
  --trial ~/.cache/huggingface/lerobot/demos/flash_insertion_trial_20261020_020-20261020-203310-655401 \
  --bundle ~/.cache/huggingface/lerobot/demos/insertion/flash/v1/prepared/flash-dowel-ring-20261028-depth-audit-v11/bundle.json \
  --held-role dowel --output-dir docs/proofs/flash-held-trial-holdout \
  --sample-row 340 --sample-row 417 --sample-row 511
PYTHONPATH=src /tmp/flash-native-benchmark-venv/bin/python docs/proofs/flash-held-trial-holdout/audit_axis.py \
  --result docs/proofs/flash-held-trial-holdout/result.json.gz --all-held \
  --output docs/proofs/flash-held-trial-holdout/axis_audit_all_held.json.gz
.venv/bin/python docs/proofs/flash-held-trial-holdout/audit_future_axis.py \
  --result docs/proofs/flash-held-trial-holdout/result.json.gz \
  --axis-audit docs/proofs/flash-held-trial-holdout/axis_audit_all_held.json.gz \
  --horizon-s 0.3 --horizon-s 0.6 --horizon-s 1.0 \
  --output docs/proofs/flash-held-trial-holdout/future_axis_comparison.json.gz
```

For the second attempt, substitute its trial path above and use `docs/proofs/flash-held-trial-holdout/second-trial` for the replay output directory and axis-audit paths.

The [future-axis check](audit_future_axis.py) compares each proposal's direction in the measured gripper frame at one camera time with the visible direction in a later camera frame. It is a check of a rigid-carry prediction, not an in-sample fit. The configured horizons below are evaluation probes, not runtime thresholds; the script accepts any positive horizon. At each probe it selects the first camera frame at or after the requested delay. The full per-frame results are saved for [20:33:10](future_axis_comparison.json.gz) and [20:18:38](second-trial/future_axis_comparison.json.gz).

| Attempt  | Future delay | Compared frames | Vision predicts later direction better | Median vision gap | Median gripper gap |
| -------- | -----------: | --------------: | -------------------------------------: | ----------------: | -----------------: |
| 20:33:10 |        0.3 s |             100 |                                     90 |             11.5° |              18.2° |
| 20:33:10 |        0.6 s |              99 |                                     84 |             11.2° |              18.2° |
| 20:33:10 |        1.0 s |              94 |                                     75 |             11.2° |              17.8° |
| 20:18:38 |        0.3 s |             147 |                                     76 |             13.7° |              14.0° |
| 20:18:38 |        0.6 s |             145 |                                     69 |             14.1° |              13.7° |
| 20:18:38 |        1.0 s |             138 |                                     64 |             14.9° |              14.1° |

This strengthens the case that the first attempt's rejected direction is useful beyond its source image, while the second still supplies no general rule to favor vision. Adjacent masks and depth errors can be correlated; the check does not validate position, end identity, contact-tip accuracy, or physical outcome.

The same future check can carry only the **direction measured from the visible
surface** at the source frame into the measured gripper frame. It makes no
claim about the dowel's center, either end, or rotation about that direction.
At a 0.6-second look-ahead it was closer to the later visible direction than
the saved gripper prediction in 93/99 and 142/145 source frames, respectively:

| Attempt  | Source frames | Median partial-direction gap | Median gripper-prediction gap |
| -------- | ------------: | ---------------------------: | ----------------------------: |
| 20:33:10 |            99 |                         7.3° |                         18.2° |
| 20:18:38 |           145 |                         5.2° |                         13.7° |

The source frames overlap heavily. Grouping them into non-overlapping
0.6-second source-time blocks gives a positive median directional improvement
in 11/11 and 13/13 blocks. These are still only two trials of the same object,
and the source and future directions share a camera, mask and depth pipeline.
This is evidence to test a **partial orientation constraint** separately from
full-pose correction; it does not establish a calibrated error bound or justify
moving the hidden insertion tip in control.

The trial audit now projects each measured line onto the closest orientation
of the saved gripper attachment, leaving its pivot position prior-based. The
median required turn is 17.8° in the first attempt and 13.3° in the second;
individual turns reach 72.6° and 50.9°. That is too large to present as a
small pose refinement without validating the task point and the source-time
attachment. The projection preserves the unmeasured roll and reports the
margin between the two possible end assignments.

This line correspondence is not implied by axial symmetry. In the saved
reference clouds, the dowel's declared symmetry axis agrees with the cloud's
dominant direction, and its 3D spreads are 3.6, 7.3 and 30.9 mm. The ring's
symmetry axis is 90° from its cloud's dominant direction, whose two largest
spreads are close (12.8 and 14.2 mm). Applying the dowel's visible-axis rule
to every symmetric object would therefore be a geometry error. A reusable
direction constraint must carry an established object-to-observation line
correspondence and its evidence, rather than infer it from a symmetry label.

The surface pipeline already divides its spatial samples into interleaved fit and validation tiles. For each source image, we also compare candidate and gripper directions with the principal direction measured from **each half separately**. A proposal gets a shadow vote only when it is closer in both halves; otherwise the vote is ambiguous. At the 0.6-second future check:

| Attempt  | Both halves favor | Source frames | Voted source predicts the later direction better |
| -------- | ----------------- | ------------: | -----------------------------------------------: |
| 20:33:10 | Vision            |            86 |                                               81 |
| 20:33:10 | Gripper           |             5 |                                                4 |
| 20:18:38 | Vision            |            75 |                                               64 |
| 20:18:38 | Gripper           |            64 |                                               59 |

The remaining 8 and 6 source frames respectively gave split disagreement. The vote uses no scene-specific angle cutoff, but the halves share a camera, mask and depth biases; their agreement is not a calibrated confidence probability. These are correlated frames of only two attempts with one physical object. The result supports carrying split directional evidence into source-frame proposals and testing a partial held-pose update in shadow; it does **not** yet justify changing control.

Both held intervals contain frames with visible masks but no accepted full pose. The table compares each proposal with the tracker's connected visible-surface direction **after the last accepted direct pose** in that attempt:

| Recorded attempt | Held RGB-D frames | Direct poses / `LOST` | Later frames | Vision closer to visible direction | Median vision gap | Median gripper gap |
| ---------------- | ----------------: | --------------------: | -----------: | ---------------------------------: | ----------------: | -----------------: |
| 20:33:10         |               284 |             115 / 169 |          103 |                           94 / 103 |             10.5° |              17.9° |
| 20:18:38         |               339 |             149 / 190 |          149 |                           81 / 149 |             13.4° |              13.3° |

This is a diagnostic comparison, **not** a measured controller success rate or full-pose ground truth. The first attempt has a strong late directional discrepancy, while the second does not justify preferring vision overall. Even in the first, some frames favor the gripper. At row 340, the tracker's visible surface favors the gripper direction (4.1° versus 10.2°); a rule that blindly accepts every vision candidate would fail there. A separate PCA of all clean mask pixels gives different angles for that weak view, demonstrating why one surface direction cannot be treated as ground truth.

## What failed

The current backend requires enough spread in _two_ visible directions to accept a complete 6D pose. It also checks the latest depth against a fixed first-view reference surface. As the dowel rotates and a different surface becomes visible, that reverse reference check rises above its acceptance limit even when the candidate follows the visible long axis. At frame 565, the visible surface-fit residual is 5.58 mm, the reverse first-view residual is 14.59 mm, and the two-direction spread check also fails. The recorded failure label mentions only the spread check; multiple gates failed. The later vision candidate has a larger fraction of its projected model points supported inside the mask than the gripper fallback (19% versus 1% at frame 565), while the fallback's small supported patch has a lower depth residual. Neither one scalar score nor the plotted axis proves a full pose.

The inference is that full-pose acceptance is conflating _directional_ evidence with all six pose coordinates, while the frozen partial reference is a poor reverse-depth comparator after rotation. A generic fix should represent supported degrees of freedom and uncertainty explicitly, test current measured geometry against each hypothesis, preserve ambiguity where the surface does not constrain a direction, and then use gripper/group priors only for the unsupported degrees of freedom. It must be checked on multiple objects and trials before changing execution. The older preparation/UI selection issue is separate: opening the newest preparation would make the display less misleading but would not remove this algorithmic failure.

The shared surface backend now records the visible patch's direction, three 3D spreads, point count, and both spatial-split directions in every source-frame proposal, including rejected full poses. This adds evidence for demo preparation, live testing and replay without altering the selected pose. Rerunning the first trial before the split addition gave identical observed/lost labels, candidate poses, accepted attachment updates and fallback states on all 553 frames. The only existing diagnostic that changed was its lateral spread at under 3 × 10⁻¹⁴ mm, numerical roundoff from requesting eigenvectors alongside eigenvalues. These fields are partial **measurements**, not yet accepted pose corrections.

An ablation that stopped rejected visual fits from seeding later visual fits made the second attempt substantially worse: held attachment unavailability rose from 5 to 178 camera frames. We reverted that behavior. The backend now retains its recovering visual hypothesis and records a separate motion-only prediction from the last RGB-D accepted pose when frame correspondence is continuous. Neither prediction is automatically chosen over the other. Replaying both attempts after this diagnostic addition reproduced the saved candidate and held-pose matrices and all accepted/lost decisions exactly (553 and 534 frames). This is behavior parity, not evidence that either pose is accurate.

### The independent motion prediction does not have a universal advantage

The updated replay artifacts also include that motion-only proposal. The table compares each source-frame proposal with the dowel's **later visible direction in gripper coordinates**, at the first available image at least 0.6 s later. It covers only frames after the last accepted direct pose in each held interval. The numbers do not measure the hidden tip, full 6D accuracy, or contact success.

| Held-out attempt | Compared frames | Median later-direction gap: vision | Motion-only | Gripper prediction |
| ---------------- | --------------: | ---------------------------------: | ----------: | -----------------: |
| 20:33:10         |              99 |                              11.2° |       26.8° |              18.2° |
| 20:18:38         |             145 |                              14.1° |       12.7° |              13.7° |

The winner reverses between the two attempts. A source-frame check against two interleaved subsets of the current masked depth surface often identifies the better direction: in the first attempt both subsets favor vision in 86 frames, and vision is closer to the later direction in 85; in the second they favor motion-only in 118 frames, and motion-only is closer in 94. It also makes errors: in the second attempt motion-only wins that source check in 24 frames where vision predicts the later direction better. The split subsets share segmentation, depth, and calibration errors, and neighboring camera frames are correlated. This is evidence for preserving and comparing independent hypotheses, not a calibrated confidence or a safe automatic controller switch.

The earlier and newer _prepared demos_ also use byte-identical saved reference point arrays, annotations, and keypoints for both insertion and gamepad stacking, with identical vision settings. Their different conflict counts reflect different code decisions on the same inputs. That is a real algorithm change, but neither the lower conflict count nor the viewer selecting the newer snapshot establishes general accuracy. The held-out trial disagreement above is the counterexample to treating the newer preparation as a solved tracker.

### What the visible-surface information adds

The current replay now records how many points on the saved first-view surface
are close to each frame's measured RGB-D patch, plus six local motion
sensitivities at the proposed fit. Removing only this new field from the two
reruns leaves **every existing result field identical** across 553 and 534
camera frames. The [audit script](audit_surface_information.py) joins these
records to the 0.6-second later-direction check:

| Trial    | Frames with the new measurement | Median matched first-view surface in compared frames | Vision closer than motion-only to later visible direction |
| -------- | ------------------------------: | ---------------------------------------------------: | --------------------------------------------------------: |
| 20:33:10 |                         535/553 |                                                 9.2% |                                                     92/99 |
| 20:18:38 |                         526/534 |                                                 9.6% |                                                    47/145 |

The similar surface coverage does not select the better pose source. The
measurement explains which motions one _proposed fit_ is locally sensitive
to; it cannot establish that the correspondence or reference geometry is
right. These future frames overlap heavily and check direction only. FLASH
records this evidence in Demo, Test and Replay, but does not use it to change
the controller. Selecting between vision, gripper and scene-motion proposals
still needs a source-frame comparison of the candidates against the same
measured image, with unresolved dimensions preserved.

### A same-image mask/depth winner is still unsafe

The existing held-pose comparison uses two measurements of a saved surface at
one camera frame: the fraction of its projected points supported by the object
mask and the 90th-percentile depth difference there. We applied that same
two-metric dominance test to each rejected vision proposal and the gripper
prediction, requiring the trial's configured 12 supported pixels for both.
The [surface-choice audit](audit_surface_choice.py) reads the recorded
same-frame measurements and joins its choices to the later visible-direction
check; it never changes the replay or controller.

| Attempt  | Depth pixels used      | Vision chosen | Chosen vision closer to later visible direction | Gripper chosen | Unresolved or insufficient |
| -------- | ---------------------- | ------------: | ----------------------------------------------: | -------------: | -------------------------: |
| 20:33:10 | All valid depth        |         24/99 |                                           19/24 |              0 |                      75/99 |
| 20:18:38 | All valid depth        |        87/145 |                                           29/87 |              4 |                     54/145 |
| 20:33:10 | Reliable mask interior |          8/99 |                                             6/8 |              1 |                      90/99 |
| 20:18:38 | Reliable mask interior |       140/145 |                                          64/140 |              0 |                      5/145 |

Even cleaner depth does not make this first-view surface comparison a reliable
full-pose selector: in the second attempt it favors vision in nearly every
frame while vision often predicts the later visible direction less well. The
test compares **direction only**, uses correlated neighboring frames, and has
no hidden-tip ground truth. It does rule out applying the current held-pose
dominance rule directly to rejected raw proposals. The next estimator needs a
partial measurement that retains competing hypotheses and establishes which
dimensions the currently visible geometry actually distinguishes; more
registered views may be necessary where the saved first view does not cover
the observed surface.

Reproduce from the saved audits with:

```bash
.venv/bin/python docs/proofs/flash-held-trial-holdout/audit_surface_choice.py \
  --axis-audit docs/proofs/flash-held-trial-holdout/axis_audit_all_held.json.gz \
  --future-axis docs/proofs/flash-held-trial-holdout/future_axis_comparison.json.gz \
  --min-supported-pixels 12
```

Use the `second-trial/` paths for the other attempt. This script's count is a
measured counterexample, not a tuned runtime threshold.
