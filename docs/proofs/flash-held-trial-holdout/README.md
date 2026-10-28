# Held-object pose: independent recorded-trial check

The newer dowel preparation does **not** establish that held-object estimation is solved. Replaying the current FLASH surface backend and gripper attachment updater on a separate, unverified 20 October insertion trial exposes a real failure: later in the carry, a visible, plausible dowel direction is rejected as a _full rigid pose_, leaving a less accurate gripper prediction in control. This is an offline counterfactual, not a physical retest or ground-truth 6D pose measurement. The current runtime policy is unchanged.

![Measured dowel against rejected vision and gripper fallback](axes_comparison.png)

At recorded camera frame 481, the rejected vision candidate's long axis differs from the tracked visible surface's dominant 3D direction by 9.3°, versus 18.7° for the gripper prediction. At frame 565, those differences are 9.7° and 33.6°. The white outline is the recorded SAM mask. The measured surface uses the tracker's cleaned, spatially sampled, connected depth points. This checks the object's **direction**, not which end is which, its roll, its tip position, or its full pose. The 2D image overlay alone is not the 3D measurement.

## Reproduction and scope

- Trial: `~/.cache/huggingface/lerobot/demos/flash_insertion_trial_20261020_020-20261020-203310-655401`, episode 0 (646 motor rows, 553 unique RGB-D frames). Its saved outcome is `unknown / unverified outcome`.
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
```

The held interval contains 284 unique RGB-D frames. The current backend returned 115 observed dowel poses and 169 `LOST` estimates; the gripper attachment was available on 265 frames. For the later recorded interval (motor rows 330 through release), the rejected vision candidate is closer to the tracker's visible-surface direction on 100 of 137 frames. Its median 3D direction gap is 10.5°, versus 15.5° for the gripper fallback. This is a diagnostic comparison, **not** a measured controller success rate. Some frames favor the gripper. At row 340, the tracker's visible surface favors the gripper direction (4.1° versus 10.2°); a rule that blindly accepts every vision candidate would fail there. A separate PCA of all clean mask pixels gives different angles for that weak view, demonstrating why one surface direction cannot be treated as ground truth.

## What failed

The current backend requires enough spread in _two_ visible directions to accept a complete 6D pose. It also checks the latest depth against a fixed first-view reference surface. As the dowel rotates and a different surface becomes visible, that reverse reference check rises above its acceptance limit even when the candidate follows the visible long axis. At frame 565, the visible surface-fit residual is 5.58 mm, the reverse first-view residual is 14.59 mm, and the two-direction spread check also fails. The recorded failure label mentions only the spread check; multiple gates failed. The later vision candidate has a larger fraction of its projected model points supported inside the mask than the gripper fallback (19% versus 1% at frame 565), while the fallback's small supported patch has a lower depth residual. Neither one scalar score nor the plotted axis proves a full pose.

The inference is that full-pose acceptance is conflating _directional_ evidence with all six pose coordinates, while the frozen partial reference is a poor reverse-depth comparator after rotation. A generic fix should represent supported degrees of freedom and uncertainty explicitly, test current measured geometry against each hypothesis, preserve ambiguity where the surface does not constrain a direction, and then use gripper/group priors only for the unsupported degrees of freedom. It must be checked on multiple objects and trials before changing execution. The older preparation/UI selection issue is separate: opening the newest preparation would make the display less misleading but would not remove this algorithmic failure.

The shared surface backend now records the visible patch's direction, three 3D spreads, and point count in every source-frame proposal, including rejected full poses. This adds evidence for demo preparation, live testing and replay without altering the selected pose. Rerunning this trial before and after the addition gave identical observed/lost labels, candidate poses, accepted attachment updates and fallback states on all 553 frames. The only existing diagnostic that changed was its lateral spread at under 3 × 10⁻¹⁴ mm, numerical roundoff from requesting eigenvectors alongside eigenvalues. This field is a partial **measurement**, not yet an accepted pose correction.
