# Cross-episode point transfer pilot

This pilot asks whether TrackEverything can transfer **some** surface-point
identities from a recorded FLASH demo to a later run of the **same physical
dowel**. It does not assume that the annotated tool tip is visible or even
queried. A successful rigid set of surface correspondences could instead carry
an annotated point expressed in the demo object's frame.

## Inputs and method

- Demo: `~/.cache/huggingface/lerobot/demos/insertion/recording`, episode 0,
  frame 100. The source mask is the dowel SAM mask from
  `trackeverything-growth-ui-20261021-a`. Interior pixels with measured depth
  provide 112 spatially spread query points.
- Trials: six `flash_insertion_trial_20261020_020-*` recordings from October 20,
  2026, at 17:52, 18:04, 19:11, 19:51, 20:13 and 20:33. RGB and **metric** depth
  come from their embedded LeRobot dataset images. Live dowel mask boundaries
  come from each run's `tracker_episode_000000.jsonl` and are used only where
  their camera frame ID exactly matches a dataset frame.
- Model: upstream TrackEverything commit
  `f0c18dd9714b9f85825c45a1c3aae685da4e6222`, with the installed
  `katefgroup/TrackEverything` checkpoint. The test gives it the source RGB-D
  frame followed immediately by 15 live RGB-D frames, identical calibrated
  intrinsics and a stationary camera. It queries source points in sparse mode.
- A predicted point counts as _surface-supported_ only when its visibility
  logit is positive, its projection lies inside the live dowel mask, and its
  predicted depth agrees with measured depth within 10 mm. A second check fits
  one rigid transform to the supported 3D pairs using a 10 mm inlier radius.
  The zero-motion control projects source 3D points unchanged into the live
  frame. These are diagnostic thresholds, not runtime acceptance rules.

The [benchmark](../../../benchmarks/flash_cross_episode_trackeverything.py)
reproduces the bridge and writes tracks, overlays and per-frame metrics. The
[CSV](startup-bridges.csv) contains the startup values, and
[this grid](bridge-evidence.jpg) shows the source, six starts and two carried
views. Red is the recorded live dowel mask; green points pass the mask and
depth check. Cyan points miss the mask; magenta points lie within the mask but
miss the depth check.

| Run   | Supported / 112 | Zero motion / 112 | Rigid inliers | Supported span of dowel axis |
| ----- | --------------: | ----------------: | ------------: | ---------------------------: |
| 17:52 |              38 |                23 |            30 |                          46% |
| 18:04 |              30 |                25 |            26 |                          42% |
| 19:11 |              63 |                33 |            51 |                          87% |
| 19:51 |              24 |                15 |            22 |                          25% |
| 20:13 |              39 |                24 |            31 |                          53% |
| 20:33 |              61 |                27 |            39 |                          72% |

The strongest case has 63 surface-supported predictions, while 33 unchanged
source points would happen to overlap the live surface. Its rigid consensus is
far stronger than chance pairing: with the same 300-trial, 10 mm RANSAC check,
52 correspondences agree; across 20 random permutations of the target-point
identities, the maximum is 14. At a stricter 5 mm depth tolerance, the six
surface-supported counts are 21, 17, 27, 20, 24 and 34 respectively. At a
5 mm rigid threshold, the 19:11 and 20:33 cases retain 29 and 28 inliers.

Viewpoint matters. Demo tray frames 120–160 progressively lose support, and
demo frames 260, 300, 340, 500 and 580 yield zero surface-supported points at
the 20:33 tray start. Eight direct hard-cut tests from demo views to _held_
live views yielded at most two surface-supported points. The model is not a
general cross-episode re-identification mechanism out of the box.

Two longer clips put demo frame 100 before the live run and continue
TrackEverything through grasp and carry. Initially supported query identities
are mostly lost at grasp: in the 20:33 run, 41 initially rigid-consistent
queries are surface-supported at the start, none at frame 85, and only 0–1
through most of frames 160–218. The 19:11 run starts with 53 such queries,
falls to zero at frame 60, and has 10 at frame 161. Other queried points later
land on the live dowel, but their accuracy as the _same material points_ has
not been established. Some predicted points visibly follow the gripper. Thus
the startup bridge does **not** yet give a reliable object frame or hidden-tip
trajectory through contact.

These measurements use the run's own SAM-derived mask as surface evidence;
they are not independent hand-labelled ground truth. A smooth, nearly
symmetric dowel also makes individual material-point identity intrinsically
hard to verify from RGB-D. Partial rigid agreement is real evidence for
cross-episode registration, but is not yet a millimetre-level endpoint error
measurement.

The next tracker experiment should retain the verified demo-to-live rigid
anchor, add newly observed live points in that anchored frame, and enforce
rigid co-motion with mask and depth agreement. It should report uncertainty
and drop or relocalize the anchor when support collapses. The live FLASH
runtime was not changed by this pilot.
