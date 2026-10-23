# Dowel insertion: candidate pose evidence across the recorded demo

The [synchronized replay](full_demo_fusion_evidence.mp4) runs through all 713 RGB frames (23.8 s). It draws the SAM3 dowel/ring boundaries over the camera image, the Point2Pose dowel axis and ring center in cyan, any scene-group fallback in green, and a gripper-propagated dowel axis in orange. The [per-frame measurements](full_demo_fusion_evidence.csv) compare projected points from each candidate with the same frame's aligned depth and SAM3 mask. The video is an **evidence comparison**, not a fused pose used for control.

The sources cover different intervals. The saved SAM3 masks and dowel Point2Pose trace cover frames 100–550; the ring Point2Pose trace covers frames 100–712. This Point2Pose trace used its internal SAM2 segmentation; the separately drawn SAM3 mask and aligned depth evaluate the proposal. The shared causal extractor uses SAM3, appearance matching, RGB-D and scene groups, and covers frames 0–590. It has direct appearance/RGB-D measurements for the dowel on 166 frames, scene-group motion for 68 frames (159–228), and object-point motion on 9 frames (238–251). It has no dowel estimate on 348 frames. The recorded grasp begins around frame 229. The ring has direct appearance/RGB-D estimates on 567 frames and scene-group fallback on 24 frames. Consequently the group source does **not** currently bridge the held dowel through insertion.

To show what a rigid attachment prior adds, the orange dowel pose starts from the Point2Pose pose at frame 340 and follows the recorded gripper pose through release at frame 514. This anchor is deliberately explicit: it is not a verified grasp-time attachment or an automatic recovery. Seeding it from the pregrasp pose instead gave a visibly worse projection, demonstrating why a gripper prior alone cannot correct a bad attachment estimate.

For the insertion interval (frames 400–513), the median fractions of _valid projected model rays_ are:

| Object and candidate                            | Mask-and-depth agreement | Measured free-space contradiction |
| ----------------------------------------------- | -----------------------: | --------------------------------: |
| Ring, direct RGB-D / scene fallback             |                      62% |                              0.8% |
| Ring, Point2Pose                                |                      44% |                               23% |
| Dowel, Point2Pose                               |                      26% |                               20% |
| Dowel, gripper prediction anchored at frame 340 |                      25% |                                7% |

The measurements use a 5 mm depth agreement tolerance and exclude locally discontinuous depth (a 10 mm span over 3×3 pixels). These are **consistency measurements, not percentages of correct poses**. The object models contain only points from the first visible view; after a turn, newly exposed surfaces need not agree with that partial model even when the pose is right. The mask and depth also contribute to some estimates, so this is not independent ground truth. Free-space contradictions are more diagnostic than a low agreement fraction, but either can be affected by registration and depth artifacts. The comparison is stable in direction at 3 and 10 mm tolerances: the ring's causal estimate still has fewer free-space contradictions than Point2Pose, and the gripper-propagated dowel still has fewer than the direct Point2Pose candidate. The script exposes the tolerance as an argument for further sensitivity checks.

A provisional fusion rule can avoid a weighted score: one candidate dominates another only if it explains at least as many valid rays **and** puts no more rays into measured free space. Otherwise retain both. During insertion, the causal ring estimate dominates Point2Pose on all 114 compared frames. For the dowel's 113 frames with both candidates, the gripper prediction dominates on 44, Point2Pose on 13, and 56 are incomparable. Even this rule only ranks these candidates against a partial first-view model; it does not certify the winner as an accurate pose. A temporal filter would also be needed to prevent source-switching jumps.

The combined architecture suggested by this evidence is a single time-aligned candidate bank: SAM + depth defines the currently visible surface, Point2Pose proposes object motion, scene groups propagate motion during environmental occlusion, and the gripper supplies a held-object prior. Candidate evaluation should compare projected visible surface, foreground occlusion, measured free space, and temporal consistency, then keep uncertainty or multiple hypotheses when they disagree. A gripper-to-object attachment update should require a supported visual observation. This recording supports that strategy for the ring, but **does not yet establish a reliable held-dowel pose during insertion**. The next experiment should build a multi-view surface representation from the demo and test whether it explains the held dowel without the first-view model's missing-surface bias, using the same full-demo replay before changing live control.

Regenerate the replay and CSV with:

```bash
python docs/proofs/flash-tracker-comparison/visualize_full_demo_fusion.py \
  --output docs/proofs/flash-tracker-comparison/full_demo_fusion_evidence.mp4 \
  --metrics docs/proofs/flash-tracker-comparison/full_demo_fusion_evidence.csv
```
