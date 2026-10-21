# Causal tracker comparison on the insertion demo

Captured 2026-10-21 on the local RTX 5090 rig, using frames 100–550 of episode 0 in `/home/feit/.cache/huggingface/lerobot/demos/insertion/recording/` (aligned 848×480 RGB-D at 30 Hz). The dataset and depth images remain local; the measurements and diagnostic images below are preserved here. No robot motion was commanded for this comparison.

## Inputs and scoring

The Point2Pose run used the pinned local fork `95bd7a8808fc1aaf015d510dffcccaa3c6971a9c` through the sealed bridge at `80d90b783`, initialized with the dowel or ring mask at frame 100. The prepared FLASH trace is `insertion-migrated-v7-02262e7ed866.npz`, made by the branch's causal SAM3 + DINO + RGB-D preparation. BundleSDF started from NVIDIA's `ffa67d425240b5b76d2e387a7dd3d3735a7cf1a1` checkout and received the sealed per-frame masks from `objects.npz`; a second dowel run received masks freshly emitted by this branch's SAM3 adapter. All methods read the same recorded RGB and metric depth, but their pose-fitting inputs differ. BundleSDF's final global NeRF pass was disabled; only pose files written during its online frame loop were scored.

The surface score is the 90th percentile distance from _currently segmented, depth-backed pixels_ to the frame-100 depth point cloud transformed by each pose. The same saved segmentation is used to score every method. It is a useful check for large pose drift, but it is **not 6D ground truth**: a single reference view lacks newly exposed surfaces, and a partial mask can hide an error. A lost pose has no score. The prepared trace stores an identity placeholder for lost poses; scoring that matrix would falsely report a large estimation error. BundleSDF frames logged as `FAIL` are likewise excluded even though its recorder writes a pose file for them.

![Surface agreement through the demo](surface_comparison_all.png)

| Object and interval      | Point2Pose                             | BundleSDF online | Prepared SAM3 + DINO + depth |
| ------------------------ | -------------------------------------- | ---------------- | ---------------------------- |
| Dowel, pregrasp 100–149  | 50/50 poses; 1.4 mm median surface p90 | 50/50; 1.3 mm    | 50/50 observed; 1.3 mm       |
| Dowel, carry 300–399     | 97/100; 23.4 mm                        | 100/100; 89.3 mm | 0/100 observed               |
| Dowel, insertion 400–500 | 100/101; 8.6 mm                        | 101/101; 68.2 mm | 0/101 observed               |
| Ring, insertion 400–500  | 101/101; 4.1 mm                        | 101/101; 3.0 mm  | 101/101 predicted; 3.5 mm    |

The BundleSDF dowel trace loses the correct object pose after the grasp occlusion, then continues issuing poses with a large surface disagreement. Projecting the _same frame-100 points_ shows the divergence directly: cyan is BundleSDF, orange is Point2Pose, and green is the scoring mask.

![Dowel pose projections](bundle_p2p_overlay_grid.png)

For the ring, Point2Pose and BundleSDF agree on the transformed reference-cloud center to a median 2.5 mm and on the ring's unoriented normal to 4.5° during insertion. The ring moves in the recording, so this is stronger than agreement on a stationary target. The prepared trace's ring poses are marked **predicted**; its surface fit does not establish direct ring observation.

A full 6D orientation difference is not automatically an insertion error: rotation around a cylindrical dowel's shaft is immaterial to fitting it through a round hole. To check the dimensions that matter, I estimated the dowel's shaft from the first-view depth cloud's principal axis and a candidate end from its 97th-percentile extent, then propagated that same point and axis with each tracker's online pose. On frames 400–500 where both emitted a pose, the trackers' candidate-end positions differ by a median **65.9 mm** and their _unoriented shaft axes_ by **72.5°**. This disagreement is not an axial-roll ambiguity. The point is a first-view geometric proxy, not a calibrated tip or independent ground truth; the image projection and surface score above supply the evidence that BundleSDF's branch is wrong in this interval.

## Segmentation and recovery checks

The current SAM3 adapter processed the 451 frames causally at 21.8 ms median and 23.5 ms p95 per frame after its first-frame load. Its dowel mask and the sealed Point2Pose mask have median intersection-over-union 0.88 during insertion. During grasp, the sealed mask is empty for 25 frames and SAM3's for 21. Both lose the dowel in the deepest occlusion; both outline it again during carry and insertion.

![SAM3 and sealed mask boundaries](sam3_p2p_mask_grid.png)

Changing BundleSDF's input to the current SAM3 masks did **not** recover the dowel in a causal frames-100–300 rerun. Against the same saved scoring mask, its surface gap was 106.7 mm at frame 230, 62.7 mm at frame 280, and 71.5 mm at frame 300. The sealed-mask run scored 71.7, 60.1 and 65.1 mm at those frames. This isolates the dominant failure to pose recovery after missing correspondence, rather than the small difference between these two masks.

A simple depth-PCA axis override was also tested and rejected. It reduced an apparent 23° visible-axis disagreement by construction but worsened held-out insertion surface p90 from 8.6 to 16.0 mm. The raw depth PCA included end caps and partial occlusion, so it was a biased axis proxy; forcing a full pose to match it was not justified.

## What the visible shaft still measures

The 23° figure above was **not** a measured Point2Pose orientation error. I re-estimated the dowel's 3D long axis from aligned depth under both saved mask streams, excluding the two axial ends before the final PCA. During frames 400–500, the median unoriented angle to Point2Pose's shaft changed from 19.2° using all SAM3-masked depth points to **2.5°** using the central 60% of points. Central 80% and 50% selections gave 2.9° and 3.3°. The sealed mask gave the same pattern (22.8° raw, 2.0° central 60%). These trims are diagnostic choices, not calibrated pose uncertainty; their agreement identifies the end-cap/occlusion bias in the original full-mask PCA.

An independent 2D check projects each proposed reference shaft into the image and compares its direction with the SAM3 mask's long direction, modulo 180°. Across insertion, the medians were **3.3° for Point2Pose**, **10.4° for a rigid-grasp prediction**, and **87.4° for BundleSDF**. The dowel mask is elongated here (median covariance eigenvalue ratio 11.6), so its long direction is a meaningful image constraint; this test would not carry the same weight for a nearly round silhouette. The rigid-grasp prediction was reconstructed from the frame-130 object pose and the dataset's frame-228 gripper pose, then propagated by recorded end-effector poses. The dataset camera/motor streams were associated by their nominal 30 Hz frame indices; shifting this association by up to three frames changed the median gripper-versus-Point2Pose task-end difference by under 0.5 mm and shaft difference by under 0.5°.

![Shaft-axis agreement against the SAM3 mask](shaft_axis_agreement.png)

The insertion end is partly occluded at the ring, but the opposite end remains exposed. Using the visible SAM3-masked depth projected onto each pose's shaft, its 5th-percentile axial position differs from the corresponding first-view end by a median **0.7 mm for Point2Pose**, **17.2 mm for the rigid-grasp prediction**, and **78.6 mm for BundleSDF** during frames 400–500. This is a **one-dimensional consistency check**, not a 3D measurement of the hidden insertion tip. Inferring that tip still needs the learned end-to-end length, a centerline estimate, the chosen end identity and their uncertainty. The picture shows why keeping a pose state matters: the same learned dowel geometry projects to different insertion-end hypotheses under the two tracked attachments.

![Partly occluded insertion-end hypotheses at two demo frames](hidden_tip_pose_comparison.png)

These checks reuse the recording's RGB-D and masks, so they are correlated with the trackers' inputs. They distinguish a catastrophic pose branch and show which parts of the carried object remain observable; they do not replace independent 3D tip labels or a physical insertion test.

## Earlier views and contact-depth quality

I tested whether the missing backside of the frame-100 reference, rather than pose drift, inflated the insertion surface score. The [reproducible analysis](analyze_multiview.py) unprojects SAM3-masked depth at frames 100, 300, 340 and 380 into the frame-100 object coordinates using the **causal Point2Pose poses at those earlier frames**. Frame-100 points are retained. A new point from the other three frames is retained only when it lies within 3 mm of a point from another source view. This two-view agreement check matters: concatenating all masked depths without verification included contact/background points and expanded the dowel model to roughly 126 × 154 × 139 mm. The checked model's bounding box is 96 × 58 × 44 mm, close to the first view's 94 × 55 × 39 mm. The 3 mm support radius is an ablation setting, **not** an object property or a calibrated acceptance rule.

Only frames **400–500** are scored, after every model-building view. This is causally held out in time, but not independent of Point2Pose: that tracker supplied the source-view registrations. Every candidate is scored against the same model and current SAM3-masked metric depth. As in the first comparison, this is a visible-surface consistency proxy, not a hidden-tip or full-pose label.

![Causal multiview and depth-edge ablation](multiview_depth_ablation.png)

| Candidate, insertion frames            | Frame-100 model; all mask depth | Confirmed earlier-view model; all mask depth | Confirmed model; interior depth |
| -------------------------------------- | ------------------------------: | -------------------------------------------: | ------------------------------: |
| Dowel Point2Pose, 100 observed frames  |                          8.5 mm |                                       5.0 mm |                      **3.8 mm** |
| Dowel BundleSDF, 100 non-failed frames |                         68.0 mm |                                      68.0 mm |                     **66.5 mm** |
| Ring Point2Pose, 101 frames            |                          3.8 mm |                                       2.9 mm |                      **2.8 mm** |
| Ring BundleSDF, 101 frames             |                          2.4 mm |                                       1.8 mm |                      **1.7 mm** |

Values are medians of each frame's one-way 90th-percentile point-to-surface distance, in millimetres. BundleSDF fits the ring surface better in this replay; a source-name priority would miss that. The dowel's candidate from BundleSDF stays grossly inconsistent even with the richer map. The later-view map improves coverage, but it **does not** by itself correct a pose or prove that Point2Pose's hidden insertion end is accurate.

The sharp early-contact spikes were largely invalid depth at an RGB-mask/occluder boundary. A 3 × 3 neighborhood check rejects a pixel when local metric depth spans more than the stated threshold or includes missing depth. It changes Point2Pose's frame-414 dowel score from 31.0 mm with all depth to 5.9 mm with the confirmed map and a 10 mm local-span threshold; the confirmed map alone still scores 31.0 mm there. The 10 mm setting retains a median 87.9% of the dowel's and 98.7% of the ring's already mask-eroded depth pixels. Sensitivity on the same frames:

| Local depth-span limit | Dowel pixels retained | Dowel median surface p90 | Across-frame p90 of surface p90 |
| ---------------------- | --------------------: | -----------------------: | ------------------------------: |
| 5 mm                   |                 82.4% |                   3.7 mm |                          5.5 mm |
| 10 mm                  |                 87.9% |                   3.8 mm |                          5.6 mm |
| 20 mm                  |                 89.1% |                   3.9 mm |                          5.6 mm |

The smoother scores are not proof that the underlying pose improved: filtering changed the measurement, not the tracker. These settings have **not** been put into the live backend. A production depth-quality rule needs a sensor-noise basis, declared parameters, and tests that it does not discard essential thin contact surfaces. The current FLASH visible-surface check accepts every positive masked depth pixel, so this recording also shows why a single unqualified surface residual should not decide whether a partial pose is valid.

The first-view pose and its propagated object geometry remain necessary when the contact tip is hidden. The camera can correct the visible shaft and opposite end, but cannot directly observe the tip or assume the grasp did not slip. The next estimator should represent those constraints and their uncertainty separately; insertion should be judged at the predicted tip and shaft relative to the opening, ignoring physically irrelevant axial roll. Independent task-feature labels or a controlled physical trial are still required to certify the actual tip error.

## Reacquisition is evidence, not an instant full-pose reset

The fresh Point2Pose dowel run reports `lost` for frames **161–248** (88 frames) and again for **263–275** (13 frames). It resumes emitting a pose at frame 249. To see when that pose becomes useful, the [reacquisition analysis](analyze_reacquisition.py) compares its projected first-view surface with the aligned SAM3 mask/depth and with a rigid-grasp prediction from the recorded end-effector. The gripper prediction attaches the frame-130 object pose at the recorded frame-228 gripper pose; it is a prior, not an observed settled grasp.

![RGB masks and depth through reacquisition](reacquisition_depth.png)

|   Frame | Mask pixels with valid depth | Point2Pose surface p90 | Rigid-grasp surface p90 | Interpretation                                                                                                                                                         |
| ------: | ---------------------------: | ---------------------: | ----------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|     249 |                          64% |                 8.5 mm |                 24.6 mm | Partial depth favors Point2Pose, but the full pose remains incompletely observed.                                                                                      |
| 260–262 |                       48–40% |             100–110 mm |               92–104 mm | The mask still follows the RGB dowel, but much of its depth is missing and some remaining pixels lie on nearby gripper surfaces. Neither residual can validate a pose. |
|     276 |                          88% |                 3.8 mm |                 20.2 mm | Depth again supports the Point2Pose branch; even shifting camera/motor association by ±3 frames gives the gripper prior no better than 14.2 mm.                        |
|     280 |                          97% |                 4.5 mm |                 15.6 mm | The better supported branch persists.                                                                                                                                  |

The two candidate poses differ on the transformed first-view surface centre by 29.2 mm at frame 276. Applying the difference to the hidden contact end could substantially alter insertion; merely remembering the grasp-time attachment would not. But a mask alone does not supply metric depth, and one visible surface does not certify the hidden tip or axial end identity. The scene estimator needs to retain a pose **prediction** through the lost interval, then admit the observable components of the direct pose as depth and silhouette evidence return. It should record low depth coverage and disagreement instead of equating `observed` with fully verified.

I also tested a direct pose correction, rather than just a score change. The [partial-refit ablation](analyze_partial_refit.py) starts from Point2Pose on the 100 insertion frames with a pose, matches a confirmed earlier-view map to the **central 60%** of current depth points along its shaft, and withholds the two visible axial ends for validation. The trial uses 12 local ICP iterations, a 15 mm correspondence gate and the best 80% of gated matches; these are explicit experiment parameters, not deployed tuning. The middle-section residual improved in 59/100 frames, but the withheld ends improved in only **48/100**. The median paired change in their p90 residual was essentially zero (+0.007 mm), while the predicted insertion end moved a median 1.6 mm and as much as **13.0 mm**. The withheld ends are themselves sometimes occluded or depth-contaminated, so this is not a ground-truth tip-error test. It gives no basis to promote this naive full-pose refit over Point2Pose; a partial measurement update needs an observability model and stronger validation.

## Runtime and compatibility boundaries

The Point2Pose bridge reported roughly 60 ms median frame time in this recording. The compatible BundleSDF dowel pass took about 190 s for 451 frames, including online reconstruction and debug output; the ring pass took about 88 s. These are observed experiment times, **not** matched implementation speed benchmarks. To run the old BundleSDF source on this CUDA 12.8 / Python 3.11 / PCL 1.12 machine, the isolated checkout changed CUDA architecture and binding compatibility, used OpenCV's CPU brute-force L2 descriptor matcher instead of its unavailable CUDA module, removed an upstream `pdb.set_trace()`, and guarded percentile denoising on empty masks. None of those changes was made in FLASH or the GUI server.

[TrackEverything](https://github.com/ayushjain1144/trackeverything) now has released code and an API for supplied RGB-D/camera geometry, but it produces point trajectories rather than object poses. Its default 16-frame window sees future frames within each window; the [authors' FAQ](https://github.com/ayushjain1144/trackeverything/blob/main/docs/faq.md) describes shorter four-frame windows and an experimental growing-buffer mode. Its published checkpoint returned `GatedRepo` to this machine's authenticated Hugging Face account on 2026-10-21, so no TrackEverything accuracy or speed result is claimed here.

The CSV files in this directory hold the frame-level measurements, including [image-axis residuals](mask_projected_axis.csv), [depth-axis ablations](depth_axis_trim.csv), [visible-end axial residuals](observed_end_axial_residual.csv), the [multiview/depth-quality ablation](multiview_depth_ablation.csv), [reacquisition scores](reacquisition_score.csv) and [partial-refit scores](partial_refit_score.csv). Re-run the latter three scripts with `--dataset /home/feit/.cache/huggingface/lerobot/demos/insertion/recording --masks /home/feit/.cache/flash-tracker-compare/sam3_masks --output /tmp/flash-analysis`; the source dataset and generated SAM3 masks are local, not committed. [pose_traces.npz](pose_traces.npz) preserves the per-frame transforms, evidence flags, camera intrinsics, calibration transform, recorded gripper poses and first-view dowel features used for these projections. No tested tracker was checked against independent 6D pose labels, and no new physical insertion was attempted. The visible drift establishes a failure case; it does not establish that any remaining tracker is accurate enough for unattended insertion.
