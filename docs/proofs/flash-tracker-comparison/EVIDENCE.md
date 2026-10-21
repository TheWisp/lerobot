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

A simple depth-PCA axis override was also tested and rejected. It reduced the apparent 23° visible-axis disagreement by construction but worsened held-out insertion surface p90 from 8.6 to 16.0 mm. The axis proxy is biased by partial visibility and the first-view surface; it cannot be promoted to an authoritative pose constraint without a joint, multi-view fit.

## Runtime and compatibility boundaries

The Point2Pose bridge reported roughly 60 ms median frame time in this recording. The compatible BundleSDF dowel pass took about 190 s for 451 frames, including online reconstruction and debug output; the ring pass took about 88 s. These are observed experiment times, **not** matched implementation speed benchmarks. To run the old BundleSDF source on this CUDA 12.8 / Python 3.11 / PCL 1.12 machine, the isolated checkout changed CUDA architecture and binding compatibility, used OpenCV's CPU brute-force L2 descriptor matcher instead of its unavailable CUDA module, removed an upstream `pdb.set_trace()`, and guarded percentile denoising on empty masks. None of those changes was made in FLASH or the GUI server.

[TrackEverything](https://github.com/ayushjain1144/trackeverything) now has released code and an API for supplied RGB-D/camera geometry, but it produces point trajectories rather than object poses. Its default 16-frame window sees future frames within each window; the [authors' FAQ](https://github.com/ayushjain1144/trackeverything/blob/main/docs/faq.md) describes shorter four-frame windows and an experimental growing-buffer mode. Its published checkpoint returned `GatedRepo` to this machine's authenticated Hugging Face account on 2026-10-21, so no TrackEverything accuracy or speed result is claimed here.

The CSV files in this directory hold the frame-level measurements. [pose_traces.npz](pose_traces.npz) preserves the per-frame transforms, evidence flags, camera intrinsics and calibration transform used for these projections. No tested tracker was checked against independent 6D pose labels, and no new physical insertion was attempted. The visible drift establishes a failure case; it does not establish that any remaining tracker is accurate enough for unattended insertion.
