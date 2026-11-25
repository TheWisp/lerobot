# FLASH frame and pose contracts — evidence

Rendered on 2026-11-25 from the two recorded fixtures that the frame and pose
contracts PR ships, using that PR's own functions. Neither fixture stores a
colour image, so the pictures show depth. These record one run at one
revision; they are not design documents.

| File                             | What it shows                                                                                                           | Inputs                                                                                                                                                                    |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01-frame-and-depth-support.png` | One recorded RGB-D frame's depth image with the dowel's segmentation mask, and which mask pixels have depth support     | `dowel_held_frame_315.npz`; `depth_support_mask(radius_px=1)`, the radius the point trackers use by default                                                               |
| `02-pose-disagreement.png`       | Two pose proposals for the dowel in the same frame, each pixel coloured the way `surface_support` classifies it         | `dowel_held_frame_315.npz`; `compare_pose_proposals` with the tolerances of `test_recorded_held_dowel_proposals_conflict_across_depth_and_mask_support`                   |
| `03-ring-registration.png`       | Matched 3D point pairs on a ring that stood still, split into the pairs one rigid motion explains and the rejected ones | `ring_static_frame_100_101.npz`; `register_object(threshold_m=0.005, min_inliers=12, trials=100)`, as in `test_recorded_static_ring_does_not_disappear_at_refit_boundary` |

Measured on these inputs: frame 315's mask has 2,497 pixels, 2,215 with a
depth reading and 2,033 with depth support. The two dowel poses are 36 mm and
82° apart; the matched-patch pose fits depth with a 4.9 mm 90th-percentile
difference and lands 29% of its pixels on the mask, the point-group pose
149.2 mm and 36%, and the verdict is `ambiguous`. On the ring, 323 of 375
pairs are kept with a 1.8 mm median distance, and the fit moves the ring
0.33 mm.
