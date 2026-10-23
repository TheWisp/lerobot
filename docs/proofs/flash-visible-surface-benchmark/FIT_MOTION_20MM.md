# FLASH common rigid-fit pilot

same estimator, pose-derived reference, no robot validation or symmetry-aware task score

Every model's first-frame depth seeds and predicted 3D positions enter the same 3-point RANSAC/Kabsch fit with a 20 mm residual threshold. The 35-trial budget follows a 99% success target with a declared 50% inlier prior. The fit receives no evaluation pose or mask. After fitting, the initial point cloud is transported by the estimated transform and compared with the dataset's annotated object motion. The source frame is excluded; missing poses count as failures. The 5/10/20 mm columns are coverage over **all later frames**, not error only when a pose was emitted.

| Model              | Clips | Pose available | Cloud within 5 mm | Within 10 mm | Within 20 mm | Median fit time |
| ------------------ | ----: | -------------: | ----------------: | -----------: | -----------: | --------------: |
| bootstapir         |     3 |          1.000 |             0.433 |        0.689 |        0.911 |         1.93 ms |
| cotracker3         |     3 |          1.000 |             0.456 |        0.656 |        0.833 |         1.54 ms |
| litetracker        |     3 |          1.000 |             0.411 |        0.711 |        0.900 |         1.87 ms |
| lk                 |     3 |          1.000 |             0.422 |        0.667 |        0.878 |         1.65 ms |
| tapip3d            |     3 |          1.000 |             0.556 |        0.733 |        0.867 |         2.04 ms |
| tapip3d_unfiltered |     3 |          1.000 |             0.556 |        0.733 |        0.867 |         2.22 ms |
| tapnext            |     3 |          1.000 |             0.467 |        0.711 |        0.900 |         2.20 ms |
| trackeverything    |     3 |          1.000 |             0.411 |        0.700 |        0.844 |         1.97 ms |
| trackon_r          |     3 |          1.000 |             0.444 |        0.711 |        0.911 |         2.07 ms |

A fit can be available and still wrong: a coherent set of drifting or background points may satisfy the inlier threshold. This estimator adds about 2 ms of CPU work on these short clips; it is not an integrated segmentation-to-robot latency measurement. Native model 3D outputs and RGB tracks lifted through measured depth use the same fitter but have different depth-error mechanisms. The raw point report remains the primary visible-surface identity test.
