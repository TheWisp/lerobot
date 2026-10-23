# FLASH common rigid-fit pilot

same estimator, pose-derived reference, no robot validation or symmetry-aware task score

Every model's first-frame depth seeds and predicted 3D positions enter the same 3-point RANSAC/Kabsch fit with a 10 mm residual threshold. The 35-trial budget follows a 99% success target with a declared 50% inlier prior. The fit receives no evaluation pose or mask. After fitting, the initial point cloud is transported by the estimated transform and compared with the dataset's annotated object motion. The source frame is excluded; missing poses count as failures. The 5/10/20 mm columns are coverage over **all later frames**, not error only when a pose was emitted.

| Model              | Clips | Pose available | Cloud within 5 mm | Within 10 mm | Within 20 mm | Median fit time |
| ------------------ | ----: | -------------: | ----------------: | -----------: | -----------: | --------------: |
| bootstapir         |     3 |          1.000 |             0.800 |        1.000 |        1.000 |         1.83 ms |
| cotracker3         |     3 |          1.000 |             0.800 |        1.000 |        1.000 |         1.62 ms |
| litetracker        |     3 |          1.000 |             0.822 |        1.000 |        1.000 |         1.95 ms |
| lk                 |     3 |          1.000 |             0.822 |        1.000 |        1.000 |         1.67 ms |
| tapip3d            |     3 |          1.000 |             0.822 |        1.000 |        1.000 |         1.81 ms |
| tapip3d_unfiltered |     3 |          1.000 |             0.822 |        1.000 |        1.000 |         1.81 ms |
| tapnext            |     3 |          1.000 |             0.822 |        1.000 |        1.000 |         1.72 ms |
| trackeverything    |     3 |          1.000 |             0.856 |        1.000 |        1.000 |         1.88 ms |
| trackon_r          |     3 |          1.000 |             0.811 |        1.000 |        1.000 |         1.77 ms |

A fit can be available and still wrong: a coherent set of drifting or background points may satisfy the inlier threshold. This estimator adds about 2 ms of CPU work on these short clips; it is not an integrated segmentation-to-robot latency measurement. Native model 3D outputs and RGB tracks lifted through measured depth use the same fitter but have different depth-error mechanisms. The raw point report remains the primary visible-surface identity test.
