# FLASH common rigid-fit pilot

same estimator, pose-derived reference, no robot validation or symmetry-aware task score

Every model's first-frame depth seeds and predicted 3D positions enter the same 3-point RANSAC/Kabsch fit with a 10 mm residual threshold. The 35-trial budget follows a 99% success target with a declared 50% inlier prior. The fit receives no evaluation pose or mask. After fitting, the initial point cloud is transported by the estimated transform and compared with the dataset's annotated object motion. The source frame is excluded; missing poses count as failures. The 5/10/20 mm columns are coverage over **all later frames**, not error only when a pose was emitted.

| Model              | Clips | Pose available | Cloud within 5 mm | Within 10 mm | Within 20 mm | Median fit time |
| ------------------ | ----: | -------------: | ----------------: | -----------: | -----------: | --------------: |
| bootstapir         |     6 |          1.000 |             0.189 |        0.789 |        0.983 |         1.97 ms |
| cotracker3         |     6 |          1.000 |             0.200 |        0.833 |        0.994 |         1.87 ms |
| litetracker        |     6 |          1.000 |             0.144 |        0.822 |        0.989 |         1.61 ms |
| lk                 |     6 |          0.967 |             0.133 |        0.567 |        0.783 |         2.06 ms |
| tapip3d            |     6 |          1.000 |             0.283 |        0.783 |        1.000 |         1.97 ms |
| tapip3d_unfiltered |     6 |          1.000 |             0.289 |        0.772 |        1.000 |         1.96 ms |
| tapnext            |     6 |          1.000 |             0.239 |        0.856 |        0.989 |         2.02 ms |
| trackeverything    |     6 |          1.000 |             0.128 |        0.561 |        0.972 |         1.66 ms |
| trackon_r          |     6 |          1.000 |             0.200 |        0.833 |        0.989 |         1.82 ms |

A fit can be available and still wrong: a coherent set of drifting or background points may satisfy the inlier threshold. This estimator adds about 2 ms of CPU work on these short clips; it is not an integrated segmentation-to-robot latency measurement. Native model 3D outputs and RGB tracks lifted through measured depth use the same fitter but have different depth-error mechanisms. The raw point report remains the primary visible-surface identity test.
