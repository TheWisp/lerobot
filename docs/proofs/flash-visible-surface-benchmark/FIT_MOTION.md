# FLASH common rigid-fit pilot

same estimator, pose-derived reference, no robot validation or symmetry-aware task score

Every model's first-frame depth seeds and predicted 3D positions enter the same 3-point RANSAC/Kabsch fit with a 10 mm residual threshold. The 35-trial budget follows a 99% success target with a declared 50% inlier prior. The fit receives no evaluation pose or mask. After fitting, the initial point cloud is transported by the estimated transform and compared with the dataset's annotated object motion. The source frame is excluded; missing poses count as failures. The 5/10/20 mm columns are coverage over **all later frames**, not error only when a pose was emitted.

| Model              | Clips | Pose available | Cloud within 5 mm | Within 10 mm | Within 20 mm | Median fit time |
| ------------------ | ----: | -------------: | ----------------: | -----------: | -----------: | --------------: |
| bootstapir         |     3 |          1.000 |             0.444 |        0.700 |        0.822 |         2.00 ms |
| cotracker3         |     3 |          1.000 |             0.444 |        0.689 |        0.856 |         1.89 ms |
| litetracker        |     3 |          1.000 |             0.433 |        0.711 |        0.856 |         1.86 ms |
| lk                 |     3 |          1.000 |             0.444 |        0.656 |        0.800 |         1.91 ms |
| spatracker2        |     3 |          0.811 |             0.067 |        0.344 |        0.622 |         1.74 ms |
| spatracker2_sensor |     3 |          1.000 |             0.489 |        0.700 |        0.844 |         1.59 ms |
| tapip3d            |     3 |          1.000 |             0.544 |        0.711 |        0.867 |         1.87 ms |
| tapip3d_unfiltered |     3 |          1.000 |             0.556 |        0.711 |        0.867 |         2.06 ms |
| tapnext            |     3 |          1.000 |             0.478 |        0.711 |        0.844 |         2.01 ms |
| trackeverything    |     3 |          1.000 |             0.400 |        0.689 |        0.844 |         1.69 ms |
| trackon_r          |     3 |          1.000 |             0.456 |        0.711 |        0.844 |         1.87 ms |

A fit can be available and still wrong: a coherent set of drifting or background points may satisfy the inlier threshold. This estimator adds about 2 ms of CPU work on these short clips; it is not an integrated segmentation-to-robot latency measurement. Native model 3D outputs and RGB tracks lifted through measured depth use the same fitter but have different depth-error mechanisms. The raw point report remains the primary visible-surface identity test.
