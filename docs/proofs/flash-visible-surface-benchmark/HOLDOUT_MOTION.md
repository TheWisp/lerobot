# FLASH visible-surface motion-holdout

frozen visible-motion sequence holdout within YCBInEOAT; not an independent source

Coverage includes missing point predictions in its denominator. The public reference transports source-depth points through object-pose annotations; its own measured-depth disagreement is shown alongside scores. Every number below is developmental.

| Clip                               | Model              | 2 px coverage | 5 mm coverage | Median reference depth gap | Model FPS | Adapter FPS |
| ---------------------------------- | ------------------ | ------------: | ------------: | -------------------------: | --------: | ----------: |
| ycb-bleach_hard_00_03_chaitanya-38 | bootstapir         |         0.193 |         0.194 |                     2.8 mm |      43.1 |        38.9 |
| ycb-bleach_hard_00_03_chaitanya-38 | cotracker3         |         0.247 |         0.247 |                     2.8 mm |     115.4 |        83.2 |
| ycb-bleach_hard_00_03_chaitanya-38 | litetracker        |         0.244 |         0.257 |                     2.8 mm |      77.4 |        64.7 |
| ycb-bleach_hard_00_03_chaitanya-38 | lk                 |         0.195 |         0.176 |                     2.8 mm |    2327.5 |       278.2 |
| ycb-bleach_hard_00_03_chaitanya-38 | point4d            |         0.059 |           n/a |                     2.8 mm |      20.2 |        19.8 |
| ycb-bleach_hard_00_03_chaitanya-38 | spatracker2        |         0.247 |         0.042 |                     2.8 mm |       6.1 |         6.0 |
| ycb-bleach_hard_00_03_chaitanya-38 | spatracker2_sensor |         0.247 |         0.246 |                     2.8 mm |       6.1 |         6.0 |
| ycb-bleach_hard_00_03_chaitanya-38 | tapip3d            |         0.360 |         0.430 |                     2.8 mm |      36.9 |         4.4 |
| ycb-bleach_hard_00_03_chaitanya-38 | tapip3d_unfiltered |         0.360 |         0.434 |                     2.8 mm |      36.5 |        31.5 |
| ycb-bleach_hard_00_03_chaitanya-38 | tapnext            |         0.302 |         0.320 |                     2.8 mm |      62.4 |        54.1 |
| ycb-bleach_hard_00_03_chaitanya-38 | trackeverything    |         0.248 |         0.252 |                     2.8 mm |      10.8 |        10.5 |
| ycb-bleach_hard_00_03_chaitanya-38 | trackon_r          |         0.232 |         0.237 |                     2.8 mm |      38.4 |        34.5 |
| ycb-cracker_box_yalehand0-1253     | bootstapir         |         0.209 |         0.097 |                     3.3 mm |      43.3 |        38.9 |
| ycb-cracker_box_yalehand0-1253     | cotracker3         |         0.207 |         0.184 |                     3.3 mm |     115.2 |        82.3 |
| ycb-cracker_box_yalehand0-1253     | litetracker        |         0.200 |         0.183 |                     3.3 mm |      75.6 |        63.2 |
| ycb-cracker_box_yalehand0-1253     | lk                 |         0.209 |         0.183 |                     3.3 mm |    1512.6 |       252.3 |
| ycb-cracker_box_yalehand0-1253     | point4d            |         0.070 |           n/a |                     3.3 mm |      20.5 |        20.1 |
| ycb-cracker_box_yalehand0-1253     | spatracker2        |         0.223 |         0.021 |                     3.3 mm |       5.9 |         5.8 |
| ycb-cracker_box_yalehand0-1253     | spatracker2_sensor |         0.223 |         0.184 |                     3.3 mm |       5.9 |         5.8 |
| ycb-cracker_box_yalehand0-1253     | tapip3d            |         0.311 |         0.270 |                     3.3 mm |      35.2 |         4.4 |
| ycb-cracker_box_yalehand0-1253     | tapip3d_unfiltered |         0.317 |         0.270 |                     3.3 mm |      35.3 |        30.5 |
| ycb-cracker_box_yalehand0-1253     | tapnext            |         0.297 |         0.184 |                     3.3 mm |      65.0 |        55.4 |
| ycb-cracker_box_yalehand0-1253     | trackeverything    |         0.181 |         0.137 |                     3.3 mm |       8.4 |         8.2 |
| ycb-cracker_box_yalehand0-1253     | trackon_r          |         0.278 |         0.209 |                     3.3 mm |      38.2 |        32.9 |
| ycb-sugar_box_yalehand0-619        | bootstapir         |         0.710 |         0.607 |                     1.2 mm |      42.3 |        38.2 |
| ycb-sugar_box_yalehand0-619        | cotracker3         |         0.694 |         0.598 |                     1.2 mm |     114.4 |        82.1 |
| ycb-sugar_box_yalehand0-619        | litetracker        |         0.783 |         0.658 |                     1.2 mm |      69.8 |        59.1 |
| ycb-sugar_box_yalehand0-619        | lk                 |         0.923 |         0.821 |                     1.2 mm |    1840.3 |       254.4 |
| ycb-sugar_box_yalehand0-619        | point4d            |         0.182 |           n/a |                     1.2 mm |      20.5 |        20.1 |
| ycb-sugar_box_yalehand0-619        | spatracker2        |         0.797 |         0.099 |                     1.2 mm |       6.1 |         6.0 |
| ycb-sugar_box_yalehand0-619        | spatracker2_sensor |         0.797 |         0.707 |                     1.2 mm |       6.1 |         6.0 |
| ycb-sugar_box_yalehand0-619        | tapip3d            |         0.901 |         0.987 |                     1.2 mm |      36.2 |         4.4 |
| ycb-sugar_box_yalehand0-619        | tapip3d_unfiltered |         0.901 |         0.986 |                     1.2 mm |      36.0 |        31.2 |
| ycb-sugar_box_yalehand0-619        | tapnext            |         0.968 |         0.856 |                     1.2 mm |      66.5 |        57.0 |
| ycb-sugar_box_yalehand0-619        | trackeverything    |         0.803 |         0.712 |                     1.2 mm |      10.0 |         9.7 |
| ycb-sugar_box_yalehand0-619        | trackon_r          |         0.857 |         0.739 |                     1.2 mm |      38.4 |        34.5 |

The Mode field in the JSON distinguishes full-clip throughput from causal per-frame throughput; their FPS values cannot be treated as equivalent publication rates. Adapter FPS includes recorded RGB-D decoding and each adapter's preprocessing. A 3D coverage of n/a means that the model predicts unscaled geometry or that the adapter does not publish 3D points. TAPIP3D's upstream depth filter dominates its wall time. Neither includes shared segmentation, pose fitting, FLASH planning, motor IO, recording, or a 30 Hz consumer. They are not measured live-loop FPS.

| Model              | Clips | Mean 2 px coverage | Mean 5 mm coverage | Median model FPS | Median adapter FPS | Peak GPU memory |
| ------------------ | ----: | -----------------: | -----------------: | ---------------: | -----------------: | --------------: |
| bootstapir         |     3 |              0.370 |              0.299 |             43.1 |               38.9 |        0.55 GiB |
| cotracker3         |     3 |              0.383 |              0.343 |            115.2 |               82.3 |        4.48 GiB |
| litetracker        |     3 |              0.409 |              0.366 |             75.6 |               63.2 |        0.25 GiB |
| lk                 |     3 |              0.442 |              0.393 |           1840.3 |              254.4 |        0.00 GiB |
| point4d            |     3 |              0.104 |                n/a |             20.5 |               20.1 |       13.49 GiB |
| spatracker2        |     3 |              0.422 |              0.054 |              6.1 |                6.0 |        5.67 GiB |
| spatracker2_sensor |     3 |              0.422 |              0.379 |              6.1 |                6.0 |        5.67 GiB |
| tapip3d            |     3 |              0.524 |              0.562 |             36.2 |                4.4 |        3.19 GiB |
| tapip3d_unfiltered |     3 |              0.526 |              0.563 |             36.0 |               31.2 |        3.19 GiB |
| tapnext            |     3 |              0.522 |              0.454 |             65.0 |               55.4 |        1.62 GiB |
| trackeverything    |     3 |              0.411 |              0.367 |             10.0 |                9.7 |        4.12 GiB |
| trackon_r          |     3 |              0.456 |              0.395 |             38.4 |               34.5 |        0.30 GiB |
