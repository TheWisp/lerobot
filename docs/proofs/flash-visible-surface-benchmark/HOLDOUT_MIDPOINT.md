# FLASH visible-surface midpoint-holdout

frozen midpoint sequence holdout within YCBInEOAT; not an independent source

Coverage includes missing point predictions in its denominator. The public reference transports source-depth points through object-pose annotations; its own measured-depth disagreement is shown alongside scores. Every number below is developmental.

| Clip                                | Model              | 2 px coverage | 5 mm coverage | Median reference depth gap | Model FPS | Adapter FPS |
| ----------------------------------- | ------------------ | ------------: | ------------: | -------------------------: | --------: | ----------: |
| ycb-bleach_hard_00_03_chaitanya-205 | bootstapir         |         0.932 |         0.835 |                     1.0 mm |      43.5 |        39.1 |
| ycb-bleach_hard_00_03_chaitanya-205 | cotracker3         |         0.945 |         0.865 |                     1.0 mm |     115.4 |        82.2 |
| ycb-bleach_hard_00_03_chaitanya-205 | litetracker        |         0.952 |         0.875 |                     1.0 mm |      76.1 |        63.6 |
| ycb-bleach_hard_00_03_chaitanya-205 | lk                 |         0.956 |         0.877 |                     1.0 mm |    2799.9 |       251.3 |
| ycb-bleach_hard_00_03_chaitanya-205 | tapip3d            |         0.960 |         0.996 |                     1.0 mm |      34.8 |         4.4 |
| ycb-bleach_hard_00_03_chaitanya-205 | tapip3d_unfiltered |         0.956 |         0.996 |                     1.0 mm |      34.6 |        30.0 |
| ycb-bleach_hard_00_03_chaitanya-205 | tapnext            |         0.930 |         0.865 |                     1.0 mm |      65.8 |        56.3 |
| ycb-bleach_hard_00_03_chaitanya-205 | trackeverything    |         0.927 |         0.994 |                     1.0 mm |      11.9 |        11.5 |
| ycb-bleach_hard_00_03_chaitanya-205 | trackon_r          |         0.964 |         0.880 |                     1.0 mm |      38.1 |        34.2 |
| ycb-cracker_box_yalehand0-648       | bootstapir         |         0.315 |         0.283 |                     2.5 mm |      43.2 |        38.9 |
| ycb-cracker_box_yalehand0-648       | cotracker3         |         0.305 |         0.405 |                     2.5 mm |     115.7 |        82.7 |
| ycb-cracker_box_yalehand0-648       | litetracker        |         0.341 |         0.425 |                     2.5 mm |      76.2 |        62.9 |
| ycb-cracker_box_yalehand0-648       | lk                 |         0.341 |         0.419 |                     2.5 mm |    3126.0 |       252.1 |
| ycb-cracker_box_yalehand0-648       | tapip3d            |         0.433 |         0.697 |                     2.5 mm |      38.4 |         4.5 |
| ycb-cracker_box_yalehand0-648       | tapip3d_unfiltered |         0.433 |         0.695 |                     2.5 mm |      38.1 |        32.4 |
| ycb-cracker_box_yalehand0-648       | tapnext            |         0.377 |         0.413 |                     2.5 mm |      65.7 |        56.2 |
| ycb-cracker_box_yalehand0-648       | trackeverything    |         0.415 |         0.623 |                     2.5 mm |      11.9 |        11.5 |
| ycb-cracker_box_yalehand0-648       | trackon_r          |         0.351 |         0.423 |                     2.5 mm |      38.3 |        34.4 |
| ycb-sugar_box_yalehand0-486         | bootstapir         |         0.942 |         0.930 |                     0.8 mm |      43.4 |        38.8 |
| ycb-sugar_box_yalehand0-486         | cotracker3         |         1.000 |         0.966 |                     0.8 mm |     115.4 |        81.3 |
| ycb-sugar_box_yalehand0-486         | litetracker        |         0.997 |         0.930 |                     0.8 mm |      76.9 |        63.4 |
| ycb-sugar_box_yalehand0-486         | lk                 |         1.000 |         0.966 |                     0.8 mm |    3281.7 |       244.1 |
| ycb-sugar_box_yalehand0-486         | tapip3d            |         1.000 |         1.000 |                     0.8 mm |      34.2 |         4.4 |
| ycb-sugar_box_yalehand0-486         | tapip3d_unfiltered |         1.000 |         1.000 |                     0.8 mm |      33.9 |        29.3 |
| ycb-sugar_box_yalehand0-486         | tapnext            |         1.000 |         0.950 |                     0.8 mm |      65.8 |        55.7 |
| ycb-sugar_box_yalehand0-486         | trackeverything    |         1.000 |         0.994 |                     0.8 mm |      12.2 |        11.7 |
| ycb-sugar_box_yalehand0-486         | trackon_r          |         1.000 |         0.961 |                     0.8 mm |      38.2 |        33.9 |

Model FPS is full-clip throughput for CoTracker3, TrackEverything and TAPIP3D, and causal per-frame tracker throughput for the other adapters. Adapter FPS includes recorded RGB-D decoding and each adapter's preprocessing. TAPIP3D's upstream depth filter dominates its wall time. Neither includes shared segmentation, pose fitting, FLASH planning, motor IO, recording, or a 30 Hz consumer. They are not measured live-loop FPS.

| Model              | Clips | Mean 2 px coverage | Mean 5 mm coverage | Median model FPS | Median adapter FPS | Peak GPU memory |
| ------------------ | ----: | -----------------: | -----------------: | ---------------: | -----------------: | --------------: |
| bootstapir         |     3 |              0.730 |              0.683 |             43.4 |               38.9 |        0.55 GiB |
| cotracker3         |     3 |              0.750 |              0.745 |            115.4 |               82.2 |        4.48 GiB |
| litetracker        |     3 |              0.763 |              0.743 |             76.2 |               63.4 |        0.25 GiB |
| lk                 |     3 |              0.765 |              0.754 |           3126.0 |              251.3 |        0.00 GiB |
| tapip3d            |     3 |              0.798 |              0.898 |             34.8 |                4.4 |        3.20 GiB |
| tapip3d_unfiltered |     3 |              0.796 |              0.897 |             34.6 |               30.0 |        3.20 GiB |
| tapnext            |     3 |              0.769 |              0.743 |             65.8 |               56.2 |        1.62 GiB |
| trackeverything    |     3 |              0.781 |              0.870 |             11.9 |               11.5 |        2.54 GiB |
| trackon_r          |     3 |              0.772 |              0.755 |             38.2 |               34.2 |        0.30 GiB |
