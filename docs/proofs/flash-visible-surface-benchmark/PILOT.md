# FLASH visible-surface pilot

development pilot; clips are not a held-out or independent-source result

Coverage includes missing point predictions in its denominator. The reference transports source-depth points through object-pose annotations; its own measured-depth disagreement is shown alongside scores. Every number below is a development result, not a tracker selection.

| Clip                              | Model           | 2 px coverage | 5 mm coverage | Median reference depth gap | Model FPS | Full decode FPS |
| --------------------------------- | --------------- | ------------: | ------------: | -------------------------: | --------: | --------------: |
| ycb-bleach0-444                   | bootstapir      |         0.136 |         0.097 |                     3.9 mm |      43.6 |            39.0 |
| ycb-bleach0-444                   | litetracker     |         0.200 |         0.179 |                     3.9 mm |      76.4 |            63.2 |
| ycb-bleach0-444                   | lk              |         0.123 |         0.120 |                     3.9 mm |    2141.5 |           225.7 |
| ycb-bleach0-444                   | tapnext         |         0.192 |         0.179 |                     3.9 mm |      66.5 |            56.2 |
| ycb-bleach0-444                   | trackeverything |         0.117 |         0.093 |                     3.9 mm |       7.9 |             7.7 |
| ycb-cracker_box_reorient-191      | bootstapir      |         0.193 |         0.210 |                     2.9 mm |      42.7 |            38.0 |
| ycb-cracker_box_reorient-191      | litetracker     |         0.230 |         0.256 |                     2.9 mm |      75.9 |            61.8 |
| ycb-cracker_box_reorient-191      | lk              |         0.224 |         0.257 |                     2.9 mm |    2974.5 |           300.0 |
| ycb-cracker_box_reorient-191      | tapnext         |         0.232 |         0.279 |                     2.9 mm |      65.4 |            55.0 |
| ycb-cracker_box_reorient-191      | trackeverything |         0.231 |         0.190 |                     2.9 mm |       7.7 |             7.5 |
| ycb-mustard-100-130               | bootstapir      |         0.457 |         0.427 |                     2.0 mm |      42.7 |            38.6 |
| ycb-mustard-100-130               | litetracker     |         0.494 |         0.469 |                     2.0 mm |      76.3 |            64.2 |
| ycb-mustard-100-130               | lk              |         0.347 |         0.309 |                     2.0 mm |    3132.6 |           359.4 |
| ycb-mustard-100-130               | tapnext         |         0.650 |         0.636 |                     2.0 mm |      66.4 |            56.9 |
| ycb-mustard-100-130               | trackeverything |         0.438 |         0.468 |                     2.0 mm |      11.7 |            11.3 |
| ycb-mustard_easy_00_02-127        | bootstapir      |         0.275 |         0.203 |                     3.6 mm |      43.5 |            39.3 |
| ycb-mustard_easy_00_02-127        | litetracker     |         0.272 |         0.233 |                     3.6 mm |      78.2 |            65.3 |
| ycb-mustard_easy_00_02-127        | lk              |         0.295 |         0.245 |                     3.6 mm |    3224.6 |           352.1 |
| ycb-mustard_easy_00_02-127        | tapnext         |         0.330 |         0.264 |                     3.6 mm |      66.8 |            57.4 |
| ycb-mustard_easy_00_02-127        | trackeverything |         0.335 |         0.188 |                     3.6 mm |      11.0 |            10.7 |
| ycb-sugar_box1-221                | bootstapir      |         0.170 |         0.183 |                     2.6 mm |      43.7 |            38.9 |
| ycb-sugar_box1-221                | litetracker     |         0.199 |         0.232 |                     2.6 mm |      78.0 |            64.0 |
| ycb-sugar_box1-221                | lk              |         0.070 |         0.069 |                     2.6 mm |    3213.4 |           316.2 |
| ycb-sugar_box1-221                | tapnext         |         0.134 |         0.170 |                     2.6 mm |      66.9 |            56.3 |
| ycb-sugar_box1-221                | trackeverything |         0.123 |         0.115 |                     2.6 mm |       8.4 |             8.2 |
| ycb-tomato_soup_can_yalehand0-654 | bootstapir      |         0.271 |         0.206 |                     2.9 mm |      43.9 |            39.6 |
| ycb-tomato_soup_can_yalehand0-654 | litetracker     |         0.235 |         0.215 |                     2.9 mm |      76.7 |            64.1 |
| ycb-tomato_soup_can_yalehand0-654 | lk              |         0.277 |         0.230 |                     2.9 mm |    3242.2 |           349.9 |
| ycb-tomato_soup_can_yalehand0-654 | tapnext         |         0.280 |         0.235 |                     2.9 mm |      66.3 |            56.9 |
| ycb-tomato_soup_can_yalehand0-654 | trackeverything |         0.294 |         0.250 |                     2.9 mm |      10.2 |             9.9 |

Model FPS is full-clip throughput for TrackEverything and causal per-frame tracker throughput for the other adapters. Full decode FPS includes recorded RGB-D loading; neither includes segmentation, pose fitting, network transfer, recording, or a 30 Hz consumer. They are not measured live-loop FPS.

| Model           | Clips | Mean 2 px coverage | Mean 5 mm coverage | Median model FPS | Peak GPU memory |
| --------------- | ----: | -----------------: | -----------------: | ---------------: | --------------: |
| bootstapir      |     6 |              0.250 |              0.221 |             43.6 |        0.55 GiB |
| litetracker     |     6 |              0.272 |              0.264 |             76.6 |        0.25 GiB |
| lk              |     6 |              0.223 |              0.205 |           3173.0 |        0.00 GiB |
| tapnext         |     6 |              0.303 |              0.294 |             66.5 |        1.62 GiB |
| trackeverything |     6 |              0.256 |              0.217 |              9.3 |        3.49 GiB |
