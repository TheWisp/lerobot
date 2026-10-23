# LSFOdyssey labelled RGB-D clips

An exploratory second labelled source checks material-point identity on two 40-frame synthetic scene videos. The clips are `cab_e_3rd/000000` and `dancingroom3_3rd/000000` from [LSFOdyssey](https://huggingface.co/datasets/wwcreator/LSFOdyssey), revision `e8a226a40a33432dc901e995a93230e1e3cf4c89`. Each uses 128 first-frame visible points selected spatially from the supplied 2D/3D trajectories. Ground-truth coordinates and later visibility are never adapter inputs; no object mask is provided. The points span both moving people and static room surfaces. This is a frontend identity check, not a rigid-object pose fit.

The source's depth differs from its labelled point depth by about 33–35 mm median at visible pixels, so a 5 mm metric score would mostly test that disagreement. The table ranks 2D identity only. The video reports 24 fps, but no independent camera timestamps are supplied; adapter FPS below is isolated compute throughput. Full-clip models may see future frames and have no causal publication rate here. The original RGB frames are kept in the local cache rather than copied into this repository. Model-training overlap with these public scenes was not audited.

The high-motion column separately scores the quarter of initial queries with the greatest sum of visible consecutive-frame pixel travel within each clip. The ranking uses source labels only for evaluation, never as model input. It is image motion, not a rigid-object label. Missing predicted visible points remain in both coverage denominators.

| Clip               | Model              | Mode                  | 2 px coverage | High-motion 2 px | 4 px coverage | Initial px error | Adapter FPS |
| ------------------ | ------------------ | --------------------- | ------------: | ---------------: | ------------: | ---------------: | ----------: |
| cab-000000-bench   | bootstapir         | causal frame by frame |         0.422 |            0.470 |         0.690 |             1.26 |        26.9 |
| cab-000000-bench   | tapnext            | causal frame by frame |         0.702 |            0.710 |         0.848 |             0.73 |        54.6 |
| cab-000000-bench   | litetracker        | causal frame by frame |         0.521 |            0.555 |         0.735 |             0.00 |        63.8 |
| cab-000000-bench   | trackon_r          | causal frame by frame |         0.632 |            0.639 |         0.822 |             0.28 |        32.9 |
| cab-000000-bench   | lk                 | causal frame by frame |         0.538 |            0.328 |         0.687 |             0.00 |       187.4 |
| cab-000000-bench   | cotracker3         | full clip             |         0.542 |            0.591 |         0.725 |             0.00 |        72.6 |
| cab-000000-bench   | tapip3d_unfiltered | full clip             |         0.654 |            0.526 |         0.770 |             0.04 |        22.9 |
| cab-000000-bench   | trackeverything    | full clip             |         0.274 |            0.168 |         0.432 |             0.07 |         0.4 |
| cab-000000-bench   | spatracker2        | full clip             |         0.553 |            0.510 |         0.737 |             0.15 |         6.1 |
| cab-000000-bench   | point4d            | full clip             |         0.185 |            0.189 |         0.568 |             1.14 |        16.8 |
| dance-000000-bench | bootstapir         | causal frame by frame |         0.423 |            0.374 |         0.706 |             1.19 |        26.6 |
| dance-000000-bench | tapnext            | causal frame by frame |         0.736 |            0.576 |         0.839 |             0.74 |        53.3 |
| dance-000000-bench | litetracker        | causal frame by frame |         0.622 |            0.489 |         0.782 |             0.00 |        59.1 |
| dance-000000-bench | trackon_r          | causal frame by frame |         0.643 |            0.534 |         0.779 |             0.32 |        32.6 |
| dance-000000-bench | lk                 | causal frame by frame |         0.495 |            0.438 |         0.605 |             0.00 |       160.3 |
| dance-000000-bench | cotracker3         | full clip             |         0.662 |            0.546 |         0.790 |             0.00 |        71.3 |
| dance-000000-bench | tapip3d_unfiltered | full clip             |         0.699 |            0.594 |         0.820 |             0.04 |        21.7 |

These two scenes provide independent point labels but are synthetic, distant (metre-scale) and mostly textured. The YCBInEOAT moving holdout and local dowel/gamepad clips remain necessary for small, weak-texture manipulation surfaces. One source or one accuracy threshold cannot select the FLASH tracker.
