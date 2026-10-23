# FLASH second-source RGB-D stress clip

second public RGB-D source; visual and throughput evidence only, no material-point labels

A manually supplied first-frame polygon seeds the same measured-depth points for each adapter. There are no later object masks or independent point identities. The video can expose obvious point bleeding or loss, but it cannot establish material-point accuracy. The source has no timestamps; 30 fps is a declared playback rate, not a measured capture rate.

| Clip     | Model              | Mode                                                                                  | Model FPS | Decode + adapter FPS | Peak GPU memory |
| -------- | ------------------ | ------------------------------------------------------------------------------------- | --------: | -------------------: | --------------: |
| mug-0-71 | bootstapir         | causal per-frame upstream Point2Pose frontend, no pose estimator                      |      53.2 |                 46.6 |        0.55 GiB |
| mug-0-71 | cotracker3         | full-clip offline; may use future frames                                              |     198.4 |                116.6 |       10.64 GiB |
| mug-0-71 | litetracker        | causal per-frame upstream Point2Pose frontend, no pose estimator                      |     115.9 |                 87.8 |        0.25 GiB |
| mug-0-71 | lk                 | causal per-frame optical flow; no object mask after seeding                           |    2636.2 |                282.1 |        0.00 GiB |
| mug-0-71 | point4d            | RGB full-clip model with predicted camera geometry; in-view, not occlusion visibility |      18.7 |                 18.4 |       22.58 GiB |
| mug-0-71 | spatracker2        | full-clip windowed RGB-D model; no causal publication rate measured                   |       8.1 |                  7.9 |       12.43 GiB |
| mug-0-71 | tapip3d_unfiltered | full-clip model; no causal publication rate measured                                  |      33.6 |                 29.0 |        3.95 GiB |
| mug-0-71 | tapnext            | causal per-frame upstream Point2Pose frontend, no pose estimator                      |      89.5 |                 72.4 |        1.62 GiB |
| mug-0-71 | trackeverything    | full-clip, windowed, may use future frames                                            |       2.6 |                  2.6 |        5.72 GiB |
| mug-0-71 | trackon_r          | causal streaming, all source frames                                                   |      46.5 |                 40.6 |        0.30 GiB |

Full-clip methods may use future frames and cannot be assigned a causal update rate from these numbers. Adapter FPS excludes segmentation, common rigid fitting, FLASH planning and robot IO. All GPU measurements used the RTX 5090. The raw per-frame predictions and stage timings remain in the local experiment cache; the video shows every source frame.
