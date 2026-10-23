# Nested point-budget sweep

One labelled 31-frame moving sugar-box clip, with 64, 256 and 1,024 queries selected as nested prefixes of a farthest-point ordering. The same initial 64 material points are scored in every row; the all-point score is secondary because adding queries changes which surfaces enter the denominator. These are isolated, causal all-frame runs, with no SAM, shared fit, robot IO, recording or 30 Hz scheduler. Model FPS excludes RGB-D decoding and depth lifting; adapter FPS includes them. GPU peak is allocated memory, not process-resident memory.

| Model       | Queries | First 64: 2 px | First 64: 5 mm | All: 2 px | Model FPS | Adapter FPS | Peak GPU GiB |
| ----------- | ------: | -------------: | -------------: | --------: | --------: | ----------: | -----------: |
| tapnext     |      64 |          0.975 |          0.894 |     0.975 |      65.2 |        56.0 |         1.62 |
| tapnext     |     256 |          0.996 |          0.908 |     0.995 |      61.8 |        53.4 |         1.66 |
| tapnext     |    1024 |          0.999 |          0.910 |     0.999 |      50.9 |        44.9 |         1.87 |
| litetracker |      64 |          0.886 |          0.792 |     0.886 |      77.0 |        64.3 |         0.25 |
| litetracker |     256 |          0.894 |          0.798 |     0.933 |      74.2 |        62.0 |         0.31 |
| litetracker |    1024 |          0.894 |          0.799 |     0.877 |      60.8 |        52.5 |         0.67 |
| trackon_r   |      64 |          0.947 |          0.874 |     0.947 |      38.3 |        34.4 |         0.30 |
| trackon_r   |     256 |          0.949 |          0.870 |     0.954 |      34.2 |        31.1 |         0.45 |
| trackon_r   |    1024 |          0.949 |          0.871 |     0.944 |      21.8 |        20.4 |         1.10 |

The sweep isolates query-count scaling on one fairly visible object. It does not establish quality on the low-texture dowel or determine a final point budget. If first-64 accuracy changes, that is a real interaction with added queries, while changes in the all-point column can also reflect the newly sampled surface locations. Three models are shown because they provide causal per-frame updates; full-clip models need a separate offline budget sweep.
