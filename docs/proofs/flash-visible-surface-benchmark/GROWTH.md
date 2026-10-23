# FLASH growing-point pilot

development clips; labels are pose-derived with measured-depth uncertainty; no held-out test

The same initial queries run with and without 32 new points at each of frames 10 and 20. New points are chosen from the object's labelled mask and measured depth, at least 5 pixels from projected existing material points. These are supplied queries, not autonomous point discovery. Each cohort's 2-pixel score starts on the frame **after** birth, so a model does not earn credit merely for repeating its supplied birth coordinate. The original 64 points are scored separately to expose interference from adding points.

| Model              | Clips | Original 2 px fixed → grown | Frame 10 cohort | Frame 20 cohort | Model FPS fixed → grown | Adapter FPS fixed → grown |
| ------------------ | ----: | --------------------------: | --------------: | --------------: | ----------------------: | ------------------------: |
| bootstapir         |     6 |               0.250 → 0.250 |           0.252 |           0.434 |             43.6 → 31.8 |               38.9 → 29.2 |
| cotracker3         |     6 |               0.271 → 0.268 |           0.262 |           0.506 |           115.6 → 110.7 |               81.7 → 79.2 |
| litetracker        |     6 |               0.272 → 0.272 |           0.282 |           0.505 |             76.6 → 74.8 |               64.0 → 62.2 |
| lk                 |     6 |               0.223 → 0.223 |           0.201 |           0.463 |         3173.0 → 2432.7 |             333.1 → 256.7 |
| tapip3d_unfiltered |     6 |               0.324 → 0.324 |           0.416 |           0.569 |             36.2 → 22.0 |               30.8 → 20.0 |
| tapnext            |     6 |               0.303 → 0.296 |           0.288 |           0.509 |             66.5 → 65.1 |               56.6 → 55.1 |
| trackeverything    |     6 |               0.256 → 0.250 |           0.302 |           0.495 |               9.3 → 9.3 |                 9.0 → 9.0 |
| trackon_r          |     6 |               0.283 → 0.281 |           0.307 |           0.499 |             38.4 → 37.4 |               34.3 → 33.6 |

Full-clip CoTracker3, TrackEverything and TAPIP3D throughput is not causal publication rate. The other frontends run on every source frame. Later cohorts have shorter lifetimes and different visible surface, so their scores cannot be read as proof that growth repaired older tracks. The points were provided using ground-truth masks for this isolation test; a live discovery policy and segmentation overhead remain unmeasured.

| Causal model | Median per-frame tracker ms: fixed points, frames 1–9 / 11–19 / 21–end | Median per-frame tracker ms: growing points, same intervals |
| ------------ | ---------------------------------------------------------------------: | ----------------------------------------------------------: |
| bootstapir   |                                                  15.12 / 15.07 / 15.10 |                                       15.04 / 26.67 / 26.74 |
| litetracker  |                                                     5.25 / 5.13 / 5.26 |                                          5.40 / 5.36 / 5.22 |
| lk           |                                                     0.33 / 0.30 / 0.31 |                                          0.40 / 0.39 / 0.41 |
| tapnext      |                                                     8.19 / 8.17 / 8.18 |                                          8.39 / 8.32 / 8.42 |
| trackon_r    |                                                  17.89 / 17.88 / 17.87 |                                       17.86 / 18.44 / 18.90 |
