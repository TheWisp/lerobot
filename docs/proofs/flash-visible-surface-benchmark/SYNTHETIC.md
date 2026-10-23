# FLASH exact-motion controls

exact-motion synthetic controls; these do not measure real-sensor generalization

Coverage includes missing point predictions in its denominator. These procedural rigid-plane clips provide exact object motion, with first-frame depth quantized to millimetres. They validate conventions and expose obvious drift, not real-scene tracking quality.

| Clip                | Model              | 2 px coverage | 5 mm coverage | Median reference depth gap | Model FPS | Adapter FPS |
| ------------------- | ------------------ | ------------: | ------------: | -------------------------: | --------: | ----------: |
| plane-tilt-smooth   | bootstapir         |         0.846 |         0.851 |                     0.3 mm |      43.3 |        41.8 |
| plane-tilt-smooth   | litetracker        |         0.789 |         0.834 |                     0.3 mm |      77.4 |        72.8 |
| plane-tilt-smooth   | lk                 |         0.378 |         0.379 |                     0.3 mm |    1143.9 |       548.4 |
| plane-tilt-smooth   | tapip3d_unfiltered |         0.713 |         0.753 |                     0.3 mm |      27.1 |        25.4 |
| plane-tilt-smooth   | tapnext            |         0.699 |         0.751 |                     0.3 mm |      66.2 |        62.9 |
| plane-tilt-smooth   | trackeverything    |         0.790 |         0.628 |                     0.3 mm |       9.3 |         9.2 |
| plane-tilt-smooth   | trackon_r          |         0.819 |         0.820 |                     0.3 mm |      38.5 |        37.1 |
| plane-tilt-textured | bootstapir         |         0.978 |         0.938 |                     0.3 mm |      43.0 |        41.5 |
| plane-tilt-textured | litetracker        |         0.998 |         0.930 |                     0.3 mm |      78.0 |        73.2 |
| plane-tilt-textured | lk                 |         0.996 |         0.929 |                     0.3 mm |    1102.8 |       534.2 |
| plane-tilt-textured | tapip3d_unfiltered |         0.999 |         0.999 |                     0.3 mm |      27.2 |        25.5 |
| plane-tilt-textured | tapnext            |         0.995 |         0.972 |                     0.3 mm |      66.5 |        63.1 |
| plane-tilt-textured | trackeverything    |         0.862 |         0.696 |                     0.3 mm |       9.2 |         9.1 |
| plane-tilt-textured | trackon_r          |         1.000 |         0.994 |                     0.3 mm |      38.6 |        37.2 |
| plane-translation   | bootstapir         |         0.980 |         0.956 |                     0.0 mm |      43.9 |        42.4 |
| plane-translation   | litetracker        |         1.000 |         0.990 |                     0.0 mm |      77.3 |        72.3 |
| plane-translation   | lk                 |         0.993 |         0.950 |                     0.0 mm |    1121.6 |       553.0 |
| plane-translation   | tapip3d_unfiltered |         1.000 |         1.000 |                     0.0 mm |      26.9 |        25.3 |
| plane-translation   | tapnext            |         1.000 |         0.995 |                     0.0 mm |      65.8 |        62.5 |
| plane-translation   | trackeverything    |         0.971 |         0.951 |                     0.0 mm |       9.2 |         9.1 |
| plane-translation   | trackon_r          |         1.000 |         0.999 |                     0.0 mm |      38.4 |        37.0 |

Model FPS is full-clip throughput for TrackEverything and TAPIP3D, and causal per-frame tracker throughput for the other adapters. Adapter FPS includes recorded RGB-D decoding and each adapter's preprocessing. Neither includes shared segmentation, pose fitting, FLASH planning, motor IO, recording, or a 30 Hz consumer. They are not measured live-loop FPS.

| Model              | Clips | Mean 2 px coverage | Mean 5 mm coverage | Median model FPS | Median adapter FPS | Peak GPU memory |
| ------------------ | ----: | -----------------: | -----------------: | ---------------: | -----------------: | --------------: |
| bootstapir         |     3 |              0.935 |              0.915 |             43.3 |               41.8 |        0.55 GiB |
| litetracker        |     3 |              0.929 |              0.918 |             77.4 |               72.8 |        0.24 GiB |
| lk                 |     3 |              0.789 |              0.753 |           1121.6 |              548.4 |        0.00 GiB |
| tapip3d_unfiltered |     3 |              0.904 |              0.917 |             27.1 |               25.4 |        3.21 GiB |
| tapnext            |     3 |              0.898 |              0.906 |             66.2 |               62.9 |        1.62 GiB |
| trackeverything    |     3 |              0.874 |              0.758 |              9.2 |                9.1 |        2.84 GiB |
| trackon_r          |     3 |              0.940 |              0.938 |             38.5 |               37.1 |        0.30 GiB |
