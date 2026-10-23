# FLASH local moving-object stress clips

local moving-object stress clips; no independent material-point truth

First-frame error checks the adapter's query-coordinate contract. The mask column measures whether published points fall in the recorded object mask, and the depth column measures native 3D agreement within 10 mm of aligned sensor depth. Neither proves material-point identity. A 2D-only or sensor-depth-lifted method has no independent depth score. FPS is isolated adapter throughput, not FLASH or robot-loop FPS.

| Clip           | Model              | Query error median | In mask | Native depth agreement | Model FPS | Adapter FPS |
| -------------- | ------------------ | -----------------: | ------: | ---------------------: | --------: | ----------: |
| gamepad-90-145 | bootstapir         |             1.1 px |   0.477 |                    n/a |      48.8 |        42.0 |
| gamepad-90-145 | cotracker3         |             0.0 px |   0.169 |                    n/a |     170.9 |        96.8 |
| gamepad-90-145 | litetracker        |             0.0 px |   0.684 |                    n/a |     101.3 |        75.9 |
| gamepad-90-145 | lk                 |             0.0 px |   0.693 |                    n/a |    2932.0 |       250.9 |
| gamepad-90-145 | point4d            |             0.6 px |   0.581 |                    n/a |      18.0 |        17.6 |
| gamepad-90-145 | spatracker2        |             0.1 px |   0.293 |                  0.140 |       7.2 |         7.0 |
| gamepad-90-145 | tapip3d_unfiltered |             0.0 px |   0.359 |                  0.305 |      31.6 |        27.0 |
| gamepad-90-145 | tapnext            |             0.8 px |   0.155 |                    n/a |      78.8 |        62.2 |
| gamepad-90-145 | trackeverything    |             0.7 px |   0.421 |                  0.272 |      19.2 |        17.9 |
| gamepad-90-145 | trackon_r          |             0.3 px |   0.528 |                    n/a |      44.1 |        37.7 |
| yellow-0-74    | bootstapir         |             0.9 px |   0.671 |                    n/a |      53.1 |        45.3 |
| yellow-0-74    | cotracker3         |             0.0 px |   0.573 |                    n/a |     198.0 |        96.5 |
| yellow-0-74    | litetracker        |             0.0 px |   0.682 |                    n/a |     116.4 |        84.5 |
| yellow-0-74    | lk                 |             0.0 px |   0.537 |                    n/a |    2270.0 |       227.2 |
| yellow-0-74    | point4d            |             1.0 px |   0.671 |                    n/a |      18.6 |        18.2 |
| yellow-0-74    | spatracker2        |             0.1 px |   0.686 |                  0.401 |       7.2 |         7.1 |
| yellow-0-74    | tapip3d_unfiltered |             0.0 px |   0.676 |                  0.654 |      26.2 |        22.3 |
| yellow-0-74    | tapnext            |             0.6 px |   0.273 |                    n/a |      83.7 |        65.1 |
| yellow-0-74    | trackeverything    |             0.4 px |   0.780 |                  0.725 |      18.5 |        17.3 |
| yellow-0-74    | trackon_r          |             0.3 px |   0.779 |                    n/a |      46.5 |        39.7 |
