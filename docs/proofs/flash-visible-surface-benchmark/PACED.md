# FLASH paced perception pilot

paced development pilot, one run per model; not steady-state or live arm FPS

The producer releases recorded frames at their source timestamps and processes the newest available frame. The independent consumer ticks at 30 Hz and holds the last published result. Unprocessed frames remain missing in accuracy scores. Model loading is excluded, but first-frame model warmup is included.

| Clip                | Model       | Source FPS | Processed / source | First result | Consumer age median / p90 | Gap p90 | 30 Hz misses |
| ------------------- | ----------- | ---------: | -----------------: | -----------: | ------------------------: | ------: | -----------: |
| dowel-280-310       | bootstapir  |       29.9 |              25/31 |       260 ms |                33 / 33 ms |   35 ms |            0 |
| dowel-280-310       | litetracker |       29.9 |              26/31 |       227 ms |                33 / 34 ms |   34 ms |            0 |
| dowel-280-310       | lk          |       29.9 |              31/31 |         7 ms |                33 / 33 ms |   34 ms |            0 |
| dowel-280-310       | tapnext     |       29.9 |              26/31 |       223 ms |                33 / 33 ms |   34 ms |            0 |
| dowel-280-370       | bootstapir  |       30.0 |              85/91 |       262 ms |                32 / 33 ms |   35 ms |            0 |
| dowel-280-370       | litetracker |       30.0 |              86/91 |       227 ms |                32 / 33 ms |   34 ms |            0 |
| dowel-280-370       | lk          |       30.0 |              91/91 |        12 ms |                32 / 33 ms |   34 ms |            0 |
| dowel-280-370       | tapnext     |       30.0 |              86/91 |       223 ms |                32 / 33 ms |   35 ms |            0 |
| dowel-280-370       | trackon_r   |       30.0 |              84/91 |       285 ms |                32 / 33 ms |   35 ms |            0 |
| ycb-mustard-100-130 | bootstapir  |       16.4 |              28/31 |       259 ms |                53 / 76 ms |   66 ms |            0 |
| ycb-mustard-100-130 | litetracker |       16.4 |              29/31 |       225 ms |                42 / 67 ms |   66 ms |            0 |
| ycb-mustard-100-130 | tapnext     |       16.4 |              29/31 |       221 ms |                43 / 66 ms |   68 ms |            0 |
| ycb-mustard-100-130 | trackon_r   |       16.4 |              28/31 |       280 ms |                55 / 78 ms |   67 ms |            0 |

This is a minimal Python scheduling probe. It does not run SAM, the rigid fit, FLASH planning, motor IO, or display work. Consumer deadline misses therefore do not certify the robot loop. The public YCBInEOAT clip's source rate is about 16 Hz; a 30 Hz consumer naturally repeats estimates there. The local dowel source is about 30 Hz.
