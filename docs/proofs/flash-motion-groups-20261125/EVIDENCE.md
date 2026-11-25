# FLASH scene motion groups — evidence

Rendered on 2026-11-25 from three recorded fixtures that the scene motion
groups PR ships, using that PR's own functions. Only the image-matching
fixture stores colour images; the group replay stores point positions only,
and the depth fixture stores one depth image. These record one run at one
revision; they are not design documents.

| File                                  | What it shows                                                                                                                                                 | Inputs                                                                                                                                                                                                                                                         |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01-matches-to-3d-pairs.png`          | Image matches that start on the gamepad, coloured by the first check each one fails: matcher confidence, target mask, usable depth, agreement with one motion | `image_matching_gamepad_168.npz`; `ufm_pairs` replaying the recorded UFM output, then `robust_fit` with default `SurfaceMotionConfig`, as in `test_recorded_matches_depth_fit_and_scene_ids_equal_original`                                                    |
| `02-motion-groups-ring-tray.png`      | Every scene point at four frames, coloured by motion group, with the ring points chosen at frame 233                                                          | `live_9103_static_ring_groups_20261116.npz`; `SceneMotionGroups(GroupConfig())` over all 152 updates, as in `test_established_ring_and_tray_edge_points_never_separate_or_disappear`                                                                           |
| `03-depth-removes-carried-points.png` | Carried group members at their predicted pixels on the recorded depth image, split into the ones measured depth contradicts and the rest                      | `group_depth_20261119/`; `SceneMotionGroups` with the recording's settings over 37 snapshots, then the last one with `SceneDepth.from_frame(tolerance_m=0.005, radius_px=2)`, the evidence `test_batched_depth_preserves_per_group_results_and_cleanup` builds |

Measured on these inputs:

- Gamepad frames 167 → 168: 413 grid matches start on the gamepad. 49 are
  below the matcher's confidence, 15 land outside the frame-168 mask, 98 lack
  usable depth at one end, and 251 become usable 3D pairs. 248 of those agree
  on one rigid motion (median distance 0.18 mm; the gamepad moved 4.5 mm).
- Ring and tray: 61 ring points and 13 tray-wall points share one group at
  frame 233. In each of the 132 updates from frame 233 to 865 they are all in
  one group, and none is ever dropped. Between frames 233 and 253 that group's
  points moved a median 5.4 mm while the rest of the table moved 0.0 mm. With
  the ring's measurements withheld for eight updates from frames 233, 530 and
  806, the group predicts 1,020 recorded ring positions with a 1.1 mm median
  and 7.0 mm 95th-percentile error.
- Frame 5381: 805 points measured; 640 group members carried. Depth removes
  66 of them, including all 15 the fixture names; at those 66 pixels the
  camera measures a surface a median 30 mm (at least 7 mm) farther than the
  predicted point. The other 574 are kept.

Re-render with `render_286.py` from the landing-build tooling:
`python render_286.py <tree with src/> tests/policies/flash/fixtures <out>`.
