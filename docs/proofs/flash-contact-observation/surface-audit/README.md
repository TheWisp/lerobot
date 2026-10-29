# Why the external dish loses its pose

Investigation of the unchanged pipeline at `c183227b2`. These are offline measurements, not tracker changes. The source calls the object a “white bowl”; its RGB shows a shallow square dish. Cache names retain `bowl` for provenance. No object-specific rules or new thresholds were added.

## Name-based lid and jar trials

Both complete 154-frame episodes used the existing SAM3 text detector, then its causal video tracker. No clicks were supplied. Outputs remain beside the local jar dataset under `flash/v1/observation-evidence-sam3-text` and `observation-evidence-sam3-text-body`.

| Prompts                 | Lid: observed / predicted / lost | Jar: observed / predicted / lost | Mask result                                                                         |
| ----------------------- | -------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------- |
| `lid`, `glass jar`      | 127 / 27 / 0                     | 2 / 78 / 74                      | Lid separated; jar still includes the lid, even after separation.                   |
| `lid`, `glass jar body` | 104 / 50 / 0                     | 1 / 26 / 127                     | Jar body first detected at frame 110, after opening; no jar mask at initialization. |

The first trial's median mask overlap with supplied masks is 0.905 for the lid and 0.454 for the jar. The body wording has zero median overlap because it is not detected for most frames; its 90th percentile is 0.936 after detection. Availability is not pose accuracy. Names solve the missing separate lid track, not reliable jar tracking throughout this episode.

## Actual saved surface

The surface backend holds the **first filtered partial depth cloud**, unchanged, and estimates a rigid transform for each new frame. It does not accumulate a complete mesh here. We reconstructed that exact cloud and each recorded candidate pose. Recomputed surface errors match all 143 saved values to within `2.8e-13 mm`.

The saved surface is roughly 200 × 200 mm, with a shallow depression, and looks like the dish. At frame 35 the median current-to-reference distance is 2.41 mm, but its 90th percentile is 8.22 mm. About 22% of the measured samples are over 5 mm away. Differences are stronger around the rim, but also occur in the interior; excluding just the immediate segmentation boundary does not remove them.

We reran the existing fitter from the recorded pose, the camera-motion-only pose (assuming this dish is stationary), and all four PCA starts. We also tried independent symmetric point-to-point ICP. These are diagnostic alternatives, not production candidate enumeration:

| Frame | Recorded surface p90 (mm) | Best existing-fit cost branch: surface p90 (mm) | Independent ICP: surface p90 (mm) |
| ----- | ------------------------: | ----------------------------------------------: | --------------------------------: |
| 8     |                      7.23 |                                            7.24 |                              7.33 |
| 15    |                      6.33 |                                            6.32 |                              6.24 |
| 35    |                      8.22 |                                            8.22 |                              8.07 |
| 70    |                      5.64 |                                            5.64 |                              5.62 |
| 120   |                      5.51 |                                            5.51 |                              5.32 |

All these remain above the configured 5 mm publication gate. This is evidence against a poor local initialization being the sole cause; it is not a proof of a global optimum. The measured surfaces differ beyond the assumed precision despite plausible RGB alignment. Original-source versus imported depth at frames 0, 8, 15, 35 and 70 differs by at most 1.23 mm, too little to explain the full discrepancy. We have not isolated the source capture's depth bias, geometry calibration, or view-dependent surface sampling into separate causes; there is no independent ground-truth surface.

## Exact loss at frame 13 (0.867 seconds)

The dish is still unobstructed. Both routes stop publishing:

1. Direct fit: surface p90 is **5.128 mm**, over the **5 mm** configured gate. The reverse depth check is 4.915 mm and the optimizer converged; 708 image-motion inliers remain. The failed condition is surface agreement, not absence of matches.
2. Environment fallback: after a new group at frame 3, membership shrinks from **57 to 12** by frame 12. The group update permanently keeps only the latest fit's inlier identities; it does not replenish its own membership. Fresh scene points exist, but they are not automatically admitted to that relationship. New accepted direct poses can establish a new relationship.

At frame 13, all **12** group IDs still have tracked image positions and valid depth. A rigid fit puts **11** within 5 mm; the last is 5.46 mm away. Replaying the actual `EnvironmentRelationship.track` with the saved points returns `None` at the unchanged 12-inlier requirement. The group therefore fails before those identities disappear in RGB. By frames 14–15, optical-flow loss and missing depth also reduce the available count. Its last successful measurement is frame 12; the group eventually expires under the existing timeout.

Replaying `ScenePointTracker` reproduced every saved group pixel exactly in the audited first 36 frames. At frame 1, all 32 previous group identities still have image tracks and depth, yet only 18 remain in the relationship; at frame 2 all 18 are measurable, yet 12 remain. Thus group erosion cannot be explained only by occlusion or lack of image features.

**Finding:** the visible dish becomes unavailable because the fixed reference/depth agreement exceeds a rig-configured precision gate, while the fallback progressively discards support and then cannot satisfy its minimum. It is not a SAM disappearance. Later errors reach 8–12 mm, so lowering one cutoff just enough for frame 13 would not address the whole clip. A useful next change must distinguish temporary measurement rejection from permanent loss of track identity, and model depth uncertainty; neither was changed in this investigation.

## Reproduce

Run `reconstruct.py`, `refit.py`, `icp.py`, then `groups.py` from the repository root with `PYTHONPATH=src` and the project Python environment. They are dataset-specific audit scripts: paths and selected diagnostic frames are explicit, inputs are read-only, and outputs go to `/tmp/flash_dish_surface_audit`. The checked-in JSON files capture this investigation's results; the scripts use the shared observation, fitting, and point-tracking code. Source RGB-D and masks remain in the local dataset cache.

For the name trials, use `benchmarks/flash_extract_observation_rgbd.py` with the jar dataset and its existing `observation-config.json`, `--prompt lid=lid`, and either `--prompt 'jar=glass jar'` or `--prompt 'jar=glass jar body'`. Pass a separate `--output` directory and no `--seed`. This requires the same SAM3/UFM inference environment as the earlier full-clip tests.
