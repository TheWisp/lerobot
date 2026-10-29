# Actionless RGB-D interaction check

We imported episode 31 of [CaryX AI's everyday-manipulation-3d dataset](https://huggingface.co/datasets/CaryxAI/everyday-manipulation-3d), a 154-frame lid-and-jar interaction, as a local LeRobot observation episode. The source is released under CC BY-NC-SA 4.0; its images and videos remain in the local dataset cache, not in this repository. The importer kept RGB-D, camera motion and two object masks, but copied **no hand state or actions**. RGB and masks were aligned to the depth camera's 256 × 192 grid from the source's declared same-ray 2× grids. It did not infer a robot gripper.

The local episode is `/home/feit/.cache/huggingface/lerobot/demos/flash_external_caryx_lid_jar_observation`. A readback of frames 0, 77 and 153 found identical saved camera transforms, preserved no-return depth pixels, and a 1.43 mm 99th-percentile difference from source depth after millimetre rounding and the second video encoding. The output has 154 distinct camera samples and no `action` or `observation.state` feature.

We ran the **same** saved-mask, surface-fit, causal scene-group and evidence code used for FLASH demo preparation. The source's object–object contact annotation was not an input to the tracker. It marks contact on 91 of the output frames and serves only as an independent interval to inspect.

| Role | Directly observed pose | Group/prior prediction |      Lost |
| ---- | ---------------------: | ---------------------: | --------: |
| Lid  |             113 frames |              41 frames |  0 frames |
| Jar  |               9 frames |              96 frames | 49 frames |

This is an informative **failure**, not a completed tracking result. The jar mask still has a median of 1,632 pixels in frames 105–153, but the estimator reports no usable pose there. Earlier direct fits increasingly fail the visible-surface support check, and tracked jar support dwindles to zero; after that, the scene estimator correctly says **unknown** instead of keeping the old pose. The current one-view reference and this sensor's depth quality are possible contributors; this run does not isolate their effects. It shows that a visible mask alone does not establish a full, stable object pose.

We then restarted the same tracker at frame 110, using the newly visible jar as its reference. It accepted that first frame, but the very next frame's proposed surface differed from depth by 24.8 mm at the 90th percentile, while the patch extended only 17.5 mm in its weaker lateral direction. Across the 44-frame late segment it produced one direct jar pose, 26 predictions and 17 lost frames. A late reference therefore does **not** solve this case by itself. The jar is transparent; at frame 110, valid depth inside its mask spans roughly 464–577 mm from the 10th to 90th percentile and includes mixed near/far patches. This is consistent with unreliable returns from glass, but it does not by itself separate material effects from ARKit depth noise, camera-pose error and fitting error. We did not relax a threshold to make it appear tracked.

In Run → FLASH, the actionless episode plays in the usual camera and 3D scene views. We marked lid-relative-to-jar review key frames at 1, 30, 96 and 120 using the editor and verified they persisted after reload. At frame 1 both poses were directly observed; at 30 and 96 the jar relation relied on a prediction; at 120 the jar was lost and the relation is explicitly unverified. No placement requirement or turn permission was asserted from this uncertain evidence. A headless browser check confirmed the observation-specific legend, visible mask alongside the lost status, four saved key frames and 3D scene loading without page errors.

The generic importer is [`benchmarks/flash_import_lerobot_rgbd_observation.py`](../../../benchmarks/flash_import_lerobot_rgbd_observation.py). It requires a depth-grid calibration and an explicit same-ray declaration before downsampling RGB; it stores source and alignment provenance. The remaining work is to recover the jar's pose from later visible views and to establish task-relevant contact-point accuracy before an adapted robot action could use this episode.

## Second interaction: device into a bowl

We also imported [episode 110 of the same source](https://huggingface.co/datasets/CaryxAI/everyday-manipulation-3d) to `/home/feit/.cache/huggingface/lerobot/demos/flash_external_caryx_device_bowl_observation`. It has 143 moving-camera RGB-D frames and masks for a small device and a large opaque bowl. The hand moves the device into the bowl; its source phase descriptions are annotations, not measured contact truth. As above, the normalized episode contains no action or hand state. FLASH lists it as an observation demo with saved perception evidence.

| Role   | Directly observed pose | Group/prior prediction |       Lost |
| ------ | ---------------------: | ---------------------: | ---------: |
| Device |              53 frames |              90 frames |   0 frames |
| Bowl   |               9 frames |               6 frames | 128 frames |

The device remains tracked, but the bowl's pose is lost after the early frames. At frames 43, 70, 95 and 123, the bowl mask still covers thousands of pixels, yet the surface proposal fails the RGB-D support check and the group has no surviving points. The visible mask does not make a verified rigid pose. Because this also happens to an opaque object, transparency is not a sufficient explanation for the jar failure. We have not established whether the limiting factor is sensor depth, camera calibration/motion, the surface fitter, group point selection or some combination. That is the next perception investigation; we did not hide the failure by loosening a threshold.

### Camera and group diagnostic

The source [defines its camera pose as camera-to-world](https://huggingface.co/datasets/CaryxAI/everyday-manipulation-3d#features), and the importer copied that feature. RGB readback against the source at frames 0, 1, 5, 15, 30, 60, 90 and 142 had a median pixel difference of about 2 intensity levels after downsampling and re-encoding. Thus the new observation video has the expected frame order and images. However, a fixed background point from frame 0 projected through those recorded camera poses increasingly misses its tracked RGB location:

| Frame | Initial-background tracks still visible | Recorded-pose median reprojection error | Held-out median after fitting camera motion from other tracks |
| ----: | --------------------------------------: | --------------------------------------: | ------------------------------------------------------------: |
|     1 |                                     257 |                                  0.3 px |                                                        0.0 px |
|    15 |                                     213 |                                 22.4 px |                                                        1.0 px |
|    60 |                                     160 |                                 27.7 px |                                                        1.4 px |
|    90 |                                     110 |                                 36.5 px |                                                        3.7 px |

The comparison excludes the two annotated objects and checks the alternative on points not used to fit it. It is repeatable with [`benchmarks/flash_camera_pose_audit.py`](../../../benchmarks/flash_camera_pose_audit.py). It established an image/pose alignment problem, but the later fixed-basis test below explains it without fitting a new camera trajectory. At frame 15, substituting the image-derived alternative in a diagnostic reduced the apparently stationary bowl's world-position change from 57.9 mm to 4.0 mm. The same change did **not** rescue the scene-group relationship: only 34 nearby background points were seeded, and the current 5 mm rigid-fit gate had fewer than 12 survivors by frame 3 even with the camera correction. Its 3D depth samples are also inconsistent.

The lid-and-jar clip shows the same camera-pose inconsistency: among frame-0 background points still tracked at frame 60, the recorded pose has 18.3 px median reprojection error, versus 1.4 px for the held-out PnP check. At frame 120 the values are 21.5 px and 0.8 px on 56 surviving tracks. This repeats across two scenes, but the fixed first-frame landmark set shrinks with time; later PnP checks are diagnostics, not a certified replacement trajectory.

### Controlled camera-pose comparison

We replayed both complete episodes through the **same** FLASH vision and scene-group code, passing the camera audit as an explicit alternative. Each frame used its recorded camera pose unless the image-derived candidate fit the audit's held-out RGB tracks better; the source episodes and normal prepared evidence were not overwritten. The comparison evidence stores its selected camera transform per frame, so an eventual replay can project its surfaces with the pose actually used during processing.

| Episode       | Source direct poses | Candidate direct poses | Source predicted/lost target frames | Candidate predicted/lost target frames |
| ------------- | ------------------: | ---------------------: | ----------------------------------: | -------------------------------------: |
| Device → bowl |   device 53, bowl 9 |      device 53, bowl 9 |                        bowl 6 / 128 |                          bowl 15 / 119 |
| Lid → jar     |      lid 113, jar 9 |         lid 113, jar 9 |                         jar 96 / 49 |                            jar 95 / 50 |

For the stationary bowl, raw world-position displacement from frame 0 to frame 15 fell from 57.9 mm with the recorded camera pose to 4.0 mm with the candidate. At frame 60 it fell from 70.7 mm to 8.9 mm. This is a useful diagnostic, but it does **not** fix direct surface acceptance or provide a reliable target throughout contact. The image-based PnP correction remains an experimental comparison, **not** a default or an action-ready calibration. The fixed-basis result below supersedes the earlier speculation that a newly fitted camera trajectory was needed.

### Root cause of the camera mismatch

The per-frame PnP comparison above was an intermediate diagnosis, **not** the appropriate correction. A follow-up check found that the source camera-pose axes are a fixed quarter-turn away from the RGB-D pixel axes as consumed here. Let `P_i` be the published camera-to-world matrix and `A` be a 90° clockwise rotation about the optical axis (`[[0,1,0],[-1,0,0],[0,0,1]]`). Reinterpreting the pose as `P_i @ A` gives the camera-relative transform `(P_i @ A)^-1 @ (P_0 @ A)`. This is a coordinate-frame conversion, not a fitted motion trajectory or a threshold change. The source [describes its pose as already in the shipped raster frame](https://huggingface.co/datasets/CaryxAI/everyday-manipulation-3d#features), so we cannot yet say whether the field or its documentation is wrong; the original source frames, before our import, show the same mismatch.

On independent SIFT matches in the original source's fixed wooden background, the bowl clip at frame 15 has 24.9 px median image reprojection error and 58 mm median disagreement with that frame's measured depth using the published pose; the fixed basis conversion gives 0.6 px and 4 mm. Giving each match any depth between 0.1 and 5 m barely improves the uncorrected image error, so noisy depth alone cannot explain it. Across all 143 imported bowl frames with at least 12 surviving background tracks, the median image error falls from 20.0 to 1.7 px; across all 154 jar frames it falls from 16.3 to 1.1 px. This also makes unmodeled lens distortion an implausible explanation for the large, coherent error, though the test still assumes the chosen background is stationary and uses the source's pinhole intrinsics.

Do **not** promote the PnP candidate or apply this basis conversion to arbitrary cameras. The importer now accepts an explicit `--pose-camera-from-rgbd-camera` transform, stores the unmodified source pose alongside the aligned pose, and leaves the default import and robot-control path unchanged. We validated the aligned RGB-D and pose readback, then reran the unchanged perception pipeline on these two recordings and a third recording whose camera axes already agree. The [three-scene retest](THREE_SCENE_RETEST.md) records both the remaining tracking failures and a depth-threshold sensitivity check.
