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
