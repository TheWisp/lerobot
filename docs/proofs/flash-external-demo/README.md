# External RGB-D workflow probe

The [YCBInEOAT real-robot sugar-box sequence](https://github.com/wenbowen123/iros20-6d-pose-tracking) was already cached for tracker benchmarking. We used source frames 619–649 (31 frames, 1.67 s), outside the two FLASH demos. The source supplies RGB, aligned depth, camera intrinsics, object masks and annotated object poses. This probe did not use an object mesh as tracker input.

`benchmarks/flash_import_external_rgbd.py` imported the frames as an ordinary local LeRobot episode at `/home/feit/.cache/huggingface/lerobot/demos/flash_external_ycb_sugar_observation`. Its `flash/v1/source.json` preserves provenance, intrinsics and the fact that the episode is **perception-only**. Original irregular timestamps and frame indices are explicit per-frame features. The standard dataset's nominal video rate is 18 fps. Readback checked frames 0, 15 and 30: all no-return depth pixels remained zero and the 99th-percentile valid-depth difference was 1 mm. The default depth-video minimum would instead have changed zero to 10 mm; this import sets the minimum to zero because the source's zero means invalid.

We ran the shared `SurfaceTrackingSession` through all 31 imported frames, once with the dataset's masks and once with a point-seeded SAM3 mask. SAM3's median mask intersection-over-union with the supplied masks was 0.669. Both tracker runs emitted a fitted pose on all 30 subsequent frames, but independent object-pose labels show that **emitting a pose is not the same as getting it right**:

| Masks    | Median orientation error | Median observed-patch center error | Median reverse depth disagreement |
| -------- | -----------------------: | ---------------------------------: | --------------------------------: |
| Supplied |                    14.2° |                            10.4 mm |                           11.4 mm |
| SAM3     |                    15.6° |                             7.1 mm |                           12.5 mm |

The orientation error remains with supplied masks, so SAM3 segmentation alone cannot explain it. This short sequence is a generalization failure to investigate, not evidence that a transferable skill was prepared. [The side-by-side video](sugar-box-external-tracking.mp4) compares the projected FLASH patch with the dataset's pose label on identical RGB frames; green traces the SAM3 mask. It runs slower than the source for inspection.

The normalized benchmark clip contains no imported robot actions, measured tool trajectory or camera-to-robot transform. The original ROS bag has gripper transforms, but its camera and robot frames are disconnected in the recorded transform tree. We therefore did not invent calibration or compile an executable skill. A complete external-demo test should use a source with synchronized RGB-D, robot actions/tool pose and calibrated camera-to-robot geometry. [RH20T's published API](https://github.com/rh20t/rh20t_api) describes all of those fields, but its public download is much larger; that is a suitable later full-workflow source.
