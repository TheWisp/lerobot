# Labeled RGB-D data for tracking accuracy

The [ten-scene sweep](ten-scene-sweep/README.md) measured whether FLASH could
publish a pose, not whether that pose was correct. A useful accuracy benchmark
needs consecutive RGB-D frames, calibration, and object poses measured or
annotated independently of FLASH. Object masks may be used as a separate
_oracle segmentation_ experiment, but object-pose labels must stay out of the
tracker.

| Source                                                                                                    | What it tests                                                                           | Label caveat                                                                                           | Access                                                                                           |
| --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| [Laval 6DOF](https://vision.gel.ulaval.ca/~jflalonde/publications/projects/6dofObjectTracking/index.html) | Kinect V2 sequences with hand occlusion, interactions, and stability trials; 11 objects | Commercial motion capture supplies object-pose reference; source has a non-commercial research license | A 203-frame clock sample is downloaded locally; full per-object archives are about 10–15 GB each |
| [YCBInEOAT](../flash-full-pose-benchmark/README.md)                                                       | Robot-held objects and occlusion                                                        | Pose annotations can jump; nine 31-frame clips already scored                                          | Local source sequences are available                                                             |
| [DexYCB](https://dex-ycb.github.io/)                                                                      | Human hand/object contact, eight calibrated RGB-D views                                 | Pose labels are annotations rather than independent MoCap                                              | Large per-subject archives                                                                       |
| [DynOPETs](https://github.com/Launch-on-Titania/DynOPETs)                                                 | Dynamic objects and moving camera                                                       | Check annotation method before treating poses as metrology                                             | UOPE56 download is 18.24 GB                                                                      |
| [ClearPose](https://github.com/opipari/ClearPose)                                                         | Transparent objects, raw versus completed depth                                         | Poses and completed geometry are annotation products; score against raw depth for sensor realism       | Separate Dropbox download                                                                        |
| [TransBiolab](https://dualtransparency.github.io/TransBiolab/)                                            | Transparent biomedical clutter and ordered camera motion                                | Objects are static while a robot-mounted camera moves                                                  | Full sequences and per-frame poses, masks, calibration; download size/access to check            |

The small [HOPE BOP validation archive](https://bop.felk.cvut.cz/datasets/)
contains 6D labels and masks, but its five views per scene repeat essentially
the same object pose. The available TUD-L BOP subset skips many source frames.
Neither is a substitute for a causal frame-by-frame tracking sequence.

## First independent-label check: Laval clock sample

I imported the 203 consecutive `clock_interaction_hard` RGB-D frames at
640×360 and ran the same FLASH observation pipeline and saved configuration
used by the Caryx sweep. SAM3 received only the text prompt “pink clock with a
round dial” and a first-frame click. FLASH did **not** receive Laval's pose
labels or mesh. For evaluation only, the scorer converts Laval's OpenGL camera
coordinates to FLASH's positive-Z camera coordinates, then compares the
_relative_ motion of 256 labeled-model surface points from frame 0. The model
projection was visually checked against the RGB image. Source timestamps are
not present in the sample; the importer used a nominal 30 Hz frame-index clock,
so this run establishes no source-timing or runtime-FPS claim.

| Output                                             | Frames after initialization | Median point error | Median rotation error |
| -------------------------------------------------- | --------------------------: | -----------------: | --------------------: |
| Published observed or predicted pose               |                    22 / 202 | 2.7 mm on those 22 |      2.5° on those 22 |
| Raw surface proposal, including rejected proposals |                   202 / 202 |            86.1 mm |                124.7° |

FLASH published three direct observations, 20 predictions, and 180 lost
frames. It rejected 158 proposals as “visible patch is too narrow for a full
rigid pose” and 42 for insufficient RGB-D support. The SAM3 mask remains on
the clock in inspected early, middle, and late frames, even when the hand
partly covers it. Around frame 96 the raw orientation jumps to a near-opposite
branch; its 3D point error stays around 90 mm afterward. The publication gate
prevents that wrong proposal from becoming an act goal, but the available
fallback runs out. This is **one sample**, not an accuracy ranking of trackers.
Its reported 2.7 mm median is conditional on the 22 published frames; it does
not imply that the whole 203-frame track is accurate.

Reproduce the score with
[`flash_score_laval_6dof.py`](../../../benchmarks/flash_score_laval_6dof.py),
passing the original `clock_interaction_hard` directory, its `geometry.ply`,
the complete FLASH evidence directory, and role `clock`. The source RGB-D,
mesh, labels, detailed score and plots stay in the local cache; no Laval media
is redistributed here. The [dataset license](https://github.com/lvsn/6DOF_tracking_evaluation#license)
allows academic/non-profit/government research and requires separate
permission for commercial use. The next labeled check should include other
object shapes and occlusion modes, then a transparent-object source. Metrics
must always report both **accuracy when available** and **fraction of frames
available**.
