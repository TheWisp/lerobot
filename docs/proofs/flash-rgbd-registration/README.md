# RGB-D registration of the held dowel

**Result: none of the tested pairwise methods kept a sufficiently stable object
frame.** RoMa v2 `fast` was the best of this controlled comparison, but the
three landmarks chosen on the dowel at frame 280 differed from a rigid-grasp
reference by 19.3 mm median and its long-axis direction differed by 17.6°
median over frames 281–399. The latter is visible in the camera image. A
10-frame clear-view re-anchor improved the numbers slightly; a single distant
anchor stopped matching. This is evidence for maintaining several object views,
not a validated live tracker or a robot test.

[Watch the synchronized 1× video](registration_comparison.mp4). It shows all
120 frames, including the missed color-ICP matches. Cyan is the _same_ SAM3
mask on every panel. Yellow A/B/C are three fixed 3D points transported from
frame 280; their connecting lines show direction. White hollow circles are
where a rigid attachment to the recorded gripper would have moved those same
points. The renderer draws a landmark even when it is hidden. The gripper
reference is approximate, not pose ground truth.

In particular, A begins near the grasped end. UFM projects A outside the
_visible_ dowel mask in 103/120 frames; even the independent rigid-gripper
prediction does so in 107/120. In the final frame, depth at UFM's projected A
is about 24 mm **nearer** than the point, consistent with a foreground
occluder. Thus A outside the outline does not alone show a tracking error.
The outside-mask projection is inconclusive, but UFM's **22.9° final
long-axis difference** from the rigid-grasp reference is a serious failure for
insertion, not a harmless rotation around the dowel. A–B spans 58.6 mm and is
within 4.1° of the measured surface's principal axis at frame 280. This
benchmark does not establish which portion of the difference comes from
tracking versus actual grasp slip or calibration/timing error; it does establish
that the estimated pose is not validated for insertion control.

![All methods at frame 399](frame_399.jpg)

## Controlled comparison

The input is the recorded, aligned RealSense RGB and depth for the insertion
demo at `~/.cache/huggingface/lerobot/demos/insertion/recording`, with the same
cached causal SAM3 dowel masks as the [earlier mask/depth experiment](../flash-mask-depth-tracker/README.md).
Every frame uses the earlier depth cleanup: reject missing depth, an inner
2-pixel mask boundary, a local depth span above 10 mm, and disconnected
surface fragments; sample the retained surface at 3 mm. RGB match endpoints
must land on this measured surface in **both** frames. No gripper pose, later
frame, CAD model, dowel shape or task landmark enters registration.

For the learned matchers, the union of the two current masks defines one 320px
RGB crop. UFM uses its published `UFM-Base-DINOv2L-init` checkpoint and
source-to-target flow/covisibility. RoMa v2 uses its published v2.0.1 checkpoint
in `turbo` (320px) or `fast` (512px) mode. We keep endpoint matches with
covisibility/overlap at least 0.5, lift both pixels using their **own** aligned
depth, reject 3D outliers with a 6 mm RANSAC threshold, and fit one rigid
transform to the surviving 3D pairs. Colored ICP instead directly aligns the
same retained colored 3D samples and normals; three geometric/color weight
settings were tried. These are declared study settings, not calibrated safety
thresholds, and were not tuned against ground-truth object poses.

We compared each result _after estimation_ with the recorded gripper's relative
motion from frame 280. This assumes a rigid, non-slipping grasp, reasonably
correct camera calibration and frame synchronization. Violations can contribute
to these errors. Video/mask agreement is a second visual check, but cannot tell
us the exact hidden surface pose. The axis error uses the directed vector from
landmark A to B; no symmetry minimization was applied.

| Adjacent-frame method              | Valid pairs / 119 | Median one-step landmark difference | Median accumulated landmark difference | Median axis difference | Median registration time |
| ---------------------------------- | ----------------: | ----------------------------------: | -------------------------------------: | ---------------------: | -----------------------: |
| Colored ICP, geometric weight .968 |               109 |                              3.2 mm |                                48.3 mm |                  46.0° |               1.0 ms CPU |
| RoMa v2 turbo                      |               119 |                              1.6 mm |                                22.5 mm |                  27.8° |              32.4 ms GPU |
| RoMa v2 fast                       |               119 |                              1.7 mm |                            **19.3 mm** |              **17.6°** |              51.9 ms GPU |
| UFM Base                           |               119 |                              1.7 mm |                                20.6 mm |                  21.4° |              32.1 ms GPU |

The best color ICP weight missed frames 306–311, 313, 315–316 and 318.
Weights .7 and .3 had only 53/119 and 41/119 valid pairs, respectively; they
did not improve coverage. The learned matchers returned an accepted transform
for every adjacent pair, but that acceptance does **not** mean the object frame
stayed correct. In particular, a small one-step error can accumulate or rotate
the same chosen landmarks away from the dowel.

The time column covers crop, matching, outlier rejection and rigid fitting. It
excludes loading the model and the input preparation, which separately took
about 36–40 ms median per frame for disk reads and full-image depth cleanup.
SAM3 inference was already cached and is also excluded. These numbers are
single-process offline measurements on an RTX 5090, not a measured live frame
rate or a guarantee that all work can be overlapped with the 30 Hz robot loop.

## Re-anchoring ablation

Using RoMa `fast`, each frame was instead matched directly to the latest
clear-view keyframe. The keyframe is replaced after a fixed number of frames;
the new global transform comes only from earlier estimated transforms, never
from the gripper reference. This is a controlled interval test, not a proposed
fixed live reseed rule.

| Maximum keyframe age | Valid / 119 | Median landmark difference | Median axis difference | Last landmark difference |
| -------------------- | ----------: | -------------------------: | ---------------------: | -----------------------: |
| 10 frames            |         119 |                    18.1 mm |                  13.5° |                  19.7 mm |
| 20 frames            |         119 |                    18.2 mm |                  14.1° |                  18.1 mm |
| 40 frames            |         106 |                    21.2 mm |                  13.7° |                  23.2 mm |
| Single anchor at 280 |          31 |                    22.3 mm |                  23.3° |                  33.5 mm |

The lower 40-frame axis number is **not** a success: 13 transforms were
withheld. A more distant image can lack enough correct shared visible surface
even when it is the same object. The improvement at 10–20 frames is too small
to solve drift; it suggests testing a multi-view object map that preserves
coordinates, identifies genuinely new surface, and relocalizes when overlap
returns. The one-step matcher should provide candidate correspondences to
that map, rather than define the object frame by chaining alone. This proposed
map was **not implemented** here.

## Reproduce

The local benchmark source is `compare.py`; `keyframe_ablation.py` and
`render.py` create the ablation and video. CSV, summaries and complete pose
traces are in `results/`. Inputs and pretrained checkpoints are local cache
files; they are not committed. Reuse the earlier experiment's mask-generation
instructions to regenerate SAM3 masks. The UFM and RoMa source revisions used
here were `ff78d6c3db807c91578acccb58ee67d4394d5e5f` and
`95c9968145c8906b7b59383258e9f73b02853d89`, respectively. The isolated
benchmark environment used PyTorch 2.8.0+cu128, OpenCV 4.12.0, and an RTX 5090. Colored ICP used the repository Open3D 0.20 environment.

UFM's constructor normally downloads a separate DINOv2 backbone before
replacing it with its full checkpoint; the benchmark skips that redundant
pretrained initialization. It does **not** omit the UFM checkpoint's encoder
weights. The UFM convenience API normalizes RGB to float32 on the input
device, so the benchmark passes GPU tensors and casts the complete checkpoint
to float32 before prediction.

Validation: two focused tests verify known 3D rigid motion with outliers and
require measured, valid depth on both match endpoints; both passed. All 119 real
transitions were scored, and the encoded
H.264/yuv420p video has 120 frames at recorded speed with fast-start metadata.
Frames 340 and 399 were inspected at full resolution. No FLASH runtime,
dataset, tracker default, GUI, or robot setting was changed.
