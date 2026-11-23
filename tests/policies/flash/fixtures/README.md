`ring_static_frame_100_101.npz` contains only metric source and target point
correspondences from the local sealed insertion recording, frames 100 and 101.
SAM3 masked the green ring, DINOv2-small matched patches against frame 100,
and aligned millimetre depth was unprojected using that recording's `cam_K.txt`.
The source array is in the frame-100 camera/object coordinate system; target
is in frame-101 camera coordinates. No raw RGB or depth is included. It
captures a real false rejection caused by an extra post-consensus refit.

`dowel_visible_growth_20261021.npz` contains the dowel's saved TrackEverything
3D query tracks and SAM3/depth support from insertion demo frames 100–489,
plus a sampled depth surface from frame 400 and the frame-489 mask/depth.
The original fixed-reference pose displacement is retained as a negative
control. It demonstrates that the old pose fit (computed before point growth)
projects the frame-400 surface mostly outside the frame-489 object mask,
whereas a causal rigid fit using newly visible point IDs remains on the
measured object. It stores no RGB, robot commands, or full episode.
It also includes a separately rerun 90-frame query set's frame-400/489 point
correspondences and initial pose, so the masked RGB-D surface correction has
an independent recorded acceptance case.

`ring_visible_motion_20261016.npz` stores 75 depth-supported TrackEverything
point frames for a visible ring moving with the tray in the 2026-10-16
11:21:49 real rollout, plus the initial measured surface and final aligned
mask/depth. It checks that a growing-point fit stays available and projects
onto the final visible RGB-D observation. The point model used future video
context, so this does not establish causal online tracking or 6D ground truth.
It stores no RGB or robot actions.

`dowel_held_frame_315.npz` contains a registered two-view dowel point model,
its frame-315 SAM3 mask and aligned depth, the frame-315 matched-patch pose,
and the same frame's point-group pose. The two reference views were frames
100 and 300, joined using the recorded group transforms. This fixture
documents a disagreement, not an independently measured ground-truth pose.

`dowel_live_visible_shape_20261020.npz` stores only the prepared dowel's
metric reference points, aligned depth inside two object masks, and camera
intrinsics from the failed 2026-10-20 startup scene. The full text-prompted
mask covers the visible dowel; the point-seeded mask covers only a fragment.
The other mask is the green ring, a same-frame negative identity case. No RGB,
robot state, or complete user episode is stored in this fixture.

`dowel_static_fit_drift_20261020.npz` stores the prepared dowel point model,
aligned depth, intrinsics, and SAM3 dowel masks from frames 0 and 13 of the
2026-10-20 11:45:32 physical FLASH trial. The dowel stayed on the tray and
both masks covered approximately the same 150-pixel span. Independent
nearest-surface fits each passed the 5 mm depth threshold, and their object
frame origins differed by more than 20 mm. The corresponding modeled surface
points moved by at most 1.5 mm, however: the arbitrary reference origin is far
from the visible surface and is not a valid continuity metric. This is a real
false-rejection regression case, without RGB or motor actions.

`dowel_axial_jitter_20261020.npz` stores the prepared dowel points, the first
demonstrated dowel reference pose, and 14 distinct direct RGB-D pose proposals
from the 2026-10-20 17:36:41 physical trial. The dowel remained on the tray;
successive proposals rotated up to 19.4 degrees, mostly about its long axis,
while their modeled surface centers stayed within a few millimetres. These
poses are tracker outputs rather than independent ground truth. The fixture
tests the explicitly annotated cylindrical symmetry without storing images or
motor actions.

`dowel_group_direct_ambiguous_20261020.npz` stores one aligned depth frame,
the corresponding segmented dowel mask boundary raster, camera calibration,
prepared model points, and the competing direct and verified-group poses from
the 2026-10-20 17:51:21 trial. The dowel was partly hidden during approach.
The two poses differ by about 13 mm at the modeled centre; the visible depth
patch does not clearly favor either pose. The fixture checks that the late
direct proposal is recorded as a disagreement rather than replacing the group
anchor. It contains no RGB or motor actions.

`ring_tray_motion_20261016.npz` contains sparse, persistent RGB-D point tracks
from camera frames 331, 360, 390 and 422 of the 2026-10-16 11:21:49 insertion
attempt. The tray and ring visibly moved together between the first two frames.
Ring centres are approximated from the old run's outline and aligned depth;
they are not independent ground truth. The fixture checks whether the new
environment relationship finds measured co-motion and predicts the later
centre within the sensor-scale tolerance. It stores no RGB, depth images or
motor actions.

`ufm_batch_depth_rounding_20261025.npz` contains the insertion demo's frame-500
and frame-501 filtered depth surfaces, source/target usable masks, intrinsics,
and UFM source/query coordinates from separate scalar and three-pair inference.
The checkpoint is `infinity1096/UFM-Base-DINOv2L-init`, with its default BF16
inference. Both calls used identical crops. Very small flow changes crossed a
depth-sampling boundary: scalar inference retained 130 metric correspondences,
batched inference 129. The actual rigid initialization and surface fitter then
returned different poses. This is evidence against assuming batching is an
exact acceleration, not a ground-truth claim that either result is better.
The fixture includes the resulting metric correspondences to check the real
pixel-to-depth consumer, and stores no RGB, motor actions or full episode.
See `docs/proofs/flash-native-acceleration/README.md` for the experiment.

`gamepad_reference_depth_20261028.npz` stores only the prepared gamepad's
first-view metric reference points, the frame-223/246/290/300/430 direct poses,
aligned depth and SAM3 masks, intrinsics and camera calibration from the `pick_place`
demo. It contains no RGB or robot commands. All five poses were labeled
observed by the earlier surface pipeline. The clean visible reference patch
agrees with measured depth at 290 and 300, but frame 430 has a large residual;
the fixture tests that a good current-to-reference patch fit alone does not
certify the whole transported reference or the hidden contact pose. A large
residual may also reflect a reference surface that is now occluded. These are
same-frame support checks, not independent ground truth or a verdict that the
old pose was wrong.

The gripper mostly hides the gamepad at 223 and 246. The cleaned connected
surface has less lateral extent than the earlier fit's own 90th-percentile
error, so it cannot certify a full rigid pose even though the fit passed its
old point-count and depth-error checks. Frames 290 and 300 provide the
complementary broader visible patch.

`image_matching_gamepad_168.npz` and `image_matching_dowel_240.npz` freeze
real-frame outputs from the implementation immediately before the 2026-11-02
LK/UFM adapter refactor. They include UFM input crops and raw network output,
SAM/depth acceptance inputs, expected correspondences and rigid motion, plus
scene-LK state and its next recorded RGB-D frame with expected IDs/pixels/3D
points. The adjacent JSON files identify source recordings, frame numbers and
SHA256 hashes of the original source files. They test refactor equivalence,
including rejected matches, not the accuracy of these estimates. No network,
checkpoint download or hardware is needed to run their tests.
These two recorded fixtures use Git LFS; run `git lfs pull` after checkout.

`gamepad_stationary.json` identifies the **real recorded integration test**, not
a saved tracker answer. It pins the RGB-D/mask input hash and UFM checkpoint.
Visual evidence covers stillness through frame 164 (7.24 s). The demonstrator
reports grasp around 9 s (±0.5 s), so the stationary check now extends through
frame 192 (8.47 s), before the earliest reported grasp. That occluded extension
uses a human annotation; it has no independent 6D pose labels. The earlier claim
of pickup in frames 165–170 was unsupported. The test runs
through frame 280 and requires subsequent movement, so freezing cannot pass.
The 5 mm / 5° stationary budgets were fixed before the fix and are never inputs
to inference. This does not certify absolute pose or post-grasp accuracy.

With the recording and checkpoint cached locally, run from the repository:

```bash
FLASH_RUN_RECORDED_TESTS=1 HF_HUB_OFFLINE=1 PYTHONPATH=src:. \
  uv run python -m pytest -q -s tests/policies/flash/test_gamepad_stationary_integration.py
```

Use the experiment's virtual environment if the project environment lacks UFM.
The test writes diagnostics to pytest's temporary directory, never to the demo.
It skips explicitly if the data/model/CUDA prerequisite is missing. It uses the
saved masks; segmentation generation is outside this test.

`green_controller_retained_view_20261107.npz` stores two cropped RGB-D frames
(430 and 436) with their SAM masks from the saved live clip
`saved-d6d3d3b38e7944e49c5b80be224f8cb4` (green game controller, a hand
approaching its right edge), the 758-point saved surface the live fit used at
frame 436, the pose accepted at frame 430, the group prediction checked at
frame 436, and the pipeline settings. Neither object nor camera moved between
the frames (0.02 mm at the surface centre). Fitting the surface against the
current mask and depth alone passes (99.3% overlap, 1.31 mm median), while the
retained frame-430 depth view rejected the prediction live: 22 of 2754 current
points, all inside both masks and 3-7 px from the silhouette, met frame-430 rim
depth 4-10 mm farther than frame 436 and the saved surface read there. The crop
(x 400-664, y 368-576, offsets multiples of the validation tile) reproduces the
recorded exposed weights exactly. The provenance array names the clip, frames,
recorded verdicts and code commit. No robot state or full episode is stored.
