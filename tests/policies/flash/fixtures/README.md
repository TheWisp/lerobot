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
