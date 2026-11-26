`ring_static_frame_100_101.npz` contains only metric source and target point
correspondences from the local sealed insertion recording, frames 100 and 101.
SAM3 masked the green ring, DINOv2-small matched patches against frame 100,
and aligned millimetre depth was unprojected using that recording's `cam_K.txt`.
The source array is in the frame-100 camera/object coordinate system; target
is in frame-101 camera coordinates. No raw RGB or depth is included. It
captures a real false rejection caused by an extra post-consensus refit.

`dowel_held_frame_315.npz` contains a registered two-view dowel point model,
its frame-315 SAM3 mask and aligned depth, the frame-315 matched-patch pose,
and the same frame's point-group pose. The two reference views were frames
100 and 300, joined using the recorded group transforms. This fixture
documents a disagreement, not an independently measured ground-truth pose.

`ring_tray_motion_20261016.npz` contains sparse, persistent RGB-D point tracks
from camera frames 331, 360, 390 and 422 of the 2026-10-16 11:21:49 insertion
attempt. The tray and ring visibly moved together between the first two frames.
Ring centres are approximated from the old run's outline and aligned depth;
they are not independent ground truth. The fixture checks whether the new
environment relationship finds measured co-motion and predicts the later
centre within the sensor-scale tolerance. It stores no RGB, depth images or
motor actions.

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
