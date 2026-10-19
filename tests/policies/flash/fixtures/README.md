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
