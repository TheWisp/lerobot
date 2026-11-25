# FLASH tracking glossary

Definitions of the records and terms FLASH tracking uses, each with the code it describes. An
entry is added in the same change as the code it describes.
These definitions describe what the system stores or computes; they do not certify an estimate.

- [RGB-D frame](#rgb-d-frame)
- [Depth support](#depth-support)
- [Published pose estimate](#published-pose-estimate)
- [Pose disagreement](#pose-disagreement)

## RGB-D frame

**One camera capture held in memory: a colour image, a depth image aligned pixel-for-pixel to
it, and what is needed to turn its pixels into metric 3D points.** This is FLASH's `RGBDFrame`.
It is not a row of a LeRobot dataset and is never written to one as such. A frame is built from
one live camera read, or from a recorded episode's colour and depth columns at one timestamp.

A frame carries the colour image (H×W×3, 8-bit) and depth image (H×W) on the same pixel grid;
the camera intrinsics and the depth scale (`depth_m_per_unit`), which together convert a pixel
and its depth into a 3D point in camera coordinates, in metres; and `base_from_camera`, the
camera's pose in robot-base coordinates. When no camera-to-robot calibration is applied, that
transform is the identity and robot-base coordinates are camera coordinates.

It also records `frame_id`, which increases for each camera, and two times: `source_time_s`,
when the image was captured, and `available_time_s`, when FLASH received it, which can never be
earlier. A pose computed from the frame refers back to it by these values (see
[Published pose estimate](#published-pose-estimate)). Optional fields hold the depth image before
depth filters and a synchronized infrared pair for learned stereo depth.

Code: [`RGBDFrame`](../observations.py#L16);
[`LatestRGBDFrameStore`](../observations.py#L65) rejects a frame whose ID or capture time does
not increase.

## Depth support

**A pixel has depth support when every pixel in a small square window around it has a depth
reading** (finite and positive). The window radius, in pixels, stands for uncertainty in where a
tracked point lands in the image. It is not a metric tolerance and does not shrink the object's
[segmentation mask](https://github.com/TheWisp/lerobot/blob/a75269aa4731297af3f19ed7d5431d873e5e7173/src/lerobot/policies/flash/docs/glossary.md#segmentation-mask).

A pixel without support is treated as missing depth, never as a far-away surface, and holes are
not filled. Support establishes that readings exist, not that they are accurate. Sampling an
object's surface additionally withholds depth discontinuities inside the mask.

Code: [`depth_support_mask`](../depth_quality.py#L9);
[`depth_quality_mask`](../surface_observation.py#L73) withholds discontinuities.

## Published pose estimate

**The record a tracker publishes for one object role: a 4×4 pose in robot-base coordinates, how
it was obtained, and which [RGB-D frame](#rgb-d-frame) it describes.** This is FLASH's
`PoseEstimate`, the handoff from tracking to action execution.

`evidence` is `observed` (measured in that frame), `predicted` (carried forward without a new
measurement) or `lost` (no pose; the record refuses one); see
[Tracking loss and recovery](https://github.com/TheWisp/lerobot/blob/a75269aa4731297af3f19ed7d5431d873e5e7173/src/lerobot/policies/flash/docs/glossary.md#tracking-loss-and-recovery) for how the tracker decides.
`source_time_s` and `source_frame_id` name the frame the pose describes, and `available_time_s`
is when it was published, which can never be earlier. `translation_supported` and
`rotation_supported` mark which position axes (robot-base X, Y, Z) and which rotation axes (the
object's own X, Y, Z) the evidence constrains. A partial pose can serve a task that does not
depend on its unsupported axes.

A consumer uses an estimate only if it is not lost, was already published, is no older than the
consumer's age limit, is allowed to be a prediction, and supports every axis the task needs.

Code: [`PoseEstimate`](../pose.py#L20) and [`usable`](../pose.py#L84).

## Pose disagreement

**A comparison of two pose proposals for the same object in the same RGB-D frame, using only
measured depth and the [segmentation mask](https://github.com/TheWisp/lerobot/blob/a75269aa4731297af3f19ed7d5431d873e5e7173/src/lerobot/policies/flash/docs/glossary.md#segmentation-mask).** Each proposal places the
object's stored 3D points in the camera and projects them into the image; on each pixel only the
nearest projected point counts.

Two numbers come out for each proposal: how closely measured depth matches the projected surface
where both exist inside the mask (the 90th-percentile depth difference, in metres), and the
fraction of projected pixels that land on measured depth inside the mask. A nearer measured
surface outside the mask is an occluder and does not count against a proposal. A projected
point hidden behind the measured surface inside the mask cannot support it.

The verdict is decided in this order. If either proposal has too few measured depth pixels inside
the mask, it is `insufficient_evidence`, even when the two poses are close. If they are within
the caller's task tolerance of each other, they `agree`. Otherwise one proposal is favoured only
if it is at least as good on both numbers and strictly better on one. When neither is, including
when the numbers are equal, the verdict is `ambiguous` and neither proposal is chosen silently.
These numbers are diagnostics, not calibrated pose likelihoods.

Code: [`surface_support`](../pose_disagreement.py#L135) and
[`compare_pose_proposals`](../pose_disagreement.py#L202).
