# FLASH tracking glossary

Definitions for the shared growing-surface tracker used by live camera, video examples,
and demo extraction. The current sources are the [tracking design](tracking_pipeline.md),
the [maintained frame architecture](../../../gui/static/flash_architecture.html), and the
code cited below.
These definitions describe what the system stores or computes; they do not certify an estimate.

The design gives usable mask and depth priority over image-motion suggestions, keeps scene-group
updates separate from object-pose decisions, and distinguishes observed poses from predictions and
lost results. Predictions supply no new object surface. See the
[core loop](tracking_pipeline.md#core-loop) and
[scene-group boundary](tracking_pipeline.md#scene-groups-and-object-association).

- [RGB-D frame](#rgb-d-frame)
- [Depth support](#depth-support)
- [Point track ID](#point-track-id)
- [Image match / 2D correspondence](#image-match--2d-correspondence)
- [Usable and validated 3D pairs](#usable-and-validated-3d-pairs)
- [Segmentation mask](#segmentation-mask)
- [Object coordinates and pose](#object-coordinates-and-pose)
- [Published pose estimate](#published-pose-estimate)
- [Pose disagreement](#pose-disagreement)
- [Stored surface / surface map](#stored-surface--surface-map)
- [Local surface fitting](#local-surface-fitting)
- [Object point anchor](#object-point-anchor)
- [Motion group](#motion-group)
- [Pose prediction / prior](#pose-prediction--prior)
- [Kalman pose update](#kalman-pose-update)
- [Accepted observation / reference image](#accepted-observation--reference-image)
- [Tracking loss and recovery](#tracking-loss-and-recovery)

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
[segmentation mask](#segmentation-mask).

A pixel without support is treated as missing depth, never as a far-away surface, and holes are
not filled. Support establishes that readings exist, not that they are accurate. Sampling an
object's surface additionally withholds depth discontinuities inside the mask.

Code: [`depth_support_mask`](../depth_quality.py#L9);
[`depth_quality_mask`](../surface_observation.py#L73) withholds discontinuities.

## Point track ID

**An integer label carried by a tracked image point across processed frames.** For example, ID 42
labels a pixel in one image and the pixel that the matcher assigns to it in the next image. This
represents a hypothesis that both pixels show the same physical point; tracking can be wrong.

The shared tracker keeps the ID when it accepts the next match. A newly selected point receives a
new ID. Matching an older accepted image through the stored history requires IDs shared with the
current frame.

Code: [`SharedFrameTracker._seed`](../shared_image_tracking.py#L93) assigns IDs;
[`update`](../shared_image_tracking.py#L163) carries them;
[`pairs`](../shared_image_tracking.py#L223) joins surviving IDs.

## Image match / 2D correspondence

**A proposed pairing of a pixel in a source image with a pixel in a target image.** LK and UFM
supply these pixel pairs. An accepted image match has passed the matcher's checks; it does not yet
establish a valid 3D measurement or the physical identity of the object.

**Actual inputs in the shared tracking path:** two full RGB images, resized to the configured image
budget, and 2D query pixels. LK uses grayscale versions. SAM masks and depth are not inputs to the
image-matching backend: the caller uses them to select queries and filter the returned pairs. This
shared pass tracks background points as well as object points. It does not create a separate crop
around each object's old and current masks.

A pair can be computed directly between two images without retaining a [point track
ID](#point-track-id) through every intervening frame. A direct match between two images and a
continuous point track are different forms of correspondence.

Code: [`SharedFrameTracker.update`](../shared_image_tracking.py#L163) and
[`image_matching.py`](../image_matching.py).

## Usable and validated 3D pairs

**A usable 3D pair is an image match whose two pixels have suitable mask membership and depth
support, converted to two 3D positions.** Each position uses its image's camera intrinsics and
depth. The two positions are in their respective camera coordinate systems, in metres.
**Validated pairs** are the subset retained by the robust rigid-motion fit.

These are separate stages: a match can survive in 2D but lack usable depth; a usable 3D pair can
still disagree with the other points' motion. Neither stage proves that the pixels depict the same
physical surface point.

The shared path checks finite positive depth around both pixels, then the object's usable-surface
mask. That mask also excludes the configured band inside the segmentation boundary and applies
local depth-continuity, surface-normal, optional depth-range, and sampled-surface coverage checks.
Therefore, a rejected pair does not necessarily lack depth: specify which check failed.

Code: [`SharedFrameTracker.pairs`](../shared_image_tracking.py#L223),
[`measured_pairs`](../surface_correspondence.py#L76), and
[`robust_fit`](../surface_correspondence.py#L120).

## Segmentation mask

**A per-image set of pixels estimated to belong to the selected object.** SAM produces these
masks in live tracking; replay can use saved masks. A mask selects the visible object region;
it does not pair old and new pixels or specify the object's 3D rotation.

Mask membership and usable depth are separate: an object pixel with rejected depth does not
become background. Current geometry uses cleaned depth inside the mask. RGB, mask and the selected
depth map must belong to the same source frame and pixel coordinates. Selected depth can come from
the sensor or a model; model depth is an estimate, not an independent sensor measurement.
See the [frame inputs](../../../gui/static/flash_architecture.html#metric-depth).

Code: [`SharedFrameTracker.update`](../shared_image_tracking.py#L163) checks mask dimensions;
[`pairs`](../shared_image_tracking.py#L223) checks source and target mask membership.

## Object coordinates and pose

**Object coordinates are the fixed 3D coordinate system used to store the object's surface.
A pose is the rotation and translation mapping those coordinates into a specified destination
coordinate system.** The surface tracker returns the 4×4 transform `camera_from_reference`,
which maps object coordinates into the current camera coordinates, in metres.

For a stored point `p`, its predicted camera position is `R × p + t`. The coordinate origin stays
fixed when the surface grows; it need not be the object's physical centre. Here “reference
coordinates” means this coordinate system, not a particular saved image.

On fresh initialization, the stored points use the first observation's camera coordinates and
the initial transform is the identity matrix. The origin is therefore not an estimated object
centre. Converting a tracker pose for robot use requires the camera-to-base transform:
`base_from_reference = base_from_camera @ camera_from_reference`.
Camera coordinates, robot-base coordinates, and the fixed object coordinates are distinct.

Code: [`SurfacePatch.points`](../experimental_surface_session.py#L291) converts measured camera
points into reference coordinates; [`PoseKalmanFilter`](../surface_kalman.py#L71) uses a fixed
object point. [`GrowingSurfaceSession.update`](../experimental_surface_session.py#L767)
initializes the fixed coordinates; [`ObjectGroupAssociation.record_observation`](../object_group_association.py#L174)
composes the camera-to-base transform.
Design: [persistent coordinates](tracking_pipeline.md#what-persists-between-frames).

## Published pose estimate

**The record a tracker publishes for one object role: a 4×4 pose in robot-base coordinates, how
it was obtained, and which [RGB-D frame](#rgb-d-frame) it describes.** This is FLASH's
`PoseEstimate`, the handoff from tracking to action execution.

`evidence` is `observed` (measured in that frame), `predicted` (carried forward without a new
measurement) or `lost` (no pose; the record refuses one); see
[Tracking loss and recovery](#tracking-loss-and-recovery) for how the tracker decides.
`source_time_s` and `source_frame_id` name the frame the pose describes, and `available_time_s`
is when it was published, which can never be earlier. `translation_supported` and
`rotation_supported` mark which position axes (robot-base X, Y, Z) and which rotation axes (the
object's own X, Y, Z) the evidence constrains. A partial pose can serve a task that does not
depend on its unsupported axes.

A consumer uses an estimate only if it is not lost, was already published, is no older than the
consumer's age limit, is allowed to be a prediction, and supports every axis the task needs.

Code: [`PoseEstimate`](../pose.py#L20) and [`usable`](../pose.py#L82).

## Pose disagreement

**A comparison of two pose proposals for the same object in the same RGB-D frame, using only
measured depth and the [segmentation mask](#segmentation-mask).** Each proposal places the
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

## Stored surface / surface map

**The accumulated 3D surface measurements stored in [object
coordinates](#object-coordinates-and-pose).** New accepted views can add measurements of newly
visible parts. The current visible surface is separately measured from the current mask and usable
depth. Neither is a complete ground-truth model of the object.

A **surface patch** retains sampled camera measurements from one observation and the transform
placing them in object coordinates. The fitting map combines confirmed patches; pending additions
cannot influence fitting before confirmation. In fresh sessions, an accepted observation also
replaces overlapping visible map samples with current measurements and removes contradicted samples.
Hidden or unknown surface is not removed merely for being unseen. This working map can change while
the coordinate origin and original patch measurements remain fixed. Rejected poses and predictions
cannot refresh or grow the map. See [surface persistence and refresh](tracking_pipeline.md#what-persists-between-frames).

Code: [`SurfacePatch`](../experimental_surface_session.py#L282) and
[`SurfaceStore`](../experimental_surface_session.py#L295), and
[`RefreshingSurfaceStore`](../surface_refresh.py#L28).

## Local surface fitting

**Moving and rotating stored 3D surface points to agree with the object's currently measured 3D
surface points.** This is also called surface registration. The starting estimate has two distinct
parts: **position** (where the object is) and **orientation** (how it is rotated).

**Actual inputs to the 3D registration solver:** stored object surface points, current 3D points
sampled from usable depth inside the SAM mask, and a starting rotation and translation. These are
object surface points, not a point cloud of the entire scene. The surrounding depth image is
retained separately for checking proposed placements.

The fit uses the confirmed map as it stood before this frame. It cannot change that map to
make its own pose pass. Image matches and group motion supply the initial motion proposal.
The default small_gicp solver then fits the object geometry; the selectable `visible_icp` solver
can also include image/group point residuals. Neither can override reliable mask/depth contradictions. The fitted pose and any
subsequent Kalman-adjusted pose must both pass the geometry checks before the map can change.
See the [fitting and publication path](../../../gui/static/flash_architecture.html#surface-fit).

Selecting object points does not itself remove their camera-space position. For example, object
centres at camera X = 100 mm and X = 140 mm remain 40 mm apart after background points are removed.
Initializing the translation from those centres is a separate operation.

“Local” means the solver improves an initial estimate using spatially nearby 3D points and a limited
number of iterations, rather than searching every possible position and rotation. A wrong starting
position, a wrong starting rotation, or both can prevent a successful fit. A rotation can move
surface points away from their corresponding measured points even if the object's centre stays in
the same place.

**Translation initialization:** keep the proposed rotation, compare its translation with one aligning
the two surface means, and start one fit from whichever has lower geometry cost. Partial visibility
can shift the measured centre, so the original translation remains eligible. Acceptance still
requires the geometry checks.

Code: [`GrowingSurfaceSession.update`](../experimental_surface_session.py#L851) chooses the starting
pose; [`WholeSurfaceFit._solve`](../surface_grounding.py#L743) and
[`fit_once`](../surface_grounding.py#L757) run one bounded fit;
[`LibraryMatcher.refine`](../surface_registration.py#L202) supplies the correspondence-distance and
iteration limits.

## Object point anchor

**A remembered 3D position of a point on the object, measured relative to the object's fixed
coordinate system.** Its stored value is three coordinates in metres. The label “anchor” names a
code data structure; it is not an extra physical marker.

For example, consider a small visible spot on a cube:

1. In the RGB image the spot has a **2D pixel position**, such as `(u, v)`. The image tracker gives
   that track an integer ID.
2. Depth at that pixel and the camera calibration give a **3D position relative to the camera**.
3. Using the accepted object position and rotation, the code expresses that 3D point **relative to
   the object** and remembers it under the track's ID: `anchors[42] = (x, y, z)`.

The remembered value is 3D; the associated image track has 2D pixel positions. When the cube moves,
the spot's camera coordinates change, while its stored coordinates relative to the cube remain
fixed. Later measurements with that track ID can be associated with this remembered point. An ID
match remains a tracking hypothesis and can be wrong.

The entry can remain stored without a current measurement of the spot. That does not establish that
the spot is physically occluded or that its predicted current position is correct. Within the same
object reference, accepted observations can add new entries but do not move existing entries.
Entries are removed when their IDs no longer survive in the scene tracker or its groups.
This object-side record cannot change scene-group membership or motion.

Code: [`ObjectGroupAssociation.record_observation`](../object_group_association.py#L174) creates
entries and preserves existing coordinates; [`query`](../object_group_association.py#L50) uses them
and can add points through an established group relation. Group-local anchors in
`motion_groups._Group` use a different coordinate system.

## Motion group

**A collection of 3D points that the tracker assumes move together as one rigid body.** This means
one common rotation and translation explains their motion; their distances from one another stay
approximately constant. The group need not equal one physical object or one SAM mask.

For example, points on a tray and a cube resting on it can appear to move together. If the cube is
then lifted, that shared-motion assumption is no longer valid for the cube. Tracking the tray
successfully does not tell us where the lifted cube went.

The scene-group tracker receives timestamped point IDs and measured 3D positions in robot-base
coordinates, plus current camera calibration and reliable scene depth for checking
carried positions. It owns group membership, fixed group-local positions, and each group's transform
into base coordinates. It receives no object masks, fitted object poses, or Kalman state.
Mask-dependent sampling happens upstream in the shared image tracker. Object association only
queries the resulting group snapshot; accepting or rejecting an object pose cannot rewrite it.
See the [group architecture](../../../gui/static/flash_architecture.html#group-update).

The implementation retains some members without accepted group measurements and predicts their
positions from the accepted members. In `GroupView`, `measured` means both currently measured and
accepted as an inlier of that group's motion. A false flag can mean either a missing measurement
or a current measurement rejected by the group fit. Those cases must be distinguished when
diagnosing failures. Remembered object points can connect an object to such a group. Predicting a
member assumes it still shares the measured members' motion; it does not establish that the point
remained physically attached.

A carried 3D point is transformed into the current camera coordinates and projected
into the current image. Reliable scene depth farther than its current camera Z by
more than the depth tolerance contradicts that position. The group removes that
membership even without enough points to form another group. Nearer depth can
occlude the point; missing depth, uncertain boundaries, and offscreen positions
leave it unknown. These checks use no object mask or object pose. A surviving
image track at another measured position remains available to the scene tracker.
If rejection removes the last remembered link to an object, the object-side
association stops using that group's fixed relation.

Code: [`GroupView`](../motion_groups.py#L76) and
[`SceneMotionGroups.update`](../motion_groups.py#L579) own the scene representation;
[`SceneDepth`](../scene_depth.py) checks current camera depth;
[`ObjectGroupAssociation.query`](../object_group_association.py#L50) consumes it.

## Pose prediction / prior

**An estimate of the current pose before accepting the current object's surface measurement.** The
Kalman filter carries the previous pose through camera changes and any usable measured group motion.
A group can also supply a separate pose hint through the stored object-to-group relation.

With Kalman enabled, measured group motion advances the previous corrected object pose; it does
not replace it with the group's absolute pose hint. Without usable measured motion, prediction
keeps the previous world position, compensates for known camera motion, and increases uncertainty.
Whether a prediction is published depends on the configured support policy and any current
geometry checks. See [prediction and correction](tracking_pipeline.md#uncertainty-and-correction).

A prediction and a fit's [starting pose](#local-surface-fitting) need not be the same: the starting
pose can instead come from current image matches. Publishing a predicted pose does not mean the
object was measured this frame.

Code: [`PoseKalmanFilter.predict`](../surface_kalman.py#L113),
[`ObjectGroupAssociation.query`](../object_group_association.py#L129), and
[`GrowingSurfaceSession.update`](../experimental_surface_session.py#L851).

## Kalman pose update

**Combining the predicted pose with a new surface-fit pose using their estimated uncertainty.** The
output is called the fused pose. Rotation and position can receive different corrections according
to the information supplied by the measurements.

The filter belongs to the object-pose estimator, not the scene-group tracker. Its 6×6 covariance
describes uncertainty around one pose; it does not represent several alternative symmetric poses
or give a calibrated probability of correctness. A large difference from the prediction is not
itself grounds to reject a geometry-compatible fit. Fresh sessions retain the prior for weighted
correction. The optional `reset_on_geometry_conflict` setting instead permits a reset when
measured geometry contradicts the prior.

The fused pose can differ from both input poses, so it must pass the geometry checks again.
If it fails, the tracker restores the prior filter state. A prediction that passes the current
checks takes precedence. When the prediction also fails, the already checked surface-fit pose
becomes the published pose and the filter's starting pose for the next frame. This fitted-pose
fallback keeps the prediction's covariance unchanged and retains the previous accepted image
and its pose. It does not grow or refresh the stored surface. A subsequent ordinary accepted
correction can update the image and surface again. The legacy comparison setting
`allow_fitted_pose_fallback = false` suppresses this fallback.
These checks test compatibility with measured geometry, not unique or correct 6D pose.
When neither a fit nor a prediction passes the applicable checks, the result is lost.

Code: [`PoseKalmanFilter.correct`](../surface_kalman.py#L168),
[`_correct`](../surface_kalman.py#L181), and
[`accept_checked_fit`](../surface_kalman.py#L93);
[`GrowingSurfaceSession.update`](../experimental_surface_session.py#L962) checks the fused pose.

## Accepted observation / reference image

**A processed RGB-D image and mask whose pose the tracker accepted for its state update.** The
tracker retains that observation, its frame ID, intrinsics and pose for subsequent matching.
In the growing-surface tracking path, **reference image** means this last accepted image.
The design also calls this a **connected observation**. It is distinct from the initial image
that established the object's fixed coordinates and from individual retained surface patches.

The last accepted image may be older than the previous processed image. Accepting a newer image does
not redefine the fixed [object coordinates](#object-coordinates-and-pose). “Accepted” reports the
estimator's decision; it is not independent ground-truth verification.

Code: [`GrowingSurfaceSession._remember_observation`](../experimental_surface_session.py#L1070)
stores `accepted_frame`, `accepted_pose`, and `accepted_observation`.

## Tracking loss and recovery

**Recovery means obtaining an accepted current object pose after the tracker could not obtain one.**
Specify what was lost: old point tracks, the mask, usable depth, or an accepted pose. Those are
different failures.

An **observed pose** (`observed_candidate`) has passed the current object-observation checks.
An ordinary accepted correction advances the accepted image; the fitted-pose fallback described
under [Kalman pose update](#kalman-pose-update) retains that image and the stored surface.
A **predicted pose** (`predicted`) is available without a newly
accepted object observation. Publishing a prediction is distinct from recovering an observed pose.
`lost` means no pose is published, even if an internal estimate is retained. These statuses
describe estimator decisions, not independent pose accuracy.

Loss of point tracks does not imply loss of segmentation or depth. An object can still have
a usable current surface observation after all old point tracks disappear. Conversely, a
segmentation mask alone does not establish an accepted 3D pose.

Code: [`GrowingSurfaceSession.update`](../experimental_surface_session.py#L767),
[`_publish_prediction`](../experimental_surface_session.py#L1102), and
[`_result`](../experimental_surface_session.py#L1158).
Design: [initialization and frame sequence](tracking_pipeline.md#initialization-and-frame-sequence).
