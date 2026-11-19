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

- [Point track ID](#point-track-id)
- [Image match / 2D correspondence](#image-match--2d-correspondence)
- [Usable and validated 3D pairs](#usable-and-validated-3d-pairs)
- [Segmentation mask](#segmentation-mask)
- [Object coordinates and pose](#object-coordinates-and-pose)
- [Stored surface / surface map](#stored-surface--surface-map)
- [Local surface fitting](#local-surface-fitting)
- [Object point anchor](#object-point-anchor)
- [Motion group](#motion-group)
- [Pose prediction / prior](#pose-prediction--prior)
- [Kalman pose update](#kalman-pose-update)
- [Accepted observation / reference image](#accepted-observation--reference-image)
- [Tracking loss and recovery](#tracking-loss-and-recovery)

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
initializes the fixed coordinates; [`ObjectGroupAssociation.record_observation`](../object_group_association.py#L167)
composes the camera-to-base transform.
Design: [persistent coordinates](tracking_pipeline.md#what-persists-between-frames).

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
pose; [`WholeSurfaceFit._solve`](../surface_grounding.py#L784) and
[`fit_once`](../surface_grounding.py#L798) run one bounded fit;
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

Code: [`ObjectGroupAssociation.record_observation`](../object_group_association.py#L167) creates
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
coordinates. It owns group membership, fixed group-local positions, and each group's transform
into base coordinates. It receives no object masks, fitted object poses, or Kalman state.
Mask-dependent sampling happens upstream in the shared image tracker. Object association only
queries the resulting group snapshot; accepting or rejecting an object pose cannot rewrite it.
See the [group architecture](../../../gui/static/flash_architecture.html#group-update).

The implementation retains some unmeasured group points and predicts their positions from the
measured members. Remembered object points can connect an object to such a group. Predicting an
unmeasured member assumes it still shares the measured members' motion; it does not establish
that the point remained physically attached.

Code: [`GroupView`](../motion_groups.py#L76) and
[`SceneMotionGroups.update`](../motion_groups.py#L546) own the scene representation;
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

Code: [`PoseKalmanFilter.predict`](../surface_kalman.py#L93),
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
If it fails, the tracker restores the prior filter state and checks whether a prediction can
be published; otherwise it reports lost.

Code: [`PoseKalmanFilter.correct`](../surface_kalman.py#L148) and
[`_correct`](../surface_kalman.py#L161);
[`GrowingSurfaceSession.update`](../experimental_surface_session.py#L958) checks the fused pose.

## Accepted observation / reference image

**A processed RGB-D image and mask whose pose the tracker accepted for its state update.** The
tracker retains that observation, its frame ID, intrinsics and pose for subsequent matching.
In the growing-surface tracking path, **reference image** means this last accepted image.
The design also calls this a **connected observation**. It is distinct from the initial image
that established the object's fixed coordinates and from individual retained surface patches.

The last accepted image may be older than the previous processed image. Accepting a newer image does
not redefine the fixed [object coordinates](#object-coordinates-and-pose). “Accepted” reports the
estimator's decision; it is not independent ground-truth verification.

Code: [`GrowingSurfaceSession._remember_observation`](../experimental_surface_session.py#L1049)
stores `accepted_frame`, `accepted_pose`, and `accepted_observation`.

## Tracking loss and recovery

**Recovery means obtaining an accepted current object pose after the tracker could not obtain one.**
Specify what was lost: old point tracks, the mask, usable depth, or an accepted pose. Those are
different failures.

An **observed pose** (`observed_candidate`) has passed the current object-observation checks
and advances the accepted image. A **predicted pose** (`predicted`) is available without a newly
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
