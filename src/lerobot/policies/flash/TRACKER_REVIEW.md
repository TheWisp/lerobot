# Offline demo tracker review

Open **Run → FLASH → Demos**, play a prepared demo, then choose **Edit demo → Trackers**.
Select a tracker (all annotated objects by default) and choose **Process demo and preview**. Processing uses the
selected version's per-object reference masks and the recorded RGB-D frames. It does not connect to
the robot. The result appears as a pink outline alongside the saved skill pose; hovering
the candidate card isolates that outline.

**Apply as new demo version** replaces the processed objects’ reference/held poses at
covered skill samples and opens the resulting version. Missing estimates retain their
previous pose and evidence. The report records changed and preserved sample counts.
The previous prepared version and raw episode remain available. No smoothing or
interpolation fabricates observations between sparse tracker outputs.

## Local runners

Runner installations live in `~/.config/lerobot/flash_trackers.json`, or the file named
by `LEROBOT_FLASH_TRACKERS_CONFIG`. The GUI only offers processing when a runner is
configured. Saved NPZ results can also be imported without installing that model.

TrackEverything has a built-in adapter:

```json
{
  "trackeverything": {
    "python": "/absolute/path/to/environment/bin/python",
    "repository": "/absolute/path/to/trackeverything",
    "checkpoint": "/absolute/path/to/model.pth",
    "query_count": 128,
    "mode": "dense",
    "timeout_s": 3600
  }
}
```

The [upstream integration guide](https://github.com/ayushjain1144/trackeverything#inference-demopy--demo_hfpy)
uses dense mode and `output.tracks()` to expose persistent 3D scene points, their birth
frames and the pixel-to-track map. It documents sparse mode as an optional way to track
specified query points; query points are not merged with scene points and may retain
higher resolution. The model outputs point trajectories, **not object identities or
6D object poses**. FLASH must supply segmentation and fit or otherwise infer poses.

This adapter presently fits a rigid pose independently per frame from points sampled
inside one reference mask per object. Dense mode is the default because it also runs
the model's persistent scene representation while preserving those high-resolution
queries. It does **not** use `output.tracks()` for pose fitting or associate newly
born scene points with an object. Thus dense mode does not by itself address a rotating
object whose original visible surface disappears. Sparse mode remains available for
comparison. The raw per-query tracks and visibility logits are retained for diagnosis.

On the evaluated 613-frame insertion demo, switching this same query fit from sparse
to dense context took 27.1 s versus 27.9 s and improved median held-dowel RGB-D
surface discrepancy over frames 400–500 from about 27 mm to 18 mm. The dowel fit still
had only 18 inliers from a median 83.5 visible queries, and the native scene output
represented its reference mask with just 9 distinct tracks after deduplication at the
upstream default 20 mm voxel size. The candidate is therefore still an inaccurate
pose estimate, not a corrected skill. Use segmentation across later views to associate
new points with an object, then check temporal rigidity and RGB-D surface agreement
before accepting a pose. Reducing voxel size may retain more scene detail but costs
memory/compute; it should be measured rather than assumed to fix correspondence.
The review card reports fits whose frame-to-frame pose motion disagrees with the
motion of tracked points that supported the previous fit. This uses the saved
registration tolerance rather than a fixed speed limit. It is a warning, not
proof that other frames are accurate.
Previewing a TrackEverything candidate also loads its retained raw query tracks
into the same demo replay. **Points** toggles them; the adjacent slider changes
the length of their visible trails. Cyan, green, and subsequent colors distinguish
objects, while each dot retains its point identity across frames. Filled dots
agree with that frame's rigid fit within its saved registration tolerance; hollow
dots disagree. Click a dot to isolate its whole visible trajectory and show its
fit residual. This layer is projected directly from TrackEverything's 3D output;
it does not use the fitted pose to position points. Occluded points are absent,
and trails do not connect across visibility gaps. Fit agreement alone is not
evidence that a track still belongs to the object.

The independently generated SAM3 masks and aligned RGB-D depths available for
this insertion experiment provide a diagnostic check, not ground truth. At camera
frame 454, 87 dowel queries are marked visible by TrackEverything; 37 land inside
the SAM3 dowel mask and within 10 mm of measured depth. Only 3 of the rigid fit's
17 inliers satisfy both checks. Rigid registration of just the 37 supported
tracks cannot find the configured 12 mutually consistent correspondences at the
5 mm fit tolerance. Thus the bad outline is partly a fitter choosing a spurious
cluster, but the available point identities are also too inconsistent to prove
a reliable 6D pose at that frame. The 10 mm depth band is a generous offline
diagnostic, not a new pose acceptance threshold.

Across frames 400–500, only 15% of the dowel fit's point supporters pass that
same mask-and-depth check, versus 75% for the ring under the same camera and
test. Using 5 or 20 mm depth bands gives 8% or 26% for the dowel, versus 56%
or 80% for the ring. This contrast persists across the diagnostic band choice;
it is not a threshold-tuned conclusion. Of all model-visible queries, 36% for
the dowel and 73% for the ring pass the 10 mm mask-and-depth check. The dowel
fit selects an especially poor subset of already mixed raw tracks.

Temporal smoothness alone gives the opposite impression. For dowel queries
visible at least ten times during held frames 400–500, the median point's median
deviation in the recorded gripper frame is only 2.4 mm (99 such tracks). That
supports stable _motion_ for many tracks, but it does not prove they stayed on
the dowel: the gripper is rigid and moves with it. The image/depth check above
is needed to catch stable tracks attached to the wrong visible surface.
Rigid fitting uses the selected demo's registration threshold, minimum inliers and
trial count. TrackEverything receives measured depth and camera intrinsics and runs
with future context within temporal windows. This is an offline demo extractor; this
integration does not make it a causal live perception source. The current adapter
requires one appearance reference per object and the existing fixed-camera RGB-D recording layout.

All selected objects share one TrackEverything inference call, including objects seeded at
different reference frames. Each object gets its own rigid fit. It tracks a fixed reference
query set; adding points on newly revealed surfaces is not implemented in the
FLASH preparation or live paths.

### Multi-view visible-surface experiment (2026-10-21)

`benchmarks/flash_expand_trackeverything_points.py` is an offline experiment
that starts from a saved query trace. It chooses a later SAM3-mask view with the
most depth-backed pixels not covered by existing mask-and-depth-supported
tracks, seeds additional TrackEverything queries there, and reruns the model
with all old and new queries. The source frame is selected from evidence, not
from an object-specific timestamp. The full raw result and query birth frames
are saved, but the coverage report counts only model-visible points that also
project inside that frame's SAM3 mask and agree with its measured depth. The
10 mm depth band is the same configurable diagnostic used above; hidden-point
predictions are deliberately excluded. This does not fit or apply an object pose.

On the insertion demo, a first dowel pass seeded 111 queries at frame 361 and a
second pass seeded 88 at frame 409. At frame 454, the supported visible dowel
points increased from 93 after one pass to 141 after two. At frame 500 the
count increased from 101 to 141. At the configured query spacing, coverage of
the visible SAM3 mask rose from 77% to 93% at frame 454 and from 77% to 91%
at frame 500. Both views visibly populate parts of the dowel missed by the
single initial view. The results, a six-frame contact sheet, and a 30 fps video are in the local
demo dataset under `flash/v1/tracker-runs/trackeverything-expanded-20261021-2/`.

More dots do not imply stable identities. Among points that still pass the mask
and depth check, the second dowel cohort's median change in pairwise 3D
separation from its birth frame is 6.8 mm at frame 454 and 11.1 mm at frame 500. A corresponding ring expansion has 0.9 mm and 1.1 mm at those frames.
This makes visible-surface expansion promising, especially for the target,
but dowel correspondence drift remains measurable even after rejecting points
outside the observed surface. Mask and depth agreement validate where a point
is now, not whether it retained the same physical identity. These outputs are
an offline research result, not a revised prepared skill or live tracker.

There is still a useful co-moving subset. With the prepared demo's 5 mm rigid
registration tolerance and 100 trials, at frame 454 the frame-361 dowel cohort
has 42 rigid-consistent tracks among its 58 mask/depth-supported tracks; the
frame-409 cohort has 28 among 47. At frame 550 those counts are 31/56 and
21/47. The ring cohort retains 74/74 at frame 454 and 71/75 at frame 550.

The Run → FLASH demo editor now exposes bounded growth for TrackEverything:
maximum points per object, maximum additions per newly exposed view, and the
depth-agreement tolerance. Processing segments the selected demo's full RGB
sequence with the saved object prompts in a separate process, tracks all query
cohorts together, and saves query birth frames and measured visible-surface
support for replay. The pink pose candidate remains the original-query fit;
the new points are review evidence, not silently promoted to planner pose.
A full two-object pass on the 613-frame insertion demo with a 320-point total
cap, 96-point per-view cap and 10 mm depth tolerance finished in 180 s. It
added four later cohorts per object without exceeding either cap.

Can those new points improve pose? A preliminary rigid fit anchored each later
cohort to the original pose at its birth view. On 38 sampled dowel frames,
held-out new-track median residual fell from 29.4 to 4.5 mm. Yet the fitted
orientation changed by 55.9° on median, the estimated object center by 20.5 mm,
and the original reference surface's mask-and-depth support improved on 24
frames but worsened on 13. This is not proof of a better 6D pose: birth anchors
inherit earlier pose error and smooth dowel surfaces permit coherent track
drift. A revised pose should be accepted only after independent surface,
free-space, temporal and task-point checks, with symmetry handled explicitly.
These are within-cohort checks anchored at each cohort's birth frame; they do
not establish alignment between cohorts or prove a stable relative dowel-ring
cloud. A next experiment must link overlapping cohorts and reject drifting
identities before using their relative motion for control.

### Adjacent-frame fits across the insertion demo (2026-10-22)

The [full-demo playback](../../../../docs/proofs/flash-cross-episode/te-demo-map/full-demo-adjacent-frames.mp4)
fits rigid motion between adjacent mask/depth-supported TrackEverything
points for **both** labeled objects. The saved run covers frames 100–712 of
the 713-frame demo; the first 100 RGB frames have no saved point tracks.
The ring has one connected visual-motion segment, 100–712. The dowel has
three, 100–172, 217–259 and 267–712. Frames 173–216 have no supported
dowel pose, and 260–266 cannot be connected to the next segment by an
adjacent fit. Reseeding an A/B/C display frame after either gap does not
establish how that frame relates to the prior one. The
[per-segment evidence](../../../../docs/proofs/flash-cross-episode/te-demo-map/full-demo-adjacent-frames.json)
records anchor IDs, fit support and raw-track disagreements.

This clarifies the earlier visually stable dense point overlay. On the same
saved tracks and original anchor IDs, replacing distant-reference refits with
adjacent fits reduced the 95th-percentile inferred anchor-position step
from 5.71 to 2.29 mm and direction step from 6.92° to 1.80°. Those numbers
measure smoothness, not accuracy. A raw dowel ID still moves onto the ring,
and the better selected triplet's maximum pairwise-distance change is 2.4 mm
at the 90th percentile of jointly supported frames. Point identity, inferred
object pose and physical task-point accuracy need separate checks.

Keep a stable abstract object frame within each **validated connected**
segment while adding newly visible point tracks to its bank. Do not require
three original surface IDs to stay visible; do not reset the object frame
merely because a manipulation stage changes. An occlusion or contradictory
co-motion may split the graph. A gripper-attachment prediction or independent
geometry registration can propose a cross-gap transform, but its provenance
and uncertainty must be checked before joining segments or extracting one
continuous demo relation. The saved TrackEverything pass took 151 s for 613
frames and uses future context. It is suitable for offline preparation and
shadow evaluation, not a drop-in causal source for the 30 Hz run engine.

Other trackers register an argv template under their tracker ID:

```json
{
  "point2pose": {
    "command": [
      "/absolute/path/to/python",
      "/absolute/path/to/adapter.py",
      "--recording",
      "{recording}",
      "--reference-mask",
      "{reference_mask}",
      "--first-frame",
      "{first_frame}",
      "--last-frame",
      "{last_frame}",
      "--output",
      "{output}"
    ]
  }
}
```

Available substitutions are `dataset_root`, `recording`, `bundle_path`, `reference_mask`,
`role`, `first_frame`, `last_frame`, `output`, and `run_dir`. Arguments are passed without
a shell. Single-object external adapters run once per selected object, sequentially,
then publish one candidate containing all results. Each run has its own dataset sidecar directory and log. The GUI offloads the
subprocess wait to a dedicated executor, separate from replay reads and robot control.
The configurable timeout bounds an offline job; closing the editor does not cancel it.

Point2Pose's existing experiment runner accepts arbitrary role names with `--initial-mask`
and an exact `--output-file`, so it can be registered directly. The registry also accepts
BundleSDF, TAPNext++, CoTracker3, LiteTracker, PANY and SAM3/DINO outputs. Those entries
do not imply that a model environment or conversion adapter has been installed.

## Pose trace contract

The output NPZ contains integer `frames[N]`, rigid `poses[N,4,4]`, and boolean `lost[N]`.
Frame indices refer to the original camera recording. The reference frame must have an
available pose. Transforms map reference-camera coordinates into the current camera;
the importer normalizes the reference self-pose and converts into the prepared object's
calibrated base frame. An object-to-camera output from another tracker must be converted
to this relative convention by its adapter before import.

The candidate manifest records its source bundle and trace hashes. A candidate cannot
be applied to a different or changed bundle. Preview coverage and rigid-fit support are not accuracy scores:
even a high-coverage pose trace may contradict the image/depth evidence.

## TrackEverything access

The [Hugging Face model page](https://huggingface.co/katefgroup/TrackEverything) gates
the checkpoint behind accepting its conditions. Download with the account that accepted
them. Code and weights are published under CC BY-NC 4.0; accepting the download gate
does not grant commercial rights. Separate permission must come from the rights holder;
the [repository license](https://github.com/ayushjain1144/trackeverything/blob/main/LICENSE)
names Meta Platforms and affiliates. No commercial-license application form is provided
in those materials.
