# Mask/depth-only tracker: full dowel demo

**Result: this first surface-alignment prototype is not a reliable replacement
for Point2Pose.** Keeping the initial geometry loses tracking through the grasp
occlusion. Replacing geometry every frame follows the currently visible patch
but lets the carried landmarks drift. Starting again at frame 280 does not
remove the drift. No live tracker, dataset pose, GUI setting or robot control
was changed by this experiment.

[Full 713-frame video, 1x](full_demo_comparison.mp4). All three panels use the
same camera frame. Top: preserve the original surface, with depth cleanup.
Middle: replace the map with each accepted observation. Bottom: depth cleanup
and a **new** initialization at frame 280; its landmarks are not carried through
the earlier occlusion. Cyan dots are the fitted map, yellow A/B/C are fixed
landmarks, and red means a pose was withheld and the last pose is shown.
The displayed map is an x-ray projection, including currently hidden samples.
Neither a cyan cloud nor the logged `surface_supported` state certifies pose.

## What was implemented

Input is aligned recorded RGB-D plus freshly computed causal SAM3 masks for all
713 frames of `~/.cache/huggingface/lerobot/demos/insertion/recording`.
RGB is used by SAM3, not by the geometric fitter. No P2P, TrackEverything,
gripper poses, CAD, cylinder assumption, future image or manual pose input feeds
the tracker. The object name is only SAM3's text prompt.

1. Reject missing depth, a two-pixel inner mask boundary and pixels neighboring
   a depth hole or a depth span above 10 mm. Keep the original measured XYZ;
   smoothing is used only for surface normals.
2. Sample the remaining surface at 3 mm spacing. Withhold alternating spatial
   tiles from fitting for a residual diagnostic. These pixels are correlated
   with fitting pixels and are **not ground truth**.
3. Find nearest surface samples and minimize point-to-plane distance, plus a
   small point-to-point stabilizer. Correspondences are recomputed during
   optimization; they are not persistent physical point identities.
4. Project the pre-update map into the current depth image. Seeing a farther
   surface where the map predicts an object is contradictory; a closer surface
   can be an occluder and is unknown. Publish/fuse only with at least six fitting
   pairs and more supported than contradictory rays. This deliberately simple
   check proved **insufficient**, not a validated confidence rule.
5. Growing-map variants keep existing coordinates fixed and admit new patches
   only after three accepted observations in a local voxel. The previous-frame
   variant instead replaces its map. The local coordinate origin and A/B/C are
   chosen once at initialization and never recentered or reselected.

This differs from our configured P2P baseline's learned BootsTAPIR point
correspondences, lifted by depth and registered by an outlier-rejecting rigid
fit, with a separate TSDF reconstruction. Both systems estimate rigid motion;
this experiment obtains its motion correspondences from nearby **surface
geometry** rather than tracked image point identity. It is ordinary geometric
ICP, not a new trained model or a claim of a new tracking algorithm.

## Depth contamination experiment

At frame 280, five of 178 sampled points formed detached patches at
431–441 mm camera depth, while the main 173-point surface lay at 331–353 mm.
One detached point became the farthest landmark A. These points were inside
the eroded SAM mask and in locally smooth depth regions: an edge-margin and
depth-gradient check alone did not remove them.

The `connected*` ablation keeps the largest connected surface patch, joining
samples within 10 mm, and withholds the others. It removes those five samples.
This rule can also discard legitimate separated visible patches behind an
occluder, and cannot reject contamination that connects smoothly to the object.
It is an explicit study option, not an object-membership guarantee.

## Measured results

| Variant                       | Frames | Initialized / supported / withheld | Outcome                                              |
| ----------------------------- | -----: | ---------------------------------- | ---------------------------------------------------- |
| Keep original surface         |    713 | 1 / 212 / 500                      | No accepted poses from frame 280 onward              |
| Replace surface every frame   |    713 | 1 / 659 / 53                       | Visible patch fits; fixed landmarks leave the object |
| Restart at 280, keep surface  |    433 | 1 / 432 / 0                        | Contaminated seed landmark and continued drift       |
| Depth cleanup, restart at 280 |    433 | 1 / 430 / 2                        | Clean seed; motion still drifts                      |
| Depth cleanup, full run       |    713 | 1 / 211 / 501                      | Still loses tracking through grasp occlusion         |

Withheld combines `pose_rejected` and `no_geometry`. These counts describe
decisions by this prototype, **not counts of correct poses**.

For rotation frames 280–399, we compared transport of the **same three clean
initial landmarks** with the recorded gripper motion, separately anchored at
280 for each tracker. The median of the per-frame mean landmark discrepancy
was 25.7 mm for previous-frame replacement, 22.7 mm for the unfiltered restart,
and 22.1 mm for the cleaned restart. Thus removing detached points fixes a real
input error but does not materially solve registration drift on this interval.
The gripper is a rigid-grasp approximation; slip, calibration and timestamp
errors contribute. It is not independent pose ground truth.

The original restart's own A/B/C discrepancy was about 60 mm, inflated by its
bad depth landmark. Do not present that against 22.1 mm as a controlled tracking
improvement: those are different landmarks. `gripper_drift.csv` uses common
clean landmarks instead. The video deliberately shows each method's own
initialized landmarks, including the failed previous-frame transport.

Small residuals are misleading here. Before cleanup, the restarted map's
median per-frame held-out p90 surface distance over 280–399 was about 3.5 mm,
despite the large landmark drift. A partial smooth patch can fit locally while
the accumulated object coordinates move incorrectly. Updating the map with
those transforms can then preserve the error. This experiment does not isolate
the remaining contributions of weak geometric constraints, noisy depth/normals,
occlusion, solver local minima and map contamination.

CPU geometric computation took about 35–37 ms median and 38–39 ms p95 per
processed frame, excluding mask inference and disk loading. About 34 ms was
full-image preprocessing; fitting itself was roughly 0.4–1.8 ms median,
depending on how many frames reached a useful fit. SAM3 warm inference was
19.8 ms median / 21.2 ms p95, with a separate 1.79 s cold first call. Disk input
was another roughly 4 ms. These are sequential offline measurements, not live
end-to-end rates or a demonstrated 30 Hz tracker. The video plays at recorded
speed regardless of processing time.

## What this establishes

Mask/depth surface tracking is implementable without learned point tracking,
but mask membership and a small local fitting error do not establish correct
landmark transport. Boundary cleanup, disconnected-depth rejection, motion
estimation, and preservation of map geometry are separate problems. A credible
next tracker must check its older geometry without allowing newly fused points
to overwhelm that check, expose unconstrained motion, and explicitly recover
after losing overlap. Those mechanisms are **not implemented here**. This
experiment does not show that geometric tracking in general cannot work.

## Reproduce and inspect

`generate_masks.py` makes a causal full-frame pass with text prompt
`wooden dowel`, resolution 504. The cache is
`~/.cache/flash-mask-depth-experiment/sam3-full`; mask hashes and inference
timings are archived in `results/`. `run_surface_tracker.py` accepts
`--recording`, `--masks`, `--config`, `--output`, `--mode previous|growing`, and
optional `--start`/`--stop`. Output directories must be new.

`config.json` records the baseline settings; `connected_surface_config.json`
enables the connected-patch ablation. These are provisional study settings,
not calibrated accuracy bounds. Margins trade boundary rejection for coverage;
depth thresholds trade sensor tolerance for contamination; matching radii
define local search range; voxel size trades detail for cost; admission counts
trade delay for map contamination. None are derived from the dowel's shape.
The first three result snapshots predate hoisting the 3-pixel depth neighborhood,
5-pixel bilateral diameter, sigma 2 and 4-pixel validation tiles into JSON;
the current baseline config uses exactly those original values.

`collect_results.py` archives traces and metrics and evaluates against
`../flash-tracker-comparison/pose_traces.npz` **after** tracking. The latter's
gripper poses and calibration are evaluation-only; none of its P2P poses or
reference shaft geometry are used. `render_comparison.py` renders all 713
images, with no temporal cuts; it crops the camera consistently for phone
legibility and does not interpolate poses. The one restart is labelled.

Validation: four tests check recovery of known synthetic rigid motion, correct
sign of a depth/occluder check, boundary/hole/depth-step removal without changing
good measured depth, and exclusion of a detached depth island. These verify
specific mechanics, not real-scene tracking quality. The real full-demo failures
above remain the result. H.264/yuv420p video has 713 frames, 23.78 seconds and
fast-start metadata; selected frames were inspected visually.
