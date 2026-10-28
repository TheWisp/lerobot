# When the opposite surface becomes visible

The live FLASH surface backend currently retains only its first measured RGB-D
surface. It transports that surface with UFM pixel motion, then adjusts the pose
to fit the current SAM mask and depth. It **does not add the newly exposed side**.
An asymmetric red hand rotated in the camera debugger exposed this limitation:
the fit visibly favored the old side. The debugger showed the resulting pose as
an unverified candidate; the same-frame surface p90 was about 7–8 mm, above the
configured 5 mm publication limit. This does not establish the hand's true 6D
pose. The archived short hand trace is stationary and only validates capture;
the rotation itself still needs to be recorded.

## Labeled reorientation control

To separate missing geometry from pose estimation, `flash_surface_multiview_oracle.py`
uses the 31-frame asymmetric YCBInEOAT cracker-box reorientation clip already
prepared under `~/.cache/flash-visible-benchmark/ycb-cracker_box_reorient-191`.
All methods use the same aligned RGB-D, supplied object masks and first-frame
surface. **Annotated poses register each earlier view into the map**; this is
an oracle ablation, not a live tracker. Each frame is scored before its own
observation is admitted, so no future surface enters its score.

| Surface available before each frame                                                                              | Median measured-surface p90, frames 1–30 | Final-frame p90 | Final map points |
| ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------: | --------------: | ---------------: |
| First view only                                                                                                  |                                  7.39 mm |        29.53 mm |            3,773 |
| Every earlier view, 3 mm voxel spacing                                                                           |                                  2.70 mm |         3.85 mm |           13,881 |
| Every earlier view, 5,000-point cap                                                                              |                                  3.21 mm |         4.08 mm |            4,984 |
| New oriented surfels seen in three earlier frames, at least 5 mm from known geometry or facing its opposite side |                                  5.24 mm |        11.02 mm |            4,139 |

The actual fixed-reference FLASH/UFM backend accepted only frames 0 and 1 of
this clip. By frame 30, its candidate transported first-frame points a median
35.4 mm from the annotated motion and differed by 8.4° in rotation. These are
relative-pose errors, not object-surface residuals.

More geometry alone does **not** make the current fit trustworthy. Giving the
fitter the annotated current pose as its starting proposal, its final-frame
surface-point error was 23.4 mm with only the first view, 17.9 mm with every
earlier view, and 16.6 mm with the repeated-view map. At frame 25 the all-view
fit was worse than the fixed-view fit (23.5 versus 14.6 mm). In each case the
local optimizer moved away from the annotated pose to lower its own objective;
its coarse PCA branch was not responsible. The annotations, masks and depth
are imperfect, but the test shows that a low visible-surface gap and a large
fused map do not certify a correct object frame.

A partial-reference ablation used the same local fitter and annotated
starting poses, but dropped the residual that requires every newly visible
depth point to match the first-view surface. It kept the terms for visible
reference points and their mask boundary. Each case scores six frames at
five-frame intervals, with the same surface-point error against annotation:

| Labeled object turn                              | Current first-view fit, median / last | Partial first-view fit, median / last | All earlier views, median / last |
| ------------------------------------------------ | ------------------------------------: | ------------------------------------: | -------------------------------: |
| [Cracker box](partial_reference_oracle.json)     |                        11.6 / 23.4 mm |                        18.0 / 24.7 mm |                    9.8 / 17.9 mm |
| [Sugar box](partial_reference_sugar_box.json)    |                        12.3 / 13.6 mm |                        22.8 / 25.7 mm |                     5.6 / 5.1 mm |
| [Mustard bottle](partial_reference_mustard.json) |                          9.5 / 7.8 mm |                         9.5 / 10.1 mm |                    7.0 / 12.0 mm |

On the cracker box at frame 25 the partial fit grew from the current fit's
14.6 mm error to 36.0 mm. The one-way fit avoids forcing an unseen backside
into the old map, but then permits drift along directions the partial view
does not constrain. It is not a safe standalone replacement. Run
`benchmarks/flash_surface_multiview_oracle.py CASE --fit-every 5 --output
OUTPUT.json` to reproduce each side-by-side measurement. This ablation starts
at annotated poses to test the fitting objective itself; it does not show
relocalization from an uncertain live prior.

`CandidateSurfaceMap` is therefore experimental and is **not connected to the
live pose output**. It keeps the initial reference and its landmarks fixed,
requires repeated accepted views before adding a new measured voxel, bounds
point count and retains source-frame provenance. It cannot itself determine
whether an input pose is correct or undo a wrongly admitted view. A runtime
version must check pose observability, current depth/mask support, free-space
contradictions, and performance before a view becomes authoritative. The
debugger now records bounded-duration aligned RGB, depth, masks and pose
decisions so the red-hand flip can be replayed through alternatives without
changing the live robot loop.

## Recorded red-hand turn

The debugger captured 527 aligned RGB-D/mask frames in
`~/.cache/huggingface/lerobot/flash_debug_traces/trace-20261025-181728-660815692`.
The hand was stationary at first, then moved and rotated between recorded frames
roughly 19520–19820. This capture does not clearly show a full 180° backside
view. The trace is local evidence, not committed image data.
The saved live backend returned `LOST` for several frames during the turn; the
first-view fit resumed later without having added the other side.

`flash_surface_motion_trace.py` re-anchors the reference at trace frame 19476
and compares three _candidate_ pose paths over the subsequent 42 processed
frames. The uncorrected consecutive UFM path turns roughly 90° by frame 19703;
the existing fixed-surface path turns roughly 38° from the same trace anchor.
This does **not** establish which rotation is physically right: a direct
first-to-frame-19703 UFM match suggests yet another orientation, and smooth
red plastic offers few persistent material features. The three answers
disagree despite each having low residuals on some of its own evidence.

Forcing pairwise matches to remain part of the surface fit produces a fourth,
intermediate path. An initial map prototype admitted any repeatedly seen
voxel farther than the _sampling spacing_ from known points. It appeared to
explain withheld red-hand frames to 1.5–2.8 mm p90, but the same rule also
admitted 64 spurious points while the hand was stationary. That was not a
valid model-completion result. Separating sample spacing from a 5 mm novelty
distance (the existing depth-fit tolerance) admits **zero** points in the
12-frame stationary trace. The constrained-pose map then admits only 17 new
points by frame 19687. Its withheld-frame p90 values at 19703, 19720, 19736
and 19783 are 3.6, 3.4, 3.6 and 3.5 mm, compared with 4.5, 4.3, 4.2 and
3.1 mm for the fixed-reference path. These are **surface-coverage** numbers,
not proof of material-point identity or orientation. The map can also store
opposite-facing surfels at similar XYZ locations, but the recorded turn did
not expose enough of a distinct back side to test that mechanism.

The labeled cracker-box control catches that exact risk. A raw chain of
consecutive UFM motion ends 6.7° and 13.8 mm from the annotated pose. Adding
the same motion-constrained _fixed-reference_ fit worsens its final point error
to 31.2 mm; fitting to the conservative growing map still ends at 25.2 mm and 9.5°.
Unconditional local ICP likewise worsens the raw chain's final point error
to 28.3 mm. The red-hand result is therefore insufficient to turn any of
these experimental paths on for live control.

The implementation next needs a way to register new visible patches without
letting their own proposed pose become their only evidence. A measured rigid
attachment or scene group can supply independent motion when available. For
vision alone, compare multiple overlapping views jointly, retain competing
registrations through ambiguous rotations, and publish a corrected object
frame only after cross-view checks resolve them. A low mask/depth residual by
itself is not such a check.

The recorded gamepad depth fixture adds a direct warning for that last point.
At held frame 430, the candidate's measured surface is only 2.35 mm p90 from
the saved reference, but that same reference projects 124 mm p90 away from
reliable depth pixels on the object. Adding measured views from held frames 290
and 300 does not remove the contradiction. These are checks of a recorded pose
candidate, not independently labeled pose errors. New prepared bundles now
record both directions of agreement for their first and combined registered
views in source-frame proposal evidence; neither number changes control yet.

Reproduce the labeled ablation with `PYTHONPATH=src python
benchmarks/flash_surface_multiview_oracle.py CASE --output OUTPUT.json` and
`--admission confirmed` for the three-view variant. The current backend run
used the same `VisionPipelineConfig` defaults and supplied masks, with UFM from
the rig's isolated tracking environment. The local trace can be replayed with
`benchmarks/flash_replay_debug_trace.py TRACE --output OUTPUT.json`; its first
saved frame establishes a new local reference unless recording began at object
selection. No supplied mesh or object-specific symmetry rule enters these
experiments.

For the recorded turn, run `benchmarks/flash_surface_motion_trace.py TRACE
--first 19476 --last 19821 --motion-constrained-fit --output OUTPUT.json`.
For the labeled control, run `benchmarks/flash_surface_motion_case.py CASE
--output OUTPUT.json`. Both scripts use the UFM-capable isolated tracking
environment. Their maps are diagnostic hypotheses, not trusted pose outputs.

## Held dowel: test a later view against earlier frames

The prepared insertion demo still has one saved dowel surface, measured before
the grasp. Its held-object correction accepts small changes to the
gripper-to-object relation immediately. In this preparation, the accepted
relation at frame 291 and the one at frame 475 differ by 12.8 mm at the saved
surface centre and 11.7° in rotation. The later frame shows more of the dowel.

`benchmarks/flash_held_surface_holdout.py` registered the clean RGB-D surface
from frame 475 using **the prepared pose at that frame**, then held each chosen
gripper-to-object relation fixed while replaying the recorded arm motion. It
scored frames 300–429, every third frame (44 frames); none supplied the new
surface. The number below is the 90th-percentile distance from measured clean
depth points to the projected surface, summarized across those frames.

| Fixed attachment   | Surface model       |   Median | 90th percentile |
| ------------------ | ------------------- | -------: | --------------: |
| Saved at frame 291 | Initial view        | 13.66 mm |        14.76 mm |
| Saved at frame 475 | Initial view        |  8.12 mm |         9.49 mm |
| Saved at frame 475 | Initial + frame 475 |  3.87 mm |         7.57 mm |

The first improvement comes from changing the attachment; the second comes
from the newly visible surface. This is independent _frame_ evidence for those
two effects, but not an independently measured 6D pose. The later attachment
could include real slip, and a partial cylinder may permit more than one pose.
The reverse depth residual remains high on some frames because projected
points can be hidden; low forward distance alone is not enough to publish a
pose. Neither the model nor the live control rule has been changed by this
audit. It motivates an interval-level attachment check and an admission test
for new views before either is used for control.

The subsequent shared attachment rule now checks the configured clean-depth
support for **every** visual correction, including small changes. Repreparing
the insertion demo from its unchanged pose trace rejects the weak corrections
at frames 291 and 475; the latter leaves the attachment conflicted instead of
silently changing the carried pose. This flags uncertainty rather than fixing
the dowel's missing geometry. A fresh preparation of the separate gamepad
stacking demo made the same 20 visual corrections and produced numerically
identical carried skill samples (32 compared). The small-correction regression
test fails against the previous implementation and passes with the check.
These results do not establish behavior on every object or a
physical insertion success.
