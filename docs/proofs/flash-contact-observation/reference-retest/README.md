# Dish, clock and mug: why poses fail, and what a slower search changes

Offline audit at `4d364d228`, 2026-10-29. **No production tracker or saved demo was changed.**
The shared pipeline was rerun with explicit alternate configurations and unchanged masks.
[Results](results.json) retain the measurements; videos and large traces stay in the local cache.

## Dish: two rules leave only a small strip

This is Caryx episode **90**, not the earlier shallow bowl in episode 110.
The clicked SAM mask follows the dish for all 245 frames. In frame 0:

- 2,193 masked pixels have depth; removing the inner boundary leaves 1,611.
- The rule rejecting a depth span over 10 mm within 3×3 pixels removes 1,295 more.
- Those rejected neighbourhoods have a median local plane RMS error of **0.205 mm**;
  **99.7% are below 1 mm**. Smooth sloping measurements are being rejected, not
  only discontinuities. This measures local smoothness, not true sensor accuracy.
- Sampling leaves 252 points; keeping one connected patch reduces these to **112**,
  on a narrow rim strip. Frame 1 has 19 matching image/depth points, but the
  proposed pose is rejected: its remaining patch is narrow and the full measured
  surface disagrees. The group fallback remains available through frame 55 and
  stops updating at frame 56 after its support shrinks from 188 to 12 points.

Keeping all patches removes the artificial same-frame discrepancy, but still
leaves the steep-surface rejection. Disabling both restrictions restores broad
coverage: frame 1 has 1,455 surface points and 203 motion inliers. Its measured
surface-to-saved-surface p90 distance is 4.1 mm, but projecting the saved surface
back to the depth image gives a 10.0 mm p90 depth difference, above the unchanged
5 mm limit. Later differences also exceed the limit. The follow-up below separates rejection, sampling and fitting effects; sensor
error versus incorrect pixel correspondence is not completely separated.

## Clock: better candidates exist, but surface score can choose wrongly

The complete labeled Laval clock clip has 203 frames. Labels and the evaluation
mesh are read **only after fitting** by `score_clock.py`. Each error below is
relative to the first frame; point error is the median displacement of 256
labeled-model points. These are raw tracking proposals, including rejected ones,
not accepted robot commands.

| Full-clip method                                                                | Median point error | Median orientation error |
| ------------------------------------------------------------------------------- | -----------------: | -----------------------: |
| Recorded original pipeline                                                      |            85.3 mm |                   123.8° |
| Keep all patches, otherwise use the same first-view tracker                     |            36.5 mm |                    31.5° |
| Keep all patches; fit each previous surface to the next and accumulate movement |           148.4 mm |                   100.9° |
| Accumulate neighbouring-frame RGB-D motion without surface correction           |            45.3 mm |                    50.3° |

All four still fail as complete pose tracks. Removing patch selection helps;
blindly replacing the first view with the previous view does not solve drift.

For a slower matching diagnostic, at **13 target frames** we tried the first,
previous, two-frames-back, four-frames-back, and an earlier view with many points.
Each pair used image/depth matches, a continuity start and all four PCA starts;
refinement had up to 200 evaluations. There were **61 source/target pairs and
413 candidates**, including unrefined image/depth fits. Every target had at least
one candidate within **3.5 mm / 3.2° of its labeled pairwise motion**. This is an
_evaluation-only best candidate_: the tracker cannot consult labels to choose it.
It also assumes the source frame is already registered correctly; it does not
establish a correct full trajectory.

At frame 150, the best surface-score choice matches the front seen in frame 0
to the currently visible back: **93.5 mm / 179.8° error**. The image/depth match
from frame 149 has **1.6 mm / 1.5° error**, but a worse surface score. The optimizer
also worsens this neighbouring-frame match to 7.3 mm / 4.1° while improving its
own score. This identifies a ranking/objective problem in addition to drift;
more search alone does not resolve it. The search is bounded, not proof of a
mathematical best fit or of an unavoidable failure of multi-view tracking.

## Mug: why the one-patch rule exists

It originated in the [dowel contamination experiment](../../flash-mask-depth-tracker/README.md#depth-contamination-experiment):
five depth points lay about 8–11 cm behind the main observed surface, despite
being inside the SAM mask. Keeping the largest connected patch removed them.
The experiment documented that this could discard real visible pieces. The
live debugger inherited the setting at `f97e395b9`.

It is **not multiple-object selection**. A single object can have separated
visible depth patches. For the mug, it removes 239 of 666 sampled points on one
side, while the later surface check still uses those points. Using this as a
general object-membership rule is incorrect.

Disabling it removes the artificial initial disagreement and reduces the next
frame's surface p90 mismatch from **67.9 to 7.9 mm**. That still exceeds the saved
5 mm limit; the complete mug pass remains 1 observed / 147 predicted / 241 lost.
Neither clip has independent mug/dish pose truth. Raising the limit or retaining
all points cannot be called a complete fix from these results.

| Full pipeline, unchanged acceptance limits          | Observed / predicted / lost |
| --------------------------------------------------- | --------------------------- |
| Clicked dish, original                              | 1 / 55 / 189                |
| Dish, all patches                                   | 1 / 7 / 237                 |
| Dish, all patches and depth-span rejection disabled | 1 / 55 / 189                |
| Mug, original                                       | 1 / 147 / 241               |
| Mug, all patches                                    | 1 / 147 / 241               |

The dish's fallback can also change because choosing a different initial cloud
changes its object frame and the surrounding points used to establish support.
These counts measure availability, not correctness.

## Reproduce

Use the repository root, `PYTHONPATH=src`, and the inference environment
`~/.cache/flash-live-tracking/.venv/bin/python`. Local datasets and configs are
named explicitly in the scripts; their masks and source recordings are read-only.

1. If absent, regenerate `/tmp/flash_dish_click_evidence_20261029` with
   `python -m lerobot.policies.flash.observation_processing` on
   `flash_benchmark_caryx_090`, its saved `flash/v1/observation-config.json`,
   `--prompt 'object_b=white dish' --seed 'object_b=91,119'`, and that `--output`.
2. Run `audit_dish.py`, `audit_clock.py`, then `score_clock.py` from this directory.
   The clock audit loads no labels; the scorer does not run or influence fitting.
3. Run `run_ablations.py` for the full dish/mug comparisons, then `summarize.py`.
4. `render.py` creates the plain-language images in the local visualization directory.

Large intermediate results are under `/tmp/flash_case_retest_20261029/`.
Raw rejected candidates remain available there for inspection. No robot is connected.

## Follow-up: root causes and alternative objectives

The additional scripts and [measurements](diagnosis_results.json) preserve the
experiments requested after the first report. No production defaults changed.

### Dish: filtering, checking, and later disagreement are separate

`diagnose_dish_depth.py` reconstructs the exact saved-to-current transform from
the completed no-span/no-component rerun. At frame 1 its reverse-depth check
reproduces **10.012 mm p90** exactly. Evaluated only where the neighboring pixels
are also retained, subpixel inverse-depth interpolation reduces 10.033 to
8.616 mm. It is **not mostly a rounding bug**. Distance perpendicular to a local
measured 3×3 surface is much smaller: **1.751 mm p90** (local plane RMS p90
0.357 mm). A fixed camera-Z cutoff and a surface-distance cutoff measure
different things, especially on a steep surface. Those subsets differ; the
reported numbers are not interchangeable sensor-accuracy estimates.

The frame-1 image/depth motion proposal was better supported before surface
refinement: interpolated camera-Z p90 3.118 mm and perpendicular p90 1.117 mm.
Thus shape refinement can worsen the separate check even at the first step.

By frame 10, the fitted surface's perpendicular discrepancy is **10.633 mm
p90**. A more expensive search (image motion, prior fit and four PCA starts,
200 evaluations each), with/without the outline-coverage term, did not produce
a selected fit below the existing 5 mm surface limit; selected surface p90
was 9.618/11.573 mm. Loosening the first-frame gate is not a demonstrated fix. An independent
symmetric point-to-point ICP probe, without our mask or visibility score,
tried five starts and retained fractions 1.0/0.9/0.75 with up to 200 iterations.
At frame 10 its best maximum of the two directional p90 distances was 14.337 mm
(10.148 mm current-to-saved). The later disagreement is therefore not confined
to the current mask-coverage objective; this still is a bounded search, not an
impossibility proof.

`check_dish_correspondences.py` independently matches frame 0 to later RGB
frames before lifting matched pixels using recorded depth. Of 193 pairs at
frame 10, 67 fit one rigid movement within the existing 6 mm RANSAC criterion.
Across all pairs, pair-to-pair 3D distances change by median 12.898 mm, p90
46.707 mm. A rigid movement cannot change those distances. This diagnoses
inconsistent **matched RGB-D measurements**, not proof that all matches are
correct or that all error is from the sensor. Source and imported depth were
checked at frames 0, 1 and 10: maximum difference within the dish mask is
1.499 mm, too small to explain the larger disagreement. RGB median absolute
readback difference is two intensity levels. The original source depth itself
has the changing shape seen in the accompanying figure.

What is established: selection is repaired; our filters removed smooth depth;
our ray-distance gate is poorly matched to sloping low-resolution surfaces;
later measurements/matches still disagree with a single saved rigid surface.
What is not established: independent pose accuracy or the exact share due to
sensor error, source RGB/depth timing/alignment, and wrong image matches.

### Clock: the score loses appearance and penalizes incomplete geometry

The image matcher proposes correspondences, but `WholeSurfaceFit` subsequently
scores only measured geometry and mask extent. Colour/pattern correspondence
and observed temporal movement are absent from the final objective. Four
residual blocks have equal aggregate weighting; they are not calibrated
likelihoods and are not guaranteed comparable across differently complete
reference views.

At target frame 150, source 149 has only 339 clean surface pixels out of its
1,303 mask pixels. Its mapped surface cannot cover the full current mask.
The candidate with 1.5° pairwise error therefore loses largely on **outline
coverage**, despite better current-depth agreement:

| Score contribution (lower is preferred)      | Wrong front-to-back fit, source 0 | Good neighboring motion, source 149 |
| -------------------------------------------- | --------------------------------: | ----------------------------------: |
| Saved points to current depth surface        |                             0.292 |                               0.269 |
| Current depth surface to saved points        |                             7.926 |                               2.995 |
| Projected points outside mask                |                             0.037 |                               0.000 |
| Current outline not covered by saved surface |                             1.450 |                              13.054 |
| Total                                        |                             9.705 |                              16.319 |

This is more specific than “shape versus pattern”: missing surface is punished
as wrong pose. Also, the image matcher itself can incorrectly match front to
back; using its score alone selected a 178° error. Neither cue is infallible.

`compare_clock_objectives.py` retains image-matched 3D correspondences inside
the surface objective with robust residuals, at explicit weights 0, 0.25, 1,
and 4. It also tests weights 0, 1 and 4 with the outline-coverage term removed.
The same cached correspondences feed all seven full-clip variants; weight 0
reproduces the previous baseline. `score_clock_objectives.py` reads the labeled
poses only after fitting has finished.

| Full 203-frame raw track                          | Median point error | Median / p90 orientation error |
| ------------------------------------------------- | -----------------: | -----------------------------: |
| Shape-only, all patches                           |            36.5 mm |                  31.5° / 43.1° |
| Add correspondence term, weight 1                 |            34.9 mm |                  26.8° / 37.8° |
| Add correspondence term, weight 4                 |            37.5 mm |                  25.6° / 37.3° |
| Remove outline coverage only                      |            34.7 mm |                  25.6° / 53.2° |
| Remove outline coverage + correspondence weight 1 |            29.9 mm |                  13.4° / 41.0° |
| Remove outline coverage + correspondence weight 4 |            32.7 mm |                  20.3° / 40.6° |

All remain inaccurate. The best median does not establish a universal setting;
removing outline coverage alone worsens the tail. Drift, incomplete reference
geometry, and correspondence mistakes remain coupled.

Separately reranking the existing 413 candidates at 13 target times gives:

| Pairwise selection                                     | Median / maximum orientation error |
| ------------------------------------------------------ | ---------------------------------: |
| Surface score only                                     |                    3.23° / 179.83° |
| Surface plus robust correspondence score               |                      2.76° / 7.52° |
| Correspondence score only                              |                    2.00° / 178.07° |
| Always use previous-frame unrefined image/depth motion |                      1.33° / 4.44° |

These are local edge errors, **not full-trajectory accuracy**. The already
measured 45.3 mm / 50.3° full-clip drift from chaining neighboring image motion
illustrates why good individual steps are insufficient. The score alternatives
are experiments, not a claimed replacement reference-view selection system.

### Mug and stray points: deleting one rule needs a replacement

`audit_stray_cleanup.py` reproduces the original dowel frame 280 exactly with
its original fixed 3 mm sampler: 178 points, including five at 431–441 mm depth
while the main dowel surface is at 331–353 mm. The five reappear if 3D patch
selection is disabled. All five belong to the **largest 2D mask component**;
image-component cleanup cannot remove this depth contamination.

| Diagnostic change                           | Dowel frame 280                           | Mug frame 0                            |
| ------------------------------------------- | ----------------------------------------- | -------------------------------------- |
| Keep largest 3D patch, 2 px margin          | Keeps 173 useful points, removes 5 strays | Keeps 427, removes 239 real cup points |
| Keep all 3D patches, 2 px margin            | All 5 strays return                       | Retains all 666 points                 |
| Keep largest 2D mask component, 2 px margin | All 5 strays remain                       | Retains all 666 points                 |
| Keep all patches, increase margin to 3 px   | Removes 5 strays **and 12 useful points** | Leaves 598 points                      |

No threshold is selected from these two examples. Image-mask fragments and
incorrect depth **inside** a correct mask require different checks. Tiny valid
parts, thin objects, occlusion-separated surfaces and different resolutions
prevent a universal point-count or pixel-margin rule.

Research checked against primary sources:

- [SAM's official mask utility](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/utils/amg.py)
  removes small disconnected **2D** islands by area. This is relevant to isolated
  mask fragments, but not the reproduced depth contamination above.
- [SAM 3](https://arxiv.org/html/2511.16719v1), sections 3 and C.3, explicitly
  handles false detections/distractors with track confirmation and periodic
  correction from confident new detections; individual masks can be refined
  using positive/negative clicks. Our adapter uses its detector at seed/loss,
  propagates with tracker memory, and periodically rebuilds from its current
  mask. That rebuild is not an independent detection-based correction.
- [SAMURAI](https://arxiv.org/html/2411.11922v2) uses motion consistency to select
  masks and higher-quality memories. It tracks 2D boxes/masks; it does not certify
  3D surface membership or solve a depth sensor's wrong returns.
- [Distractor-Aware Memory](https://arxiv.org/html/2411.17576v2) retains past
  examples that distinguish the target from distractors, while avoiding memory
  updates when the target is absent. The reported gains concern segmentation
  and object identity, not our rigid-pose accuracy.
- [SAM2Long](https://arxiv.org/html/2410.16268v3) keeps multiple segmentation
  histories instead of one greedy history; its final best-path selection suits
  offline demo processing, with extra computation and future-frame dependence.

Promising but **not implemented or validated here**: retain separated patches
as provisional observations, admit them to the persistent object surface when
they agree with its observed motion across frames, and use confident negative
clicks/detector corrections for foreign mask regions. An unseen real part and a
stray can both initially lack temporal support; withholding map promotion must
not mean permanently deleting either. Co-motion alone cannot separate objects
that happen to move together. Depth uncertainty must remain distinct from SAM
confidence, and partial observations must not be scored as complete geometry.

Reproduce the follow-up, using the same inference environment as above:
`diagnose_dish_depth.py`, `compare_dish_objectives.py`,
`check_dish_correspondences.py`, `check_dish_import.py`,
`check_dish_geometry_only.py`,
`compare_clock_objectives.py`, `score_clock_objectives.py`,
`audit_stray_cleanup.py`, then `render_diagnosis.py` and `verify_diagnosis.py`.
The clock comparison reads the correspondence cache from its first completed
pass; if absent, it builds the cache from the recorded RGB-D inputs.
