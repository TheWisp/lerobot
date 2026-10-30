# FLASH perception failure ledger (research snapshot)

Updated 2026-10-29. **Rigid objects only. No general fix is selected yet.**

“Our current method failed” does not mean “this cannot be fixed.” A limitation
below means that cleaning or matching the **same available measurements** cannot
recover information those measurements never contained. Other views, RGB cues,
physical constraints or learned priors may still help.

## Confirmed causes in our implementation

| Cause                                                                  | Evidence and conventional checks                                                                                                                                                                                                               | Verdict                                                                                                          |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Object selection fails                                                 | Clicking the pale dish repairs the failed name prompt throughout 245 frames.                                                                                                                                                                   | Selection engineering; not a fundamental limitation. Pose tracking still needs separate diagnosis.               |
| Depth cleaning rejects useful surface                                  | The dish's 10 mm local span rule rejects smooth slopes. Keeping all 3D pieces restores a real mug side but also restores five stray dowel points.                                                                                              | Confirmed filtering trade-off. Neither remove-all nor largest-piece-only is a general fix.                       |
| A partial shape is scored as if complete                               | On the clock, the good previous-frame match cannot cover the whole current outline and loses to a wrong front/back match. More search finds good candidates; mixed image/shape scoring improves selection but still drifts over the full clip. | Confirmed objective/ranking problem, not proof of unavoidable ambiguity.                                         |
| Starting or maintaining a track can be wrong independently of matching | The box's tiny initial surface is published as observed despite 147.5 mm disagreement. The fast object-point tracker clears valid identities at a time gap; an isolated gap-rule change retains 48 consistent points.                          | Initialization, confidence and recovery need their own evaluation. See the [new sweep](failure-sweep/README.md). |

The [dish/clock/mug experiments](reference-retest/README.md) include failed
alternatives, full trajectories and the stray-point counterexample. A lower
average error or more accepted frames alone is not sufficient to adopt one.

## Observation limits and unresolved cases

| Pattern                                   | What is established                                                                                                                                                                                                                         | What is **not** established                                                                                                                                          |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Transparent / translucent surfaces        | Glass-jar depth often measures the table behind the object. Filtering cannot reveal an unmeasured glass wall.                                                                                                                               | Not all transparent objects fail alike; another jar gives mixed depths. Depth completion would be inference, not a recovered measurement.                            |
| Missing depth on an opaque object         | On the new cracker-box frame, SAM3 covers the object but 71% of pixels have zero depth.                                                                                                                                                     | Why the sensor/source missed it; whether RGB plus the surviving measurements can recover an accurate pose.                                                           |
| Good masks but inconsistent depth/matches | The dish still disagrees after restoring its surface, changing interpolation, trying more starts and independent ICP. Imported depth differs from the source by less than 1.5 mm, insufficient to explain the later 10–14 mm discrepancies. | The share due to sensor error, source RGB/depth timing, and incorrect image matches. No independent dish pose labels exist. **Unresolved, not “unfixable.”**         |
| Full occlusion or indistinguishable views | Without any view or valid motion relationship, current object motion is not measured. Identical observations cannot determine which visually identical material point is which.                                                             | The clock failure is not proof of this limit: a correct neighboring match was available. Group/gripper predictions are additional evidence, not direct observations. |
| Accepted pose disagrees with labels       | Some new clips pass surface checks while differing from annotated motion.                                                                                                                                                                   | Exact error attribution: annotations can jump; symmetric rotations may be task-equivalent. Image/depth residual is not calibrated pose confidence.                   |

## What the wider research adds

These are evidence about failure mechanisms, **not models adopted or benchmarked
in this work**:

| Source                                                                                                                                                               | Relevant finding                                                                                                                                         | Implication / remaining coverage gap                                                                                                                                                      |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [RealSense projection and occlusion](https://www.realsenseai.com/stereo-depth/occlusion/)                                                                            | RGB and depth sensors have different viewpoints; visibility can cause color/depth boundary mismatches even with correct calibration.                     | Separate mask error from cross-sensor occlusion. We have not isolated this cause on the dish.                                                                                             |
| [RealSense depth filtering](https://dev.realsenseai.com/docs/post-processing-filters/)                                                                               | Temporal depth filtering can smear moving scenes.                                                                                                        | Hole filling/smoothing needs moving-object counterexamples, not only improved static surfaces.                                                                                            |
| [Controlled transparency/translucency measurements](https://pmc.ncbi.nlm.nih.gov/articles/PMC9572012/)                                                               | Different RGB-D sensing technologies behave differently through transparent/scattering media.                                                            | Keep sensor provenance. Do not assume an iPhone result transfers quantitatively to the rig's camera.                                                                                      |
| [Sight View Constraint, 2024](https://arxiv.org/html/2409.05065v1)                                                                                                   | Tests whether a proposed alignment would place an opaque surface in front of another measured surface along the camera ray.                              | Useful distinction between missing overlap and actual contradiction; its assumptions do not hold unchanged for transparent surfaces or unreliable depth.                                  |
| [UA-Pose, CVPR 2025, appendix F](https://arxiv.org/html/2506.07996v1)                                                                                                | A learned selector can prefer an obviously wrong pose when its object model is incomplete.                                                               | Ranking is a separate failure even with better models. Its known-pose references / generated model assumptions differ from our causal pipeline.                                           |
| [RegScorer, 2026](https://github.com/WHU-USI3DV/RegScorer)                                                                                                           | Conventional distance/inlier scores can select incorrect alignments under symmetry, repeated structure and low overlap.                                  | Independently supports auditing candidate selection. Scene-registration results do not establish accuracy or speed for our small objects.                                                 |
| [Tracking with built-in recovery, 2026](https://arxiv.org/html/2607.23468v1)                                                                                         | Uses earlier-view memory and consistency checks; still reports unseen-view recovery limits and identity switches between same-kind objects.              | Evaluate recovery and identity separately. The method requires a CAD/reconstructed mesh, so it is not a direct model-free replacement.                                                    |
| [SAM3, video design](https://arxiv.org/html/2511.16719v1)                                                                                                            | Meta periodically checks tracked masks against new detections and re-prompts on confident agreement; it also describes detector-guided drift correction. | Our GUI/FLASH wrapper detects at seed and after reported loss, but does not independently check a confident track. The published correction strategy has not been evaluated on our clips. |
| [CDTB RGB-D benchmark](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lukezic_CDTB_A_Color_and_Depth_Visual_Object_Tracking_Dataset_and_ICCV_2019_paper.pdf) | Tests motion, clutter, disappearance and multiple sensing setups; fast motion is difficult.                                                              | Boxes are not 6D labels. Our current sweep does not cover controlled lighting, fast motion, distractor identity and all sensor combinations.                                              |

## What the evidence does and does not support

The [temporal/depth pilot](temporal-depth-pilot/README.md) adds two distinctions:
background-depth contamination recurs inside the main SAM mask, so persistence
alone cannot certify it; and RGB-plus-sensor depth models can fill holes but may
also move valid surfaces. LingBot improves one withheld-reading dish check, not
all five cases. No true-hole geometry or full tracking improvement is established.
The [current pipeline map](../../../src/lerobot/policies/flash/README.md#perception-at-a-glance)
separates production layers from these experiments.

The [ten-scene Caryx sweep](ten-scene-sweep/README.md), labeled clock audit, local
dowel/mug counterexamples, and [nine longer YCBInEOAT intervals](failure-sweep/README.md)
now separate selection, measurement, cleaning, pose fitting and track lifecycle.
They do **not** exhaust all failure patterns. The longer sweep adds 1,629 frames
plus a separate 181-frame SAM3 comparison, with the same configuration throughout.

Still missing are controlled shiny/black/metal surfaces, multiple indistinguishable
objects crossing, abrupt motion versus dropped frames, systematic RGB-depth time
offsets, and independent labeled moving-camera scenes. Deformation is deliberately
outside this rigid-object stage. The mostly stationary sugar-box control must not
be counted as evidence for successful reorientation.

A future change is justified only by improving the intended measurement while
preserving these counterexamples: valid separated pieces **and** stray points;
partial overlap **and** front/back discrimination; recoverable gaps **and** real
unexpected motion. Preserve raw proposals, published poses, uncertainty and loss
duration so improved availability cannot conceal incorrect tracking.
