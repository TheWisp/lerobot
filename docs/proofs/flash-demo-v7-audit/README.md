# FLASH demo pose comparison, 2026-10-28

The pose arbitration now compares the frontmost model surface on each camera ray with aligned RGB-D. A nearer measured surface outside the object's mask is treated as an occluder. This avoids counting hidden model points as visible depth errors. The same comparison is used by offline demo preparation and the live causal/held-object estimators. The fitted object model and mask source are unchanged.

Both demos were prepared again from their v6 annotation and keypoint files with pose-trace selection version 6; neither used the old cached selections. These are candidate bundles, not physically validated trials.

| Demo              | v6 held replay                            | v7 held replay                                              | Remaining concern                                                                                                                                                                           |
| ----------------- | ----------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dowel into ring   | 184 conflicted carry frames, 39 corrected | 0 conflicted carry frames, 223 carrying a visual correction | The long middle span propagates the attachment accepted at frame 291 from the gripper; it is not 223 independent visual measurements. The ring remains group/point predicted.               |
| Gamepad onto cube | 51 conflicted frames, 135 corrected       | 47 conflicted frames, 139 corrected                         | A late direct fit flips the gamepad orientation around frames 429–430. The comparison still cannot choose it over the gripper prediction, so the replay remains conflicted through release. |

The contact sheets show the held prediction in orange and the independent demo extraction in cyan at the listed recorded frames. [Insertion](insertion.jpg) looks better aligned through carry and insertion; [gamepad placement](pick_place.jpg) shows early alignment and the late unresolved fit. These are visual checks, not metric pose ground truth.
