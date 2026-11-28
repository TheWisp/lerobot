# FLASH demo execution — evidence

Captured on 2026-11-27 in the GUI's demo editor (from the GUI PR, #283) at
source commit 29a575fe8. Its demo document, demo tracking and execution code
is the same as the demo-execution PR's (#282). It records one capture at one
revision; it is not a design document.

| File                                | What it shows                                                                                                                                                                                                                                       |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01-demo-document-and-tracking.png` | The recorded demo "Gamepad onto green ring" (1227 frames) at frame 320: two selected objects, both observed; tracking complete at pipeline version 1; three keyframe ranges, relative to the robot base, to the green gamepad and to the green ring |

The tracking shown was re-run with that code on 2026-11-27: the saved result
was stale, and opening the demo re-tracked it. Apply to robot was not pressed;
no robot was run with this code.
