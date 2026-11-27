<!-- Captured evidence for one change: what was observed, not what was designed.
     Design documents live beside the code they describe. -->

# SAM3 identity after occlusion — captured runs

Issue #270: once the gamepad covers the green cube, the mask labelled "green
cube" moves to another object. The files here are recorded masks drawn over the
camera frames: first as the issue observed them, then main and the fix on the
same frames.

## The recording

The local `pick_place` demo: 628 frames at about 22.5 fps from the gripper
camera, the arm placing a green gamepad onto a green cube next to a tape roll.
Prompt `green gamepad . green cube`, one instance per name, no clicks. Every run
below goes through the shared adapter the way Process dataset does
(`build_adapter` → `set_control` → `set_camera` → `reset` → `segment` per
frame), on an RTX 5090 with main's lockfile `transformers 5.5.4`. The cube never
comes back into view in this recording, so recovery onto the cube itself is
pinned by `tests/gui/test_overlay_adapter_detect.py`, not by a run.

## Files

| File                                      | Shows                                                                                                                   |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `cube-mask-switch.png`                    | The issue's figure: the cube tracked, covered, then its label inside the tape roll (research branch `3bfe3743`, 448 px) |
| `recovery-still-selects-wrong-object.png` | The issue's second figure: frame 500 with the stock restart (tape roll) and with empty seeds dropped (gamepad)          |
| `03-before-after-448.png`                 | Frames 150, 480 and 500 at 448 px, the issue's setting: main `13f05f36` above, fix `1d1d648e` below                     |
| `04-before-after-672.png`                 | The same frames at 672 px, the GUI default                                                                              |

## What the runs measured

|                                                                          | main `13f05f36`, 448                 | main, 672                        | fix `1d1d648e`, 448                                  | fix, 672                       |
| ------------------------------------------------------------------------ | ------------------------------------ | -------------------------------- | ---------------------------------------------------- | ------------------------------ |
| frames with a "green cube" mask                                          | 0–426, 494–627                       | 0–341, 348–349, 367–425, 496–627 | 0–426                                                | 0–341, 348–349, 367–425        |
| frames 428–627 with a "green cube" mask                                  | 134                                  | 132                              | 0                                                    | 0                              |
| frames with an empty "green cube" mask and a score ≥ 0.30 (counted held) | 44                                   | 34                               | 0                                                    | 0                              |
| the periodic restart at frame 450                                        | `lost none`, re-seeds the empty cube | same                             | `lost ['green cube'] · re-seeding ['green gamepad']` | same                           |
| where "green cube" lands afterwards                                      | tape roll from 495, score 0.87–1.0   | tape roll from 496               | nowhere                                              | nowhere                        |
| recovery searches declined because the detector offered the held gamepad |                                      |                                  | 28, frames 490–625                                   | 0 (the detector found nothing) |
| "green gamepad" masks against main, frame for frame                      |                                      |                                  | IoU min 0.999, median 1.00                           | IoU min 0.969, median 1.00     |
| "green cube" masks before the cover (frames 0–426) against main          |                                      |                                  | identical, IoU 1.00                                  | identical                      |
| per-frame time, median / p95                                             | 22 / 24 ms                           | 31 / 33 ms                       | 23 / 45 ms                                           | 32 / 61 ms                     |

The p95 rises because a lost object is probed by the detector every
`RECOVER_EVERY` frames, the documented cost of a loss. On main the empty mask
hid the loss, so those probes never ran.

Adapter log of the fix at 448, frame index prefixed by the driver:

```
f00427 track[cam]: lost ['green cube'] (no pixels, or score < 0.30)
f00450 flush[cam]: lost ['green cube'] · re-seeding ['green gamepad']
f00490 detect[cam]: 'green cube' offered the mask held as 'green gamepad'; skipped
...every fifth frame...
f00625 detect[cam]: 'green cube' offered the mask held as 'green gamepad'; skipped
```

Main's masks at 448 agree with the masks saved by the run the issue drew its
figures from (IoU ≥ 0.9 on all 628 frames, 241 frames bit-identical), so the
issue's figures and the "before" rows here show the same failure.
