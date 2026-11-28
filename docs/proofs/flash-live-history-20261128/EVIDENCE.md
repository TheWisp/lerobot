# FLASH live history export — evidence

Captured on 2026-11-28 with the live worker and history PR's code (layer commit
66d888d4d), on a copy of one saved live run: 317 RealSense D435 frames with the
infrared pair, 160 of them also with Fast-FoundationStereo depth. It records one run at one revision; it is
not a design document.

| File                    | What it shows                                                                                                        |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `01-history-export.png` | Frame 158 of the saved run, the same frame read back from the exported LeRobot dataset, and its exported depth image |

Measured on this run, reading the export back with `LeRobotDataset`:

- 317 saved frames became one episode of 317 frames at 30 fps.
- Every depth pixel of every frame equals the saved sensor depth in millimetres,
  and every frame ID matches the saved one.
- Color is H.264 video: its mean absolute difference from the saved color is
  3.68 of 255, and 4.1 on the worst frame.
- The export also carries the infrared pair, the Fast-FoundationStereo depth, and the
  saved tracking files (110 processed frames, 38 overlays).

The run was saved by an earlier build; only its export and read-back used this
code. The live worker itself was not run for this capture: the camera on this
machine faced an unlit scene.
