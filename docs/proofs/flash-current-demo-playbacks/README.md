# Current FLASH demo playback (2026-10-28)

These are captures of Run → FLASH camera playback on the `docs/flash-redesign` branch, refreshed after the held-pose depth-support gate. They show the pose saved by each prepared demo on the original RGB frames, with the timeline visible. They are demonstrations, not autonomous trials.

Each capture samples 36 fully loaded UI frames and repeats them at roughly the recording's source-time pace in phone-compatible H.264 video. The 30 fps video file does not claim 30 fps pose inference or UI rendering. The current 3D scene and source-estimate comparison remain available as separate controls in the UI.

- `dowel-insertion-camera.mp4`: source frames 200–550 of the insertion demo, from `partial-center-support-gate` prepared with the current depth-support rule. Its object poses still need review.
- `gamepad-stacking-camera.mp4`: source frames 210–440 of the gamepad-on-cube demo, from `flash-gamepad-cube-20261028-depth-audit-v11`. Its object poses still need review.

The older 3D screenshots in this folder predate these captures and are retained only as historical inspection evidence.
