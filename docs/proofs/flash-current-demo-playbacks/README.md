# Current FLASH demo playback (2026-10-28)

These are fresh captures of Run → FLASH camera playback on the `docs/flash-redesign` branch. They show the pose saved by each prepared demo on the original RGB frames, with the timeline visible. They are demonstrations, not autonomous trials.

The captures sample a fully loaded UI frame about every three source images, then repeat those images at the recording's source-time pace in phone-compatible H.264 video. The 30 fps video file does not claim 30 fps pose inference or UI rendering. The current 3D scene and source-estimate comparison remain available as separate controls in the UI.

- `dowel-insertion-camera.mp4`: about 6.7–18.3 seconds of the insertion demo, from the isolated preparation rebuilt with the measured-ring-center fix (pose-trace version 7). Its object poses still need review.
- `gamepad-stacking-camera.mp4`: about 9.1–25.3 seconds of the gamepad-on-cube demo, from `flash-gamepad-cube-20261028-depth-audit-v11`. Its object poses still need review.

The older 3D screenshots in this folder predate these captures and are retained only as historical inspection evidence.
