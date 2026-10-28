# FLASH demo review, 2026-10-28

Screenshots from the actual prepared dowel/ring and gamepad/cube recordings in
Run → FLASH on port 9103. The camera view now starts with one displayed estimate
per object. It prefers the pose saved in the skill; for objects outside the
current skill interval it shows the extracted perception pose and labels it as
not saved. Predictions are marked as predictions. Source comparisons are opt-in.

- [Dowel camera, frame 501](dowel-camera-used-pose.jpg)
- [Gamepad camera, frame 290](gamepad-camera-used-pose.jpg)
- [Dowel focused in 3D](dowel-3d-focus.png)
- [Gamepad focused in 3D](gamepad-3d-focus.png)
- [Dowel 3D editor, frame 522](dowel-3d-editor-aligned.png)
- [Dowel camera editor, frame 522](dowel-camera-editor-aligned.jpg)

The 3D surface contains only measured, visible depth from the selected RGB-D
frame. Its missing reverse side is not a reconstructed object model. These
screenshots show the UI and the remaining pose disagreement; they do not
validate the prepared poses for control.
