# Run-tab Overlays panel resize

Captured from a GUI booted on a random free port by `scripts/gui/screenshot_gui.py`,
on the branch's own code. Cropped to the right-hand region so the panel's left
edge is visible against the camera area, which is what moves.

| File                 | State                                                             |
| -------------------- | ----------------------------------------------------------------- |
| `1-default-320.png`  | No stored width — the `var(--run-overlays-width, 320px)` fallback |
| `2-widened-600.png`  | Dragged past the maximum, clamped to 600px                        |
| `3-narrowed-220.png` | Dragged past the minimum, clamped to 220px                        |
