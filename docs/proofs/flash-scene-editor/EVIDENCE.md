# FLASH scene editor, insertion demo

Captured 2026-10-28 from insertion episode 0, prepared version
`insertion-migrated-v10-preferred-hover`, in the Run → FLASH demo viewer.

![Frame 180 with measured surfaces and key-frame editor](demo-ui-frame180.png)

In the camera-aligned 3D view, measured points from the saved masks and aligned
depth projected back to their RGB pixel locations. At frames 180, 291, and 450,
the maximum discrepancy across 85, 64, and 174 sampled masked depth pixels was
below `3e-13` display pixels. The browser reported no page errors. This checks
the renderer's camera projection and frame synchronization; it does not certify
the saved object pose, segmentation, or depth accuracy.

The GUI regression test in `tests/gui/test_flash_inspect.py` also exercises
clicking a measured 3D surface, changing a key frame's reference, and sending
that edit for preparation. The legacy importer tests check that the authored
reference changes the generated relation except in an explicit start-anchored
interval.
