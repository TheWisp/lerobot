# FLASH scene editor, insertion demo

Captured 2026-10-28 from insertion episode 0, prepared version
`insertion-migrated-v10-preferred-hover`, in the Run → FLASH demo viewer.

![Frame 180 with measured surfaces and key-frame editor](demo-ui-frame180.png)

The editor shows one selected key frame beside playback. Yellow timeline markers
select key frames, arrows move between them, and clicking a measured object opens
its mask controls. Other settings stay closed until requested.

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

## Turn editor check, 2026-10-28

![Simplified turn editor on insertion demo frame 343](turn-rule-preview.png)

In the insertion demo, key frame 6 can now be given a turn around the ring's
saved center with one button. The angle range is under **Refine turn**. The
browser saved the rule without page errors. The cyan measured patch is on the
green ring beneath the dowel; the blue object farther right is unrelated.

The center and direction are unchanged by the ring's unobserved spin, although
the saved center's _accuracy_ is not proven by this screenshot. The UI rule is
validated against the prepared object data. The editor regenerated a candidate
on the actual insertion dataset (`flash-prepared-20261028-030548-3f5318`), and
its compiled skill contains the rule on interval 5. On this demo's recorded connection
(frames 290–342), endpoint IK from the initial arm state missed the demonstrated
goal by 21.7 mm. Following the same connection with the runtime planner's
intermediate IK checks found the recorded zero-angle choice and timed it at
0.89 s. That check took about 19 ms; the bounded multi-angle search took about
1.0 s. This is a kinematic check, not collision validation or a robot trial.
