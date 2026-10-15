# Basic motion checks, 2026-10-16

These measurements concern the white left SO-107 arm. They do not establish
that insertion is reliably smooth or that the recurring shoulder stop is fixed.

## Free-waypoint interpolation

Commit `1cb3f657b` coalesces generated free-line samples into their actual
waypoint, with checked cubic joint interpolation. Contact/follow samples and
gripper events retain their phases. The targeted motion/replay/generic suite
passed 146 tests. Restoring old target selection failed both new stationary
and moving-target release tests; restoring the old curve failed endpoint
braking. Three older group-view tests failed identically before and after
the change because they still observe the unused Cartesian target entry point.

Physical Act `insertion/acts/20261016_001205`, speed 2, P32/I0/D32,
gravity feed-forward 3.2, 80 mm/s and 60 degrees/s base limits, stopped during
key frame 5. The shoulder stayed at 25.76 degrees against an untrimmed target
of 20.79 degrees. Register readback confirmed its feed-forward-adjusted goal
at 18.81 degrees, zero velocity, load 632/1000 and no shoulder error flag.
The queue held future samples. Side-camera images show the held dowel's end
touching the tray, but opening the gripper and trying the same small lift
through the gizmo still left the shoulder at 25.85 degrees for 1.2 seconds.
This rules out the held object as the sole cause in that configuration;
contact elsewhere, effort and mechanics remain unresolved. Parking succeeded.

## Consumer timing

That act also recorded moving command intervals as short as approximately
5.2 ms after a delayed read. Commit `f062c76c8` spaces writes by a nominal
tick and checks playback lag against the newly validated encoder read.
A failed read no longer consumes an unsent command. The targeted jog,
queue, cadence and replay suite passed 139 tests. All four new regression
cases failed with the original consumer restored.

An empty-gripper Cartesian lift of 60 mm on `f062c76c8`, at 80 mm/s with
the same gains, completed without a halt. Twelve intermediate IK/mesh checks
validated its approach before motion. The saved trace has 29 motor samples
over 1.000 seconds: 28.01 Hz mean; command intervals 35.42 ms minimum,
36.17 ms p95 and 36.21 ms maximum. Encoder-derived endpoint error was 2.86 mm;
peak reported arm load was 308/1000. Command timestamps are reconstructed
from encoder-read monotonic time plus recorded command duration.

This was a short free-space gizmo check with the vision worker off, not a
paired insertion trial under tracking load. It does not verify the new
consumer throughout a pick/place or measure perceptual smoothness. The arm
was parked and disconnected afterward; the worker was stopped. The dowel
rolled beside the tape roll after release, so a scene reset was requested
before another insertion.

Raw local records, images, probe scripts and mutation logs are archived at
`/home/feit/Documents/showservo-experiments/20261016/basic-motion/`.
The original Act retains its RGB-D recording and motor trace in the demo store.
