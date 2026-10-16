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

The shoulder stop did not first appear with `1cb3f657b`. In the earlier
successful Act `20261015_233027`, the shoulder stayed near 26.81 degrees
while the key-frame-5 command dipped to 22.23 degrees. Its 4.58-degree
maximum lag stayed inside the 5-degree transport limit; the command then
rose and the stage completed. Act `20261015_234819`, before the free-waypoint
interpolation, stopped in the same stage at 26.20 observed / 21.20 commanded.
The new interpolation run stopped at 25.76 / 20.79 degrees. The latest pose
and trajectory differed slightly across physical resets, so these runs show
a pre-existing narrow margin, not a same-scene comparison of interpolation.
The interpolation's contribution to that margin remains unmeasured.

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

## Later insertion trials

Act `20261016_005114` completed insertion. Its shoulder passed through
6.9 to 6.0 degrees during withdrawal while following a monotone decreasing
command, with load near 400/1000. Act `20261016_010345` stopped earlier,
during carry, at 20.4 observed / 15.5 commanded degrees and load 584/1000.
Act `20261016_010735` inserted the dowel (confirmed in the camera timeline)
but stopped on withdrawal at 6.8 observed / 2.7 commanded degrees and load
504/1000. The actuator read the same 6.8-degree encoder position for over
0.8 seconds while the other joints followed their commands. Its temperature
was normal. All three used speed 2, P32/I0/D32, gravity feed-forward 3.2,
and the same 80 mm/s and 60 degrees/s limits.

The calibrated rigid-arm mesh reported no self-collision in the two stops.
Against the static RGB-D scene, the nearest modeled link was approximately
46 mm from observed obstacles at the carry stop and 37 mm at the withdrawal
stop. These checks exclude neither a cable snag nor geometry hidden from the
overhead camera. They make a visible tray collision an unsupported explanation
for the later stops. In `010735`, the command briefly reversed from 5.5 to
6.6 degrees at the end of the post-release key frame, then headed toward the
world-frame withdrawal target. The shoulder was already at 6.8 degrees when
the command reversed; the trace does not establish whether this caused the
sticking. The IK solves are seeded from the previous solution, and the saved
joint commands show no branch jump at the stop.

The tracker's on-screen FPS was 7.45 at the start of the successful Act and
7.05 at the start of `010735`. This is the pose-result rate, not the camera
or motor rate. The latter Act's 30 Hz camera timeline saved 590 distinct
RGB-D frames in 21.5 seconds; the motor trace recorded 499 ticks in 21.35
seconds and a largest inter-tick gap of 315 ms. For `010735`, median P2P
compute, server pose verification and overlay times were 88, 58 and 92 ms,
respectively. One-off find/reference work currently shares the track's
single render executor; its queueing contribution requires a retest after
executor isolation. A disconnected-arm live check after server restart showed
about 10 FPS, with median verification and render times of 47 and 26 ms.
