# Recorded-scene reset for the SO-107 stacking experiment

This preserves the Astra reset tool used around the `pick_place` show-and-servo
experiment. It was previously only in a temporary scratch directory. It is a
rig-specific experimental tool: the included target image, camera transform,
grasp orientation, jaw meshes, colour cues, workspace and servo profile describe
the left arm named `white` and its overhead camera. These assumptions belong to
the reset scaffold, not to show-and-servo's object tracking or landing rule.

The GUI on `http://127.0.0.1:9100` owns the arm, serial connection and camera.
Every hardware call goes through its API. The tool needs the local `white_left`
motor and geometric calibrations and this checkout's SO-107 kinematics. Run in
the environment used by the GUI, with one controller owning the arm at a time:

```bash
python benchmarks/showservo_reset/rig.py state
python benchmarks/showservo_reset/reset_policy.py --tol-mm 10 --tol-deg 10
python benchmarks/showservo_reset/reset_policy.py --scramble --seed 7
python benchmarks/showservo_reset/rig.py park
```

Each reset captures a fresh scene, chooses reachable transfers or observed-floor
slides, checks the actual lift clearance before translation, and verifies the
result from another camera view. It parks on completion or failure. It writes
images, source snapshots, motor telemetry and `report.json` beneath `runs/`.
Creating `runs/<active-run>/STOP` interrupts guarded motion and starts recovery
and parking. The arm can still move while parking; wait for `connected: false`
before touching the scene.

The reference depth and CAD jaw samples are stored as compressed NPZ files
without changing their numeric values. `fixtures/slide_after.*` is the recorded
scene from reset `20261008_120533_335094` used by the slide regression tests.
All three NPZ types are local data, loaded without pickle.

## October 12 measurements and fixes

Reset `20261012_132417_711160` moved the cube, but falsely reported a descent
stall 35 mm above contact. The outer controller requested z=30.5 mm from an
observed 40.5 mm. Its move helper split that into two targets and returned
`held_short` at the first, z=35.5 mm. The contact detector compared the observed
height to the unsent z=30.5 mm target. A separate 5 mm arrival shortcut could
also skip the detector's final 3 mm step.

Descent now sends each bounded step to the controller exactly once, without a
second subdivision or the approximate-arrival shortcut. Contact detection still
checks downward progress, rebases the command at the observed pose when it
stops, and rejects more than 7 mm of sideways drift. Two regressions fail with
the original subdivision and shortcut restored.

Temperature telemetry produced isolated spikes: reset `20261012_132938_502324`
logged wrist roll at 57 °C between approximately 34 °C readings. An independent
12-second reconnect check returned 23 populated samples between 29 and 34 °C.
A hot reading still interrupts movement and parks. Before imposing a ten-minute
cooldown, the reset now reconnects at rest and requires five complete, cool
readings. Any hot reading or missing telemetry keeps the cooldown. Three
transient stops end the run. In `20261012_133304_645478`, this recheck returned
five samples at 29–34 °C and the reset resumed without the false cooldown.
The 70% arm-load guard and hardware limits are unchanged. Parking an already
disconnected arm is now a no-op, so interrupting a cooldown does not create a
spurious “no arm connected” failure.

**The reset has not yet passed a full cycle with these changes.** In
`20261012_132838_605662`, a loaded lift commanded z=50.28 mm but reached only
36.7 mm and stopped at its compensation bound. In
`20261012_133304_645478`, the approach to the distant gamepad drifted beyond
the 7 mm corridor; the arm parked, and physical trials stopped for a manual
scene reset. Static IK reachability does not establish that the loaded arm
will follow the path accurately.

The preceding stack miss, act `20261012_130212`, had **no injected error**.
Its pre-place hold correction was 12.76 mm / 10.63°. The observed fingertip
was about 2.5 mm from its commanded position just before release. A fit to the
visible green surface suggests an orientation discrepancy, but a partial
surface fit can be biased by depth noise and the curved shell. This is not
yet an established cause. An offline replay from a fresh tracker state kept
all 117 reconstructed model points inside the initial object mask, at a median
0.75 mm from the initial observed surface; that replay does not reconstruct
the live tracker's prior history exactly.

## Recorded failure at 16:04 and subsequent reset

Act `20261012_160437` did not inject a grasp error and aborted before placement.
The hold estimate differed from the demo by 7.97 mm / 18.08 degrees. Keeping the
previously selected landing turn after applying that correction drove the
wrist to its limit. Replanning the corrected hold and remaining placement,
including permitted landing turns, fixes that planning failure offline; it
does not establish the accuracy of the measured hold or physical stacking success.

The saved images do not independently establish 18 degrees of physical slip.
At the pre-place check only a narrow part of the gamepad and one button are
visible. The fitted points are predominantly on that visible part, rather
than on the gripper. Seven demo carry frames have valid depth on only 31–42%
of the visible green pixels. Independent surface-normal fits disagree with
the tracker and include an unstable demo-frame outlier. They therefore cannot
serve as ground truth. The demonstrated and corrected holds both project
plausibly onto the small visible patch. Agreement among successive tracker
estimates establishes repeatability, not accuracy; slip, a different initial
grasp, and estimation bias remain unresolved without a clearer held-object view.

In reset `20261012_160640_008685`, the third attempt already stopped 4.85 mm
sideways from its approach target while the modeled jaws were 37 mm above the
nominal tray. It then accepted ten consecutive `held_short` descent results.
The final requested tool position was `[-144.287, -168.917, -8.400]` mm; calibrated
FK of the commanded joints matched it to about 0.005 mm. Encoder FK was
`[-144.275, -176.108, -2.086]` mm: 7.19 mm sideways and 6.31 mm high. Thus IK
reachability was not the cause of this reset's position error. Joint deviations
below 1.5 degrees accumulated into a 9.57 mm tool error, which the legacy
six-degree stopped-short allowance accepted before its lateral guard aborted.

The shoulder/elbow gravity offsets were -2.14/-1.11 degrees at that endpoint;
observed deviations from the unmodified IK goals were -1.48/-0.82 degrees.
This is consistent with excessive compensation contributing to the error,
but does not isolate it from servo deadband, friction, or contact. No gain or
compensation setting was changed based on this single descent. The jaw mesh
also extends roughly 6 mm below the nominal tray at the last observed pose,
despite the tool origin remaining above the old API floor. That exposes the
tool-only clearance assumption; it is not proof of physical penetration or
the precise onset of contact. The motion planner checks the complete jaw geometry.

`tests/gui/fixtures/reset_20261012_160640_arrival.json` retains two actual endpoint
readings and their calibrated FK. The guarded-path regressions prove the new
executor cannot advance from the first stopped-short endpoint and immediately
rejects the final following error. This is a check of arrival decisions, not a
physical replay. Widening arrival tolerances or the hard following-error limit
in process-local mutations makes the corresponding regression fail.

## Offline verification

```bash
cd benchmarks/showservo_reset
python -m unittest test_contact_control test_slide_planner
```

The contact tests use fake hardware and include descent-command fidelity,
short steps, hot/missing telemetry, transient retries and idempotent parking.
The slide tests use the included image/depth fixture and the local calibrated
kinematics. The original scratch suite also covered perception and transfer
planning against older local recordings; those larger historical recordings
are not part of this directory.

## Experimental motion-planning reset

`planned_reset.py` builds an **offline, single-object reset plan**. It does not
connect to the arm or send commands. An experimental live runner is described
below; physical validation is pending. In particular,
sending the resulting tool poses through the Cartesian jog would let IK choose
a different joint path from the one checked here.

```bash
python benchmarks/showservo_reset/planned_reset.py \
  --rgb /absolute/path/frame.jpg \
  --depth /absolute/path/depth.npy \
  --joints /absolute/path/observed_joints.json \
  --object gamepad --out /absolute/path/plan.json
```

Depth is in metres; the joints file maps the seven motor names to their observed
positions, in motor degrees except for the gripper's 0–100 units. Use joints
from the same stationary frame. Both objects must be visible. The command uses
the local left-arm calibration and requires Pinocchio, Coal, Pink, NumPy,
SciPy and OpenCV in the GUI environment. No new GPU model is required.
The report contains the model's start configuration, phase-by-phase joint
samples and attachment transforms, or a rejection reason. A rejection exits
with status 1. `ok: true` means a path passed this geometric model, **not** that
a grasp or reset succeeded on hardware.

The bounded RRT-Connect search checks intermediate joint configurations, keeps
the gripper fixed during transit, and tries alternative central grasps on both
sides. Grasp search varies yaw, height and small lateral offsets. It derives
height from the jaw meshes through closure and requires both jaw bodies to
reach the selected object's box. Descent, lift and retreat follow a straight
tool corridor. Carry checks transform the held object's whole box with the
planned tip-to-object transform; release and retreat use the destination scene.
The default preserves the object's current yaw, consistent with this reset's
orientation-free use case. Box contact is a geometric prefilter, not a force
closure or slip model; a live executor would need to measure the actual grasp
and validate/replan the remaining trajectory before carrying.

The scene model uses the calibrated arm meshes, nonadjacent-link self-collision,
the sloped tray, measured boxes for the two objects and 5 mm columns for other
RGB-D returns. Clutter columns and ordinary object checks reserve 7 mm.
Intentional finger contact is restricted to the selected object when closing
or releasing; the final approach can enter its margin but cannot intersect it.
The table is not exempted. Planning samples arm motion at no more than 0.5 motor
degrees and gripper-only motion at one unit. This is discrete collision checking,
not a continuous swept-volume or tracking-error guarantee.

Model coverage has explicit limits. Missing columns inside the tray view are
blocked up to one metre. The flat tray is **assumed** beneath recognized object
masks and depth returns within 12 mm of the measured robot mesh; those hidden
surfaces were not observed. Object masks are protected from robot removal.
Cables and attachments missing from the robot model can remain classified as
clutter. Geometry outside the camera's tray footprint and the fixed mount region
is not modeled as an observed environment. These assumptions must be resolved
or validated before live execution; never clear collisions just to obtain a plan.

The search has per-path iteration, collision-check and time budgets, plus a
120-second limit checked between grasp candidates. Individual bounded IK/path
calls can extend the latter. It refuses a colliding start instead of inventing
an unchecked escape or park motion. It changes none of the existing load,
temperature or descent guards.

### Recorded evidence, October 12

On the idealized positive fixture (tray and the two target boxes, with all other
clutter explicitly omitted), it plans a 20 mm gamepad transfer including open,
approach, descent, close, lift, carry, place, release and retreat. The original
pinch offset and a 2 mm lateral adjustment fail the two-jaw contact check; a
4 mm adjustment admits a plan. This is an offline test, not a successful
physical transfer or evidence that the recorded clutter can be ignored.

The calibrated mesh predicts jaw vertices about 16–21 mm below the virtual tool
point at the tested demo grasp orientation. The planner now accounts for that
offset. This observation does not establish that tool calibration caused the
previous physical miss; the mesh, touch calibration and actual fingers still
need to be compared on the rig.

The full `20261012_145240_395213/connected_view` scene rejects its initial
configuration: contacts include the robot silhouette/unknown view boundary and
a jaw close to the tray's reserved margin. Its available idle joint snapshot
precedes the image by about 0.85 seconds, so this is a diagnostic reconstruction,
not a synchronized execution input. The refusal remains in place. No arm motion
was made while developing this planner, and no collision or telemetry guard was
relaxed to make the recorded scene pass.

### Planner regressions

```bash
cd benchmarks/showservo_reset
python -m unittest test_motion_planner test_collision_scene test_planned_reset
```

The search tests include an obstacle detour, collision between clear endpoints,
locked gripper, invalid states and bounded failure. Analytic collision fixtures
cover payload collisions, support height, unknown depth and narrow contact
exceptions. The end-to-end positive test uses a frozen calibration fixture and
checks every phase again, with all GUI calls forbidden. That fixture is only for
regression reproducibility and is never loaded by the hardware-facing defaults.

### Guarded segment execution

The GUI now offers `/api/jog/path/{start,state,heartbeat,finish,stop}` for
externally collision-checked joint segments. `guarded_execution.GuiExecutor`
bridges a segment from fresh observed joints, rechecks every interpolated sample
(including intermediate gripper openings when carrying), and monitors actual
arm/payload clearance before renewing the GUI's one-second lease. The GUI checks
calibration, servo/model/workspace limits, speed, observed arrival, following
error, load and temperature. It excludes competing motion writers, applies no
Cartesian trim, and never ramps to an unverified first sample. A final sample
must actually arrive before the next phase or return to Cartesian control.

Failure freezes the arm on its last bounded command. It does **not** automatically
park: a checked retreat or operator intervention is required. Reported hot
samples stop this path for recheck; a later cool sample cannot resume it by
itself. Hardware fault handling remains active. The caller must supply a current
scene and the measured payload transform; this API does not establish collision
clearance on its own. A stopped jaw may be wider than the closing command, so
that opening range must be included in planning.

The executor is tested with a fake bus and fake GUI. The recorded scene's
initial collision refusals still need physical investigation before claiming a
successful reset.

### Reset with measured grasp geometry

`live_reset.py` adds the experimental runner. It requires an already connected,
stationary arm and a start pose that passes the full collision model. It never
connects the serial port or substitutes the old ready/park ramps. Without
`--execute`, it captures a fresh stationary scene and writes a plan only:

```bash
python benchmarks/showservo_reset/live_reset.py --object gamepad \
  --destination-mm -70 -148 --out /absolute/path/new-run
```

Adding `--execute` runs the checked transfer. A `STOP` file in the run folder
interrupts execution. The GUI must include `/api/reset/frame.npz`,
`/api/reset/object_pose`, and the guarded path API described above. Snapshots
pair a fresh RGB-D image with stable encoder readings and the connected arm's
camera calibration. Object poses use a consensus of strong finds against the
same demo reference. Weak/inconsistent fits and camera movement are refusals.

The October 12 17:49 startup retest found that a raw matrix comparison rejected
a stationary arm 0.107 mm / 0.173 degrees from its target. Comparing every matrix
entry to 0.001 mixed metre tolerances with rotation coefficients. Cartesian
observation now uses the same translation, rotation and joint arrival limits
as guarded execution. The before/after capture checks still reject movement,
stale telemetry, active act/follow, wrong ownership and camera displacement.
The snapshot always records the measured pose. This sensing fix does not make
an initial configuration collision-valid: that retest's jaw mesh had only
1.07 mm of tray clearance, below the planner's 2 mm floor reserve, and its
scene reconstruction also reported robot/unknown-space contacts. The arm was
parked without attempting a transfer or stack.

The 17:57 hardware recheck on `9176b70e3` accepted the stationary snapshot and
captured synchronized RGB-D/encoders. Reset preflight then refused
`observed start: scene collision L4_1_0`. The reconstructed contacts include
the camera boundary and robot silhouette; their classification remains
unresolved. This capture estimated 3.17 mm of jaw/tray clearance and no floor
collision, versus 1.07 mm in the earlier diagnostic pair. That variation means
the earlier clearance result should not be treated as a stable measurement
of physical contact. No collision guard was bypassed. Temperatures during the
bounded startup check were normal, and the arm was parked and disconnected
without a transfer. This validates the sensing fix, not a complete reset.

The runner measures the actual hold after closure and lift, replans placement,
and checks the hold again after carrying. A shift changes the gripper target
while preserving the object destination. Repeated slip stops the run. It checks
the complete release and retreat before carry, verifies the released object's
position, tilt and support, and replans retreat around its measured pose. It
returns to the initial arm pose through the updated collision scene and hands
control back while connected. The enclosing rig experiment remains responsible
for a checked park/recovery; this runner does not claim the old fold ramp is
collision-checked.

**Physical validation is pending.** The calibrated idealized-scene regression
handles a measured 4 mm grasp shift; an opposite hypothetical shift is refused
because the fixed jaw cannot retreat. Neither result establishes a physical
reset or stack. In the recorded connected pose, the modeled fixed jaw is inside
the 2 mm tray margin. Fresh synchronized geometry must resolve that refusal;
the runner will not erase it or widen a threshold to begin moving.
