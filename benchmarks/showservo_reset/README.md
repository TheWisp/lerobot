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
