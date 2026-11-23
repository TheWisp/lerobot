# SO-107 gravity compensation

The SO-107's STS3215 servos accept position goals. Under a steady gravity load, the joint settles short of its goal. The gravity robot sends `goal = desired + alpha × gravity_torque(desired)` so the resulting position is closer to the desired joint angle. It uses P=32, I=0, D=32 and a measured default `alpha=3.2°/(N·m)`; the plain `so107_follower` still uses P=16 with no gravity offset.

This is a separate single-arm robot type, `so107_follower_gravity`; the bimanual SO-107 profiles do not use it. The servo zero/range calibration file is shared with `so107_follower`; the motor-to-URDF alignment is selected by `arm="left"` or `arm="right"`. The gravity model needs the optional `pin` dependency (available through the `kinematics` extra). Its dedicated `SO107_gravity.urdf` retains SO-107's CAD centres of mass but uses masses scaled by link role from the SO-101 model, because the CAD export had uniform-density placeholders totaling 6.16 kg. The masses have not been weighed link by link. The shared `SO107.urdf` remains unchanged and is still used by visualization and Cartesian IK.

## Evidence and limits

On the left white-pair arm, 30 encoder-only static stations fitted compliance at P=32 of approximately 3.1, 3.2 and 3.4°/(N·m) for shoulder lift, elbow and wrist flex. The single 3.2 default covers those joints. A 60 mm horizontal circle at constant commanded height gave these encoder/FK results on 2026-09-19/20:

| Configuration                | In-plane RMS / max | Height mean / max             |
| ---------------------------- | ------------------ | ----------------------------- |
| P=16, I=0, no compensation   | 12.0 / 19.4 mm     | −33.1 / 93.4 mm; lift stalled |
| P=32, I=0, gravity alpha=3.2 | 4.7 / 5.8 mm       | −0.1 / 3.9 mm                 |

The measurements compare FK of encoder positions with FK of commanded positions. They cannot measure link flexure or fingertip error. The default gain was fitted on one left arm, and P=32 has not been retested under leader teleop. Test and refit for another assembly or gain setting. The shape and static-station measurements were performed in the sealed Show-and-Servo history; this PR does not claim a new hardware run.

## Measure and fit on an arm

The benchmark connects directly to the serial bus. Disconnect the arm from the GUI before running it. It logs tick-level commanded, present and sent joints, plus dwell stations and servo effort when available. It stops on excessive joint divergence, prolonged lag or a `STOP` file in the run directory. The start pose must be inside the IK workspace and away from the folded URDF seam.

```bash
uv run --extra kinematics python benchmarks/so107_gravity_sag.py --p 32 --ff-alpha 0 --start-joints -45 74 -41
uv run --extra kinematics python benchmarks/so107_gravity_sag.py --fit-stations outputs/gravity_sag/<run>
uv run --extra kinematics python benchmarks/so107_gravity_sag.py --p 32 --ff-alpha 3.2 --start-joints -45 74 -41
```

The fit command reads the first run's `stations.json` and `meta.json` offline. It rejects runs with feed-forward or integral action (onboard or outer) because their command/present residual no longer measures the uncompensated compliance. Its output reports a joint-wise and combined slope. For the confirmation run, compare the saved residuals before adopting a new default. The benchmark is fixed to the left-arm alignment; the `arm` config supports right-arm control, but the benchmark does not calibrate the right arm yet.

Servo position calibration and touch-based fingertip/camera calibration are separate from this gravity compliance fit. The latter tooling is in the `proto/show-and-servo` history and is not required to use this robot type.
