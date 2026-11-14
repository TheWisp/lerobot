# Gamepad-on-ring Apply, 2026-11-14

The recorded demo now compiles and completes in the motor-free controller replay.
The live-camera test acquired both references but stalled at the first placement
target with the wrist at its calibrated limit. No robot motors were connected.
Physical transfer is not yet verified.

[Video reconstruction](http://127.0.0.1:9103/static/flash_apply_stall_20261114.mp4):
left is the actual demonstration at the keyframe currently attempted; right is
the calibrated arm skeleton and tool target recomputed from the captured live
references over the final live camera preview. The right background is static.
This is explicitly a no-motion reconstruction, not a recording of robot execution.
It reproduces all 131 pickup keys completing and the first placement approach
(frame 570) stalling 7.47 mm short. The 14.87-second clip ends with a three-second
hold at the stall. `render_apply_replay.py` generates it and `video-trace.json`
preserves every simulated tick. No contacts or object movement are simulated.

**Follow-up inspection:** a solution for the first transferred placement exists
within the existing joint limits and 3 mm / 3° execution tolerance. The production
solver does not find it. A weighting-only diagnostic passes that target but does
not complete the placement. See the sealed-branch comparison and numeric checks
below; no production IK behavior was changed by this investigation.

## Reference handling

The 342 annotated frames are unchanged:

| Range   | Reference                  | Keys |
| ------- | -------------------------- | ---: |
| 420–550 | green gamepad at frame 420 |  131 |
| 570–780 | green ring at frame 570    |  211 |

Consecutive keys sharing a reference form a range. Compilation expresses each
measured tool pose relative to the reference at that range's entry. Execution
acquires a current published reference before entering the range and holds it
through its samples. A gap or a different reference starts another range. World
means the robot base. This preserves the recorded lift when the held gamepad
moves with the tool, and keeps the ring's missing pose at 601 from invalidating
placement. It does not continuously follow a separately moving placement target.

## Recorded-demo controller check

`check_recorded_demo.py` reads the actual LeRobot episode and tracking output.
It uses the selected robot's calibrated FK/IK and the production executor.
Robot connect/send methods raise if called; actions instead update simulated
joint state. Perception supplies only recorded range-entry references, with no
observations during the ranges. This is not a contact or perception test.

| Input                                    | Completed | Simulated duration |
| ---------------------------------------- | --------: | -----------------: |
| Recorded placement                       |   342/342 |             22.2 s |
| Both references shifted 10 mm in robot X |   342/342 |             22.4 s |
| Approach from recorded frame 0           |   342/342 |             26.2 s |

Every completed sample is within the existing 3 mm / 3° controller tolerance;
the maximum rotation residual is 2.74°. Gripper commands are retained. Duration
is the simulated 30 Hz servo clock with speed limits, not a measured robot run.
Per-frame results are in `result.json`; `compiled-plan.json` preserves the exact
input to the executor.

The first run stopped after 339 keys. The target at 778 was within calibration,
but IK did not know the calibrated limits: it proposed wrist flex 0.071° beyond
the lower limit, which the executor refused. The unchanged motor limits now
reach Pink in calibrated URDF coordinates. FK still describes measured joints,
including measurements outside planning limits. No limit was widened.

## Live-camera test

On port 9103, Apply prepared both ranges and Test run used the actual RealSense,
selected depth model, and shared tracker. Both references were published as
observed. The 131 pickup keys completed; placement stalled at frame 570.

The captured target is about 7.5 mm from the final simulated tool pose, with
wrist flex at -93.313° (configured lower bound -93.319°). IK from the stalled
seed also leaves about 7.5 mm residual when given the full target. This is
evidence of a constrained IK stall, not proof that no solution exists anywhere
in joint space. The current reference differs from the recorded entry reference;
whether the transformed target or its tracked reference should change is not
established. The captured state is `live-execution.json`.

Previously the controller kept sending the unchanged joints and displayed
"moving". It now sends no action and reports "waiting: IK made no progress"
with the remaining translation and rotation. The same captured target reproduces
that message in `live-stall-probe.log`. Stop ended the camera worker normally.

## Sealed-branch comparison and target inspection

Read from the actual seals, not their later descendants:

| Seal                                                           | Recorded failure                                                                                                                                                    | Implemented response                                                                                                                                                   | Relationship to this run                                                                                                                                                    |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `80d90b783`, first `proto/show-and-servo`                      | October 8 placement asked for wrist flex −94° against a calibrated −93° limit. Later approaches commanded wrist roll +125–127° while the encoder stopped near +97°. | Intersect the model and servo/profile limits inside IK; use the same checked joint stream for execution. Refuse required targets whose achieved residual is too large. | The calibrated-limit wiring bug found in the new editor had already been addressed here. The new wiring restores this invariant; current stall occurs with bounds enforced. |
| `flash-redesign-sealed-20261031`, second `docs/flash-redesign` | October 20 19:11: dowel grasp/lift succeeded, then the carry failed because a local IK solve tried the distant endpoint first.                                      | Subdivide failed spatial steps, preserving the 3 mm / 3° checks. The recorded carry passed offline afterward.                                                          | A failed local solve is not proof of unreachability. The present executor already takes small Cartesian steps, so merely copying subdivision does not address this stall.   |

Primary historical sources:

- First seal `src/lerobot/showservo/docs/stage1_approach.md:1299`,
  `docs/act_loop.md:352–384` under `src/lerobot/showservo/`,
  `src/lerobot/gui/api/jog.py:_limit_to_servos`, and
  `src/lerobot/showservo/motion.py:solve_plan_joints`.
- Second seal `src/lerobot/policies/flash/docs/evidence_status.md:338–354`,
  `src/lerobot/policies/flash/planner.py:428`, commit `6c041becd`.

The first seal also allowed **explicitly annotated** best-effort waypoints to
end at their achieved pose and searched operator-permitted landing turns. Those
are different task semantics, not a generic fix for a required placement. This
inspection does not introduce either behavior into the simple editor.

### Captured target, without hardware

The transfer composition agrees exactly with
`live_base_from_ring @ reference_from_tool`. Relative to the recorded frame-570
tool pose, the captured target moves (−4.39, −11.77, −2.41) mm and rotates 10.06°.
This checks the transform arithmetic; it does not establish that the tracked
ring rotation is physically correct.

`inspect_target.py` reconstructs the calibrated model without connecting motors.
It uses the same captured target and actual joint limits. Results are in
`target-inspection.json`:

| Solve for frame 570                                                     | Position error | Rotation error | Within existing tolerance?       |
| ----------------------------------------------------------------------- | -------------: | -------------: | -------------------------------- |
| Current Pink objective, 50 iterations allowed                           |        7.50 mm |         0.053° | No                               |
| Current objective, 500 allowed                                          |        7.50 mm |         0.053° | No; both stop after 2 iterations |
| Pink position/orientation costs scaled by existing 3 mm / 3° tolerances |        1.39 mm |          2.76° | Yes                              |
| Independent bounded least-squares, actual tool point                    |        0.67 mm |          2.15° | Yes                              |

The current objective compares metre and radian errors with equal numeric costs.
It preserves orientation much more tightly than required while missing position.
Increasing iterations or removing posture cost does not resolve the first target.
The independent solve supplies a constructive counterexample to calling that
target unreachable. Its additional seeds are diagnostics, not a production retry
strategy. An additional diagnostic ignoring the calibrated bounds needs about
−101.2° wrist flex; that result is **not** a permissible robot command.

`compare_target_path.py` runs the production controller with the two captured
range-entry references and ideal simulated motor feedback. It only patches the
Pink cost in its own process:

| Objective        | Keys completed | First remaining key         |
| ---------------- | -------------: | --------------------------- |
| Current          |        131/342 | 570; 7.5 mm / 0.1° residual |
| Tolerance-scaled |        138/342 | 577; 3.5 mm / 3.1° residual |

Thus the weighting trial is **not a completed fix**. Pink optimizes the upstream
URDF anchor and the controller judges the measured tool point, and the motor loop
requests an intermediate goal rather than the full remaining target. These are
important differences from the independent tool-space solve; their individual
contributions have not been separated.

`inspect_placement.py` independently minimizes the larger of the two normalized
tool errors with the same joint bounds (and a 0.006° inset). It finds solutions
within tolerance for sampled frames 570, 577, 590, 610, 650, 700 and 740. Frame 577
has a solution with 1.81 mm / 1.81° error. At final frame 780 its local solution
remains 6.73 mm / 6.73° off. That last result is **not** a proof of global
unreachability, and isolated feasible endpoints do not establish a continuous,
collision-free or physically successful path. Results: `placement-feasibility.json`.

No motor connection, tracker change, limit widening or production cost change
was made. These diagnostics use `PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=2` and the same `coarse-open3d-env/bin/python` as the GUI.

## Verification

Focused suites cover editor Apply, episode compilation, the actual controller,
SO-107 calibration/kinematics, robot geometry and Pink. `tests.log` is the result.
`check_mutations.py` restores each defect in process; each named regression fails:

- Per-keyframe moving reference erases the demonstrated lift.
- Reacquiring every motor tick changes the reference during the range.
- Omitting calibrated IK limits overshoots the wrist limit.
- Calling an unchanged IK result "moving" sends a misleading command.

See the four `mutation-*.log` files. No tracker algorithm was changed. Only the
editor section of the shared local TODO was updated; no GitHub issue was opened.
