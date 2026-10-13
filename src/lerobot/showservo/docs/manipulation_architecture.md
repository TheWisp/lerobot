# Key-frame manipulation runtime

The manipulation API has one execution architecture. Every act validates its demo
reference poses, then executes the key-frame runner. `line`, `follow` and `lift`
are key-frame movement modes within that runner. The old marks runner and the
separate grasp/place/world planners have been removed.

Older recordings can still contain named marks such as `pregrasp`, `grasp_end`
and `preplace`. `showservo.motion.key_frames` converts that saved representation
into key frames. This is data-format compatibility, not another execution path.
Act requests cannot select a runner; an obsolete `runner` field is rejected.

## Object-independent interaction sequences

Objects are designated on the RGB-D recording and identified by user labels.
The recorder, worker, tracker and act runner do not select a policy by object
name or class. Shapes, poses and symmetry declarations belong to the demo.
Names in historical tests and failure comments describe evidence, not runtime
dispatch rules. The old `benchmarks/showservo_reset/reset_policy.py` experiment
is a separate, scene-specific reset program; it is not imported or invoked by
the demonstration framework and must not be used to reset a different demo.

`demo_holdings` reads every recorded attachment from gripper commands and
observations. Each grasp/release interval is executed in the same key-frame
runner, with its own measured or explicitly predicted attachment correction.
A held tool can follow several interaction segments before release. Completing
the first such segment does not detach it from the gripper in group tracking.
Later grasps repeat the reference, grasp and held-pose checks. A missing or
failed second grasp cannot inherit the first grasp's correction. The editor
shows every attachment interval and replay records each selection separately.

For a subsequent attachment interval, an object used again is referenced at its
visible pose before that interaction, after the preceding release. This avoids
applying the object's earlier demonstrated displacement twice. That reference
passes the same depth-surface verification as the initial reference. An unseen
later reference is refused, rather than silently reusing the original pose.
Grasp and release transitions need separate recorded follow segments; an
attachment is to the object that segment names. A held key frame must reference
another object or the world.

**New demo** clears the previous demo's live object state, labels and injected
error settings, without moving the arm or deleting saved data. A new recording
cannot overwrite a different saved demo with the same name. Record with the
leader or gizmo, save, designate the objects, then author the key frames. RGB-D
recording does not require choosing a tracking object beforehand.

This supports recording general hand-object and object-object motion sequences
on the single gripper, with rigid pose-based replay. It does not infer contact
forces, friction, deformation, liquid motion, or success for an unseen task.
Several contacts inside a streamed segment remain a replayed demonstrated path;
there is no general contact-dynamics solver or per-contact replanning. Occluded
held-object accuracy remains limited by the documented visual checks and the
explicit no-slip fallback. The sequence regressions use a simulated arm and
supplied poses; a new physical task still needs its own observed trial.

## Code ownership

`lerobot.gui.api.manipulation` assembles the endpoint routers. Code imports the
owning module explicitly; there is no `pregrasp.py` forwarding facade.

| Module                  | Responsibility                                                            |
| ----------------------- | ------------------------------------------------------------------------- |
| `runtime`               | Shared session data, configuration and dedicated executors                |
| `session`, `camera`     | State readout, frames and calibration from the GUI-owned camera           |
| `acquisition`, `worker` | Object registration, reacquisition and worker job transport               |
| `tracking`              | Geometry verification, acceptance, accepted pose history and following    |
| `holds`                 | Measured object-to-gripper relationships and checked demo reference poses |
| `planning`              | Object-relative key-frame composition, reach checks and landing choices   |
| `execution`             | Reference preflight, the key-frame runner and manual approach controls    |
| `demos`                 | Demo persistence, object designation and key-frame editing                |
| `recording`, `replay`   | Raw/control evidence, trial records, playback and exports                 |
| `rendering`, `groups`   | Overlays and conditional environment-group propagation                    |

Pure registration, rigid-motion calculations and key-frame sample planning live
in `lerobot.showservo.motion`, outside the GUI API. The worker entrypoint is
`benchmarks/manipulation_worker.py`. The HTTP prefix remains `/api/pregrasp` for
compatibility with existing clients and saved scripts; this name does not select
an execution architecture. Camera and arm ownership remain with the GUI.

## Measured-surface correction

P2P proposes a rigid pose. Before control accepts it, `tracking` compares the
reference object's visible surface with the current masked, aligned depth.
`showservo.surface_constraint` can align a supported matching planar patch's
normal and normal offset. This measures two orientation components and one
translation component; tangential position and twist remain from the proposal.
The correction rotates around the predicted reference patch centre and then
moves only along the observed normal. It does not assume a level table or an
upright object and does not change P2P's internal map.

Split-sample agreement, surface span, curvature and local overlap checks decide
whether that plane supports a correction. When a reference plane is supported
but the current plane is not verified, the observation is rejected. For a
non-planar reference, the general surface check may retain an unchanged proposal
when it agrees. Existing visibility, depth and centre-support gates still apply.
Group-motion comparison remains diagnostic; it is not a blanket pose override.

The same surface policy checks acquisition, secondary targets, observations used
for held-object correction, and the saved demo reference poses inverted during
planning. Raw proposals and actual control decisions remain separate in records
and replay. This prevents a corrected live pose from being compared against an
uncorrected, tilted demo reference.

See [pose verification](pose_verification.md) for measured results and limitations.
Visible surface agreement is not independent full-pose ground truth. In-hand
ambiguity and active inspection remain unresolved; this reorganization changes
neither the estimator policy nor its measured performance.

## Regression coverage

The key-frame runner is tested against expected grasp, carry, placement, drop,
reach, occlusion and hold-correction behavior. Planner tests verify reference
poses, recorded follow samples, timing, gripper commands, joint hints and floor
references directly. They no longer require a second implementation as an oracle.
Persisted mark-to-key-frame conversion, old replay records and the HTTP routes
retain their own compatibility checks.
