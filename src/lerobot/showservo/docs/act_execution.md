# Act execution: Progress and Milestones

Status: proposed
State of the work: no tracking issue yet

An act replays a demo on objects that have moved, and today it does so as a string of separate motions. Each key
frame is planned on its own, streamed, and waited on until the arm stands still at it; the remaining key frames are
planned again whenever a stage ends. At speed 1.0 an act takes 1.4 times the demo's time, stands still 12 to 16
times, and stood 2.5 s after a grasp while the carry was planned again. Execution only moves forward: if the object
is moved while the arm grasps or inserts, the act keeps going with the next action, and an act that failed to grasp
or dropped what it held has no way back. The same runner is meant to drive the OpenArm 2 as well as the SO-107.

**Proposal.** The act runs as one continuous replay of the demo, adapted to the scene. A single
[Progress](#glossary) marks how far along the demo's timeline the act is; a queue of
[pose entries](#glossary), each relative to the object its [leg](#glossary) follows, is filled from Progress a second
ahead; a command stage composes each entry with the object's newest pose and solves the IK a few ticks before the
motor loop sends it. An object update changes one pose in the composition, not the trajectory. [Milestones](#glossary)
— conditions on the observed state, carried by key frames or legs, such as "the gripper closed on the object" —
gate Progress: it cannot move past one that has not been met. When an object is moved, or a Milestone fails or stops
holding, Progress is placed again at the nearest state on the timeline it may reach: never past an unmet Milestone,
and back across met ones only when each of them may be repeated; otherwise the act stops and reports the failure to
whatever runs it. Choices that need many IK solves, such as which allowed turn to land at, are made beside the
motion and change the queue only where that is safe.

## Scope

In: executing one act from its first key frame to its last on one arm: the timeline, Progress, Milestones, the pose
queue, the command stage, choosing between allowed turns, speed, and what the act reports at its end.

Out, and why: how an object's pose is decided from the tracker's views, which
[act_loop.md](act_loop.md) designs and this design consumes; recording and editing demos, beyond the marks a
Milestone needs; the motor loop itself (`gui/api/jog.py`), which already sends the newest due command each tick;
two arms moving together, whose Progress would have to be coupled, a design of its own.

Non-goals: collision checking against the scene beyond the workspace box and floor the act checks today; planning a
path around obstacles; deciding what to do after a failure, which belongs to the agent above the act.

## Requirements

Safety first: Progress must never act on a state that is not there. Then the motion: an act that nothing disturbs
should look like the demo. Generality to other arms and diagnostics come last.

| #   | Pri | Requirement                                                                                                                                                                 | Target                                                                                                               | Why that target                                                                                                                                                                        | Checked by                                                                      |
| --- | --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| R1  | P0  | Progress never moves past a Milestone that has not been met                                                                                                                 | 0 crossings                                                                                                          | A missed grasp followed by a carry is the failure this design exists to stop ([O3](#o3))                                                                                               | Unit tests on the Progress update; every act's record                           |
| R2  | P0  | A Milestone that fails, or stops holding, moves Progress back to the leg that establishes it when every Milestone crossed may be repeated; otherwise the act ends as failed | Grasp missed: back to the approach; object dropped in the carry: back to it; a final Milestone crossed: the act ends | The cases named by the user on 2026-10-17; the recovery proposal in [manipulation_architecture.md](manipulation_architecture.md#proposed-key-frame-recovery-and-task-outcome-contract) | Simulated acts for each case                                                    |
| R3  | P0  | Only a view that places an object can move Progress                                                                                                                         | 0 Progress updates from views [act_loop.md](act_loop.md) turns away                                                  | A stale or conflicting view must not trigger a rewind (recovery proposal); views off by 7.6 to 29.8 mm were accepted by the tracker ([act_loop.md](act_loop.md) O3)                    | A test that feeds views that do not place the object                            |
| R4  | P0  | An object update never stops the commands                                                                                                                                   | No standstill of more than one tick caused by an update                                                              | 2.5 s after a grasp, 0.06 to 0.8 s at other stage ends, with the arm standing ([O2](#o2))                                                                                              | A simulated act whose target moves during each leg; the act's record on the rig |
| R5  | P1  | An act nothing disturbs, at speed 1.0, takes about the demo's time over the same legs                                                                                       | At most 1.15 × the demo's time                                                                                       | 1.38 × today ([E7](#e7)); what is left by design is the arm trailing its commands by about 0.1 s ([O5](#o5)) and the Milestone checks                                                  | Per-leg duration against the demo's, on the rig                                 |
| R6  | P1  | The arm stands still only where the demo did, or at a Milestone                                                                                                             | Standstills over 100 ms only there                                                                                   | 12 to 16 per act today, most at key frames ([O1](#o1))                                                                                                                                 | The act's command record                                                        |
| R7  | P1  | A moved object places Progress at the nearest state it may reach                                                                                                            | Object moved away during an insertion: back on the approach; moved closer: ahead, up to the next unmet Milestone     | The user's cases of 2026-10-17; today Progress only moves forward ([O3](#o3))                                                                                                          | Simulated acts with the object moved at each leg                                |
| R8  | P1  | Choosing between allowed turns never delays a command                                                                                                                       | A choice within 200 ms of the object moving; the queue keeps playing meanwhile                                       | Two tracker frames at the rig's 9.5 frames/s ([E8](#e8)); the stand that held the landing chosen again lasted 2.5 s ([O2](#o2))                                                        | The act's record of each choice                                                 |
| R9  | P1  | One engine drives the SO-107 and the OpenArm 2                                                                                                                              | Both through the same robot interface                                                                                | The OpenArm 2 is the next arm; the runner names the SO-107's motors today ([O9](#o9))                                                                                                  | The engine's tests run on both arms' kinematics                                 |
| R10 | P2  | Joint speed is bounded by what the arm can do, not by the jog's ready/park speed                                                                                            | The arm's measured limit; the lag guard expects a following error that grows with speed                              | The cap is 30 °/s; the demo moved joints at up to 158 °/s on the same arm ([O5](#o5))                                                                                                  | Lines at the demo's pace on the rig without a lag freeze                        |
| R11 | P2  | Every Progress update and Milestone decision is in the act's record, with its cause                                                                                         | One record per decision                                                                                              | Failures have been diagnosed from the act's own record ([E7](#e7)); a rewind nobody can see cannot be judged                                                                           | A test of the record                                                            |

Conditions: the rig is the SO-107 (left arm) with its RealSense camera on this desktop; the acts are the insertion
demo of 2026-10-17, listed in the appendix. IK timings were taken on the same desktop with the GUI idle.

## Observations

<a id="o1"></a>**O1. Every key frame is a stop.** Before the grasp, each key frame is planned on its own and streamed,
and the stream returns only when the arm is within 2° of its last command, or has stopped within 6°
(`_act_key_frames_task` in `gui/api/manipulation/execution.py`; the end check in `live_execution.stream`). Within a
stream, a line brakes at the key frame it ends on (`reactive.checked_joint_samples`): between the withdrawal and the
lift away the command fell from 26 to 3 °/s and rose again ([E3](#e3)). After the grasp the rest is planned as one but
streamed in pieces, each waited on. So stopping at key frames is built into how the act is cut, not into the demo.

<a id="o2"></a>**O2. Planning happens with the arm standing.** At each stage end the remaining key frames are planned
again from where the arm stands. The arm stood 0.06 to 0.8 s at those ends, up to 0.7 s of it settling under load, and
2.5 s after the grasp of act `20261017_130202`, which held two plans of the carry: the first failed at the landing turn
the start had taken, the second ranked every allowed turn ([E4](#e4)). Planning ahead of the arm
(commit `a3e0cf48c`) shortened these where its inputs still held. So retargeting must not depend on planning the
trajectory again.

<a id="o3"></a>**O3. Progress only moves forward.** The phase a stream aims at only increases
(`test_update_during_contact_retargets_remaining_phase_and_preserves_release` asserts it), and the runner walks the
key frames in order. A moved object re-aims the commands still queued; nothing re-runs an approach already passed. So
an object moved during a grasp is followed by a closing gripper, and a missed grasp by a carry.

<a id="o4"></a>**O4. IK is cheap; choosing among candidates is not.** One IK call takes 0.2 to 0.9 ms, a solve to
0.5 mm 0.3 to 2.6 ms, on both arms' kinematics; one call closes about half of a 10 mm step, so a solve repeats it 2 to
6 times ([E1](#e1)). Under the rig's camera and tracker load the act's solves took 1.7 ms at the median and 21 ms at
the 95th percentile ([E1](#e1)). Ranking the allowed turns runs these solves over every sample of every turn, and it
sat inside the 2.5 s stand in [O2](#o2). So IK can run for every tick, with a few ticks of slack, and ranking turns
cannot.

<a id="o5"></a>**O5. The arm can move several times faster than the act lets it, and it trails its commands.** In the
demo the shoulder, elbow and wrist reached 119 to 137 °/s; the following error grew with speed, about 2° at 10 to 30
°/s and 6° at 60 to 100 °/s, and the arm trailed its command by about 3 ticks ([E2](#e2)). The act limits each joint to
the jog's ready/park speed, 30 °/s, less a 1.5 margin for easing: its lines ran at 20 °/s ([E5](#e5)). The jog holds the
playback when a joint is more than 5° behind and freezes it after 0.5 s. So the joint limit has to come from the arm,
and the lag guard has to expect the error that the speed brings.

<a id="o6"></a>**O6. A servo takes the newest goal; only the act queues.** The SO-107's Feetech servos are written a
goal position each tick and move toward the newest; the OpenArm 2's Damiao motors take a target position, stiffness,
damping, zero velocity and gravity torque each tick (MIT mode), a pull toward the newest target. The jog's playback is
the only queue, and the jog limits any goal to 12° from the joint's reading. So replacing queued commands is always
safe, and a goal far beyond what a tick can cover is chased in joint space, off the Cartesian path.

<a id="o7"></a>**O7. The grasp never lands where it was aimed.** The hold seeded at the grasp was 4.6 to 7.4 mm and 1.2
to 1.8° from the demo's in every act of 2026-10-16/17 ([E6](#e6)). So the relation between gripper and object after a
grasp is a measured state, and what follows the grasp must be composed with it.

<a id="o8"></a>**O8. Each object's pose arrives at the tracker's rate, with its own error.** The rig's tracker ran at
about 9.5 frames/s ([E8](#e8)); [act_loop.md](act_loop.md) records views accepted by the tracker that were tens of
millimetres off. So a change of Progress needs a view the pose rule accepts, and a margin against jitter.

<a id="o9"></a>**O9. The two arms differ in kinematics, gripper and motor, not in what the act needs from them.** The
SO-107 runs Pink (`so107_description/cartesian_ik.py`); the OpenArm 2 follower runs the vendor's mink solver
(`openarm_description/mink_ik.py`), and a Pink build of its URDF exists beside it. The OpenArm 2 has 7 arm joints, so
its IK has a null space the SO-107's does not. The runner and the stream take the SO-107's motor names
(`so107_description.joint_alignment.MOTOR_NAMES`). So the engine needs forward and inverse kinematics, joint ranges and
speed limits, the gripper's units and closing direction, and nothing else from the arm, and today it has the SO-107's
motor names built in.

## Constraints and freedoms

<a id="c1"></a>**C1.** Commands are dense: one pose per motor tick, each within what the arm covers in a tick
([O5](#o5), [O6](#o6)).

<a id="c2"></a>**C2.** IK runs per entry, seeded from the previous command, a few ticks ahead of the motor loop and
not in its thread ([O4](#o4)).

<a id="c3"></a>**C3.** An object update is applied by composition: entries hold poses relative to their frame, and the
frame's newest pose is applied when the entry is solved ([O2](#o2), [R4](#requirements)).

<a id="c4"></a>**C4.** Choosing between candidates runs beside the motion and is applied at safe points ([O4](#o4),
[R8](#requirements)).

<a id="c5"></a>**C5.** Progress is a position on a continuous timeline, and Milestones are the only places it is
gated ([R1](#requirements), [R6](#requirements)).

<a id="c6"></a>**C6.** Free: how a Milestone's condition is measured, as long as it uses views the pose rule accepts
([R3](#requirements), [O8](#o8)); how far ahead the queue reaches; whether candidate feasibility is computed online or
looked up ([Q4](#open-questions)).

## Architecture

```
 demo + marks ──► timeline: legs, each relative to its frame; Milestones on key frames and legs
                     │
 observations ──►  Progress ──► generator: pose entries from Progress, ~1 s ahead, up to the next unmet Milestone
 (pose rule)         ▲                     │
                     │                     ▼
         Progress update          command stage (own thread, 2-3 ticks ahead): compose with the frame's newest
         (on a moved object,       pose and the hold, IK from the previous command, checks ──► jog playback
          a failed Milestone,                                                                      │
          an unreachable entry)                                                     motor loop (30 Hz) sends the newest due
                     ▲
 candidates (beside): allowed turns ranked when a frame moves; applied before a leg is entered
```

1. **The timeline.** Built once per act from the demo and its marks, and again when a candidate changes. A
   [leg](#glossary) runs between consecutive key frames and is held in its [leg frame](#glossary): an object's pose,
   turned by the leg's chosen candidate, or the world. A follow leg is the demo's samples of the
   [guided pose](#glossary) relative to the leg frame, with the gripper command and the demo's clock. A line leg is
   generated (item 3). Time on the timeline is the demo's time divided by the act's speed. ([C5](#c5), [O1](#o1))
2. **Progress.** A leg and a time within it. It advances as the motor loop sends the entries generated from it, so
   it follows what the arm was told, not the clock alone; it stops at a Milestone until the Milestone is met.
   ([R1](#requirements), [R6](#requirements))
3. **Pose entries.** One per motor tick, from Progress to about a second ahead. A follow entry is a pose relative to
   its leg frame. A line entry has two [anchors](#glossary): where the line starts, in the arm's base frame, fixed
   when the line is made, and where it ends, relative to its leg frame; plus how far along it is. When the object
   moves, the end moves with it and the start stays where the arm was, so the line bends toward the new end instead
   of jumping. Entries are paced at the demo's speed over the leg ([commit `e5988b6ff`](#e7)). ([C1](#c1), [C3](#c3))
4. **The command stage.** Its own thread, a few ticks ahead of the motor loop. For each entry it composes the pose:
   the leg frame's newest pose, the entry's relative pose, and, while an object is attached, the measured hold
   ([O7](#o7)). It solves the IK from the previous command, checks the joint step against the arm's limit, the
   workspace box, the floor and the joint ranges, and writes the command into the jog's playback, replacing what was
   queued for those ticks. An entry it cannot solve stops the queue there and asks for a Progress update.
   ([C2](#c2), [O4](#o4), [O6](#o6))
5. **Milestones.** A condition on the observed state, of one of a few kinds, none named for a task: _arrived_ (the
   guided pose within tolerance of a key frame in its leg frame), _attached_ (the gripper stopped short of its
   command on an object, and the object's pose follows the gripper once it moves), _released_ (the gripper open and
   the object no longer following), _related_ (one object's pose within tolerance of another's, as the demo left
   them). The gripper's closing and opening carry _attached_ and _released_ by default; any key frame may carry one
   marked in the demo editor. Each has a regression policy: _repeatable_, or _final_ for an effect that cannot be
   undone. A met Milestone stays watched while it matters: _attached_ until the release. ([R1](#requirements),
   [R2](#requirements))
6. **The Progress update.** Run when a view the pose rule accepts moves a leg frame by more than a margin, when a
   Milestone fails or stops holding, or when the command stage cannot solve an entry. For each leg Progress may
   reach, it expresses the guided pose now in that leg's frame and finds the demo sample nearest to it: distance in
   position, plus rotation and gripper opening weighted into the same units, plus a cost for moving away from where
   Progress is. The legs it may reach run from the last met Milestone to the next unmet one; earlier legs only when
   every Milestone between is repeatable, and those are then unmet again. If the nearest state needs a final Milestone
   crossed, the act ends as failed. Progress is placed there; the queue is regenerated: a line from the guided pose
   to that state, then the leg from there on. A change smaller than the margin is ignored, so jitter does not move
   Progress. ([R2](#requirements), [R3](#requirements), [R7](#requirements), [O3](#o3), [O8](#o8))
7. **Candidates.** A leg frame may have allowed turns, from the object's symmetry or the demo's landing rule. When
   its object moves, a task beside the motion ranks them: IK of the leg's key frames and a few of its samples, seeded
   from the current joints, ranked by reach and by distance from the demo's joints; the ranking is kept until the
   object moves again. A different turn is taken only before its leg is entered, and only when it ranks better by a
   margin, so the choice does not flip between frames. ([C4](#c4), [R8](#requirements), [O4](#o4))
8. **Speed.** The act's speed scales the demo's time everywhere: follows on the demo's clock, lines at the demo's pace
   between their key frames. Each arm declares its joint speed limit, measured; the lag guard allows a following error
   that grows with the commanded speed. ([R5](#requirements), [R10](#requirements), [O5](#o5))
9. **The arm.** One interface: forward and inverse kinematics, joint ranges and speed limits, the gripper's units and
   closing direction. The SO-107 and the OpenArm 2 each provide it. ([R9](#requirements), [O9](#o9))
10. **What the act reports.** Each Milestone met or failed, and each Progress update with what caused it, in the act's
    record; at its end, whether every Milestone was met, which one failed and why, or that a fault aborted it — kept
    apart, as the recovery proposal asks, so a motor fault after a placement is not reported as a failed task.
    ([R11](#requirements))

## Alternatives, and what this costs

**Keep the runner and plan ahead of the arm** (what `a3e0cf48c` and `d8d36d606` do). It shortens the stops at stage
ends where its inputs still hold, but keeps the stops at key frames ([O1](#o1)), plans again when an object moves
([O2](#o2)), and never goes back ([O3](#o3)).

**Plan whole joint trajectories and re-plan them fast**, for instance with batched GPU optimisation. Still needs
Progress and Milestones to know what to plan for; puts the act's planning on the GPU the trackers already share
([act_loop.md](act_loop.md) O6).

**A state machine authored per task, with its retry edges.** Every task would carry its own recovery, which the act
keeps out of the runtime: interactions are generic, named by the demo, never by a task.

**Progress as the nearest key frame.** Coarse, and wrong where the path doubles back on itself: over and then down
into a ring, the nearest key frame can be the wrong side of the descent. The time within a leg, with a cost for
moving, keeps the choice continuous.

**IK in the motor loop's thread.** Simplest, but a solve took 21 ms at the 95th percentile under load ([O4](#o4)), and
the motor loop's timing was the reason its own thread was cleaned of other work
([action_queue_timing.md](../../gui/docs/action_queue_timing.md)).

**A reachability map instead of ranking candidates online.** Common practice for choosing grasps and robot bases:
fill a map of position cells times orientations offline, by sampling the forward kinematics, and look a pose up in it.
Megabytes as bits for this workspace. It says a pose has an IK solution somewhere; it does not say the arm reaches it
from where it is without changing configuration, or within the servo ranges calibrated on this arm. Kept as an option
to screen candidates ([Q4](#open-questions)).

What the design costs. The runner, `_act_key_frames_task`, and the stream, `live_execution.stream`, are replaced rather
than edited. Per-tick IK finds a configuration change only when it comes, where a plan made in advance would have
seen it; the candidate ranking and the joint-step check stand in for that. Every Progress update depends on the views
the pose rule lets through; a bad view that passes it moves the arm. The Milestone checks that measure the hold take
seconds, and Progress waits on them. There are more threads to keep correct.

## Open questions

<a id="q1"></a>**Q1. Where do Milestones come from?** At the gripper's closing and opening automatically, and on any
key frame the operator marks; or only where marked. Automatic ones cover the cases in [R2](#requirements) without
editing every demo; marked ones are the only way to say "inserted" or "placed upright". Leaning: both.

<a id="q2"></a>**Q2. Which Milestones may be repeated by default?** Attaching and releasing can be undone by opening
the gripper and starting the approach again; a marked Milestone may stand for something that cannot. Leaning:
attached and released repeatable, marked ones repeatable unless marked final.

<a id="q3"></a>**Q3. How does the arm leave contact when Progress goes back?** Back along the demo's own path in the
leg frame, to the leg's start; or straight out along the approach. The demo's path is known to have been free of
collision relative to the object; the recovery proposal warns that reversing it blindly proves nothing if the object
was displaced. Leaning: back along the demo's path, with the same joint, workspace and floor checks, and the act
ending as failed if the arm cannot follow it.

<a id="q4"></a>**Q4. Rank candidates online only, or screen them with a precomputed reachability map?** Online is
built on the IK the act already uses and takes the arm's calibrated ranges into account; a map answers in
microseconds but per arm and per calibration ([Alternatives](#alternatives-and-what-this-costs)). Leaning: online,
with the map if the ranking misses [R8](#requirements)'s 200 ms.

To measure: the tracker's pose jitter on an object at rest, which sets the margin for a Progress update and the
weights in its distance; the OpenArm 2's following error and its mink IK's timing (its solver is not installed on
this desktop); the time to rank the ring's allowed turns; the command stage's latency on the rig under load.

## Glossary

- **Progress:** where along the act's timeline the act is: a leg, and a time within it.
- **Milestone:** a condition on the observed state that Progress cannot move past until it is met: _arrived_,
  _attached_, _released_ or _related_, carried by a key frame or a leg, _repeatable_ or _final_.
- **Leg:** the part of the demo between two consecutive key frames, as in [act_loop.md](act_loop.md).
- **Leg frame:** the pose a leg's motion is relative to: its object's, turned by the leg's chosen candidate, or the
  world.
- **Guided pose:** what a leg moves as the demo did: the fingertip, or the attached object while one is attached.
- **Pose entry:** one motor tick of the queue: a pose relative to its leg frame, or a line's two anchors and how far
  along it is, with the gripper command and the tick it is due.
- **Anchor:** one end of a line: its start in the arm's base frame, or its end relative to the leg frame.

## Appendix: evidence

Taken on 2026-10-17 on `proto/show-and-servo` (between `be2e1822e` and `d8d36d606`), from the insertion demo and its
acts' own records on the rig, and from benchmarks on the same desktop.

<a id="e1"></a>**E1. IK timing.** Pink, 50 QP iterations a call, targets 1, 10, 50 and 100 mm (0.5 to 30°) from a
working pose, median of 5 targets each; the GUI idle.

| Arm (kinematics)       | One call   | Solve to 0.5 mm       | Checked act step | One call leaves of a 10 mm step |
| ---------------------- | ---------- | --------------------- | ---------------- | ------------------------------- |
| SO-107 (Pink)          | 0.2–0.8 ms | 0.6–2.6 ms, max 4 ms  | 1.0–3.1 ms       | 5.2 mm                          |
| OpenArm 2 (Pink, URDF) | 0.2–0.9 ms | 0.3–1.6 ms, max 12 ms | 0.5–1.8 ms       | 1.6 mm                          |

FK: 3 to 5 µs. On the rig, during act `20261017_125408`, the act's checked steps: 1.7 ms median, 20.8 ms at the 95th
percentile, 74.8 ms at most (`stream_diagnostics.solve_ms`).

<a id="e2"></a>**E2. Joint speed and following error.** The insertion demo (teleoperated) reached 119 °/s at the
shoulder lift, 137 °/s at the elbow, 116 °/s at the wrist flex (observed). Following error against commanded speed:

| Commanded    | Demo, median / 95th percentile | Acts `125408`, `150843`, median / 95th percentile |
| ------------ | ------------------------------ | ------------------------------------------------- |
| 10–30 °/s    | 1.9 / 4.7°                     | 1.4 / 2.9°                                        |
| 30–60 °/s    | 3.4 / 22.6°                    | 2.0–2.2 / 2.5–3.1°                                |
| 60–100 °/s   | 6.3 / 9.7°                     | —                                                 |
| over 100 °/s | 10.6 / 14.7°                   | —                                                 |

The arm's reading trails its command by 2 to 4 ticks (67 to 133 ms).

<a id="e3"></a>**E3. Stops.** Command standstills over 100 ms between the first and last key frame: 15 (3.1 s) in act
`125408`, 16 (5.1 s) in `130202`, 12 (2.6 s) in `150843`. Command speed across the withdrawal-to-lift boundary in
`150843`, by tick: 26, 25, 23, 20, 17, 13, 8, 3, 6, 10, 14, 18, 20, 22, 23 °/s.

<a id="e4"></a>**E4. Planning with the arm standing.** Between the last sample of one stream and the first of the
next: 2.13 s in `125408` (1.09 s of it the arm settling), 3.53 s in `130202` (2.52 s after the grasp: the carry failed
at the start's landing turn — "the planned approach to key frame 6 has no solved joint path" — and every allowed turn
was ranked again), 1.41 s in `150843`.

<a id="e5"></a>**E5. The joint limit.** `jog.RAMP_DEG_S` = 30 °/s, "joint-space moves (ready, park): slow enough to
watch"; the act's step takes ticks for 1.5 times its mean joint speed. In `150843` every line ran at 0.64 to 0.67 ° a
tick: the lift away turned the shoulder 62° in 3.3 s, which the demo did in 1.2 s. The demo moved joints faster than
30 °/s on 31 of the lift away's 36 samples (at most 158 °/s). Bus clamp: `jog.MAX_STEP_DEG` = 12° from the reading.

<a id="e6"></a>**E6. The hold at the grasp.** The hold seeded at the grasp, against the demo's, in the eight acts of
2026-10-16 23:19 to 2026-10-17 13:02: 4.6 to 7.4 mm and 1.2 to 1.8°.

<a id="e7"></a>**E7. Act durations against the demo**, key frames 2 to 9 of the insertion: demo 13.66 s; act `125408`
19.77 s (at `be2e1822e`); act `150843` 18.89 s (at `a3e0cf48c`: lines at the demo's pace, follows free of the walk's
cap, the carry planned ahead). The insertion's follow: demo 4.81 s, 6.20 s, then 5.37 s.

<a id="e8"></a>**E8. Tracker rate.** About 9.5 frames/s on the rig with Point2Pose tracking, 2026-10-16
([action_queue_timing.md](../../gui/docs/action_queue_timing.md)).
