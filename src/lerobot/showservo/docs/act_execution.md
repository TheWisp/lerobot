# Act execution: Progress, Milestones and the state checks

Status: proposed
State of the work: no tracking issue yet

An act replays a demo on objects that have moved, and today it does so as a string of separate motions. Each key
frame is planned on its own, streamed, and waited on until the arm stands still at it; the remaining key frames are
planned again whenever a stage ends. At speed 1.0 an act takes 1.4 times the demo's time, stands still 12 to 16
times, and stood 2.5 s after a grasp while the carry was planned again. Execution only moves forward, and nothing
checks what the gripper did: if the object is moved while the arm grasps or inserts, the act keeps going with the
next action; an act that failed to grasp carries nothing to the target; a placed object that falls over is reported
as placed. The same runner is meant to drive the OpenArm 2 as well as the SO-107.

**Proposal.** A new act engine, beside the current runner and chosen per act, runs the act as one continuous replay
of the demo adapted to the scene. A single [Progress](#glossary) marks how far along the demo's timeline the act is.
[Pose entries](#glossary), one per motor tick and each relative to the object its [leg](#glossary) follows, are made
from Progress and solved a few ticks before the motor loop sends them, with the object's newest pose: an object
update changes one pose in the composition, not the trajectory. [Milestones](#glossary) sit at the gripper's closing
and opening and at any key frame the operator marks. Each is a [predicate](#glossary) from a small vocabulary —
`follows`, `present_at`, `related`, `resting` — read from the camera as yes, no or cannot tell, against what the
demo shows at that point: at its closing the object comes with the gripper; at its opening it stays where it was
let go, relative to what it was let go on, after the gripper has left. A Milestone whose check says no moves
Progress back to the leg that establishes it, when every Milestone crossed may be repeated; otherwise the act ends
and reports the failure to whatever runs it.

## Scope

In: executing one act from its first key frame to its last on one arm: the timeline, Progress, Milestones and their
state checks, the pose queue, the command stage, choosing between allowed turns, speed, what the act reports at its
end, and how the engine sits beside the current runner.

Out, and why: how an object's pose is decided from the tracker's views, which [act_loop.md](act_loop.md) designs and
this design consumes; recording and editing demos, beyond the marks a Milestone needs; the motor loop itself
(`gui/api/jog.py`), which already sends the newest due command each tick; two arms moving together, whose Progress
would have to be coupled, a design of its own; checks that need a model of cause, such as "the arm knocked
something over", since nothing in the demo says which contacts were meant; coupled motions such as a screw's
advance per turn, which need a law fitted to the demo.

Non-goals: collision checking against the scene beyond the workspace box and floor; planning a path around
obstacles; deciding what to do after a failure, which belongs to the agent above the act; telling a corner of a
cloth from an edge, which takes touch ([state_check_research.md](state_check_research.md)).

## Requirements

Safety first: Progress must never act on a state that is not there, and a check that cannot tell must not say yes.
Then the motion: an act that nothing disturbs should look like the demo. Latency of the checks, generality to other
arms and diagnostics come last.

| #   | Pri | Requirement                                                                                                                                                                 | Target                                                                                                               | Why that target                                                                                                                                                                          | Checked by                                                                               |
| --- | --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| R1  | P0  | Progress never moves past a Milestone whose check says no, and past one whose check cannot tell only where that Milestone allows it                                         | 0 crossings                                                                                                          | A missed grasp followed by a carry is the failure this design exists to stop ([O3](#o3))                                                                                                 | Unit tests on the Progress update; every act's record                                    |
| R2  | P0  | A Milestone that fails, or stops holding, moves Progress back to the leg that establishes it when every Milestone crossed may be repeated; otherwise the act ends as failed | Grasp missed: back to the approach; object dropped in the carry: back to it; a final Milestone crossed: the act ends | The cases named by the user on 2026-10-17; the recovery proposal in [manipulation_architecture.md](manipulation_architecture.md#proposed-key-frame-recovery-and-task-outcome-contract)   | Simulated acts for each case                                                             |
| R3  | P0  | A check says yes only on evidence that separates the outcomes                                                                                                               | 0 yes on the recorded misses and on negatives made by substitution                                                   | A false yes is the missed grasp followed by a carry of [R1](#requirements), with the check's approval                                                                                    | Unit tests; the replay of the act records ([E19](#e19))                                  |
| R4  | P0  | Only a view that places an object can move Progress                                                                                                                         | 0 Progress updates from views [act_loop.md](act_loop.md) turns away                                                  | A stale or conflicting view must not trigger a rewind (recovery proposal); views off by 7.6 to 29.8 mm were accepted by the tracker ([act_loop.md](act_loop.md) O3)                      | A test that feeds views that do not place the object                                     |
| R5  | P0  | An object update never stops the commands                                                                                                                                   | No standstill of more than one tick caused by an update                                                              | 2.5 s after a grasp, 0.06 to 0.8 s at other stage ends, with the arm standing ([O2](#o2))                                                                                                | A simulated act whose target moves during each leg; the act's record on the rig          |
| R6  | P1  | An act nothing disturbs, at speed 1.0, takes about the demo's time over the same legs, whatever the demo's frame rate                                                       | At most 1.15 × the demo's time, for demos recorded at 15 to 60 frames/s                                              | 1.38 × today ([E7](#e7)); what is left by design is the arm trailing its commands by about 0.1 s ([O5](#o5)); the demos on record run at 29.93 to 30 frames/s, other cameras at 15 or 60 | A simulated act at each frame rate; per-leg duration against the demo's on the rig       |
| R7  | P1  | The arm stands still only where the demo did, at a Milestone that waits, or after a check said no                                                                           | Standstills over 100 ms only there                                                                                   | 12 to 16 per act today, most at key frames ([O1](#o1))                                                                                                                                   | The act's command record                                                                 |
| R8  | P1  | A moved object places Progress at the nearest state it may reach                                                                                                            | Object moved away during an insertion: back on the approach; moved closer: ahead, up to the next unmet Milestone     | The user's cases of 2026-10-17; today Progress only moves forward ([O3](#o3))                                                                                                            | Simulated acts with the object moved at each leg                                         |
| R9  | P1  | A Milestone's check answers soon after the demo's own motion makes the answer visible                                                                                       | At the closing: within 1.0 s of the fingertip moving 30 mm; at the opening: within 2.0 s of the gripper clearing     | The demo's lift moved 35 mm 0.57 s after its closing ([E16](#e16)); tracker frames come at 9.5/s ([E8](#e8)); the placed check needs five frames over 0.5 s ([C7](#c7))                  | The replay of the act records ([E19](#e19)); simulated acts                              |
| R10 | P1  | Choosing between allowed turns never delays a command                                                                                                                       | A choice within 200 ms of the object moving; the queue keeps playing meanwhile                                       | Two tracker frames at the rig's 9.5 frames/s ([E8](#e8)); the stand that held the landing chosen again lasted 2.5 s ([O2](#o2))                                                          | The act's record of each choice                                                          |
| R11 | P1  | One engine drives the SO-107 and the OpenArm 2                                                                                                                              | Both through the same arm interface                                                                                  | The OpenArm 2 is the next arm; the runner names the SO-107's motors today ([O9](#o9))                                                                                                    | The engine's tests run on both arms' kinematics                                          |
| R12 | P1  | The engine runs beside the current runner, chosen per act, and leaves it unchanged                                                                                          | Either runner from the same page and API; the current runner's tests pass unchanged                                  | The engine replaces `_act_key_frames_task` and `live_execution.stream`; acts with each on the same rig and scene are what decides the switch                                             | The existing tests; the act's record names its runner                                    |
| R13 | P2  | Joint speed is bounded by what the arm and the motor loop can do, not by the jog's ready/park speed                                                                         | The live command bound, 90 °/s on the SO-107; the timeline slows where a step would exceed it                        | The cap is 30 °/s; the demo moved joints at up to 158 °/s on the same arm ([O5](#o5)); the jog bounds each live command to 3° a tick ([O6](#o6))                                         | A simulated act whose demo is faster than the bound; lines at the demo's pace on the rig |
| R14 | P2  | Every Progress update and Milestone verdict is in the act's record, with its evidence                                                                                       | One record per decision                                                                                              | Failures have been diagnosed from the act's own record ([E7](#e7)); a rewind nobody can see cannot be judged                                                                             | A test of the record                                                                     |
| R15 | P2  | A check can take another sensor, a fingertip skin or a camera on the gripper, without changing the Milestones                                                               | A new evidence source is one function                                                                                | The SO-107's gripper has no touch; cloth Milestones need it ([state_check_research.md](state_check_research.md))                                                                         | A test that adds a source                                                                |

Conditions: the rig is the SO-107 (left arm) with its RealSense camera on this desktop; the acts are the insertion
demo of 2026-10-17 and the acts of 2026-10-15 to 17 with saved frames, listed in the appendix. IK timings were taken
on the same desktop with the GUI idle.

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
6 times ([E1](#e1)). Under the rig's camera and tracker load the act's solves took 1.7 ms at the median, 21 ms at
the 95th percentile and 75 ms at most ([E1](#e1)). Ranking the allowed turns runs these solves over every sample of
every turn, and it sat inside the 2.5 s stand in [O2](#o2). So IK can run for every tick, a few ticks ahead, and
ranking turns cannot.

<a id="o5"></a>**O5. The arm can move several times faster than the act lets it, and it trails its commands.** In the
demo the shoulder, elbow and wrist reached 119 to 137 °/s; the following error grew with speed, about 2° at 10 to 30
°/s and 6° at 60 to 100 °/s, and the arm trailed its command by about 3 ticks ([E2](#e2)). The act limits each joint to
the jog's ready/park speed, 30 °/s, less a 1.5 margin for easing: its lines ran at 20 °/s ([E5](#e5)). The jog holds the
playback when a joint is more than 5° behind and freezes it after 0.5 s. So the joint limit has to come from the arm,
and the lag guard has to expect the error that the speed brings.

<a id="o6"></a>**O6. A servo takes the newest goal; only the act queues.** The SO-107's Feetech servos are written a
goal position each tick and move toward the newest; the OpenArm 2's Damiao motors take a target position, stiffness,
damping, zero velocity and gravity torque each tick (MIT mode), a pull toward the newest target. The jog's playback is
the only queue: it sends the newest due sample each tick, never skipping a gripper start, stop or reversal, and a
live playback's command moves at most 3° a tick from the one before (`jog._live_joint_command`); any goal is held to
12° from the joint's reading. So replacing queued commands is always safe, and a goal further than a tick covers is
chased in joint space, off the Cartesian path.

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

<a id="o10"></a>**O10. The gripper's own readings tell a hold from a miss within a quarter second, but not what is
held.** Its reading settles 0.10 s after the closing command ends, and its load, once the command is still, sits at
128 to 240 with an object between the fingers and at most 34 without, on both demos' objects ([E9](#e9),
[E10](#e10)). They say that something stops the fingers, never which object or that it comes along; the field
answers "what is pinched" with touch ([state_check_research.md](state_check_research.md)). So the readings are
evidence a check may add, not the one it rests on.

<a id="o11"></a>**O11. The gripper's load is effort, not contact force.** While the command moves and the reading
trails it by 4 to 5 units, the load reads 210 to 264 with the fingers open on nothing ([E11](#e11)). So the load is a
hold signal only while the command is still and the reading has settled.

<a id="o12"></a>**O12. The camera sees an object come along with the gripper soon, and its pose in the gripper late
or never.** The object's colour tracks move with the fingertip's projection, 0.84 to 1.0 of it when held and 0.01 to
0.03 of it at rest, once the fingertip has moved 25 px; the demo's lift gets there in about 0.5 s ([E15](#e15),
[E16](#e16)). The resting place reads empty 4.5 s after the closing, once the carry has moved the arm off it
([E12](#e12)). The first view that placed the held object came 10 to 22 s after the closing, in 8 acts of 57
([E10](#e10)); the tracker's 3D pose of the held object was wrong by hundreds of millimetres, and a pose from the
tracks alone is degenerate on a thin object ([E17](#e17)); after the release, the object's reference view placed it
in 0 of 33 views ([E13](#e13)). So "it came with the gripper" is a fast camera check, and "it sits as the demo's did"
cannot wait for a pose of the object.

<a id="o13"></a>**O13. The pose the act aims with is not steady at the millimetre.** Trusted views of the ring at
rest spread 15 to 20 mm at the 95th percentile within an act; the pose the act aimed its insertion with drifted
13 mm over 2.3 s while the ring stood still ([E14](#e14)). So "the object moved" needs a margin above that spread,
and a check that depends on the object's pose is no better than it.

<a id="o14"></a>**O14. Games and simulators need few predicates.** Held, in a place, on or at a relation, part moved,
at rest, and their order cover the tasks the benchmarks ship; two of their success checks count a placement only once
the effector has stopped acting on it, and the VR grab-break rule declares a grab broken when the object drifts past
a threshold from where hand times grab offset puts it, assuming no rigidity
([state_check_research.md](state_check_research.md#how-games-and-simulators-define-the-same-predicates)). They read
contacts and exact poses; none needs to say "cannot tell". So a handful of predicates covers carry, stack and place,
and each needs a camera version with a third answer.

<a id="o15"></a>**O15. Right after a grasp the camera often cannot tell.** In 20 of 49 acts the gripper covered all
but 0 to 4 of the object's tracks; the resting place is covered by the arm until the carry moves off it; the placed
object is not placed by its reference view ([E15](#e15), [E12](#e12), [E13](#e13)). So "cannot tell" is a common
answer, and each Milestone must say what Progress does on it.

<a id="o16"></a>**O16. The demo is the only statement of success, and the camera sits in the robot's frame.** No
model of tables, trays or supports exists, and none is in the demo: a deliberate drop and a slip look alike unless
the demo shows which was meant. The camera is calibrated to the arm's base (`camera._t_base_cam`), which is not a
fixed world on a robot that moves. So every predicate is relative to a reference that perception tracks — the
gripper, another object, or the base — and every expectation is read off the demo.

## Constraints and freedoms

<a id="c1"></a>**C1.** Commands are dense: one pose per motor tick, each within what the arm covers in a tick
([O5](#o5), [O6](#o6)).

<a id="c2"></a>**C2.** IK runs per entry, seeded from the previous command, a few ticks ahead of the motor loop and
not in its thread ([O4](#o4)).

<a id="c3"></a>**C3.** An object update is applied by composition: entries hold poses relative to their frame, and the
frame's newest pose is applied when the entry is solved ([O2](#o2), [R5](#requirements)).

<a id="c4"></a>**C4.** Choosing between candidates runs beside the motion and is applied at safe points ([O4](#o4),
[R10](#requirements)).

<a id="c5"></a>**C5.** Progress is a position on a continuous timeline, and Milestones are the only places it is
gated ([R1](#requirements), [R7](#requirements)).

<a id="c6"></a>**C6.** Free: how far ahead the queue reaches; whether candidate feasibility is computed online or
looked up ([Q4](#open-questions)); which evidence a predicate reads, as long as it answers in the three values
([R15](#requirements)).

<a id="c7"></a>**C7.** Each predicate answers yes, no or cannot tell, from whichever evidence it has, with thresholds
taken from measurements on the rig; a verdict that moves Progress must repeat on consecutive frames ([O8](#o8),
[O13](#o13), [O15](#o15), [R3](#requirements)).

<a id="c8"></a>**C8.** Expectations come from the demo's gripper and key frames, relative to a tracked reference: from
the closing, the object follows the gripper; from the opening, it stays where it was let go, relative to the frame it
was let go in, and keeps staying once the gripper has left ([O14](#o14), [O16](#o16)).

<a id="c9"></a>**C9.** The engine shares the jog's playback and the perception with the current runner and nothing
else; it holds no GUI state ([R12](#requirements), [R11](#requirements)).

## Architecture

```
 demo + marks ──► timeline: legs, each relative to its frame; Milestones at the gripper's events and marked key frames
                     │
 scene log  ──►  Milestone checks ──► Progress ──► generator: entries from Progress, up to the lead, not past a gate
 (views, tracks,  (predicates:         ▲   │                 │
  depth, arm)      yes / no / ?)       │   │                 ▼
                                       │   │      command stage: compose with the frames' poses (rate-limited) and
                    Progress update ───┘   │      the hold, IK from the previous command, checks ──► jog playback
                    (a no, a moved frame,  │                                                           │
                     an unreachable entry) │                                         motor loop (30 Hz) sends the newest due
                                           ▼
                                 the act's record: every verdict, update and entry
```

1. **The timeline.** Built once per act from the demo and its marks. A [leg](#glossary) runs from one key frame to the
   next and is held in its [leg frame](#glossary): an object's pose, turned by the leg's chosen candidate, or the
   base. A follow leg is the demo's fingertip samples, with the gripper command, the demo's joints and its clock. A
   line or lift leg is generated (item 3). ([C5](#c5), [O1](#o1))
2. **Progress.** A leg and how far into it, 0 to 1. It is the position of the last entry the motor loop has sent, so
   it follows what the arm was told, not the clock alone. ([R1](#requirements), [R7](#requirements))
3. **Pose entries and the clock.** One entry per motor tick, made from Progress up to the [lead](#glossary). Each
   advances the timeline by the act's speed over the leg's duration, so a demo recorded at any frame rate plays in
   its own wall-clock time: a follow's duration is the demo's, its pose interpolated between the demo's samples; a
   line's is its length at the demo's pace between its key frames (`core.demo_pace`, `core.walk_scale`). A follow
   entry is the demo's fingertip pose at that moment, relative to its leg frame. A line entry has two
   [anchors](#glossary): its start in the base frame, fixed when the line is entered, and its end relative to its leg
   frame; when the object moves the end moves with it and the start stays, so the line bends toward the new end
   instead of jumping. A line eases in only where the arm starts from standing and eases out only where the demo
   stood still at its key frame or the next leg starts by standing, so the arm passes key frames at speed. Gripper
   commands are the demo's, interpolated. ([C1](#c1), [C3](#c3), [R6](#requirements), [R7](#requirements))
4. **The command stage.** A thread beside the motor loop, solving each entry when it comes within the lead, four ticks
   (133 ms), of being due: longer than the 75 ms the slowest solve took on the rig ([E1](#e1)). It composes the
   entry's pose — the leg frame's pose, the entry, and while an object is attached the measured hold ([O7](#o7)) —
   solves the IK seeded from the previous command plus the demo's joint change over the same moment, and checks the
   joint step against the arm's bound, the workspace box, the floor and the joint ranges. A step over the bound is
   made shorter by advancing the timeline less on that tick, so the arm slows on the path instead of leaving it
   ([R13](#requirements)); a step that stays over the bound however short is a configuration change, and stops the
   queue. Solved entries are appended to the jog's live playback, which keeps its own clock and lag guards.
   ([C2](#c2), [O4](#o4), [O6](#o6))
5. **Retargeting.** A frame's pose reaches the composition through a rate limit, 0.10 m/s and 60 °/s: a correction the
   size of the pose rule's spread (20 mm, [O13](#o13)) is spread over six ticks, about 3 mm each, so a newer view moves
   the commands without a step. Entries already solved keep the pose they were solved with; the lead bounds how old it
   is. ([R5](#requirements), [C3](#c3))
6. **Milestones.** Each has a kind (item 7), the object it is about, its [reference](#glossary), and four timeline
   positions: where its check begins, its gate (Progress waits there for a yes, or for its rule on cannot-tell), where
   its watch ends (a no until then is a failure), and where Progress restarts after a no. Every closing and opening
   of the demo's gripper carries one, and any key frame may carry one marked in the demo editor. Each is
   _repeatable_, or _final_ for an effect that cannot be undone; a repeatable one is tried at most a set number of
   times. ([R1](#requirements), [R2](#requirements), [C8](#c8))
7. **The state checks.** A Milestone's check combines predicates from a small vocabulary over the scene log; the
   vocabulary and how each kind of Milestone uses it are the next section. ([C7](#c7), [O14](#o14))
8. **The Progress update.** Run when a check says no, when a view the pose rule accepts moves the frame of the follow
   leg Progress is in by more than a margin, or when the command stage cannot solve an entry. On a no, Progress goes to
   the failed Milestone's restart when every Milestone between there and Progress is repeatable and has tries left,
   and those are pending again; otherwise the act ends as failed. On a moved frame, the guided pose now is expressed in
   that leg's frame and matched to the nearest demo sample — distance in position, plus rotation weighted into the
   same units, plus a cost for moving away from where Progress is — among the samples between the last met Milestone
   and the next unmet one; Progress is placed there when that beats staying by a margin. Either way the queue after
   the last sent entry is dropped, and a [transition](#glossary) leads from the last command to the new state: the
   gripper opens first if it must open, the arm moves on a line at the walk's pace times the act's speed, and the
   gripper closes last if it must close. ([R2](#requirements), [R4](#requirements), [R8](#requirements), [O3](#o3),
   [O8](#o8))
9. **Candidates.** A leg frame may have allowed turns, from the object's symmetry or the demo's landing rule. When
   its object moves, a task beside the motion ranks them: IK of the leg's key frames and a few of its samples, seeded
   from the current joints, ranked by reach and by distance from the demo's joints; the ranking is kept until the
   object moves again. A different turn is taken only before its leg is entered, and only when it ranks better by a
   margin, so the choice does not flip between frames. ([C4](#c4), [R10](#requirements), [O4](#o4))
10. **Speed.** The act's speed scales the demo's time everywhere: follows on the demo's clock, lines at the demo's pace
    between their key frames, transitions at the walk's pace. ([R6](#requirements), [O5](#o5))
11. **The seams.** The engine (`lerobot.showservo.engine`) holds no GUI state and talks to three things: an arm
    (forward and inverse kinematics, joint names, the gripper's column and closing direction, joint ranges, the
    workspace and the per-tick joint bound), a playback (the jog's live queue: what was sent, append, replace), and a
    scene log that perception writes into (each frame's pose, accepted views, an object's 2D tracks, depth frames, the
    arm's observed joints, the camera's intrinsics and its pose in the base). The GUI's adapter fills them from the
    jog, the tracker's per-frame results and the act's finds, and the act request names the runner. ([C9](#c9),
    [R11](#requirements), [R12](#requirements), [O9](#o9))
12. **What the act reports.** Each verdict with the evidence that gave it, each Progress update with what caused it,
    and each entry sent, in the act's record; at its end, whether every Milestone was met, which one failed and why,
    which were passed without a verdict, or that a fault aborted it — kept apart, as the recovery proposal asks, so a
    motor fault after a placement is not reported as a failed task. ([R14](#requirements))

## The state checks

A check reads the scene log — what perception reported, each with the time its frame was read — and answers yes, no
or cannot tell, with the evidence. It needs no model of the scene beyond the objects the demo names, the gripper,
and the base ([O16](#o16)). The thresholds below are the rig's measurements, named where they come from.

**The vocabulary.**

| Predicate                | Question                                                           | Evidence it reads                                                                                                                                                                                                                                                                                                                        | Yes / no / cannot tell                                                                                                                                                                                                                                                                                                                                               |
| ------------------------ | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `follows(obj, gripper)`  | Since a reference moment, has the object moved as the gripper did? | 2D: the object's tracks visible at the reference frame and now, their median displacement against the fingertip's projected displacement over the same frames. 3D: views that place the object, against where the gripper's motion since the reference takes its pose then — the grab-break rule, with no rigidity assumed ([O14](#o14)) | Cannot tell until the fingertip has moved 25 px in the image (3D: 30 mm), or with fewer than five tracks in common. Yes: the displacement along the fingertip's is 0.5 to 2.0 of it, and across it at most half. No: the object moved at most 0.15 of the fingertip's displacement. Between: cannot tell. From [E15](#e15): held 0.29 to 2.38, at rest at most 0.034 |
| `present_at(obj, pose)`  | Is the object's surface where a pose puts it?                      | The object's surface points at that pose, nearest per pixel, against the depth frame: each reads present (within half the object's thickness, at least 4 mm), absent (farther: the camera sees past where the object would be), or covered (nearer: the arm or another object in front)                                                  | Cannot tell with fewer than 20 points not covered. Yes: 60% of those present. No: 60% absent. From [E12](#e12): the resting place read empty at 0.50 and present at 0.00 to 0.02 once uncovered                                                                                                                                                                      |
| `related(obj, ref, rel)` | Is the object's pose relative to the reference the demo's?         | Views that place both, within 1 s of each other: the object's centre and rotation in the reference's frame against the demo's, rotation about a symmetry axis the object declares not counted                                                                                                                                            | Cannot tell without such views. Yes: within 20 mm and 10°. No: past 40 mm or 20°. Between: cannot tell. From [E14](#e14): the pose rule's spread at rest 15 to 20 mm, 2 to 2.6°                                                                                                                                                                                      |
| `resting(obj)`           | Has the object stopped moving?                                     | Its tracks over the last 0.5 s, against its reference                                                                                                                                                                                                                                                                                    | Cannot tell with fewer than five tracks. Yes: median displacement within 3 px. No: past 8 px. From [E15](#e15): at rest no track moved more than 8 px                                                                                                                                                                                                                |

**Milestones from the predicates.** What each of the demo's gripper events promises, and the predicates that check
it ([C8](#c8)):

| Milestone  | Carried by                                      | Expectation from the demo                                                                                                         | Yes when                                                                                                                     | No when                                                                                                                                         | Cannot tell                                                           |
| ---------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| _attached_ | the closing (the firm grip); a marked key frame | the object comes with the gripper from the closing until the opening                                                              | `follows` says yes, from a reference frame 0.2 s after the grip                                                              | `follows` says no, or `present_at(obj, where it lay)` says yes once the fingertip is 30 mm from it                                              | passed and watched: the carry is the probe that will tell ([Q2](#q2)) |
| _placed_   | the opening (the release); a marked key frame   | from the opening, the object stays where it was let go, relative to the leg frame it was let go in, after the gripper has left it | the gripper is 30 mm from the release, `follows` says no from the opening, and `present_at` or `related` says yes, for 0.5 s | `follows` says yes from the opening (it came away with the gripper), `present_at` says no (it is not where it was let go), or `related` says no | at the act's end the arm waits 2 s, then reports it unverified        |

_On top_ and _at a goal_ are both _placed_: the difference is the reference, another object or the base. The pose
`present_at` tests after a release is the gripper's pose at the opening times the hold measured at the grasp, kept
relative to the reference, so a stack pushed as a whole still reads as placed. A Milestone that is met stays watched
until its watch ends: _attached_ until the opening, _placed_ until the act's end. A no repeats on two consecutive
frames before it moves Progress ([C7](#c7)).

**What a failure looks like, and where Progress goes.** Every failure is the demo's effect gone wrong in one of four
ways; the check names which, and the Progress update ([item 8](#architecture)) decides where the act goes:

| Failure     | Against the demo                                             | Read as                                                            | Progress goes                                  |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------------ | ---------------------------------------------- |
| Absent      | the effect did not start: the object stayed at the closing   | _attached_ no: `follows` no, or the object still lies where it lay | back to the approach                           |
| Incomplete  | it started and stopped: the object fell out during the carry | _attached_ no while watched                                        | back to the approach, at the object's new pose |
| Misdirected | it ended elsewhere: let go off target, or carried away       | _placed_ no: `present_at` no, or `follows` yes after the opening   | back to the approach, at the object's new pose |
| Unsustained | it reached the demo's end state and reverted: tipped, rolled | _placed_ no while watched                                          | back to the approach, at the object's new pose |

**Extending the vocabulary.** A case these four predicates do not cover is one more predicate with the same three
answers, written for it however specific it is, and named by the Milestones that use it: an object that should
vanish into a hole (`present_at` no where it was let go, with the demo's own frames showing it gone), a part turned
relative to its body (`related` between the part and the body, past the demo's fraction of its turn), a fingertip
skin or a camera on the gripper ([R15](#requirements)).

## Alternatives, and what this costs

**Keep the runner and plan ahead of the arm** (what `a3e0cf48c` and `d8d36d606` do). It shortens the stops at stage
ends where its inputs still hold, but keeps the stops at key frames ([O1](#o1)), plans again when an object moves
([O2](#o2)), never goes back ([O3](#o3)), and checks nothing after the grasp.

**Change the current runner in place.** Its stops come from how the act is cut into streams waited on one by one
([O1](#o1)), and its retargeting from re-planning ([O2](#o2)); both are the structure, not a part of it. A rewrite in
place leaves no runner to compare against on the same scene while the new one is unproven, and the act records of
both are what decides the switch ([R12](#requirements)).

**Plan whole joint trajectories and re-plan them fast**, for instance with batched GPU optimisation. Still needs
Progress and Milestones to know what to plan for; puts the act's planning on the GPU the trackers already share
([act_loop.md](act_loop.md) O6).

**A state machine authored per task, with its retry edges.** Every task would carry its own recovery, which the act
keeps out of the runtime: interactions are generic, named by the demo, never by a task.

**The gripper's own readings as the check.** Width and load answer within a quarter second ([O10](#o10)), but say only
that something stops the fingers, the load is effort while the gripper moves ([O11](#o11)), and neither sees a placed
object fall over. They can join a check as evidence ([R15](#requirements)).

**A learned success or stage detector**, such as SARM, which is in this repository, or a VLM judge. SARM needs
annotated demos of each task and judges stages over seconds; VLM judges take 10 to 20 s per answer at 70 to 96%
([state_check_research.md](state_check_research.md)). Neither says which local fact failed. Both fit as a slow
layer that contradicts a Milestone late.

**A model of supports and surfaces.** "Resting on the table" needs the table found and tracked, and a deliberate drop
would read as a failure; the demo already says where the object ended relative to what it was let go on
([O16](#o16)).

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

What the design costs. Two runners exist side by side until the switch, and a fix to one is not a fix to the other.
Per-tick IK finds a configuration change only when it comes, where a plan made in advance would have seen it; the
candidate ranking and the joint-step check stand in for that. Every Progress update depends on the views the pose
rule lets through; a bad view that passes it moves the arm. The checks answer "cannot tell" often right after a grasp
([O15](#o15)), so an _attached_ Milestone is passed on the carry's promise more often than on a yes, and a miss is
caught at the lift or, when the tracks are covered, at the resting place seconds later. Every threshold is from one
rig and one camera. There are more threads to keep correct.

## Open questions

<a id="q1"></a>**Q1. Which Milestones may be repeated, and how often?** Attaching and placing can be undone by opening
the gripper and starting the approach again; a marked Milestone may stand for something that cannot. More tries
recover more failures and spend more time on a task that cannot succeed. Leaning: attached and placed repeatable, two
tries each; marked ones repeatable unless marked final.

<a id="q2"></a>**Q2. What does Progress do when the grasp's check cannot tell?** Wait standing at the gate, which tells
nothing, since the check needs the gripper to move ([O12](#o12)); end the act, which ends the 20 acts of 49 whose
tracks were covered ([O15](#o15)); or go on and keep watching, so a no during the carry or at the resting place sends
it back. Leaning: go on and watch, and record the pass as made without a verdict.

<a id="q3"></a>**Q3. How does the arm leave contact when Progress goes back?** Along a line to the restart's state, or
back along the demo's own path in the leg frame. The demo's path is known to have been free of collision relative to
the object; the recovery proposal warns that reversing it blindly proves nothing if the object was displaced.
Leaning: the line, with the gripper opened first, and the same joint, workspace and floor checks; the demo's path
when a measured case shows the line dragging the object.

<a id="q4"></a>**Q4. Rank candidates online only, or screen them with a precomputed reachability map?** Online is
built on the IK the act already uses and takes the arm's calibrated ranges into account; a map answers in
microseconds but per arm and per calibration ([Alternatives](#alternatives-and-what-this-costs)). Leaning: online,
with the map if the ranking misses [R10](#requirements)'s 200 ms.

<a id="q5"></a>**Q5. When two kinds of evidence disagree, which wins?** The 2D tracks may say the object came along
while the depth at its resting place still shows it there. Trusting either one stops a false answer from the other
only half the time. Leaning: a disagreement is "cannot tell", recorded with both.

To measure: the OpenArm 2's following error and its mink IK's timing (its solver is not installed on this
desktop), and what its gripper (a 1 N·m finger in position-force mode) reports for width and load; the time to rank
the ring's allowed turns; the command stage's latency on the rig under load; an act on the rig with each runner on
the same scene, per leg.

## Glossary

- **Progress:** where along the act's timeline the act is: a leg, and how far into it, 0 to 1.
- **Milestone:** a condition on the observed state that Progress cannot move past while its check says no: _attached_
  or _placed_, carried by a gripper event or a marked key frame, _repeatable_ or _final_.
- **Predicate:** one question about the scene from the vocabulary — `follows`, `present_at`, `related`, `resting` —
  answered yes, no or cannot tell, with its evidence.
- **Reference:** what a predicate's answer is relative to: the gripper, another object, or the arm's base.
- **Leg:** the part of the demo between two consecutive key frames, as in [act_loop.md](act_loop.md).
- **Leg frame:** the pose a leg's motion is relative to: its object's, turned by the leg's chosen candidate, or the
  base.
- **Guided pose:** what a leg moves as the demo did: the fingertip, or the attached object while one is attached.
- **Pose entry:** one motor tick of the queue: a pose relative to its leg frame, or a line's two anchors and how far
  along it is, with the gripper command and the time it is due.
- **Anchor:** one end of a line: its start in the base frame, or its end relative to the leg frame.
- **Lead:** how far ahead of being due an entry is solved: four ticks.
- **Transition:** the line from the last command to the state a Progress update chose, with the gripper's opening
  before it and its closing after.

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
30 °/s on 31 of the lift away's 36 samples (at most 158 °/s). Bus clamp: `jog.MAX_STEP_DEG` = 12° from the reading;
live playback commands: `jog.PLAYBACK_LAG_DEG` = 3° a tick from the previous command.

<a id="e6"></a>**E6. The hold at the grasp.** The hold seeded at the grasp, against the demo's, in the eight acts of
2026-10-16 23:19 to 2026-10-17 13:02: 4.6 to 7.4 mm and 1.2 to 1.8°.

<a id="e7"></a>**E7. Act durations against the demo**, key frames 2 to 9 of the insertion: demo 13.66 s; act `125408`
19.77 s (at `be2e1822e`); act `150843` 18.89 s (at `a3e0cf48c`: lines at the demo's pace, follows free of the walk's
cap, the carry planned ahead). The insertion's follow: demo 4.81 s, 6.20 s, then 5.37 s.

<a id="e8"></a>**E8. Tracker rate.** About 9.5 frames/s on the rig with Point2Pose tracking, 2026-10-16
([action_queue_timing.md](../../gui/docs/action_queue_timing.md)).

<a id="e9"></a>**E9. Gripper width.** 264 act records; 77 with a grasp check (`core.grasp_held`). Holds (75): the
reading stopped 0.85 to 2.40 units short of the closing command (10th percentile 1.19, median 2.13). Misses (2, acts
`20261015_213310`, `_221031`): 0.69 and 0.41 short, readings 99.3 and 99.6 at a command of 100. The demo's own grip:
97.87 at a command of 100 (2.13 short). Held readings over 34 carries: 97.94 to 98.35; the reading's deviation from
its median within a carry 0.27 to 0.41 units at the 95th percentile, 0.82 at most. 1 unit = 14.9 encoder counts
(calibration range 1991 to 3480). The reading settles (three ticks moving under 0.3) 0.10 s after the closing command
ends, 0.14 s at the 90th percentile, 0.40 s at most (95 acts).

<a id="e10"></a>**E10. Gripper load at rest.** `Present_Load` of the gripper, median over 0.5 to 2 s after the closing
command ended: holds 144 to 232 (10th percentile 144 over 73 acts; by demo: insertion 224, pick-and-place 128), misses
20 and 34, before the closing 8 (at most 24). Over the carries of acts `20261017_125408` and `_150843`: 10th
percentile 200, minimum 92 and 180, no tick under 64. Hold evidence from vision (`hold_selections` not from
prediction): 8 acts, 9.9 to 21.8 s after the closing ended, all "held-view position; rotation predicted"; 110 failed
attempts over 48 acts.

<a id="e11"></a>**E11. The release.** Acts `20261017_150843` and `_125408`, by 0.1 s from the start of the opening
command: the reading follows the command within 0.03 s; the load falls from 200 to 28 and 44 at 0.4 to 0.5 s, as the
command pauses at 83.9; it rises to 210 to 264 at 1.0 to 1.2 s while the command goes on to 65.7 and the reading
trails it by 4 to 5 units, and falls to 2 to 16 once the reading has caught up.

<a id="e12"></a>**E12. The resting place.** The object's mask and depth on the act's first frame; per later frame, the
share of those pixels reading the object (within half its height), the table, or something nearer. Act
`20261017_150843`: nearer (the arm) from 3 s before the closing to 4 s after; empty from 4.45 s after the closing
ended (the carry began at 0.82 s), then empty at the 10th percentile 0.50 and object at the 90th 0.00 over the next
6 s. Act `_125408`: empty from 4.50 s, then 0.83 and 0.02. The dowel lay 7 to 13 mm above the table in those
frames. In the missed act `_213310` the arm never lifted, and the place read nearer throughout.

<a id="e13"></a>**E13. The placed object.** Act `20261017_150843`: 33 frames after the release, the dowel's
Point2Pose view against its reference accepted in none (surface state "disagreement" in all 33); its colour tracks
still visible on 179 of 323. Finds: 679 pairs of consecutive dowel locates in the acts of 2026-10-15 to 17, 0.75 s
apart at the median, 1.22 s at the 90th percentile.

<a id="e14"></a>**E14. Pose spread at rest.** Trusted views of the ring before the carry, 41 acts, 14 views each at the
median: the constrained pose's translation from its median 15.2 mm at the 95th percentile per act (median act; 44.9
in the worst), the raw tracker pose 19.8 mm; rotation 2.0° (raw 2.6°). Act `_150843`: the ring motion its targets
were composed with moved from (−13.1, 44.4, 6.3) to (−16.3, 44.9, 19.7) mm between 0 and 2.3 s of the carry.

<a id="e15"></a>**E15. 2D co-motion.** 49 acts of 2026-10-15 to 17 with frames saved and a closing; the reference
frame 0.3 s after the closing ended, the verdict at the first frame with the fingertip's projection 25 px from the
reference. 22 verdicts (five or more tracks in common): tracks' median displacement over the fingertip's, 0.29 to
2.38, median 0.91; 20 acts had 0 to 4 tracks left; 7 never moved 25 px. At rest, over the approach (key frame 1) of
39 acts: 33 at 0.006 to 0.034 with no track moved more than 8 px; 6 at 0.20 to 1.85 with every track moved, the
median shift 7 to 50 px, and the act's aimed pose of the object moved 15 to 34 mm over the same key frames (steps of
12 to 29 mm). Track ids are stable: 323 distinct over an act, none jumping more than 30 px between frames.

<a id="e16"></a>**E16. The demo's lift.** After its gripper command reached its closing value the demo's fingertip
had moved 20 mm in 0.27 s, 35 mm in 0.57 s, 50 mm in 0.63 s. Under the runner of 2026-10-17 the fingertip reached
25 px of projected motion 1.7 to 5.3 s after the closing (standing for the grasp check and the carry's planning).

<a id="e17"></a>**E17. The pose in the hand.** Act `20261017_150843`: the tracker's pose of the dowel existed in all
86 carry frames, accepted in none; its motion since the first frame after the closing came to 413 mm against the
fingertip's 136 mm, under every frame convention tried. Perspective-n-point (OpenCV, RANSAC, 3 px) on the original model points
still tracked: 6 to 13 inliers, reprojection 0.4 to 2.0 px, the object-in-fingertip offset scattered over 1.2 m and
111° at the median across the carry; at rest with 24 to 25 inliers it differed from the accepted tracker pose by 19
to 58 mm. In the demo's own carry the tracker saw the dowel in 0 of 101 samples.

<a id="e18"></a>**E18. What each frame records.** Per saved frame (`frames/NNNNNN.npz`, by `p2p_bridge`): the tracker's
key-point rows as `track_idx` (ids, growing as it re-seeds), `track_uv`, `track_vis`; the original designation's 3D
key points `model_xyz` with `model_valid`; the points the fit used, `fit_uv`, `fit_inlier`; the pose `delta`; the
followed objects under `other_*`; the point groups' points with depth. The held object's pose is stored relative to
the fingertip (`held_0`), not in the camera frame.
