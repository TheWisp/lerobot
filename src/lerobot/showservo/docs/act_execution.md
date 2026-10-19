# Act execution: Progress, Milestones and the state checks

Status: proposed
State of the work: no tracking issue yet

An act replays a demo on objects that have moved. Today it does so as a string of separate motions: the arm goes to
each key frame, stops there, and while it stands the rest of the act is planned again. At speed 1.0 an act takes 1.4
times as long as the demo and stops 12 to 16 times. It only ever moves on, and it never looks at what happened: after
a missed grasp it carries nothing to the target, when the target is moved during an insertion it goes on with the
next step, and when a placed object falls over it reports success. The OpenArm 2 is meant to use the same runner.

**Proposal.** A new act engine, run beside the current one and chosen per act, plays the whole act as one motion.
[Progress](#glossary) marks how far through the demo the act is. For every motor tick the engine makes one
[command](#glossary) from the demo — the demo's pose at that moment, moved with the object the arm is heading for —
a few ticks before the tick, so when that object moves, the motion still to come bends toward it and the arm does not
stop. Where the demo shows an object being picked up or put down, or where the operator marks one, a
[Milestone](#glossary) checks with the camera that the same happened in the act. The arm never waits for a check:
when one fails, Progress jumps back to the nearest point of the demo that matches what the camera sees, and the act
goes on from there.

## Scope

In: one act on one arm, first key frame to last: legs, Progress, the command queue, the checks, going back or ahead,
the allowed turns (symmetry and landing), speed, the act's record, and how the engine sits beside the current runner.

Out, and why: the trackers that measure object poses (SAM, Point2Pose and the point groups, [act_loop.md](act_loop.md));
this design takes their measurements as they come. The servo I/O (`gui/api/jog.py`), which already sends the newest
due command each tick; two arms at once, whose Progress would have
to be tied together; failures that need a model of cause ("the arm knocked something over") or a law fitted to the
demo (how far a screw goes in per turn); what is pinched in a cloth grasp, which takes touch
([state_check_research.md](state_check_research.md)). Not goals: collision checking beyond the workspace box and the
floor; deciding what to do after the act gives up, which belongs to whatever runs the act.

## Requirements

Safety first: the act must not act on a state that is not there. Then the motion should look like the demo's.

| #   | Pri | Requirement                                                                                                                                                        | Target                                                                                                         | Why                                                                                                             | Checked by                                                                  |
| --- | --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| R1  | P0  | The act never goes on past a failed Milestone, nor past a final one before every earlier Milestone has succeeded                                                   | Never                                                                                                          | A missed grasp is followed by a carry today ([O2](#o2)); what cannot be undone must not be done on a guess      | Unit tests; every act's record                                              |
| R2  | P0  | A failure sends Progress back, and a moved object back or ahead, to the nearest point of the demo that matches what the camera sees                                | The cases in [Going back, or ahead](#going-back-or-ahead)                                                      | The user's cases of 2026-10-17; finding a failure and recovering from it are one mechanism                      | A simulated act per case                                                    |
| R3  | P0  | A check reports success only when the camera can tell success from failure                                                                                         | No success on the recorded misses, nor on made-up misses (a frame from before a lift paired with one after it) | A false success is R1's carry, approved                                                                         | Unit tests; a replay of the act records                                     |
| R4  | P0  | Every camera frame, every object's pose is updated from the trackers' measurement when the trackers are confident in it; no rule by object, role or state gates it | No measurement the trackers are confident in is dropped                                                        | A rule that dropped views of a covered object left the held dowel unmeasured on 207 of 214 frames ([E25](#e25)) | A test feeding measurements of a covered, held object                       |
| R5  | P0  | No check, moved object or choice of turn holds up a command                                                                                                        | No stop longer than one tick from any of them                                                                  | The arm stood 0.06 to 2.5 s while the act planned again ([O1](#o1))                                             | Simulated acts with moving objects; the rig                                 |
| R6  | P1  | An undisturbed act takes the demo's time divided by the speed, whatever rate the demo was recorded at                                                              | At most 1.15 × that, for demos at 15 to 60 frames/s, key frames one demo frame apart included                  | 1.38 × today ([E7](#e7)); the arm runs about 0.1 s behind its commands ([O4](#o4))                              | A simulated act per frame rate and speed; the rig                           |
| R7  | P1  | The arm stops only where the demo stopped, at a final Milestone, while the gripper opens before going back, or while the arm catches up with its commands          | No other stop longer than 100 ms                                                                               | 12 to 16 stops per act today, most at key frames ([O1](#o1))                                                    | The act's command record                                                    |
| R8  | P1  | A failed grasp is acted on soon after the camera can see it                                                                                                        | Decided within 1 s of the lift making it visible; Progress moved within two camera frames of that              | The demo's lift rises 35 mm in 0.57 s ([E16](#e16)); the camera gives 9.5 frames/s ([E8](#e8))                  | A replay of the act records; simulated acts                                 |
| R9  | P1  | Every threshold a check uses is measured during the act or set by the operator; none is a constant in the code                                                     | Each threshold in the act's record, with the measurement or setting it came from                               | How far the rig's views of an object at rest wander is 15 mm in one act and 45 mm in another ([E14](#e14))      | A test: the same scene with noisier views reads "can't tell", not "failure" |
| R10 | P1  | One engine drives the SO-107 and the OpenArm 2                                                                                                                     | The same arm interface                                                                                         | The runner names the SO-107's motors ([O10](#o10))                                                              | The engine's tests on both arms' kinematics                                 |
| R11 | P1  | The engine runs beside the current runner, chosen per act, and leaves it unchanged                                                                                 | Either from the same page and API; the current runner's tests pass unchanged                                   | Acts with both on the same scene decide the switch                                                              | The existing tests; the record names its runner                             |
| R12 | P2  | Joint speed is limited by the arm and the motor loop, not by the slow ready/park speed                                                                             | 90 °/s on the SO-107; where a step would be faster, the act slows down there                                   | The limit is 30 °/s; the demo moved at up to 158 °/s ([O4](#o4))                                                | A simulated act faster than the limit                                       |
| R13 | P2  | Every check result and every jump of Progress is in the act's record, with what it was based on                                                                    | One record per decision                                                                                        | Failures are diagnosed from the act's own record ([E7](#e7))                                                    | A test of the record                                                        |

Conditions: the SO-107 (left arm) and its RealSense camera on this desktop; the insertion demo of 2026-10-17 and the
acts of 2026-10-15 to 17 with saved frames.

## Observations

<a id="o1"></a>**O1. The act stops at every key frame and plans again with the arm standing.** Each key frame is sent
and waited on until the arm is within 2° of it (`_act_key_frames_task`, `live_execution.stream`), and a line slows
into the key frame it ends on: from 26 °/s down to 3 and back up ([E3](#e3)). The arm stood 0.06 to 0.8 s at the end
of each stage, and 2.5 s after a grasp while the carry was planned twice ([E4](#e4)). So the stops come from how the
act is cut up, and following a moved object must not mean planning again.

<a id="o2"></a>**O2. The act only moves on.** The step a stream aims for only increases, and the runner takes the key
frames in order: an object moved during a grasp is followed by a closing gripper, a missed grasp by a carry.

<a id="o3"></a>**O3. IK is cheap; choosing between allowed turns is not.** Solving one pose to 0.5 mm takes 0.3 to
2.6 ms on both arms; under the rig's camera and tracker load, 1.7 ms typically, 21 ms at the 95th percentile and 75 ms
at the slowest ([E1](#e1)). Ranking the allowed turns solves every sample of every turn, and that was the 2.5 s stand
of [O1](#o1). So IK can run for every tick, a few ticks ahead, and ranking must run beside the motion.

<a id="o4"></a>**O4. The arm can move several times faster than the act lets it, and runs behind its commands.** The
demo moved joints at up to 158 °/s and the arm ran about 3 ticks behind; the act caps joints at 30 °/s, the speed
meant for ready and park ([E2](#e2), [E5](#e5)). The motor loop sends the newest due command each tick, never skips
one where the gripper starts, stops or turns back, and holds each goal within 12° of the joint's reading
(`jog._advance_playback`, `jog._clamped`); commands still queued can always be replaced. It once also moved each live
command at most 3° a tick from the one before (90 °/s) and held its playback for a joint 5° behind; neither is measured
against the arm, which turns its joints at up to 224 °/s and in end-to-end runs trails by 8 to 12° at the 95th
percentile ([E22](#e22)). Both are gone.

<a id="o5"></a>**O5. The grasp never lands exactly where it was aimed:** 4.6 to 7.4 mm and 1.2 to 1.8° from the demo's
in every act of 2026-10-16 and 17 ([E6](#e6)). So what follows a grasp must allow for how the object actually sits in
the gripper.

<a id="o6"></a>**O6. Object poses arrive 9.5 times a second, and how much they wander at rest differs between acts.**
Views of the ring standing untouched spread 15 mm in the median act and 45 mm in the worst (95th percentile within an
act; [E8](#e8), [E14](#e14)). So "the object moved" and "the object is where it should be" can only be judged against
the spread measured in the same act.

<a id="o7"></a>**O7. The camera can tell whether an object came along with the gripper only once the gripper has
moved away from where the object lay, and sometimes not even then.** Once the fingertip had moved 25 px in the image,
the object's tracked points had moved 0.84 to 1.0 as far when it was held and 0.01 to 0.03 as far when it was not
([E15](#e15)); where it lay reads empty once the arm stops covering it ([E12](#e12)). In 20 of 49 acts the gripper hid
the tracked points; the held object's 3D pose was wrong in every frame of the carry; after being put down, the object
was recognised from its reference view in 0 of 33 frames ([E13](#e13), [E17](#e17)). So a check chooses between two
outcomes that cannot be told apart until they would look different, and "can't tell" is a normal answer.

<a id="o8"></a>**O8. The gripper's readings say that something stopped the fingers, not what, nor where it is
later.** The gripper's position and load tell a hold from a miss 0.25 s after it closes, and the load measures the
servo's effort while the gripper moves ([E9](#e9) to [E11](#e11)). So they can add to a check but not make it.

<a id="o9"></a>**O9. Games and simulators need few checks, and never "can't tell".** Held, inside a place, on top of,
a part turned, at rest; a VR grab counts as broken when the object drifts too far from where the hand should be
holding it, with no assumption that the grip is rigid; ManiSkill and robosuite count a placement only once the robot
has let go and moved off
([state_check_research.md](state_check_research.md#how-games-and-simulators-define-the-same-predicates)). Here the demo
is the only statement of success, and the camera is calibrated to the arm's base, which moves on a mobile robot. So the
checks ask what an object moves with — the gripper, another object, the base — as the demo shows it.

<a id="o10"></a>**O10. The two arms differ only in kinematics, gripper and motors.** The SO-107 solves its IK with
Pink, the OpenArm 2 with the vendor's mink solver (a Pink build of its URDF exists); the OpenArm 2 has 7 arm joints.
The runner uses the SO-107's motor names (`joint_alignment.MOTOR_NAMES`). So the engine needs kinematics, joint ranges,
the joint speed limit, and which gripper column closes which way, nothing else.

## Design

### The loops, and who writes what

Two loops. Perception runs at the camera's pace, about 10 frames a second; everything else runs on one control loop at
a fixed 30 Hz. Data flows one way — measurements and joint readings, then the world, then the task, then commands, then
the servos — and each piece of data has exactly one writer.

| Data           | Holds                                                                                                                                 | Written by                     | Read by                   |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ | ------------------------- |
| Measurements   | each object's pose the pose estimator takes from the trackers' output (SAM, Point2Pose, point groups), per frame, at its capture time | perception: the pose estimator | the world step            |
| Joint readings | the joint angles the servos report, and the fingertip computed from them, at their read time                                          | the motor step                 | world, task, motion steps |
| World          | each object's pose, and the time of the measurement it came from                                                                      | the world step                 | task and motion steps     |
| Task state     | Progress, each Milestone's state, how each held object sits in the gripper                                                            | the task step                  | the motion step           |
| Commands       | timed joint commands, each with the pose it came from                                                                                 | the motion step                | the motor step            |

```mermaid
flowchart LR
    subgraph PER["Perception, every camera frame (about 10 a second)"]
        P1["The trackers: each object's pose, and their confidence in it"]
        P2["The pose estimator: which of those poses is a measurement"]
        P1 --> P2
    end
    subgraph CTL["Control loop, 30 times a second"]
        C1["Motor step: read the joints; send the command due now"]
        C2["World step: each new measurement becomes that object's pose"]
        C3["Task step: Progress from what was sent; each Milestone answered from the world; the leg's target"]
        C4["Motion step: top the command queue up, each command the demo's pose at its tick moved with the target's pose, then IK"]
        C1 --> C2 --> C3 --> C4
    end
    P2 -- "measurements" --> C2
    C4 -- "commands" --> C1
```

- **Which tracker output is a measurement is its own layer:** the pose estimator (`lerobot.showservo.estimation`)
  takes each frame's candidates, a pose per source with that source's own verdict, and says per object which pose
  to take, or none and why; the act's record counts the refusals. Neither the world nor the engine sees a tracker, so
  a better estimator replaces the one in use without touching either. Two sources today: Point2Pose's own verdict
  (its fit found, the object not lost), which passes views every other check rejects ([E30](#e30)); and the
  key-frame runner's checks, which refuse nearly every view instead ([E31](#e31)). The one in use takes Point2Pose's
  view unless the camera sees through it: the camera's ray to a point of the object's surface, at that pose, met
  something farther than the point, so the surface the pose puts there is not there (a point the ray met something
  nearer on is only hidden). A view is refused when more of the surface is seen through than the object's cut-off;
  the tolerance and the cut-off are measured on the object's own views in its first second ([E33](#e33)).
- **One rule for every object.** A new measurement replaces the object's pose. No rule
  names an object, or depends on its role (the one picked up, the one placed onto) or its state (held, touched,
  moving). Between measurements an object is where it was last measured, or where the gripper carried it, whichever
  came last: while the act holds it, its one pose is where the gripper has it, in the world too, and let go it stays
  where the gripper left it until a view of it ([E34](#e34)). ([R4](#requirements))
- **A turn its symmetry cannot show is not a move.** Of the poses an object's declared symmetry makes look the same,
  the world keeps the one nearest its demo pose, and how it sits in the gripper the one nearest the demo's grip; a
  symmetric object is measured at a point on its axis, which those turns leave still ([E29](#e29)).
- **The gripper is the only special thing,** and it is in the joint readings, not the world. A held object's pose
  between its measurements is the fingertip now times how it sits in the gripper. How it sits is taken at the grasp,
  from the object's pose and the fingertip at that moment, and estimated from then on like any pose, from every
  measurement but those the grasp's check answers "not held" (they say nothing about how it sits in the gripper). The
  check answers each measurement from how it sat before that measurement ([E27](#e27)). Whether it is held is the
  task's question, never the world's.
- **The motion step solves ahead.** IK takes 1.7 ms at the median and up to 75 and 299 ms on the rig ([E1](#e1),
  [E25](#e25)), so commands are solved a few ticks before they are due and queued: as many ticks as the slowest solve
  of this run and the gap between control ticks cover, plus one. The motor step sends whatever is due and never waits
  on a solve; on the rig it runs on its own thread at the same 30 Hz, so a slow solve can delay the motion step, never
  the servos.
- **A moved object becomes motion, never a jump.** What the motion step makes commands from — each object's pose,
  and how each held object sits in the gripper — follows its estimate at no more than the walk speed during a leg; a
  new path (going back after a failure, backing out of a moved target) starts from the estimates as they are, being
  itself a move from where the arm stands ([E26](#e26), [E27](#e27)).
- **One clock** stamps capture times, joint reads and due times. The world step pairs a measurement with the joint
  reading at its capture time, interpolated between two reads and never extrapolated.
- **A failure or a moved target takes effect at once:** the task step moves Progress, the commands not yet sent are
  cancelled, and the motion step fills the queue again from there.

### Legs: the path between two key frames

A leg is the demo between two key frames. It starts from wherever the arm is when the leg begins and ends at its key
frame, which is held relative to an object or to the arm's base. Each command of the leg is

    pose(s) = blend( C₀ · D(0)⁻¹ · D(s),   M · D(s),   w(s) )

- `s`: how far through the leg the command is, 0 to 1;
- `D(s)`: the demo's pose at that point (for a line, the straight path between the demo's poses at its two key frames);
- `C₀`: the command the leg began from, frozen;
- `M`: how far the leg's end object has moved since the demo, read anew for every command;
- `w(s)`: rises smoothly from 0 to 1 over the leg.

What this does:

- The object the arm is leaving no longer matters once the leg has begun: going from above A to above B, A moving
  changes nothing.
- The object the arm is heading for is read for every command: when it moves, the rest of the leg bends toward it, a
  little at first and fully by the key frame, and the arm keeps moving.
- A leg that starts and ends at the same object — a grasp, an insertion — follows that object fully all the way
  (`w = 1`): inside a ring, the arm must move with the ring. Only if the arm starts away from the leg's start (after a
  best-effort stop) does `w` rise over the first part, to close the gap.
- With nothing moved, both sides are the demo's own pose.
- While an object is held, the pose is corrected for how it sits in the gripper ([O5](#o5)), as the task state has
  it, so the held object, not the fingertip, goes where the demo's went. Once it is let go, the rest of that leg keeps
  the correction it had then: the gripper goes on from where it let go ([E27](#e27)).
- A new measurement of the target reaches every command made after it; commands already queued keep the pose they
  were made with.

### The demo's clock

Commands are made from the demo's clock, not from its frames or its key frames. Each tick moves the clock on by the
speed times 1/30 s, and the command is the demo's pose at that time, interpolated between the two demo frames around
it. That is what keeps the act on the demo's time at any recording rate and any speed:

- A demo recorded at 60 frames/s, or played at 2×, gives about two demo frames per tick; one at 15 frames/s, about one
  every two ticks. Either way the act takes the demo's time divided by the speed.
- Key frames marked on every demo frame, as a grasp or a place may be, are legs a thirtieth of a second long, and one
  tick can cross several. Its leftover time carries into the next leg: no leg is rounded up to a whole tick, so no
  time is gained or lost per key frame.
- Where the demo kept moving through a key frame, the act does too: no slowing down, no gripper moving while the arm
  waits.
- A leg where the demo stood still (the gripper closing) stands for the demo's time; nothing is divided by its length.
- A line takes the demo's time between its key frames, scaled by how much longer or shorter the line is than the path
  the demo took there.
- The gripper's turning points — where the demo's gripper command starts, stops or turns back — are never smoothed
  away: the tick nearest each one gets its exact value, so a closing reaches its full command even when it falls
  between ticks. The motor loop never skips such a command when it catches up.
- Where a joint would move further in a tick than the motor loop's clamp (12°), the clock runs slower there, and the
  act's record says by how much. The bound guards against a solve that jumps; it is not a speed limit, and on the rig
  it has not slowed an act ([E22](#e22)).
- Nothing waits for an arm that trails its commands. A joint that sticks breaks away as its goal moves on, as in
  end-to-end runs; a goal held still stops the effort that frees it ([E22](#e22)). A joint 25° behind its command is
  blocked, and the motor loop freezes the arm. Only a gripper opening waits, at most 0.5 s, for every joint within 3°.

([R6](#requirements), [R7](#requirements), [R12](#requirements))

### Milestones: where, which object, when

**Where.** Only where the demo shows an object start or stop moving with something:

- at a grasp: the object starts moving with the gripper (in the demo, the gripper stopped closing before its command
  because the object held it open: `core.demo_holdings`);
- at the release of a held object: it stops moving with the gripper and stays where it was put, relative to the
  object the leg ends at, or to the base;
- at a key frame where the operator marks one: an object, and what it moves with from there (a block pushed by the
  gripper without a grasp; a handle turning with the toy's body).

Opening or closing the gripper with nothing in it makes no Milestone.

**Which object.** Each Milestone names one object: the one picked up, put down or marked. It is checked from that
moment until the object's next Milestone, or the end of the act: the dowel from the grasp until it is put down, then
from being put down to the end. Only those objects are checked, usually one at a time; the others matter only as
what legs are relative to.

**When.** On every camera frame, for each object being checked: 0.06 to 0.18 ms per check ([E19](#e19)), beside the
motion and never in its way. Every frame rather than at key frames, because the answer appears when the gripper has
moved far enough from where the object lay for the two outcomes to look different, and no key frame marks that
moment ([O7](#o7)).

### What a check answers, and how close counts

- **Success:** what the demo shows happened. After a grasp the object rises with the gripper; after a release it
  stays where it was put while the gripper moves off.
- **Failure:** the other outcome happened. After a grasp, the object stayed where it lay. After a release, it came
  away with the gripper, or it is not where it was put: it fell, rolled, or was put down off target.
- **Can't tell:** the camera cannot see the object, or the gripper has not moved far enough yet for the two outcomes
  to look different. A grasp can't be told until the lift; a release, until the gripper has moved off.

Every answer comes from the world: the object's measured pose, against two predictions of it, one for each
outcome. After a grasp: with the gripper (the fingertip at the measurement's moment times how it sat at the grasp), or
where it lay. After a release: where it was put (relative to what it was put on), or with the gripper. The nearer
prediction wins once the two are further apart than the object's noise; a measurement far from both — it fell
somewhere else — is a failure; no new measurement, or the two still too close, is can't tell. A failure counts once
two measurements in a row show it.

**How close counts.** No threshold is a number written into the engine ([R9](#requirements)):

- _The noise is measured, per object, in this act:_ how far each view lands from the pose the views before it make,
  while it stands untouched — before the arm reaches it, and all act long for those it never touches — counting only
  the views the world took as still: a view that shows a move is not noise ([E28](#e28)). On the rig its views
  wandered 15 mm in one act and 45 mm in another ([O6](#o6)).
- _Where the object should be is the demo's:_ relative to what it was put on, as the demo left it.
- _How far off the task allows is set by the operator,_ per Milestone in the demo editor, the way the landing freedom
  is set today: one demo cannot say whether a place needs a millimetre or a few centimetres. By default there is no
  allowance, so the check is as strict as the camera can see, and no stricter.
- _Equivalent poses pass:_ the object's declared symmetry (its repeating turns, a continuous axis, swapped ends) and
  the landing freedom of the leg (exact, the target's symmetric turns, any turn about its axis).
- _One is a choice, not a measurement:_ two measurements in a row for a failure, the least that ignores a single bad
  frame.

**What it does to Progress.** A failure sends Progress back. Success or can't tell lets the act go on, and the check
keeps running while the object is being checked, so a drop during the carry or a fall after the release is still
caught. Only at a Milestone the operator marks final — something that cannot be undone, such as a pour — does the arm
wait: it does not go past until every earlier Milestone has succeeded, and the act gives up if that does not come
within a time the operator sets. When the motion is over, an act whose every Milestone succeeded is done; one with a
Milestone the camera still cannot tell is "not confirmed", never a success ([E24](#e24)). ([R1](#requirements),
[R3](#requirements), [R8](#requirements))

### Going back, or ahead

**When a check fails**, or a command cannot be reached, Progress moves to the nearest point of the demo that matches
what the camera sees:

- not past the next Milestone that has not yet succeeded;
- never back past a final Milestone;
- nearest to where the arm is now, compared relative to each point's own object, with a cost for jumping far through
  the demo.

A Milestone that Progress goes back past must succeed again; one that fails a second time makes the act give up
([Q1](#open-questions)). The arm first waits for a new view of the object, which may have fallen somewhere new, then
gets to the new point by a short straight move at the walk speed, opening the gripper first if it has to open, closing
it last if it has to close. ([R2](#requirements), [R4](#requirements))

**When the object the arm is heading for moves** by more than its measured noise, on two views in a row:

- during a follow — in contact, or nearly, as in a grasp or an insertion — the arm backs out along the demo's own path
  relative to where the object was, to the start of that follow, and goes in again relative to where it is now. The
  nearest point is not enough here: after a sideways move every point of the descent is as far, so nothing would take
  the arm up out of the ring. Until the second view confirms the move, the arm keeps to where the object was rather
  than drifting sideways inside it.
- during a line, the line re-aims from where the arm is and keeps its speed: a target moved nearer is reached sooner,
  one moved away later.

A missed grasp, step by step:

```mermaid
sequenceDiagram
    participant A as Arm, 30 times a second
    participant P as Planner, 4 ticks ahead
    participant C as Camera check
    participant G as Going back
    Note over A: the gripper has closed on the dowel and the lift starts
    A->>C: from now on, check that the dowel rises with the gripper
    loop every tick
        P->>A: the next command of the lift
    end
    C->>C: frame 1: the gripper has risen far enough to tell, and the dowel has not moved (failure 1 of 2)
    C->>C: frame 2: still not moved (failure 2 of 2)
    C->>G: the grasp failed
    G->>A: cancel the queued commands
    G->>P: go back to the point of the descent nearest the arm, gripper open
    P->>A: open the gripper, go down, close it again
    Note over A,C: the arm lifted about a third of a second longer than it needed to
```

| What happened                                             | Where Progress goes                                 |
| --------------------------------------------------------- | --------------------------------------------------- |
| The grasp missed                                          | back to just above the object: open, grasp again    |
| The object fell out during the carry                      | back to the approach to the object, where it now is |
| It was put down off target, or came away with the gripper | back to the approach to the object, where it now is |
| It fell over or rolled after being put down               | back to the approach to the object, where it now is |
| The target moved nearer                                   | ahead: the part the arm no longer needs is skipped  |
| The target moved away during an insertion                 | back up out of it, and in again                     |
| A final Milestone was passed                              | nowhere: the act gives up and says why              |

### What the engine keeps of the key frames

| Today                                                         | In the engine                                                                                                                                                             |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A key frame's object: an object, the taught object, the world | the object the leg arriving at it is heading for                                                                                                                          |
| (new) Its frame: start, where its leg begins                  | the demo's motion since the key frame before, from wherever the arm began the leg: C0·D(t0)⁻¹·D(t); a lift that turns as the demo turned. The key-frame runner refuses it |
| Its move: line, follow, lift                                  | line: straight from where the arm is; follow: the demo's own path; lift: straight up by `up_mm` from where it is                                                          |
| Its gripper: first, along                                     | on a line, the gripper moves before the arm, or with it                                                                                                                   |
| Its constraint: pose, orientation                             | the whole pose; or only the rotation, the position staying where the leg started                                                                                          |
| Its constraint: best effort                                   | the key frame is a target: when the arm stops short, the leg ends where the arm is and the next starts from there ([E23](#e23))                                           |
| A held object, from the grasp to the release                  | the legs between allow for how it sits in the gripper; the grasp and the release are Milestones                                                                           |
| An object's symmetry: repeating turns, an axis, swapped ends  | poses that look the same: the world and the hold keep the one nearest the demo's ([E29](#e29)). **NOT IMPLEMENTED:** ranking them by reach, beside the motion             |
| The demo's landing: exact, symmetric turns, any turn          | turns the legs onto the target may take, chosen beside the motion; poses the checks count as the same                                                                     |
| The point groups carrying objects between views               | another source of object poses                                                                                                                                            |

Allowed turns are ranked again when their object moves — IK of the leg's key frames and a few of its points, from the
arm's present joints, by whether the arm reaches them and how close its joints stay to the demo's — and a different
turn is taken only before its leg begins, and only when it is clearly better. ([O3](#o3), [R5](#requirements))

### What the engine connects to

The engine (`lerobot.showservo.engine`) holds no GUI state. It reads three things: the arm (kinematics, joint names,
which gripper column closes which way, joint ranges, workspace, joint speed limit), the servo I/O's queue (what was
sent; add to it; cancel what is not sent; its clock), and a log that perception and the joint readings write into
(each object's pose the pose estimator takes, per frame; the fingertip from the joints; the camera). The GUI fills them in, and the act request
says which runner to use. The act's record holds every check result with what it was based on, every move of
Progress with its cause, every command sent, and on each camera frame every object where the engine
believed it (a held one where the gripper carries it), which the act's replay draws over the camera image
([E32](#e32)); at the end it says which Milestones succeeded, which failed and why,
and which were passed while the camera couldn't tell, apart from a fault that stopped the act. ([R10](#requirements),
[R11](#requirements), [R13](#requirements), [O10](#o10))

## Alternatives, and what this costs

- **Keep the runner and plan ahead of the arm** (`a3e0cf48c`, `d8d36d606`): fewer stops at the end of each stage, but
  it still stops at key frames, plans again whenever something moves, never goes back, and checks nothing.
- **Change the runner in place:** its stops and its planning again are how it is built ([O1](#o1)), and without it
  there is nothing to compare acts against while the new one is unproven.
- **Check at key frames, or make every Milestone wait:** a key frame comes before or after the moment the outcomes
  look different, and waiting stops the arm where the check needs it moving ([O7](#o7)).
- **The gripper's readings as the check:** fast, but they say nothing about where the object is later ([O8](#o8)).
- **A learned success detector (SARM, in this repository; a vision-language model):** seconds per answer, needs
  labelled demos, cannot say which fact failed; it could come later as a slow second opinion.
- **A model of tables and supports:** the table would have to be found and tracked, and a deliberate drop would read as
  a failure; the demo already says where the object ended up, relative to what.
- **A state machine written per task:** every task would carry its own recovery; the engine stays the same for every
  task.

What it costs. Two runners exist until the switch. Running ahead of the checks spends motion that a failure undoes:
about a third of a second, plus the move back. A grasp whose object the trackers cannot measure goes on unconfirmed, and
a miss is then caught only where the object turns out not to be: where it lay, once the arm uncovers it, or at the
release. IK solved tick by tick notices a change of arm configuration only when it comes. Every update depends on the
trackers' own confidence in their measurements.

## Open questions

<a id="q1"></a>**Q1. How many times may a Milestone fail before the act gives up?** More tries recover more failures,
and waste more time on a task that cannot succeed. Leaning: twice.

<a id="q2"></a>**Q2. After a failure, how does the arm get back to the earlier point?** Straight there, after opening
the gripper; or by playing the demo's motion backwards. Straight is shorter; backwards follows a path the demo showed
was clear of the object. Leaning: straight there. In contact the question does not arise: a moved object
always backs the arm out along the demo's path (see [Going back, or ahead](#going-back-or-ahead)).

To measure: the checks on the act records, with real and made-up misses ([R3](#requirements), [R8](#requirements));
how much each object's measurements wander at rest on the rig, per act; the OpenArm 2's following error and gripper readings; how long the command queue takes to fill on the rig
under load; how long ranking the ring's allowed turns takes.

## Glossary

- **Progress:** how far through the demo the act is: the leg, and how far into it, of the last command sent to the arm.
- **Command:** what the arm is told for one motor tick: a pose and a gripper value, turned into joint angles.
- **Leg:** the demo between two key frames.
- **Milestone:** a point where the demo shows an object picked up or put down, or where the operator marks one, with
  the check that the act did the same; _final_ when it cannot be undone.

## Appendix: evidence

Taken 2026-10-17 on `proto/show-and-servo` (`be2e1822e` to `d8d36d606`), from the insertion demo, the acts' own
records on the rig, and benchmarks on the same desktop.

<a id="e1"></a>**E1. IK timing.** Pink, 50 QP iterations a call, targets 1 to 100 mm and 0.5 to 30° from a working
pose. SO-107: one call 0.2–0.8 ms, a solve to 0.5 mm 0.6–2.6 ms (4 at most). OpenArm 2 (Pink, URDF): 0.2–0.9 ms and
0.3–1.6 ms (12 at most). On the rig during act `20261017_125408`: 1.7 ms median, 20.8 ms 95th percentile, 74.8 ms
at most (`stream_diagnostics.solve_ms`).

<a id="e2"></a>**E2. Joint speed.** The teleoperated insertion demo reached 119 °/s at the shoulder lift, 137 °/s at
the elbow, 116 °/s at the wrist flex; following error 1.9° at the median at 10 to 30 °/s, 6.3° at 60 to 100 °/s; the
reading trails the command by 2 to 4 ticks.

<a id="e3"></a>**E3. Stops.** Command standstills over 100 ms between the first and last key frame: 15 (3.1 s) in act
`125408`, 16 (5.1 s) in `130202`, 12 (2.6 s) in `150843`. Across the withdrawal-to-lift boundary of `150843`, by tick:
26, 25, 23, 20, 17, 13, 8, 3, 6, 10, 14, 18, 20, 22, 23 °/s.

<a id="e4"></a>**E4. Re-planning.** Between streams: 2.13 s in `125408` (1.09 s settling), 3.53 s in `130202` (2.52 s
after the grasp: the carry failed at the start's landing turn and every turn was ranked again), 1.41 s in `150843`.

<a id="e5"></a>**E5. The joint cap.** `jog.RAMP_DEG_S` = 30 °/s with a 1.5 easing margin: the lines of `150843` ran at
0.64 to 0.67° a tick; its lift away turned the shoulder 62° in 3.3 s, which the demo did in 1.2 s, faster than 30 °/s on
31 of 36 samples (158 °/s at most). Live playback, until 2026-10-18: 3° a tick from the previous command
(`jog.PLAYBACK_LAG_DEG`, [E22](#e22)).

<a id="e6"></a>**E6. The hold at the grasp.** Against the demo's, in the eight acts of 2026-10-16 23:19 to 10-17 13:02:
4.6 to 7.4 mm, 1.2 to 1.8°.

<a id="e7"></a>**E7. Durations**, key frames 2 to 9 of the insertion: demo 13.66 s; act `125408` 19.77 s
(`be2e1822e`); `150843` 18.89 s (`a3e0cf48c`). The insertion's follow: demo 4.81 s; acts 6.20 s, then 5.37 s.

<a id="e8"></a>**E8. Tracker rate.** About 9.5 frames/s with Point2Pose, 2026-10-16
([action_queue_timing.md](../../gui/docs/action_queue_timing.md)).

<a id="e9"></a>**E9. Gripper width.** 264 act records, 77 with a grasp check: holds (75) stopped 0.85 to 2.40 units
short of the command, misses (`20261015_213310`, `_221031`) 0.69 and 0.41; the reading settles 0.10 s after the
closing command ends (0.40 at most, 95 acts). 1 unit = 14.9 encoder counts.

<a id="e10"></a>**E10. Gripper load at rest.** Median over 0.5 to 2 s after the closing: holds 144 to 232, misses 20 and
34, open at most 24. Hold evidence from vision came 9.9 to 21.8 s after the closing in 8 acts; 110 attempts failed
over 48 acts.

<a id="e11"></a>**E11. The release.** The reading follows the opening command within 0.03 s; the load falls from 200 to
28–44 at 0.4–0.5 s, then reads 210 to 264 while the reading trails the command by 4 to 5 units.

<a id="e12"></a>**E12. The resting place.** The object's mask and depth on the act's first frame; per later frame, the
share of those pixels reading the object (within half its height), the table, or nearer. Act `20261017_150843`: nearer
(the arm) until 4 s after the closing; empty from 4.45 s, then empty at 0.50 or more (10th percentile) and the object
at 0.00 (90th). Act `_125408`: empty from 4.50 s, then 0.83 and 0.02. The dowel lay 7 to 13 mm above the table.

<a id="e13"></a>**E13. The placed object.** Act `20261017_150843`: of 33 frames after the release, the dowel's view
against its reference accepted none; its tracks were visible on 179 of 323.

<a id="e14"></a>**E14. Pose spread at rest.** Trusted views of the ring standing untouched, 41 acts, 14 views each at
the median: the spread of their translation, at the 95th percentile within an act, 15.2 mm in the median act and
44.9 mm in the worst (raw tracker 19.8 mm in the median act); rotation 2.0° (raw 2.6°); the pose the act aimed with drifted 13 mm over 2.3 s
with the ring untouched.

<a id="e15"></a>**E15. 2D co-motion.** 49 acts of 2026-10-15 to 17 with frames and a closing; reference 0.3 s after
the closing, verdict at the first frame with the fingertip's projection 25 px away. 22 verdicts (five or more tracks
in common): the tracks' median displacement over the fingertip's 0.29 to 2.38, median 0.91; 20 acts had 0 to 4 tracks
left. At rest over the approach of 39 acts: 33 at 0.006 to 0.034, no track over 8 px; 6 at 0.20 to 1.85, every track
shifted 7 to 50 px while the act's pose of the object moved 15 to 34 mm. Track ids are stable over an act.

<a id="e16"></a>**E16. The demo's lift.** After its closing command ended, the demo's fingertip had moved 20 mm in
0.27 s, 35 mm in 0.57 s, 50 mm in 0.63 s.

<a id="e17"></a>**E17. The pose in the hand.** Act `20261017_150843`: the tracker's pose of the held dowel existed in
all 86 carry frames, accepted in none, moving 413 mm against the fingertip's 136; perspective-n-point on the 6 to 13
model points still tracked scattered over 1.2 m. The demo's tracker saw the held dowel in 0 of 101 samples.

<a id="e18"></a>**E18. What each frame records.** `frames/NNNNNN.npz`: `track_idx`, `track_uv`, `track_vis`, `model_xyz`,
`model_valid`, `fit_uv`, `fit_inlier`, `delta`, the followed objects under `other_*`; the held object's pose relative
to the fingertip (`held_0`).

<a id="e19"></a>**E19. What a check costs.** On a rig-sized synthetic frame (848 × 480, the rig's intrinsics and camera
pose), mean of 200 calls on this desktop, idle: depth at 1,372 surface points 0.18 ms; 60 tracks against the gripper's
motion 0.06 ms (`lerobot.showservo.engine.predicates`, working tree on `eb29ecba7`).

<a id="e20"></a>**E20. The first engine act on the rig**, 2026-10-18, `d66681568`, the insertion demo at speed 0.5 (act
`20261018_005946`). The grasp's follow was sent over 0.93 s, the demo's time over the speed exactly; the two lines
before it over 1.13 s and 1.20 s against the demo's 1.67 s and 1.61 s, their straight share. Joints trailed their
commands by 1.5° at the median and 3.0° at most, at up to 40 °/s; the motor loop never held the playback. The grasp's
check said success from the tracked points: 3.1 px from "came along" against 21.1 px from "stayed". The straight line
the demo marks after the grasp then drove the held dowel's free end into the ring's side in this layout (the dowel
pointing at the ring); the shoulder's load rose to its maximum and the jog froze the arm. The demo's own path went up
before over; a lift key frame or a follow from the dowel's frame into the ring's would replay that.

<a id="e21"></a>**E21. The second engine act on the rig**, 2026-10-18, `271f1717b` with a lift marked after the grasp, speed
0.5 (act `20261018_052107`). The dowel lay about 6 cm further from the arm's base than in the acts of 2026-10-17.
The gripper held it throughout (reading 95.5 to 97.6 at a command of 100). Through the lift the shoulder moved in
steps — its reading stood still while its command went on, until the error broke it loose — at 24 to 31° with its
load at 400 to 624; in act `20261017_150843` the same lift kept the shoulder at 1.5 to 21°, its load at most 404 and
its lag at most 3.0°. The fingertip followed its commanded path across within 4 mm and fell behind it in height (47
against 71 mm after 3.2 s); the dowel's tracked points moving 40 px toward the image's edge are the lift's perspective,
not a move across. The shoulder then stood 5° behind a still goal at a load of 624, at the same 24 to 26° where the
morning's act (E20) had stopped, until the jog froze the arm. No contact shows in the frames. It stood there because
the motor loop's hold had stopped its goal 5° ahead of it ([E22](#e22)).

<a id="e22"></a>**E22. The motor loop's own hold stalled the shoulder**, 2026-10-18. Act `20261018_052107`'s commands
replayed with nothing held (`/api/jog/replay`, P 32, feed-forward 3.2): lifting, the shoulder stood still at 30.6°,
27.1° and 24.4° while its load rose to about 530, then moved on, each time. At 24.35° the replay's commands had ended
4.96° from it (6.8° with the feed-forward's −1.87°), where the act's hold had stopped them; its load settled at 624
and it stood for 4 s. Four end-to-end SO-107 datasets on this machine (`cylinder_ring_assembly`,
`cylinder_socket_merged_head`, `eval_cylinder_socket_feb_15`, `eval_act_pickup_socket_mar_01`; the last two from before
the follower's gains were configurable, so P 16 and no feed-forward), sent every frame with nothing held back: the
shoulder lift trailed its command by 1.5 to 1.9° at the median and 7.8 to 12.5° at the 95th percentile, more than 5°
behind on 17 to 29% of frames, and the arm followed. Its joints turned at 42 to 161 °/s at the 99.9th percentile; in
the jog's act records (2026-10-05 to 18, 106,545 ticks) at 56 to 136 °/s, 224 °/s at most. Neither act of 2026-10-18
was slowed by the engine's 3° joint bound (`stretched_s` 0.0).

<a id="e23"></a>**E23. A lift stopped short by a joint's range**, 2026-10-18, act `20261018_094848` (speed 0.5, P 16, no
hold): the grasp and the lift went through. The dowel lay near the arm's base, so lifting it straight up at the grasp's
turn folded the wrist: wrist flex reached its servo range (−93.3°) 58 mm into the 100 mm lift, with the forearm roll
at −2°. The lift is best effort, so it went on with its commands below their poses, 30.4 mm at the top (orientation
within 0.2°). The line after it started from the lift's planned top, no joint's reach: its first sample was 31.9 mm
out and the act gave up, though both its ends were reachable (0.2 to 0.4 mm, solved from the lift's last joints).
The demo itself had turned the forearm to 43° during its lift and to 76° before the pre-place. Fixed: a best-effort
command's pose is where its joints get, and the run after a best-effort leg starts there.

<a id="e24"></a>**E24. The hold taken from the command, and a fall reported as a success**, 2026-10-18, act
`20261018_111834` (`6facafcb7`, speed 0.5, `white_left`: P 32, feed-forward 3.2). The grasp, lift and carry ran; the
grasp's check said success (tracks 17-27 px from "came along", 123-132 px from "stayed"). The dowel hovered about 5 mm
off the ring's hole (seen by the operator; the camera above cannot tell inside from above), was let go, fell over and
rolled away, and the act ended "ok" with the release "could not tell" (no tracked point in two frames, the depth
wandering 24 mm). At the firm grip the fingertip read 6.3-7.5 mm and 1.7-2.1° from its command, the grasp's usual
miss (O5). The hold is measured from the fingertip at that moment, but the GUI's fingertip lookup interpolates
between two encoder reads and none comes after "now", so the engine took the commanded pose and held the dowel as the
demo did. The key-frame runner's acts that inserted (`20261017_130202`, `_150843`) read the fingertip after the
gripper settled: holds of 5.9 and 4.8 mm. Fixed: "now" is the latest read, at most two ticks old; the grasp's record
says where the fingertip came from and the hold it gave; an act whose checks cannot tell ends "not confirmed".

<a id="e25"></a>**E25. What reached the engine of the trackers' measurements, and its slowest solves**, 2026-10-18.
Act `20261018_113106`: Point2Pose measured the dowel on all 214 frames; the old runner's acceptance chain let 7
through, none once the arm approached it. Of the 207 it turned away, 163 failed its point-group agreement (3 mm, 3°),
37 had agreeing groups and failed its other rules (share of tracked points seen, 97%, `runtime.TRUST_SHARE_DEFAULT`,
chosen on 5 replayed acts because tracked points slid onto the arm under the gripper; inlier counts; depth), 7 had no
group pose to compare. The ring: 41 of 206 frames. The slowest IK solve in act `20261018_111834` took 299 ms; in
`20261018_113106`, 12 ms.

<a id="e26"></a>**E26. A pushed object, taken by the world and jumped to by the commands**, 2026-10-18, act
`20261018_135723` (`7442baa2f`), replayed offline from its record (its measurements, joint readings and finds, on its
own clock; the replay's commands within 0.1-2.2 mm of the act's, the same abort at the same leg). As the fingers came
down for the grasp they pushed the object 11.1 mm and turned it 10.2 deg (the frames before and after differ along its
whole body). The world took the move on the second measurement showing it, the grasp's commands — made from the
object's pose each tick — moved by that between two ticks, no arm configuration one tick away reached it, and the act
gave up: "the arm would have to jump to another configuration". The moved-target rule, which looks at distance only
(threshold twice the noise, 11.3 mm), did not back out. With the commands' poses following the world at the walk
speed, the replay goes through the descent and the grasp begins.

<a id="e27"></a>**E27. How a held object sits, gated on its own prediction**, 2026-10-18, act `20261018_113106`
(`7442baa2f`), replayed offline as E26. The closing fingers pushed the object 14 mm (7.1-7.8 s; the fingertip moved 5 mm) just
before the grasp began, and the world had not taken the push: its noise, measured over every view until the
grasp, counted the push's own views and rose from 4.5 to 6.6 mm, so the move threshold stood at 13.3 mm. How the
object sits was taken from that pose, 14-17 mm from where every view after the grasp put it (the views drawn on the frames
lie on the object; the pose taken at the grasp does not). The views could replace it only where the grasp's check
answered "still held", and the check predicted from that same pose: for 2.9 s the views were "cannot tell" or
"neither" (one counted a failure, of the two that send Progress back), then the first "held" replaced it by 19.7 mm at
once. At 12.1-13.3 s, turned upright for the insertion, the object slid and turned in the fingers: 24.1 mm and 20.8 deg between the
medians of its views carried lying (7.9-11.8 s) and standing (14.0-21.5 s, 95% of them within 3.1 mm of theirs); how it sits changed 18.4 mm and
12.9 deg on one update, the commands carrying it jumped by that, and the act gave up at key frame 5. The live act, on
the pose taken at the grasp, put the object in the ring.

<a id="e28"></a>**E28. A push counted as noise**, 2026-10-18, acts `20261018_113106` and `20261018_135723`, replayed
offline. An object's noise was the spread of every view of it until the act touched it, and a view counted toward
that noise before the move test ran on it. The fingers coming down pushed the object 14 mm and 11.1 mm; the push's own
views raised the noise (4.5 to 6.6 mm, 4.9 to 5.7 mm) and with it the move threshold (13.3 mm, 11.3 mm), just past the
push, and no move was seen. Counting only the views taken as still, each by its distance from the pose the views
before it made, the first act's noise is 2.4 mm before the push and the push is a move at 7.47 s, before the grasp; in
the second the push is seen during the grasp's descent, and the arm backs out and comes in again.

<a id="e29"></a>**E29. A turn no view can tell, driving the wrist**, 2026-10-18, act `20261018_113106`, replayed
offline with E27's changes. The object is declared round about its long axis with interchangeable ends. Of the 20.8
deg between how it sat carried lying and standing (E27), its long axis tilted 5.7 deg; the rest was a turn about that
axis, which the fit of a half-seen round surface leaves free. How it sits took the whole turn, and to stand the
object as the demo did the gripper turned against it until the wrist's bend reached its bound (-93.31 of -93.32 deg):
"key frame 7: the arm reaches only within 5.4 mm". Keeping, of the equivalent poses, the one nearest the demo's grip,
the first move in the gripper falls from 11.0 to 3.8 deg, the wrist stays off its bound, and the act goes on into key
frame 7, where the ring's views stop it ([E30](#e30)).

<a id="e30"></a>**E30. Views every other check rejected, taken**, 2026-10-18, act `20261018_113106`, replayed offline
with E27-E29. Once the object is in the ring's hole and the gripper covers part of the ring, Point2Pose's fit of the
ring is left with 14-31 points (50 at rest), sitting 10-13 mm higher than before, and the pose that fits them is
tilted 45-70 deg and 7-16 mm higher; the ring's own visible surface in the depth image rose 2-3 mm, the frames show it
flat, and the tilted views go on with the arm gone (44 deg at 25.9 s). Point2Pose marked them "ok", which is all the
engine's feed reads; the check recorded beside it (Point2Pose against the point groups, within 3 mm and 3 deg) and the
old runner's trust rejected them, and the point groups, not seeing enough of the ring, kept it where it was. That
check also rejects every view of a held object, the right ones included: the point groups carry a held object with
the table. The world moved the ring, the moved-target rule backed out, and the way out needed a joint jump. Two
checks tried on the act do not separate these views from the held object's good ones: model points the depth sees
past (0-10% of the ring's against 4-33% of the object's), and the posed demo surface against the depth inside the
object's mask (median 3.6-8.1 mm against 4.3-5.0 mm).

<a id="e31"></a>**E31. The key-frame runner's checks as the estimator**, 2026-10-18, acts `20261018_113106` and
`20261018_135723` replayed offline with each estimator (their verdicts as recorded live).

| Act                             | Point2Pose's verdict                                  | The key-frame runner's checks                                                                                                                                                                                                 |
| ------------------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 113106                          | stops in key frame 7 on the ring's tilted views (E30) | the whole motion; the insertion's commands within 4.8-7.0 mm of the live act's, which put the object in the ring; ends "not confirmed": no view of the object certified after 0.5 s, so neither grasp nor release was checked |
| 135723 (its record ends at 8 s) | backs out of a 4 mm move, then grasps                 | grasps                                                                                                                                                                                                                        |

In 113106 the checks certified 7 of the object's 214 views, all in the first 0.5 s, before the arm moved (its views
carried in the gripper are right: E27), and 41 of the ring's 112 before the insertion, none of its 102 after. They avoid E30 by refusing nearly
every view: the act then runs on the poses it had at the grasp, as the live act did, and checks nothing.

<a id="e32"></a>**E32. Confidently wrong, and the record could not say why**, 2026-10-18, rig act `20261018_162243`
(`ae25eece4`, speed 0.5). The fingers pushed the object 6 mm on the way down and the arm backed out and came in again;
the grasp's check answered no failure; the carry recorded the object moving in the gripper four times (5.2 mm
2.4 deg, 7.0 mm 3.4 deg, 11.5 mm 8.9 deg, 15.1 mm 23.5 deg). The act stopped 39% into key frame 6 ("the arm reaches
only within 3.2 mm"), the wrist at its bound, holding the object north of the ring, off the hole; at the same point
the demo's own camera has it over the hole. Just before the stop the object stood upright in the gripper and
Point2Pose's view, which the engine took, had it lying diagonally. The record held the trackers' views, the commands
and the size of each change the engine made, not where it believed the object was, so which views moved the
commands, and by how much, could not be read from it. The object was dropped afterwards by the recovery (lowered and
opened above the table), not by the act.

<a id="e33"></a>**E33. Poses the camera sees through**, 2026-10-18, the four rig acts of the day replayed offline. For
each point of an object's demo surface at a view's pose, the depth the same camera frame measured on the ray to it,
against the point's: at rest a median 1.4-1.7 mm apart (95th percentile 5-7 mm for the ring, 10-12 mm for the
object, at its ends). On the wrong views the rays pass through the surface: the upright object seen end-on, the
model lying diagonally where the camera sees the tray about 100 mm farther; the ring with the object in its hole,
the tilted model 10-25 mm in front of the ring's measured top. Over the frames labelled from the images, with each
object's tolerance twice its 95th percentile at rest and its cut-off twice its largest share seen through at rest
(at least two points): every bad view refused (the ring's 93 after the insertion, the object's 6 turning upright),
5% of the good ones refused (36 of 673; a refused view leaves the estimate where it was). Point2Pose's own fitted
points miss the ring's (at most 7% seen through): the check needs the whole surface. Replayed with it, act
`20261018_162243` runs to the end of its motion instead of stopping in key frame 7; `20261018_113106` reaches the
release (its check then answers "neither", 15 mm from staying put and 19 from going with the gripper: the object
leaned in the ring's hole, and the ring's pose stood where its last view before the insertion put it, while the frames
show the insertion pushed it); `20261018_191804` no longer drifts its grip, and stops in key frame 6 with the wrist at its
bound (the demo holds it 0.6 deg from it there), 3 mm short.

<a id="e34"></a>**E34. The first insertion, and a belief left on the tray**, 2026-10-19, rig act `20261019_005211`
(speed 0.5), the engine taking the point groups' poses. It ran all nine key frames and the object stood upright in
the ring after it. No view of the object was taken from the grasp on: the world's pose of it stood at its last view,
from before the grasp, to the end (a view taken would have replaced it). Of the act's 219 frames, 100 had no groups
pose yet (the groups run behind the camera), 76 were refused as seen through (the cut-off measured on the object
lying still: 0.3% of its points), 6 were in a switch of group. So the engine carried the object by how it sat at the
grasp, never corrected, and the insertion came from that grip holding. At the release it went back to the world's
pose, on the tray, and the release could not be told. Replayed with the act's frames and the groups run offline, the
groups' poses taken were 57 of 64 lying, 21 of 38 carried, 1 of 65 upright and 0 of 36 after the release (median 46%
and 91% seen through there). Since then one belief per object: carried, the world's pose is where the gripper has it
(`test_an_object_the_camera_never_sees_again_after_the_grasp_stays_where_the_gripper_left_it`).
