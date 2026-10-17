# Act execution: Progress, Milestones and the state checks

Status: proposed
State of the work: no tracking issue yet

An act replays a demo on objects that have moved. Today it is a string of separate motions: each key frame is
planned, streamed and waited on, and the rest is planned again with the arm standing whenever a stage ends. At
speed 1.0 an act takes 1.4 times the demo's time and stands still 12 to 16 times. It only moves forward and checks
nothing: a missed grasp is followed by a carry, an object moved during an insertion by the next action, and a
placed object that falls over is reported as placed. The same runner is meant to drive the OpenArm 2.

**Proposal.** A new act engine, beside the current runner and chosen per act, plays the act as one continuous
replay. [Progress](#glossary) is a position on the demo's timeline. Each motor tick gets one pose entry, made from
Progress and solved a few ticks ahead with the objects' newest poses, so an object update changes the composition,
not the trajectory. [Milestones](#glossary) sit where the demo changes what an object moves with — the gripper's
closing and opening, or a key frame the operator marks — and are checked from the camera, on every frame, by two
questions: which of two references the object moves with, and whether it sits where the demo's relation puts it.
Progress runs ahead of the checks; when a check says no, an object moves, or a pose cannot be reached, one rule
places Progress at the nearest state of the demo consistent with what the camera shows, backward or forward.

## Scope

In: one act on one arm, first key frame to last: legs, Progress, the queue, the checks, the Progress update,
candidates (symmetry and landing turns), speed, the act's record, and how the engine sits beside the current runner.

Out, and why: estimating object poses ([act_loop.md](act_loop.md) designs it; this design consumes it); the motor
loop (`gui/api/jog.py`), which already sends the newest due command each tick; two arms at once, whose Progress would
need coupling; failures that need a model of cause ("knocked something over") or a law fitted to the demo (a screw's
advance per turn); what is pinched in a cloth grasp, which takes touch
([state_check_research.md](state_check_research.md)). Non-goals: collision checking beyond the workspace box and
floor; choosing what to do after the act fails, which belongs to the agent above it.

## Requirements

Safety first: the act must not act on a state that is not there. Then the motion should look like the demo's.

| #   | Pri | Requirement                                                                                                                                                                                           | Target                                                                                                                                                           | Why                                                                                                                           | Checked by                                                |
| --- | --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| R1  | P0  | Progress never passes a Milestone whose check says no, nor a _final_ one before every check before it has said yes                                                                                    | 0 crossings                                                                                                                                                      | A missed grasp followed by a carry ([O2](#o2)); a step that cannot be undone must not be taken on a guess                     | Unit tests; every act's record                            |
| R2  | P0  | One Progress update for every disturbance: the nearest state consistent with what is observed, backward across repeatable Milestones, forward up to the next unmet one, never back across a final one | Grasp missed: grasp again; dropped: the approach at its new pose; target moved nearer: ahead; moved away mid-insertion: back up; final crossed: the act fails up | The user's cases of 2026-10-17; detecting a failure and readjusting are one mechanism                                         | A simulated act per case                                  |
| R3  | P0  | A check says yes only on evidence that separates its two outcomes                                                                                                                                     | 0 yes on the recorded misses and on negatives made by substitution                                                                                               | A false yes is R1's carry, approved                                                                                           | Unit tests; a replay of the act records                   |
| R4  | P0  | Only views the pose rule accepts move an object, or Progress                                                                                                                                          | 0 updates from views it turns away                                                                                                                               | Views 7.6 to 29.8 mm off were accepted by the tracker ([act_loop.md](act_loop.md) O3)                                         | A test feeding such views                                 |
| R5  | P0  | No check, object update or candidate choice delays a command                                                                                                                                          | No standstill over one tick from any of them                                                                                                                     | The arm stood 0.06 to 2.5 s while the act re-planned ([O1](#o1))                                                              | Simulated acts with moving objects; the record on the rig |
| R6  | P1  | An act nothing disturbs, at speed 1.0, takes the demo's time, whatever the demo's frame rate                                                                                                          | At most 1.15 × the demo's, for demos at 15 to 60 frames/s                                                                                                        | 1.38 × today ([E7](#e7)); the arm trails its commands by about 0.1 s ([O4](#o4))                                              | A simulated act per frame rate; per leg on the rig        |
| R7  | P1  | The arm stands still only where the demo did, at a final Milestone, or while a Progress update's gripper moves                                                                                        | No other standstill over 100 ms                                                                                                                                  | 12 to 16 per act today, most at key frames ([O1](#o1))                                                                        | The act's command record                                  |
| R8  | P1  | A check answers soon after the demo's motion makes the answer visible, and a no reaches Progress at once                                                                                              | Within 1.0 s of the references separating by 30 mm; the update within two camera frames of the one that shows it                                                 | The demo's lift separates gripper and table by 35 mm 0.57 s after its closing ([E16](#e16)); frames come at 9.5/s ([E8](#e8)) | A replay of the act records; simulated acts               |
| R9  | P1  | One engine drives the SO-107 and the OpenArm 2                                                                                                                                                        | The same arm interface                                                                                                                                           | The OpenArm 2 is the next arm; the runner names the SO-107's motors ([O9](#o9))                                               | The engine's tests on both arms' kinematics               |
| R10 | P1  | The engine runs beside the current runner, chosen per act, and leaves it unchanged                                                                                                                    | Either from the same page and API; the current runner's tests pass unchanged                                                                                     | Acts with both on the same scene decide the switch                                                                            | The existing tests; the record names its runner           |
| R11 | P2  | Joint speed is bounded by the arm and the motor loop, not the ready/park ramp                                                                                                                         | 90 °/s on the SO-107; the timeline slows where a step would exceed it                                                                                            | The cap is 30 °/s; the demo reached 158 °/s ([O4](#o4))                                                                       | A simulated act faster than the bound                     |
| R12 | P2  | Every verdict and Progress update is in the act's record with its evidence                                                                                                                            | One record per decision                                                                                                                                          | Failures are diagnosed from the act's own record ([E7](#e7))                                                                  | A test of the record                                      |

Conditions: the SO-107 (left arm) and its RealSense on this desktop; the insertion demo of 2026-10-17 and the acts of
2026-10-15 to 17 with saved frames.

## Observations

<a id="o1"></a>**O1. Every key frame is a stop, and every stage end a re-plan.** Each key frame is streamed and waited
on until the arm is within 2° of it (`_act_key_frames_task`, `live_execution.stream`); lines brake at the key frame
they end on: 26 °/s down to 3 and back up ([E3](#e3)). The arm stood 0.06 to 0.8 s at stage ends and 2.5 s after a
grasp while the carry was planned twice ([E4](#e4)). So the stops come from how the act is cut, and retargeting must
not depend on planning again.

<a id="o2"></a>**O2. Progress only moves forward.** The phase a stream aims at only increases, and the runner walks the
key frames in order: an object moved during a grasp is followed by a closing gripper, a missed grasp by a carry.

<a id="o3"></a>**O3. IK is cheap; choosing among candidates is not.** A solve to 0.5 mm takes 0.3 to 2.6 ms on both arms;
under the rig's load, 1.7 ms at the median, 21 ms at the 95th percentile, 75 ms at most ([E1](#e1)). Ranking the allowed
turns solves every sample of every turn and sat inside the 2.5 s stand of [O1](#o1). So IK can run per tick, a few
ticks ahead; ranking must run beside the motion.

<a id="o4"></a>**O4. The arm can go several times faster than the act lets it, and trails its commands.** The demo
moved joints at up to 158 °/s with about 3 ticks of lag; the act caps them at the ready/park ramp's 30 °/s ([E2](#e2),
[E5](#e5)). The jog's live playback moves each command at most 3° a tick from the one before, sends the newest due
sample each tick, and never skips a gripper start, stop or reversal (`jog._advance_playback`). So the bound is 90 °/s,
and queued commands can always be replaced.

<a id="o5"></a>**O5. The grasp never lands where it was aimed:** 4.6 to 7.4 mm and 1.2 to 1.8° from the demo's in
every act of 2026-10-16/17 ([E6](#e6)). So what follows a grasp is composed with the hold measured there.

<a id="o6"></a>**O6. Object poses come at 9.5/s and spread 15 to 20 mm at rest** (95th percentile per act; [E8](#e8),
[E14](#e14)). So "the object moved" needs a margin above that, and a pose comparison is no finer than it.

<a id="o7"></a>**O7. The camera tells which reference an object moves with once the references separate, and often
cannot tell before.** Once the fingertip's projection has moved 25 px, the object's tracks have moved 0.84 to 1.0 of
it when held and 0.01 to 0.03 of it at rest ([E15](#e15)); the resting place reads empty once the arm uncovers it
([E12](#e12)). In 20 of 49 acts the gripper covered the tracks; the held object's 3D pose was wrong in every carry
frame; the placed object's reference view placed it in 0 of 33 frames ([E13](#e13), [E17](#e17)). So a check is a
choice between two references that is undecidable until they separate, and "cannot tell" is a normal answer.

<a id="o8"></a>**O8. The gripper's readings say that something stopped the fingers, not what, nor where it is
later.** Width and load separate a hold from a miss 0.25 s after the closing, and the load is servo effort while the
gripper moves ([E9](#e9) to [E11](#e11)). So they can join a check as evidence, not carry it.

<a id="o9"></a>**O9. Games and simulators need few predicates, and none of them knows "cannot tell".** Held, in a
place, on, part moved, at rest; the VR grab-break rule declares a grab broken when the object drifts past a threshold
from hand times grab offset, with no rigidity assumed; ManiSkill and robosuite count a placement only once the
effector has left it ([state_check_research.md](state_check_research.md#how-games-and-simulators-define-the-same-predicates)).
The demo is the only statement of success here, and the camera is calibrated to the arm's base, which moves on a
mobile robot. So the checks compare an object against references perception tracks — the gripper, other objects, the
base — as the demo had them.

<a id="o10"></a>**O10. The two arms differ in kinematics, gripper and motor only.** The SO-107 runs Pink, the OpenArm
2 the vendor's mink solver (a Pink build of its URDF exists); the OpenArm 2 has 7 arm joints. The runner takes the
SO-107's motor names (`joint_alignment.MOTOR_NAMES`). So the engine needs kinematics, joint ranges, the per-tick bound
and the gripper's column and closing direction, nothing else.

## Architecture

```
 demo + key frames ──► legs (start frame → end frame)  +  Milestones at the demo's reference changes
                                       │
 perception ──► scene log ──► checks: each camera frame, the Milestones whose window holds Progress ──► verdicts
                                       │                                                                │
                      Progress ◄── Progress update: a verdict changed, an object moved, a pose unreachable
                         │
                         ▼
                 generator ──► command stage (IK, 4 ticks ahead) ──► jog playback ──► motor loop, 30 Hz
```

**The queue.** What is sent, what is solved, and what each event changes:

```
 timeline    ──┬── leg 3: follow, dowel → dowel ──┬── leg 4: lift ──┬── leg 5: line, arm → ring ──┬──
                         closing ▲ attached's window opens ─────────────────────────────────────► (until the opening)
                                                      ▲ Progress = the last entry sent

 playback    … e41  e42 │ e43  e44  e45  e46 │ e47 …
               sent     │ solved: the lead,  │ not made yet: each is made when it comes within the lead, from
                        │ 4 ticks, 133 ms    │ its timeline position and its frames' poses at that moment

 each tick     the motor loop sends the oldest due entry; the generator makes and solves one more
 object moves  nothing queued changes; entries made from now on compose with the new pose, rate-limited
 a check: no,  the unsent e43 … e46 are dropped; the update picks a state; a transition from e42's command to that
 a big move    state, then the timeline from there, refill the lead before the motor loop next ticks
```

1. **Legs.** A leg spans two key frames. Its _start frame_ is the frame of the key frame it starts at — an object or
   the base — or, for a line, the arm's pose where the line begins; its _end frame_ is the frame of the key frame it
   ends at. An entry at fraction `s` of the leg is the demo's guided pose at that moment composed with each frame's
   motion since the demo, blended from the start frame's to the end frame's as `s` goes from 0 to 1 (a line: from
   where the arm was to the key frame). With nothing moved both are the demo's pose. A follow within one object, a
   line from where the arm stands to an object, and a carry recorded from a pick straight to a place are one rule,
   and an object that moves bends what is left of the leg instead of making it jump. While an object is attached the
   composition includes the hold measured at the grasp ([O5](#o5)).
2. **Progress and the clock.** Progress is the position of the last entry sent. Each entry advances the timeline by
   one tick times the act's speed: a follow's duration is the demo's, its poses interpolated between the demo's
   samples, so any demo frame rate plays in its own wall-clock time; a line's is its length at the demo's pace
   between its key frames (`core.demo_pace`, `core.walk_scale`). A line eases in only where the arm starts from
   standing and eases out only where the demo stood still, so the arm passes key frames at speed. ([R6](#requirements),
   [R7](#requirements), [O1](#o1))
3. **The command stage.** It solves each entry as it comes within the lead, four ticks, longer than the slowest solve
   on the rig (75 ms, [O3](#o3)): IK seeded from the previous command plus the demo's joint change over the same
   moment, then the per-tick joint bound, the workspace box, the floor and the joint ranges. A step over the bound is
   shortened by advancing the timeline less on that tick, so the arm slows on its path ([R11](#requirements)); one
   that stays over it however short is a change of configuration and is unreachable. Entries go to the jog's live
   playback, which keeps its own clock and lag guards ([O4](#o4)).
4. **Retargeting.** A frame's pose reaches the composition through a rate limit, 0.10 m/s and 60 °/s: a correction
   the size of the pose rule's spread (20 mm, [O6](#o6)) spreads over six ticks. Solved entries keep the pose they
   were solved with; the lead bounds how stale that is. ([R5](#requirements))
5. **Milestones and when they are checked.** A Milestone sits where the demo changes what an object moves with: the
   gripper's closing (the object of the leg the closing is in goes from that leg's start frame to the gripper), its
   opening (from the gripper to the end frame of the leg it opens in), or a key frame the operator marks with the
   object and its new reference. Its _window_ runs from that event to the object's next one, or the act's end. While
   Progress is in the window, every camera frame updates its verdict: yes, the object moves with the new reference
   and sits relative to it as the demo's did; no, it moves with the old one or is not where the demo's relation puts
   it; cannot tell, until the two references have separated by 25 px in the image or 30 mm. A check costs 0.06 to
   0.18 ms per frame ([E19](#e19)), so all the open windows are checked on every frame, beside the command stage.
   Progress does not wait for them: it runs on the demo's promise, and a no, repeated on two frames, reaches the
   Progress update within two frames of showing (0.21 s) plus the lead (0.13 s) — motion the update then undoes. A
   Milestone the operator marks _final_, for an effect that cannot be undone, is the one place Progress waits: it
   does not pass it until every check before it has said yes, and fails the act up if that does not come within
   2 s. ([R1](#requirements), [R5](#requirements), [R8](#requirements), [O7](#o7))
6. **The predicates.** Two questions, each answered from whichever evidence there is ([O7](#o7), [O9](#o9)):

   | Predicate                   | Evidence                                                                                                                                                                                                                                                                     | Answers, with the measurement the thresholds come from                                                                                                                                                                                                                                                       |
   | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
   | `moves_with(obj, new, old)` | The object's tracks since the event against where each reference's motion would have taken them in the image (the gripper's from the joints, an object's from its views); views that place it, the same in 3D, the grab-break rule; the gripper's width and load, when added | Cannot tell until the references' predictions are 25 px (30 mm) apart, or with under five tracks seen in both frames. New: the observed displacement is 0.5 to 2.0 of the way from old to new, at most half of it across. Old: within 0.15 of it. From [E15](#e15): held 0.29 to 2.38, at rest at most 0.034 |
   | `sits_at(obj, pose)`        | The object's surface points at the pose, nearest per pixel, against the depth frame: present within half its thickness (at least 4 mm), absent where the camera sees past it, covered where something is in front; views that place it, against the pose                     | Depth: cannot tell under 20 points not covered; yes with 60% of them present, no with 60% absent ([E12](#e12)). Views: yes within 20 mm and 10°, no past 40 mm or 20°, turns about a declared symmetry not counted ([E14](#e14))                                                                             |

   A Milestone's verdict combines `moves_with(obj, new, old)` and `sits_at(obj, new reference × the demo's relation)`:
   a yes and a no together cannot tell ([Q3](#open-questions)). Neither predicate names the gripper: at a closing the
   new reference is the gripper and the old is what the object lay on; at an opening it is the other way round, and
   "after the gripper has left" is the separation that makes it decidable. Another sensor — a fingertip skin, a camera
   on the gripper — is more evidence for the same two questions.

7. **The Progress update.** One rule, run when a verdict changes, when a view the pose rule accepts moves an object
   by more than 20 mm or 5° ([O6](#o6)), or when an entry is unreachable: Progress may be placed at any state of the
   timeline whose expectations no current verdict contradicts, no further forward than the next Milestone not yet
   met, and never back across a final Milestone; of those, at the one nearest to where the guided pose is now, in
   each candidate state's frames, plus a cost for moving along the timeline. A Milestone gone back across is unmet
   again; one that has failed twice fails the act up ([Q1](#open-questions)). The transition to the chosen state is a
   line at the walk's pace times the act's speed, the gripper moved before it if that opens it and after it if that
   closes it. What each disturbance becomes ([R2](#requirements)):

   | Disturbance                                      | Verdict                                                                  | The nearest consistent state                                |
   | ------------------------------------------------ | ------------------------------------------------------------------------ | ----------------------------------------------------------- |
   | Grasp missed (absent)                            | the closing's: moves with what it lay on                                 | before the closing, near where the arm is: grasp again      |
   | Dropped in the carry (incomplete)                | the closing's, while its window is open                                  | before the closing, at the object's new pose: the approach  |
   | Let go off target, or carried off (misdirected)  | the opening's: not where the relation puts it, or moves with the gripper | before the closing, at its new pose                         |
   | Tipped or rolled after the release (unsustained) | the opening's, while its window is open                                  | before the closing, at its new pose                         |
   | Target moved nearer or away, no verdict changed  | —                                                                        | ahead or back along the same stretch, as near as the arm is |

8. **What the engine keeps of the key frames.** Every option a key frame or a demo has today has a place:

   | Today                                                     | In the engine                                                                                                                                                                        |
   | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
   | `frame`: an object, the taught object, the world          | the end frame of the leg arriving at it, the start frame of the leg leaving it                                                                                                       |
   | `move`: line, follow, lift                                | line: from where the arm is to the key frame; follow: the demo's samples; lift: up by `up_mm` from where the arm is, in the base                                                     |
   | `grip`: first, along                                      | a line's gripper moves before its motion, or with it                                                                                                                                 |
   | `constraint`: pose, orientation                           | the whole pose; or the rotation only, the position kept from the leg's start                                                                                                         |
   | `constraint`: best_effort                                 | the key frame is a target, not a requirement: when the arm stops short (the lag guard holds it, no joint progress), the leg ends where the arm stands and the next starts from there |
   | a firm grip until the release (`core.demo_holdings`)      | the legs between compose with the hold measured at the grasp, so the held object goes where the demo's did                                                                           |
   | an object's symmetry: discrete, continuous, end swap      | the turns its frame may take (candidates); checks do not count them                                                                                                                  |
   | the demo's landing: exact, symmetry, turn, about its axis | candidates for the legs framed on the object landed on                                                                                                                               |
   | point groups carrying objects between views               | another source of frame poses                                                                                                                                                        |

   Candidates are ranked beside the motion when their object moves — IK of the leg's key frames and a few samples,
   seeded from the current joints, by reach and by distance from the demo's joints — and a different turn is taken
   only before its leg is entered and only when it ranks better by a margin. ([O3](#o3), [R5](#requirements))

9. **The seams.** The engine (`lerobot.showservo.engine`) holds no GUI state. It reads an arm (kinematics, joint names,
   the gripper's column and closing direction, ranges, workspace, per-tick bound), a playback (the jog's live queue:
   sent, append, drop unsent, its clock) and a scene log perception writes into (views, tracks, depth frames, the
   arm's fingertip, the camera); the GUI's adapter fills them, and the act request names the runner. It records every
   verdict with its evidence, every update with its cause and every entry sent; the act's end says which Milestones
   were met, which failed and why, and which were passed while the check could not tell, kept apart from a fault that
   aborted it. ([R9](#requirements), [R10](#requirements), [R12](#requirements), [O10](#o10))

## Alternatives, and what this costs

- **Keep the runner and plan ahead of the arm** (`a3e0cf48c`, `d8d36d606`): shortens the stops at stage ends, keeps
  the stops at key frames, re-plans on every move, never goes back, checks nothing.
- **Change the runner in place:** its stops and re-plans are its structure ([O1](#o1)); and without the old runner
  there is nothing to compare acts against while the new one is unproven.
- **Check at key frames, or gate every Milestone:** a key frame comes before or after the moment the references
  separate, and gating stands the arm still where the check needs it moving ([O7](#o7)).
- **The gripper's readings as the check:** fast, but they say nothing of where the object is later ([O8](#o8)).
- **A learned success or stage detector (SARM, in this repository; a VLM judge):** seconds per answer, needs
  annotated demos, cannot say which fact failed; fits as a slow layer that contradicts a Milestone late.
- **A model of supports and surfaces:** needs the table found and tracked, and reads a deliberate drop as a failure;
  the demo already says where the object ended relative to what.
- **Re-plan whole joint trajectories fast (batched GPU):** still needs Progress and Milestones, and takes the GPU the
  trackers share.
- **A state machine per task:** each task would carry its own recovery; interactions stay generic.

What it costs. Two runners exist until the switch. Running ahead of the checks spends motion that a no then undoes,
up to about a third of a second's worth plus the transition back; an attached Milestone whose tracks are covered is
passed on the carry's promise, and its miss is caught only where the object turns out not to be — at the resting
place, or at the release. Per-tick IK sees a change of configuration only when it comes. Every update depends on the
views the pose rule lets through. Every threshold is from one rig and one camera. There are more threads to keep
correct.

## Open questions

<a id="q1"></a>**Q1. How many times may a Milestone fail before the act fails up?** More tries recover more failures
and waste more time on a task that cannot succeed. Leaning: twice, then up.

<a id="q2"></a>**Q2. Leave contact along a line, or back along the demo's path?** The demo's path was free of
collision relative to the object; a line is shorter and does not assume the object is where it was. Leaning: the line,
gripper first if it opens, until a measured case shows it dragging the object.

<a id="q3"></a>**Q3. When two kinds of evidence disagree, which wins?** The tracks may say the object came along while
the depth still shows it lying there. Leaning: neither; it cannot tell, recorded with both.

To measure: the predicates on the act records, misses and substituted negatives included ([R3](#requirements),
[R8](#requirements)); the OpenArm 2's following error and gripper readings; the command stage's latency on the rig
under load; the time to rank the ring's allowed turns.

## Glossary

- **Progress:** the position on the act's timeline of the last entry sent: a leg, and how far into it.
- **Leg:** the demo between two key frames, with a start frame and an end frame.
- **Milestone:** where the demo changes what an object moves with, and the check that it did; _final_ when it cannot
  be undone.
- **Window:** the stretch of the timeline over which a Milestone's check runs: from its event to the object's next.
- **Lead:** how far ahead of being due an entry is solved: four ticks.

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
31 of 36 samples (158 °/s at most). Live playback: `jog.PLAYBACK_LAG_DEG` = 3° a tick from the previous command.

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

<a id="e14"></a>**E14. Pose spread at rest.** Trusted views of the ring, 41 acts: translation 15.2 mm at the 95th
percentile per act (raw tracker 19.8), rotation 2.0° (raw 2.6°); the pose the act aimed with drifted 13 mm over 2.3 s
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
