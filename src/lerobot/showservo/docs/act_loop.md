# The act as one loop between marks

Status: proposed
State of the work: no tracking issue yet

The act replays a demo on objects that have moved. Each of its stages decides on its own where the objects are and
how the arm follows them, and only one stage guards against an object the gripper partly covers. On 2026-10-09 two
acts failed at the same moment, the wrist over the gamepad. One planned its grasp from a view that put the gamepad 22
degrees off, and closed on nothing. The other re-aimed the arm at each new view for 2.6 s while the views disagreed by
tens of degrees, pushed the gamepad round, and had its grasp refused by the planner. Each stage's rule was added where
a failure showed itself; fixing this one in the stage where it showed would add one more.

**Proposal.** The act runs one loop over the demo from mark to mark. Every tick, each object's pose is its last pose
moved with whatever supports it, and a new tracker view replaces it only when the view's own fit points place the
object's middle within the act's reach tolerance. The arm's target is the demo's next sample carried by the pose of
the object the current [leg](#glossary) follows, and the arm steps toward it within its speed limits. Grasp and place
are legs like the others, in which the gripper command changes. Nothing is played back without looking, and there is
no wait for the object to hold still.

## Scope

In: the act from the first pre-grasp to the end of the place; the rule that decides each object's pose, for the
object picked and the one placed onto; how the arm follows.

Out, and why: the finds that register the live view against the demo's (they start the act, and none of these
failures came from them); the reset policy (a separate program); recording and editing demos; the internals of the two
estimators, Point2Pose and the point groups, beyond what the loop needs from them ([C3](#constraints-and-freedoms)).

Non-goals: faster acts; grasps that need force or contact sensing.

## Requirements

The arm must not move an object it has not grasped; then the grasp must land; then the rules must be the same
everywhere, so the next failure is fixed in one place.

| #   | Pri | Requirement                                                                                    | Target                                                                                                                    | Why that target                                                                                                                                                             | Checked by                                                                                |
| --- | --- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| R1  | P0  | The arm's target never moves because of a view that does not place the object                  | 0 mm of target motion on a view turned away                                                                               | The wait at 20:35 moved the target 136 mm on such views and pushed the gamepad ([E2](#appendix-evidence))                                                                   | Replay of recorded acts; a unit test that feeds the loop views it must turn away          |
| R2  | P0  | While the object is untouched, the grasp is aimed from a pose within the act's reach tolerance | 3 mm and 3 degrees at the pre-grasp target                                                                                | The reach tolerance (`ACT_REACH_TOL_M`, `ACT_REACH_TOL_DEG`) is set under the hand-eye calibration's own error: an aim worse than it is the estimate's fault, not the arm's | Replay: worst target error while untouched, over recorded acts ([E4](#appendix-evidence)) |
| R3  | P1  | One rule decides every object's pose, in every leg                                             | One function takes or turns away a view; no rule that belongs to one stage                                                | The failures came from rules that differ by stage ([O1](#observations), [O2](#observations))                                                                                | Reading the loop: one pose update, called for every leg                                   |
| R4  | P1  | A covered object keeps its last placed pose, moved with its support                            | Unchanged while it and its support are untouched; within the reach tolerance of where its support took it when that moved | [O3](#observations): covered-object views were 7.6 to 29.8 mm off; the support moving under it is what the point groups are for                                             | Replay of recorded acts; a unit test of an act whose covered object's tray is pushed      |
| R5  | P2  | The gripper closes and opens where the demo's did along each leg                               | The gripper command as a function of progress along the leg                                                               | The grasp is replayed 1:1 today; the close must still come with the fingers around the object                                                                               | Replay of a leg's progress against the gripper; a live trial                              |

Conditions: the replays read five acts recorded on the rig, two on 2026-10-08 and three on 2026-10-09, listed in the
evidence ([E](#appendix-evidence)).

## Observations

**O1. Each stage of the act decides where the object is, and how to follow it, in its own way.** Source:
`_act_task`, `walk_to`, `_still_decision`, `stream` and `_apply_others` in `src/lerobot/gui/api/pregrasp.py` at
`c2f8bedd4`.

| Stage                      | Object       | A new view is taken when                       | The arm follows by                              |
| -------------------------- | ------------ | ---------------------------------------------- | ----------------------------------------------- |
| Walk to the pre-grasps     | picked       | the tracker accepts it                         | a target re-aimed every tick                    |
| Wait at the last pre-grasp | picked       | two views agree; once hidden, the last one     | a target re-aimed every tick                    |
| Grasp                      | picked       | never                                          | a plan made once, played back                   |
| Carry and place            | placed onto  | 97% of its points are seen; otherwise it stays | plans made once, played back; one re-aimed walk |
| Hold in the gripper        | picked, held | views taken while the arm stands still         | one correction before the place                 |

So there is no one place where a covered object is handled: a rule added to one stage leaves the others as they were.

**O2. The object placed onto already has a rule for being covered; the object picked has none.** Its track is
followed only while 97% of its points are seen, because "covered in part, its points drift onto what covers it while
the frame still looks trusted" (`_apply_others`). So the rule exists; it is applied to one object.

**O3. Views of a partly covered gamepad were followed while it lay untouched, 7.6 to 29.8 mm and 13.8 to 21.9
degrees off; the tracker accepted every one** ([E1](#appendix-evidence)). So the tracker's own acceptance does not
tell a view that can place the object from one that cannot.

**O4. The wait at the last pre-grasp moves the arm to every new view** (`_act_task`: "stay with the object"). At
20:35 the fingertip travelled 70.5 mm in 2.6 s, to 5 mm below the pre-grasp, and the gamepad turned
([E2](#appendix-evidence)). So moving the arm on a view is safe only when the view places the object.

**O5. A view's own fit points predict its error at the object's middle, and the prediction separates the views.**
Predicted at 0.78 or less per mm of noise on the points, every view was within 4.9 mm and 3.4 degrees; at 1.20 or
more, all but two were 6.3 to 29.8 mm off. The number of fit points does not separate them; one view of 35 points was
11.3 mm off ([E3](#appendix-evidence)). So whether a view places the object can be decided from the view alone, before
it is used.

**O6. With the point groups view running, the act's tracker took 223 to 278 ms a view, against 147 to 161 without;
the one act since without it stacked** ([E5](#appendix-evidence)). So whatever estimator the loop uses has to be
measured with everything else that shares the GPU.

**O7. The demo's object tracks are exact at each object's click frame and drift elsewhere: at the cube's pose mark
it reads tilted 5.2 degrees though the cube never moved** ([E6](#appendix-evidence)). So a demo pose read from a track
at another frame carries the track's error, as the act's views do.

**O8. In the point groups, a covered object went wherever most of its tracks went, and its hidden tracks voted with
its own seen ones.** When 18 of a gamepad's 30 tracks stuck to a wrist moving on at 4 mm a frame and the rest were
hidden, the seen ones split off into the arm's group, the hidden ones followed them, and the gamepad was carried off
with the arm (`test_tracks_of_a_covered_object_that_slide_onto_the_wrist_do_not_carry_it_off` in
`tests/showservo/test_groups.py`, against `groups.py` before `_place_objects` took the rule). So an object's support
may change only on a view that places it, or with the body it lies hidden in, as its support's points decide.

**O9. In the point groups, a small object at rest wobbled with its own points.** Each frame its own points passed
the rule they re-placed it outright: a 36 mm cube 1 to 5 degrees from one frame to the next while it lay still, and
in the act of 2026-10-10 09:20 the carry turned the cube's place target 2.1 degrees though nothing had moved (act
`20261010_092029`; the same scene in `test_an_object_at_rest_stays_with_its_group_until_its_own_points_show_it_moved`,
4.6 degrees). So an object at rest stays with its group, fitted on hundreds of points, until its own points show it
moved.

**O10. The point groups lose an object in the gripper.** In the act of 2026-10-10 00:55 the gamepad's outline stayed
on the tray while the gripper carried it to the cube, as it had in a replay of the demo before the rule: the fingers
hide most of it, and its hidden points stay with what it lay on. So while the gripper holds it, its pose is the arm's,
which neither the fingers nor the camera hide.

**O11. Views that place a still object still jump its pose.** In the acts of 2026-10-10 00:55 and 09:20, where
nothing moved the cube, each of the 39 and 42 views that placed it moved its middle a median 0.2-0.3 mm but turned it
a median 1.4-1.7 degrees (up to 4.6), 1.1-1.3 mm at the place's fingertip 45 mm above it (up to 3.9); every one
replaced the pose ([E7](#appendix-evidence)). So a view that agrees with the pose held is better averaged in than
taken whole.

**O12. With steady poses, a run of turned views kept the pose turned after the views came right.** In the act of
2026-10-10 11:15, while the grasp lifted the gamepad, the cube's track grew from 25 points to 50, and for 19 views
(2.6 s) it read the still cube turned 5 to 12 degrees from the find, mostly about another axis than the other 31
views (2.6 to 7.6 degrees). The rule placed every one: all their tracks seen, their middles 3.7 to 5.1 mm from the
find's. Steady poses averaged all 50 into one pin, none putting the cube's points 3 mm RMS from where the pin held
them, and when the act planned its carry the pose still held the 19: the place came out 5.4 mm beyond reach at the
landing turn taken, where the newest view alone planned within 0.5 mm, and so did 0 degrees ranked again on the same
pose ([E8](#appendix-evidence)). So a turn the rule and the steady option do not see at the object's points is what
moved the place.

**O13. Beside a nearer object the depth camera misreads the points the colour tracker still sees.** Those 19 views
came while the wrist, 12 to 13 cm nearer the camera, stood beside the cube and hid its lower corner. Replayed through
Point2Pose on the act's own frames, the cube turned a median 8.4 degrees there (at most 11.4), against 2.2 to 2.5
before and after; with the same colour frames and the cube's depth from just before the wrist came, a median 3.2 (at
most 3.7). Under 4 of its 25 tracked points the depth read nothing, or the wrist, while the colour tracker counted all
25 seen; Point2Pose dropped those 4 from its fit, and the few millimetres of depth error left at others turned a 36 mm
cube 5 to 10 degrees by themselves ([E9](#appendix-evidence)). So whether a point is seen needs its depth as well as
its colour.

## Constraints and freedoms

**C1.** The test of a view needs the noise on its fit points, measured: the prediction in [O5](#observations) is per
millimetre of that noise.

**C2.** Every pose is judged against the reach tolerance, 3 mm and 3 degrees ([R2](#requirements)).

**C3.** The estimators are free. The loop needs, per object, views registered against the demo's with the fit points
behind them ([O5](#observations)), and between them how the object moved since each view's frame.

**C4.** How the arm steps toward a target is free within the jog's speed limits; a plan made once and played back is
needed only where timing matters ([R5](#requirements)).

## Architecture

1. **Legs.** The demo is cut at its marks, and each leg follows one object: the pre-grasps and the grasp end follow
   the object picked, the pre-places and the place end the object placed onto. `GRASP_KINDS` and `PLACE_KINDS` in
   `_pregrasp_core.py` already name the split. ([R3](#requirements), [O1](#observations))
2. **The pose rule, one function for both objects.** An object's pose is its last pose moved with its
   [support](#glossary): with what it rests on before the grasp (item 5); with the fingertip, from the arm's own
   kinematics, while held. A view
   replaces the pose when it shows enough of the object to pin it: the share of its tracked points the object placed
   onto already needs ([O2](#observations)), against points drifting onto what covers it; and its predicted error at
   the object's middle, times the measured noise, within the reach tolerance ([O5](#observations)), against too few or
   too bunched points to fix a turn. The share counts the tracker's tracks, which it seeds on what it sees, so it cannot
   alone say whether the points seen pin the pose; the prediction cannot see points that have drifted. Both halves,
   for both objects. With the steady option (off by default, a box on the Approach tab), a view that puts the
   object's points within the reach tolerance of where the pose holds them is averaged in with the others since the
   object last moved, and only a farther one replaces the pose ([O11](#observations)). With the depth check (on by
   default, 20 mm, a box and a tolerance on the Approach tab), a tracked point counts toward the share only while the
   depth under it puts it within the tolerance of where it was when the last view placed the object, moved since with
   its support ([O13](#observations)). ([R1](#requirements), [R2](#requirements), [R4](#requirements), [O3](#observations),
   [C1](#constraints-and-freedoms), [C2](#constraints-and-freedoms))
3. **The loop.** Every tick: the target is the leg's next sample carried by the pose of the leg's object; the arm steps
   toward it within the jog's limits; the gripper command follows the demo's along the leg's progress. A leg ends when
   the arm reaches its last sample within the reach tolerance. ([R1](#requirements), [R3](#requirements),
   [R5](#requirements), [C4](#constraints-and-freedoms))
4. **What the loop replaces.** The wait for the object to hold still and its rule of taking the last view once the
   object is hidden (`_still_decision` at `c2f8bedd4`); the grasp, carry and place played back from plans made once (`stream`); the
   separate correction for how the object sits in the gripper, since a held object's pose moves with the fingertip.
   ([R3](#requirements), [O4](#observations))
5. **Point2Pose's views place an object; the point groups carry it between them.** The pose is `G(t) · G(t_v)⁻¹ · V`:
   `V` the last view that placed the object, registered against the demo's view, `t_v` the time its frame was read,
   and `G` the point groups' pose of the object at a frame read at `t`, interpolated between their frames
   (`carried_motion` in `_pregrasp_core.py`). The point groups place an object by its own points only under the rule
   of item 2, and move it to another group only on such a view or with a body that splits off with it hidden among its
   points, by the vote of the points around it ([O8](#observations)). With the point groups off, `G(t) · G(t_v)⁻¹` is
   the identity and the view is held. An act designates the objects it follows in the point groups before the arm
   moves, and stops when they draw no frame for 5 s, since the objects would stand still in them whatever happened.
   An object at rest stays with its group until its own points put it beyond the place tolerance from where the group
   carries it ([O9](#observations)). From the grip the act measures until the place lets go, the held object's pose
   in the point groups is the arm's: the fingertip by the joints, times how it sat in the fingers when they closed; let
   go, it rests in the world where it was put ([O10](#observations)). The point groups run in the act's own flow: an
   object the act's tracker finds is designated in them at once, and the act's camera view paints them, with each
   object where the act would aim. ([R4](#requirements), [C3](#constraints-and-freedoms), [O6](#observations))

## Alternatives, and what this costs

**Do nothing, and keep the point groups view off during acts.** The one act since stacked. But on 2026-10-08 an act
stacked while following views up to 9.8 mm and 11.4 degrees off ([E4](#appendix-evidence)): whether the act survives a
covered object depends on the tracker calling it occluded in time.

**Patch the wait.** Keep the arm still there, and plan from the last view that agreed with the one before. That fixes
the 20:35 failure and leaves the rule: the walks still follow any view the tracker accepts, and the stages still
differ ([R3](#requirements)).

**Use the share of points seen alone, for both objects.** It is in the code already and costs nothing to compute.
But it counts the tracker's tracks, whose number grows as it seeds new ones on what it sees (25 to 124 in the 20:35
act), so it measures the tracker's bookkeeping rather than whether the points seen pin the pose: two partly covered
views that were within 2 mm saw 78% and 82% ([E1](#appendix-evidence)). It is kept as one half of the rule.

**The point groups alone.** They place an object by its own points under the same rule, but against where it was
designated, not against the demo's view of it, which only a find registers. They carry; Point2Pose's views place.

**A prior that objects stay flat on the table.** Replayed against ground truth on the nine YCBInEOAT videos it scored
below or equal to the raw fit on every one (`_compose_motion`), and it assumes the scene.

What the design costs: the point groups run beside the act's tracker and slow it ([O6](#observations)); an act
starts them when they are not running, and waits before the arm moves while they load and take the objects. The grasp
and the place stop being exact replays of the demo's timing, so the close is timed
by progress along the leg ([R5](#requirements)), and a slow arm closes later than the demo did. The act's core,
`_act_task`, is replaced rather than edited. A held object's pose comes from the fingertip and the grasp pose, so a
slip in the fingers goes unseen until a view places the object again.

## Open questions

**Q1. Where are the demo's object poses read: at each object's click frame, or at the operator's pose marks?** At the
click frame the pose is exact by construction ([O7](#observations)); a pose mark reads the track wherever it is
placed. Leaning: the click frame, with a pose mark only for an object that moved before its leg begins.

**Q2. What should a view's turn count for, in the rule and in the steady option?** Both judge a view at the object's
points, where a small object's turn hardly shows, while the place uses its pose 45 mm above it ([O12](#observations)).
The turns of the 11:15 act came from the depth beside the wrist ([O13](#observations)), which the depth check now
catches; whether a turn by itself should still count is open. A limit on the turn alone does not separate them: on
that act normal views already differed from the steady pose by up to 5.4 degrees and the turned ones by as little as
3.9.

To measure: the noise on the fit points ([C1](#constraints-and-freedoms)), from repeated views of an object standing
still; the act's stacking rate over trials with the point groups on and off.

## Glossary

- **Mark:** one of the demo's keypoints, as the demo editor calls them: pre-grasp, grasp end, pre-place, place end,
  pose.
- **Leg:** the part of the demo between two consecutive marks.
- **Support:** what moves an object while no view places it: before the grasp, what it rests on, as the point groups
  have it (with them off, nothing: it stays where it was last placed); the fingertip while the gripper holds it.
- **Places the object:** said of a view whose predicted error at the object's middle, times the noise on its points,
  is within the reach tolerance.

## Appendix: evidence

Taken 2026-10-09 on `proto/show-and-servo` at `c2f8bedd4`, from the acts' own recordings; the captures, the
pictures and the method are in [`docs/proofs/act-loop/EVIDENCE.md`](../../../../docs/proofs/act-loop/EVIDENCE.md).

- **E1.** Views of the gamepad while untouched, with what each saw and how far off it was (its section 1).
- **E2.** The 20:35 wait: the fingertip's travel and the targets' spread; the gamepad before and after (section 2).
- **E3.** Every untouched-gamepad view of five acts sorted by predicted error (section 3).
- **E4.** The rule replayed over the five acts against what each act did (section 4).
- **E5.** The act's tracker with and without the point groups view (section 5).
- **E6.** The demo's cube track at its click frame and at its pose mark (section 6).
- **E7.** The cube's pose at each view that re-pinned it, in two acts where nothing moved it (section 7, taken
  2026-10-10).
- **E8.** The 11:15 act's carry planning replayed on each view of the cube, on the steady pose, and over every landing
  turn (section 8, taken 2026-10-10).
- **E9.** The 11:15 act's frames replayed through Point2Pose, with and without the cube's depth from before the wrist
  came; the depth check and a check against the point groups scored on 18 replayed acts (section 9, taken
  2026-10-10).
