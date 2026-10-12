# Average the held object's position at its centre

A measured hold `H = inverse(tip) @ object_motion` maps coordinates from the
demo's reference view, expressed in the base frame, into the fingertip frame.
Its translation column describes where the reference **coordinate origin**
went. It does not describe the object's centre.

`average_hold` rejects inconsistent views by evaluating each transform at the
reference object centre `c`. It must also average position at that point. With
the retained views, choose the mean rotation `R` and set the translation to
`mean(H_i @ c) - R @ c`. The resulting rigid transform maps the reference
centre to the mean measured centre. Changing the source coordinate origin
does not change the physical result.

Previously, the rotation and translation columns were averaged separately.
Rotational noise about a common centre then introduced a position bias: the
mean rotation matrix differs from the elementwise mean of the input rotation
matrices. A regression with four views turned ±4° about a common centre
produced a 1.032 mm shift despite zero measured positional spread. The corrected
result preserves the centre to floating-point precision and passes a change
of coordinate origin check. That position fix did not change outlier selection,
rotation averaging or the reported spreads.

For the `pick_place` demo used by act `20261012_130212`, seven still carry views
yield six retained measurements. The previous average put the centre
`[0.115095, 0.006595, 0.000350]` mm away from their mean in the fingertip frame,
a 0.115284 mm bias. This measures the demo component of the hold correction;
it does not establish the whole live correction's change or explain that act's
failed stack.

An independent consistency check of the same demo's tracks found its settled
gamepad surface centre about 6.5 mm horizontally from the cube's tracked
surface centre. The cube was pinned at its marked pre-occlusion frame, since
its later track drifted while hidden. The planner's rigid carry-hold prediction
at release was 2.71 mm from that settled tracked position. These are tracker
measurements, not independent geometric truth. The intended landing remains
the relationship taught by the demo; these observations do not justify
replacing it with an exactly centred stack.

The later act `20261012_203608` exposed a separate aggregation failure. Large
rotation outliers pulled the initial mean away from a tight majority, so the
five-degree gate rejected good views too. In recorded checkpoint frames 93–99,
six poses cluster with median scatter of 0.352 mm and 0.207 degrees; the old
averaging returned no hold. This window demonstrates the aggregation defect,
not the exact unlogged seven-frame window the live hold check consumed or the
accuracy of the surviving pose.

Rotation filtering now considers each observation and the all-view mean as
candidate centres. It selects the largest consensus under the existing
five-millimetre position and five-degree orientation gates, averages that set,
and removes any remaining views outside the angular gate about its new mean.
The all-view candidate preserves symmetric noise cases such as the four ±4°
views above. A strict majority and the existing minimum of three are required:
a small consistent cluster amid mostly contradictory evidence, or two equally
supported alternatives, cannot authorize placement. Thus the change also
tightens acceptance when only a minority agrees.

The synthetic regression includes five nearly identical orientations plus
30° and 60° outliers, in two input orders. A compact recorded fixture checks
the checkpoint window without hardware or GPU models. Process-local mutations
restoring the initial all-view mean fail all three cases; removing the majority
requirement fails both ambiguity cases. Consistency alone cannot rule out a
systematic tracking error and does not establish ground-truth pose accuracy.
