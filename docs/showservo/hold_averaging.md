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
of coordinate origin check. Outlier selection, rotation averaging and the
reported spreads remain unchanged.

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
