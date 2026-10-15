# Act timing and goal jitter, 2026-10-15

The successful `insertion` Act `20261015_165734` was run with `speed=2`. Its
25-second record separates three rates that the GUI can make look like one:

| Source                    |                                                                      Measured in this Act |
| ------------------------- | ----------------------------------------------------------------------------------------: |
| Camera timeline           |                  751 slots at 30 Hz; 637 distinct frames, 107 held frames, 7 missed slots |
| P2P results               | 51, about 2.0 per second; result intervals 0.291 s median, 0.866 s at the 90th percentile |
| Motor observations        |                                                                704, about 28.1 per second |
| Distinct streamed targets |       456, about 18.4 per second; 0.034 s median interval, 0.072 s at the 90th percentile |

P2P's reported model time was 0.096 s median. The time from a queued frame to
an applied result was 0.288 s median and 0.864 s at the 90th percentile. Pose
verification accounted for about 0.041 s median. The pump feeds
only one frame to the worker at a time, waiting for the complete result and
render before requesting the next. The new camera timeline did not change the
motor control algorithm. A pre-timeline Act (`20261015_141339`, different
speed and outcome) already had 0.255 s median result intervals and 0.033 s
median target intervals. These runs are not a controlled recorder A/B test.

The motor loop repeats its last goal when the next checked IK step has not
arrived; 36.8% of observed motor ticks in this Act had no new arm-joint goal.
There were also plan-boundary waits, a 0.33-second mid-stream gap with only two
motor samples, and a 0.55-second gap while a joint lagged. Thus a 30 Hz motor
read and a 30 Hz camera recording do not imply a 30 Hz changing command.

The target ring remained nearly still: its 304 group poses ranged by only
0.68, 0.48 and 0.47 mm in camera x, y and z. The corresponding overhead depth
samples were 421–422 mm. Nevertheless, consecutive goals within the same
ring-relative action phase changed by 4.1 mm / 4.0 degrees at +10.44 s and
9.2 mm / 4.5 degrees at +12.47 s. The pose arbitration compared group motion
with raw P2P; a depth-constrained pose could then introduce a new tilt without
that same comparison. A narrow planar patch (about 11 mm span, about 0.7 mm
residual in this Act) makes such corrections sensitive to view changes. The
record establishes unwanted goal changes while the ring stayed still; it does
not alone apportion every degree between the raw fit and plane correction.

Control now also compares the _corrected_ pose's motion with synchronized
group motion, against the first and the last accepted corrected views. A view
that contradicts either holds the last accepted pose. The same recorded
sequence, passed through this new motion check offline, would reject ten
corrected ring candidates, seven of which the original Act accepted, including
the 4-degree change at +10.44 s. This is an
offline counterfactual, not a hardware demonstration that all jitter is gone.
The next Act record stamps worker-result arrival, NPZ decode, verification
and rendering separately. Live tracking status also reports worker frame fetch,
compute, depth-geometry stages and these server stages. The GUI was restarted
with these changes after folding and disconnecting the arm. The `m1_left`
profile lacked a rest pose; its same-port `white` left-arm profile already had
one, so that recorded pose was copied to `m1_left` before parking.

A passive measurement after the successful placement, with the arm disconnected,
reproduced a 1.3–1.9 Hz pose-update rate. The tracker then designated about
211,000 pixels—much larger than the 1,400–4,500-pixel masks in the successful
Act. Its worker step took about 636 ms median, split into 72 ms for P2P, 4 ms
for the table, 53 ms for the footprint, **430 ms for the visible-face fit**,
and 70 ms for footprint rotation. Frame transfer took about 4 ms, server
verification 48 ms and rendering under 15 ms. The face fit evaluates 200 plane
hypotheses against this large designation; the cost rises with mask size.
This identifies the current slow passive case but does **not** prove the same
cost dominated the successful Act, whose saved masks were small. In that Act,
pose results were roughly 0.2–0.3 s apart in key frames 8–9 but roughly
0.8–0.9 s apart in key frames 5 and 7, despite similarly small masks and
similar reported P2P times. The next physical Act's stage stamps are needed to
attribute those phase-dependent pauses.
Re-evaluating twelve of the Act's saved depth/mask pairs found face-fit times
of about 5–18 ms in both fast and slow phases. The large-mask face-fit cost
therefore does not explain those pauses.

For masks larger than 8,192 depth points, the worker now ranks its 200 plane
hypotheses on a reproducible 8,192-point sample and verifies the best 16 on
the entire cloud before the existing full-cloud refits. Normal-sized masks
follow the original calculation exactly. On the saved large-mask frame, the
face fit fell from 642–665 ms to 57–98 ms across repeated measurements, with
the same face normal and inlier count. Six normal-sized views from the Act
produced identical face metadata. This bounds the cost of a broad or incorrect
designation; it does not certify that such a designation is correct.
After restarting the worker, a second passive sample on the same broad
designation rose from about 1.3–1.4 to about 2.8–2.9 pose updates per second.
The arm was disconnected for both samples. The remaining median worker step
was about 284 ms: P2P about 72 ms, table 4 ms, footprint 54 ms, face 59 ms,
and footprint turn 92 ms. Server pose verification added about 50 ms. These
numbers describe this post-Act scene, not the earlier moving Act.
