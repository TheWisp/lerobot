# Can a demo dowel endpoint transfer to a recorded live run?

**Partly, but not reliably enough to use as an insertion landmark yet.** An
explicit TrackEverything query on the demo's visible right endpoint lands near
a visible endpoint in two of three replay starts, but misses the entire dowel
in the third. Tracking a successful live seed through a full rollout works on
some visible frames and loses support during grasp. Because the dowel is
unmarked and approximately symmetric, these observations cannot prove that
the same _physical end_ was identified after each reset or occlusion.

## Test

I marked pixel `(218, 300)` on the visible right end of the dowel in demo
episode 0, frame 100. Its RGB-D depth is about 417 mm. All RGB and depth
images come from the calibrated LeRobot recordings; live masks are matched by
camera-frame ID. These are **recorded** trials, with no robot act in this test.

For each replay start, I sent the demo and live RGB-D frames to the installed
TrackEverything checkpoint, querying that exact demo pixel. Green circles in
[the image](landmark-transfer.jpg) are the two approximate endpoint locations
derived from the live SAM mask. Red is the **direct** landmark prediction. Blue
is a separate rigid-transform prediction fitted from other surface queries;
the last fifth of the demo dowel, including the marked end, was held out of
that fit. The endpoint comparison is a 2D diagnostic, not ground-truth 3D pose
or proof of material-point identity.

| Replay start   | Direct query to nearest mask endpoint | On live dowel mask? | Live depth error at query | Held-out rigid fit to nearest endpoint |
| -------------- | ------------------------------------: | :-----------------: | ------------------------: | -------------------------------------: |
| 19:11, frame 4 |                                6.5 px |         yes         |                    1.8 mm |                                14.5 px |
| 20:33, frame 0 |                               10.6 px |         yes         |                    5.4 mm |                                25.1 px |
| 19:51, frame 5 |                               88.7 px |       **no**        |                   23.4 mm |                               112.3 px |

The 19:51 failure is a different initial dowel angle. Its rigid fit still has
22 inliers and a 3.0 mm median residual, but those surviving matches cover
only about a quarter of the dowel's axis. They agree locally while giving a
bad extrapolation to the opposite end. The direct query's visibility logit is
positive despite its prediction being off the object. Mask and measured depth
are therefore essential checks; a high point count or visibility alone is
not sufficient.

Next I followed the 20:33 two-frame prediction through 500 adjacent _live_
frames, starting with its predicted live pixel. This is the sensible two-stage
use: cross-episode match once, then temporal tracking within the rollout.
Across 82 frames with an exactly aligned recorded dowel mask, the predicted
point is inside the mask in 59 and has mask plus depth agreement within the
diagnostic 10 mm tolerance in 50. The [six-frame view](landmark-two-stage-rollout.jpg)
shows the red prediction and green mask boundary. Before grasp, 12 of 13
masked frames have surface agreement. During frames 60–299 the prediction
often has no surface support, mainly while the object is gripped and
occluded; from frames 300–499, 38 of 41 masked frames have support again.
Reappearance on the surface does **not** certify that it returned to the same
material point on a featureless cylinder.

I also tried treating the demo frame and all 500 live frames as one hard-cut
video with a single source query. [That comparison](landmark-through-rollout.jpg)
shows the point staying near the old tray location instead of transferring at
the cut. It had no mask-plus-depth-supported frame among the same 82 masked
frames. Inference over two frames and over the full clip gives different
startup answers; the two-stage test avoids relying on that behavior.

The repeatable offline probes are
[`flash_cross_episode_landmark.py`](../../../benchmarks/flash_cross_episode_landmark.py)
and [`flash_cross_episode_landmark_follow.py`](../../../benchmarks/flash_cross_episode_landmark_follow.py).
The 10 mm surface and rigid tolerances, and 3% mask-tail endpoint definition,
are named benchmark parameters, not FLASH runtime acceptance rules. The next
validation needs a temporary distinguishable endpoint mark or another
independent landmark measurement across episodes. That mark is for _measuring_
identity; FLASH need not require one. More demo viewpoints and a
mask/depth-gated, spatially distributed rigid fit are promising, but the
current evidence does not justify committing a transferred tip as truth.
