# Offline tracking check, 2026-10-16

The two successful insertion acts do **not** establish that the target tracker
always followed the ring. The live group pose lagged a clearly visible push in
`20261016_022104`, then missed a second, partly occluded ring move for several
seconds in `20261016_075037`. A cheap appearance check can detect these
contradictions without another neural-network inference. It has not yet been
connected to live control, so no live frame-rate or insertion improvement is
claimed.

![Visible ring versus the live group pose](offline_ring_disagreement_20261016.png)

The green mark is a local RGB appearance match to the ring's initial SAM mask;
the magenta mark is the **recorded live** group pose projected into the same
camera frame. These are images from the acts, not a new physical test.

| Act and camera frame         | Appearance support |        Live-group discrepancy | Observation                                                                                                                                          |
| ---------------------------- | -----------------: | ----------------------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `022104`, frame 202, +7.33 s |               0.96 | 25.6 px, about 17 mm at 0.4 m | The ring is almost fully visible while moving; the group is still at the old position. It catches up around frame 204.                               |
| `075037`, frame 350, +14.2 s |               0.43 | 20.6 px, about 14 mm at 0.4 m | The gripper covers part of the ring; the visible green rim and aligned depth put it above the group estimate. The disagreement persists for seconds. |

The appearance match measures image translation, not full 6D pose. The second
measurement is weaker because the ring is partly hidden. Its local color
template, learned from the SAM mask, matches 43% of the original colored
pixels, or roughly 1,600 pixels, at frame 350. The measured depth of those
pixels is about 407 mm. The visible outer rim also ends near the expected
radius from the proposed center. Artificially hiding a contiguous strip of
40–60% of the ring in clear frames gave a 95th-percentile center error of
about 2.2 px over 4,000 samples across both acts; real occlusion and changed
viewpoint may be harder. The late 20.6 px discrepancy is therefore compelling
evidence of a missed move, but not a calibrated ground-truth error bound.

Before occlusion in `075037`, the live group and appearance check agree within
1.0 px median and 2.0 px at the 95th percentile on frames with at least 90%
appearance support. The ring's original color is distinctive: 83–85% of its
SAM mask passes the learned color filter, compared with 1.5–2.1% of the
surrounding pixels. That is why this cue is useful here. It would not be a
universal tracker for a grey object on a grey support, or for a held object
whose viewpoint and visible shape change substantially.

The current point-group rule requires 97% of an object's modelled points to
be seen before its own tracks can replace the group-carried pose. This is why
the target stayed attached to the tray after the hand partly covered it. An
offline replay with 50% instead of 97% followed this ring better and cost no
extra tracker time. But a synthetic gripper-occlusion case already in
`tests/showservo/test_groups.py` then falsely carried a stationary gamepad
176 mm with the wrist; the 97% setting kept it within 0.3 mm. Lowering the
global gate is therefore not a safe fix.

A second performance experiment reduced TAPIR's PIPS refinement from two
iterations to one. Median GPU tracking work fell from about 12.4 to 8.5 ms
per sampled group frame. The ring poses then diverged by roughly 3–4 mm for
long stretches after motion, and by up to 15 mm during the earlier group
split. With no independent reference under occlusion, that speed change has
not earned a default switch.

`benchmarks/group_appearance.py` reproduces the RGB/depth comparison directly
from each act's fixed 30 Hz camera recording and its live group poses. The
local match itself takes 0.11 ms median and 0.12–0.14 ms at the 95th
percentile per camera frame on this host, excluding shared RGB decode and the
one-time SAM mask. The candidate can therefore be scheduled beside the
existing group tracker with little expected computation cost, but a real
concurrent A/B run is still needed to establish unchanged end-to-end FPS.

For example:

```bash
PYTHONPATH=src /home/feit/.cache/point2pose/venv/bin/python benchmarks/group_appearance.py \
  --act ~/.cache/huggingface/lerobot/demos/insertion/acts/20261016_022104 \
  --click 368,340 --object ring --duration 20 --out /tmp/ring-022104.json
```

The controlled next change is a _separate, conditional measurement_: retain
the 97% point-track gate, use the initial SAM mask to learn whether an object
has appearance distinct from its local background, and compare a small
search region only while point/group evidence is missing or contradictory.
A highly supported match can correct translation with aligned depth. A
partly supported match should first require temporal consistency and depth
agreement; it must not silently override the group's orientation or claim a
full pose. Objects without a distinctive cue continue to use the existing
point/group/P2P evidence. Validate this against other object shapes, colors,
occlusions, and distractors before connecting it to control. The held dowel's
pose remains unresolved by this ring translation experiment.

The earlier claim that the `075037` success showed the controller tracking the
final target movement was too strong. It followed the tray's earlier motion,
but missed the later independent ring movement. The physical insertion
succeeded, yet this act cannot prove that its final alignment was due to an
accurate updated ring pose.
