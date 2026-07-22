# OpenArm2 integration — TODO

This is the central tracker for OpenArm2 work in LeRobot. Quest-specific
transport and browser items remain in
`src/lerobot/teleoperators/quest_vr/TODO.md`; items affecting the robot,
control loop, state semantics, safety, or datasets belong here.

## P0 — Define feedback-loss behavior while supporting the arm

The current Damiao path waits up to the upstream 10 ms batch deadline and then
continues with the last valid state for any missing motor. It logs the age of
each reused value. It must not automatically torque-off solely because one
state batch is incomplete: on a gravity-loaded arm, abrupt torque disable can
cause a drop.

Design and validate explicit degraded modes on hardware:

- hold the last command while preserving gravity support;
- ramp velocity and commanded motion to zero;
- distinguish a single late state, repeated feedback loss, bus-off, estimator
  failure, and an operator emergency stop;
- define when controlled hold, controlled descent, or torque disable is the
  appropriate response;
- verify every transition with the operator next to the robot before enabling
  it by default.

## P0 — Decouple CAN transmit, receive, and state publication

The official OpenArm CAN library sends commands independently, then refreshes
and drains available SocketCAN feedback. Our current LeRobot driver performs a
bounded request/collect operation inside the application loop. Move toward a
dedicated receive path without losing observability:

- continuously drain each CAN interface in its own receive worker;
- store a monotonic timestamp and generation/sequence for every motor update;
- expose snapshot age and cross-joint/cross-arm skew;
- define whether J8 POS_FORCE belongs to the arm-state completeness barrier;
- monitor SocketCAN errors, drops, overruns, queue depth, and bus state;
- use a bounded-age snapshot instead of silently describing cached data as a
  simultaneous hardware sample.

Do not replace the current path until hardware logs establish normal and tail
latency distributions for both CAN interfaces.

## P0 — Separate high-rate control from policy and recording rates

The long-term architecture may run a controller at 200–500 Hz while Quest,
cameras, policy inference, and dataset recording run at lower independent
rates. Specify the contract between them:

- a real-time or near-real-time controller owns CAN I/O and actuator safety;
- policy/teleop publishes timestamped targets at its natural rate;
- the controller interpolates, filters, or holds targets and enforces joint,
  velocity, acceleration, and stale-command limits;
- observations are timestamped snapshots of the controller state cache;
- dataset frames retain source timestamps and sample age rather than implying
  that camera, Quest, policy, and every motor were sampled simultaneously;
- controller deadline misses and policy stalls are logged separately.

A 500 Hz controller has a 2 ms cycle budget, so it cannot contain a 10 ms
blocking receive deadline or synchronous disk logging.

## P1 — Measure and choose the production state contract

Run representative 5-minute Quest sessions and report:

- per-motor first/last response latency and missing-response counts;
- reused-state age and cross-joint/cross-arm skew;
- controller/application-loop p50, p95, p99, and maximum latency;
- Quest pose arrival rate, sequence gaps, and pose age;
- IK solve/hold timing and final post-clamp actions;
- SocketCAN and USB adapter statistics before and after each run;
- logging-disabled versus logging-enabled A/B results.

Use these measurements to choose between a strict refresh generation, a
bounded-age asynchronous snapshot, or a hybrid policy. Keep timeout and age
thresholds configurable and derived from measured distributions.

## P1 — Dual-arm timing

The current bimanual observation path reads the two arms sequentially. Measure
the resulting cross-arm skew, then evaluate concurrent refresh triggers or
hardware timestamps. Similarly, preserve the validated near-simultaneous
dual-arm command launch when consolidating the remaining hardware checkpoint
work.

## P1 — J8 POS_FORCE feedback

Confirm the exact J8 response behavior and callback mode under sustained
POS_FORCE control. Determine whether missing J8 refresh responses are caused by
motor mode, frame identifiers, receive ordering, USB batching, or scheduling.
Do not tune its timeout or exclude it from state validity without evidence.

## Related trackers

- Quest/WebXR transport, tracking, and browser metrics:
  `src/lerobot/teleoperators/quest_vr/TODO.md`
- GUI and latency-monitoring follow-ups: `src/lerobot/gui/TODO.md`
