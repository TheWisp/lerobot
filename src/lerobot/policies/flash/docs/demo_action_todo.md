# FLASH demo action: open work

Open work beyond the group update and display changes. The running design is in the [architecture diagram](../../../gui/static/flash_architecture.html); recorded group regressions live in [the tests](../../../../../tests/policies/flash/test_live_static_ring_groups.py).

- [ ] **Medium:** bound group memory without silently losing useful hidden identities. The 4,096-point and 32-group capacities can still erase established state without measured separation.
- [ ] Investigate the two deferred pose changes independently: which current observations contribute Kalman information, and when a group hint can skip object fitting or reset the pose filter. Neither is required for the group-only regression.
- [ ] Prevent stale scene-group predictions from steering a reappearing held object. Verify measured object evidence and the gripper-to-object relation during grasp, release, and slip.
- [ ] Verify one physical placement end to end: stable ring reference, demo-to-live object identity, commanded versus measured tool path, capture-to-command latency, contact, IK limits, and actual object-on-ring outcome. Queue completion alone is insufficient.
- [ ] Stamp every saved tracking result with a monotonically increasing integer pipeline version. Increment it when group or pose behavior changes; detect version mismatches and rebuild stale demo, rollout, live-history, and example results from retained inputs before presenting them as current.
- [ ] Validate group membership and carried positions across more recordings with independent physical evidence, including new-point joining and genuine object separation. Benchmark full replay/live latency and memory on the target hardware; the recorded group-core benchmark does not cover camera capture or object pose fitting.
