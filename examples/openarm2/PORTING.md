# OpenArm 2.0 port — what changed on `feat/openarm2-port` and why

This branch ports the validated OpenArm 2.0 work from the dora stack
(`dora-openarm` / `dora-openarm-vr`, branch `local/vr-standard-edition`) into
thewisp/lerobot, making **lerobot the single stack** for collect / train /
orchestrate ("option B", agreed 2026-07-19 in the dora workspace TODO).
The dora stack stays as the working fallback until on-hardware validation
passes.

It is a **port, not a merge**: the two stacks share no code layout, so each
validated dora feature was re-implemented inside lerobot's existing OpenArm
integration. Base: `sync/upstream-2026-07` (untouched).

## Port map (dora → lerobot)

| dora (validated) | lerobot (this branch) |
| --- | --- |
| `gravity_ff.py`: MuJoCo `qfrc_bias` → MIT `tff` slot, gain 0.9, 1 s fade-in, 20 Hz LPF, clamp 0.5×actuatorfrcrange, non-finite→0 | `src/lerobot/robots/openarm_follower/gravity_ff.py` — same math/safeguards, wired into `OpenArmFollower.send_action`, gated by `gravity_ff_gain` (default 0 = off) |
| (stock) MIT velocity/torque slots unreachable | New public `DamiaoMotorsBus.mit_control()` / `mit_control_batch()` (`src/lerobot/motors/damiao/damiao.py`) — `send_action` no longer calls privates |
| Velocity FF (tracking-goal lever 1, was planned not implemented) | Implemented: finite-difference of commanded positions → MIT `q̇*` slot, clamped to joint delta limits, gated by `velocity_ff_gain` |
| Align ramp 0.003 rad/step, **gripper (J8) excluded**; jump-guard logs naming the joint | Same in `send_action`: `align_step_limit` (None = off), gripper unclamped (1 N·m POS_FORCE finger), `align_jump_threshold` warnings, rate-limited |
| 30 s follower telemetry (err_max, τ_ext = qtorque − tff, MOS temp) | `src/lerobot/robots/openarm_follower/telemetry.py` — fed from the bus response cache, zero extra CAN traffic |
| `openarm_standard.yaml`: zero offsets, arms-down zero, v2 model joint limits, standard PD gains | Defaults in `OpenArmFollowerConfig` (per-side `joint_limits`, `position_kp/kd`); docstring states the zero convention |
| dora-openarm-vr: settle window after tracking recovery; jump-guard diagnostics | `teleoperators/quest_vr`: 0.25 s post-recovery settle (clutch treated released while pose settles), per-tick target-step warnings |
| dora-openarm-vr: yaw-only gravity-aligned anchor | **Not ported** — that fixed dora's live-head-pose anchoring; lerobot's WebXR stage frame is already fixed and gravity-aligned |
| UDP receiver + custom dataflow | **Replaced** by the fork's existing `quest_vr` WebXR teleop (browser on the Quest → WebSocket), extended with an OpenArm IK path (below) |

## Quest VR teleop: which solution

thewisp/lerobot's `quest_vr` (WebXR page on the Quest + aiohttp WSS server),
**not** the dora UDP receiver. It already had clutch (mouse-lift relative
control), glitch clamps, and tracking-loss re-anchor; only the settle window
and jump-guard logging were missing and were ported.

The teleop outputs Cartesian EE targets; driving OpenArm arms required new
pieces:

- `src/lerobot/robots/openarm_description/` — per-arm OpenArm 2.0 URDFs
  (kinematics-only, mechanically extracted from enactic/openarm_description
  v2.0, Apache-2.0 `LICENSE.txt` vendored; provenance in that README) plus
  `cartesian_ik.py` reusing the fork's pin-pink machinery
  (`BimanualOpenArmIKTransform`, `build_openarm_bimanual_ik_transform`).
- `BiOpenArmFollower.attach_teleop` — detects the Cartesian teleop and
  installs the IK transform (mirrors the SO-107 pattern), so
  `lerobot-record` / `lerobot-teleoperate` work unchanged.
- FK at q = 0 verified equal to the dora `_EE_ZERO` pose (arms hanging
  down), so **no joint-alignment shim** is needed (unlike SO-107).
- `QuestVRTeleopConfig.robot_forward_in_urdf` / `robot_up_in_urdf` make the
  quest→robot frame configurable (OpenArm: forward `[1,0,0]`, up `[0,0,1]`;
  defaults preserve SO-107 behavior). Gripper motor ranges are per-arm
  config (right −45°..0, left 0..+45°).

## Dependencies

New extra `openarm-ff = ["lerobot[openarms]", "mujoco>=3.2,<4",
"openarm-mujoco>=2.0"]` (pyproject.toml + regenerated `uv.lock`). The MuJoCo
model resolves via the `openarm-mujoco` package itself — no XML vendored for
gravity FF. IK needs `quest-vr` (aiohttp + pin-pink). All optional imports
are guarded (`lerobot/utils/import_utils.py`).

## Validation

- 348 tests passing (`tests/robots tests/teleoperators tests/motors` with
  `test`+`quest-vr`+`openarm-ff` extras), incl. gravity-torque vs MuJoCo
  (front-mid J1 ≈ +8.64 N·m, matching the physically validated band),
  MIT packet round-trips, ramp/gripper-exclusion, telemetry math, real
  pin-pink FK/IK round-trips, dropout-no-slew full-stack, and draccus-parsed
  `RecordConfig`s for both `README.md` record commands
  (`tests/robots/test_openarm2_record_wiring.py`).
- Hardware (2026-07-22): PCAN-USB Pro FD up on can0/can1 (1 M / 5 M FD),
  16/16 motors detected, read-only joint states sane.
- Not yet done: calibration + first torque-enabled connect, then the
  circle/square bench vs the dora atlas baseline (8–10 mm mean, ~100 ms
  delay) before switching data collection over.

## Control-regime rule (unchanged from the dora TODO)

One FF/ramp regime per dataset. Changing `gravity_ff_gain`,
`velocity_ff_gain`, or `align_step_limit` = new control regime → new
`dataset.repo_id`; never mix mid-dataset; teleop and inference must run the
same stack. Validated values: gravity FF 0.9; ramp 0.003 rad/step if used.

## Deliberately deferred

- dora's 500 Hz closed-loop MuJoCo plant sim test (`test_gravity_ff_sim.py`
  intent) — offline torque/filter coverage exists; add if wanted.
- τ_ext / friction observation processor (FORCE_TORQUE.md Stage 1/2) —
  follow-up; template is the fork's `feature/motor-current-observation`.
- The dora dead-man side-grip gate on the follower (no such input exists in
  lerobot; the quest_vr clutch/hold covers the same safety role).
