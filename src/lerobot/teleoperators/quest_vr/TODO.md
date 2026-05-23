# Quest VR teleop — TODO

Follow-ups identified during PR review. Not blockers for landing — captured
here so they don't get lost.

## Safety: make limits explicit and configurable

The Cartesian-teleop safety limits are currently a mix of hardcoded module
constants and constructor defaults. They work, but a buried limit is a
_hidden limitation_ — if the arm later feels throttled or unresponsive,
nobody will know to look here. Safety belongs in one explicit, documented,
user-configurable place (a dataclass surfaced on the robot/teleop config,
so it shows up in the GUI):

- `robots/so107_description/cartesian_ik.py`
  - `_MAX_JOINT_STEP_DEG` — per-tick joint-jump cap (IK glitch backstop). 20°/tick.
  - `_MAX_EE_STEP_M` — per-tick EE-position cap. 0.10 m.
  - `SO107_WORKSPACE_MIN/MAX` — reachable-box clip; hand-tuned estimates.
    (Replace with _measured_ bounds once the guided-calibration tool lands —
    see `gui/TODO.md`.)
- `teleoperators/quest_vr/configuration_quest_vr.py`
  - `max_pos_step_m_per_tick`, `max_rot_step_rad_per_tick` — already config
    fields; fold them into the same safety section so every limit lives together.

Each limit should: be a named, documented config field; log when it engages
(the joint-jump cap already does — a throttled arm is visible in the run
log, not silent); and be tunable without editing source.

## Extract the generic IK classes when a second arm needs them

`CartesianIKController`, `JointMappedKinematics`, and `make_bimanual_ik_transform`
in `robots/so107_description/cartesian_ik.py` are arm-agnostic — they sit
beside their one consumer's factory only because there is one consumer today.
When a second arm wants Cartesian VR teleop (e.g. a unimanual SO-101), move
those three to a generic home (`lerobot/model/cartesian_ik.py`) and keep
only the SO-107 factory + workspace constants in
`so107_description/cartesian_ik.py`. Don't extract preemptively.

## Production-grade WebXR page

`webxr_teleop.html` + `server.py` are solid for fork dev / research on a
trusted LAN, but not for shipping:

- **Authentication.** Today anyone on the LAN with the URL can drive the
  robot once they accept the cert. At minimum a one-shot token in the URL;
  ideally a short-lived shared secret in localStorage.
- **Proper certificate path.** The self-signed cert means every Quest sees
  a browser warning once per device. Document a Let's Encrypt / mkcert flow.
- **In-headset status UI.** The 2D log `<div>` is invisible during the
  immersive session. Show "controller not tracked", disconnect / RTT, and
  "session about to suspend (headset off-face)" inside the XR layer.
- **Haptic feedback.** Use `gamepad.hapticActuators` to confirm clutch
  engage/release and to indicate workspace-clip / IK-hold events.

## Tighter tracking on moving targets

What's left after the structural fix below is one source of drift,
intrinsic to PinkKinematics: per call the QP returns a bounded joint
velocity, so lag scales with target velocity. Position drift visible in
PR #9's bare-IK numbers: 0.2 mm at 1 cm/s to 3 mm at 17 cm/s on a smooth
circle. Same shape in this PR's stack tracks within ~mm at typical
teleop speeds — fine for VR-hand motion (drift → 0 during pauses), worth
knowing for higher-speed use cases like trajectory replay or autonomous
policy execution.

Lever for tightening: the `FrameTask` vs `PostureTask` cost ratio inside
`PinkKinematics`, or a velocity feed-forward layer. Tighten only if
hardware reveals a need — aggressive tuning can re-introduce the
null-space chatter the `PostureTask` was added to fix.

Two paired sweeps in `benchmarks/cartesian_ik_tracking.py` make the
remaining floor easy to read:

**SO-101 stack** (5-DOF, position-only, identity alignment, single-arm
controller). The numbers come out _bit-identical_ to PR #9's bare-IK
sweep — same URDF, same shapes, same speeds. The controller wrapper
adds no overhead here: no S7 to pin (gripper joint is on a branch off
the EE chain), no per-arm alignment math, no bimanual transform.

**SO-107 bimanual stack** (6-DOF, full SE(3), per-arm calibration, with
the L6 anchor + `TIP_OFFSET` structural fix below). Position drift in
the same band as SO-101; rotation drift sits at the IK floor (≤ 0.05°)
because the IK target is now S7-independent and the gripper-pos
overwrite no longer fights the solver:

|        | speed | SO-101 pos | SO-107 pos | SO-107 rot |
| ------ | ----- | ---------- | ---------- | ---------- |
| circle | ~4.4  | 0.46 mm    | 1.63 mm    | 0.01°      |
| square | ~9.4  | 1.65 mm    | 2.96 mm    | 0.03°      |
| line   | ~8.8  | 4.61 mm    | 5.75 mm    | 0.02°      |

Historical record — same shapes / speeds before the structural fix,
when the IK targeted `L7_1` directly and the post-IK S7 overwrite
removed one DOF from a 6-DOF task: circle 4.95°, square 8.04°, line
0.62° at the same speeds. That drift is now gone.

## EE-point / tip-frame calibration

**Status:** the structural half landed in this PR. `make_so107_arm_kinematics`
now targets `URDF_ANCHOR_FRAME = "L6_1"` (upstream of `S7`) and applies
a fixed `TIP_OFFSET` SE(3) to the virtual EE — implemented as a thin
`TipOffsetKinematics` wrapper, so `PinkKinematics` stays generic. The
gripper-DOF rotation drift is gone (see the table above).

**What's still missing — the calibration half.** The `TIP_OFFSET`
currently shipped is the S7 joint origin straight from the URDF: it
defines "where `L7_1` sits when `S7=0`". That removes the structural
problem but doesn't yet move the virtual EE to the **physical gripping
point** — the place a user reaches toward to pick something. For the
SO-107 (soft fin-ray fingers) that's where the two soft fingertips meet
when squeezed; for a parallel gripper it would be the midpoint between
the two finger faces. Either way the exact offset is per-physical-arm
calibration, not a property of the CAD URDF — PR #9's "absolute EE
position needs further calibration" issue.

To finish the calibration story:

- Make `TIP_OFFSET` per-arm calibration data — same shape as `JointAlignment`
  already is. Hardcoded default stays in `joint_alignment.py`; the robot
  profile JSON overrides per physical robot.
- The guided-calibration GUI tool (`gui/TODO.md`) is the natural place
  for the user to measure / set it: jog the arm to a known fiducial,
  click "this is where the tip is now", solve for the offset.
- The next section (on-hardware benchmark) lets the user verify the
  calibration is good before relying on it.

Once that lands, the structural and calibration halves of the EE-point
model fit together: IK target = `L6_1 @ TIP_OFFSET` where the offset is
the user's measured closed-gripper tip. Matches the intuitive user
model, resolves PR #9's calibration caveat, keeps rotation tracking at
the IK floor.

## Configurable IK + on-hardware benchmark + usage analytics

Once the EE anchor frame and tip offset are per-arm calibration (above),
the user needs tools to set them and verify they're right — and we need
data on how the configured stack behaves on the real robot over time.
Three related pieces:

### Configurable IK (the calibration surface)

- `URDF_ANCHOR_FRAME` and `TIP_OFFSET` (closed-gripper-tip offset) become
  per-arm fields on the robot profile JSON — same shape as the per-arm
  `JointAlignment` already is. Hardcoded defaults stay in
  `so107_description/joint_alignment.py` as a sensible factory baseline;
  the profile overrides per physical robot.
- The guided-calibration GUI tool (`gui/TODO.md`) is the place to set
  them: jog the arm to a known target, the user clicks "this is where
  the tip is now", and the tool solves for `TIP_OFFSET` from the
  touched position. Same UX as the planned joint-alignment calibration.

### On-hardware benchmark (validate your calibration)

The plots in `docs/cartesian_ik_tradeoff_so101.png` and
`docs/cartesian_ik_tradeoff_so107.png` characterise the IK + stack
floor _in software_ — useful but blind to the hardware. The
runtime analog is a benchmark routine the user can run with the
configured robot:

- Command a scripted sequence of EE waypoints — a small grid or a few
  fiducials, plus the same shapes the software benchmark uses (circle,
  square, line at known speeds).
- Read the achieved motor positions, FK with the configured
  `TIP_OFFSET`, and compare to the commanded EE poses. The residual
  includes the IK floor _plus_ everything the math-only plot misses:
  motor following error, backlash, compliance under load, URDF vs
  measured geometry, calibration error. Surface in the GUI as a per-
  axis residual + a pass / re-calibrate signal.
- Differences between this and the software plot tell the user which
  side needs attention — IK tuning vs hardware calibration.

### Usage logging for analytics

Per-tick capture during normal teleop / record / replay sessions:
commanded EE (from the teleop), achieved EE (from FK on motor
observations), commanded joints, motor positions, the active
`TIP_OFFSET`, controller-tracking-lost events. Anonymised — no user
identifiers, no scene contents, just numbers about the system's own
behaviour. Aggregate into:

- Drift histograms over time — a wearing gripper or a shifted mount
  shows up as creeping residuals before the user notices in feel.
- Dropouts per session, common workspace regions, per-task success
  signals.
- Data-driven cases for tightening the IK QP gains specifically where
  it matters in practice, vs. blanket tuning.

Privacy: opt-in, local-first, anonymised by default; upload only on
explicit user action. Goes in the same place run logs already live
(`outputs/teleop/`); the analytics layer reads from there.

## VR mode: default to AR, and add camera-feed telepresence later

Immersive VR is intentionally pitch-dark today — we render nothing into the
world. **AR passthrough is the right mode for collocated teleop** and the
page already exposes "Enter AR".

- **Default to AR.** Right now the page offers AR and VR with equal weight
  and the README mentions both. Make AR the visually prominent button and
  the documented default; a user who picks VR by accident gets a pitch-dark
  headset and wonders why.
- **Camera-feed telepresence.** For _remote_ teleop, stream the robot's
  cameras into the headset and render them as quads in an immersive-VR
  session. Sketch: a WebRTC track per camera from the PC to the Quest
  browser; in `onXRFrame`, sample the `<video>` into a GL texture and render
  it on positioned quads. Camera intrinsics + the rig's eye-baseline let
  stereo cameras present as proper stereo depth; mono cameras render to
  both eyes. Real work — latency budget, codec, intrinsics calibration, rig
  geometry — but not exotic. Probably its own PR.
