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

Two sources of drift on continuously-moving targets, characterized by
`benchmarks/cartesian_ik_tracking.py` (this PR) and
`benchmarks/pink_ik_tracking.py` (PR #9). For VR teleop this is **not a
UX blocker** — drift is small in absolute terms during typical hand
motion, and goes to 0 during pauses — but worth knowing for higher-speed
use cases (trajectory replay, autonomous policy execution).

**Source 1 — iterative IK lag (intrinsic to PinkKinematics).** Per call
the QP returns a bounded joint velocity, so lag scales with target
velocity. Position drift visible in PR #9's bare-IK numbers: 0.2 mm at
1 cm/s to 3 mm at 17 cm/s on a smooth circle. Lever for tightening: the
`FrameTask` vs `PostureTask` cost ratio inside `PinkKinematics`, or a
velocity feed-forward layer. Tighten only if hardware reveals a need —
aggressive tuning can re-introduce the null-space chatter the
`PostureTask` was added to fix.

**Source 2 — gripper-DOF cost (introduced by this PR's stack).** Bare-IK
has 7 DOFs and rotation drift is ≈0° everywhere. This PR's
`CartesianIKController` pins the gripper joint (S7) to the teleop's
`gripper_pos` after each IK solve — required so the user's open/close
actuates the gripper — which removes one DOF from the IK. The IK now has
exactly 6 DOFs for a 6-DOF SE(3) task, no redundancy, so the small
S7 contribution to wrist orientation it used to lean on is gone. Shows
up as 1–8° rotation drift on circle/square (2D paths with held orientation,
5 constraints, 1 redundant DOF lost), but only 0.3–0.6° on a line (1D
path, 4 constraints, still has slack).

Two paired sweeps in `benchmarks/cartesian_ik_tracking.py` make the
contribution easy to read:

**SO-101 stack** (5-DOF, position-only, identity alignment, single-arm
controller). The numbers come out _bit-identical_ to PR #9's bare-IK
sweep — same URDF, same shapes, same speeds. The controller wrapper
adds no overhead here: no S7 to pin (gripper joint is on a branch off
the EE chain), no per-arm alignment math, no bimanual transform. So
SO-101 sets the bare-IK floor for the stack, free of the L7-frame
caveat that complicates SO-107's read.

**SO-107 bimanual stack** (6-DOF, full SE(3), per-arm calibration). At
the same shapes / speeds the position drift is in the same band as
SO-101 (~mm at typical teleop), but rotation drift shows the gripper-
DOF cost cleanly:

|        | speed | SO-101 pos | SO-107 pos | SO-107 rot |
| ------ | ----- | ---------- | ---------- | ---------- |
| circle | ~4.4  | 0.46 mm    | 1.22 mm    | 4.95°      |
| square | ~9.4  | 1.65 mm    | 2.57 mm    | 8.04°      |
| line   | ~8.8  | 4.61 mm    | 5.66 mm    | 0.62°      |

The clean fix is to make the IK target frame independent of S7 (use a
frame upstream of the gripper joint, e.g. `L6_1`, plus a fixed SE(3)
offset to the logical EE tip). Tracked in the next section.

## EE-point / tip-frame calibration

The IK currently targets `L7_1` — the SO-107 URDF's CAD-exported tip
frame. Two problems with that:

1. It sits on the gripper body, _downstream_ of S7. That's the root cause
   of the gripper-DOF rotation drift in the section above — S7 affects
   the EE pose, so pinning S7 for teleop discards the IK's S7 choice.
2. It's not actually at the physical gripping point. PR #9's "known
   issues" already flags this: there is a fixed offset to where the
   _closed_-gripper tip would be. For a parallel gripper that's the
   midpoint between the two finger faces; for the SO-107 (soft fin-ray
   fingers) it's where the two soft fingertips meet when squeezed. The
   exact offset is per-physical-arm calibration, not a property of the
   CAD URDF.

The right model: the IK target should be the **hypothetical closed-
gripper tip**, defined as a fixed SE(3) offset from an upstream-of-S7
anchor frame. Sketch:

- `PinkKinematics` gains a `tip_offset` SE(3) parameter (default
  identity = current behaviour). FK returns `FK(anchor) @ tip_offset`;
  the IK target is pre-multiplied by `tip_offset.inv()` so the QP still
  optimizes against the URDF anchor frame.
- `URDF_TIP_FRAME` switches from `L7_1` to an upstream frame (`L6` or
  the wrist link, before the gripper joint).
- Per-arm `TIP_OFFSET` lives in `joint_alignment.py` alongside
  `JointAlignment` — both are "this physical arm's calibration data".
- The guided-calibration GUI tool (`gui/TODO.md`) is the natural place
  for the user to measure / set this; until that lands, profile JSON
  override + a measured default.

Solves multiple things at once:

- The rotation drift from the gripper-DOF cost goes away structurally
  (S7 no longer affects the EE pose → the overwrite is genuinely free).
- PR #9's "absolute EE position needs further calibration" issue is
  resolved by the same mechanism.
- Matches the intuitive user model — teleop targets the closed-gripper
  tip, which is what you reach toward to pick something.

Lives across two PRs to land cleanly: the `tip_offset` parameter belongs
in `PinkKinematics` (PR #9's territory); the SO-107 anchor + per-arm
defaults belong here. Worth folding both at the time we do the URDF /
EE-point redesign.

## Configurable IK + on-hardware benchmark + usage analytics

Once the EE anchor frame and tip offset are per-arm calibration (above),
the user needs tools to set them and verify they're right — and we need
data on how the configured stack behaves on the real robot over time.
Three related pieces:

### Configurable IK (the calibration surface)

- `URDF_TIP_FRAME` (anchor) and `TIP_OFFSET` (closed-gripper-tip offset)
  become per-arm fields on the robot profile JSON — same shape as the
  per-arm `JointAlignment` already is. Hardcoded defaults stay in
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
