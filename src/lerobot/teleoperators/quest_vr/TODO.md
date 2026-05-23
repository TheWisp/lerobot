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

Side-by-side at the same speeds (max drift, left arm, 30 Hz loop):

|        | speed | bare-IK pos | stack pos | bare-IK rot | stack rot |
| ------ | ----- | ----------- | --------- | ----------- | --------- |
| circle | 4.4   | 0.8 mm      | 1.2 mm    | 0.00°       | 5.0°      |
| square | 9.4   | 2.7 mm      | 2.6 mm    | 0.01°       | 8.0°      |
| line   | 8.8   | 5.6 mm      | 5.7 mm    | 0.03°       | 0.6°      |

The clean fix is to make the IK target frame independent of S7 (use a
wrist tip frame upstream of the gripper joint in the URDF). Until then,
the trade-off is "user controls gripper" vs "perfect orientation tracking
during sustained 2D motion" — picked the former.

## EE-point / tip-frame calibration

The IK currently targets `L7_1` — the SO-107 URDF's CAD-exported tip
frame. Two problems with that:

1. It sits on the gripper body, _downstream_ of S7. That's the root cause
   of the gripper-DOF rotation drift in the section above — S7 affects
   the EE pose, so pinning S7 for teleop discards the IK's S7 choice.
2. It's not actually at the physical gripping point. PR #9's "known
   issues" already flags this: there is a fixed offset to where the
   _closed_-gripper tip would be (mid-point between the two fingertips
   for a parallel gripper), and that offset is per-physical-arm
   calibration data, not a property of the CAD URDF.

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
