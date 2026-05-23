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

## Tighter IK tracking on moving targets

`PinkKinematics` (from #9) is conservatively tuned: per call its QP returns
a bounded joint velocity, so on a continuously-moving teleop target the
lag scales with target velocity. Bumping `max_iters` doesn't help — the
IK exits on the convergence threshold before consuming the budget. The
actual lever is the `FrameTask` cost versus `PostureTask` cost inside
`PinkKinematics`, or a velocity feed-forward layer.

For VR teleop this is **not a UX blocker**. Numbers from
`benchmarks/cartesian_ik_tracking.py` (a 30 mm-radius circle through the
bimanual stack at varying speeds — see `docs/cartesian_ik_tradeoff.png`):

| Peak EE speed                  | Position drift | Rotation drift |
| ------------------------------ | -------------- | -------------- |
| 1.1 cm/s (slow precision)      | 0.4 mm         | 1.6°           |
| 4.4 cm/s (moderate sustained)  | 1.2 mm         | 5.0°           |
| 8.8 cm/s (fast reach)          | 1.8 mm         | 6.9°           |
| 17 cm/s (very fast continuous) | 3.0 mm         | 8.0°           |

The numbers are _worst-case during sustained continuous motion_; real
teleop has pauses where the IK converges and the drift → 0. Tighten the
QP gains only if hardware reveals a need (carefully — aggressive tuning
can re-introduce the null-space chatter the PostureTask was added to fix
in the first place), or for higher-speed use cases like trajectory replay
or autonomous policy execution.

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
