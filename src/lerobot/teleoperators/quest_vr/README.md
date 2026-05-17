# Quest VR teleoperator

WebXR-based 6-DOF teleop for Cartesian-IK-capable robots. Conforms to LeRobot's
`Teleoperator` interface. Emits the same EE-delta action format as the existing
`keyboard_ee` and `phone` teleops, so it slots into any pipeline that already
handles Cartesian control.

## How it works

```
[Quest 3 browser]                  [PC]
─────────────────                  ─────
WebXR session     ──HTTPS+WSS──>   QuestServer (daemon thread)
controller poses                         │
@ 90 Hz                                  ▼ on_frame callback
                                   QuestVRTeleop._on_frame
                                         │
                                         ▼ cache update (under lock)
                                   QuestVRTeleop._cached_action
                                         ▲
                                         │ lock-free read
                                   get_action()
                                         │
                                         ▼
                                   EEReferenceAndDelta -> EEBoundsAndSafety
                                   -> GripperVelocityToJoint
                                   -> PinkInverseKinematicsEEToJoints
                                         │
                                         ▼ <motor>.pos
                                   robot.send_action()
```

Sample rate is fully decoupled from poll rate (same pattern as
`HighRateLeaderMixin`): Quest streams at ~90 Hz, the run loop polls
`get_action()` at whatever frequency it wants and always gets the most recent
cached action.

## Setup

1. **Install pink + a QP solver** (optional dependency for the recommended IK path):
   ```bash
   uv pip install pin-pink qpsolvers[open_source_solvers]
   ```
2. **Open port 8443 in the PC's firewall**: `sudo ufw allow 8443/tcp`.
3. **Quest 3 on the same WiFi as the PC**.
4. **Disable the proximity sensor on the Quest** if you want to wear the headset
   on your chest/neck during collocated teleop (otherwise the immersive session
   pauses when the headset comes off your face). Tape over the sensor (inside,
   top-center between the lenses) is the most reliable approach.

## Action format

`get_action()` returns:

| key                               | type        | meaning                                                 |
| --------------------------------- | ----------- | ------------------------------------------------------- |
| `enabled`                         | float (0/1) | clutch state (right grip pressed)                       |
| `target_x, target_y, target_z`    | float       | position delta from engage snapshot, in robot frame (m) |
| `target_wx, target_wy, target_wz` | float       | rotation delta as rotvec                                |
| `gripper_vel`                     | float       | gripper velocity command derived from trigger delta     |

Matches what `EEReferenceAndDelta` consumes upstream.

## CLI test (manual composition)

The example at `examples/quest_vr_to_so107/teleoperate.py` shows how to compose
the full pipeline (teleop → EEReferenceAndDelta → EEBoundsAndSafety →
GripperVelocityToJoint → PinkInverseKinematicsEEToJoints → robot):

```bash
python examples/quest_vr_to_so107/teleoperate.py \
    --robot-port /dev/ttyACM2 --robot-id white_right
```

It prints a URL like `https://192.168.x.x:8443/`. Open it in the Quest browser,
accept the self-signed cert warning, tap **Connect** + **Enter AR**, then squeeze
the right grip to engage tracking.

## GUI integration status

The teleop config (`QuestVRTeleopConfig`) is registered with `TeleoperatorConfig`,
so the GUI's auto-discovery (`pkgutil.walk_packages` in
`src/lerobot/gui/api/robot.py:_ensure_configs_loaded`) picks it up — **it
appears in the GUI's teleop dropdown without any GUI-side changes**.

**However**, the GUI's standard `lerobot-teleoperate` flow uses _identity_
processor pipelines by default. For Cartesian teleops (`quest_vr`, `keyboard_ee`,
`phone`) to work end-to-end, the script needs to compose the Cartesian IK
pipeline. That auto-composition is **not yet implemented in `lerobot-teleoperate`**;
the current end-to-end path uses the example script above.

To unlock GUI launches with quest_vr, the path is:

- detect Cartesian teleops (e.g., `"target_x" in teleop.action_features["names"]`)
- inject the right pipeline (EEReferenceAndDelta + bounds + gripper + IK)
  in `lerobot-teleoperate`'s default processors

That's tracked as a follow-up.

## What's tested

- `tests/teleoperators/test_quest_vr.py`: 10 unit tests covering config
  registration, factory dispatch, action shape, clutch state machine, axis
  mapping, gripper_vel derivation, hand filtering.
- `tests/robots/test_pink_kinematic_processor.py`: 7 tests covering the IK
  ProcessorStep.
- `tests/model/test_pink_kinematics.py`: 8 tests covering the underlying IK.

End-to-end physical testing requires a Quest 3 + SO-107 hardware; run the
example script.
