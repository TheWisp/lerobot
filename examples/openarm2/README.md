# OpenArm 2 integration and first-run safety guide

This directory documents the staged bring-up of the OpenArm 2 follower in the
A00 LeRobot tree. Complete every stage for one arm before enabling bimanual or
Quest VR control.

## Verified hardware state and mapping

- The target has one PEAK PCAN-USB Pro FD device. That single USB device exposes
  two CAN channels, `can0` and `can1`.
- Both Linux interfaces were observed UP and ERROR-ACTIVE at 1 Mbit/s nominal
  and 5 Mbit/s data rate.
- Torque-disabled physical identification and the subsequent motion tests
  established the channel mapping:

  | Physical side | CAN channel | Status |
  | --- | --- | --- |
  | right | `can0` | verified on the physical robot |
  | left | `can1` | verified on the physical robot |
- J8 on both channels was explicitly placed in POS_FORCE for the current
  bring-up. Treat this as a temporary test mode: read it back before every
  torque-enable session and do not assume it survives a restart or external
  configuration change.
- The completed hardware checkpoints are separate-arm motion at 15°,
  bimanual motion within ±10°, and MJCF gravity
  compensation on the physical robot.

## Model and actuator rules

- The integration uses the OpenArm bimanual MJCF model
  `openarm_bimanual.xml`. Gravity compensation and Cartesian kinematics must use
  this MJCF route. Do not introduce a URDF fallback.
- MJCF arm joint names are `openarm_left_joint1` through
  `openarm_left_joint7` and `openarm_right_joint1` through
  `openarm_right_joint7`.
- Cartesian end-effector sites are `left_ee_control_point` and
  `right_ee_control_point`.
- J1 through J7 use MIT control. J8 is the gripper and uses POS_FORCE in the
  standard OpenArm configuration. Do not include J8 in MIT gravity torque or
  arm-joint telemetry.
- `connect()` is transport-only by default. The default values
  `handshake_on_connect=false`, `configure_on_connect=false`, and
  `enable_torque_on_connect=false` do not enable motors or write a zero.

## Commands excluded from integration testing

Do not run either zero-setting command. They change the persistent motor zero
reference and can invalidate the verified pose and limits:

```bash
openarm-can-cli -i can0 set_zero
openarm-can-cli -i can1 set_zero
```

Do not reconfigure either CAN interface while the verified 1 Mbit/s nominal and
5 Mbit/s data-rate links are working:

```bash
openarm-can-cli -i can0 can_configure -d 5000000
openarm-can-cli -i can1 can_configure -d 5000000
```

`can_configure` changes interface state and is unnecessary while both links are
already UP at the verified rates. Do not use `monitor` as a read-only check: it
can enable motors. Use the passive host checks below and the integration's
refresh/mode-readback paths instead.

## Staged bring-up

## Standard Quest teleoperation entrypoint

The checked-in launcher uses the OpenArm standard-edition mapping and control
regime carried by the validated Dora workspace while keeping LeRobot's native
teleoperation lifecycle:

```bash
source .venv/bin/activate
PYTHONPATH=src:.deps python examples/openarm2/teleoperate_quest_vr.py
```

It configures `can1` as the left arm, `can0` as the right arm, the standard
joint limits and gains, MJCF gravity feed-forward gain 0.9, the standard
OpenArm Quest frame axes and signed gripper ranges. The relative motor limits
are derived from `openarm_standard.yaml`'s rad/s values at the 60 Hz LeRobot
loop rate. Torque is enabled through the follower's full safe-arming sequence
when the launcher starts and disabled on exit.

Open `https://<target-LAN-IP>:8443/` in the Quest browser after the launcher is
ready. The side grip is the motion clutch and the index trigger controls the
gripper. A stale WebXR stream forces idle/hold before the MJCF transform runs.

The launcher intentionally keeps velocity feed-forward disabled because the
Dora backup lists it as future work rather than part of the validated control
regime. It also does not copy `dataflow-vr-custom.yaml` verbatim: the pinned
receiver bundle and that YAML have output-name drift, while LeRobot already
provides the maintained teleoperation and recording loop.

For a local joint/action/observation episode without cameras, use the thin
OpenArm profile over LeRobot's native recorder:

```bash
lerobot-openarm2-record \
  --repo-id local/openarm2-smoke \
  --task "OpenArm 2 Quest teleoperation smoke" \
  --root outputs/openarm2-smoke \
  --episodes 1 \
  --episode-seconds 30
```

This entrypoint does not implement a second recorder. It delegates episode
creation, action/observation fields and dataset finalization to
`lerobot-record`. Camera configuration is intentionally deferred until stable
device names are available on the target; the Dora backup also marks its
post-control-rework recording pipeline as not yet revalidated.

### Stage 0: passive host checks

No CAN frames are transmitted by these commands:

```bash
ip -details -statistics link show can0
ip -details -statistics link show can1
```

Confirm all of the following before continuing:

- both links remain UP and ERROR-ACTIVE;
- nominal bitrate is 1000000 and data bitrate is 5000000;
- transmit errors, receive errors, and bus-off counters do not increase;
- no other process owns or writes either CAN channel;
- the robot workspace is clear and a person is at the physical stop control.

### Stage 1: transport-only software check

Instantiate one `OpenArmFollower` with these settings:

```yaml
port: can0                 # verified right arm
side: right
handshake_on_connect: false
configure_on_connect: false
enable_torque_on_connect: false
disable_torque_on_disconnect: false
gravity_ff_gain: 0.0
max_relative_target: 0.2
gripper_control_mode: pos_force
gripper_speed_rad_s: 1.0
gripper_torque_pu: 0.0
cameras: {}
```

`can0` is the verified right arm. Connect and disconnect without requesting an
observation or action. This stage
checks Python dependencies and SocketCAN access. It must not enable, configure,
zero, or command a motor. Repeat on `can1` with `side: left`.

### Stage 2: refresh-only identification

Keep all Stage 1 settings, set `handshake_on_connect=true`, and connect to only
one channel. The Damiao handshake uses the refresh request and does not enable
torque. Verify that all expected motor IDs answer with finite position,
velocity, and torque values.

This identification has been completed on the physical robot: `can0` is the
right arm and `can1` is the left arm. Repeat identification only after cabling
changes. Keep torque disabled, move one arm by hand through a small comfortable
range, and stop on unexpected resistance, response loss, bus error, or a
non-finite value.

### Stage 3: single-arm hold and minimal displacement

This stage transmits motion commands and requires explicit operator agreement.
Keep Quest VR, cameras, gravity feed-forward, and the second arm disabled.

First-run constraints:

| Parameter | First-run value |
| --- | --- |
| `configure_on_connect` | `false` |
| `enable_torque_on_connect` | `false`; enable only after reading the pose |
| `gravity_ff_gain` | `0.0` |
| `max_relative_target` | `0.2` motor degrees per command |
| J8 mode | `pos_force` |
| `gripper_speed_rad_s` | `1.0` |
| `gripper_torque_pu` | `0.0` during J1-J7 identification |
| action rate | manual single-step commands |
| arms active | one |

Read a complete fresh pose first. Reject the test if any J1-J8 value is
missing, stale, non-finite, or outside the configured absolute limit. Enable
torque explicitly, then issue a hold command equal to that fresh pose. After a
stable hold, test one selected joint with a displacement no greater than 0.2
motor degrees. Return to the measured hold pose before testing another joint.
Disable torque immediately after the test.

Hardware checkpoint completed: each arm was tested separately with a bounded
15° motion. This result validates the tested path and mapping; it does not
remove the fresh-pose, limit, and operator-presence checks for later sessions.

Do not invent reduced Kp/Kd values during this stage. Validate the gain set
against the OpenArm motor configuration before the first enable. A low gain can
allow the arm to fall; an excessive gain can create a hard correction.

### Stage 4: J8 POS_FORCE validation

J8 was temporarily switched to POS_FORCE on both verified channels. Read the
mode back before enabling torque in every new session. Validate gripper
direction separately after J1-J7 pass. Start at the measured
J8 position with `gripper_speed_rad_s=1.0` and a conservative nonzero current
limit approved by the operator. Confirm the left/right sign convention before
increasing travel or force. Never send J8 through the MIT batch in the standard
configuration.

### Stage 5: MJCF gravity feed-forward

Enable gravity feed-forward only after single-arm position control is stable.
Use the verified OpenArm bimanual MJCF path and begin from
`gravity_ff_gain=0.0`. Check joint-name resolution, torque sign, finite output,
torque clamping, fade-in, and low-pass behavior offline before increasing the
gain on hardware. Any dataset must retain one fixed feed-forward and gain
regime.

Hardware checkpoint completed: MJCF gravity compensation was exercised on the
physical robot after the separate-arm tests. The bimanual path was also tested
with bounded commands within ±10°. Keep the same staged
enable, fresh-feedback, joint-limit, and stop-condition checks when repeating
either test.

### Stage 6: computer Cartesian targets, then Quest VR

Verify bounded Cartesian targets generated on the computer before attaching
Quest VR. The Cartesian solver must use the MJCF joint names and end-effector
sites listed above. A stale, disconnected, or discontinuous target stream must
release the control lease and force idle behavior. Add Quest VR only after both
single-arm and bimanual computer-target tests pass.

Quest's generic defaults target a different robot and must be overridden for
OpenArm before the first session:

```yaml
left_gripper_open_motor: 0.0
left_gripper_closed_motor: 45.0
right_gripper_open_motor: 0.0
right_gripper_closed_motor: -45.0
robot_forward_in_urdf: [1.0, 0.0, 0.0]
robot_up_in_urdf: [0.0, 0.0, 1.0]
```

The historical option names contain `urdf`, but the OpenArm solver and model
path in this integration remain MJCF-only.

## Offline follower telemetry

`lerobot.robots.openarm_follower.telemetry.FollowerTelemetry` aggregates the
following values for J1 through J7 without performing CAN I/O:

- maximum absolute commanded-to-measured position error;
- mean, mean absolute, and maximum absolute external torque;
- maximum motor MOS temperature.

When MJCF gravity feed-forward is enabled, pass the torque actually sent in the
MIT torque slot as `tff`. The telemetry helper subtracts it
from measured torque. Do not pass a planned torque that was not transmitted.
J8 remains excluded because its standard mode is POS_FORCE.

Telemetry validates vector length and finite values before updating a window.
A malformed sample raises an error without contaminating the existing
aggregate. Reporting is a logging operation only and never sends a hardware
command.

## Stop conditions

Disable torque and end the test immediately when any of these conditions is
observed:

- physical motion before explicit enable;
- a channel-to-side mismatch;
- missing, stale, or non-finite feedback;
- a target jump beyond the configured relative limit;
- growing CAN error counters or any bus-off event;
- unexpected motor heating, noise, vibration, resistance, or direction;
- loss of the active operator or physical-stop observer.
