# Motor swap migration — preserve calibration without recalibrating

When a Feetech motor on a LeRobot arm dies, replacing it normally requires
running the full `robot.calibrate()` flow, which re-records every joint's
`homing_offset` / `range_min` / `range_max` via human-eyeballed reference
poses. That introduces ±a few degrees of drift on every joint on the arm,
not just the replaced one, and invalidates the alignment with existing
datasets and trained models.

This procedure replaces one motor and writes the **old motor's exact
calibration values** to the new motor's EEPROM. The rest of the arm's
calibration JSON is left unchanged. Datasets, FK, and IK frames stay
valid.

Captured from a real session that replaced the right-arm `shoulder_lift`
on the `white` SO-107 follower.

## Prerequisites

- The dying motor is still electrically readable (you can `ping` it and
  read registers). If the bus can't see it at all, you can't capture its
  state; use the standard calibration flow as a fallback.
- The new motor is the same model and same gear ratio as the old one.
  Different gear ratios produce different encoder ticks per joint degree,
  and the range_min / range_max values from the old calibration won't
  describe the same joint angles on a different ratio.
- You can isolate the new motor on the bus before assigning its ID (break
  the daisy-chain, or use a dedicated USB-to-Feetech adapter).

## Procedure

All commands run from the repo root via the project's Python:

```bash
.venv/bin/python scripts/motor_tools.py <subcommand> ...
```

### 1. Capture the dying motor's calibrated state

Before disconnecting the old motor:

```bash
python scripts/motor_tools.py read /dev/ttyACM2 2
```

Record `Present_Position` (any reading is fine — joint can be free-hanging,
±10 ticks of drift is well inside the 14.4° spline-tooth tolerance) and
`Homing_Offset`. Also note the matching `range_min` and `range_max` from
the robot's calibration JSON
(`~/.cache/huggingface/lerobot/calibration/robots/.../<arm_side>.json`).

For the white right shoulder_lift the calibration JSON has:

```json
"shoulder_lift": {
    "id": 2,
    "homing_offset": -1017,
    "range_min": 803,
    "range_max": 3188
}
```

### 2. Isolate the new motor on the bus

Disconnect the old motor and any other motor that might also answer at
the source ID (most fresh-from-factory motors come at id=1, which is
where the arm's shoulder_pan already lives — wire conflict). Two safe
options:

- Break the daisy-chain at the connector between shoulder_pan and where
  shoulder_lift used to sit. Plug only the new motor into that gap.
- Use a dedicated USB-to-Feetech adapter on a different port for ID
  assignment, then move the motor to the arm bus.

Verify only one motor is on the bus:

```bash
python -c "
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech.feetech import FeetechMotorsBus
PORT = '/dev/ttyACM2'
real = []
for mid in range(1, 30):
    bus = FeetechMotorsBus(port=PORT, motors={f'm{mid}': Motor(mid, 'sts3215', MotorNormMode.RANGE_M100_100)})
    bus.connect(handshake=False)
    try:
        m = bus.ping(f'm{mid}')
        if m is not None: real.append(mid)
    except Exception: pass
    bus.disconnect(disable_torque=False)
print(f'IDs answering: {real}')
"
```

Expect a single ID. If more than one, do not proceed with `set-id`.

### 3. Reassign the new motor's ID

If the bus scan in step 2 reported the new motor at id=5 and you need it
at id=2:

```bash
python scripts/motor_tools.py set-id /dev/ttyACM2 5 2
```

Expected: `OK — motor now answers at id=2`.

### 4. Migrate calibration state and drive shaft to the matching angle

Write the old motor's `Homing_Offset`, `Min_Position_Limit`, and
`Max_Position_Limit` to the new motor's EEPROM, and drive the shaft to
the angle you read in step 1:

```bash
python scripts/motor_tools.py write /dev/ttyACM2 2 69.79 \
    --homing-offset=-1017 \
    --min-position-limit=803 \
    --max-position-limit=3188
```

Without `--min-position-limit` / `--max-position-limit`, a fresh motor
from a previous role may have tight limits that silently clip
`Goal_Position`. The new motor's previous role's limits are unrelated to
your arm — overwrite them.

The motor will drive the shaft to within a few ticks of the target
(P-only steady-state error against gearbox friction), then disable
torque so you can install without it fighting you. The 1° or so residual
is far inside one spline tooth's 14.4° tolerance.

### 5. Mechanical install

1. Attach the horn to the new motor's shaft, picking the spline tooth that
   visually matches how the old horn sat at the joint's install pose.
2. Screw horn to the link via the same screw holes the old horn used.
3. Screw the motor housing into the bracket via the same 4 bolt holes.

If the horn is one tooth off, the arm will sit visibly tilted by ~14° at
this joint when other joints are at rest. Pop the horn off, advance one
spline tooth, retry.

### 6. Verify

Reconnect the rest of the arm's daisy chain. Then:

```bash
python scripts/motor_tools.py read /dev/ttyACM2 2
```

With the arm at whatever pose it settles into, the reported position
should be reasonable. The strongest test is positioning the left and
right arms identically by hand and comparing the two `shoulder_lift`
readings — they should match within ~10 ticks.

Run a regular leader-follower teleop briefly to confirm tracking is
smooth before re-engaging the Quest VR (or whatever else) path.

## Why this works

The motor's encoder reports a 12-bit single-turn position (0–4095). The
calibration normalization layer maps raw ticks to a logical range via
`range_min` and `range_max`, **after** subtracting `homing_offset`:

```
logical = raw_encoder - homing_offset
norm = (logical - range_min) / (range_max - range_min)  # 0..1
```

The mechanical end-stops define `range_min` and `range_max` in physical
joint space. Those positions don't move when you swap the motor — same
bracket, same link, same hard plastic stops. The encoder reading at
those positions, however, does change between motors: each motor has its
own factory zero-mark alignment on the encoder ring, and each install can
pick a different one of the 25 spline teeth.

By writing the old motor's `homing_offset` and `range_min` / `range_max`
to the new motor and matching the install (housing orientation +
horn-to-spline tooth + horn-to-link screws), you ensure the new motor
reports the same `logical` value at every physical joint angle. The
calibration JSON's `range_min` / `range_max` then describe the same
physical joint angles on the new motor as they did on the old one.

## Honest limits

- Manufacturing variation in encoder zero-mark alignment between two
  motors is ~sub-degree. Not zero, but well below visual policy noise.
- Mechanical compliance at the end-stops (plastic-on-plastic deflection,
  user push pressure) means even your "same calibration values" describe
  the same physical pose to ±2–3°. The closed-loop visual policies in
  the LeRobot stack tolerate this on the same noise floor they already
  tolerate.
- Open-loop replay of pre-swap recordings will inherit any drift.

If your downstream use needs better than this, you need a vision-based
external calibration loop. None of that is in this codebase today.
