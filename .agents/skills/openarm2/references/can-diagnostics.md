# OpenArm CAN diagnostics

Use the official
[OpenArm CAN CLI reference](https://docs.openarm.dev/api-reference/can/cli/)
for current syntax and behavior. Resolve interface names, motor IDs, bitrates,
and motor models from the active profile and hardware; do not assume copied
values remain correct.

## Install the official CLI

Install only when the CLI is unavailable and system changes are authorized.
Follow the official repository setup:

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:openarm/main
sudo apt update
sudo apt install -y libopenarm-can-dev openarm-can-utils
```

## Protect active control sessions

Before diagnostics, confirm that teleoperation, inference, recording,
calibration, and other motor-control processes are stopped for the target
interface.

- `show_param` and `monitor` send diagnostic traffic but do not reconfigure the
  interface or motor.
- `can_configure` changes the host SocketCAN interface and can disrupt an
  active controller.
- `discover` scans multiple interface bitrates. The official CLI may leave the
  interface at the final scanned bitrate, so always restore the profile's
  interface configuration afterward.
- Treat `enable`, `set_zero`, `clear_error`, `write_param`, `change_id`, and
  `change_baud` as state-changing commands. Use them only with an explicit
  target and safety plan. Flash-saving options have finite write endurance.

## Diagnose before changing configuration

Start with interface and error-counter status:

```bash
ip -details -statistics link show <interface>
```

Read the target motor parameters, then compare with a responding neighboring
motor or an equivalent motor on the other arm:

```bash
openarm-can-cli -i <interface> show_param --id <motor-id>
openarm-can-cli -i <interface> monitor --id <motor-id>
```

If the expected ID does not respond and no active controller owns the
interface, discover IDs and bitrates:

```bash
openarm-can-cli -i <interface> discover
```

After discovery, restore the nominal and data-phase bitrates from the active
profile rather than copying historical values:

```bash
openarm-can-cli -i <interface> can_configure \
  -b <nominal-bitrate> -d <data-bitrate>
```

Use `--full-scan` only when the normal discovery scan is insufficient.

## Interpret a single missing motor

Check the target motor's model-specific manual for LED and fault semantics.
For the Damiao motors documented for OpenArm, a solid red LED indicates
powered but disabled, green indicates enabled, and flashing red indicates a
fault. A healthy SocketCAN error counter does not prove that the final cable
segment reaches the missing motor.

When neighboring motors respond, the bus counters remain healthy, and one
powered motor gives no parameter, enable, refresh, or control response:

1. Do not change IDs, baudrates, or zero positions as a first reaction.
2. Power off the arm.
3. Reseat and inspect the missing motor's local power/CAN connector and cable.
4. Power on and repeat the targeted parameter read.
5. If it remains absent, use the manufacturer UART debugger or a known-good
   cable to distinguish a CAN wiring fault from a motor controller or
   transceiver fault.

If the motor moves but feedback is absent, inspect its response/master ID and
return path. If feedback contains a documented fault code, diagnose that fault
before clearing it.
