#!/usr/bin/env python
"""Feetech motor-bus diagnostics, inventory, and migration helpers.

A single CLI with subcommands. Use this when you need to inspect or modify
a single motor's state without booting the whole robot stack — e.g. to
diagnose a dying motor, audit which physical motor variants are wired
across your arms, or migrate the calibration of one motor onto a fresh
replacement without recalibrating the rest of the chain.

All subcommands operate against a directly-opened ``FeetechMotorsBus``, so
they bypass robot configs, the GUI, and any predictive-controller threads
that might otherwise hold the port.

Subcommands:

  inventory   Scan one or more ports and print every responding motor's
              firmware version, voltage configuration, and current voltage
              reading. Useful for spotting which motors have had their
              ``Max_Voltage_Limit`` overwritten (a writable EEPROM field
              that does NOT reliably distinguish 7.4V vs 12V variants —
              ground truth is the printed label on the motor housing).

  health      Walk a structured diagnostic on one motor: ping, register
              snapshot, EEPROM config, commanded +/- moves with a watch
              loop. Diagnostic verdicts:
                * Ping fails              -> bus / wiring problem
                * Goal_Position latches,  -> firmware OK, MCU rejects writes
                  Torque_Enable won't       (replace motor)
                * Goal/Pres diverge,      -> driver/power stage dead
                  Current = 0               (replace motor)
                * Current spikes,         -> mechanical jam
                  Pres doesn't change       (inspect linkage)
                * Pres stops short of     -> stale Min/Max_Position_Limit
                  Goal at a flat number     (use ``write`` with
                                             --min-position-limit / --max-position-limit)

  read        Read one motor's Present_Position, Homing_Offset, and
              torque state. Useful before pulling a dying motor (capture
              its calibrated state) so the replacement can be configured
              to match without recalibrating.

  write       Drive one motor's shaft to a target angle (degrees by
              default, or ticks with --raw-ticks). Optionally also writes
              Homing_Offset, Min_Position_Limit, and Max_Position_Limit
              to the motor's EEPROM in the same call — the typical
              migration workflow (see scripts/MOTOR_MIGRATION.md).

  set-id      Reassign a motor's bus ID. The bus MUST have only the one
              motor during this call (broadcast traps in stock LeRobot
              scripts can reassign the wrong motor; this script uses a
              targeted write, but a duplicate-ID hazard remains if the
              new ID is already in use by another motor on the chain).

Examples:

  # Inventory all 4 arms
  python scripts/motor_tools.py inventory \\
      /dev/ttyACM0 /dev/ttyACM1 /dev/ttyACM2 /dev/ttyACM3

  # Health-probe right-arm shoulder_lift
  python scripts/motor_tools.py health /dev/ttyACM2 2

  # Capture the dying motor's calibrated angle + homing_offset before
  # pulling it (Homing_Offset value goes into the migration write).
  python scripts/motor_tools.py read /dev/ttyACM2 2

  # On a fresh replacement (alone on the bus), set ID and migrate state.
  python scripts/motor_tools.py set-id /dev/ttyACM2 1 2
  python scripts/motor_tools.py write /dev/ttyACM2 2 69.79 \\
      --homing-offset=-1017 \\
      --min-position-limit=803 \\
      --max-position-limit=3188

Calibration values come from the original motor's entry in the robot's
calibration JSON (``~/.cache/huggingface/lerobot/calibration/robots/.../<arm>.json``).
"""

from __future__ import annotations

import argparse
import contextlib
import time
from collections.abc import Iterable

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech.feetech import FeetechMotorsBus

TICKS_PER_TURN = 4096
MOTOR_NAMES_SO107 = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


# ── conversion ────────────────────────────────────────────────────────────


def ticks_to_deg(ticks: int) -> float:
    return ticks * 360.0 / TICKS_PER_TURN


def deg_to_ticks(deg: float) -> int:
    return int(round(deg * TICKS_PER_TURN / 360.0))


# ── bus open ──────────────────────────────────────────────────────────────


def _open_single(port: str, motor_id: int) -> FeetechMotorsBus:
    bus = FeetechMotorsBus(
        port=port,
        motors={"motor": Motor(motor_id, "sts3215", MotorNormMode.RANGE_M100_100)},
    )
    bus.connect(handshake=False)
    return bus


# ── subcommand: inventory ─────────────────────────────────────────────────


def cmd_inventory(ports: Iterable[str]) -> None:
    """Scan each port for motors at every SO-107 motor id (1-7) and print
    firmware + voltage info. Caveats:

      * STS3215 variants (7.4V vs 12V; 1:147, 1:191, 1:345 gear ratios)
        all share Model_Number=777 and identical firmware. Variant is not
        reliably distinguishable in software.
      * Max_Voltage_Limit is a writable EEPROM field. A heuristic that
        calls 8.0V "7.4V variant" and 12.0V "12V variant" breaks the
        moment anyone has written that register — real example: leader
        arms in ``white`` have per-motor overwrites that don't reflect the
        physical motor variant.
      * Gear ratio (1:345 etc.) is NOT exposed in any register — the MCU
        doesn't know what's on the output shaft.

    Ground truth for variant identification is the printed label on the
    motor housing. This report is honest diagnostic info; it does NOT
    pretend to identify variants.
    """
    for port in ports:
        print(f"\n=== {port} ===")
        try:
            motors = {
                name: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100)
                for i, name in enumerate(MOTOR_NAMES_SO107)
            }
            bus = FeetechMotorsBus(port=port, motors=motors)
            bus.connect(handshake=False)
        except Exception as e:
            print(f"  open failed: {e}")
            continue

        print(f"  {'name':<14} {'id':>2}  {'FW':>6}  {'MaxV':>5}  {'PresV':>5}")
        try:
            for name in MOTOR_NAMES_SO107:
                try:
                    model = bus.ping(name)
                    if model is None:
                        # Silent no-response. Skip.
                        continue
                    fw_maj = bus.read("Firmware_Major_Version", name, normalize=False)
                    fw_min = bus.read("Firmware_Minor_Version", name, normalize=False)
                    max_v = bus.read("Max_Voltage_Limit", name, normalize=False)
                    pres_v = bus.read("Present_Voltage", name, normalize=False)
                    print(
                        f"  {name:<14} {motors[name].id:>2}  "
                        f"{fw_maj}.{fw_min:<4}  {max_v / 10:>4.1f}V  {pres_v / 10:>4.1f}V"
                    )
                except Exception:
                    pass
        finally:
            bus.disconnect(disable_torque=False)


# ── subcommand: health ────────────────────────────────────────────────────


def cmd_health(port: str, motor_id: int) -> None:
    """Structured diagnostic on one motor. Reports a verdict at the end."""
    bus = _open_single(port, motor_id)
    try:
        try:
            model = bus.ping("motor")
            print(f"[1] ping ok: model={model!r}")
        except Exception as e:
            print(f"[1] PING FAILED — bus or wiring problem: {e}")
            return

        def snap(label: str) -> tuple[int | None, int | None]:
            try:
                te = bus.read("Torque_Enable", "motor", normalize=False)
                gp = bus.read("Goal_Position", "motor", normalize=False)
                pp = bus.read("Present_Position", "motor", normalize=False)
                pc = bus.read("Present_Current", "motor", normalize=False)
                mv = bus.read("Moving", "motor", normalize=False)
                print(
                    f"[{label}] "
                    f"Torque_Enable={te}  Goal_Position={gp}  Present_Position={pp}  "
                    f"Present_Current={pc}  Moving={mv}"
                )
                return pp, te
            except Exception as e:
                print(f"[{label}] register read FAILED: {e}")
                return None, None

        print("\n[2] initial state:")
        pos0, _ = snap("initial")

        print("\n[2b] EEPROM / mode registers:")
        for reg in (
            "Operating_Mode",
            "Lock",
            "Min_Position_Limit",
            "Max_Position_Limit",
            "Max_Torque_Limit",
            "Torque_Limit",
            "P_Coefficient",
        ):
            try:
                print(f"    {reg} = {bus.read(reg, 'motor', normalize=False)}")
            except Exception as e:
                print(f"    {reg} read FAILED: {e}")

        print("\n[3] enabling torque (Lock=0 first so EEPROM is writable) ...")
        try:
            bus.write("Lock", "motor", 0, normalize=False)
            bus.write("Torque_Enable", "motor", 1, normalize=False)
        except Exception as e:
            print(f"  write FAILED: {e}")
        time.sleep(0.05)
        snap("after torque enable")

        if pos0 is None:
            print("\n[4] skipping move test — couldn't read initial position")
            return

        for direction, delta in (("UP", +100), ("DOWN", -100)):
            target = pos0 + delta
            print(f"\n[4-{direction}] commanding Goal_Position={target} (was {pos0}) ...")
            try:
                bus.write("Goal_Position", "motor", target, normalize=False)
            except Exception as e:
                print(f"  write FAILED: {e}")
                continue

            t_start = time.monotonic()
            for _ in range(15):
                time.sleep(0.1)
                try:
                    pp = bus.read("Present_Position", "motor", normalize=False)
                    pc = bus.read("Present_Current", "motor", normalize=False)
                    gp = bus.read("Goal_Position", "motor", normalize=False)
                    mv = bus.read("Moving", "motor", normalize=False)
                    print(
                        f"    t={time.monotonic() - t_start:4.1f}s  Goal={gp}  Pos={pp}  "
                        f"d(Pos-pos0)={pp - pos0:+4d}  Current={pc}  Moving={mv}"
                    )
                except Exception as e:
                    print(f"    read FAILED: {e}")
                    break

        print("\n[5] disabling torque ...")
        with contextlib.suppress(Exception):
            bus.write("Torque_Enable", "motor", 0, normalize=False)
    finally:
        bus.disconnect(disable_torque=False)


# ── subcommand: read ──────────────────────────────────────────────────────


def cmd_read(port: str, motor_id: int) -> None:
    bus = _open_single(port, motor_id)
    try:
        bus.ping("motor")
        pres = bus.read("Present_Position", "motor", normalize=False)
        ho = bus.read("Homing_Offset", "motor", normalize=False)
        torque = bus.read("Torque_Enable", "motor", normalize=False)
        print(f"port={port} id={motor_id}")
        print(f"  Present_Position = {pres} ticks  ({ticks_to_deg(pres):.2f}°)")
        print(f"  Homing_Offset    = {ho} ticks    ({ticks_to_deg(ho):.2f}°)")
        print(f"  Torque_Enable    = {torque}")
    finally:
        bus.disconnect(disable_torque=False)


# ── subcommand: write ─────────────────────────────────────────────────────


def cmd_write(
    port: str,
    motor_id: int,
    target: float,
    raw_ticks: bool,
    homing_offset: int | None,
    min_position_limit: int | None,
    max_position_limit: int | None,
) -> None:
    target_ticks = int(target) if raw_ticks else deg_to_ticks(target)
    target_deg = ticks_to_deg(target_ticks)
    bus = _open_single(port, motor_id)
    try:
        bus.ping("motor")
        start_pos = bus.read("Present_Position", "motor", normalize=False)
        ho_before = bus.read("Homing_Offset", "motor", normalize=False)
        min_before = bus.read("Min_Position_Limit", "motor", normalize=False)
        max_before = bus.read("Max_Position_Limit", "motor", normalize=False)
        print(f"port={port} id={motor_id}")
        print(f"  start Present_Position    = {start_pos} ticks  ({ticks_to_deg(start_pos):.2f}°)")
        print(f"  current Homing_Offset     = {ho_before} ticks       ({ticks_to_deg(ho_before):.2f}°)")
        print(f"  current Min_Position_Limit = {min_before}")
        print(f"  current Max_Position_Limit = {max_before}")

        needs_eeprom_write = (
            (homing_offset is not None and homing_offset != ho_before)
            or (min_position_limit is not None and min_position_limit != min_before)
            or (max_position_limit is not None and max_position_limit != max_before)
        )
        if needs_eeprom_write:
            bus.write("Lock", "motor", 0, normalize=False)
            bus.write("Torque_Enable", "motor", 0, normalize=False)

        if homing_offset is not None and homing_offset != ho_before:
            print(f"  writing Homing_Offset = {homing_offset} ticks ({ticks_to_deg(homing_offset):.2f}°) ...")
            bus.write("Homing_Offset", "motor", homing_offset, normalize=False)
            time.sleep(0.05)
            print(f"    Homing_Offset readback = {bus.read('Homing_Offset', 'motor', normalize=False)}")

        if min_position_limit is not None and min_position_limit != min_before:
            print(f"  writing Min_Position_Limit = {min_position_limit} ...")
            bus.write("Min_Position_Limit", "motor", min_position_limit, normalize=False)
            time.sleep(0.05)
            print(
                f"    Min_Position_Limit readback = "
                f"{bus.read('Min_Position_Limit', 'motor', normalize=False)}"
            )

        if max_position_limit is not None and max_position_limit != max_before:
            print(f"  writing Max_Position_Limit = {max_position_limit} ...")
            bus.write("Max_Position_Limit", "motor", max_position_limit, normalize=False)
            time.sleep(0.05)
            print(
                f"    Max_Position_Limit readback = "
                f"{bus.read('Max_Position_Limit', 'motor', normalize=False)}"
            )

        print(f"  target = {target_ticks} ticks  ({target_deg:.2f}°)")
        print("  enabling torque + writing Goal_Position ...")
        bus.write("Torque_Enable", "motor", 1, normalize=False)
        bus.write("Goal_Position", "motor", target_ticks, normalize=False)

        t0 = time.monotonic()
        last = start_pos
        settled = 0
        while time.monotonic() - t0 < 8.0:
            time.sleep(0.1)
            pres = bus.read("Present_Position", "motor", normalize=False)
            err = pres - target_ticks
            print(
                f"  t={time.monotonic() - t0:4.1f}s  Present={pres} ({ticks_to_deg(pres):.2f}°)  "
                f"err={err:+d} ticks ({ticks_to_deg(abs(err)):.2f}°)"
            )
            if abs(err) < 5 and abs(pres - last) < 2:
                settled += 1
                if settled >= 2:
                    print("  settled.")
                    break
            else:
                settled = 0
            last = pres

        print("  disabling torque so you can swap horn / install ...")
        bus.write("Torque_Enable", "motor", 0, normalize=False)
    finally:
        bus.disconnect(disable_torque=False)


# ── subcommand: set-id ────────────────────────────────────────────────────


def cmd_set_id(port: str, current_id: int, new_id: int) -> None:
    """Change a motor's bus ID.

    WARNING: the motor must be the ONLY motor on the bus during this call.
    If any other motor on the chain also answers at ``current_id``, this
    targeted write hits it too. Either physically isolate the motor (break
    the daisy-chain) or use a dedicated USB-to-Feetech adapter.
    """
    assert 1 <= new_id <= 253, "new_id must be in [1, 253]"
    bus = _open_single(port, current_id)
    try:
        try:
            bus.ping("motor")
        except Exception as e:
            print(f"ping at id={current_id} FAILED: {e}")
            return
        print(f"port={port}  current id={current_id} -> new id={new_id}")
        bus.write("Lock", "motor", 0, normalize=False)
        bus.write("Torque_Enable", "motor", 0, normalize=False)
        bus.write("ID", "motor", new_id, normalize=False)
        time.sleep(0.1)
    finally:
        bus.port_handler.closePort()

    print(f"  re-opening at id={new_id} to confirm ...")
    bus2 = _open_single(port, new_id)
    try:
        bus2.ping("motor")
        readback = bus2.read("ID", "motor", normalize=False)
        if readback == new_id:
            print(f"  OK — motor now answers at id={readback}")
        else:
            print(f"  UNEXPECTED — readback id={readback} (wanted {new_id})")
    finally:
        bus2.disconnect(disable_torque=False)


# ── argparse ──────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="Feetech motor-bus diagnostics, inventory, and migration helpers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    inv = sub.add_parser("inventory", help="Scan ports for motors; print firmware + voltage info.")
    inv.add_argument("ports", nargs="+", help="one or more serial ports (e.g. /dev/ttyACM0)")

    hp = sub.add_parser("health", help="Structured diagnostic on one motor.")
    hp.add_argument("port")
    hp.add_argument("id", type=int)

    r = sub.add_parser("read", help="Read one motor's Present_Position + Homing_Offset.")
    r.add_argument("port")
    r.add_argument("id", type=int)

    w = sub.add_parser("write", help="Drive one motor to a target angle, optionally writing EEPROM.")
    w.add_argument("port")
    w.add_argument("id", type=int)
    w.add_argument("angle", type=float, help="target angle (degrees by default)")
    w.add_argument(
        "--raw-ticks",
        action="store_true",
        help="interpret 'angle' as raw encoder ticks instead of degrees",
    )
    w.add_argument(
        "--homing-offset",
        type=int,
        default=None,
        help="also write this Homing_Offset (ticks) to EEPROM",
    )
    w.add_argument(
        "--min-position-limit",
        type=int,
        default=None,
        help="also write this Min_Position_Limit (ticks) to EEPROM",
    )
    w.add_argument(
        "--max-position-limit",
        type=int,
        default=None,
        help="also write this Max_Position_Limit (ticks) to EEPROM",
    )

    s = sub.add_parser("set-id", help="Reassign a motor's bus id (bus must have only this motor).")
    s.add_argument("port")
    s.add_argument("current_id", type=int)
    s.add_argument("new_id", type=int)

    args = p.parse_args()
    if args.cmd == "inventory":
        cmd_inventory(args.ports)
    elif args.cmd == "health":
        cmd_health(args.port, args.id)
    elif args.cmd == "read":
        cmd_read(args.port, args.id)
    elif args.cmd == "write":
        cmd_write(
            args.port,
            args.id,
            args.angle,
            args.raw_ticks,
            args.homing_offset,
            args.min_position_limit,
            args.max_position_limit,
        )
    else:
        cmd_set_id(args.port, args.current_id, args.new_id)


if __name__ == "__main__":
    main()
