"""Time the jog's checked bus read and goal write while other Python threads of the process compete for the GIL.

A fake bus of seven Feetech servos answers on a pseudo-terminal from a separate process (``--serve``), one
os.write per reply, so the host reads them in pieces as it does from the USB adapter. The measured thread runs a
30 Hz tick: ``sync_read_block`` of the jog's state registers, then ``sync_write`` of Goal_Position. Load:

  py:N    N threads of small-array numpy and Python arithmetic (an IK solve), busy for --duty of every 20 ms
  proc:N  the same work in N separate processes: CPU load without GIL contention
  json:N  N threads each encoding a 100k-item dict every 250 ms: one long C call that never yields the GIL

``--sdk-poll`` reads replies with the SDK's own polling ``rxPacket`` instead of waiting for the block.
``--throughput`` instead measures what the switch interval costs threads that only compute.
No robot is touched.

usage: motor_bus_gil.py [--seconds 10] [--load py:2] [--duty 0.5] [--switch-ms 5] [--sdk-poll] [--throughput]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import select
import subprocess
import sys
import threading
import time
import tty

import numpy as np

HZ = 30.0
STATE = ("Present_Position", "Present_Velocity", "Present_Load", "Present_Voltage", "Present_Temperature")
NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "forearm_roll", "wrist_flex", "wrist_roll", "gripper")


def serve(reply_delay_s: float = 300e-6, gap_s: float = 100e-6) -> None:
    """The fake servos: answer SYNC_READ (0x82) with each servo's reply in id order; swallow writes."""
    master, slave = os.openpty()
    tty.setraw(master)
    tty.setraw(slave)
    print(os.ttyname(slave), flush=True)
    buf = bytearray()
    while True:
        ready, _, _ = select.select([master, sys.stdin], [], [], 1.0)
        if sys.stdin in ready and not sys.stdin.buffer.read1(1):
            return
        if master not in ready:
            continue
        buf += os.read(master, 4096)
        while (
            (start := buf.find(b"\xff\xff")) >= 0
            and len(buf) - start >= 4
            and len(buf) - start >= buf[start + 3] + 4
        ):
            del buf[:start]
            packet, buf = bytes(buf[: buf[3] + 4]), buf[buf[3] + 4 :]
            if packet[4] != 0x82:
                continue
            addr, length, ids = packet[5], packet[6], packet[7:-1]
            time.sleep(reply_delay_s)
            for id_ in ids:
                data = [{56: 0x00, 57: 0x08, 62: 120, 63: 30}.get(addr + k, 0) for k in range(length)]
                body = [id_, length + 2, 0, *data]
                os.write(master, bytes([0xFF, 0xFF, *body, ~sum(body) & 0xFF]))
                time.sleep(gap_s)


def ik_like(stop: threading.Event, duty: float) -> None:
    a = np.random.default_rng(0).normal(size=(6, 6)) + 6 * np.eye(6)
    t = np.eye(4)
    while not stop.is_set():
        started = time.perf_counter()
        while time.perf_counter() - started < 0.02 * duty:
            for _ in range(20):
                t = t @ np.eye(4)
                np.linalg.solve(a, np.ones(6))
                sum(i * 0.5 for i in range(50))
        time.sleep(0.02 * (1.0 - duty))


def encode_json(stop: threading.Event) -> None:
    big = {f"k{i}": [float(i), i * 0.5, {"a": [1, 2, 3], "b": "x" * 10}] for i in range(100_000)}
    while not stop.is_set():
        json.dumps(big)
        time.sleep(0.25)


def throughput(seconds: float = 3.0) -> None:
    """Work done per second by 1 and 3 always-busy IK-like threads at each switch interval."""

    def work(stop, out, i):
        a = np.random.default_rng(i).normal(size=(6, 6)) + 6 * np.eye(6)
        while not stop.is_set():
            np.linalg.solve(a, np.ones(6))
            sum(k * 0.5 for k in range(50))
            out[i] += 1

    for switch_s in (0.005, 0.001, 0.0005):
        sys.setswitchinterval(switch_s)
        for n in (1, 3):
            stop, out = threading.Event(), [0] * n
            threads = [threading.Thread(target=work, args=(stop, out, i)) for i in range(n)]
            for t in threads:
                t.start()
            time.sleep(seconds)
            stop.set()
            for t in threads:
                t.join()
            print(f"switch {switch_s * 1000:.1f} ms, {n} threads: {sum(out) / seconds:,.0f} iterations/s")


def _proc_load(duty: float) -> None:
    ik_like(threading.Event(), duty)


def pct(x: np.ndarray) -> str:
    return "p50 {:6.2f}  p95 {:6.2f}  p99 {:6.2f}  max {:7.2f}".format(
        *np.percentile(x, [50, 95, 99]), x.max()
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--load", nargs="*", default=[])
    p.add_argument("--duty", type=float, default=0.5)
    p.add_argument("--switch-ms", type=float, default=5.0)
    p.add_argument("--sdk-poll", action="store_true")
    p.add_argument("--throughput", action="store_true")
    p.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.serve:
        serve()
        return
    if a.throughput:
        throughput()
        return

    import lerobot.motors.feetech.feetech as feetech
    from lerobot.motors import Motor, MotorNormMode

    if a.sdk_poll:
        feetech._receive = lambda port, n: b""
    sys.setswitchinterval(a.switch_ms / 1000.0)
    fake = subprocess.Popen(
        [sys.executable, __file__, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    bus = feetech.FeetechMotorsBus(
        port=fake.stdout.readline().strip(),
        motors={n: Motor(i + 1, "sts3215", MotorNormMode.DEGREES) for i, n in enumerate(NAMES)},
    )
    bus.connect(handshake=False)
    stop = threading.Event()
    threads, procs = [], []
    for spec in a.load:
        kind, n = spec.split(":")
        for _ in range(int(n)):
            if kind == "py":
                threads.append(threading.Thread(target=ik_like, args=(stop, a.duty), daemon=True))
            elif kind == "json":
                threads.append(threading.Thread(target=encode_json, args=(stop,), daemon=True))
            elif kind == "proc":
                procs.append(mp.get_context("spawn").Process(target=_proc_load, args=(a.duty,), daemon=True))
            else:
                raise SystemExit(f"unknown load {kind!r}")
    for w in (*threads, *procs):
        w.start()
    time.sleep(0.5)
    rows = []
    goal = dict.fromkeys(NAMES, 2048)
    next_tick = time.perf_counter()
    end = next_tick + a.seconds
    while next_tick < end:
        time.sleep(max(0.0, next_tick - time.perf_counter()))
        t0, c0 = time.perf_counter(), time.thread_time()
        block = bus.sync_read_block(STATE, normalize=False)
        t1, c1 = time.perf_counter(), time.thread_time()
        bus.sync_write("Goal_Position", goal, normalize=False)
        t2 = time.perf_counter()
        rows.append(
            (
                t0 - next_tick,
                t1 - t0,
                c1 - c0,
                block.timing.tx_ms / 1e3,
                block.timing.rx_ms / 1e3,
                t2 - t1,
                t2,
            )
        )
        next_tick += 1.0 / HZ
        if next_tick <= time.perf_counter():
            next_tick = time.perf_counter()
    stop.set()
    for pr in procs:
        pr.terminate()
    fake.stdin.close()
    fake.wait(timeout=5)
    late, read, cpu, tx, rx, write, sent = (1000.0 * np.array(c) for c in zip(*rows, strict=True))
    print(
        f"load={a.load} duty={a.duty} switch={a.switch_ms} ms {'sdk-poll' if a.sdk_poll else 'block wait'}: "
        f"{len(rows)} ticks, {1000.0 * (len(rows) - 1) / (sent[-1] - sent[0]):.2f} writes/s"
    )
    for name, x in (
        ("read", read),
        ("read CPU", cpu),
        ("tx", tx),
        ("rx", rx),
        ("write", write),
        ("wake late", late),
    ):
        print(f"  {name:9s} ms  {pct(x)}")


if __name__ == "__main__":
    main()
