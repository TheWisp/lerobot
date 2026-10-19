"""Does a replayed demo take the demo's wall-clock time, whatever the demo's frame rate and the motor loop's?

A joint-command surrogate of the Act's follow replay, run on a recorded demo segment (default: the insertion demo's
key frame 6 to 7). The producer turns demo samples into motor commands with due times; the motor loop sends them.
No IK, servo dynamics or contact; the producer is assumed to keep up (queue starvation is a separate effect).

Producers:
  ceil   each sample gets max(1, ceil(dt * hz - 1e-8)) ticks (the producer of 2026-10-16)
  round  each sample gets max(1, round(dt * hz)) ticks
  clock  as live_execution.stream (removed 2026-10-19): a step takes the ticks between its two samples' ticks on the demo's clock
         (rounded half up); samples sharing a tick are merged while the gripper keeps its direction (still,
         opening or closing); ticks between samples are interpolated
Consumers:
  one     each tick sends the oldest due command (the jog of 2026-10-16)
  newest  as jog._advance_playback: each tick sends the newest due command, never skipping one where the gripper
          starts, stops or reverses, nor one further than PLAYBACK_LAG_DEG from the arm (here: the last command sent)

usage: playback_clock_sim.py DEMO_NPZ [--from-t 12.616 --to-t 17.425] [--act-arm ARM_NPZ]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

PLAYBACK_LAG_DEG = 3.0  # jog.PLAYBACK_LAG_DEG


def resample(t: np.ndarray, q: np.ndarray, fps: float | None) -> tuple[np.ndarray, np.ndarray]:
    """The same motion over the same duration at ``fps`` (None keeps the recorded samples)."""
    if fps is None:
        return t, q
    nt = np.arange(t[0], t[-1] + 1e-9, 1.0 / fps)
    return nt, np.stack([np.interp(nt, t, q[:, j]) for j in range(q.shape[1])], axis=1)


def direction(d: np.ndarray | float) -> np.ndarray:
    """jog.gripper_direction: -1, 0 or 1 within 1e-6."""
    d = np.asarray(d, float)
    return np.where(np.abs(d) <= 1e-6, 0, np.sign(d)).astype(int)


def gripper_turns(g: np.ndarray) -> np.ndarray:
    """Commands where the gripper starts, stops or reverses: never skipped (jog._gripper_turns_at)."""
    d = direction(np.diff(g))
    out = np.ones(len(g), bool)
    out[1:-1] = d[:-1] != d[1:]
    return out


def produce(t: np.ndarray, q: np.ndarray, hz: float, producer: str) -> tuple[np.ndarray, np.ndarray]:
    """Commands and their due times (s from the start) for one replayed segment. Column -1 of ``q`` is the gripper."""
    cmds, due = [q[0]], [0.0]
    if producer in ("ceil", "round"):
        tick = 0
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            n = max(1, int(np.ceil(dt * hz - 1e-8)) if producer == "ceil" else round(dt * hz))
            for k in range(1, n + 1):
                cmds.append(q[i - 1] + (q[i] - q[i - 1]) * k / n)
                due.append((tick + k) / hz)
            tick += n
        return np.array(cmds), np.array(due)
    assert producer == "clock"

    def tick_of(k: int) -> int:
        return int(np.floor((t[k] - t[0]) * hz + 0.5 + 1e-9))

    tick, last = 0, 0
    while last < len(t) - 1:
        i = last + 1
        gripping = direction(q[i, -1] - q[last, -1])
        while (
            i + 1 < len(t) and tick_of(i) <= tick_of(last) and direction(q[i + 1, -1] - q[i, -1]) == gripping
        ):
            i += 1  # merged: due on the same tick, the gripper going the same way
        n = max(1, tick_of(i) - tick_of(last))
        for k in range(1, n + 1):
            cmds.append(q[last] + (q[i] - q[last]) * k / n)
            due.append((tick + k) / hz)
        tick, last = tick + n, i
    return np.array(cmds), np.array(due)


def consume(
    cmds: np.ndarray, due: np.ndarray, intervals: np.ndarray, consumer: str
) -> tuple[np.ndarray, np.ndarray]:
    """The commands the motor loop sends and when, ticking at ``intervals`` (cycled)."""
    keep = gripper_turns(cmds[:, -1])
    sent, times, k, now, n = [], [], -1, 0.0, 0
    while k < len(cmds) - 1 and n < 100000:
        if due[k + 1] <= now + 1e-9:
            k += 1
            if consumer == "newest":
                arm = cmds[max(k - 1, 0), :-1]
                while (
                    k + 1 < len(cmds)
                    and due[k + 1] <= now + 1e-9
                    and not keep[k]
                    and np.abs(cmds[k + 1, :-1] - arm).max() <= PLAYBACK_LAG_DEG
                ):
                    k += 1
        if k >= 0:
            sent.append(cmds[k])
            times.append(now)
        now += intervals[n % len(intervals)]
        n += 1
    return np.array(sent), np.array(times)


def metrics(sent: np.ndarray, times: np.ndarray, demo_s: float) -> dict:
    arm = sent[:, :-1]
    step = np.abs(np.diff(arm, axis=0)).max(axis=1)
    dt = np.diff(times)
    speed = step / np.maximum(dt, 1e-6)
    accel = np.abs(np.diff(np.diff(arm, axis=0) / np.maximum(dt, 1e-6)[:, None], axis=0)).max(
        axis=1
    ) / np.maximum(dt[1:], 1e-6)
    return {
        "duration_s": round(float(times[-1] - times[0]), 3),
        "vs_demo": round(float((times[-1] - times[0]) / demo_s), 3),
        "max_step_deg": round(float(step.max()), 3),
        "p99_speed_deg_s": round(float(np.percentile(speed, 99)), 1),
        "p99_accel_deg_s2": round(float(np.percentile(accel, 99)), 0),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("demo", help="showservo_demo.npz (t, q_cmd)")
    p.add_argument("--from-t", type=float, default=12.616)
    p.add_argument("--to-t", type=float, default=17.425)
    p.add_argument(
        "--act-arm", help="an act's arm.npz: its measured motor tick intervals are one of the loops"
    )
    p.add_argument("--json")
    a = p.parse_args()
    d = np.load(a.demo)
    w = (d["t"] >= a.from_t) & (d["t"] <= a.to_t)
    t0, q0 = d["t"][w], d["q_cmd"][w]
    demo_s = float(t0[-1] - t0[0])
    loops = {
        "30 Hz": (30.0, np.array([1 / 30])),
        "20 Hz": (20.0, np.array([1 / 20])),
        "50 Hz": (50.0, np.array([1 / 50])),
    }
    if a.act_arm:
        arm = np.load(a.act_arm)
        gaps = np.diff(arm["t_mono"])
        loops["30 Hz, measured ticks"] = (30.0, gaps[(gaps > 0) & (gaps < 1.0)])
    rows = []
    for fps in (None, 15.0, 30.0, 60.0):
        t, q = resample(t0, q0, fps)
        for loop, (hz, intervals) in loops.items():
            for producer, consumer in (
                ("ceil", "one"),
                ("round", "one"),
                ("clock", "one"),
                ("clock", "newest"),
            ):
                cmds, due = produce(t, q, hz, producer)
                sent, times = consume(cmds, due, intervals, consumer)
                rows.append(
                    {
                        "demo_fps": "recorded" if fps is None else fps,
                        "loop": loop,
                        "approach": f"{producer}+{consumer}",
                        **metrics(sent, times, demo_s),
                    }
                )
    print(f"demo segment {a.from_t}-{a.to_t} s: {demo_s:.2f} s, {len(t0)} samples")
    print(
        f"{'demo fps':>9} {'loop':>22} {'approach':>13} {'duration':>9} {'vs demo':>8} {'max step':>9} {'p99 deg/s':>10}"
    )
    for r in rows:
        print(
            f"{str(r['demo_fps']):>9} {r['loop']:>22} {r['approach']:>13} {r['duration_s']:>8.2f}s {r['vs_demo']:>8.2f}"
            f" {r['max_step_deg']:>9.2f} {r['p99_speed_deg_s']:>10.1f}"
        )
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"demo_s": demo_s, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
