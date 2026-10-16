"""Deterministic queue/latency experiment using a recorded FLASH insertion trace.

This is a joint-command surrogate, not an IK or contact simulation. It compares
the scheduling consequences of three retarget designs while the motor consumes
one command per tick. The old queue remains available during each simulated
inference job. The six-command case reproduces the timing mistake, not its old
path checker. Run with an act.json and its adjacent arm.npz; no robot is touched.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

HZ = 30.0
PERIOD = 1 / HZ
EVENTS_S = (0.7, 1.5, 2.4, 3.3, 4.0)
# Small, reversible target-pose corrections in joint-equivalent degrees. They
# are illustrative; IK and collision feasibility must be checked separately.
PERTURBATIONS = np.array(
    [
        [0.0, 0.8, -0.3, 0.2, 0.0, 0.0],
        [0.0, -0.6, 0.2, -0.1, 0.0, 0.0],
        [0.0, 1.0, -0.4, 0.2, 0.0, 0.0],
        [0.0, -0.9, 0.3, -0.2, 0.0, 0.0],
        [0.0, 0.7, -0.2, 0.1, 0.0, 0.0],
    ]
)


@dataclass
class Action:
    phase: int
    q: np.ndarray


def load_trace(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    act = json.loads((root / "act.json").read_text())
    insertion = [x for x in act["targets"] if x["step"] == "key frame 7"]
    by_phase = {int(x["phase"]): x["joints"] for x in insertion}
    phases = sorted(by_phase)
    if phases != list(range(phases[0], phases[-1] + 1)):
        raise ValueError("insertion trace has missing phase indices")
    raw = np.asarray([by_phase[i][:6] for i in phases], dtype=float)
    # The logged last command per demo phase omits interpolated subcommands.
    # Smooth just that sampling artifact for a representative 30 Hz reference.
    baseline = savgol_filter(raw, 9, 3, axis=0)
    solve_ms = np.asarray(act["stream_diagnostics"][6]["solve_ms"], dtype=float)
    arm = np.load(root / "arm.npz")
    low, high = insertion[0]["t"], insertion[-1]["t"]
    mask = (arm["t"] >= low) & (arm["t"] <= high)
    read_ms = np.asarray(arm["read_ms"][mask], dtype=float)
    command_ms = np.asarray(arm["command_ms"][mask], dtype=float)
    observed_intervals = np.diff(np.asarray(arm["t_mono"][mask], dtype=float))
    if len(read_ms) < len(baseline) or len(observed_intervals) < 10:
        raise ValueError("arm trace does not cover the insertion")
    return baseline, solve_ms, read_ms + command_ms, observed_intervals


def step_latency_s(solve_ms: np.ndarray, event: int) -> float:
    # Stratify measured per-sample costs to exercise an ordinary, slow, and
    # worst observed solve without inventing a computation-time distribution.
    percentile = (50, 95, 100, 75, 90)[event]
    return float(np.percentile(solve_ms, percentile) / 1000)


def chunk_latency_s(solve_ms: np.ndarray, event: int, horizon: int) -> float:
    # Sum contiguous measured solves. The actual producer also checks each
    # generated joint chord, so this is an optimistic chunk-compute estimate.
    if len(solve_ms) <= horizon:
        return float(np.sum(solve_ms) / 1000)
    start = ((event + 1) * 37) % (len(solve_ms) - horizon)
    return float(np.sum(solve_ms[start : start + horizon]) / 1000)


def smoothstep(u: float) -> float:
    u = float(np.clip(u, 0, 1))
    return u * u * (3 - 2 * u)


def run(
    baseline: np.ndarray,
    intervals: np.ndarray,
    solve_ms: np.ndarray,
    mode: str,
    perturbation_scale: float,
) -> dict:
    queue = [Action(i, q.copy()) for i, q in enumerate(baseline)]
    sent: list[np.ndarray] = []
    sent_phases: list[int] = []
    sent_times: list[float] = []
    pending = None
    offset = np.zeros(6)
    event = 0
    t = 0.0
    first_response: dict[int, float] = {}
    awaiting_response: dict[int, dict[int, np.ndarray]] = {}
    retarget_compute_s = 0.0
    refill_compute_s = 0.0
    rejected = 0
    starved_ticks = 0
    refill_phase = None
    refill_ready = 0.0
    solve_cursor = 0
    ticks = 0
    while (queue or refill_phase is not None or pending is not None) and ticks < 1000:
        while (
            pending is None
            and refill_phase is not None
            and refill_ready <= t
            and refill_phase < len(baseline)
        ):
            queue.append(Action(refill_phase, baseline[refill_phase] + offset))
            refill_phase += 1
            if refill_phase < len(baseline):
                next_solve = float(solve_ms[solve_cursor % len(solve_ms)] / 1000)
                refill_ready += next_solve
                refill_compute_s += next_solve
                solve_cursor += 1
        if refill_phase == len(baseline):
            refill_phase = None
        if event < len(EVENTS_S) and t >= EVENTS_S[event] and pending is None:
            correction = PERTURBATIONS[event] * perturbation_scale
            horizon = min(8 if mode == "timed_chunk_short" else 30, len(queue))
            latency = (
                chunk_latency_s(solve_ms, event, horizon)
                if mode.startswith("timed_chunk")
                else step_latency_s(solve_ms, event)
            )
            retarget_compute_s += latency
            snapshot = [Action(item.phase, item.q.copy()) for item in queue]
            pending = (event, t + latency, latency, correction, snapshot, horizon)
            event += 1
        if pending is not None and t >= pending[1]:
            event_id, _ready_at, latency, correction, snapshot, horizon = pending
            consumed = len(snapshot) - len(queue)
            if mode.startswith("timed_chunk"):
                # Prefix actions expected to execute during computation are
                # unchanged. Generate a smooth correction in existing slots,
                # then discard the slots actually consumed before the swap.
                freeze = max(1, math.ceil(latency / PERIOD))
                candidate = [
                    Action(item.phase, item.q + correction * smoothstep((k - freeze + 1) / 6))
                    for k, item in enumerate(snapshot[:horizon])
                ]
                replacement = candidate[consumed:]
                if replacement and [x.phase for x in replacement] == [
                    x.phase for x in queue[: len(replacement)]
                ]:
                    before = sent[-1] if sent else baseline[0]
                    first = replacement[0].q
                    incoming = before - (sent[-2] if len(sent) > 1 else before)
                    outgoing = first - before
                    if np.max(np.abs(outgoing)) <= 2 and np.max(np.abs(outgoing - incoming)) <= 2:
                        queue = replacement
                        refill_phase = replacement[-1].phase + 1
                        if refill_phase < len(baseline):
                            next_solve = float(solve_ms[solve_cursor % len(solve_ms)] / 1000)
                            refill_ready = t + next_solve
                            refill_compute_s += next_solve
                            solve_cursor += 1
                    else:
                        rejected += 1
                else:
                    rejected += 1
            elif mode == "six_ticks":
                if queue:
                    next_action = queue[0]
                    before = sent[-1] if sent else baseline[0]
                    after = baseline[next_action.phase] + offset + correction
                    inserted = [
                        Action(next_action.phase, before + (k / 6) * (after - before)) for k in range(1, 7)
                    ]
                    queue = inserted
                    refill_phase = next_action.phase + 1
                    if refill_phase < len(baseline):
                        next_solve = float(solve_ms[solve_cursor % len(solve_ms)] / 1000)
                        refill_ready = t + next_solve
                        refill_compute_s += next_solve
                        solve_cursor += 1
            elif mode == "one_tick":
                if queue:
                    next_phase = queue[0].phase
                    queue = [Action(next_phase, baseline[next_phase] + offset + correction)]
                    refill_phase = next_phase + 1
                    if refill_phase < len(baseline):
                        next_solve = float(solve_ms[solve_cursor % len(solve_ms)] / 1000)
                        refill_ready = t + next_solve
                        refill_compute_s += next_solve
                        solve_cursor += 1
            else:
                raise ValueError(mode)
            offset += correction
            pending = None
            awaiting_response[event_id] = {item.phase: item.q for item in snapshot}
        if queue:
            item = queue.pop(0)
            sent.append(item.q)
            sent_phases.append(item.phase)
            sent_times.append(t)
            for event_id, before in list(awaiting_response.items()):
                if item.phase in before and np.max(np.abs(item.q - before[item.phase])) > 0.01:
                    first_response[event_id] = t
                    del awaiting_response[event_id]
        elif refill_phase is not None:
            sent.append(sent[-1].copy())
            sent_phases.append(sent_phases[-1])
            sent_times.append(t)
            starved_ticks += 1
        t += float(intervals[ticks % len(intervals)])
        ticks += 1
    q = np.asarray(sent)
    v = np.diff(q, axis=0)
    a = np.diff(q, n=2, axis=0)
    return {
        "method": mode,
        "commands": len(sent),
        "duration_s": round(sent_times[-1] - sent_times[0], 3),
        "retarget_compute_s": round(retarget_compute_s, 3),
        "refill_compute_s": round(refill_compute_s, 3),
        "response_ms": [round(1000 * (first_response[i] - EVENTS_S[i])) for i in sorted(first_response)],
        "rejected_splices": rejected,
        "max_joint_step_deg": round(float(np.max(np.abs(v))), 3),
        "p95_joint_step_deg": round(float(np.percentile(np.max(np.abs(v), axis=1), 95)), 3),
        "max_joint_accel_deg_per_tick2": round(float(np.max(np.abs(a))), 3),
        "motor_gaps_over_50ms": int(np.sum(np.diff(sent_times) > 0.05)),
        "repeated_demo_slots": len(sent_phases) - len(set(sent_phases)),
        "starved_motor_ticks": starved_ticks,
        "times": np.asarray(sent_times),
        "joints": q,
        "phases": np.asarray(sent_phases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("act_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--perturbation-scale", type=float, default=1.0)
    args = parser.parse_args()
    baseline, solves, bus_work_ms, observed_intervals = load_trace(args.act_dir)
    # Same measured encoder and command work, two scheduling policies. The old
    # loop waits a full period after each write; the fixed loop budgets the read
    # within the period and sends late only when work exceeds it.
    raw_old = PERIOD + bus_work_ms / 1000
    old = raw_old + max(0.0, float(np.mean(observed_intervals) - np.mean(raw_old)))
    paced = np.maximum(PERIOD, bus_work_ms / 1000)
    methods = ("six_ticks", "one_tick", "timed_chunk", "timed_chunk_short")
    results = [
        run(baseline, cadence, solves, method, args.perturbation_scale)
        for cadence in (old, paced)
        for method in methods
    ]
    for result, cadence_name in zip(
        results, ("post_write",) * len(methods) + ("fixed_rate",) * len(methods), strict=True
    ):
        result["cadence"] = cadence_name
    args.output.mkdir(parents=True, exist_ok=True)
    summary = [
        {k: v for k, v in result.items() if k not in {"times", "joints", "phases"}} for result in results
    ]
    (args.output / "queue_simulation.json").write_text(
        json.dumps(
            {
                "source": str(args.act_dir),
                "baseline_phases": len(baseline),
                "perturbation_scale": args.perturbation_scale,
                "measured_motor_interval_ms": round(float(1000 * np.mean(observed_intervals)), 2),
                "measured_read_command_ms": round(float(np.mean(bus_work_ms)), 2),
                "assumptions": "Joint-command surrogate; 5 synthetic target shifts; no physics, IK feasibility or collision check.",
                "results": summary,
            },
            indent=2,
        )
        + "\n"
    )
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for result in results:
        if result["cadence"] != "fixed_rate":
            continue
        label = {
            "six_ticks": "6 added ticks",
            "one_tick": "one-tick correction",
            "timed_chunk": "30-slot chunk",
            "timed_chunk_short": "8-slot chunk",
        }[result["method"]]
        axes[0].plot(result["times"], result["joints"][:, 1], label=label)
        v = np.diff(result["joints"][:, 1]) / np.diff(result["times"])
        axes[1].plot(result["times"][1:], v, label=label, alpha=0.8)
    for axis in axes:
        for at in EVENTS_S:
            axis.axvline(at, color="gray", lw=0.7, ls="--", alpha=0.5)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left")
    axes[0].set_ylabel("shoulder lift command (deg)")
    axes[1].set_ylabel("command velocity (deg/s)")
    axes[1].set_xlabel("simulated time (s)")
    fig.suptitle("Insertion replay surrogate: five retargets, measured solve and motor costs")
    fig.tight_layout()
    fig.savefig(args.output / "queue_simulation.png", dpi=150)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
