# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Host CPU/GPU utilization for a training run.

Answers one question the loss curve cannot: *is this run using the machine it
is paying for?* A run that is dataloader-bound looks identical to a healthy one
in every existing chart — same loss, same lr, just fewer steps per hour.

**Where this runs.** Sampling happens on the machine executing the training,
reached through :class:`~lerobot.gui.training.transport.TransportClient` like
every other host operation. That keeps the training image vanilla, as
``DESIGN.md § Polling`` requires: the container prints, the orchestrator
structures. It also keeps a remote run honest — sampling the GUI server's own
CPU for a run executing on a pod would produce a chart that is confidently
about the wrong computer.

The transport returns raw text; every parser here is pure, so the semantics are
tested without a GPU, a pod, or a training run.

## What the numbers mean, and what they do not

**CPU is a delta, never an instant.** ``/proc/stat`` reports cumulative jiffies
since boot; a single read says how busy the machine has been *since it booted*,
which is never what anyone wants. Utilization only exists between two reads, so
:func:`cpu_percent_between` takes both and the caller keeps the previous one.

**Device memory here is not the run's peak allocation.** The Metrics card
reports `torch.cuda.max_memory_allocated()`, which PyTorch can give because
PyTorch is the allocator: it is this process's tensor high-water mark. The
figure here is `nvidia-smi memory.used` — every process on the card, right now,
including the caching allocator's reserved pool and CUDA context that the
PyTorch number excludes. The device figure is always the larger one, and both
are correct about different questions.

That asymmetry is about vantage point, not capability. No framework can
self-report SM utilization, because a process cannot see how the driver
scheduled its kernels — which is why memory has a per-run number and busy does
not. From outside, `nvidia-smi pmon` reports per-process `sm%` and both become
attributable; see the note on container attribution below.

**Everything here is the host's, not the run's.** ``/proc/stat`` and
``nvidia-smi`` both report the whole machine, so a second training job, a
compile, or a browser counts toward these numbers. On a dedicated training box
that distinction is empty; on a shared one these are an upper bound on what the
run is using, and nothing here can tell the difference.

Attributing CPU to the run itself would mean reading the training container's
own cgroup (``cpu.stat``), which needs the container id — the recipe does not
name the container or write a cidfile today — and a cgroup path that differs
between docker's cgroupfs and systemd drivers. Worth doing; not done here, and
the UI says so rather than implying an attribution it cannot make.

**CPU is reported twice, and the second number is the useful one.** The
aggregate is normalized across cores — 100% means every core is busy, which is
what "saturated" has to mean when there are 32 of them. But a single-threaded
bottleneck — the classic being a dataloader pinning one core while the rest
idle — reads as 3% aggregate on a 32-core box, indistinguishable from an idle
machine. ``busiest_core_pct`` is what makes *some* thread being pinned visible.
Which thread it is, this cannot say.

**GPU "busy" is not GPU "saturated", and this module does not pretend
otherwise.** NVML defines ``utilization.gpu`` as the percent of time in the
sample period during which one or more kernels was executing. It is a
time-occupancy flag: one small kernel resident on 1 of 170 SMs for the whole
window reads 100% while the card is nearly idle. That is a well-documented trap
(arthurchiao.art/blog/understanding-gpu-performance), not an edge case.

True saturation wants SM-activity counters — DCGM's ``DCGM_FI_PROF_SM_ACTIVE``
— which are unavailable on the GeForce parts this repo is developed against.
So the honest proxy shipped alongside is **power draw as a fraction of the
board limit**: a genuinely compute-bound run pulls near its cap, and a run that
is merely *resident* on the GPU does not. Weights & Biases surfaces the same
quantity as ``gpu.N.powerPercent`` for the same reason. Both are recorded; the
UI labels them for what they are.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass, field

# One `nvidia-smi` call, CSV, no units, no header — the field order here is the
# field order parsed below. Kept as a list so the transport can pass it to
# either subprocess or ssh without re-quoting.
NVIDIA_SMI_QUERY = "index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit"
NVIDIA_SMI_ARGV = [
    "nvidia-smi",
    f"--query-gpu={NVIDIA_SMI_QUERY}",
    "--format=csv,noheader,nounits",
]

# Marks the boundary between the two payloads in one combined probe, so a
# single round trip serves both (an SSH hop per metric is the expensive part).
SECTION_SEPARATOR = "__LEROBOT_RESOURCE_SECTION__"

# Shortest interval that yields a meaningful CPU reading. Polls are not evenly
# spaced — two can land in the same second when a state change triggers an
# extra one — and a delta taken over 50 ms covers a handful of jiffies, so
# rounding alone swings it tens of percent. Measured on a real run: two polls
# 0.05 s apart reported 73.5% and then 86.6% for a workload that had not
# changed. Below this, the sample is dropped rather than charted as a spike.
MIN_SAMPLE_INTERVAL_S = 1.0

# One shell command producing both sections. `|| true` on the GPU half so a
# machine without a GPU still yields CPU numbers rather than a failed probe.
PROBE_COMMAND = f"cat /proc/stat; echo {SECTION_SEPARATOR}; {' '.join(NVIDIA_SMI_ARGV)} 2>/dev/null || true"


@dataclass(frozen=True)
class GpuSample:
    """One GPU at one instant.

    ``busy_pct`` is NVML's time-occupancy figure — see the module docstring for
    why it is not saturation. ``power_pct`` is the saturation proxy.
    """

    index: int
    name: str
    busy_pct: float
    memory_used_mb: float
    memory_total_mb: float
    power_w: float
    power_limit_w: float

    @property
    def memory_pct(self) -> float:
        if self.memory_total_mb <= 0:
            return 0.0
        return 100.0 * self.memory_used_mb / self.memory_total_mb

    @property
    def power_pct(self) -> float:
        if self.power_limit_w <= 0:
            return 0.0
        return 100.0 * self.power_w / self.power_limit_w


@dataclass(frozen=True)
class CpuTotals:
    """Cumulative jiffies from one ``/proc/stat`` read, and when it was taken.

    Meaningless alone — see :func:`cpu_percent_between`. ``per_core`` is indexed
    by core number; ``total`` is the kernel's own aggregate line. ``ts`` is
    carried so the next sample can tell whether enough time passed to be worth
    dividing by.
    """

    total: tuple[int, int]  # (busy, total) jiffies
    per_core: tuple[tuple[int, int], ...]
    ts: float = 0.0
    # Runnable tasks at the instant of the read (``procs_running``). This is
    # vmstat's ``r`` — the saturation half of the USE method, and not a delta:
    # it is a queue depth, meaningful from a single reading.
    runnable: int = 0


@dataclass(frozen=True)
class ResourceSample:
    """One poll's worth of host utilization, ready to serialize.

    ``cpu_pct`` is None on the first sample of a run: utilization is a delta and
    there is nothing yet to subtract from. The chart starts one poll late rather
    than opening on a fabricated zero.
    """

    ts: float
    cpu_pct: float | None
    busiest_core_pct: float | None
    cpu_cores: int
    # Every core's utilization over the interval, in core order. The whole
    # distribution rather than a summary of it: one bar per core is how htop,
    # mpstat and every other tool shows this, and it is the only form that
    # cannot be misread as a claim about *which* thread is hot.
    per_core_pct: list[float] = field(default_factory=list)
    # Runnable tasks vs cores. Over 1.0 means work is queued for a CPU that is
    # not free — CPU saturation proper, as distinct from utilization.
    runnable: int = 0
    gpus: list[GpuSample] = field(default_factory=list)

    def to_row(self) -> dict:
        """The ``resources.jsonl`` row. Percentages rounded — a chart cannot
        show more, and the file is read in full on every dashboard open."""
        return {
            "ts": round(self.ts, 3),
            "cpu_pct": None if self.cpu_pct is None else round(self.cpu_pct, 1),
            "busiest_core_pct": (None if self.busiest_core_pct is None else round(self.busiest_core_pct, 1)),
            "cpu_cores": self.cpu_cores,
            "per_core_pct": [round(v, 1) for v in self.per_core_pct],
            "runnable": self.runnable,
            "gpus": [
                {
                    **{k: v for k, v in asdict(g).items() if k != "name"},
                    "name": g.name,
                    "memory_pct": round(g.memory_pct, 1),
                    "power_pct": round(g.power_pct, 1),
                }
                for g in self.gpus
            ],
        }


def parse_proc_stat(text: str, *, ts: float = 0.0) -> CpuTotals | None:
    """Cumulative CPU jiffies from ``/proc/stat`` text.

    Pre: ``text`` is the file's contents (any platform's; non-Linux yields None).
    Post: on success ``per_core`` is ordered by core index and every entry has
    ``total > 0``.

    ``idle`` for the purposes of utilization is idle + iowait: a core waiting on
    disk is not doing work, and counting iowait as busy would make a
    dataloader-starved run look compute-bound — the exact confusion this chart
    exists to resolve.
    """
    total: tuple[int, int] | None = None
    cores: list[tuple[int, tuple[int, int]]] = []
    runnable = 0
    for line in text.splitlines():
        # The same file carries the run queue: `procs_running` is vmstat's `r`,
        # which is the CPU *saturation* metric — how many tasks want a CPU right
        # now, as opposed to how busy the CPUs have been. Not `/proc/loadavg`:
        # Linux load averages fold in uninterruptible I/O waiters, which is why
        # the USE checklist excludes them for CPU.
        if line.startswith("procs_running"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                runnable = int(parts[1])
            continue
        if not line.startswith("cpu"):
            continue
        parts = line.split()
        label = parts[0]
        try:
            values = [int(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(values) < 5:
            continue
        # user nice system idle iowait irq softirq steal ...
        idle = values[3] + values[4]
        busy = sum(values) - idle
        pair = (busy, sum(values))
        if label == "cpu":
            total = pair
        elif label[3:].isdigit():
            cores.append((int(label[3:]), pair))
    if total is None or total[1] <= 0:
        return None
    cores.sort()
    return CpuTotals(total=total, per_core=tuple(p for _, p in cores), ts=ts, runnable=runnable)


def _delta_percent(before: tuple[int, int], after: tuple[int, int]) -> float | None:
    """Busy fraction between two cumulative readings, as a percentage.

    None when the counters did not advance (same reading twice, or a counter
    reset): there is no interval to average over, and reporting 0% would claim
    the machine was idle when in fact nothing was measured.
    """
    busy = after[0] - before[0]
    total = after[1] - before[1]
    if total <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * busy / total))


def cpu_percent_between(
    before: CpuTotals, after: CpuTotals
) -> tuple[float | None, float | None, list[float]]:
    """``(aggregate_pct, busiest_core_pct, per_core_pct)`` over the interval.

    Pre: both readings come from the same machine, ``after`` taken later.
    Post: each value is either None (no interval elapsed) or in [0, 100].

    The aggregate is normalized across cores by construction — it is the
    kernel's own ``cpu`` line, whose total already sums every core — so 100%
    means every core was busy for the whole interval, which is the only
    definition of "saturated" that survives a machine with 32 of them.
    """
    aggregate = _delta_percent(before.total, after.total)
    per_core = [
        pct
        for b, a in zip(before.per_core, after.per_core, strict=False)
        if (pct := _delta_percent(b, a)) is not None
    ]
    return aggregate, (max(per_core) if per_core else None), per_core


def parse_nvidia_smi_csv(text: str) -> list[GpuSample]:
    """GPUs from ``nvidia-smi --format=csv,noheader,nounits`` output.

    Pre: ``text`` is that command's stdout, or empty on a machine without one.
    Post: one entry per parsed row, in the order reported; rows that do not
    parse are skipped rather than failing the sample — a chart missing one GPU
    beats a poll that raises.

    ``[N/A]`` appears for fields a card does not report (power on some laptop
    parts), and is read as zero, which the percentage properties then report as
    0% rather than dividing by it.
    """
    out: list[GpuSample] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue

        def num(raw: str) -> float:
            try:
                return float(raw)
            except ValueError:
                return 0.0  # "[N/A]" and friends

        try:
            index = int(parts[0])
        except ValueError:
            continue
        out.append(
            GpuSample(
                index=index,
                name=parts[1],
                busy_pct=num(parts[2]),
                memory_used_mb=num(parts[3]),
                memory_total_mb=num(parts[4]),
                power_w=num(parts[5]),
                power_limit_w=num(parts[6]),
            )
        )
    return out


def split_probe_output(text: str) -> tuple[str, str]:
    """``(proc_stat_text, nvidia_smi_text)`` from one :data:`PROBE_COMMAND` run.

    A machine with no GPU yields an empty second half rather than an error, so
    CPU charts still work on a CPU-only host.
    """
    head, _, tail = text.partition(SECTION_SEPARATOR)
    return head, tail


def run_probe_locally() -> str:
    """Run :data:`PROBE_COMMAND`'s work in-process on this machine.

    Pre: none. Post: text in :func:`split_probe_output`'s format; the GPU half
    is empty when ``nvidia-smi`` is absent or fails.

    Not a shell-out for the CPU half — reading the file directly avoids a
    process spawn on every poll of every run.
    """
    try:
        stat = open("/proc/stat").read()  # noqa: SIM115 — read once, closed by refcount
    except OSError:
        stat = ""
    gpu = ""
    if shutil.which("nvidia-smi"):
        try:
            gpu = subprocess.run(NVIDIA_SMI_ARGV, capture_output=True, text=True, timeout=5).stdout
        except (subprocess.SubprocessError, OSError):
            gpu = ""
    return f"{stat}\n{SECTION_SEPARATOR}\n{gpu}"


def sample_from_probe(
    text: str, *, previous: CpuTotals | None, now: float
) -> tuple[ResourceSample, CpuTotals | None]:
    """Turn one probe's raw text into a sample, plus the state for the next one.

    Pre: ``text`` is :func:`run_probe_locally`-shaped; ``previous`` is the
    ``CpuTotals`` returned by the last call for this run, or None to start.
    Post: returns ``(sample, totals)``; ``totals`` is what the caller must pass
    back as ``previous``. ``sample.cpu_pct`` is None exactly when no interval
    could be measured.
    """
    stat_text, gpu_text = split_probe_output(text)
    totals = parse_proc_stat(stat_text, ts=now)
    cpu_pct: float | None = None
    busiest: float | None = None
    per_core: list[float] = []
    if previous is not None and totals is not None:
        # Too short an interval is worse than no reading: it is a reading that
        # looks real. Keep the older baseline so the next poll measures from a
        # sensible distance rather than restarting the clock each time.
        if now - previous.ts >= MIN_SAMPLE_INTERVAL_S:
            cpu_pct, busiest, per_core = cpu_percent_between(previous, totals)
        else:
            totals = previous
    return (
        ResourceSample(
            ts=now,
            cpu_pct=cpu_pct,
            busiest_core_pct=busiest,
            cpu_cores=len(totals.per_core) if totals else 0,
            per_core_pct=per_core,
            runnable=totals.runnable if totals else 0,
            gpus=parse_nvidia_smi_csv(gpu_text),
        ),
        totals,
    )
