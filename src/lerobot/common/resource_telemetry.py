"""Resource utilization sampling for training runs.

Answers the one question step timing cannot: is the *machine* the constraint?
``updt_s``/``data_s`` already separate compute from dataloading, but neither
says whether the CPU is saturated, whether one core is pinned, or whether
another process owns the GPU.

The design and its rationale live in ``lerobot/gui/training/DESIGN.md``
(§ Resource telemetry). The parts that shape this module:

* **Utilization is not saturation.** Every "how busy" figure is paired with a
  saturation signal — run-queue depth for the CPU, power against the board
  limit for the GPU — because a resource can be 100% busy and not the limit,
  or below 100% and already the limit.
* **Sampling happens inside the training process**, which is the only place
  the attributable numbers exist: ``/proc/stat`` is not namespaced, so a
  container sees all host cores, while ``/sys/fs/cgroup/cpu.stat`` is the
  container's own.
* **Nothing here forks.** GPU counters are read in-process through NVML
  rather than by running ``nvidia-smi``. PyTorch's DataLoader installs a
  SIGCHLD handler to notice worker death, and a subprocess spawned from a
  sampler thread alongside it intermittently loses its reap: the child becomes
  a zombie, ``subprocess.run`` waits on it forever holding the sampler's lock,
  and the next drain hangs the training loop. Measured on the reference rig —
  a 60-step run wedged with an unreaped ``nvidia-smi`` and the main thread in
  ``futex_do_wait``.
* **Missing is never zero.** A resource that cannot be read omits its values
  and reports a status code, because a zero meaning "could not measure" is
  indistinguishable from a real idle.
* **Means are paired with maxima**, since a mean over a log window hides the
  stall the window exists to reveal.

Nothing here raises into the training loop: a sampler that breaks training to
report on training is worse than no sampler.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Status codes shared by every resource group, matching the design's three
# states. They are numeric because they ride the metric line, which carries
# only numbers.
STAT_MEASURED = 0
STAT_ABSENT = 1
STAT_UNREADABLE = 2

_PROC_STAT = "/proc/stat"
_CGROUP_CPU_STAT = "/sys/fs/cgroup/cpu.stat"
_CGROUP_CPU_MAX = "/sys/fs/cgroup/cpu.max"

_MILLIWATTS_PER_WATT = 1000.0


@dataclass
class _Window:
    """Running mean/max accumulator for one field over one log window."""

    total: float = 0.0
    count: int = 0
    peak: float = float("-inf")

    def add(self, value: float) -> None:
        self.total += value
        self.count += 1
        self.peak = max(self.peak, value)

    @property
    def mean(self) -> float:
        assert self.count > 0, "mean of an empty window"
        return self.total / self.count


@dataclass
class _Accumulator:
    """Every window in flight, plus the status of each resource group."""

    windows: dict[str, _Window] = field(default_factory=dict)
    cpu_stat: int = STAT_MEASURED
    gpu_stat: dict[int, int] = field(default_factory=dict)
    # Constants (core count, power limit, total memory) repeat on every line
    # rather than being hoisted, so a line is self-describing.
    constants: dict[str, float] = field(default_factory=dict)

    def add(self, key: str, value: float) -> None:
        self.windows.setdefault(key, _Window()).add(value)

    def merge(self, other: _Accumulator) -> None:
        """Fold one sample's readings into this window."""
        for key, window in other.windows.items():
            target = self.windows.setdefault(key, _Window())
            target.total += window.total
            target.count += window.count
            target.peak = max(target.peak, window.peak)
        self.constants.update(other.constants)
        self.cpu_stat = other.cpu_stat
        self.gpu_stat.update(other.gpu_stat)


def _read_first_line(path: str, prefix: str) -> str | None:
    try:
        with open(path) as handle:
            for line in handle:
                if line.startswith(prefix):
                    return line
    except OSError:
        return None
    return None


def _cpu_core_count() -> int | None:
    """Cores available to this container — the denominator for both CPU figures.

    Prefers the cgroup CPU quota (``docker run --cpus``), then the affinity
    mask (``--cpuset-cpus``), then the host count. The recipe sets neither
    today, so these coincide; the field's meaning must not depend on that.
    """
    try:
        with open(_CGROUP_CPU_MAX) as handle:
            quota_text, period_text = handle.read().split()
        if quota_text != "max":
            quota, period = int(quota_text), int(period_text)
            if period > 0 and quota > 0:
                return max(1, round(quota / period))
    except (OSError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count()


class ResourceSampler:
    """Samples host and container resource use on a background thread.

    Pre: ``start()`` is called at most once, before the first ``drain()``.
    Post: ``drain()`` returns only finite floats, and resets the window so the
    next call reports a disjoint interval.

    The caller drains once per log emission rather than per step: ``AverageMeter``
    has no window maximum and ``MetricsTracker`` routes every assignment through
    ``update()``, so a maximum written each step would be averaged away.
    """

    def __init__(self, interval_s: float = 2.0) -> None:
        assert interval_s > 0, f"interval must be positive, got {interval_s}"
        self._interval_s = interval_s
        self._lock = threading.Lock()
        self._acc = _Accumulator()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Deltas need a previous reading; utilization does not exist until the
        # second sample.
        self._prev_cpu: tuple[float, float] | None = None
        self._prev_cgroup: tuple[float, float] | None = None
        self._gpu_available = True
        # Devices seen at least once, so a sampler that starts failing can say
        # which device it lost rather than looking like a machine without one.
        self._known_devices: set[int] = set()
        self._nvml_module = None
        # Per-device cursor into NVML's process-utilization samples.
        self._nvml_last_ts: dict[int, int] = {}

    def start(self) -> None:
        assert self._thread is None, "sampler already started"
        self._thread = threading.Thread(target=self._run, name="resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s + 5.0)

    def __enter__(self) -> ResourceSampler:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def _run(self) -> None:
        # Prime the deltas immediately so the first drain has data even when the
        # first log window is short.
        self._sample_once()
        while not self._stop.wait(self._interval_s):
            self._sample_once()

    def _sample_once(self) -> None:
        # Sample into a private accumulator and merge under the lock, so the
        # lock is never held across a read. DataLoader forks worker processes
        # while this thread runs; a lock held for the duration of a sample is a
        # lock a forked child can inherit locked.
        scratch = _Accumulator()
        try:
            self._sample_cpu(scratch)
            self._sample_cgroup(scratch)
            self._sample_gpu(scratch)
        except Exception:
            # A sampler must never take training down with it.
            logger.exception("resource sampler iteration failed")
        with self._lock:
            self._acc.merge(scratch)

    # ── CPU ────────────────────────────────────────────────────────────────
    def _sample_cpu(self, acc: _Accumulator) -> None:
        line = _read_first_line(_PROC_STAT, "cpu ")
        if line is None:
            # /proc/stat is absent off Linux and unreadable if it exists but
            # cannot be opened; the reader cannot tell those apart, and the
            # actionable distinction for the card is the same.
            acc.cpu_stat = STAT_ABSENT if not os.path.exists(_PROC_STAT) else STAT_UNREADABLE
            return

        try:
            values = [float(field_text) for field_text in line.split()[1:]]
        except ValueError:
            acc.cpu_stat = STAT_UNREADABLE
            return
        if len(values) < 8:
            acc.cpu_stat = STAT_UNREADABLE
            return

        user, nice, system, idle, iowait, irq, softirq, steal = values[:8]
        # iowait is excluded from busy: a core waiting on I/O is not working,
        # and counting it is what makes load average misleading here.
        busy = user + nice + system + irq + softirq + steal
        total = busy + idle + iowait

        previous = self._prev_cpu
        self._prev_cpu = (busy, total)
        if previous is not None:
            busy_delta, total_delta = busy - previous[0], total - previous[1]
            if total_delta > 0:
                acc.add("cpu", 100.0 * busy_delta / total_delta)

        cores = _cpu_core_count()
        if cores:
            acc.constants["cores"] = float(cores)

        run_queue = _read_first_line(_PROC_STAT, "procs_running")
        if run_queue is not None:
            with contextlib.suppress(IndexError, ValueError):
                acc.add("rq", float(run_queue.split()[1]))
        acc.cpu_stat = STAT_MEASURED

    def _sample_cgroup(self, acc: _Accumulator) -> None:
        """This container's own CPU use, which /proc/stat cannot give.

        /proc/stat is not namespaced, so inside a container it reports the whole
        host. cpu.stat is the container's own at a fixed path under a cgroup
        namespace, which every shipped run has (training always launches via
        ``docker run``).
        """
        line = _read_first_line(_CGROUP_CPU_STAT, "usage_usec")
        if line is None:
            return
        try:
            usage_usec = float(line.split()[1])
        except (IndexError, ValueError):
            return

        now = time.monotonic()
        previous = self._prev_cgroup
        self._prev_cgroup = (usage_usec, now)
        if previous is None:
            return
        usage_delta = usage_usec - previous[0]
        elapsed_us = (now - previous[1]) * 1e6
        cores = acc.constants.get("cores") or _cpu_core_count()
        if elapsed_us > 0 and cores:
            acc.add("pcpu", 100.0 * usage_delta / elapsed_us / float(cores))

    # ── GPU ────────────────────────────────────────────────────────────────
    def _nvml(self):
        """The NVML module, initialized once, or None when there is no GPU.

        Post: on None, ``_gpu_available`` is False and no further attempt is
        made — a machine without a driver must not pay an import per interval.
        """
        if self._nvml_module is not None:
            return self._nvml_module
        try:
            import pynvml
        except ImportError:
            self._gpu_available = False
            return None
        try:
            pynvml.nvmlInit()
        except Exception:
            # No driver, or one this binding cannot talk to. Not a failure to
            # report — there is simply no device here.
            self._gpu_available = False
            return None
        self._nvml_module = pynvml
        return pynvml

    def _sample_gpu(self, acc: _Accumulator) -> None:
        if not self._gpu_available:
            return
        nvml = self._nvml()
        if nvml is None:
            return
        try:
            count = nvml.nvmlDeviceGetCount()
        except Exception:
            self._mark_gpu_unreadable(acc)
            return
        if count == 0:
            self._gpu_available = False
            return

        for index in range(count):
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
            except Exception:
                acc.gpu_stat[index] = STAT_UNREADABLE
                continue
            measured = self._sample_one_gpu(nvml, handle, index, acc)
            acc.gpu_stat[index] = STAT_MEASURED if measured else STAT_UNREADABLE
            if measured:
                self._known_devices.add(index)

    def _sample_one_gpu(self, nvml, handle, index: int, acc: _Accumulator) -> bool:
        """Read one device. Returns whether anything at all could be read.

        Each counter is read independently: a part that does not expose power
        must still report occupancy, rather than losing the whole device.
        """
        read_any = False
        with contextlib.suppress(Exception):
            rates = nvml.nvmlDeviceGetUtilizationRates(handle)
            acc.add(f"g{index}busy", float(rates.gpu))
            read_any = True
        with contextlib.suppress(Exception):
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            acc.add(f"g{index}mem", float(memory.used))
            acc.constants[f"g{index}memtot"] = float(memory.total)
            read_any = True
        with contextlib.suppress(Exception):
            acc.add(f"g{index}pw", nvml.nvmlDeviceGetPowerUsage(handle) / _MILLIWATTS_PER_WATT)
            read_any = True
        with contextlib.suppress(Exception):
            limit = nvml.nvmlDeviceGetEnforcedPowerLimit(handle) / _MILLIWATTS_PER_WATT
            acc.constants[f"g{index}pwlim"] = limit
        if self._sample_process_occupancy(nvml, handle, index, acc):
            read_any = True
        return read_any

    def _sample_process_occupancy(self, nvml, handle, index: int, acc: _Accumulator) -> bool:
        """This run's own share of the device, attributed by pid.

        NVML reports pids in the container's namespace, so the training process
        matches on its own pid without translation. Samples are consumed from
        the last seen timestamp so each reading covers only new activity;
        finding none means no sample in the interval, which is not a zero.
        """
        try:
            samples = nvml.nvmlDeviceGetProcessUtilization(handle, self._nvml_last_ts.get(index, 0))
        except Exception:
            # NVML_ERROR_NOT_FOUND is the ordinary "nothing since last time".
            return False
        ours = os.getpid()
        newest = self._nvml_last_ts.get(index, 0)
        occupancy = None
        for sample in samples or ():
            timestamp = getattr(sample, "timeStamp", 0)
            newest = max(newest, timestamp)
            if getattr(sample, "pid", None) == ours:
                # Several samples can fall in one interval; the peak is what
                # matters for "did this run keep the device busy".
                util = float(getattr(sample, "smUtil", 0))
                occupancy = util if occupancy is None else max(occupancy, util)
        self._nvml_last_ts[index] = newest
        if occupancy is None:
            return False
        acc.add(f"g{index}sm", occupancy)
        return True

    def _mark_gpu_unreadable(self, acc: _Accumulator) -> None:
        """A driver that is present but stopped answering.

        Distinct from having no GPU, and it must not render as one: a silent
        sampler failure would otherwise read as an idle device.
        """
        for index in self._known_devices or {0}:
            acc.gpu_stat[index] = STAT_UNREADABLE

    # ── Emission ───────────────────────────────────────────────────────────
    def drain(self) -> dict[str, float]:
        """Return this window's fields and start a new window.

        Post: every value is finite, so nothing here can be dropped by the log
        parser or poison a chart series.
        """
        with self._lock:
            acc, self._acc = self._acc, _Accumulator()
            # Status is a property of the sampler, not of the window, so it
            # survives the reset.
            self._acc.cpu_stat = acc.cpu_stat
            self._acc.gpu_stat = dict(acc.gpu_stat)

        fields: dict[str, float] = {}
        for key, window in acc.windows.items():
            if window.count == 0:
                continue
            fields[key] = window.mean
            # Saturation-relevant series carry a max; a mean alone hides the
            # spike the window exists to reveal.
            fields[f"{key}_max"] = window.peak
        fields.update(acc.constants)

        # 'rq' is a depth, not a rate: its max over the window is the signal
        # and its mean is noise.
        if "rq_max" in fields:
            fields["rq"] = fields.pop("rq_max")

        fields["cpu_stat"] = float(acc.cpu_stat)
        for index, status in acc.gpu_stat.items():
            fields[f"g{index}_stat"] = float(status)

        assert all(math.isfinite(value) for value in fields.values()), (
            f"non-finite telemetry field would be dropped by the parser: {fields}"
        )
        return fields


def format_with_resources(tracker: object, sampler: ResourceSampler | None) -> str:
    """Render the metric line with this window's resource fields appended.

    Pre: called once per log emission, on the process that emits — draining
    resets the window, so a second call reports an empty interval.
    Post: the line still carries every field the tracker renders, and each
    appended value round-trips through the GUI's log parser.

    Fields are appended rather than registered on the tracker because the field
    set varies with device count and status, while ``reduce_across_ranks`` zips
    the tracker's metrics with ``strict=True``.
    """
    line = str(tracker)
    if sampler is None:
        return line
    try:
        fields = sampler.drain()
    except Exception:
        # Telemetry must never cost a log line, which is the run's only record.
        logger.exception("resource telemetry drain failed")
        return line
    if not fields:
        return line
    # %g keeps large byte counts compact without emitting a trailing magnitude
    # letter, which the parser would read as a multiplier.
    return line + " " + " ".join(f"{key}:{value:.6g}" for key, value in sorted(fields.items()))
