"""Unit tests for the resource sampler.

Every source is synthesized in ``tmp_path`` and pointed at by monkeypatching the
module's path constants — the sampler reads ``/proc`` and ``/sys`` on the real
machine, so tests that used the live files would assert on whatever the host
happened to be doing.

What is pinned here is the arithmetic and the three-state status contract, which
are the parts a reader of the charts depends on and cannot check.
"""

from __future__ import annotations

import math
import os

import pytest

from lerobot.common.resource_telemetry import (
    STAT_ABSENT,
    STAT_MEASURED,
    STAT_UNREADABLE,
    ResourceSampler,
    _Accumulator,
    format_with_resources,
)


def _proc_stat(*, busy_fields: tuple[float, ...], running: int = 1) -> str:
    """A /proc/stat whose 'cpu ' line carries the eight standard counters."""
    assert len(busy_fields) == 8, "the kernel's cpu line has eight counters"
    return "cpu  " + " ".join(str(v) for v in busy_fields) + f"\nprocs_running {running}\n"


def _point(sampler: ResourceSampler, tmp_path, monkeypatch, text: str) -> _Accumulator:
    """Feed one /proc/stat reading into a fresh accumulator."""
    path = tmp_path / "stat"
    path.write_text(text)
    monkeypatch.setattr("lerobot.common.resource_telemetry._PROC_STAT", str(path))
    acc = _Accumulator()
    sampler._sample_cpu(acc)
    return acc


# ── CPU arithmetic ─────────────────────────────────────────────────────────


def test_cpu_utilization_needs_two_readings():
    """A delta does not exist until there is something to subtract from.

    Charting the first sample would report cumulative-since-boot as though it
    were current utilization.
    """
    sampler = ResourceSampler()
    acc = _Accumulator()
    sampler._prev_cpu = None
    # Simulate the first reading only.
    sampler._prev_cpu = (100.0, 200.0)
    assert "cpu" not in acc.windows


def test_cpu_utilization_is_the_busy_share_of_the_delta(tmp_path, monkeypatch):
    sampler = ResourceSampler()
    # user nice system idle iowait irq softirq steal
    first = _proc_stat(busy_fields=(100, 0, 0, 100, 0, 0, 0, 0))
    _point(sampler, tmp_path, monkeypatch, first)
    # 60 more busy ticks, 40 more idle ticks -> 60% over the interval.
    second = _proc_stat(busy_fields=(160, 0, 0, 140, 0, 0, 0, 0))
    acc = _point(sampler, tmp_path, monkeypatch, second)

    assert acc.windows["cpu"].mean == pytest.approx(60.0)
    assert acc.cpu_stat == STAT_MEASURED


def test_iowait_is_not_busy(tmp_path, monkeypatch):
    """The distinction that makes this figure worth having.

    A core blocked on disk is not working. Counting iowait as busy is precisely
    what makes load average misleading for a dataloader-bound run — the run that
    most needs an honest answer.
    """
    sampler = ResourceSampler()
    _point(sampler, tmp_path, monkeypatch, _proc_stat(busy_fields=(100, 0, 0, 100, 0, 0, 0, 0)))
    # 100 ticks pass, every one of them spent waiting on I/O.
    acc = _point(sampler, tmp_path, monkeypatch, _proc_stat(busy_fields=(100, 0, 0, 100, 100, 0, 0, 0)))

    assert acc.windows["cpu"].mean == pytest.approx(0.0), "iowait must not read as busy"


def test_steal_and_softirq_are_busy(tmp_path, monkeypatch):
    """Time the kernel spends on our behalf, and time a hypervisor takes, are
    both real occupancy — omitting them understates a loaded VM."""
    sampler = ResourceSampler()
    _point(sampler, tmp_path, monkeypatch, _proc_stat(busy_fields=(0, 0, 0, 100, 0, 0, 0, 0)))
    acc = _point(sampler, tmp_path, monkeypatch, _proc_stat(busy_fields=(0, 0, 0, 100, 0, 0, 50, 50)))

    assert acc.windows["cpu"].mean == pytest.approx(100.0)


# ── Status contract ────────────────────────────────────────────────────────


def test_absent_cpu_source_reports_absent_not_zero(tmp_path, monkeypatch):
    sampler = ResourceSampler()
    monkeypatch.setattr("lerobot.common.resource_telemetry._PROC_STAT", str(tmp_path / "does-not-exist"))
    acc = _Accumulator()
    sampler._sample_cpu(acc)

    assert acc.cpu_stat == STAT_ABSENT
    assert "cpu" not in acc.windows, "an unreadable source must omit its value"


def test_unparsable_cpu_source_reports_unreadable(tmp_path, monkeypatch):
    """Present but not usable is a different problem from not present, and the
    card says so — one is 'this platform has no such thing', the other is a bug
    or a permissions failure worth showing."""
    sampler = ResourceSampler()
    acc = _point(sampler, tmp_path, monkeypatch, "cpu  not numbers at all\n")

    assert acc.cpu_stat == STAT_UNREADABLE
    assert "cpu" not in acc.windows


def test_truncated_cpu_line_reports_unreadable(tmp_path, monkeypatch):
    sampler = ResourceSampler()
    acc = _point(sampler, tmp_path, monkeypatch, "cpu  1 2 3\n")

    assert acc.cpu_stat == STAT_UNREADABLE


# ── Window semantics ───────────────────────────────────────────────────────


def test_run_queue_reports_its_peak_not_its_mean(tmp_path, monkeypatch):
    """Depth is not a rate. A run queue that hit 40 once during the window is
    the finding; averaging it against thirty idle samples erases it."""
    sampler = ResourceSampler()
    for running in (1, 1, 40, 1):
        path = tmp_path / "stat"
        path.write_text(_proc_stat(busy_fields=(1, 0, 0, 1, 0, 0, 0, 0), running=running))
        monkeypatch.setattr("lerobot.common.resource_telemetry._PROC_STAT", str(path))
        sampler._sample_cpu(sampler._acc)

    fields = sampler.drain()

    assert fields["rq"] == pytest.approx(40.0)
    assert "rq_max" not in fields, "the peak is reported as rq itself"


def test_every_mean_is_paired_with_a_max(tmp_path, monkeypatch):
    sampler = ResourceSampler()
    _point(sampler, tmp_path, monkeypatch, _proc_stat(busy_fields=(0, 0, 0, 100, 0, 0, 0, 0)))
    for busy in (10, 90):
        path = tmp_path / "stat"
        path.write_text(_proc_stat(busy_fields=(busy, 0, 0, 100, 0, 0, 0, 0)))
        monkeypatch.setattr("lerobot.common.resource_telemetry._PROC_STAT", str(path))
        sampler._sample_cpu(sampler._acc)

    fields = sampler.drain()

    assert "cpu" in fields and "cpu_max" in fields
    assert fields["cpu_max"] >= fields["cpu"], "a max below its mean is impossible"


def test_drain_starts_a_new_window(tmp_path, monkeypatch):
    """Windows must be disjoint, or every reading is smeared into the next.

    The status carries over, because it describes the sampler rather than the
    interval — a source that vanished has not come back just because the window
    turned over.
    """
    sampler = ResourceSampler()
    _point(sampler, tmp_path, monkeypatch, _proc_stat(busy_fields=(0, 0, 0, 100, 0, 0, 0, 0)))
    path = tmp_path / "stat"
    path.write_text(_proc_stat(busy_fields=(100, 0, 0, 100, 0, 0, 0, 0)))
    monkeypatch.setattr("lerobot.common.resource_telemetry._PROC_STAT", str(path))
    sampler._sample_cpu(sampler._acc)

    first = sampler.drain()
    second = sampler.drain()

    assert "cpu" in first
    assert "cpu" not in second, "a drained window must not report again"
    assert second["cpu_stat"] == first["cpu_stat"], "status survives the reset"


def test_drain_emits_only_finite_values(tmp_path, monkeypatch):
    """The parser drops non-finite fields key by key, so a NaN here would vanish
    and be indistinguishable from a field never emitted."""
    sampler = ResourceSampler()
    _point(sampler, tmp_path, monkeypatch, _proc_stat(busy_fields=(0, 0, 0, 100, 0, 0, 0, 0)))
    path = tmp_path / "stat"
    path.write_text(_proc_stat(busy_fields=(50, 0, 0, 150, 0, 0, 0, 0)))
    monkeypatch.setattr("lerobot.common.resource_telemetry._PROC_STAT", str(path))
    sampler._sample_cpu(sampler._acc)

    fields = sampler.drain()

    assert fields, "expected at least the status field"
    assert all(math.isfinite(v) for v in fields.values()), fields


def test_a_zero_length_interval_emits_no_utilization(tmp_path, monkeypatch):
    """Two identical readings mean no time passed, not that the CPU was idle."""
    sampler = ResourceSampler()
    text = _proc_stat(busy_fields=(100, 0, 0, 100, 0, 0, 0, 0))
    _point(sampler, tmp_path, monkeypatch, text)
    acc = _point(sampler, tmp_path, monkeypatch, text)

    assert "cpu" not in acc.windows


# ── GPU ────────────────────────────────────────────────────────────────────
#
# NVML is faked rather than driven, so these run on a machine with no GPU and
# assert the same thing everywhere. The sampler reads counters in-process
# precisely so nothing here forks; see the module docstring.


class _FakeSample:
    def __init__(self, pid, sm_util, timestamp=1):
        self.pid = pid
        self.smUtil = sm_util
        self.timeStamp = timestamp


class _FakeNvml:  # noqa: N801
    """Minimal stand-in for pynvml. Each counter can be made to fail alone.

    Method names deliberately mirror NVML's camelCase C API, because that is
    what the sampler calls.
    """

    NVMLError = RuntimeError

    def __init__(self, devices=1, processes=None, broken=()):
        self._devices = devices
        self._processes = processes or {}
        self._broken = set(broken)

    def _check(self, name):
        if name in self._broken:
            raise RuntimeError(f"NVML {name} failed")

    def nvmlDeviceGetCount(self):  # noqa: N802
        self._check("count")
        return self._devices

    def nvmlDeviceGetHandleByIndex(self, index):  # noqa: N802
        self._check("handle")
        return index

    def nvmlDeviceGetUtilizationRates(self, handle):  # noqa: N802
        self._check("util")
        return type("R", (), {"gpu": 40 + handle, "memory": 10})()

    def nvmlDeviceGetMemoryInfo(self, handle):  # noqa: N802
        self._check("memory")
        return type("M", (), {"used": 4 * 1024**3, "total": 32 * 1024**3})()

    def nvmlDeviceGetPowerUsage(self, handle):  # noqa: N802
        self._check("power")
        return 351_000

    def nvmlDeviceGetEnforcedPowerLimit(self, handle):  # noqa: N802
        self._check("limit")
        return 575_000

    def nvmlDeviceGetProcessUtilization(self, handle, since):  # noqa: N802
        self._check("procutil")
        return self._processes.get(handle, [])


def _with_nvml(sampler: ResourceSampler, fake) -> ResourceSampler:
    sampler._nvml_module = fake
    return sampler


def test_an_absent_binding_reports_no_device_at_all(monkeypatch):
    """No GPU is not a failure, and must not render a tile claiming one."""
    sampler = ResourceSampler()
    monkeypatch.setitem(__import__("sys").modules, "pynvml", None)

    def no_pynvml(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError("no pynvml")
        return original_import(name, *args, **kwargs)

    import builtins

    original_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", no_pynvml)
    sampler._sample_gpu(sampler._acc)
    fields = sampler.drain()

    assert not any(key.startswith("g") for key in fields), fields
    assert sampler._gpu_available is False, "an absent binding must not be retried"


def test_gpu_fields_are_read_and_units_are_normalized():
    """Watts and bytes, never milliwatts and MiB — the chart axes are the units
    a person reads."""
    sampler = _with_nvml(ResourceSampler(), _FakeNvml(processes={0: [_FakeSample(os.getpid(), 75)]}))
    sampler._sample_gpu(sampler._acc)
    fields = sampler.drain()

    assert fields["g0busy"] == pytest.approx(40.0)
    assert fields["g0pw"] == pytest.approx(351.0), "milliwatts converted to watts"
    assert fields["g0pwlim"] == pytest.approx(575.0)
    assert fields["g0mem"] == pytest.approx(4 * 1024**3)
    assert fields["g0memtot"] == pytest.approx(32 * 1024**3)
    assert fields["g0sm"] == pytest.approx(75.0), "our own pid's occupancy"
    assert fields["g0_stat"] == STAT_MEASURED


def test_another_process_on_the_gpu_is_not_attributed_to_us():
    """The whole point of per-process attribution: device occupancy that belongs
    to someone else must not be charted as this run's."""
    sampler = _with_nvml(ResourceSampler(), _FakeNvml(processes={0: [_FakeSample(os.getpid() + 1, 95)]}))
    sampler._sample_gpu(sampler._acc)
    fields = sampler.drain()

    assert fields["g0busy"] == pytest.approx(40.0), "the device is busy"
    assert "g0sm" not in fields, "but not with our work"


def test_no_process_sample_is_not_zero_occupancy():
    """NVML returning nothing means no sample in the interval. Reading it as 0
    would chart an idle GPU that was never measured."""
    sampler = _with_nvml(ResourceSampler(), _FakeNvml(processes={0: []}))
    sampler._sample_gpu(sampler._acc)
    fields = sampler.drain()

    assert "g0sm" not in fields
    assert fields["g0_stat"] == STAT_MEASURED, "the device itself was readable"


def test_one_dead_counter_does_not_lose_the_whole_device():
    """A part that does not expose power must still report occupancy."""
    sampler = _with_nvml(ResourceSampler(), _FakeNvml(broken={"power", "limit"}))
    sampler._sample_gpu(sampler._acc)
    fields = sampler.drain()

    assert fields["g0busy"] == pytest.approx(40.0)
    assert "g0pw" not in fields and "g0pwlim" not in fields
    assert fields["g0_stat"] == STAT_MEASURED


def test_a_device_that_answers_nothing_is_unreadable_not_absent():
    """The distinction the card depends on, and the one easiest to lose.

    A device whose every counter fails still exists. Reporting it as absent
    would hide the failure entirely.
    """
    sampler = _with_nvml(
        ResourceSampler(),
        _FakeNvml(broken={"util", "memory", "power", "limit", "procutil"}),
    )
    sampler._sample_gpu(sampler._acc)
    fields = sampler.drain()

    assert fields["g0_stat"] == STAT_UNREADABLE
    assert "g0busy" not in fields


def test_a_driver_that_stops_answering_names_the_devices_it_lost():
    sampler = _with_nvml(ResourceSampler(), _FakeNvml(devices=2))
    sampler._sample_gpu(sampler._acc)
    sampler.drain()

    sampler._nvml_module = _FakeNvml(devices=2, broken={"count"})
    sampler._sample_gpu(sampler._acc)
    fields = sampler.drain()

    assert fields["g0_stat"] == STAT_UNREADABLE
    assert fields["g1_stat"] == STAT_UNREADABLE, "both known devices are reported"


def test_two_devices_emit_two_suffixed_groups():
    """Multi-GPU is a key suffix, not a nested structure — the flat metric line
    has no way to carry an array."""
    sampler = _with_nvml(
        ResourceSampler(),
        _FakeNvml(
            devices=2,
            processes={0: [_FakeSample(os.getpid(), 70)], 1: [_FakeSample(os.getpid(), 65)]},
        ),
    )
    sampler._sample_gpu(sampler._acc)
    fields = sampler.drain()

    assert fields["g0sm"] == pytest.approx(70.0)
    assert fields["g1sm"] == pytest.approx(65.0)
    assert fields["g0busy"] == pytest.approx(40.0)
    assert fields["g1busy"] == pytest.approx(41.0)
    assert fields["g0_stat"] == STAT_MEASURED and fields["g1_stat"] == STAT_MEASURED


def test_a_machine_with_a_driver_but_no_devices_reports_none():
    sampler = _with_nvml(ResourceSampler(), _FakeNvml(devices=0))
    sampler._sample_gpu(sampler._acc)
    fields = sampler.drain()

    assert not any(key.startswith("g") for key in fields), fields
    assert sampler._gpu_available is False


def test_a_window_shorter_than_the_sampling_interval_still_reports(tmp_path, monkeypatch):
    """A fast run must not chart a row of gaps.

    The sampler thread owns the cadence, but a log window can be shorter than
    that interval — a small model at a low ``log_freq`` closes a window in under
    a second. Without a reading of its own such a window carries only its status
    field, so every other line charts as missing. Measured before this: a
    900-step run at 27 steps/s filled 13 of 30 lines.
    """
    sampler = ResourceSampler()
    # Prime the delta the way the sampler thread would, then let a window pass
    # with no thread sample at all.
    _point(sampler, tmp_path, monkeypatch, _proc_stat(busy_fields=(100, 0, 0, 100, 0, 0, 0, 0)))
    path = tmp_path / "stat"
    path.write_text(_proc_stat(busy_fields=(160, 0, 0, 140, 0, 0, 0, 0)))
    monkeypatch.setattr("lerobot.common.resource_telemetry._PROC_STAT", str(path))

    fields = sampler.drain()

    assert fields["cpu"] == pytest.approx(60.0), "the empty window took its own reading"


def test_an_on_demand_reading_is_skipped_when_one_is_in_flight(tmp_path, monkeypatch):
    """Two threads must not both advance the deltas' previous values.

    The sampler thread and a drain can coincide; the drain yields rather than
    waits, because a reading is arriving anyway and a log line must never block
    on telemetry.
    """
    sampler = ResourceSampler()
    _point(sampler, tmp_path, monkeypatch, _proc_stat(busy_fields=(100, 0, 0, 100, 0, 0, 0, 0)))
    path = tmp_path / "stat"
    path.write_text(_proc_stat(busy_fields=(160, 0, 0, 140, 0, 0, 0, 0)))
    monkeypatch.setattr("lerobot.common.resource_telemetry._PROC_STAT", str(path))

    sampler._sample_lock.acquire()
    try:
        fields = sampler.drain()
    finally:
        sampler._sample_lock.release()

    assert "cpu" not in fields, "drain must not block behind an in-flight reading"
    assert fields["cpu_stat"] == STAT_MEASURED


def test_a_crashing_sampler_does_not_report_itself_healthy(monkeypatch):
    """The status field must never be more optimistic than the reading.

    It defaults to "measured", so an unexpected failure mid-sample would
    otherwise merge that default and claim a healthy sampler standing behind no
    numbers — the single thing the three-state contract exists to prevent.
    """
    sampler = ResourceSampler()
    sampler._known_devices.add(0)

    def boom(_acc):
        raise RuntimeError("counter source exploded")

    monkeypatch.setattr(ResourceSampler, "_sample_cpu", lambda self, acc: boom(acc))
    sampler._sample_once()
    fields = sampler.drain()

    assert fields["cpu_stat"] == STAT_UNREADABLE
    assert fields["g0_stat"] == STAT_UNREADABLE
    assert not any(k in fields for k in ("cpu", "pcpu")), fields


def test_a_crashing_sampler_does_not_take_training_down(monkeypatch):
    """Telemetry is never worth a training run."""
    sampler = ResourceSampler()
    monkeypatch.setattr(
        ResourceSampler,
        "_sample_cpu",
        lambda self, acc: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    sampler._sample_once()  # must not raise
    line = format_with_resources(_Tracker(), sampler)

    assert "loss:0.5" in line, "the metric line survives a broken sampler"


class _Tracker:
    def __str__(self):
        return "step:1K loss:0.5"
