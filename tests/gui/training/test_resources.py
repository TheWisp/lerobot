# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Host utilization sampling: the arithmetic, and the claims the UI makes.

The chart asserts that 100% means saturated. These tests are what make that
sentence true, so they are about the *semantics* of each number rather than the
plumbing: that CPU is a delta and not an instant, that the all-core average
cannot hide a pinned single core, that a GPU reporting "100% busy" is not the
same claim as "this GPU is working hard", and that a machine with no GPU still
charts its CPU.

Every parser is pure, so none of this needs a GPU, a pod, or a training run.
"""

from __future__ import annotations

from lerobot.gui.training.resources import (
    SECTION_SEPARATOR,
    cpu_percent_between,
    parse_nvidia_smi_csv,
    parse_proc_stat,
    sample_from_probe,
    split_probe_output,
)


def _stat(total: tuple[int, int, int], cores: list[tuple[int, int, int]]) -> str:
    """A /proc/stat body from (user, system, idle) triples, in jiffies."""
    lines = [f"cpu  {total[0]} 0 {total[1]} {total[2]} 0 0 0 0 0 0"]
    for i, (user, system, idle) in enumerate(cores):
        lines.append(f"cpu{i} {user} 0 {system} {idle} 0 0 0 0 0 0")
    lines.append("intr 12345")
    return "\n".join(lines) + "\n"


class TestCpuIsADeltaNotAnInstant:
    """`/proc/stat` counts jiffies since boot, so one read describes the machine's
    whole uptime. Utilization only exists between two reads."""

    def test_one_reading_alone_yields_no_percentage(self):
        sample, totals = sample_from_probe(
            _stat((500, 100, 400), [(500, 100, 400)]) + SECTION_SEPARATOR, previous=None, now=1.0
        )
        assert sample.cpu_pct is None
        assert totals is not None, "the reading must still be kept for the next interval"

    def test_the_second_reading_measures_the_interval_between_them(self):
        before = parse_proc_stat(_stat((100, 0, 100), [(100, 0, 100)]))
        # 75 busy jiffies out of 100 elapsed.
        after = parse_proc_stat(_stat((175, 0, 125), [(175, 0, 125)]))
        aggregate, _, _ = cpu_percent_between(before, after)
        assert aggregate == 75.0

    def test_counters_that_did_not_advance_report_nothing_rather_than_idle(self):
        """Two identical reads mean no interval was observed. Reporting 0% would
        claim the machine was idle when in fact nothing was measured."""
        same = parse_proc_stat(_stat((100, 0, 100), [(100, 0, 100)]))
        aggregate, busiest, _ = cpu_percent_between(same, same)
        assert aggregate is None
        assert busiest is None

    def test_iowait_counts_as_idle(self):
        """A core blocked on disk is not doing work. Counting iowait as busy
        would make a dataloader-starved run look compute-bound, which is the
        exact confusion this chart exists to resolve."""
        before = parse_proc_stat("cpu  100 0 0 100 0 0 0 0 0 0\ncpu0 100 0 0 100 0 0 0 0 0 0\n")
        # 100 jiffies elapsed, all of them in iowait.
        after = parse_proc_stat("cpu  100 0 0 100 100 0 0 0 0 0\ncpu0 100 0 0 100 100 0 0 0 0\n")
        aggregate, _, _ = cpu_percent_between(before, after)
        assert aggregate == 0.0


class TestOnePinnedCoreIsVisible:
    """The failure this chart is for: a single-threaded dataloader saturating one
    core while the GPU starves. On a many-core box the average cannot show it."""

    def test_the_all_core_average_alone_would_hide_it(self):
        # 100 jiffies pass for all 32 cores. Core 7 spends them working; the
        # other 31 spend them idle, which is what makes the aggregate small.
        before = parse_proc_stat(_stat((0, 0, 3200), [(0, 0, 100)] * 32))
        after_cores = [(0, 0, 200)] * 32
        after_cores[7] = (100, 0, 100)
        after = parse_proc_stat(_stat((100, 0, 6300), after_cores))

        aggregate, busiest, _ = cpu_percent_between(before, after)

        assert aggregate is not None and aggregate < 5, "one of 32 cores is a small average"
        assert busiest == 100.0, "the busiest-core line is what makes the bottleneck visible"

    def test_a_genuinely_saturated_machine_reads_100(self):
        before = parse_proc_stat(_stat((0, 0, 400), [(0, 0, 100)] * 4))
        after = parse_proc_stat(_stat((400, 0, 400), [(100, 0, 100)] * 4))
        aggregate, busiest, _ = cpu_percent_between(before, after)
        assert aggregate == 100.0
        assert busiest == 100.0


NVIDIA_SMI_REAL = "0, NVIDIA GeForce RTX 5090, 97, 21197, 32607, 448.31, 575.00\n"


class TestGpuBusyIsNotGpuSaturated:
    """NVML defines utilization.gpu as the share of time at least one kernel was
    resident. A tiny kernel on one multiprocessor reads 100% while the card is
    nearly idle, so the UI charts power against the board limit beside it."""

    def test_power_distinguishes_a_working_card_from_a_merely_occupied_one(self):
        working = parse_nvidia_smi_csv(NVIDIA_SMI_REAL)[0]
        occupied = parse_nvidia_smi_csv("0, NVIDIA GeForce RTX 5090, 100, 900, 32607, 34.10, 575.00\n")[0]

        # Indistinguishable on the metric everyone watches...
        assert working.busy_pct >= 97 and occupied.busy_pct == 100
        # ...and unambiguous on the one that reflects work done.
        assert working.power_pct > 75
        assert occupied.power_pct < 10

    def test_percentages_are_derived_from_the_board_limit_not_a_constant(self):
        gpu = parse_nvidia_smi_csv(NVIDIA_SMI_REAL)[0]
        assert round(gpu.power_pct, 1) == 78.0  # 448.31 / 575.00
        assert round(gpu.memory_pct, 1) == 65.0  # 21197 / 32607

    def test_a_card_that_reports_no_power_does_not_divide_by_it(self):
        """Some parts report [N/A]. That must read as 0%, not crash the poll."""
        gpu = parse_nvidia_smi_csv("0, Some GPU, 50, 100, 200, [N/A], [N/A]\n")[0]
        assert gpu.power_pct == 0.0
        assert gpu.memory_pct == 50.0

    def test_every_gpu_is_kept_separately(self):
        gpus = parse_nvidia_smi_csv("0, A, 100, 10, 100, 300, 300\n1, B, 0, 1, 100, 10, 300\n")
        assert [g.index for g in gpus] == [0, 1]
        assert gpus[0].power_pct == 100.0 and gpus[1].power_pct < 5, (
            "averaging these would hide an idle card beside a busy one"
        )

    def test_an_unparsable_row_does_not_cost_the_others(self):
        gpus = parse_nvidia_smi_csv("garbage\n0, A, 100, 10, 100, 300, 300\n")
        assert len(gpus) == 1


class TestTheProbeSurvivesTheMachineItLandsOn:
    def test_a_host_with_no_gpu_still_reports_cpu(self):
        text = _stat((100, 0, 100), [(100, 0, 100)]) + SECTION_SEPARATOR + "\n"
        first, totals = sample_from_probe(text, previous=None, now=1.0)
        second, _ = sample_from_probe(
            _stat((175, 0, 125), [(175, 0, 125)]) + SECTION_SEPARATOR + "\n",
            previous=totals,
            now=2.0,
        )
        assert first.gpus == [] and second.gpus == []
        assert second.cpu_pct == 75.0

    def test_a_probe_that_returned_nothing_is_not_charted_as_zero(self):
        sample, totals = sample_from_probe("", previous=None, now=1.0)
        assert totals is None
        assert sample.cpu_pct is None and sample.gpus == []

    def test_the_two_halves_are_split_on_the_sentinel(self):
        stat, gpu = split_probe_output(f"cpu 1 2 3\n{SECTION_SEPARATOR}\n0, A, 1, 2, 3, 4, 5\n")
        assert stat.strip() == "cpu 1 2 3"
        assert "0, A" in gpu

    def test_the_serialized_row_carries_what_the_chart_reads(self):
        text = _stat((0, 0, 400), [(0, 0, 100)] * 4) + SECTION_SEPARATOR + "\n" + NVIDIA_SMI_REAL
        _, totals = sample_from_probe(text, previous=None, now=1.0)
        busy = _stat((400, 0, 400), [(100, 0, 100)] * 4) + SECTION_SEPARATOR + "\n" + NVIDIA_SMI_REAL
        sample, _ = sample_from_probe(busy, previous=totals, now=2.0)

        row = sample.to_row()
        assert row["cpu_pct"] == 100.0
        assert row["busiest_core_pct"] == 100.0
        assert row["cpu_cores"] == 4
        assert row["gpus"][0]["power_pct"] == 78.0
        assert row["gpus"][0]["name"] == "NVIDIA GeForce RTX 5090"
        assert row["ts"] == 2.0


class TestAnIntervalTooShortToMeasure:
    """Polls are not evenly spaced: a state change can trigger one moments after
    the last. A CPU delta over 50 ms covers a handful of jiffies, so rounding
    alone swings it tens of percent — and it charts as a spike that looks like a
    real event. Found on a real run, where two polls 0.05 s apart reported 73.5%
    and then 86.6% for a workload that had not changed.
    """

    def test_a_sample_taken_too_soon_reports_no_cpu_reading(self):
        first = _stat((100, 0, 100), [(100, 0, 100)]) + SECTION_SEPARATOR + "\n"
        _, totals = sample_from_probe(first, previous=None, now=100.0)

        soon = _stat((101, 0, 100), [(101, 0, 100)]) + SECTION_SEPARATOR + "\n"
        sample, _ = sample_from_probe(soon, previous=totals, now=100.05)

        assert sample.cpu_pct is None
        assert sample.busiest_core_pct is None

    def test_the_older_baseline_is_kept_so_the_next_poll_still_measures(self):
        """Restarting the clock on every dropped sample would mean a burst of
        rapid polls left the series with no readings at all."""
        first = _stat((100, 0, 100), [(100, 0, 100)]) + SECTION_SEPARATOR + "\n"
        _, totals = sample_from_probe(first, previous=None, now=100.0)

        soon = _stat((101, 0, 100), [(101, 0, 100)]) + SECTION_SEPARATOR + "\n"
        _, after_short = sample_from_probe(soon, previous=totals, now=100.05)
        assert after_short is not None and after_short.ts == 100.0, (
            "the dropped sample must not become the new baseline"
        )

        # 75 busy of 100 elapsed jiffies, measured from the original baseline.
        later = _stat((175, 0, 125), [(175, 0, 125)]) + SECTION_SEPARATOR + "\n"
        sample, _ = sample_from_probe(later, previous=after_short, now=103.0)
        assert sample.cpu_pct == 75.0

    def test_a_gpu_reading_still_lands_when_the_cpu_interval_is_too_short(self):
        """GPU figures are instantaneous, not deltas, so they are unaffected."""
        first = _stat((100, 0, 100), [(100, 0, 100)]) + SECTION_SEPARATOR + "\n"
        _, totals = sample_from_probe(first, previous=None, now=100.0)
        soon = _stat((101, 0, 100), [(101, 0, 100)]) + SECTION_SEPARATOR + "\n" + NVIDIA_SMI_REAL
        sample, _ = sample_from_probe(soon, previous=totals, now=100.05)

        assert sample.cpu_pct is None
        assert sample.gpus and sample.gpus[0].power_pct > 75


class TestSaturationIsNotUtilization:
    """Gregg's USE checklist treats CPU the way this module treats the GPU: how
    busy a resource has been is a different question from whether work is queued
    behind it. `procs_running` is vmstat's `r`, the saturation half."""

    def test_the_run_queue_is_read_from_the_same_file(self):
        text = (
            _stat((100, 0, 100), [(100, 0, 100)] * 4)
            + "procs_running 9\nprocs_blocked 2\n"
            + SECTION_SEPARATOR
        )
        totals = parse_proc_stat(text)
        assert totals is not None and totals.runnable == 9

    def test_a_queue_deeper_than_the_cores_is_what_oversubscribed_looks_like(self):
        """Four cores, nine tasks wanting one: fully utilized *and* saturated,
        which utilization alone cannot distinguish from merely fully utilized."""
        before = parse_proc_stat(_stat((0, 0, 400), [(0, 0, 100)] * 4) + "procs_running 1\n", ts=1.0)
        after = parse_proc_stat(_stat((400, 0, 400), [(100, 0, 100)] * 4) + "procs_running 9\n", ts=3.0)
        aggregate, _, _ = cpu_percent_between(before, after)
        assert aggregate == 100.0
        assert after.runnable > len(after.per_core)

    def test_the_whole_per_core_distribution_survives_to_the_row(self):
        """The UI draws a bar per core. A summary could not show the shape, and a
        'busiest core' number invites reading the hot core as this run's thread."""
        before = parse_proc_stat(_stat((0, 0, 400), [(0, 0, 100)] * 4))
        after_cores = [(0, 0, 200)] * 4
        after_cores[2] = (100, 0, 100)
        after = parse_proc_stat(_stat((100, 0, 700), after_cores))

        _, busiest, per_core = cpu_percent_between(before, after)

        assert per_core == [0.0, 0.0, 100.0, 0.0]
        assert busiest == 100.0
