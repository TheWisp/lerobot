# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the lerobot-train stdout parser (progress + auto metrics)."""

from __future__ import annotations

import math

import pytest

from lerobot.common.training_log import (
    TRAINING_LOG_JSON_MARKER,
    TrainingHealthTracker,
    format_training_log_record,
)
from lerobot.gui.training.log_parse import (
    ProgressSample,
    parse_metric_sample,
    parse_progress,
)

# ── Progress (tqdm bar) ──────────────────────────────────────────────────────


def test_progress_full_bar():
    line = "Training:   1%|▏         | 125/10000 [02:02<2:36:10,  1.05step/s]"
    s = parse_progress(line)
    assert s == ProgressSample(step=125, total_steps=10000, eta_seconds=2 * 3600 + 36 * 60 + 10)


def test_progress_mm_ss_eta():
    s = parse_progress("Training:  50%|##### | 5/10 [00:03<00:03,  1.5step/s]")
    assert s.step == 5 and s.total_steps == 10
    assert s.eta_seconds == 3


def test_progress_eta_unknown_is_none():
    # tqdm prints '?' before it can estimate.
    s = parse_progress("Training:   0%| | 0/10000 [00:00<?, ?step/s]")
    assert s is not None
    assert s.step == 0
    assert s.total_steps == 10000
    assert s.eta_seconds is None


def test_progress_structured_record():
    record = format_training_log_record(
        step=600,
        total_steps=50_000,
        eta_seconds=11_954.8,
        loss=0.5179,
    )
    line = f"2026-07-24 06:34:18,471 [INFO] readable text | {record}"

    assert parse_progress(line) == ProgressSample(step=600, total_steps=50_000, eta_seconds=11_954.8)


def test_progress_legacy_hvla_step_record():
    line = (
        "2026-07-24 06:34:18,471 [INFO] "
        "step 600/50000 | loss: 0.5179 | flow_loss: 0.5179 | lr: 1.5e-05 | 242ms"
    )

    assert parse_progress(line) == ProgressSample(step=600, total_steps=50_000, eta_seconds=11_954.8)


@pytest.mark.parametrize(
    "line",
    [
        "INFO some unrelated log line",
        "step:100 loss:0.5",  # metric line, not a tqdm bar
        "",
        "Training: starting…",  # no counts
    ],
)
def test_progress_non_match_returns_none(line):
    assert parse_progress(line) is None


# ── Metrics (auto-captured bag) ──────────────────────────────────────────────


def test_metric_sample_core_fields():
    line = "INFO 2026-06-17 step:1000 smpl:32K loss:0.0343 grdn:1.234 lr:1.0e-05 updt_s:0.123"
    bag = parse_metric_sample(line)
    assert bag["step"] == 1000
    assert bag["loss"] == pytest.approx(0.0343)
    assert bag["grdn"] == pytest.approx(1.234)
    assert bag["lr"] == pytest.approx(1e-5)
    assert bag["updt_s"] == pytest.approx(0.123)


def test_metric_sample_is_auto_not_fixed():
    # A brand-new metric lerobot might add must be captured with no code change.
    line = "step:500 loss:0.2 some_new_metric:42.0 another_one:3"
    bag = parse_metric_sample(line)
    assert bag["some_new_metric"] == 42.0
    assert bag["another_one"] == 3.0


def test_metric_sample_magnitude_suffix():
    bag = parse_metric_sample("step:2000 smpl:1.5M loss:0.1")
    assert bag["smpl"] == pytest.approx(1.5e6)


def test_metric_sample_real_lerobot_line():
    # The exact shape lerobot's MetricsTracker.__str__ emits: step/smpl/ep via
    # format_big_number (K/M/B/T/Q suffixes), epch float, then the AverageMeters
    # (loss/grdn ':.3f', lr ':0.1e'). Locks the parser to the real format.
    line = "INFO 2026-06-17 12:00:00 step:10K smpl:320K ep:1K epch:0.50 loss:0.034 grdn:1.234 lr:1.0e-05 updt_s:0.123 data_s:0.001"
    bag = parse_metric_sample(line)
    assert bag["step"] == pytest.approx(10_000)
    assert bag["smpl"] == pytest.approx(320_000)
    assert bag["ep"] == pytest.approx(1_000)
    assert bag["epch"] == pytest.approx(0.50)
    assert bag["loss"] == pytest.approx(0.034)
    assert bag["grdn"] == pytest.approx(1.234)
    assert bag["lr"] == pytest.approx(1e-5)


def test_metric_sample_captures_lerobot_throughput_name():
    bag = parse_metric_sample("step:100 loss:0.2 updt_s:0.12 data_s:0.01 smp/s:61")

    assert bag["smp/s"] == pytest.approx(61)


def test_metric_sample_structured_record():
    record = format_training_log_record(
        step=600,
        total_steps=50_000,
        loss=0.5179,
        flow_loss=0.5179,
        grdn=1.234,
        lr=1.5e-05,
        updt_s=0.2,
        data_s=0.042,
        samples_per_s=66.1,
        mem_gb=7.8,
        step_time_ms=242.0,
    )
    bag = parse_metric_sample(f"2026-07-24 06:34:18,471 [INFO] readable text | {record}")

    assert bag == {
        "step": 600.0,
        "total_steps": 50_000.0,
        "loss": pytest.approx(0.5179),
        "flow_loss": pytest.approx(0.5179),
        "grdn": pytest.approx(1.234),
        "lr": pytest.approx(1.5e-05),
        "updt_s": pytest.approx(0.2),
        "data_s": pytest.approx(0.042),
        "samples_per_s": pytest.approx(66.1),
        "mem_gb": pytest.approx(7.8),
        "step_time_ms": 242.0,
    }


def test_custom_trainer_health_record_round_trips_through_parser():
    timestamps = iter((10.0, 12.0))
    tracker = TrainingHealthTracker(
        batch_size=8,
        total_steps=100,
        clock=lambda: next(timestamps),
        peak_memory_gb=lambda: 4.5,
    )
    tracker.step()

    sample = tracker.sample(step=1, values={"loss": 0.25, "grdn": 1.5, "lr": 1e-5})
    bag = parse_metric_sample(sample.record)

    assert bag is not None
    assert bag["step"] == 1
    assert bag["total_steps"] == 100
    assert bag["loss"] == pytest.approx(0.25)
    assert bag["grdn"] == pytest.approx(1.5)
    assert bag["samples_per_s"] == pytest.approx(4.0)
    assert bag["mem_gb"] == pytest.approx(4.5)


def test_metric_sample_legacy_hvla_step_record():
    line = (
        "2026-07-24 06:34:18,471 [INFO] "
        "step 600/50000 | loss: 0.5179 | flow_loss: 0.5179 | lr: 1.5e-05 | 242ms"
    )
    bag = parse_metric_sample(line)

    assert bag == {
        "step": 600.0,
        "loss": pytest.approx(0.5179),
        "flow_loss": pytest.approx(0.5179),
        "lr": pytest.approx(1.5e-05),
    }


@pytest.mark.parametrize(
    "payload",
    [
        '{"version":2,"step":1,"total_steps":10,"loss":0.5}',
        '{"version":true,"step":1,"total_steps":10,"loss":0.5}',
        '{"version":1,"step":-1,"total_steps":10,"loss":0.5}',
        '{"version":1,"step":1.5,"total_steps":10,"loss":0.5}',
        '{"version":1,"step":1,"total_steps":0,"loss":0.5}',
        '{"version":1,"step":1,"total_steps":10,"eta_seconds":-1,"loss":0.5}',
        '{"version":1,"total_steps":10,"loss":0.5}',
        '{"version":1,"step":1,"loss":0.5}',
        '{"version":1,"step":1,"total_steps":10,"loss":0.5} trailing',
        "not-json",
    ],
)
def test_structured_record_rejects_unknown_or_invalid_payload(payload):
    line = f"2026-07-24 06:34:18,471 [INFO] step 7/10 | loss: 0.5 | LEROBOT_TRAINING_JSON:{payload}"
    assert parse_progress(line) is None
    assert parse_metric_sample(line) is None


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"step": -1, "total_steps": 10}, ValueError),
        ({"step": 1.5, "total_steps": 10}, ValueError),
        ({"step": 1, "total_steps": 0}, ValueError),
        ({"step": 1, "total_steps": 10, "loss": float("inf")}, ValueError),
        ({"step": 1, "total_steps": 10, "loss": "bad"}, TypeError),
    ],
)
def test_structured_record_writer_rejects_invalid_values(kwargs, error):
    with pytest.raises(error):
        format_training_log_record(**kwargs)


def test_metric_sample_billion_suffix_not_dropped():
    # format_big_number uses 'B' for billion (not 'G') — a step/smpl in the
    # billions must scale, not silently drop.
    bag = parse_metric_sample("step:2B loss:0.1")
    assert bag["step"] == pytest.approx(2e9)


def test_metric_sample_skips_non_numeric_values():
    bag = parse_metric_sample("level:INFO tag:run_a step:10 loss:0.9")
    assert "level" not in bag and "tag" not in bag
    assert bag["step"] == 10 and bag["loss"] == 0.9


def test_metric_sample_requires_loss():
    # The tqdm bar and other lines carry step but no loss → not a metric sample.
    assert parse_metric_sample("step:100 lr:1e-5 grdn:0.5") is None
    assert parse_metric_sample("INFO checkpoint saved at step:100") is None


def test_metric_sample_all_values_finite_floats():
    bag = parse_metric_sample("step:1 loss:0.5 lr:1e-3")
    assert bag is not None
    assert all(isinstance(v, float) and math.isfinite(v) for v in bag.values())


def test_metric_and_progress_are_disjoint_on_their_lines():
    # The tqdm line is progress-only; the metric line is metric-only.
    tqdm = "Training:  10%| | 1000/10000 [00:30<04:30, 33.0step/s]"
    metric = "step:1000 loss:0.04 lr:1e-5"
    assert parse_progress(tqdm) is not None and parse_metric_sample(tqdm) is None
    assert parse_metric_sample(metric) is not None and parse_progress(metric) is None


def test_metric_sample_ignores_logging_prefix_and_glued_tqdm_bar():
    # The shape REAL lerobot emits: tqdm holds the line open with \r and the
    # logging handler appends the metric record, so the bar, the logging
    # prefix (LEVEL date time file.py:lineno), and the metrics all land on one
    # physical line. The parser must pull ONLY the metrics out of that noise —
    # not "py:611" from the file:lineno, not "Training:39" from the bar.
    line = (
        "Training:  39%|███▉      | 1156/3000 [00:17<00:24, 73.83step/s]"
        "INFO 2026-06-17 11:55:12 lerobot_train.py:611 step:1K smpl:9K ep:75 epch:0.36 "
        "loss:1.596 grdn:58.077 lr:1.0e-05 updt_s:0.013 data_s:0.000"
    )
    bag = parse_metric_sample(line)
    assert bag is not None
    assert "py" not in bag, f"logging file.py:lineno leaked: {bag}"
    assert "Training" not in bag, f"tqdm percent leaked: {bag}"
    assert bag["loss"] == pytest.approx(1.596)
    assert bag["grdn"] == pytest.approx(58.077)
    assert bag["lr"] == pytest.approx(1.0e-5)
    # Only the real metric keys survive.
    assert set(bag) == {"step", "smpl", "ep", "epch", "loss", "grdn", "lr", "updt_s", "data_s"}


# ── Future-proofing: parse lerobot's OWN MetricsTracker output ───────────────
#
# These tests drive the *upstream* formatter (MetricsTracker / AverageMeter /
# format_big_number) rather than a hand-typed string. If lerobot renames a
# metric (e.g. ``grdn``), changes a format spec, or swaps the magnitude
# suffixes, these break — which is exactly the alert we want, because the
# parser would otherwise silently start dropping fields on a `lerobot` bump.


def _train_tracker(initial_step: int):
    """Build a MetricsTracker exactly as ``lerobot_train.py`` does."""
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

    metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    # batch_size, num_frames, num_episodes — arbitrary but realistic.
    return MetricsTracker(8, 50_000, 100, metrics, initial_step=initial_step)


def test_parses_real_metricstracker_output():
    pytest.importorskip("lerobot.utils.logging_utils")
    # Values chosen to be exact at the trackers' format specs (loss/grdn/_s
    # are ':.3f', lr is ':0.1e') so the round-trip is lossless — we're testing
    # the parser, not float-formatting rounding.
    t = _train_tracker(initial_step=1000)
    t.loss = 0.034
    t.grad_norm = 1.234
    t.lr = 1.0e-5
    t.update_s = 0.123
    t.dataloading_s = 0.002

    line = str(t)  # the exact string lerobot logs
    bag = parse_metric_sample(line)
    assert bag is not None, f"parser failed on real lerobot line: {line!r}"
    # Display names are the AverageMeter names, not the dict keys.
    assert bag["loss"] == pytest.approx(0.034)
    assert bag["grdn"] == pytest.approx(1.234)
    assert bag["lr"] == pytest.approx(1.0e-5)
    assert bag["updt_s"] == pytest.approx(0.123)
    assert bag["data_s"] == pytest.approx(0.002)


def test_parses_real_output_with_big_number_step():
    # Drive step into the K/M range so the parser is exercised against the
    # actual format_big_number output (not a hand-written "10K").
    pytest.importorskip("lerobot.utils.logging_utils")
    from lerobot.utils.utils import format_big_number

    t = _train_tracker(initial_step=12_500)
    t.loss = 0.5
    line = str(t)
    bag = parse_metric_sample(line)
    # step is rendered via format_big_number; whatever suffix it uses, the
    # parser must recover the real magnitude (±1 unit of rounding).
    assert bag["step"] == pytest.approx(12_500, rel=0.05)
    # Sanity: confirm the line really did use a magnitude suffix (else this
    # test isn't proving anything about suffix handling).
    assert any(c in format_big_number(12_500) for c in "KMBTQ")


# ── Orchestrator ingest: stderr.log → progress.json + metrics.jsonl ──────────


def test_ingest_writes_progress_and_metrics(tmp_path):
    """End-to-end: a real-shaped stderr.log is parsed into progress.json
    (position) and metrics.jsonl (series) by the orchestrator's poll-path
    ingest — the path that makes real runs show data."""
    from lerobot.gui.training.hosts import HostRegistry
    from lerobot.gui.training.orchestrator import Orchestrator
    from lerobot.gui.training.runs import RunPaths, RunRegistry
    from lerobot.gui.training.transport import SubprocessClient, SubprocessTransport

    rr = RunRegistry(runs_dir=tmp_path / "runs")
    orch = Orchestrator(HostRegistry(hosts=[]), rr)
    paths = RunPaths.for_run("r1", rr.runs_dir)
    paths.ensure_exists()
    paths.stderr_log.write_text(
        "INFO starting\n"
        "Training:   1%| | 100/10000 [00:05<08:15, 20.0step/s]\n"
        "step:100 smpl:800 loss:0.500 grdn:2.0 lr:1.0e-04\n"
        "Training:   2%| | 250/10000 [00:12<07:50, 20.0step/s]\n"
        "step:250 smpl:2K loss:0.300 grdn:1.5 lr:1.0e-04\n"
    )
    client = SubprocessClient(SubprocessTransport(workdir=paths.root))

    orch._ingest_training_log(client, paths)

    prog = orch._read_progress(client, paths.progress_json)
    assert prog["step"] == 250  # freshest (metric line beats earlier tqdm)
    assert prog["total_steps"] == 10000
    assert prog["eta_seconds"] == 7 * 60 + 50

    series = orch._read_metrics(paths.metrics_jsonl)
    assert [s["step"] for s in series] == [100, 250]
    assert series[-1]["loss"] == pytest.approx(0.3)
    assert series[-1]["grdn"] == pytest.approx(1.5)


@pytest.mark.parametrize("structured", [True, False], ids=["structured", "legacy"])
def test_ingest_hvla_progress_and_metrics(tmp_path, structured):
    from lerobot.gui.training.hosts import HostRegistry
    from lerobot.gui.training.orchestrator import Orchestrator
    from lerobot.gui.training.runs import RunPaths, RunRegistry
    from lerobot.gui.training.transport import SubprocessClient, SubprocessTransport

    rr = RunRegistry(runs_dir=tmp_path / "runs")
    orch = Orchestrator(HostRegistry(hosts=[]), rr)
    paths = RunPaths.for_run("hvla", rr.runs_dir)
    paths.ensure_exists()

    def line(step, loss):
        readable = (
            f"2026-07-24 06:34:18,471 [INFO] step {step}/50000 | "
            f"loss: {loss:.4f} | flow_loss: {loss:.4f} | lr: 1.5e-05 | 242ms"
        )
        if not structured:
            return readable
        record = format_training_log_record(
            step=step,
            total_steps=50_000,
            loss=loss,
            flow_loss=loss,
            lr=1.5e-05,
            step_time_ms=242.0,
        )
        return f"{readable} | {record}"

    paths.stderr_log.write_text(line(500, 0.4977) + "\n" + line(600, 0.5179) + "\n")
    client = SubprocessClient(SubprocessTransport(workdir=paths.root))

    orch._ingest_training_log(client, paths)

    progress = orch._read_progress(client, paths.progress_json)
    assert progress["step"] == 600
    assert progress["total_steps"] == 50_000
    series = orch._read_metrics(paths.metrics_jsonl)
    assert [sample["step"] for sample in series] == [500, 600]
    assert series[-1]["loss"] == pytest.approx(0.5179)


def test_ingest_glued_real_lerobot_lines(tmp_path):
    """The real lerobot stderr.log interleaves the tqdm bar and the metric log
    on ONE physical line. Ingest must still capture metrics (an earlier
    progress-then-continue dropped them all), and stamp each metric with the
    PRECISE step from the bar — the metric line's own step is coarsely
    formatted (``format_big_number`` renders 1156 as "1K")."""
    from lerobot.gui.training.hosts import HostRegistry
    from lerobot.gui.training.orchestrator import Orchestrator
    from lerobot.gui.training.runs import RunPaths, RunRegistry
    from lerobot.gui.training.transport import SubprocessClient, SubprocessTransport

    rr = RunRegistry(runs_dir=tmp_path / "runs")
    orch = Orchestrator(HostRegistry(hosts=[]), rr)
    paths = RunPaths.for_run("r1", rr.runs_dir)
    paths.ensure_exists()
    # Two glued lines, exactly as captured from a live act/pusht run.
    paths.stderr_log.write_text(
        "Training:  39%|###9      | 1156/3000 [00:17<00:24, 73.83step/s]"
        "INFO 2026-06-17 11:55:12 lerobot_train.py:611 step:1K smpl:9K ep:75 epch:0.36 "
        "loss:1.596 grdn:58.077 lr:1.0e-05 updt_s:0.013 data_s:0.000\n"
        "Training:  79%|#######9  | 2378/3000 [00:33<00:08, 73.0step/s]"
        "INFO 2026-06-17 11:55:29 lerobot_train.py:611 step:2K smpl:19K ep:154 epch:0.74 "
        "loss:0.421 grdn:12.300 lr:1.0e-05 updt_s:0.013 data_s:0.000\n"
    )
    client = SubprocessClient(SubprocessTransport(workdir=paths.root))

    orch._ingest_training_log(client, paths)

    series = orch._read_metrics(paths.metrics_jsonl)
    assert len(series) == 2, "glued metric lines must not be dropped"
    # Precise step from the tqdm bar, NOT the coarse "1K"/"2K" (1000/2000).
    assert [s["step"] for s in series] == [1156, 2378]
    assert series[0]["loss"] == pytest.approx(1.596)
    assert series[1]["loss"] == pytest.approx(0.421)
    # No logging-prefix / bar noise in the series.
    assert "py" not in series[0] and "Training" not in series[0]

    prog = orch._read_progress(client, paths.progress_json)
    assert prog["step"] == 2378
    assert prog["total_steps"] == 3000


def test_ingest_is_idempotent_no_duplicate_metrics(tmp_path):
    # Re-running ingest on the same (unchanged) log must not duplicate rows —
    # full reparse + rewrite, so polls and restarts are safe.
    from lerobot.gui.training.hosts import HostRegistry
    from lerobot.gui.training.orchestrator import Orchestrator
    from lerobot.gui.training.runs import RunPaths, RunRegistry
    from lerobot.gui.training.transport import SubprocessClient, SubprocessTransport

    rr = RunRegistry(runs_dir=tmp_path / "runs")
    orch = Orchestrator(HostRegistry(hosts=[]), rr)
    paths = RunPaths.for_run("r1", rr.runs_dir)
    paths.ensure_exists()
    paths.stderr_log.write_text("step:100 loss:0.5\nstep:200 loss:0.4\n")
    client = SubprocessClient(SubprocessTransport(workdir=paths.root))

    orch._ingest_training_log(client, paths)
    orch._ingest_training_log(client, paths)
    assert [s["step"] for s in orch._read_metrics(paths.metrics_jsonl)] == [100, 200]


def test_ingest_does_not_clobber_externally_written_progress(tmp_path):
    # The test fake-runner writes progress.json itself and prints nothing
    # parseable. Ingest must leave that progress.json untouched.
    from lerobot.gui.training.hosts import HostRegistry
    from lerobot.gui.training.jobs import atomic_write_json
    from lerobot.gui.training.orchestrator import Orchestrator
    from lerobot.gui.training.runs import RunPaths, RunRegistry
    from lerobot.gui.training.transport import SubprocessClient, SubprocessTransport

    rr = RunRegistry(runs_dir=tmp_path / "runs")
    orch = Orchestrator(HostRegistry(hosts=[]), rr)
    paths = RunPaths.for_run("r1", rr.runs_dir)
    paths.ensure_exists()
    atomic_write_json(paths.progress_json, {"step": 42, "source": "fake-runner"})
    paths.stderr_log.write_text("[runner] starting fake training\nsome non-metric output\n")
    client = SubprocessClient(SubprocessTransport(workdir=paths.root))

    orch._ingest_training_log(client, paths)
    assert orch._read_progress(client, paths.progress_json) == {"step": 42, "source": "fake-runner"}


# ── Characterization: the contract resource telemetry will ride on ──────────
#
# DESIGN.md § Resource telemetry adds numeric utilization fields to the line
# lerobot-train already prints, on the strength of the parser auto-capturing
# every numeric key:value. These pin that guarantee BEFORE the feature exists,
# so breaking it fails a test rather than silently emptying a chart.
#
# They are equally the regression net for anything else that widens the line.


def test_telemetry_fields_do_not_displace_the_training_metrics():
    """The whole point: extra numeric fields are additive, never substitutive.

    A widened line must still yield loss/grdn/lr/updt_s/data_s. If this fails,
    the transport chosen in DESIGN.md is invalid and the design must change —
    not the trainer.
    """
    t = _train_tracker(initial_step=2000)
    t.loss = 0.526
    t.grad_norm = 41.688
    t.lr = 1.0e-5
    t.update_s = 0.053
    t.dataloading_s = 0.002
    line = str(t) + " cpu:71.2 cpu_max:88.0 rq:10 cores:32 pcpu:68.4 g0sm:75 g0pw:351"

    bag = parse_metric_sample(line)

    assert bag is not None
    assert bag["loss"] == pytest.approx(0.526)
    assert bag["grdn"] == pytest.approx(41.688)
    assert bag["lr"] == pytest.approx(1.0e-5)
    assert bag["updt_s"] == pytest.approx(0.053)
    assert bag["data_s"] == pytest.approx(0.002)
    assert bag["cpu"] == pytest.approx(71.2)
    assert bag["rq"] == pytest.approx(10)
    assert bag["g0sm"] == pytest.approx(75)


def test_a_metric_the_parser_has_never_heard_of_is_captured():
    """The zero-code-change promise, enforced rather than described.

    DESIGN.md § Integrating a trainer states that a policy logging a new metric
    charts with no change on either side. Every telemetry field is exactly such
    a metric, so the promise is this feature's foundation.
    """
    from lerobot.utils.logging_utils import AverageMeter

    t = _train_tracker(initial_step=500)
    t.loss = 0.1
    # A name no test and no parser knows, added the way a policy would add one.
    t.metrics["never_seen_before"] = AverageMeter("g7memtot", ":.1f")
    t.never_seen_before = 34359738368.0

    bag = parse_metric_sample(str(t))

    assert bag is not None
    assert bag["g7memtot"] == pytest.approx(34359738368.0)


def test_position_still_parses_from_a_widened_line():
    """Progress and metrics ride the same segment — real lerobot glues the
    metric log onto the tqdm bar — so widening must not cost the position."""
    bar = "Training:  50%|█████     | 2000/4000 [02:02<02:02,  1.05step/s]"
    t = _train_tracker(initial_step=2000)
    t.loss = 0.5
    line = bar + str(t) + " cpu:71.2 g0sm:75 g0_stat:0"

    progress = parse_progress(line)

    assert progress is not None
    assert progress.step == 2000
    assert progress.total_steps == 4000
    assert progress.eta_seconds == pytest.approx(122)


def test_the_two_ingestion_grades_stay_equivalent():
    """A flat line and a structured record must produce the same bag.

    HVLA S1 emits the record; lerobot-train emits the flat line. Telemetry is
    designed onto the flat path, so the two must not drift into disagreeing
    about what a metric bag contains.
    """
    flat = parse_metric_sample("INFO lerobot_train.py:611 step:2K loss:0.526 grdn:41.688 lr:1.0e-05 cpu:71.2")
    structured = parse_metric_sample(
        format_training_log_record(step=2000, total_steps=4000, loss=0.526, grdn=41.688, lr=1.0e-05, cpu=71.2)
    )

    assert flat is not None and structured is not None
    for key in ("loss", "grdn", "lr", "cpu"):
        assert flat[key] == pytest.approx(structured[key]), key


def test_a_non_finite_reading_costs_its_own_field_and_no_other():
    """A sampler that cannot read a source must emit a sentinel, never NaN.

    This is the reason DESIGN.md gives telemetry explicit ``cpu_stat``/``g0_stat``
    columns rather than letting an unreadable source produce a non-number: the
    parser drops non-finite values key by key, so a NaN would silently vanish
    and be indistinguishable from a field the trainer never emitted.
    """
    bag = parse_metric_sample("INFO step:1K loss:0.5 grdn:41.688 cpu:nan g0sm:75")

    assert bag is not None
    assert bag["loss"] == pytest.approx(0.5)
    assert bag["grdn"] == pytest.approx(41.688)
    assert bag["g0sm"] == pytest.approx(75)
    assert "cpu" not in bag, "a non-finite value must not reach the series"
    assert all(math.isfinite(v) for v in bag.values()), bag


def test_a_diverging_policy_drops_the_whole_sample():
    """A NaN *loss* is a real training outcome, and it costs the entire sample.

    Pinned because it bounds what telemetry can promise: during divergence the
    resource charts go blank too, since the sample carrying them is discarded.
    Telemetry must not be designed to be the thing that explains a NaN loss.
    """
    assert parse_metric_sample("INFO step:1K loss:nan grdn:41.688 cpu:71.2") is None
    assert parse_metric_sample("INFO step:1K loss:inf cpu:71.2") is None

    # And it is the loss specifically — not the presence of any NaN.
    assert parse_metric_sample("INFO step:1K loss:0.5 grdn:nan") is not None


def test_the_structured_record_refuses_a_non_finite_value():
    """The other grade rejects rather than drops, which is why telemetry rides
    the flat line: routing a diverging loss through the record would turn a
    diverging run into a crashed one."""
    with pytest.raises(ValueError):
        format_training_log_record(step=1, total_steps=10, loss=float("nan"))


def test_a_hand_built_record_cannot_smuggle_a_non_finite_value():
    """The record path needs its own finite check, and it is not reachable
    through ``format_training_log_record`` — that refuses NaN at the source.

    But a record is a wire format: HVLA writes one, and anything else may too.
    ``json.loads`` accepts bare ``NaN``/``Infinity``, so a hand-built record is
    the one way a non-finite value can reach the series. Pinned because a
    mutation removing this check passed every other test in this file.
    """
    smuggled = (
        TRAINING_LOG_JSON_MARKER
        + '{"loss":0.5,"step":1,"total_steps":10,"version":1,"cpu":NaN,"g0pw":Infinity}'
    )

    bag = parse_metric_sample(smuggled)

    assert bag is not None
    assert bag["loss"] == pytest.approx(0.5)
    assert "cpu" not in bag
    assert "g0pw" not in bag
    assert all(math.isfinite(v) for v in bag.values()), bag


def test_data_path_decision_is_extracted_for_display():
    """Which pipeline ran is not inferable from timings, and the fallback is
    quiet, so the GUI reads the decision (and the reason) off the log line."""
    from lerobot.gui.training.log_parse import parse_data_path

    gpu = parse_data_path("2026-08-22 [INFO] Data path: GPU (NVDEC decode + on-device composite/resize)")
    assert gpu == ("gpu", "NVDEC decode + on-device composite/resize")

    fell_back = parse_data_path(
        "2026-08-22 [WARNING] Data path: CPU (GPU path unavailable - RuntimeError: GPU decode of "
        "/x.mp4 does not reproduce the CPU decoder's pixels)"
    )
    assert fell_back is not None
    assert fell_back[0] == "cpu"
    assert "does not reproduce" in fell_back[1]

    assert parse_data_path("Data path: CPU (requested)") == ("cpu", "requested")
    assert parse_data_path("step 100/800 | loss: 0.5") is None
