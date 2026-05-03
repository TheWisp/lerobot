"""Tests for MetricsLogger file outputs."""

from __future__ import annotations

import json

from lerobot.policies.hvla.flash_dagger.metrics import CycleMetrics, MetricsLogger


def _sample_metrics(cycle: int = 0) -> CycleMetrics:
    return CycleMetrics(
        cycle=cycle,
        episode=cycle * 2,
        correction_id=cycle,
        n_intervention_frames=200,
        n_train_frames=160,
        n_val_frames=40,
        n_steps=100,
        wall_seconds=12.5,
        loss_new_train_final=0.04,
        loss_new_val_pre=0.27,
        loss_new_val_post=0.04,
        loss_old_val_pre=0.025,
        loss_old_val_post=0.029,
        loss_flashed_val_pre=None,
        loss_flashed_val_post=None,
        swap_accepted=True,
        swap_reject_reason="",
        n_lora_layers=12,
        frobenius_max=0.42,
        effective_rank_max=14,
    )


def test_metrics_logger_creates_dirs(tmp_path):
    MetricsLogger(tmp_path / "out")
    assert (tmp_path / "out").is_dir()
    assert (tmp_path / "out" / "curves").is_dir()
    assert (tmp_path / "out" / "layer_diag").is_dir()


def test_metrics_logger_writes_summary_jsonl(tmp_path):
    log = MetricsLogger(tmp_path / "out")
    log.write_cycle(_sample_metrics(0))
    log.write_cycle(_sample_metrics(1))
    summary = (tmp_path / "out" / "summary.jsonl").read_text().strip().split("\n")
    assert len(summary) == 2
    row0 = json.loads(summary[0])
    assert row0["cycle"] == 0
    assert row0["swap_accepted"] is True
    assert row0["loss_new_val_post"] == 0.04


def test_metrics_logger_curve_csv(tmp_path):
    log = MetricsLogger(tmp_path / "out")
    log.write_curve(7, [0.5, 0.4, 0.3, 0.2, 0.1])
    body = (tmp_path / "out" / "curves" / "cycle_0007.csv").read_text()
    lines = body.strip().split("\n")
    assert lines[0] == "step,train_loss"
    assert len(lines) == 6  # header + 5 steps


def test_metrics_logger_layer_diag_csv(tmp_path):
    log = MetricsLogger(tmp_path / "out")
    diag = [
        {
            "layer": "model.decoder_layers.0.self_attn.q_proj",
            "rank": 16,
            "frobenius": 0.1,
            "effective_rank": 8,
        },
        {"layer": "model.decoder_layers.0.linear1", "rank": 16, "frobenius": 0.2, "effective_rank": 12},
    ]
    log.write_layer_diag(3, diag)
    body = (tmp_path / "out" / "layer_diag" / "cycle_0003.csv").read_text()
    lines = body.strip().split("\n")
    assert lines[0] == "layer,rank,frobenius,effective_rank"
    assert len(lines) == 3  # header + 2 layers


def test_metrics_logger_layer_diag_skips_empty(tmp_path):
    log = MetricsLogger(tmp_path / "out")
    log.write_layer_diag(5, [])
    assert not (tmp_path / "out" / "layer_diag" / "cycle_0005.csv").exists()
