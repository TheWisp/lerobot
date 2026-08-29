"""Structured metrics emitted by the HVLA S1 generalization probes."""

from __future__ import annotations

from lerobot.policies.hvla.s1.flow_matching.train import generalization_log_metrics


def test_generalization_results_are_flattened_for_the_training_gui() -> None:
    metrics = generalization_log_metrics(
        0.125,
        {
            "train": {"chunk_error": 0.31, "null_error": 0.98, "ratio": 0.316},
            "held_out": {"chunk_error": 0.57, "null_error": 1.0, "ratio": 0.569},
            "ratio_gap": 0.253,
        },
    )

    assert metrics == {
        "validation_flow_loss": 0.125,
        "generation_train_chunk_error": 0.31,
        "generation_train_null_error": 0.98,
        "generation_train_ratio": 0.316,
        "generation_held_out_chunk_error": 0.57,
        "generation_held_out_null_error": 1.0,
        "generation_held_out_ratio": 0.569,
        "generation_ratio_gap": 0.253,
    }


def test_missing_or_non_finite_probe_values_are_not_logged() -> None:
    metrics = generalization_log_metrics(
        None,
        {
            "train": {"chunk_error": 0.31, "null_error": float("nan"), "ratio": 0.316},
            "held_out": None,
            "ratio_gap": float("inf"),
        },
    )

    assert metrics == {
        "generation_train_chunk_error": 0.31,
        "generation_train_ratio": 0.316,
    }
