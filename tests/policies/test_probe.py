# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The probe has to tell a collapsed policy from a working one, or it is decoration.

Written against synthetic policies whose failure mode is known by construction:
a real checkpoint cannot prove the detector fires, because if it were healthy
there would be nothing to detect. The three fakes below stand in for the three
things "the robot idles and shakes" can actually mean.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lerobot.policies.probe import probe_frames
from lerobot.utils.constants import ACTION

ACTION_DIM = 4
HORIZON = 8


class _Dataset:
    """Frames whose recorded actions genuinely differ from each other."""

    repo_id = "test/probe"

    def __init__(self, n: int = 6):
        rng = np.random.default_rng(0)
        self._items = [
            {
                "observation.state": torch.tensor(rng.normal(size=ACTION_DIM), dtype=torch.float32),
                ACTION: torch.tensor(rng.normal(size=(HORIZON, ACTION_DIM)), dtype=torch.float32),
                "episode_index": torch.tensor(0),
                "frame_index": torch.tensor(i),
            }
            for i in range(n)
        ]

    def __getitem__(self, i):
        return self._items[i]

    def __len__(self):
        return len(self._items)


class _Collapsed:
    """Emits the same chunk whatever it is shown — a model that learned nothing."""

    def predict_action_chunk(self, batch):
        return torch.zeros(1, HORIZON, ACTION_DIM)


class _FlatButResponsive:
    """Responds to input, but each chunk is constant over the horizon."""

    def predict_action_chunk(self, batch):
        value = batch["observation.state"].mean()
        return value.expand(1, HORIZON, ACTION_DIM).clone()


class _Perfect:
    """Reproduces the recorded action exactly."""

    def __init__(self, dataset):
        self._ds = dataset
        self._i = 0

    def predict_action_chunk(self, batch):
        truth = self._ds[self._i][ACTION]
        self._i += 1
        return truth.unsqueeze(0)


def test_a_perfect_policy_scores_zero_error():
    ds = _Dataset()

    report = probe_frames(_Perfect(ds), ds, range(len(ds)), device="cpu")

    assert report.mae == pytest.approx(0.0, abs=1e-6)
    assert "RESPONSIVE" in report.verdict()


def test_a_collapsed_policy_is_named_as_collapsed():
    """The headline case: this is what 'idles in place' looks like in numbers."""
    ds = _Dataset()

    report = probe_frames(_Collapsed(), ds, range(len(ds)), device="cpu")

    assert report.between_frame_spread == pytest.approx(0.0, abs=1e-9)
    assert "COLLAPSED" in report.verdict()
    assert report.mae > 0.1, "a collapsed policy must not look accurate"


def test_flat_chunks_are_distinguished_from_collapse():
    """Different bug, different fix — the verdict must not conflate them."""
    ds = _Dataset()

    report = probe_frames(_FlatButResponsive(), ds, range(len(ds)), device="cpu")

    assert report.action_spread == pytest.approx(0.0, abs=1e-6), "chunk is flat by construction"
    assert report.between_frame_spread > 0.0, "but it does respond to input"
    assert "FLAT CHUNKS" in report.verdict()


def test_per_joint_error_is_reported():
    """An aggregate MAE hides a single mis-ordered joint; the per-joint one does not."""
    ds = _Dataset()

    report = probe_frames(_Collapsed(), ds, range(len(ds)), device="cpu")

    assert len(report.frames[0].mae_per_joint) == ACTION_DIM
    assert report.frames[0].action_dim == ACTION_DIM


class TestItRefusesToCompareThingsThatDoNotLineUp:
    """A wrong number is worse than a skipped frame."""

    def test_mismatched_action_width_is_skipped(self, caplog):
        ds = _Dataset()

        class _WrongWidth:
            def predict_action_chunk(self, batch):
                return torch.zeros(1, HORIZON, ACTION_DIM + 1)

        with caplog.at_level("WARNING"):
            report = probe_frames(_WrongWidth(), ds, range(len(ds)), device="cpu")

        assert report.frames == []
        assert "do not line up" in caplog.text

    def test_shorter_horizon_compares_only_the_overlap(self):
        """Dataset window and policy horizon are configured independently."""
        ds = _Dataset()

        class _ShortHorizon:
            def predict_action_chunk(self, batch):
                return torch.zeros(1, 3, ACTION_DIM)

        report = probe_frames(_ShortHorizon(), ds, range(len(ds)), device="cpu")

        assert report.frames[0].ground_truth_frames == 3
        assert report.frames[0].horizon == 3

    def test_a_frame_without_a_recorded_action_is_skipped(self, caplog):
        ds = _Dataset(n=2)
        del ds._items[0][ACTION]

        with caplog.at_level("WARNING"):
            report = probe_frames(_Collapsed(), ds, range(2), device="cpu")

        assert len(report.frames) == 1


def test_chunk_queue_policies_are_reset_between_frames():
    """Without reset(), a queueing policy serves frame N-1's chunk for frame N."""
    ds = _Dataset()
    resets = []

    class _Queued:
        def reset(self):
            resets.append(True)

        def predict_action_chunk(self, batch):
            return torch.zeros(1, HORIZON, ACTION_DIM)

    probe_frames(_Queued(), ds, range(len(ds)), device="cpu")

    assert len(resets) == len(ds)


def test_the_report_serialises_for_a_cli_or_ui():
    ds = _Dataset()

    payload = probe_frames(_Collapsed(), ds, range(2), device="cpu").as_dict()

    assert payload["n_frames"] == 2
    assert "verdict" in payload
    assert len(payload["frames"]) == 2
