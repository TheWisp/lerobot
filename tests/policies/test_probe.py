# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The probe must separate pathologies that look identical on the robot.

Written against policies whose failure mode is known by construction, because a
real checkpoint cannot prove a detector fires — if it were healthy there would
be nothing to detect.

The load-bearing test is ``test_a_multimodal_policy_is_not_called_broken``: the
whole reason MAE was dropped as the headline metric is that a policy which
correctly represents two valid behaviours scores *worse* on pointwise distance
than one that averages them into an invalid action. If that test ever passes
for the wrong reason, the metric has regressed to the thing it replaced.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lerobot.policies.probe import probe_frames
from lerobot.utils.constants import ACTION

ACTION_DIM = 4
HORIZON = 6


class _Dataset:
    """Frames whose recorded actions genuinely differ, with unit action std."""

    repo_id = "test/probe"

    def __init__(self, n: int = 6, std: float = 1.0):
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

        class _Meta:
            stats = {ACTION: {"std": np.full(ACTION_DIM, std)}}

        self.meta = _Meta()

    def __getitem__(self, i):
        return self._items[i]

    def __len__(self):
        return len(self._items)


class _Collapsed:
    """Same chunk whatever it is shown."""

    def predict_action_chunk(self, batch):
        return torch.zeros(1, HORIZON, ACTION_DIM)


def _frame_of(batch) -> int:
    """Which frame the probe is asking about.

    The fakes must answer per-frame or they are not modelling a policy at all:
    a fake that returns frame 0 forever is indistinguishable from a collapsed
    one, which silently inverts what these tests claim to show.
    """
    return int(batch["frame_index"].reshape(-1)[0])


class _Perfect:
    """Reproduces the recorded action for whichever frame it is shown."""

    def __init__(self, ds):
        self._ds = ds

    def predict_action_chunk(self, batch):
        return self._ds[_frame_of(batch)][ACTION].unsqueeze(0)


class _Bimodal:
    """Alternates between two valid behaviours, one of which is the recorded one.

    Its *average* would be neither. This is the policy a pointwise metric
    punishes and a best-of-K metric does not.
    """

    def __init__(self, ds):
        self._ds, self._flip = ds, 0

    def predict_action_chunk(self, batch):
        truth = self._ds[_frame_of(batch)][ACTION]
        self._flip += 1
        # Odd samples reproduce the demonstration; even ones take the other mode.
        return (truth if self._flip % 2 else -truth).unsqueeze(0)


def test_a_perfect_policy_shows_no_pathology():
    ds = _Dataset()
    policy = _Perfect(ds)

    report = probe_frames(policy, ds, range(len(ds)), device="cpu", samples=2)

    assert report.best_of_k == pytest.approx(0.0, abs=1e-6)
    assert any("NO PATHOLOGY" in f for f in report.findings())


def test_a_collapsed_policy_is_reported_as_ignoring_input():
    """The signal multimodality cannot explain away."""
    ds = _Dataset()

    report = probe_frames(_Collapsed(), ds, range(len(ds)), device="cpu", samples=2)

    assert report.between_frame_spread == pytest.approx(0.0, abs=1e-9)
    assert any("IGNORES ITS INPUT" in f for f in report.findings())


def test_a_multimodal_policy_is_not_called_broken():
    """The reason MAE was dropped.

    This policy is *correct*: it represents two valid behaviours and one of them
    is what the operator did. Pointwise mean distance punishes it; best-of-K
    sees that it covers the demonstrated mode.
    """
    ds = _Dataset()

    report = probe_frames(_Bimodal(ds), ds, range(len(ds)), device="cpu", samples=4)

    assert report.best_of_k < 1e-6, "best-of-K must find the mode the operator used"
    assert report.best_of_k < report.frames[0].mean_distance, (
        "a pointwise mean must be strictly worse here — otherwise the metric is not doing its job"
    )
    assert not any("IGNORES ITS INPUT" in f for f in report.findings())


def test_a_wide_conditional_is_flagged_as_a_deployment_problem():
    """The candidate mechanism for shaking: same input, different chunk."""
    ds = _Dataset()

    report = probe_frames(_Bimodal(ds), ds, range(len(ds)), device="cpu", samples=4)

    assert report.sample_spread > 0.0
    assert any("WIDE CONDITIONAL" in f for f in report.findings())


def test_training_loss_is_reported_when_the_policy_exposes_one():
    ds = _Dataset()

    class _WithLoss(_Collapsed):
        def forward(self, batch):
            return torch.tensor(2.5), {"loss": 2.5}

    report = probe_frames(_WithLoss(), ds, range(2), device="cpu", samples=2)

    assert report.training_loss == pytest.approx(2.5)
    assert any("DID NOT FIT" in f for f in report.findings())


def test_a_policy_without_a_training_forward_is_not_an_error():
    ds = _Dataset()

    report = probe_frames(_Collapsed(), ds, range(2), device="cpu", samples=2)

    assert report.training_loss is None
    assert not any("DID NOT FIT" in f for f in report.findings())


def test_distances_are_normalised_per_joint():
    """Raw units weight a shoulder and a gripper equally; std-units do not."""
    wide = _Dataset(std=10.0)

    report = probe_frames(_Collapsed(), wide, range(len(wide)), device="cpu", samples=2)
    raw = probe_frames(_Collapsed(), _Dataset(std=1.0), range(6), device="cpu", samples=2)

    assert report.best_of_k == pytest.approx(raw.best_of_k / 10.0, rel=1e-3)


class TestItRefusesToCompareThingsThatDoNotLineUp:
    def test_mismatched_action_width_is_skipped(self, caplog):
        ds = _Dataset()

        class _WrongWidth:
            def predict_action_chunk(self, batch):
                return torch.zeros(1, HORIZON, ACTION_DIM + 1)

        with caplog.at_level("WARNING"):
            report = probe_frames(_WrongWidth(), ds, range(len(ds)), device="cpu", samples=2)

        assert report.frames == []
        assert "do not line up" in caplog.text

    def test_a_shorter_horizon_compares_only_the_overlap(self):
        ds = _Dataset()

        class _Short:
            def predict_action_chunk(self, batch):
                return torch.zeros(1, 3, ACTION_DIM)

        report = probe_frames(_Short(), ds, range(len(ds)), device="cpu", samples=2)

        assert report.frames[0].ground_truth_frames == 3


def test_every_sample_re_runs_inference():
    """Without reset(), a queueing policy returns its cached chunk and the
    sampling-spread signal silently reads zero."""
    ds = _Dataset()
    resets = []

    class _Queued(_Collapsed):
        def reset(self):
            resets.append(True)

    probe_frames(_Queued(), ds, range(2), device="cpu", samples=3)

    assert len(resets) == 2 * 3


def test_the_report_serialises():
    ds = _Dataset()

    payload = probe_frames(_Collapsed(), ds, range(2), device="cpu", samples=2).as_dict()

    assert payload["n_frames"] == 2
    assert isinstance(payload["findings"], list)
