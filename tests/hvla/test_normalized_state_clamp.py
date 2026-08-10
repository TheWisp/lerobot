"""The normalized-state clamp, driven by the numbers that motivated it.

A joint held still for a whole recording gets a training std at the 1e-6
numerical floor. Dividing by that, a reading difference below sensor resolution
becomes enormous, and one oversized channel corrupts the first linear layer for
every joint -- not only the still one.

Measured on GPU/0803_20260803_174402, the dataset behind the checkpoint under
investigation: left_joint_3.pos has mean 0.9508 and std 1.000e-06 across all
40,101 frames. The rig read 0.732. That 0.22-degree difference normalized to
218,569 sigma.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

from lerobot.policies.hvla.s1.flow_matching.model import (
    NORMALIZED_STATE_CLAMP,
    FlowMatchingS1Policy,
)

# Measured, not invented: left_joint_3.pos in GPU/0803_20260803_174402.
TRAIN_MEAN = 0.9508
TRAIN_STD_DEGENERATE = 1.0e-06  # torch's own clamp(min=1e-6), i.e. zero variance
TRAIN_STD_FLOORED = 0.5  # what state_position_std_floor substitutes
RIG_VALUE = 0.732  # 0.22 degrees from the recorded mean


class _Bare(FlowMatchingS1Policy):
    """Only the normalization path; no network, no weights."""

    def __init__(self, names: list[str], mean: torch.Tensor, std: torch.Tensor):
        torch.nn.Module.__init__(self)
        self._clamped_state_features = set()
        self._state_mean = mean
        self._state_std = std

        class _Cfg:
            image_features = None
            state_feature_names = names

        self.config = _Cfg()


def _policy(std: float, names=("left_joint_3.pos",)):
    return _Bare(list(names), torch.tensor([TRAIN_MEAN]), torch.tensor([std]))


def _normalize(policy, value):
    return policy.prepare_batch_for_encode_observations({"observation.state": torch.tensor([[value]])})[
        "observation.state"
    ]


def test_a_quarter_of_a_degree_is_enough_to_break_it():
    """State the size of the problem, so the fix has something to be measured against.

    This is the part that makes it worth guarding: the trigger is not operator
    error, it is a difference no operator could notice or correct.
    """
    offset_degrees = abs(RIG_VALUE - TRAIN_MEAN)
    sigma = offset_degrees / TRAIN_STD_DEGENERATE

    assert offset_degrees < 0.25, "the trigger is a sub-degree difference"
    assert sigma > 200_000


@pytest.mark.parametrize("std", [TRAIN_STD_DEGENERATE, TRAIN_STD_FLOORED])
def test_the_still_joint_is_bounded_with_and_without_the_floor(std):
    """The floor is not sufficient on its own -- it covers ``.pos`` only, and
    torque channels on this same frame still reach 13.7 sigma -- so the clamp
    has to hold in both configurations."""
    assert _normalize(_policy(std), RIG_VALUE).abs().max().item() == pytest.approx(
        min(NORMALIZED_STATE_CLAMP, abs(RIG_VALUE - TRAIN_MEAN) / std)
    )


def test_an_ordinary_observation_is_untouched():
    """The clamp must be invisible for a typical frame.

    Not for every frame -- see ``test_the_clamp_is_not_free``.
    """
    ordinary = TRAIN_MEAN + 4.0 * TRAIN_STD_FLOORED
    assert _normalize(_policy(TRAIN_STD_FLOORED), ordinary).item() == pytest.approx(4.0)


def test_healthy_joints_survive_a_neighbour_being_clamped():
    """The point of the clamp: one degenerate channel must not disturb the rest."""
    policy = _Bare(
        ["left_joint_3.pos", "right_joint_3.pos"],
        torch.tensor([TRAIN_MEAN, 0.0]),
        torch.tensor([TRAIN_STD_DEGENERATE, 25.4]),  # right_joint_3.pos native std
    )
    out = policy.prepare_batch_for_encode_observations(
        {"observation.state": torch.tensor([[RIG_VALUE, 63.5]])}
    )
    still, working = out["observation.state"][0].tolist()

    assert abs(still) == pytest.approx(NORMALIZED_STATE_CLAMP)
    assert working == pytest.approx(2.5), "a joint that moves must pass through unchanged"


def test_the_warning_names_the_joint_and_its_std(caplog):
    """Silent clamping would turn a loud failure into a quiet one. The std is in
    the message because it is what identifies the cause; the sigma count alone
    does not distinguish a degenerate channel from a genuinely wild reading."""
    with caplog.at_level(logging.WARNING):
        _normalize(_policy(TRAIN_STD_DEGENERATE), RIG_VALUE)

    assert "left_joint_3.pos" in caplog.text
    assert "1e-06" in caplog.text or "1.0e-06" in caplog.text


def test_it_warns_once_per_feature_not_once_per_frame(caplog):
    """Inference runs at 30 Hz; a per-frame warning would bury the log."""
    policy = _policy(TRAIN_STD_DEGENERATE)
    with caplog.at_level(logging.WARNING):
        for _ in range(50):
            _normalize(policy, RIG_VALUE)

    assert caplog.text.count("left_joint_3.pos reached") == 1


def test_no_warning_when_nothing_is_clamped(caplog):
    with caplog.at_level(logging.WARNING):
        _normalize(_policy(TRAIN_STD_FLOORED), TRAIN_MEAN)

    assert "clamped" not in caplog.text


def test_the_clamp_is_not_free():
    """Record what a bound of 10 costs, so it stays a decision and not an assumption.

    It truncates real motion, not only degenerate channels: across
    GPU/0803_20260803_174402, 0.85% of frames have at least one feature past 10
    sigma, and right_joint_7.vel peaks at 20.5 on a joint whose native std is
    17 deg/s. Raising or lowering the bound should be argued against that
    distribution rather than against the belief that it never fires.
    """
    measured_peak_on_a_moving_joint = 20.5  # right_joint_7.vel

    assert measured_peak_on_a_moving_joint > NORMALIZED_STATE_CLAMP, (
        "if real motion no longer exceeds the bound, re-measure and update this"
    )


def test_training_and_inference_apply_the_same_bound():
    """A clamp on one side only is a train/serve skew.

    The trainer normalizes in ``FlowMatchingDataset`` and the policy in
    ``prepare_batch_for_encode_observations``. Both must clamp, and against the
    same constant -- a bound that drifts between them is silent, surfacing only
    as degraded rollouts.
    """
    trainer = (
        Path(__file__).resolve().parents[2] / "src/lerobot/policies/hvla/s1/flow_matching/train.py"
    ).read_text()

    assert "clamp(-NORMALIZED_STATE_CLAMP, NORMALIZED_STATE_CLAMP)" in trainer, (
        "the trainer must clamp its normalized states"
    )
    assert "NORMALIZED_STATE_CLAMP = " not in trainer, (
        "the trainer must import the bound from the policy, not redefine it"
    )
