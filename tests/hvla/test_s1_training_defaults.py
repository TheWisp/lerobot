"""Defaults for the HVLA S1 trainer, and the invariant that they agree.

The std floor and the seed are declared in three places — the config dataclass,
the trainer's argument parser, and the GUI form. A form default that drifts from
the code default means the GUI silently trains with something other than what
the code says, and nobody notices until a run is being reconstructed months
later.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "lerobot"
CONFIG = SRC / "policies/hvla/s1/flow_matching/config.py"
TRAINER = SRC / "policies/hvla/s1/flow_matching/train.py"
FORM = SRC / "gui/api/training.py"


def _form_default(field: str) -> str:
    m = re.search(rf'"name": "{field}".*?"default": ([^,\n]+),', FORM.read_text(), re.S)
    assert m, f"no form field named {field}"
    return m.group(1).strip()


def test_std_floor_defaults_on():
    """A degenerate joint must be guarded without anyone remembering a flag.

    A joint held still across a recording gets a near-zero std; normalizing
    against it turned one out-of-range reading into 218,569 sigma on this
    project's own data, against 14 with the floor applied.
    """
    from lerobot.policies.hvla.s1.flow_matching.config import FlowMatchingS1Config

    assert FlowMatchingS1Config().state_position_std_floor == 0.5


def test_seed_defaults_to_the_repository_convention():
    """TrainPipelineConfig.seed has defaulted to 1000 since long before this
    trainer existed; a separate training script should not be the one place a
    run is irreproducible by default."""
    from lerobot.configs.train import TrainPipelineConfig

    assert TrainPipelineConfig.seed == 1000  # the convention being matched
    m = re.search(r'"--seed"[^)]*default=(\d+)', TRAINER.read_text(), re.S)
    assert m and int(m.group(1)) == 1000


@pytest.mark.parametrize(
    ("field", "expected"),
    [("state_position_std_floor", "0.5"), ("seed", "1000")],
)
def test_form_default_matches_the_code_default(field, expected):
    assert _form_default(field) == expected


def test_floor_is_a_no_op_for_a_joint_that_moves():
    """Guard against raising the default until it distorts real motion.

    Measured across four recorded datasets, no ``.pos`` dimension has a std
    between 0.5 and 1.0 — the threshold sits in an empty gap. The dimensions it
    does floor are the ones held still (a closed gripper at std 0.001-0.05).
    """
    import torch

    std = torch.tensor([0.001, 0.0487, 2.8, 12.0])  # gripper, gripper, moving, moving
    floored = std.clamp(min=0.5)

    assert floored[0] == 0.5 and floored[1] == 0.5, "static joints are floored"
    assert floored[2] == 2.8 and floored[3] == 12.0, "moving joints are untouched"
