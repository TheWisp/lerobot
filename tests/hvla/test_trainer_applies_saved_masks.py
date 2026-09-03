# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""A masked dataset must reach the HVLA trainer masked.

`LeRobotDataset.apply_saved_masks` defaults to **False**. The `True` default
lives in `configs/default.py`, which reaches the dataset through
`datasets/factory.py` — and this trainer does not use that factory. It builds
`LeRobotDataset(repo_id)` directly, so it inherited nothing and read raw frames
from a dataset that had been deliberately masked.

Nothing failed. Loss fell, throughput was normal, the log named a dataset with
"mask" in its title, and the run was worthless: it was measuring the effect of
masking on pixels that were never masked. It was caught by reading the source,
not by any signal from the run.

So the source is what is checked. Both halves matter: the construction has to
pass the flag, and the flag has to default to ON — a `--apply-saved-masks`
opt-in would reproduce the same failure for anyone who forgets it, which is why
the escape hatch is `--ignore-saved-masks` instead.
"""

import ast
from pathlib import Path

TRAINER = Path(__file__).resolve().parents[2] / "src/lerobot/policies/hvla/s1/flow_matching/train.py"


def _dataset_construction() -> ast.Call:
    """The `LeRobotDataset(...)` call the trainer builds its dataset with."""
    tree = ast.parse(TRAINER.read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "LeRobotDataset"
    ]
    assert len(calls) == 1, f"expected one LeRobotDataset construction, found {len(calls)}"
    return calls[0]


def _flag_registration(flag: str) -> ast.Call:
    """The trainer's own `parser.add_argument(flag, ...)` call."""
    tree = ast.parse(TRAINER.read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "add_argument"
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == flag
    ]
    assert len(calls) == 1, f"expected one registration of {flag}, found {len(calls)}"
    return calls[0]


def test_the_trainer_asks_for_saved_masks():
    """Without this argument the dataset silently yields raw frames."""
    kwargs = {k.arg for k in _dataset_construction().keywords}
    assert "apply_saved_masks" in kwargs, (
        "the HVLA trainer builds LeRobotDataset without apply_saved_masks, so a masked "
        "dataset trains on raw pixels and nothing about the run says so"
    )


def test_masks_are_on_unless_explicitly_refused():
    """The default must be ON. An opt-in flag would leave the same trap for
    whoever forgets it, which is the failure this test exists for."""
    src = TRAINER.read_text()
    assert "--ignore-saved-masks" in src, "the escape hatch should be opt-OUT, not opt-in"
    assert "--apply-saved-masks" not in src, (
        "an opt-in flag reintroduces the defect: forgetting it trains on raw frames"
    )

    # Read the flag off the TRAINER's own add_argument call. Building a parser
    # here and asserting its default would only be re-testing argparse: it
    # would stay green if the trainer registered the flag with `default=True`,
    # which is precisely the regression this guards.
    reg = _flag_registration("--ignore-saved-masks")
    kwargs = {k.arg: k.value for k in reg.keywords}
    assert "default" not in kwargs, "an explicit default on a store_true flag is a way to make it default-on"
    assert isinstance(kwargs.get("action"), ast.Constant) and kwargs["action"].value == "store_true", (
        "the escape hatch must be a store_true flag, so omitting it applies the masks"
    )


def test_the_run_log_states_which_it_did():
    """A run has to be answerable after the fact — the previous one was not."""
    src = TRAINER.read_text()
    assert "Saved masks ACTIVE" in src
    assert "Saved masks IGNORED" in src
    assert "no saved masks" in src
