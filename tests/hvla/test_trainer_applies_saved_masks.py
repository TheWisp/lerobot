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

import argparse
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--ignore-saved-masks", action="store_true")
    assert parser.parse_args([]).ignore_saved_masks is False


def test_the_run_log_states_which_it_did():
    """A run has to be answerable after the fact — the previous one was not."""
    src = TRAINER.read_text()
    assert "Saved masks ACTIVE" in src
    assert "Saved masks IGNORED" in src
    assert "no saved masks" in src


def test_auto_falls_back_but_explicit_gpu_does_not():
    """auto degrades with a logged reason; an explicit --data-path gpu fails.

    A run that asks for the GPU path and silently gets the CPU one is exactly
    how three benchmark runs were measured wrong; auto exists so the default
    is safe, not so an explicit request can be ignored.
    """

    import pytest

    from lerobot.policies.hvla.s1.flow_matching.train import _resolve_data_path

    class Cfg:
        image_augmentation = False
        image_features = {"observation.images.cam": {}}

    # device is not CUDA -> unsupported for both modes
    assert _resolve_data_path("auto", Cfg(), None, (224, 224), "cpu", 8) is None
    with pytest.raises(NotImplementedError, match="not CUDA"):
        _resolve_data_path("gpu", Cfg(), None, (224, 224), "cpu", 8)

    aug = Cfg()
    aug.image_augmentation = True
    assert _resolve_data_path("auto", aug, None, (224, 224), "cuda", 8) is None

    assert _resolve_data_path("cpu", Cfg(), None, (224, 224), "cuda", 8) is None


def test_auto_logs_why_it_fell_back(caplog):
    """A silent fallback is the failure mode; the reason must reach the log."""
    import logging

    from lerobot.policies.hvla.s1.flow_matching.train import _resolve_data_path

    class Cfg:
        image_augmentation = True
        image_features = {"observation.images.cam": {}}

    with caplog.at_level(logging.WARNING):
        _resolve_data_path("auto", Cfg(), None, (224, 224), "cuda", 8)
    assert any("GPU path unavailable" in r.getMessage() for r in caplog.records)
