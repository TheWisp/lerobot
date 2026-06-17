# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Shared fixtures for the GUI test suite.

The fake-training worker (:mod:`tests.gui.training.fake_runner`) is a test
fixture, not shipped production code — the real training path is docker +
``lerobot-train`` (see ``recipes.build_lerobot_train_command``). Production
therefore leaves ``recipes.FAKE_RUNNER_PATH`` unset; the autouse fixture below
points it at the test worker so any test using the ``__recipe__=__fake__``
recipe can spawn it (by absolute file path — ``tests`` isn't importable from
the worker's subprocess cwd, so ``python -m`` won't resolve it).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from lerobot.gui.training import recipes

FAKE_RUNNER_PATH = Path(__file__).parent / "training" / "fake_runner.py"


@pytest.fixture(autouse=True)
def _inject_fake_runner() -> Iterator[None]:
    """Wire the test fake-training worker into the fake recipe for the
    duration of each GUI test, then restore production's unset default."""
    assert FAKE_RUNNER_PATH.exists(), f"fake runner missing at {FAKE_RUNNER_PATH}"
    prev = recipes.FAKE_RUNNER_PATH
    recipes.FAKE_RUNNER_PATH = str(FAKE_RUNNER_PATH)
    try:
        yield
    finally:
        recipes.FAKE_RUNNER_PATH = prev
