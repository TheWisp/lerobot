# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Shared fixtures for the GUI test suite.

The fake-training worker is a test fixture (production trains via docker +
``lerobot-train``). The autouse fixture points ``recipes.FAKE_RUNNER_PATH``
at it so ``__recipe__=__fake__`` runs can spawn it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from lerobot.gui.training import recipes

FAKE_RUNNER_PATH = Path(__file__).parent / "training" / "fake_runner.py"


@pytest.fixture(autouse=True)
def _inject_fake_runner() -> Iterator[None]:
    assert FAKE_RUNNER_PATH.exists(), f"fake runner missing at {FAKE_RUNNER_PATH}"
    prev = recipes.FAKE_RUNNER_PATH
    recipes.FAKE_RUNNER_PATH = str(FAKE_RUNNER_PATH)
    try:
        yield
    finally:
        recipes.FAKE_RUNNER_PATH = prev


@pytest.fixture(autouse=True)
def isolate_hub_transfer_history(tmp_path, monkeypatch) -> Iterator[Path]:
    """Keep the transfer-outcome history out of the developer's real config.

    Every terminal transfer now appends to
    ``~/.config/lerobot/gui/hub_transfers.jsonl``, and the GUI shows that file
    to the user as their transfer history. Before this fixture the suite wrote
    into it — 105 fixture entries for repos like ``user/repo`` and ``u/ds``
    were found in a real one — so the tray would have offered invented
    transfers as fact.

    Autouse and suite-wide on purpose. Patching the individual fixtures missed
    tests that build a ``_WorkerState`` in-process, and would keep missing each
    new one; the property wanted is "no test anywhere touches it".

    Both channels are covered: the module constant for in-process writers, and
    the env var for the worker subprocesses, which inherit the environment and
    cannot see a monkeypatched module.
    """
    from lerobot.gui import hub_history

    path = tmp_path / "hub_transfers.jsonl"
    monkeypatch.setattr(hub_history, "HISTORY_PATH", path)
    monkeypatch.setenv(hub_history.HISTORY_PATH_ENV, str(path))
    yield path
