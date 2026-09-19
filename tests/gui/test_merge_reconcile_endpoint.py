# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The merge dialog's reconcile checkbox reaches the merge itself.

The flag crosses an untyped boundary: the browser sends JSON, the endpoint
builds a request model, and something further down does the merging. Nothing
links the three at import time, so a field can be declared on the model, sent
by the client, and dropped by the endpoint in between while each layer looks
correct on its own -- the run then quietly takes the default. These tests
watch the value the merge is actually invoked with.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def merge_calls(monkeypatch):
    """POST to the real endpoint; record what it passes on, without merging."""
    from lerobot.gui import server as gui_server_mod
    from lerobot.gui.api import _edits_core

    calls: list[dict] = []

    async def fake_merge(_state, source_id, target_id, *, force=False, reconcile_features=False):
        calls.append(
            {"source": source_id, "target": target_id, "force": force, "reconcile": reconcile_features}
        )
        # The shape the endpoint reformats for the frontend.
        return {
            "source_episodes_merged": 0,
            "source_frames_merged": 0,
            "target_id": target_id,
            "target_episodes_after": 0,
            "target_frames_after": 0,
        }

    monkeypatch.setattr(_edits_core, "merge_dataset_into", fake_merge)
    return TestClient(gui_server_mod.app), calls


def _post(client, **over):
    body = {"source_dataset_id": "a/src", "target_dataset_id": "b/dst"}
    body.update(over)
    return client.post("/api/edits/merge-into", json=body)


def test_a_ticked_box_reaches_the_merge(merge_calls):
    client, calls = merge_calls
    assert _post(client, reconcile_features=True).status_code == 200
    assert calls and calls[-1]["reconcile"] is True


def test_an_unticked_box_does_not(merge_calls):
    """The complement. Without it, code that always reconciled would satisfy
    the test above -- and silently reshaping a dataset nobody asked to reshape
    is the outcome the opt-in exists to prevent."""
    client, calls = merge_calls
    assert _post(client, reconcile_features=False).status_code == 200
    assert calls[-1]["reconcile"] is False


def test_the_default_is_not_to_reconcile(merge_calls):
    """A client that never heard of the flag must get the old behaviour."""
    client, calls = merge_calls
    assert _post(client).status_code == 200
    assert calls[-1]["reconcile"] is False


def test_reconcile_and_force_stay_independent(merge_calls):
    """Different decisions: reconciling makes the schemas agree and then
    validates; forcing skips validation. Neither may imply the other."""
    client, calls = merge_calls
    _post(client, reconcile_features=True)
    assert calls[-1]["force"] is False

    _post(client, force=True)
    assert calls[-1]["reconcile"] is False

    _post(client, force=True, reconcile_features=True)
    assert (calls[-1]["force"], calls[-1]["reconcile"]) == (True, True)
